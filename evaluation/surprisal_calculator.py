"""
Surprisal calculation utilities for language model evaluation.
Implements the surprisal equation from the outline document:
S(w_i) = -log₂ P(w_i | w_1, w_2, ..., w_{i-1})
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import sentencepiece as spm
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class SurprisalCalculator:
    """Calculate surprisal values for text using language models."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: spm.SentencePieceProcessor,
        device: str = None
    ):
        """
        Initialize the surprisal calculator.
        
        Args:
            model: Loaded language model
            tokenizer: SentencePiece tokenizer
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure model is on correct device
        if not next(model.parameters()).is_cuda and self.device == 'cuda':
            self.model = self.model.to(self.device)
        
        # Get special token IDs
        self.bos_id = tokenizer.bos_id() if tokenizer.bos_id() >= 0 else None
        self.eos_id = tokenizer.eos_id() if tokenizer.eos_id() >= 0 else None
        self.pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else 0
        
    def tokenize_text(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Tokenize text using SentencePiece.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        # Encode text
        tokens = self.tokenizer.encode(text)
        
        # Add special tokens if requested
        if add_special_tokens:
            if self.bos_id is not None:
                tokens = [self.bos_id] + tokens
            if self.eos_id is not None:
                tokens = tokens + [self.eos_id]
        
        return tokens
    
    def calculate_surprisal(
        self,
        text: str,
        context: Optional[str] = None,
        hotspot: Optional[str] = None,
        return_tokens: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], List[str]]]:
        """
        Calculate surprisal for text, optionally with context.
        
        Args:
            text: Target text to calculate surprisal for
            context: Optional context preceding the target text
            hotspot: Optional specific word to measure surprisal at
            return_tokens: Whether to return token strings along with surprisal
            
        Returns:
            Dictionary with surprisal values (and optionally token list)
        """
        # Prepare full text
        if context:
            full_text = context + " " + text
            context_tokens = self.tokenize_text(context, add_special_tokens=True)
            context_len = len(context_tokens)
        else:
            full_text = text
            context_len = 0
        
        # Tokenize
        tokens = self.tokenize_text(full_text, add_special_tokens=True)
        token_strings = [self.tokenizer.decode([t]) for t in tokens]
        
        # Find hotspot position if specified
        hotspot_idx = None
        if hotspot:
            # Simple search for hotspot in token strings
            for i, tok_str in enumerate(token_strings):
                if hotspot.lower() in tok_str.lower():
                    hotspot_idx = i
                    break
        
        # Calculate surprisal
        with torch.no_grad():
            # Convert to tensor
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            # Get model outputs
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Calculate log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Get surprisal for each token (except first)
            surprisals = []
            for i in range(1, len(tokens)):
                token_id = tokens[i]
                # Surprisal = -log2(P(token|context))
                log_prob = log_probs[i-1, token_id].item()
                surprisal = -log_prob / np.log(2)  # Convert to log2
                surprisals.append(surprisal)
        
        # Compile results
        results = {
            "total_surprisal": sum(surprisals[context_len:]) if context else sum(surprisals),
            "mean_surprisal": np.mean(surprisals[context_len:]) if context else np.mean(surprisals),
            "tokens": len(tokens) - context_len if context else len(tokens),
        }
        
        # Add hotspot surprisal if found
        if hotspot_idx is not None and hotspot_idx > 0:
            results["hotspot_surprisal"] = surprisals[hotspot_idx - 1]
            results["hotspot_position"] = hotspot_idx
        
        # Add per-token surprisals for target text only
        if context:
            target_surprisals = surprisals[context_len:]
            target_tokens = token_strings[context_len + 1:]  # Skip one for alignment
        else:
            target_surprisals = surprisals
            target_tokens = token_strings[1:]  # Skip BOS token
        
        results["per_token_surprisal"] = list(zip(target_tokens, target_surprisals))
        
        if return_tokens:
            return results, token_strings
        return results
    
    def batch_calculate(
        self,
        texts: List[str],
        contexts: Optional[List[str]] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, float]]:
        """
        Calculate surprisal for multiple texts in batches.
        
        Args:
            texts: List of target texts
            contexts: Optional list of contexts (must match texts length)
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            List of surprisal dictionaries
        """
        if contexts and len(contexts) != len(texts):
            raise ValueError("Contexts list must match texts length")
        
        results = []
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Calculating surprisal")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size] if contexts else [None] * len(batch_texts)
            
            # Process each item in batch
            # Note: True batching would require padding and masking
            for text, context in zip(batch_texts, batch_contexts):
                result = self.calculate_surprisal(text, context)
                results.append(result)
        
        return results
    
    def compare_minimal_pair(
        self,
        good_sentence: str,
        bad_sentence: str,
        return_difference: bool = True
    ) -> Dict[str, float]:
        """
        Compare surprisal between a minimal pair (e.g., for BLIMP evaluation).
        
        Args:
            good_sentence: Grammatical sentence
            bad_sentence: Ungrammatical sentence
            return_difference: Whether to return the surprisal difference
            
        Returns:
            Dictionary with comparison results
        """
        good_result = self.calculate_surprisal(good_sentence)
        bad_result = self.calculate_surprisal(bad_sentence)
        
        results = {
            "good_surprisal": good_result["mean_surprisal"],
            "bad_surprisal": bad_result["mean_surprisal"],
            "good_total": good_result["total_surprisal"],
            "bad_total": bad_result["total_surprisal"],
            "correct": good_result["mean_surprisal"] < bad_result["mean_surprisal"]
        }
        
        if return_difference:
            results["surprisal_difference"] = (
                bad_result["mean_surprisal"] - good_result["mean_surprisal"]
            )
        
        return results
    
    def calculate_perplexity(
        self,
        texts: List[str],
        aggregate: bool = True
    ) -> Union[float, List[float]]:
        """
        Calculate perplexity for a list of texts.
        
        Args:
            texts: List of texts
            aggregate: Whether to return aggregate perplexity or per-text
            
        Returns:
            Perplexity value(s)
        """
        total_surprisal = 0
        total_tokens = 0
        per_text_perplexity = []
        
        for text in texts:
            result = self.calculate_surprisal(text)
            text_perplexity = 2 ** result["mean_surprisal"]
            per_text_perplexity.append(text_perplexity)
            
            if aggregate:
                total_surprisal += result["total_surprisal"]
                total_tokens += result["tokens"]
        
        if aggregate:
            # Calculate aggregate perplexity
            mean_surprisal = total_surprisal / total_tokens
            return 2 ** mean_surprisal
        else:
            return per_text_perplexity


class NullSubjectSurprisalCalculator(SurprisalCalculator):
    """Extended calculator for null-subject stimuli evaluation."""
    
    def evaluate_null_subject_pair(
        self,
        context: str,
        overt_target: str,
        null_target: str,
        hotspot: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a null-subject stimulus pair.
        
        Args:
            context: Context sentence
            overt_target: Target with overt pronoun
            null_target: Target without pronoun
            hotspot: Word to measure surprisal at
            
        Returns:
            Dictionary with evaluation results
        """
        # Calculate surprisal for both conditions
        overt_result = self.calculate_surprisal(overt_target, context, hotspot)
        null_result = self.calculate_surprisal(null_target, context, hotspot)
        
        results = {
            "overt_mean_surprisal": overt_result["mean_surprisal"],
            "null_mean_surprisal": null_result["mean_surprisal"],
            "overt_total_surprisal": overt_result["total_surprisal"],
            "null_total_surprisal": null_result["total_surprisal"],
            "prefers_overt": overt_result["mean_surprisal"] < null_result["mean_surprisal"],
            "surprisal_difference": null_result["mean_surprisal"] - overt_result["mean_surprisal"]
        }
        
        # Add hotspot surprisals if available
        if "hotspot_surprisal" in overt_result:
            results["overt_hotspot_surprisal"] = overt_result["hotspot_surprisal"]
        if "hotspot_surprisal" in null_result:
            results["null_hotspot_surprisal"] = null_result["hotspot_surprisal"]
            
        if "hotspot_surprisal" in overt_result and "hotspot_surprisal" in null_result:
            results["hotspot_difference"] = (
                null_result["hotspot_surprisal"] - overt_result["hotspot_surprisal"]
            )
        
        return results