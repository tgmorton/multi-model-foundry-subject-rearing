"""
Perplexity evaluation module for test corpus assessment.
Supports streaming evaluation for large datasets.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import sentencepiece as spm
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """Calculate perplexity on test corpora."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: spm.SentencePieceProcessor,
        device: str = None,
        max_length: int = 1000
    ):
        """
        Initialize perplexity evaluator.
        
        Args:
            model: Language model
            tokenizer: SentencePiece tokenizer
            device: Device for computation
            max_length: Maximum sequence length for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        # Get special tokens
        self.bos_id = tokenizer.bos_id() if tokenizer.bos_id() >= 0 else None
        self.eos_id = tokenizer.eos_id() if tokenizer.eos_id() >= 0 else None
    
    def read_corpus_streaming(
        self, 
        corpus_path: str, 
        max_samples: Optional[int] = None
    ) -> Iterator[str]:
        """
        Stream text from a corpus file.
        
        Args:
            corpus_path: Path to text file
            max_samples: Maximum number of lines to read
            
        Yields:
            Individual lines of text
        """
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if line:  # Skip empty lines
                    yield line
    
    def read_corpus_directory(
        self,
        corpus_dir: str,
        max_files: Optional[int] = None,
        max_samples_per_file: Optional[int] = None
    ) -> Iterator[str]:
        """
        Stream text from all files in a directory.
        
        Args:
            corpus_dir: Directory containing text files
            max_files: Maximum number of files to process
            max_samples_per_file: Maximum samples per file
            
        Yields:
            Individual lines of text
        """
        corpus_dir = Path(corpus_dir)
        
        # Get all text files
        text_files = []
        for pattern in ['*.train', '*.test', '*.txt']:
            text_files.extend(corpus_dir.glob(pattern))
        
        text_files = sorted(text_files)[:max_files] if max_files else sorted(text_files)
        
        logger.info(f"Processing {len(text_files)} files from {corpus_dir}")
        
        for filepath in text_files:
            logger.info(f"Reading {filepath.name}")
            yield from self.read_corpus_streaming(filepath, max_samples_per_file)
    
    def calculate_sequence_perplexity(self, text: str) -> Dict[str, float]:
        """
        Calculate perplexity for a single text sequence.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with perplexity metrics
        """
        # Tokenize
        tokens = self.tokenizer.encode(text)
        if self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if self.eos_id is not None:
            tokens = tokens + [self.eos_id]
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        if len(tokens) < 2:
            return {'perplexity': float('inf'), 'tokens': 0, 'total_log_prob': 0.0}
        
        # Calculate log probabilities
        with torch.no_grad():
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Get log probabilities for actual tokens
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Calculate negative log likelihood (excluding first token)
            total_log_prob = 0.0
            for i in range(1, len(tokens)):
                token_id = tokens[i]
                log_prob = log_probs[i-1, token_id].item()
                total_log_prob += log_prob
            
            # Calculate perplexity
            num_tokens = len(tokens) - 1  # Exclude first token from count
            avg_log_prob = total_log_prob / num_tokens
            perplexity = np.exp(-avg_log_prob)
        
        return {
            'perplexity': perplexity,
            'tokens': num_tokens,
            'total_log_prob': total_log_prob,
            'avg_log_prob': avg_log_prob
        }
    
    def calculate_corpus_perplexity(
        self,
        corpus_source: str,
        is_directory: bool = True,
        max_samples: Optional[int] = None,
        batch_size: int = 1,
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Calculate perplexity over an entire corpus.
        
        Args:
            corpus_source: Path to corpus file or directory
            is_directory: Whether source is a directory or single file
            max_samples: Maximum number of samples to evaluate
            batch_size: Number of sequences to process at once (currently only 1 supported)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with aggregate perplexity metrics
        """
        if batch_size != 1:
            logger.warning("Batch processing not yet implemented, using batch_size=1")
        
        # Set up text iterator
        if is_directory:
            text_iter = self.read_corpus_directory(corpus_source)
        else:
            text_iter = self.read_corpus_streaming(corpus_source, max_samples)
        
        # Process sequences
        total_log_prob = 0.0
        total_tokens = 0
        num_sequences = 0
        perplexities = []
        
        # Add progress bar if requested
        if show_progress and max_samples:
            text_iter = tqdm(text_iter, total=max_samples, desc="Calculating perplexity")
        elif show_progress:
            text_iter = tqdm(text_iter, desc="Calculating perplexity")
        
        for text in text_iter:
            if max_samples and num_sequences >= max_samples:
                break
                
            result = self.calculate_sequence_perplexity(text)
            
            if result['tokens'] > 0:  # Skip empty sequences
                total_log_prob += result['total_log_prob']
                total_tokens += result['tokens']
                perplexities.append(result['perplexity'])
                num_sequences += 1
        
        if total_tokens == 0:
            logger.warning("No valid tokens found in corpus")
            return {
                'perplexity': float('inf'),
                'num_sequences': 0,
                'total_tokens': 0
            }
        
        # Calculate aggregate metrics
        avg_log_prob = total_log_prob / total_tokens
        aggregate_perplexity = np.exp(-avg_log_prob)
        
        results = {
            'perplexity': aggregate_perplexity,
            'num_sequences': num_sequences,
            'total_tokens': total_tokens,
            'avg_log_prob': avg_log_prob,
            'total_log_prob': total_log_prob,
            'mean_sequence_perplexity': np.mean(perplexities),
            'median_sequence_perplexity': np.median(perplexities),
            'std_sequence_perplexity': np.std(perplexities)
        }
        
        logger.info(
            f"Perplexity evaluation complete: "
            f"PPL={results['perplexity']:.2f}, "
            f"sequences={num_sequences}, "
            f"tokens={total_tokens}"
        )
        
        return results
    
    def evaluate_by_domain(
        self,
        corpus_dir: str,
        max_samples_per_domain: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Calculate perplexity separately for each domain in the corpus.
        
        Args:
            corpus_dir: Directory with domain-specific files
            max_samples_per_domain: Maximum samples per domain file
            
        Returns:
            Dictionary mapping domain names to perplexity results
        """
        corpus_dir = Path(corpus_dir)
        
        # Find all corpus files
        corpus_files = []
        for pattern in ['*.train', '*.test', '*.txt']:
            corpus_files.extend(corpus_dir.glob(pattern))
        
        results = {}
        
        for filepath in sorted(corpus_files):
            domain_name = filepath.stem
            logger.info(f"Evaluating domain: {domain_name}")
            
            domain_result = self.calculate_corpus_perplexity(
                str(filepath),
                is_directory=False,
                max_samples=max_samples_per_domain,
                show_progress=False
            )
            
            results[domain_name] = domain_result
            logger.info(f"{domain_name}: PPL={domain_result['perplexity']:.2f}")
        
        return results