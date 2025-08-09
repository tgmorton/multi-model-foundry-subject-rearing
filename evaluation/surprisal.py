"""
Surprisal Calculation for Subject Drop Evaluation

This script calculates surprisal values for minimal linguistic pairs to evaluate
how well models have learned subject-drop patterns in English.

Surprisal: S(w_i) = -log_2 P(w_i | w_1, ..., w_{i-1})
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import sys

# Add model_foundry to path for logging_utils import
sys.path.insert(0, str(Path(__file__).parent.parent / "model_foundry"))
from logging_utils import setup_logging


class SurprisalCalculator:
    """Calculate surprisal values for linguistic stimuli."""
    
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "auto"):
        """
        Initialize the surprisal calculator.
        
        Args:
            model_path: Path to the trained model checkpoint
            tokenizer_path: Path to the tokenizer
            device: Device to use for computation
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)
        
        # Load model and tokenizer
        print(f"Loading model from: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=False)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def calculate_surprisal(self, text: str, target_token: Optional[str] = None) -> float:
        """
        Calculate surprisal for a given text.
        
        Args:
            text: Input text
            target_token: Specific token to calculate surprisal for (if None, uses last token)
            
        Returns:
            Surprisal value in bits
        """
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get the target position
            if target_token is not None:
                # Find the target token in the sequence
                target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)
                if len(target_ids) > 0:
                    target_id = target_ids[0]  # Use first token if multi-token
                else:
                    target_id = inputs['input_ids'][0, -1]  # Fallback to last token
            else:
                # Use the last token
                target_id = inputs['input_ids'][0, -1]
                target_pos = -1
            
            # Get the position of the target token
            if target_token is not None:
                # Find the position of the target token
                input_ids = inputs['input_ids'][0]
                target_pos = None
                for i, token_id in enumerate(input_ids):
                    if token_id == target_id:
                        target_pos = i
                        break
                if target_pos is None:
                    target_pos = -1  # Fallback to last position
            else:
                target_pos = -1
            
            # Get logits for the target position
            if target_pos >= 0 and target_pos < logits.shape[1]:
                logits_at_pos = logits[0, target_pos, :]
                
                # Calculate probability distribution
                probs = F.softmax(logits_at_pos, dim=-1)
                
                # Get probability of the target token
                target_prob = probs[target_id].item()
                
                # Calculate surprisal: -log2(p)
                surprisal = -np.log2(target_prob) if target_prob > 0 else float('inf')
                
                return surprisal
            else:
                return float('inf')
    
    def calculate_surprisal_difference(self, preferred: str, dispreferred: str, 
                                     target_token: Optional[str] = None) -> float:
        """
        Calculate the difference in surprisal between preferred and dispreferred sentences.
        
        Args:
            preferred: The preferred (grammatical) sentence
            dispreferred: The dispreferred (ungrammatical) sentence
            target_token: Specific token to calculate surprisal for
            
        Returns:
            Surprisal difference (dispreferred - preferred)
        """
        surprisal_preferred = self.calculate_surprisal(preferred, target_token)
        surprisal_dispreferred = self.calculate_surprisal(dispreferred, target_token)
        
        return surprisal_dispreferred - surprisal_preferred
    
    def evaluate_stimuli_set(self, stimuli: List[Dict]) -> Dict:
        """
        Evaluate a set of linguistic stimuli.
        
        Args:
            stimuli: List of stimulus dictionaries with 'preferred' and 'dispreferred' fields
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            'stimuli': [],
            'summary': {
                'total_pairs': len(stimuli),
                'valid_pairs': 0,
                'mean_surprisal_diff': 0.0,
                'std_surprisal_diff': 0.0,
                'correct_predictions': 0
            }
        }
        
        surprisal_diffs = []
        
        for i, stimulus in enumerate(tqdm(stimuli, desc="Evaluating stimuli")):
            try:
                preferred = stimulus['preferred']
                dispreferred = stimulus['dispreferred']
                target_token = stimulus.get('target_token', None)
                
                # Calculate surprisal difference
                surprisal_diff = self.calculate_surprisal_difference(
                    preferred, dispreferred, target_token
                )
                
                # Check if prediction is correct (dispreferred should have higher surprisal)
                is_correct = surprisal_diff > 0
                
                result = {
                    'id': i,
                    'preferred': preferred,
                    'dispreferred': dispreferred,
                    'target_token': target_token,
                    'surprisal_preferred': self.calculate_surprisal(preferred, target_token),
                    'surprisal_dispreferred': self.calculate_surprisal(dispreferred, target_token),
                    'surprisal_difference': surprisal_diff,
                    'is_correct': is_correct
                }
                
                results['stimuli'].append(result)
                surprisal_diffs.append(surprisal_diff)
                
                if is_correct:
                    results['summary']['correct_predictions'] += 1
                
            except Exception as e:
                print(f"Error processing stimulus {i}: {e}")
                continue
        
        # Calculate summary statistics
        if surprisal_diffs:
            results['summary']['valid_pairs'] = len(surprisal_diffs)
            results['summary']['mean_surprisal_diff'] = np.mean(surprisal_diffs)
            results['summary']['std_surprisal_diff'] = np.std(surprisal_diffs)
            results['summary']['accuracy'] = results['summary']['correct_predictions'] / len(surprisal_diffs)
        
        return results


def load_stimuli(file_path: str) -> List[Dict]:
    """Load linguistic stimuli from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_results(results: Dict, output_path: str):
    """Save evaluation results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    """Main function for surprisal evaluation."""
    parser = argparse.ArgumentParser(description="Calculate surprisal for linguistic stimuli")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "tokenizer_path", 
        type=str,
        help="Path to the tokenizer"
    )
    parser.add_argument(
        "stimuli_path",
        type=str,
        help="Path to the stimuli JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="surprisal_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="surprisal_evaluation",
        help="Experiment name for logging"
    )
    
    args = parser.parse_args()
    
    # Set up unified logging
    logger = setup_logging("surprisal_evaluation", experiment=args.experiment_name, log_dir="logs")
    logger.info("=== Starting Surprisal Evaluation ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Tokenizer path: {args.tokenizer_path}")
    logger.info(f"Stimuli path: {args.stimuli_path}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Device: {args.device}")
    
    # Load stimuli
    logger.info(f"Loading stimuli from: {args.stimuli_path}")
    stimuli = load_stimuli(args.stimuli_path)
    
    # Initialize calculator
    calculator = SurprisalCalculator(args.model_path, args.tokenizer_path, args.device)
    
    # Evaluate stimuli
    logger.info("Evaluating stimuli...")
    results = calculator.evaluate_stimuli_set(stimuli)
    
    # Print summary
    summary = results['summary']
    print(f"\n=== Evaluation Summary ===")
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"Valid pairs: {summary['valid_pairs']}")
    print(f"Mean surprisal difference: {summary['mean_surprisal_diff']:.4f}")
    print(f"Std surprisal difference: {summary['std_surprisal_diff']:.4f}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Correct predictions: {summary['correct_predictions']}/{summary['valid_pairs']}")
    
    # Save results
    save_results(results, args.output)
    logger.info(f"Results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 