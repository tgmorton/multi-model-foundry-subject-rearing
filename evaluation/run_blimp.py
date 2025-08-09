"""
BLIMP Evaluation Script

This script evaluates trained models on the BLIMP (Broad-coverage Linguistic 
Minimal Pairs) benchmark to assess general linguistic performance.

BLIMP contains 67 minimal pairs across 17 linguistic phenomena.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import sys

# Add model_foundry to path for logging_utils import
sys.path.insert(0, str(Path(__file__).parent.parent / "model_foundry"))
from logging_utils import setup_logging


class BLIMPEvaluator:
    """Evaluate models on the BLIMP benchmark."""
    
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "auto"):
        """
        Initialize the BLIMP evaluator.
        
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
    
    def calculate_surprisal(self, text: str) -> float:
        """
        Calculate surprisal for a given text.
        
        Args:
            text: Input text
            
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
            
            # Use the last token for surprisal calculation
            last_logits = logits[0, -1, :]
            last_token_id = inputs['input_ids'][0, -1]
            
            # Calculate probability distribution
            probs = F.softmax(last_logits, dim=-1)
            
            # Get probability of the last token
            last_token_prob = probs[last_token_id].item()
            
            # Calculate surprisal: -log2(p)
            surprisal = -np.log2(last_token_prob) if last_token_prob > 0 else float('inf')
            
            return surprisal
    
    def evaluate_blimp_pair(self, sentence_good: str, sentence_bad: str) -> Dict:
        """
        Evaluate a single BLIMP pair.
        
        Args:
            sentence_good: The grammatical sentence
            sentence_bad: The ungrammatical sentence
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            surprisal_good = self.calculate_surprisal(sentence_good)
            surprisal_bad = self.calculate_surprisal(sentence_bad)
            
            surprisal_diff = surprisal_bad - surprisal_good
            is_correct = surprisal_diff > 0  # Bad sentence should have higher surprisal
            
            return {
                'sentence_good': sentence_good,
                'sentence_bad': sentence_bad,
                'surprisal_good': surprisal_good,
                'surprisal_bad': surprisal_bad,
                'surprisal_diff': surprisal_diff,
                'is_correct': is_correct
            }
        except Exception as e:
            return {
                'sentence_good': sentence_good,
                'sentence_bad': sentence_bad,
                'error': str(e),
                'is_correct': False
            }
    
    def evaluate_blimp_dataset(self, dataset_name: str = "blimp") -> Dict:
        """
        Evaluate the model on the full BLIMP dataset.
        
        Args:
            dataset_name: Name of the BLIMP dataset to use
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Loading BLIMP dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        
        results = {
            'phenomena': {},
            'summary': {
                'total_pairs': 0,
                'valid_pairs': 0,
                'correct_predictions': 0,
                'overall_accuracy': 0.0,
                'mean_surprisal_diff': 0.0,
                'std_surprisal_diff': 0.0
            }
        }
        
        all_surprisal_diffs = []
        
        # Group by linguistic phenomenon
        phenomena = {}
        for item in dataset['train']:
            phenomenon = item['linguistics_term']
            if phenomenon not in phenomena:
                phenomena[phenomenon] = []
            phenomena[phenomenon].append(item)
        
        print(f"Found {len(phenomena)} linguistic phenomena")
        
        # Evaluate each phenomenon
        for phenomenon, pairs in tqdm(phenomena.items(), desc="Evaluating phenomena"):
            phenomenon_results = {
                'pairs': [],
                'correct_predictions': 0,
                'total_pairs': len(pairs),
                'accuracy': 0.0,
                'mean_surprisal_diff': 0.0
            }
            
            phenomenon_surprisal_diffs = []
            
            for pair in pairs:
                result = self.evaluate_blimp_pair(
                    pair['sentence_good'], 
                    pair['sentence_bad']
                )
                
                phenomenon_results['pairs'].append(result)
                
                if 'error' not in result:
                    if result['is_correct']:
                        phenomenon_results['correct_predictions'] += 1
                        results['summary']['correct_predictions'] += 1
                    
                    phenomenon_surprisal_diffs.append(result['surprisal_diff'])
                    all_surprisal_diffs.append(result['surprisal_diff'])
                
                results['summary']['total_pairs'] += 1
            
            # Calculate phenomenon statistics
            if phenomenon_surprisal_diffs:
                phenomenon_results['valid_pairs'] = len(phenomenon_surprisal_diffs)
                phenomenon_results['accuracy'] = phenomenon_results['correct_predictions'] / len(phenomenon_surprisal_diffs)
                phenomenon_results['mean_surprisal_diff'] = np.mean(phenomenon_surprisal_diffs)
                results['summary']['valid_pairs'] += len(phenomenon_surprisal_diffs)
            
            results['phenomena'][phenomenon] = phenomenon_results
        
        # Calculate overall statistics
        if all_surprisal_diffs:
            results['summary']['overall_accuracy'] = results['summary']['correct_predictions'] / len(all_surprisal_diffs)
            results['summary']['mean_surprisal_diff'] = np.mean(all_surprisal_diffs)
            results['summary']['std_surprisal_diff'] = np.std(all_surprisal_diffs)
        
        return results
    
    def evaluate_custom_blimp_data(self, data_path: str) -> Dict:
        """
        Evaluate the model on custom BLIMP-formatted data.
        
        Args:
            data_path: Path to custom BLIMP data file
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Loading custom BLIMP data from: {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        results = {
            'pairs': [],
            'summary': {
                'total_pairs': len(data),
                'valid_pairs': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'mean_surprisal_diff': 0.0,
                'std_surprisal_diff': 0.0
            }
        }
        
        surprisal_diffs = []
        
        for i, pair in enumerate(tqdm(data, desc="Evaluating pairs")):
            result = self.evaluate_blimp_pair(
                pair['sentence_good'],
                pair['sentence_bad']
            )
            
            result['id'] = i
            results['pairs'].append(result)
            
            if 'error' not in result:
                if result['is_correct']:
                    results['summary']['correct_predictions'] += 1
                
                surprisal_diffs.append(result['surprisal_diff'])
                results['summary']['valid_pairs'] += 1
        
        # Calculate summary statistics
        if surprisal_diffs:
            results['summary']['accuracy'] = results['summary']['correct_predictions'] / len(surprisal_diffs)
            results['summary']['mean_surprisal_diff'] = np.mean(surprisal_diffs)
            results['summary']['std_surprisal_diff'] = np.std(surprisal_diffs)
        
        return results


def save_results(results: Dict, output_path: str):
    """Save evaluation results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def print_summary(results: Dict):
    """Print a summary of the evaluation results."""
    summary = results['summary']
    
    print(f"\n=== BLIMP Evaluation Summary ===")
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"Valid pairs: {summary['valid_pairs']}")
    print(f"Correct predictions: {summary['correct_predictions']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.2%}")
    print(f"Mean surprisal difference: {summary['mean_surprisal_diff']:.4f}")
    print(f"Std surprisal difference: {summary['std_surprisal_diff']:.4f}")
    
    if 'phenomena' in results:
        print(f"\n=== Performance by Linguistic Phenomenon ===")
        for phenomenon, phenom_results in results['phenomena'].items():
            accuracy = phenom_results.get('accuracy', 0.0)
            print(f"{phenomenon}: {accuracy:.2%} ({phenom_results['correct_predictions']}/{phenom_results['valid_pairs']})")


def main():
    """Main function for BLIMP evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model on BLIMP benchmark")
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
        "--output",
        type=str,
        default="blimp_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--custom-data",
        type=str,
        help="Path to custom BLIMP-formatted data file (optional)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="blimp",
        help="BLIMP dataset name (default: blimp)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="blimp_evaluation",
        help="Experiment name for logging"
    )
    
    args = parser.parse_args()
    
    # Set up unified logging
    logger = setup_logging("blimp_evaluation", experiment=args.experiment_name, log_dir="logs")
    logger.info("=== Starting BLIMP Evaluation ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Tokenizer path: {args.tokenizer_path}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Device: {args.device}")
    
    # Initialize evaluator
    evaluator = BLIMPEvaluator(args.model_path, args.tokenizer_path, args.device)
    
    # Run evaluation
    if args.custom_data:
        logger.info("Evaluating custom BLIMP data...")
        results = evaluator.evaluate_custom_blimp_data(args.custom_data)
    else:
        logger.info("Evaluating BLIMP benchmark...")
        results = evaluator.evaluate_blimp_dataset(args.dataset_name)
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output)
    logger.info(f"Results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 