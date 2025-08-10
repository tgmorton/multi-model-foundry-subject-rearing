"""
Parallel evaluation runner for faster multi-GPU evaluation.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, List
import yaml
import torch
from datetime import datetime
import time

from .model_loader import ModelLoader, clear_gpu_cache
from .surprisal_calculator import SurprisalCalculator, NullSubjectSurprisalCalculator
from .parallel_blimp_evaluator import ParallelBLIMPEvaluator
from .null_subject_evaluator import NullSubjectEvaluator
from .perplexity_evaluator import PerplexityEvaluator
from .evaluation_runner import EvaluationConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParallelEvaluationRunner:
    """Parallel evaluation pipeline runner."""
    
    def __init__(self, config: EvaluationConfig, parallel_workers: int = 4, gpu_ids: Optional[List[int]] = None, force_rerun: bool = False):
        """
        Initialize the parallel evaluation runner.
        
        Args:
            config: Evaluation configuration
            parallel_workers: Number of parallel workers for BLIMP evaluation
            gpu_ids: List of GPU IDs to use (e.g., [0, 1])
            force_rerun: If True, re-evaluate checkpoints even if results exist
        """
        self.config = config
        self.parallel_workers = parallel_workers
        self.gpu_ids = gpu_ids or [0]
        self.force_rerun = force_rerun
        self.results = {}
        
        # Set up output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model loader for non-parallel tasks
        device = config.device
        if device == "auto":
            device = f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
        self.model_loader = ModelLoader(device=device)
    
    def run_perplexity_evaluation(self, checkpoint_path: str, checkpoint_name: str) -> Dict:
        """Run perplexity evaluation (single GPU)."""
        if not self.config.run_perplexity or not self.config.test_corpus:
            return {}
        
        logger.info("Running perplexity evaluation...")
        
        # Load model for perplexity
        model, tokenizer = self.model_loader.load_model_and_tokenizer(
            checkpoint_path=checkpoint_path,
            tokenizer_path=self.config.tokenizer_path,
            use_fp16=self.config.use_fp16
        )
        
        evaluator = PerplexityEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=self.model_loader.device,
            max_length=self.config.max_length
        )
        
        # Determine if test corpus is directory or file
        corpus_path = Path(self.config.test_corpus)
        is_directory = corpus_path.is_dir()
        
        # Calculate max_samples based on sample ratio if provided
        max_samples = self.config.max_samples
        if self.config.perplexity_sample_ratio is not None and max_samples is None:
            # Count total lines/samples in corpus
            if is_directory:
                total_samples = 0
                for pattern in ['*.train', '*.test', '*.txt']:
                    for filepath in corpus_path.glob(pattern):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            total_samples += sum(1 for line in f if line.strip())
            else:
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    total_samples = sum(1 for line in f if line.strip())
            
            max_samples = int(total_samples * self.config.perplexity_sample_ratio)
            logger.info(f"Sampling {max_samples} out of {total_samples} total samples ({self.config.perplexity_sample_ratio:.1%})")
        
        results = evaluator.calculate_corpus_perplexity(
            corpus_source=str(corpus_path),
            is_directory=is_directory,
            max_samples=max_samples
        )
        
        # Clean up
        del model, tokenizer
        clear_gpu_cache()
        
        # Save detailed results if requested
        if self.config.save_detailed:
            output_file = self.output_dir / f"{checkpoint_name}_perplexity.json"
            with open(output_file, 'w') as f:
                safe_results = self._make_json_serializable(results)
                json.dump(safe_results, f, indent=2)
        
        return results
    
    def run_parallel_blimp_evaluation(self, checkpoint_path: str, checkpoint_name: str) -> Dict:
        """Run BLIMP evaluation in parallel across multiple GPUs."""
        if not self.config.run_blimp or not self.config.blimp_dir:
            return {}
        
        logger.info(f"Running parallel BLIMP evaluation with {self.parallel_workers} workers...")
        
        # Create parallel evaluator
        evaluator = ParallelBLIMPEvaluator(
            model_checkpoint=checkpoint_path,
            tokenizer_path=self.config.tokenizer_path,
            num_workers=self.parallel_workers,
            device_ids=self.gpu_ids,
            use_fp16=self.config.use_fp16,
            batch_size=self.config.batch_size
        )
        
        # Run evaluation
        start_time = time.time()
        detailed_results = evaluator.evaluate_all(
            blimp_dir=self.config.blimp_dir,
            max_samples_per_file=self.config.max_samples
        )
        elapsed_time = time.time() - start_time
        
        logger.info(f"BLIMP evaluation completed in {elapsed_time:.2f} seconds")
        
        # Save detailed results if requested
        if self.config.save_detailed and not detailed_results.empty:
            output_file = self.output_dir / f"{checkpoint_name}_blimp_detailed.jsonl"
            detailed_results.to_json(output_file, orient='records', lines=True)
        
        # Get summary statistics
        summary = evaluator.get_summary_statistics(detailed_results)
        summary['evaluation_time'] = elapsed_time
        
        return summary
    
    def run_null_subject_evaluation(self, checkpoint_path: str, checkpoint_name: str) -> Dict:
        """Run null-subject evaluation (single GPU for now)."""
        if not self.config.run_null_subject or not self.config.null_subject_dir:
            return {}
        
        logger.info("Running null-subject evaluation...")
        
        # Load model
        model, tokenizer = self.model_loader.load_model_and_tokenizer(
            checkpoint_path=checkpoint_path,
            tokenizer_path=self.config.tokenizer_path,
            use_fp16=self.config.use_fp16
        )
        
        # Set up calculators and evaluator
        surprisal_calc = NullSubjectSurprisalCalculator(model, tokenizer, self.model_loader.device)
        evaluator = NullSubjectEvaluator(surprisal_calc)
        
        # Run evaluation
        detailed_results = evaluator.evaluate_all(
            stimuli_dir=self.config.null_subject_dir,
            max_items_per_file=self.config.max_samples
        )
        
        # Clean up
        del model, tokenizer
        clear_gpu_cache()
        
        # Save detailed results if requested
        if self.config.save_detailed:
            output_file = self.output_dir / f"{checkpoint_name}_null_subject_detailed.jsonl"
            # Convert any non-JSON-serializable values in detailed results
            detailed_clean = detailed_results.copy()
            for col in detailed_clean.columns:
                if detailed_clean[col].dtype == 'object':
                    detailed_clean[col] = detailed_clean[col].apply(lambda x: str(x) if isinstance(x, tuple) else x)
            detailed_clean.to_json(output_file, orient='records', lines=True)
        
        # Get summary statistics
        summary = evaluator.get_summary_statistics(detailed_results)
        
        # Add linguistic analysis
        if len(detailed_results) > 0:
            linguistic_analysis = evaluator.analyze_by_person_number(detailed_results)
            summary['linguistic_analysis'] = linguistic_analysis
        
        return summary
    
    def checkpoint_has_results(self, checkpoint_name: str) -> bool:
        """
        Check if a checkpoint already has evaluation results.
        
        Args:
            checkpoint_name: Name of the checkpoint
            
        Returns:
            True if all requested evaluations exist for this checkpoint
        """
        existing_files = []
        missing_files = []
        
        # Check for each type of evaluation result
        if self.config.run_perplexity:
            perplexity_file = self.output_dir / f"{checkpoint_name}_perplexity.json"
            if perplexity_file.exists():
                existing_files.append("perplexity")
            else:
                missing_files.append("perplexity")
        
        if self.config.run_blimp:
            blimp_file = self.output_dir / f"{checkpoint_name}_blimp_detailed.jsonl"
            if blimp_file.exists():
                existing_files.append("blimp")
            else:
                missing_files.append("blimp")
        
        if self.config.run_null_subject:
            null_subject_file = self.output_dir / f"{checkpoint_name}_null_subject_detailed.jsonl"
            if null_subject_file.exists():
                existing_files.append("null_subject")
            else:
                missing_files.append("null_subject")
        
        # All requested evaluations must exist
        has_all_results = len(missing_files) == 0 and len(existing_files) > 0
        
        if has_all_results:
            logger.info(f"Checkpoint {checkpoint_name} already has results for: {', '.join(existing_files)}")
        elif existing_files:
            logger.info(f"Checkpoint {checkpoint_name} has partial results. Existing: {', '.join(existing_files)}, Missing: {', '.join(missing_files)}")
        
        return has_all_results
    
    def evaluate_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Evaluate a single checkpoint with parallel BLIMP processing.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Dictionary with all evaluation results
        """
        checkpoint_name = Path(checkpoint_path).name
        
        # Check if checkpoint already has results
        if not self.force_rerun and self.checkpoint_has_results(checkpoint_name):
            logger.info(f"Skipping checkpoint {checkpoint_name} - results already exist (use --force-rerun to override)")
            return {
                'checkpoint': checkpoint_name,
                'checkpoint_path': str(checkpoint_path),
                'skipped': True,
                'reason': 'Results already exist'
            }
        
        logger.info(f"Evaluating checkpoint: {checkpoint_name}")
        
        # Initialize results
        results = {
            'checkpoint': checkpoint_name,
            'checkpoint_path': str(checkpoint_path),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Run evaluations
            if self.config.run_perplexity:
                results['perplexity'] = self.run_perplexity_evaluation(checkpoint_path, checkpoint_name)
            
            if self.config.run_blimp:
                results['blimp'] = self.run_parallel_blimp_evaluation(checkpoint_path, checkpoint_name)
            
            if self.config.run_null_subject:
                results['null_subject'] = self.run_null_subject_evaluation(checkpoint_path, checkpoint_name)
            
        except Exception as e:
            logger.error(f"Error evaluating checkpoint {checkpoint_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def run(self) -> Dict:
        """
        Run the full evaluation pipeline with parallel processing.
        
        Returns:
            Dictionary with all results
        """
        logger.info(f"Starting parallel evaluation pipeline with {self.parallel_workers} workers...")
        
        # Find checkpoints
        checkpoints = self.model_loader.find_checkpoints(self.config.model_checkpoint_dir)
        
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {self.config.model_checkpoint_dir}")
        
        if self.config.max_checkpoints:
            checkpoints = checkpoints[:self.config.max_checkpoints]
        
        logger.info(f"Found {len(checkpoints)} checkpoints to evaluate")
        
        # Evaluate each checkpoint
        all_results = []
        for checkpoint_path in checkpoints:
            result = self.evaluate_checkpoint(checkpoint_path)
            all_results.append(result)
        
        # Save aggregated results
        output_file = self.output_dir / "evaluation_results.jsonl"
        with open(output_file, 'w') as f:
            for result in all_results:
                # Ensure JSON serializable
                safe_result = self._make_json_serializable(result)
                json.dump(safe_result, f)
                f.write('\n')
        
        logger.info(f"Evaluation complete. Results saved to {output_file}")
        
        return {'results': all_results, 'summary': self.generate_summary(all_results)}
    
    def _make_json_serializable(self, obj):
        """Recursively convert any tuple keys or non-serializable values to JSON-safe format."""
        if isinstance(obj, dict):
            return {
                str(key) if isinstance(key, tuple) else key: 
                self._make_json_serializable(value) 
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def generate_summary(self, all_results: list) -> Dict:
        """Generate summary statistics across all checkpoints."""
        summary = {
            'total_checkpoints': len(all_results),
            'successful_evaluations': len([r for r in all_results if 'error' not in r])
        }
        
        # Aggregate metrics across checkpoints
        if self.config.run_perplexity:
            perplexities = [r['perplexity'].get('perplexity') for r in all_results 
                          if 'perplexity' in r and r['perplexity'].get('perplexity')]
            if perplexities:
                summary['perplexity_trend'] = {
                    'initial': perplexities[0],
                    'final': perplexities[-1],
                    'best': min(perplexities),
                    'improvement': perplexities[0] - perplexities[-1]
                }
        
        if self.config.run_blimp:
            accuracies = [r['blimp'].get('overall_accuracy') for r in all_results
                         if 'blimp' in r and r['blimp'].get('overall_accuracy')]
            if accuracies:
                summary['blimp_trend'] = {
                    'initial': accuracies[0],
                    'final': accuracies[-1],
                    'best': max(accuracies),
                    'improvement': accuracies[-1] - accuracies[0]
                }
        
        return summary


def main():
    """Command-line interface for parallel evaluation."""
    parser = argparse.ArgumentParser(description="Parallel evaluation of language models")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint", help="Evaluate specific checkpoint only")
    parser.add_argument("--parallel-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--gpu-ids", nargs='+', type=int, help="GPU IDs to use (e.g., 0 1)")
    parser.add_argument("--force-rerun", action="store_true", help="Re-evaluate checkpoints even if results exist")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Handle nested evaluation config
    if 'evaluation' in config_dict:
        config_dict = config_dict['evaluation']
    
    config = EvaluationConfig(**config_dict)
    
    # Override for single checkpoint evaluation
    if args.checkpoint:
        config.model_checkpoint_dir = args.checkpoint
        config.max_checkpoints = 1
    
    # Run parallel evaluation
    runner = ParallelEvaluationRunner(
        config=config,
        parallel_workers=args.parallel_workers,
        gpu_ids=args.gpu_ids or [0, 1],  # Default to using both GPUs
        force_rerun=args.force_rerun
    )
    results = runner.run()
    
    print(f"Evaluation complete. Results saved to {runner.output_dir}")


if __name__ == "__main__":
    main()