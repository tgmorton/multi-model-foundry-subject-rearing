"""
Main evaluation runner script.
Orchestrates evaluation of language models on multiple tasks with config support.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
import yaml
import torch
from pydantic import BaseModel, Field
from datetime import datetime

from ..core.model_loader import ModelLoader, clear_gpu_cache
from ..core.surprisal_calculator import SurprisalCalculator, NullSubjectSurprisalCalculator
from ..evaluators.blimp_evaluator import BLIMPEvaluator
from ..evaluators.null_subject_evaluator import NullSubjectEvaluator
from ..evaluators.perplexity_evaluator import PerplexityEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation pipeline."""
    
    # Model paths
    model_checkpoint_dir: str = Field(..., description="Directory containing model checkpoints")
    tokenizer_path: str = Field(..., description="Path to tokenizer")
    
    # Test datasets
    test_corpus: Optional[str] = Field(None, description="Path to test corpus")
    blimp_dir: Optional[str] = Field(None, description="Path to BLIMP stimuli directory")
    null_subject_dir: Optional[str] = Field(None, description="Path to null-subject stimuli directory")
    
    # Evaluation settings
    batch_size: int = Field(32, description="Batch size for evaluation")
    device: str = Field("auto", description="Device to use (cuda/cpu/auto)")
    use_fp16: bool = Field(False, description="Use mixed precision")
    max_length: int = Field(1000, description="Maximum sequence length")
    
    # Analysis options
    run_perplexity: bool = Field(True, description="Calculate perplexity")
    run_blimp: bool = Field(True, description="Evaluate BLIMP")
    run_null_subject: bool = Field(True, description="Evaluate null-subject stimuli")
    
    # Speed optimizations
    num_workers: int = Field(4, description="Number of data loading workers")
    pin_memory: bool = Field(True, description="Pin memory for faster GPU transfer")
    prefetch_factor: int = Field(2, description="Prefetch factor for data loading")
    
    # Testing/debugging options
    max_samples: Optional[int] = Field(None, description="Maximum samples per evaluation (for testing)")
    perplexity_sample_ratio: Optional[float] = Field(None, description="Fraction of data to sample for perplexity (e.g., 0.1 for 10%)")
    max_checkpoints: Optional[int] = Field(None, description="Maximum checkpoints to evaluate")
    
    # Output configuration
    output_dir: str = Field("evaluation/results", description="Output directory")
    save_detailed: bool = Field(True, description="Save detailed per-item results")
    save_format: str = Field("jsonl", description="Output format (jsonl/csv)")


class EvaluationRunner:
    """Main evaluation pipeline runner."""
    
    def __init__(self, config: EvaluationConfig, force_rerun: bool = False):
        """
        Initialize the evaluation runner.
        
        Args:
            config: Evaluation configuration
            force_rerun: If True, re-evaluate checkpoints even if results exist
        """
        self.config = config
        self.force_rerun = force_rerun
        self.results = {}
        
        # Set up output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model loader
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loader = ModelLoader(device=device)
    
    def load_model_and_tokenizer(self, checkpoint_path: str):
        """
        Load model and tokenizer for evaluation.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model from {checkpoint_path}")
        
        model, tokenizer = self.model_loader.load_model_and_tokenizer(
            checkpoint_path=checkpoint_path,
            tokenizer_path=self.config.tokenizer_path,
            use_fp16=self.config.use_fp16
        )
        
        return model, tokenizer
    
    def run_perplexity_evaluation(self, model, tokenizer, checkpoint_name: str) -> Dict:
        """Run perplexity evaluation."""
        if not self.config.run_perplexity or not self.config.test_corpus:
            return {}
        
        logger.info("Running perplexity evaluation...")
        
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
        
        # Save detailed results if requested
        if self.config.save_detailed:
            output_file = self.output_dir / f"{checkpoint_name}_perplexity.json"
            with open(output_file, 'w') as f:
                safe_results = self._make_json_serializable(results)
                json.dump(safe_results, f, indent=2)
        
        return results
    
    def run_blimp_evaluation(self, model, tokenizer, checkpoint_name: str) -> Dict:
        """Run BLIMP evaluation."""
        if not self.config.run_blimp or not self.config.blimp_dir:
            return {}
        
        logger.info("Running BLIMP evaluation...")
        
        # Set up calculators and evaluator
        surprisal_calc = SurprisalCalculator(model, tokenizer, self.model_loader.device)
        evaluator = BLIMPEvaluator(surprisal_calc)
        
        # Run evaluation
        detailed_results = evaluator.evaluate_all(
            blimp_dir=self.config.blimp_dir,
            max_samples_per_file=self.config.max_samples
        )
        
        # Save detailed results if requested
        if self.config.save_detailed:
            output_file = self.output_dir / f"{checkpoint_name}_blimp_detailed.jsonl"
            detailed_results.to_json(output_file, orient='records', lines=True)
        
        # Get summary statistics
        summary = evaluator.get_summary_statistics(detailed_results)
        
        return summary
    
    def run_null_subject_evaluation(self, model, tokenizer, checkpoint_name: str) -> Dict:
        """Run null-subject evaluation."""
        if not self.config.run_null_subject or not self.config.null_subject_dir:
            return {}
        
        logger.info("Running null-subject evaluation...")
        
        # Set up calculators and evaluator
        surprisal_calc = NullSubjectSurprisalCalculator(model, tokenizer, self.model_loader.device)
        evaluator = NullSubjectEvaluator(surprisal_calc)
        
        # Run evaluation - always use all samples for null-subject
        detailed_results = evaluator.evaluate_all(
            stimuli_dir=self.config.null_subject_dir,
            max_items_per_file=None  # Always use all samples
        )
        
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
    
    def save_task_summaries(self, all_results: list):
        """
        Save individual task summaries with detailed statistics.
        
        Args:
            all_results: List of per-checkpoint results
        """
        import numpy as np
        
        # Filter out skipped results
        valid_results = [r for r in all_results if 'skipped' not in r]
        
        if not valid_results:
            logger.warning("No valid results to summarize")
            return
        
        # Perplexity summary
        if self.config.run_perplexity:
            perplexities = []
            checkpoint_names = []
            for result in valid_results:
                if 'perplexity' in result and result['perplexity'].get('perplexity'):
                    perplexities.append(result['perplexity']['perplexity'])
                    checkpoint_names.append(result['checkpoint'])
            
            if perplexities:
                perp_array = np.array(perplexities)
                perp_summary = {
                    'task': 'perplexity',
                    'n_checkpoints': len(perplexities),
                    'checkpoints': checkpoint_names,
                    'values': perplexities,
                    'statistics': {
                        'mean': float(np.mean(perp_array)),
                        'std': float(np.std(perp_array, ddof=1)) if len(perp_array) > 1 else 0.0,
                        'se': float(np.std(perp_array, ddof=1) / np.sqrt(len(perp_array))) if len(perp_array) > 1 else 0.0,
                        'min': float(np.min(perp_array)),
                        'max': float(np.max(perp_array)),
                        'median': float(np.median(perp_array)),
                        'initial': perplexities[0] if perplexities else None,
                        'final': perplexities[-1] if perplexities else None
                    }
                }
                
                # Save perplexity summary
                perp_file = self.output_dir / "perplexity_summary.json"
                with open(perp_file, 'w') as f:
                    json.dump(perp_summary, f, indent=2)
                logger.info(f"Saved perplexity summary to {perp_file}")
        
        # BLIMP summary
        if self.config.run_blimp:
            accuracies = []
            checkpoint_names = []
            for result in valid_results:
                if 'blimp' in result and result['blimp'].get('overall_accuracy'):
                    accuracies.append(result['blimp']['overall_accuracy'])
                    checkpoint_names.append(result['checkpoint'])
            
            if accuracies:
                acc_array = np.array(accuracies)
                blimp_summary = {
                    'task': 'blimp',
                    'n_checkpoints': len(accuracies),
                    'checkpoints': checkpoint_names,
                    'values': accuracies,
                    'statistics': {
                        'mean': float(np.mean(acc_array)),
                        'std': float(np.std(acc_array, ddof=1)) if len(acc_array) > 1 else 0.0,
                        'se': float(np.std(acc_array, ddof=1) / np.sqrt(len(acc_array))) if len(acc_array) > 1 else 0.0,
                        'min': float(np.min(acc_array)),
                        'max': float(np.max(acc_array)),
                        'median': float(np.median(acc_array)),
                        'initial': accuracies[0] if accuracies else None,
                        'final': accuracies[-1] if accuracies else None
                    }
                }
                
                # Save BLIMP summary
                blimp_file = self.output_dir / "blimp_summary.json"
                with open(blimp_file, 'w') as f:
                    json.dump(blimp_summary, f, indent=2)
                logger.info(f"Saved BLIMP summary to {blimp_file}")
        
        # Null-subject summary
        if self.config.run_null_subject:
            # Extract null-subject accuracies (could be overall or specific metrics)
            null_subj_data = []
            checkpoint_names = []
            for result in valid_results:
                if 'null_subject' in result:
                    # Try to get overall accuracy or a representative metric
                    ns_result = result['null_subject']
                    if 'overall_accuracy' in ns_result:
                        null_subj_data.append(ns_result['overall_accuracy'])
                        checkpoint_names.append(result['checkpoint'])
                    elif 'accuracy' in ns_result:
                        null_subj_data.append(ns_result['accuracy'])
                        checkpoint_names.append(result['checkpoint'])
            
            if null_subj_data:
                ns_array = np.array(null_subj_data)
                ns_summary = {
                    'task': 'null_subject',
                    'n_checkpoints': len(null_subj_data),
                    'checkpoints': checkpoint_names,
                    'values': null_subj_data,
                    'statistics': {
                        'mean': float(np.mean(ns_array)),
                        'std': float(np.std(ns_array, ddof=1)) if len(ns_array) > 1 else 0.0,
                        'se': float(np.std(ns_array, ddof=1) / np.sqrt(len(ns_array))) if len(ns_array) > 1 else 0.0,
                        'min': float(np.min(ns_array)),
                        'max': float(np.max(ns_array)),
                        'median': float(np.median(ns_array)),
                        'initial': null_subj_data[0] if null_subj_data else None,
                        'final': null_subj_data[-1] if null_subj_data else None
                    }
                }
                
                # Save null-subject summary
                ns_file = self.output_dir / "null_subject_summary.json"
                with open(ns_file, 'w') as f:
                    json.dump(ns_summary, f, indent=2)
                logger.info(f"Saved null-subject summary to {ns_file}")
    
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
        Evaluate a single checkpoint.
        
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
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer(checkpoint_path)
        
        # Initialize results
        results = {
            'checkpoint': checkpoint_name,
            'checkpoint_path': str(checkpoint_path),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Run evaluations
            if self.config.run_perplexity:
                results['perplexity'] = self.run_perplexity_evaluation(model, tokenizer, checkpoint_name)
            
            if self.config.run_blimp:
                results['blimp'] = self.run_blimp_evaluation(model, tokenizer, checkpoint_name)
            
            if self.config.run_null_subject:
                results['null_subject'] = self.run_null_subject_evaluation(model, tokenizer, checkpoint_name)
            
        except Exception as e:
            logger.error(f"Error evaluating checkpoint {checkpoint_name}: {e}")
            results['error'] = str(e)
        
        finally:
            # Clean up GPU memory
            if hasattr(model, 'cpu'):
                model.cpu()
            del model, tokenizer
            clear_gpu_cache()
        
        return results
    
    def run(self) -> Dict:
        """
        Run the full evaluation pipeline.
        
        Returns:
            Dictionary with all results
        """
        logger.info("Starting evaluation pipeline...")
        
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
        
        # Generate and save individual task summaries
        self.save_task_summaries(all_results)
        
        # Generate comprehensive summaries
        from .summary_generator import SummaryGenerator
        from .item_level_aggregator import ItemLevelAggregator
        
        experiment_name = self.output_dir.name
        SummaryGenerator.save_comprehensive_summaries(self.output_dir, experiment_name)
        
        # Generate item-level datasets for mixed effects models
        ItemLevelAggregator.create_mixed_effects_datasets(self.output_dir)
        
        logger.info(f"Evaluation complete. Results saved to {output_file}")
        
        return {'results': all_results, 'summary': self.generate_summary(all_results)}
    
    def _make_json_serializable(self, obj):
        """
        Recursively convert any tuple keys or non-serializable values to JSON-safe format.
        """
        if isinstance(obj, dict):
            return {
                str(key) if isinstance(key, tuple) else key: 
                self._make_json_serializable(value) 
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return str(obj)  # Convert tuples to strings
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Convert complex objects to strings
        else:
            return obj
    
    def generate_summary(self, all_results: list) -> Dict:
        """
        Generate summary statistics across all checkpoints.
        
        Args:
            all_results: List of per-checkpoint results
            
        Returns:
            Summary statistics
        """
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


def load_config(config_path: str) -> EvaluationConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Handle nested evaluation config
    if 'evaluation' in config_dict:
        config_dict = config_dict['evaluation']
    
    return EvaluationConfig(**config_dict)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Evaluate language models")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint", help="Evaluate specific checkpoint only")
    parser.add_argument("--force-rerun", action="store_true", help="Re-evaluate checkpoints even if results exist")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override for single checkpoint evaluation
    if args.checkpoint:
        config.model_checkpoint_dir = args.checkpoint
        config.max_checkpoints = 1
    
    # Run evaluation
    runner = EvaluationRunner(config, force_rerun=args.force_rerun)
    results = runner.run()
    
    print(f"Evaluation complete. Results saved to {runner.output_dir}")


if __name__ == "__main__":
    main()