"""
Parallel BLIMP evaluation module for faster processing using multiple GPUs/processes.
"""

import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch
import os

from .model_loader import ModelLoader
from .surprisal_calculator import SurprisalCalculator
from .blimp_evaluator import BLIMPEvaluator

logger = logging.getLogger(__name__)


class ParallelBLIMPEvaluator:
    """Parallel BLIMP evaluator that distributes tasks across multiple processes."""
    
    def __init__(
        self,
        model_checkpoint: str,
        tokenizer_path: str,
        num_workers: int = 4,
        device_ids: Optional[List[int]] = None,
        use_fp16: bool = False,
        batch_size: int = 32
    ):
        """
        Initialize parallel BLIMP evaluator.
        
        Args:
            model_checkpoint: Path to model checkpoint
            tokenizer_path: Path to tokenizer
            num_workers: Number of parallel workers
            device_ids: List of GPU device IDs to use (e.g., [0, 1])
            use_fp16: Use mixed precision
            batch_size: Batch size for evaluation
        """
        self.model_checkpoint = model_checkpoint
        self.tokenizer_path = tokenizer_path
        self.num_workers = num_workers
        self.device_ids = device_ids or [0]
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        
    @staticmethod
    def _evaluate_file_batch(
        file_paths: List[str],
        model_checkpoint: str,
        tokenizer_path: str,
        device_id: int,
        use_fp16: bool,
        batch_size: int,
        max_samples_per_file: Optional[int] = None,
        worker_id: int = 0
    ) -> List[Tuple[str, pd.DataFrame]]:
        """
        Worker function to evaluate a batch of BLIMP files.
        
        Args:
            file_paths: List of file paths to evaluate
            model_checkpoint: Path to model checkpoint
            tokenizer_path: Path to tokenizer
            device_id: GPU device ID to use
            use_fp16: Use mixed precision
            batch_size: Batch size for evaluation
            max_samples_per_file: Maximum samples per file
            worker_id: Worker ID for logging
            
        Returns:
            List of (filename, results_dataframe) tuples
        """
        # Set device for this worker
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        
        # Load model and tokenizer for this worker
        model_loader = ModelLoader(device=device)
        model, tokenizer = model_loader.load_model_and_tokenizer(
            checkpoint_path=model_checkpoint,
            tokenizer_path=tokenizer_path,
            use_fp16=use_fp16
        )
        
        # Create evaluator
        surprisal_calc = SurprisalCalculator(model, tokenizer, device)
        evaluator = BLIMPEvaluator(surprisal_calc)
        
        results = []
        for filepath in file_paths:
            try:
                logger.info(f"Worker {worker_id} evaluating {Path(filepath).name} on device {device_id}")
                df = evaluator.evaluate_file(filepath, max_samples_per_file)
                df['file'] = Path(filepath).stem
                results.append((Path(filepath).stem, df))
            except Exception as e:
                logger.error(f"Worker {worker_id} failed on {filepath}: {e}")
                continue
        
        # Clean up
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def evaluate_all(
        self,
        blimp_dir: str,
        output_path: Optional[str] = None,
        max_files: Optional[int] = None,
        max_samples_per_file: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate all BLIMP files in parallel.
        
        Args:
            blimp_dir: Directory containing BLIMP JSONL files
            output_path: Optional path to save results
            max_files: Maximum number of files to process
            max_samples_per_file: Maximum samples per file
            
        Returns:
            DataFrame with all results
        """
        blimp_dir = Path(blimp_dir)
        
        if not blimp_dir.exists():
            raise FileNotFoundError(f"BLIMP directory not found: {blimp_dir}")
        
        # Find all JSONL files
        jsonl_files = sorted(blimp_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {blimp_dir}")
        
        if max_files:
            jsonl_files = jsonl_files[:max_files]
        
        logger.info(f"Found {len(jsonl_files)} BLIMP files to evaluate with {self.num_workers} workers")
        
        # Convert to string paths
        file_paths = [str(f) for f in jsonl_files]
        
        # Split files into batches for workers
        batch_size = max(1, len(file_paths) // self.num_workers)
        file_batches = [
            file_paths[i:i + batch_size]
            for i in range(0, len(file_paths), batch_size)
        ]
        
        # Ensure we don't have more batches than workers
        if len(file_batches) > self.num_workers:
            # Merge extra batches into the last one
            extra_batches = file_batches[self.num_workers:]
            for batch in extra_batches:
                file_batches[self.num_workers - 1].extend(batch)
            file_batches = file_batches[:self.num_workers]
        
        # Distribute work across processes
        all_results = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks
            futures = {}
            for i, batch in enumerate(file_batches):
                # Assign GPU in round-robin fashion
                device_id = self.device_ids[i % len(self.device_ids)]
                
                future = executor.submit(
                    self._evaluate_file_batch,
                    batch,
                    self.model_checkpoint,
                    self.tokenizer_path,
                    device_id,
                    self.use_fp16,
                    self.batch_size,
                    max_samples_per_file,
                    i
                )
                futures[future] = i
            
            # Collect results with progress bar
            with tqdm(total=len(file_paths), desc="Evaluating BLIMP files") as pbar:
                for future in as_completed(futures):
                    worker_id = futures[future]
                    try:
                        batch_results = future.result()
                        for filename, df in batch_results:
                            all_results.append(df)
                            pbar.update(1)
                    except Exception as e:
                        logger.error(f"Worker {worker_id} failed: {e}")
                        pbar.update(len(file_batches[worker_id]))
        
        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Save if requested
            if output_path:
                combined_df.to_json(output_path, orient='records', lines=True)
                logger.info(f"Saved results to {output_path}")
            
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_summary_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics from results.
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            Dictionary with summary statistics
        """
        if results_df.empty:
            return {}
        
        # Overall metrics
        summary = {
            'overall_accuracy': results_df['correct'].mean(),
            'overall_surprisal_diff': results_df['surprisal_difference'].mean(),
            'total_items': len(results_df)
        }
        
        # Per-file metrics
        if 'file' in results_df.columns:
            file_stats = results_df.groupby('file').agg({
                'correct': 'mean',
                'surprisal_difference': 'mean'
            }).to_dict('index')
            summary['per_file'] = file_stats
        
        # Per-phenomenon metrics
        if 'linguistics_term' in results_df.columns:
            phenomenon_stats = results_df.groupby('linguistics_term').agg({
                'correct': 'mean',
                'surprisal_difference': 'mean'
            }).to_dict('index')
            summary['per_phenomenon'] = phenomenon_stats
        
        return summary