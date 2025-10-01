"""
Threaded BLIMP evaluation module for faster processing using threading instead of multiprocessing.
This avoids CUDA multiprocessing issues while still providing parallelism.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from ..core.model_loader import ModelLoader
from ..core.surprisal_calculator import SurprisalCalculator
from ..evaluators.blimp_evaluator import BLIMPEvaluator

logger = logging.getLogger(__name__)


class ThreadedBLIMPEvaluator:
    """Threaded BLIMP evaluator that uses threading for parallel file processing."""
    
    def __init__(
        self,
        model_checkpoint: str,
        tokenizer_path: str,
        num_threads: int = 4,
        device_id: int = 0,
        use_fp16: bool = False,
        batch_size: int = 32
    ):
        """
        Initialize threaded BLIMP evaluator.
        
        Args:
            model_checkpoint: Path to model checkpoint
            tokenizer_path: Path to tokenizer
            num_threads: Number of threads for parallel evaluation
            device_id: GPU device ID to use
            use_fp16: Use mixed precision
            batch_size: Batch size for evaluation
        """
        self.model_checkpoint = model_checkpoint
        self.tokenizer_path = tokenizer_path
        self.num_threads = num_threads
        self.device_id = device_id
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        
        # Load model once and share across threads
        if device_id >= 0:
            device = f"cuda:{device_id}"
        elif device_id == -1:
            # Use default CUDA device (respects CUDA_VISIBLE_DEVICES)
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"
        self.model_loader = ModelLoader(device=device)
        self.model, self.tokenizer = self.model_loader.load_model_and_tokenizer(
            checkpoint_path=model_checkpoint,
            tokenizer_path=tokenizer_path,
            use_fp16=use_fp16
        )
        
        # Create evaluator
        self.surprisal_calc = SurprisalCalculator(self.model, self.tokenizer, device)
        self.evaluator = BLIMPEvaluator(self.surprisal_calc)
        
        # Thread lock for thread-safe model access
        self.model_lock = threading.Lock()
        
    def _evaluate_file_threaded(self, filepath: str, max_samples_per_file: Optional[int] = None) -> Tuple[str, pd.DataFrame]:
        """
        Thread-safe file evaluation function.
        
        Args:
            filepath: Path to BLIMP file
            max_samples_per_file: Maximum samples per file
            
        Returns:
            Tuple of (filename, results_dataframe)
        """
        thread_id = threading.current_thread().ident
        logger.debug(f"Thread {thread_id} evaluating {Path(filepath).name}")
        
        try:
            # Thread-safe evaluation
            with self.model_lock:
                df = self.evaluator.evaluate_file(filepath, max_samples_per_file)
            
            df['file'] = Path(filepath).stem
            return (Path(filepath).stem, df)
        except Exception as e:
            logger.error(f"Thread {thread_id} failed on {filepath}: {e}")
            raise
    
    def evaluate_all(
        self,
        blimp_dir: str,
        output_path: Optional[str] = None,
        max_files: Optional[int] = None,
        max_samples_per_file: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate all BLIMP files using threading.
        
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
        
        logger.info(f"Found {len(jsonl_files)} BLIMP files to evaluate with {self.num_threads} threads")
        
        # Convert to string paths
        file_paths = [str(f) for f in jsonl_files]
        
        # Evaluate using threading
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit tasks
            futures = {
                executor.submit(self._evaluate_file_threaded, filepath, max_samples_per_file): filepath
                for filepath in file_paths
            }
            
            # Collect results with progress bar
            with tqdm(total=len(file_paths), desc="Evaluating BLIMP files") as pbar:
                for future in as_completed(futures):
                    filepath = futures[future]
                    try:
                        filename, df = future.result()
                        all_results.append(df)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"File {filepath} failed: {e}")
                        pbar.update(1)
        
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
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer