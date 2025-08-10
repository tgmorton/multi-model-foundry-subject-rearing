"""
BLIMP (Benchmark of Linguistic Minimal Pairs) evaluation module.
Processes JSONL files and evaluates model performance on linguistic phenomena.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from .surprisal_calculator import SurprisalCalculator

logger = logging.getLogger(__name__)


class BLIMPEvaluator:
    """Evaluate language models on BLIMP dataset."""
    
    def __init__(self, surprisal_calculator: SurprisalCalculator):
        """
        Initialize BLIMP evaluator.
        
        Args:
            surprisal_calculator: Instance of SurprisalCalculator
        """
        self.calculator = surprisal_calculator
        
    def load_blimp_file(self, filepath: str) -> List[Dict]:
        """
        Load a BLIMP JSONL file.
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            List of stimulus dictionaries
        """
        stimuli = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    stimuli.append(json.loads(line))
        
        logger.info(f"Loaded {len(stimuli)} stimuli from {Path(filepath).name}")
        return stimuli
    
    def evaluate_stimulus(self, stimulus: Dict) -> Dict:
        """
        Evaluate a single BLIMP stimulus.
        
        Args:
            stimulus: Dictionary with 'sentence_good' and 'sentence_bad'
            
        Returns:
            Evaluation results
        """
        result = self.calculator.compare_minimal_pair(
            stimulus['sentence_good'],
            stimulus['sentence_bad']
        )
        
        # Add metadata
        result['UID'] = stimulus.get('UID', 'unknown')
        result['linguistics_term'] = stimulus.get('linguistics_term', 'unknown')
        result['field'] = stimulus.get('field', 'unknown')
        result['pairID'] = stimulus.get('pairID', 'unknown')
        
        return result
    
    def evaluate_file(
        self,
        filepath: str,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate all stimuli in a BLIMP file.
        
        Args:
            filepath: Path to JSONL file
            max_samples: Maximum number of samples to evaluate (for testing)
            
        Returns:
            DataFrame with results
        """
        # Load stimuli
        stimuli = self.load_blimp_file(filepath)
        
        if max_samples:
            stimuli = stimuli[:max_samples]
        
        # Evaluate each stimulus
        results = []
        for stimulus in tqdm(stimuli, desc=f"Evaluating {Path(filepath).stem}"):
            result = self.evaluate_stimulus(stimulus)
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Calculate aggregate metrics
        accuracy = df['correct'].mean()
        mean_diff = df['surprisal_difference'].mean()
        
        logger.info(
            f"{Path(filepath).stem}: Accuracy={accuracy:.3f}, "
            f"Mean surprisal diff={mean_diff:.3f}"
        )
        
        return df
    
    def evaluate_all(
        self,
        blimp_dir: str,
        output_path: Optional[str] = None,
        max_files: Optional[int] = None,
        max_samples_per_file: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate all BLIMP files in a directory.
        
        Args:
            blimp_dir: Directory containing BLIMP JSONL files
            output_path: Optional path to save results
            max_files: Maximum number of files to process (for testing)
            max_samples_per_file: Maximum samples per file (for testing)
            
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
        
        logger.info(f"Found {len(jsonl_files)} BLIMP files to evaluate")
        
        # Evaluate each file
        all_results = []
        for filepath in jsonl_files:
            df = self.evaluate_file(filepath, max_samples_per_file)
            df['filename'] = filepath.stem
            all_results.append(df)
        
        # Combine results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Calculate overall metrics
        overall_accuracy = combined_df['correct'].mean()
        logger.info(f"Overall BLIMP accuracy: {overall_accuracy:.3f}")
        
        # Calculate per-phenomenon accuracy
        phenomenon_accuracy = combined_df.groupby('linguistics_term')['correct'].mean()
        
        # Save if requested
        if output_path:
            combined_df.to_json(output_path, orient='records', lines=True)
            logger.info(f"Results saved to {output_path}")
        
        return combined_df
    
    def get_summary_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics from BLIMP results.
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'overall_accuracy': results_df['correct'].mean(),
            'total_stimuli': len(results_df),
            'mean_surprisal_difference': results_df['surprisal_difference'].mean(),
            'std_surprisal_difference': results_df['surprisal_difference'].std(),
        }
        
        # Per-phenomenon statistics
        phenomenon_stats = results_df.groupby('linguistics_term').agg({
            'correct': 'mean',
            'surprisal_difference': 'mean'
        }).to_dict('index')
        
        summary['by_phenomenon'] = phenomenon_stats
        
        # Per-field statistics
        if 'field' in results_df.columns:
            field_stats = results_df.groupby('field')['correct'].mean().to_dict()
            summary['by_field'] = field_stats
        
        # Per-file statistics
        if 'filename' in results_df.columns:
            file_stats = results_df.groupby('filename')['correct'].mean().to_dict()
            summary['by_file'] = file_stats
        
        return summary