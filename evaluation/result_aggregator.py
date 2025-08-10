"""
Result aggregation and export utilities for evaluation pipeline.
Converts evaluation results to R-compatible formats.
"""

import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregate and export evaluation results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize result aggregator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_evaluation_results(self, results_file: str) -> List[Dict]:
        """
        Load evaluation results from JSONL file.
        
        Args:
            results_file: Path to evaluation results file
            
        Returns:
            List of result dictionaries
        """
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        logger.info(f"Loaded {len(results)} evaluation results")
        return results
    
    def extract_scalar_metrics(self, results: List[Dict]) -> pd.DataFrame:
        """
        Extract scalar metrics for R analysis.
        
        Args:
            results: List of evaluation result dictionaries
            
        Returns:
            DataFrame with one row per checkpoint and metric
        """
        rows = []
        
        for result in results:
            checkpoint = result.get('checkpoint', 'unknown')
            
            # Extract checkpoint metadata
            base_row = {
                'checkpoint': checkpoint,
                'timestamp': result.get('timestamp', ''),
            }
            
            # Extract epoch/step number if possible
            if 'epoch' in checkpoint:
                try:
                    base_row['epoch'] = int(checkpoint.split('epoch_')[1])
                except:
                    base_row['epoch'] = 0
            elif 'checkpoint' in checkpoint:
                try:
                    base_row['step'] = int(checkpoint.split('checkpoint-')[1])
                except:
                    base_row['step'] = 0
            
            # Perplexity metrics
            if 'perplexity' in result:
                ppl_data = result['perplexity']
                if isinstance(ppl_data, dict):
                    rows.append({
                        **base_row,
                        'metric': 'perplexity',
                        'value': ppl_data.get('perplexity', np.nan),
                        'n_samples': ppl_data.get('num_sequences', 0),
                        'n_tokens': ppl_data.get('total_tokens', 0)
                    })
            
            # BLIMP metrics
            if 'blimp' in result:
                blimp_data = result['blimp']
                if isinstance(blimp_data, dict):
                    rows.append({
                        **base_row,
                        'metric': 'blimp_overall',
                        'value': blimp_data.get('overall_accuracy', np.nan),
                        'n_samples': blimp_data.get('total_stimuli', 0)
                    })
                    
                    # Per-phenomenon BLIMP results
                    if 'by_phenomenon' in blimp_data:
                        for phenomenon, stats in blimp_data['by_phenomenon'].items():
                            rows.append({
                                **base_row,
                                'metric': f'blimp_{phenomenon}',
                                'value': stats.get('correct', np.nan),
                                'phenomenon': phenomenon
                            })
            
            # Null-subject metrics
            if 'null_subject' in result:
                ns_data = result['null_subject']
                if isinstance(ns_data, dict):
                    rows.append({
                        **base_row,
                        'metric': 'null_subject_overt_pref',
                        'value': ns_data.get('overt_preference_rate', np.nan),
                        'n_samples': ns_data.get('total_pairs', 0)
                    })
                    
                    rows.append({
                        **base_row,
                        'metric': 'null_subject_surprisal_diff',
                        'value': ns_data.get('mean_surprisal_difference', np.nan),
                        'n_samples': ns_data.get('total_pairs', 0)
                    })
                    
                    # Per-condition null-subject results
                    if 'by_condition' in ns_data:
                        for condition, stats in ns_data['by_condition'].items():
                            rows.append({
                                **base_row,
                                'metric': f'null_subject_{condition}_overt_pref',
                                'value': stats.get('prefers_overt', np.nan),
                                'condition': condition
                            })
        
        return pd.DataFrame(rows)
    
    def create_learning_curves(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create learning curve data for visualization.
        
        Args:
            metrics_df: DataFrame with extracted metrics
            
        Returns:
            DataFrame optimized for plotting learning curves
        """
        # Filter to main metrics only
        main_metrics = [
            'perplexity', 'blimp_overall', 
            'null_subject_overt_pref', 'null_subject_surprisal_diff'
        ]
        
        curve_data = metrics_df[metrics_df['metric'].isin(main_metrics)].copy()
        
        # Sort by training progress
        if 'epoch' in curve_data.columns:
            curve_data = curve_data.sort_values(['metric', 'epoch'])
        elif 'step' in curve_data.columns:
            curve_data = curve_data.sort_values(['metric', 'step'])
        
        return curve_data
    
    def compare_experiments(self, experiment_dirs: List[str]) -> pd.DataFrame:
        """
        Compare results across multiple experiments.
        
        Args:
            experiment_dirs: List of experiment result directories
            
        Returns:
            DataFrame with cross-experiment comparison
        """
        all_data = []
        
        for exp_dir in experiment_dirs:
            exp_path = Path(exp_dir)
            results_file = exp_path / "evaluation_results.jsonl"
            
            if not results_file.exists():
                logger.warning(f"No results file found in {exp_dir}")
                continue
            
            # Load results
            results = self.load_evaluation_results(results_file)
            
            # Extract metrics
            metrics_df = self.extract_scalar_metrics(results)
            metrics_df['experiment'] = exp_path.name
            
            all_data.append(metrics_df)
        
        if not all_data:
            raise ValueError("No valid experiment results found")
        
        # Combine data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        return combined_df
    
    def generate_summary_report(self, results: List[Dict]) -> Dict:
        """
        Generate a summary report of evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Summary report dictionary
        """
        report = {
            'generation_time': datetime.now().isoformat(),
            'total_checkpoints': len(results),
            'successful_evaluations': len([r for r in results if 'error' not in r])
        }
        
        # Extract final checkpoint metrics
        if results:
            final_result = results[-1]
            
            if 'perplexity' in final_result:
                report['final_perplexity'] = final_result['perplexity'].get('perplexity')
            
            if 'blimp' in final_result:
                report['final_blimp_accuracy'] = final_result['blimp'].get('overall_accuracy')
            
            if 'null_subject' in final_result:
                report['final_overt_preference'] = final_result['null_subject'].get('overt_preference_rate')
        
        # Calculate improvement metrics
        if len(results) > 1:
            first_result = results[0]
            
            # Perplexity improvement
            if ('perplexity' in first_result and 'perplexity' in final_result and
                first_result['perplexity'].get('perplexity') and final_result['perplexity'].get('perplexity')):
                initial_ppl = first_result['perplexity']['perplexity']
                final_ppl = final_result['perplexity']['perplexity']
                report['perplexity_improvement'] = initial_ppl - final_ppl
                report['perplexity_improvement_pct'] = ((initial_ppl - final_ppl) / initial_ppl) * 100
            
            # BLIMP improvement
            if ('blimp' in first_result and 'blimp' in final_result):
                initial_acc = first_result['blimp'].get('overall_accuracy', 0)
                final_acc = final_result['blimp'].get('overall_accuracy', 0)
                report['blimp_improvement'] = final_acc - initial_acc
                report['blimp_improvement_pct'] = ((final_acc - initial_acc) / initial_acc) * 100 if initial_acc > 0 else 0
        
        return report
    
    def export_for_r(
        self,
        results_file: str,
        output_prefix: str = "evaluation_data"
    ) -> Dict[str, str]:
        """
        Export evaluation results in R-compatible formats.
        
        Args:
            results_file: Path to evaluation results JSONL file
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary mapping data types to output file paths
        """
        # Load results
        results = self.load_evaluation_results(results_file)
        
        # Extract scalar metrics
        metrics_df = self.extract_scalar_metrics(results)
        
        # Create learning curves
        curves_df = self.create_learning_curves(metrics_df)
        
        # Generate summary report
        summary = self.generate_summary_report(results)
        
        # Save files
        output_files = {}
        
        # Main metrics file (long format for mixed-effects models)
        metrics_file = self.output_dir / f"{output_prefix}_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        output_files['metrics'] = str(metrics_file)
        
        # Learning curves file
        curves_file = self.output_dir / f"{output_prefix}_learning_curves.csv"
        curves_df.to_csv(curves_file, index=False)
        output_files['learning_curves'] = str(curves_file)
        
        # Summary report
        summary_file = self.output_dir / f"{output_prefix}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        output_files['summary'] = str(summary_file)
        
        # Wide format for simple analyses
        try:
            wide_df = metrics_df.pivot(
                index=['checkpoint', 'epoch', 'step'], 
                columns='metric', 
                values='value'
            ).reset_index()
            wide_file = self.output_dir / f"{output_prefix}_wide.csv"
            wide_df.to_csv(wide_file, index=False)
            output_files['wide_format'] = str(wide_file)
        except Exception as e:
            logger.warning(f"Could not create wide format: {e}")
        
        logger.info(f"Exported evaluation data to {len(output_files)} files")
        return output_files