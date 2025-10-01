"""
Comprehensive summary generator for evaluation results.
Generates detailed hierarchical statistics for analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """Generate comprehensive summaries with hierarchical statistics."""
    
    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for a list of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary with statistics
        """
        if not values:
            return {}
        
        # Convert to numpy array and ensure float type for boolean inputs
        arr = np.array(values, dtype=float)
        n = len(arr)
        
        stats = {
            'n': n,
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr, ddof=1)) if n > 1 else 0.0,
            'se': float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
            'q1': float(np.percentile(arr, 25)),
            'q3': float(np.percentile(arr, 75))
        }
        
        # Add 95% CI if we have enough data
        if n > 1:
            stats['ci95_lower'] = stats['mean'] - 1.96 * stats['se']
            stats['ci95_upper'] = stats['mean'] + 1.96 * stats['se']
        
        return stats
    
    @staticmethod
    def generate_blimp_summary(checkpoint_name: str, blimp_file: Path) -> Dict[str, Any]:
        """
        Generate comprehensive BLIMP summary for a checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            blimp_file: Path to BLIMP detailed results file
            
        Returns:
            Comprehensive summary dictionary
        """
        # Load detailed results
        try:
            df = pd.read_json(blimp_file, lines=True)
        except Exception as e:
            logger.error(f"Failed to load BLIMP results: {e}")
            return {}
        
        summary = {
            'checkpoint': checkpoint_name,
            'overall': {},
            'by_phenomenon': {},
            'by_field': {}
        }
        
        # Overall statistics
        if 'correct' in df.columns:
            summary['overall'] = {
                'accuracy': float(df['correct'].mean()),
                'n_items': len(df),
                **SummaryGenerator.calculate_statistics(df['correct'].tolist())
            }
        
        if 'surprisal_difference' in df.columns:
            summary['overall']['surprisal'] = SummaryGenerator.calculate_statistics(
                df['surprisal_difference'].tolist()
            )
        
        # By phenomenon (linguistics_term)
        if 'linguistics_term' in df.columns:
            for phenomenon, group_df in df.groupby('linguistics_term'):
                summary['by_phenomenon'][phenomenon] = {
                    'accuracy': float(group_df['correct'].mean()),
                    'n_items': len(group_df),
                    **SummaryGenerator.calculate_statistics(group_df['correct'].tolist())
                }
                
                if 'surprisal_difference' in group_df.columns:
                    summary['by_phenomenon'][phenomenon]['surprisal'] = \
                        SummaryGenerator.calculate_statistics(group_df['surprisal_difference'].tolist())
        
        # By field (syntax, morphology, etc.)
        if 'field' in df.columns:
            for field, group_df in df.groupby('field'):
                summary['by_field'][field] = {
                    'accuracy': float(group_df['correct'].mean()),
                    'n_items': len(group_df),
                    **SummaryGenerator.calculate_statistics(group_df['correct'].tolist())
                }
        
        # By file (if available)
        if 'file' in df.columns:
            summary['by_file'] = {}
            for file_name, group_df in df.groupby('file'):
                summary['by_file'][file_name] = {
                    'accuracy': float(group_df['correct'].mean()),
                    'n_items': len(group_df),
                    **SummaryGenerator.calculate_statistics(group_df['correct'].tolist())
                }
        
        return summary
    
    @staticmethod
    def generate_null_subject_summary(checkpoint_name: str, ns_file: Path) -> Dict[str, Any]:
        """
        Generate comprehensive null-subject summary for a checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            ns_file: Path to null-subject detailed results file
            
        Returns:
            Comprehensive summary dictionary
        """
        # Load detailed results
        try:
            df = pd.read_json(ns_file, lines=True)
        except Exception as e:
            logger.error(f"Failed to load null-subject results: {e}")
            return {}
        
        summary = {
            'checkpoint': checkpoint_name,
            'overall': {},
            'by_item_group': {},
            'by_form': {},
            'by_person_number': {},
            'by_item_group_and_form': {}
        }
        
        # Overall statistics
        if 'correct' in df.columns:
            summary['overall'] = {
                'accuracy': float(df['correct'].mean()),
                'n_items': len(df),
                **SummaryGenerator.calculate_statistics(df['correct'].tolist())
            }
        
        if 'surprisal_difference' in df.columns:
            summary['overall']['surprisal'] = SummaryGenerator.calculate_statistics(
                df['surprisal_difference'].tolist()
            )
        
        # By item_group
        if 'item_group' in df.columns:
            for item_group, group_df in df.groupby('item_group'):
                summary['by_item_group'][item_group] = {
                    'accuracy': float(group_df['correct'].mean()),
                    'n_items': len(group_df),
                    **SummaryGenerator.calculate_statistics(group_df['correct'].tolist())
                }
                
                if 'surprisal_difference' in group_df.columns:
                    summary['by_item_group'][item_group]['surprisal'] = \
                        SummaryGenerator.calculate_statistics(group_df['surprisal_difference'].tolist())
        
        # By form
        if 'form' in df.columns:
            for form, group_df in df.groupby('form'):
                summary['by_form'][form] = {
                    'accuracy': float(group_df['correct'].mean()),
                    'n_items': len(group_df),
                    **SummaryGenerator.calculate_statistics(group_df['correct'].tolist())
                }
                
                if 'surprisal_difference' in group_df.columns:
                    summary['by_form'][form]['surprisal'] = \
                        SummaryGenerator.calculate_statistics(group_df['surprisal_difference'].tolist())
        
        # By person-number combination
        if 'person' in df.columns and 'number' in df.columns:
            df['person_number'] = df['person'].astype(str) + '_' + df['number'].astype(str)
            for pn, group_df in df.groupby('person_number'):
                summary['by_person_number'][pn] = {
                    'accuracy': float(group_df['correct'].mean()),
                    'n_items': len(group_df),
                    **SummaryGenerator.calculate_statistics(group_df['correct'].tolist())
                }
        
        # By item_group AND form (nested)
        if 'item_group' in df.columns and 'form' in df.columns:
            for item_group, ig_df in df.groupby('item_group'):
                summary['by_item_group_and_form'][item_group] = {}
                for form, form_df in ig_df.groupby('form'):
                    summary['by_item_group_and_form'][item_group][form] = {
                        'accuracy': float(form_df['correct'].mean()),
                        'n_items': len(form_df),
                        **SummaryGenerator.calculate_statistics(form_df['correct'].tolist())
                    }
                    
                    if 'surprisal_difference' in form_df.columns:
                        summary['by_item_group_and_form'][item_group][form]['surprisal'] = \
                            SummaryGenerator.calculate_statistics(form_df['surprisal_difference'].tolist())
        
        return summary
    
    @staticmethod
    def generate_perplexity_summary(checkpoint_name: str, perp_file: Path) -> Dict[str, Any]:
        """
        Generate comprehensive perplexity summary for a checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            perp_file: Path to perplexity results file
            
        Returns:
            Comprehensive summary dictionary
        """
        # Load perplexity results
        try:
            with open(perp_file, 'r') as f:
                perp_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load perplexity results: {e}")
            return {}
        
        summary = {
            'checkpoint': checkpoint_name,
            'overall': {
                'perplexity': perp_data.get('perplexity'),
                'loss': perp_data.get('loss'),
                'n_tokens': perp_data.get('total_tokens'),
                'n_sequences': perp_data.get('num_sequences')
            }
        }
        
        # Add per-file stats if available
        if 'per_file_stats' in perp_data:
            summary['by_file'] = perp_data['per_file_stats']
        
        return summary
    
    @staticmethod
    def save_comprehensive_summaries(output_dir: Path, experiment_name: str = "experiment"):
        """
        Generate and save comprehensive summaries for all checkpoints.
        
        Args:
            output_dir: Directory containing evaluation results
            experiment_name: Name of the experiment
        """
        output_dir = Path(output_dir)
        
        # Collect all checkpoints
        checkpoints = set()
        for f in output_dir.glob("checkpoint-*_*.json*"):
            checkpoint = f.name.split('_')[0]
            checkpoints.add(checkpoint)
        
        # Sort checkpoints numerically
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
        
        # Generate summaries for each task type
        all_summaries = {
            'experiment': experiment_name,
            'checkpoints': {},
            'cross_checkpoint_stats': {}
        }
        
        # Process each checkpoint
        for checkpoint in checkpoints:
            all_summaries['checkpoints'][checkpoint] = {}
            
            # BLIMP summary
            blimp_file = output_dir / f"{checkpoint}_blimp_detailed.jsonl"
            if blimp_file.exists():
                blimp_summary = SummaryGenerator.generate_blimp_summary(checkpoint, blimp_file)
                all_summaries['checkpoints'][checkpoint]['blimp'] = blimp_summary
            
            # Null-subject summary
            ns_file = output_dir / f"{checkpoint}_null_subject_detailed.jsonl"
            if ns_file.exists():
                ns_summary = SummaryGenerator.generate_null_subject_summary(checkpoint, ns_file)
                all_summaries['checkpoints'][checkpoint]['null_subject'] = ns_summary
            
            # Perplexity summary
            perp_file = output_dir / f"{checkpoint}_perplexity.json"
            if perp_file.exists():
                perp_summary = SummaryGenerator.generate_perplexity_summary(checkpoint, perp_file)
                all_summaries['checkpoints'][checkpoint]['perplexity'] = perp_summary
        
        # Calculate cross-checkpoint statistics
        all_summaries['cross_checkpoint_stats'] = \
            SummaryGenerator._calculate_cross_checkpoint_stats(all_summaries['checkpoints'])
        
        # Save comprehensive summary
        summary_file = output_dir / "comprehensive_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        
        logger.info(f"Saved comprehensive summary to {summary_file}")
        
        # Also save separate CSV files for easy analysis
        SummaryGenerator._save_csv_summaries(all_summaries, output_dir)
    
    @staticmethod
    def _calculate_cross_checkpoint_stats(checkpoints_data: Dict) -> Dict:
        """Calculate statistics across checkpoints."""
        stats = {
            'blimp': {'overall': [], 'by_phenomenon': {}},
            'null_subject': {'overall': [], 'by_item_group': {}, 'by_form': {}},
            'perplexity': []
        }
        
        for checkpoint, data in checkpoints_data.items():
            # BLIMP
            if 'blimp' in data and data['blimp']:
                if 'overall' in data['blimp']:
                    stats['blimp']['overall'].append(data['blimp']['overall'].get('accuracy'))
                
                for phenomenon, pdata in data['blimp'].get('by_phenomenon', {}).items():
                    if phenomenon not in stats['blimp']['by_phenomenon']:
                        stats['blimp']['by_phenomenon'][phenomenon] = []
                    stats['blimp']['by_phenomenon'][phenomenon].append(pdata.get('accuracy'))
            
            # Null-subject
            if 'null_subject' in data and data['null_subject']:
                if 'overall' in data['null_subject']:
                    stats['null_subject']['overall'].append(data['null_subject']['overall'].get('accuracy'))
                
                for item_group, igdata in data['null_subject'].get('by_item_group', {}).items():
                    if item_group not in stats['null_subject']['by_item_group']:
                        stats['null_subject']['by_item_group'][item_group] = []
                    stats['null_subject']['by_item_group'][item_group].append(igdata.get('accuracy'))
                
                for form, fdata in data['null_subject'].get('by_form', {}).items():
                    if form not in stats['null_subject']['by_form']:
                        stats['null_subject']['by_form'][form] = []
                    stats['null_subject']['by_form'][form].append(fdata.get('accuracy'))
            
            # Perplexity
            if 'perplexity' in data and data['perplexity']:
                if 'overall' in data['perplexity']:
                    stats['perplexity'].append(data['perplexity']['overall'].get('perplexity'))
        
        # Calculate statistics for each metric
        return {
            'blimp_overall': SummaryGenerator.calculate_statistics(stats['blimp']['overall']),
            'null_subject_overall': SummaryGenerator.calculate_statistics(stats['null_subject']['overall']),
            'perplexity': SummaryGenerator.calculate_statistics(stats['perplexity'])
        }
    
    @staticmethod
    def _save_csv_summaries(all_summaries: Dict, output_dir: Path):
        """Save CSV files for easy analysis in R or Excel."""
        # BLIMP by checkpoint and phenomenon
        blimp_rows = []
        for checkpoint, data in all_summaries['checkpoints'].items():
            if 'blimp' in data and data['blimp']:
                for phenomenon, pdata in data['blimp'].get('by_phenomenon', {}).items():
                    blimp_rows.append({
                        'checkpoint': checkpoint,
                        'checkpoint_num': int(checkpoint.split('-')[1]),
                        'phenomenon': phenomenon,
                        'accuracy': pdata.get('accuracy'),
                        'n_items': pdata.get('n_items'),
                        'mean': pdata.get('mean'),
                        'std': pdata.get('std'),
                        'se': pdata.get('se')
                    })
        
        if blimp_rows:
            blimp_df = pd.DataFrame(blimp_rows)
            blimp_df.to_csv(output_dir / 'blimp_by_phenomenon.csv', index=False)
        
        # Null-subject by checkpoint, item_group, and form
        ns_rows = []
        for checkpoint, data in all_summaries['checkpoints'].items():
            if 'null_subject' in data and data['null_subject']:
                for item_group, ig_forms in data['null_subject'].get('by_item_group_and_form', {}).items():
                    for form, fdata in ig_forms.items():
                        ns_rows.append({
                            'checkpoint': checkpoint,
                            'checkpoint_num': int(checkpoint.split('-')[1]),
                            'item_group': item_group,
                            'form': form,
                            'accuracy': fdata.get('accuracy'),
                            'n_items': fdata.get('n_items'),
                            'mean': fdata.get('mean'),
                            'std': fdata.get('std'),
                            'se': fdata.get('se')
                        })
        
        if ns_rows:
            ns_df = pd.DataFrame(ns_rows)
            ns_df.to_csv(output_dir / 'null_subject_by_item_group_form.csv', index=False)