"""
Null-subject stimuli evaluation module.
Processes CSV files with overt vs. null subject minimal pairs.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from ..core.surprisal_calculator import NullSubjectSurprisalCalculator
from ..unigram import UnigramTable

logger = logging.getLogger(__name__)


class NullSubjectEvaluator:
    """Evaluate language models on null-subject stimuli."""

    def __init__(
        self,
        surprisal_calculator: NullSubjectSurprisalCalculator,
        unigram: Optional[UnigramTable] = None,
    ):
        """
        Initialize null-subject evaluator.

        Args:
            surprisal_calculator: Instance of NullSubjectSurprisalCalculator
            unigram: Optional per-corpus unigram baseline (FIT-CLAMS style).
                When supplied, the evaluator adds `overt_slor`, `null_slor`,
                and related fields to each pair's result. When None, SLOR
                fields are omitted.
        """
        self.calculator = surprisal_calculator
        self.unigram = unigram
    
    # v2 schema columns (from docs/eval_stimuli/design.md §1)
    _V2_REQUIRED = [
        'item_id', 'category', 'condition', 'pronoun_status',
        'context', 'target', 'hotspot_token', 'hotspot_position',
        'language',
    ]

    @staticmethod
    def _is_v2_schema(df: pd.DataFrame) -> bool:
        return 'item_id' in df.columns and 'hotspot_position' in df.columns

    def load_stimuli_file(self, filepath: str) -> pd.DataFrame:
        """
        Load a null-subject stimuli CSV file.

        Accepts either the v1 schema (item, item_group, pronoun_status,
        c_english, target, hotspot_english) or the v2 schema
        (item_id, category, condition, pronoun_status, context, target,
        hotspot_token, hotspot_position, language, ...). Internally
        normalizes to a common DataFrame using v1 column names plus a
        `schema_version` tag and (for v2) the resolved `hotspot_position`.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with stimuli, normalized to a common internal schema.
        """
        df = pd.read_csv(filepath)

        if self._is_v2_schema(df):
            missing = [c for c in self._V2_REQUIRED if c not in df.columns]
            if missing:
                raise ValueError(f"v2 schema missing required columns: {missing}")
            # Normalize to internal representation (v1 column names)
            df = df.rename(columns={
                'item_id': 'item',
                'category': 'item_group',
                'context': 'c_english',
                'hotspot_token': 'hotspot_english',
            })
            df['schema_version'] = 'v2'
            # Validate hotspot_position points to hotspot_token in target
            for _, row in df.iterrows():
                toks = str(row['target']).split()
                pos = int(row['hotspot_position'])
                if pos < 0 or pos >= len(toks) or toks[pos] != row['hotspot_english']:
                    raise ValueError(
                        f"v2 hotspot mismatch in {Path(filepath).name} "
                        f"item_id={row['item']} ps={row['pronoun_status']}: "
                        f"hotspot_position={pos} but target.split()[{pos}]"
                        f"={toks[pos] if 0 <= pos < len(toks) else '<OOB>'!r}"
                        f" vs hotspot_token={row['hotspot_english']!r}"
                    )
        elif 'form' in df.columns:
            # v1 master file format
            required = ['item', 'item_group', 'form', 'pronoun_status',
                        'c_english', 'target', 'hotspot_english']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"v1 master schema missing columns: {missing}")
            df['schema_version'] = 'v1_master'
        else:
            # v1 individual file format
            required = ['item', 'item_group', 'pronoun_status',
                        'c_english', 'target', 'hotspot_english']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"v1 schema missing columns: {missing}")
            df['schema_version'] = 'v1'

        logger.info(f"Loaded {len(df)} stimuli from {Path(filepath).name} "
                    f"(schema={df['schema_version'].iloc[0]})")
        return df
    
    def process_item_group(self, group_df: pd.DataFrame) -> Dict:
        """
        Process a group of stimuli (overt vs null pairs).
        
        Args:
            group_df: DataFrame subset for one item group
            
        Returns:
            Dictionary with evaluation results
        """
        # Separate overt (1) and null (0) conditions
        overt_rows = group_df[group_df['pronoun_status'] == 1]
        null_rows = group_df[group_df['pronoun_status'] == 0]
        
        if len(overt_rows) == 0 or len(null_rows) == 0:
            logger.warning(f"Incomplete pair in group: {group_df['item_group'].iloc[0]}")
            return {}
        
        # Take first row of each condition (in case of multiple)
        overt_row = overt_rows.iloc[0]
        null_row = null_rows.iloc[0]
        
        # Extract components
        context = overt_row['c_english'].strip()
        overt_target = overt_row['target'].strip()
        null_target = null_row['target'].strip()
        hotspot = overt_row['hotspot_english'].strip() if pd.notna(overt_row['hotspot_english']) else None
        
        # Evaluate the pair
        result = self.calculator.evaluate_null_subject_pair(
            context=context,
            overt_target=overt_target,
            null_target=null_target,
            hotspot=hotspot
        )
        
        # Add metadata
        result['item'] = int(overt_row['item'])
        result['item_group'] = overt_row['item_group']

        # v2 schema: pass condition + language through
        if 'condition' in overt_row.index:
            result['condition'] = overt_row['condition']
        if 'language' in overt_row.index:
            result['language'] = overt_row['language']
        if 'schema_version' in overt_row.index:
            result['schema_version'] = overt_row['schema_version']

        # Add form information if available (v1 master file)
        if 'form' in overt_row.index:
            result['form'] = overt_row['form']

        result['context'] = context
        result['overt_target'] = overt_target
        result['null_target'] = null_target
        result['hotspot'] = hotspot

        # D4: SLOR per side + difference + preference.
        # SLOR(s) = (Σ log P_M(w) - Σ log P_u(w)) / |s|
        # where the sum is over target tokens (natural log throughout).
        if self.unigram is not None:
            overt_ids = result.get('overt_per_token_ids', [])
            null_ids = result.get('null_per_token_ids', [])
            if overt_ids and null_ids:
                overt_model_sum = result['overt_total_log_prob']
                null_model_sum = result['null_total_log_prob']
                overt_unigram_sum = self.unigram.sum_log_prob(overt_ids)
                null_unigram_sum = self.unigram.sum_log_prob(null_ids)
                overt_n = max(len(overt_ids), 1)
                null_n = max(len(null_ids), 1)
                overt_slor = (overt_model_sum - overt_unigram_sum) / overt_n
                null_slor = (null_model_sum - null_unigram_sum) / null_n
                result['overt_slor'] = float(overt_slor)
                result['null_slor'] = float(null_slor)
                result['overt_unigram_sum_log_prob'] = float(overt_unigram_sum)
                result['null_unigram_sum_log_prob'] = float(null_unigram_sum)
                result['slor_diff_overt_minus_null'] = float(overt_slor - null_slor)
                result['prefers_overt_slor'] = bool(overt_slor > null_slor)
                result['unigram_corpus_id'] = self.unigram.corpus_id
                result['unigram_tokenizer_id'] = self.unigram.tokenizer_id

        return result
    
    def evaluate_file(
        self,
        filepath: str,
        max_items: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate all stimuli in a file.
        
        Args:
            filepath: Path to CSV file
            max_items: Maximum number of items to evaluate (for testing)
            
        Returns:
            DataFrame with results
        """
        # Load stimuli
        df = self.load_stimuli_file(filepath)
        
        # Group by appropriate columns depending on file format
        schema = df['schema_version'].iloc[0] if 'schema_version' in df.columns else 'v1'
        if schema == 'v2':
            # v2: pair on (item, item_group, condition) — condition distinguishes
            # sub-conditions within a category (e.g. subj_3sg vs subj_3pl).
            groups = df.groupby(['item', 'item_group', 'condition'])
        elif 'form' in df.columns:
            # v1 master: group by item, item_group, and form
            groups = df.groupby(['item', 'item_group', 'form'])
        else:
            # v1 individual: group by item only
            groups = df.groupby('item')
        
        if max_items:
            groups = list(groups)[:max_items]
        else:
            groups = list(groups)
        
        # Process each item group
        results = []
        for group_key, group_df in tqdm(groups, desc=f"Evaluating {Path(filepath).stem}"):
            result = self.process_item_group(group_df)
            if result:  # Only add non-empty results
                results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Calculate summary statistics
            overt_preference = results_df['prefers_overt'].mean()
            mean_diff = results_df['surprisal_difference'].mean()
            
            logger.info(
                f"{Path(filepath).stem}: Overt preference={overt_preference:.3f}, "
                f"Mean surprisal diff={mean_diff:.3f}"
            )
        
        return results_df
    
    def evaluate_all(
        self,
        stimuli_dir: str,
        output_path: Optional[str] = None,
        max_files: Optional[int] = None,
        max_items_per_file: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate all null-subject files in a directory.
        
        Args:
            stimuli_dir: Directory containing CSV files
            output_path: Optional path to save results
            max_files: Maximum number of files to process (for testing)
            max_items_per_file: Maximum items per file (for testing)
            
        Returns:
            DataFrame with all results
        """
        stimuli_dir = Path(stimuli_dir)
        
        if not stimuli_dir.exists():
            raise FileNotFoundError(f"Stimuli directory not found: {stimuli_dir}")
        
        # Look for master file first, then fall back to individual files
        master_file = stimuli_dir / "master_stimuli_transformed.csv"
        
        if master_file.exists():
            csv_files = [master_file]
            logger.info("Using master stimuli file: master_stimuli_transformed.csv")
        else:
            # Find all individual CSV files
            csv_files = sorted(stimuli_dir.glob("*.csv"))
            
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {stimuli_dir}")
            
            # Filter out master file if present
            csv_files = [f for f in csv_files if 'master' not in f.name.lower()]
            
            if max_files:
                csv_files = csv_files[:max_files]
            
            logger.info(f"Found {len(csv_files)} individual null-subject files to evaluate")
        
        # Evaluate each file
        all_results = []
        for filepath in csv_files:
            df = self.evaluate_file(filepath, max_items_per_file)
            if len(df) > 0:
                df['filename'] = filepath.stem
                all_results.append(df)
        
        if not all_results:
            logger.warning("No results obtained from any files")
            return pd.DataFrame()
        
        # Combine results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Calculate overall metrics
        if len(combined_df) > 0:
            overall_overt_pref = combined_df['prefers_overt'].mean()
            overall_mean_diff = combined_df['surprisal_difference'].mean()
            
            logger.info(
                f"Overall null-subject results: Overt preference={overall_overt_pref:.3f}, "
                f"Mean surprisal diff={overall_mean_diff:.3f}"
            )
        
        # Save if requested
        if output_path:
            combined_df.to_json(output_path, orient='records', lines=True)
            logger.info(f"Results saved to {output_path}")
        
        return combined_df
    
    def get_summary_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics from null-subject results.
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            Dictionary with summary statistics
        """
        if len(results_df) == 0:
            return {'error': 'No results to summarize'}
        
        summary = {
            'total_pairs': len(results_df),
            'overt_preference_rate': results_df['prefers_overt'].mean(),
            'mean_surprisal_difference': results_df['surprisal_difference'].mean(),
            'std_surprisal_difference': results_df['surprisal_difference'].std(),
            'mean_overt_surprisal': results_df['overt_mean_surprisal'].mean(),
            'mean_null_surprisal': results_df['null_mean_surprisal'].mean(),
        }
        
        # Per-condition statistics (by item_group)
        if 'item_group' in results_df.columns:
            condition_stats = results_df.groupby('item_group').agg({
                'prefers_overt': 'mean',
                'surprisal_difference': 'mean',
                'overt_mean_surprisal': 'mean',
                'null_mean_surprisal': 'mean'
            }).to_dict('index')
            
            summary['by_condition'] = condition_stats
            
        # Per-form statistics (if using master file)
        if 'form' in results_df.columns:
            form_stats = results_df.groupby('form').agg({
                'prefers_overt': 'mean',
                'surprisal_difference': 'mean',
                'overt_mean_surprisal': 'mean', 
                'null_mean_surprisal': 'mean'
            }).to_dict('index')
            
            summary['by_form'] = form_stats
            
            # Combined item_group and form statistics
            if 'item_group' in results_df.columns:
                combo_stats_raw = results_df.groupby(['item_group', 'form']).agg({
                    'prefers_overt': 'mean',
                    'surprisal_difference': 'mean'
                }).to_dict('index')
                
                # Convert tuple keys to strings for JSON serialization
                combo_stats = {f"{item_group}_{form}": stats 
                              for (item_group, form), stats in combo_stats_raw.items()}
                
                summary['by_condition_and_form'] = combo_stats
        
        # Per-file statistics (for individual files)
        if 'filename' in results_df.columns:
            file_stats = results_df.groupby('filename').agg({
                'prefers_overt': 'mean',
                'surprisal_difference': 'mean'
            }).to_dict('index')
            
            summary['by_file'] = file_stats
        
        # Hotspot analysis if available
        hotspot_results = results_df[results_df['hotspot_difference'].notna()]
        if len(hotspot_results) > 0:
            summary['hotspot_analysis'] = {
                'mean_hotspot_difference': hotspot_results['hotspot_difference'].mean(),
                'std_hotspot_difference': hotspot_results['hotspot_difference'].std(),
                'n_hotspot_pairs': len(hotspot_results)
            }
        
        # Ensure JSON serializable by converting any remaining tuple keys
        summary = self._make_json_serializable(summary)
        
        return summary
    
    def _make_json_serializable(self, obj):
        """
        Recursively convert any tuple keys to strings for JSON serialization.
        """
        if isinstance(obj, dict):
            return {
                str(key) if isinstance(key, tuple) else key: 
                self._make_json_serializable(value) 
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def analyze_by_person_number(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze results by person and number conditions.
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            Analysis by linguistic conditions
        """
        analysis = {}
        
        # Extract person/number from item_group
        if 'item_group' in results_df.columns:
            # Map item groups to linguistic conditions
            condition_mapping = {
                '1a_3rdSG': '3rd_person_singular',
                '1b_3rdPL': '3rd_person_plural',
                '2a_2ndSG': '2nd_person_singular', 
                '2b_2ndPL': '2nd_person_plural',
                '3a_1stSg': '1st_person_singular',
                '3b_1stPL': '1st_person_plural',
                '4a_subject_control': 'subject_control',
                '4b_object_control': 'object_control',
                '5a_expletive_seems': 'expletive_seems',
                '5b_expletive_be': 'expletive_be',
                '6_long_distance_binding': 'long_distance_binding',
                '7a_conjunction_no_topic_shift': 'conjunction_no_shift',
                '7b_conjunction_topic_shift': 'conjunction_shift'
            }
            
            # Group by condition type
            for group, condition in condition_mapping.items():
                subset = results_df[results_df['item_group'] == group]
                if len(subset) > 0:
                    analysis[condition] = {
                        'overt_preference': subset['prefers_overt'].mean(),
                        'surprisal_difference': subset['surprisal_difference'].mean(),
                        'n_items': len(subset)
                    }
        
        return analysis