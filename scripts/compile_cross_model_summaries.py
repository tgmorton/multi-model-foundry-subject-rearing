#!/usr/bin/env python3
"""
Compile cross-model summary CSV files combining all models' data.
Creates master CSV files for BLIMP and null_subject data across all models.
"""

import json
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import argparse


def extract_checkpoint_number(checkpoint_name: str) -> int:
    """Extract checkpoint number from checkpoint name."""
    return int(checkpoint_name.split('-')[1])


def compile_blimp_cross_model(results_dir: Path, model_dirs: List[Path]) -> pd.DataFrame:
    """Compile BLIMP data across all models."""
    all_data = []
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"  Processing BLIMP data for {model_name}...")
        
        # Find all BLIMP detailed files
        blimp_files = sorted(model_dir.glob("checkpoint-*_blimp_detailed.jsonl"))
        
        for file_path in blimp_files:
            checkpoint_name = file_path.stem.replace("_blimp_detailed", "")
            checkpoint_num = extract_checkpoint_number(checkpoint_name)
            
            with open(file_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    
                    # Create record with model name
                    record = {
                        'model': model_name,
                        'checkpoint': checkpoint_name,
                        'checkpoint_num': checkpoint_num,
                        'item_id': f"{item['filename']}_{item['pairID']}",
                        'phenomenon': item.get('filename', 'unknown'),
                        'field': item.get('field', 'unknown'),
                        'linguistics_term': item.get('linguistics_term', 'unknown'),
                        'correct': int(item['correct']),
                        'surprisal_difference': item['surprisal_difference'],
                        'good_surprisal': item['good_surprisal'],
                        'bad_surprisal': item['bad_surprisal']
                    }
                    
                    all_data.append(record)
    
    return pd.DataFrame(all_data)


def compile_null_subject_cross_model(results_dir: Path, model_dirs: List[Path]) -> pd.DataFrame:
    """Compile null_subject data across all models."""
    all_data = []
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"  Processing null_subject data for {model_name}...")
        
        # Find all null_subject detailed files
        ns_files = sorted(model_dir.glob("checkpoint-*_null_subject_detailed.jsonl"))
        
        for file_path in ns_files:
            checkpoint_name = file_path.stem.replace("_null_subject_detailed", "")
            checkpoint_num = extract_checkpoint_number(checkpoint_name)
            
            with open(file_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    
                    # Create records for both null and overt forms
                    item_id = f"item_{item['item']:03d}"
                    
                    # Handle potential None values
                    surprisal_diff = item.get('surprisal_difference', 0)
                    hotspot_diff = item.get('hotspot_difference', 0)
                    
                    # Null form record
                    null_record = {
                        'model': model_name,
                        'checkpoint': checkpoint_name,
                        'checkpoint_num': checkpoint_num,
                        'item_id': item_id,
                        'item_group': item['item_group'],
                        'form': item['form'],
                        'form_type': 'null',
                        'correct': int(not item['prefers_overt']),
                        'surprisal_difference': surprisal_diff if surprisal_diff else 0,
                        'mean_surprisal': item.get('null_mean_surprisal', 0),
                        'hotspot_surprisal': item.get('null_hotspot_surprisal', 0),
                        'hotspot_difference': hotspot_diff if hotspot_diff else 0,
                        'context': item.get('context', ''),
                        'target': item.get('null_target', ''),
                        'hotspot': item.get('hotspot', '')
                    }
                    all_data.append(null_record)
                    
                    # Overt form record
                    overt_record = {
                        'model': model_name,
                        'checkpoint': checkpoint_name,
                        'checkpoint_num': checkpoint_num,
                        'item_id': item_id,
                        'item_group': item['item_group'],
                        'form': item['form'],
                        'form_type': 'overt',
                        'correct': int(item['prefers_overt']),
                        'surprisal_difference': -surprisal_diff if surprisal_diff else 0,
                        'mean_surprisal': item.get('overt_mean_surprisal', 0),
                        'hotspot_surprisal': item.get('overt_hotspot_surprisal', 0),
                        'hotspot_difference': -hotspot_diff if hotspot_diff else 0,
                        'context': item.get('context', ''),
                        'target': item.get('overt_target', ''),
                        'hotspot': item.get('hotspot', '')
                    }
                    all_data.append(overt_record)
    
    return pd.DataFrame(all_data)


def create_summary_statistics(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """Create summary statistics DataFrame."""
    
    if task_type == 'blimp':
        # Group by model and checkpoint
        summary = df.groupby(['model', 'checkpoint_num']).agg({
            'correct': ['mean', 'std', 'count'],
            'surprisal_difference': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['model', 'checkpoint_num', 'accuracy_mean', 'accuracy_std', 
                          'n_items', 'surprisal_diff_mean', 'surprisal_diff_std']
        
        # Add phenomenon-level summaries
        phenomenon_summary = df.groupby(['model', 'checkpoint_num', 'phenomenon']).agg({
            'correct': 'mean'
        }).reset_index()
        phenomenon_summary = phenomenon_summary.pivot(
            index=['model', 'checkpoint_num'],
            columns='phenomenon',
            values='correct'
        ).reset_index()
        
        # Merge summaries
        summary = summary.merge(phenomenon_summary, on=['model', 'checkpoint_num'], how='left')
        
    else:  # null_subject
        # Overall summary
        summary = df.groupby(['model', 'checkpoint_num', 'form_type']).agg({
            'correct': ['mean', 'std', 'count'],
            'surprisal_difference': ['mean', 'std'],
            'mean_surprisal': 'mean'
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['model', 'checkpoint_num', 'form_type', 'accuracy_mean', 
                          'accuracy_std', 'n_items', 'surprisal_diff_mean', 
                          'surprisal_diff_std', 'mean_surprisal']
        
        # Pivot to get null vs overt in same row
        summary_wide = summary.pivot(
            index=['model', 'checkpoint_num'],
            columns='form_type',
            values=['accuracy_mean', 'surprisal_diff_mean', 'mean_surprisal']
        ).reset_index()
        
        # Flatten column names
        summary_wide.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                for col in summary_wide.columns.values]
        
        # Add form-specific summaries
        form_summary = df.groupby(['model', 'checkpoint_num', 'form', 'form_type']).agg({
            'correct': 'mean'
        }).reset_index()
        
        # Create separate columns for each form
        for form in df['form'].unique():
            form_data = form_summary[form_summary['form'] == form]
            form_pivot = form_data.pivot(
                index=['model', 'checkpoint_num'],
                columns='form_type',
                values='correct'
            ).reset_index()
            
            safe_form = form.replace(' ', '_').replace('/', '_')
            form_pivot.columns = ['model', 'checkpoint_num', 
                                 f'{safe_form}_null_accuracy', f'{safe_form}_overt_accuracy']
            
            summary_wide = summary_wide.merge(
                form_pivot[['model', 'checkpoint_num', f'{safe_form}_null_accuracy', 
                          f'{safe_form}_overt_accuracy']],
                on=['model', 'checkpoint_num'],
                how='left'
            )
        
        summary = summary_wide
    
    return summary


def create_lme4_ready_cross_model(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """Create LME4-ready dataset for cross-model analysis."""
    df = df.copy()
    
    # Create model contrasts (treatment coding with baseline as reference)
    baseline_model = sorted(df['model'].unique())[0]  # Use first model alphabetically as baseline
    df['is_baseline'] = (df['model'] == baseline_model).astype(int)
    
    # Create dummy variables for each model
    for model in df['model'].unique():
        df[f'model_{model}'] = (df['model'] == model).astype(int)
    
    # Center and scale checkpoint numbers (within each model)
    for model in df['model'].unique():
        model_mask = df['model'] == model
        df.loc[model_mask, 'checkpoint_centered'] = (
            df.loc[model_mask, 'checkpoint_num'] - 
            df.loc[model_mask, 'checkpoint_num'].mean()
        )
        df.loc[model_mask, 'checkpoint_scaled'] = (
            df.loc[model_mask, 'checkpoint_centered'] / 
            df.loc[model_mask, 'checkpoint_num'].std()
        )
    
    # Add polynomial terms
    df['checkpoint_squared'] = df['checkpoint_scaled'] ** 2
    df['checkpoint_cubed'] = df['checkpoint_scaled'] ** 3
    
    # Add logit transformation
    epsilon = 0.001
    df['correct_adjusted'] = df['correct'].clip(epsilon, 1 - epsilon)
    df['correct_logit'] = np.log(df['correct_adjusted'] / (1 - df['correct_adjusted']))
    
    if task_type == 'null_subject':
        # Add form contrasts
        df['form_contrast'] = (df['form_type'] == 'overt').astype(int)
        
        # Add interaction terms
        df['checkpoint_x_form'] = df['checkpoint_scaled'] * df['form_contrast']
        df['model_x_form'] = df['is_baseline'] * df['form_contrast']
        df['checkpoint_x_model_x_form'] = df['checkpoint_scaled'] * df['is_baseline'] * df['form_contrast']
    
    # Add model × checkpoint interactions
    df['checkpoint_x_model'] = df['checkpoint_scaled'] * df['is_baseline']
    
    return df




def main():
    parser = argparse.ArgumentParser(description="Compile cross-model evaluation summaries")
    parser.add_argument("--results-dir", type=str, default="evaluation/results",
                      help="Path to results directory")
    parser.add_argument("--output-dir", type=str, default="evaluation/results",
                      help="Directory for output files")
    parser.add_argument("--models", nargs="+", help="Specific models to include")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model directories
    if args.models:
        model_dirs = [results_dir / model for model in args.models]
    else:
        model_dirs = sorted([d for d in results_dir.iterdir() 
                           if d.is_dir() and d.name.startswith("exp") and not d.name.endswith(".bak")])
    
    print(f"Found {len(model_dirs)} models to process")
    
    # Compile BLIMP data
    print("\nCompiling BLIMP data across models...")
    blimp_df = compile_blimp_cross_model(results_dir, model_dirs)
    
    if not blimp_df.empty:
        # Save raw combined data
        blimp_df.to_csv(output_dir / "all_models_blimp_raw.csv", index=False)
        print(f"  Saved {len(blimp_df)} BLIMP records")
        
        # Create summary statistics
        blimp_summary = create_summary_statistics(blimp_df, 'blimp')
        blimp_summary.to_csv(output_dir / "all_models_blimp_summary.csv", index=False)
        print(f"  Created summary with {len(blimp_summary)} checkpoint records")
        
        # Create LME4-ready dataset
        blimp_lme4 = create_lme4_ready_cross_model(blimp_df, 'blimp')
        blimp_lme4.to_csv(output_dir / "all_models_blimp_lme4_ready.csv", index=False)
        print(f"  Created LME4-ready dataset")
    
    # Compile null_subject data
    print("\nCompiling null_subject data across models...")
    ns_df = compile_null_subject_cross_model(results_dir, model_dirs)
    
    if not ns_df.empty:
        # Save raw combined data
        ns_df.to_csv(output_dir / "all_models_null_subject_raw.csv", index=False)
        print(f"  Saved {len(ns_df)} null_subject records")
        
        # Create summary statistics
        ns_summary = create_summary_statistics(ns_df, 'null_subject')
        ns_summary.to_csv(output_dir / "all_models_null_subject_summary.csv", index=False)
        print(f"  Created summary with {len(ns_summary)} checkpoint records")
        
        # Create LME4-ready dataset
        ns_lme4 = create_lme4_ready_cross_model(ns_df, 'null_subject')
        ns_lme4.to_csv(output_dir / "all_models_null_subject_lme4_ready.csv", index=False)
        print(f"  Created LME4-ready dataset")
    
    # Create master summary JSON
    master_summary = {
        'models': [m.name for m in model_dirs],
        'blimp': {
            'total_records': len(blimp_df) if not blimp_df.empty else 0,
            'n_models': blimp_df['model'].nunique() if not blimp_df.empty else 0,
            'n_checkpoints': blimp_df['checkpoint'].nunique() if not blimp_df.empty else 0,
            'n_items': blimp_df['item_id'].nunique() if not blimp_df.empty else 0,
            'phenomena': list(blimp_df['phenomenon'].unique()) if not blimp_df.empty else []
        },
        'null_subject': {
            'total_records': len(ns_df) if not ns_df.empty else 0,
            'n_models': ns_df['model'].nunique() if not ns_df.empty else 0,
            'n_checkpoints': ns_df['checkpoint'].nunique() if not ns_df.empty else 0,
            'n_items': ns_df['item_id'].nunique() if not ns_df.empty else 0,
            'forms': list(ns_df['form'].unique()) if not ns_df.empty else []
        }
    }
    
    with open(output_dir / "all_models_summary.json", 'w') as f:
        json.dump(master_summary, f, indent=2)
    
    print(f"\nAll cross-model summaries created successfully!")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()