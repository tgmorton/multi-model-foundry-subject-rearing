#!/usr/bin/env python3
"""
Compile detailed evaluation files into comprehensive summary files for each model.
This script processes the existing detailed JSONL files and creates all the summary formats
needed for mixed effects analysis.
"""

import json
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import argparse


def extract_checkpoint_number(checkpoint_name: str) -> int:
    """Extract checkpoint number from checkpoint name."""
    return int(checkpoint_name.split('-')[1])


def process_blimp_data(model_dir: Path) -> Tuple[Dict, List[Dict]]:
    """Process all BLIMP detailed files in a model directory."""
    checkpoints = []
    all_items = []
    item_data = defaultdict(lambda: defaultdict(dict))
    
    # Find all BLIMP files
    blimp_files = sorted(model_dir.glob("checkpoint-*_blimp_detailed.jsonl"))
    
    for file_path in blimp_files:
        checkpoint_name = file_path.stem.replace("_blimp_detailed", "")
        checkpoint_num = extract_checkpoint_number(checkpoint_name)
        checkpoints.append(checkpoint_name)
        
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                
                # Create unique item ID
                item_id = f"{item['filename']}_{item['pairID']}"
                item_group = item.get('linguistics_term', item.get('filename', 'unknown'))
                
                # Store for long format
                all_items.append({
                    'checkpoint': checkpoint_name,
                    'checkpoint_num': checkpoint_num,
                    'item_id': item_id,
                    'item_group': item_group,
                    'phenomenon': item.get('filename', 'unknown'),
                    'field': item.get('field', 'unknown'),
                    'linguistics_term': item.get('linguistics_term', 'unknown'),
                    'correct': int(item['correct']),
                    'surprisal_difference': item['surprisal_difference'],
                    'good_surprisal': item['good_surprisal'],
                    'bad_surprisal': item['bad_surprisal']
                })
                
                # Store for wide format
                item_data[item_id][checkpoint_name] = {
                    'correct': int(item['correct']),
                    'surprisal_difference': item['surprisal_difference']
                }
                
                # Store metadata
                if 'metadata' not in item_data[item_id]:
                    item_data[item_id]['metadata'] = {
                        'item_group': item_group,
                        'phenomenon': item.get('filename', 'unknown'),
                        'field': item.get('field', 'unknown'),
                        'linguistics_term': item.get('linguistics_term', 'unknown')
                    }
    
    return item_data, all_items


def process_null_subject_data(model_dir: Path) -> Tuple[Dict, List[Dict]]:
    """Process all null_subject detailed files in a model directory."""
    checkpoints = []
    all_items = []
    item_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Find all null_subject files
    ns_files = sorted(model_dir.glob("checkpoint-*_null_subject_detailed.jsonl"))
    
    for file_path in ns_files:
        checkpoint_name = file_path.stem.replace("_null_subject_detailed", "")
        checkpoint_num = extract_checkpoint_number(checkpoint_name)
        checkpoints.append(checkpoint_name)
        
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                
                # Create unique item ID
                item_id = f"item_{item['item']:03d}"
                item_group = item['item_group']
                form = item['form']
                
                # Store for long format - both null and overt forms
                for form_type in ['null', 'overt']:
                    # Handle potential None values
                    surprisal_diff = item.get('surprisal_difference', 0)
                    hotspot_diff = item.get('hotspot_difference', 0)
                    
                    all_items.append({
                        'checkpoint': checkpoint_name,
                        'checkpoint_num': checkpoint_num,
                        'item_id': item_id,
                        'item_group': item_group,
                        'form': form,
                        'form_type': form_type,
                        'correct': int(item['prefers_overt']) if form_type == 'overt' else int(not item['prefers_overt']),
                        'surprisal_difference': surprisal_diff if form_type == 'null' else -surprisal_diff if surprisal_diff else 0,
                        'mean_surprisal': item.get(f'{form_type}_mean_surprisal', 0),
                        'hotspot_surprisal': item.get(f'{form_type}_hotspot_surprisal', 0),
                        'hotspot_difference': hotspot_diff if form_type == 'null' else -hotspot_diff if hotspot_diff else 0
                    })
                
                # Store for wide format
                item_data[form][item_id][checkpoint_name] = {
                    'prefers_overt': int(item['prefers_overt']),
                    'surprisal_difference': item['surprisal_difference'],
                    'null_mean_surprisal': item['null_mean_surprisal'],
                    'overt_mean_surprisal': item['overt_mean_surprisal']
                }
                
                # Store metadata
                if 'metadata' not in item_data[form][item_id]:
                    item_data[form][item_id]['metadata'] = {
                        'item_group': item_group,
                        'form': form,
                        'context': item.get('context', ''),
                        'null_target': item.get('null_target', ''),
                        'overt_target': item.get('overt_target', '')
                    }
    
    return item_data, all_items


def create_lme4_ready_data(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """Create LME4-ready dataset with preprocessing."""
    df = df.copy()
    
    # Center and scale checkpoint numbers
    df['checkpoint_centered'] = df['checkpoint_num'] - df['checkpoint_num'].mean()
    df['checkpoint_scaled'] = (df['checkpoint_num'] - df['checkpoint_num'].mean()) / df['checkpoint_num'].std()
    
    # Add logit transformation of accuracy (with adjustment for 0 and 1)
    epsilon = 0.001
    df['correct_adjusted'] = df['correct'].clip(epsilon, 1 - epsilon)
    df['correct_logit'] = np.log(df['correct_adjusted'] / (1 - df['correct_adjusted']))
    
    # Add polynomial terms for checkpoint
    df['checkpoint_squared'] = df['checkpoint_scaled'] ** 2
    df['checkpoint_cubed'] = df['checkpoint_scaled'] ** 3
    
    if task_type == 'null_subject':
        # Add form contrasts (treatment coding)
        df['form_contrast'] = (df['form_type'] == 'overt').astype(int)
        
        # Add interaction terms
        df['checkpoint_x_form'] = df['checkpoint_scaled'] * df['form_contrast']
        df['checkpoint_squared_x_form'] = df['checkpoint_squared'] * df['form_contrast']
    
    return df




def create_comprehensive_summary(model_dir: Path, blimp_items: List[Dict], ns_items: List[Dict]) -> Dict:
    """Create comprehensive summary JSON."""
    summary = {
        'model': model_dir.name,
        'checkpoints': {},
        'phenomena': {},
        'overall': {}
    }
    
    # Process BLIMP data
    if blimp_items:
        blimp_df = pd.DataFrame(blimp_items)
        
        # Checkpoint-level summaries
        for checkpoint in blimp_df['checkpoint'].unique():
            cp_data = blimp_df[blimp_df['checkpoint'] == checkpoint]
            summary['checkpoints'][checkpoint] = {
                'blimp': {
                    'accuracy': cp_data['correct'].mean(),
                    'n_items': len(cp_data),
                    'mean_surprisal_diff': cp_data['surprisal_difference'].mean(),
                    'std_surprisal_diff': cp_data['surprisal_difference'].std()
                }
            }
        
        # Phenomenon-level summaries
        for phenomenon in blimp_df['phenomenon'].unique():
            ph_data = blimp_df[blimp_df['phenomenon'] == phenomenon]
            summary['phenomena'][phenomenon] = {
                'accuracy_trajectory': ph_data.groupby('checkpoint_num')['correct'].mean().to_dict(),
                'n_items': ph_data['item_id'].nunique(),
                'field': ph_data['field'].iloc[0] if 'field' in ph_data else 'unknown'
            }
    
    # Process null_subject data
    if ns_items:
        ns_df = pd.DataFrame(ns_items)
        
        # Add to checkpoint summaries
        for checkpoint in ns_df['checkpoint'].unique():
            cp_data = ns_df[ns_df['checkpoint'] == checkpoint]
            
            if checkpoint not in summary['checkpoints']:
                summary['checkpoints'][checkpoint] = {}
            
            summary['checkpoints'][checkpoint]['null_subject'] = {
                'overall_accuracy': cp_data['correct'].mean(),
                'null_preference': cp_data[cp_data['form_type'] == 'null']['correct'].mean(),
                'overt_preference': cp_data[cp_data['form_type'] == 'overt']['correct'].mean(),
                'n_items': cp_data['item_id'].nunique(),
                'forms': {}
            }
            
            # Form-specific summaries
            for form in cp_data['form'].unique():
                form_data = cp_data[cp_data['form'] == form]
                summary['checkpoints'][checkpoint]['null_subject']['forms'][form] = {
                    'null_accuracy': form_data[form_data['form_type'] == 'null']['correct'].mean(),
                    'overt_accuracy': form_data[form_data['form_type'] == 'overt']['correct'].mean()
                }
    
    # Overall summaries
    if blimp_items:
        blimp_df = pd.DataFrame(blimp_items)
        summary['overall']['blimp'] = {
            'mean_accuracy': blimp_df['correct'].mean(),
            'final_accuracy': blimp_df[blimp_df['checkpoint_num'] == blimp_df['checkpoint_num'].max()]['correct'].mean(),
            'n_checkpoints': blimp_df['checkpoint'].nunique(),
            'n_items': blimp_df['item_id'].nunique()
        }
    
    if ns_items:
        ns_df = pd.DataFrame(ns_items)
        summary['overall']['null_subject'] = {
            'mean_accuracy': ns_df['correct'].mean(),
            'null_preference': ns_df[ns_df['form_type'] == 'null']['correct'].mean(),
            'overt_preference': ns_df[ns_df['form_type'] == 'overt']['correct'].mean(),
            'n_checkpoints': ns_df['checkpoint'].nunique(),
            'n_items': ns_df['item_id'].nunique()
        }
    
    return summary


def process_model_directory(model_dir: Path):
    """Process a single model directory and create all summary files."""
    print(f"Processing {model_dir.name}...")
    
    # Process BLIMP data
    blimp_item_data, blimp_items = process_blimp_data(model_dir)
    
    if blimp_items:
        # Create long format CSV
        blimp_df = pd.DataFrame(blimp_items)
        blimp_df.to_csv(model_dir / "blimp_item_level_long.csv", index=False)
        
        # Create wide format CSV
        wide_data = []
        for item_id, item_info in blimp_item_data.items():
            row = {'item_id': item_id}
            row.update(item_info['metadata'])
            
            for checkpoint, values in item_info.items():
                if checkpoint != 'metadata':
                    row[f"{checkpoint}_correct"] = values['correct']
                    row[f"{checkpoint}_surprisal_diff"] = values['surprisal_difference']
            
            wide_data.append(row)
        
        if wide_data:
            wide_df = pd.DataFrame(wide_data)
            wide_df.to_csv(model_dir / "blimp_item_level_wide.csv", index=False)
        
        # Create LME4-ready dataset
        lme4_df = create_lme4_ready_data(blimp_df, 'blimp')
        lme4_df.to_csv(model_dir / "blimp_lme4_ready.csv", index=False)
        
        # Create item summaries JSON
        item_summaries = {}
        for phenomenon in blimp_df['phenomenon'].unique():
            ph_data = blimp_df[blimp_df['phenomenon'] == phenomenon]
            item_summaries[phenomenon] = {
                'items': {},
                'summary': {
                    'mean_accuracy': ph_data['correct'].mean(),
                    'n_items': ph_data['item_id'].nunique(),
                    'n_checkpoints': ph_data['checkpoint'].nunique()
                }
            }
            
            for item_id in ph_data['item_id'].unique():
                item_data = ph_data[ph_data['item_id'] == item_id]
                item_summaries[phenomenon]['items'][item_id] = {
                    'accuracy_trajectory': item_data.set_index('checkpoint_num')['correct'].to_dict(),
                    'mean_accuracy': item_data['correct'].mean()
                }
        
        with open(model_dir / "blimp_item_summaries.json", 'w') as f:
            json.dump(item_summaries, f, indent=2)
    
    # Process null_subject data
    ns_item_data, ns_items = process_null_subject_data(model_dir)
    
    if ns_items:
        # Create long format CSV
        ns_df = pd.DataFrame(ns_items)
        ns_df.to_csv(model_dir / "null_subject_item_level_long.csv", index=False)
        
        # Create wide format CSVs per form
        for form, form_items in ns_item_data.items():
            wide_data = []
            for item_id, item_info in form_items.items():
                row = {'item_id': item_id}
                row.update(item_info['metadata'])
                
                for checkpoint, values in item_info.items():
                    if checkpoint != 'metadata':
                        row[f"{checkpoint}_prefers_overt"] = values['prefers_overt']
                        row[f"{checkpoint}_surprisal_diff"] = values['surprisal_difference']
                
                wide_data.append(row)
            
            if wide_data:
                wide_df = pd.DataFrame(wide_data)
                safe_form = form.replace(' ', '_').replace('/', '_')
                wide_df.to_csv(model_dir / f"null_subject_{safe_form}_item_level_wide.csv", index=False)
        
        # Create LME4-ready dataset
        lme4_df = create_lme4_ready_data(ns_df, 'null_subject')
        lme4_df.to_csv(model_dir / "null_subject_lme4_ready.csv", index=False)
        
        # Create item summaries JSON
        item_summaries = {}
        for form in ns_df['form'].unique():
            form_data = ns_df[ns_df['form'] == form]
            item_summaries[form] = {
                'items': {},
                'summary': {
                    'null_preference': form_data[form_data['form_type'] == 'null']['correct'].mean(),
                    'overt_preference': form_data[form_data['form_type'] == 'overt']['correct'].mean(),
                    'n_items': form_data['item_id'].nunique()
                }
            }
            
            for item_id in form_data['item_id'].unique():
                item_data = form_data[form_data['item_id'] == item_id]
                item_summaries[form]['items'][item_id] = {
                    'null_trajectory': item_data[item_data['form_type'] == 'null'].set_index('checkpoint_num')['correct'].to_dict(),
                    'overt_trajectory': item_data[item_data['form_type'] == 'overt'].set_index('checkpoint_num')['correct'].to_dict()
                }
        
        with open(model_dir / "null_subject_item_summaries.json", 'w') as f:
            json.dump(item_summaries, f, indent=2)
    
    # Create comprehensive summary
    comprehensive_summary = create_comprehensive_summary(model_dir, blimp_items, ns_items)
    with open(model_dir / "comprehensive_summary.json", 'w') as f:
        json.dump(comprehensive_summary, f, indent=2)
    
    print(f"Completed {model_dir.name}")
    return blimp_items, ns_items


def main():
    parser = argparse.ArgumentParser(description="Compile model evaluation summaries")
    parser.add_argument("--results-dir", type=str, default="evaluation/results",
                      help="Path to results directory")
    parser.add_argument("--models", nargs="+", help="Specific models to process")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        sys.exit(1)
    
    # Get model directories
    if args.models:
        model_dirs = [results_dir / model for model in args.models]
    else:
        model_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("exp")])
    
    # Process each model
    for model_dir in model_dirs:
        if model_dir.exists():
            try:
                process_model_directory(model_dir)
            except Exception as e:
                print(f"Error processing {model_dir.name}: {e}")
                continue
    
    print("All models processed successfully!")


if __name__ == "__main__":
    main()