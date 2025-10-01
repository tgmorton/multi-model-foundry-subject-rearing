"""
Item-level data aggregator for mixed effects model analysis.
Generates long-format data with item-level information across checkpoints.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ItemLevelAggregator:
    """Aggregate item-level data across checkpoints for mixed effects modeling."""
    
    @staticmethod
    def aggregate_blimp_items(output_dir: Path) -> pd.DataFrame:
        """
        Aggregate BLIMP item-level data across all checkpoints.
        
        Args:
            output_dir: Directory containing evaluation results
            
        Returns:
            DataFrame with item-level data in long format
        """
        all_data = []
        
        # Find all BLIMP detailed files
        blimp_files = sorted(output_dir.glob("checkpoint-*_blimp_detailed.jsonl"))
        
        for file_path in blimp_files:
            # Extract checkpoint info
            checkpoint_name = file_path.name.split('_')[0]
            checkpoint_num = int(checkpoint_name.split('-')[1])
            
            try:
                # Load detailed results
                df = pd.read_json(file_path, lines=True)
                
                # Add checkpoint information
                df['checkpoint'] = checkpoint_name
                df['checkpoint_num'] = checkpoint_num
                
                # Ensure we have key columns
                required_cols = ['UID', 'correct', 'surprisal_difference']
                if all(col in df.columns for col in required_cols):
                    # Add to collection
                    all_data.append(df)
                else:
                    logger.warning(f"Missing required columns in {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Reorganize columns for mixed effects modeling
        column_order = [
            'checkpoint', 'checkpoint_num', 'UID', 'pairID',
            'linguistics_term', 'field', 'correct', 'surprisal_difference',
            'sentence_good', 'sentence_bad'
        ]
        
        # Only include columns that exist
        column_order = [col for col in column_order if col in combined_df.columns]
        combined_df = combined_df[column_order]
        
        # Sort by checkpoint and item
        combined_df = combined_df.sort_values(['checkpoint_num', 'UID'])
        
        return combined_df
    
    @staticmethod
    def aggregate_null_subject_items(output_dir: Path) -> pd.DataFrame:
        """
        Aggregate null-subject item-level data across all checkpoints.
        
        Args:
            output_dir: Directory containing evaluation results
            
        Returns:
            DataFrame with item-level data in long format
        """
        all_data = []
        
        # Find all null-subject detailed files
        ns_files = sorted(output_dir.glob("checkpoint-*_null_subject_detailed.jsonl"))
        
        for file_path in ns_files:
            # Extract checkpoint info
            checkpoint_name = file_path.name.split('_')[0]
            checkpoint_num = int(checkpoint_name.split('-')[1])
            
            try:
                # Load detailed results
                df = pd.read_json(file_path, lines=True)
                
                # Add checkpoint information
                df['checkpoint'] = checkpoint_name
                df['checkpoint_num'] = checkpoint_num
                
                # Add to collection
                all_data.append(df)
                    
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Reorganize columns for mixed effects modeling
        base_columns = ['checkpoint', 'checkpoint_num']
        
        # Identify key columns if they exist
        id_columns = []
        for col in ['item_id', 'item', 'stimulus_id', 'UID']:
            if col in combined_df.columns:
                id_columns.append(col)
                break
        
        grouping_columns = []
        for col in ['item_group', 'form', 'person', 'number', 'verb_type', 'context_type']:
            if col in combined_df.columns:
                grouping_columns.append(col)
        
        outcome_columns = []
        for col in ['correct', 'surprisal_difference', 'null_surprisal', 'overt_surprisal']:
            if col in combined_df.columns:
                outcome_columns.append(col)
        
        # Combine column order
        column_order = base_columns + id_columns + grouping_columns + outcome_columns
        
        # Add any remaining columns
        remaining_cols = [col for col in combined_df.columns if col not in column_order]
        column_order.extend(remaining_cols)
        
        combined_df = combined_df[column_order]
        
        # Sort by checkpoint and item
        sort_cols = ['checkpoint_num'] + id_columns
        combined_df = combined_df.sort_values(sort_cols)
        
        return combined_df
    
    @staticmethod
    def create_nested_item_summaries(df: pd.DataFrame, task_type: str = 'null_subject') -> Dict[str, Any]:
        """
        Create nested summaries for items within forms/groups.
        
        Args:
            df: DataFrame with item-level data
            task_type: Type of task ('blimp' or 'null_subject')
            
        Returns:
            Nested dictionary with summaries
        """
        summaries = {
            'task': task_type,
            'by_form': {},
            'by_item': {},
            'by_form_and_item': {}
        }
        
        if task_type == 'null_subject' and 'form' in df.columns:
            # Summary by form across all checkpoints
            for form in df['form'].unique():
                form_df = df[df['form'] == form]
                summaries['by_form'][form] = {
                    'n_observations': len(form_df),
                    'n_items': form_df['item_id'].nunique() if 'item_id' in form_df.columns else None,
                    'n_checkpoints': form_df['checkpoint'].nunique(),
                    'accuracy': {
                        'mean': float(form_df['correct'].mean()),
                        'std': float(form_df['correct'].std()),
                        'se': float(form_df['correct'].std() / np.sqrt(len(form_df)))
                    }
                }
                
                # Add surprisal stats if available
                if 'surprisal_difference' in form_df.columns:
                    summaries['by_form'][form]['surprisal_diff'] = {
                        'mean': float(form_df['surprisal_difference'].mean()),
                        'std': float(form_df['surprisal_difference'].std()),
                        'se': float(form_df['surprisal_difference'].std() / np.sqrt(len(form_df)))
                    }
            
            # Summary by item across all checkpoints and forms
            if 'item_id' in df.columns:
                for item_id in df['item_id'].unique():
                    item_df = df[df['item_id'] == item_id]
                    summaries['by_item'][str(item_id)] = {
                        'n_observations': len(item_df),
                        'n_forms': item_df['form'].nunique() if 'form' in item_df.columns else None,
                        'n_checkpoints': item_df['checkpoint'].nunique(),
                        'accuracy': {
                            'mean': float(item_df['correct'].mean()),
                            'std': float(item_df['correct'].std()),
                            'se': float(item_df['correct'].std() / np.sqrt(len(item_df)))
                        }
                    }
                
                # Nested: by form and then by item
                for form in df['form'].unique():
                    summaries['by_form_and_item'][form] = {}
                    form_df = df[df['form'] == form]
                    
                    for item_id in form_df['item_id'].unique():
                        item_df = form_df[form_df['item_id'] == item_id]
                        summaries['by_form_and_item'][form][str(item_id)] = {
                            'n_checkpoints': item_df['checkpoint'].nunique(),
                            'accuracy_mean': float(item_df['correct'].mean()),
                            'accuracy_std': float(item_df['correct'].std()),
                            'checkpoints': item_df['checkpoint'].tolist(),
                            'correct_values': item_df['correct'].tolist()
                        }
                        
                        if 'surprisal_difference' in item_df.columns:
                            summaries['by_form_and_item'][form][str(item_id)]['surprisal_values'] = \
                                item_df['surprisal_difference'].tolist()
        
        elif task_type == 'blimp':
            # For BLIMP, organize by phenomenon and item
            if 'linguistics_term' in df.columns:
                for phenomenon in df['linguistics_term'].unique():
                    phenom_df = df[df['linguistics_term'] == phenomenon]
                    summaries['by_phenomenon'] = summaries.get('by_phenomenon', {})
                    summaries['by_phenomenon'][phenomenon] = {
                        'n_observations': len(phenom_df),
                        'n_items': phenom_df['UID'].nunique() if 'UID' in phenom_df.columns else None,
                        'n_checkpoints': phenom_df['checkpoint'].nunique(),
                        'accuracy': {
                            'mean': float(phenom_df['correct'].mean()),
                            'std': float(phenom_df['correct'].std()),
                            'se': float(phenom_df['correct'].std() / np.sqrt(len(phenom_df)))
                        }
                    }
        
        return summaries
    
    @staticmethod
    def create_mixed_effects_datasets(output_dir: Path):
        """
        Create datasets formatted for mixed effects modeling.
        
        Args:
            output_dir: Directory containing evaluation results
        """
        output_dir = Path(output_dir)
        
        # Process BLIMP data
        logger.info("Aggregating BLIMP item-level data...")
        blimp_df = ItemLevelAggregator.aggregate_blimp_items(output_dir)
        
        if not blimp_df.empty:
            # Save long-format data for mixed effects models
            blimp_long_file = output_dir / "blimp_item_level_long.csv"
            blimp_df.to_csv(blimp_long_file, index=False)
            logger.info(f"Saved BLIMP item-level data to {blimp_long_file}")
            
            # Create wide format for repeated measures
            if 'UID' in blimp_df.columns:
                blimp_wide = blimp_df.pivot_table(
                    index='UID',
                    columns='checkpoint',
                    values=['correct', 'surprisal_difference'],
                    aggfunc='first'
                )
                blimp_wide.columns = ['_'.join(col).strip() for col in blimp_wide.columns.values]
                blimp_wide_file = output_dir / "blimp_item_level_wide.csv"
                blimp_wide.to_csv(blimp_wide_file)
                logger.info(f"Saved BLIMP wide-format data to {blimp_wide_file}")
            
            # Generate nested summaries
            blimp_summaries = ItemLevelAggregator.create_nested_item_summaries(blimp_df, 'blimp')
            blimp_summary_file = output_dir / "blimp_item_summaries.json"
            with open(blimp_summary_file, 'w') as f:
                json.dump(blimp_summaries, f, indent=2)
        
        # Process null-subject data
        logger.info("Aggregating null-subject item-level data...")
        ns_df = ItemLevelAggregator.aggregate_null_subject_items(output_dir)
        
        if not ns_df.empty:
            # Save long-format data for mixed effects models
            ns_long_file = output_dir / "null_subject_item_level_long.csv"
            ns_df.to_csv(ns_long_file, index=False)
            logger.info(f"Saved null-subject item-level data to {ns_long_file}")
            
            # Create wide format if we have item identifiers
            id_col = None
            for col in ['item_id', 'item', 'stimulus_id']:
                if col in ns_df.columns:
                    id_col = col
                    break
            
            if id_col:
                # Create separate wide formats for each form if applicable
                if 'form' in ns_df.columns:
                    for form in ns_df['form'].unique():
                        form_df = ns_df[ns_df['form'] == form]
                        form_wide = form_df.pivot_table(
                            index=id_col,
                            columns='checkpoint',
                            values=['correct', 'surprisal_difference'],
                            aggfunc='first'
                        )
                        form_wide.columns = ['_'.join(col).strip() for col in form_wide.columns.values]
                        form_wide_file = output_dir / f"null_subject_{form}_item_level_wide.csv"
                        form_wide.to_csv(form_wide_file)
                        logger.info(f"Saved null-subject {form} wide-format data to {form_wide_file}")
                else:
                    # Single wide format
                    ns_wide = ns_df.pivot_table(
                        index=id_col,
                        columns='checkpoint',
                        values=['correct', 'surprisal_difference'],
                        aggfunc='first'
                    )
                    ns_wide.columns = ['_'.join(col).strip() for col in ns_wide.columns.values]
                    ns_wide_file = output_dir / "null_subject_item_level_wide.csv"
                    ns_wide.to_csv(ns_wide_file)
                    logger.info(f"Saved null-subject wide-format data to {ns_wide_file}")
            
            # Generate nested summaries
            ns_summaries = ItemLevelAggregator.create_nested_item_summaries(ns_df, 'null_subject')
            ns_summary_file = output_dir / "null_subject_item_summaries.json"
            with open(ns_summary_file, 'w') as f:
                json.dump(ns_summaries, f, indent=2)
        
        # Create R-ready mixed effects model dataset
        ItemLevelAggregator._create_lme4_datasets(output_dir, blimp_df, ns_df)
    
    @staticmethod
    def _create_lme4_datasets(output_dir: Path, blimp_df: pd.DataFrame, ns_df: pd.DataFrame):
        """
        Create datasets specifically formatted for lme4 in R.
        
        Args:
            output_dir: Output directory
            blimp_df: BLIMP dataframe
            ns_df: Null-subject dataframe
        """
        # BLIMP dataset for lme4
        if not blimp_df.empty:
            lme4_blimp = blimp_df.copy()
            
            # Center checkpoint number for better model convergence
            if 'checkpoint_num' in lme4_blimp.columns:
                lme4_blimp['checkpoint_centered'] = lme4_blimp['checkpoint_num'] - lme4_blimp['checkpoint_num'].mean()
                lme4_blimp['checkpoint_scaled'] = lme4_blimp['checkpoint_num'] / lme4_blimp['checkpoint_num'].max()
            
            # Add log-odds transformation of accuracy for better model properties
            lme4_blimp['correct_logit'] = lme4_blimp['correct'].apply(
                lambda x: np.log(x / (1 - x)) if 0 < x < 1 else (np.log(0.999/0.001) if x == 1 else np.log(0.001/0.999))
            )
            
            # Save
            lme4_blimp_file = output_dir / "blimp_lme4_ready.csv"
            lme4_blimp.to_csv(lme4_blimp_file, index=False)
            
            # Create R script template
            r_script = '''# R script for mixed effects analysis of BLIMP data
library(lme4)
library(lmerTest)
library(ggplot2)

# Load data
data <- read.csv("blimp_lme4_ready.csv")

# Convert factors
data$checkpoint <- as.factor(data$checkpoint)
data$linguistics_term <- as.factor(data$linguistics_term)
data$UID <- as.factor(data$UID)

# Basic mixed effects model
# Random intercepts for items, fixed effect of checkpoint
model1 <- glmer(correct ~ checkpoint_scaled + (1|UID) + (1|linguistics_term), 
                data = data, family = binomial)

# More complex model with random slopes
model2 <- glmer(correct ~ checkpoint_scaled + (checkpoint_scaled|UID) + (1|linguistics_term), 
                data = data, family = binomial)

# Model comparison
anova(model1, model2)

# Summary
summary(model1)

# Plot predictions
data$predicted <- predict(model1, type = "response")
ggplot(data, aes(x = checkpoint_num, y = predicted, group = linguistics_term)) +
  stat_summary(fun = mean, geom = "line") +
  facet_wrap(~linguistics_term)
'''
            
            r_script_file = output_dir / "blimp_analysis.R"
            with open(r_script_file, 'w') as f:
                f.write(r_script)
        
        # Null-subject dataset for lme4
        if not ns_df.empty:
            lme4_ns = ns_df.copy()
            
            # Center checkpoint number
            if 'checkpoint_num' in lme4_ns.columns:
                lme4_ns['checkpoint_centered'] = lme4_ns['checkpoint_num'] - lme4_ns['checkpoint_num'].mean()
                lme4_ns['checkpoint_scaled'] = lme4_ns['checkpoint_num'] / lme4_ns['checkpoint_num'].max()
            
            # Add log-odds transformation
            lme4_ns['correct_logit'] = lme4_ns['correct'].apply(
                lambda x: np.log(x / (1 - x)) if 0 < x < 1 else (np.log(0.999/0.001) if x == 1 else np.log(0.001/0.999))
            )
            
            # Save
            lme4_ns_file = output_dir / "null_subject_lme4_ready.csv"
            lme4_ns.to_csv(lme4_ns_file, index=False)
            
            # Create R script template
            r_script = '''# R script for mixed effects analysis of null-subject data
library(lme4)
library(lmerTest)
library(ggplot2)

# Load data
data <- read.csv("null_subject_lme4_ready.csv")

# Convert factors
data$checkpoint <- as.factor(data$checkpoint)
data$form <- as.factor(data$form)
data$item_group <- as.factor(data$item_group)

# Identify item column
item_col <- if("item_id" %in% names(data)) "item_id" else "item"
data[[item_col]] <- as.factor(data[[item_col]])

# Mixed effects model with form contrast
model1 <- glmer(correct ~ checkpoint_scaled * form + (1|item_id) + (1|item_group), 
                data = data, family = binomial)

# Model with random slopes for form effect
model2 <- glmer(correct ~ checkpoint_scaled * form + (form|item_id) + (1|item_group), 
                data = data, family = binomial)

# Model comparison
anova(model1, model2)

# Summary
summary(model1)

# Test form effect at different checkpoints
library(emmeans)
emmeans(model1, pairwise ~ form | checkpoint)

# Plot
ggplot(data, aes(x = checkpoint_num, y = correct, color = form)) +
  stat_summary(fun = mean, geom = "point") +
  stat_summary(fun = mean, geom = "line") +
  facet_wrap(~item_group) +
  theme_minimal()
'''
            
            r_script_file = output_dir / "null_subject_analysis.R"
            with open(r_script_file, 'w') as f:
                f.write(r_script)