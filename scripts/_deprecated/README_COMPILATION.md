# Evaluation Summary Compilation Scripts

## Overview

These scripts compile the detailed evaluation results from individual checkpoint files into comprehensive summary files suitable for statistical analysis, particularly mixed effects modeling in R.

## Scripts

### 1. `compile_model_summaries.py`
Processes individual model directories to create summary files for each model.

**Usage:**
```bash
# Process all models
python scripts/compile_model_summaries.py

# Process specific models
python scripts/compile_model_summaries.py --models exp0_baseline exp1_remove_expletives

# Specify custom results directory
python scripts/compile_model_summaries.py --results-dir path/to/results
```

**Files Created per Model:**
- `blimp_item_level_long.csv` - Long format data for mixed effects analysis
- `blimp_item_level_wide.csv` - Wide format with checkpoints as columns
- `blimp_lme4_ready.csv` - Pre-processed data with scaled predictors
- `blimp_item_summaries.json` - Nested JSON summaries by phenomenon
- `blimp_analysis.R` - R script template for analysis
- `null_subject_item_level_long.csv` - Long format with form contrasts
- `null_subject_*_item_level_wide.csv` - Wide format per form type
- `null_subject_lme4_ready.csv` - Ready for mixed effects regression
- `null_subject_item_summaries.json` - Form-specific summaries
- `null_subject_analysis.R` - R analysis template
- `comprehensive_summary.json` - Complete model summary

### 2. `compile_cross_model_summaries.py`
Combines data across multiple models for comparative analysis.

**Usage:**
```bash
# Process all models
python scripts/compile_cross_model_summaries.py

# Process specific models
python scripts/compile_cross_model_summaries.py --models exp0_baseline exp1_remove_expletives

# Specify output directory
python scripts/compile_cross_model_summaries.py --output-dir analysis/results
```

**Files Created:**
- `all_models_blimp_raw.csv` - Combined raw BLIMP data
- `all_models_blimp_summary.csv` - Summary statistics by model/checkpoint
- `all_models_blimp_lme4_ready.csv` - Ready for cross-model analysis
- `all_models_blimp_analysis.R` - R script for model comparisons
- `all_models_null_subject_raw.csv` - Combined null subject data
- `all_models_null_subject_summary.csv` - Form preferences by model
- `all_models_null_subject_lme4_ready.csv` - With model contrasts
- `all_models_null_subject_analysis.R` - Cross-model R analysis
- `all_models_summary.json` - Master summary file

## Data Formats

### Long Format CSV Structure

**BLIMP:**
```
checkpoint,checkpoint_num,item_id,item_group,phenomenon,field,linguistics_term,correct,surprisal_difference,good_surprisal,bad_surprisal
checkpoint-0,0,adjunct_island_0,island_effects,adjunct_island,syntax,island_effects,0,-0.107,15.689,15.582
```

**Null Subject:**
```
checkpoint,checkpoint_num,item_id,item_group,form,form_type,correct,surprisal_difference,mean_surprisal,hotspot_surprisal,hotspot_difference
checkpoint-0,0,item_000,1a_3rdSG,default,null,1,0.284,15.524,15.408,-0.372
checkpoint-0,0,item_000,1a_3rdSG,default,overt,0,-0.284,15.808,15.779,0.372
```

### LME4-Ready Features

Both datasets include:
- Centered and scaled checkpoint numbers
- Polynomial terms (squared, cubed) for growth curves
- Logit transformations of accuracy
- Treatment-coded contrasts for categorical variables
- Interaction terms for factorial designs

## R Analysis Examples

### Basic Mixed Effects Model (BLIMP)
```r
library(lme4)
data <- read.csv("blimp_lme4_ready.csv")

model <- glmer(correct ~ checkpoint_scaled + 
               (1 | item_id) + 
               (1 | item_group),
               data = data,
               family = binomial)
```

### Form × Checkpoint Interaction (Null Subject)
```r
data <- read.csv("null_subject_lme4_ready.csv")

model <- glmer(correct ~ checkpoint_scaled * form_contrast + 
               (form_contrast | item_id) + 
               (1 | item_group),
               data = data,
               family = binomial)
```

### Cross-Model Comparison
```r
data <- read.csv("all_models_blimp_lme4_ready.csv")

model <- glmer(correct ~ checkpoint_scaled * model + 
               (1 | item_id),
               data = data,
               family = binomial)
```

## Requirements

- Python 3.7+
- pandas
- numpy
- pathlib (standard library)
- json (standard library)

## Notes

1. **Processing Time**: Full compilation can take several minutes for large datasets
2. **Memory Usage**: Cross-model files can be large (>500MB for complete datasets)
3. **Missing Data**: Scripts handle missing values gracefully with defaults
4. **File Naming**: Form names with spaces/slashes are sanitized for filenames
5. **Checkpoint Extraction**: Assumes checkpoint names follow pattern "checkpoint-{number}"

## Troubleshooting

**Error: "Results directory does not exist"**
- Ensure you're running from the project root directory
- Check that evaluation/results/ exists

**Error: "bad operand type for unary -"**
- Fixed in current version - handles None values in surprisal differences

**Large file sizes**
- Use --models flag to process subset of models
- Consider sampling checkpoints for initial exploration

**R script errors**
- Ensure required R packages are installed: lme4, lmerTest, emmeans, ggplot2, tidyverse
- Adjust optimizer settings if convergence fails

## Output Directory Structure
```
evaluation/results/
├── exp0_baseline/
│   ├── checkpoint-*_blimp_detailed.jsonl     # Original detailed files
│   ├── checkpoint-*_null_subject_detailed.jsonl
│   ├── blimp_item_level_long.csv            # Generated summaries
│   ├── blimp_lme4_ready.csv
│   ├── comprehensive_summary.json
│   └── ...
├── exp1_remove_expletives/
│   └── ... (same structure)
├── all_models_blimp_raw.csv                 # Cross-model summaries
├── all_models_blimp_summary.csv
├── all_models_null_subject_raw.csv
├── all_models_null_subject_summary.csv
└── all_models_summary.json
```

## Contact

For issues or questions about these scripts, please refer to the main project documentation or create an issue in the repository.