# Analysis Methods Audit Report

## Overview
This document provides a comprehensive audit of the statistical methods implemented for the null/overt subject acquisition analysis in language models. All methods are extracted from the codebase with specific file and line citations.

## 1. Data Preparation

### Input Data
- **Primary file**: `evaluation/results/all_models_null_subject_lme4_ready.csv`
- **Preference coding**: Binary variable `correct` where 1 = null preferred, 0 = overt preferred (based on lower surprisal)
- **Key factors**:
  - `model`: 6 experimental conditions (exp0_baseline through exp5_remove_subject_pronominals)
  - `form_type`: null vs overt subject forms
  - `item_group`: Linguistic item categories
  - `item_id`: Random effect grouping variable

### Baseline Releveling
- **Implementation**: `analysis/scripts/analysis_with_models.R:L119-L121`
- **Method**: `forcats::fct_relevel(data$model, baseline_model)`
- Ensures exp0_baseline is the reference level for all contrasts

## 2. Model Selection (Spline Degrees of Freedom)

### Implementation
- **Location**: `analysis/scripts/analysis_with_models.R:L139-L230`
- **Method**: Automatic df selection via AIC comparison

### Parameters
- **Degrees tested**: 3, 4, 5, 6, 7
- **Model formula**: `correct ~ ns(log_chk, df = K) + (1|item_id)`
- **Family**: binomial
- **Optimizer**: nloptwrap with maxeval = 200000
- **Selection criterion**: Lowest AIC among converged models

### Fallback Strategy
1. If no non-singular models converge: Accept lowest AIC even if singular
2. If all models fail: Default to df = 3

### Output Files
- `analysis/tables/tests/df_selection_by_aic.csv`
- `analysis/tables/tests/df_comparison_detailed.csv`

## 3. t₅₀ (Chance Crossing) Analysis

### Implementation
- **Location**: `analysis/scripts/analysis_with_models.R:L232-L324`

### Model Specification
- **Formula**: `correct ~ ns(log_chk, df=K) + (1|item_id)`
- **Log transformation**: `log_chk = log10(checkpoint_num + 1)`
- **Burn-in**: checkpoint ≥ 100

### Crossing Detection Algorithm
- **Tolerance**: ±0.05 from 0.5
- **Halfway point**: 60% of max checkpoint (conservative)
- **Preference**: Last crossing within first 60% of training
- **Fallback**: First crossing if no early crossings exist
- **Interpolation**: Linear interpolation between adjacent points

### Bootstrap Configuration
- **Type**: Parametric bootstrap
- **Simulations**: nsim = 1000
- **Method**: mvrnorm with fixed effects covariance matrix
- **Random effects**: re.form = NA (fixed effects only)
- **Seed**: 42
- **CI**: 2.5% and 97.5% quantiles

### Output Files
- `analysis/tables/tests/t50_by_model_robust.csv`
- `analysis/tables/tests/delta_t50_vs_baseline_robust.csv`

## 4. AoA½ (Halfway to Asymptote)

### Implementation
- **Status**: IMPLEMENTED
- **Location**: `analysis/scripts/analysis_with_models.R:L859-L1012`

### Definition
- **End-state window**: Last 10% of training (`checkpoint_num >= quantile(checkpoint_num, 0.9)`)
- **Asymptote**: p∞ = mean preference in end-state window
- **Threshold**: θ = (p∞ + 0.5) / 2
- **Interpretation**: Halfway between chance (0.5) and asymptotic preference

### Crossing Detection
- **Burn-in**: checkpoint ≥ 100
- **Stability requirement**: Minimum run of 3 consecutive points above threshold
- **Rule**: Last crossing of threshold

### Bootstrap
- **Type**: Parametric
- **Simulations**: nsim = 1000
- **Seed**: 99
- **CI computation**: 2.5% and 97.5% quantiles

### Output Files
- `analysis/tables/tests/aoa_halfway_by_model.csv`
- `analysis/tables/tests/delta_aoa_half_vs_baseline.csv`

## 5. End-State Analysis

### Implementation
- **Location**: `analysis/scripts/analysis_with_models.R:L466-L552`

### Window Definition
- **Method**: Last 10% of training
- **Formula**: `checkpoint_num >= quantile(checkpoint_num, 0.9)`

### Null Preference Model
- **Formula**: `correct ~ model + (1|item_id)`
- **Family**: binomial
- **Optimizer**: bobyqa with maxfun = 200000
- **EMM computation**: emmeans with type = "response"

### Null-Overt Gap Analysis
- **Formula**: `correct ~ model * form_type + (1|item_id)`
- **Contrast definition**: NullMinusOvert = c(1, -1)
- **Scale**: Response scale (probability difference)

### Multiple Comparisons
- **Primary adjustment**: Tukey HSD
- **Secondary adjustment**: FDR (False Discovery Rate)
- **Reporting scale**: Probability

### Output Files
- `analysis/tables/tests/endstate_null_pref.csv`
- `analysis/tables/tests/endstate_null_pref_pairwise.csv`
- `analysis/tables/tests/endstate_gap_null_minus_overt.csv`

## 6. Item-Group Analysis

### Implementation
- **Location**: `analysis/scripts/within_model_itemgroup_posthoc.R:L1-L198`

### Model Specification
- **Formula**: `correct ~ item_group * form_type + (1|item_id)`
- **Window**: Last 1000 checkpoints
- **Optimizer**: bobyqa with maxfun = 100000

### EMM Settings
- **Fixed at**: form_type = 'null'
- **Type**: response scale
- **Adjustment**: FDR

### Key Contrasts Examined
- Person contrasts: 3rd vs 1st, singular vs plural
- Control types: subject vs object control
- Expletive types: seems vs be
- Topic shift: conjunction with vs without topic shift

### Output Files
- `analysis/tables/within_model_itemgroup_emmeans.csv`
- `analysis/tables/within_model_itemgroup_pairwise.csv`
- `analysis/tables/within_model_key_contrasts.csv`

## 7. Processing/Form Analysis

### Implementation
- **Location**: `analysis/scripts/form_pairwise_comparisons.R:L1-L141`

### Model Specification
- **Formula**: `correct ~ form + (1|item_id)`
- **Window**: Last 1000 checkpoints
- **Data filter**: form_type == 'overt' (overt subjects only)

### Form Pairs Compared
1. complex_emb vs complex_long
2. target_negation vs context_negation
3. target_negation vs both_negation

### Statistical Details
- **Optimizer**: bobyqa
- **EMM type**: response scale
- **Inference**: Odds ratios with 95% CI

### Output Files
- `analysis/tables/form_pairwise_comparisons.csv`

## 8. First-Epoch Analysis

### Implementation
- **Location**: `analysis/scripts/first_epoch_analysis.R:L1-L140`

### Epoch Definition
- **Formula**: max_checkpoint / 20
- **Window**: 4 checkpoints closest to and before first epoch boundary

### Statistical Methods
- **Summary statistics**: Mean and SD across 4 checkpoints
- **Hypothesis test**: Pooled binomial test vs 0.5
- **Test type**: Two-sided exact binomial test
- **Significance level**: α = 0.05

### Output Files
- `analysis/tables/first_epoch_checkpoints.csv` (detailed)
- `analysis/tables/first_epoch_summary.csv` (summary)

## 9. Technical Implementation Details

### Optimization Settings
- **Primary optimizer**: bobyqa
- **Spline fitting optimizer**: nloptwrap
- **Maximum function evaluations**: 200000
- **Derivative calculation**: Disabled (calc.derivs = FALSE)

### Model Caching
- **Enabled**: Yes
- **Directory**: `analysis/models/`
- **Format**: RDS files
- **Validation**: Smoke test with predict() function

### Convergence Handling
- **Singularity check**: isSingular() function
- **Singular models**: Accepted if lowest AIC
- **Convergence failures**: Automatic fallback to simpler models

### Computational Features
- **Random seed**: 123 for reproducibility
- **Parallel processing**: Available via future/furrr packages
- **Bootstrap method**: Parametric with mvrnorm for multivariate normal draws
- **Covariance regularization**: Automatic jitter (1e-8 to 1e-4) if matrix not positive definite

## Key Methodological Notes

1. **Log transformation**: All checkpoint numbers transformed as log10(checkpoint_num + 1) to handle 0 and improve model fit

2. **Bootstrap coherence**: Same random draws used for both point estimates and confidence intervals to ensure coherent uncertainty quantification

3. **Multiple testing correction**: Different adjustment methods used appropriately:
   - Tukey for all pairwise comparisons among models
   - FDR for within-model item-group comparisons
   - No adjustment for planned contrasts

4. **Window consistency**: End-state consistently defined as last 10% of training across all analyses

5. **Robust estimation**: All mixed models use restricted maximum likelihood (REML) with appropriate optimizer settings for convergence

## Quality Assurance

All methods include:
- Explicit burn-in periods to avoid early training noise
- Convergence checks with fallback strategies
- Bootstrap validation with adequate sample size (nsim ≥ 1000)
- Model caching with validation to ensure reproducibility
- Multiple comparison corrections where appropriate

---

*Generated from codebase analysis on `analysis/scripts/` directory*