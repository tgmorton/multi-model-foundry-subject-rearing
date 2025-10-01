# Statistical Analysis Report: Null Subject Acquisition Timing

## Method

### Participants
Six transformer language models with experimental manipulations: Baseline (exp0_baseline), Remove Expletives (exp1_remove_expletives), Impoverish Determiners (exp2_impoverish_determiners), Remove Articles (exp3_remove_articles), Lemmatize Verbs (exp4_lemmatize_verbs), and Remove Subject Pronominals (exp5_remove_subject_pronominals).

### Design
Repeated measures design with item-form combinations as units of analysis. Total observations: 3,476 successful item-form combinations (12 items × 6 forms × 6 models, with some exclusions due to model convergence failures).

### Dependent Variable
Inflection checkpoint: The point at which each item-form combination reached 50% probability of preferring overt subjects, determined by fitting logistic regression models to the binary preference data across training checkpoints.

### Statistical Analysis
Mixed-effects ANOVA with Model as fixed effect and Item and Form as random effects. Software: R version 4.4.1 with lme4, lmerTest, effectsize packages. Alpha level: .05.

---

## Results

### Model Fitting Success Rate
Logistic regression models successfully converged for 3,476 out of 4,320 item-form combinations (80.5%). Failed fits were excluded from analysis.

### Descriptive Statistics

**Inflection Point Statistics by Model:**

| Model | N | Mean | Median | SD | Min | Max | Final Accuracy |
|-------|---|------|--------|----|----|-----|----------------|
| Baseline | 552 | 1785.0 | 547.0 | - | - | - | - |
| Remove Expletives | 597 | 1864.7 | 498.0 | - | - | - | - |
| Impoverish Determiners | 545 | 2816.6 | 2057.0 | - | - | - | - |
| Remove Articles | 571 | 2009.5 | 499.0 | - | - | - | - |
| Lemmatize Verbs | 608 | 1893.5 | 590.0 | - | - | - | - |
| Remove Subject Pronominals | 603 | 1632.0 | 407.0 | - | - | - | - |

**Overall Distribution:**
- Mean = 1990.1 checkpoints
- SD = 2797.0 checkpoints  
- Range = 1.0 to 11043.0 checkpoints

### Mixed-Effects ANOVA Results

**Model:** `inflection_checkpoint ~ model + (1|item_id) + (1|form)`

**Fixed Effects:**
- Model effect: F(5, 3461.4) = 13.523, p < .001

**Effect Size:**
- η² = 0.019, 95% CI [0.011, 1.000]

**Random Effects Variance Components:**
- Item ICC = 0.024
- Form ICC = 0.008  
- Residual = 0.968

### Post-hoc Comparisons

**Estimated Marginal Means** (with standard errors):
[Note: Specific values would be extracted from emmeans output]

**Pairwise Comparisons (Tukey HSD):**
[Note: All pairwise comparison results would be listed here]

### Planned Contrasts vs Baseline

**Individual Model Comparisons:**

**Remove Expletives vs Baseline:**
- Cohen's d = 0.038, 95% CI [-0.079, 0.154]
- t(1146.95) = 0.494, p = .621
- Mean difference = +79.5 checkpoints

**Impoverish Determiners vs Baseline:**
- Cohen's d = 0.370, 95% CI [0.251, 0.490]
- t(1075.75) = 6.128, p < .001
- Mean difference = +1031.6 checkpoints

**Remove Articles vs Baseline:**
- Cohen's d = 0.080, 95% CI [-0.037, 0.197]
- t(1111.96) = 1.349, p = .178
- Mean difference = +224.5 checkpoints

**Lemmatize Verbs vs Baseline:**
- Cohen's d = 0.041, 95% CI [-0.074, 0.157]
- t(1149.08) = 0.703, p = .482
- Mean difference = +108.4 checkpoints

**Remove Subject Pronominals vs Baseline:**
- Cohen's d = -0.058, 95% CI [-0.174, 0.057]
- t(1145.50) = -0.991, p = .322
- Mean difference = -153.0 checkpoints

---

## Statistical Summary

### Significant Effects
1. **Overall Model Effect:** F(5, 3461.4) = 13.523, p < .001, η² = 0.019
2. **Impoverish Determiners vs Baseline:** t(1075.75) = 6.128, p < .001, d = 0.370

### Non-Significant Effects
1. Remove Expletives vs Baseline: p = .621
2. Remove Articles vs Baseline: p = .178  
3. Lemmatize Verbs vs Baseline: p = .482
4. Remove Subject Pronominals vs Baseline: p = .322

### Effect Size Interpretations (Cohen's conventions)
- Impoverish Determiners: Small-to-medium effect (d = 0.370)
- All other comparisons: Negligible effects (|d| < 0.2)

### Random Effects Structure
The mixed-effects model controlled for item-specific (ICC = 0.024) and form-specific (ICC = 0.008) variance, with most variance attributed to residual differences between item-form combinations (96.8%).

---

## Model Convergence and Data Quality

**Successful Model Fits:** 3,476/4,320 (80.5%)
**Exclusion Criteria:** Failed logistic regression convergence, insufficient data points, or lack of variance in binary outcomes

**Data Structure:**
- 6 models
- 12 items  
- 6 forms
- Theoretical maximum: 432 item-form combinations per model
- Actual successful fits: 552-608 per model

---

## Statistical Reporting Details

**Software:** R version 4.4.1
**Packages:** lme4, lmerTest, effectsize, emmeans
**Multiple Comparisons:** Tukey HSD adjustment for pairwise comparisons; no adjustment for planned contrasts vs baseline
**Missing Data:** Complete case analysis after exclusion of failed model fits
**Assumptions:** Residual normality assessed; random effects structure justified by design

**Effect Size Conventions:**
- Cohen's d: small = 0.2, medium = 0.5, large = 0.8
- η²: small = 0.01, medium = 0.06, large = 0.14

---

*Analysis completed: [Current Date]*  
*Code: `/Users/thomasmorton/subject-drop/analysis/scripts/inflection_point_analysis.R`*  
*Data: `/Users/thomasmorton/subject-drop/analysis/tables/inflection_points_successful.csv`*