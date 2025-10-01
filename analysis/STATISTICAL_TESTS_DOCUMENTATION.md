# Statistical Tests Documentation

This document provides a comprehensive overview of all statistical tests conducted in the null subject acquisition analysis, their associated files, and APA-style descriptions.

## Overview

All statistical analyses used mixed-effects logistic regression models with subject identity as a random effect, unless otherwise noted. Multiple comparisons were controlled using False Discovery Rate (FDR) adjustment. Bootstrap confidence intervals used n=200 iterations with parametric resampling.

---

## 1. MODEL SELECTION TESTS

### 1.1 Degrees of Freedom Selection via AIC
**File:** `analysis/tables/tests/df_selection_by_aic.csv`  
**File:** `analysis/tables/tests/df_comparison_detailed.csv`

**Description:** Model selection procedure comparing natural spline models with 3-7 degrees of freedom using Akaike Information Criterion (AIC). For each experimental condition, generalized linear mixed-effects models (GLMMs) with varying spline complexity were fitted and compared. The model with the lowest AIC was selected as optimal.

**APA Reporting Format:**
*"AIC-based model selection indicated optimal spline complexity varied across conditions, with the Baseline model selecting 6 degrees of freedom (AIC = 146,242), while Impoverish Determiners selected 7 degrees of freedom (AIC = 139,485)."*

**Parameters to report:** Model name, selected df, AIC value

---

## 2. ACQUISITION TIMING TESTS

### 2.1 Robust t50 Estimates  
**File:** `analysis/tables/tests/t50_by_model_robust.csv`

**Description:** Bootstrap estimation of t50 (time to 50% null subject accuracy) using natural spline GLMMs with last-crossing detection. For each model, 200 bootstrap samples were generated using parametric resampling from fixed-effects uncertainty. The t50 was defined as the last checkpoint where the fitted curve crossed 0.5 null subject probability.

**APA Reporting Format:**
*"The Baseline model achieved t50 at checkpoint 482.86 (95% CI [396.19, 576.63]), while the Impoverish Determiners condition showed severely delayed acquisition (t50 = 2,751.93, 95% CI [2,622.93, 2,879.39])."*

**Parameters to report:** t50 estimate, 95% CI bounds

### 2.2 t50 Differences from Baseline
**File:** `analysis/tables/tests/delta_t50_vs_baseline_robust.csv`

**Description:** Pairwise comparisons of t50 estimates between experimental conditions and baseline using bootstrap difference distributions. Empirical p-values were calculated as the proportion of bootstrap samples where the experimental condition showed equal or earlier t50 than baseline.

**APA Reporting Format:**
*"Relative to baseline, the Remove Expletives condition showed a significant delay in acquisition (Δt50 = 48.30, 95% CI [31.53, 61.83], p < .001), while the Impoverish Determiners condition exhibited extreme delays (Δt50 = 2,269.06, 95% CI [2,200.12, 2,333.51], p < .001)."*

**Parameters to report:** Δt50, 95% CI bounds, empirical p-value

### 2.3 Age of Acquisition (AoA) Estimates
**File:** `analysis/tables/tests/aoa_halfway_by_model.csv`  
**File:** `analysis/tables/tests/delta_aoa_half_vs_baseline.csv`

**Description:** Bootstrap estimation of Age of Acquisition defined as time to halfway point between final null and overt subject preferences. Similar bootstrap methodology as t50 analysis, but using halfway point between asymptotic performance levels rather than fixed 0.5 threshold.

**APA Reporting Format:**
*"Age of acquisition analysis revealed that Lemmatize Verbs achieved earliest AoA (t_half = 705.19, 95% CI [660.94, 748.55]), representing a significant 22.27 checkpoint advantage over baseline (95% CI [-43.31, -1.65], p = .034)."*

**Parameters to report:** AoA estimate, 95% CI bounds, difference from baseline, empirical p-value

---

## 3. END-STATE PERFORMANCE TESTS

### 3.1 Final Null Subject Preference
**File:** `analysis/tables/tests/endstate_null_pref.csv`

**Description:** Estimated marginal means for null subject preference in final 1000 checkpoints using GLMM with subject identity random effects. Preferences were estimated on the probability scale using logistic link function.

**APA Reporting Format:**
*"End-state analysis revealed significant variation in null subject preferences across conditions, with Remove Subject Pronominals showing the highest preference (p = .457, 95% CI [.412, .502]) and Impoverish Determiners the lowest (p = .239, 95% CI [.207, .274])."*

**Parameters to report:** Probability estimate, 95% CI bounds

### 3.2 Pairwise Model Comparisons (Null Preference)
**File:** `analysis/tables/tests/endstate_null_pref_pairwise.csv`

**Description:** Post-hoc pairwise comparisons of final null subject preferences between all experimental conditions using estimated marginal means. Comparisons used odds ratios with FDR adjustment for multiple testing.

**APA Reporting Format:**
*"Pairwise comparisons revealed that Remove Subject Pronominals significantly exceeded baseline null preference (OR = 1.95, SE = 0.05, z = 25.70, p_FDR < .001), while Lemmatize Verbs showed significantly lower preference than baseline (OR = 0.70, SE = 0.02, z = -13.62, p_FDR < .001)."*

**Parameters to report:** Odds ratio, standard error, z-ratio, FDR-adjusted p-value

### 3.3 Null-Overt Performance Gap
**File:** `analysis/tables/tests/endstate_gap_null_minus_overt.csv`  
**File:** `analysis/tables/tests/endstate_gap_pairwise_models.csv`

**Description:** Analysis of performance differences between null and overt subject conditions at end-state. Gap calculated as accuracy on null subjects minus accuracy on overt subjects, with pairwise model comparisons using linear contrasts.

**APA Reporting Format:**
*"Performance gap analysis showed that Remove Subject Pronominals exhibited the largest null-overt accuracy difference (gap = .717, 95% CI [.683, .752], p < .001), while Impoverish Determiners showed the smallest gap (gap = .104, 95% CI [.098, .110], p < .001)."*

**Parameters to report:** Gap estimate, 95% CI bounds, p-value

---

## 4. LINGUISTIC FORM ANALYSIS

### 4.1 Form × Model Interactions  
**File:** `analysis/tables/tests/endstate_form_model_diffs.csv`

**Description:** Two-way analysis examining interactions between linguistic form types (both_negation, complex_emb, complex_long, context_negation, default, target_negation) and experimental conditions using GLMM with form × model interaction terms.

**APA Reporting Format:**
*"Form × model interaction analysis revealed that complex embedding contexts showed significantly different null preferences between Baseline and Lemmatize Verbs conditions (OR = 1.73, SE = 0.04, z = 23.37, p_FDR < .001), while default contexts showed minimal cross-model variation."*

**Parameters to report:** Odds ratio, standard error, z-ratio, FDR-adjusted p-value, form type, model contrast

### 4.2 Within-Model Form Comparisons
**File:** `analysis/tables/tests/endstate_forms_within_model.csv`

**Description:** Within-model analysis comparing null subject preferences across different linguistic form types using estimated marginal means with FDR correction for multiple comparisons within each model.

**APA Reporting Format:**
*"Within the Baseline model, complex embedding contexts showed significantly higher null preference than default contexts (OR = 1.42, SE = 0.10, z = 5.20, p_FDR < .001), while target negation contexts did not differ significantly from default (OR = 1.64, SE = 0.11, z = 7.36, p_FDR < .001)."*

**Parameters to report:** Odds ratio, standard error, z-ratio, FDR-adjusted p-value, form contrast, model name

---

## 5. ITEM GROUP ANALYSIS

### 5.1 Baseline Item Group Effects
**File:** `analysis/tables/tests/endstate_itemgroup_baseline.csv`  
**File:** `analysis/tables/tests/endstate_itemgroup_baseline_pairwise.csv`

**Description:** Analysis of syntactic item group effects (1a_3rdSG, 1b_3rdPL, 2a_2ndSG, 2b_2ndPL, 3a_1stSg, 3b_1stPL, 4a_subject_control, 4b_object_control, 5a_expletive_seems, 5b_expletive_be, 6_long_distance_binding, 7a_conjunction_no_topic_shift, 7b_conjunction_topic_shift) in baseline condition using GLMM with comprehensive pairwise comparisons.

**APA Reporting Format:**
*"Item group analysis in the baseline condition revealed a hierarchy of null subject preferences, with object control showing highest preference (p = 1.000, 95% CI [.992, 1.000]) and long-distance binding showing lowest (p < .001, 95% CI [.000, .873]). Pairwise comparisons indicated significant differences between subject and object control (OR = 39.43, SE = 5.78, z = 25.08, p_FDR < .001)."*

**Parameters to report:** Probability estimates with CI for marginal means, OR/SE/z-ratio/p_FDR for pairwise comparisons

### 5.2 Item Group × Model Interactions
**File:** `analysis/tables/tests/endstate_itemgroup_model_diffs.csv`

**Description:** Cross-model analysis examining how item group effects vary across experimental conditions using three-way interactions (item_group × model × form_type) in comprehensive GLMM.

**APA Reporting Format:**
*"Item group × model interactions revealed that third-person singular contexts showed significantly different patterns between Baseline and Impoverish Determiners (OR = 0.58, SE = 0.06, z = -5.74, p_FDR < .001), while first-person contexts remained stable across conditions."*

**Parameters to report:** Odds ratio, standard error, z-ratio, FDR-adjusted p-value, item group, model contrast

### 5.3 Within-Model Item Group Analysis  
**File:** `analysis/tables/tests/endstate_itemgroup_within_model.csv` (truncated due to size)

**Description:** Detailed within-model pairwise comparisons of all item groups for each experimental condition separately, using model-specific GLMMs with comprehensive post-hoc testing.

**APA Reporting Format:**
*"Within-model comparisons in the Remove Subject Pronominals condition revealed that conjunction with topic shift contexts significantly exceeded third-person singular (OR = 29.32, SE = 3.39, z = 11.51, p_FDR < .001), while first-person contexts remained lowest (OR = 0.38, SE = 0.04, z = -9.08, p_FDR < .001)."*

**Parameters to report:** Odds ratio, standard error, z-ratio, FDR-adjusted p-value, item group contrast, model name

---

## 6. FIRST EPOCH ANALYSIS

### 6.1 Early Learning Assessment
**File:** `analysis/tables/first_epoch_checkpoints.csv`  
**File:** `analysis/tables/first_epoch_summary.csv`

**Description:** Binomial tests of null subject preference in the final 4 checkpoints before end of first epoch (defined as maximum_checkpoint/20 for each model). Tests whether models showed significant learning (preference ≠ 0.5) within first epoch using exact binomial tests.

**APA Reporting Format:**
*"First epoch analysis demonstrated that all models achieved significant null subject learning by epoch end, with Baseline showing 65.4% null preference (SD = 1.3%, p < 2e-16) and Remove Expletives showing 60.5% preference (SD = 3.8%, p < 2e-16), both significantly above chance."*

**Parameters to report:** Mean percentage, standard deviation, binomial test p-value

---

## 7. WITHIN-MODEL ITEM GROUP POST-HOC TESTS

### 7.1 Null Subject Preference by Item Group
**File:** `analysis/tables/within_model_itemgroup_emmeans.csv`  
**File:** `analysis/tables/within_model_itemgroup_pairwise.csv`

**Description:** Model-specific analysis of item group effects on null subject preference using separate GLMMs for each experimental condition. Estimated marginal means calculated for null subject trials only, with comprehensive pairwise comparisons using FDR adjustment.

**APA Reporting Format:**
*"Within-model analysis revealed that in the Baseline condition, object control contexts showed highest null preference (p = 1.000, SE = 0.000), significantly exceeding subject control (OR = 31.44, SE = 4.61, z = 25.08, p_FDR < .001). First-person contexts consistently showed lowest preferences across models."*

**Parameters to report:** Probability estimates with SE for marginal means, OR/SE/z-ratio/p_FDR for pairwise comparisons, model name

### 7.2 Overt Subject Preference by Item Group  
**File:** `analysis/tables/within_model_itemgroup_emmeans_overt.csv`  
**File:** `analysis/tables/within_model_itemgroup_pairwise_overt.csv`

**Description:** Complementary analysis examining item group effects on overt subject preference using identical methodology as null subject analysis but focusing on overt subject trials. Provides complete picture of subject type preferences across syntactic contexts.

**APA Reporting Format:**
*"Overt subject analysis showed that long-distance binding contexts required overt subjects most strongly (p = 1.000, SE = 0.000) in the Baseline condition, significantly exceeding expletive 'be' contexts (OR = 13.11, SE = 1.39, z = 23.37, p_FDR < .001). Object control showed lowest overt preference across models."*

**Parameters to report:** Probability estimates with SE for marginal means, OR/SE/z-ratio/p_FDR for pairwise comparisons, model name

---

## 8. COMPREHENSIVE OVERT PREFERENCE ANALYSIS

### 8.1 End-State Overt Preference vs. Chance
**File:** `analysis/tables/endstate_overt_vs_chance_models.csv`  
**File:** `analysis/tables/endstate_overt_vs_chance_itemgroups.csv`  
**File:** `analysis/tables/endstate_overt_vs_chance_forms.csv`  
**File:** `analysis/tables/endstate_overt_itemgroup_consistency.csv`

**Description:** Comprehensive binomial tests comparing end-state overt subject preference to chance (0.5) across models, item groups, and linguistic forms. Analysis included within-model comparisons by item group, within-model comparisons by form, and item group × form interactions. Cross-model consistency analysis identified patterns of overt vs. null preference across experimental conditions.

**APA Reporting Format:**
*"End-state overt preference analysis revealed that all models showed significant preference for overt subjects above chance, with Impoverish Determiners showing strongest preference (74.9%, 95% CI [74.0, 75.7], p < .001) and Remove Subject Pronominals showing weakest preference (54.4%, 95% CI [53.4, 55.3], p < .001). Item group analysis identified consistent patterns, with 1st person contexts showing overt preference across all models (mean = 94.0%) and object control consistently preferring null subjects (mean = 6.8%)."*

**Parameters to report:** Overt preference percentage, 95% CI bounds, binomial test p-value, model name, item group consistency patterns

### 8.2 Pairwise Item Group Comparisons  
**File:** `analysis/tables/pairwise_b1_number_contrasts.csv`  
**File:** `analysis/tables/pairwise_b2_person_contrasts.csv`  
**File:** `analysis/tables/pairwise_b3_control_contrasts.csv`  
**File:** `analysis/tables/pairwise_b4_expletive_contrasts.csv`  
**File:** `analysis/tables/pairwise_b5_topic_shift_contrasts.csv`

**Description:** Systematic pairwise comparisons of overt preferences between related item groups using chi-square tests of independence. Analysis included (B1) number contrasts within person categories, (B2) person hierarchy comparisons, (B3) subject vs. object control contrasts, (B4) expletive construction contrasts, and (B5) topic shift effects in conjunction contexts.

**APA Reporting Format:**
*"Number contrast analysis revealed systematic singular-plural differences across person categories. In the Baseline model, 2nd person contexts showed significant plural advantage (χ²(1) = 21.81, p < .001), with plural forms achieving 75.6% overt preference compared to 65.7% for singular forms. Person hierarchy analysis confirmed 1st > 2nd > 3rd person overt preference across all models, with 1st vs. 2nd person differences ranging from 20.2% to 25.4% (all p < .001)."*

**Parameters to report:** Chi-square statistic, degrees of freedom, p-value, effect size (proportion difference), contrast type, model name

### 8.3 Form/Processing Manipulation Comparisons
**File:** `analysis/tables/pairwise_c1_forms_vs_default.csv`  
**File:** `analysis/tables/pairwise_c2_complex_embedding.csv`  
**File:** `analysis/tables/pairwise_c3_negation_types.csv`

**Description:** Analysis of processing manipulation effects on overt preference using chi-square tests. Included (C1) comparisons of all linguistic forms against default baseline, (C2) complex embedding comparisons (long-distance vs. embedded), and (C3) negation type hierarchy testing (context vs. target vs. both negation conditions).

**APA Reporting Format:**
*"Form manipulation analysis demonstrated consistent negation advantages across all models. Target negation showed significant overt preference advantage over default contexts in all 6 models, with effect sizes ranging from 10.3% (Baseline, χ²(1) = 47.63, p < .001) to 20.8% (Lemmatize Verbs, χ²(1) = 189.45, p < .001). Complex embedding comparisons revealed no systematic preference for long-distance vs. embedded complexity (all p > .05), suggesting equivalent processing demands."*

**Parameters to report:** Chi-square statistic, degrees of freedom, p-value, effect size (proportion difference), form contrast, model name, consistency across models

---

## Statistical Software and Packages

All analyses were conducted in R (version 4.0+) using the following packages:
- **lme4**: Mixed-effects models
- **lmerTest**: Significance testing for mixed models  
- **emmeans**: Estimated marginal means and post-hoc comparisons
- **splines**: Natural spline fitting
- **MASS**: Multivariate normal sampling for bootstrap
- **tidyverse**: Data manipulation and analysis

## Multiple Comparison Corrections

All post-hoc pairwise comparisons used False Discovery Rate (FDR) adjustment to control Type I error rate across multiple tests within each analysis family. Bootstrap procedures used n=200 iterations with parametric resampling from estimated fixed-effects distributions. The comprehensive overt preference analysis conducted 330 statistical comparisons across Parts A, B, and C, with appropriate correction procedures applied within each analysis family.

## Convergence and Model Diagnostics

All reported models achieved successful convergence. Models showing boundary (singular) fits were noted but included in analysis as the substantive results remained stable. For complex models that failed to converge, simplified random effect structures or alternative optimizers were employed.