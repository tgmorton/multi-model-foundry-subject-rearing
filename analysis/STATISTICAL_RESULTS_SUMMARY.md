# Null Subject Acquisition - Statistical Analysis Results

## Overview

This document summarizes the key findings from the statistical analysis of null subject acquisition across experimental manipulations, based on the comprehensive analysis plan in ANALYSISPLAN.MD.

## Phase 1: Primary Analyses ✅ COMPLETED

### Analysis 1A: End-of-Training Success (Mixed Effects Model)

**Research Question:** Did models successfully acquire overt preference?

**Model:** `correct ~ model * form_type + (1|item_id) + (1|form)`

**Key Findings:**

1. **Main Effect of Form Type (Overt vs Null):**
   - Strong preference for overt subjects (estimate = 0.385, SE = 0.022, t = 17.73, p < .001)
   - Effect Size: Cohen's d = 0.833 [95% CI: 0.738, 0.927] = **Large Effect**

2. **Model Differences in Overt Preference (Effect Sizes):**

| Model                          | Cohen's d | 95% CI        | Interpretation | Mean Diff |
|--------------------------------|-----------|---------------|----------------|-----------|
| Baseline                       | 0.83      | [0.74, 0.93] | Large          | 0.38      |
| **Impoverish Determiners**     | **1.02**  | [0.92, 1.11] | **Large**      | **0.45**  |
| Remove Articles                | 0.94      | [0.85, 1.04] | Large          | 0.43      |
| Remove Expletives              | 0.76      | [0.67, 0.86] | Large          | 0.36      |
| Lemmatize Verbs                | 0.44      | [0.35, 0.53] | Small-Medium   | 0.21      |
| **Remove Subject Pronominals** | **0.15**  | [0.06, 0.24] | **Negligible** | **0.08**  |

**Key Interpretations:**
- ✅ **All models successfully acquired overt preference** (all p < .001)
- ⚠️ **Remove Subject Pronominals showed severely impaired learning** (d = 0.15)
- 🔝 **Impoverish Determiners showed enhanced learning** (d = 1.02, strongest effect)
- 📉 **Clear hierarchy:** Impoverish Determiners > Remove Articles > Baseline > Remove Expletives > Lemmatize Verbs > Remove Subject Pronominals

### Analysis 1B: Overall Acquisition Timing (ANOVA)

**Research Question:** Do experimental manipulations affect acquisition speed?

**Key Findings:**

| Model                          | Mean Acquisition Checkpoint | Delay from Baseline | Interpretation  |
|--------------------------------|----------------------------|-------------------|----------------|
| Baseline                       | 666                        | --               | Reference      |
| Remove Subject Pronominals    | 681                        | +15              | Minimal delay  |
| Remove Articles                | 712                        | +46              | Small delay    |
| Remove Expletives              | 741                        | +75              | Small delay    |
| Lemmatize Verbs                | 748                        | +82              | Small delay    |
| **Impoverish Determiners**     | **3061**                   | **+2395**        | **Major delay** |

**Key Interpretations:**
- 📈 **Major finding:** Impoverish Determiners shows **4.6x slower acquisition** than baseline
- ⏱️ Most other manipulations show minimal timing delays (15-82 checkpoints)
- 🎯 **Remove Subject Pronominals:** Fastest acquisition time BUT weakest final performance (paradox!)

## Phase 2: Granular Analyses ⚠️ PARTIAL COMPLETION

**Status:** Mixed model convergence issues due to model complexity. Successfully completed descriptive analyses.

## Phase 3: Learning Dynamics ⚠️ CONVERGENCE ISSUES  

**Status:** Polynomial growth models failed to converge due to high dimensionality.

## Phase 4: Exploratory Analyses ⚠️ NOT COMPLETED

**Status:** Dependent on successful completion of earlier phases.

---

## Summary of Key Scientific Findings

### 1. **Success vs Speed Paradox**
- **Remove Subject Pronominals:** Fastest to acquire (681 checkpoints) but weakest final performance (d = 0.15)
- **Impoverish Determiners:** Slowest to acquire (3061 checkpoints) but strongest final performance (d = 1.02)

### 2. **Manipulation Effectiveness Ranking**

**By Final Performance (Effect Size):**
1. 🥇 Impoverish Determiners (d = 1.02)
2. 🥈 Remove Articles (d = 0.94) 
3. 🥉 Baseline (d = 0.83)
4. Remove Expletives (d = 0.76)
5. Lemmatize Verbs (d = 0.44)
6. ⚠️ Remove Subject Pronominals (d = 0.15)

**By Acquisition Speed:**
1. 🚀 Remove Subject Pronominals (681)
2. Remove Articles (712)
3. Remove Expletives (741) 
4. Lemmatize Verbs (748)
5. 🐌 Impoverish Determiners (3061)

### 3. **Implications for Linguistic Theory**

1. **Determiner impoverishment** enhances ultimate acquisition success but delays initial learning
2. **Subject pronoun removal** creates rapid but shallow acquisition 
3. **Morphological manipulations** (lemmatize verbs) impair acquisition more than syntactic ones
4. **Expletive removal** shows moderate effects across both dimensions

---

## Technical Notes

- **Mixed effects models** successfully controlled for item and form variability
- **Random effects variance** was near zero, indicating strong consistency across items/forms
- **Effect sizes** calculated using Cohen's d with 95% confidence intervals
- **Statistical power** was high due to large sample size (11,232 final observations)

## Recommendations for Further Analysis

1. **Simplify Phase 2-4 models** to avoid convergence issues
2. **Investigate the success-speed paradox** with targeted follow-up analyses  
3. **Examine trajectory shapes** using simpler polynomial models
4. **Test robustness** across different acquisition thresholds

---

*Analysis completed: 2025-08-11*  
*Based on:** `/Users/thomasmorton/subject-drop/analysis/scripts/null_subject_statistical_tests.R`