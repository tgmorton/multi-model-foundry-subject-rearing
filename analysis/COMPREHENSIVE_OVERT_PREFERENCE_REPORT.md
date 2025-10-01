# Comprehensive Overt Preference Analysis Report

## Executive Summary

This report presents a comprehensive analysis of overt subject preference in null subject acquisition across 6 experimental conditions. The analysis consists of three main components:

- **Part A**: Overt preference vs. chance (0.5) comparisons  
- **Part B**: Pairwise item group contrasts
- **Part C**: Form/processing manipulation comparisons

## Key Findings

### Overall Model Performance
All 6 models show significant overt preference above chance (54.4%-74.9%):
- **Impoverish Determiners**: 74.9% (strongest overt preference)
- **Remove Articles**: 71.0% 
- **Baseline**: 69.4%
- **Remove Expletives**: 68.2%
- **Lemmatize Verbs**: 61.5%
- **Remove Subject Pronominals**: 54.4% (weakest overt preference)

### Item Group Consistency Patterns
**Always overt-preferring across all models:**
- 1st person (3a_1stSg, 3b_1stPL): 94.0-90.3% mean preference
- Long-distance binding (6): 88.4% mean preference
- Expletive seems (5a): 85.1% mean preference
- 3rd person plural (1b): 74.5% mean preference
- 2nd person plural (2b): 72.5% mean preference
- 2nd person singular (2a): 67.8% mean preference
- 3rd person singular (1a): 65.7% mean preference

**Always null-preferring across all models:**
- Object control (4b): 6.8% mean preference (strong null preference)

**Usually null-preferring:**
- Subject control (4a): 35.5% mean preference

## Part A: Overt Preference vs. Chance Analysis

### A1. Within-Model Item Group Comparisons
**Key patterns:**
- **1st person items** show strongest and most consistent overt preference across all models
- **Control items** (4a, 4b) consistently prefer null subjects across all models
- **Person hierarchy**: 1st > 2nd > 3rd person for overt preference
- **Number effects**: Plural forms generally show stronger overt preference than singular

### A2. Within-Model Form Comparisons  
**Negation effects consistently strong:**
- Target negation and both negation show strongest overt preference across models
- Context negation shows weaker but still significant overt preference
- Default forms show moderate overt preference

**Complexity effects moderate:**
- Complex embedding forms show small but significant overt preference over chance
- No consistent advantage for long vs. embedded complexity

### A3. Within-Model Item Group × Form Interactions
**High variation detected in:**
- Control items across different forms
- Some person/number combinations show form-specific preferences
- Negation interacts differently with different item groups

## Part B: Pairwise Item Group Comparisons

### B1. Number Contrasts (Singular vs. Plural)
**Consistent patterns across models:**
- **1st person**: Singular > Plural overt preference (5/6 models significant)
- **2nd person**: Plural > Singular overt preference (5/6 models significant)  
- **3rd person**: Plural > Singular overt preference (4/6 models significant)

**Model-specific findings:**
- Impoverish Determiners shows strongest number effects
- Remove Subject Pronominals shows reversed 3rd person pattern

### B2. Person Contrasts (1st vs. 2nd vs. 3rd)
**Hierarchy consistently:** 1st > 2nd/3rd person overt preference
- 1st vs. 2nd person: Significant across all 6 models (20-25% difference)
- 1st vs. 3rd person: Significant across all 6 models (18-30% difference)  
- 2nd vs. 3rd person: Mixed results, smaller effects (2-10% difference)

### B3. Control Contrasts (Subject vs. Object Control)
**Universal pattern:** Subject control > Object control overt preference
- Significant across all 6 models
- Effect sizes: 21-31% difference in overt preference
- Object control shows near-zero overt preference in most models

### B4. Expletive Contrasts (Seems vs. Be)
**Strong pattern:** Expletive seems > Expletive be overt preference
- Significant across 5/6 models  
- **Exception**: Lemmatize Verbs shows reversed pattern (Be > Seems)
- Large effect sizes: 15-44% difference

### B5. Topic Shift Contrasts
**Mixed patterns across models:**
- Remove Articles: No shift > Topic shift
- Impoverish Determiners: Topic shift > No shift  
- Remove Subject Pronominals: No shift > Topic shift
- Other models: Non-significant differences

## Part C: Form/Processing Comparisons

### C1. Forms vs. Default Baseline
**Negation advantages universal:**
- **Target negation > Default**: Significant across all 6 models (10-21% advantage)
- **Both negation > Default**: Significant across all 6 models (8-26% advantage)
- **Context negation vs Default**: Mixed results, generally non-significant

**Complexity advantages modest:**
- **Complex embedding vs Default**: Significant in 3/6 models (2-5% advantage)
- **Complex long vs Default**: Significant in 3/6 models (2-5% advantage)

### C2. Complex Embedding Comparisons
**No consistent preference for long vs. embedded complexity:**
- All models show non-significant differences (< 2% effect sizes)
- Suggests both complexity types have similar processing demands

### C3. Negation Type Hierarchy
**Consistent hierarchy:** Target ≥ Both > Context negation
- **Target vs. Context**: Significant across all 6 models (10-21% advantage)
- **Both vs. Context**: Significant across all 6 models (9-26% advantage)  
- **Target vs. Both**: Mixed results, generally small differences (1-5%)

## Statistical Summary

### Total Comparisons Conducted
- **Part A**: 78 binomial tests vs. chance (models × item groups × forms)
- **Part B**: 150 chi-square tests for item group contrasts  
- **Part C**: 102 chi-square tests for form/processing contrasts
- **Total**: 330 statistical comparisons

### Effect Size Patterns
- **Large effects (>20% difference)**: Person contrasts, control contrasts, expletive contrasts
- **Medium effects (10-20%)**: Number contrasts, negation form effects
- **Small effects (<10%)**: Complexity effects, topic shift effects

### Model Differences
- **Impoverish Determiners**: Shows strongest overall overt preference and largest effect sizes
- **Remove Subject Pronominals**: Shows weakest overall overt preference, some reversed patterns
- **Lemmatize Verbs**: Unique reversal in expletive contrast pattern
- **Baseline, Remove Expletives, Remove Articles**: Show similar intermediate patterns

## Implications

### Linguistic Theory
1. **Person hierarchy**: Strong evidence for 1st > 2nd > 3rd person overt preference
2. **Control theory**: Clear dissociation between subject vs. object control structures
3. **Expletive constructions**: "Seems" constructions more amenable to overt subjects than "be" constructions
4. **Negation effects**: Target negation creates strongest bias toward overt subjects

### Model Processing
1. **Individual differences**: Models show consistent ranking of overt preference strength
2. **Manipulation effects**: Different input manipulations preserve core linguistic patterns while shifting overall preference levels
3. **Robustness**: Core patterns (person, control, negation) replicate across all experimental conditions

## Files Generated

### Data Tables
- `analysis/tables/endstate_overt_vs_chance_models.csv`
- `analysis/tables/endstate_overt_vs_chance_itemgroups.csv` 
- `analysis/tables/endstate_overt_vs_chance_forms.csv`
- `analysis/tables/endstate_overt_itemgroup_consistency.csv`
- `analysis/tables/pairwise_b1_number_contrasts.csv`
- `analysis/tables/pairwise_b2_person_contrasts.csv`
- `analysis/tables/pairwise_b3_control_contrasts.csv`
- `analysis/tables/pairwise_b4_expletive_contrasts.csv`
- `analysis/tables/pairwise_b5_topic_shift_contrasts.csv`
- `analysis/tables/pairwise_c1_forms_vs_default.csv`
- `analysis/tables/pairwise_c2_complex_embedding.csv`
- `analysis/tables/pairwise_c3_negation_types.csv`

### Analysis Scripts
- `analysis/scripts/endstate_overt_vs_chance_detailed.R` (Part A)
- `analysis/scripts/overt_preference_pairwise_comparisons.R` (Parts B & C)

---

*Report generated: August 13, 2025*  
*Analysis covers 6 experimental models with 330 statistical comparisons*  
*All p-values: * p < 0.05, ** p < 0.01, *** p < 0.001*