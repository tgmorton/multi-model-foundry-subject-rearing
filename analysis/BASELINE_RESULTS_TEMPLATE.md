# Results Section Template: Baseline (exp0_baseline) Condition

*Fill in the blanks using the statistical test documentation and data files*

## Model Selection

AIC-based model selection indicated that the Baseline model achieved optimal fit with **[SELECTED_DF]** degrees of freedom (AIC = **[AIC_VALUE]**), selected from candidates ranging 3-7 df. **[CONVERGENCE_STATUS]**.

## Timing of Acquisition

### t50 Analysis
The Baseline model achieved t50 at checkpoint **[T50_ESTIMATE]** (95% CI [**[T50_CI_LOW]**, **[T50_CI_HIGH]**]). **[COMPARISON_TO_OTHER_MODELS]**.

### Age of Acquisition (AoA)
Age of acquisition analysis revealed that Baseline achieved AoA at checkpoint **[AOA_ESTIMATE]** (95% CI [**[AOA_CI_LOW]**, **[AOA_CI_HIGH]**]). **[COMPARISON_TO_OTHER_MODELS_AOA]**.

## End-State Performance

### Null Subject Preference
End-state analysis revealed that the Baseline model showed a null subject preference of **[NULL_PREF_PROB]** (95% CI [**[NULL_PREF_CI_LOW]**, **[NULL_PREF_CI_HIGH]**]). **[COMPARISON_DESCRIPTION]**.

### Null-Overt Performance Gap
Performance gap analysis showed that the Baseline model exhibited a null-overt accuracy difference of **[GAP_ESTIMATE]** (95% CI [**[GAP_CI_LOW]**, **[GAP_CI_HIGH]**], p **[GAP_P_VALUE]**). **[INTERPRETATION_OF_GAP]**.

## First-Epoch Learning

First epoch analysis demonstrated that the Baseline model achieved significant null subject learning by epoch end, showing **[FIRST_EPOCH_PERCENT]**% null preference (SD = **[FIRST_EPOCH_SD]**%, p **[FIRST_EPOCH_P_VALUE]**), **[SIGNIFICANCE_INTERPRETATION]**.

## Item Group Analysis

### Overall Item Group Effects
Item group analysis in the Baseline condition revealed a hierarchy of null subject preferences across syntactic contexts:

**Highest null preference contexts:**
1. **[HIGHEST_ITEM_GROUP_1]**: **[HIGHEST_PROB_1]** (95% CI [**[HIGHEST_CI_LOW_1]**, **[HIGHEST_CI_HIGH_1]**])
2. **[HIGHEST_ITEM_GROUP_2]**: **[HIGHEST_PROB_2]** (95% CI [**[HIGHEST_CI_LOW_2]**, **[HIGHEST_CI_HIGH_2]**])
3. **[HIGHEST_ITEM_GROUP_3]**: **[HIGHEST_PROB_3]** (95% CI [**[HIGHEST_CI_LOW_3]**, **[HIGHEST_CI_HIGH_3]**])

**Lowest null preference contexts:**
1. **[LOWEST_ITEM_GROUP_1]**: **[LOWEST_PROB_1]** (95% CI [**[LOWEST_CI_LOW_1]**, **[LOWEST_CI_HIGH_1]**])
2. **[LOWEST_ITEM_GROUP_2]**: **[LOWEST_PROB_2]** (95% CI [**[LOWEST_CI_LOW_2]**, **[LOWEST_CI_HIGH_2]**])
3. **[LOWEST_ITEM_GROUP_3]**: **[LOWEST_PROB_3]** (95% CI [**[LOWEST_CI_LOW_3]**, **[LOWEST_CI_HIGH_3]**])

### Key Theoretical Contrasts

**Person Effects:**
- Third-person vs. first-person contexts: **[3RD_VS_1ST_OR]** (SE = **[3RD_VS_1ST_SE]**, z = **[3RD_VS_1ST_Z]**, p_FDR **[3RD_VS_1ST_P]**). **[INTERPRETATION]**.

**Number Effects:**
- Singular vs. plural contexts: **[SG_VS_PL_OR]** (SE = **[SG_VS_PL_SE]**, z = **[SG_VS_PL_Z]**, p_FDR **[SG_VS_PL_P]**). **[INTERPRETATION]**.

**Control Construction Effects:**
- Subject vs. object control: **[SUBJ_VS_OBJ_CONTROL_OR]** (SE = **[SUBJ_VS_OBJ_CONTROL_SE]**, z = **[SUBJ_VS_OBJ_CONTROL_Z]**, p_FDR **[SUBJ_VS_OBJ_CONTROL_P]**). **[INTERPRETATION]**.

**Expletive Effects:**
- "Seems" vs. "be" expletives: **[SEEMS_VS_BE_OR]** (SE = **[SEEMS_VS_BE_SE]**, z = **[SEEMS_VS_BE_Z]**, p_FDR **[SEEMS_VS_BE_P]**). **[INTERPRETATION]**.

### Linguistic Form Analysis

Analysis of linguistic form effects within the Baseline model revealed:

**Context-dependent patterns:**
- Complex embedding vs. default contexts: **[COMPLEX_EMB_OR]** (SE = **[COMPLEX_EMB_SE]**, z = **[COMPLEX_EMB_Z]**, p_FDR **[COMPLEX_EMB_P]**). **[INTERPRETATION]**.
- Target negation vs. default contexts: **[TARGET_NEG_OR]** (SE = **[TARGET_NEG_SE]**, z = **[TARGET_NEG_Z]**, p_FDR **[TARGET_NEG_P]**). **[INTERPRETATION]**.
- Context negation vs. default contexts: **[CONTEXT_NEG_OR]** (SE = **[CONTEXT_NEG_SE]**, z = **[CONTEXT_NEG_Z]**, p_FDR **[CONTEXT_NEG_P]**). **[INTERPRETATION]**.

## Summary

The Baseline model demonstrated **[OVERALL_LEARNING_PATTERN]** with **[T50_SUMMARY]** and final null preference of **[FINAL_PREF_SUMMARY]**. Item group analysis revealed **[ITEM_GROUP_SUMMARY]** with **[KEY_CONTRASTS_SUMMARY]**. These patterns establish the foundation for comparison with experimental manipulations.

---

## Data Sources for Fill-in Values:

**Model Selection:** `analysis/tables/tests/df_selection_by_aic.csv` & `df_comparison_detailed.csv`
**t50 Analysis:** `analysis/tables/tests/t50_by_model_robust.csv`
**AoA Analysis:** `analysis/tables/tests/aoa_halfway_by_model.csv`
**End-state Preference:** `analysis/tables/tests/endstate_null_pref.csv`
**Performance Gap:** `analysis/tables/tests/endstate_gap_null_minus_overt.csv`
**First Epoch:** `analysis/tables/first_epoch_summary.csv`
**Item Groups:** `analysis/tables/tests/endstate_itemgroup_baseline.csv` & `endstate_itemgroup_baseline_pairwise.csv`
**Form Analysis:** `analysis/tables/tests/endstate_forms_within_model.csv`

## Instructions:
1. Extract values from the corresponding CSV files
2. Round appropriately (3 decimal places for probabilities, 2 for ORs, etc.)
3. Use p < .001 format for very small p-values
4. Add interpretive language for significant vs. non-significant findings
5. Maintain parallel structure across sections