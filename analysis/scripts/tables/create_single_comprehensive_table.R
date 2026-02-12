# Create Single Comprehensive CSV with All Corrected Comparisons
# =============================================================

library(tidyverse)

cat("Creating single comprehensive corrected comparisons file...\n")

# Initialize comprehensive results
all_comparisons <- tibble()

# 1. FIRST EPOCH VS BASELINE
if (file.exists("analysis/tables/full_first_epoch_vs_baseline_corrected.csv")) {
  first_epoch <- read_csv("analysis/tables/full_first_epoch_vs_baseline_corrected.csv", show_col_types = FALSE) %>%
    mutate(
      analysis_type = "First Epoch vs Baseline",
      comparison_description = paste(Model, "vs Baseline (First Epoch)"),
      measure = "Null Subject Preference",
      baseline_value = baseline_null_pref,
      treatment_value = treatment_null_pref,
      corrected_p_value = p_value_bonferroni,
      correction_method = "Bonferroni",
      significant_corrected = significant_bonferroni
    )
  
  all_comparisons <- bind_rows(all_comparisons, first_epoch)
}

# 2. END-STATE VS BASELINE  
if (file.exists("analysis/tables/endstate_vs_baseline_corrected.csv")) {
  endstate <- read_csv("analysis/tables/endstate_vs_baseline_corrected.csv", show_col_types = FALSE) %>%
    mutate(
      analysis_type = "End-State vs Baseline", 
      comparison_description = paste(Model, "vs Baseline (End-State)"),
      measure = "Overt Subject Preference",
      baseline_value = baseline_overt_pref,
      treatment_value = treatment_overt_pref,
      corrected_p_value = p_value_bonferroni,
      correction_method = "Bonferroni",
      significant_corrected = significant_bonferroni
    )
  
  all_comparisons <- bind_rows(all_comparisons, endstate)
}

# 3. FORM PAIRWISE COMPARISONS
if (file.exists("analysis/tables/form_pairwise_comparisons_corrected.csv")) {
  forms <- read_csv("analysis/tables/form_pairwise_comparisons_corrected.csv", show_col_types = FALSE) %>%
    mutate(
      analysis_type = "Form Pairwise",
      comparison_description = paste(Model, "-", 
                                   case_when(
                                     comparison == "complex_emb_vs_complex_long" ~ "Complex Emb vs Long",
                                     comparison == "target_negation_vs_context_negation" ~ "Target vs Context Negation",
                                     comparison == "target_negation_vs_both_negation" ~ "Target vs Both Negation",
                                     TRUE ~ comparison
                                   )),
      measure = "Overt Subject Preference",
      baseline_value = form2_prob,  # Form 2 as reference
      treatment_value = form1_prob, # Form 1 as treatment
      difference = form1_prob - form2_prob,
      corrected_p_value = p_value_fdr,
      correction_method = "FDR",
      significant_corrected = significant_fdr
    )
}

# Clean up form data to match structure
if (exists("forms")) {
  forms <- forms %>%
    select(model, Model, analysis_type, comparison_description, measure, 
           baseline_value, treatment_value, difference, odds_ratio, or_ci_low, or_ci_high,
           p_value, corrected_p_value, correction_method, significant_corrected,
           form1, form2, comparison)
  
  all_comparisons <- bind_rows(all_comparisons, forms)
}

# CREATE FINAL COMPREHENSIVE TABLE
comprehensive_table <- all_comparisons %>%
  # Ensure consistent columns across all analyses
  mutate(
    baseline_pct = sprintf("%.1f%%", baseline_value * 100),
    treatment_pct = sprintf("%.1f%%", treatment_value * 100),
    difference_pct = sprintf("%+.1f%%", difference * 100),
    odds_ratio_formatted = sprintf("%.3f", odds_ratio),
    or_ci_formatted = sprintf("[%.3f, %.3f]", or_ci_low, or_ci_high),
    p_value_formatted = ifelse(p_value < 0.001, "< .001", sprintf("%.4f", p_value)),
    corrected_p_formatted = ifelse(corrected_p_value < 0.001, "< .001", sprintf("%.4f", corrected_p_value)),
    significant_symbol = ifelse(significant_corrected, "***", "ns"),
    effect_direction = case_when(
      !significant_corrected ~ "No difference",
      difference > 0 ~ paste("Higher", str_to_lower(measure)),
      difference < 0 ~ paste("Lower", str_to_lower(measure)),
      TRUE ~ "No difference"
    )
  ) %>%
  # Select final columns in logical order
  select(
    analysis_type,
    Model,
    comparison_description,
    measure,
    baseline_pct,
    treatment_pct, 
    difference_pct,
    odds_ratio_formatted,
    or_ci_formatted,
    p_value_formatted,
    corrected_p_formatted,
    correction_method,
    significant_corrected,
    significant_symbol,
    effect_direction
  ) %>%
  # Rename columns for clarity
  rename(
    `Analysis Type` = analysis_type,
    `Model` = Model,
    `Comparison` = comparison_description,
    `Measure` = measure,
    `Baseline %` = baseline_pct,
    `Treatment %` = treatment_pct,
    `Difference` = difference_pct,
    `Odds Ratio` = odds_ratio_formatted,
    `OR 95% CI` = or_ci_formatted,
    `p-value` = p_value_formatted,
    `p-value (corrected)` = corrected_p_formatted,
    `Correction` = correction_method,
    `Significant` = significant_corrected,
    `Sig.` = significant_symbol,
    `Effect` = effect_direction
  ) %>%
  # Sort logically
  arrange(`Analysis Type`, `Model`, `Comparison`)

# Save comprehensive table
write_csv(comprehensive_table, "analysis/tables/all_corrected_comparisons_comprehensive.csv")

cat(sprintf("Created comprehensive table with %d comparisons\n", nrow(comprehensive_table)))
cat("Saved to: analysis/tables/all_corrected_comparisons_comprehensive.csv\n\n")

# Print preview
cat("Preview of comprehensive table:\n")
print(comprehensive_table, n = 10, width = Inf)

cat(sprintf("\nColumns included:\n"))
cat(paste("  -", names(comprehensive_table), collapse = "\n"))

cat(sprintf("\n\nBreakdown by analysis type:\n"))
breakdown <- comprehensive_table %>%
  group_by(`Analysis Type`) %>%
  summarise(
    Count = n(),
    Significant = sum(Significant),
    .groups = "drop"
  )
print(breakdown)