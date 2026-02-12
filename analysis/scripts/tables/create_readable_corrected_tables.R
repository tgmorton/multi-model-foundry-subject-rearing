# Create Human-Readable Corrected Comparison Tables
# =================================================

library(tidyverse)

cat("=== CREATING READABLE CORRECTED TABLES ===\n\n")

# 1. CROSS-MODEL COMPARISONS (First Epoch & End-State)
cat("1. Cross-Model Comparisons\n")
cat("==========================\n")

# First Epoch vs Baseline (corrected)
first_epoch_corrected <- read_csv("analysis/tables/full_first_epoch_vs_baseline_corrected.csv", show_col_types = FALSE)

first_epoch_readable <- first_epoch_corrected %>%
  select(Model, baseline_null_pref, treatment_null_pref, difference, odds_ratio, 
         or_ci_low, or_ci_high, p_value, p_value_bonferroni, significant_bonferroni) %>%
  mutate(
    `Baseline Null %` = sprintf("%.1f%%", baseline_null_pref * 100),
    `Treatment Null %` = sprintf("%.1f%%", treatment_null_pref * 100),
    `Difference` = sprintf("%+.1f%%", difference * 100),
    `Odds Ratio` = sprintf("%.3f", odds_ratio),
    `OR 95% CI` = sprintf("[%.3f, %.3f]", or_ci_low, or_ci_high),
    `p-value` = ifelse(p_value < 0.001, "< .001", sprintf("%.4f", p_value)),
    `p-value (Bonf.)` = ifelse(p_value_bonferroni < 0.001, "< .001", sprintf("%.4f", p_value_bonferroni)),
    `Significant` = ifelse(significant_bonferroni, "Yes", "No"),
    `Effect` = case_when(
      !significant_bonferroni ~ "No difference",
      difference > 0 ~ "Higher null pref",
      difference < 0 ~ "Lower null pref"
    )
  ) %>%
  select(Model, `Baseline Null %`, `Treatment Null %`, `Difference`, 
         `Odds Ratio`, `OR 95% CI`, `p-value`, `p-value (Bonf.)`, `Significant`, `Effect`) %>%
  arrange(desc(parse_number(`Treatment Null %`)))

cat("\nFIRST EPOCH vs BASELINE (Bonferroni corrected, α = 0.01)\n")
write_csv(first_epoch_readable, "analysis/tables/first_epoch_vs_baseline_readable.csv")
print(first_epoch_readable, n = Inf)

# End-State vs Baseline (corrected)
endstate_corrected <- read_csv("analysis/tables/endstate_vs_baseline_corrected.csv", show_col_types = FALSE)

endstate_readable <- endstate_corrected %>%
  select(Model, baseline_overt_pref, treatment_overt_pref, difference, odds_ratio,
         or_ci_low, or_ci_high, p_value, p_value_bonferroni, significant_bonferroni) %>%
  mutate(
    `Baseline Overt %` = sprintf("%.1f%%", baseline_overt_pref * 100),
    `Treatment Overt %` = sprintf("%.1f%%", treatment_overt_pref * 100),
    `Difference` = sprintf("%+.1f%%", difference * 100),
    `Odds Ratio` = sprintf("%.3f", odds_ratio),
    `OR 95% CI` = sprintf("[%.3f, %.3f]", or_ci_low, or_ci_high),
    `p-value` = ifelse(p_value < 0.001, "< .001", sprintf("%.4f", p_value)),
    `p-value (Bonf.)` = ifelse(p_value_bonferroni < 0.001, "< .001", sprintf("%.4f", p_value_bonferroni)),
    `Significant` = ifelse(significant_bonferroni, "Yes", "No"),
    `Effect` = case_when(
      !significant_bonferroni ~ "No difference",
      difference > 0 ~ "Higher overt pref",
      difference < 0 ~ "Lower overt pref"
    )
  ) %>%
  select(Model, `Baseline Overt %`, `Treatment Overt %`, `Difference`,
         `Odds Ratio`, `OR 95% CI`, `p-value`, `p-value (Bonf.)`, `Significant`, `Effect`) %>%
  arrange(desc(parse_number(`Treatment Overt %`)))

cat("\nEND-STATE vs BASELINE (Bonferroni corrected, α = 0.01)\n")
write_csv(endstate_readable, "analysis/tables/endstate_vs_baseline_readable.csv")
print(endstate_readable, n = Inf)

# 2. FORM PAIRWISE COMPARISONS
cat("\n\n2. Form Pairwise Comparisons\n")
cat("============================\n")

form_corrected <- read_csv("analysis/tables/form_pairwise_comparisons_corrected.csv", show_col_types = FALSE)

form_readable <- form_corrected %>%
  select(Model, comparison, form1, form1_prob, form2, form2_prob, 
         odds_ratio, or_ci_low, or_ci_high, p_value, p_value_fdr, significant_fdr) %>%
  mutate(
    `Comparison` = case_when(
      comparison == "complex_emb_vs_complex_long" ~ "Complex Emb vs Long",
      comparison == "target_negation_vs_context_negation" ~ "Target vs Context Neg",
      comparison == "target_negation_vs_both_negation" ~ "Target vs Both Neg",
      TRUE ~ comparison
    ),
    `Form 1` = str_replace_all(form1, "_", " "),
    `Form 1 %` = sprintf("%.1f%%", form1_prob * 100),
    `Form 2` = str_replace_all(form2, "_", " "),
    `Form 2 %` = sprintf("%.1f%%", form2_prob * 100),
    `Odds Ratio` = sprintf("%.3f", odds_ratio),
    `OR 95% CI` = sprintf("[%.3f, %.3f]", or_ci_low, or_ci_high),
    `p-value` = ifelse(p_value < 0.001, "< .001", sprintf("%.4f", p_value)),
    `p-value (FDR)` = ifelse(p_value_fdr < 0.001, "< .001", sprintf("%.4f", p_value_fdr)),
    `Significant` = ifelse(significant_fdr, "Yes", "No"),
    `Winner` = case_when(
      !significant_fdr ~ "No difference",
      odds_ratio < 1 ~ paste(form1, ">", form2),
      odds_ratio > 1 ~ paste(form2, ">", form1),
      TRUE ~ "Tie"
    )
  ) %>%
  select(Model, `Comparison`, `Form 1`, `Form 1 %`, `Form 2`, `Form 2 %`,
         `Odds Ratio`, `OR 95% CI`, `p-value`, `p-value (FDR)`, `Significant`, `Winner`) %>%
  arrange(Model, `Comparison`)

cat("\nFORM PAIRWISE COMPARISONS (FDR corrected)\n")
write_csv(form_readable, "analysis/tables/form_pairwise_comparisons_readable.csv")
print(form_readable, n = Inf)

# 3. ITEM GROUP COMPARISONS SUMMARY
cat("\n\n3. Item Group Comparisons Summary\n")
cat("=================================\n")

# Read the comprehensive corrected item group data
if (file.exists("analysis/tables/all_item_group_comparisons_corrected.csv")) {
  item_groups_corrected <- read_csv("analysis/tables/all_item_group_comparisons_corrected.csv", show_col_types = FALSE)
  
  # Create summary by analysis type
  item_groups_summary <- item_groups_corrected %>%
    group_by(analysis_group) %>%
    summarise(
      `Total Tests` = n(),
      `Significant (Uncorrected)` = sum(significant_uncorrected, na.rm = TRUE),
      `Significant (FDR)` = sum(significant_fdr_global, na.rm = TRUE),
      `% Significant (Uncorrected)` = sprintf("%.1f%%", 100 * `Significant (Uncorrected)` / `Total Tests`),
      `% Significant (FDR)` = sprintf("%.1f%%", 100 * `Significant (FDR)` / `Total Tests`),
      .groups = "drop"
    ) %>%
    mutate(
      `Analysis Type` = case_when(
        analysis_group == "pairwise_b1" ~ "Number Contrasts",
        analysis_group == "pairwise_b2" ~ "Person Contrasts", 
        analysis_group == "pairwise_b3" ~ "Control Contrasts",
        analysis_group == "pairwise_b4" ~ "Expletive Contrasts",
        analysis_group == "pairwise_b5" ~ "Topic Shift Contrasts",
        analysis_group == "pairwise_c1" ~ "Forms vs Default",
        analysis_group == "pairwise_c2" ~ "Complex Embedding",
        analysis_group == "pairwise_c3" ~ "Negation Types",
        TRUE ~ analysis_group
      )
    ) %>%
    select(`Analysis Type`, `Total Tests`, `Significant (Uncorrected)`, `Significant (FDR)`,
           `% Significant (Uncorrected)`, `% Significant (FDR)`)
  
  cat("\nITEM GROUP COMPARISONS SUMMARY (FDR corrected across 108 tests)\n")
  write_csv(item_groups_summary, "analysis/tables/item_group_comparisons_summary_readable.csv")
  print(item_groups_summary, n = Inf)
  
  # Overall totals
  total_summary <- item_groups_corrected %>%
    summarise(
      `Total Tests` = n(),
      `Significant (Uncorrected)` = sum(significant_uncorrected, na.rm = TRUE),
      `Significant (FDR)` = sum(significant_fdr_global, na.rm = TRUE),
      `% Lost to Correction` = sprintf("%.1f%%", 100 * (`Significant (Uncorrected)` - `Significant (FDR)`) / `Significant (Uncorrected)`)
    )
  
  cat(sprintf("\nOVERALL ITEM GROUP SUMMARY:\n"))
  cat(sprintf("Total tests: %d\n", total_summary$`Total Tests`))
  cat(sprintf("Significant before correction: %d\n", total_summary$`Significant (Uncorrected)`))
  cat(sprintf("Significant after FDR correction: %d\n", total_summary$`Significant (FDR)`))
  cat(sprintf("Percentage lost to correction: %s\n", total_summary$`% Lost to Correction`))
  
  # Show which specific results changed
  changes <- item_groups_corrected %>%
    filter(significant_uncorrected != significant_fdr_global) %>%
    select(analysis_group, model, p_value, p_value_fdr_global, significant_uncorrected, significant_fdr_global) %>%
    mutate(
      `Analysis` = case_when(
        analysis_group == "pairwise_b1" ~ "Number Contrasts",
        TRUE ~ analysis_group
      ),
      `Model` = str_replace_all(model, "exp[0-9]_", ""),
      `Model` = str_replace_all(`Model`, "_", " "),
      `Model` = str_to_title(`Model`),
      `p-value` = sprintf("%.4f", p_value),
      `p-value (FDR)` = sprintf("%.4f", p_value_fdr_global),
      `Change` = "Lost significance"
    ) %>%
    select(`Analysis`, `Model`, `p-value`, `p-value (FDR)`, `Change`)
  
  if (nrow(changes) > 0) {
    cat("\nSPECIFIC CHANGES DUE TO CORRECTION:\n")
    print(changes, n = Inf)
    write_csv(changes, "analysis/tables/item_group_significance_changes.csv")
  } else {
    cat("\nNo specific changes in significance after FDR correction.\n")
  }
}

cat("\n=== READABLE TABLES CREATED ===\n")
cat("Files saved:\n")
cat("  - analysis/tables/first_epoch_vs_baseline_readable.csv\n")
cat("  - analysis/tables/endstate_vs_baseline_readable.csv\n") 
cat("  - analysis/tables/form_pairwise_comparisons_readable.csv\n")
cat("  - analysis/tables/item_group_comparisons_summary_readable.csv\n")
cat("  - analysis/tables/item_group_significance_changes.csv\n")

cat("\nUse these files for reporting - all include multiple comparisons corrections!\n")