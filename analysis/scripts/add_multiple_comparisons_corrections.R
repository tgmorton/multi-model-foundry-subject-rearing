# Add Multiple Comparisons Corrections to All Analyses
# ====================================================

library(tidyverse)

cat("=== ADDING MULTIPLE COMPARISONS CORRECTIONS ===\n\n")

# 1. FIRST EPOCH VS BASELINE COMPARISONS (Bonferroni for 5 tests)
cat("1. First Epoch vs Baseline Comparisons (5 tests - Bonferroni)\n")
cat("================================================================\n")

first_epoch_comp <- read_csv("analysis/tables/full_first_epoch_vs_baseline.csv", show_col_types = FALSE)

first_epoch_corrected <- first_epoch_comp %>%
  mutate(
    p_value_bonferroni = p.adjust(p_value, method = "bonferroni"),
    p_value_fdr = p.adjust(p_value, method = "fdr"),
    significant_uncorrected = p_value < 0.05,
    significant_bonferroni = p_value_bonferroni < 0.05,
    significant_fdr = p_value_fdr < 0.05,
    interpretation_bonferroni = case_when(
      !significant_bonferroni ~ "No difference (after Bonferroni)",
      difference > 0 ~ "Higher null pref than baseline (Bonferroni corrected)",
      difference < 0 ~ "Lower null pref than baseline (Bonferroni corrected)"
    ),
    interpretation_fdr = case_when(
      !significant_fdr ~ "No difference (after FDR)",
      difference > 0 ~ "Higher null pref than baseline (FDR corrected)",
      difference < 0 ~ "Lower null pref than baseline (FDR corrected)"
    )
  )

# Print results
comparison_summary <- first_epoch_corrected %>%
  select(Model, p_value, p_value_bonferroni, p_value_fdr, 
         significant_uncorrected, significant_bonferroni, significant_fdr) %>%
  mutate(
    p_uncorrected = sprintf("%.4f", p_value),
    p_bonf = sprintf("%.4f", p_value_bonferroni),
    p_fdr_adj = sprintf("%.4f", p_value_fdr)
  ) %>%
  select(Model, p_uncorrected, p_bonf, p_fdr_adj, 
         significant_uncorrected, significant_bonferroni, significant_fdr)

print(comparison_summary)

write_csv(first_epoch_corrected, "analysis/tables/full_first_epoch_vs_baseline_corrected.csv")

cat("\nChanges after correction:\n")
changes <- first_epoch_corrected %>%
  filter(significant_uncorrected != significant_bonferroni) %>%
  select(Model, p_value, p_value_bonferroni, significant_uncorrected, significant_bonferroni)

if(nrow(changes) > 0) {
  print(changes)
} else {
  cat("  No changes in significance after Bonferroni correction\n")
}

# 2. END-STATE VS BASELINE COMPARISONS (Bonferroni for 5 tests)
cat("\n\n2. End-State vs Baseline Comparisons (5 tests - Bonferroni)\n")
cat("============================================================\n")

endstate_comp <- read_csv("analysis/tables/endstate_vs_baseline.csv", show_col_types = FALSE)

endstate_corrected <- endstate_comp %>%
  mutate(
    p_value_bonferroni = p.adjust(p_value, method = "bonferroni"),
    p_value_fdr = p.adjust(p_value, method = "fdr"),
    significant_uncorrected = p_value < 0.05,
    significant_bonferroni = p_value_bonferroni < 0.05,
    significant_fdr = p_value_fdr < 0.05,
    interpretation_bonferroni = case_when(
      !significant_bonferroni ~ "No difference (after Bonferroni)",
      difference > 0 ~ "Higher overt pref than baseline (Bonferroni corrected)",
      difference < 0 ~ "Lower overt pref than baseline (Bonferroni corrected)"
    )
  )

endstate_summary <- endstate_corrected %>%
  select(Model, p_value, p_value_bonferroni, p_value_fdr, 
         significant_uncorrected, significant_bonferroni, significant_fdr) %>%
  mutate(
    p_uncorrected = sprintf("%.4f", p_value),
    p_bonf = sprintf("%.4f", p_value_bonferroni),
    p_fdr_adj = sprintf("%.4f", p_value_fdr)
  ) %>%
  select(Model, p_uncorrected, p_bonf, p_fdr_adj, 
         significant_uncorrected, significant_bonferroni, significant_fdr)

print(endstate_summary)

write_csv(endstate_corrected, "analysis/tables/endstate_vs_baseline_corrected.csv")

cat("\nChanges after correction:\n")
endstate_changes <- endstate_corrected %>%
  filter(significant_uncorrected != significant_bonferroni) %>%
  select(Model, p_value, p_value_bonferroni, significant_uncorrected, significant_bonferroni)

if(nrow(endstate_changes) > 0) {
  print(endstate_changes)
} else {
  cat("  No changes in significance after Bonferroni correction\n")
}

# 3. FORM PAIRWISE COMPARISONS (FDR for 18 tests: 3 comparisons × 6 models)
cat("\n\n3. Form Pairwise Comparisons (18 tests - FDR)\n")
cat("==============================================\n")

form_comp <- read_csv("analysis/tables/form_pairwise_comparisons.csv", show_col_types = FALSE)

form_corrected <- form_comp %>%
  mutate(
    p_value_fdr = p.adjust(p_value, method = "fdr"),
    p_value_bonferroni = p.adjust(p_value, method = "bonferroni"),
    significant_uncorrected = p_value < 0.05,
    significant_fdr = p_value_fdr < 0.05,
    significant_bonferroni = p_value_bonferroni < 0.05,
    interpretation_fdr = case_when(
      !significant_fdr ~ "No difference (after FDR)",
      odds_ratio > 1 ~ paste(form1, ">", form2, "(FDR corrected)"),
      odds_ratio < 1 ~ paste(form2, ">", form1, "(FDR corrected)")
    )
  ) %>%
  left_join(
    tribble(
      ~model, ~Model,
      "exp0_baseline", "Baseline",
      "exp1_remove_expletives", "Remove Expletives", 
      "exp2_impoverish_determiners", "Impoverish Determiners",
      "exp3_remove_articles", "Remove Articles",
      "exp4_lemmatize_verbs", "Lemmatize Verbs",
      "exp5_remove_subject_pronominals", "Remove Subject Pronominals"
    ), by = "model"
  )

form_summary <- form_corrected %>%
  select(Model, comparison, p_value, p_value_fdr, p_value_bonferroni,
         significant_uncorrected, significant_fdr, significant_bonferroni) %>%
  mutate(
    p_uncorrected = sprintf("%.4f", pmin(p_value, 0.9999)),
    p_fdr_adj = sprintf("%.4f", p_value_fdr),
    p_bonf = sprintf("%.4f", pmin(p_value_bonferroni, 0.9999))
  ) %>%
  select(Model, comparison, p_uncorrected, p_fdr_adj, p_bonf,
         significant_uncorrected, significant_fdr, significant_bonferroni) %>%
  arrange(Model, comparison)

print(form_summary)

write_csv(form_corrected, "analysis/tables/form_pairwise_comparisons_corrected.csv")

cat("\nChanges after FDR correction:\n")
form_changes <- form_corrected %>%
  filter(significant_uncorrected != significant_fdr) %>%
  select(Model, comparison, p_value, p_value_fdr, significant_uncorrected, significant_fdr)

if(nrow(form_changes) > 0) {
  print(form_changes)
} else {
  cat("  No changes in significance after FDR correction\n")
}

# SUMMARY
cat("\n\n=== SUMMARY OF CORRECTIONS ===\n")
cat("1. First Epoch vs Baseline: Bonferroni correction (5 tests)\n")
cat("2. End-State vs Baseline: Bonferroni correction (5 tests)\n") 
cat("3. Form Pairwise: FDR correction (18 tests)\n")

cat("\nSaved corrected results to:\n")
cat("  - analysis/tables/full_first_epoch_vs_baseline_corrected.csv\n")
cat("  - analysis/tables/endstate_vs_baseline_corrected.csv\n")
cat("  - analysis/tables/form_pairwise_comparisons_corrected.csv\n")

cat("\nIMPORTANT: Use corrected p-values for all reporting!\n")