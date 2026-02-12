# Create Final Comprehensive Table with ALL Mixed-Effects Comparisons
# ====================================================================

library(tidyverse)

cat("Creating final comprehensive table with consistent mixed-effects methodology...\n")

all_comparisons <- tibble()

# 1. CROSS-MODEL COMPARISONS (First Epoch vs Baseline)
first_epoch <- read_csv("analysis/tables/full_first_epoch_vs_baseline_corrected.csv", show_col_types = FALSE) %>%
  mutate(
    analysis_category = "Cross-Model",
    analysis_type = "First Epoch vs Baseline",
    comparison_name = paste(Model, "vs Baseline"),
    group1_name = "Baseline",
    group2_name = Model,
    group1_pref = baseline_null_pref,
    group2_pref = treatment_null_pref,
    measure_type = "Null Subject Preference",
    corrected_p = p_value_bonferroni,
    correction_type = "Bonferroni",
    significant_corrected = significant_bonferroni,
    statistical_test = "Mixed-Effects Logistic",
    # Default values for Fisher's exact (not applicable)
    group1_n_success = NA_integer_,
    group1_n_total = NA_integer_,
    group2_n_success = NA_integer_,
    group2_n_total = NA_integer_
  ) %>%
  select(model, Model, analysis_category, analysis_type, comparison_name,
         group1_name, group1_pref, group2_name, group2_pref, measure_type,
         difference, odds_ratio, or_ci_low, or_ci_high, p_value, 
         corrected_p, correction_type, significant_corrected, statistical_test,
         group1_n_success, group1_n_total, group2_n_success, group2_n_total)

all_comparisons <- bind_rows(all_comparisons, first_epoch)

# 2. CROSS-MODEL COMPARISONS (End-State vs Baseline)  
endstate <- read_csv("analysis/tables/endstate_vs_baseline_corrected.csv", show_col_types = FALSE) %>%
  mutate(
    analysis_category = "Cross-Model",
    analysis_type = "End-State vs Baseline", 
    comparison_name = paste(Model, "vs Baseline"),
    group1_name = "Baseline",
    group2_name = Model,
    group1_pref = baseline_overt_pref,
    group2_pref = treatment_overt_pref,
    measure_type = "Overt Subject Preference",
    corrected_p = p_value_bonferroni,
    correction_type = "Bonferroni",
    significant_corrected = significant_bonferroni,
    statistical_test = "Mixed-Effects Logistic",
    # Default values for Fisher's exact (not applicable)
    group1_n_success = NA_integer_,
    group1_n_total = NA_integer_,
    group2_n_success = NA_integer_,
    group2_n_total = NA_integer_
  ) %>%
  select(model, Model, analysis_category, analysis_type, comparison_name,
         group1_name, group1_pref, group2_name, group2_pref, measure_type,
         difference, odds_ratio, or_ci_low, or_ci_high, p_value,
         corrected_p, correction_type, significant_corrected, statistical_test,
         group1_n_success, group1_n_total, group2_n_success, group2_n_total)

all_comparisons <- bind_rows(all_comparisons, endstate)

# 3. FORM PAIRWISE COMPARISONS (Using full dataset version)
forms <- read_csv("analysis/tables/form_full_dataset_pairwise_corrected.csv", show_col_types = FALSE) %>%
  mutate(
    analysis_category = "Form Pairwise",
    analysis_type = case_when(
      comparison == "complex_emb_vs_complex_long" ~ "Complex Forms",
      comparison == "target_negation_vs_context_negation" ~ "Negation Types",  
      comparison == "target_negation_vs_both_negation" ~ "Negation Types",
      TRUE ~ "Form Comparison"
    ),
    comparison_name = case_when(
      comparison == "complex_emb_vs_complex_long" ~ "Complex Embedded vs Long",
      comparison == "target_negation_vs_context_negation" ~ "Target vs Context Negation",
      comparison == "target_negation_vs_both_negation" ~ "Target vs Both Negation",
      TRUE ~ comparison
    ),
    group1_name = str_replace_all(form1, "_", " "),
    group2_name = str_replace_all(form2, "_", " "),
    group1_pref = form1_prob,
    group2_pref = form2_prob,
    measure_type = "Overt Subject Preference",
    difference = form1_prob - form2_prob,
    corrected_p = p_value_fdr,
    correction_type = "FDR",
    significant_corrected = significant_fdr,
    statistical_test = "Mixed-Effects Logistic",
    # Default values for Fisher's exact (not applicable)
    group1_n_success = NA_integer_,
    group1_n_total = NA_integer_,
    group2_n_success = NA_integer_,
    group2_n_total = NA_integer_
  ) %>%
  select(model, Model, analysis_category, analysis_type, comparison_name,
         group1_name, group1_pref, group2_name, group2_pref, measure_type,
         difference, odds_ratio, or_ci_low, or_ci_high, p_value,
         corrected_p, correction_type, significant_corrected, statistical_test,
         group1_n_success, group1_n_total, group2_n_success, group2_n_total)

all_comparisons <- bind_rows(all_comparisons, forms)

# 4. ITEM GROUP PAIRWISE COMPARISONS (Using full dataset version)
item_groups <- read_csv("analysis/tables/item_group_full_dataset_pairwise_corrected.csv", show_col_types = FALSE) %>%
  mutate(
    analysis_category = "Item Group Pairwise",
    analysis_type = case_when(
      str_detect(comparison_type, "B1_Number") ~ "Number Contrasts",
      str_detect(comparison_type, "B2_Person") ~ "Person Contrasts", 
      str_detect(comparison_type, "B3_Control") ~ "Control Contrasts",
      str_detect(comparison_type, "B4_Expletive") ~ "Expletive Contrasts",
      str_detect(comparison_type, "B5_Topic") ~ "Topic Shift Contrasts",
      TRUE ~ "Other"
    ),
    comparison_name = case_when(
      comparison_type == "B1_Number_1st_Sing_vs_Plural" ~ "1st Person: Singular vs Plural",
      comparison_type == "B1_Number_2nd_Sing_vs_Plural" ~ "2nd Person: Singular vs Plural",
      comparison_type == "B1_Number_3rd_Sing_vs_Plural" ~ "3rd Person: Singular vs Plural",
      comparison_type == "B2_Person_1st_vs_2nd" ~ "1st vs 2nd Person",
      comparison_type == "B2_Person_1st_vs_3rd" ~ "1st vs 3rd Person", 
      comparison_type == "B2_Person_2nd_vs_3rd" ~ "2nd vs 3rd Person",
      comparison_type == "B3_Control_Subject_vs_Object" ~ "Subject vs Object Control",
      comparison_type == "B4_Expletive_Seems_vs_Be" ~ "Seems vs Be Expletives",
      comparison_type == "B5_Topic_NoShift_vs_Shift" ~ "No Topic Shift vs Topic Shift",
      TRUE ~ comparison_type
    ),
    group1_name = str_replace_all(group1, "_", " "),
    group2_name = str_replace_all(group2, "_", " "),
    group1_pref = group1_prob,
    group2_pref = group2_prob,
    measure_type = "Overt Subject Preference",
    difference = group1_prob - group2_prob,
    corrected_p = p_value_fdr,
    correction_type = "FDR",
    significant_corrected = significant_fdr,
    # Add statistical test indicator
    statistical_test = ifelse(is.na(z_value), "Fisher's Exact", "Mixed-Effects Logistic"),
    # Include Fisher's exact test specific information
    group1_n_success = ifelse(is.na(z_value), group1_n_success, NA_integer_),
    group1_n_total = ifelse(is.na(z_value), group1_n_total, NA_integer_),
    group2_n_success = ifelse(is.na(z_value), group2_n_success, NA_integer_),  
    group2_n_total = ifelse(is.na(z_value), group2_n_total, NA_integer_)
  ) %>%
  select(model, Model, analysis_category, analysis_type, comparison_name,
         group1_name, group1_pref, group2_name, group2_pref, measure_type,
         difference, odds_ratio, or_ci_low, or_ci_high, p_value,
         corrected_p, correction_type, significant_corrected, statistical_test,
         group1_n_success, group1_n_total, group2_n_success, group2_n_total)

all_comparisons <- bind_rows(all_comparisons, item_groups)

# FORMAT FOR FINAL TABLE
final_table <- all_comparisons %>%
  mutate(
    `Group 1 %` = sprintf("%.1f%%", group1_pref * 100),
    `Group 2 %` = sprintf("%.1f%%", group2_pref * 100),
    `Difference` = sprintf("%+.1f%%", difference * 100),
    `Odds Ratio` = sprintf("%.3f", odds_ratio),
    `OR 95% CI` = sprintf("[%.3f, %.3f]", or_ci_low, or_ci_high),
    `p-value` = ifelse(p_value < 0.001, "< .001", sprintf("%.4f", p_value)),
    `p-corrected` = ifelse(corrected_p < 0.001, "< .001", sprintf("%.4f", corrected_p)),
    `Significant` = ifelse(significant_corrected, "***", "ns"),
    `Effect` = case_when(
      !significant_corrected ~ "No difference",
      difference > 0 ~ paste(group1_name, ">", group2_name),
      difference < 0 ~ paste(group2_name, ">", group1_name),
      TRUE ~ "No difference"
    ),
    # Format Fisher's exact test information
    `Group 1 N Success` = ifelse(!is.na(group1_n_success), as.character(group1_n_success), NA_character_),
    `Group 1 N Total` = ifelse(!is.na(group1_n_total), as.character(group1_n_total), NA_character_),
    `Group 2 N Success` = ifelse(!is.na(group2_n_success), as.character(group2_n_success), NA_character_),
    `Group 2 N Total` = ifelse(!is.na(group2_n_total), as.character(group2_n_total), NA_character_)
  ) %>%
  select(
    `Model` = Model,
    `Analysis Category` = analysis_category,
    `Analysis Type` = analysis_type,
    `Comparison` = comparison_name,
    `Group 1` = group1_name,
    `Group 1 %`,
    `Group 2` = group2_name, 
    `Group 2 %`,
    `Difference`,
    `Odds Ratio`,
    `OR 95% CI`,
    `p-value`,
    `p-corrected`,
    `Significant`,
    `Effect`,
    `Measure` = measure_type,
    `Correction` = correction_type,
    `Statistical Test` = statistical_test,
    `Group 1 N Success`,
    `Group 1 N Total`, 
    `Group 2 N Success`,
    `Group 2 N Total`
  ) %>%
  arrange(`Model`, `Analysis Category`, `Analysis Type`, `Comparison`)

# Save final comprehensive table
write_csv(final_table, "analysis/tables/final_comprehensive_mixed_effects_comparisons.csv")

cat(sprintf("Created final comprehensive table with %d comparisons\n", nrow(final_table)))

# Summary by category
summary_by_category <- final_table %>%
  group_by(`Analysis Category`, `Correction`) %>%
  summarise(
    `Total Tests` = n(),
    `Significant` = sum(`Significant` == "***"),
    `% Significant` = sprintf("%.1f%%", 100 * `Significant` / `Total Tests`),
    .groups = "drop"
  )

cat("\nSummary by Analysis Category:\n")
print(summary_by_category)

cat(sprintf("\nOverall Summary:\n"))
cat(sprintf("Total comparisons: %d\n", nrow(final_table)))
cat(sprintf("Total significant (corrected): %d (%.1f%%)\n", 
            sum(final_table$Significant == "***"),
            100 * sum(final_table$Significant == "***") / nrow(final_table)))

cat("\nSaved to: analysis/tables/final_comprehensive_mixed_effects_comparisons.csv\n")
cat("\nAll comparisons now use consistent mixed-effects + odds ratios with appropriate corrections!\n")