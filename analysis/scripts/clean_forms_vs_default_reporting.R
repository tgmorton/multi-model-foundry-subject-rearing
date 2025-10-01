# Clean Forms vs Default - Reporting Version
# ==========================================
# Create a clean version of forms vs default with key reporting measures

library(tidyverse)

# Load the corrected forms vs default data
forms_vs_default <- read_csv("analysis/tables/pairwise_c1_forms_vs_default_corrected.csv", show_col_types = FALSE)

# Clean and format for reporting
clean_forms_reporting <- forms_vs_default %>%
  # Keep only essential columns and clean up
  select(
    Model,
    form,
    form_prop, 
    default_prop,
    diff,
    p_value,
    p_value_fdr_within,
    significant_fdr_within
  ) %>%
  # Format percentages and differences
  mutate(
    `Form %` = sprintf("%.1f%%", form_prop * 100),
    `Default %` = sprintf("%.1f%%", default_prop * 100), 
    `Difference` = sprintf("%+.1f%%", diff * 100),
    `p-value` = case_when(
      p_value < 0.001 ~ "< .001",
      p_value < 0.01 ~ sprintf("%.3f", p_value),
      TRUE ~ sprintf("%.3f", p_value)
    ),
    `p-corrected` = case_when(
      p_value_fdr_within < 0.001 ~ "< .001", 
      p_value_fdr_within < 0.01 ~ sprintf("%.3f", p_value_fdr_within),
      TRUE ~ sprintf("%.3f", p_value_fdr_within)
    ),
    `Significant` = ifelse(significant_fdr_within, "***", "ns"),
    `Effect` = case_when(
      !significant_fdr_within ~ "No difference",
      diff > 0 ~ sprintf("%s > default", str_replace_all(form, "_", " ")),
      diff < 0 ~ sprintf("default > %s", str_replace_all(form, "_", " ")),
      TRUE ~ "No difference"
    ),
    # Clean form names for display
    `Form` = case_when(
      form == "both_negation" ~ "Both Negation",
      form == "complex_emb" ~ "Complex Embedded", 
      form == "complex_long" ~ "Complex Long",
      form == "context_negation" ~ "Context Negation",
      form == "target_negation" ~ "Target Negation",
      TRUE ~ str_to_title(str_replace_all(form, "_", " "))
    )
  ) %>%
  # Select final columns for reporting
  select(
    Model,
    Form,
    `Form %`,
    `Default %`, 
    Difference,
    `p-value`,
    `p-corrected`,
    Significant,
    Effect
  ) %>%
  # Order models and forms properly
  mutate(
    model_order = case_when(
      Model == "Baseline" ~ 1,
      Model == "Remove Expletives" ~ 2,
      Model == "Impoverish Determiners" ~ 3,
      Model == "Remove Articles" ~ 4,
      Model == "Lemmatize Verbs" ~ 5,
      Model == "Remove Subject Pronominals" ~ 6,
      TRUE ~ 99
    ),
    form_order = case_when(
      Form == "Target Negation" ~ 1,
      Form == "Context Negation" ~ 2,
      Form == "Both Negation" ~ 3,
      Form == "Complex Embedded" ~ 4,
      Form == "Complex Long" ~ 5,
      TRUE ~ 99
    )
  ) %>%
  arrange(model_order, form_order) %>%
  select(-model_order, -form_order)

# Save the clean reporting version
write_csv(clean_forms_reporting, "analysis/tables/forms_vs_default_reporting.csv")

# Print summary
cat("=== FORMS VS DEFAULT - REPORTING VERSION ===\n")
cat(sprintf("Generated clean reporting table with %d comparisons\n", nrow(clean_forms_reporting)))
cat("Saved to: analysis/tables/forms_vs_default_reporting.csv\n\n")

# Show sample for each model
cat("Sample entries:\n")
sample_data <- clean_forms_reporting %>%
  group_by(Model) %>%
  slice_head(n = 2) %>%
  ungroup()

print(sample_data, n = 12)

cat("\n=== SUMMARY BY SIGNIFICANCE ===\n")
significance_summary <- clean_forms_reporting %>%
  group_by(Model) %>%
  summarise(
    `Total Comparisons` = n(),
    `Significant` = sum(Significant == "***"),
    `% Significant` = sprintf("%.1f%%", 100 * `Significant` / `Total Comparisons`),
    .groups = "drop"
  )

print(significance_summary, n = 6)