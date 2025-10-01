# Form Pairwise Comparisons LaTeX Tables
# =======================================
# Generates LaTeX tables for form pairwise comparisons

library(tidyverse)
library(xtable)

# Create directory
dir.create("analysis/tables/latex_tables/form_pairwise", recursive = TRUE, showWarnings = FALSE)

# Load results
results <- read_csv("analysis/tables/form_pairwise_comparisons.csv")

# Model labels
model_labels <- tribble(
  ~model, ~Model,
  "exp0_baseline", "Baseline",
  "exp1_remove_expletives", "Remove Expletives", 
  "exp2_impoverish_determiners", "Impoverish Determiners",
  "exp3_remove_articles", "Remove Articles",
  "exp4_lemmatize_verbs", "Lemmatize Verbs",
  "exp5_remove_subject_pronominals", "Remove Subject Pronominals"
)

results <- results %>%
  left_join(model_labels, by = "model")

# Create readable comparison names
comparison_labels <- tribble(
  ~comparison, ~Comparison,
  "complex_emb_vs_complex_long", "Complex Embedded vs Complex Long",
  "target_negation_vs_context_negation", "Target Negation vs Context Negation", 
  "target_negation_vs_both_negation", "Target Negation vs Both Negation"
)

results <- results %>%
  left_join(comparison_labels, by = "comparison")

# Generate tables for each model
for (model_name in unique(results$model)) {
  model_label <- results %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  model_data <- results %>% 
    filter(model == model_name) %>%
    # Order comparisons consistently
    mutate(comparison = factor(comparison, levels = c(
      "complex_emb_vs_complex_long",
      "target_negation_vs_context_negation", 
      "target_negation_vs_both_negation"
    ))) %>%
    arrange(comparison) %>%
    mutate(comparison = as.character(comparison))
  
  model_clean_name <- str_replace_all(tolower(model_name), "_", "")
  
  # Format for LaTeX
  latex_table <- model_data %>%
    mutate(
      Comparison = case_when(
        comparison == "complex_emb_vs_complex_long" ~ "Complex Emb vs Long",
        comparison == "target_negation_vs_context_negation" ~ "Target vs Context Negation",
        comparison == "target_negation_vs_both_negation" ~ "Target vs Both Negation",
        TRUE ~ comparison
      ),
      `Form 1` = case_when(
        form1 == "complex_emb" ~ "Complex Emb",
        form1 == "target_negation" ~ "Target Negation",
        TRUE ~ str_replace_all(form1, "_", " ")
      ),
      `Pref 1` = sprintf("%.3f", form1_prob),
      `CI 1` = sprintf("[%.3f, %.3f]", form1_ci_low, form1_ci_high),
      `Form 2` = case_when(
        form2 == "complex_long" ~ "Complex Long", 
        form2 == "context_negation" ~ "Context Negation",
        form2 == "both_negation" ~ "Both Negation",
        TRUE ~ str_replace_all(form2, "_", " ")
      ),
      `Pref 2` = sprintf("%.3f", form2_prob),
      `CI 2` = sprintf("[%.3f, %.3f]", form2_ci_low, form2_ci_high),
      `Odds Ratio` = sprintf("%.3f", odds_ratio),
      `OR CI` = sprintf("[%.3f, %.3f]", or_ci_low, or_ci_high),
      `p value` = ifelse(p_value < 0.001, "p < .001", sprintf("p = %.3f", p_value))
    ) %>%
    select(Comparison, `Form 1`, `Pref 1`, `CI 1`, `Form 2`, `Pref 2`, `CI 2`, 
           `Odds Ratio`, `OR CI`, `p value`)
  
  # Generate LaTeX table
  xtab <- xtable(latex_table, 
                 caption = sprintf("%s: Pairwise Form Comparisons", model_label),
                 label = sprintf("tab:%s_form_pairwise", model_clean_name),
                 align = c("l", "l", "l", "r", "l", "l", "r", "l", "r", "l", "l"))
  
  # Save to file
  table_file <- sprintf("analysis/tables/latex_tables/form_pairwise/%s_form_pairwise.tex", model_clean_name)
  cat(print(xtab, 
            include.rownames = FALSE,
            caption.placement = "top",
            table.placement = "",
            floating = FALSE,
            print.results = FALSE),
      file = table_file)
  
  cat(sprintf("Saved: %s\n", table_file))
}

# Create a combined summary table across all models
cat("\n=== Creating combined summary table ===\n")

# Summary table showing significant differences
summary_data <- results %>%
  select(Model, comparison, form1, form1_prob, form2, form2_prob, odds_ratio, p_value) %>%
  mutate(
    significant = ifelse(p_value < 0.05, "Yes", "No"),
    direction = case_when(
      odds_ratio > 1 ~ paste(form1, ">", form2),
      odds_ratio < 1 ~ paste(form2, ">", form1),
      TRUE ~ "No difference"
    )
  ) %>%
  mutate(
    Comparison = case_when(
      comparison == "complex_emb_vs_complex_long" ~ "Complex Emb vs Long",
      comparison == "target_negation_vs_context_negation" ~ "Target vs Context", 
      comparison == "target_negation_vs_both_negation" ~ "Target vs Both",
      TRUE ~ comparison
    ),
    `p value` = ifelse(p_value < 0.001, "p < .001", sprintf("p = %.3f", p_value)),
    `Odds Ratio` = sprintf("%.3f", odds_ratio)
  ) %>%
  select(Model, Comparison, `Odds Ratio`, `p value`, significant, direction)

# Generate combined LaTeX table
xtab_combined <- xtable(summary_data,
                       caption = "Form Pairwise Comparisons Summary Across All Models", 
                       label = "tab:form_pairwise_summary",
                       align = c("l", "l", "l", "r", "l", "l", "l"))

# Save combined table
combined_file <- "analysis/tables/latex_tables/form_pairwise/combined_form_pairwise_summary.tex"
cat(print(xtab_combined,
          include.rownames = FALSE,
          caption.placement = "top",
          table.placement = "",
          floating = FALSE, 
          print.results = FALSE),
    file = combined_file)

cat(sprintf("Saved combined summary: %s\n", combined_file))

cat("\nAll form pairwise comparison tables generated!\n")
cat("\nDirectory structure:\n")
cat("analysis/tables/latex_tables/form_pairwise/\n")
cat("├── exp0baseline_form_pairwise.tex\n")
cat("├── exp1removeexpletives_form_pairwise.tex\n")
cat("├── exp2impoverishdeterminers_form_pairwise.tex\n") 
cat("├── exp3removearticles_form_pairwise.tex\n")
cat("├── exp4lemmatizeverbs_form_pairwise.tex\n")
cat("├── exp5removesubjectpronominals_form_pairwise.tex\n")
cat("└── combined_form_pairwise_summary.tex\n")

cat("\nTo include in LaTeX:\n")
cat("\\input{analysis/tables/latex_tables/form_pairwise/exp0baseline_form_pairwise.tex}\n")
cat("\\input{analysis/tables/latex_tables/form_pairwise/combined_form_pairwise_summary.tex}\n")