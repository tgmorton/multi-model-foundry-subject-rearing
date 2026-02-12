# First Epoch vs Baseline Comparisons
# ====================================
# Compare each preprocessing condition to baseline in first epoch performance

library(tidyverse)
library(lme4)
library(emmeans)

# Load data
dat <- read_csv("evaluation/results/all_models_null_subject_lme4_ready.csv")

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

dat <- dat %>%
  left_join(model_labels, by = "model")

# Calculate first epoch boundaries (5% of total checkpoints)
first_epoch_bounds <- dat %>%
  group_by(model) %>%
  summarise(
    max_checkpoint = max(checkpoint_num),
    first_epoch_end = ceiling(max_checkpoint * 0.05),
    .groups = "drop"
  )

# Get first epoch data (last 4 checkpoints before epoch boundary)
first_epoch_data <- dat %>%
  left_join(first_epoch_bounds, by = "model") %>%
  group_by(model) %>%
  filter(checkpoint_num <= first_epoch_end) %>%
  slice_max(checkpoint_num, n = 4) %>%
  ungroup() %>%
  filter(form_type == "overt") %>%
  # Convert to null preference (1 - overt preference)
  mutate(null_pref = 1 - correct)

cat("First epoch data prepared. Models included:\n")
first_epoch_data %>% 
  group_by(Model) %>% 
  summarise(checkpoints = n_distinct(checkpoint_num), .groups = "drop") %>%
  print()

# Initialize results storage
all_comparisons <- tibble()

# Get list of non-baseline models
comparison_models <- setdiff(unique(first_epoch_data$model), "exp0_baseline")

# Compare each model to baseline
for (model_name in comparison_models) {
  cat(sprintf("\n=== Comparing %s to Baseline ===\n", model_name))
  
  # Filter to baseline + current model
  comparison_data <- first_epoch_data %>%
    filter(model %in% c("exp0_baseline", model_name)) %>%
    mutate(
      is_baseline = ifelse(model == "exp0_baseline", "Baseline", "Treatment"),
      is_baseline = factor(is_baseline, levels = c("Baseline", "Treatment"))
    )
  
  if (nrow(comparison_data) == 0) {
    cat("No data found for this comparison\n")
    next
  }
  
  # Fit mixed-effects model
  tryCatch({
    mod <- glmer(null_pref ~ is_baseline + (1|item_id), 
                 data = comparison_data, 
                 family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))
    
    # Get estimated marginal means
    emm <- emmeans(mod, ~ is_baseline)
    emm_summary <- summary(emm, type = "response")
    
    # Pairwise comparison
    contrast <- pairs(emm, type = "response")
    contrast_summary <- summary(contrast, infer = TRUE)
    
    # Extract model label
    model_label <- first_epoch_data %>% 
      filter(model == model_name) %>% 
      pull(Model) %>% 
      unique()
    
    # Store results
    result_row <- tibble(
      model = model_name,
      Model = model_label,
      baseline_null_pref = emm_summary$prob[emm_summary$is_baseline == "Baseline"],
      baseline_se = emm_summary$SE[emm_summary$is_baseline == "Baseline"],
      baseline_ci_low = emm_summary$asymp.LCL[emm_summary$is_baseline == "Baseline"],
      baseline_ci_high = emm_summary$asymp.UCL[emm_summary$is_baseline == "Baseline"],
      treatment_null_pref = emm_summary$prob[emm_summary$is_baseline == "Treatment"],
      treatment_se = emm_summary$SE[emm_summary$is_baseline == "Treatment"],
      treatment_ci_low = emm_summary$asymp.LCL[emm_summary$is_baseline == "Treatment"],
      treatment_ci_high = emm_summary$asymp.UCL[emm_summary$is_baseline == "Treatment"],
      odds_ratio = contrast_summary$odds.ratio,
      or_ci_low = contrast_summary$asymp.LCL,
      or_ci_high = contrast_summary$asymp.UCL,
      z_value = contrast_summary$z.ratio,
      p_value = contrast_summary$p.value
    )
    
    all_comparisons <- bind_rows(all_comparisons, result_row)
    
    # Print summary
    cat(sprintf("  Baseline: %.3f [%.3f, %.3f]\n", 
                result_row$baseline_null_pref, result_row$baseline_ci_low, result_row$baseline_ci_high))
    cat(sprintf("  %s: %.3f [%.3f, %.3f]\n", 
                model_label, result_row$treatment_null_pref, result_row$treatment_ci_low, result_row$treatment_ci_high))
    cat(sprintf("  Odds Ratio: %.3f [%.3f, %.3f], p = %.3f\n", 
                result_row$odds_ratio, result_row$or_ci_low, result_row$or_ci_high, result_row$p_value))
    
    # Interpretation
    if (result_row$p_value < 0.05) {
      if (result_row$odds_ratio > 1) {
        cat(sprintf("  → %s shows HIGHER null preference than baseline (significant)\n", model_label))
      } else {
        cat(sprintf("  → %s shows LOWER null preference than baseline (significant)\n", model_label))
      }
    } else {
      cat(sprintf("  → No significant difference from baseline\n"))
    }
    
  }, error = function(e) {
    cat(sprintf("Error fitting model for %s: %s\n", model_name, e$message))
  })
}

# Save results
write_csv(all_comparisons, "analysis/tables/first_epoch_vs_baseline_comparisons.csv")

# Print summary table
cat("\n=== SUMMARY: FIRST EPOCH vs BASELINE ===\n")
summary_table <- all_comparisons %>%
  mutate(
    difference = treatment_null_pref - baseline_null_pref,
    significance = ifelse(p_value < 0.05, "Significant", "Non-significant"),
    direction = case_when(
      p_value >= 0.05 ~ "No difference",
      odds_ratio > 1 ~ "Higher than baseline",
      odds_ratio < 1 ~ "Lower than baseline"
    )
  ) %>%
  select(Model, baseline_null_pref, treatment_null_pref, difference, odds_ratio, p_value, direction) %>%
  arrange(desc(treatment_null_pref))

print(summary_table, n = Inf, width = Inf)

cat(sprintf("\nResults saved to: analysis/tables/first_epoch_vs_baseline_comparisons.csv\n"))
cat("\nInterpretation:\n")
cat("- Positive difference = higher null preference than baseline\n")
cat("- Negative difference = lower null preference than baseline\n")
cat("- Odds Ratio > 1 = treatment has higher null preference\n")
cat("- Odds Ratio < 1 = treatment has lower null preference\n")