# Full First Epoch Analysis
# =========================
# Analyze null subject preference across the ENTIRE first epoch (all checkpoints)

library(tidyverse)

# Load data
cat("Loading data...\n")
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
  group_by(model, Model) %>%
  summarise(
    max_checkpoint = max(checkpoint_num),
    first_epoch_end = ceiling(max_checkpoint * 0.05),
    .groups = "drop"
  )

cat("\nFirst epoch boundaries (5% of training):\n")
first_epoch_bounds %>%
  mutate(boundary_info = sprintf("%s: %d (from max %d)", Model, first_epoch_end, max_checkpoint)) %>%
  pull(boundary_info) %>%
  walk(~cat("  ", .x, "\n"))

# Get ENTIRE first epoch data (all checkpoints <= boundary)
first_epoch_data <- dat %>%
  left_join(first_epoch_bounds, by = c("model", "Model")) %>%
  filter(checkpoint_num <= first_epoch_end) %>%
  filter(form_type == "overt")

cat("\nCheckpoints included per model:\n")
checkpoint_summary <- first_epoch_data %>%
  group_by(Model) %>%
  summarise(
    n_checkpoints = n_distinct(checkpoint_num),
    checkpoints = paste(sort(unique(checkpoint_num)), collapse = ", "),
    .groups = "drop"
  )

checkpoint_summary %>%
  mutate(summary = sprintf("%s: %d checkpoints (%s)", Model, n_checkpoints, 
                          ifelse(nchar(checkpoints) > 50, 
                                paste0(substr(checkpoints, 1, 47), "..."), 
                                checkpoints))) %>%
  pull(summary) %>%
  walk(~cat("  ", .x, "\n"))

# Calculate null subject preference for entire first epoch
first_epoch_summary <- first_epoch_data %>%
  group_by(model, Model) %>%
  summarise(
    total_items = n(),
    overt_correct = sum(correct),
    null_correct = total_items - overt_correct,
    overt_pref = overt_correct / total_items,
    null_pref = null_correct / total_items,
    .groups = "drop"
  ) %>%
  # Test against 50% chance
  mutate(
    # Binomial test against 50% for null preference
    p_value = map2_dbl(null_correct, total_items, 
                      ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$p.value),
    ci_result = map2(null_correct, total_items, 
                     ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$conf.int),
    ci_low = map_dbl(ci_result, 1),
    ci_high = map_dbl(ci_result, 2),
    # Determine direction
    direction = case_when(
      null_pref > 0.5 & p_value < 0.05 ~ "above chance",
      null_pref < 0.5 & p_value < 0.05 ~ "below chance", 
      TRUE ~ "at chance"
    ),
    # Create interpretation
    interpretation = sprintf("%.1f%% null preference (%s)", null_pref * 100, direction)
  ) %>%
  select(-ci_result) %>%
  arrange(desc(null_pref))

# Save detailed data
write_csv(first_epoch_data, "analysis/tables/full_first_epoch_data.csv")

# Save summary table  
first_epoch_table <- first_epoch_summary %>%
  mutate(
    `Null Pref (95% CI)` = sprintf("%.3f [%.3f, %.3f]", null_pref, ci_low, ci_high),
    `p-value` = format.pval(p_value, digits = 3),
    Direction = direction,
    Interpretation = interpretation
  ) %>%
  select(Model, `Null Pref (95% CI)`, `p-value`, Direction, Interpretation)

write_csv(first_epoch_table, "analysis/tables/full_first_epoch_summary.csv")

cat("\n===========================================\n")
cat("FULL FIRST EPOCH NULL SUBJECT PREFERENCE\n")
cat("(ALL checkpoints in first 5% of training)\n") 
cat("===========================================\n\n")

print(first_epoch_table, n = Inf)

cat("\n===========================================\n")
cat("DETAILED STATISTICS\n")
cat("===========================================\n\n")

detailed_stats <- first_epoch_summary %>%
  select(Model, total_items, null_correct, overt_correct, null_pref, p_value) %>%
  mutate(
    null_pref_pct = sprintf("%.1f%%", null_pref * 100),
    significance = ifelse(p_value < 0.001, "***", 
                         ifelse(p_value < 0.01, "**",
                               ifelse(p_value < 0.05, "*", "ns")))
  )

print(detailed_stats, n = Inf)

# Compare to chance (50%)
cat("\n=== COMPARISON TO CHANCE (50%) ===\n")
first_epoch_summary %>%
  select(Model, null_pref, p_value, direction) %>%
  mutate(
    null_pct = sprintf("%.1f%%", null_pref * 100),
    vs_chance = case_when(
      direction == "above chance" ~ sprintf("Significantly ABOVE 50%% (p < %.3f)", p_value),
      direction == "below chance" ~ sprintf("Significantly BELOW 50%% (p < %.3f)", p_value),
      TRUE ~ sprintf("Not different from 50%% (p = %.3f)", p_value)
    )
  ) %>%
  select(Model, null_pct, vs_chance) %>%
  arrange(desc(null_pct)) %>%
  pwalk(~cat(sprintf("  %s: %s - %s\n", ..1, ..2, ..3)))

# CROSS-MODEL PAIRWISE COMPARISONS
cat("\n=== CROSS-MODEL PAIRWISE COMPARISONS ===\n")
library(lme4)
library(emmeans)

# Get list of models for comparisons
model_list <- unique(first_epoch_data$model)
all_comparisons <- tibble()

# Compare each model to baseline
cat("\nBaseline vs Other Models:\n")
for (model_name in setdiff(model_list, "exp0_baseline")) {
  cat(sprintf("\n--- %s vs Baseline ---\n", 
              model_labels$Model[model_labels$model == model_name]))
  
  # Filter to baseline + current model
  comparison_data <- first_epoch_data %>%
    filter(model %in% c("exp0_baseline", model_name)) %>%
    mutate(
      is_baseline = ifelse(model == "exp0_baseline", "Baseline", "Treatment"),
      is_baseline = factor(is_baseline, levels = c("Baseline", "Treatment")),
      # Convert to null preference
      null_pref = 1 - correct
    )
  
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
    model_label <- model_labels$Model[model_labels$model == model_name]
    
    # Store results
    result_row <- tibble(
      model = model_name,
      Model = model_label,
      baseline_null_pref = emm_summary$prob[emm_summary$is_baseline == "Baseline"],
      treatment_null_pref = emm_summary$prob[emm_summary$is_baseline == "Treatment"],
      odds_ratio = contrast_summary$odds.ratio,
      or_ci_low = contrast_summary$asymp.LCL,
      or_ci_high = contrast_summary$asymp.UCL,
      p_value = contrast_summary$p.value,
      difference = emm_summary$prob[emm_summary$is_baseline == "Treatment"] - 
                  emm_summary$prob[emm_summary$is_baseline == "Baseline"]
    )
    
    all_comparisons <- bind_rows(all_comparisons, result_row)
    
    # Print results
    cat(sprintf("  Baseline: %.1f%% null preference\n", result_row$baseline_null_pref * 100))
    cat(sprintf("  %s: %.1f%% null preference\n", model_label, result_row$treatment_null_pref * 100))
    cat(sprintf("  Difference: %.1f percentage points\n", result_row$difference * 100))
    cat(sprintf("  Odds Ratio: %.3f [%.3f, %.3f], p = %.3f\n", 
                result_row$odds_ratio, result_row$or_ci_low, result_row$or_ci_high, result_row$p_value))
    
    # Interpretation
    if (result_row$p_value < 0.05) {
      if (result_row$difference > 0) {
        cat(sprintf("  → %s shows HIGHER null preference than baseline\n", model_label))
      } else {
        cat(sprintf("  → %s shows LOWER null preference than baseline\n", model_label))
      }
    } else {
      cat(sprintf("  → No significant difference from baseline\n"))
    }
    
  }, error = function(e) {
    cat(sprintf("Error comparing %s to baseline: %s\n", model_name, e$message))
  })
}

# Save comparison results
write_csv(all_comparisons, "analysis/tables/full_first_epoch_vs_baseline.csv")

# Summary of all comparisons
cat("\n=== SUMMARY: FIRST EPOCH vs BASELINE ===\n")
if (nrow(all_comparisons) > 0) {
  summary_comp <- all_comparisons %>%
    mutate(
      significance = ifelse(p_value < 0.05, "Significant", "Non-significant"),
      direction = case_when(
        p_value >= 0.05 ~ "No difference",
        difference > 0 ~ "Higher than baseline",
        difference < 0 ~ "Lower than baseline"
      ),
      diff_pct = sprintf("%.1f%%", difference * 100)
    ) %>%
    select(Model, baseline_null_pref, treatment_null_pref, diff_pct, odds_ratio, p_value, direction) %>%
    arrange(desc(treatment_null_pref))
  
  print(summary_comp, n = Inf, width = Inf)
  
  cat("\nSignificant differences from baseline:\n")
  significant_diffs <- all_comparisons %>%
    filter(p_value < 0.05) %>%
    mutate(
      direction = ifelse(difference > 0, "higher", "lower"),
      diff_pct = sprintf("%.1f%%", abs(difference) * 100)
    )
  
  if (nrow(significant_diffs) > 0) {
    significant_diffs %>%
      select(Model, direction, diff_pct, p_value) %>%
      pwalk(~cat(sprintf("  %s: %s null preference (%s points, p = %.3f)\n", 
                        ..1, ..2, ..3, ..4)))
  } else {
    cat("  No significant differences found.\n")
  }
}

cat("\nFull first epoch analysis complete.\n")
cat("Results saved to:\n")
cat("  - analysis/tables/full_first_epoch_data.csv (all data)\n")
cat("  - analysis/tables/full_first_epoch_summary.csv (summary)\n")
cat("  - analysis/tables/full_first_epoch_vs_baseline.csv (comparisons)\n")