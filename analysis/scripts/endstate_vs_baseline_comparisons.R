# End-State vs Baseline Comparisons
# ==================================
# Compare each preprocessing condition to baseline in end-state performance

library(tidyverse)
library(lme4)
library(emmeans)

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

# Get end-state data (last 1000 checkpoints)
endstate_data <- dat %>%
  group_by(model, Model) %>%
  filter(checkpoint_num >= max(checkpoint_num) - 1000) %>%
  ungroup() %>%
  filter(form_type == "overt")

cat("End-state data prepared (last 1000 checkpoints):\n")
checkpoint_summary <- endstate_data %>%
  group_by(Model) %>%
  summarise(
    n_checkpoints = n_distinct(checkpoint_num),
    checkpoint_range = paste(range(checkpoint_num), collapse = " to "),
    .groups = "drop"
  )

checkpoint_summary %>%
  mutate(summary = sprintf("%s: %d checkpoints (%s)", Model, n_checkpoints, checkpoint_range)) %>%
  pull(summary) %>%
  walk(~cat("  ", .x, "\n"))

# Calculate end-state overt subject preference 
endstate_summary <- endstate_data %>%
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
    # Binomial test against 50% for overt preference
    p_value = map2_dbl(overt_correct, total_items, 
                      ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$p.value),
    ci_result = map2(overt_correct, total_items, 
                     ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$conf.int),
    ci_low = map_dbl(ci_result, 1),
    ci_high = map_dbl(ci_result, 2),
    # Determine direction
    direction = case_when(
      overt_pref > 0.5 & p_value < 0.05 ~ "above chance",
      overt_pref < 0.5 & p_value < 0.05 ~ "below chance", 
      TRUE ~ "at chance"
    ),
    # Create interpretation
    interpretation = sprintf("%.1f%% overt preference (%s)", overt_pref * 100, direction)
  ) %>%
  select(-ci_result) %>%
  arrange(desc(overt_pref))

cat("\n===========================================\n")
cat("END-STATE OVERT SUBJECT PREFERENCE\n")
cat("(Last 1000 checkpoints of training)\n") 
cat("===========================================\n\n")

# Save summary table  
endstate_table <- endstate_summary %>%
  mutate(
    `Overt Pref (95% CI)` = sprintf("%.3f [%.3f, %.3f]", overt_pref, ci_low, ci_high),
    `p-value` = format.pval(p_value, digits = 3),
    Direction = direction,
    Interpretation = interpretation
  ) %>%
  select(Model, `Overt Pref (95% CI)`, `p-value`, Direction, Interpretation)

print(endstate_table, n = Inf)

# Save data
write_csv(endstate_summary, "analysis/tables/endstate_summary.csv")
write_csv(endstate_table, "analysis/tables/endstate_formatted.csv")

# CROSS-MODEL PAIRWISE COMPARISONS
cat("\n=== CROSS-MODEL PAIRWISE COMPARISONS ===\n")

# Get list of models for comparisons
model_list <- unique(endstate_data$model)
all_comparisons <- tibble()

# Compare each model to baseline
cat("\nBaseline vs Other Models:\n")
for (model_name in setdiff(model_list, "exp0_baseline")) {
  cat(sprintf("\n--- %s vs Baseline ---\n", 
              model_labels$Model[model_labels$model == model_name]))
  
  # Filter to baseline + current model
  comparison_data <- endstate_data %>%
    filter(model %in% c("exp0_baseline", model_name)) %>%
    mutate(
      is_baseline = ifelse(model == "exp0_baseline", "Baseline", "Treatment"),
      is_baseline = factor(is_baseline, levels = c("Baseline", "Treatment"))
    )
  
  # Fit mixed-effects model
  tryCatch({
    mod <- glmer(correct ~ is_baseline + (1|item_id), 
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
      baseline_overt_pref = emm_summary$prob[emm_summary$is_baseline == "Baseline"],
      treatment_overt_pref = emm_summary$prob[emm_summary$is_baseline == "Treatment"],
      odds_ratio = contrast_summary$odds.ratio,
      or_ci_low = contrast_summary$asymp.LCL,
      or_ci_high = contrast_summary$asymp.UCL,
      p_value = contrast_summary$p.value,
      difference = emm_summary$prob[emm_summary$is_baseline == "Treatment"] - 
                  emm_summary$prob[emm_summary$is_baseline == "Baseline"]
    )
    
    all_comparisons <- bind_rows(all_comparisons, result_row)
    
    # Print results
    cat(sprintf("  Baseline: %.1f%% overt preference\n", result_row$baseline_overt_pref * 100))
    cat(sprintf("  %s: %.1f%% overt preference\n", model_label, result_row$treatment_overt_pref * 100))
    cat(sprintf("  Difference: %.1f percentage points\n", result_row$difference * 100))
    cat(sprintf("  Odds Ratio: %.3f [%.3f, %.3f], p = %.3f\n", 
                result_row$odds_ratio, result_row$or_ci_low, result_row$or_ci_high, result_row$p_value))
    
    # Interpretation
    if (result_row$p_value < 0.05) {
      if (result_row$difference > 0) {
        cat(sprintf("  → %s shows HIGHER overt preference than baseline\n", model_label))
      } else {
        cat(sprintf("  → %s shows LOWER overt preference than baseline\n", model_label))
      }
    } else {
      cat(sprintf("  → No significant difference from baseline\n"))
    }
    
  }, error = function(e) {
    cat(sprintf("Error comparing %s to baseline: %s\n", model_name, e$message))
  })
}

# Save comparison results
write_csv(all_comparisons, "analysis/tables/endstate_vs_baseline.csv")

# Summary of all comparisons
cat("\n=== SUMMARY: END-STATE vs BASELINE ===\n")
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
    select(Model, baseline_overt_pref, treatment_overt_pref, diff_pct, odds_ratio, p_value, direction) %>%
    arrange(desc(treatment_overt_pref))
  
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
      pwalk(~cat(sprintf("  %s: %s overt preference (%s points, p = %.3f)\n", 
                        ..1, ..2, ..3, ..4)))
  } else {
    cat("  No significant differences found.\n")
  }
}

# Compare to chance (50%)
cat("\n=== COMPARISON TO CHANCE (50%) ===\n")
endstate_summary %>%
  select(Model, overt_pref, p_value, direction) %>%
  mutate(
    overt_pct = sprintf("%.1f%%", overt_pref * 100),
    vs_chance = case_when(
      direction == "above chance" ~ sprintf("Significantly ABOVE 50%% (p < %.3f)", p_value),
      direction == "below chance" ~ sprintf("Significantly BELOW 50%% (p < %.3f)", p_value),
      TRUE ~ sprintf("Not different from 50%% (p = %.3f)", p_value)
    )
  ) %>%
  select(Model, overt_pct, vs_chance) %>%
  arrange(desc(overt_pct)) %>%
  pwalk(~cat(sprintf("  %s: %s - %s\n", ..1, ..2, ..3)))

cat("\nEnd-state analysis complete.\n")
cat("Results saved to:\n")
cat("  - analysis/tables/endstate_summary.csv (raw data)\n")
cat("  - analysis/tables/endstate_formatted.csv (formatted)\n")
cat("  - analysis/tables/endstate_vs_baseline.csv (comparisons)\n")