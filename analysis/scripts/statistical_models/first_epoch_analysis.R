# First Epoch Analysis - Standalone Script
# ========================================

library(tidyverse)
library(broom)

# Load the data
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

# Calculate first epoch checkpoint per model (may vary slightly)
model_checkpoints <- dat %>%
  group_by(model, Model) %>%
  summarise(
    max_chk = max(checkpoint_num, na.rm = TRUE),
    first_epoch_target = round(max_chk / 20),
    .groups = "drop"
  )

cat("\nFirst epoch checkpoints by model:\n")
for(i in 1:nrow(model_checkpoints)) {
  cat(sprintf("  %s: %d (from max %d)\n", 
              model_checkpoints$Model[i], 
              model_checkpoints$first_epoch_target[i],
              model_checkpoints$max_chk[i]))
}

# Extract data for each model's specific checkpoints
first_epoch_data <- dat %>%
  inner_join(model_checkpoints, by = c("model", "Model")) %>%
  group_by(model, Model) %>%
  # Get the 4 checkpoints closest to and before/at the first epoch target
  filter(checkpoint_num <= first_epoch_target) %>%
  arrange(desc(checkpoint_num)) %>%
  filter(checkpoint_num %in% head(unique(checkpoint_num), 4)) %>%
  ungroup() %>%
  group_by(model, Model, checkpoint_num, form_type) %>%
  summarise(
    accuracy = mean(correct, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = form_type, values_from = c(accuracy, n)) %>%
  mutate(
    null_pref = accuracy_null,
    # Calculate binomial test for null preference vs 0.5
    p_value = map2_dbl(round(accuracy_null * n_null), n_null, 
                       ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$p.value),
    # Determine direction
    direction = case_when(
      null_pref > 0.5 ~ "above",
      null_pref < 0.5 ~ "below",
      TRUE ~ "at"
    ),
    significant = p_value < 0.05
  )

# Show which checkpoints were actually used per model
checkpoints_used <- first_epoch_data %>%
  group_by(Model) %>%
  summarise(
    checkpoints = paste(sort(unique(checkpoint_num)), collapse=", "),
    .groups = "drop"
  )

cat("\nActual checkpoints analyzed:\n")
for(i in 1:nrow(checkpoints_used)) {
  cat(sprintf("  %s: %s\n", checkpoints_used$Model[i], checkpoints_used$checkpoints[i]))
}

# Summarize across the 4 checkpoints
first_epoch_summary <- first_epoch_data %>%
  group_by(model, Model) %>%
  summarise(
    mean_null_pref = mean(null_pref, na.rm = TRUE),
    sd_null_pref = sd(null_pref, na.rm = TRUE),
    # Pool the binomial test across checkpoints
    total_null_correct = sum(accuracy_null * n_null, na.rm = TRUE),
    total_null_n = sum(n_null, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    # Binomial test on pooled data
    p_value = map2_dbl(round(total_null_correct), total_null_n,
                       ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$p.value),
    direction = case_when(
      mean_null_pref > 0.5 ~ "above",
      mean_null_pref < 0.5 ~ "below",
      TRUE ~ "at"
    ),
    significant = ifelse(p_value < 0.05, "*", ""),
    # Format for display
    null_pref_pct = sprintf("%.1f%%", mean_null_pref * 100),
    interpretation = paste0(
      null_pref_pct, " ",
      "(", direction, " 50%", significant, ")"
    )
  ) %>%
  arrange(desc(mean_null_pref))

# Save detailed checkpoint data
write_csv(first_epoch_data, "analysis/tables/first_epoch_checkpoints.csv")

# Save summary table
first_epoch_table <- first_epoch_summary %>%
  select(Model, mean_null_pref, sd_null_pref, direction, p_value, interpretation) %>%
  mutate(
    `Mean (SD)` = sprintf("%.3f (%.3f)", mean_null_pref, sd_null_pref),
    `p-value` = format.pval(p_value, digits = 3),
    Direction = direction,
    Interpretation = interpretation
  ) %>%
  select(Model, `Mean (SD)`, Direction, `p-value`, Interpretation)

write_csv(first_epoch_table, "analysis/tables/first_epoch_summary.csv")

cat("\n===========================================\n")
cat("First Epoch Null Subject Preference\n")
cat("(Last 4 checkpoints before epoch 1 ends)\n")
cat("===========================================\n\n")
print(first_epoch_table, n = Inf)

cat("\nFirst epoch analysis complete.\n")
cat("Results saved to:\n")
cat("  - analysis/tables/first_epoch_checkpoints.csv (detailed)\n")
cat("  - analysis/tables/first_epoch_summary.csv (summary)\n")