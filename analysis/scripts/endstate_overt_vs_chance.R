# End-State Overt Preference vs. Chance (0.5) Analysis
# =====================================================

library(tidyverse)
library(lme4)
library(emmeans)
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

# Get end-state data (last 1000 checkpoints)
endstate_data <- dat %>%
  group_by(model) %>%
  filter(checkpoint_num >= max(checkpoint_num) - 1000) %>%
  ungroup()

cat("\n============================================\n")
cat("END-STATE OVERT PREFERENCE VS. CHANCE (0.5)\n")
cat("============================================\n\n")

# Analyze each model
all_results <- list()

for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  cat(sprintf("\nAnalyzing %s...\n", model_label))
  
  # Get overt subject data
  overt_data <- endstate_data %>%
    filter(model == model_name, form_type == "overt")
  
  # Calculate overall overt preference
  overt_summary <- overt_data %>%
    summarise(
      n_correct = sum(correct),
      n_total = n(),
      overt_pref = n_correct / n_total,
      se = sqrt(overt_pref * (1 - overt_pref) / n_total),
      .groups = "drop"
    )
  
  # Binomial test vs chance (0.5)
  binom_test <- binom.test(
    overt_summary$n_correct, 
    overt_summary$n_total, 
    p = 0.5,
    alternative = "two.sided"
  )
  
  # Store results
  result <- tibble(
    model = model_name,
    Model = model_label,
    overt_pref = overt_summary$overt_pref,
    se = overt_summary$se,
    ci_low = binom_test$conf.int[1],
    ci_high = binom_test$conf.int[2],
    n = overt_summary$n_total,
    p_value = binom_test$p.value,
    direction = case_when(
      overt_pref > 0.5 ~ "above",
      overt_pref < 0.5 ~ "below",
      TRUE ~ "at"
    ),
    significant = p_value < 0.05,
    interpretation = sprintf(
      "%.1f%% (%.3f ± %.3f), %s chance%s",
      overt_pref * 100,
      overt_pref,
      se,
      direction,
      ifelse(significant, "*", "")
    )
  )
  
  all_results[[model_name]] <- result
  
  # Print summary
  cat(sprintf("  Overt preference: %.3f (SE = %.4f)\n", 
              overt_summary$overt_pref, overt_summary$se))
  cat(sprintf("  95%% CI: [%.3f, %.3f]\n", 
              binom_test$conf.int[1], binom_test$conf.int[2]))
  cat(sprintf("  vs. chance (0.5): %s\n", 
              ifelse(binom_test$p.value < 0.001, "p < 0.001",
                     sprintf("p = %.3f", binom_test$p.value))))
  cat(sprintf("  Direction: %s chance\n", result$direction))
  
  # Item group analysis
  cat("\n  By item group:\n")
  
  itemgroup_results <- endstate_data %>%
    filter(model == model_name, form_type == "overt") %>%
    group_by(item_group) %>%
    summarise(
      n_correct = sum(correct),
      n_total = n(),
      overt_pref = n_correct / n_total,
      .groups = "drop"
    ) %>%
    mutate(
      # Binomial test for each item group
      p_value = map2_dbl(n_correct, n_total, 
                         ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$p.value),
      direction = case_when(
        overt_pref > 0.5 ~ "above",
        overt_pref < 0.5 ~ "below",
        TRUE ~ "at"
      ),
      sig = ifelse(p_value < 0.05, "*", "")
    ) %>%
    arrange(desc(overt_pref))
  
  # Show top and bottom 3
  cat("    Highest overt preference (vs chance):\n")
  itemgroup_results %>%
    head(3) %>%
    mutate(
      display = sprintf("      %s: %.1f%% %s (p %s)",
                       item_group,
                       overt_pref * 100,
                       sig,
                       ifelse(p_value < 0.001, "< 0.001", sprintf("= %.3f", p_value)))
    ) %>%
    pull(display) %>%
    cat(sep = "\n")
  
  cat("\n    Lowest overt preference (vs chance):\n")
  itemgroup_results %>%
    tail(3) %>%
    arrange(overt_pref) %>%
    mutate(
      display = sprintf("      %s: %.1f%% %s (p %s)",
                       item_group,
                       overt_pref * 100,
                       sig,
                       ifelse(p_value < 0.001, "< 0.001", sprintf("= %.3f", p_value)))
    ) %>%
    pull(display) %>%
    cat(sep = "\n")
  
  cat("\n")
  
  # Save item group results
  itemgroup_results <- itemgroup_results %>%
    mutate(
      model = model_name,
      Model = model_label,
      .before = 1
    )
  
  # Store for saving
  if (!exists("all_itemgroup_results")) {
    all_itemgroup_results <- itemgroup_results
  } else {
    all_itemgroup_results <- bind_rows(all_itemgroup_results, itemgroup_results)
  }
}

# Combine and save results
combined_results <- bind_rows(all_results)

# Create summary table
summary_table <- combined_results %>%
  mutate(
    `Overt Pref (%)` = sprintf("%.1f%%", overt_pref * 100),
    `Mean (SE)` = sprintf("%.3f (%.4f)", overt_pref, se),
    `95% CI` = sprintf("[%.3f, %.3f]", ci_low, ci_high),
    `p-value` = ifelse(p_value < 0.001, "< 0.001", sprintf("%.3f", p_value)),
    Direction = paste0(direction, ifelse(significant, "*", ""))
  ) %>%
  select(Model, `Overt Pref (%)`, `Mean (SE)`, `95% CI`, Direction, `p-value`)

cat("\n============================================\n")
cat("SUMMARY: End-State Overt Preference vs Chance\n")
cat("============================================\n\n")
print(summary_table, n = Inf)

# Save results
write_csv(combined_results, "analysis/tables/endstate_overt_vs_chance.csv")
write_csv(all_itemgroup_results, "analysis/tables/endstate_overt_vs_chance_itemgroup.csv")

# Cross-model comparison
cat("\n\n============================================\n")
cat("CROSS-MODEL COMPARISON\n")
cat("============================================\n\n")

# Count how many models are significantly above/below/at chance
direction_summary <- combined_results %>%
  filter(significant) %>%
  count(direction) %>%
  mutate(
    description = case_when(
      direction == "above" ~ "significantly prefer overt subjects",
      direction == "below" ~ "significantly prefer null subjects",
      TRUE ~ "at chance"
    )
  )

cat("Models by preference direction:\n")
for (i in 1:nrow(direction_summary)) {
  cat(sprintf("  %d models %s\n", 
              direction_summary$n[i], 
              direction_summary$description[i]))
}

# Models significantly different from chance
sig_models <- combined_results %>%
  filter(significant) %>%
  arrange(desc(overt_pref))

if (nrow(sig_models) > 0) {
  cat("\nModels significantly different from chance:\n")
  for (i in 1:nrow(sig_models)) {
    cat(sprintf("  %s: %.1f%% %s chance (p %s)\n",
                sig_models$Model[i],
                sig_models$overt_pref[i] * 100,
                sig_models$direction[i],
                ifelse(sig_models$p_value[i] < 0.001, "< 0.001", 
                       sprintf("= %.3f", sig_models$p_value[i]))))
  }
}

cat("\n* = significant at p < 0.05 (two-tailed binomial test)\n")
cat("\nAnalysis complete!\n")
cat("Results saved to:\n")
cat("  - analysis/tables/endstate_overt_vs_chance.csv\n")
cat("  - analysis/tables/endstate_overt_vs_chance_itemgroup.csv\n")