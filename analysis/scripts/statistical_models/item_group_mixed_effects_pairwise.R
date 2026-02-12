# Item Group Mixed-Effects Pairwise Comparisons
# ==============================================
# Re-run ALL item group comparisons using consistent mixed-effects + odds ratios

library(tidyverse)
library(lme4)
library(emmeans)

cat("=== ITEM GROUP MIXED-EFFECTS PAIRWISE COMPARISONS ===\n\n")

# Load data
dat <- read_csv("evaluation/results/all_models_null_subject_lme4_ready.csv", show_col_types = FALSE)

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
  ungroup() %>%
  filter(form_type == "overt")

# Initialize storage for all comparisons
all_item_comparisons <- tibble()

# PART B1: NUMBER CONTRASTS
cat("B1. NUMBER CONTRASTS\n")
cat("====================\n")

# Helper function for mixed-effects pairwise comparison
run_mixed_effects_comparison <- function(data, grouping_var, group1, group2, model_name, comparison_name) {
  comparison_data <- data %>%
    filter(!!sym(grouping_var) %in% c(group1, group2)) %>%
    mutate(
      group = factor(!!sym(grouping_var), levels = c(group1, group2))
    )
  
  if (nrow(comparison_data) == 0) return(NULL)
  
  tryCatch({
    mod <- glmer(correct ~ group + (1|item_id), 
                 data = comparison_data, 
                 family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))
    
    emm <- emmeans(mod, ~ group)
    emm_summary <- summary(emm, type = "response")
    contrast <- pairs(emm, type = "response") 
    contrast_summary <- summary(contrast, infer = TRUE)
    
    tibble(
      model = model_name,
      comparison_type = comparison_name,
      group1 = group1,
      group2 = group2,
      group1_prob = emm_summary$prob[1],
      group1_se = emm_summary$SE[1],
      group1_ci_low = emm_summary$asymp.LCL[1],
      group1_ci_high = emm_summary$asymp.UCL[1],
      group2_prob = emm_summary$prob[2], 
      group2_se = emm_summary$SE[2],
      group2_ci_low = emm_summary$asymp.LCL[2],
      group2_ci_high = emm_summary$asymp.UCL[2],
      odds_ratio = contrast_summary$odds.ratio,
      or_ci_low = contrast_summary$asymp.LCL,
      or_ci_high = contrast_summary$asymp.UCL,
      z_value = contrast_summary$z.ratio,
      p_value = contrast_summary$p.value
    )
  }, error = function(e) {
    cat(sprintf("Error in %s - %s vs %s: %s\n", model_name, group1, group2, e$message))
    NULL
  })
}

# B1: Number contrasts within person
for (model_name in unique(endstate_data$model)) {
  model_data <- endstate_data %>% filter(model == model_name)
  
  cat(sprintf("\nModel: %s\n", model_name))
  
  # 3rd person: singular vs plural 
  third_data <- model_data %>%
    filter(str_detect(item_group, "^1[ab]_")) %>%
    mutate(number = ifelse(str_detect(item_group, "1a_"), "3rd_singular", "3rd_plural"))
  
  if (nrow(third_data) > 0) {
    result <- run_mixed_effects_comparison(third_data, "number", "3rd_singular", "3rd_plural", 
                                         model_name, "B1_Number_3rd_Sing_vs_Plural")
    if (!is.null(result)) all_item_comparisons <- bind_rows(all_item_comparisons, result)
  }
  
  # 2nd person: singular vs plural
  second_data <- model_data %>%
    filter(str_detect(item_group, "^2[ab]_")) %>%
    mutate(number = ifelse(str_detect(item_group, "2a_"), "2nd_singular", "2nd_plural"))
  
  if (nrow(second_data) > 0) {
    result <- run_mixed_effects_comparison(second_data, "number", "2nd_singular", "2nd_plural",
                                         model_name, "B1_Number_2nd_Sing_vs_Plural")
    if (!is.null(result)) all_item_comparisons <- bind_rows(all_item_comparisons, result)
  }
  
  # 1st person: singular vs plural
  first_data <- model_data %>%
    filter(str_detect(item_group, "^3[ab]_")) %>%
    mutate(number = ifelse(str_detect(item_group, "3a_"), "1st_singular", "1st_plural"))
  
  if (nrow(first_data) > 0) {
    result <- run_mixed_effects_comparison(first_data, "number", "1st_singular", "1st_plural",
                                         model_name, "B1_Number_1st_Sing_vs_Plural")
    if (!is.null(result)) all_item_comparisons <- bind_rows(all_item_comparisons, result)
  }
}

# B2: PERSON CONTRASTS
cat("\nB2. PERSON CONTRASTS\n")
cat("====================\n")

for (model_name in unique(endstate_data$model)) {
  model_data <- endstate_data %>% 
    filter(model == model_name) %>%
    mutate(person = case_when(
      str_detect(item_group, "^1[ab]_") ~ "3rd_person",
      str_detect(item_group, "^2[ab]_") ~ "2nd_person", 
      str_detect(item_group, "^3[ab]_") ~ "1st_person",
      TRUE ~ "other"
    )) %>%
    filter(person != "other")
  
  cat(sprintf("\nModel: %s\n", model_name))
  
  # 1st vs 2nd person
  result <- run_mixed_effects_comparison(model_data, "person", "1st_person", "2nd_person",
                                       model_name, "B2_Person_1st_vs_2nd")
  if (!is.null(result)) all_item_comparisons <- bind_rows(all_item_comparisons, result)
  
  # 1st vs 3rd person  
  result <- run_mixed_effects_comparison(model_data, "person", "1st_person", "3rd_person",
                                       model_name, "B2_Person_1st_vs_3rd")
  if (!is.null(result)) all_item_comparisons <- bind_rows(all_item_comparisons, result)
  
  # 2nd vs 3rd person
  result <- run_mixed_effects_comparison(model_data, "person", "2nd_person", "3rd_person",
                                       model_name, "B2_Person_2nd_vs_3rd")
  if (!is.null(result)) all_item_comparisons <- bind_rows(all_item_comparisons, result)
}

# B3: CONTROL CONTRASTS
cat("\nB3. CONTROL CONTRASTS\n")
cat("=====================\n")

for (model_name in unique(endstate_data$model)) {
  control_data <- endstate_data %>%
    filter(model == model_name, str_detect(item_group, "^4[ab]_")) %>%
    mutate(control_type = ifelse(str_detect(item_group, "4a_"), "subject_control", "object_control"))
  
  cat(sprintf("\nModel: %s\n", model_name))
  
  result <- run_mixed_effects_comparison(control_data, "control_type", "subject_control", "object_control",
                                       model_name, "B3_Control_Subject_vs_Object")
  if (!is.null(result)) all_item_comparisons <- bind_rows(all_item_comparisons, result)
}

# B4: EXPLETIVE CONTRASTS  
cat("\nB4. EXPLETIVE CONTRASTS\n")
cat("=======================\n")

for (model_name in unique(endstate_data$model)) {
  expletive_data <- endstate_data %>%
    filter(model == model_name, str_detect(item_group, "^5[ab]_")) %>%
    mutate(expletive_type = ifelse(str_detect(item_group, "5a_"), "seems", "be"))
  
  cat(sprintf("\nModel: %s\n", model_name))
  
  result <- run_mixed_effects_comparison(expletive_data, "expletive_type", "seems", "be",
                                       model_name, "B4_Expletive_Seems_vs_Be")
  if (!is.null(result)) all_item_comparisons <- bind_rows(all_item_comparisons, result)
}

# B5: TOPIC SHIFT CONTRASTS
cat("\nB5. TOPIC SHIFT CONTRASTS\n") 
cat("=========================\n")

for (model_name in unique(endstate_data$model)) {
  topic_data <- endstate_data %>%
    filter(model == model_name, str_detect(item_group, "^7[ab]_")) %>%
    mutate(topic_type = ifelse(str_detect(item_group, "7a_"), "no_topic_shift", "topic_shift"))
  
  cat(sprintf("\nModel: %s\n", model_name))
  
  result <- run_mixed_effects_comparison(topic_data, "topic_type", "no_topic_shift", "topic_shift",
                                       model_name, "B5_Topic_NoShift_vs_Shift")
  if (!is.null(result)) all_item_comparisons <- bind_rows(all_item_comparisons, result)
}

# APPLY MULTIPLE COMPARISONS CORRECTIONS
cat("\n=== APPLYING MULTIPLE COMPARISONS CORRECTIONS ===\n")

total_tests <- nrow(all_item_comparisons)
cat(sprintf("Total item group pairwise tests: %d\n", total_tests))

all_item_comparisons_corrected <- all_item_comparisons %>%
  mutate(
    p_value_fdr = p.adjust(p_value, method = "fdr"),
    p_value_bonferroni = p.adjust(p_value, method = "bonferroni"),
    significant_uncorrected = p_value < 0.05,
    significant_fdr = p_value_fdr < 0.05,
    significant_bonferroni = p_value_bonferroni < 0.05
  ) %>%
  left_join(model_labels, by = "model")

# Save results
write_csv(all_item_comparisons_corrected, "analysis/tables/item_group_pairwise_mixed_effects_corrected.csv")

# Summary statistics
cat("\n=== SUMMARY STATISTICS ===\n")
summary_stats <- all_item_comparisons_corrected %>%
  group_by(comparison_type) %>%
  summarise(
    n_tests = n(),
    significant_uncorrected = sum(significant_uncorrected),
    significant_fdr = sum(significant_fdr),
    significant_bonferroni = sum(significant_bonferroni),
    .groups = "drop"
  )

print(summary_stats)

overall_summary <- all_item_comparisons_corrected %>%
  summarise(
    total_tests = n(),
    significant_uncorrected = sum(significant_uncorrected),
    significant_fdr = sum(significant_fdr),
    significant_bonferroni = sum(significant_bonferroni)
  )

cat(sprintf("\nOverall Summary:\n"))
cat(sprintf("Total tests: %d\n", overall_summary$total_tests))
cat(sprintf("Significant (uncorrected): %d (%.1f%%)\n", 
            overall_summary$significant_uncorrected,
            100 * overall_summary$significant_uncorrected / overall_summary$total_tests))
cat(sprintf("Significant (FDR): %d (%.1f%%)\n", 
            overall_summary$significant_fdr,
            100 * overall_summary$significant_fdr / overall_summary$total_tests))
cat(sprintf("Significant (Bonferroni): %d (%.1f%%)\n", 
            overall_summary$significant_bonferroni,
            100 * overall_summary$significant_bonferroni / overall_summary$total_tests))

cat("\n=== ITEM GROUP MIXED-EFFECTS ANALYSIS COMPLETE ===\n")
cat("Results saved to: analysis/tables/item_group_pairwise_mixed_effects_corrected.csv\n")
cat("This replaces the previous chi-square approach with consistent mixed-effects + odds ratios\n")