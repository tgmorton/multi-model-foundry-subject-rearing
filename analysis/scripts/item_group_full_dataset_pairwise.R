# Item Group Pairwise Comparisons Using FULL DATASET
# ===================================================
# This version uses the complete dataset for all comparisons to ensure
# consistent marginal means across all analyses

library(tidyverse)
library(lme4)
library(emmeans)

cat("=== ITEM GROUP PAIRWISE COMPARISONS (FULL DATASET METHOD) ===\n\n")

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

# Add comprehensive group labels to ALL data
endstate_data <- endstate_data %>%
  mutate(
    # Individual group labels
    group_label = case_when(
      str_detect(item_group, "^1a_") ~ "3rd_singular",
      str_detect(item_group, "^1b_") ~ "3rd_plural",
      str_detect(item_group, "^2a_") ~ "2nd_singular",
      str_detect(item_group, "^2b_") ~ "2nd_plural",
      str_detect(item_group, "^3a_") ~ "1st_singular",
      str_detect(item_group, "^3b_") ~ "1st_plural",
      str_detect(item_group, "^4a_") ~ "subject_control",
      str_detect(item_group, "^4b_") ~ "object_control",
      str_detect(item_group, "^5a_") ~ "seems",
      str_detect(item_group, "^5b_") ~ "be",
      str_detect(item_group, "^6_") ~ "raising",
      str_detect(item_group, "^7a_") ~ "no_topic_shift",
      str_detect(item_group, "^7b_") ~ "topic_shift",
      TRUE ~ "other"
    ),
    # Person categories
    person = case_when(
      str_detect(item_group, "^1[ab]_") ~ "3rd",
      str_detect(item_group, "^2[ab]_") ~ "2nd",
      str_detect(item_group, "^3[ab]_") ~ "1st",
      TRUE ~ NA_character_
    ),
    # Number categories
    number = case_when(
      str_detect(item_group, "^[123]a_") ~ "singular",
      str_detect(item_group, "^[123]b_") ~ "plural",
      TRUE ~ NA_character_
    ),
    # Control type
    control_type = case_when(
      str_detect(item_group, "^4a_") ~ "subject",
      str_detect(item_group, "^4b_") ~ "object",
      TRUE ~ NA_character_
    ),
    # Expletive type
    expletive_type = case_when(
      str_detect(item_group, "^5a_") ~ "seems",
      str_detect(item_group, "^5b_") ~ "be",
      TRUE ~ NA_character_
    ),
    # Topic shift type
    topic_type = case_when(
      str_detect(item_group, "^7a_") ~ "no_shift",
      str_detect(item_group, "^7b_") ~ "shift",
      TRUE ~ NA_character_
    )
  )

# Initialize storage for all comparisons
all_comparisons <- tibble()

# (Helper function removed - using direct implementation for each comparison type)

# Process each model
for (model_name in unique(endstate_data$model)) {
  cat(sprintf("\n=== MODEL: %s ===\n", model_labels$Model[model_labels$model == model_name]))
  
  model_data <- endstate_data %>% filter(model == model_name)
  model_results <- tibble()
  
  # 1. PERSON COMPARISONS (using full person data)
  cat("\nPerson comparisons (using all person items):\n")
  person_data <- model_data %>% filter(!is.na(person))
  
  if (nrow(person_data) > 0) {
    # Fit single model with person as factor
    person_mod <- glmer(correct ~ person + (1|item_id), 
                       data = person_data %>% mutate(person = factor(person, levels = c("1st", "2nd", "3rd"))),
                       family = binomial,
                       control = glmerControl(optimizer = "bobyqa"))
    
    emm_person <- emmeans(person_mod, ~ person)
    emm_person_summary <- summary(emm_person, type = "response")
    contrast_person <- pairs(emm_person, type = "response")
    contrast_person_summary <- summary(contrast_person, infer = TRUE)
    
    # Extract all three person comparisons
    person_comparisons <- tribble(
      ~comp_name, ~g1, ~g2,
      "B2_Person_1st_vs_2nd", "1st_person", "2nd_person",
      "B2_Person_1st_vs_3rd", "1st_person", "3rd_person", 
      "B2_Person_2nd_vs_3rd", "2nd_person", "3rd_person"
    )
    
    for (i in 1:nrow(person_comparisons)) {
      # Map the comparison to the contrast results
      if (i <= nrow(contrast_person_summary)) {
        comp_result <- tibble(
          comparison_type = person_comparisons$comp_name[i],
          group1 = person_comparisons$g1[i],
          group2 = person_comparisons$g2[i],
          group1_prob = emm_person_summary$prob[ifelse(str_detect(person_comparisons$g1[i], "1st"), 1, 
                                                        ifelse(str_detect(person_comparisons$g1[i], "2nd"), 2, 3))],
          group1_se = emm_person_summary$SE[ifelse(str_detect(person_comparisons$g1[i], "1st"), 1,
                                                   ifelse(str_detect(person_comparisons$g1[i], "2nd"), 2, 3))],
          group1_ci_low = emm_person_summary$asymp.LCL[ifelse(str_detect(person_comparisons$g1[i], "1st"), 1,
                                                              ifelse(str_detect(person_comparisons$g1[i], "2nd"), 2, 3))],
          group1_ci_high = emm_person_summary$asymp.UCL[ifelse(str_detect(person_comparisons$g1[i], "1st"), 1,
                                                               ifelse(str_detect(person_comparisons$g1[i], "2nd"), 2, 3))],
          group2_prob = emm_person_summary$prob[ifelse(str_detect(person_comparisons$g2[i], "1st"), 1,
                                                       ifelse(str_detect(person_comparisons$g2[i], "2nd"), 2, 3))],
          group2_se = emm_person_summary$SE[ifelse(str_detect(person_comparisons$g2[i], "1st"), 1,
                                                   ifelse(str_detect(person_comparisons$g2[i], "2nd"), 2, 3))],
          group2_ci_low = emm_person_summary$asymp.LCL[ifelse(str_detect(person_comparisons$g2[i], "1st"), 1,
                                                              ifelse(str_detect(person_comparisons$g2[i], "2nd"), 2, 3))],
          group2_ci_high = emm_person_summary$asymp.UCL[ifelse(str_detect(person_comparisons$g2[i], "1st"), 1,
                                                               ifelse(str_detect(person_comparisons$g2[i], "2nd"), 2, 3))],
          odds_ratio = contrast_person_summary$odds.ratio[i],
          or_ci_low = contrast_person_summary$asymp.LCL[i],
          or_ci_high = contrast_person_summary$asymp.UCL[i],
          z_value = contrast_person_summary$z.ratio[i],
          p_value = contrast_person_summary$p.value[i],
          # Default values for Fisher's exact (not applicable)
          group1_n_success = NA_integer_,
          group1_n_total = NA_integer_,
          group2_n_success = NA_integer_,
          group2_n_total = NA_integer_
        )
        model_results <- bind_rows(model_results, comp_result)
        cat(sprintf("  %s vs %s: %.1f%% vs %.1f%% (OR = %.3f, p = %.4f)\n",
                   person_comparisons$g1[i], person_comparisons$g2[i],
                   comp_result$group1_prob * 100, comp_result$group2_prob * 100,
                   comp_result$odds_ratio, comp_result$p_value))
      }
    }
  }
  
  # 2. NUMBER COMPARISONS WITHIN PERSON (using full data with interaction)
  cat("\nNumber comparisons (using person*number interaction model):\n")
  
  if (nrow(person_data) > 0) {
    # Fit model with person*number interaction
    number_mod <- glmer(correct ~ person * number + (1|item_id),
                       data = person_data %>% 
                         mutate(person = factor(person, levels = c("1st", "2nd", "3rd")),
                               number = factor(number, levels = c("singular", "plural"))),
                       family = binomial,
                       control = glmerControl(optimizer = "bobyqa"))
    
    # Get emmeans for person:number combinations
    emm_number <- emmeans(number_mod, ~ number | person)
    
    # Get contrasts for each person
    for (p in c("1st", "2nd", "3rd")) {
      contrast_number <- pairs(emm_number, by = "person", type = "response")
      contrast_summary <- summary(contrast_number, infer = TRUE)
      
      # Filter to this person's contrast
      person_contrast <- contrast_summary %>% filter(person == p)
      
      if (nrow(person_contrast) > 0) {
        # Get marginal means for this person
        emm_summary <- summary(emm_number, type = "response") %>%
          filter(person == p)
        
        comp_result <- tibble(
          comparison_type = paste0("B1_Number_", p, "_Sing_vs_Plural"),
          group1 = paste0(p, "_singular"),
          group2 = paste0(p, "_plural"),
          group1_prob = emm_summary$prob[emm_summary$number == "singular"],
          group1_se = emm_summary$SE[emm_summary$number == "singular"],
          group1_ci_low = emm_summary$asymp.LCL[emm_summary$number == "singular"],
          group1_ci_high = emm_summary$asymp.UCL[emm_summary$number == "singular"],
          group2_prob = emm_summary$prob[emm_summary$number == "plural"],
          group2_se = emm_summary$SE[emm_summary$number == "plural"],
          group2_ci_low = emm_summary$asymp.LCL[emm_summary$number == "plural"],
          group2_ci_high = emm_summary$asymp.UCL[emm_summary$number == "plural"],
          odds_ratio = person_contrast$odds.ratio[1],
          or_ci_low = person_contrast$asymp.LCL[1],
          or_ci_high = person_contrast$asymp.UCL[1],
          z_value = person_contrast$z.ratio[1],
          p_value = person_contrast$p.value[1],
          # Default values for Fisher's exact (not applicable)
          group1_n_success = NA_integer_,
          group1_n_total = NA_integer_,
          group2_n_success = NA_integer_,
          group2_n_total = NA_integer_
        )
        
        model_results <- bind_rows(model_results, comp_result)
        cat(sprintf("  %s singular vs plural: %.1f%% vs %.1f%% (OR = %.3f, p = %.4f)\n",
                   p, comp_result$group1_prob * 100, comp_result$group2_prob * 100,
                   comp_result$odds_ratio, comp_result$p_value))
      }
    }
  }
  
  # 3. CONTROL COMPARISONS (using observed proportions due to perfect separation)
  cat("\nControl comparisons:\n")
  control_data <- model_data %>% filter(!is.na(control_type))
  
  if (nrow(control_data) > 0) {
    # Calculate observed proportions directly (mixed-effects fails due to perfect separation)
    control_summary <- control_data %>%
      group_by(control_type) %>%
      summarise(
        n_correct = sum(correct),
        n_total = n(),
        prop = n_correct / n_total,
        se = sqrt(prop * (1 - prop) / n_total),
        .groups = "drop"
      )
    
    subject_stats <- control_summary %>% filter(control_type == "subject")
    object_stats <- control_summary %>% filter(control_type == "object")
    
    # Fisher's exact test for significance (better for 0% groups)
    contingency_table <- control_data %>%
      count(control_type, correct) %>%
      pivot_wider(names_from = correct, values_from = n, values_fill = 0)
    
    fisher_test <- fisher.test(matrix(c(contingency_table$`1`[1], contingency_table$`0`[1],
                                       contingency_table$`1`[2], contingency_table$`0`[2]), 
                                     nrow = 2))
    
    comp_result <- tibble(
      comparison_type = "B3_Control_Subject_vs_Object",
      group1 = "subject_control",
      group2 = "object_control",
      group1_prob = subject_stats$prop,
      group1_se = subject_stats$se,
      group1_ci_low = max(0, subject_stats$prop - 1.96 * subject_stats$se),
      group1_ci_high = min(1, subject_stats$prop + 1.96 * subject_stats$se),
      group2_prob = object_stats$prop,
      group2_se = object_stats$se,
      group2_ci_low = max(0, object_stats$prop - 1.96 * object_stats$se),
      group2_ci_high = min(1, object_stats$prop + 1.96 * object_stats$se),
      odds_ratio = fisher_test$estimate,
      or_ci_low = fisher_test$conf.int[1],
      or_ci_high = fisher_test$conf.int[2],
      z_value = NA_real_, # Not applicable for Fisher's test
      p_value = fisher_test$p.value,
      # Add Fisher's exact test specific information
      group1_n_success = subject_stats$n_correct,
      group1_n_total = subject_stats$n_total,
      group2_n_success = object_stats$n_correct,
      group2_n_total = object_stats$n_total
    )
    
    model_results <- bind_rows(model_results, comp_result)
    cat(sprintf("  Subject vs Object control: %.1f%% vs %.1f%% (OR = %.3f, p = %.4f) [Fisher's exact]\n",
               comp_result$group1_prob * 100, comp_result$group2_prob * 100,
               comp_result$odds_ratio, comp_result$p_value))
  }
  
  # 4. EXPLETIVE COMPARISONS (using all expletive items)
  cat("\nExpletive comparisons:\n")
  expletive_data <- model_data %>% filter(!is.na(expletive_type))
  
  if (nrow(expletive_data) > 0) {
    tryCatch({
      expletive_mod <- glmer(correct ~ expletive_type + (1|item_id),
                            data = expletive_data %>% mutate(expletive_type = factor(expletive_type)),
                            family = binomial,
                            control = glmerControl(optimizer = "bobyqa"))
      
      emm_expletive <- emmeans(expletive_mod, ~ expletive_type)
      emm_expletive_summary <- summary(emm_expletive, type = "response")
      contrast_expletive <- pairs(emm_expletive, type = "response")
      contrast_expletive_summary <- summary(contrast_expletive, infer = TRUE)
      
      comp_result <- tibble(
        comparison_type = "B4_Expletive_Seems_vs_Be",
        group1 = "seems",
        group2 = "be",
        group1_prob = emm_expletive_summary$prob[emm_expletive_summary$expletive_type == "seems"],
        group1_se = emm_expletive_summary$SE[emm_expletive_summary$expletive_type == "seems"],
        group1_ci_low = emm_expletive_summary$asymp.LCL[emm_expletive_summary$expletive_type == "seems"],
        group1_ci_high = emm_expletive_summary$asymp.UCL[emm_expletive_summary$expletive_type == "seems"],
        group2_prob = emm_expletive_summary$prob[emm_expletive_summary$expletive_type == "be"],
        group2_se = emm_expletive_summary$SE[emm_expletive_summary$expletive_type == "be"],
        group2_ci_low = emm_expletive_summary$asymp.LCL[emm_expletive_summary$expletive_type == "be"],
        group2_ci_high = emm_expletive_summary$asymp.UCL[emm_expletive_summary$expletive_type == "be"],
        odds_ratio = contrast_expletive_summary$odds.ratio[1],
        or_ci_low = contrast_expletive_summary$asymp.LCL[1],
        or_ci_high = contrast_expletive_summary$asymp.UCL[1],
        z_value = contrast_expletive_summary$z.ratio[1],
        p_value = contrast_expletive_summary$p.value[1],
        # Default values for Fisher's exact (not applicable)
        group1_n_success = NA_integer_,
        group1_n_total = NA_integer_,
        group2_n_success = NA_integer_,
        group2_n_total = NA_integer_
      )
      
      model_results <- bind_rows(model_results, comp_result)
      cat(sprintf("  Seems vs Be: %.1f%% vs %.1f%% (OR = %.3f, p = %.4f)\n",
                 comp_result$group1_prob * 100, comp_result$group2_prob * 100,
                 comp_result$odds_ratio, comp_result$p_value))
    }, error = function(e) {
      cat(sprintf("  Error in expletive comparison: %s\n", e$message))
    })
  }
  
  # 5. TOPIC SHIFT COMPARISONS (using all topic items)
  cat("\nTopic shift comparisons:\n")
  topic_data <- model_data %>% filter(!is.na(topic_type))
  
  if (nrow(topic_data) > 0) {
    tryCatch({
      topic_mod <- glmer(correct ~ topic_type + (1|item_id),
                        data = topic_data %>% mutate(topic_type = factor(topic_type)),
                        family = binomial,
                        control = glmerControl(optimizer = "bobyqa"))
      
      emm_topic <- emmeans(topic_mod, ~ topic_type)
      emm_topic_summary <- summary(emm_topic, type = "response")
      contrast_topic <- pairs(emm_topic, type = "response")
      contrast_topic_summary <- summary(contrast_topic, infer = TRUE)
      
      comp_result <- tibble(
        comparison_type = "B5_Topic_NoShift_vs_Shift",
        group1 = "no_topic_shift",
        group2 = "topic_shift",
        group1_prob = emm_topic_summary$prob[emm_topic_summary$topic_type == "no_shift"],
        group1_se = emm_topic_summary$SE[emm_topic_summary$topic_type == "no_shift"],
        group1_ci_low = emm_topic_summary$asymp.LCL[emm_topic_summary$topic_type == "no_shift"],
        group1_ci_high = emm_topic_summary$asymp.UCL[emm_topic_summary$topic_type == "no_shift"],
        group2_prob = emm_topic_summary$prob[emm_topic_summary$topic_type == "shift"],
        group2_se = emm_topic_summary$SE[emm_topic_summary$topic_type == "shift"],
        group2_ci_low = emm_topic_summary$asymp.LCL[emm_topic_summary$topic_type == "shift"],
        group2_ci_high = emm_topic_summary$asymp.UCL[emm_topic_summary$topic_type == "shift"],
        odds_ratio = contrast_topic_summary$odds.ratio[1],
        or_ci_low = contrast_topic_summary$asymp.LCL[1],
        or_ci_high = contrast_topic_summary$asymp.UCL[1],
        z_value = contrast_topic_summary$z.ratio[1],
        p_value = contrast_topic_summary$p.value[1],
        # Default values for Fisher's exact (not applicable)
        group1_n_success = NA_integer_,
        group1_n_total = NA_integer_,
        group2_n_success = NA_integer_,
        group2_n_total = NA_integer_
      )
      
      model_results <- bind_rows(model_results, comp_result)
      cat(sprintf("  No shift vs Shift: %.1f%% vs %.1f%% (OR = %.3f, p = %.4f)\n",
                 comp_result$group1_prob * 100, comp_result$group2_prob * 100,
                 comp_result$odds_ratio, comp_result$p_value))
    }, error = function(e) {
      cat(sprintf("  Error in topic comparison: %s\n", e$message))
    })
  }
  
  # Add model info to results
  model_results <- model_results %>%
    mutate(
      model = model_name,
      Model = model_labels$Model[model_labels$model == model_name]
    )
  
  all_comparisons <- bind_rows(all_comparisons, model_results)
}

# Apply multiple comparisons corrections
cat("\n\n=== APPLYING MULTIPLE COMPARISONS CORRECTIONS ===\n")
total_tests <- nrow(all_comparisons)
cat(sprintf("Total tests: %d\n", total_tests))

all_comparisons_corrected <- all_comparisons %>%
  mutate(
    p_value_fdr = p.adjust(p_value, method = "fdr"),
    p_value_bonferroni = p.adjust(p_value, method = "bonferroni"),
    significant_uncorrected = p_value < 0.05,
    significant_fdr = p_value_fdr < 0.05,
    significant_bonferroni = p_value_bonferroni < 0.05
  )

# Save results
write_csv(all_comparisons_corrected, "analysis/tables/item_group_full_dataset_pairwise_corrected.csv")

# Summary
cat("\n=== SUMMARY ===\n")
summary_stats <- all_comparisons_corrected %>%
  summarise(
    total_tests = n(),
    significant_uncorrected = sum(significant_uncorrected),
    significant_fdr = sum(significant_fdr),
    significant_bonferroni = sum(significant_bonferroni)
  )

cat(sprintf("Significant (uncorrected): %d / %d (%.1f%%)\n", 
           summary_stats$significant_uncorrected, 
           summary_stats$total_tests,
           100 * summary_stats$significant_uncorrected / summary_stats$total_tests))
cat(sprintf("Significant (FDR): %d / %d (%.1f%%)\n", 
           summary_stats$significant_fdr,
           summary_stats$total_tests,
           100 * summary_stats$significant_fdr / summary_stats$total_tests))
cat(sprintf("Significant (Bonferroni): %d / %d (%.1f%%)\n", 
           summary_stats$significant_bonferroni,
           summary_stats$total_tests,
           100 * summary_stats$significant_bonferroni / summary_stats$total_tests))

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Results saved to: analysis/tables/item_group_full_dataset_pairwise_corrected.csv\n")
cat("\nKEY IMPROVEMENTS:\n")
cat("- All comparisons now use FULL dataset (not subsets)\n")
cat("- Marginal means are consistent across all comparisons\n")
cat("- Person*Number interaction properly modeled\n")
cat("- More statistical power from using all available data\n")