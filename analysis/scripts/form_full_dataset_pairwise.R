# Form Pairwise Comparisons Using FULL DATASET
# =============================================
# This version uses complete dataset to ensure consistent marginal means

library(tidyverse)
library(lme4)
library(emmeans)

cat("=== FORM PAIRWISE COMPARISONS (FULL DATASET METHOD) ===\n\n")

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

# Initialize results storage
all_results <- tibble()

# Define form pairs to compare
form_pairs <- list(
  c("complex_emb", "complex_long"),
  c("target_negation", "context_negation"),
  c("target_negation", "both_negation")
)

pair_names <- c(
  "complex_emb_vs_complex_long",
  "target_negation_vs_context_negation", 
  "target_negation_vs_both_negation"
)

# Analyze each model
for (model_name in unique(endstate_data$model)) {
  cat(sprintf("\n=== MODEL: %s ===\n", model_labels$Model[model_labels$model == model_name]))
  
  model_data <- endstate_data %>% filter(model == model_name)
  
  # Fit single model with ALL forms
  cat("Fitting model with all forms for consistent marginal means...\n")
  
  tryCatch({
    # Convert form to factor with specific ordering
    model_data <- model_data %>%
      mutate(form = factor(form, levels = c("default", "complex_long", "complex_emb",
                                           "context_negation", "target_negation", "both_negation")))
    
    # Fit model on FULL dataset
    full_mod <- glmer(correct ~ form + (1|item_id),
                     data = model_data,
                     family = binomial,
                     control = glmerControl(optimizer = "bobyqa"))
    
    # Get marginal means for ALL forms
    emm_all <- emmeans(full_mod, ~ form)
    emm_all_summary <- summary(emm_all, type = "response")
    
    cat("\nMarginal means for all forms:\n")
    for (f in levels(model_data$form)) {
      prob <- emm_all_summary$prob[emm_all_summary$form == f]
      cat(sprintf("  %s: %.1f%%\n", f, prob * 100))
    }
    
    # Now extract specific pairwise comparisons
    cat("\nPairwise comparisons:\n")
    
    for (i in seq_along(form_pairs)) {
      pair <- form_pairs[[i]]
      pair_name <- pair_names[i]
      
      # Get specific contrast
      contrast_spec <- paste0(pair[1], " - ", pair[2])
      contrast_result <- pairs(emm_all, type = "response")
      contrast_summary <- summary(contrast_result, infer = TRUE)
      
      # Find the specific comparison
      contrast_row <- contrast_summary %>%
        filter(str_detect(contrast, pair[1]) & str_detect(contrast, pair[2]))
      
      if (nrow(contrast_row) == 0) {
        cat(sprintf("  WARNING: Could not find comparison for %s vs %s\n", pair[1], pair[2]))
        next
      }
      
      # Extract values
      form1_stats <- emm_all_summary %>% filter(form == pair[1])
      form2_stats <- emm_all_summary %>% filter(form == pair[2])
      
      result_row <- tibble(
        model = model_name,
        Model = model_labels$Model[model_labels$model == model_name],
        comparison = pair_name,
        form1 = pair[1],
        form2 = pair[2],
        form1_prob = form1_stats$prob,
        form1_se = form1_stats$SE,
        form1_ci_low = form1_stats$asymp.LCL,
        form1_ci_high = form1_stats$asymp.UCL,
        form2_prob = form2_stats$prob,
        form2_se = form2_stats$SE,
        form2_ci_low = form2_stats$asymp.LCL,
        form2_ci_high = form2_stats$asymp.UCL,
        odds_ratio = contrast_row$odds.ratio[1],
        or_ci_low = contrast_row$asymp.LCL[1],
        or_ci_high = contrast_row$asymp.UCL[1],
        z_value = contrast_row$z.ratio[1],
        p_value = contrast_row$p.value[1]
      )
      
      all_results <- bind_rows(all_results, result_row)
      
      cat(sprintf("  %s vs %s: %.1f%% vs %.1f%% (OR = %.3f, p = %.4f)\n",
                 pair[1], pair[2],
                 form1_stats$prob * 100, form2_stats$prob * 100,
                 contrast_row$odds.ratio[1], contrast_row$p.value[1]))
    }
    
  }, error = function(e) {
    cat(sprintf("  Error fitting model: %s\n", e$message))
  })
}

# Apply multiple comparisons corrections
cat("\n\n=== APPLYING MULTIPLE COMPARISONS CORRECTIONS ===\n")
total_tests <- nrow(all_results)
cat(sprintf("Total form pairwise tests: %d\n", total_tests))

all_results_corrected <- all_results %>%
  mutate(
    p_value_fdr = p.adjust(p_value, method = "fdr"),
    p_value_bonferroni = p.adjust(p_value, method = "bonferroni"),
    significant_uncorrected = p_value < 0.05,
    significant_fdr = p_value_fdr < 0.05,
    significant_bonferroni = p_value_bonferroni < 0.05
  )

# Save results
write_csv(all_results_corrected, "analysis/tables/form_full_dataset_pairwise_corrected.csv")

# Summary statistics
cat("\n=== SUMMARY STATISTICS ===\n")
summary_stats <- all_results_corrected %>%
  group_by(comparison) %>%
  summarise(
    n_tests = n(),
    significant_uncorrected = sum(significant_uncorrected),
    significant_fdr = sum(significant_fdr),
    significant_bonferroni = sum(significant_bonferroni),
    .groups = "drop"
  )

print(summary_stats)

overall_summary <- all_results_corrected %>%
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

cat("\n=== FORM COMPARISONS COMPLETE ===\n")
cat("Results saved to: analysis/tables/form_full_dataset_pairwise_corrected.csv\n")
cat("\nKEY IMPROVEMENTS:\n")
cat("- All comparisons use FULL dataset (not subsets)\n")
cat("- Marginal means are consistent across all comparisons\n")
cat("- More statistical power from using all available data\n")