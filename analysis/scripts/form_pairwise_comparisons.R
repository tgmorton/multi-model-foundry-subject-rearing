# Form Pairwise Comparisons Analysis
# ===================================
# Compares specific form pairs within each model:
# 1. complex_emb vs complex_long
# 2. target_negation vs context_negation  
# 3. target_negation vs both_negation

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
  cat(sprintf("\n=== Analyzing %s ===\n", model_name))
  
  model_data <- endstate_data %>% filter(model == model_name)
  
  # Analyze each form pair
  for (i in seq_along(form_pairs)) {
    pair <- form_pairs[[i]]
    pair_name <- pair_names[i]
    
    cat(sprintf("\nForm pair: %s vs %s\n", pair[1], pair[2]))
    
    # Filter to only the two forms in this comparison
    pair_data <- model_data %>%
      filter(form %in% pair)
    
    if (nrow(pair_data) == 0) {
      cat("No data found for this pair\n")
      next
    }
    
    # Fit mixed-effects model
    tryCatch({
      mod <- glmer(correct ~ form + (1|item_id), 
                   data = pair_data, 
                   family = binomial,
                   control = glmerControl(optimizer = "bobyqa"))
      
      # Get estimated marginal means
      emm <- emmeans(mod, ~ form)
      emm_summary <- summary(emm, type = "response")
      
      # Pairwise comparison
      contrast <- pairs(emm, type = "response")
      contrast_summary <- summary(contrast, infer = TRUE)
      
      # Store results
      result_row <- tibble(
        model = model_name,
        comparison = pair_name,
        form1 = pair[1],
        form2 = pair[2],
        form1_prob = emm_summary$prob[emm_summary$form == pair[1]],
        form1_se = emm_summary$SE[emm_summary$form == pair[1]],
        form1_ci_low = emm_summary$asymp.LCL[emm_summary$form == pair[1]],
        form1_ci_high = emm_summary$asymp.UCL[emm_summary$form == pair[1]],
        form2_prob = emm_summary$prob[emm_summary$form == pair[2]],
        form2_se = emm_summary$SE[emm_summary$form == pair[2]],
        form2_ci_low = emm_summary$asymp.LCL[emm_summary$form == pair[2]],
        form2_ci_high = emm_summary$asymp.UCL[emm_summary$form == pair[2]],
        odds_ratio = contrast_summary$odds.ratio,
        or_ci_low = contrast_summary$asymp.LCL,
        or_ci_high = contrast_summary$asymp.UCL,
        z_value = contrast_summary$z.ratio,
        p_value = contrast_summary$p.value
      )
      
      all_results <- bind_rows(all_results, result_row)
      
      # Print summary
      cat(sprintf("  %s: %.3f [%.3f, %.3f]\n", 
                  pair[1], result_row$form1_prob, result_row$form1_ci_low, result_row$form1_ci_high))
      cat(sprintf("  %s: %.3f [%.3f, %.3f]\n", 
                  pair[2], result_row$form2_prob, result_row$form2_ci_low, result_row$form2_ci_high))
      cat(sprintf("  Odds Ratio: %.3f [%.3f, %.3f], p = %.3f\n", 
                  result_row$odds_ratio, result_row$or_ci_low, result_row$or_ci_high, result_row$p_value))
      
    }, error = function(e) {
      cat(sprintf("Error fitting model for %s: %s\n", pair_name, e$message))
    })
  }
}

# Save results
write_csv(all_results, "analysis/tables/form_pairwise_comparisons.csv")

# Print summary table
cat("\n=== SUMMARY OF ALL FORM PAIRWISE COMPARISONS ===\n")
cat("Available columns:", paste(names(all_results), collapse = ", "), "\n")

summary_table <- all_results %>%
  left_join(model_labels, by = "model") %>%
  select(model, Model, comparison, form1, form1_prob, form2, form2_prob, odds_ratio, p_value) %>%
  arrange(Model, comparison)

print(summary_table, n = Inf, width = Inf)

cat(sprintf("\nResults saved to: analysis/tables/form_pairwise_comparisons.csv\n"))
cat("Next: Run LaTeX table generation script\n")