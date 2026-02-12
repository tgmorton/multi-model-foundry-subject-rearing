# Forms vs Default - Mixed-Effects Logistic Regression
# ===================================================
# Generate forms vs default comparisons using mixed-effects logistic regression
# with odds ratios (consistent with all other analyses)

library(tidyverse)
library(lme4)
library(emmeans)

cat("=== FORMS VS DEFAULT - MIXED-EFFECTS APPROACH ===\n\n")

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
all_comparisons <- tibble()

# Process each model
for (model_name in unique(endstate_data$model)) {
  cat(sprintf("\n=== MODEL: %s ===\n", model_labels$Model[model_labels$model == model_name]))
  
  model_data <- endstate_data %>% filter(model == model_name)
  model_results <- tibble()
  
  # Check available forms
  available_forms <- unique(model_data$form)
  cat(sprintf("Available forms: %s\n", paste(available_forms, collapse = ", ")))
  
  if (!"default" %in% available_forms) {
    cat("Warning: No 'default' form found in this model. Skipping.\n")
    next
  }
  
  # Set factor levels with default as reference
  model_data <- model_data %>%
    mutate(form = factor(form, levels = c("default", setdiff(available_forms, "default"))))
  
  # Fit mixed-effects model with form as predictor (default is reference)
  tryCatch({
    cat("  Fitting mixed-effects model...\n")
    form_mod <- glmer(correct ~ form + (1|item_id), 
                     data = model_data,
                     family = binomial,
                     control = glmerControl(optimizer = "bobyqa"))
    
    cat("  Getting estimated marginal means...\n")
    # Get estimated marginal means for each form
    emm_forms <- emmeans(form_mod, ~ form)
    emm_forms_summary <- summary(emm_forms, type = "response")
    
    cat("  Computing contrasts...\n")
    # Get contrasts of each form vs default (reference level)
    contrasts_forms <- contrast(emm_forms, method = "trt.vs.ctrl", ref = 1)  # ref = 1 means first level (default)
    contrasts_summary <- summary(contrasts_forms, type = "response", infer = TRUE)
    
    cat("Forms vs default comparisons:\n")
    cat(sprintf("  Found %d contrasts to process\n", nrow(contrasts_summary)))
    
    # Extract results for each non-default form
    for (i in 1:nrow(contrasts_summary)) {
      form_name <- str_replace(contrasts_summary$contrast[i], " / default", "")
      
      # Get marginal means
      default_prob <- emm_forms_summary$prob[emm_forms_summary$form == "default"]
      form_prob <- emm_forms_summary$prob[emm_forms_summary$form == form_name]
      
      default_se <- emm_forms_summary$SE[emm_forms_summary$form == "default"]
      form_se <- emm_forms_summary$SE[emm_forms_summary$form == form_name]
      
      default_ci_low <- emm_forms_summary$asymp.LCL[emm_forms_summary$form == "default"]
      default_ci_high <- emm_forms_summary$asymp.UCL[emm_forms_summary$form == "default"]
      
      form_ci_low <- emm_forms_summary$asymp.LCL[emm_forms_summary$form == form_name]
      form_ci_high <- emm_forms_summary$asymp.UCL[emm_forms_summary$form == form_name]
      
      comp_result <- tibble(
        comparison_type = paste0("Form_", form_name, "_vs_default"),
        form = form_name,
        default_form = "default",
        form_prob = form_prob,
        form_se = form_se,
        form_ci_low = form_ci_low,
        form_ci_high = form_ci_high,
        default_prob = default_prob,
        default_se = default_se,
        default_ci_low = default_ci_low,
        default_ci_high = default_ci_high,
        odds_ratio = contrasts_summary$odds.ratio[i],
        or_ci_low = contrasts_summary$asymp.LCL[i],
        or_ci_high = contrasts_summary$asymp.UCL[i],
        z_value = contrasts_summary$z.ratio[i],
        p_value = contrasts_summary$p.value[i],
        model = model_name,
        Model = model_labels$Model[model_labels$model == model_name]
      )
      
      model_results <- bind_rows(model_results, comp_result)
      cat(sprintf("  %s vs default: %.1f%% vs %.1f%% (OR = %.3f, p = %.4f)\n",
                 form_name, form_prob * 100, default_prob * 100,
                 contrasts_summary$odds.ratio[i], contrasts_summary$p.value[i]))
    }
    
  }, error = function(e) {
    cat(sprintf("  Error fitting model for %s: %s\n", model_name, e$message))
    print(e)
  })
  
  all_comparisons <- bind_rows(all_comparisons, model_results)
}

# Apply multiple comparisons corrections
cat("\n\n=== APPLYING MULTIPLE COMPARISONS CORRECTIONS ===\n")

all_comparisons_corrected <- all_comparisons %>%
  group_by(Model) %>%
  mutate(
    # Within-model FDR correction
    p_value_fdr_within = p.adjust(p_value, method = "fdr"),
    significant_fdr_within = p_value_fdr_within < 0.05
  ) %>%
  ungroup() %>%
  mutate(
    # Global corrections across all tests
    p_value_fdr_global = p.adjust(p_value, method = "fdr"),
    p_value_bonferroni_global = p.adjust(p_value, method = "bonferroni"),
    significant_uncorrected = p_value < 0.05,
    significant_fdr_global = p_value_fdr_global < 0.05,
    significant_bonferroni_global = p_value_bonferroni_global < 0.05
  )

# Save results
write_csv(all_comparisons_corrected, "analysis/tables/forms_vs_default_mixed_effects_corrected.csv")

# Create clean reporting version
clean_reporting <- all_comparisons_corrected %>%
  mutate(
    `Form %` = sprintf("%.1f%%", form_prob * 100),
    `Default %` = sprintf("%.1f%%", default_prob * 100),
    `Difference` = sprintf("%+.1f%%", (form_prob - default_prob) * 100),
    `Odds Ratio` = sprintf("%.3f", odds_ratio),
    `OR 95% CI` = sprintf("[%.3f, %.3f]", or_ci_low, or_ci_high),
    `p-value` = case_when(
      p_value < 0.001 ~ "< .001",
      p_value < 0.01 ~ sprintf("%.3f", p_value),
      TRUE ~ sprintf("%.3f", p_value)
    ),
    `p-corrected` = case_when(
      p_value_fdr_within < 0.001 ~ "< .001",
      p_value_fdr_within < 0.01 ~ sprintf("%.3f", p_value_fdr_within),
      TRUE ~ sprintf("%.3f", p_value_fdr_within)
    ),
    `Significant` = ifelse(significant_fdr_within, "***", "ns"),
    `Effect` = case_when(
      !significant_fdr_within ~ "No difference",
      odds_ratio > 1 ~ sprintf("%s > default", str_replace_all(form, "_", " ")),
      odds_ratio < 1 ~ sprintf("default > %s", str_replace_all(form, "_", " ")),
      TRUE ~ "No difference"
    ),
    `Form` = case_when(
      form == "both_negation" ~ "Both Negation",
      form == "complex_emb" ~ "Complex Embedded", 
      form == "complex_long" ~ "Complex Long",
      form == "context_negation" ~ "Context Negation",
      form == "target_negation" ~ "Target Negation",
      TRUE ~ str_to_title(str_replace_all(form, "_", " "))
    )
  ) %>%
  select(
    Model, Form, `Form %`, `Default %`, Difference, `Odds Ratio`, `OR 95% CI`,
    `p-value`, `p-corrected`, Significant, Effect
  ) %>%
  arrange(Model, Form)

# Save clean reporting version
write_csv(clean_reporting, "analysis/tables/forms_vs_default_mixed_effects_reporting.csv")

# Summary
cat("\n=== SUMMARY ===\n")
total_tests <- nrow(all_comparisons_corrected)
significant_within <- sum(all_comparisons_corrected$significant_fdr_within)
significant_global <- sum(all_comparisons_corrected$significant_fdr_global)

cat(sprintf("Total tests: %d\n", total_tests))
cat(sprintf("Significant (FDR within-model): %d / %d (%.1f%%)\n", 
           significant_within, total_tests, 100 * significant_within / total_tests))
cat(sprintf("Significant (FDR global): %d / %d (%.1f%%)\n", 
           significant_global, total_tests, 100 * significant_global / total_tests))

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Results saved to:\n")
cat("- Raw data: analysis/tables/forms_vs_default_mixed_effects_corrected.csv\n")
cat("- Reporting: analysis/tables/forms_vs_default_mixed_effects_reporting.csv\n")
cat("\nKEY IMPROVEMENTS:\n")
cat("- Uses mixed-effects logistic regression (not chi-square)\n")
cat("- Provides odds ratios with confidence intervals\n")
cat("- Full dataset approach for consistent marginal means\n")
cat("- Proper multiple comparisons corrections\n")