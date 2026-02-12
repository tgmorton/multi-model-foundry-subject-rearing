# Within-Model, Within-Item-Group Post-hoc Analysis
# ==================================================

library(tidyverse)
library(lme4)
library(emmeans)
library(knitr)

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

cat("\nFitting models for within-model item group comparisons...\n")

# Store results
all_results <- list()
all_pairwise <- list()

# Analyze each model separately
for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  cat(sprintf("\nAnalyzing %s...\n", model_label))
  
  # Subset data for this model
  model_data <- endstate_data %>%
    filter(model == model_name)
  
  # Fit the model with item_group as fixed effect
  tryCatch({
    # Simpler model for stability
    m <- glmer(correct ~ item_group * form_type + (1|item_id),
               data = model_data,
               family = binomial,
               control = glmerControl(
                 optimizer = "bobyqa",
                 optCtrl = list(maxfun = 100000)
               ))
    
    # Get estimated marginal means for null subjects
    emm <- emmeans(m, ~ item_group, 
                   at = list(form_type = "null"),
                   type = "response")
    
    # Get pairwise comparisons
    pairs <- pairs(emm, adjust = "fdr")
    
    # Convert to data frames
    emm_df <- as.data.frame(emm) %>%
      mutate(
        model = model_name,
        Model = model_label,
        .before = 1
      )
    
    pairs_df <- as.data.frame(pairs) %>%
      mutate(
        model = model_name,
        Model = model_label,
        .before = 1
      )
    
    # Store results
    all_results[[model_name]] <- emm_df
    all_pairwise[[model_name]] <- pairs_df
    
    # Print summary
    cat("  Item group null subject preferences:\n")
    print(emm_df %>% 
            select(item_group, prob, SE, asymp.LCL, asymp.UCL) %>%
            mutate(prob = round(prob, 3),
                   SE = round(SE, 4),
                   CI = sprintf("[%.3f, %.3f]", asymp.LCL, asymp.UCL)) %>%
            select(item_group, prob, SE, CI))
    
    # Print significant pairwise differences
    sig_pairs <- pairs_df %>%
      filter(p.value < 0.05) %>%
      arrange(p.value)
    
    if (nrow(sig_pairs) > 0) {
      cat("\n  Significant pairwise differences (FDR-adjusted p < 0.05):\n")
      print(sig_pairs %>%
              select(contrast, odds.ratio, p.value) %>%
              mutate(odds.ratio = round(odds.ratio, 3),
                     p.value = format.pval(p.value, digits = 3)),
            n = 10)
    } else {
      cat("\n  No significant pairwise differences found.\n")
    }
    
  }, error = function(e) {
    cat(sprintf("  Error fitting model: %s\n", e$message))
  })
}

# Combine all results
if (length(all_results) > 0) {
  combined_emmeans <- bind_rows(all_results)
  combined_pairwise <- bind_rows(all_pairwise)
  
  # Save results
  write_csv(combined_emmeans, "analysis/tables/within_model_itemgroup_emmeans.csv")
  write_csv(combined_pairwise, "analysis/tables/within_model_itemgroup_pairwise.csv")
  
  # Create summary table of key contrasts
  cat("\n\n========================================\n")
  cat("SUMMARY: Key Item Group Contrasts\n")
  cat("========================================\n\n")
  
  # Focus on theoretically important contrasts
  key_contrasts <- c(
    "1a_3rdSG - 1b_3rdPL",  # 3rd person singular vs plural
    "2a_2ndSG - 2b_2ndPL",  # 2nd person singular vs plural  
    "3a_1stSg - 3b_1stPL",  # 1st person singular vs plural
    "1a_3rdSG - 3a_1stSg",  # 3rd vs 1st person
    "4a_subject_control - 4b_object_control",  # Subject vs object control
    "5a_expletive_seems - 5b_expletive_be",  # Expletive types
    "7a_conjunction_no_topic_shift - 7b_conjunction_topic_shift"  # Topic shift
  )
  
  summary_table <- combined_pairwise %>%
    filter(contrast %in% key_contrasts) %>%
    select(Model, contrast, odds.ratio, SE, p.value) %>%
    mutate(
      odds.ratio = round(odds.ratio, 3),
      SE = round(SE, 4),
      p.value = format.pval(p.value, digits = 3),
      sig = case_when(
        p.value < 0.001 ~ "***",
        p.value < 0.01 ~ "**",
        p.value < 0.05 ~ "*",
        TRUE ~ ""
      )
    ) %>%
    pivot_wider(names_from = Model, 
                values_from = c(odds.ratio, p.value, sig),
                names_glue = "{Model}_{.value}")
  
  write_csv(summary_table, "analysis/tables/within_model_key_contrasts.csv")
  
  # Print model-by-model summary
  for (model_label in unique(combined_emmeans$Model)) {
    cat(sprintf("\n%s:\n", model_label))
    cat("---------------------\n")
    
    # Get top 3 highest and lowest item groups
    model_emm <- combined_emmeans %>%
      filter(Model == model_label) %>%
      arrange(desc(prob))
    
    cat("  Highest null preference:\n")
    print(model_emm %>%
            head(3) %>%
            select(item_group, prob) %>%
            mutate(prob = sprintf("%.1f%%", prob * 100)),
          row.names = FALSE)
    
    cat("  Lowest null preference:\n")
    print(model_emm %>%
            tail(3) %>%
            select(item_group, prob) %>%
            mutate(prob = sprintf("%.1f%%", prob * 100)),
          row.names = FALSE)
  }
  
  cat("\n\nAnalysis complete!\n")
  cat("Results saved to:\n")
  cat("  - analysis/tables/within_model_itemgroup_emmeans.csv\n")
  cat("  - analysis/tables/within_model_itemgroup_pairwise.csv\n")
  cat("  - analysis/tables/within_model_key_contrasts.csv\n")
  
} else {
  cat("\nNo models successfully analyzed.\n")
}