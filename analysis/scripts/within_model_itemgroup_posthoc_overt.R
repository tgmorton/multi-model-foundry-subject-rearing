# Within-Model, Within-Item-Group Post-hoc Analysis - OVERT PREFERENCE
# ====================================================================

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

cat("\nFitting models for within-model item group comparisons (OVERT preference)...\n")

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
    
    # Get estimated marginal means for OVERT subjects (key change)
    emm <- emmeans(m, ~ item_group, 
                   at = list(form_type = "overt"),
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
    cat("  Item group OVERT subject preferences:\n")
    summary_table <- emm_df %>% 
      select(item_group, prob, SE, asymp.LCL, asymp.UCL) %>%
      mutate(prob = round(prob, 3),
             SE = round(SE, 4),
             CI = sprintf("[%.3f, %.3f]", asymp.LCL, asymp.UCL)) %>%
      select(item_group, prob, SE, CI)
    print(summary_table)
    
    # Print significant pairwise differences
    sig_pairs <- pairs_df %>%
      filter(p.value < 0.05) %>%
      arrange(p.value)
    
    if (nrow(sig_pairs) > 0) {
      cat("\n  Significant pairwise differences (FDR-adjusted p < 0.05):\n")
      pairwise_table <- sig_pairs %>%
        select(contrast, odds.ratio, p.value) %>%
        mutate(odds.ratio = round(odds.ratio, 3),
               p.value = format.pval(p.value, digits = 3)) %>%
        head(10)
      print(pairwise_table)
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
  write_csv(combined_emmeans, "analysis/tables/within_model_itemgroup_emmeans_overt.csv")
  write_csv(combined_pairwise, "analysis/tables/within_model_itemgroup_pairwise_overt.csv")
  
  cat("\n\n========================================\n")
  cat("SUMMARY: OVERT Subject Preferences\n")
  cat("========================================\n\n")
  
  # Print model-by-model summary
  for (model_label in unique(combined_emmeans$Model)) {
    cat(sprintf("\n%s:\n", model_label))
    cat("---------------------\n")
    
    # Get top 3 highest and lowest item groups for OVERT preference
    model_emm <- combined_emmeans %>%
      filter(Model == model_label) %>%
      arrange(desc(prob))
    
    cat("  Highest OVERT preference:\n")
    print(model_emm %>%
            head(3) %>%
            select(item_group, prob) %>%
            mutate(prob = sprintf("%.1f%%", prob * 100)),
          row.names = FALSE)
    
    cat("  Lowest OVERT preference:\n")
    print(model_emm %>%
            tail(3) %>%
            select(item_group, prob) %>%
            mutate(prob = sprintf("%.1f%%", prob * 100)),
          row.names = FALSE)
    
    # Show full ranking
    cat("\n  Full item group ranking (by OVERT preference):\n")
    ranking <- model_emm %>%
      mutate(
        rank = row_number(),
        pct = sprintf("%.1f%%", prob * 100)
      ) %>%
      select(Rank = rank, `Item Group` = item_group, `Overt Pref` = pct)
    
    print(ranking, row.names = FALSE)
  }
  
  # Key contrasts comparison
  cat("\n\n========================================\n")
  cat("KEY CONTRASTS FOR OVERT PREFERENCE\n")
  cat("========================================\n\n")
  
  # Focus on theoretically important contrasts (same as before but now for overt)
  for (model_label in unique(combined_emmeans$Model)) {
    cat(sprintf("\n%s - Key Theoretical Contrasts:\n", model_label))
    cat("---------------------------------------\n")
    
    model_pairs <- combined_pairwise %>%
      filter(Model == model_label, p.value < 0.05) %>%
      filter(
        grepl("3rdSG.*3rdPL|3rdPL.*3rdSG", contrast) |  # Person number
        grepl("2ndSG.*2ndPL|2ndPL.*2ndSG", contrast) |
        grepl("1stSg.*1stPL|1stPL.*1stSg", contrast) |
        grepl("subject_control.*object_control|object_control.*subject_control", contrast) |  # Control
        grepl("expletive_seems.*expletive_be|expletive_be.*expletive_seems", contrast) |  # Expletives
        grepl("no_topic_shift.*topic_shift|topic_shift.*no_topic_shift", contrast)  # Topic shift
      ) %>%
      mutate(
        direction = ifelse(odds.ratio > 1, ">", "<"),
        interpretation = sprintf("%s %s %s", 
                                 sub(" / .*", "", contrast),
                                 direction,
                                 sub(".* / ", "", contrast))
      ) %>%
      select(interpretation, odds.ratio, p.value) %>%
      mutate(
        OR = sprintf("%.2f", odds.ratio),
        p = format.pval(p.value, digits = 3),
        sig = case_when(
          p.value < 0.001 ~ "***",
          p.value < 0.01 ~ "**",
          p.value < 0.05 ~ "*",
          TRUE ~ ""
        )
      ) %>%
      select(Comparison = interpretation, OR, p, sig)
    
    if (nrow(model_pairs) > 0) {
      print(model_pairs, row.names = FALSE)
    } else {
      cat("  No significant theoretical contrasts\n")
    }
  }
  
  cat("\n\nAnalysis complete!\n")
  cat("Results saved to:\n")
  cat("  - analysis/tables/within_model_itemgroup_emmeans_overt.csv\n")
  cat("  - analysis/tables/within_model_itemgroup_pairwise_overt.csv\n")
  
} else {
  cat("\nNo models successfully analyzed.\n")
}