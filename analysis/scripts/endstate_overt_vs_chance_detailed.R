# End-State Overt Preference vs. Chance - Detailed Analysis
# Within Models and Within Models × Item Groups
# ==========================================================

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

cat("\n=======================================================\n")
cat("DETAILED OVERT PREFERENCE VS. CHANCE ANALYSIS\n")
cat("=======================================================\n")

# Store all results
all_model_results <- list()
all_itemgroup_results <- list()
all_form_results <- list()

# Analyze each model
for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  cat(sprintf("\n\n%s\n", model_label))
  cat(paste(rep("=", nchar(model_label)), collapse=""), "\n\n")
  
  model_data <- endstate_data %>%
    filter(model == model_name)
  
  # =======================
  # 1. OVERALL MODEL ANALYSIS
  # =======================
  
  overt_overall <- model_data %>%
    filter(form_type == "overt") %>%
    summarise(
      n_correct = sum(correct),
      n_total = n(),
      overt_pref = n_correct / n_total,
      se = sqrt(overt_pref * (1 - overt_pref) / n_total)
    )
  
  binom_overall <- binom.test(
    overt_overall$n_correct, 
    overt_overall$n_total, 
    p = 0.5,
    alternative = "two.sided"
  )
  
  cat("OVERALL MODEL PERFORMANCE:\n")
  cat(sprintf("  Overt preference: %.3f (SE = %.4f)\n", 
              overt_overall$overt_pref, overt_overall$se))
  cat(sprintf("  95%% CI: [%.3f, %.3f]\n", 
              binom_overall$conf.int[1], binom_overall$conf.int[2]))
  cat(sprintf("  vs. chance: %s %s\n", 
              ifelse(binom_overall$p.value < 0.001, "p < 0.001",
                     sprintf("p = %.3f", binom_overall$p.value)),
              ifelse(binom_overall$p.value < 0.05, "***", "")))
  
  # Store overall result
  all_model_results[[model_name]] <- tibble(
    model = model_name,
    Model = model_label,
    overt_pref = overt_overall$overt_pref,
    se = overt_overall$se,
    ci_low = binom_overall$conf.int[1],
    ci_high = binom_overall$conf.int[2],
    p_value = binom_overall$p.value,
    significant = binom_overall$p.value < 0.05
  )
  
  # =======================
  # 2. BY LINGUISTIC FORM
  # =======================
  
  cat("\nBY LINGUISTIC FORM:\n")
  
  form_results <- model_data %>%
    filter(form_type == "overt") %>%
    group_by(form) %>%
    summarise(
      n_correct = sum(correct),
      n_total = n(),
      overt_pref = n_correct / n_total,
      .groups = "drop"
    ) %>%
    mutate(
      p_value = map2_dbl(n_correct, n_total, 
                         ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$p.value),
      sig = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        TRUE ~ ""
      ),
      direction = case_when(
        overt_pref > 0.5 ~ "above",
        overt_pref < 0.5 ~ "below",
        TRUE ~ "at"
      )
    ) %>%
    arrange(desc(overt_pref))
  
  # Print form results
  for (i in 1:nrow(form_results)) {
    cat(sprintf("  %s: %.1f%% %s chance %s\n",
                str_pad(form_results$form[i], 20),
                form_results$overt_pref[i] * 100,
                form_results$direction[i],
                form_results$sig[i]))
  }
  
  # Store form results
  form_results <- form_results %>%
    mutate(model = model_name, Model = model_label, .before = 1)
  all_form_results[[paste0(model_name, "_form")]] <- form_results
  
  # =======================
  # 3. BY ITEM GROUP
  # =======================
  
  cat("\nBY ITEM GROUP:\n")
  
  itemgroup_results <- model_data %>%
    filter(form_type == "overt") %>%
    group_by(item_group) %>%
    summarise(
      n_correct = sum(correct),
      n_total = n(),
      overt_pref = n_correct / n_total,
      se = sqrt(overt_pref * (1 - overt_pref) / n_total),
      .groups = "drop"
    ) %>%
    mutate(
      p_value = map2_dbl(n_correct, n_total, 
                         ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$p.value),
      ci_low = map2_dbl(n_correct, n_total, 
                        ~binom.test(.x, .y, p = 0.5)$conf.int[1]),
      ci_high = map2_dbl(n_correct, n_total, 
                         ~binom.test(.x, .y, p = 0.5)$conf.int[2]),
      sig = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        TRUE ~ ""
      ),
      direction = case_when(
        overt_pref > 0.5 ~ "above",
        overt_pref < 0.5 ~ "below",
        TRUE ~ "at"
      )
    ) %>%
    arrange(desc(overt_pref))
  
  # Print top and bottom performers
  cat("  Strongest overt preference:\n")
  itemgroup_results %>%
    head(3) %>%
    mutate(
      display = sprintf("    %s: %.1f%% [%.1f, %.1f] %s",
                       str_pad(item_group, 30),
                       overt_pref * 100,
                       ci_low * 100,
                       ci_high * 100,
                       sig)
    ) %>%
    pull(display) %>%
    cat(sep = "\n")
  
  cat("\n  Weakest overt preference:\n")
  itemgroup_results %>%
    tail(3) %>%
    arrange(overt_pref) %>%
    mutate(
      display = sprintf("    %s: %.1f%% [%.1f, %.1f] %s",
                       str_pad(item_group, 30),
                       overt_pref * 100,
                       ci_low * 100,
                       ci_high * 100,
                       sig)
    ) %>%
    pull(display) %>%
    cat(sep = "\n")
  
  # Count significant effects
  n_sig_above <- sum(itemgroup_results$p_value < 0.05 & itemgroup_results$overt_pref > 0.5)
  n_sig_below <- sum(itemgroup_results$p_value < 0.05 & itemgroup_results$overt_pref < 0.5)
  n_ns <- sum(itemgroup_results$p_value >= 0.05)
  
  cat(sprintf("\n  Summary: %d above chance, %d below chance, %d non-significant\n",
              n_sig_above, n_sig_below, n_ns))
  
  # Store item group results
  itemgroup_results <- itemgroup_results %>%
    mutate(model = model_name, Model = model_label, .before = 1)
  all_itemgroup_results[[model_name]] <- itemgroup_results
  
  # =======================
  # 4. ITEM GROUP × FORM INTERACTION
  # =======================
  
  cat("\nITEM GROUP × FORM INTERACTIONS:\n")
  
  # For each item group, test forms
  for (ig in unique(model_data$item_group)) {
    ig_data <- model_data %>%
      filter(item_group == ig, form_type == "overt")
    
    # Get form-specific results for this item group
    form_by_ig <- ig_data %>%
      group_by(form) %>%
      summarise(
        n_correct = sum(correct),
        n_total = n(),
        overt_pref = n_correct / n_total,
        .groups = "drop"
      ) %>%
      mutate(
        p_value = map2_dbl(n_correct, n_total, 
                           ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$p.value),
        sig = p_value < 0.05
      )
    
    # Check if there's variation across forms
    if (sd(form_by_ig$overt_pref) > 0.1) {  # Substantial variation
      cat(sprintf("\n  %s (high variation across forms):\n", ig))
      
      # Show extremes
      best_form <- form_by_ig %>% filter(overt_pref == max(overt_pref))
      worst_form <- form_by_ig %>% filter(overt_pref == min(overt_pref))
      
      cat(sprintf("    Highest: %s = %.1f%% %s\n",
                  best_form$form[1],
                  best_form$overt_pref[1] * 100,
                  ifelse(best_form$sig[1], "*", "")))
      cat(sprintf("    Lowest:  %s = %.1f%% %s\n",
                  worst_form$form[1],
                  worst_form$overt_pref[1] * 100,
                  ifelse(worst_form$sig[1], "*", "")))
      cat(sprintf("    Range: %.1f%% points\n",
                  (max(form_by_ig$overt_pref) - min(form_by_ig$overt_pref)) * 100))
    }
  }
  
  cat("\n")
}

# =======================
# SAVE ALL RESULTS
# =======================

# Combine and save
all_models_df <- bind_rows(all_model_results)
all_itemgroups_df <- bind_rows(all_itemgroup_results)
all_forms_df <- bind_rows(all_form_results)

write_csv(all_models_df, "analysis/tables/endstate_overt_vs_chance_models.csv")
write_csv(all_itemgroups_df, "analysis/tables/endstate_overt_vs_chance_itemgroups.csv")
write_csv(all_forms_df, "analysis/tables/endstate_overt_vs_chance_forms.csv")

# =======================
# CROSS-MODEL SUMMARY
# =======================

cat("\n\n=======================================================\n")
cat("CROSS-MODEL SUMMARY\n")
cat("=======================================================\n\n")

# Overall summary
summary_table <- all_models_df %>%
  mutate(
    pref_pct = sprintf("%.1f%%", overt_pref * 100),
    ci = sprintf("[%.1f, %.1f]", ci_low * 100, ci_high * 100),
    p_display = ifelse(p_value < 0.001, "p < .001", sprintf("p = %.3f", p_value)),
    sig = ifelse(significant, "***", "")
  ) %>%
  select(Model, `Overt %` = pref_pct, `95% CI` = ci, p_display, sig) %>%
  arrange(desc(as.numeric(str_extract(`Overt %`, "[0-9.]+"))))

cat("Overall Model Performance vs. Chance:\n\n")
print(summary_table, n = Inf)

# Item group summary across models
cat("\n\nItem Groups Consistently Above/Below Chance:\n")

ig_consistency <- all_itemgroups_df %>%
  group_by(item_group) %>%
  summarise(
    mean_pref = mean(overt_pref),
    n_sig_above = sum(p_value < 0.05 & overt_pref > 0.5),
    n_sig_below = sum(p_value < 0.05 & overt_pref < 0.5),
    n_models = n(),
    consistency = case_when(
      n_sig_above == n_models ~ "Always above",
      n_sig_below == n_models ~ "Always below",
      n_sig_above > n_models/2 ~ "Usually above",
      n_sig_below > n_models/2 ~ "Usually below",
      TRUE ~ "Mixed"
    ),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_pref))

cat("\nConsistently ABOVE chance (overt-preferring):\n")
ig_consistency %>%
  filter(consistency %in% c("Always above", "Usually above")) %>%
  mutate(display = sprintf("  %s: %.1f%% mean (above in %d/%d models)",
                          item_group, mean_pref * 100, n_sig_above, n_models)) %>%
  pull(display) %>%
  cat(sep = "\n")

cat("\n\nConsistently BELOW chance (null-preferring):\n")
ig_consistency %>%
  filter(consistency %in% c("Always below", "Usually below")) %>%
  mutate(display = sprintf("  %s: %.1f%% mean (below in %d/%d models)",
                          item_group, mean_pref * 100, n_sig_below, n_models)) %>%
  pull(display) %>%
  cat(sep = "\n")

cat("\n\nMixed or near-chance:\n")
ig_consistency %>%
  filter(consistency == "Mixed") %>%
  mutate(display = sprintf("  %s: %.1f%% mean (above: %d, below: %d, ns: %d)",
                          item_group, mean_pref * 100, 
                          n_sig_above, n_sig_below, 
                          n_models - n_sig_above - n_sig_below)) %>%
  pull(display) %>%
  cat(sep = "\n")

write_csv(ig_consistency, "analysis/tables/endstate_overt_itemgroup_consistency.csv")

cat("\n\n*** p < 0.001, ** p < 0.01, * p < 0.05\n")
cat("\nAnalysis complete!\n")
cat("Results saved to:\n")
cat("  - analysis/tables/endstate_overt_vs_chance_models.csv\n")
cat("  - analysis/tables/endstate_overt_vs_chance_itemgroups.csv\n")
cat("  - analysis/tables/endstate_overt_vs_chance_forms.csv\n")
cat("  - analysis/tables/endstate_overt_itemgroup_consistency.csv\n")