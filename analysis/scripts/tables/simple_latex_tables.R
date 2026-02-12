# Simple LaTeX Table Generator - Overt vs Chance Tables
# ======================================================

library(tidyverse)
library(xtable)

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

# Function to create proper item group ordering
order_item_groups <- function(item_groups) {
  item_groups %>%
    mutate(
      sort_num = case_when(
        str_detect(item_group, "^1[ab]_") ~ 1,
        str_detect(item_group, "^2[ab]_") ~ 2, 
        str_detect(item_group, "^3[ab]_") ~ 3,
        str_detect(item_group, "^4[ab]_") ~ 4,
        str_detect(item_group, "^5[ab]_") ~ 5,
        str_detect(item_group, "^6_") ~ 6,
        str_detect(item_group, "^7[ab]_") ~ 7,
        TRUE ~ 99
      ),
      sort_letter = case_when(
        str_detect(item_group, "a_") ~ "a",
        str_detect(item_group, "b_") ~ "b", 
        TRUE ~ "z"
      )
    ) %>%
    arrange(sort_num, sort_letter) %>%
    select(-sort_num, -sort_letter)
}

# Function to format p-values for display
format_pvalue <- function(p) {
  case_when(
    p < 0.001 ~ "< 0.001",
    p < 0.01 ~ sprintf("%.3f", p),
    TRUE ~ sprintf("%.4f", p)
  )
}

# Generate tables for each model
for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  model_data <- endstate_data %>% filter(model == model_name)
  
  # ===================
  # ITEM GROUPS TABLE
  # ===================
  
  itemgroup_results <- model_data %>%
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
    order_item_groups()
  
  # Format for LaTeX
  latex_itemgroups <- itemgroup_results %>%
    mutate(
      item_group = str_replace_all(item_group, "_", "\\\\\\\\_"),
      overt_pref = sprintf("%.3f", overt_pref),
      p_value = ifelse(p_value < 0.001, "p < .001", sprintf("p = %.3f", p_value)),
      CI = sprintf("[%.3f, %.3f]", ci_low, ci_high)
    ) %>%
    select(item_group, overt_pref, CI, p_value)
  
  # Generate LaTeX table
  cat(sprintf("\n%% %s - Item Groups\n", model_label))
  xtab <- xtable(latex_itemgroups, 
                 caption = sprintf("%s: Overt Subject Preference vs. Chance (0.5) by Item Group", model_label),
                 label = sprintf("tab:%s_itemgroups", str_replace_all(tolower(model_name), "_", "")),
                 align = c("l", "l", "r", "l", "l"))
  
  print(xtab, 
        include.rownames = FALSE,
        caption.placement = "top",
        table.placement = "H")
  
  cat("\n")
  
  # =================
  # FORMS TABLE
  # =================
  
  form_results <- model_data %>%
    group_by(form) %>%
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
    # Order forms logically
    mutate(form = factor(form, levels = c("default", "context_negation", "target_negation", 
                                         "both_negation", "complex_long", "complex_emb"))) %>%
    arrange(form) %>%
    mutate(form = as.character(form))
  
  # Format for LaTeX
  latex_forms <- form_results %>%
    mutate(
      form = str_replace_all(form, "_", "\\\\\\\\_"),
      overt_pref = sprintf("%.3f", overt_pref),
      p_value = ifelse(p_value < 0.001, "p < .001", sprintf("p = %.3f", p_value)),
      CI = sprintf("[%.3f, %.3f]", ci_low, ci_high)
    ) %>%
    select(form, overt_pref, CI, p_value)
  
  # Generate LaTeX table
  cat(sprintf("\n%% %s - Forms\n", model_label))
  xtab <- xtable(latex_forms,
                 caption = sprintf("%s: Overt Subject Preference vs. Chance (0.5) by Linguistic Form", model_label),
                 label = sprintf("tab:%s_forms", str_replace_all(tolower(model_name), "_", "")),
                 align = c("l", "l", "r", "l", "l"))
  
  print(xtab,
        include.rownames = FALSE,
        caption.placement = "top", 
        table.placement = "H")
  
  cat("\n\\clearpage\n\n")
}

cat("% LaTeX tables generated successfully!\n")