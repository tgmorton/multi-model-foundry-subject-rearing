# Chance-Level LaTeX Table Generator
# =================================
# Generate LaTeX tables comparing each item group and form to 50% chance
# within each model (not pairwise comparisons)

library(tidyverse)
library(xtable)

# Create directories
dir.create("analysis/tables/latex_tables", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/tables/latex_tables/itemgroups", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/tables/latex_tables/forms", recursive = TRUE, showWarnings = FALSE)

# Load data
dat <- read_csv("evaluation/results/all_models_null_subject_lme4_ready.csv", show_col_types = FALSE)

# Model labels
model_labels <- tribble(
  ~model, ~Model, ~model_clean_name,
  "exp0_baseline", "Baseline", "exp0baseline",
  "exp1_remove_expletives", "Remove Expletives", "exp1removeexpletives",
  "exp2_impoverish_determiners", "Impoverish Determiners", "exp2impoverishdeterminers",
  "exp3_remove_articles", "Remove Articles", "exp3removearticles",
  "exp4_lemmatize_verbs", "Lemmatize Verbs", "exp4lemmatizeverbs",
  "exp5_remove_subject_pronominals", "Remove Subject Pronominals", "exp5removesubjectpronominals"
)

dat <- dat %>%
  left_join(model_labels, by = "model")

# Get end-state data (last 1000 checkpoints)
endstate_data <- dat %>%
  group_by(model) %>%
  filter(checkpoint_num >= max(checkpoint_num) - 1000) %>%
  ungroup() %>%
  filter(form_type == "overt")

# Function to perform binomial test against 50%
test_against_chance <- function(n_success, n_total) {
  if (n_total == 0) return(list(p_value = NA, estimate = NA, conf_low = NA, conf_high = NA))
  
  test_result <- binom.test(n_success, n_total, p = 0.5)
  list(
    p_value = test_result$p.value,
    estimate = test_result$estimate,
    conf_low = test_result$conf.int[1],
    conf_high = test_result$conf.int[2]
  )
}

# ITEM GROUP TABLES
cat("Generating item group LaTeX tables...\n")

# Create item group labels
endstate_data <- endstate_data %>%
  mutate(
    item_group_label = case_when(
      str_detect(item_group, "^1a_") ~ "3rd Singular",
      str_detect(item_group, "^1b_") ~ "3rd Plural", 
      str_detect(item_group, "^2a_") ~ "2nd Singular",
      str_detect(item_group, "^2b_") ~ "2nd Plural",
      str_detect(item_group, "^3a_") ~ "1st Singular",
      str_detect(item_group, "^3b_") ~ "1st Plural",
      str_detect(item_group, "^4a_") ~ "Subject Control",
      str_detect(item_group, "^4b_") ~ "Object Control",
      str_detect(item_group, "^5a_") ~ "Seems Expletive",
      str_detect(item_group, "^5b_") ~ "Be Expletive",
      str_detect(item_group, "^6_") ~ "Long Distance Binding",
      str_detect(item_group, "^7a_") ~ "No Topic Shift",
      str_detect(item_group, "^7b_") ~ "Topic Shift",
      TRUE ~ item_group
    )
  )

# Generate item group chance-level comparisons for each model
for (model_name in unique(endstate_data$model)) {
  model_display <- model_labels$Model[model_labels$model == model_name]
  model_clean <- model_labels$model_clean_name[model_labels$model == model_name]
  
  # Calculate item group performance vs chance
  itemgroup_results <- endstate_data %>%
    filter(model == model_name) %>%
    group_by(item_group_label) %>%
    summarise(
      n_correct = sum(correct),
      n_total = n(),
      pct_correct = n_correct / n_total * 100,
      .groups = "drop"
    ) %>%
    filter(!is.na(item_group_label), item_group_label != "Long Distance Binding") %>%
    rowwise() %>%
    mutate(
      test_result = list(test_against_chance(n_correct, n_total)),
      p_value = test_result$p_value,
      conf_low = test_result$conf_low * 100,
      conf_high = test_result$conf_high * 100
    ) %>%
    select(-test_result) %>%
    ungroup() %>%
    mutate(
      p_formatted = case_when(
        p_value < 0.001 ~ "< .001",
        p_value < 0.01 ~ sprintf("%.3f", p_value),
        TRUE ~ sprintf("%.3f", p_value)
      ),
      ci_formatted = sprintf("[%.1f, %.1f]", conf_low, conf_high),
      significance = ifelse(p_value < 0.05, "*", ""),
      direction = case_when(
        pct_correct > 50 & p_value < 0.05 ~ "Above",
        pct_correct < 50 & p_value < 0.05 ~ "Below", 
        TRUE ~ "At"
      )
    ) %>%
    # Order groups properly
    mutate(
      sort_order = case_when(
        item_group_label == "1st Singular" ~ 1,
        item_group_label == "1st Plural" ~ 2,
        item_group_label == "2nd Singular" ~ 3,
        item_group_label == "2nd Plural" ~ 4,
        item_group_label == "3rd Singular" ~ 5,
        item_group_label == "3rd Plural" ~ 6,
        item_group_label == "Subject Control" ~ 7,
        item_group_label == "Object Control" ~ 8,
        item_group_label == "Seems Expletive" ~ 9,
        item_group_label == "Be Expletive" ~ 10,
        item_group_label == "No Topic Shift" ~ 11,
        item_group_label == "Topic Shift" ~ 12,
        TRUE ~ 99
      )
    ) %>%
    arrange(sort_order) %>%
    select(item_group_label, pct_correct, ci_formatted, direction, p_formatted)
  
  # Create LaTeX table
  itemgroup_table <- itemgroup_results %>%
    transmute(
      `Item Group` = item_group_label,
      `Accuracy` = sprintf("%.1f%%", pct_correct),
      `95% CI` = ci_formatted,
      `vs Chance` = direction,
      `p-value` = p_formatted
    )
  
  # Generate LaTeX
  latex_table <- xtable(itemgroup_table, 
                       caption = sprintf("Item Group Performance vs 50\\%% Chance - %s", model_display),
                       label = sprintf("tab:%s_itemgroups", model_clean))
  
  # Write to file
  output_file <- sprintf("analysis/tables/latex_tables/itemgroups/%s_itemgroups.tex", model_clean)
  print(latex_table, 
        file = output_file,
        include.rownames = FALSE,
        tabular.environment = "tabular",
        floating = FALSE,
        hline.after = c(-1, 0, nrow(itemgroup_table)))
  
  cat(sprintf("  Generated: %s\n", output_file))
}

# FORM TABLES  
cat("\nGenerating form LaTeX tables...\n")

# Generate form chance-level comparisons for each model
for (model_name in unique(endstate_data$model)) {
  model_display <- model_labels$Model[model_labels$model == model_name]
  model_clean <- model_labels$model_clean_name[model_labels$model == model_name]
  
  # Calculate form performance vs chance
  form_results <- endstate_data %>%
    filter(model == model_name) %>%
    group_by(form) %>%
    summarise(
      n_correct = sum(correct),
      n_total = n(),
      pct_correct = n_correct / n_total * 100,
      .groups = "drop"
    ) %>%
    rowwise() %>%
    mutate(
      test_result = list(test_against_chance(n_correct, n_total)),
      p_value = test_result$p_value,
      conf_low = test_result$conf_low * 100,
      conf_high = test_result$conf_high * 100
    ) %>%
    select(-test_result) %>%
    ungroup() %>%
    mutate(
      p_formatted = case_when(
        p_value < 0.001 ~ "< .001",
        p_value < 0.01 ~ sprintf("%.3f", p_value),
        TRUE ~ sprintf("%.3f", p_value)
      ),
      ci_formatted = sprintf("[%.1f, %.1f]", conf_low, conf_high),
      significance = ifelse(p_value < 0.05, "*", ""),
      direction = case_when(
        pct_correct > 50 & p_value < 0.05 ~ "Above",
        pct_correct < 50 & p_value < 0.05 ~ "Below",
        TRUE ~ "At"
      ),
      form_clean = str_replace_all(form, "_", " ") %>% str_to_title(),
      form_order = case_when(
        form == "default" ~ 1,
        form == "complex_long" ~ 2,
        form == "complex_emb" ~ 3,
        form == "context_negation" ~ 4,
        form == "target_negation" ~ 5,
        form == "both_negation" ~ 6,
        TRUE ~ 99
      )
    ) %>%
    arrange(form_order) %>%
    select(form_clean, pct_correct, ci_formatted, direction, p_formatted)
  
  # Create LaTeX table
  form_table <- form_results %>%
    transmute(
      `Form` = form_clean,
      `Accuracy` = sprintf("%.1f%%", pct_correct),
      `95% CI` = ci_formatted,
      `vs Chance` = direction,
      `p-value` = p_formatted
    )
  
  # Generate LaTeX
  latex_table <- xtable(form_table,
                       caption = sprintf("Form Performance vs 50\\%% Chance - %s", model_display),
                       label = sprintf("tab:%s_forms", model_clean))
  
  # Write to file
  output_file <- sprintf("analysis/tables/latex_tables/forms/%s_forms.tex", model_clean)
  print(latex_table,
        file = output_file,
        include.rownames = FALSE,
        tabular.environment = "tabular",
        floating = FALSE,
        hline.after = c(-1, 0, nrow(form_table)))
  
  cat(sprintf("  Generated: %s\n", output_file))
}

cat("\n=== LATEX TABLE GENERATION COMPLETE ===\n")
cat("Generated chance-level comparison tables for all models\n")
cat("- Item group tables: analysis/tables/latex_tables/itemgroups/\n")
cat("- Form tables: analysis/tables/latex_tables/forms/\n")