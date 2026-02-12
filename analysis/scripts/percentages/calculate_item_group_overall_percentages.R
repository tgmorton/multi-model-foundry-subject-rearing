# Calculate Overall Item Group Percentages (Consistent Across All Comparisons)
# ===========================================================================

library(tidyverse)

cat("=== CALCULATING OVERALL ITEM GROUP PERCENTAGES ===\n\n")

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

# Calculate overall percentages for each item group
overall_percentages <- endstate_data %>%
  group_by(model, Model, item_group) %>%
  summarise(
    n_correct = sum(correct),
    n_total = n(),
    overt_pref = n_correct / n_total,
    se = sqrt(overt_pref * (1 - overt_pref) / n_total),
    ci_low = overt_pref - 1.96 * se,
    ci_high = overt_pref + 1.96 * se,
    .groups = "drop"
  ) %>%
  # Add readable group labels
  mutate(
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
    # Also create person and number categories for aggregation
    person = case_when(
      str_detect(item_group, "^1[ab]_") ~ "3rd_person",
      str_detect(item_group, "^2[ab]_") ~ "2nd_person",
      str_detect(item_group, "^3[ab]_") ~ "1st_person",
      TRUE ~ NA_character_
    ),
    number = case_when(
      str_detect(item_group, "^[123]a_") ~ "singular",
      str_detect(item_group, "^[123]b_") ~ "plural",
      TRUE ~ NA_character_
    )
  )

# Save overall percentages
write_csv(overall_percentages, "analysis/tables/item_group_overall_percentages.csv")

# Display summary for each model
for (model_name in unique(overall_percentages$model)) {
  cat(sprintf("\n=== %s ===\n", model_name))
  
  model_data <- overall_percentages %>%
    filter(model == model_name) %>%
    arrange(item_group)
  
  # Print person/number groups
  cat("\nPerson/Number Groups:\n")
  person_number <- model_data %>%
    filter(!is.na(person)) %>%
    select(group_label, overt_pref, ci_low, ci_high) %>%
    mutate(
      display = sprintf("%-15s: %.1f%% [%.1f%%, %.1f%%]", 
                       group_label, overt_pref * 100, ci_low * 100, ci_high * 100)
    )
  
  cat(paste(person_number$display, collapse = "\n"))
  
  # Print other groups
  cat("\n\nOther Groups:\n")
  other_groups <- model_data %>%
    filter(is.na(person)) %>%
    select(group_label, overt_pref, ci_low, ci_high) %>%
    mutate(
      display = sprintf("%-15s: %.1f%% [%.1f%%, %.1f%%]", 
                       group_label, overt_pref * 100, ci_low * 100, ci_high * 100)
    )
  
  cat(paste(other_groups$display, collapse = "\n"))
  cat("\n")
}

# Also calculate aggregated person and number percentages
cat("\n\n=== AGGREGATED PERSON PERCENTAGES ===\n")
person_aggregated <- endstate_data %>%
  mutate(
    person = case_when(
      str_detect(item_group, "^1[ab]_") ~ "3rd_person",
      str_detect(item_group, "^2[ab]_") ~ "2nd_person",
      str_detect(item_group, "^3[ab]_") ~ "1st_person",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(person)) %>%
  group_by(model, Model, person) %>%
  summarise(
    n_correct = sum(correct),
    n_total = n(),
    overt_pref = n_correct / n_total,
    .groups = "drop"
  )

for (model_name in unique(person_aggregated$model)) {
  cat(sprintf("\n%s:\n", model_name))
  model_person <- person_aggregated %>%
    filter(model == model_name) %>%
    mutate(display = sprintf("  %-12s: %.1f%%", person, overt_pref * 100))
  cat(paste(model_person$display, collapse = "\n"))
}

cat("\n\n=== ANALYSIS COMPLETE ===\n")
cat("Results saved to: analysis/tables/item_group_overall_percentages.csv\n")