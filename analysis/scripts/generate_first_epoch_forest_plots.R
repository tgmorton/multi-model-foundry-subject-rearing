# Generate First Epoch Forest Plots
# This script creates first epoch versions of all forest plots
# Based on the existing forest plot code in null_subject_analysis.R

library(tidyverse)
library(ggplot2)
library(scales)

# Load paper figure specifications
source("analysis/scripts/paper_figures/figure_dimensions.R")

# Load data
cat("Loading data...\n")
data <- read.csv("evaluation/results/all_models_null_subject_lme4_ready.csv")

# Add readable model names
data <- data %>%
  mutate(model_label = case_when(
    model == "exp0_baseline" ~ "Baseline",
    model == "exp1_remove_expletives" ~ "Remove Expletives",
    model == "exp2_impoverish_determiners" ~ "Impoverish Determiners",
    model == "exp3_remove_articles" ~ "Remove Articles",
    model == "exp4_lemmatize_verbs" ~ "Lemmatize Verbs",
    model == "exp5_remove_subject_pronominals" ~ "Remove Subject Pronominals",
    TRUE ~ model
  ))

# Convert to factors
data$model <- as.factor(data$model)
data$model_label <- factor(data$model_label, levels = unique(data$model_label))
data$form_type <- factor(data$form_type, levels = c("null", "overt"))
data$item_group <- as.factor(data$item_group)
data$form <- factor(data$form, levels = c("default", "complex_long", "complex_emb", "context_negation", "target_negation", "both_negation"))

cat("Data loaded. Dimensions:", dim(data), "\n")

# Define PAPER_COLORS (from original script)
PAPER_COLORS <- list(
  models = c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
)

# Create output directory
dir.create("analysis/paper_figures/supplementary", recursive = TRUE, showWarnings = FALSE)

cat("Creating first epoch forest charts for each model...\n")

# Calculate first-epoch preferences for forest charts (using first 10% of checkpoints - like begin_state_data)
first_epoch_data <- data %>%
  group_by(model_label) %>%
  filter(checkpoint_num <= quantile(checkpoint_num, 0.1)) %>%
  ungroup()

# Get list of models
models_list <- unique(data$model_label)

for(model_name in models_list) {
  cat("  Creating first epoch forest charts for:", model_name, "\n")
  
  model_first_data <- first_epoch_data %>% filter(model_label == model_name)
  safe_model_name <- gsub("[^A-Za-z0-9]", "_", tolower(model_name))
  
  # Assign model-specific color
  model_color <- case_when(
    model_name == "Baseline" ~ PAPER_COLORS$models[1],
    model_name == "Remove Expletives" ~ PAPER_COLORS$models[2],
    model_name == "Impoverish Determiners" ~ PAPER_COLORS$models[3],
    model_name == "Remove Articles" ~ PAPER_COLORS$models[4],
    model_name == "Lemmatize Verbs" ~ PAPER_COLORS$models[5],
    model_name == "Remove Subject Pronominals" ~ PAPER_COLORS$models[6],
    TRUE ~ PAPER_COLORS$models[1]  # fallback
  )
  
  # 1. FOREST CHART BY ITEM GROUP (OVERT SUBJECTS ONLY) - FIRST EPOCH
  if ("item_group" %in% names(model_first_data)) {
    # Create mapping for human-readable item group names
    item_group_names <- c(
      "1a_3rdSG" = "1a. 3rd Person Singular",
      "1b_3rdPL" = "1b. 3rd Person Plural", 
      "2a_2ndSG" = "2a. 2nd Person Singular",
      "2b_2ndPL" = "2b. 2nd Person Plural",
      "3a_1stSg" = "3a. 1st Person Singular",
      "3b_1stPL" = "3b. 1st Person Plural",
      "4a_subject_control" = "4a. Subject Control",
      "4b_object_control" = "4b. Object Control",
      "5a_expletive_seems" = "5a. Expletive 'Seems'",
      "5b_expletive_be" = "5b. Expletive 'Be'",
      "6_long_distance_binding" = "6. Long Distance Binding",
      "7a_conjunction_no_topic_shift" = "7a. Conjunction (No Topic Shift)",
      "7b_conjunction_topic_shift" = "7b. Conjunction (Topic Shift)"
    )
    
    # Define the order based on the numbering
    item_group_order <- c(
      "1a_3rdSG", "1b_3rdPL", "2a_2ndSG", "2b_2ndPL", "3a_1stSg", "3b_1stPL",
      "4a_subject_control", "4b_object_control", "5a_expletive_seems", "5b_expletive_be",
      "6_long_distance_binding", "7a_conjunction_no_topic_shift", "7b_conjunction_topic_shift"
    )
    
    item_group_forest <- model_first_data %>%
      filter(form_type == "overt") %>%  # Only overt subjects
      group_by(item_group) %>%
      summarise(
        mean_pref = mean(correct, na.rm = TRUE),
        se_pref = sd(correct, na.rm = TRUE) / sqrt(n()),
        n = n(),
        .groups = "drop"
      ) %>%
      mutate(
        ci_lower = mean_pref - 1.96 * se_pref,
        ci_upper = mean_pref + 1.96 * se_pref,
        ci_lower = pmax(0, ci_lower),  # Bound at 0
        ci_upper = pmin(1, ci_upper),  # Bound at 1
        # Add human-readable names and ordering (reversed for ascending order)
        item_group_readable = factor(item_group_names[item_group], 
                                   levels = rev(item_group_names[item_group_order]))
      ) %>%
      filter(!is.na(item_group_readable))  # Remove any unmapped groups
    
    p_forest_item_group <- ggplot(item_group_forest, aes(x = mean_pref, y = item_group_readable)) +
      geom_point(size = 2.0, color = model_color) +
      geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper), 
                     height = 0.2, color = model_color) +
      geom_vline(xintercept = 0.5, linetype = "dotted", color = "gray50", linewidth = 0.5) +
      scale_x_continuous(labels = percent_format(), limits = c(0, 1)) +
      scale_y_discrete(position = "right") +  # Move y-axis labels to right
      theme_minimal() +
      theme(
        legend.position = "none",  # Remove legend
        axis.text.y = element_text(hjust = 0),  # Left-align y-axis text
        plot.title = element_text(size = 10, face = "bold"),
        plot.subtitle = element_text(size = 9),
        axis.title = element_text(size = 9),
        axis.text = element_text(size = 8)
      ) +
      labs(
        title = paste("Overt Subject Preference by Item Group:\n", model_name, "(First Epoch)"),
        subtitle = "First epoch overt subject preferences with 95% confidence intervals",
        x = "Overt Subject Preference",
        y = "Evaluation Set"
      )
    
    # Save forest chart by item group - first epoch
    ggsave(paste0("analysis/paper_figures/supplementary/forest_item_group_first_epoch_", safe_model_name, ".pdf"), 
           p_forest_item_group, 
           width = get_figure_specs("regular")$width, 
           height = get_figure_specs("regular")$height, 
           dpi = get_figure_specs("regular")$dpi)
    ggsave(paste0("analysis/paper_figures/supplementary/forest_item_group_first_epoch_", safe_model_name, ".png"), 
           p_forest_item_group, 
           width = get_figure_specs("regular")$width, 
           height = get_figure_specs("regular")$height, 
           dpi = get_figure_specs("regular")$dpi)
  }
  
  # 2. FOREST CHART BY FORM (OVERT SUBJECTS ONLY) - FIRST EPOCH
  if ("form" %in% names(model_first_data)) {
    # Create mapping for human-readable form names and ordering
    form_names <- c(
      "default" = "Default",
      "complex_long" = "Complex Long", 
      "complex_emb" = "Complex Embedded",
      "context_negation" = "Context Negation",
      "target_negation" = "Target Negation",
      "both_negation" = "Both Negation"
    )
    
    # Define the order as requested
    form_order <- c("default", "complex_long", "complex_emb", "context_negation", "target_negation", "both_negation")
    
    form_forest <- model_first_data %>%
      filter(form_type == "overt") %>%  # Only overt subjects
      group_by(form) %>%
      summarise(
        mean_pref = mean(correct, na.rm = TRUE),
        se_pref = sd(correct, na.rm = TRUE) / sqrt(n()),
        n = n(),
        .groups = "drop"
      ) %>%
      mutate(
        ci_lower = mean_pref - 1.96 * se_pref,
        ci_upper = mean_pref + 1.96 * se_pref,
        ci_lower = pmax(0, ci_lower),  # Bound at 0
        ci_upper = pmin(1, ci_upper),  # Bound at 1
        # Add human-readable names and ordering (reversed for ascending order)
        form_readable = factor(form_names[form], 
                              levels = rev(form_names[form_order]))
      ) %>%
      filter(!is.na(form_readable))  # Remove any unmapped forms
    
    p_forest_form <- ggplot(form_forest, aes(x = mean_pref, y = form_readable)) +
      geom_point(size = 2.0, color = model_color) +
      geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper), 
                     height = 0.2, color = model_color) +
      geom_vline(xintercept = 0.5, linetype = "dotted", color = "gray50", linewidth = 0.5) +
      scale_x_continuous(labels = percent_format(), limits = c(0, 1)) +
      scale_y_discrete(position = "right") +  # Move y-axis labels to right
      theme_minimal() +
      theme(
        legend.position = "none",  # Remove legend
        axis.text.y = element_text(hjust = 0),  # Left-align y-axis text
        plot.title = element_text(size = 10, face = "bold"),
        plot.subtitle = element_text(size = 9),
        axis.title = element_text(size = 9),
        axis.text = element_text(size = 8)
      ) +
      labs(
        title = paste("Overt Subject Preference by Linguistic Form:\n", model_name, "(First Epoch)"),
        subtitle = "First epoch overt subject preferences with 95% confidence intervals",
        x = "Overt Subject Preference",
        y = "Processing Manipulation"
      )
    
    # Save forest chart by form - first epoch
    ggsave(paste0("analysis/paper_figures/supplementary/forest_form_first_epoch_", safe_model_name, ".pdf"), 
           p_forest_form, 
           width = get_figure_specs("regular")$width, 
           height = get_figure_specs("regular")$height, 
           dpi = get_figure_specs("regular")$dpi)
    ggsave(paste0("analysis/paper_figures/supplementary/forest_form_first_epoch_", safe_model_name, ".png"), 
           p_forest_form, 
           width = get_figure_specs("regular")$width, 
           height = get_figure_specs("regular")$height, 
           dpi = get_figure_specs("regular")$dpi)
  }
}

cat("First epoch forest charts created for all models.\n")
cat("Figures saved in: analysis/paper_figures/supplementary/\n")
cat("Files created:\n")
for(model_name in models_list) {
  safe_model_name <- gsub("[^A-Za-z0-9]", "_", tolower(model_name))
  cat(paste("  - forest_item_group_first_epoch_", safe_model_name, ".pdf/.png\n", sep=""))
  cat(paste("  - forest_form_first_epoch_", safe_model_name, ".pdf/.png\n", sep=""))
}