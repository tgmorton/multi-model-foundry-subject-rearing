# Generate Overt-Only Faceted Comparison Figures
# This script creates versions of the faceted comparison figures showing only the overt condition
# Based on the existing code in null_subject_analysis.R

library(tidyverse)
library(ggplot2)
library(scales)

# Load paper figure specifications
source("analysis/scripts/paper_figures/figure_dimensions.R")

# Load data
cat("Loading data...\n")
data <- read.csv("evaluation/results/all_models_null_subject_lme4_ready.csv")

# Load t50 acquisition timing data
t50_data <- read.csv("analysis/tables/tests/t50_by_model_robust.csv")
t50_data <- t50_data %>%
  mutate(
    t50_checkpoint_log = log10(t50_checkpoint + 1),
    ci_lower_log = log10(CI_lo + 1),
    ci_upper_log = log10(CI_hi + 1)
  )

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

# Prepare model comparison data (same as original script)
model_comparison_data <- data %>%
  group_by(model, model_label, checkpoint_num, form_type, item_id, form) %>%
  summarise(item_form_correct = mean(correct, na.rm = TRUE), .groups = "drop") %>%
  group_by(model, model_label, checkpoint_num, form_type) %>%
  summarise(
    mean_correct = mean(item_form_correct, na.rm = TRUE),
    se_correct = sd(item_form_correct, na.rm = TRUE) / sqrt(n()),
    n_item_form_combinations = n(),
    ci_lower = pmax(0, mean_correct - 1.96 * se_correct),
    ci_upper = pmin(1, mean_correct + 1.96 * se_correct),
    .groups = "drop"
  )

# Calculate crossover data
model_comparison_crossover <- t50_data %>%
  select(Model, model, t50_checkpoint, CI_lo, CI_hi) %>%
  rename(model_label = Model) %>%
  mutate(
    crossover_checkpoint_log = log10(t50_checkpoint + 1),
    ci_lower_log = log10(CI_lo + 1),
    ci_upper_log = log10(CI_hi + 1)
  )

# Define PAPER_COLORS (from original script)
PAPER_COLORS <- list(
  models = c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
)

# Create output directory
dir.create("analysis/paper_figures/wide", recursive = TRUE, showWarnings = FALSE)

cat("Creating overt-only faceted comparison figures...\n")

# Get list of ablated models (exclude baseline)
ablated_models <- setdiff(unique(data$model_label), "Baseline")

for(target_model in ablated_models) {
  cat("  Creating overt-only comparison for:", target_model, "vs Baseline\n")
  
  # Filter data for baseline and target model, OVERT ONLY
  comparison_data <- model_comparison_data %>%
    filter(model_label %in% c("Baseline", target_model)) %>%
    filter(form_type == "overt")  # Only overt subjects
  
  # Create log-transformed version
  comparison_data_log <- comparison_data %>%
    mutate(checkpoint_num_log = log10(checkpoint_num + 1))
  
  # Get crossover data for both models
  comparison_crossover <- model_comparison_crossover %>%
    filter(model_label %in% c("Baseline", target_model))
  
  comparison_crossover_log <- comparison_crossover
  
  # Create safe filename
  safe_target_name <- gsub("[^A-Za-z0-9]", "_", tolower(target_model))
  
  # Get the correct color index for each model
  baseline_color <- PAPER_COLORS$models[1]
  target_color <- case_when(
    target_model == "Remove Expletives" ~ PAPER_COLORS$models[2],
    target_model == "Impoverish Determiners" ~ PAPER_COLORS$models[3],
    target_model == "Remove Articles" ~ PAPER_COLORS$models[4],
    target_model == "Lemmatize Verbs" ~ PAPER_COLORS$models[5],
    target_model == "Remove Subject Pronominals" ~ PAPER_COLORS$models[6],
    TRUE ~ PAPER_COLORS$models[2]  # fallback
  )
  
  # Create overt-only comparison figure (no faceting needed since only one condition)
  p_vs_baseline_overt_only <- ggplot(comparison_data_log, 
                                   aes(x = checkpoint_num_log, y = mean_correct, 
                                       color = model_label, fill = model_label)) +
    geom_line(linewidth = 0.6) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                alpha = 0.15, color = NA) +
    # Add t50 confidence intervals as rectangles (behind lines)
    geom_rect(data = comparison_crossover_log,
             aes(xmin = ci_lower_log, xmax = ci_upper_log, fill = model_label),
             ymin = 0, ymax = 1, alpha = 0.1, inherit.aes = FALSE) +
    # Add log-transformed acquisition lines
    geom_vline(data = comparison_crossover_log, 
               aes(xintercept = crossover_checkpoint_log, color = model_label),
               linetype = "dashed", linewidth = 0.5, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", linewidth = 0.3) +
    # Use correct colors
    scale_color_manual(values = setNames(c(baseline_color, target_color), c("Baseline", target_model))) +
    scale_fill_manual(values = setNames(c(baseline_color, target_color), c("Baseline", target_model))) +
    scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
    scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                       labels = c("0", "10", "100", "1K", "10K")) +
    theme_minimal() +
    theme(
      legend.text = element_text(size = 8),
      legend.key.size = unit(0.3, "cm"),
      legend.spacing = unit(0.1, "cm"),
      legend.title = element_blank(),
      legend.position = "bottom",
      plot.title = element_text(size = 7, face = "bold"),
      plot.subtitle = element_text(size = 6),
      axis.title = element_text(size = 8),
      axis.text = element_text(size = 7)
    ) +
    guides(
      color = guide_legend(title = "Model"),
      fill = guide_legend(title = "Model")
    ) +
    labs(
      title = paste("Model Comparison:", target_model, "vs Baseline (Overt Subjects Only)"),
      subtitle = "Overt subject acquisition (log scale). Dashed lines = 50/50 acquisition points.",
      x = "Training Checkpoint (Log Scale)",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold"
    )
  
  # Save overt-only version with standard width
  spec_regular <- get_figure_specs("regular")
  ggsave(paste0("analysis/paper_figures/wide/comparison_vs_baseline_overt_only_", safe_target_name, ".pdf"), 
         p_vs_baseline_overt_only, 
         width = spec_regular$width, height = spec_regular$height, dpi = spec_regular$dpi)
  ggsave(paste0("analysis/paper_figures/wide/comparison_vs_baseline_overt_only_", safe_target_name, ".png"), 
         p_vs_baseline_overt_only, 
         width = spec_regular$width, height = spec_regular$height, dpi = spec_regular$dpi)
}

cat("Overt-only faceted comparison figures created.\n")
cat("Figures saved in: analysis/paper_figures/wide/\n")
cat("Files created:\n")
for(target_model in ablated_models) {
  safe_target_name <- gsub("[^A-Za-z0-9]", "_", tolower(target_model))
  cat(paste("  - comparison_vs_baseline_overt_only_", safe_target_name, ".pdf\n", sep=""))
  cat(paste("  - comparison_vs_baseline_overt_only_", safe_target_name, ".png\n", sep=""))
}