# Null Subject Comprehensive Analysis
# This script analyzes null vs overt subject preferences across models
# with detailed descriptive statistics and visualizations

library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(knitr)
library(kableExtra)
library(scales)

# Load paper figure specifications
source("analysis/scripts/paper_figures/figure_dimensions.R")

# Create output directories for paper figures
dir.create("analysis/paper_figures", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/paper_figures/main", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/paper_figures/supplementary", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/paper_figures/wide", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/paper_tables", recursive = TRUE, showWarnings = FALSE)

# Load data (only if not already in environment or if it's not a dataframe)
if (!exists("data") || !is.data.frame(data)) {
  cat("Loading data...\n")
  data <- read.csv("evaluation/results/all_models_null_subject_lme4_ready.csv")
  cat("Data loaded from file.\n")
} else {
  cat("Data already exists in environment - skipping load.\n")
}

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
data$form <- as.factor(data$form)

cat("Data loaded. Dimensions:", dim(data), "\n")
cat("Models:", unique(data$model), "\n")
cat("Forms:", unique(data$form), "\n")
cat("Item groups:", unique(data$item_group), "\n\n")

# ============================================================================
# VISUALIZATION: Null vs Overt Preference Over Training
# ============================================================================

cat("Creating preference visualization...\n")

# Calculate mean preferences by checkpoint and model
pref_summary <- data %>%
  group_by(model, model_label, checkpoint_num, form_type) %>%
  summarise(
    mean_correct = mean(correct, na.rm = TRUE),
    se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
    mean_surprisal = mean(mean_surprisal, na.rm = TRUE),
    .groups = "drop"
  )

# Create WIDE figure showing all models
p_preference_wide <- ggplot(pref_summary, 
                           aes(x = checkpoint_num, y = mean_correct, 
                               color = form_type, fill = form_type)) +
  geom_line(linewidth = 0.8) +
  geom_ribbon(aes(ymin = mean_correct - se_correct, 
                  ymax = mean_correct + se_correct),
              alpha = 0.2, color = NA) +
  facet_wrap(~ model_label, nrow = 2, ncol = 3) +
  scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                     labels = c("Null Subject", "Overt Subject")) +
  scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                    labels = c("Null Subject", "Overt Subject")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  labs(
    title = "Learning Trajectories Across All Models",
    x = "Training Checkpoint", 
    y = "Proportion Preferred"
  ) +
  paper_theme(get_figure_specs("wide"))

# Save as wide figure
save_paper_figure(p_preference_wide, "analysis/paper_figures/wide/fig_learning_curves_all", "wide")

# Create REGULAR figure for main text (baseline only)
baseline_data <- pref_summary %>% filter(model == "exp0_baseline")

p_preference_main <- ggplot(baseline_data, 
                           aes(x = checkpoint_num, y = mean_correct, 
                               color = form_type, fill = form_type)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = mean_correct - se_correct, 
                  ymax = mean_correct + se_correct),
              alpha = 0.2, color = NA) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray60", linewidth = 0.5) +
  scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                     labels = c("Null Subject", "Overt Subject")) +
  scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                    labels = c("Null Subject", "Overt Subject")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  labs(
    title = "Subject Preference Learning (Baseline Model)",
    x = "Training Checkpoint", 
    y = "Proportion Preferred"
  ) +
  paper_theme(get_figure_specs("regular"))

# Save as regular figure
save_paper_figure(p_preference_main, "analysis/paper_figures/main/fig1_baseline_learning", "regular")

cat("Paper-optimized preference plots saved\n")

# ============================================================================
# ACQUISITION POINT ESTIMATION - 50/50 Crossover Analysis
# ============================================================================

cat("Calculating 50/50 acquisition points...\n")

# Helper function to find crossover point where null preference = 0.5
find_crossover_point <- function(df, checkpoint_col = "checkpoint_num", pref_col = "null_pref") {
  df <- df %>% arrange(!!sym(checkpoint_col))
  
  # Calculate null preference (proportion null is preferred)
  if (!"null_pref" %in% names(df)) {
    df <- df %>%
      mutate(null_pref = ifelse(form_type == "null", mean_correct, 1 - mean_correct)) %>%
      group_by(across(-c(form_type, mean_correct, se_correct))) %>%
      summarise(null_pref = mean(null_pref), .groups = "drop")
  }
  
  # Filter out very early checkpoints to avoid noise (after step 100)
  df_filtered <- df %>% 
    filter(!!sym(checkpoint_col) > 100) %>%
    arrange(!!sym(checkpoint_col))
  
  if (nrow(df_filtered) == 0) {
    df_filtered <- df  # Fallback
  }
  
  # Find the training halfway point to avoid marking very late crossovers
  max_checkpoint <- max(df_filtered[[checkpoint_col]], na.rm = TRUE)
  halfway_point <- max_checkpoint * 0.6  # Use 60% as "halfway" to be conservative
  
  # Find points close to 50% (within 5% tolerance)
  tolerance <- 0.05
  near_50_points <- df_filtered %>%
    mutate(dist_from_50 = abs(null_pref - 0.5)) %>%
    filter(dist_from_50 <= tolerance) %>%
    arrange(!!sym(checkpoint_col))
  
  if (nrow(near_50_points) > 0) {
    # Look for the first good crossing point in the early-to-mid training
    early_crossings <- near_50_points %>%
      filter(!!sym(checkpoint_col) <= halfway_point)
    
    if (nrow(early_crossings) > 0) {
      # Take the LAST crossing in the first half of training (most stable early point)
      crossover_point <- early_crossings %>%
        slice_tail(n = 1) %>%
        pull(!!sym(checkpoint_col))
    } else {
      # If no early crossings, take the very first crossing point found
      # (this handles cases where crossing only happens late in training)
      crossover_point <- near_50_points %>%
        slice_head(n = 1) %>%
        pull(!!sym(checkpoint_col))
    }
  } else {
    # No points within tolerance - find closest point, preferring earlier ones
    all_distances <- df_filtered %>%
      mutate(
        dist_from_50 = abs(null_pref - 0.5),
        is_early = !!sym(checkpoint_col) <= halfway_point
      )
    
    min_distance <- min(all_distances$dist_from_50, na.rm = TRUE)
    
    # Prefer early points if they're close to the minimum distance
    best_candidates <- all_distances %>%
      filter(dist_from_50 <= (min_distance + 0.02))  # Allow 2% tolerance for preferring early points
    
    early_candidates <- best_candidates %>% filter(is_early)
    
    if (nrow(early_candidates) > 0) {
      # Take latest early candidate
      crossover_point <- early_candidates %>%
        slice_tail(n = 1) %>%
        pull(!!sym(checkpoint_col))
    } else {
      # Take earliest late candidate
      crossover_point <- best_candidates %>%
        slice_head(n = 1) %>%
        pull(!!sym(checkpoint_col))
    }
  }
  
  return(crossover_point)
}

# Calculate overall crossover points by model
overall_crossover <- pref_summary %>%
  group_by(model, model_label, checkpoint_num) %>%
  summarise(
    null_pref = mean(mean_correct[form_type == "null"]),
    .groups = "drop"
  ) %>%
  group_by(model, model_label) %>%
  summarise(
    crossover_checkpoint = find_crossover_point(cur_data()),
    .groups = "drop"
  )

# Calculate preferences by item group for plotting
itemgroup_summary <- data %>%
  group_by(model, model_label, checkpoint_num, item_group, form_type) %>%
  summarise(
    mean_correct = mean(correct, na.rm = TRUE),
    se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Calculate crossover points by item group
itemgroup_crossover <- itemgroup_summary %>%
  group_by(model, model_label, item_group, checkpoint_num) %>%
  summarise(
    null_pref = mean(mean_correct[form_type == "null"]),
    .groups = "drop"
  ) %>%
  group_by(model, model_label, item_group) %>%
  summarise(
    crossover_checkpoint = find_crossover_point(cur_data()),
    .groups = "drop"
  )

# Calculate preferences by form and item group
form_itemgroup_summary <- data %>%
  group_by(model, model_label, checkpoint_num, item_group, form, form_type) %>%
  summarise(
    mean_correct = mean(correct, na.rm = TRUE),
    se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Calculate crossover points by form within item group
form_itemgroup_crossover <- form_itemgroup_summary %>%
  group_by(model, model_label, item_group, form, checkpoint_num) %>%
  summarise(
    null_pref = mean(mean_correct[form_type == "null"]),
    .groups = "drop"
  ) %>%
  group_by(model, model_label, item_group, form) %>%
  summarise(
    crossover_checkpoint = find_crossover_point(cur_data()),
    .groups = "drop"
  )

# Calculate preferences by form only
form_summary <- data %>%
  group_by(model, model_label, checkpoint_num, form, form_type) %>%
  summarise(
    mean_correct = mean(correct, na.rm = TRUE),
    se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Calculate crossover points by form
form_crossover <- form_summary %>%
  group_by(model, model_label, form, checkpoint_num) %>%
  summarise(
    null_pref = mean(mean_correct[form_type == "null"]),
    .groups = "drop"
  ) %>%
  group_by(model, model_label, form) %>%
  summarise(
    crossover_checkpoint = find_crossover_point(cur_data()),
    .groups = "drop"
  )

# Save crossover analysis
write.csv(overall_crossover, "analysis/tables/acquisition_points_overall.csv", row.names = FALSE)
write.csv(itemgroup_crossover, "analysis/tables/acquisition_points_by_itemgroup.csv", row.names = FALSE)
write.csv(form_crossover, "analysis/tables/acquisition_points_by_form.csv", row.names = FALSE)
write.csv(form_itemgroup_crossover, "analysis/tables/acquisition_points_by_form_itemgroup.csv", row.names = FALSE)

cat("Acquisition points calculated and saved\n")

# ============================================================================
# MODEL COMPARISON FIGURES - GENERAL ACQUISITION PATTERNS
# ============================================================================

cat("Creating model comparison figures...\n")

# Overall model comparison (collapsed across items and forms)
# First calculate means by item and form, then aggregate for confidence intervals
model_comparison_data <- data %>%
  # Step 1: Get means by item and form within each model/checkpoint
  group_by(model, model_label, checkpoint_num, form_type, item_id, form) %>%
  summarise(item_form_correct = mean(correct, na.rm = TRUE), .groups = "drop") %>%
  # Step 2: Calculate overall means and confidence intervals based on item/form variation
  group_by(model, model_label, checkpoint_num, form_type) %>%
  summarise(
    mean_correct = mean(item_form_correct, na.rm = TRUE),
    se_correct = sd(item_form_correct, na.rm = TRUE) / sqrt(n()),
    n_item_form_combinations = n(),
    # Calculate 95% confidence interval
    ci_lower = mean_correct - 1.96 * se_correct,
    ci_upper = mean_correct + 1.96 * se_correct,
    .groups = "drop"
  )

# Add acquisition lines for model comparison
model_comparison_crossover <- overall_crossover %>%
  mutate(crossover_checkpoint_log = log10(crossover_checkpoint + 1))

# Regular scale model comparison
p_models_comparison <- ggplot(model_comparison_data, 
                             aes(x = checkpoint_num, y = mean_correct, 
                                 color = model_label, fill = model_label)) +
  geom_line(size = 0.8) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.15, color = NA) +
  # Add acquisition lines
  geom_vline(data = model_comparison_crossover, 
             aes(xintercept = crossover_checkpoint, color = model_label),
             linetype = "dashed", size = 0.7, alpha = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  facet_wrap(~ form_type, labeller = labeller(form_type = c("null" = "Null Subject Preference", 
                                                            "overt" = "Overt Subject Preference"))) +
  scale_color_manual(values = c("Baseline" = "#1f77b4", 
                                "Remove Expletives" = "#ff7f0e",
                                "Impoverish Determiners" = "#2ca02c", 
                                "Remove Articles" = "#d62728",
                                "Lemmatize Verbs" = "#9467bd",
                                "Remove Subject Pronominals" = "#8c564b")) +
  scale_fill_manual(values = c("Baseline" = "#1f77b4", 
                               "Remove Expletives" = "#ff7f0e",
                               "Impoverish Determiners" = "#2ca02c", 
                               "Remove Articles" = "#d62728",
                               "Lemmatize Verbs" = "#9467bd",
                               "Remove Subject Pronominals" = "#8c564b")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.text = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  ) +
  guides(
    color = guide_legend(title = "Model"),
    fill = guide_legend(title = "Model")
  ) +
  labs(
    title = "Model Comparison: General Null Subject Acquisition",
    subtitle = "Collapsed across all items and linguistic forms. Ribbons show 95% CIs based on item/form variation. Dashed lines = 50/50 acquisition points.",
    x = "Training Checkpoint",
    y = "Proportion Preferred",
    caption = "Gray dotted line = 50% preference threshold"
  )

# Save model comparison plot
ggsave("analysis/figures/combined/models_comparison_general.pdf", 
       p_models_comparison, width = 14, height = 8)
ggsave("analysis/figures/combined/models_comparison_general.png", 
       p_models_comparison, width = 14, height = 8, dpi = 300)

# Log scale model comparison  
model_comparison_data_log <- model_comparison_data %>%
  mutate(checkpoint_num_log = log10(checkpoint_num + 1))

p_models_comparison_log <- ggplot(model_comparison_data_log, 
                                 aes(x = checkpoint_num_log, y = mean_correct, 
                                     color = model_label, fill = model_label)) +
  geom_line(size = 0.8) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.15, color = NA) +
  # Add log-transformed acquisition lines
  geom_vline(data = model_comparison_crossover, 
             aes(xintercept = crossover_checkpoint_log, color = model_label),
             linetype = "dashed", size = 0.7, alpha = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  facet_wrap(~ form_type, labeller = labeller(form_type = c("null" = "Null Subject Preference", 
                                                            "overt" = "Overt Subject Preference"))) +
  scale_color_manual(values = c("Baseline" = "#1f77b4", 
                                "Remove Expletives" = "#ff7f0e",
                                "Impoverish Determiners" = "#2ca02c", 
                                "Remove Articles" = "#d62728",
                                "Lemmatize Verbs" = "#9467bd",
                                "Remove Subject Pronominals" = "#8c564b")) +
  scale_fill_manual(values = c("Baseline" = "#1f77b4", 
                               "Remove Expletives" = "#ff7f0e",
                               "Impoverish Determiners" = "#2ca02c", 
                               "Remove Articles" = "#d62728",
                               "Lemmatize Verbs" = "#9467bd",
                               "Remove Subject Pronominals" = "#8c564b")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                     labels = c("0", "10", "100", "1K", "10K")) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.text = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  ) +
  guides(
    color = guide_legend(title = "Model"),
    fill = guide_legend(title = "Model")
  ) +
  labs(
    title = "Model Comparison: General Null Subject Acquisition (Log Scale)",
    subtitle = "Collapsed across all items and linguistic forms. Ribbons show 95% CIs based on item/form variation. Dashed lines = 50/50 acquisition points.",
    x = "Training Checkpoint (Log Scale)",
    y = "Proportion Preferred",
    caption = "Gray dotted line = 50% preference threshold"
  )

# Save log scale model comparison plot
ggsave("analysis/figures/combined/models_comparison_general_log.pdf", 
       p_models_comparison_log, width = 14, height = 8)
ggsave("analysis/figures/combined/models_comparison_general_log.png", 
       p_models_comparison_log, width = 14, height = 8, dpi = 300)

# Combined null+overt model comparison (like other figures)
p_models_combined <- ggplot(model_comparison_data, 
                           aes(x = checkpoint_num, y = mean_correct, 
                               color = model_label, fill = model_label, linetype = form_type)) +
  geom_line(size = 0.8) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.15, color = NA) +
  # Add acquisition lines
  geom_vline(data = model_comparison_crossover, 
             aes(xintercept = crossover_checkpoint, color = model_label),
             linetype = "dashed", size = 0.7, alpha = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  scale_color_manual(values = c("Baseline" = "#1f77b4", 
                                "Remove Expletives" = "#ff7f0e",
                                "Impoverish Determiners" = "#2ca02c", 
                                "Remove Articles" = "#d62728",
                                "Lemmatize Verbs" = "#9467bd",
                                "Remove Subject Pronominals" = "#8c564b")) +
  scale_fill_manual(values = c("Baseline" = "#1f77b4", 
                               "Remove Expletives" = "#ff7f0e",
                               "Impoverish Determiners" = "#2ca02c", 
                               "Remove Articles" = "#d62728",
                               "Lemmatize Verbs" = "#9467bd",
                               "Remove Subject Pronominals" = "#8c564b")) +
  scale_linetype_manual(values = c("null" = "dotted", "overt" = "solid"),
                       labels = c("Null Subject", "Overt Subject")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.text = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  ) +
  guides(
    color = guide_legend(title = "Model", order = 1),
    fill = guide_legend(title = "Model", order = 1),
    linetype = guide_legend(title = "Form Type", order = 2)
  ) +
  labs(
    title = "Model Comparison: Combined Null & Overt Subject Acquisition",
    subtitle = "All models shown together. Ribbons show 95% CIs based on item/form variation. Dotted=null, solid=overt, dashed lines = 50/50 acquisition points.",
    x = "Training Checkpoint",
    y = "Proportion Preferred",
    caption = "Gray dotted line = 50% preference threshold"
  )

# Save combined model comparison plot
ggsave("analysis/figures/combined/models_comparison_combined.pdf", 
       p_models_combined, width = 14, height = 8)
ggsave("analysis/figures/combined/models_comparison_combined.png", 
       p_models_combined, width = 14, height = 8, dpi = 300)

# Combined log scale model comparison  
p_models_combined_log <- ggplot(model_comparison_data_log, 
                               aes(x = checkpoint_num_log, y = mean_correct, 
                                   color = model_label, fill = model_label, linetype = form_type)) +
  geom_line(size = 0.8) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.15, color = NA) +
  # Add log-transformed acquisition lines
  geom_vline(data = model_comparison_crossover, 
             aes(xintercept = crossover_checkpoint_log, color = model_label),
             linetype = "dashed", size = 0.7, alpha = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  scale_color_manual(values = c("Baseline" = "#1f77b4", 
                                "Remove Expletives" = "#ff7f0e",
                                "Impoverish Determiners" = "#2ca02c", 
                                "Remove Articles" = "#d62728",
                                "Lemmatize Verbs" = "#9467bd",
                                "Remove Subject Pronominals" = "#8c564b")) +
  scale_fill_manual(values = c("Baseline" = "#1f77b4", 
                               "Remove Expletives" = "#ff7f0e",
                               "Impoverish Determiners" = "#2ca02c", 
                               "Remove Articles" = "#d62728",
                               "Lemmatize Verbs" = "#9467bd",
                               "Remove Subject Pronominals" = "#8c564b")) +
  scale_linetype_manual(values = c("null" = "dotted", "overt" = "solid"),
                       labels = c("Null Subject", "Overt Subject")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                     labels = c("0", "10", "100", "1K", "10K")) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.text = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  ) +
  guides(
    color = guide_legend(title = "Model", order = 1),
    fill = guide_legend(title = "Model", order = 1),
    linetype = guide_legend(title = "Form Type", order = 2)
  ) +
  labs(
    title = "Model Comparison: Combined Null & Overt Subject Acquisition (Log Scale)",
    subtitle = "All models shown together. Ribbons show 95% CIs based on item/form variation. Dotted=null, solid=overt, dashed lines = 50/50 acquisition points.",
    x = "Training Checkpoint (Log Scale)",
    y = "Proportion Preferred",
    caption = "Gray dotted line = 50% preference threshold"
  )

# Save combined log scale model comparison plot
ggsave("analysis/figures/combined/models_comparison_combined_log.pdf", 
       p_models_combined_log, width = 14, height = 8)
ggsave("analysis/figures/combined/models_comparison_combined_log.png", 
       p_models_combined_log, width = 14, height = 8, dpi = 300)

cat("Model comparison figures saved\n")

# ============================================================================
# NULL-ONLY MODEL COMPARISON FIGURES
# ============================================================================

cat("Creating null-only model comparison figures...\n")

# Filter data for null subjects only
model_comparison_data_null <- model_comparison_data %>%
  filter(form_type == "null")

model_comparison_data_log_null <- model_comparison_data_null %>%
  mutate(checkpoint_num_log = log10(checkpoint_num + 1))

# Regular scale null-only model comparison
p_models_null_only <- ggplot(model_comparison_data_null, 
                            aes(x = checkpoint_num, y = mean_correct, 
                                color = model_label, fill = model_label)) +
  geom_line(size = 0.8) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.15, color = NA) +
  # Add acquisition lines
  geom_vline(data = model_comparison_crossover, 
             aes(xintercept = crossover_checkpoint, color = model_label),
             linetype = "dashed", size = 0.7, alpha = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  scale_color_manual(values = c("Baseline" = "#1f77b4", 
                                "Remove Expletives" = "#ff7f0e",
                                "Impoverish Determiners" = "#2ca02c", 
                                "Remove Articles" = "#d62728",
                                "Lemmatize Verbs" = "#9467bd",
                                "Remove Subject Pronominals" = "#8c564b")) +
  scale_fill_manual(values = c("Baseline" = "#1f77b4", 
                               "Remove Expletives" = "#ff7f0e",
                               "Impoverish Determiners" = "#2ca02c", 
                               "Remove Articles" = "#d62728",
                               "Lemmatize Verbs" = "#9467bd",
                               "Remove Subject Pronominals" = "#8c564b")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  ) +
  guides(
    color = guide_legend(title = "Model"),
    fill = guide_legend(title = "Model")
  ) +
  labs(
    title = "Model Comparison: Null Subject Preference Only",
    subtitle = "Collapsed across all items and linguistic forms. Ribbons show 95% CIs. Dashed lines = 50/50 acquisition points.",
    x = "Training Checkpoint",
    y = "Null Subject Preference",
    caption = "Gray dotted line = 50% preference threshold"
  )

# Save null-only model comparison plot
ggsave("analysis/figures/combined/models_comparison_null_only.pdf", 
       p_models_null_only, width = 12, height = 8)
ggsave("analysis/figures/combined/models_comparison_null_only.png", 
       p_models_null_only, width = 12, height = 8, dpi = 300)

# Log scale null-only model comparison  
p_models_null_only_log <- ggplot(model_comparison_data_log_null, 
                                aes(x = checkpoint_num_log, y = mean_correct, 
                                    color = model_label, fill = model_label)) +
  geom_line(size = 0.8) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.15, color = NA) +
  # Add log-transformed acquisition lines
  geom_vline(data = model_comparison_crossover, 
             aes(xintercept = crossover_checkpoint_log, color = model_label),
             linetype = "dashed", size = 0.7, alpha = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  scale_color_manual(values = c("Baseline" = "#1f77b4", 
                                "Remove Expletives" = "#ff7f0e",
                                "Impoverish Determiners" = "#2ca02c", 
                                "Remove Articles" = "#d62728",
                                "Lemmatize Verbs" = "#9467bd",
                                "Remove Subject Pronominals" = "#8c564b")) +
  scale_fill_manual(values = c("Baseline" = "#1f77b4", 
                               "Remove Expletives" = "#ff7f0e",
                               "Impoverish Determiners" = "#2ca02c", 
                               "Remove Articles" = "#d62728",
                               "Lemmatize Verbs" = "#9467bd",
                               "Remove Subject Pronominals" = "#8c564b")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                     labels = c("0", "10", "100", "1K", "10K")) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  ) +
  guides(
    color = guide_legend(title = "Model"),
    fill = guide_legend(title = "Model")
  ) +
  labs(
    title = "Model Comparison: Null Subject Preference Only (Log Scale)",
    subtitle = "Collapsed across all items and linguistic forms. Ribbons show 95% CIs. Dashed lines = 50/50 acquisition points.",
    x = "Training Checkpoint (Log Scale)",
    y = "Null Subject Preference",
    caption = "Gray dotted line = 50% preference threshold"
  )

# Save log scale null-only model comparison plot
ggsave("analysis/figures/combined/models_comparison_null_only_log.pdf", 
       p_models_null_only_log, width = 12, height = 8)
ggsave("analysis/figures/combined/models_comparison_null_only_log.png", 
       p_models_null_only_log, width = 12, height = 8, dpi = 300)

cat("Null-only model comparison figures saved\n")

# ============================================================================
# MODEL COMPARISON BY FORM - SHOWING FORM DIFFERENCES WITHIN EACH MODEL
# ============================================================================

cat("Creating model comparison by form figures...\n")

# Model comparison data aggregated by form (not collapsed across forms)
model_form_comparison_data <- data %>%
  # Step 1: Get means by item within each model/checkpoint/form combination
  group_by(model, model_label, checkpoint_num, form_type, form, item_id) %>%
  summarise(item_correct = mean(correct, na.rm = TRUE), .groups = "drop") %>%
  # Step 2: Calculate overall means and confidence intervals based on item variation
  group_by(model, model_label, checkpoint_num, form_type, form) %>%
  summarise(
    mean_correct = mean(item_correct, na.rm = TRUE),
    se_correct = sd(item_correct, na.rm = TRUE) / sqrt(n()),
    n_items = n(),
    # Calculate 95% confidence interval
    ci_lower = mean_correct - 1.96 * se_correct,
    ci_upper = mean_correct + 1.96 * se_correct,
    .groups = "drop"
  )

# Regular scale model-form comparison (faceted by null/overt)
p_models_by_form <- ggplot(model_form_comparison_data, 
                          aes(x = checkpoint_num, y = mean_correct, 
                              color = form, fill = form)) +
  geom_line(size = 0.6) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.1, color = NA) +
  # Add form-specific acquisition lines
  geom_vline(data = form_crossover, 
             aes(xintercept = crossover_checkpoint, color = form),
             linetype = "dashed", size = 0.5, alpha = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  facet_grid(model_label ~ form_type, 
             labeller = labeller(form_type = c("null" = "Null Subject", 
                                               "overt" = "Overt Subject"))) +
  scale_color_manual(values = c("both_negation" = "#E63946",
                               "complex_emb" = "#F77F00", 
                               "complex_long" = "#FCBF49",
                               "context_negation" = "#2A9D8F",
                               "default" = "#264653",
                               "target_negation" = "#7209B7")) +
  scale_fill_manual(values = c("both_negation" = "#E63946",
                              "complex_emb" = "#F77F00", 
                              "complex_long" = "#FCBF49",
                              "context_negation" = "#2A9D8F",
                              "default" = "#264653",
                              "target_negation" = "#7209B7")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.text = element_text(size = 9, face = "bold"),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 11),
    panel.spacing = unit(0.2, "lines")
  ) +
  guides(
    color = guide_legend(title = "Linguistic Form", ncol = 6),
    fill = guide_legend(title = "Linguistic Form", ncol = 6)
  ) +
  labs(
    title = "Model Comparison by Linguistic Form",
    subtitle = "Each model shows different linguistic forms. Ribbons show 95% CIs. Dashed lines = form-specific 50/50 acquisition points.",
    x = "Training Checkpoint",
    y = "Proportion Preferred",
    caption = "Gray dotted line = 50% preference threshold"
  )

# Save model-form comparison plot
ggsave("analysis/figures/combined/models_comparison_by_form.pdf", 
       p_models_by_form, width = 12, height = 16)
ggsave("analysis/figures/combined/models_comparison_by_form.png", 
       p_models_by_form, width = 12, height = 16, dpi = 300)

# Log scale version
model_form_comparison_data_log <- model_form_comparison_data %>%
  mutate(checkpoint_num_log = log10(checkpoint_num + 1))

# Add log-transformed crossover data for forms
form_crossover_log <- form_crossover %>%
  mutate(crossover_checkpoint_log = log10(crossover_checkpoint + 1))

p_models_by_form_log <- ggplot(model_form_comparison_data_log, 
                              aes(x = checkpoint_num_log, y = mean_correct, 
                                  color = form, fill = form)) +
  geom_line(size = 0.6) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.1, color = NA) +
  # Add form-specific log-transformed acquisition lines
  geom_vline(data = form_crossover_log, 
             aes(xintercept = crossover_checkpoint_log, color = form),
             linetype = "dashed", size = 0.5, alpha = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  facet_grid(model_label ~ form_type, 
             labeller = labeller(form_type = c("null" = "Null Subject", 
                                               "overt" = "Overt Subject"))) +
  scale_color_manual(values = c("both_negation" = "#E63946",
                               "complex_emb" = "#F77F00", 
                               "complex_long" = "#FCBF49",
                               "context_negation" = "#2A9D8F",
                               "default" = "#264653",
                               "target_negation" = "#7209B7")) +
  scale_fill_manual(values = c("both_negation" = "#E63946",
                              "complex_emb" = "#F77F00", 
                              "complex_long" = "#FCBF49",
                              "context_negation" = "#2A9D8F",
                              "default" = "#264653",
                              "target_negation" = "#7209B7")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                     labels = c("0", "10", "100", "1K", "10K")) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.text = element_text(size = 9, face = "bold"),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 11),
    panel.spacing = unit(0.2, "lines")
  ) +
  guides(
    color = guide_legend(title = "Linguistic Form", ncol = 6),
    fill = guide_legend(title = "Linguistic Form", ncol = 6)
  ) +
  labs(
    title = "Model Comparison by Linguistic Form (Log Scale)",
    subtitle = "Each model shows different linguistic forms. Ribbons show 95% CIs. Dashed lines = form-specific 50/50 acquisition points.",
    x = "Training Checkpoint (Log Scale)",
    y = "Proportion Preferred",
    caption = "Gray dotted line = 50% preference threshold"
  )

# Save log scale model-form comparison plot
ggsave("analysis/figures/combined/models_comparison_by_form_log.pdf", 
       p_models_by_form_log, width = 12, height = 16)
ggsave("analysis/figures/combined/models_comparison_by_form_log.png", 
       p_models_by_form_log, width = 12, height = 16, dpi = 300)

cat("Model comparison by form figures saved\n")

# ============================================================================
# ACQUISITION TIMING SUMMARY TABLE
# ============================================================================

cat("Creating acquisition timing summary table...\n")

# Simple table showing 50/50 checkpoint numbers across models, items, and forms
acquisition_timing_table <- overall_crossover %>%
  dplyr::select(model, model_label, crossover_checkpoint) %>%
  arrange(crossover_checkpoint) %>%
  mutate(
    crossover_checkpoint = round(crossover_checkpoint, 0),
    rank = row_number()
  ) %>%
  dplyr::select(
    Rank = rank,
    Model = model_label,
    `50/50 Checkpoint` = crossover_checkpoint
  )

# Save as CSV
write.csv(acquisition_timing_table, "analysis/tables/acquisition_timing_summary.csv", 
          row.names = FALSE)

# Save as formatted table
kable(acquisition_timing_table, format = "latex", booktabs = TRUE,
      caption = "Null subject acquisition timing by model (50/50 preference checkpoint)") %>%
  kable_styling(latex_options = c("striped", "hold_position")) %>%
  save_kable("analysis/tables/acquisition_timing_summary.tex")

# Print to console
cat("\nAcquisition Timing Summary:\n")
cat(paste(rep("-", 50), collapse = ""), "\n")
print(acquisition_timing_table)
cat(paste(rep("-", 50), collapse = ""), "\n\n")

cat("Acquisition timing summary table saved\n")

# ============================================================================
# INDIVIDUAL MODEL FIGURES WITH ACQUISITION LINES
# ============================================================================

cat("Creating figures with acquisition lines...\n")

# Item Group figures with acquisition lines
for (mod in unique(itemgroup_summary$model)) {
  mod_label <- unique(itemgroup_summary$model_label[itemgroup_summary$model == mod])
  mod_data <- itemgroup_summary %>% filter(model == mod)
  
  # Get crossover data for this model
  mod_crossover <- itemgroup_crossover %>% filter(model == mod)
  
  p_itemgroup <- ggplot(mod_data, 
                        aes(x = checkpoint_num, y = mean_correct, 
                            color = form_type, fill = form_type)) +
    geom_line(size = 0.8) +
    geom_ribbon(aes(ymin = mean_correct - se_correct, 
                    ymax = mean_correct + se_correct),
                alpha = 0.2, color = NA) +
    # Add acquisition lines
    geom_vline(data = mod_crossover, 
               aes(xintercept = crossover_checkpoint),
               linetype = "dashed", color = "red", size = 0.7, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
    facet_wrap(~ item_group, scales = "free_x") +
    scale_color_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      axis.text = element_text(size = 9),
      axis.title = element_text(size = 11)
    ) +
    labs(
      title = paste("Item Group Acquisition Points:", mod_label),
      subtitle = "Red dashed lines show 50/50 preference crossover points",
      x = "Training Checkpoint",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold"
    )
  
  # Save plot
  safe_name <- gsub(" ", "_", tolower(mod_label))
  model_folder <- case_when(
    mod == "exp0_baseline" ~ "baseline",
    mod == "exp1_remove_expletives" ~ "remove_expletives", 
    mod == "exp2_impoverish_determiners" ~ "impoverish_determiners",
    mod == "exp3_remove_articles" ~ "remove_articles",
    mod == "exp4_lemmatize_verbs" ~ "lemmatize_verbs",
    mod == "exp5_remove_subject_pronominals" ~ "remove_subject_pronominals"
  )
  ggsave(paste0("analysis/figures/", model_folder, "/itemgroup_acquisition.pdf"), 
         p_itemgroup, width = 12, height = 8)
  ggsave(paste0("analysis/figures/", model_folder, "/itemgroup_acquisition.png"), 
         p_itemgroup, width = 12, height = 8, dpi = 300)
}

cat("Item group acquisition plots saved\n")


# Form-only figures with acquisition lines
for (mod in unique(form_summary$model)) {
  mod_label <- unique(form_summary$model_label[form_summary$model == mod])
  mod_data <- form_summary %>% filter(model == mod)
  
  # Get crossover data for this model
  mod_crossover <- form_crossover %>% filter(model == mod)
  
  p_forms <- ggplot(mod_data, 
                    aes(x = checkpoint_num, y = mean_correct, 
                        color = form_type, fill = form_type)) +
    geom_line(size = 0.8) +
    geom_ribbon(aes(ymin = mean_correct - se_correct, 
                    ymax = mean_correct + se_correct),
                alpha = 0.2, color = NA) +
    # Add acquisition lines
    geom_vline(data = mod_crossover, 
               aes(xintercept = crossover_checkpoint),
               linetype = "dashed", color = "red", size = 0.7, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
    facet_wrap(~ form, scales = "free_x") +
    scale_color_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      axis.text = element_text(size = 9),
      axis.title = element_text(size = 11)
    ) +
    labs(
      title = paste("Form Acquisition Points:", mod_label),
      subtitle = "Red dashed lines show 50/50 preference crossover points by linguistic form",
      x = "Training Checkpoint",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold"
    )
  
  # Save plot
  safe_name <- gsub(" ", "_", tolower(mod_label))
  model_folder <- case_when(
    mod == "exp0_baseline" ~ "baseline",
    mod == "exp1_remove_expletives" ~ "remove_expletives", 
    mod == "exp2_impoverish_determiners" ~ "impoverish_determiners",
    mod == "exp3_remove_articles" ~ "remove_articles",
    mod == "exp4_lemmatize_verbs" ~ "lemmatize_verbs",
    mod == "exp5_remove_subject_pronominals" ~ "remove_subject_pronominals"
  )
  ggsave(paste0("analysis/figures/", model_folder, "/form_acquisition.pdf"), 
         p_forms, width = 12, height = 8)
  ggsave(paste0("analysis/figures/", model_folder, "/form_acquisition.png"), 
         p_forms, width = 12, height = 8, dpi = 300)
}

cat("All acquisition figures created!\n\n")

# ============================================================================
# DESCRIPTIVE STATISTICS TABLES
# ============================================================================

cat("Generating descriptive statistics tables...\n")

# Helper function to calculate preference statistics  
calc_preference_stats <- function(df) {
  df %>%
    group_by(form_type, .add = TRUE) %>%  # .add = TRUE preserves existing grouping
    summarise(
      n_obs = n(),
      prop_correct = mean(correct, na.rm = TRUE),
      se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
      mean_surprisal = mean(mean_surprisal, na.rm = TRUE),
      sd_surprisal = sd(mean_surprisal, na.rm = TRUE),
      mean_surprisal_diff = mean(surprisal_difference, na.rm = TRUE),
      sd_surprisal_diff = sd(surprisal_difference, na.rm = TRUE),
      .groups = "drop_last"  # Keep original grouping, drop form_type
    ) %>%
    pivot_wider(
      names_from = form_type,
      values_from = c(n_obs, prop_correct, se_correct, mean_surprisal, 
                     sd_surprisal, mean_surprisal_diff, sd_surprisal_diff)
    ) %>%
    mutate(
      # Calculate preference difference (null - overt)
      preference_diff = prop_correct_null - prop_correct_overt,
      surprisal_diff = mean_surprisal_overt - mean_surprisal_null  # Higher overt surprisal = null preference
    ) %>%
    ungroup()
}

# ----------------------------------------------------------------------------
# Table 1: Overall Model Preferences
# ----------------------------------------------------------------------------

table1_model_prefs <- data %>%
  group_by(model, model_label) %>%
  calc_preference_stats() %>%
  arrange(model) %>%
  dplyr::select(
    Model = model_label,
    `Null Preference` = prop_correct_null,
    `Null SE` = se_correct_null,
    `Overt Preference` = prop_correct_overt,
    `Overt SE` = se_correct_overt,
    `Preference Diff` = preference_diff,
    `Null Surprisal` = mean_surprisal_null,
    `Overt Surprisal` = mean_surprisal_overt,
    `Surprisal Diff` = surprisal_diff
  ) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

# Calculate crossover points by item group
itemgroup_crossover <- itemgroup_summary %>%
  group_by(model, model_label, item_group, checkpoint_num) %>%
  summarise(
    null_pref = mean(mean_correct[form_type == "null"]),
    .groups = "drop"
  ) %>%
  group_by(model, model_label, item_group) %>%
  summarise(
    crossover_checkpoint = find_crossover_point(cur_data()),
    .groups = "drop"
  )

# Calculate crossover points by form
form_crossover <- form_summary %>%
  group_by(model, model_label, form, checkpoint_num) %>%
  summarise(
    null_pref = mean(mean_correct[form_type == "null"]),
    .groups = "drop"
  ) %>%
  group_by(model, model_label, form) %>%
  summarise(
    crossover_checkpoint = find_crossover_point(cur_data()),
    .groups = "drop"
  )

# Calculate crossover points by form within item group
form_itemgroup_crossover <- form_itemgroup_summary %>%
  group_by(model, model_label, item_group, form, checkpoint_num) %>%
  summarise(
    null_pref = mean(mean_correct[form_type == "null"]),
    .groups = "drop"
  ) %>%
  group_by(model, model_label, item_group, form) %>%
  summarise(
    crossover_checkpoint = find_crossover_point(cur_data()),
    .groups = "drop"
  )

# Save crossover analysis
write.csv(overall_crossover, "analysis/tables/acquisition_points_overall.csv", row.names = FALSE)
write.csv(itemgroup_crossover, "analysis/tables/acquisition_points_by_itemgroup.csv", row.names = FALSE)
write.csv(form_crossover, "analysis/tables/acquisition_points_by_form.csv", row.names = FALSE)
write.csv(form_itemgroup_crossover, "analysis/tables/acquisition_points_by_form_itemgroup.csv", row.names = FALSE)

cat("Acquisition points calculated and saved\n")

cat("All acquisition figures created!\n\n")

# Continue to the descriptive statistics and understanding sections below...

# Note: Some duplicate code sections exist below but have been left intact for safety
# The main analysis should work from the sections above

# Skip to the working descriptive statistics section below...

# ============================================================================
# DESCRIPTIVE STATISTICS TABLES
# ============================================================================

cat("Generating descriptive statistics tables...\n")

# Helper function to calculate preference statistics  
calc_preference_stats <- function(df) {
  df %>%
    group_by(form_type, .add = TRUE) %>%  # .add = TRUE preserves existing grouping
    summarise(
      n_obs = n(),
      prop_correct = mean(correct, na.rm = TRUE),
      se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
      mean_surprisal = mean(mean_surprisal, na.rm = TRUE),
      sd_surprisal = sd(mean_surprisal, na.rm = TRUE),
      mean_surprisal_diff = mean(surprisal_difference, na.rm = TRUE),
      sd_surprisal_diff = sd(surprisal_difference, na.rm = TRUE),
      .groups = "drop_last"  # Keep original grouping, drop form_type
    ) %>%
    pivot_wider(
      names_from = form_type,
      values_from = c(n_obs, prop_correct, se_correct, mean_surprisal, 
                     sd_surprisal, mean_surprisal_diff, sd_surprisal_diff)
    ) %>%
    mutate(
      # Calculate preference difference (null - overt)
      preference_diff = prop_correct_null - prop_correct_overt,
      surprisal_diff = mean_surprisal_overt - mean_surprisal_null  # Higher overt surprisal = null preference
    ) %>%
    ungroup()
}

# ----------------------------------------------------------------------------
# Table 1: Overall Model Preferences
# ----------------------------------------------------------------------------

table1_model_prefs <- data %>%
  group_by(model, model_label) %>%
  calc_preference_stats() %>%
  arrange(model) %>%
  dplyr::select(
    Model = model_label,
    `Null Preference` = prop_correct_null,
    `Null SE` = se_correct_null,
    `Overt Preference` = prop_correct_overt,
    `Overt SE` = se_correct_overt,
    `Preference Diff` = preference_diff,
    `Null Surprisal` = mean_surprisal_null,
    `Overt Surprisal` = mean_surprisal_overt,
    `Surprisal Diff` = surprisal_diff
  ) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

# Save as CSV
write.csv(table1_model_prefs, "analysis/tables/table1_model_preferences.csv", 
          row.names = FALSE)

# Save as LaTeX
kable(table1_model_prefs, format = "latex", booktabs = TRUE,
      caption = "Overall null vs overt subject preferences by model") %>%
  kable_styling(latex_options = c("striped", "scale_down")) %>%
  save_kable("analysis/tables/table1_model_preferences.tex")

cat("Table 1 (Model Preferences) saved\n")

# ============================================================================
# LOG-TRANSFORMED FIGURES FOR BETTER LEARNING CURVE VISUALIZATION
# ============================================================================

cat("Creating log-transformed versions of figures...\n")

# ----------------------------------------------------------------------------
# Log-transformed Item Group Figures
# ----------------------------------------------------------------------------

for (mod in unique(itemgroup_summary$model)) {
  mod_label <- unique(itemgroup_summary$model_label[itemgroup_summary$model == mod])
  mod_data <- itemgroup_summary %>% filter(model == mod)
  
  # Get crossover data for this model and log transform the checkpoint positions
  mod_crossover <- itemgroup_crossover %>% 
    filter(model == mod) %>%
    mutate(crossover_checkpoint_log = log10(crossover_checkpoint + 1))  # +1 to handle checkpoint 0
  
  # Add log-transformed checkpoint to data
  mod_data <- mod_data %>%
    mutate(checkpoint_num_log = log10(checkpoint_num + 1))
  
  p_itemgroup_log <- ggplot(mod_data, 
                           aes(x = checkpoint_num_log, y = mean_correct, 
                               color = form_type, fill = form_type)) +
    geom_line(size = 0.8) +
    geom_ribbon(aes(ymin = mean_correct - se_correct, 
                    ymax = mean_correct + se_correct),
                alpha = 0.2, color = NA) +
    # Add log-transformed acquisition lines
    geom_vline(data = mod_crossover, 
               aes(xintercept = crossover_checkpoint_log),
               linetype = "dashed", color = "red", size = 0.7, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
    facet_wrap(~ item_group, scales = "free_x") +
    scale_color_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                       labels = c("0", "10", "100", "1K", "10K")) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      axis.text = element_text(size = 9),
      axis.title = element_text(size = 11)
    ) +
    labs(
      title = paste("Item Group Acquisition (Log Scale):", mod_label),
      subtitle = "Red dashed lines show 50/50 preference crossover points by item group",
      x = "Training Checkpoint (Log Scale)",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold"
    )
  
  # Save log-transformed plot
  safe_name <- gsub(" ", "_", tolower(mod_label))
  model_folder <- case_when(
    mod == "exp0_baseline" ~ "baseline",
    mod == "exp1_remove_expletives" ~ "remove_expletives", 
    mod == "exp2_impoverish_determiners" ~ "impoverish_determiners",
    mod == "exp3_remove_articles" ~ "remove_articles",
    mod == "exp4_lemmatize_verbs" ~ "lemmatize_verbs",
    mod == "exp5_remove_subject_pronominals" ~ "remove_subject_pronominals"
  )
  ggsave(paste0("analysis/figures/", model_folder, "/itemgroup_acquisition_log.pdf"), 
         p_itemgroup_log, width = 12, height = 8)
  ggsave(paste0("analysis/figures/", model_folder, "/itemgroup_acquisition_log.png"), 
         p_itemgroup_log, width = 12, height = 8, dpi = 300)
}

# ----------------------------------------------------------------------------
# Log-transformed Form-only Figures  
# ----------------------------------------------------------------------------

for (mod in unique(form_summary$model)) {
  mod_label <- unique(form_summary$model_label[form_summary$model == mod])
  mod_data <- form_summary %>% filter(model == mod)
  
  # Get crossover data for this model and log transform
  mod_crossover <- form_crossover %>% 
    filter(model == mod) %>%
    mutate(crossover_checkpoint_log = log10(crossover_checkpoint + 1))
  
  # Add log-transformed checkpoint to data
  mod_data <- mod_data %>%
    mutate(checkpoint_num_log = log10(checkpoint_num + 1))
  
  p_forms_log <- ggplot(mod_data, 
                       aes(x = checkpoint_num_log, y = mean_correct, 
                           color = form_type, fill = form_type)) +
    geom_line(size = 0.8) +
    geom_ribbon(aes(ymin = mean_correct - se_correct, 
                    ymax = mean_correct + se_correct),
                alpha = 0.2, color = NA) +
    # Add log-transformed acquisition lines
    geom_vline(data = mod_crossover, 
               aes(xintercept = crossover_checkpoint_log),
               linetype = "dashed", color = "red", size = 0.7, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
    facet_wrap(~ form, scales = "free_x") +
    scale_color_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                       labels = c("0", "10", "100", "1K", "10K")) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      axis.text = element_text(size = 9),
      axis.title = element_text(size = 11)
    ) +
    labs(
      title = paste("Form Acquisition (Log Scale):", mod_label),
      subtitle = "Red dashed lines show 50/50 preference crossover points by linguistic form",
      x = "Training Checkpoint (Log Scale)",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold"
    )
  
  # Save log-transformed plot
  safe_name <- gsub(" ", "_", tolower(mod_label))
  model_folder <- case_when(
    mod == "exp0_baseline" ~ "baseline",
    mod == "exp1_remove_expletives" ~ "remove_expletives", 
    mod == "exp2_impoverish_determiners" ~ "impoverish_determiners",
    mod == "exp3_remove_articles" ~ "remove_articles",
    mod == "exp4_lemmatize_verbs" ~ "lemmatize_verbs",
    mod == "exp5_remove_subject_pronominals" ~ "remove_subject_pronominals"
  )
  ggsave(paste0("analysis/figures/", model_folder, "/form_acquisition_log.pdf"), 
         p_forms_log, width = 12, height = 8)
  ggsave(paste0("analysis/figures/", model_folder, "/form_acquisition_log.png"), 
         p_forms_log, width = 12, height = 8, dpi = 300)
}

# ============================================================================
# INDIVIDUAL MODEL DETAILED FIGURES - NEW LAYOUTS
# ============================================================================

cat("Creating individual model detailed figures...\n")

# ----------------------------------------------------------------------------
# Type 1: Simple null vs overt for each model (collapsed across items/forms)
# ----------------------------------------------------------------------------

for (mod in unique(data$model)) {
  mod_label <- unique(data$model_label[data$model == mod])
  
  # Collapse across items and forms
  simple_data <- data %>%
    filter(model == mod) %>%
    group_by(checkpoint_num, form_type) %>%
    summarise(
      mean_correct = mean(correct, na.rm = TRUE),
      se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
      ci_lower = mean_correct - 1.96 * se_correct,
      ci_upper = mean_correct + 1.96 * se_correct,
      .groups = "drop"
    )
  
  # Get overall crossover for this model
  mod_crossover <- overall_crossover %>% filter(model == mod)
  
  p_simple <- ggplot(simple_data, 
                    aes(x = checkpoint_num, y = mean_correct, 
                        color = form_type, fill = form_type)) +
    geom_line(size = 1.0) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                alpha = 0.2, color = NA) +
    # Add acquisition line
    geom_vline(data = mod_crossover, 
               aes(xintercept = crossover_checkpoint),
               linetype = "dashed", color = "red", size = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
    facet_wrap(~ form_type, labeller = labeller(form_type = c("null" = "Null Subject", 
                                                              "overt" = "Overt Subject"))) +
    scale_color_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    theme_bw() +
    theme(
      legend.position = "none",
      strip.text = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 11)
    ) +
    labs(
      title = paste("Simple Acquisition Overview:", mod_label),
      subtitle = "Collapsed across all items and forms. Dashed line = 50/50 acquisition point.",
      x = "Training Checkpoint",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold"
    )
  
  # Save simple plot
  safe_name <- gsub(" ", "_", tolower(mod_label))
  model_folder <- case_when(
    mod == "exp0_baseline" ~ "baseline",
    mod == "exp1_remove_expletives" ~ "remove_expletives", 
    mod == "exp2_impoverish_determiners" ~ "impoverish_determiners",
    mod == "exp3_remove_articles" ~ "remove_articles",
    mod == "exp4_lemmatize_verbs" ~ "lemmatize_verbs",
    mod == "exp5_remove_subject_pronominals" ~ "remove_subject_pronominals"
  )
  ggsave(paste0("analysis/figures/", model_folder, "/simple_acquisition.pdf"), 
         p_simple, width = 10, height = 6)
  ggsave(paste0("analysis/figures/", model_folder, "/simple_acquisition.png"), 
         p_simple, width = 10, height = 6, dpi = 300)
}

# ----------------------------------------------------------------------------
# Type 2: Item groups faceted vertically for each model
# ----------------------------------------------------------------------------

for (mod in unique(itemgroup_summary$model)) {
  mod_label <- unique(itemgroup_summary$model_label[itemgroup_summary$model == mod])
  mod_data <- itemgroup_summary %>% filter(model == mod)
  
  # Get crossover data for this model
  mod_crossover <- itemgroup_crossover %>% filter(model == mod)
  
  p_itemgroups_vertical <- ggplot(mod_data, 
                                 aes(x = checkpoint_num, y = mean_correct, 
                                     color = form_type, fill = form_type)) +
    geom_line(size = 0.8) +
    geom_ribbon(aes(ymin = mean_correct - se_correct, 
                    ymax = mean_correct + se_correct),
                alpha = 0.2, color = NA) +
    # Add acquisition lines
    geom_vline(data = mod_crossover, 
               aes(xintercept = crossover_checkpoint, color = item_group),
               linetype = "dashed", size = 0.6, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
    facet_grid(item_group ~ form_type, 
               labeller = labeller(form_type = c("null" = "Null Subject", 
                                                 "overt" = "Overt Subject"))) +
    scale_color_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    theme_bw() +
    theme(
      legend.position = "none",
      strip.text = element_text(size = 9, face = "bold"),
      axis.text = element_text(size = 8),
      axis.title = element_text(size = 10),
      panel.spacing = unit(0.2, "lines")
    ) +
    labs(
      title = paste("Item Group Acquisition:", mod_label),
      subtitle = "Item groups stacked vertically, null vs overt side-by-side. Dashed lines = item-specific 50/50 points.",
      x = "Training Checkpoint",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold"
    )
  
  # Save item groups plot
  safe_name <- gsub(" ", "_", tolower(mod_label))
  model_folder <- case_when(
    mod == "exp0_baseline" ~ "baseline",
    mod == "exp1_remove_expletives" ~ "remove_expletives", 
    mod == "exp2_impoverish_determiners" ~ "impoverish_determiners",
    mod == "exp3_remove_articles" ~ "remove_articles",
    mod == "exp4_lemmatize_verbs" ~ "lemmatize_verbs",
    mod == "exp5_remove_subject_pronominals" ~ "remove_subject_pronominals"
  )
  ggsave(paste0("analysis/figures/", model_folder, "/itemgroups_vertical.pdf"), 
         p_itemgroups_vertical, width = 8, height = 14)
  ggsave(paste0("analysis/figures/", model_folder, "/itemgroups_vertical.png"), 
         p_itemgroups_vertical, width = 8, height = 14, dpi = 300)
}

# ----------------------------------------------------------------------------  
# Type 3: Forms faceted vertically for each model
# ----------------------------------------------------------------------------

for (mod in unique(form_summary$model)) {
  mod_label <- unique(form_summary$model_label[form_summary$model == mod])
  mod_data <- form_summary %>% filter(model == mod)
  
  # Get crossover data for this model
  mod_crossover <- form_crossover %>% filter(model == mod)
  
  p_forms_vertical <- ggplot(mod_data, 
                            aes(x = checkpoint_num, y = mean_correct, 
                                color = form_type, fill = form_type)) +
    geom_line(size = 0.8) +
    geom_ribbon(aes(ymin = mean_correct - se_correct, 
                    ymax = mean_correct + se_correct),
                alpha = 0.2, color = NA) +
    # Add acquisition lines
    geom_vline(data = mod_crossover, 
               aes(xintercept = crossover_checkpoint, color = form),
               linetype = "dashed", size = 0.6, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
    facet_grid(form ~ form_type, 
               labeller = labeller(form_type = c("null" = "Null Subject", 
                                                 "overt" = "Overt Subject"))) +
    scale_color_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = "#2E86AB", "overt" = "#A23B72"),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    theme_bw() +
    theme(
      legend.position = "none",
      strip.text = element_text(size = 9, face = "bold"),
      axis.text = element_text(size = 8),
      axis.title = element_text(size = 10),
      panel.spacing = unit(0.2, "lines")
    ) +
    labs(
      title = paste("Form Acquisition:", mod_label),
      subtitle = "Linguistic forms stacked vertically, null vs overt side-by-side. Dashed lines = form-specific 50/50 points.",
      x = "Training Checkpoint",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold"
    )
  
  # Save forms plot
  safe_name <- gsub(" ", "_", tolower(mod_label))
  model_folder <- case_when(
    mod == "exp0_baseline" ~ "baseline",
    mod == "exp1_remove_expletives" ~ "remove_expletives", 
    mod == "exp2_impoverish_determiners" ~ "impoverish_determiners",
    mod == "exp3_remove_articles" ~ "remove_articles",
    mod == "exp4_lemmatize_verbs" ~ "lemmatize_verbs",
    mod == "exp5_remove_subject_pronominals" ~ "remove_subject_pronominals"
  )
  ggsave(paste0("analysis/figures/", model_folder, "/forms_vertical.pdf"), 
         p_forms_vertical, width = 8, height = 12)
  ggsave(paste0("analysis/figures/", model_folder, "/forms_vertical.png"), 
         p_forms_vertical, width = 8, height = 12, dpi = 300)
}

cat("Individual model detailed figures saved\n")

# ============================================================================
# BASELINE COMPARISON FIGURES (FOR NON-BASELINE MODELS ONLY)
# ============================================================================

cat("Creating baseline comparison figures...\n")

# Get baseline data
baseline_comparison_data <- model_comparison_data %>%
  filter(model_label %in% c("Baseline", "Remove Expletives", "Impoverish Determiners", 
                            "Remove Articles", "Lemmatize Verbs", "Remove Subject Pronominals"))

baseline_crossover_data <- model_comparison_crossover %>%
  filter(model_label %in% c("Baseline", "Remove Expletives", "Impoverish Determiners", 
                            "Remove Articles", "Lemmatize Verbs", "Remove Subject Pronominals"))

# Create comparison figure for each non-baseline model
non_baseline_models <- c("Remove Expletives", "Impoverish Determiners", "Remove Articles", 
                        "Lemmatize Verbs", "Remove Subject Pronominals")

non_baseline_folders <- c("remove_expletives", "impoverish_determiners", "remove_articles",
                         "lemmatize_verbs", "remove_subject_pronominals")

for (i in seq_along(non_baseline_models)) {
  target_model <- non_baseline_models[i]
  target_folder <- non_baseline_folders[i]
  
  # Filter data for baseline + target model
  comparison_data <- baseline_comparison_data %>%
    filter(model_label %in% c("Baseline", target_model))
  
  comparison_crossover <- baseline_crossover_data %>%
    filter(model_label %in% c("Baseline", target_model))
  
  # Create comparison plot
  p_baseline_comparison <- ggplot(comparison_data, 
                                 aes(x = checkpoint_num, y = mean_correct, 
                                     color = model_label, fill = model_label, linetype = form_type)) +
    geom_line(size = 0.8) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                alpha = 0.15, color = NA) +
    # Add acquisition lines
    geom_vline(data = comparison_crossover, 
               aes(xintercept = crossover_checkpoint, color = model_label),
               linetype = "dashed", size = 0.7, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
    scale_color_manual(values = c("Baseline" = "#1f77b4", 
                                  "Remove Expletives" = "#ff7f0e",
                                  "Impoverish Determiners" = "#2ca02c", 
                                  "Remove Articles" = "#d62728",
                                  "Lemmatize Verbs" = "#9467bd",
                                  "Remove Subject Pronominals" = "#8c564b")) +
    scale_fill_manual(values = c("Baseline" = "#1f77b4", 
                                 "Remove Expletives" = "#ff7f0e",
                                 "Impoverish Determiners" = "#2ca02c", 
                                 "Remove Articles" = "#d62728",
                                 "Lemmatize Verbs" = "#9467bd",
                                 "Remove Subject Pronominals" = "#8c564b")) +
    scale_linetype_manual(values = c("null" = "dotted", "overt" = "solid"),
                         labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 11)
    ) +
    guides(
      color = guide_legend(title = "Model", order = 1),
      fill = guide_legend(title = "Model", order = 1),
      linetype = guide_legend(title = "Form Type", order = 2)
    ) +
    labs(
      title = paste("Baseline vs", target_model, "Comparison"),
      subtitle = "Direct comparison of null and overt subject acquisition. Ribbons show 95% CIs. Dashed lines = 50/50 acquisition points.",
      x = "Training Checkpoint",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold. Dotted=null, solid=overt"
    )
  
  # Save comparison plot
  ggsave(paste0("analysis/figures/", target_folder, "/baseline_comparison.pdf"), 
         p_baseline_comparison, width = 12, height = 8)
  ggsave(paste0("analysis/figures/", target_folder, "/baseline_comparison.png"), 
         p_baseline_comparison, width = 12, height = 8, dpi = 300)
  
  # Also create log scale version
  comparison_data_log <- comparison_data %>%
    mutate(checkpoint_num_log = log10(checkpoint_num + 1))
  
  comparison_crossover_log <- comparison_crossover %>%
    mutate(crossover_checkpoint_log = log10(crossover_checkpoint + 1))
  
  p_baseline_comparison_log <- ggplot(comparison_data_log, 
                                     aes(x = checkpoint_num_log, y = mean_correct, 
                                         color = model_label, fill = model_label, linetype = form_type)) +
    geom_line(size = 0.8) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                alpha = 0.15, color = NA) +
    # Add log-transformed acquisition lines
    geom_vline(data = comparison_crossover_log, 
               aes(xintercept = crossover_checkpoint_log, color = model_label),
               linetype = "dashed", size = 0.7, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
    scale_color_manual(values = c("Baseline" = "#1f77b4", 
                                  "Remove Expletives" = "#ff7f0e",
                                  "Impoverish Determiners" = "#2ca02c", 
                                  "Remove Articles" = "#d62728",
                                  "Lemmatize Verbs" = "#9467bd",
                                  "Remove Subject Pronominals" = "#8c564b")) +
    scale_fill_manual(values = c("Baseline" = "#1f77b4", 
                                 "Remove Expletives" = "#ff7f0e",
                                 "Impoverish Determiners" = "#2ca02c", 
                                 "Remove Articles" = "#d62728",
                                 "Lemmatize Verbs" = "#9467bd",
                                 "Remove Subject Pronominals" = "#8c564b")) +
    scale_linetype_manual(values = c("null" = "dotted", "overt" = "solid"),
                         labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                       labels = c("0", "10", "100", "1K", "10K")) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 11)
    ) +
    guides(
      color = guide_legend(title = "Model", order = 1),
      fill = guide_legend(title = "Model", order = 1),
      linetype = guide_legend(title = "Form Type", order = 2)
    ) +
    labs(
      title = paste("Baseline vs", target_model, "Comparison (Log Scale)"),
      subtitle = "Direct comparison of null and overt subject acquisition. Ribbons show 95% CIs. Dashed lines = 50/50 acquisition points.",
      x = "Training Checkpoint (Log Scale)",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold. Dotted=null, solid=overt"
    )
  
  # Save log scale comparison plot
  ggsave(paste0("analysis/figures/", target_folder, "/baseline_comparison_log.pdf"), 
         p_baseline_comparison_log, width = 12, height = 8)
  ggsave(paste0("analysis/figures/", target_folder, "/baseline_comparison_log.png"), 
         p_baseline_comparison_log, width = 12, height = 8, dpi = 300)
}

cat("Baseline comparison figures saved\n")

cat("Updated figures with acquisition lines saved\n")
cat("Acquisition analysis complete!\n\n")

# ============================================================================
# MISSING DESCRIPTIVE STATISTICS TABLES (Tables 2-5)
# ============================================================================

cat("Generating additional descriptive statistics tables...\n")

# ----------------------------------------------------------------------------
# Table 2: Model × Item Group Preferences  
# ----------------------------------------------------------------------------

table2_model_itemgroup <- data %>%
  group_by(model, model_label, item_group) %>%
  calc_preference_stats() %>%
  arrange(model, item_group) %>%
  dplyr::select(
    Model = model_label,
    `Item Group` = item_group,
    `Null Preference` = prop_correct_null,
    `Overt Preference` = prop_correct_overt,
    `Preference Diff` = preference_diff,
    `N Observations` = n_obs_null
  ) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

# Save as CSV
write.csv(table2_model_itemgroup, "analysis/tables/table2_model_itemgroup_preferences.csv", 
          row.names = FALSE)

cat("Table 2 (Model × Item Group Preferences) saved\n")

# ----------------------------------------------------------------------------  
# Table 3: Model × Individual Items
# ----------------------------------------------------------------------------

table3_model_items <- data %>%
  group_by(model, model_label, item_id, item_group) %>%
  calc_preference_stats() %>%
  arrange(model, item_group, item_id) %>%
  dplyr::select(
    Model = model_label,
    `Item Group` = item_group,
    `Item ID` = item_id,
    `Null Preference` = prop_correct_null,
    `Overt Preference` = prop_correct_overt,
    `Preference Diff` = preference_diff
  ) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

# Save as CSV
write.csv(table3_model_items, "analysis/tables/table3_model_items_preferences.csv", 
          row.names = FALSE)

cat("Table 3 (Model × Individual Items) saved\n")

# ----------------------------------------------------------------------------
# Table 4: Model × Forms
# ----------------------------------------------------------------------------

table4_model_forms <- data %>%
  group_by(model, model_label, form) %>%
  calc_preference_stats() %>%
  arrange(model, form) %>%
  dplyr::select(
    Model = model_label,
    Form = form,
    `Null Preference` = prop_correct_null,
    `Overt Preference` = prop_correct_overt,
    `Preference Diff` = preference_diff,
    `N Observations` = n_obs_null
  ) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

# Save as CSV
write.csv(table4_model_forms, "analysis/tables/table4_model_forms_preferences.csv", 
          row.names = FALSE)

cat("Table 4 (Model × Forms) saved\n")

# ----------------------------------------------------------------------------
# Table 5: Item Group × Forms  
# ----------------------------------------------------------------------------

table5_itemgroup_forms <- data %>%
  group_by(item_group, form) %>%
  calc_preference_stats() %>%
  arrange(item_group, form) %>%
  dplyr::select(
    `Item Group` = item_group,
    Form = form,
    `Null Preference` = prop_correct_null,
    `Overt Preference` = prop_correct_overt,
    `Preference Diff` = preference_diff,
    `N Observations` = n_obs_null
  ) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

# Save as CSV
write.csv(table5_itemgroup_forms, "analysis/tables/table5_itemgroup_forms_preferences.csv", 
          row.names = FALSE)

cat("Table 5 (Item Group × Forms) saved\n")

# ============================================================================
# HOTSPOT SURPRISAL ANALYSIS
# ============================================================================

cat("Creating hotspot surprisal analysis...\n")

# Hotspot analysis disabled for now
# Create hotspot analysis function (DISABLED)
analyze_hotspot_surprisal_disabled <- function(data_subset, model_name, model_folder) {
  # Create hotspot folder within model folder
  hotspot_dir <- paste0("analysis/figures/", model_folder, "/hotspots/")
  dir.create(hotspot_dir, recursive = TRUE, showWarnings = FALSE)
  
  # This function assumes we have surprisal data for different positions
  # We'll simulate/structure this assuming the data contains:
  # - subject_surprisal, verb_surprisal, object_surprisal columns
  # - or we can calculate from token-level data
  
  # For now, we'll create a framework assuming surprisal data exists or can be extracted
  # You may need to modify this based on actual data structure
  
  # Check if we have hotspot surprisal columns
  has_hotspot_data <- all(c("subject_surprisal", "verb_surprisal") %in% names(data_subset))
  
  if (!has_hotspot_data) {
    cat("  Note: Hotspot surprisal data not found for", model_name, "- creating placeholder analysis\n")
    cat("  Available columns:", paste(names(data_subset), collapse = ", "), "\n")
    cat("  Form values:", paste(unique(data_subset$form), collapse = ", "), "\n")
    
    # Create simulated hotspot data for demonstration
    # In real implementation, this would extract from actual model outputs
    set.seed(42)  # For reproducible simulated data
    hotspot_data <- data_subset %>%
      dplyr::select(checkpoint_num, item_id, form, correct, model, model_label) %>%
      # Make sure we have both null and overt for each checkpoint/item combination
      distinct(checkpoint_num, item_id, form, .keep_all = TRUE) %>%
      mutate(
        # Simulate more realistic surprisal values based on form
        subject_surprisal = rnorm(n(), mean = 8 + ifelse(form == "null", 2, 0), sd = 1.5),
        verb_surprisal = rnorm(n(), mean = 6 + ifelse(form == "null", -0.5, 0.5), sd = 1.2),
        object_surprisal = rnorm(n(), mean = 7 + ifelse(form == "null", 0.3, -0.3), sd = 1.0)
      )
  } else {
    hotspot_data <- data_subset
  }
  
  # 1. SURPRISAL OVER TIME BY HOTSPOT
  hotspot_summary <- hotspot_data %>%
    group_by(checkpoint_num, form) %>%
    summarise(
      subject_mean = mean(subject_surprisal, na.rm = TRUE),
      subject_se = sd(subject_surprisal, na.rm = TRUE) / sqrt(n()),
      verb_mean = mean(verb_surprisal, na.rm = TRUE),
      verb_se = sd(verb_surprisal, na.rm = TRUE) / sqrt(n()),
      object_mean = mean(object_surprisal, na.rm = TRUE),
      object_se = sd(object_surprisal, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  # Reshape for plotting
  hotspot_long <- hotspot_summary %>%
    pivot_longer(
      cols = c(subject_mean, verb_mean, object_mean),
      names_to = "hotspot",
      values_to = "surprisal",
      names_pattern = "(.*)_mean"
    ) %>%
    left_join(
      hotspot_summary %>%
        pivot_longer(
          cols = c(subject_se, verb_se, object_se),
          names_to = "hotspot_se",
          values_to = "se",
          names_pattern = "(.*)_se"
        ) %>%
        mutate(hotspot_se = gsub("_se", "", hotspot_se)),
      by = c("checkpoint_num", "form", "hotspot" = "hotspot_se")
    )
  
  # Plot surprisal over time by hotspot
  # Check what form values actually exist
  form_values <- unique(hotspot_long$form)
  color_mapping <- setNames(c("#2E86AB", "#A23B72")[1:length(form_values)], form_values)
  
  p_hotspot_time <- ggplot(hotspot_long, aes(x = checkpoint_num, y = surprisal, 
                                            color = form, fill = form)) +
    geom_line(size = 0.8) +
    geom_ribbon(aes(ymin = surprisal - se, ymax = surprisal + se),
                alpha = 0.2, color = NA) +
    facet_wrap(~ hotspot, scales = "free_y", 
               labeller = labeller(hotspot = c(subject = "Subject Position",
                                              verb = "Verb Position", 
                                              object = "Object Position"))) +
    scale_color_manual(values = color_mapping) +
    scale_fill_manual(values = color_mapping) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      strip.text = element_text(size = 10, face = "bold")
    ) +
    labs(
      title = paste("Hotspot Surprisal Over Time:", model_name),
      subtitle = "Lower surprisal indicates higher model expectation",
      x = "Training Checkpoint",
      y = "Surprisal (bits)"
    )
  
  ggsave(paste0(hotspot_dir, "surprisal_over_time.pdf"), p_hotspot_time, 
         width = 12, height = 8)
  ggsave(paste0(hotspot_dir, "surprisal_over_time.png"), p_hotspot_time, 
         width = 12, height = 8, dpi = 300)
  
  # 2. BINARY PREFERENCE BY HOTSPOT (which form has lower surprisal)
  hotspot_preference <- hotspot_data %>%
    dplyr::select(checkpoint_num, item_id, form, subject_surprisal, verb_surprisal, object_surprisal) %>%
    pivot_wider(
      names_from = form,
      values_from = c(subject_surprisal, verb_surprisal, object_surprisal),
      names_sep = "_",
      values_fn = mean  # Handle duplicates by taking mean
    ) %>%
    mutate(
      subject_prefers_null = if_else(!is.na(subject_surprisal_null) & !is.na(subject_surprisal_overt),
                                    subject_surprisal_null < subject_surprisal_overt, NA),
      verb_prefers_null = if_else(!is.na(verb_surprisal_null) & !is.na(verb_surprisal_overt),
                                 verb_surprisal_null < verb_surprisal_overt, NA),
      object_prefers_null = if_else(!is.na(object_surprisal_null) & !is.na(object_surprisal_overt),
                                   object_surprisal_null < object_surprisal_overt, NA)
    ) %>%
    group_by(checkpoint_num) %>%
    summarise(
      subject_null_pref = mean(subject_prefers_null, na.rm = TRUE),
      verb_null_pref = mean(verb_prefers_null, na.rm = TRUE),
      object_null_pref = mean(object_prefers_null, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Reshape for plotting  
  preference_long <- hotspot_preference %>%
    pivot_longer(
      cols = c(subject_null_pref, verb_null_pref, object_null_pref),
      names_to = "hotspot",
      values_to = "null_preference",
      names_pattern = "(.*)_null_pref"
    ) %>%
    mutate(
      overt_preference = 1 - null_preference,
      hotspot = factor(hotspot, levels = c("subject", "verb", "object"),
                      labels = c("Subject Position", "Verb Position", "Object Position"))
    )
  
  # Plot binary preferences
  p_hotspot_binary <- ggplot(preference_long, aes(x = checkpoint_num)) +
    geom_line(aes(y = null_preference, color = "Null Preferred"), size = 1) +
    geom_line(aes(y = overt_preference, color = "Overt Preferred"), size = 1) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50") +
    facet_wrap(~ hotspot) +
    scale_color_manual(values = c("Null Preferred" = "#2E86AB", "Overt Preferred" = "#A23B72")) +
    scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      strip.text = element_text(size = 10, face = "bold")
    ) +
    labs(
      title = paste("Hotspot Preference Over Time:", model_name),
      subtitle = "Proportion of items where each form has lower surprisal (higher expectation)",
      x = "Training Checkpoint",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold"
    )
  
  ggsave(paste0(hotspot_dir, "binary_preference.pdf"), p_hotspot_binary, 
         width = 12, height = 8)
  ggsave(paste0(hotspot_dir, "binary_preference.png"), p_hotspot_binary, 
         width = 12, height = 8, dpi = 300)
  
  # 3. HOTSPOT COMPARISON HEATMAP
  hotspot_matrix <- hotspot_preference %>%
    mutate(
      checkpoint_bin = cut(checkpoint_num, breaks = 10, labels = FALSE)
    ) %>%
    group_by(checkpoint_bin) %>%
    summarise(
      Subject = mean(subject_null_pref, na.rm = TRUE),
      Verb = mean(verb_null_pref, na.rm = TRUE), 
      Object = mean(object_null_pref, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    pivot_longer(cols = c(Subject, Verb, Object), 
                names_to = "Hotspot", values_to = "Null_Preference")
  
  p_heatmap <- ggplot(hotspot_matrix, aes(x = checkpoint_bin, y = Hotspot, fill = Null_Preference)) +
    geom_tile() +
    scale_fill_gradient2(low = "#A23B72", mid = "white", high = "#2E86AB", 
                        midpoint = 0.5, labels = scales::percent_format(),
                        name = "Null\nPreference") +
    theme_minimal() +
    labs(
      title = paste("Hotspot Preference Heatmap:", model_name),
      subtitle = "Blue = prefers null subjects, Red = prefers overt subjects",
      x = "Training Progress (binned)",
      y = "Hotspot Position"
    )
  
  ggsave(paste0(hotspot_dir, "preference_heatmap.pdf"), p_heatmap, 
         width = 10, height = 6)
  ggsave(paste0(hotspot_dir, "preference_heatmap.png"), p_heatmap, 
         width = 10, height = 6, dpi = 300)
  
  cat("  Hotspot analysis saved for", model_name, "\n")
}

# Run hotspot analysis for each model
models_to_analyze <- unique(data$model)

for (mod in models_to_analyze) {
  mod_data <- data %>% filter(model == mod)
  mod_label <- unique(mod_data$model_label)
  
  # Determine model folder name
  model_folder <- case_when(
    mod == "exp0_baseline" ~ "baseline",
    mod == "exp1_remove_expletives" ~ "remove_expletives", 
    mod == "exp2_impoverish_determiners" ~ "impoverish_determiners",
    mod == "exp3_remove_articles" ~ "remove_articles",
    mod == "exp4_lemmatize_verbs" ~ "lemmatize_verbs",
    mod == "exp5_remove_subject_pronominals" ~ "remove_subject_pronominals",
    TRUE ~ gsub("[^a-z0-9_]", "_", tolower(mod))
  )
  
  # Run analysis (DISABLED)
  # analyze_hotspot_surprisal_disabled(mod_data, mod_label, model_folder)
}

cat("Hotspot analysis section skipped.\n\n")

# ============================================================================
# SUMMARY AND COMPLETION
# ============================================================================

cat(paste("\n", paste(rep("=", 50), collapse = ""), "\n", sep = ""))
cat("NULL SUBJECT ANALYSIS COMPLETE\n") 
cat(paste(paste(rep("=", 50), collapse = ""), "\n\n", sep = ""))

cat("Files saved:\n")
cat("- Figures: analysis/figures/\n")
cat("- Tables: analysis/tables/\n") 
cat("- CSV files: acquisition points and descriptive statistics\n\n")

cat("Analysis complete!\n")

