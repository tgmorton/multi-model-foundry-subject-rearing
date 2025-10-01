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
library(ggpattern)
library(patchwork)
library(cowplot)

# Load paper figure specifications
source("scripts/paper_figures/figure_dimensions.R")

# Create output directories if they don't exist
dir.create("analysis/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/figures/combined", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/figures/baseline", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/figures/remove_expletives", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/figures/impoverish_determiners", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/figures/remove_articles", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/figures/lemmatize_verbs", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/figures/remove_subject_pronominals", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/tables", recursive = TRUE, showWarnings = FALSE)

# Create paper figure directories
dir.create("analysis/paper_figures", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/paper_figures/main", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/paper_figures/wide", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/paper_figures/supplementary", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/figures/individual", recursive = TRUE, showWarnings = FALSE)

# Load data (only if not already in environment or if it's not a dataframe)
if (!exists("data") || !is.data.frame(data)) {
  cat("Loading data...\n")
  data <- read.csv("../evaluation/results/all_models_null_subject_lme4_ready.csv")
  cat("Data loaded from file.\n")
} else {
  cat("Data already exists in environment - skipping load.\n")
}

# Load t50 acquisition timing data
cat("Loading t50 acquisition timing data...\n")
t50_data <- read.csv("tables/tests/t50_by_model_robust.csv")
t50_data <- t50_data %>%
  mutate(
    t50_checkpoint_log = log10(t50_checkpoint + 1),
    ci_lower_log = log10(CI_lo + 1),
    ci_upper_log = log10(CI_hi + 1)
  )
cat("t50 data loaded.\n")

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
# Set form factor with correct ordering: default, complex_long, complex_emb, context_negation, target_negation, both_negation
data$form <- factor(data$form, levels = c("default", "complex_long", "complex_emb", "context_negation", "target_negation", "both_negation"))

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

# Create the preference plot with paper specifications
spec_regular <- get_figure_specs("regular")
p_preference <- ggplot(pref_summary, 
                       aes(x = checkpoint_num, y = mean_correct, 
                           color = form_type, fill = form_type)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = mean_correct - se_correct, 
                  ymax = mean_correct + se_correct),
              alpha = 0.2, color = NA) +
  facet_wrap(~ model_label, nrow = 2) +
  scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                     labels = c("Null Subject", "Overt Subject")) +
  scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                    labels = c("Null Subject", "Overt Subject")) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  paper_theme(spec_regular) +
  labs(
    title = "Null vs Overt Subject Preference Across Models",
    subtitle = "Proportion of items where each form is preferred (lower surprisal)",
    x = "Training Checkpoint",
    y = "Proportion Preferred"
  )

# Save both traditional and paper versions
ggsave("analysis/figures/combined/null_overt_preference_by_model.pdf", 
       p_preference, width = 12, height = 8)
ggsave("analysis/figures/combined/null_overt_preference_by_model.png", 
       p_preference, width = 12, height = 8, dpi = 300)

# Save paper version (Figure 1: Regular width for main text)
save_paper_figure(p_preference, "analysis/paper_figures/main/fig1_null_overt_preference", "regular")

cat("Overall preference plot saved to analysis/figures/\n")

# ============================================================================
# INDIVIDUAL MODEL FIGURES FOR PAPER (LOG SCALE)
# ============================================================================

cat("Creating individual model figures for paper...\n")

# Create individual model plots with smaller scale and log transformation
models_list <- unique(data$model_label)

for(model_name in models_list) {
  cat("  Creating figure for:", model_name, "\n")
  
  # Filter data for this model
  model_data <- data %>% filter(model_label == model_name)
  
  # Create log-transformed checkpoint variable
  model_data <- model_data %>%
    mutate(checkpoint_log = log10(checkpoint_num + 1))
  
  # Calculate summary statistics for this model
  model_summary <- model_data %>%
    group_by(checkpoint_log, form_type) %>%
    summarise(
      mean_correct = mean(correct, na.rm = TRUE),
      se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(
      ci_lower = mean_correct - 1.96 * se_correct,
      ci_upper = mean_correct + 1.96 * se_correct
    )
  
  # Calculate model-specific epoch marker (final checkpoint / 20)
  model_final_checkpoint <- max(model_data$checkpoint_num, na.rm = TRUE)
  model_epoch_marker <- log10((model_final_checkpoint / 20) + 1)
  
  # Create figure with less cramped appearance
  spec_regular_open <- get_figure_specs("regular")
  spec_regular_open$base_size <- 8  # Slightly larger font
  spec_regular_open$title_size <- 9
  spec_regular_open$height <- spec_regular_open$height * 0.9  # Less height reduction
  
  safe_model_name <- gsub("[^A-Za-z0-9]", "_", tolower(model_name))
  
  p_model <- ggplot(model_summary, aes(x = checkpoint_log, y = mean_correct, 
                                       color = form_type, fill = form_type)) +
    geom_line(linewidth = 0.5) +  # Thinner lines
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                alpha = 0.15, color = NA) +  # More subtle ribbons
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
    # Add model-specific epoch marker
    geom_vline(xintercept = model_epoch_marker, linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
    scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                       labels = c("Null", "Overt")) +
    scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                      labels = c("Null", "Overt")) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +  # Full 0-100% range
    scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                       labels = c("0", "10", "100", "1K", "10K")) +
    paper_theme(spec_regular_open) +
    theme(
      legend.position = "right",
      legend.justification = c(0, 0.5),
      legend.margin = margin(l = 2, unit = "pt"),
      legend.key.size = unit(8, "pt"),
      legend.text = element_text(size = 6),
      legend.title = element_blank(),
      plot.margin = margin(5, 25, 5, 5, "pt")  # More right margin for legend
    ) +
    labs(
      title = model_name,
      subtitle = "Null vs overt preference across training",
      x = "Training Step (Log)",
      y = "Preference"
    )
  
  # Save as paper figure with custom smaller specs
  dir.create(dirname(paste0("analysis/paper_figures/main/model_", safe_model_name)), recursive = TRUE, showWarnings = FALSE)
  
  for (format in c("pdf", "png")) {
    full_filename <- paste0("analysis/paper_figures/main/model_", safe_model_name, ".", format)
    
    if (format == "pdf") {
      ggsave(full_filename, p_model, 
             width = spec_regular_open$width, height = spec_regular_open$height, 
             units = "in", dpi = spec_regular_open$dpi,
             device = "pdf", useDingbats = FALSE)
    } else if (format == "png") {
      ggsave(full_filename, p_model, 
             width = spec_regular_open$width, height = spec_regular_open$height, 
             units = "in", dpi = spec_regular_open$dpi,
             device = "png", type = "cairo")
    }
  }
  
  # Also save traditional version
  ggsave(paste0("analysis/figures/individual/", safe_model_name, "_log.pdf"), 
         p_model, width = 5, height = 3)
  ggsave(paste0("analysis/figures/individual/", safe_model_name, "_log.png"), 
         p_model, width = 5, height = 3, dpi = 300)
}

cat("Individual model figures created.\n")

# ============================================================================
# WIDE LOG GRAPHS BY ITEM GROUP AND FORM FOR EACH MODEL
# ============================================================================

cat("Creating wide log graphs by item group and form for each model...\n")

# Create wide graphs for each model
for(model_name in models_list) {
  cat("  Creating wide graphs for:", model_name, "\n")
  
  # Filter data for this model
  model_data <- data %>% filter(model_label == model_name)
  model_data <- model_data %>%
    mutate(checkpoint_log = log10(checkpoint_num + 1))
  
  safe_model_name <- gsub("[^A-Za-z0-9]", "_", tolower(model_name))
  
  # Calculate model-specific epoch marker (final checkpoint / 20)
  model_final_checkpoint <- max(model_data$checkpoint_num, na.rm = TRUE)
  model_epoch_marker <- log10((model_final_checkpoint / 20) + 1)
  
  # Wide specs with legend positioned in margin space
  spec_wide_legend <- get_figure_specs("wide")
  spec_wide_legend$base_size <- 8
  spec_wide_legend$height <- spec_wide_legend$height * 1.25  # Taller for better visibility
  
  # 1. BY ITEM GROUP
  if ("item_group" %in% names(model_data)) {
    cat("    - Item group analysis for:", model_name, "\n")
    
    item_group_summary <- model_data %>%
      group_by(checkpoint_log, form_type, item_group) %>%
      summarise(
        mean_correct = mean(correct, na.rm = TRUE),
        se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
        .groups = "drop"
      ) %>%
      mutate(
        ci_lower = mean_correct - 1.96 * se_correct,
        ci_upper = mean_correct + 1.96 * se_correct
      )
    
    p_item_group <- ggplot(item_group_summary, aes(x = checkpoint_log, y = mean_correct, 
                                                   color = form_type, fill = form_type)) +
      geom_line(size = 0.4) +  # Thin lines for wide plots
      geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                  alpha = 0.1, color = NA) +  # Very subtle ribbons
      geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
      # Add model-specific epoch marker
      geom_vline(xintercept = model_epoch_marker, linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
      facet_wrap(~ item_group, ncol = 3, scales = "free_y") +
      scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                         labels = c("Null", "Overt")) +
      scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                        labels = c("Null", "Overt")) +
      scale_y_continuous(labels = percent_format()) +
      scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                         labels = c("0", "10", "100", "1K", "10K")) +
      paper_theme(spec_wide_legend) +
      theme(
        legend.position = c(0.98, 0.02),  # Position in bottom-right "margin space"
        legend.justification = c(1, 0),
        legend.background = element_rect(fill = "white", color = "gray90", size = 0.3),
        legend.key.size = unit(6, "pt"),
        legend.text = element_text(size = 5),
        legend.title = element_blank(),
        legend.margin = margin(2, 2, 2, 2, "pt"),
        strip.text = element_text(size = 6),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6)
      ) +
      labs(
        title = paste(model_name, "- By Item Group"),
        subtitle = "Log-scale preference by item group",
        x = "Training Step (Log)",
        y = "Preference"
      )
    
    # Save wide figure
    for (format in c("pdf", "png")) {
      full_filename <- paste0("analysis/paper_figures/wide/", safe_model_name, "_by_item_group.", format)
      dir.create(dirname(full_filename), recursive = TRUE, showWarnings = FALSE)
      
      if (format == "pdf") {
        ggsave(full_filename, p_item_group, 
               width = spec_wide_legend$width, height = spec_wide_legend$height, 
               units = "in", dpi = spec_wide_legend$dpi,
               device = "pdf", useDingbats = FALSE)
      } else {
        ggsave(full_filename, p_item_group, 
               width = spec_wide_legend$width, height = spec_wide_legend$height, 
               units = "in", dpi = spec_wide_legend$dpi,
               device = "png", type = "cairo")
      }
    }
  }
  
  # 2. BY FORM
  if ("form" %in% names(model_data)) {
    cat("    - Form analysis for:", model_name, "\n")
    
    form_summary <- model_data %>%
      group_by(checkpoint_log, form_type, form) %>%
      summarise(
        mean_correct = mean(correct, na.rm = TRUE),
        se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
        .groups = "drop"
      ) %>%
      mutate(
        ci_lower = mean_correct - 1.96 * se_correct,
        ci_upper = mean_correct + 1.96 * se_correct
      )
    
    p_form <- ggplot(form_summary, aes(x = checkpoint_log, y = mean_correct, 
                                       color = form_type, fill = form_type)) +
      geom_line(size = 0.4) +  # Thin lines for wide plots
      geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                  alpha = 0.1, color = NA) +  # Very subtle ribbons
      geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
      # Add model-specific epoch marker
      geom_vline(xintercept = model_epoch_marker, linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
      facet_wrap(~ form, ncol = 3, scales = "free_y") +
      scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                         labels = c("Null", "Overt")) +
      scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                        labels = c("Null", "Overt")) +
      scale_y_continuous(labels = percent_format()) +
      scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                         labels = c("0", "10", "100", "1K", "10K")) +
      paper_theme(spec_wide_legend) +
      theme(
        legend.position = c(0.98, 0.02),  # Position in bottom-right "margin space"
        legend.justification = c(1, 0),
        legend.background = element_rect(fill = "white", color = "gray90", size = 0.3),
        legend.key.size = unit(6, "pt"),
        legend.text = element_text(size = 5),
        legend.title = element_blank(),
        legend.margin = margin(2, 2, 2, 2, "pt"),
        strip.text = element_text(size = 6),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6)
      ) +
      labs(
        title = paste(model_name, "- By Form"),
        subtitle = "Log-scale preference by linguistic form",
        x = "Training Step (Log)",
        y = "Preference"
      )
    
    # Save wide figure
    for (format in c("pdf", "png")) {
      full_filename <- paste0("analysis/paper_figures/wide/", safe_model_name, "_by_form.", format)
      dir.create(dirname(full_filename), recursive = TRUE, showWarnings = FALSE)
      
      if (format == "pdf") {
        ggsave(full_filename, p_form, 
               width = spec_wide_legend$width, height = spec_wide_legend$height, 
               units = "in", dpi = spec_wide_legend$dpi,
               device = "pdf", useDingbats = FALSE)
      } else {
        ggsave(full_filename, p_form, 
               width = spec_wide_legend$width, height = spec_wide_legend$height, 
               units = "in", dpi = spec_wide_legend$dpi,
               device = "png", type = "cairo")
      }
    }
  }
}

cat("Wide log graphs by item group and form created for all models.\n")

# ============================================================================
# ADDITIONAL LOG VARIANTS FOR PAPER
# ============================================================================

cat("Creating additional log-transformed figures for paper...\n")

# 1. COMBINED ALL MODELS LOG COMPARISON (WIDE)
cat("  Creating combined model comparison (log)...\n")

all_models_summary <- data %>%
  mutate(checkpoint_log = log10(checkpoint_num + 1)) %>%
  group_by(checkpoint_log, form_type, model_label) %>%
  summarise(
    mean_correct = mean(correct, na.rm = TRUE),
    se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    ci_lower = mean_correct - 1.96 * se_correct,
    ci_upper = mean_correct + 1.96 * se_correct
  )

spec_wide_log <- get_figure_specs("wide")
spec_wide_log$base_size <- 7
spec_wide_log$height <- spec_wide_log$height * 0.8

p_all_models_log <- ggplot(all_models_summary, aes(x = checkpoint_log, y = mean_correct,
                                                   color = model_label, fill = model_label)) +
  geom_line(size = 0.4, alpha = 0.9) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.08, color = NA) +
  # Note: 50/50 acquisition points will be added after crossover data is calculated
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
  facet_wrap(~ form_type, labeller = labeller(form_type = c("null" = "Null Subject", "overt" = "Overt Subject"))) +
  scale_color_manual(values = PAPER_COLORS$models[1:6]) +
  scale_fill_manual(values = PAPER_COLORS$models[1:6]) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1),
                     labels = c("0", "10", "100", "1K", "10K")) +
  paper_theme(spec_wide_log) +
  theme(
    legend.position = "bottom",
    legend.key.size = unit(4, "pt"),
    legend.text = element_text(size = 5),
    legend.title = element_blank(),
    legend.margin = margin(t = 2, unit = "pt"),
    strip.text = element_text(size = 7)
  ) +
  labs(
    title = "All Models Comparison (Log Scale)",
    subtitle = "Null vs overt preference across training. Dashed lines = 50/50 acquisition points.",
    x = "Training Step (Log)",
    y = "Preference"
  )

save_paper_figure(p_all_models_log, "analysis/paper_figures/wide/all_models_comparison_log", "wide")

# 2. ACQUISITION TIMING COMPARISON (LOG) - REGULAR WIDTH
cat("  Creating acquisition timing comparison (log)...\n")

# Calculate acquisition points for each model
acquisition_points <- data %>%
  mutate(checkpoint_log = log10(checkpoint_num + 1)) %>%
  group_by(model_label, checkpoint_num, checkpoint_log, form_type) %>%
  summarise(mean_correct = mean(correct, na.rm = TRUE), .groups = "drop") %>%
  group_by(model_label, form_type) %>%
  arrange(checkpoint_num) %>%
  mutate(
    crossed_50 = mean_correct >= 0.5,
    first_cross = !duplicated(crossed_50) & crossed_50
  ) %>%
  filter(first_cross) %>%
  ungroup()

spec_regular_timing <- get_figure_specs("regular")
spec_regular_timing$base_size <- 7

p_timing_log <- ggplot(acquisition_points, aes(x = checkpoint_log, y = model_label, 
                                               color = form_type, shape = form_type)) +
  geom_point(size = 2.5, alpha = 0.8) +
  geom_vline(xintercept = log10(c(100, 1000) + 1), linetype = "dotted", color = "gray70", size = 0.3) +
  scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                     labels = c("Null", "Overt")) +
  scale_shape_manual(values = c("null" = 16, "overt" = 17),
                     labels = c("Null", "Overt")) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1),
                     labels = c("0", "10", "100", "1K", "10K")) +
  paper_theme(spec_regular_timing) +
  theme(
    legend.position = c(0.85, 0.2),
    legend.key.size = unit(8, "pt"),
    legend.text = element_text(size = 6),
    legend.title = element_blank(),
    legend.background = element_rect(fill = "white", color = NA)
  ) +
  labs(
    title = "Acquisition Timing",
    subtitle = "First 50% crossing point",
    x = "Training Step (Log)",
    y = NULL
  )

save_paper_figure(p_timing_log, "analysis/paper_figures/main/acquisition_timing_log", "regular")

# 3. END-STATE PERFORMANCE LOG VIEW (REGULAR)
cat("  Creating end-state performance (log view)...\n")

end_state_data <- data %>%
  group_by(model_label) %>%
  filter(checkpoint_num >= quantile(checkpoint_num, 0.9)) %>%
  ungroup() %>%
  mutate(checkpoint_log = log10(checkpoint_num + 1))

end_state_summary <- end_state_data %>%
  group_by(model_label, form_type) %>%
  summarise(
    mean_correct = mean(correct, na.rm = TRUE),
    se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    ci_lower = mean_correct - 1.96 * se_correct,
    ci_upper = mean_correct + 1.96 * se_correct
  )

p_endstate <- ggplot(end_state_summary, aes(x = model_label, y = mean_correct, 
                                            fill = form_type)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7, alpha = 0.8) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                position = position_dodge(0.8), width = 0.2, size = 0.3) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
  scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                    labels = c("Null", "Overt")) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  coord_flip() +
  paper_theme(spec_regular_timing) +
  theme(
    legend.position = c(0.85, 0.9),
    legend.key.size = unit(8, "pt"),
    legend.text = element_text(size = 6),
    legend.title = element_blank(),
    legend.background = element_rect(fill = "white", color = NA),
    axis.text.y = element_text(size = 6)
  ) +
  labs(
    title = "End-State Performance",
    subtitle = "Final 10% of training",
    x = NULL,
    y = "Preference"
  )

save_paper_figure(p_endstate, "analysis/paper_figures/main/endstate_performance", "regular")

# 3B. BEGIN-STATE PERFORMANCE (FIRST 10% OF CHECKPOINTS)
cat("  Creating begin-state performance (first epoch)...\n")

# Calculate begin-state preferences (using first 10% of checkpoints)
begin_state_data <- data %>%
  group_by(model_label) %>%
  filter(checkpoint_num <= quantile(checkpoint_num, 0.1)) %>%
  ungroup() %>%
  group_by(model_label, form_type) %>%
  summarise(
    mean_correct = mean(correct, na.rm = TRUE),
    se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    ci_lower = mean_correct - 1.96 * se_correct,
    ci_upper = mean_correct + 1.96 * se_correct
  )

# Create begin-state performance plot with model colors
p_beginstate <- ggplot(begin_state_data, 
                      aes(x = model_label, y = mean_correct, fill = model_label, pattern = form_type)) +
  geom_bar_pattern(stat = "identity", position = position_dodge(width = 0.9), width = 0.8,
                   pattern_fill = "white", pattern_angle = 45, pattern_density = 0.1,
                   pattern_spacing = 0.025, pattern_key_scale_factor = 0.6) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                position = position_dodge(width = 0.9), width = 0.25, size = 0.3) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50") +
  scale_fill_manual(values = PAPER_COLORS$models[1:6], name = "Model") +
  scale_pattern_manual(values = c("null" = "stripe", "overt" = "none"),
                      name = "Subject Type",
                      labels = c("null" = "Null", "overt" = "Overt")) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  paper_theme(get_figure_specs("regular")) +
  theme(
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "none"
  ) +
  labs(
    title = "Begin-State Performance",
    subtitle = "First 10% of training",
    y = "Preference"
  )

# Save begin-state as separate figure
save_paper_figure(p_beginstate, "analysis/paper_figures/main/beginstate_performance", "regular")

# 3C. COMBINED BANNER GRAPHIC
cat("  Creating combined banner graphic...\n")

# Create begin-state plot with model colors and white hatching for null
p_beginstate_banner <- ggplot(begin_state_data, aes(x = model_label, y = mean_correct, 
                                                    fill = model_label, pattern = form_type)) +
  geom_bar_pattern(stat = "identity", position = position_dodge(0.8), width = 0.7, alpha = 0.8,
                   pattern_fill = "white", pattern_angle = 45, pattern_density = 0.1,
                   pattern_spacing = 0.025, pattern_key_scale_factor = 0.6) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                position = position_dodge(0.8), width = 0.2, size = 0.3) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
  scale_fill_manual(values = PAPER_COLORS$models[1:6], name = "Model") +
  scale_pattern_manual(values = c("null" = "stripe", "overt" = "none"),
                      name = "Subject Type") +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  coord_flip() +
  paper_theme(get_figure_specs("regular")) +
  theme(
    legend.position = "none",  # Remove legend for banner
    axis.text.y = element_text(size = 6),
    axis.text.x = element_blank(),   # Remove x-axis text
    axis.title.y = element_blank(),  # Remove y-axis label
    axis.title.x = element_blank()   # Remove x-axis label (preference)
  ) +
  labs(
    title = "Begin-State Performance",
    subtitle = "First 10% of training",
    x = NULL,
    y = NULL  # Remove y-axis label
  )

# Create end-state plot with model colors and white hatching for null
p_endstate_banner <- ggplot(end_state_summary, aes(x = model_label, y = mean_correct, 
                                                   fill = model_label, pattern = form_type)) +
  geom_bar_pattern(stat = "identity", position = position_dodge(0.8), width = 0.7, alpha = 0.8,
                   pattern_fill = "white", pattern_angle = 45, pattern_density = 0.1,
                   pattern_spacing = 0.025, pattern_key_scale_factor = 0.6) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                position = position_dodge(0.8), width = 0.2, size = 0.3) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
  scale_fill_manual(values = PAPER_COLORS$models[1:6], name = "Model") +
  scale_pattern_manual(values = c("null" = "stripe", "overt" = "none"),
                      name = "Subject Type") +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  coord_flip() +
  paper_theme(get_figure_specs("regular")) +
  theme(
    legend.position = "none",  # Remove legend for banner
    axis.text.y = element_text(size = 6),
    axis.text.x = element_blank(),   # Remove x-axis text
    axis.title.y = element_blank(),  # Remove y-axis label
    axis.title.x = element_blank()   # Remove x-axis label (preference)
  ) +
  labs(
    title = "End-State Performance", 
    subtitle = "Final 10% of training",
    x = NULL,
    y = NULL  # Remove y-axis label
  )

# Create trajectories plot without legend
p_trajectories_banner <- ggplot(all_models_summary, 
                               aes(x = checkpoint_log, y = mean_correct,
                                   color = model_label, fill = model_label)) +
  geom_line(size = 0.5, alpha = 0.9) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.08, color = NA) +
  # Add t50 confidence intervals as rectangles
  geom_rect(data = t50_data,
           aes(xmin = ci_lower_log, xmax = ci_upper_log, fill = Model),
           ymin = 0, ymax = 1, alpha = 0.1, inherit.aes = FALSE) +
  # Add 50/50 acquisition points
  geom_vline(data = t50_data, 
             aes(xintercept = t50_checkpoint_log, color = Model),
             linetype = "dashed", size = 0.5, alpha = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
  facet_wrap(~ form_type, labeller = labeller(form_type = c("null" = "Null Subject", "overt" = "Overt Subject"))) +
  scale_color_manual(values = PAPER_COLORS$models[1:6], name = "Model") +
  scale_fill_manual(values = PAPER_COLORS$models[1:6], name = "Model") +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1),
                     labels = c("0", "10", "100", "1K", "10K")) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "none",  # Remove legend for banner
    strip.text = element_text(size = 11, face = "bold"),
    axis.text = element_text(size = 9),
    axis.title = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1, "cm")
  ) +
  labs(
    title = "Subject Drop Acquisition Trajectories",
    x = "Training Step (Log)",
    y = "Preference"
  )

# Create unified legend with both model colors and subject patterns
p_legend_source <- ggplot(begin_state_data, aes(x = model_label, y = mean_correct, fill = model_label, pattern = form_type)) +
  geom_bar_pattern(stat = "identity", pattern_fill = "white", pattern_angle = 45, pattern_density = 0.1) +
  scale_fill_manual(values = PAPER_COLORS$models[1:6], name = "Model") +
  scale_pattern_manual(values = c("null" = "stripe", "overt" = "none"),
                      name = "Subject Type",
                      labels = c("null" = "Null", "overt" = "Overt")) +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.box = "horizontal",
        legend.text = element_text(size = 9),
        legend.title = element_text(size = 10, face = "bold")) +
  guides(fill = guide_legend(nrow = 1, order = 1),
         pattern = guide_legend(nrow = 1, order = 2))

# Extract legend
legend <- cowplot::get_legend(p_legend_source)
print("Legend extracted successfully")

# Combine into banner using patchwork
final_banner <- p_trajectories_banner / (p_beginstate_banner + p_endstate_banner) / legend +
  plot_layout(heights = c(1.2, 1.0, 0.5)) +
  plot_annotation(
    title = "Subject Drop Acquisition: Trajectories and Performance Comparison",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    )
  )

# Save banner graphic
ggsave("analysis/paper_figures/wide/banner_graphic.pdf", final_banner, 
       width = get_figure_specs("wide")$width * 1.5, 
       height = get_figure_specs("wide")$height * 2.5, 
       dpi = 300)
ggsave("analysis/paper_figures/wide/banner_graphic.png", final_banner, 
       width = get_figure_specs("wide")$width * 1.5, 
       height = get_figure_specs("wide")$height * 2.5, 
       dpi = 300)
# Create test version with timestamp
ggsave("analysis/paper_figures/wide/banner_graphic_TEST.pdf", final_banner, 
       width = get_figure_specs("wide")$width * 1.5, 
       height = get_figure_specs("wide")$height * 2.5, 
       dpi = 300)

cat("  Banner graphic created.\n")

# 4. FORM-SPECIFIC LOG TRAJECTORIES (MULTIPANEL WIDE)
cat("  Creating form-specific trajectories (log)...\n")

form_trajectories <- data %>%
  mutate(checkpoint_log = log10(checkpoint_num + 1)) %>%
  group_by(checkpoint_log, form_type, form, model_label) %>%
  summarise(
    mean_correct = mean(correct, na.rm = TRUE),
    .groups = "drop"
  )

spec_multipanel <- get_figure_specs("multipanel")
spec_multipanel$base_size <- 8  # Increased from 6

p_form_trajectories <- ggplot(form_trajectories, aes(x = checkpoint_log, y = mean_correct,
                                                     color = model_label)) +
  geom_line(size = 0.4, alpha = 0.8) +  # Slightly thicker lines
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
  facet_grid(form ~ form_type, 
             labeller = labeller(form_type = c("null" = "Null", "overt" = "Overt"))) +
  scale_color_manual(values = PAPER_COLORS$models[1:6]) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1),
                     labels = c("0", "10", "100", "1K", "10K")) +
  paper_theme(spec_multipanel) +
  theme(
    legend.position = "bottom",
    legend.key.size = unit(4, "pt"),  # Bigger legend keys
    legend.text = element_text(size = 6),  # Bigger legend text
    legend.title = element_blank(),
    legend.margin = margin(t = 2, unit = "pt"),
    strip.text = element_text(size = 7),  # Bigger facet labels
    strip.text.y = element_text(angle = 0),
    axis.text = element_text(size = 6),  # Bigger axis text
    axis.title = element_text(size = 7),  # Bigger axis titles
    panel.spacing = unit(0.15, "lines")
  ) +
  labs(
    title = "Form-Specific Trajectories",
    subtitle = "All models and linguistic forms",
    x = "Training Step (Log)",
    y = "Pref"
  )

save_paper_figure(p_form_trajectories, "analysis/paper_figures/wide/form_trajectories_log", "multipanel")

# 4B. ITEM GROUP-SPECIFIC LOG TRAJECTORIES (MULTIPANEL WIDE)
cat("  Creating item group-specific trajectories (log)...\n")

if ("item_group" %in% names(data)) {
  # Create item group trajectories data
  item_group_trajectories <- data %>%
    mutate(checkpoint_log = log10(checkpoint_num + 1)) %>%
    group_by(checkpoint_log, form_type, item_group, model_label) %>%
    summarise(
      mean_correct = mean(correct, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Create grouped categories as requested
  item_group_trajectories <- item_group_trajectories %>%
    mutate(
      item_group_category = case_when(
        item_group %in% c("1a_3rdSG", "1b_3rdPL", "2a_2ndSG", "2b_2ndPL", "3a_1stSg", "3b_1stPL") ~ "Person/Number",
        item_group %in% c("4a_subject_control", "4b_object_control") ~ "Control Contexts", 
        item_group %in% c("7a_conjunction_no_topic_shift", "7b_conjunction_topic_shift") ~ "Conjunction",
        item_group == "6_long_distance_binding" ~ "Long-Distance Binding",
        item_group %in% c("5a_expletive_seems", "5b_expletive_be") ~ "Expletives",
        TRUE ~ "Other"
      ),
      # Create readable item group names within categories
      item_group_readable = case_when(
        item_group == "1a_3rdSG" ~ "3rd Singular",
        item_group == "1b_3rdPL" ~ "3rd Plural", 
        item_group == "2a_2ndSG" ~ "2nd Singular",
        item_group == "2b_2ndPL" ~ "2nd Plural",
        item_group == "3a_1stSg" ~ "1st Singular",
        item_group == "3b_1stPL" ~ "1st Plural",
        item_group == "4a_subject_control" ~ "Subject Control",
        item_group == "4b_object_control" ~ "Object Control",
        item_group == "5a_expletive_seems" ~ "Expletive Seems",
        item_group == "5b_expletive_be" ~ "Expletive Be",
        item_group == "6_long_distance_binding" ~ "Long-Distance Binding",
        item_group == "7a_conjunction_no_topic_shift" ~ "No Topic Shift",
        item_group == "7b_conjunction_topic_shift" ~ "Topic Shift",
        TRUE ~ item_group
      )
    ) %>%
    # Set factor levels for proper ordering
    mutate(
      item_group_category = factor(item_group_category, 
                                 levels = c("Person/Number", "Control Contexts", "Conjunction", 
                                           "Long-Distance Binding", "Expletives")),
      item_group_readable = factor(item_group_readable,
                                 levels = c("3rd Singular", "3rd Plural", "2nd Singular", "2nd Plural", 
                                           "1st Singular", "1st Plural", "Subject Control", "Object Control",
                                           "No Topic Shift", "Topic Shift", "Long-Distance Binding", 
                                           "Expletive Seems", "Expletive Be"))
    )
  
  # Create separate plots for each linguistic category
  categories <- c("Person/Number", "Control Contexts", "Conjunction", "Long-Distance Binding", "Expletives")
  
  for(category in categories) {
    cat(paste("    Creating", category, "trajectories...\n"))
    
    # Filter data for this category
    category_data <- item_group_trajectories %>%
      filter(item_group_category == category)
    
    # Create safe filename
    safe_category_name <- gsub("[^A-Za-z0-9]", "_", tolower(category))
    
    # Calculate height: 1 inch per facet + 1.5 inches for legend/margins
    n_groups <- length(unique(category_data$item_group_readable))
    spec_category <- get_figure_specs("wide")
    spec_category$height <- (n_groups * 1.0) + 1.5  # 1 inch per facet + fixed space for legend
    spec_category$base_size <- 8
    cat(paste("      ->", n_groups, "groups, height =", spec_category$height, "inches\n"))
    
    p_category <- ggplot(category_data, 
                        aes(x = checkpoint_log, y = mean_correct, color = model_label)) +
      geom_line(size = 0.4, alpha = 0.8) +
      geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
      # First epoch marker at checkpoint 500
      geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.5) +
      facet_grid(item_group_readable ~ form_type, 
                 labeller = labeller(form_type = c("null" = "Null", "overt" = "Overt"))) +
      scale_color_manual(values = PAPER_COLORS$models[1:6]) +
      scale_y_continuous(labels = NULL, limits = c(0, 1)) +
      scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1),
                         labels = c("0", "10", "100", "1K", "10K")) +
      paper_theme(spec_category) +
      theme(
        strip.text.y = element_text(size = 7, angle = 0),
        strip.text.x = element_text(size = 8),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6),
        legend.position = "bottom",
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.4, "cm"),
        panel.spacing.y = unit(0.2, "cm"),
        panel.spacing.x = unit(0.3, "cm")
      ) +
      guides(color = guide_legend(title = NULL, nrow = 1)) +
      labs(
        title = paste(category, "Trajectories"),
        subtitle = "All models across training",
        x = "Training Step (Log)",
        y = "Preference"
      )
    
    # Save each category with custom height (not using save_paper_figure to preserve custom height)
    filename <- paste0("analysis/paper_figures/wide/item_group_trajectories_", safe_category_name)
    ggsave(paste0(filename, ".pdf"), p_category, 
           width = spec_category$width, height = spec_category$height, dpi = spec_category$dpi)
    ggsave(paste0(filename, ".png"), p_category, 
           width = spec_category$width, height = spec_category$height, dpi = spec_category$dpi)
  }
  
  # Create collapsed versions (aggregated across models)
  cat("  Creating collapsed item group trajectories (aggregated across models)...\n")
  
  # Create properly aggregated data across all models (smooth average trajectory)
  item_group_collapsed <- data %>%
    mutate(checkpoint_log = log10(checkpoint_num + 1)) %>%
    # Aggregate across all models for each checkpoint-item_group (simple average)
    group_by(checkpoint_log, form_type, item_group) %>%
    summarise(
      mean_correct = mean(correct, na.rm = TRUE),
      se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(
      # Much smaller confidence intervals based on the individual data points
      ci_lower = pmax(0, mean_correct - 1.96 * se_correct),
      ci_upper = pmin(1, mean_correct + 1.96 * se_correct),
      # Add the grouping variables back
      item_group_category = case_when(
        item_group %in% c("1a_3rdSG", "1b_3rdPL", "2a_2ndSG", "2b_2ndPL", "3a_1stSg", "3b_1stPL") ~ "Person/Number",
        item_group %in% c("4a_subject_control", "4b_object_control") ~ "Control Contexts", 
        item_group %in% c("7a_conjunction_no_topic_shift", "7b_conjunction_topic_shift") ~ "Conjunction",
        item_group == "6_long_distance_binding" ~ "Long-Distance Binding",
        item_group %in% c("5a_expletive_seems", "5b_expletive_be") ~ "Expletives",
        TRUE ~ "Other"
      ),
      item_group_readable = case_when(
        item_group == "1a_3rdSG" ~ "3rd Singular",
        item_group == "1b_3rdPL" ~ "3rd Plural", 
        item_group == "2a_2ndSG" ~ "2nd Singular",
        item_group == "2b_2ndPL" ~ "2nd Plural",
        item_group == "3a_1stSg" ~ "1st Singular",
        item_group == "3b_1stPL" ~ "1st Plural",
        item_group == "4a_subject_control" ~ "Subject Control",
        item_group == "4b_object_control" ~ "Object Control",
        item_group == "5a_expletive_seems" ~ "Expletive Seems",
        item_group == "5b_expletive_be" ~ "Expletive Be",
        item_group == "6_long_distance_binding" ~ "Long-Distance Binding",
        item_group == "7a_conjunction_no_topic_shift" ~ "No Topic Shift",
        item_group == "7b_conjunction_topic_shift" ~ "Topic Shift",
        TRUE ~ item_group
      )
    ) %>%
    mutate(
      item_group_category = factor(item_group_category, 
                                 levels = c("Person/Number", "Control Contexts", "Conjunction", 
                                           "Long-Distance Binding", "Expletives")),
      item_group_readable = factor(item_group_readable,
                                 levels = c("3rd Singular", "3rd Plural", "2nd Singular", "2nd Plural", 
                                           "1st Singular", "1st Plural", "Subject Control", "Object Control",
                                           "No Topic Shift", "Topic Shift", "Long-Distance Binding", 
                                           "Expletive Seems", "Expletive Be"))
    )
  
  for(category in categories) {
    cat(paste("    Creating collapsed", category, "trajectories...\n"))
    
    # Filter data for this category
    category_collapsed <- item_group_collapsed %>%
      filter(item_group_category == category)
    
    # Create safe filename
    safe_category_name <- gsub("[^A-Za-z0-9]", "_", tolower(category))
    
    # Calculate height: 1 inch per facet + 1.5 inches for legend/margins
    n_groups <- length(unique(category_collapsed$item_group_readable))
    spec_category <- get_figure_specs("wide")
    spec_category$height <- (n_groups * 1.0) + 1.5  # 1 inch per facet + fixed space for legend
    spec_category$base_size <- 8
    
    p_category_collapsed <- ggplot(category_collapsed, 
                                  aes(x = checkpoint_log, y = mean_correct)) +
      geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.15, fill = "#3498db") +
      geom_line(size = 0.8, color = "#2980b9") +
      geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
      # First epoch marker at checkpoint 500
      geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.5) +
      facet_grid(item_group_readable ~ form_type, 
                 labeller = labeller(form_type = c("null" = "Null", "overt" = "Overt"))) +
      scale_y_continuous(labels = NULL, limits = c(0, 1)) +
      scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1),
                         labels = c("0", "10", "100", "1K", "10K")) +
      paper_theme(spec_category) +
      theme(
        strip.text.y = element_text(size = 7, angle = 0),
        strip.text.x = element_text(size = 8),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6),
        panel.spacing.y = unit(0.2, "cm"),
        panel.spacing.x = unit(0.3, "cm")
      ) +
      labs(
        title = paste(category, "Trajectories (Collapsed)"),
        subtitle = "Average across all models with 95% confidence intervals",
        x = "Training Step (Log)",
        y = "Preference"
      )
    
    # Save collapsed version with custom height (not using save_paper_figure to preserve custom height)
    filename_collapsed <- paste0("analysis/paper_figures/wide/item_group_trajectories_", safe_category_name, "_collapsed")
    ggsave(paste0(filename_collapsed, ".pdf"), p_category_collapsed, 
           width = spec_category$width, height = spec_category$height, dpi = spec_category$dpi)
    ggsave(paste0(filename_collapsed, ".png"), p_category_collapsed, 
           width = spec_category$width, height = spec_category$height, dpi = spec_category$dpi)
  }
  
} else {
  cat("  Skipping item group trajectories - item_group column not found.\n")
}

# 5. ITEM GROUP HEATMAP (LOG SCALE)
cat("  Creating item group heatmap (log)...\n")

if ("item_group" %in% names(data)) {
  heatmap_data <- data %>%
    mutate(checkpoint_bin = cut(checkpoint_num, 
                                breaks = c(0, 10, 100, 1000, 10000, Inf),
                                labels = c("0-10", "10-100", "100-1K", "1K-10K", "10K+"),
                                include.lowest = TRUE)) %>%
    group_by(model_label, item_group, checkpoint_bin, form_type) %>%
    summarise(mean_correct = mean(correct, na.rm = TRUE), .groups = "drop") %>%
    filter(form_type == "null")  # Focus on null for heatmap
  
  p_heatmap <- ggplot(heatmap_data, aes(x = checkpoint_bin, y = item_group, fill = mean_correct)) +
    geom_tile(color = "white", size = 0.1) +
    facet_wrap(~ model_label, ncol = 3) +
    scale_fill_gradient2(low = PAPER_COLORS$overt, mid = "white", high = PAPER_COLORS$null,
                        midpoint = 0.5, limits = c(0, 1),
                        labels = percent_format()) +
    paper_theme(spec_wide_log) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
      axis.text.y = element_text(size = 5),
      strip.text = element_text(size = 6),
      legend.position = "right",
      legend.key.height = unit(15, "pt"),
      legend.key.width = unit(3, "pt"),
      legend.text = element_text(size = 5),
      legend.title = element_text(size = 6)
    ) +
    labs(
      title = "Item Group Performance Heatmap",
      subtitle = "Null preference across training bins",
      x = "Training Checkpoint Bin",
      y = "Item Group",
      fill = "Null\nPref"
    )
  
  save_paper_figure(p_heatmap, "analysis/paper_figures/wide/item_group_heatmap_log", "wide")
}

# 6. DELTA FROM BASELINE (LOG SCALE)
cat("  Creating delta from baseline (log)...\n")

baseline_data <- data %>%
  filter(model_label == "Baseline") %>%
  mutate(checkpoint_log = log10(checkpoint_num + 1)) %>%
  group_by(checkpoint_log, form_type) %>%
  summarise(baseline_correct = mean(correct, na.rm = TRUE), .groups = "drop")

delta_data <- data %>%
  filter(model_label != "Baseline") %>%
  mutate(checkpoint_log = log10(checkpoint_num + 1)) %>%
  group_by(checkpoint_log, form_type, model_label) %>%
  summarise(mean_correct = mean(correct, na.rm = TRUE), .groups = "drop") %>%
  left_join(baseline_data, by = c("checkpoint_log", "form_type")) %>%
  mutate(delta = mean_correct - baseline_correct)

p_delta <- ggplot(delta_data, aes(x = checkpoint_log, y = delta, 
                                  color = model_label, fill = model_label)) +
  geom_hline(yintercept = 0, linetype = "solid", color = "gray50", size = 0.3) +
  geom_line(size = 0.4) +
  geom_ribbon(aes(ymin = 0, ymax = delta), alpha = 0.1, color = NA) +
  facet_wrap(~ form_type, labeller = labeller(form_type = c("null" = "Null", "overt" = "Overt"))) +
  scale_color_manual(values = PAPER_COLORS$models[2:6]) +
  scale_fill_manual(values = PAPER_COLORS$models[2:6]) +
  scale_y_continuous(labels = percent_format()) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1),
                     labels = c("0", "10", "100", "1K", "10K")) +
  paper_theme(spec_wide_log) +
  theme(
    legend.position = "bottom",
    legend.key.size = unit(4, "pt"),
    legend.text = element_text(size = 5),
    legend.title = element_blank(),
    strip.text = element_text(size = 7)
  ) +
  labs(
    title = "Delta from Baseline",
    subtitle = "Difference in preference relative to baseline model",
    x = "Training Step (Log)",
    y = "Δ Preference"
  )

save_paper_figure(p_delta, "analysis/paper_figures/wide/delta_from_baseline_log", "wide")

cat("Additional log-transformed paper figures created.\n")

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

# Add acquisition lines for model comparison (using t50 data)
cat("Available columns in t50_data:", names(t50_data), "\n")
model_comparison_crossover <- t50_data %>%
  dplyr::select(Model, t50_checkpoint_log, ci_lower_log, ci_upper_log) %>%
  dplyr::rename(model_label = Model, crossover_checkpoint_log = t50_checkpoint_log)

# Now recreate the all_models_comparison_log figure with acquisition points
cat("Updating all_models_comparison_log with 50/50 acquisition points...\n")

p_all_models_log_clean <- ggplot(all_models_summary %>% filter(form_type == "null"), aes(x = checkpoint_log, y = mean_correct,
                                                   color = model_label, fill = model_label)) +
  geom_line(size = 0.4, alpha = 0.9) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.08, color = NA) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
  scale_color_manual(values = PAPER_COLORS$models[1:6]) +
  scale_fill_manual(values = PAPER_COLORS$models[1:6]) +
  scale_y_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), 
                     labels = c("0", ".25", ".5", ".75", "1"),
                     limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1),
                     labels = c("0", "10", "100", "1K", "10K")) +
  paper_theme(spec_wide_log) +
  theme(
    legend.position = "none",
    strip.text = element_text(size = 7),
    axis.text.x = element_text(size = 10),
    axis.text.y = element_text(size = 10)
  ) +
  labs(
    title = "All Models Comparison (Log Scale)",
    subtitle = "Null subject preference across training.",
    x = "Training Step (Log)",
    y = "Preference"
  )

# Save the clean figure with new filename
save_paper_figure(p_all_models_log_clean, "analysis/paper_figures/wide/all_models_comparison_log_clean", "wide")

# Now update individual model figures with acquisition points
cat("Updating individual model figures with 50/50 acquisition points...\n")

for(model_name in models_list) {
  cat("  Updating figure for:", model_name, "\n")
  
  # Filter data for this model
  model_data <- data %>% filter(model_label == model_name)
  
  # Create log-transformed checkpoint variable
  model_data <- model_data %>%
    mutate(checkpoint_log = log10(checkpoint_num + 1))
  
  # Calculate summary statistics for this model
  model_summary <- model_data %>%
    group_by(checkpoint_log, form_type) %>%
    summarise(
      mean_correct = mean(correct, na.rm = TRUE),
      se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(
      ci_lower = mean_correct - 1.96 * se_correct,
      ci_upper = mean_correct + 1.96 * se_correct
    )
  
  # Get acquisition point for this model
  model_crossover <- model_comparison_crossover %>% filter(model_label == model_name)
  
  # Calculate model-specific epoch marker (final checkpoint / 20)
  model_final_checkpoint <- max(model_data$checkpoint_num, na.rm = TRUE)
  model_epoch_marker <- log10((model_final_checkpoint / 20) + 1)
  
  # Create figure specs
  spec_regular_open <- get_figure_specs("regular")
  spec_regular_open$base_size <- 8
  spec_regular_open$title_size <- 9
  spec_regular_open$height <- spec_regular_open$height * 0.9
  
  safe_model_name <- gsub("[^A-Za-z0-9]", "_", tolower(model_name))
  
  p_model_updated <- ggplot(model_summary, aes(x = checkpoint_log, y = mean_correct, 
                                       color = form_type, fill = form_type)) +
    geom_line(linewidth = 0.5) +  # Thinner lines
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                alpha = 0.15, color = NA) +  # More subtle ribbons
    # Add t50 confidence interval as rectangle (behind lines)
    geom_rect(data = model_crossover,
             aes(xmin = ci_lower_log, xmax = ci_upper_log),
             ymin = 0, ymax = 1, fill = "red", alpha = 0.1, inherit.aes = FALSE) +
    # Add 50/50 acquisition point for this model (line on top of ribbon)
    geom_vline(xintercept = model_crossover$crossover_checkpoint_log,
               linetype = "dashed", color = "red", linewidth = 0.4, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", linewidth = 0.3) +
    # Add model-specific epoch marker
    geom_vline(xintercept = model_epoch_marker, linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
    scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                       labels = c("Null", "Overt")) +
    scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                      labels = c("Null", "Overt")) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +  # Full 0-100% range
    scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                       labels = c("0", "10", "100", "1K", "10K")) +
    paper_theme(spec_regular_open) +
    theme(
      legend.position = "right",
      legend.justification = c(0, 0.5),
      legend.margin = margin(l = 2, unit = "pt"),
      legend.key.size = unit(8, "pt"),
      legend.text = element_text(size = 6),
      legend.title = element_blank(),
      plot.margin = margin(5, 25, 5, 5, "pt")  # More right margin for legend
    ) +
    labs(
      title = model_name,
      subtitle = "Null vs overt preference. Red line = 50/50 acquisition point.",
      x = "Training Step (Log)",
      y = "Preference"
    )
  
  # Save updated paper figure
  dir.create("analysis/paper_figures/main", recursive = TRUE, showWarnings = FALSE)
  
  for (format in c("pdf", "png")) {
    full_filename <- paste0("analysis/paper_figures/main/model_", safe_model_name, ".", format)
    
    if (format == "pdf") {
      ggsave(full_filename, p_model_updated, 
             width = spec_regular_open$width, height = spec_regular_open$height, 
             units = "in", dpi = spec_regular_open$dpi,
             device = "pdf", useDingbats = FALSE)
    } else if (format == "png") {
      ggsave(full_filename, p_model_updated, 
             width = spec_regular_open$width, height = spec_regular_open$height, 
             units = "in", dpi = spec_regular_open$dpi,
             device = "png", type = "cairo")
    }
  }
}

cat("Individual model figures updated with 50/50 acquisition points.\n")

# ============================================================================
# FOREST CHARTS FOR EACH MODEL BY ITEM GROUP AND FORM
# ============================================================================

cat("Creating forest charts for each model...\n")

# Calculate end-state preferences for forest charts (using last 10% of checkpoints)
end_state_data <- data %>%
  group_by(model_label) %>%
  filter(checkpoint_num >= quantile(checkpoint_num, 0.9)) %>%
  ungroup()

for(model_name in models_list) {
  cat("  Creating forest charts for:", model_name, "\n")
  
  model_end_data <- end_state_data %>% filter(model_label == model_name)
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
  
  # 1. FOREST CHART BY ITEM GROUP (OVERT SUBJECTS ONLY)
  if ("item_group" %in% names(model_end_data)) {
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
    
    item_group_forest <- model_end_data %>%
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
      paper_theme(get_figure_specs("regular")) +
      theme(
        legend.position = "none",  # Remove legend
        axis.text.y = element_text(hjust = 0)  # Left-align y-axis text
      ) +
      labs(
        title = paste("Overt Subject Preference by Item Group:\n", model_name),
        subtitle = "End-state overt subject preferences with 95% confidence intervals",
        x = "Overt Subject Preference",
        y = "Evaluation Set"
      )
    
    # Save forest chart by item group
    save_paper_figure(p_forest_item_group, 
                     paste0("analysis/paper_figures/supplementary/forest_item_group_", safe_model_name), 
                     "regular")
  }
  
  # 2. FOREST CHART BY FORM (OVERT SUBJECTS ONLY)
  if ("form" %in% names(model_end_data)) {
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
    
    form_forest <- model_end_data %>%
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
      paper_theme(get_figure_specs("regular")) +
      theme(
        legend.position = "none",  # Remove legend
        axis.text.y = element_text(hjust = 0)  # Left-align y-axis text
      ) +
      labs(
        title = paste("Overt Subject Preference by Linguistic Form:\n", model_name),
        subtitle = "End-state overt subject preferences with 95% confidence intervals",
        x = "Overt Subject Preference",
        y = "Processing Manipulation"
      )
    
    # Save forest chart by form
    save_paper_figure(p_forest_form, 
                     paste0("analysis/paper_figures/supplementary/forest_form_", safe_model_name), 
                     "regular")
  }
}

cat("Forest charts created for all models.\n")

# ============================================================================
# CONJUNCTION-ONLY FOREST CHART (FACETED BY MODEL)
# ============================================================================

cat("Creating conjunction-only forest chart faceted by model...\n")

# Filter for conjunction contexts only (using end-state data with last 10% of checkpoints)
conjunction_forest_data <- end_state_data %>%
  filter(item_group %in% c("7a_conjunction_no_topic_shift", "7b_conjunction_topic_shift")) %>%
  filter(form_type == "overt") %>%  # Only overt subjects for forest chart
  group_by(model_label, item_group) %>%
  summarise(
    mean_pref = mean(correct, na.rm = TRUE),
    se_pref = sd(correct, na.rm = TRUE) / sqrt(n()),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    ci_lower = mean_pref - 1.96 * se_pref,
    ci_upper = mean_pref + 1.96 * se_pref,
    ci_lower = pmax(0, ci_lower),
    ci_upper = pmin(1, ci_upper),
    # Create readable names
    item_group_readable = case_when(
      item_group == "7a_conjunction_no_topic_shift" ~ "No Topic Shift",
      item_group == "7b_conjunction_topic_shift" ~ "Topic Shift",
      TRUE ~ item_group
    ),
    # Get model-specific colors
    model_color = case_when(
      model_label == "Baseline" ~ PAPER_COLORS$models[1],
      model_label == "Remove Expletives" ~ PAPER_COLORS$models[2],
      model_label == "Impoverish Determiners" ~ PAPER_COLORS$models[3],
      model_label == "Remove Articles" ~ PAPER_COLORS$models[4],
      model_label == "Lemmatize Verbs" ~ PAPER_COLORS$models[5],
      model_label == "Remove Subject Pronominals" ~ PAPER_COLORS$models[6],
      TRUE ~ PAPER_COLORS$models[1]
    )
  ) %>%
  # Order factors properly
  mutate(
    item_group_readable = factor(item_group_readable, 
                                 levels = c("No Topic Shift", "Topic Shift")),
    model_label = factor(model_label,
                        levels = c("Baseline", "Remove Expletives", "Impoverish Determiners",
                                  "Remove Articles", "Lemmatize Verbs", "Remove Subject Pronominals"))
  )

# Create the faceted forest chart without facet labels
p_conjunction_forest <- ggplot(conjunction_forest_data, 
                              aes(x = mean_pref, y = item_group_readable)) +
  geom_point(aes(color = model_label), size = 2.0) +
  geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper, color = model_label), 
                 height = 0.15, size = 0.4) +
  geom_vline(xintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.3) +
  facet_wrap(~ model_label, ncol = 1) +
  scale_x_continuous(labels = percent_format(), 
                     limits = c(0, 1),
                     breaks = seq(0, 1, 0.25)) +
  scale_y_discrete(position = "right") +
  scale_color_manual(values = PAPER_COLORS$models[1:6],
                    name = "Model") +
  paper_theme(get_figure_specs("regular")) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 6),
    legend.title = element_text(size = 7),
    legend.key.size = unit(0.3, "cm"),
    strip.text = element_blank(),  # Remove facet labels
    strip.background = element_blank(),  # Remove facet background
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(size = 6),
    panel.spacing = unit(0.1, "cm"),  # Minimal spacing between facets
    plot.margin = unit(c(0.3, 0.3, 0.3, 0.3), "cm")
  ) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE)) +  # 2-row legend
  labs(
    title = "Conjunction Context Preferences by Model",
    subtitle = "Overt subject preferences with 95% confidence intervals",
    x = "Overt Subject Preference (%)",
    y = NULL
  )

# Calculate appropriate height (0.35 inches per model facet + margins for legend)
n_models <- length(unique(conjunction_forest_data$model_label))
forest_height <- (n_models * 0.35) + 1.5  # Even smaller facets, compact legend

# Save with custom height
filename <- "analysis/paper_figures/supplementary/forest_conjunction_by_model"
ggsave(paste0(filename, ".pdf"), p_conjunction_forest, 
       width = get_figure_specs("regular")$width, 
       height = forest_height, 
       dpi = get_figure_specs("regular")$dpi)
ggsave(paste0(filename, ".png"), p_conjunction_forest, 
       width = get_figure_specs("regular")$width, 
       height = forest_height, 
       dpi = get_figure_specs("regular")$dpi)

cat("Conjunction-only forest chart created.\n")

# ============================================================================
# CONTROL CONTEXT TRAJECTORIES BY MODEL (INDIVIDUAL GRAPHS)
# ============================================================================

cat("Creating individual control context trajectory graphs for each model...\n")

# Filter for control contexts only (overt preference)
control_trajectories <- data %>%
  filter(item_group %in% c("4a_subject_control", "4b_object_control")) %>%
  filter(form_type == "overt") %>%  # Only overt subjects
  mutate(checkpoint_log = log10(checkpoint_num + 1)) %>%
  group_by(checkpoint_log, model_label, item_group) %>%
  summarise(
    mean_correct = mean(correct, na.rm = TRUE),
    se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    ci_lower = pmax(0, mean_correct - 1.96 * se_correct),
    ci_upper = pmin(1, mean_correct + 1.96 * se_correct),
    # Create readable names
    control_type = case_when(
      item_group == "4a_subject_control" ~ "Subject Control",
      item_group == "4b_object_control" ~ "Object Control",
      TRUE ~ item_group
    )
  )

# Create individual graphs for each model
models <- unique(control_trajectories$model_label)

for(model_name in models) {
  cat(paste("  Creating control context graph for:", model_name, "\n"))
  
  # Filter data for this model
  model_control_data <- control_trajectories %>%
    filter(model_label == model_name)
  
  # Get model color
  model_color <- case_when(
    model_name == "Baseline" ~ PAPER_COLORS$models[1],
    model_name == "Remove Expletives" ~ PAPER_COLORS$models[2],
    model_name == "Impoverish Determiners" ~ PAPER_COLORS$models[3],
    model_name == "Remove Articles" ~ PAPER_COLORS$models[4],
    model_name == "Lemmatize Verbs" ~ PAPER_COLORS$models[5],
    model_name == "Remove Subject Pronominals" ~ PAPER_COLORS$models[6],
    TRUE ~ PAPER_COLORS$models[1]
  )
  
  # Create the plot
  p_control <- ggplot(model_control_data, 
                     aes(x = checkpoint_log, y = mean_correct, 
                         color = control_type, fill = control_type)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2, color = NA) +
    geom_line(linewidth = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", linewidth = 0.4) +
    # First epoch marker at checkpoint 500
    geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.5) +
    scale_color_manual(values = c("Subject Control" = "#2c3e50", "Object Control" = "#e74c3c")) +
    scale_fill_manual(values = c("Subject Control" = "#2c3e50", "Object Control" = "#e74c3c")) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
    scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1),
                       labels = c("0", "10", "100", "1K", "10K")) +
    paper_theme(get_figure_specs("regular")) +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 8),
      legend.key.size = unit(0.5, "cm")
    ) +
    labs(
      title = paste(model_name, "- Control Context Preferences"),
      subtitle = "Overt subject preference over training",
      x = "Training Step (Log)",
      y = "Overt Subject Preference"
    )
  
  # Save the plot
  safe_model_name <- gsub("[^A-Za-z0-9]", "_", tolower(model_name))
  save_paper_figure(p_control,
                   paste0("analysis/paper_figures/supplementary/control_trajectories_", safe_model_name),
                   "regular")
}

cat("Individual control context trajectory graphs created.\n")

# ============================================================================
# INDIVIDUAL MODEL VS BASELINE COMPARISONS
# ============================================================================

cat("Creating individual model vs baseline comparison figures...\n")

# Get the baseline model data
baseline_data <- model_comparison_data %>% filter(model_label == "Baseline")

# Get list of ablated models (exclude baseline)
ablated_models <- setdiff(unique(data$model_label), "Baseline")

for(target_model in ablated_models) {
  cat("  Creating comparison for:", target_model, "vs Baseline\n")
  
  # Filter data for baseline and target model
  comparison_data <- model_comparison_data %>%
    filter(model_label %in% c("Baseline", target_model))
  
  # Create log-transformed version
  comparison_data_log <- comparison_data %>%
    mutate(checkpoint_num_log = log10(checkpoint_num + 1))
  
  # Get crossover data for both models
  comparison_crossover <- model_comparison_crossover %>%
    filter(model_label %in% c("Baseline", target_model))
  
  comparison_crossover_log <- comparison_crossover
  
  # Create safe filename
  safe_target_name <- gsub("[^A-Za-z0-9]", "_", tolower(target_model))
  
  # Get the correct color index for each model from the full model colors
  baseline_color <- PAPER_COLORS$models[1]
  target_color <- case_when(
    target_model == "Remove Expletives" ~ PAPER_COLORS$models[2],
    target_model == "Impoverish Determiners" ~ PAPER_COLORS$models[3],
    target_model == "Remove Articles" ~ PAPER_COLORS$models[4],
    target_model == "Lemmatize Verbs" ~ PAPER_COLORS$models[5],
    target_model == "Remove Subject Pronominals" ~ PAPER_COLORS$models[6],
    TRUE ~ PAPER_COLORS$models[2]  # fallback
  )
  
  # Log scale comparison (only version - as requested)
  p_vs_baseline_log <- ggplot(comparison_data_log, 
                             aes(x = checkpoint_num_log, y = mean_correct, 
                                 color = model_label, fill = model_label, linetype = form_type)) +
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
    # Use correct colors from the full model comparison
    scale_color_manual(values = setNames(c(baseline_color, target_color), c("Baseline", target_model))) +
    scale_fill_manual(values = setNames(c(baseline_color, target_color), c("Baseline", target_model))) +
    scale_linetype_manual(values = c("null" = "dotted", "overt" = "solid"),
                         labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
    scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                       labels = c("0", "10", "100", "1K", "10K")) +
    paper_theme(get_figure_specs("wide")) +
    theme(
      legend.text = element_text(size = 6),
      legend.key.size = unit(0.3, "cm"),
      legend.spacing = unit(0.1, "cm"),
      legend.title = element_blank()
    ) +
    guides(
      color = guide_legend(title = "Model", order = 1),
      fill = guide_legend(title = "Model", order = 1),
      linetype = guide_legend(title = "Form Type", order = 2)
    ) +
    labs(
      title = paste("Model Comparison:", target_model, "vs Baseline"),
      subtitle = "Null vs overt subject acquisition (log scale). Dashed lines = 50/50 acquisition points.",
      x = "Training Checkpoint (Log Scale)",
      y = "Proportion Preferred",
      caption = "Dotted = null subjects, solid = overt subjects"
    )
  
  # Save log scale comparison only
  save_paper_figure(p_vs_baseline_log, 
                   paste0("analysis/paper_figures/supplementary/comparison_vs_baseline_", safe_target_name), 
                   "wide")
  
  # Create faceted version (split by null/overt) for wide section
  p_vs_baseline_faceted <- ggplot(comparison_data_log, 
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
    # Split by null/overt like the main comparison graphs
    facet_wrap(~ form_type, labeller = labeller(form_type = c("null" = "Null Subject", "overt" = "Overt Subject"))) +
    # Use correct colors from the full model comparison
    scale_color_manual(values = setNames(c(baseline_color, target_color), c("Baseline", target_model))) +
    scale_fill_manual(values = setNames(c(baseline_color, target_color), c("Baseline", target_model))) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
    scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                       labels = c("0", "10", "100", "1K", "10K")) +
    paper_theme(get_figure_specs("wide")) +
    theme(
      legend.text = element_text(size = 6),
      legend.key.size = unit(0.3, "cm"),
      legend.spacing = unit(0.1, "cm"),
      legend.title = element_blank(),
      legend.position = "bottom"
    ) +
    guides(
      color = guide_legend(title = "Model"),
      fill = guide_legend(title = "Model")
    ) +
    labs(
      title = paste("Model Comparison:", target_model, "vs Baseline"),
      subtitle = "Null vs overt subject acquisition (log scale). Dashed lines = 50/50 acquisition points.",
      x = "Training Checkpoint (Log Scale)",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold"
    )
  
  # Save faceted version to wide section
  save_paper_figure(p_vs_baseline_faceted, 
                   paste0("analysis/paper_figures/wide/comparison_vs_baseline_faceted_", safe_target_name), 
                   "wide")
}

cat("Individual model vs baseline comparisons created.\n")

# ============================================================================
# ITEM GROUP TRAJECTORIES ACROSS ALL MODELS BY FORM
# ============================================================================

cat("Creating item group trajectory graphs across all models...\n")

if ("item_group" %in% names(data)) {
  # Get list of unique item groups
  item_groups <- unique(data$item_group)
  
  for (item_group_name in item_groups) {
    cat("  Creating trajectory for item group:", item_group_name, "\n")
    
    # Filter data for this item group across all models
    item_group_data <- data %>%
      filter(item_group == item_group_name) %>%
      mutate(checkpoint_log = log10(checkpoint_num + 1))
    
    # Calculate summary statistics by model, checkpoint, form, and form_type
    item_group_summary <- item_group_data %>%
      group_by(model_label, checkpoint_log, form, form_type) %>%
      summarise(
        mean_correct = mean(correct, na.rm = TRUE),
        se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
        .groups = "drop"
      ) %>%
      mutate(
        ci_lower = mean_correct - 1.96 * se_correct,
        ci_upper = mean_correct + 1.96 * se_correct
      )
    
    # Create safe filename
    safe_item_group <- gsub("[^A-Za-z0-9]", "_", tolower(item_group_name))
    
    # Create faceted plot: forms as columns, form_type (null/overt) as rows
    p_item_group_trajectory <- ggplot(item_group_summary, 
                                     aes(x = checkpoint_log, y = mean_correct, 
                                         color = model_label, fill = model_label)) +
      geom_line(linewidth = 0.4, alpha = 0.8) +
      geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
                  alpha = 0.1, color = NA) +
      geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", linewidth = 0.3) +
      # Multi-model graph, so use fixed epoch marker at 500
      geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
      facet_grid(form_type ~ form, 
                 labeller = labeller(form_type = c("null" = "Null Subject", "overt" = "Overt Subject"))) +
      scale_color_manual(values = PAPER_COLORS$models[1:6]) +
      scale_fill_manual(values = PAPER_COLORS$models[1:6]) +
      scale_y_continuous(labels = NULL, limits = c(0, 1)) +
      scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                         labels = c("0", "10", "100", "1K", "10K")) +
      paper_theme(get_figure_specs("wide")) +
      theme(
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.3, "cm"),
        legend.spacing = unit(0.1, "cm"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(size = 7)
      ) +
      guides(
        color = guide_legend(title = "Model", ncol = 3),
        fill = guide_legend(title = "Model", ncol = 3)
      ) +
      labs(
        title = paste("Item Group Trajectory Across All Models:", item_group_name),
        subtitle = "Preference by linguistic form and null/overt type. Gray line = epoch 1.",
        x = "Training Checkpoint (Log Scale)",
        y = "Proportion Preferred",
        caption = "Dotted line = 50% preference threshold"
      )
    
    # Save the item group trajectory figure
    save_paper_figure(p_item_group_trajectory, 
                     paste0("analysis/paper_figures/wide/item_group_trajectory_", safe_item_group), 
                     "wide")
  }
  
  cat("Item group trajectory graphs created.\n")
} else {
  cat("No item_group column found in data. Skipping item group trajectories.\n")
}

# Regular scale model comparison
spec_wide <- get_figure_specs("wide")
p_models_comparison <- ggplot(model_comparison_data, 
                             aes(x = checkpoint_num, y = mean_correct, 
                                 color = model_label, fill = model_label)) +
  geom_line(size = 0.8) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.15, color = NA) +
  # Add acquisition lines
  geom_vline(data = model_comparison_crossover, 
             aes(xintercept = crossover_checkpoint_log, color = model_label),
             linetype = "dashed", size = 0.7, alpha = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
  facet_wrap(~ form_type, labeller = labeller(form_type = c("null" = "Null Subject Preference", 
                                                            "overt" = "Overt Subject Preference"))) +
  scale_color_manual(values = PAPER_COLORS$models[1:6]) +
  scale_fill_manual(values = PAPER_COLORS$models[1:6]) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  paper_theme(spec_wide) +
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

# Save paper version (Figure 6: Wide figure for complete model comparison)
save_paper_figure(p_models_comparison, "analysis/paper_figures/wide/fig6_model_comparison_general", "wide")

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
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
  facet_wrap(~ form_type, labeller = labeller(form_type = c("null" = "Null Subject Preference", 
                                                            "overt" = "Overt Subject Preference"))) +
  scale_color_manual(values = PAPER_COLORS$models[1:6]) +
  scale_fill_manual(values = PAPER_COLORS$models[1:6]) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                     labels = c("0", "10", "100", "1K", "10K")) +
  paper_theme(get_figure_specs("wide")) +
  theme(
    legend.text = element_text(size = 6),     # Smaller legend text
    legend.key.size = unit(0.3, "cm"),       # Smaller legend keys
    legend.spacing = unit(0.1, "cm"),        # Tighter legend spacing
    legend.title = element_blank(),
    plot.margin = margin(5, 5, 5, 5, "pt")   # Adjust margins for taller view
  ) +
  guides(
    color = guide_legend(title = "Model", ncol = 2),  # 2 columns to save space
    fill = guide_legend(title = "Model", ncol = 2)
  ) +
  labs(
    title = "Model Comparison: General Null Subject Acquisition (Log Scale)",
    subtitle = "Collapsed across all items and forms. Dashed lines = 50/50 acquisition points.",
    x = "Training Checkpoint (Log Scale)",
    y = "Proportion Preferred",
    caption = "Gray dotted line = 50% preference threshold"
  )

# Save log scale model comparison plot
ggsave("analysis/figures/combined/models_comparison_general_log.pdf", 
       p_models_comparison_log, width = 14, height = 8)
ggsave("analysis/figures/combined/models_comparison_general_log.png", 
       p_models_comparison_log, width = 14, height = 8, dpi = 300)

# Create custom taller specs for supplementary figures
spec_supplementary <- get_figure_specs("wide")
spec_supplementary$height <- spec_supplementary$height * 1.3  # 30% taller

# Save paper version (Wide figure - supplementary log scale version) with taller dimensions
dir.create("analysis/paper_figures/supplementary", recursive = TRUE, showWarnings = FALSE)
ggsave("analysis/paper_figures/supplementary/fig6_model_comparison_general_log.pdf", p_models_comparison_log, 
       width = spec_supplementary$width, height = spec_supplementary$height, units = "in", dpi = spec_supplementary$dpi,
       device = "pdf", useDingbats = FALSE)
ggsave("analysis/paper_figures/supplementary/fig6_model_comparison_general_log.png", p_models_comparison_log, 
       width = spec_supplementary$width, height = spec_supplementary$height, units = "in", dpi = spec_supplementary$dpi,
       device = "png", type = "cairo")
cat("Saved taller supplementary figure: analysis/paper_figures/supplementary/fig6_model_comparison_general_log\n")

# Combined null+overt model comparison (like other figures)
p_models_combined <- ggplot(model_comparison_data, 
                           aes(x = checkpoint_num, y = mean_correct, 
                               color = model_label, fill = model_label, linetype = form_type)) +
  geom_line(size = 0.8) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.15, color = NA) +
  # Add acquisition lines
  geom_vline(data = model_comparison_crossover, 
             aes(xintercept = crossover_checkpoint_log, color = model_label),
             linetype = "dashed", size = 0.7, alpha = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
  scale_color_manual(values = c("Baseline" = PAPER_COLORS$models[1], 
                                "Remove Expletives" = PAPER_COLORS$models[2],
                                "Impoverish Determiners" = PAPER_COLORS$models[3], 
                                "Remove Articles" = PAPER_COLORS$models[4],
                                "Lemmatize Verbs" = PAPER_COLORS$models[5],
                                "Remove Subject Pronominals" = PAPER_COLORS$models[6])) +
  scale_fill_manual(values = c("Baseline" = PAPER_COLORS$models[1], 
                               "Remove Expletives" = PAPER_COLORS$models[2],
                               "Impoverish Determiners" = PAPER_COLORS$models[3], 
                               "Remove Articles" = PAPER_COLORS$models[4],
                               "Lemmatize Verbs" = PAPER_COLORS$models[5],
                               "Remove Subject Pronominals" = PAPER_COLORS$models[6])) +
  scale_linetype_manual(values = c("null" = "dotted", "overt" = "solid"),
                       labels = c("Null Subject", "Overt Subject")) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  paper_theme(get_figure_specs("wide")) +
  theme(
    legend.title = element_blank()
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

# Save paper version (Wide figure for main text)
save_paper_figure(p_models_combined, "analysis/paper_figures/wide/fig7_model_comparison_combined", "wide")

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
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
  scale_color_manual(values = c("Baseline" = PAPER_COLORS$models[1], 
                                "Remove Expletives" = PAPER_COLORS$models[2],
                                "Impoverish Determiners" = PAPER_COLORS$models[3], 
                                "Remove Articles" = PAPER_COLORS$models[4],
                                "Lemmatize Verbs" = PAPER_COLORS$models[5],
                                "Remove Subject Pronominals" = PAPER_COLORS$models[6])) +
  scale_fill_manual(values = c("Baseline" = PAPER_COLORS$models[1], 
                               "Remove Expletives" = PAPER_COLORS$models[2],
                               "Impoverish Determiners" = PAPER_COLORS$models[3], 
                               "Remove Articles" = PAPER_COLORS$models[4],
                               "Lemmatize Verbs" = PAPER_COLORS$models[5],
                               "Remove Subject Pronominals" = PAPER_COLORS$models[6])) +
  scale_linetype_manual(values = c("null" = "dotted", "overt" = "solid"),
                       labels = c("Null Subject", "Overt Subject")) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                     labels = c("0", "10", "100", "1K", "10K")) +
  paper_theme(get_figure_specs("wide")) +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 6),  # Smaller legend text
    legend.key.size = unit(0.3, "cm"),     # Smaller legend keys
    legend.spacing = unit(0.1, "cm")       # Tighter legend spacing
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

# Save paper version (Wide figure - supplementary log scale version)
save_paper_figure(p_models_combined_log, "analysis/paper_figures/supplementary/fig8_model_comparison_combined_log", "wide")

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

# Get custom specifications for Figure 3 - wider and taller
spec_fig3 <- get_figure_specs("regular")
spec_fig3$width <- spec_fig3$width * 1.15  # 15% wider to fit title
spec_fig3$height <- spec_fig3$height * 1.2  # 20% taller
spec_fig3$base_size <- 8  # Smaller base font size

# Regular scale null-only model comparison
p_models_null_only <- ggplot(model_comparison_data_null, 
                            aes(x = checkpoint_num, y = mean_correct, 
                                color = model_label, fill = model_label)) +
  geom_line(linewidth = 0.5) +  # Thinner lines
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.15, color = NA) +
  # Add acquisition lines
  geom_vline(data = model_comparison_crossover, 
             aes(xintercept = crossover_checkpoint_log, color = model_label),
             linetype = "dashed", linewidth = 0.4, alpha = 0.8) +  # Thinner dashed lines
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", linewidth = 0.3) +
  scale_color_manual(values = c("Baseline" = PAPER_COLORS$models[1], 
                                "Remove Expletives" = PAPER_COLORS$models[2],
                                "Impoverish Determiners" = PAPER_COLORS$models[3], 
                                "Remove Articles" = PAPER_COLORS$models[4],
                                "Lemmatize Verbs" = PAPER_COLORS$models[5],
                                "Remove Subject Pronominals" = PAPER_COLORS$models[6])) +
  scale_fill_manual(values = c("Baseline" = PAPER_COLORS$models[1], 
                               "Remove Expletives" = PAPER_COLORS$models[2],
                               "Impoverish Determiners" = PAPER_COLORS$models[3], 
                               "Remove Articles" = PAPER_COLORS$models[4],
                               "Lemmatize Verbs" = PAPER_COLORS$models[5],
                               "Remove Subject Pronominals" = PAPER_COLORS$models[6])) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  paper_theme(spec_fig3) +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 5),     # Much smaller legend text
    legend.key.size = unit(0.25, "cm"),       # Smaller legend keys
    legend.spacing = unit(0.05, "cm"),        # Tighter legend spacing
    plot.title = element_text(size = 9),      # Smaller title
    plot.subtitle = element_text(size = 7)    # Smaller subtitle
  ) +
  guides(
    color = guide_legend(title = "Model", ncol = 2),  # 2 columns to save space
    fill = guide_legend(title = "Model", ncol = 2)
  ) +
  labs(
    title = "Model Comparison: Null Subject Preference",
    subtitle = "Collapsed across all items and forms. Dashed lines = 50/50 acquisition.",
    x = "Training Checkpoint",
    y = "Null Subject Preference",
    caption = "Gray dotted line = 50% preference threshold"
  )

# Save null-only model comparison plot
ggsave("analysis/figures/combined/models_comparison_null_only.pdf", 
       p_models_null_only, width = 12, height = 8)
ggsave("analysis/figures/combined/models_comparison_null_only.png", 
       p_models_null_only, width = 12, height = 8, dpi = 300)

# Save paper version with custom specifications
dir.create("analysis/paper_figures/main", recursive = TRUE, showWarnings = FALSE)
ggsave("analysis/paper_figures/main/fig3_model_comparison_null_only.pdf", p_models_null_only, 
       width = spec_fig3$width, height = spec_fig3$height, units = "in", dpi = spec_fig3$dpi,
       device = "pdf", useDingbats = FALSE)
ggsave("analysis/paper_figures/main/fig3_model_comparison_null_only.png", p_models_null_only, 
       width = spec_fig3$width, height = spec_fig3$height, units = "in", dpi = spec_fig3$dpi,
       device = "png", type = "cairo")
cat("Saved custom figure: analysis/paper_figures/main/fig3_model_comparison_null_only\n")

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
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
  scale_color_manual(values = c("Baseline" = PAPER_COLORS$models[1], 
                                "Remove Expletives" = PAPER_COLORS$models[2],
                                "Impoverish Determiners" = PAPER_COLORS$models[3], 
                                "Remove Articles" = PAPER_COLORS$models[4],
                                "Lemmatize Verbs" = PAPER_COLORS$models[5],
                                "Remove Subject Pronominals" = PAPER_COLORS$models[6])) +
  scale_fill_manual(values = c("Baseline" = PAPER_COLORS$models[1], 
                               "Remove Expletives" = PAPER_COLORS$models[2],
                               "Impoverish Determiners" = PAPER_COLORS$models[3], 
                               "Remove Articles" = PAPER_COLORS$models[4],
                               "Lemmatize Verbs" = PAPER_COLORS$models[5],
                               "Remove Subject Pronominals" = PAPER_COLORS$models[6])) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                     labels = c("0", "10", "100", "1K", "10K")) +
  paper_theme(get_figure_specs("regular")) +
  theme(
    legend.title = element_blank()
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

# Save paper version (Regular figure - supplementary log scale version)
save_paper_figure(p_models_null_only_log, "analysis/paper_figures/supplementary/fig4_model_comparison_null_only_log", "regular")

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
             aes(xintercept = crossover_checkpoint_log, color = form),
             linetype = "dashed", size = 0.5, alpha = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
  facet_grid(model_label ~ form_type, 
             labeller = labeller(form_type = c("null" = "Null Subject", 
                                               "overt" = "Overt Subject"))) +
  scale_color_manual(values = PAPER_COLORS$forms) +
  scale_fill_manual(values = PAPER_COLORS$forms) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  paper_theme(get_figure_specs("multipanel")) +
  theme(
    legend.title = element_blank()
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

# Save paper version (Multi-panel figure for main text)
save_paper_figure(p_models_by_form, "analysis/paper_figures/wide/fig9_model_comparison_by_form", "multipanel")

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
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
  facet_grid(model_label ~ form_type, 
             labeller = labeller(form_type = c("null" = "Null Subject", 
                                               "overt" = "Overt Subject"))) +
  scale_color_manual(values = PAPER_COLORS$forms) +
  scale_fill_manual(values = PAPER_COLORS$forms) +
  scale_y_continuous(labels = NULL, limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1), 
                     labels = c("0", "10", "100", "1K", "10K")) +
  paper_theme(get_figure_specs("wide")) +
  theme(
    legend.title = element_blank()
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

# Save paper version (Wide figure - supplementary log scale version)
save_paper_figure(p_models_by_form_log, "analysis/paper_figures/supplementary/fig10_model_comparison_by_form_log", "wide")

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
               aes(xintercept = crossover_checkpoint_log),
               linetype = "dashed", color = "red", size = 0.7, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
    facet_wrap(~ item_group, scales = "free_x") +
    scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
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
               aes(xintercept = crossover_checkpoint_log),
               linetype = "dashed", color = "red", size = 0.7, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
    facet_wrap(~ form, scales = "free_x") +
    scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
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
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
    facet_wrap(~ item_group, scales = "free_x") +
    scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
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
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
    facet_wrap(~ form, scales = "free_x") +
    scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
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
               aes(xintercept = crossover_checkpoint_log),
               linetype = "dashed", color = "red", size = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
    facet_wrap(~ form_type, labeller = labeller(form_type = c("null" = "Null Subject", 
                                                              "overt" = "Overt Subject"))) +
    scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
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
               aes(xintercept = crossover_checkpoint_log, color = item_group),
               linetype = "dashed", size = 0.6, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
    facet_grid(item_group ~ form_type, 
               labeller = labeller(form_type = c("null" = "Null Subject", 
                                                 "overt" = "Overt Subject"))) +
    scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
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
               aes(xintercept = crossover_checkpoint_log, color = form),
               linetype = "dashed", size = 0.6, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
    facet_grid(form ~ form_type, 
               labeller = labeller(form_type = c("null" = "Null Subject", 
                                                 "overt" = "Overt Subject"))) +
    scale_color_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                       labels = c("Null Subject", "Overt Subject")) +
    scale_fill_manual(values = c("null" = PAPER_COLORS$null, "overt" = PAPER_COLORS$overt),
                      labels = c("Null Subject", "Overt Subject")) +
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
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
               aes(xintercept = crossover_checkpoint_log, color = model_label),
               linetype = "dashed", size = 0.7, alpha = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
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
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
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
  
  comparison_crossover_log <- comparison_crossover
  
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
  # First epoch marker at checkpoint 500
  geom_vline(xintercept = log10(500 + 1), linetype = "solid", color = "gray30", linewidth = 0.3, alpha = 0.6) +
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
    scale_y_continuous(labels = NULL, limits = c(0, 1)) +
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

# Hotspot analysis removed for now - can be added back later if needed
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
  
  # Check if we have the actual hotspot surprisal columns
  has_hotspot_data <- all(c("hotspot_surprisal", "hotspot", "form") %in% names(data_subset))
  
  if (!has_hotspot_data) {
    cat("  Note: Expected hotspot columns not found for", model_name, "\n")
    cat("  Available columns:", paste(names(data_subset), collapse = ", "), "\n")
    cat("  Hotspot analysis skipped\n")
    return(invisible(NULL))
  }
  
  cat("  Using collected hotspot data for", model_name, "\n")
  
  # Use the collected hotspot data as-is
  hotspot_data <- data_subset %>%
    dplyr::select(checkpoint_num, item_id, form, form_type, hotspot, hotspot_surprisal, hotspot_difference, correct) %>%
    filter(!is.na(hotspot_surprisal))
  
  # 1. SURPRISAL OVER TIME BY FORM TYPE (aggregated across all hotspots)
  hotspot_summary <- hotspot_data %>%
    group_by(checkpoint_num, form_type) %>%
    summarise(
      mean_surprisal = mean(hotspot_surprisal, na.rm = TRUE),
      se_surprisal = sd(hotspot_surprisal, na.rm = TRUE) / sqrt(n()),
      mean_difference = mean(hotspot_difference, na.rm = TRUE),
      se_difference = sd(hotspot_difference, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(
      ci_lower = mean_surprisal - 1.96 * se_surprisal,
      ci_upper = mean_surprisal + 1.96 * se_surprisal
    )
  
  # Plot surprisal over time by hotspot
  # Check what form_type values actually exist
  form_type_values <- unique(hotspot_summary$form_type)
  color_mapping <- setNames(c(PAPER_COLORS$null, PAPER_COLORS$overt)[1:length(form_type_values)], form_type_values)
  
  p_hotspot_time <- ggplot(hotspot_summary, aes(x = checkpoint_num, y = mean_surprisal, 
                                            color = form_type, fill = form_type)) +
    geom_line(size = 0.8) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                alpha = 0.2, color = NA) +
    scale_color_manual(values = color_mapping) +
    scale_fill_manual(values = color_mapping) +
    paper_theme(get_figure_specs("wide")) +
    theme(
      legend.title = element_blank()
    ) +
    labs(
      title = paste("Hotspot Surprisal Over Time:", model_name),
      subtitle = "Average surprisal at critical positions (aggregated across all hotspots)",
      x = "Training Checkpoint",
      y = "Mean Surprisal (bits)"
    )
  
  ggsave(paste0(hotspot_dir, "surprisal_over_time.pdf"), p_hotspot_time, 
         width = 12, height = 8)
  ggsave(paste0(hotspot_dir, "surprisal_over_time.png"), p_hotspot_time, 
         width = 12, height = 8, dpi = 300)
  
  # Save paper version (Wide figure for supplementary)
  safe_model_name <- gsub("[^A-Za-z0-9]", "_", tolower(model_name))
  save_paper_figure(p_hotspot_time, paste0("analysis/paper_figures/supplementary/hotspot_time_", safe_model_name), "wide")
  
  # 2. BINARY PREFERENCE BY HOTSPOT (which form has lower surprisal)
  # First check what forms actually exist in the data
  if (nrow(hotspot_data) == 0) {
    cat("    No hotspot data available for", model_name, "- skipping preference analysis\n")
    cat("  Hotspot analysis completed for", model_name, "(no data)\n")
    return(invisible(NULL))
  }
  
  unique_forms <- unique(hotspot_data$form)
  cat("    Available forms:", paste(unique_forms, collapse = ", "), "\n")
  
  # The data has linguistic forms, not null/overt, so we'll analyze by form_type instead
  if (!"form_type" %in% names(hotspot_data) || !all(c("null", "overt") %in% unique(hotspot_data$form_type))) {
    cat("    Missing null or overt form_type - skipping preference analysis for", model_name, "\n")
    cat("  Hotspot analysis completed for", model_name, "(missing form types)\n")
    return(invisible(NULL))
  }
  
  # 2. BINARY PREFERENCE ANALYSIS (aggregated across hotspots)
  hotspot_preference <- hotspot_data %>%
    group_by(checkpoint_num, item_id, form_type) %>%
    summarise(mean_hotspot_surprisal = mean(hotspot_surprisal, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(
      names_from = form_type,
      values_from = mean_hotspot_surprisal,
      names_prefix = "surprisal_",
      values_fn = mean  # Handle duplicates by taking mean
    )
  
  # Check what columns were actually created
  created_cols <- names(hotspot_preference)
  cat("    Created columns:", paste(created_cols, collapse = ", "), "\n")
  
  # Safely create preference variables (aggregated across all hotspots)
  hotspot_preference <- hotspot_preference %>%
    mutate(
      prefers_null = case_when(
        "surprisal_null" %in% names(.) & "surprisal_overt" %in% names(.) ~
          if_else(!is.na(surprisal_null) & !is.na(surprisal_overt),
                  surprisal_null < surprisal_overt, NA),
        TRUE ~ NA
      )
    ) %>%
    group_by(checkpoint_num) %>%
    summarise(
      null_pref = mean(prefers_null, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      overt_pref = 1 - null_pref
    )
  
  # Reshape for plotting  
  preference_long <- hotspot_preference %>%
    pivot_longer(
      cols = c(null_pref, overt_pref),
      names_to = "preference_type",
      values_to = "preference_value"
    ) %>%
    mutate(
      preference_type = case_when(
        preference_type == "null_pref" ~ "Null Preferred",
        preference_type == "overt_pref" ~ "Overt Preferred",
        TRUE ~ preference_type
      )
    )
  
  # Plot binary preferences
  p_hotspot_binary <- ggplot(preference_long, aes(x = checkpoint_num, y = preference_value, color = preference_type)) +
    geom_line(size = 1) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50") +
    scale_color_manual(values = c("Null Preferred" = PAPER_COLORS$null, "Overt Preferred" = PAPER_COLORS$overt)) +
    scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
    paper_theme(get_figure_specs("wide")) +
    theme(
      legend.title = element_blank()
    ) +
    labs(
      title = paste("Hotspot Preference Over Time:", model_name),
      subtitle = "Proportion preference aggregated across all hotspots",
      x = "Training Checkpoint",
      y = "Proportion Preferred",
      caption = "Gray dotted line = 50% preference threshold"
    )
  
  ggsave(paste0(hotspot_dir, "binary_preference.pdf"), p_hotspot_binary, 
         width = 12, height = 8)
  ggsave(paste0(hotspot_dir, "binary_preference.png"), p_hotspot_binary, 
         width = 12, height = 8, dpi = 300)
  
  # Save paper version (Wide figure for supplementary)
  save_paper_figure(p_hotspot_binary, paste0("analysis/paper_figures/supplementary/hotspot_binary_", safe_model_name), "wide")
  
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
    scale_fill_gradient2(low = PAPER_COLORS$overt, mid = "white", high = PAPER_COLORS$null, 
                        midpoint = 0.5, labels = scales::percent_format(),
                        name = "Null\nPreference") +
    paper_theme(get_figure_specs("wide")) +
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
  
  # Save paper version (Wide figure for supplementary)
  save_paper_figure(p_heatmap, paste0("analysis/paper_figures/supplementary/hotspot_heatmap_", safe_model_name), "wide")
  
  cat("  Hotspot analysis saved for", model_name, "\n")
}

# Hotspot analysis disabled for now
# # Run hotspot analysis for each model
# models_to_analyze <- unique(data$model)
# 
# for (mod in models_to_analyze) {
#   mod_data <- data %>% filter(model == mod)
#   mod_label <- unique(mod_data$model_label)
#   
#   # Determine model folder name
#   model_folder <- case_when(
#     mod == "exp0_baseline" ~ "baseline",
#     mod == "exp1_remove_expletives" ~ "remove_expletives", 
#     mod == "exp2_impoverish_determiners" ~ "impoverish_determiners",
#     mod == "exp3_remove_articles" ~ "remove_articles",
#     mod == "exp4_lemmatize_verbs" ~ "lemmatize_verbs",
#     mod == "exp5_remove_subject_pronominals" ~ "remove_subject_pronominals",
#     TRUE ~ gsub("[^a-z0-9_]", "_", tolower(mod))
#   )
#   
#   # Run analysis
#   # analyze_hotspot_surprisal_disabled(mod_data, mod_label, model_folder)
# }

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

