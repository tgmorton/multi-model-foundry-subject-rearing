#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) != 1) stop("Run this file with Rscript")
script_path <- normalizePath(sub("^--file=", "", file_arg), mustWork = TRUE)
repo <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)

figure_dir <- file.path(
  repo, "analysis", "eval_v2", "figures", "foundry_trajectories"
)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

trajectory_csv <- file.path(
  figure_dir, "gpt2_medium_old_vs_matched_h2_h4_cell_trajectories.csv"
)
builder <- file.path(repo, "scripts", "build_old_vs_matched_trajectory.py")
builder_output <- system2(
  "python3",
  c(
    shQuote(builder),
    "--architecture", "gpt2_medium",
    "--hp-rank", "2", "--hp-rank", "3", "--hp-rank", "4",
    "--old-root", shQuote(file.path(repo, "data", "eval_results", "null_subj_v2")),
    "--matched-root", shQuote(file.path(
      repo, "data", "eval_results", "null_subj_v2_condition_matched_v1"
    )),
    "--unigram-root", shQuote(file.path(repo, "data", "unigrams")),
    "--output", shQuote(trajectory_csv)
  ),
  stdout = TRUE,
  stderr = TRUE
)
builder_status <- attr(builder_output, "status")
if (!is.null(builder_status) && builder_status != 0) {
  stop(paste(builder_output, collapse = "\n"))
}
message(paste(builder_output, collapse = "\n"))

category_order <- c(
  "subject_drop", "subject_drop_no_agreement", "expletive", "object_drop",
  "embedded_drop", "extraction", "conjunction", "control"
)
category_labels <- c(
  subject_drop = "Subject drop",
  subject_drop_no_agreement = "Subject drop (no agreement)",
  expletive = "Expletive",
  object_drop = "Object drop",
  embedded_drop = "Embedded drop",
  extraction = "Extraction",
  conjunction = "Conjunction",
  control = "Control"
)
intervention_labels <- c(
  baseline = "Baseline",
  remove_expletive_sentences = "Remove expletives",
  impoverish_case = "Impoverished case",
  lemmatize_verbs = "Lemmatized verbs",
  enrich_verbal_morphology = "Enriched morphology"
)
intervention_colors <- c(
  "Baseline" = "#1b9e77",
  "Remove expletives" = "#e7298a",
  "Impoverished case" = "#d95f02",
  "Lemmatized verbs" = "#66a61e",
  "Enriched morphology" = "#7570b3"
)

raw <- read.csv(trajectory_csv, stringsAsFactors = FALSE) |>
  filter(
    category %in% category_order,
    intervention %in% names(intervention_labels),
    hp_rank %in% c(2, 3, 4)
  ) |>
  mutate(
    category = factor(
      category,
      levels = category_order,
      labels = category_labels[category_order]
    ),
    intervention = factor(
      intervention_labels[intervention],
      levels = unname(intervention_labels)
    ),
    eval_version = factor(
      eval_version,
      levels = c("Old generic", "Condition-matched")
    )
  )

positive_tokens <- raw$tokens_seen[raw$tokens_seen > 0]
token_min <- min(positive_tokens)
token_max <- max(positive_tokens)
n_bins <- 30L
edges <- 10 ^ seq(log10(token_min), log10(token_max), length.out = n_bins + 1L)

binned <- raw |>
  mutate(
    bin_id = ifelse(
      tokens_seen == 0,
      0L,
      pmin(n_bins, pmax(1L, findInterval(tokens_seen, edges, all.inside = TRUE)))
    ),
    bin_center = ifelse(
      bin_id == 0,
      0,
      sqrt(edges[pmax(1L, bin_id)] * edges[pmin(n_bins + 1L, bin_id + 1L)])
    )
  ) |>
  group_by(
    cell_id, hp_rank, seed, intervention, eval_version,
    category, bin_id, bin_center
  ) |>
  summarise(
    preference = mean(preference),
    slor_diff = mean(slor_diff),
    .groups = "drop"
  )

summary_df <- binned |>
  group_by(intervention, eval_version, category, bin_id, bin_center) |>
  summarise(
    mean_preference = mean(preference),
    sd = sd(preference),
    n_cells = n_distinct(cell_id),
    se = ifelse(n_cells > 1, sd / sqrt(n_cells), 0),
    .groups = "drop"
  ) |>
  mutate(x = log10(bin_center + 1))

delta_cells <- binned |>
  select(
    cell_id, hp_rank, seed, intervention, category,
    bin_id, bin_center, eval_version, preference
  ) |>
  pivot_wider(names_from = eval_version, values_from = preference) |>
  filter(!is.na(`Old generic`), !is.na(`Condition-matched`)) |>
  mutate(delta = `Condition-matched` - `Old generic`)

delta_df <- delta_cells |>
  group_by(intervention, category, bin_id, bin_center) |>
  summarise(
    mean_delta = mean(delta),
    sd = sd(delta),
    n_cells = n_distinct(cell_id),
    se = ifelse(n_cells > 1, sd / sqrt(n_cells), 0),
    lower = mean_delta - se,
    upper = mean_delta + se,
    .groups = "drop"
  ) |>
  mutate(x = log10(bin_center + 1))

slor_summary_df <- binned |>
  group_by(intervention, eval_version, category, bin_id, bin_center) |>
  summarise(
    mean_slor_diff = mean(slor_diff),
    sd = sd(slor_diff),
    n_cells = n_distinct(cell_id),
    se = ifelse(n_cells > 1, sd / sqrt(n_cells), 0),
    .groups = "drop"
  ) |>
  mutate(x = log10(bin_center + 1))

slor_delta_cells <- binned |>
  select(
    cell_id, hp_rank, seed, intervention, category,
    bin_id, bin_center, eval_version, slor_diff
  ) |>
  pivot_wider(names_from = eval_version, values_from = slor_diff) |>
  filter(!is.na(`Old generic`), !is.na(`Condition-matched`)) |>
  mutate(delta = `Condition-matched` - `Old generic`)

slor_delta_df <- slor_delta_cells |>
  group_by(intervention, category, bin_id, bin_center) |>
  summarise(
    mean_delta = mean(delta),
    sd = sd(delta),
    n_cells = n_distinct(cell_id),
    se = ifelse(n_cells > 1, sd / sqrt(n_cells), 0),
    lower = mean_delta - se,
    upper = mean_delta + se,
    .groups = "drop"
  ) |>
  mutate(x = log10(bin_center + 1))

write.csv(
  summary_df,
  file.path(figure_dir, "gpt2_medium_old_vs_matched_h2_h4_summary.csv"),
  row.names = FALSE
)
write.csv(
  delta_df,
  file.path(figure_dir, "gpt2_medium_matched_minus_old_h2_h4_summary.csv"),
  row.names = FALSE
)
write.csv(
  slor_summary_df,
  file.path(figure_dir, "gpt2_medium_old_vs_matched_h2_h4_slor_summary.csv"),
  row.names = FALSE
)
write.csv(
  slor_delta_df,
  file.path(figure_dir, "gpt2_medium_matched_minus_old_h2_h4_slor_summary.csv"),
  row.names = FALSE
)

token_axis <- data.frame(
  value = c(256000, 1e6, 1e7, 1e8, 1e9, 4e9),
  label = c("256K", "1M", "10M", "100M", "1B", "4B")
) |>
  filter(value >= token_min, value <= token_max * 1.05)
token_breaks <- token_axis$value
token_labels <- token_axis$label
x_limits <- c(log10(token_min + 1) - 0.04, log10(token_max + 1) + 0.04)

matched_lines <- filter(summary_df, eval_version == "Condition-matched")
old_lines <- filter(summary_df, eval_version == "Old generic")

overlay <- ggplot(
  summary_df,
  aes(x = x, y = mean_preference, color = intervention)
) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "grey55", linewidth = 0.4) +
  geom_line(
    data = matched_lines,
    aes(group = intervention, linetype = eval_version),
    linewidth = 1.05,
    alpha = 0.95
  ) +
  geom_line(
    data = old_lines,
    aes(group = intervention, linetype = eval_version),
    linewidth = 0.85,
    alpha = 0.95
  ) +
  facet_wrap(~ category, ncol = 2) +
  scale_color_manual(values = intervention_colors, drop = TRUE) +
  scale_linetype_manual(
    values = c("Old generic" = "22", "Condition-matched" = "solid"),
    drop = TRUE
  ) +
  scale_x_continuous(
    breaks = log10(token_breaks + 1),
    labels = token_labels,
    limits = x_limits
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(
    title = "GPT-2 medium: old versus condition-matched evaluation trajectories",
    subtitle = paste0(
      "Binary length-normalized likelihood preference; H2-H4, seeds 42 and 137 ",
      "(6 fixed cells per condition). Dashed = old generic eval; solid = matched eval."
    ),
    x = "Tokens seen (log scale)",
    y = "P(overt preferred)",
    color = "Condition",
    linetype = "Evaluation"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 10),
    strip.text = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

delta_plot <- ggplot(
  delta_df,
  aes(x = x, y = mean_delta, color = intervention, fill = intervention)
) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey45", linewidth = 0.45) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.12, linewidth = 0) +
  geom_line(linewidth = 1.05) +
  facet_wrap(~ category, ncol = 2) +
  scale_color_manual(values = intervention_colors, drop = TRUE) +
  scale_fill_manual(values = intervention_colors, drop = TRUE) +
  guides(fill = "none") +
  scale_x_continuous(
    breaks = log10(token_breaks + 1),
    labels = token_labels,
    limits = x_limits
  ) +
  labs(
    title = "GPT-2 medium: trajectory change after condition matching",
    subtitle = paste0(
      "Matched minus old binary preference, mean ± 1 SE across paired H2-H4 cells. ",
      "Positive values indicate a higher overt-pronoun preference under the matched eval."
    ),
    x = "Tokens seen (log scale)",
    y = "Matched − old P(overt preferred)",
    color = "Condition",
    fill = "Condition"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 10),
    strip.text = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

slor_matched_lines <- filter(
  slor_summary_df, eval_version == "Condition-matched"
)
slor_old_lines <- filter(slor_summary_df, eval_version == "Old generic")

slor_overlay <- ggplot(
  slor_summary_df,
  aes(x = x, y = mean_slor_diff, color = intervention)
) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey55", linewidth = 0.4) +
  geom_line(
    data = slor_matched_lines,
    aes(group = intervention, linetype = eval_version),
    linewidth = 1.05,
    alpha = 0.95
  ) +
  geom_line(
    data = slor_old_lines,
    aes(group = intervention, linetype = eval_version),
    linewidth = 0.85,
    alpha = 0.95
  ) +
  facet_wrap(~ category, ncol = 2) +
  scale_color_manual(values = intervention_colors, drop = TRUE) +
  scale_linetype_manual(
    values = c("Old generic" = "22", "Condition-matched" = "solid"),
    drop = TRUE
  ) +
  scale_x_continuous(
    breaks = log10(token_breaks + 1),
    labels = token_labels,
    limits = x_limits
  ) +
  labs(
    title = "GPT-2 medium: old versus condition-matched ΔSLOR trajectories",
    subtitle = paste0(
      "Continuous SLOR(overt) − SLOR(null); H2-H4, seeds 42 and 137 ",
      "(6 fixed cells per condition). Dashed = old; solid = matched."
    ),
    x = "Tokens seen (log scale)",
    y = "ΔSLOR (overt − null)",
    color = "Condition",
    linetype = "Evaluation"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 10),
    strip.text = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

slor_delta_plot <- ggplot(
  slor_delta_df,
  aes(x = x, y = mean_delta, color = intervention, fill = intervention)
) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey45", linewidth = 0.45) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.12, linewidth = 0) +
  geom_line(linewidth = 1.05) +
  facet_wrap(~ category, ncol = 2) +
  scale_color_manual(values = intervention_colors, drop = TRUE) +
  scale_fill_manual(values = intervention_colors, drop = TRUE) +
  guides(fill = "none") +
  scale_x_continuous(
    breaks = log10(token_breaks + 1),
    labels = token_labels,
    limits = x_limits
  ) +
  labs(
    title = "GPT-2 medium: ΔSLOR trajectory change after condition matching",
    subtitle = paste0(
      "Matched minus old continuous ΔSLOR, mean ± 1 SE across paired H2-H4 cells. ",
      "Positive values shift toward the overt member."
    ),
    x = "Tokens seen (log scale)",
    y = "Matched − old ΔSLOR",
    color = "Condition"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 10),
    strip.text = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

overlay_path <- file.path(
  figure_dir, "gpt2_medium_old_vs_matched_h2_h4_trajectories.png"
)
delta_path <- file.path(
  figure_dir, "gpt2_medium_matched_minus_old_h2_h4_trajectories.png"
)
slor_overlay_path <- file.path(
  figure_dir, "gpt2_medium_old_vs_matched_h2_h4_slor_trajectories.png"
)
slor_delta_path <- file.path(
  figure_dir, "gpt2_medium_matched_minus_old_h2_h4_slor_trajectories.png"
)
ggsave(overlay_path, overlay, width = 14, height = 16, dpi = 180, bg = "white")
ggsave(delta_path, delta_plot, width = 14, height = 16, dpi = 180, bg = "white")
ggsave(slor_overlay_path, slor_overlay, width = 14, height = 16, dpi = 180, bg = "white")
ggsave(slor_delta_path, slor_delta_plot, width = 14, height = 16, dpi = 180, bg = "white")
message("Wrote ", overlay_path)
message("Wrote ", delta_path)
message("Wrote ", slor_overlay_path)
message("Wrote ", slor_delta_path)
