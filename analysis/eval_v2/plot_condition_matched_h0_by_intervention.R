#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(scales)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) != 1) stop("Run this file with Rscript")
script_path <- normalizePath(sub("^--file=", "", file_arg), mustWork = TRUE)
repo <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
cli_args <- commandArgs(trailingOnly = TRUE)
all_hp <- "--all-hp" %in% cli_args
architecture <- "gpt2_small"
arch_arg <- which(cli_args == "--architecture")
if (length(arch_arg) == 1 && arch_arg < length(cli_args)) {
  architecture <- cli_args[arch_arg + 1]
}
architecture_labels <- c(
  gpt2_small = "GPT-2 small",
  gpt2_medium = "GPT-2 medium",
  gpt2_large = "GPT-2 large",
  bert_large = "BERT-large",
  lstm = "LSTM",
  mamba_370m = "Mamba 370M"
)
if (!architecture %in% names(architecture_labels)) stop("Unknown architecture: ", architecture)
architecture_label <- unname(architecture_labels[architecture])

figure_dir <- file.path(repo, "analysis", "eval_v2", "figures", "foundry_trajectories")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
output_stem <- if (all_hp) {
  paste0(architecture, "_condition_matched_all_hp_by_intervention")
} else {
  paste0(architecture, "_condition_matched_h0_by_intervention")
}
trajectory_csv <- file.path(figure_dir, paste0(output_stem, "_cell_trajectories.csv"))
summary_csv <- file.path(figure_dir, paste0(output_stem, "_summary.csv"))

builder_args <- c(
    shQuote(file.path(repo, "scripts", "build_condition_matched_h0_trajectory.py")),
    "--results-root", shQuote(file.path(
      repo, "data", "eval_results", "null_subj_v2_condition_matched_v1"
    )),
    "--architecture", architecture,
    "--output", shQuote(trajectory_csv)
)
if (!all_hp) builder_args <- c(builder_args, "--hp-rank", "0")
builder_output <- system2(
  "python3",
  builder_args,
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
    intervention %in% names(intervention_labels)
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
    )
  )

stopifnot(
  n_distinct(raw$benchmark) == 1,
  unique(raw$benchmark) == "null_subj_v2_condition_matched_v1",
  n_distinct(raw$scoring_version) == 1,
  unique(raw$scoring_version) == "null-subj-v2-condition-matched-v1",
  n_distinct(raw$stimuli_manifest_sha256) == 1,
  unique(raw$stimuli_manifest_sha256) ==
    "056ca1d2a5df745662ba501c97c434a2dde3a7cc857362d36b34caacd05d5de7"
)

plot_data <- if (all_hp) {
  shared_tokens <- exp(seq(
    log(min(raw$tokens_seen)), log(max(raw$tokens_seen)), length.out = 45L
  ))
  raw |>
    group_by(cell_id, intervention, hp_rank, seed, category) |>
    group_modify(function(.x, .y) {
      eligible <- shared_tokens[
        shared_tokens >= min(.x$tokens_seen) & shared_tokens <= max(.x$tokens_seen)
      ]
      data.frame(
        tokens_seen = eligible,
        preference = approx(
          x = log(.x$tokens_seen), y = .x$preference,
          xout = log(eligible), ties = mean
        )$y
      )
    }) |>
    ungroup()
} else {
  raw
}

summary_df <- plot_data |>
  group_by(intervention, category, tokens_seen) |>
  summarise(
    mean_preference = mean(preference),
    sd = sd(preference),
    n_cells = n_distinct(cell_id),
    n_hp_ranks = n_distinct(hp_rank),
    se = ifelse(n_cells > 1, sd / sqrt(n_cells), 0),
    lower = pmax(0, mean_preference - se),
    upper = pmin(1, mean_preference + se),
    .groups = "drop"
  )
write.csv(summary_df, summary_csv, row.names = FALSE)

cell_coverage <- raw |>
  group_by(intervention, cell_id) |>
  summarise(max_tokens = max(tokens_seen), .groups = "drop") |>
  group_by(intervention) |>
  summarise(
    cells = n(),
    full_horizon = sum(max_tokens > 1e9),
    .groups = "drop"
  ) |>
  mutate(
    label = paste0(as.character(intervention), " ", cells, " (", full_horizon, " full)")
  )
coverage_text <- paste(cell_coverage$label, collapse = "; ")

token_breaks <- c(128000, 1e6, 1e7, 1e8, 1e9, 4e9)
token_labels <- c("128K", "1M", "10M", "100M", "1B", "4B")

plot <- ggplot(
  summary_df,
  aes(
    x = tokens_seen,
    y = mean_preference,
    color = intervention,
    fill = intervention,
    group = intervention
  )
) +
  geom_hline(yintercept = 0.5, color = "grey55", linewidth = 0.35, linetype = "dotted") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.10, color = NA) +
  geom_line(linewidth = 0.72) +
  facet_wrap(~category, ncol = 2) +
  scale_x_log10(breaks = token_breaks, labels = token_labels) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.25),
    labels = label_number(accuracy = 0.01)
  ) +
  scale_color_manual(values = intervention_colors, drop = TRUE) +
  scale_fill_manual(values = intervention_colors, drop = TRUE) +
  labs(
    title = if (all_hp) {
      paste0(architecture_label, " trajectories across H0-H4 by training intervention")
    } else {
      paste0(architecture_label, " H0 trajectories by training intervention")
    },
    subtitle = if (all_hp) {
      paste0(
        "Condition-matched evaluation; binary length-normalized preference, mean ± 1 SE across ",
        "seed × hyperparameter cells.\nAvailable cells (full-horizon in parentheses): ",
        coverage_text, "."
      )
    } else {
      paste0(
        "Condition-matched evaluation; binary length-normalized preference, mean ± 1 SE across seeds. ",
        "Twelve seeds through epoch 2; six continued thereafter."
      )
    },
    x = "Tokens seen (log scale)",
    y = "P(overt form preferred)",
    color = "Training / evaluation condition",
    fill = "Training / evaluation condition",
    caption = if (all_hp) {
      paste0(
        "Items are averaged within each cell; cells are interpolated to a shared log-token grid without ",
        "extrapolation, then equally averaged. Dotted line = chance."
      )
    } else {
      "Each checkpoint is averaged over items within seed before seeds are averaged. Dotted line = chance."
    }
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
    strip.text = element_text(face = "bold", hjust = 0),
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 10),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.key.width = grid::unit(1.2, "cm")
  ) +
  guides(
    fill = "none",
    color = guide_legend(nrow = 2, byrow = TRUE)
  )

png_out <- file.path(figure_dir, paste0(output_stem, ".png"))
pdf_out <- file.path(figure_dir, paste0(output_stem, ".pdf"))
ggsave(png_out, plot, width = 12, height = 15, dpi = 220, bg = "white")
ggsave(pdf_out, plot, width = 12, height = 15, device = "pdf")
message("Wrote ", png_out)
message("Wrote ", pdf_out)
message("Wrote ", summary_csv)
