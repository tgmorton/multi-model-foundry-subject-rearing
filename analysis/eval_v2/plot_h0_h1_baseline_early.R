#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

repo <- normalizePath(file.path(dirname(commandArgs(trailingOnly = FALSE)[1]), "..", ".."), mustWork = FALSE)
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) == 1) {
  script_path <- normalizePath(sub("^--file=", "", file_arg), mustWork = TRUE)
  repo <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
}

fig_dir <- file.path(repo, "analysis", "eval_v2", "figures", "foundry_trajectories")
input_path <- file.path(fig_dir, "baseline_by_hyperparameter_aggregates.csv")
token_limit <- 256000000

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
arch_labels <- c(gpt2_small = "GPT-2 small", gpt2_medium = "GPT-2 medium")
hp_labels <- c(`0` = "H0", `1` = "H1")
hp_colors <- c(H0 = "#1f77b4", H1 = "#e66101")

raw <- read.csv(input_path, stringsAsFactors = FALSE) |>
  filter(
    architecture %in% names(arch_labels),
    intervention == "baseline",
    hp_rank %in% c(0, 1),
    category %in% category_order,
    tokens_seen <= token_limit
  ) |>
  mutate(
    hp = factor(hp_labels[as.character(hp_rank)], levels = c("H0", "H1")),
    category = factor(category, levels = category_order, labels = category_labels[category_order])
  )

bin_arch <- function(d, n_positive_bins = 24) {
  positive <- d$tokens_seen[d$tokens_seen > 0]
  lo <- min(positive)
  hi <- token_limit
  edges <- 10 ^ seq(log10(lo), log10(hi), length.out = n_positive_bins + 1)
  d |>
    mutate(
      bin_id = ifelse(
        tokens_seen == 0,
        0L,
        pmin(n_positive_bins, pmax(1L, findInterval(tokens_seen, edges, all.inside = TRUE)))
      ),
      bin_center = ifelse(
        bin_id == 0,
        0,
        sqrt(edges[pmax(1L, bin_id)] * edges[pmin(n_positive_bins + 1L, bin_id + 1L)])
      )
    )
}

binned <- raw |>
  group_by(architecture) |>
  group_modify(~ bin_arch(.x)) |>
  ungroup() |>
  group_by(architecture, hp, seed, category, bin_id, bin_center) |>
  summarise(preference = mean(preference), .groups = "drop")

summary_df <- binned |>
  group_by(architecture, hp, category, bin_id, bin_center) |>
  summarise(
    mean_preference = mean(preference),
    sd = sd(preference),
    n_seeds = n_distinct(seed),
    se = ifelse(n_seeds > 1, sd / sqrt(n_seeds), 0),
    lower = pmax(0, mean_preference - se),
    upper = pmin(1, mean_preference + se),
    .groups = "drop"
  ) |>
  mutate(x = log10(bin_center + 1))

write.csv(
  summary_df,
  file.path(fig_dir, "baseline_h0_vs_h1_early_all_seeds_summary.csv"),
  row.names = FALSE
)

x_values <- c(0, 1e5, 1e6, 1e7, 1e8, token_limit)
x_labels <- c("0", "100K", "1M", "10M", "100M", "256M")

for (arch in names(arch_labels)) {
  d <- filter(summary_df, architecture == arch)
  seed_counts <- raw |>
    filter(architecture == arch) |>
    distinct(hp, seed) |>
    count(hp, name = "n")
  count_text <- paste0(seed_counts$hp, " n=", seed_counts$n, collapse = "; ")

  p <- ggplot(d, aes(x = x, y = mean_preference, color = hp, fill = hp)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.16, linewidth = 0) +
    geom_line(linewidth = 1.05) +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "grey50", linewidth = 0.45) +
    geom_vline(xintercept = log10(128000000 + 1), linetype = "dashed", color = "grey55", linewidth = 0.4) +
    facet_wrap(~ category, ncol = 2) +
    scale_color_manual(values = hp_colors, drop = FALSE) +
    scale_fill_manual(values = hp_colors, drop = FALSE) +
    scale_x_continuous(
      breaks = log10(x_values + 1),
      labels = x_labels,
      limits = c(-0.04, log10(token_limit + 1) + 0.02)
    ) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    labs(
      title = paste0(arch_labels[[arch]], ": baseline H0 versus H1 during the first two epochs"),
      subtitle = paste0(
        "Binary length-normalized likelihood preference, mean ± 1 SE across seeds (",
        count_text, "). Dashed line = end of epoch 1."
      ),
      x = "Tokens seen (log scale; 0 is exact initialization)",
      y = "P(overt preferred)",
      color = "Hyperparameter rank",
      fill = "Hyperparameter rank"
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 15),
      plot.subtitle = element_text(size = 10),
      strip.text = element_text(face = "bold", size = 10),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )

  out <- file.path(fig_dir, paste0("baseline_h0_vs_h1_early_all_seeds_", arch, ".png"))
  ggsave(out, p, width = 14, height = 16, dpi = 180, bg = "white")
  message("Wrote ", out)
}
