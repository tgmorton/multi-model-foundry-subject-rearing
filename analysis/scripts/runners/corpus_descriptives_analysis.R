###############################################################################
# Corpus Descriptive Analysis — All 8 Analyzers
#
# Loads CSV output from the corpus descriptive pipeline (per-split), produces
# summary tables, genre-comparison figures, CHILDES child-vs-adult contrasts,
# and split-consistency checks.
#
# Expects results in:
#   analysis/output/corpus_descriptives/data/{split_name}/{analyzer}.csv
#
# Outputs:
#   analysis/output/corpus_descriptives/figures/   (PDF + PNG)
#   analysis/output/corpus_descriptives/tables/    (LaTeX .tex)
###############################################################################

# ── Section 0: Setup ─────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
  library(patchwork)
  library(kableExtra)
})

# Shared figure infrastructure
source("analysis/scripts/paper_figures/figure_dimensions.R")

# Output directories
fig_dir <- "analysis/output/corpus_descriptives/figures"
tab_dir <- "analysis/output/corpus_descriptives/tables"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

# Genre colour palette (8 categories: 6 base genres + CHILDES child/adult)
GENRE_COLORS <- c(
  "BNC"             = "#1f77b4",
  "Gutenberg"       = "#ff7f0e",
  "OpenSubtitles"   = "#2ca02c",
  "SimpleWikipedia" = "#d62728",
  "Switchboard"     = "#9467bd",
  "CHILDES_child"   = "#17becf",
  "CHILDES_adult"   = "#e377c2",
  "CHILDES"         = "#8c564b"
)

GENRE_ORDER <- c(
  "CHILDES_child", "CHILDES_adult", "CHILDES",
  "BNC", "Gutenberg", "OpenSubtitles", "SimpleWikipedia", "Switchboard"
)

# ── Data loading helpers ─────────────────────────────────────────────────────

ANALYZER_NAMES <- c(
  "pronoun_inventory", "expletives", "verb_finiteness", "clause_structure",
  "wh_questions", "relative_clauses", "negation", "that_trace"
)

SPLITS <- c("train_90M", "test_10M", "pull_10M")

load_analyzer_csv <- function(analyzer, split, base_dir = "analysis/output/corpus_descriptives/data") {

  path <- file.path(base_dir, split, paste0(analyzer, ".csv"))
  if (!file.exists(path)) {
    return(NULL)
  }
  df <- read.csv(path, stringsAsFactors = FALSE)
  df
}

load_all <- function() {
  all_data <- list()
  for (analyzer in ANALYZER_NAMES) {
    frames <- list()
    for (split in SPLITS) {
      df <- load_analyzer_csv(analyzer, split)
      if (!is.null(df)) {
        frames[[split]] <- df
      }
    }
    if (length(frames) > 0) {
      all_data[[analyzer]] <- bind_rows(frames)
    }
  }
  all_data
}

cat("Loading data...\n")
DATA <- load_all()
cat("Loaded analyzers:", paste(names(DATA), collapse = ", "), "\n")

# Helper: add genre metadata columns
add_genre_helpers <- function(df) {
  df %>%
    mutate(
      genre_base = case_when(
        genre %in% c("CHILDES_child", "CHILDES_adult") ~ "CHILDES",
        TRUE ~ genre
      ),
      is_childes = genre_base == "CHILDES",
      speaker_role = case_when(
        genre == "CHILDES_child" ~ "child",
        genre == "CHILDES_adult" ~ "adult",
        TRUE ~ NA_character_
      )
    )
}

# Helper: filter to train_90M, exclude "overall" rows
train_genre <- function(df) {
  df %>%
    filter(split == "train_90M", genre != "overall") %>%
    add_genre_helpers()
}

# Helper: order genre factor for plots
order_genres <- function(df) {
  present <- intersect(GENRE_ORDER, unique(df$genre))
  df %>% mutate(genre = factor(genre, levels = present))
}

# Detect whether CHILDES speaker split data is available
has_speaker_split <- function() {
  if (is.null(DATA[["pronoun_inventory"]])) return(FALSE)
  any(DATA[["pronoun_inventory"]]$genre %in% c("CHILDES_child", "CHILDES_adult"))
}
SPEAKER_SPLIT_AVAILABLE <- has_speaker_split()
if (SPEAKER_SPLIT_AVAILABLE) {
  cat("CHILDES speaker split data detected.\n")
} else {
  cat("CHILDES speaker split data NOT yet available (Section 3 will be stubbed).\n")
}


###############################################################################
# ── Section 1: Overall Descriptives (train_90M) ─────────────────────────────
###############################################################################

cat("\n=== Section 1: Overall Descriptives ===\n")

# --- 1.1 Pronoun Inventory ---------------------------------------------------

if (!is.null(DATA[["pronoun_inventory"]])) {
  cat("  Pronoun inventory...\n")
  pi_overall <- DATA[["pronoun_inventory"]] %>%
    filter(split == "train_90M", genre == "overall")

  # Frequency by function x person x number (only rows that have these cols)
  if (all(c("function.", "person", "number", "count") %in% names(pi_overall))) {
    pi_freq <- pi_overall %>%
      group_by(function., person, number) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(total))
    cat("    Function x Person x Number (top 10):\n")
    print(head(pi_freq, 10))
  } else if (all(c("function", "person", "number", "count") %in% names(pi_overall))) {
    # Column might not have trailing dot
    pi_freq <- pi_overall %>%
      rename(function_ = `function`) %>%
      group_by(function_, person, number) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(total))
    cat("    Function x Person x Number (top 10):\n")
    print(head(pi_freq, 10))
  }

  # Top lemmas
  if ("lemma" %in% names(pi_overall) && "count" %in% names(pi_overall)) {
    top_lemmas <- pi_overall %>%
      group_by(lemma) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(total))
    cat("    Top 20 pronoun lemmas:\n")
    print(head(top_lemmas, 20))
  }

  # Subject / object ratio
  fn_col <- if ("function." %in% names(pi_overall)) "function." else "function"
  if (fn_col %in% names(pi_overall) && "count" %in% names(pi_overall)) {
    so_ratio <- pi_overall %>%
      rename(fn = all_of(fn_col)) %>%
      group_by(fn) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop")
    subj_n <- so_ratio %>% filter(fn == "subject") %>% pull(total)
    obj_n  <- so_ratio %>% filter(fn == "object")  %>% pull(total)
    if (length(subj_n) > 0 && length(obj_n) > 0 && obj_n > 0) {
      cat("    Subject/object ratio:", round(subj_n / obj_n, 2), "\n")
    }
  }
}

# --- 1.2 Expletives ----------------------------------------------------------

if (!is.null(DATA[["expletives"]])) {
  cat("  Expletives...\n")
  exp_overall <- DATA[["expletives"]] %>%
    filter(split == "train_90M", genre == "overall")

  if ("class" %in% names(exp_overall) && "count" %in% names(exp_overall)) {
    # Class counts (rows without verb_lemma, or with verb_lemma as metric)
    class_counts <- exp_overall %>%
      filter(!is.na(class)) %>%
      group_by(class) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(total))
    cat("    Expletive class counts:\n")
    print(class_counts)
  }

  if (all(c("class", "verb_lemma", "count") %in% names(exp_overall))) {
    top_verbs <- exp_overall %>%
      filter(!is.na(verb_lemma), verb_lemma != "") %>%
      group_by(class, verb_lemma) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      arrange(class, desc(total)) %>%
      group_by(class) %>%
      slice_head(n = 5)
    cat("    Top verb lemmas per class:\n")
    print(top_verbs)
  }
}

# --- 1.3 Verb Finiteness -----------------------------------------------------

if (!is.null(DATA[["verb_finiteness"]])) {
  cat("  Verb finiteness...\n")
  vf_overall <- DATA[["verb_finiteness"]] %>%
    filter(split == "train_90M", genre == "overall")

  if (all(c("verb_form", "clause_position", "count") %in% names(vf_overall))) {
    vf_xtab <- vf_overall %>%
      group_by(verb_form, clause_position) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      pivot_wider(names_from = clause_position, values_from = total, values_fill = 0)
    cat("    Verb form x Clause position:\n")
    print(vf_xtab)
  }
}

# --- 1.4 Clause Structure ----------------------------------------------------

if (!is.null(DATA[["clause_structure"]])) {
  cat("  Clause structure...\n")
  cs_overall <- DATA[["clause_structure"]] %>%
    filter(split == "train_90M", genre == "overall")

  if (all(c("clause_dep", "subject_status", "count") %in% names(cs_overall))) {
    cs_xtab <- cs_overall %>%
      filter(!is.na(clause_dep), clause_dep != "",
             !is.na(subject_status), subject_status != "") %>%
      group_by(clause_dep, subject_status) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      pivot_wider(names_from = subject_status, values_from = total, values_fill = 0)
    cat("    Subject realization by clause type:\n")
    print(cs_xtab)
  }

  if ("verb_lemma" %in% names(cs_overall)) {
    xcomp_verbs <- cs_overall %>%
      filter(!is.na(verb_lemma), verb_lemma != "") %>%
      group_by(verb_lemma) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(total))
    cat("    Top xcomp matrix verbs:\n")
    print(head(xcomp_verbs, 10))
  }
}

# --- 1.5 Wh-Questions --------------------------------------------------------

if (!is.null(DATA[["wh_questions"]])) {
  cat("  Wh-questions...\n")
  wh_overall <- DATA[["wh_questions"]] %>%
    filter(split == "train_90M", genre == "overall")

  if (all(c("extraction_type", "clause_level", "count") %in% names(wh_overall))) {
    wh_xtab <- wh_overall %>%
      filter(!is.na(extraction_type), extraction_type != "",
             !is.na(clause_level), clause_level != "") %>%
      group_by(extraction_type, clause_level) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      pivot_wider(names_from = clause_level, values_from = total, values_fill = 0)
    cat("    Extraction type x Clause level:\n")
    print(wh_xtab)
  }

  # Embedded complementizer rate
  if ("metric" %in% names(wh_overall)) {
    comp_rows <- wh_overall %>% filter(metric == "embedded_complementizer")
    if (nrow(comp_rows) > 0) {
      cat("    Embedded complementizer:\n")
      print(comp_rows %>% select(-split, -genre))
    }
  }
}

# --- 1.6 Relative Clauses ----------------------------------------------------

if (!is.null(DATA[["relative_clauses"]])) {
  cat("  Relative clauses...\n")
  rc_overall <- DATA[["relative_clauses"]] %>%
    filter(split == "train_90M", genre == "overall")

  if ("metric" %in% names(rc_overall) && "value" %in% names(rc_overall)) {
    rc_summary <- rc_overall %>%
      select(metric, value)
    cat("    Counts and proportions:\n")
    print(rc_summary)
  }
}

# --- 1.7 Negation ------------------------------------------------------------

if (!is.null(DATA[["negation"]])) {
  cat("  Negation...\n")
  neg_overall <- DATA[["negation"]] %>%
    filter(split == "train_90M", genre == "overall")

  if (all(c("position", "subject_status", "count") %in% names(neg_overall))) {
    neg_xtab <- neg_overall %>%
      filter(!is.na(position), position != "",
             !is.na(subject_status), subject_status != "") %>%
      group_by(position, subject_status) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      pivot_wider(names_from = subject_status, values_from = total, values_fill = 0)
    cat("    Position x Subject status:\n")
    print(neg_xtab)
  }

  # Position distribution
  if ("metric" %in% names(neg_overall) && "value" %in% names(neg_overall)) {
    pos_dist <- neg_overall %>%
      filter(str_detect(metric, "position_counts|total_negations")) %>%
      select(metric, value)
    if (nrow(pos_dist) > 0) {
      cat("    Position distribution:\n")
      print(pos_dist)
    }
  }
}

# --- 1.8 That-Trace ----------------------------------------------------------

if (!is.null(DATA[["that_trace"]])) {
  cat("  That-trace...\n")
  tt_overall <- DATA[["that_trace"]] %>%
    filter(split == "train_90M", genre == "overall")

  # Complementizer rate
  if ("metric" %in% names(tt_overall) && "value" %in% names(tt_overall)) {
    comp_rate <- tt_overall %>%
      filter(metric == "complementizer_rate") %>%
      select(-split, -genre)
    if (nrow(comp_rate) > 0) {
      cat("    Complementizer rate:\n")
      print(comp_rate)
    }
  }

  # Cross-tab: comp x extraction
  if (all(c("comp_present", "extraction_type", "count") %in% names(tt_overall))) {
    tt_xtab <- tt_overall %>%
      filter(!is.na(comp_present), comp_present != "",
             !is.na(extraction_type), extraction_type != "") %>%
      group_by(comp_present, extraction_type) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      pivot_wider(names_from = extraction_type, values_from = total, values_fill = 0)
    cat("    Comp x Extraction:\n")
    print(tt_xtab)
  }

  # Top bridge verbs
  if ("verb_lemma" %in% names(tt_overall) && "count" %in% names(tt_overall)) {
    bridge <- tt_overall %>%
      filter(!is.na(verb_lemma), verb_lemma != "") %>%
      group_by(verb_lemma) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(total))
    cat("    Top bridge verbs:\n")
    print(head(bridge, 10))
  }
}


###############################################################################
# ── Section 2: Genre Comparisons (train_90M) ────────────────────────────────
###############################################################################

cat("\n=== Section 2: Genre Comparisons ===\n")

# Helper: apply genre colours consistently
scale_fill_genre <- function(...) {
  scale_fill_manual(values = GENRE_COLORS, ...)
}
scale_color_genre <- function(...) {
  scale_color_manual(values = GENRE_COLORS, ...)
}

# --- 2.1 Pronoun function by genre -------------------------------------------

if (!is.null(DATA[["pronoun_inventory"]])) {
  cat("  Fig: Pronoun function by genre\n")
  fn_col <- if ("function." %in% names(DATA[["pronoun_inventory"]])) "function." else "function"

  pi_genre <- DATA[["pronoun_inventory"]] %>%
    train_genre() %>%
    order_genres() %>%
    rename(fn = all_of(fn_col)) %>%
    filter(fn %in% c("subject", "object")) %>%
    group_by(genre, fn) %>%
    summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
    group_by(genre) %>%
    mutate(prop = total / sum(total)) %>%
    ungroup()

  spec <- get_figure_specs("wide")
  p_pronoun_fn <- ggplot(pi_genre, aes(x = genre, y = prop, fill = fn)) +
    geom_col(position = "dodge", width = 0.7) +
    scale_fill_manual(values = c("subject" = PAPER_COLORS$primary,
                                 "object"  = PAPER_COLORS$secondary)) +
    scale_y_continuous(labels = percent_format()) +
    paper_theme(spec) +
    labs(title = "Pronoun Function by Genre",
         x = NULL, y = "Proportion of pronouns", fill = "Function") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  save_paper_figure(p_pronoun_fn,
                    file.path(fig_dir, "fig_pronoun_function_by_genre"), "wide")
}

# --- 2.2 Expletive class by genre --------------------------------------------

if (!is.null(DATA[["expletives"]])) {
  cat("  Fig: Expletive class by genre\n")
  exp_genre <- DATA[["expletives"]] %>%
    train_genre() %>%
    order_genres() %>%
    filter(!is.na(class), class != "") %>%
    group_by(genre, class) %>%
    summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
    group_by(genre) %>%
    mutate(prop = total / sum(total)) %>%
    ungroup()

  spec <- get_figure_specs("wide")
  p_expletive <- ggplot(exp_genre, aes(x = genre, y = prop, fill = class)) +
    geom_col(width = 0.7) +
    scale_fill_brewer(palette = "Set2") +
    scale_y_continuous(labels = percent_format()) +
    paper_theme(spec) +
    labs(title = "Expletive Class by Genre",
         x = NULL, y = "Proportion", fill = "Class") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  save_paper_figure(p_expletive,
                    file.path(fig_dir, "fig_expletive_class_by_genre"), "wide")
}

# --- 2.3 Verb finiteness by genre (heatmap) ----------------------------------

if (!is.null(DATA[["verb_finiteness"]])) {
  cat("  Fig: Verb finiteness heatmap\n")
  vf_genre <- DATA[["verb_finiteness"]] %>%
    train_genre() %>%
    order_genres() %>%
    filter(!is.na(verb_form), !is.na(clause_position)) %>%
    group_by(genre, verb_form, clause_position) %>%
    summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
    group_by(genre) %>%
    mutate(prop = total / sum(total)) %>%
    ungroup() %>%
    mutate(cell = paste(verb_form, clause_position, sep = " / "))

  spec <- get_figure_specs("multipanel")
  p_vf_heat <- ggplot(vf_genre, aes(x = cell, y = genre, fill = prop)) +
    geom_tile(color = "white") +
    scale_fill_viridis_c(labels = percent_format(), option = "C") +
    paper_theme(spec) +
    labs(title = "Verb Form / Clause Position by Genre",
         x = NULL, y = NULL, fill = "Proportion") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  save_paper_figure(p_vf_heat,
                    file.path(fig_dir, "fig_verb_finiteness_heatmap"), "multipanel")
}

# --- 2.4 Subject realization by genre (ROOT only) ----------------------------

if (!is.null(DATA[["clause_structure"]])) {
  cat("  Fig: Subject realization at ROOT by genre\n")
  cs_genre <- DATA[["clause_structure"]] %>%
    train_genre() %>%
    order_genres() %>%
    filter(!is.na(clause_dep), clause_dep == "ROOT",
           !is.na(subject_status)) %>%
    group_by(genre, subject_status) %>%
    summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
    group_by(genre) %>%
    mutate(prop = total / sum(total)) %>%
    ungroup()

  spec <- get_figure_specs("wide")
  p_subj_real <- ggplot(cs_genre, aes(x = genre, y = prop, fill = subject_status)) +
    geom_col(width = 0.7) +
    scale_fill_manual(values = c("overt"     = PAPER_COLORS$primary,
                                 "expletive" = PAPER_COLORS$accent,
                                 "none"      = PAPER_COLORS$neutral)) +
    scale_y_continuous(labels = percent_format()) +
    paper_theme(spec) +
    labs(title = "Subject Realization at ROOT by Genre",
         x = NULL, y = "Proportion", fill = "Subject status") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  save_paper_figure(p_subj_real,
                    file.path(fig_dir, "fig_subject_realization_by_genre"), "wide")
}

# --- 2.5 Wh-question type by genre -------------------------------------------

if (!is.null(DATA[["wh_questions"]])) {
  cat("  Fig: Wh-question type by genre\n")
  wh_genre <- DATA[["wh_questions"]] %>%
    train_genre() %>%
    order_genres() %>%
    filter(!is.na(extraction_type), !is.na(count)) %>%
    group_by(genre, extraction_type) %>%
    summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
    group_by(genre) %>%
    mutate(prop = total / sum(total)) %>%
    ungroup()

  spec <- get_figure_specs("wide")
  p_wh <- ggplot(wh_genre, aes(x = genre, y = prop, fill = extraction_type)) +
    geom_col(position = "dodge", width = 0.7) +
    scale_fill_brewer(palette = "Set1") +
    scale_y_continuous(labels = percent_format()) +
    paper_theme(spec) +
    labs(title = "Wh-Question Extraction Type by Genre",
         x = NULL, y = "Proportion", fill = "Extraction") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  save_paper_figure(p_wh,
                    file.path(fig_dir, "fig_wh_question_type_by_genre"), "wide")
}

# --- 2.6 Relative clause type by genre ---------------------------------------

if (!is.null(DATA[["relative_clauses"]])) {
  cat("  Fig: Relative clause type by genre\n")
  rc_genre <- DATA[["relative_clauses"]] %>%
    train_genre() %>%
    order_genres() %>%
    filter(metric %in% c("counts", "proportions"))

  # Reshape depending on structure
  if ("value" %in% names(rc_genre)) {
    # Scalar rows with metric=counts.subject / counts.object etc.
    rc_plot <- rc_genre %>%
      filter(str_detect(metric, "counts")) %>%
      mutate(rel_type = str_remove(metric, "counts\\.?")) %>%
      group_by(genre) %>%
      mutate(total = sum(as.numeric(value), na.rm = TRUE),
             prop = as.numeric(value) / total) %>%
      ungroup()
  } else if (all(c("subject", "object") %in% names(rc_genre))) {
    rc_plot <- rc_genre %>%
      filter(metric == "counts") %>%
      pivot_longer(cols = c(subject, object), names_to = "rel_type", values_to = "n") %>%
      mutate(n = as.numeric(n)) %>%
      group_by(genre) %>%
      mutate(prop = n / sum(n)) %>%
      ungroup()
  } else {
    rc_plot <- NULL
  }

  if (!is.null(rc_plot) && nrow(rc_plot) > 0) {
    spec <- get_figure_specs("wide")
    p_rc <- ggplot(rc_plot, aes(x = genre, y = prop, fill = rel_type)) +
      geom_col(position = "dodge", width = 0.7) +
      scale_fill_manual(values = c("subject" = PAPER_COLORS$primary,
                                   "object"  = PAPER_COLORS$secondary)) +
      scale_y_continuous(labels = percent_format()) +
      paper_theme(spec) +
      labs(title = "Relative Clause Type by Genre",
           x = NULL, y = "Proportion", fill = "Type") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))

    save_paper_figure(p_rc,
                      file.path(fig_dir, "fig_relative_clause_type_by_genre"), "wide")
  }
}

# --- 2.7 Negation position by genre ------------------------------------------

if (!is.null(DATA[["negation"]])) {
  cat("  Fig: Negation position by genre\n")
  neg_genre <- DATA[["negation"]] %>%
    train_genre() %>%
    order_genres() %>%
    filter(!is.na(position), !is.na(count)) %>%
    group_by(genre, position) %>%
    summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
    group_by(genre) %>%
    mutate(prop = total / sum(total)) %>%
    ungroup()

  spec <- get_figure_specs("wide")
  p_neg <- ggplot(neg_genre, aes(x = genre, y = prop, fill = position)) +
    geom_col(width = 0.7) +
    scale_fill_brewer(palette = "Dark2") +
    scale_y_continuous(labels = percent_format()) +
    paper_theme(spec) +
    labs(title = "Negation Position by Genre",
         x = NULL, y = "Proportion", fill = "Position") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  save_paper_figure(p_neg,
                    file.path(fig_dir, "fig_negation_position_by_genre"), "wide")
}

# --- 2.8 That-trace: complementizer rate by genre ----------------------------

if (!is.null(DATA[["that_trace"]])) {
  cat("  Fig: Complementizer rate by genre\n")
  tt_genre <- DATA[["that_trace"]] %>%
    train_genre() %>%
    order_genres()

  # Build comp rate from metric rows or from comp_present column
  if ("metric" %in% names(tt_genre) && "value" %in% names(tt_genre)) {
    comp_rate <- tt_genre %>%
      filter(metric == "complementizer_rate") %>%
      select(genre, everything())
    # May have with_that / without_that as columns
    if (all(c("with_that", "without_that") %in% names(comp_rate))) {
      comp_rate <- comp_rate %>%
        mutate(with_that = as.numeric(with_that),
               without_that = as.numeric(without_that),
               total = with_that + without_that,
               rate = with_that / total)
    }
  } else if ("comp_present" %in% names(tt_genre)) {
    comp_rate <- tt_genre %>%
      filter(!is.na(comp_present)) %>%
      mutate(comp_label = ifelse(comp_present == "True" | comp_present == TRUE,
                                 "with_that", "without_that")) %>%
      group_by(genre, comp_label) %>%
      summarise(n = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      pivot_wider(names_from = comp_label, values_from = n, values_fill = 0) %>%
      mutate(total = with_that + without_that,
             rate = with_that / total)
  } else {
    comp_rate <- NULL
  }

  if (!is.null(comp_rate) && nrow(comp_rate) > 0 && "rate" %in% names(comp_rate)) {
    spec <- get_figure_specs("wide")
    p_comp <- ggplot(comp_rate, aes(x = genre, y = rate, fill = genre)) +
      geom_col(width = 0.7, show.legend = FALSE) +
      scale_fill_genre() +
      scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
      paper_theme(spec) +
      labs(title = "That-Complementizer Rate by Genre",
           x = NULL, y = "Rate (with 'that')") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))

    save_paper_figure(p_comp,
                      file.path(fig_dir, "fig_comp_rate_by_genre"), "wide")
  }
}


###############################################################################
# ── Section 3: CHILDES Child vs Adult ────────────────────────────────────────
###############################################################################

cat("\n=== Section 3: CHILDES Child vs Adult ===\n")

if (!SPEAKER_SPLIT_AVAILABLE) {
  cat("  Speaker-split data not yet available.\n")
  cat("  Re-run pipeline with CHILDES speaker split, then re-run this script.\n")
} else {
  cat("  Generating child vs adult comparisons...\n")

  # Helper: extract CHILDES child/adult rows for an analyzer
  childes_pair <- function(df) {
    df %>%
      filter(split == "train_90M",
             genre %in% c("CHILDES_child", "CHILDES_adult")) %>%
      mutate(role = ifelse(genre == "CHILDES_child", "child", "adult"))
  }

  # --- 3.1 Pronouns: person distribution by role ---
  if (!is.null(DATA[["pronoun_inventory"]])) {
    cat("    Pronoun person by role\n")
    fn_col <- if ("function." %in% names(DATA[["pronoun_inventory"]])) "function." else "function"
    pi_ch <- DATA[["pronoun_inventory"]] %>%
      childes_pair() %>%
      rename(fn = all_of(fn_col)) %>%
      group_by(role, person) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      group_by(role) %>%
      mutate(prop = total / sum(total)) %>%
      ungroup()

    spec <- get_figure_specs("regular")
    p_pi_ch <- ggplot(pi_ch, aes(x = person, y = prop, fill = role)) +
      geom_col(position = "dodge", width = 0.7) +
      scale_fill_manual(values = c("child" = GENRE_COLORS[["CHILDES_child"]],
                                   "adult" = GENRE_COLORS[["CHILDES_adult"]])) +
      scale_y_continuous(labels = percent_format()) +
      paper_theme(spec) +
      labs(title = "Pronoun Person: Child vs Adult",
           x = "Person", y = "Proportion", fill = "Role")

    save_paper_figure(p_pi_ch,
                      file.path(fig_dir, "fig_childes_pronoun_person"), "regular")
  }

  # --- 3.2 Verb finiteness: form at ROOT by role ---
  if (!is.null(DATA[["verb_finiteness"]])) {
    cat("    Verb finiteness at ROOT by role\n")
    vf_ch <- DATA[["verb_finiteness"]] %>%
      childes_pair() %>%
      filter(clause_position == "ROOT") %>%
      group_by(role, verb_form) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      group_by(role) %>%
      mutate(prop = total / sum(total)) %>%
      ungroup()

    spec <- get_figure_specs("regular")
    p_vf_ch <- ggplot(vf_ch, aes(x = verb_form, y = prop, fill = role)) +
      geom_col(position = "dodge", width = 0.7) +
      scale_fill_manual(values = c("child" = GENRE_COLORS[["CHILDES_child"]],
                                   "adult" = GENRE_COLORS[["CHILDES_adult"]])) +
      scale_y_continuous(labels = percent_format()) +
      paper_theme(spec) +
      labs(title = "Verb Finiteness at ROOT: Child vs Adult",
           x = "Verb form", y = "Proportion", fill = "Role")

    save_paper_figure(p_vf_ch,
                      file.path(fig_dir, "fig_childes_verb_finiteness"), "regular")
  }

  # --- 3.3 Clause structure: subject realization at ROOT by role ---
  if (!is.null(DATA[["clause_structure"]])) {
    cat("    Subject realization at ROOT by role\n")
    cs_ch <- DATA[["clause_structure"]] %>%
      childes_pair() %>%
      filter(clause_dep == "ROOT", !is.na(subject_status)) %>%
      group_by(role, subject_status) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      group_by(role) %>%
      mutate(prop = total / sum(total)) %>%
      ungroup()

    spec <- get_figure_specs("regular")
    p_cs_ch <- ggplot(cs_ch, aes(x = subject_status, y = prop, fill = role)) +
      geom_col(position = "dodge", width = 0.7) +
      scale_fill_manual(values = c("child" = GENRE_COLORS[["CHILDES_child"]],
                                   "adult" = GENRE_COLORS[["CHILDES_adult"]])) +
      scale_y_continuous(labels = percent_format()) +
      paper_theme(spec) +
      labs(title = "Subject Realization at ROOT: Child vs Adult",
           x = "Subject status", y = "Proportion", fill = "Role")

    save_paper_figure(p_cs_ch,
                      file.path(fig_dir, "fig_childes_subject_realization"), "regular")
  }

  # --- 3.4 Wh-questions: extraction type by role ---
  if (!is.null(DATA[["wh_questions"]])) {
    cat("    Wh-question extraction by role\n")
    wh_ch <- DATA[["wh_questions"]] %>%
      childes_pair() %>%
      filter(!is.na(extraction_type)) %>%
      group_by(role, extraction_type) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      group_by(role) %>%
      mutate(prop = total / sum(total)) %>%
      ungroup()

    spec <- get_figure_specs("regular")
    p_wh_ch <- ggplot(wh_ch, aes(x = extraction_type, y = prop, fill = role)) +
      geom_col(position = "dodge", width = 0.7) +
      scale_fill_manual(values = c("child" = GENRE_COLORS[["CHILDES_child"]],
                                   "adult" = GENRE_COLORS[["CHILDES_adult"]])) +
      scale_y_continuous(labels = percent_format()) +
      paper_theme(spec) +
      labs(title = "Wh-Question Type: Child vs Adult",
           x = "Extraction type", y = "Proportion", fill = "Role")

    save_paper_figure(p_wh_ch,
                      file.path(fig_dir, "fig_childes_wh_questions"), "regular")
  }

  # --- 3.5 Negation: position by role ---
  if (!is.null(DATA[["negation"]])) {
    cat("    Negation position by role\n")
    neg_ch <- DATA[["negation"]] %>%
      childes_pair() %>%
      filter(!is.na(position)) %>%
      group_by(role, position) %>%
      summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
      group_by(role) %>%
      mutate(prop = total / sum(total)) %>%
      ungroup()

    spec <- get_figure_specs("regular")
    p_neg_ch <- ggplot(neg_ch, aes(x = position, y = prop, fill = role)) +
      geom_col(position = "dodge", width = 0.7) +
      scale_fill_manual(values = c("child" = GENRE_COLORS[["CHILDES_child"]],
                                   "adult" = GENRE_COLORS[["CHILDES_adult"]])) +
      scale_y_continuous(labels = percent_format()) +
      paper_theme(spec) +
      labs(title = "Negation Position: Child vs Adult",
           x = "Position", y = "Proportion", fill = "Role")

    save_paper_figure(p_neg_ch,
                      file.path(fig_dir, "fig_childes_negation"), "regular")
  }

  # --- 3.6 Summary table: child/adult proportions and ratios ---
  cat("    Building child vs adult summary table\n")
  summary_rows <- list()

  # Helper to add a ratio row
  add_ratio <- function(label, child_val, adult_val) {
    ratio <- if (!is.na(adult_val) && adult_val > 0) child_val / adult_val else NA_real_
    data.frame(
      measure = label,
      child = round(child_val, 4),
      adult = round(adult_val, 4),
      child_adult_ratio = round(ratio, 2),
      stringsAsFactors = FALSE
    )
  }

  # Pronoun 1st person proportion
  if (exists("pi_ch") && nrow(pi_ch) > 0) {
    p1_child <- pi_ch %>% filter(role == "child", person == "1") %>% pull(prop)
    p1_adult <- pi_ch %>% filter(role == "adult", person == "1") %>% pull(prop)
    if (length(p1_child) > 0 && length(p1_adult) > 0) {
      summary_rows[[length(summary_rows) + 1]] <-
        add_ratio("Pronoun: 1st person proportion", p1_child, p1_adult)
    }
  }

  # Subject realization: missing subjects
  if (exists("cs_ch") && nrow(cs_ch) > 0) {
    none_child <- cs_ch %>% filter(role == "child", subject_status == "none") %>% pull(prop)
    none_adult <- cs_ch %>% filter(role == "adult", subject_status == "none") %>% pull(prop)
    if (length(none_child) > 0 && length(none_adult) > 0) {
      summary_rows[[length(summary_rows) + 1]] <-
        add_ratio("Missing subjects at ROOT", none_child, none_adult)
    }
  }

  # Negation: pre-verbal proportion
  if (exists("neg_ch") && nrow(neg_ch) > 0) {
    pre_child <- neg_ch %>% filter(role == "child", position == "pre") %>% pull(prop)
    pre_adult <- neg_ch %>% filter(role == "adult", position == "pre") %>% pull(prop)
    if (length(pre_child) > 0 && length(pre_adult) > 0) {
      summary_rows[[length(summary_rows) + 1]] <-
        add_ratio("Negation: pre-verbal proportion", pre_child, pre_adult)
    }
  }

  if (length(summary_rows) > 0) {
    summary_table <- bind_rows(summary_rows)
    cat("    Child vs Adult summary:\n")
    print(summary_table)
  }
}


###############################################################################
# ── Section 4: Split Consistency ─────────────────────────────────────────────
###############################################################################

cat("\n=== Section 4: Split Consistency ===\n")

# For each analyzer, compute per-category proportions in train_90M vs
# test_10M / pull_10M, then scatterplot.

make_split_consistency_plot <- function(df, analyzer_name, category_cols) {
  # Build proportions per split x genre x category
  props <- df %>%
    filter(genre != "overall") %>%
    mutate(count = as.numeric(count)) %>%
    group_by(split, genre, across(all_of(category_cols))) %>%
    summarise(n = sum(count, na.rm = TRUE), .groups = "drop") %>%
    group_by(split, genre) %>%
    mutate(prop = n / sum(n)) %>%
    ungroup()

  # Create a category label for joining
  props <- props %>%
    unite("category", all_of(category_cols), sep = "_", remove = FALSE)

  # Pivot: one column per split
  wide <- props %>%
    select(split, genre, category, prop) %>%
    pivot_wider(names_from = split, values_from = prop, values_fill = 0)

  plots <- list()

  for (comp_split in c("test_10M", "pull_10M")) {
    if (!comp_split %in% names(wide)) next

    plot_df <- wide %>%
      filter(!is.na(.data[["train_90M"]]), !is.na(.data[[comp_split]]))

    if (nrow(plot_df) == 0) next

    r_val <- tryCatch(
      cor(plot_df$train_90M, plot_df[[comp_split]], use = "complete.obs"),
      error = function(e) NA_real_
    )
    r_label <- if (!is.na(r_val)) paste0("r = ", round(r_val, 3)) else ""

    spec <- get_figure_specs("square")
    p <- ggplot(plot_df, aes(x = train_90M, y = .data[[comp_split]], color = genre)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
      geom_point(alpha = 0.7, size = 1.5) +
      scale_color_genre() +
      paper_theme(spec) +
      labs(title = paste0(analyzer_name, ": train vs ", comp_split),
           subtitle = r_label,
           x = "train_90M proportion",
           y = paste0(comp_split, " proportion"),
           color = "Genre")

    plots[[comp_split]] <- p
  }

  plots
}

# Run split consistency for analyzers with suitable category columns
split_configs <- list(
  list(name = "verb_finiteness", cols = c("verb_form", "clause_position")),
  list(name = "negation",        cols = c("position", "subject_status")),
  list(name = "wh_questions",    cols = c("extraction_type", "clause_level"))
)

# Also add pronoun_inventory if function column exists
if (!is.null(DATA[["pronoun_inventory"]])) {
  fn_col <- if ("function." %in% names(DATA[["pronoun_inventory"]])) "function." else "function"
  if (all(c(fn_col, "person", "count") %in% names(DATA[["pronoun_inventory"]]))) {
    split_configs[[length(split_configs) + 1]] <-
      list(name = "pronoun_inventory", cols = c(fn_col, "person"))
  }
}

for (cfg in split_configs) {
  df <- DATA[[cfg$name]]
  if (is.null(df)) next
  if (!all(c(cfg$cols, "count") %in% names(df))) next

  cat("  Split consistency:", cfg$name, "\n")
  plots <- make_split_consistency_plot(df, cfg$name, cfg$cols)
  for (split_name in names(plots)) {
    fname <- paste0("fig_split_consistency_", cfg$name, "_", split_name)
    save_paper_figure(plots[[split_name]], file.path(fig_dir, fname), "square")
  }
}


###############################################################################
# ── Section 5: Export (LaTeX tables) ─────────────────────────────────────────
###############################################################################

cat("\n=== Section 5: Export ===\n")

# --- 5.1 Overall summary table -----------------------------------------------

if (!is.null(DATA[["pronoun_inventory"]]) && !is.null(DATA[["expletives"]]) &&
    !is.null(DATA[["verb_finiteness"]]) && !is.null(DATA[["clause_structure"]]) &&
    !is.null(DATA[["negation"]]) && !is.null(DATA[["relative_clauses"]]) &&
    !is.null(DATA[["wh_questions"]]) && !is.null(DATA[["that_trace"]])) {

  cat("  Building overall summary LaTeX table\n")

  # Collect one key statistic per analyzer from train_90M overall
  overall_stats <- list()

  # Pronoun inventory: total pronouns
  pi_ov <- DATA[["pronoun_inventory"]] %>%
    filter(split == "train_90M", genre == "overall", !is.na(count))
  overall_stats[["Pronoun inventory"]] <- paste0(
    format(sum(as.numeric(pi_ov$count), na.rm = TRUE), big.mark = ","), " tokens"
  )

  # Expletives: total
  exp_ov <- DATA[["expletives"]] %>%
    filter(split == "train_90M", genre == "overall", !is.na(count))
  overall_stats[["Expletives"]] <- paste0(
    format(sum(as.numeric(exp_ov$count), na.rm = TRUE), big.mark = ","), " instances"
  )

  # Verb finiteness: total
  vf_ov <- DATA[["verb_finiteness"]] %>%
    filter(split == "train_90M", genre == "overall", !is.na(count))
  overall_stats[["Verb finiteness"]] <- paste0(
    format(sum(as.numeric(vf_ov$count), na.rm = TRUE), big.mark = ","), " clauses"
  )

  # Clause structure: total finite clauses
  cs_ov <- DATA[["clause_structure"]] %>%
    filter(split == "train_90M", genre == "overall", !is.na(clause_dep), !is.na(count))
  overall_stats[["Clause structure"]] <- paste0(
    format(sum(as.numeric(cs_ov$count), na.rm = TRUE), big.mark = ","), " clauses"
  )

  # Wh-questions: total
  wh_ov <- DATA[["wh_questions"]] %>%
    filter(split == "train_90M", genre == "overall", !is.na(extraction_type), !is.na(count))
  overall_stats[["Wh-questions"]] <- paste0(
    format(sum(as.numeric(wh_ov$count), na.rm = TRUE), big.mark = ","), " questions"
  )

  # Relative clauses
  rc_ov <- DATA[["relative_clauses"]] %>%
    filter(split == "train_90M", genre == "overall")
  if ("metric" %in% names(rc_ov) && "value" %in% names(rc_ov)) {
    total_rc <- rc_ov %>% filter(metric == "total") %>% pull(value)
    if (length(total_rc) > 0) {
      overall_stats[["Relative clauses"]] <- paste0(
        format(as.numeric(total_rc[1]), big.mark = ","), " clauses"
      )
    }
  }

  # Negation: total
  neg_ov <- DATA[["negation"]] %>%
    filter(split == "train_90M", genre == "overall", !is.na(position), !is.na(count))
  overall_stats[["Negation"]] <- paste0(
    format(sum(as.numeric(neg_ov$count), na.rm = TRUE), big.mark = ","), " negations"
  )

  # That-trace: total bridge verb environments
  tt_ov <- DATA[["that_trace"]] %>%
    filter(split == "train_90M", genre == "overall", !is.na(verb_lemma), !is.na(count))
  overall_stats[["That-trace"]] <- paste0(
    format(sum(as.numeric(tt_ov$count), na.rm = TRUE), big.mark = ","), " bridge-verb comps"
  )

  summary_df <- data.frame(
    Analyzer = names(overall_stats),
    Summary  = unlist(overall_stats),
    stringsAsFactors = FALSE
  )
  rownames(summary_df) <- NULL

  cat("  Overall summary:\n")
  print(summary_df)

  # Export as LaTeX
  tex_path <- file.path(tab_dir, "tab_overall_summary.tex")
  kbl(summary_df, format = "latex", booktabs = TRUE,
      caption = "Corpus descriptive analysis: overall counts (train\\_90M)",
      label = "tab:corpus_descriptives_summary") %>%
    kable_styling(latex_options = "hold_position") %>%
    writeLines(tex_path)
  cat("  Saved:", tex_path, "\n")
}

# --- 5.2 Genre comparison tables (one per analyzer) --------------------------

write_genre_table <- function(df, caption, label, filename) {
  tex_path <- file.path(tab_dir, filename)
  kbl(df, format = "latex", booktabs = TRUE,
      caption = caption, label = label) %>%
    kable_styling(latex_options = c("hold_position", "scale_down")) %>%
    writeLines(tex_path)
  cat("  Saved:", tex_path, "\n")
}

# Subject realization at ROOT (key null-subject table)
if (!is.null(DATA[["clause_structure"]])) {
  cs_tab <- DATA[["clause_structure"]] %>%
    train_genre() %>%
    filter(clause_dep == "ROOT", !is.na(subject_status)) %>%
    group_by(genre, subject_status) %>%
    summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
    group_by(genre) %>%
    mutate(prop = round(total / sum(total), 3)) %>%
    ungroup() %>%
    select(genre, subject_status, prop) %>%
    pivot_wider(names_from = subject_status, values_from = prop, values_fill = 0)

  write_genre_table(
    cs_tab,
    "Subject realization at ROOT by genre (proportions, train\\_90M)",
    "tab:subject_realization_genre",
    "tab_subject_realization_genre.tex"
  )
}

# Negation position by genre
if (!is.null(DATA[["negation"]])) {
  neg_tab <- DATA[["negation"]] %>%
    train_genre() %>%
    filter(!is.na(position), position != "") %>%
    group_by(genre, position) %>%
    summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
    group_by(genre) %>%
    mutate(prop = round(total / sum(total), 3)) %>%
    ungroup() %>%
    select(genre, position, prop) %>%
    pivot_wider(names_from = position, values_from = prop, values_fill = 0)

  write_genre_table(
    neg_tab,
    "Negation position by genre (proportions, train\\_90M)",
    "tab:negation_genre",
    "tab_negation_genre.tex"
  )
}

# Verb finiteness at ROOT by genre
if (!is.null(DATA[["verb_finiteness"]])) {
  vf_tab <- DATA[["verb_finiteness"]] %>%
    train_genre() %>%
    filter(clause_position == "ROOT") %>%
    group_by(genre, verb_form) %>%
    summarise(total = sum(as.numeric(count), na.rm = TRUE), .groups = "drop") %>%
    group_by(genre) %>%
    mutate(prop = round(total / sum(total), 3)) %>%
    ungroup() %>%
    select(genre, verb_form, prop) %>%
    pivot_wider(names_from = verb_form, values_from = prop, values_fill = 0)

  write_genre_table(
    vf_tab,
    "Verb finiteness at ROOT by genre (proportions, train\\_90M)",
    "tab:verb_finiteness_genre",
    "tab_verb_finiteness_genre.tex"
  )
}

# --- 5.3 CHILDES child vs adult table ----------------------------------------

if (SPEAKER_SPLIT_AVAILABLE && exists("summary_table") && nrow(summary_table) > 0) {
  write_genre_table(
    summary_table,
    "CHILDES child vs adult: key proportions (train\\_90M)",
    "tab:childes_child_adult",
    "tab_childes_child_adult.tex"
  )
}

cat("\n=== Done ===\n")
