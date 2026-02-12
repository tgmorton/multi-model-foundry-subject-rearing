# Null Subject: Comprehensive Acquisition and End-State Analysis (v5)
# -------------------------------------------------------------------
# This script integrates a robust t50/end-state analysis with a
# comprehensive descriptive and visualization framework.
#
# Key Features:
#   - Progress Indicators: Provides a main progress bar for models, a
#     bootstrap progress bar, and verbose output from the optimizer.
#   - Model Caching: Saves/loads fitted glmer models to avoid re-running.
#   - Robust t50: Uses glmer with splines and bootstrapping for accuracy.
#   - End-State Analysis: Compares final model performance.
#   - Combined & Individual Visuals: Creates panel figures and individual plots.
#   - Organized Outputs: Saves all outputs to the `analysis/` directory.
# -------------------------------------------------------------------

# ============================================================================
# 1. SETUP: LIBRARIES, DIRECTORIES, AND DATA
# ============================================================================

cat("Step 1: Setting up the environment...\n")

# Load all necessary packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
  library(knitr)
  library(kableExtra)
  library(scales)
  library(splines)
  library(MASS)
  library(cowplot)
  library(forcats)
  library(progress) # NEW: For the main progress bar
  library(parallel) # For M1 Max optimization
  library(future)   # For parallel processing
  library(furrr)    # For parallel mapping
})

# Load paper figure specifications for consistent colors
source("analysis/scripts/paper_figures/figure_dimensions.R")

# Create a comprehensive directory structure for all outputs
dir.create("analysis/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/models", recursive = TRUE, showWarnings = FALSE)

# Directories for robust t50/end-state analysis
for (p in c("analysis/figures/tests/t50",
            "analysis/figures/tests/endstate",
            "analysis/figures/tests/combined",
            "analysis/tables/tests")) {
  dir.create(p, recursive = TRUE, showWarnings = FALSE)
}

# Directories for detailed descriptive/exploratory figures
for (p in c("analysis/figures/combined", "analysis/figures/baseline",
            "analysis/figures/remove_expletives", "analysis/figures/impoverish_determiners",
            "analysis/figures/remove_articles", "analysis/figures/lemmatize_verbs",
            "analysis/figures/remove_subject_pronominals")) {
  dir.create(p, recursive = TRUE, showWarnings = FALSE)
}


# Load data (only if not already in environment or if it's not a dataframe)
if (!exists("data") || !is.data.frame(data)) {
  cat("Loading data from file...\n")
  data <- read.csv("evaluation/results/all_models_null_subject_lme4_ready.csv")
} else {
  cat("Data already exists in environment - skipping load.\n")
}

# Ensure robust log_chk computation
stopifnot("checkpoint_num" %in% names(data))
data <- data %>%
  mutate(checkpoint_num = as.numeric(checkpoint_num)) %>%
  mutate(log_chk = log10(pmax(checkpoint_num, 0) + 1))

# Labeling, factors, and data prep
if (!"model_label" %in% names(data)) {
  data <- data %>%
    mutate(model_label = case_when(
      model == "exp0_baseline" ~ "Baseline",
      model == "exp1_remove_expletives" ~ "Remove Expletives",
      model == "exp2_impoverish_determiners" ~ "Impoverish Determiners",
      model == "exp3_remove_articles" ~ "Remove Articles",
      model == "exp4_lemmatize_verbs" ~ "Lemmatize Verbs",
      model == "exp5_remove_subject_pronominals" ~ "Remove Subject Pronominals",
      TRUE ~ as.character(model)
    ))
}

# Ensure correct data types for factors
data$model <- as.factor(data$model)
data$model_label <- factor(data$model_label, levels = unique(data$model_label))
data$form_type <- factor(data$form_type, levels = c("null", "overt"))
if ("item_group" %in% names(data)) data$item_group <- as.factor(data$item_group)
if ("form" %in% names(data)) data$form <- as.factor(data$form)

cat("Data loaded and prepped. Dimensions:", paste(dim(data), collapse = "x"), "\n\n")


# ============================================================================
# 2. ROBUST ANALYSIS: t50 ACQUISITION & END-STATE
# (Using GLMM, Splines, and Bootstrapping)
# ============================================================================

cat("Step 2: Running robust acquisition (t50) + end-state analysis...\n")

# Prep data and set baseline reference
stopifnot(all(c("item_id","correct","model","model_label","form_type","checkpoint_num") %in% names(data)))
set.seed(123)

data <- data %>% mutate(log_chk = log10(checkpoint_num + 1))
labels <- distinct(data, model, model_label)

baseline_model <- as.character(labels$model[labels$model_label == "Baseline"][1])
data$model <- forcats::fct_relevel(data$model, baseline_model)
cat("Baseline model set to:", baseline_model, "\n")

# Helper functions for robust analysis
grid_for_model <- function(df, n = 1500, spacing = c("linear", "log")) {
  spacing <- match.arg(spacing)
  rng <- range(df$checkpoint_num, na.rm = TRUE)
  if (spacing == "linear") {
    chk <- seq(rng[1], rng[2], length.out = n)
  } else {
    lo <- log1p(max(rng[1], 0))
    hi <- log1p(rng[2])
    chk <- unique(round(expm1(seq(lo, hi, length.out = n))))
  }
  out <- data.frame(checkpoint_num = pmax(chk, 0))
  out$log_chk <- log10(out$checkpoint_num + 1)
  out
}

fit_curve <- function(df) {
  # Auto-select optimal df using AIC comparison
  df_candidates <- 3:7  # Test df from 3 to 7
  model_comparisons <- list()
  best_fit <- NULL
  best_aic <- Inf
  best_df <- NA
  
  cat("    Testing spline degrees of freedom:\n")
  
  for (df_test in df_candidates) {
    cat("      df =", df_test, "... ")
    tryCatch({
      fit <- glmer(correct ~ ns(log_chk, df = df_test) + (1|item_id),
                   data = df, family = binomial,
                   control = glmerControl(optimizer = "nloptwrap", 
                                          optCtrl = list(maxeval = 2e5),
                                          calc.derivs = FALSE))
      aic_val <- AIC(fit)
      bic_val <- BIC(fit)
      
      # Store comparison info
      model_comparisons[[paste0("df_", df_test)]] <- list(
        df = df_test, 
        aic = aic_val, 
        bic = bic_val,
        converged = !isSingular(fit)
      )
      
      cat("AIC =", round(aic_val, 1), "BIC =", round(bic_val, 1))
      
      if (aic_val < best_aic && !isSingular(fit)) {
        best_aic <- aic_val
        best_fit <- fit
        best_df <- df_test
        cat("  ← BEST (AIC)")
      } else if (isSingular(fit)) {
        cat(" (singular)")
      }
      cat("\n")
      
    }, error = function(e) {
      cat("FAILED:", conditionMessage(e), "\n")
      model_comparisons[[paste0("df_", df_test)]] <- list(
        df = df_test, 
        aic = NA, 
        bic = NA,
        converged = FALSE,
        error = conditionMessage(e)
      )
    })
  }
  
  if (is.null(best_fit)) {
    # Check if we have any successful fits (including singular ones)
    successful_fits <- model_comparisons[!is.na(sapply(model_comparisons, `[[`, "aic"))]
    
    if (length(successful_fits) > 0) {
      # Pick lowest AIC regardless of singularity
      aics <- sapply(successful_fits, `[[`, "aic")
      best_idx <- which.min(aics)
      best_df <- successful_fits[[best_idx]]$df
      best_aic <- successful_fits[[best_idx]]$aic
      
      cat("    No non-singular fits found. Selecting lowest AIC (including singular): df =", best_df, "\n")
      
      best_fit <- glmer(correct ~ ns(log_chk, df = best_df) + (1|item_id),
                        data = df, family = binomial,
                        control = glmerControl(optimizer = "nloptwrap", 
                                               optCtrl = list(maxeval = 2e5),
                                               calc.derivs = FALSE))
    } else {
      # Ultimate fallback to df=3 if all fail
      cat("    All df tests failed, using df=3 fallback\n")
      best_fit <- glmer(correct ~ ns(log_chk, df = 3) + (1|item_id),
                        data = df, family = binomial,
                        control = glmerControl(optimizer = "nloptwrap", 
                                               optCtrl = list(maxeval = 2e5),
                                               calc.derivs = FALSE))
      best_df <- 3
      best_aic <- AIC(best_fit)
    }
  }
  
  cat("    → Selected df =", best_df, "with AIC =", round(best_aic, 1), "\n")
  
  # Store model comparison info as attribute
  attr(best_fit, "df_comparison") <- model_comparisons
  attr(best_fit, "selected_df") <- best_df
  
  return(best_fit)
}

t50_from_pred <- function(chk, p, burnin_chk = 100, find_overt = FALSE,
                          last = TRUE, min_run = 3) {
  keep <- chk >= burnin_chk & is.finite(chk) & is.finite(p)
  if (!any(keep)) return(NA_real_)
  x <- chk[keep]; y <- p[keep]
  if (length(x) < 2) return(NA_real_)

  # Boolean "on the target side" of the threshold
  on_side <- if (find_overt) (y <= 0.5) else (y >= 0.5)

  # Optional stability: require at least 'min_run' consecutive points on the side
  if (min_run > 1) {
    r <- rle(on_side)
    run_ids <- inverse.rle(with(r, list(values = seq_along(values), lengths = lengths)))
    keep_runs <- r$lengths >= min_run
    on_side <- on_side & keep_runs[run_ids]
  }

  # Where does the side flip? (sign change)
  flips <- which(diff(on_side) != 0)
  if (length(flips) == 0) return(Inf)

  idx <- if (last) flips[length(flips)] else flips[1]

  # Linear interpolation across (x[idx], x[idx+1])
  x1 <- x[idx]; x2 <- x[idx + 1]
  y1 <- y[idx]; y2 <- y[idx + 1]
  thr <- 0.5
  if (!is.finite(x1) || !is.finite(x2) || !is.finite(y1) || !is.finite(y2) || y2 == y1)
    return(x2)  # fallback

  x1 + (thr - y1) * (x2 - x1) / (y2 - y1)
}

# --- Bootstrap utilities -----------------------------------------------------

rmvnorm_safe <- function(n, mu, Sigma) {
  # draw from N(mu, Sigma); add a tiny jitter if Sigma isn't PD
  tryCatch(
    MASS::mvrnorm(n, mu = mu, Sigma = Sigma),
    error = function(e) {
      for (eps in c(1e-8, 1e-6, 1e-4)) {
        Sigma2 <- Sigma; diag(Sigma2) <- diag(Sigma2) + eps
        out <- try(MASS::mvrnorm(n, mu = mu, Sigma = Sigma2), silent = TRUE)
        if (!inherits(out, "try-error")) return(out)
      }
      # ultimate fallback: ignore covariances
      warning("mvrnorm failed; falling back to independent draws.")
      sweep(matrix(rnorm(n * length(mu), mu, sqrt(pmax(diag(Sigma), 1e-8))),
                   nrow = n, byrow = TRUE), 2, 0, `+`)
    }
  )
}

build_X <- function(fit, newdata) {
  nd <- as.data.frame(newdata[, c("checkpoint_num","log_chk")])
  stopifnot(all(c("checkpoint_num","log_chk") %in% names(nd)))
  tt <- delete.response(terms(fit))
  X  <- model.matrix(tt, data = nd)
  list(nd = nd, X = X)
}

# Draw fixed effects, predict on grid, compute ribbons + t50 CI
boot_t50 <- function(fit, newdata, nsim = 1000, burnin_chk = 100, seed = 42) {
  set.seed(seed)
  bx <- build_X(fit, newdata); nd <- bx$nd; X <- bx$X
  mu <- fixef(fit); Sigma <- as.matrix(vcov(fit))
  betas <- rmvnorm_safe(nsim, mu, Sigma)                # nsim x k
  eta_draws <- betas %*% t(X)                           # nsim x n
  p_draws   <- plogis(eta_draws)                        # nsim x n

  # point estimate (using MLE)
  p_hat <- plogis(drop(X %*% mu))

  # t50 per draw
  draws <- apply(p_draws, 1, function(p) t50_from_pred(nd$checkpoint_num, p, burnin_chk = burnin_chk))
  finite_draws <- draws[is.finite(draws)]
  ci <- if (length(finite_draws) >= 10) quantile(finite_draws, c(0.025, 0.975), na.rm = TRUE) else c(NA_real_, NA_real_)

  # point t50 from p_hat
  t50_pt <- t50_from_pred(nd$checkpoint_num, p_hat, burnin_chk = burnin_chk)

  # ribbons from same draws (coherent with CI)
  band <- tibble(
    checkpoint_num = nd$checkpoint_num,
    p_hat = p_hat,
    p_lo  = apply(p_draws, 2, quantile, 0.025, na.rm = TRUE),
    p_hi  = apply(p_draws, 2, quantile, 0.975, na.rm = TRUE)
  )

  list(t50 = t50_pt, ci = as.numeric(ci), draws = as.numeric(draws), band = band)
}

# --- 2a. Fit curves and calculate t50 for each model (with caching) ---
models <- levels(data$model)
nsim_t50 <- 1000
results <- vector("list", length(models)); names(results) <- models

# NEW: Initialize the main progress bar
pb <- progress_bar$new(
  format = "[:bar] :percent | Fitting model :model_name",
  total = length(models),
  width = 60
)

for (m in models) {
  model_name_label <- labels$model_label[labels$model == m]
  pb$tick(tokens = list(model_name = model_name_label)) # Update progress bar

  # Caching logic with smoke test
  model_filepath <- file.path("analysis/models", paste0(m, ".rds"))

  if (file.exists(model_filepath)) {
    cat("\n  -> Loading cached model for:", model_name_label, "\n")
    fit <- readRDS(model_filepath)

    # Smoke test: can we predict with a tiny, valid newdata?
    grid_test <- data.frame(checkpoint_num = c(0, 10, 100))
    grid_test$log_chk <- log10(grid_test$checkpoint_num + 1)

    ok <- TRUE
    tryCatch(
      { predict(fit, newdata = grid_test, re.form = NA) },
      error = function(e) { ok <<- FALSE; message("    ! Cached model failed predict(): ", conditionMessage(e)) }
    )

    if (!ok) {
      cat("  -> Re-fitting model due to stale/invalid cache:", model_name_label, "\n")
      dfm <- data %>% filter(model == m, form_type == "null")
      fit <- fit_curve(dfm)
      saveRDS(fit, file = model_filepath)
      cat("  -> Model re-saved to:", model_filepath, "\n")
    }
  } else {
    cat("\n  -> Fitting model (will be cached):", model_name_label, "\n")
    dfm <- data %>% filter(model == m, form_type == "null")
    fit <- fit_curve(dfm)
    saveRDS(fit, file = model_filepath)
    cat("  -> Model saved to:", model_filepath, "\n")
  }

  # Proceed with the (now loaded or fitted) model
  cat("  -> Bootstrapping t50 for:", model_name_label, "\n")
  grid <- grid_for_model(data %>% filter(model == m, form_type == "null"), 
                         n = 2000, spacing = "log")
  bt   <- boot_t50(fit, grid, nsim = nsim_t50, burnin_chk = 100)
  band <- bt$band %>% mutate(model = m)
  # Store results with df selection info
  results[[m]] <- list(
    fit = fit, 
    grid = grid, 
    t50 = bt$t50, 
    ci = bt$ci, 
    draws = bt$draws, 
    band = band,
    selected_df = attr(fit, "selected_df"),
    df_comparison = attr(fit, "df_comparison")
  )
}

# Helper: empirical two-sided p from paired draws vs baseline
emp_p_two_sided <- function(d) {
  d <- d[is.finite(d)]
  if (!length(d)) return(NA_real_)
  2 * min(mean(d <= 0), mean(d >= 0))
}

# --- 2b. Generate t50 tables and df selection summary ---
# Create df selection summary
df_selection_summary <- tibble(
  model = names(results),
  selected_df = purrr::map_dbl(results, "selected_df")
) %>%
  dplyr::left_join(labels, by = "model") %>%
  dplyr::transmute(Model = model_label, model, selected_df)

cat("Degrees of freedom selected by AIC:\n")
print(df_selection_summary)
write.csv(df_selection_summary, "analysis/tables/tests/df_selection_by_aic.csv", row.names = FALSE)

# Create detailed df comparison table
df_comparison_detailed <- purrr::map_dfr(names(results), function(m) {
  comparisons <- results[[m]]$df_comparison
  purrr::map_dfr(comparisons, function(comp) {
    tibble(
      model = m,
      df = comp$df,
      aic = comp$aic,
      bic = comp$bic,
      converged = comp$converged,
      selected = (comp$df == results[[m]]$selected_df)
    )
  })
}) %>%
  dplyr::left_join(labels, by = "model") %>%
  dplyr::select(Model = model_label, model, df, aic, bic, converged, selected) %>%
  dplyr::arrange(model, df)

write.csv(df_comparison_detailed, "analysis/tables/tests/df_comparison_detailed.csv", row.names = FALSE)
cat("Model comparison tables saved.\n")
t50_table <- tibble(
  model = names(results),
  t50   = purrr::map_dbl(results, "t50"),
  ci_lo = purrr::map(results, "ci") %>% purrr::map_dbl(1),
  ci_hi = purrr::map(results, "ci") %>% purrr::map_dbl(2)
) %>%
  left_join(labels, by = "model") %>%
  transmute(Model = model_label, model, t50_checkpoint = t50, CI_lo = ci_lo, CI_hi = ci_hi) %>%
  arrange(t50_checkpoint)

write.csv(t50_table, "analysis/tables/tests/t50_by_model_robust.csv", row.names = FALSE)

# Paired bootstrap for Δt50 vs Baseline with empirical p-values
base_draws <- results[[baseline_model]]$draws
delta_tbl <- t50_table %>%
  mutate(
    base_t50 = results[[baseline_model]]$t50,
    delta    = t50_checkpoint - base_t50,
    draws    = purrr::map(model, ~{
      d <- results[[.x]]$draws
      n <- min(length(base_draws), length(d))
      d[seq_len(n)] - base_draws[seq_len(n)]
    }),
    d_lo = purrr::map_dbl(draws, ~quantile(na.omit(.x[is.finite(.x)]), .025)),
    d_hi = purrr::map_dbl(draws, ~quantile(na.omit(.x[is.finite(.x)]), .975)),
    p_emp = purrr::map_dbl(draws, emp_p_two_sided)
  ) %>%
  filter(model != baseline_model) %>%
  dplyr::transmute(Model = Model, model, delta, d_lo, d_hi, p_emp)

write.csv(delta_tbl, "analysis/tables/tests/delta_t50_vs_baseline_robust.csv", row.names = FALSE)
cat("\nRobust t50 tables saved.\n")


# --- 2c. Comprehensive End-State Analysis (last 10% of training) ---
end_marks <- data %>%
  group_by(model) %>%
  mutate(in_last = checkpoint_num >= quantile(checkpoint_num, 0.9, na.rm = TRUE)) %>%
  ungroup()

## (1) Final NULL preference: model main effect (null only)
end_null <- end_marks %>% filter(in_last, form_type == "null")
fit_end_null <- glmer(correct ~ model + (1|item_id), data = end_null, family = binomial,
                      control = glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

emm_null <- emmeans(fit_end_null, ~ model, type = "response")
emm_null_tbl <- as_tibble(summary(emm_null)) %>%
  left_join(labels, by = "model") %>%
  transmute(Model = model_label, model, prob = prob, lo = asymp.LCL, hi = asymp.UCL)
write.csv(emm_null_tbl, "analysis/tables/tests/endstate_null_pref.csv", row.names = FALSE)

# Pairwise model comparisons (final NULL pref), Tukey + FDR
pw_null <- contrast(emm_null, method = "pairwise", adjust = "tukey", type = "response")
pw_null_tbl <- as_tibble(summary(pw_null)) %>%
  mutate(p.value.fdr = p.adjust(p.value, method = "fdr"))
write.csv(pw_null_tbl, "analysis/tables/tests/endstate_null_pref_pairwise.csv", row.names = FALSE)

# End-state NULL–OVERT gap (robust to emmeans column naming)
end_both <- end_marks %>% filter(in_last)

fit_end_gap <- glmer(
  correct ~ model * form_type + (1|item_id),
  data = end_both, family = binomial,
  control = glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))
)

emm_gap <- emmeans(fit_end_gap, ~ form_type | model, type = "response")

# Ask for CIs explicitly; compute Null - Overt on the response scale
gap_contrast <- contrast(
  emm_gap,
  method = list("NullMinusOvert" = c(1, -1)),
  by = "model", type = "response", infer = c(TRUE, TRUE)
)

gap_raw <- as_tibble(summary(gap_contrast))

# Standardize column names across emmeans versions
col_est <- intersect(c("estimate", "prob", "response", "odds.ratio"), names(gap_raw))
col_lo  <- intersect(c("lower.CL", "asymp.LCL", "LCL"), names(gap_raw))
col_hi  <- intersect(c("upper.CL", "asymp.UCL", "UCL"), names(gap_raw))
col_pval <- intersect(c("p.value", "prob"), names(gap_raw))

if (length(col_est) == 0) {
  # Ultimate fallback: create empty table with correct structure since emmeans failed
  gap_tbl <- tibble(
    model = unique(end_both$model),
    gap = NA_real_,
    lo = NA_real_, 
    hi = NA_real_,
    p.value = NA_real_
  )
} else {
  gap_tbl <- gap_raw %>%
    mutate(
      gap = .data[[col_est[1]]],
      lo  = if (length(col_lo)) .data[[col_lo[1]]] else NA_real_,
      hi  = if (length(col_hi)) .data[[col_hi[1]]] else NA_real_,
      p.value = if (length(col_pval)) .data[[col_pval[1]]] else NA_real_
    ) %>%
    transmute(model, gap, lo, hi, p.value)
}

gap_tbl <- gap_tbl %>%
  dplyr::left_join(labels, by = "model") %>%
  dplyr::transmute(Model = model_label, model, gap, lo, hi, p.value)

write.csv(gap_tbl, "analysis/tables/tests/endstate_gap_null_minus_overt.csv", row.names = FALSE)

# Pairwise: compare models on the gap
tryCatch({
  gap_by_model <- emmeans(fit_end_gap, ~ model | form_type, type = "response")
  gap_wide <- contrast(emm_gap, "revpairwise", by = "model", type = "response")
  gap_pairs <- pairs(contrast(emm_gap, list("NullMinusOvert" = c(1, -1)), by = "model"), adjust = "tukey")
  gap_pairs_tbl <- as_tibble(summary(gap_pairs))
  write.csv(gap_pairs_tbl, "analysis/tables/tests/endstate_gap_pairwise_models.csv", row.names = FALSE)
}, error = function(e) {
  cat("Gap pairwise comparisons failed:", conditionMessage(e), "\n")
})

cat("End-state analysis tables saved.\n")

# --- 2d. Generate Primary Figures from Robust Analysis ---
curves_df <- bind_rows(lapply(results, `[[`, "band")) %>%
  left_join(labels, by = "model")

t50_lines <- t50_table %>%
  transmute(model, model_label = Model, xintercept = t50_checkpoint) %>%
  filter(is.finite(xintercept))

# Faceted plot for overall comparison
p_curves_faceted <- ggplot(curves_df, aes(x = checkpoint_num, y = p_hat)) +
  geom_ribbon(aes(ymin = p_lo, ymax = p_hi), alpha = 0.15, fill = "grey70", color = NA) +
  geom_line(size = 0.8, color = "#2E86AB") +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50") +
  geom_vline(data = t50_lines, aes(xintercept = xintercept), color = "red", linetype = "dashed") +
  facet_wrap(~ model_label, scales = "free_x") +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0,1)) +
  labs(title = "Null Preference Learning Curves with Robust t50",
       subtitle = "Blue = model-based fit; grey band = 95% CI; red dashed = bootstrapped t50",
       x = "Training Checkpoint", y = "P(NULL preferred)") +
  theme_bw() +
  theme(strip.text = element_text(face = "bold"))

# Other panel plots
p_forest <- ggplot(delta_tbl, aes(y = reorder(Model, delta), x = delta)) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "gray50") +
  geom_point(size = 2) +
  geom_errorbarh(aes(xmin = d_lo, xmax = d_hi), height = 0) +
  labs(title = "Δt50 vs Baseline", x = "Δt50 (checkpoints; negative = earlier)", y = NULL) +
  theme_bw()

p_end_null <- ggplot(emm_null_tbl, aes(y = reorder(Model, prob), x = prob)) +
  geom_vline(xintercept = emm_null_tbl$prob[emm_null_tbl$Model == "Baseline"], linetype = "dashed", color = "#1f77b4") +
  geom_point(size = 2) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0) +
  scale_x_continuous(labels = scales::percent_format()) +
  labs(title = "End-State Null Preference", x = "P(NULL preferred)", y = NULL) +
  theme_bw()

p_end_gap <- ggplot(gap_tbl, aes(y = reorder(Model, gap), x = gap)) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "gray50") +
  geom_point(size = 2) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0) +
  scale_x_continuous(labels = scales::percent_format()) +
  labs(title = "End-State Gap: NULL − OVERT", x = "Probability Difference", y = NULL) +
  theme_bw()

# Combine into a final panel
panel <- cowplot::plot_grid(
  p_curves_faceted + theme(legend.position = "none"),
  cowplot::plot_grid(p_forest, p_end_null, p_end_gap, ncol = 3, rel_widths = c(1,1,1)),
  nrow = 2, rel_heights = c(1.4, 1)
)

ggsave("analysis/figures/tests/combined/acquisition_endstate_panel.pdf", panel, width = 16, height = 12)
ggsave("analysis/figures/tests/combined/acquisition_endstate_panel.png", panel, width = 16, height = 12, dpi = 300)
cat("Primary analysis panel figure saved.\n")


# --- 2e. Generate and Save Individual Model Curve Figures ---
cat("Generating individual curve plots for each model...\n")
for (m in models) {
  model_name_label <- labels$model_label[labels$model == m]

  # Filter data for the current model
  curve_data_single <- curves_df %>% filter(model == m)
  t50_line_single <- t50_lines %>% filter(model == m)

  # Create the plot for the single model
  p_single_curve <- ggplot(curve_data_single, aes(x = checkpoint_num, y = p_hat)) +
    geom_ribbon(aes(ymin = p_lo, ymax = p_hi), alpha = 0.2, fill = "grey70", color = NA) +
    geom_line(size = 1, color = "#2E86AB") +
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50") +
    geom_vline(data = t50_line_single, aes(xintercept = xintercept), color = "red", linetype = "dashed", size = 0.8) +
    scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
    labs(
      title = paste("Null Preference Learning Curve:", model_name_label),
      subtitle = "Curve from GLMM fit with 95% CI. Red dashed line indicates bootstrapped t50.",
      x = "Training Checkpoint",
      y = "P(NULL preferred)"
    ) +
    theme_bw(base_size = 14)

  # Define safe filename
  safe_filename <- gsub(" ", "_", tolower(model_name_label))

  # Save the individual plot
  ggsave(paste0("analysis/figures/tests/t50/t50_curve_", safe_filename, ".pdf"), p_single_curve, width = 8, height = 6)
  ggsave(paste0("analysis/figures/tests/t50/t50_curve_", safe_filename, ".png"), p_single_curve, width = 8, height = 6, dpi = 300)
}
cat("Individual model curve plots saved to analysis/figures/tests/t50/\n\n")


# ============================================================================
# 3. EXPLORATORY & DETAILED VISUALIZATIONS
# ============================================================================

cat("Step 3: Creating detailed and exploratory visualizations...\n")

# --- 3a. Model Comparison plots using robust curve fits ---
model_comparison_data_robust <- curves_df %>%
  transmute(
    model, model_label, checkpoint_num,
    mean_correct = p_hat,
    ci_lower = p_lo,
    ci_upper = p_hi,
    form_type = "null" # This analysis is null-only
  ) %>%
  mutate(checkpoint_num_log = log10(checkpoint_num + 1))

t50_table_log <- t50_table %>%
    mutate(t50_checkpoint_log = log10(t50_checkpoint + 1))

p_models_null_only_log <- ggplot(model_comparison_data_robust,
                                aes(x = checkpoint_num_log, y = mean_correct,
                                    color = model_label, fill = model_label)) +
  geom_line(size = 0.8) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.15, color = NA) +
  geom_vline(data = t50_table_log, aes(xintercept = t50_checkpoint_log, color = Model),
             linetype = "dashed", size = 0.7, alpha = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
  scale_x_continuous(breaks = log10(c(1, 10, 100, 1000, 10000) + 1),
                     labels = c("0", "10", "100", "1K", "10K")) +
  theme_bw() +
  labs(
    title = "Model Comparison: Null Subject Acquisition (Log Scale)",
    subtitle = "Curves from GLMM fits. Ribbons = 95% CI. Dashed lines = bootstrapped t50.",
    x = "Training Checkpoint (Log Scale)",
    y = "Null Subject Preference"
  ) +
  theme(legend.position = "bottom", legend.title = element_blank())

ggsave("analysis/figures/combined/models_comparison_null_only_log_robust.png",
       p_models_null_only_log, width = 12, height = 8, dpi = 300)
ggsave("analysis/figures/combined/models_comparison_null_only_log_robust.pdf",
       p_models_null_only_log, width = 12, height = 8)
cat("Robust model comparison figure saved.\n")

# --- 3a2. Enhanced log-scale figures with proper axis transform ---
cat("Creating enhanced log-scale acquisition figures...\n")

# Global max for consistent axis range
max_chk <- max(curves_df$checkpoint_num, na.rm = TRUE)
nice_breaks <- c(0, 10, 100, 1000, 10000)
nice_breaks <- nice_breaks[nice_breaks <= max_chk]

# Filter t50 lines to only finite values
t50_lines_clean <- t50_table %>%
  transmute(model, model_label = Model, xintercept = t50_checkpoint) %>%
  filter(is.finite(xintercept))

# Individual acquisition curves - true log axis
p_curves_log_individual <- ggplot(curves_df, aes(x = checkpoint_num, y = p_hat)) +
  geom_ribbon(aes(ymin = p_lo, ymax = p_hi), alpha = 0.15, fill = "grey70", color = NA) +
  geom_line(linewidth = 0.8, color = "#2E86AB") +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50") +
  geom_vline(data = t50_lines_clean, aes(xintercept = xintercept),
             color = "red", linetype = "dashed", linewidth = 0.6) +
  facet_wrap(~ model_label, scales = "fixed") +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
  scale_x_continuous(
    trans = scales::log1p_trans(),
    breaks = nice_breaks,
    labels = c("0","10","100","1K","10K")[seq_along(nice_breaks)]
  ) +
  labs(title = "Null Subject Acquisition Curves (Log Scale)",
       subtitle = "Fitted curves with 95% CI. Red dashed lines indicate t50.",
       x = "Training Checkpoint (log scale)", y = "P(NULL preferred)") +
  theme_bw() +
  theme(strip.text = element_text(face = "bold", size = 10),
        panel.grid.minor = element_blank())

ggsave("analysis/figures/tests/combined/acquisition_curves_log_scale.pdf", 
       p_curves_log_individual, width = 14, height = 10)
ggsave("analysis/figures/tests/combined/acquisition_curves_log_scale.png", 
       p_curves_log_individual, width = 14, height = 10, dpi = 300)

# Overlay comparison - true log axis with consistent colors
p_curves_log_overlay <- ggplot(curves_df, aes(x = checkpoint_num, y = p_hat, 
                                              color = model_label, fill = model_label)) +
  geom_ribbon(aes(ymin = p_lo, ymax = p_hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50") +
  geom_vline(data = t50_lines_clean, aes(xintercept = xintercept, color = model_label), 
             linetype = "dashed", linewidth = 0.6, alpha = 0.8) +
  scale_color_manual(values = PAPER_COLORS$models) +
  scale_fill_manual(values = PAPER_COLORS$models) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
  scale_x_continuous(
    trans = scales::log1p_trans(),
    breaks = nice_breaks,
    labels = c("0","10","100","1K","10K")[seq_along(nice_breaks)]
  ) +
  labs(title = "Model Comparison: Null Subject Acquisition (Log Scale)",
       subtitle = "GLMM fitted curves with 95% CI. Dashed lines indicate t50 points.",
       x = "Training Checkpoint (log scale)", 
       y = "P(NULL preferred)",
       color = "Model", fill = "Model") +
  theme_bw() +
  theme(legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  guides(color = guide_legend(ncol = 3), fill = guide_legend(ncol = 3))

ggsave("analysis/figures/tests/combined/acquisition_overlay_log_scale.pdf", 
       p_curves_log_overlay, width = 12, height = 8)
ggsave("analysis/figures/tests/combined/acquisition_overlay_log_scale.png", 
       p_curves_log_overlay, width = 12, height = 8, dpi = 300)

cat("Enhanced log-scale acquisition figures saved.\n")


# --- 3b. Detailed breakdowns by linguistic form (using raw summaries) ---
cat("Creating detailed plots by linguistic form...\n")

model_form_comparison_data <- data %>%
  group_by(model, model_label, checkpoint_num, form_type, form) %>%
  summarise(
    mean_correct = mean(correct, na.rm = TRUE),
    se_correct = sd(correct, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    ci_lower = mean_correct - 1.96 * se_correct,
    ci_upper = mean_correct + 1.96 * se_correct
  )

p_models_by_form <- ggplot(model_form_comparison_data,
                          aes(x = checkpoint_num, y = mean_correct, color = form, fill = form)) +
  geom_line(size = 0.6) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.1, color = NA) +
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50", size = 0.5) +
  facet_grid(model_label ~ form_type,
             labeller = labeller(form_type = c("null" = "Null Subject", "overt" = "Overt Subject"))) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
  theme_bw() +
  labs(
    title = "Model Comparison by Linguistic Form (Descriptive)",
    subtitle = "Each model shows performance on different linguistic forms. Ribbons are SE-based 95% CIs.",
    x = "Training Checkpoint",
    y = "Proportion Preferred"
  ) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.text = element_text(size = 9, face = "bold"),
    panel.spacing = unit(0.2, "lines")
  )

ggsave("analysis/figures/combined/models_comparison_by_form_descriptive.png",
       p_models_by_form, width = 12, height = 16, dpi = 300)
cat("Descriptive plots by linguistic form saved.\n\n")


# ============================================================================
# 4. DESCRIPTIVE STATISTICS TABLES
# ============================================================================

cat("Step 4: Generating descriptive statistics tables...\n")

# Helper function for descriptive stats
calc_preference_stats <- function(df) {
  df %>%
    group_by(form_type, .add = TRUE) %>%
    summarise(
      n_obs = n(),
      prop_correct = mean(correct, na.rm = TRUE),
      mean_surprisal = mean(mean_surprisal, na.rm = TRUE),
      .groups = "drop_last"
    ) %>%
    pivot_wider(names_from = form_type, values_from = c(n_obs, prop_correct, mean_surprisal)) %>%
    mutate(
      preference_diff = prop_correct_null - prop_correct_overt,
      surprisal_diff_favoring_null = mean_surprisal_overt - mean_surprisal_null
    ) %>%
    ungroup()
}

# Table 1: Overall Model Preferences (descriptive)
table1_model_prefs <- data %>%
  group_by(model, model_label) %>%
  calc_preference_stats() %>%
  transmute(Model = model_label, `Null Pref` = prop_correct_null, `Overt Pref` = prop_correct_overt, `Pref Diff` = preference_diff) %>%
  mutate(across(where(is.numeric), ~round(., 3)))
write.csv(table1_model_prefs, "analysis/tables/table1_model_preferences.csv", row.names = FALSE)

# Table 2: Model × Item Group Preferences
table2_model_itemgroup <- data %>%
  group_by(model, model_label, item_group) %>%
  calc_preference_stats() %>%
  transmute(Model = model_label, `Item Group` = item_group, `Null Pref` = prop_correct_null, `Overt Pref` = prop_correct_overt, `Pref Diff` = preference_diff) %>%
  mutate(across(where(is.numeric), ~round(., 3)))
write.csv(table2_model_itemgroup, "analysis/tables/table2_model_itemgroup_preferences.csv", row.names = FALSE)

# Table 3: Model × Form Preferences
table3_model_forms <- data %>%
  group_by(model, model_label, form) %>%
  calc_preference_stats() %>%
  transmute(Model = model_label, Form = form, `Null Pref` = prop_correct_null, `Overt Pref` = prop_correct_overt, `Pref Diff` = preference_diff) %>%
  mutate(across(where(is.numeric), ~round(., 3)))
write.csv(table3_model_forms, "analysis/tables/table3_model_forms_preferences.csv", row.names = FALSE)

cat("All descriptive statistics tables saved.\n\n")


# ============================================================================
# 5. CHANG & BERGEN HALFWAY-TO-ASYMPTOTE ANALYSIS
# ============================================================================

cat("Step 5: Computing AoA at halfway-to-asymptote (Chang & Bergen style)...\n")

# Function to find halfway-to-asymptote crossing
t_half_stat <- function(fit, grid, last_frac = 0.1, burnin_chk = 100, find_overt = FALSE) {
  p <- plogis(predict(fit, newdata = grid, re.form = NA))  # P(null)
  end_mask <- grid$checkpoint_num >= quantile(grid$checkpoint_num, 1 - last_frac, na.rm = TRUE)
  
  if (find_overt) {
    # For overt preference, work with P(overt) = 1 - P(null)
    p_end_overt <- 1 - mean(p[end_mask], na.rm = TRUE)
    thresh <- 0.5 + 0.5 * (p_end_overt - 0.5)  # halfway to overt asymptote
    # But we need to find where P(null) drops below (1 - thresh)
    thresh_null <- 1 - thresh
    
    keep <- grid$checkpoint_num >= burnin_chk
    x <- grid$checkpoint_num[keep]; y <- p[keep]
    i <- which(diff(y <= thresh_null) != 0)[1]  # P(null) drops below threshold
  } else {
    # Original: for null preference
    p_end <- mean(p[end_mask], na.rm = TRUE)
    thresh <- 0.5 + 0.5 * (p_end - 0.5)
    
    keep <- grid$checkpoint_num >= burnin_chk
    x <- grid$checkpoint_num[keep]; y <- p[keep]
    i <- which(diff(y >= thresh) != 0)[1]  # P(null) rises above threshold
  }
  
  if (is.na(i)) return(list(t = Inf, p_end = p_end, thresh = thresh))
  # linear interpolation to the crossing
  t_cross <- x[i] + (thresh - y[i]) * (x[i+1] - x[i]) / (y[i+1] - y[i])
  list(t = t_cross, p_end = p_end, thresh = thresh)
}

# AoA halfway-to-asymptote per draw
boot_t_half <- function(fit, grid, nsim = 1000, last_frac = 0.1, burnin_chk = 100, seed = 99) {
  set.seed(seed)
  bx <- build_X(fit, grid); nd <- bx$nd; X <- bx$X
  mu <- fixef(fit); Sigma <- as.matrix(vcov(fit))
  betas <- rmvnorm_safe(nsim, mu, Sigma)
  eta_draws <- betas %*% t(X)         # nsim x n
  p_draws   <- plogis(eta_draws)

  # point estimate curve
  p_hat <- plogis(drop(X %*% mu))

  # helper computes t_half for a probability curve with last crossing
  t_half_from_p <- function(chk, p, last_frac, burnin_chk, last = TRUE, min_run = 3) {
    keep <- is.finite(chk) & is.finite(p)
    chk <- chk[keep]; p <- p[keep]

    end_mask <- chk >= quantile(chk, 1 - last_frac, na.rm = TRUE)
    p_end <- mean(p[end_mask], na.rm = TRUE)
    thresh <- 0.5 + 0.5 * (p_end - 0.5)  # halfway to asymptote on P(null)

    keep2 <- chk >= burnin_chk
    x <- chk[keep2]; y <- p[keep2]
    if (length(x) < 2) return(NA_real_)

    on_side <- y >= thresh

    if (min_run > 1) {
      r <- rle(on_side)
      run_ids <- inverse.rle(with(r, list(values = seq_along(values), lengths = lengths)))
      keep_runs <- r$lengths >= min_run
      on_side <- on_side & keep_runs[run_ids]
    }

    flips <- which(diff(on_side) != 0)
    if (length(flips) == 0) return(Inf)

    idx <- if (last) flips[length(flips)] else flips[1]

    x1 <- x[idx]; x2 <- x[idx + 1]
    y1 <- y[idx]; y2 <- y[idx + 1]
    if (!is.finite(x1) || !is.finite(x2) || !is.finite(y1) || !is.finite(y2) || y2 == y1)
      return(x2)

    x1 + (thresh - y1) * (x2 - x1) / (y2 - y1)
  }

  # point estimate
  t_pt <- t_half_from_p(nd$checkpoint_num, p_hat, last_frac, burnin_chk, last = TRUE)

  # bootstrap draws
  draws <- apply(p_draws, 1, function(p) t_half_from_p(nd$checkpoint_num, p, last_frac, burnin_chk, last = TRUE))
  finite_draws <- draws[is.finite(draws)]
  ci <- if (length(finite_draws) >= 10) quantile(finite_draws, c(0.025, 0.975), na.rm = TRUE) else c(NA_real_, NA_real_)

  list(t = t_pt, ci = as.numeric(ci), draws = as.numeric(draws))
}

# Compute AoA_half for each model using existing fits/grids
cat("  -> Bootstrapping AoA_half for each model...\n")
half_res <- list()
for(m in names(results)) {
  cat("    - Processing:", m, "\n")
  half_res[[m]] <- boot_t_half(results[[m]]$fit, results[[m]]$grid, 
                               nsim = 1000, last_frac = 0.1, burnin_chk = 100)
}

# Create summary table
t_half_table <- tibble(
  model = names(half_res),
  t_half = sapply(half_res, function(x) x$t),
  ci_lo  = sapply(half_res, function(x) x$ci[1]),
  ci_hi  = sapply(half_res, function(x) x$ci[2])
) %>% 
  left_join(labels, by = "model") %>%
  transmute(Model = model_label, model, t_half_checkpoint = t_half, CI_lo = ci_lo, CI_hi = ci_hi) %>%
  arrange(t_half_checkpoint)

write.csv(t_half_table, "analysis/tables/tests/aoa_halfway_by_model.csv", row.names = FALSE)
cat("  -> AoA_half table saved to analysis/tables/tests/aoa_halfway_by_model.csv\n")

# Compute deltas vs baseline with empirical p-values
base_draws_half <- half_res[[baseline_model]]$draws
delta_half_tbl <- t_half_table %>%
  mutate(
    base_t = half_res[[baseline_model]]$t,
    delta  = t_half_checkpoint - base_t,
    draws  = purrr::map(model, ~{
      d <- half_res[[.x]]$draws
      n <- min(length(base_draws_half), length(d))
      d <- d[seq_len(n)]; b <- base_draws_half[seq_len(n)]
      d - b
    }),
    d_lo = purrr::map_dbl(draws, ~quantile(na.omit(.x[is.finite(.x)]), .025)),
    d_hi = purrr::map_dbl(draws, ~quantile(na.omit(.x[is.finite(.x)]), .975)),
    p_emp = purrr::map_dbl(draws, emp_p_two_sided)
  ) %>%
  dplyr::left_join(labels, by = "model") %>%
  filter(model != baseline_model) %>%
  dplyr::transmute(Model = model_label, model, delta, d_lo, d_hi, p_emp)

write.csv(delta_half_tbl, "analysis/tables/tests/delta_aoa_half_vs_baseline.csv", row.names = FALSE)
cat("  -> Delta AoA_half table saved to analysis/tables/tests/delta_aoa_half_vs_baseline.csv\n")

# Create forest plot
p_half_forest <- ggplot(delta_half_tbl, aes(y = reorder(Model, delta), x = delta)) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "gray50") +
  geom_point(size = 2) +
  geom_errorbarh(aes(xmin = d_lo, xmax = d_hi), height = 0) +
  labs(title = "ΔAoA (halfway-to-asymptote) vs Baseline",
       x = "Δ checkpoints (negative = earlier than Baseline)", y = NULL) +
  theme_bw()

ggsave("analysis/figures/tests/t50/delta_aoa_half_forest.pdf", p_half_forest, width = 8, height = 5)
ggsave("analysis/figures/tests/t50/delta_aoa_half_forest.png", p_half_forest, width = 8, height = 5, dpi = 300)
cat("  -> Forest plot saved to analysis/figures/tests/t50/delta_aoa_half_forest.png\n")

cat("Chang & Bergen AoA analysis complete!\n\n")

# ============================================================================
# 6. END-STATE BY ITEM GROUP AND FORM ANALYSIS
# ============================================================================

cat("Step 6: End-state analysis by item groups and forms...\n")

# End-state subset for detailed analyses
end_last <- data %>%
  group_by(model) %>%
  mutate(in_last = checkpoint_num >= quantile(checkpoint_num, 0.9, na.rm = TRUE)) %>%
  ungroup() %>%
  filter(in_last, form_type == "null")

## (1) Baseline only: item_group differences
cat("  -> Analyzing item group differences within Baseline...\n")
end_baseline <- end_last %>% filter(model == baseline_model)
fit_ig_base <- glmer(correct ~ item_group + (1|item_id), data = end_baseline, family = binomial,
                     control = glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
emm_ig_base <- emmeans(fit_ig_base, ~ item_group, type = "response")
ig_base_tbl <- as_tibble(summary(emm_ig_base))
ig_base_pairs <- summary(pairs(emm_ig_base, adjust = "tukey", type = "response")) %>%
  as_tibble() %>% mutate(p.value.fdr = p.adjust(p.value, "fdr"))
write.csv(ig_base_tbl,   "analysis/tables/tests/endstate_itemgroup_baseline.csv", row.names = FALSE)
write.csv(ig_base_pairs, "analysis/tables/tests/endstate_itemgroup_baseline_pairwise.csv", row.names = FALSE)

## (2) All models: model × item_group (with timeout and simplification)
cat("  -> Analyzing model differences by item group...\n")
cat("    Data size:", nrow(end_last), "rows. This may take several minutes...\n")

tryCatch({
  # M1 Max optimized approach: parallel by item_group + optimized settings
  library(parallel)
  library(furrr)
  
  # Set up parallel processing for M1 Max (use performance cores)
  n_cores <- min(8, parallel::detectCores())  # M1 Max has 8 performance cores
  cat("    Using", n_cores, "cores for M1 Max optimization...\n")
  
  # Option 1: Try the full interaction with M1-optimized settings
  cat("    Attempting M1-optimized full model...\n")
  fit_ig_all <- glmer(correct ~ model * item_group + (1|item_id), 
                      data = end_last, family = binomial,
                      control = glmerControl(
                        optimizer = "nloptwrap",  # Often faster on M1
                        optCtrl = list(
                          maxeval = 5e4,          # Reduced for speed
                          xtol_abs = 1e-6,        # Relaxed tolerance
                          ftol_abs = 1e-6
                        ),
                        calc.derivs = FALSE,
                        boundary.tol = 1e-5     # More lenient boundary tolerance
                      ))
  
  cat("    Model fitting completed. Computing contrasts...\n")
  
  # Parallel emmeans computation
  plan(multisession, workers = n_cores)
  
  emm_ig_all <- emmeans(fit_ig_all, ~ model | item_group, type = "response")
  # model differences within each item_group  
  ig_model_pairs <- as_tibble(summary(pairs(emm_ig_all, adjust = "tukey"))) %>%
    mutate(p.value.fdr = p.adjust(p.value, "fdr"))
  write.csv(ig_model_pairs, "analysis/tables/tests/endstate_itemgroup_model_diffs.csv", row.names = FALSE)

  # item_group differences within each model
  emm_ig_all2 <- emmeans(fit_ig_all, ~ item_group | model, type = "response")
  ig_within_model <- as_tibble(summary(pairs(emm_ig_all2, adjust = "tukey"))) %>%
    mutate(p.value.fdr = p.adjust(p.value, "fdr"))
  write.csv(ig_within_model, "analysis/tables/tests/endstate_itemgroup_within_model.csv", row.names = FALSE)
  
  plan(sequential)  # Reset to sequential
  cat("    Item group × model analysis completed successfully.\n")
  
}, error = function(e1) {
  cat("    M1-optimized approach failed. Trying parallel by-group approach...\n")
  
  # Option 2: Fit separate models by item_group in parallel (much faster)
  tryCatch({
    library(furrr)
    plan(multisession, workers = min(6, parallel::detectCores()))
    
    # Get unique item groups
    item_groups <- unique(end_last$item_group)
    cat("    Fitting", length(item_groups), "separate models in parallel...\n")
    
    # Function to fit model for one item group
    fit_one_group <- function(ig) {
      data_ig <- end_last %>% filter(item_group == ig)
      if(nrow(data_ig) < 50) return(NULL)  # Skip groups with too little data
      
      tryCatch({
        fit <- glmer(correct ~ model + (1|item_id), 
                     data = data_ig, family = binomial,
                     control = glmerControl(optimizer = "nloptwrap",
                                          optCtrl = list(maxeval = 2e4),
                                          calc.derivs = FALSE))
        
        emm <- emmeans(fit, ~ model, type = "response")
        pairs_result <- summary(pairs(emm, adjust = "tukey"))
        pairs_result$item_group <- ig
        return(pairs_result)
      }, error = function(e) NULL)
    }
    
    # Run in parallel
    results_by_group <- future_map(item_groups, fit_one_group, .progress = TRUE)
    results_by_group <- results_by_group[!sapply(results_by_group, is.null)]
    
    # Combine results
    if(length(results_by_group) > 0) {
      ig_model_pairs <- map_dfr(results_by_group, as_tibble) %>%
        mutate(p.value.fdr = p.adjust(p.value, "fdr"))
      write.csv(ig_model_pairs, "analysis/tables/tests/endstate_itemgroup_model_diffs.csv", row.names = FALSE)
      
      cat("    Parallel by-group analysis completed successfully.\n")
    }
    
    plan(sequential)
    
  }, error = function(e2) {
    cat("    Parallel approach also failed:", conditionMessage(e2), "\n")
    plan(sequential)  # Make sure to reset
  })
  
  # If we get here, both approaches failed
  cat("    All item_group × model approaches failed. Creating placeholder files...\n")
  write.csv(data.frame(note = "Analysis failed due to model complexity"), 
            "analysis/tables/tests/endstate_itemgroup_model_diffs.csv", row.names = FALSE)
  write.csv(data.frame(note = "Analysis failed due to model complexity"), 
            "analysis/tables/tests/endstate_itemgroup_within_model.csv", row.names = FALSE)
})

## (3) End-state by form (forms between models) 
cat("  -> Analyzing form differences between models...\n")
end_last_forms <- data %>%
  group_by(model) %>%
  mutate(in_last = checkpoint_num >= quantile(checkpoint_num, 0.9, na.rm = TRUE)) %>%
  ungroup() %>%
  filter(in_last, form_type == "null")  # analyzing null forms

cat("    Form analysis data size:", nrow(end_last_forms), "rows...\n")

tryCatch({
  fit_form_all <- glmer(correct ~ model * form + (1|item_id), data = end_last_forms, family = binomial,
                        control = glmerControl(optimizer="bobyqa", 
                                             optCtrl=list(maxfun=1e5),  # Reduced iterations
                                             calc.derivs = FALSE))
  
  cat("    Form model fitting completed. Computing contrasts...\n")
  emm_form_all <- emmeans(fit_form_all, ~ model | form, type = "response")
  form_model_pairs <- as_tibble(summary(pairs(emm_form_all, adjust = "tukey"))) %>%
    mutate(p.value.fdr = p.adjust(p.value, "fdr"))
  write.csv(form_model_pairs, "analysis/tables/tests/endstate_form_model_diffs.csv", row.names = FALSE)

  # forms within each model
  emm_form_all2 <- emmeans(fit_form_all, ~ form | model, type = "response")
  form_within_model <- as_tibble(summary(pairs(emm_form_all2, adjust = "tukey"))) %>%
    mutate(p.value.fdr = p.adjust(p.value, "fdr"))
  write.csv(form_within_model, "analysis/tables/tests/endstate_forms_within_model.csv", row.names = FALSE)
  
  cat("    Form analysis completed successfully.\n")
  
}, error = function(e) {
  cat("    Complex form × model analysis failed:", conditionMessage(e), "\n")
  cat("    Creating simplified analysis files...\n")
  
  # Create placeholder files to indicate what was attempted
  write.csv(data.frame(note = "Analysis failed due to model complexity"), 
            "analysis/tables/tests/endstate_form_model_diffs.csv", row.names = FALSE)
  write.csv(data.frame(note = "Analysis failed due to model complexity"), 
            "analysis/tables/tests/endstate_forms_within_model.csv", row.names = FALSE)
})

cat("Item group and form analyses complete!\n\n")


# ============================================================================
# 7.5 FIRST EPOCH ANALYSIS
# ============================================================================

cat("\n", cyan("→"), " Analyzing first epoch performance...\n")

# Calculate first epoch checkpoint per model (may vary slightly)
model_checkpoints <- dat %>%
  group_by(model, Model) %>%
  summarise(
    max_chk = max(checkpoint_num, na.rm = TRUE),
    first_epoch_target = round(max_chk / 20),
    .groups = "drop"
  )

cat("  First epoch checkpoints by model:\n")
for(i in 1:nrow(model_checkpoints)) {
  cat(sprintf("    %s: %d (from max %d)\n", 
              model_checkpoints$Model[i], 
              model_checkpoints$first_epoch_target[i],
              model_checkpoints$max_chk[i]))
}

# Extract data for each model's specific checkpoints
first_epoch_data <- dat %>%
  inner_join(model_checkpoints, by = c("model", "Model")) %>%
  mutate(
    # Find closest available checkpoint to target
    target_distance = abs(checkpoint_num - first_epoch_target)
  ) %>%
  group_by(model, Model) %>%
  # Get the 4 checkpoints closest to and before/at the first epoch target
  filter(checkpoint_num <= first_epoch_target) %>%
  arrange(desc(checkpoint_num)) %>%
  filter(checkpoint_num %in% head(unique(checkpoint_num), 4)) %>%
  ungroup() %>%
  group_by(model, Model, checkpoint_num, form_type) %>%
  summarise(
    accuracy = mean(correct, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = form_type, values_from = c(accuracy, n)) %>%
  mutate(
    null_pref = accuracy_null,
    # Calculate binomial test for null preference vs 0.5
    p_value = map2_dbl(accuracy_null * n_null, n_null, 
                       ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$p.value),
    # Determine direction
    direction = case_when(
      null_pref > 0.5 ~ "above",
      null_pref < 0.5 ~ "below",
      TRUE ~ "at"
    ),
    significant = p_value < 0.05
  )

# Show which checkpoints were actually used per model
checkpoints_used <- first_epoch_data %>%
  group_by(Model) %>%
  summarise(
    checkpoints = paste(sort(unique(checkpoint_num)), collapse=", "),
    .groups = "drop"
  )

cat("\n  Actual checkpoints analyzed:\n")
for(i in 1:nrow(checkpoints_used)) {
  cat(sprintf("    %s: %s\n", checkpoints_used$Model[i], checkpoints_used$checkpoints[i]))
}

# Summarize across the 4 checkpoints
first_epoch_summary <- first_epoch_data %>%
  group_by(model, Model) %>%
  summarise(
    mean_null_pref = mean(null_pref, na.rm = TRUE),
    sd_null_pref = sd(null_pref, na.rm = TRUE),
    # Pool the binomial test across checkpoints
    total_null_correct = sum(accuracy_null * n_null, na.rm = TRUE),
    total_null_n = sum(n_null, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    # Binomial test on pooled data
    p_value = map2_dbl(round(total_null_correct), total_null_n,
                       ~binom.test(.x, .y, p = 0.5, alternative = "two.sided")$p.value),
    direction = case_when(
      mean_null_pref > 0.5 ~ "above",
      mean_null_pref < 0.5 ~ "below",
      TRUE ~ "at"
    ),
    significant = ifelse(p_value < 0.05, "*", ""),
    # Format for display
    null_pref_pct = sprintf("%.1f%%", mean_null_pref * 100),
    interpretation = paste0(
      null_pref_pct, " ",
      "(", direction, " 50%", significant, ")"
    )
  ) %>%
  arrange(desc(mean_null_pref))

# Save detailed checkpoint data
write_csv(first_epoch_data, "analysis/tables/first_epoch_checkpoints.csv")

# Save summary table
first_epoch_table <- first_epoch_summary %>%
  select(Model, mean_null_pref, sd_null_pref, direction, p_value, interpretation) %>%
  mutate(
    `Mean (SD)` = sprintf("%.3f (%.3f)", mean_null_pref, sd_null_pref),
    `p-value` = format.pval(p_value, digits = 3)
  ) %>%
  select(Model, `Mean (SD)`, Direction = direction, `p-value`, Interpretation = interpretation)

write_csv(first_epoch_table, "analysis/tables/first_epoch_summary.csv")

cat("\nFirst Epoch Null Subject Preference (checkpoints 495-498):\n")
print(first_epoch_table)

cat("\n", green("✓"), " First epoch analysis complete.\n")

# ============================================================================
# 7. FINAL SUMMARY
# ============================================================================

cat(paste(rep("=", 60), collapse = ""), "\n")
cat("NULL SUBJECT ANALYSIS COMPLETE\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")
cat("Primary findings (robust analysis) are in:\n")
cat("  - Figures: analysis/figures/tests/\n")
cat("  - Tables:  analysis/tables/tests/\n")
cat("  - First epoch: analysis/tables/first_epoch_*.csv\n\n")
cat("Individual model plots have been saved to analysis/figures/tests/t50/\n")
cat("Exploratory figures and descriptive tables also saved under analysis/*\n")
cat("\nAnalysis finished!\n")

