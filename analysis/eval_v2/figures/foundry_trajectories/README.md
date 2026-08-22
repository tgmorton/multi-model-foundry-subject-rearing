# Foundry trajectory figures

These figures use the local `null_subj_v2` pair/checkpoint exports.

- `baseline_by_hyperparameter_<arch>.png`: baseline only, h0–h4, equal-weighted across seeds.
- `intervention_collapsed_<arch>.png`: five conditions, collapsed across HP ranks and seeds.
- `early_vs_continuation_<arch>.png`: early-only starts, early segments of continuation runs, and late continuation.

Scores are item-level binary preferences from normalized likelihood comparisons, averaged first within each cell/category/checkpoint and then equally across cells at shared `tokens_seen`. Initialization is checkpoint -1 at exactly 0 tokens; the x-axis uses `log10(tokens_seen + 1)` only to display that zero on a log-like axis. A cell is classified as early-only when its resolved checkpoint metadata ends by epoch 2; otherwise it is a continuation cell.

See `run_stage_counts.csv` for the actual cell counts used.

## Category-resolved early starts

- `early_starts_by_seed_category_<arch>.png`: eight evaluation-category panels per architecture for the baseline condition, with individual early-only seed trajectories and mean ±1 SE.
