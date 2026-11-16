# Talk-deck exports: which script writes which file

All files land in `analysis/eval_v2/figures/deck_exports/` of the main checkout. Run from the main checkout
with `python3.13`; `<main>` is the main checkout, `<key>` is
`evaluation/stimuli/null-subj-test10m-robbi-v1/item_key.csv` on branch `eval/bert-v2`.
Results are pulled first with `scripts/pull_eval_results.py --benchmark <b> --dest data/eval_results`.

| Files | Script | Command |
|---|---|---|
| `s1_baseline_trajectories`, `s1_condition_trajectories`, `s1_item_group_trajectories`, `s2_order_trajectories`, `s2_order_endstate_by_percent`, `s2_order_effect_by_percent`, `README.txt` | `deck_exports.py` | `--data data/eval_results --out <exports>` |
| `s1_item_group_by_condition_trajectories`, pooled-GPT-2 rows appended to the two `s1_*` files, `README_item_groups.txt` | `deck_exports_item_groups.py` | same arguments; run after `deck_exports.py`, which rewrites the `s1_*` files |
| `arg_decile_by_order`, `arg_decile_order_minus_reference`, `arg_balanced_endstate`, `arg_matched_item_deciles` | `deck_exports_argument.py --only arg` | `--data data/eval_results --key <key> --rater-scores data/recoverability/analysis/eval_stimuli --out <exports>` |
| `bridge_order_effect_by_condition`, `s2_item_group_endstate`, `s2_item_group_endstate_by_condition`, `s1_path_endpoint`, `s1_distal_endstate` | `deck_exports_argument.py --only bridge groups groups_by_condition path distal` | as above |
| `arg_rater_free_slopes`, `arg_rater_free_by_decile`, `arg_rater_free_agreement`, `arg_rater_free_crosstab`, `README_rater_free.txt` | `rater_free_regrade.py` | `--results data/eval_results/null_subj_test10m_robbi_v1 --key <key> --out <exports>` |
| `arg_decile_trajectories`, `arg_decile_trajectories_minus_random` | `deck_exports_decile_trajectories.py` | `--data data/eval_results --key <nat-root>/item_key.csv --out <exports>` |

Reports built from the same data:

| PDF | Script |
|---|---|
| `figures/eval_recoverability/eval_testset_results_robbi.pdf` | `eval_recoverability_testset_results.py` |
| `figures/eval_recoverability/decile_reanalysis.pdf` | `decile_reanalysis_report.py` (reads the export tables) |
| `figures/production_interventions/enrichment_rescore_by_intervention.pdf` | `enrich_rescore_by_intervention.py` |
| `figures/natural_drop/natural_drop_recoverability.pdf` | `analysis/recoverability/natural_drop/{build_sites,score_sites,analyze,report}.py` (`score_sites` and `analyze` need `.venv/bin/python`) |

Conventions: Study 1 = full-length seeds at the winning hyperparameter setting, early-only seeds added up to
256M tokens, untrained model at `tokens_seen = 0`, enrichment from the fixed items (matched v2). Study 2 =
one run per cell; intervals are over items. End state = mean of a run's last three checkpoints.
One-off analyses are in `adhoc/`; its README lists the deck numbers that have no script yet.
