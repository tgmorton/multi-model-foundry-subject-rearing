# Condition-matched evaluation results v1

This directory is the local mirror for the intervention-matched English
Foundry evaluation wave. Durable canonical Parquets live at:

`s3://thomas-subject-drop-artifacts/eval_results/null_subj_v2_condition_matched_v1/`

Expected tables are `items/`, `pairs/`, `per_token/`, and `checkpoints/`, with
one `cell_id=<run_id>.parquet` file per evaluated training run. Do not mix files
from the original `null_subj_v2` benchmark into this directory.

The frozen stimulus and analysis policy is documented in
`docs/eval_stimuli/condition_matched_v1.md`. The final PVC inventory and
coverage audit will be stored here alongside the pulled Parquets.
