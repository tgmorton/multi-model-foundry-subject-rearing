# Condition-matched evaluation results v1

This directory is the local mirror for the intervention-matched English
Foundry evaluation wave. Durable canonical Parquets live at:

`s3://thomas-subject-drop-artifacts/eval_results/null_subj_v2_condition_matched_v1/`

Expected tables are `items/`, `pairs/`, `per_token/`, and `checkpoints/`, with
one `cell_id=<run_id>.parquet` file per evaluated training run. Do not mix files
from the original `null_subj_v2` benchmark into this directory.

Pre-training initialization (`checkpoint_step=-1`, `tokens_seen=0`) is kept in
the separate durable benchmark
`null_subj_v2_condition_matched_init_v1`.  Its four result tables can be
combined with these trained-checkpoint tables only after both benchmark audits
pass.  Per-seed initialization records are namespaced by frozen inventory hash,
so a later H1 delta tranche cannot overwrite the stable tranche's provenance.

The frozen checkpoint inventory has SHA-256
`25f044a67b9eb9f2f1da168fd2f8a53cf12b6a88f919fbc5f0cc29fa669b5f37`;
the gold stimulus manifest has SHA-256
`056ca1d2a5df745662ba501c97c434a2dde3a7cc857362d36b34caacd05d5de7`.
These hashes identify the initial stable tranche. A later delta inventory must
be retained and audited separately before the tranche outputs are combined.

Pull the immutable S3 objects with metadata verification:

```bash
AWS_PROFILE=nrp python scripts/pull_eval_results.py \
  --benchmark null_subj_v2_condition_matched_v1 \
  --tables items pairs per_token checkpoints --require-sha256

AWS_PROFILE=nrp python scripts/pull_eval_results.py \
  --benchmark null_subj_v2_condition_matched_init_v1 \
  --tables items pairs per_token checkpoints initialization_records \
  --require-sha256
```

Then run `scripts/audit_condition_matched_eval_results.py` and
`scripts/audit_condition_matched_init_results.py` against the exact frozen
inventory selection. A green Kubernetes Job or an S3 object listing is not a
completion proof.

The frozen stimulus and analysis policy is documented in
`docs/eval_stimuli/condition_matched_v1.md`. The final PVC inventory and
coverage audit will be stored here alongside the pulled Parquets.
