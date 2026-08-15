# Condition-matched initialization evaluation

The rendered manifests in this directory describe the original one-seed-per-GPU
plan for deterministic `checkpoint_step=-1` evaluation. They passed client and
server dry-run, but the full fleet has **not** been launched.

On 2026-08-14, only `thomas-fdy-matched-init-gpt2s-v1` was submitted. A bounded
utilization measurement on its first two indexes (seeds 42 and 137) found about
2.3 GiB peak VRAM but only 0--1% sampled mean GPU utilization: the five
condition score passes are brief, while the subsequent per-cell Parquet and S3
fan-out is CPU/I/O-bound. Before either index completed, the live Job's
`spec.completions` was reduced from 12 to 2 with a server-validated patch. This
lets those two countable seeds finish without automatically starting the other
ten. No other architecture's initialization Job has been submitted.

Do not apply `job-init-all-v1.yaml` as rendered. The safe next design is a
two-stage pipeline:

1. A short GPU stage constructs and hashes one deterministic state per
   `(architecture, seed)`, scores each of the five condition-specific stimulus
   sets once, and publishes immutable representative outputs.
2. A CPU-only stage fans those scores out to the run/cell identities that share
   the same `(architecture, seed, condition)`, writes the per-cell sidecars and
   inventory-scoped provenance record, and performs all repeated Parquet/S3
   work without holding a GPU.

The two stages must retain the existing benchmark names, stimulus and inventory
hashes, state hashes, immutable-upload collision checks, and final audits. They
require a separate finite dry-run and explicit launch decision before use.
