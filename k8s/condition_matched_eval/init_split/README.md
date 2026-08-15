# Split condition-matched initialization evaluation (v2)

These manifests are a **prepared plan, not launch authorization**. Do not apply
either `job-init-gpu-all-v2.yaml` or `job-init-cpu-all-v2.yaml` directly.

The earlier one-stage GPT-2-small measurement showed that checkpoint -1 model
scoring uses a GPU briefly, while repeated per-cell Parquet/S3 fanout is
CPU/I/O-bound. The v2 plan therefore separates the work:

1. `job-init-gpu-*-v2.yaml` constructs and hashes one deterministic state per
   `(architecture, seed)`, evaluates the five matched stimulus conditions once,
   and publishes one representative result per condition.
2. Only after the corresponding GPU Job is terminal and its representative
   outputs pass integrity checks, `job-init-cpu-*-v2.yaml` materializes the
   remaining HP/cell identities without requesting a GPU.

The frozen trained-checkpoint inventory SHA-256 is
`25f044a67b9eb9f2f1da168fd2f8a53cf12b6a88f919fbc5f0cc29fa669b5f37`.
The gold matched-stimulus manifest SHA-256 is
`056ca1d2a5df745662ba501c97c434a2dde3a7cc857362d36b34caacd05d5de7`.
Both stages target `null_subj_v2_condition_matched_init_v1` and pin code commit
`ebf8a6c` plus the established evaluator image digest.

The plan contains 70 remaining initialization groups: 10 GPT-2-small seeds and
12 seeds for each other architecture. GPT-2-small seeds 42 and 137 are excluded
because all 50 of their stable-inventory cells already passed the exact
initialization audit. Architectures whose H1 training lanes remain mutable
exclude HP rank 1; those cells can be added later through CPU fanout from the
same state/condition representatives, without another GPU scoring pass.

Safety gates before launch:

- First render and submit exactly one GPT-2-small representative-only GPU index
  as the v2 resource sentinel. It must verify the state/stimulus/inventory
  hashes and measure GPU, VRAM, CPU, RAM, and elapsed scoring time.
- Do not scale the GPU stage until that new workload shape demonstrates useful
  GPU utilization and appropriate requests on the established 24 GB pool.
- Run exactly one matching CPU fanout index next and audit its expected cells
  and S3 SHA metadata before launching the remaining CPU indexes.
- GPU Jobs must finish before their CPU counterparts start. The CPU Jobs do not
  wait or poll for missing representatives.
- Apply only in context `nautilus`, namespace `lemn-lab`, after a fresh client
  and server dry-run. The generated full manifests passed both dry-runs on
  2026-08-14, but that is schema/policy validation, not launch approval.

Regenerate with:

```bash
python3 scripts/generate_condition_matched_init_manifests.py \
  --inventory data/eval_results/null_subj_v2_condition_matched_v1/checkpoint_inventory.json \
  --output-dir k8s/condition_matched_eval/init_split --split \
  --exclude-hp1-arch bert_large \
  --exclude-hp1-arch gpt2_large \
  --exclude-hp1-arch mamba_370m \
  --completed-arch-seed gpt2_small:42 \
  --completed-arch-seed gpt2_small:137
```
