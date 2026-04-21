# eval_v2 — NRP jobs

Kubernetes jobs for the new evaluation pipeline (D1–D14). Uses the
existing `corpus-analysis-data` PVC. No custom image build — jobs run on
stock `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` and install deps at
pod startup.

## Jobs

### `job-eval-v2-smoke.yaml` — end-to-end smoke

Exercises the full pipeline on real trained checkpoints. Runs twice:
- `/mnt/data/models/exp0_baseline_90M_smoke/checkpoint-0` — untrained
- `/mnt/data/models/exp0_baseline_90M_smoke_resume/checkpoint-5` — after 5 training steps

For each, it:
1. Pre-tokenizes the v2 EN + ES `subject_drop.csv` stimuli (D6).
2. Builds the model from config.json (fresh weights via
   `AutoModelForCausalLM.from_config`).
3. Loads the checkpoint's `pytorch_model.bin` into the same model
   instance via `load_state_dict` (D9 — the load-once/swap pattern).
4. Runs batched forward passes (D7, no KV-reuse in smoke) → computes
   MeanLP (D2) and saves per-token log-probs (D5).
5. Writes partitioned parquet + DuckDB queries (D10).
6. Re-runs once more to confirm the cache layer (D11) makes the second
   run a no-op.
7. Cross-checks ckpt0 vs ckpt5 overt-preference rates via DuckDB.

Output goes to:
- `/mnt/data/eval_v2/smoke_ckpt0/`
- `/mnt/data/eval_v2/smoke_ckpt5/`

### Run it

```bash
kubectl apply -f k8s/v2/job-eval-v2-smoke.yaml

# Watch logs
kubectl logs -f job/eval-v2-smoke

# After completion
kubectl describe job/eval-v2-smoke
kubectl delete job eval-v2-smoke   # when you're done inspecting
```

Expected runtime: ~3–5 minutes (dominated by pip install, not eval).

### What success looks like

The logs should show, for each run:

1. `Found 1 checkpoint(s): checkpoint-N at /mnt/data/models/...`
2. Stimuli cache build: `n_rows=1152 languages=['en','es'] categories=['subject_drop']`
3. `RUN 1 summary: n_processed=1 n_forward_passes=1`
4. `RUN 2 summary: n_cached=1 n_processed=0 n_forward_passes=0`
5. `✓ D11 cache invariant holds: 0 forward passes on re-run.`
6. `✓ No NaN/Inf in item metrics.`
7. `Per-checkpoint overt-preference rate (MeanLP):` — a line per (step, lang)
8. `✅ SMOKE PASS`

At the very end, a cross-check table showing `overt_pref` for ckpt0 vs
ckpt5 per language. Untrained (ckpt0) should be ≈ 0.5 (random
preference); trained-5-steps (ckpt5) may shift slightly but after only 5
steps is still essentially untrained.

### Debugging

If it fails, in order:

- `pip install` errors → check the image has Python 3.10+
- `No checkpoint-* directories` → verify PVC paths on a scratch pod
- `OOM` → reduce `--batch_size 8` in the job args
- Parquet empty → check stimuli CSV paths in the job args match the repo tree
- NaN/Inf in metrics → the surprisal calculator hit an edge case; rerun
  with `--verbose` and inspect logs

## Why init container + stock image?

Same reason the rest of `k8s/` does it: zero per-change Docker rebuild.
Each commit to main is immediately runnable because the init container
clones the latest repo.

## Follow-ups (out of scope for smoke)

- `job-eval-v2-stage1-tokenize.yaml`: single-shot pre-tokenization per
  tokenizer family (write `stimuli_tokenized/...parquet` once, reuse).
- `job-eval-v2-stage2-unigram.yaml`: train unigram baselines per
  corpus-variant (one indexed job per `(intervention, tokenizer_family)`).
- `job-eval-v2-cell.yaml`: fleet fan-out — one indexed job per
  `(architecture, intervention, rep)` once we have a full manifest.
- `job-eval-v2-ngram.yaml`: CPU fast-path for the n-gram families.

Defer these until the smoke passes and we've validated D9 on real
checkpoints.
