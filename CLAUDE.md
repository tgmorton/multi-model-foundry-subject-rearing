# Project CLAUDE.md

This file is loaded at the start of every Claude session in this repo.
It's the minimum priors a new agent needs to not re-derive things we
already decided. Detail lives in `docs/` and in project memory.

## What the project is

Controlled-rearing study of subject-drop in language models. Training
a matrix of (architecture × language × condition × seed) models on
90M-word corpora (English BabyLM + Spanish BebeLM), with 7 ablations
per (arch × lang) plus a baseline. The goal is to compare generative
linguistics vs information-theoretic accounts of subject-drop.

## Where we are

- **Optimization stack** (phase 0.2–0.6, 1.1–1.3) landed and verified:
  content-addressed data cache, fused AdamW, tensor-loss (one sync per
  optimizer step, not per micro-batch), save-policy split (last N
  checkpoints keep `training_state.pt`, rest are analysis-only),
  timing instrumentation, Inductor backend.
- **Custom GitLab image** (`gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:latest`)
  ships torch + requirements + flash-attn + mamba-ssm + causal-conv1d
  pre-installed. Pod startup went from ~300 s (pip install) to 5 s.
- **Registry substrate** in place: bucket, K8s secret, `registry.py`
  module, per-run JSON + materialized `registry.parquet` plan, write
  API wired into `cli.py::run` + training-loop heartbeat.
- **Smokes pass** for all 5 architectures (GPT-2 medium/large, BERT
  large, LSTM, Mamba 370M). First full production baseline finished
  (GPT-2 medium, English, 10 epochs, loss 3.15).

## Study parameters (locked)

| Parameter | Value |
|---|---|
| Architectures | n-gram 1–5, GPT-2 small/medium/large, BERT large, LSTM, Mamba 370M |
| Languages | English + Spanish (Italian is dormant) |
| Conditions | baseline + 7 ablations per (arch, lang) |
| Seeds per cell | **30** |
| Epochs (production) | **30** |
| Checkpoints per run | **80**, must include anchors `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512`, then appropriate spacing |
| HP sweep horizon | **3 epochs** per trial |
| HP sweep metric | Both training loss (A) and held-out perplexity (B) — decide between after first round |
| HP sweep per language | **Separate sweeps** per language (no English→Spanish transfer) |

## Storage architecture

See `memory/reference_storage_diagram.md` for the canonical ASCII
picture. Summary:

- **CephFS PVC `corpus-analysis-data`** (200 GiB, RWX) — hot operational
  storage. Holds raw corpora, tokenizers, tokenized+chunked data,
  in-flight checkpoints.
- **CephFS PVC `subject-drop-archive`** (40 TiB, RWX) — cold
  reference-rep storage. Holds analysis-only checkpoints for the
  reference replicate of each cell after post-eval pruning.
- **Ceph RGW S3 bucket `thomas-subject-drop-artifacts`** — portable
  science outputs that travel with the paper. Registry, eval parquets,
  WandB exports, env snapshots.
- **Nautilus retention is 6 months of inactivity, not 6 months
  absolute.** Storage ceases to be a binding constraint during the
  active study. Post-publication migration is an institutional
  archiving question (Zenodo / OSF / UCSD service).

## S3 details (everything needed to talk to the bucket)

| Field | Value |
|---|---|
| Bucket | `s3://thomas-subject-drop-artifacts/` |
| Pool | West (default) |
| Inside-cluster endpoint (pods) | `http://rook-ceph-rgw-nautiluss3.rook` — high bandwidth, multi-OSD |
| Outside-cluster endpoint (laptops) | `https://s3-west.nrp-nautilus.io` — via load balancer |
| K8s secret | `s3-secret-thomas` (keys `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) |
| Local profile | `~/.aws/credentials` section `[nrp]` |

**Every training / eval pod needs this env block** (already added to
the 7 active training templates under `k8s/`):

```yaml
env:
- name: AWS_ACCESS_KEY_ID
  valueFrom: {secretKeyRef: {name: s3-secret-thomas, key: AWS_ACCESS_KEY_ID}}
- name: AWS_SECRET_ACCESS_KEY
  valueFrom: {secretKeyRef: {name: s3-secret-thomas, key: AWS_SECRET_ACCESS_KEY}}
- {name: AWS_ENDPOINT_URL,   value: "http://rook-ceph-rgw-nautiluss3.rook"}
- {name: AWS_DEFAULT_REGION, value: "us-west-1"}
- {name: REGISTRY_BUCKET,    value: "thomas-subject-drop-artifacts"}
```

**Tool of choice**: `boto3` for programmatic writes (registry JSONs,
small eval parquets), `pyarrow.fs.S3FileSystem` for native parquet I/O,
`rclone` for bulk / multi-GB transfers. **Do NOT use `aws s3 cp`** — the
NRP docs flag a multipart upload bug on files >80 MB; boto3 handles
multipart correctly.

Bucket layout (source of truth: `docs/RUN_REGISTRY.md` + `docs/S3_INTEGRATION.md`):

```
s3://thomas-subject-drop-artifacts/
├── run_registry/
│   ├── by_run/<arch>/<lang>/<condition>/<run_id>.json   ← authoritative
│   └── registry.parquet                                 ← hourly-compacted view
├── eval_results/{blimp,perplexity,null_subj,...}/<run_id>.parquet
├── training_curves/<run_id>.parquet
└── env_snapshots/<run_id>.txt
```

## Code conventions that matter

- **Tokenizer = one-shot operator setup.** Not a per-training-pod step.
  Sentencepiece unigram has stochastic tie-breaking; re-training would
  shift the vocabulary and invalidate within-(arch, lang) ablation
  comparisons. Train once per (arch, lang) via
  `k8s/job-train-tokenizer.yaml`.
- **Content-addressed cache keys** (`model_foundry/cache_keys.py`) are
  hashed from (corpus_path, tokenizer bytes, max_sequence_length,
  dataset_manipulation). Paths: `data/tokenized/<hash>/` and
  `data/chunked/<hash>/`. Different experiments with same ingredients
  share cache.
- **Registry is organizational, not auto-scheduling.** User invokes
  launchers (`scripts/launch_training.py`, `scripts/launch_evals.py`)
  for group operations. Only two CronJobs: compactor + reaper.
  Everything else is user-driven or automatic-within-a-launched-pod.
- **Checkpoint save policy**: last N (default 3) scheduled checkpoints
  save full `training_state.pt`, earlier ones save weights-only. Set
  via `training.save_resume_state_last_n` in the config.
- **AMP** (`use_amp: true`) is on for every transformer/BERT/Mamba
  config — needed so FA2 sees fp16 activations via autocast.
- **Registry writes are non-fatal**. S3 outage never kills a 20-hour
  training job — the `_safe_*` wrappers log and move on. Stale records
  get reconciled on next run.

## Known bugs / WIP

- **Tokenizer CLI-path idempotency bug**: `python -m model_foundry.cli
  train-tokenizer <config>` re-trains even when the cache is present.
  Calling the factory directly from a REPL hits the cache correctly.
  Currently has `[debug]` prints committed in
  `model_foundry/tokenizer/tokenizer_factory.py`; next run reveals the
  faulty predicate. Workaround: only invoke via
  `k8s/job-train-tokenizer.yaml` once per (arch, lang).
- **Not yet built**: launcher scripts, eval runner (other agent),
  pruners (1.4 / 1.5), reference-rep selector (1.6), reaper CronJob,
  compactor CronJob, HP sweep infra (decisions locked, code not
  written).

## Key docs to read (in order of relevance for most tasks)

1. `docs/RUN_REGISTRY.md` — full v1 registry schema + state machine +
   writer contract + launcher shape + read patterns.
2. `docs/S3_INTEGRATION.md` — endpoint, secret, layout, API, failure
   modes, K8s env-var block.
3. `memory/reference_storage_diagram.md` — canonical ASCII data-flow
   diagram.
4. `memory/reference_storage_plan.md` — detailed sizing, retention,
   pruning policy.
5. `memory/reference_hp_sweep_plan.md` — HP sweep methodology and
   per-arch optimization priorities.
6. `memory/reference_implementation_roadmap.md` — phased plan P0 → P3,
   agent handoff map.

## Common operator commands

```bash
# Train a tokenizer (one-shot per arch/lang)
kubectl apply -f k8s/job-train-tokenizer.yaml
# Edit TOKENIZER_CONFIG env if needed for a different config file

# Launch a training run (smoke pattern; production launcher TBD)
kubectl apply -f k8s/job-smoke-gpt2-medium.yaml

# Read the registry locally
aws --profile nrp s3 cp \
  s3://thomas-subject-drop-artifacts/run_registry/registry.parquet - | \
  python -c "import sys, pyarrow.parquet as pq; \
             t = pq.read_table(sys.stdin.buffer); \
             print(t.to_pandas().head())"

# Force-compact the registry (run on-demand before an analysis)
python scripts/compact_registry.py --bucket thomas-subject-drop-artifacts
```
