# S3 integration guide

This document is the single source of truth for how training and eval
pods talk to the Nautilus Ceph S3 bucket. It covers what's stored, how
it's written, and how to consume it. Keep it short — detail lives in
code.

## Bucket + credentials

| Setting | Value |
|---|---|
| Bucket | `s3://thomas-subject-drop-artifacts/` |
| Pool | West (default) |
| Endpoint (inside-cluster, high bandwidth) | `http://rook-ceph-rgw-nautiluss3.rook` |
| Endpoint (outside-cluster, via load balancer) | `https://s3-west.nrp-nautilus.io` |
| K8s secret | `s3-secret-thomas` (keys `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) |
| Local profile | `~/.aws/credentials` `[nrp]` |

The NRP docs explicitly recommend the **inside** endpoint for pod-side
clients because it can hit multiple OSDs in parallel. Use the outside
endpoint only when writing from a laptop or outside the cluster.

## Layout

```
s3://thomas-subject-drop-artifacts/
├── run_registry/
│   ├── by_run/<arch>/<lang>/<condition>/<run_id>.json   ← authoritative
│   └── registry.parquet                                 ← materialized view
├── eval_results/
│   ├── perplexity/<run_id>.parquet
│   ├── blimp/<run_id>.parquet
│   └── null_subj/<run_id>_<category>.parquet
├── training_curves/<run_id>.parquet                     ← WandB export (future)
└── env_snapshots/<run_id>.txt                           ← pip freeze + nvidia-smi
```

**The `by_run/` shard is the source of truth.** Every writer
(`register_run_start`, `heartbeat`, `register_run_end`,
`register_eval_end`, `mark_reference_rep`) reads/writes that single key.
Concurrent writers never collide because distinct runs have distinct
keys. The `registry.parquet` file is a periodic read-optimization rebuilt
by `scripts/compact_registry.py`.

**`eval_results/` lives on S3 from day 1** — don't backfill-migrate
parquets from CephFS. Clean portability for the data-availability
statement.

**`env_snapshots/` is a sidecar**, written once at run-start from
`register_run_start(write_env=True)`. `pip freeze` + `nvidia-smi -L` +
`git_commit`. 1 KB each, useful for audits.

## What pods write with

| Payload | Size | Tool |
|---|---|---|
| Registry JSON (~2 KB) | tiny | `boto3.put_object` via `model_foundry.registry` |
| Env snapshot (~1 KB) | tiny | `boto3.put_object` via `registry.write_env_snapshot` |
| Eval parquet (1–10 MB) | small | `pyarrow.fs.S3FileSystem` + `pq.write_table` |
| Archive blobs (future, >80 MB) | large | `boto3.upload_fileobj` (multipart-correct) or `rclone` |
| Bulk inter-bucket ops | any | `rclone` (not `aws s3 cp` — broken on >80 MB per NRP docs) |

**Do not use `aws s3 cp`** for anything non-trivial. The NRP docs call
out a known multipart bug on files >80 MB.

## Write API (Python, inside training/eval pods)

```python
from model_foundry import registry

registry.register_run_start(
    run_id=registry.build_run_id("gpt2_medium", "en", "baseline", seed=0),
    arch="gpt2_medium", lang="en", condition="baseline", seed=0,
    config_hash=cfg_hash, git_commit=git_sha,
    cache_key=data_cache_key, tokenizer_dir="tokenizers/en_sp_50004",
    wandb_run_id=wandb.run.id, wandb_project="just-drop-the-subject",
    gpu_product=os.environ.get("GPU_PRODUCT"),
    train_steps=5000,
)

# Every ~5 minutes from the training loop
registry.heartbeat("gpt2_medium", "en", "baseline",
                   "gpt2_medium-en-baseline-s0",
                   current_step=global_step, current_loss=last_loss)

# On completion
registry.register_run_end(
    run_id="gpt2_medium-en-baseline-s0",
    arch="gpt2_medium", lang="en", condition="baseline",
    status="COMPLETE", final_loss=3.21,
    total_tokens_processed=880_000_000,
    checkpoint_paths=[...],  # list of ceph://... or /mnt/data/... paths
)
```

Eval runner calls:

```python
registry.register_eval_end(
    run_id="gpt2_medium-en-baseline-s0",
    arch="gpt2_medium", lang="en", condition="baseline",
    eval_status="COMPLETE",
    eval_parquet_paths=[
        "s3://thomas-subject-drop-artifacts/eval_results/blimp/gpt2_medium-en-baseline-s0.parquet",
        ...
    ],
)
```

Reference-rep selector (1.6):

```python
registry.mark_reference_rep(
    run_id="gpt2_medium-en-baseline-s0",
    arch="gpt2_medium", lang="en", condition="baseline",
    rationale="median BLiMP score across 10 seeds at final checkpoint",
)
```

## Read API

Common case — load the materialized Parquet and analyze:

```python
import pandas as pd
df = pd.read_parquet("s3://thomas-subject-drop-artifacts/run_registry/registry.parquet")
```

Rare case — need the absolute latest (compactor runs hourly; if you
need it live):

```python
from model_foundry.registry import iter_all_records
import pandas as pd
df = pd.DataFrame(list(iter_all_records()))
```

## Compaction

`scripts/compact_registry.py` reads `run_registry/by_run/**/*.json` and
writes `run_registry/registry.parquet`. Runs on-demand or as a K8s
CronJob at ~hourly cadence. See the script for CLI flags.

## K8s env vars (add to every pod that talks to S3)

```yaml
env:
- name: AWS_ACCESS_KEY_ID
  valueFrom: {secretKeyRef: {name: s3-secret-thomas, key: AWS_ACCESS_KEY_ID}}
- name: AWS_SECRET_ACCESS_KEY
  valueFrom: {secretKeyRef: {name: s3-secret-thomas, key: AWS_SECRET_ACCESS_KEY}}
- name: AWS_ENDPOINT_URL
  value: "http://rook-ceph-rgw-nautiluss3.rook"
- name: AWS_DEFAULT_REGION
  value: "us-west-1"
- name: REGISTRY_BUCKET
  value: "thomas-subject-drop-artifacts"
```

## Failure modes

- **Transient S3 unavailability**: boto3 retries 5× with exponential
  backoff (configured in `registry._client`). Beyond that the call
  raises; callers treat it as non-fatal (log + keep the checkpoint) for
  `heartbeat` and `register_run_end`, and as fatal for
  `register_run_start` (can't start without being registered).
- **Compactor sees malformed JSON**: it logs and skips. The bad record
  stays in `by_run/` for manual inspection.
- **Pod crashes before `register_run_end`**: record stays as
  `status=RUNNING` with stale `last_heartbeat_at`. A separate reaper
  (not yet written; roadmap item) marks runs stale after ≥2 h of no
  heartbeat.
- **Re-runs of the same (arch × lang × condition × seed) cell**: same
  `run_id`, overwrites the record, `attempt_count` increments. Postmortems
  can tell re-runs apart from `attempt_count > 1`.

## Integration points (files touched when this lands)

1. `requirements.txt` → add `boto3>=1.34`.
2. `model_foundry/cli.py` entrypoint `run` → call
   `register_run_start` / `register_run_end`.
3. `model_foundry/training/loop.py` → call `heartbeat` every ~5 min
   (simple: `if global_step % heartbeat_steps == 0`).
4. Eval runner (existing, separate agent) → call `register_eval_end`.
5. `scripts/select_reference_reps.py` (future, 1.6) → call
   `mark_reference_rep`.
6. `k8s/job-*.yaml` templates → add the env-var block above.
7. `k8s/job-compact-registry.yaml` (CronJob) → run
   `scripts/compact_registry.py` hourly.

Steps 2, 3, 6, 7 are the remaining work once this lands.
