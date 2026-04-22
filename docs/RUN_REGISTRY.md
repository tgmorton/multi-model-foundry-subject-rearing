# Run Registry — full design

The registry is the portable, authoritative index of every training run in
the study. This document specifies its complete schema, state machine,
write contracts, read patterns, and integration points. It's the
contract that every writer (training, eval, pruners, reference-rep
selector, HP sweeps) implements.

For the lower-level "how do pods talk to S3" plumbing, see
[`S3_INTEGRATION.md`](S3_INTEGRATION.md). This doc focuses on **what
the records contain and who changes what when**.

## 1. Why the registry exists

Four questions become unanswerable at 5,720 runs without one:

1. **Which cells still need training?** (Gate the meta-orchestrator.)
2. **Which runs have eval parquets written?** (Gate the eval runner and the post-eval pruner.)
3. **Which seed is the reference rep for each cell?** (Gate analyses that should use one rep per cell, not 10.)
4. **What exactly produced this result?** (`git_commit` + `config_hash` + `cache_key` + `docker_image` travel with the run forever, for the data-availability statement.)

WandB covers parts of (1)–(3) live but is a dashboard, not an archive —
it won't be reliable API-accessible in 5 years when a reader reproduces
the paper. The registry is.

## 2. Storage layout

```
s3://thomas-subject-drop-artifacts/
└── run_registry/
    ├── by_run/                                   ← source of truth
    │   └── <run_kind>/<arch>/<lang>/<condition>/
    │       └── <run_id>.json
    └── registry.parquet                          ← materialized view,
                                                    compactor rebuilds
```

- **`by_run/` is the source of truth.** One file per run. Writers do
  get-modify-put to a unique key, so concurrent writers for different
  runs never collide. Writers for the same run (training, eval, pruner,
  ref-rep selector) touch disjoint fields — which they do by design
  (see §6).
- **`registry.parquet` is a read optimization** rebuilt by
  `scripts/compact_registry.py` on an hourly K8s CronJob. Analyses load
  this single file.

The `<run_kind>` prefix (`hp_sweep`, `production`, `analysis_only`)
makes it cheap to list just the production runs during a compaction
pass if we ever need to skip sweep noise in the main parquet.

## 3. The run identifier

Deterministic: `{arch}-{lang}-{condition}-s{seed}`, built via
`registry.build_run_id(arch, lang, condition, seed)`.

For HP-sweep trials (which don't have a canonical seed), we use:
`sweep-{sweep_id}-trial-{trial_id}`.

Re-runs of the same cell (e.g. after preemption) overwrite the same
record but increment `attempt_count`, so postmortems can tell re-runs
apart.

## 4. Schema v1

Fields grouped by purpose. Every field is optional except where noted;
missing fields deserialize as `null` from the compacted Parquet.

### 4.1 Identity & classification (required at run start)

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | int | ✓ | Currently `1`. Bump on breaking change. |
| `run_id` | str | ✓ | Deterministic (§3). |
| `run_kind` | str | ✓ | `hp_sweep` \| `production` \| `analysis_only`. Drives compaction partitioning and consumer filters. |
| `arch` | str | ✓ | `gpt2_small`, `gpt2_medium`, `gpt2_large`, `bert_large`, `lstm`, `mamba_370m`, `ngram_1`..`ngram_5` |
| `lang` | str | ✓ | `en` \| `es` |
| `condition` | str | ✓ | `baseline` \| `impoverish_case` \| `lemmatize_verbs` \| ... |
| `seed` | int | ✓ | 0–9 for production. For sweep trials, the sweep trial index. |
| `attempt_count` | int | ✓ | Bumped on each `register_run_start`. |

### 4.2 Reproducibility (required)

The "can a reader reproduce this result in 2031?" set.

| Field | Type | Notes |
|---|---|---|
| `config_hash` | str | MD5 of the resolved, post-validation config (`config.model_dump()` canonicalised). |
| `git_commit` | str | Short SHA of the code commit. |
| `docker_image` | str | Full image reference including digest, once the GitLab CI image lands. E.g. `registry.nautilus.optiputer.net/thmorton/mmf:abc123@sha256:...`. |
| `cache_key` | str | Content-addressed tokenized+chunked cache key (0.2). |
| `tokenizer_dir` | str | Path to the tokenizer directory used. Hash of `tokenizer.model` also inside `cache_key`. |
| `dataset_manipulation_hash` | str | Canonical JSON hash of the `dataset_manipulation` pipeline. `[]` for baseline. |
| `hyperparameters` | dict | Flat map of the HP vector — `learning_rate`, `adam_beta1`, `adam_beta2`, `warmup_ratio`, `dropout`, `effective_batch_size`, etc. Same values the sweep optimizer ranged over. |

### 4.3 Provenance

| Field | Type | Notes |
|---|---|---|
| `wandb_run_id` | str | For joining against WandB API while it's live. |
| `wandb_project` | str | E.g. `just-drop-the-subject`. |
| `wandb_sweep_id` | str? | Only populated for `run_kind=hp_sweep`. |
| `node_name` | str | K8s node, for postmortems. |
| `gpu_product` | str | `NVIDIA-GeForce-RTX-3090`, etc. |
| `k8s_pod_name` | str | For postmortems. |

### 4.4 Training lifecycle

State machine: `QUEUED → RUNNING → COMPLETE | FAILED | PREEMPTED`.
`QUEUED` is optional (only the orchestrator writes it — see §7).

| Field | Type | Notes |
|---|---|---|
| `status` | str | Current training state. |
| `started_at` | iso8601 | First `register_run_start`. |
| `finished_at` | iso8601? | When `register_run_end` was called. |
| `duration_seconds` | int? | `finished_at - started_at`, precomputed for convenience. |
| `last_heartbeat_at` | iso8601 | Bumped on every `heartbeat()`. Reaper uses this. |
| `current_step` | int? | Set by `heartbeat()`. |
| `current_loss` | float? | Set by `heartbeat()`. |
| `failure_reason` | str? | Free-text when `status=FAILED`. First 500 chars of the exception. |
| `oom_count` | int? | Number of OOM events during this attempt (observable from training loop). |

### 4.5 Training outputs (populated at `register_run_end`)

| Field | Type | Notes |
|---|---|---|
| `final_loss` | float | Loss at the last optimizer step. |
| `train_steps` | int | Scheduled total (same as `config.training.train_steps`). |
| `steps_completed` | int | Actual — may be less than `train_steps` if `FAILED/PREEMPTED`. |
| `epochs_completed` | int | How many epochs finished. |
| `total_tokens_processed` | int | Same counter the checkpoint metadata tracks. |
| `checkpoint_count` | int | Number of checkpoint dirs written. Should be 80 for production. |
| `checkpoint_paths` | list[str] | Sorted by step. `/mnt/data/models/...` or `ceph://...` — not S3. |
| `tokens_per_sec_avg` | float? | From 1.1 timing, for HP-sweep fitness and postmortem. |
| `data_fraction_avg` | float? | Fraction of step time spent in DataLoader. |

### 4.6 Eval lifecycle (populated by eval writer)

Eval is per-benchmark. Different benchmarks may finish at different times,
so we track each independently plus an overall aggregate.

| Field | Type | Notes |
|---|---|---|
| `eval_status` | str | Aggregate. `NONE` \| `PARTIAL` \| `COMPLETE` \| `FAILED`. |
| `eval_started_at` | iso8601? | First benchmark started. |
| `eval_finished_at` | iso8601? | Last benchmark finished (or gave up). |
| `eval_benchmarks` | dict[str, dict] | Map benchmark → `{status, parquet_path, finished_at, metric_summary}`. See §4.6.1. |
| `eval_duration_seconds` | int? | `eval_finished_at - eval_started_at`. |

#### 4.6.1 `eval_benchmarks` shape

Keyed by benchmark name. Values:

```json
{
  "blimp": {
    "status": "COMPLETE",
    "parquet_path": "s3://thomas-subject-drop-artifacts/eval_results/blimp/gpt2_medium-en-baseline-s0.parquet",
    "finished_at": "2026-04-22T10:05:00Z",
    "metric_summary": {"accuracy_final_ckpt": 0.6321, "accuracy_best_ckpt": 0.6344}
  },
  "perplexity": { "status": "COMPLETE", "parquet_path": "...", "finished_at": "...", "metric_summary": {"ppl_final_ckpt": 44.7} },
  "null_subj": { "status": "PARTIAL", "parquet_path": "...", "finished_at": "...", "metric_summary": null },
  "stimuli_pronoun_sweep": { "status": "NONE", ... }
}
```

`metric_summary` is an optional thin summary for ranking (drives
reference-rep selection without reloading every parquet). Payload-heavy
detail stays in the parquet.

Benchmark names we plan to run: `blimp`, `perplexity`,
`null_subject_expletive`, `null_subject_morph`, `null_subject_case`,
`pronoun_sweep` (+ condition-specific variants TBD).

### 4.7 Pruning & archive lifecycle

Three pruner operations happen over a run's life, each records itself.

| Field | Type | Notes |
|---|---|---|
| `resume_state_steps` | list[int] | Steps whose checkpoints carry `training_state.pt`. Typically the latest 3 during training. |
| `post_run_pruned_at` | iso8601? | When the in-training pruner (1.4) stripped older `training_state.pt` files. |
| `post_eval_pruned_at` | iso8601? | When the post-eval pruner (1.5) stripped ALL `training_state.pt` files and deleted non-ref-rep replicates. |
| `archived_at` | iso8601? | When the ref rep moved from hot PVC to archive PVC. |
| `archive_paths` | list[str]? | Replaces `checkpoint_paths` after archival. `/mnt/data/archive/...`. |
| `archive_size_bytes` | int? | Final disk size after all pruning. Useful for the storage-planning dashboards. |

### 4.8 Reference-rep selection

Written once per (arch × lang × condition) cell, on exactly one seed.

| Field | Type | Notes |
|---|---|---|
| `is_reference_rep` | bool? | `true` on exactly one seed per cell. `null` until selection. |
| `reference_rep_rationale` | str? | E.g. `"median BLiMP at final checkpoint among 10 seeds"`. |
| `reference_rep_metric` | str? | Machine-readable: e.g. `"blimp.metric_summary.accuracy_final_ckpt"`. |
| `reference_rep_score` | float? | The actual value the winner scored on the selection metric. |
| `reference_rep_selected_at` | iso8601? | |

### 4.9 HP-sweep specific

Only populated when `run_kind=hp_sweep`.

| Field | Type | Notes |
|---|---|---|
| `hp_sweep_id` | str | WandB sweep id. |
| `hp_sweep_trial_id` | str | WandB run id for this trial. |
| `hp_sweep_rank` | int? | 1 = best, set after sweep completes. |
| `hp_proxy_metric` | str? | What the sweep ranked on (e.g. `"perplexity_after_2000_steps"`). |
| `hp_proxy_score` | float? | Value of that metric. |
| `is_hp_winner` | bool? | `true` for the one trial we freeze as the HP vector for this arch. Its `hyperparameters` become the production config. |

### 4.10 Bookkeeping

| Field | Type | Notes |
|---|---|---|
| `created_at` | iso8601 | First time this record was written. Immutable. |
| `updated_at` | iso8601 | Last merge. Bumped by every `_merge_record`. |

## 5. State machine

```
                     ┌──────────────────────────────────────────────────┐
                     │  Orchestrator decides a cell needs training       │
                     └──────────────────────────────────────────────────┘
                                         │
                                         ▼
                                    ┌─────────┐
                                    │ QUEUED  │  (optional — only the orchestrator writes this)
                                    └────┬────┘
                                         │  training pod starts
                                         ▼
                                    ┌─────────┐
                        ┌──────────►│ RUNNING │◄──── heartbeat every ~5 min
                        │  retry    └────┬────┘
                        │                │
                        │  ┌─────────────┼─────────────┬────────────────┐
                        │  │             │             │                │
                        │  │ finishes    │ crashes     │ node preempted │ times out
                        │  ▼             ▼             ▼                ▼
                        │ COMPLETE     FAILED       PREEMPTED         FAILED
                        │  │             │             │                │
                        └──┘             │             │                │
                        (attempt++)      │             │                │
                                         ▼             ▼                ▼
                                    ┌─────────────────────────────────────┐
                                    │  Eval runner picks up COMPLETE runs │
                                    └──────────────────────┬──────────────┘
                                                           │
                                                           ▼
                                                    eval_status=RUNNING
                                                           │
                                                           ▼
                                                    eval_status=COMPLETE
                                                           │
                                                           ▼
                                   ┌───────────────────────────────────────┐
                                   │  Ref-rep selector picks median seed   │
                                   └──────────────────────┬────────────────┘
                                                          │
                                                          ▼
                                                  is_reference_rep=true
                                                          │
                                                          ▼
                                      Post-eval pruner strips non-ref reps,
                                      deletes training_state.pt, optionally
                                      migrates ref to archive PVC
                                                          │
                                                          ▼
                                                  archived_at=...
```

## 6. Who writes what, when — the writer contract

Every writer owns **exactly one column set** from §4. Writers never
modify fields outside their set — that's how we avoid coordination.

| Writer | Trigger | Fields written | Call site |
|---|---|---|---|
| **orchestrator** (optional) | decides to launch a cell | 4.1 identity, `status=QUEUED`, `created_at` | `scripts/orchestrator.py` (TBD) |
| **training entrypoint** | CLI `run` start | 4.1 identity, 4.2 reproducibility, 4.3 provenance, `status=RUNNING`, `started_at`, `last_heartbeat_at`, `attempt_count++`, `hyperparameters`, `train_steps` | `model_foundry/cli.py::run` → `register_run_start` |
| **training loop** | every ~5 min | `last_heartbeat_at`, `current_step`, `current_loss` | `model_foundry/training/loop.py` → `heartbeat` |
| **training entrypoint (end)** | CLI `run` completion | `status` (COMPLETE/FAILED/PREEMPTED), `finished_at`, `duration_seconds`, `final_loss`, `steps_completed`, `epochs_completed`, `total_tokens_processed`, `checkpoint_count`, `checkpoint_paths`, `resume_state_steps`, `tokens_per_sec_avg`, `data_fraction_avg`, `failure_reason`, `oom_count` | `model_foundry/cli.py::run` → `register_run_end` |
| **eval runner** | eval start | `eval_status=RUNNING`, `eval_started_at`, `eval_benchmarks[X]={status: RUNNING}` | eval runner (other agent) → `register_eval_start(benchmark=X)` |
| **eval runner** | per-benchmark completion | `eval_benchmarks[X]={status, parquet_path, finished_at, metric_summary}`, aggregate `eval_status` recomputed | → `register_eval_benchmark_done` |
| **eval runner** | all benchmarks done | `eval_status` aggregate, `eval_finished_at`, `eval_duration_seconds` | → `register_eval_end` |
| **in-training pruner (1.4)** | after each run completes | `post_run_pruned_at`, updates `checkpoint_paths` (analysis-only subset) | `scripts/prune_in_training.py` (TBD) → `register_pruner_event('post_run')` |
| **post-eval pruner (1.5)** | after eval_status=COMPLETE for all reps of a cell | `post_eval_pruned_at`, removes non-ref-rep `checkpoint_paths`, compacts `resume_state_steps` to empty | `scripts/prune_post_eval.py` (TBD) → `register_pruner_event('post_eval')` |
| **ref-rep selector (1.6)** | after all N seeds in a cell have `eval_status=COMPLETE` | On exactly one seed: `is_reference_rep=true`, `reference_rep_rationale`, `reference_rep_metric`, `reference_rep_score`, `reference_rep_selected_at` | `scripts/select_reference_reps.py` (TBD) → `mark_reference_rep` |
| **archiver** | after post-eval pruner | `archived_at`, `archive_paths`, `archive_size_bytes` | `scripts/archive_runs.py` (TBD) → `register_archive_event` |
| **reaper** | CronJob, looks for stale RUNNING | `status=PREEMPTED`, `failure_reason="stale heartbeat > 2h"` | `scripts/reap_stale_runs.py` (TBD) |
| **HP sweep agent** | per trial | 4.1 identity (as `hp_sweep`), 4.2 reproducibility, 4.3 provenance, 4.4 lifecycle, 4.5 outputs, 4.9 sweep fields | `scripts/sweep_agent.py` (existing, needs registry wiring) |
| **HP sweep coordinator** | after sweep completes | `hp_sweep_rank`, `is_hp_winner` on one trial | `scripts/select_hp_winner.py` (TBD) |
| **compactor** | hourly CronJob | (reads only; writes `registry.parquet`) | `scripts/compact_registry.py` |

Rule: **field ownership is single-writer.** If two writers want the
same field, we make one of them canonical and the other read-only.

## 7. Read patterns

These are the queries analyses and orchestration scripts will actually
run. The compacted Parquet handles all of them natively.

### Which production cells still need training?

```python
df = pd.read_parquet("s3://thomas-subject-drop-artifacts/run_registry/registry.parquet")
target_cells = {
    (arch, lang, cond, seed)
    for arch in ARCHS for lang in LANGS for cond in CONDS for seed in range(10)
}
done = set(df[(df.run_kind == "production") & (df.status == "COMPLETE")]
           [["arch", "lang", "condition", "seed"]].itertuples(index=False, name=None))
still_needed = target_cells - done
```

### Which completed runs have no eval yet?

```python
q = df[(df.run_kind == "production") &
       (df.status == "COMPLETE") &
       (df.eval_status.isna() | (df.eval_status != "COMPLETE"))]
```

### Show me the reference rep for each cell

```python
ref = df[df.is_reference_rep == True]   # exactly 1 per (arch, lang, condition)
```

### Paper figure: "final loss by arch × condition × lang"

```python
ref_only = df[df.is_reference_rep == True]
fig_data = ref_only.groupby(["arch", "lang", "condition"]).final_loss.mean().reset_index()
```

### Paper figure: "trajectory across 80 checkpoints for reference reps"

Load the eval parquets (which are indexed per-checkpoint):

```python
ref = df[df.is_reference_rep == True]
# For each ref rep, its blimp eval parquet:
paths = ref.eval_benchmarks.apply(lambda d: d.get("blimp", {}).get("parquet_path"))
blimp = pd.concat([pd.read_parquet(p) for p in paths.dropna()])
# blimp has (run_id, checkpoint, stimulus, response) rows
```

### HP sweep audit: what HP did we freeze for gpt2_medium and why?

```python
winner = df[(df.run_kind == "hp_sweep") & (df.is_hp_winner == True) & (df.arch == "gpt2_medium")]
print(winner[["hyperparameters", "hp_proxy_metric", "hp_proxy_score"]].iloc[0])
```

### Postmortem: any production runs with >5 OOMs?

```python
df[df.oom_count > 5][["run_id", "oom_count", "gpu_product", "node_name"]]
```

### Data-availability table for the paper

```python
report = df[df.run_kind == "production"][[
    "run_id", "arch", "lang", "condition", "seed",
    "config_hash", "git_commit", "cache_key", "docker_image",
    "final_loss", "checkpoint_count", "archive_paths",
    "eval_benchmarks",
]]
report.to_parquet("paper_data_availability.parquet")
```

## 8. Rollup views beyond `registry.parquet`

We may also produce secondary materialized views from the same
`by_run/` source. Each is its own scheduled job.

| View | Key | Rows | Purpose |
|---|---|---|---|
| `registry.parquet` | one row per run | ~17K at peak | Primary: covers 99% of queries. |
| `cells.parquet` | one row per (arch, lang, condition) | ~320 | Per-cell aggregates: how many seeds done, reference rep id, mean final loss. For the orchestrator dashboard. |
| `archs.parquet` | one row per (arch, lang) | ~20 | Total tokens processed, total GPU hours, reference-rep final loss, etc. Top-level dashboard. |

None of these are strictly needed now. `registry.parquet` is — the
others are follow-ups when the analysis code starts feeling slow.

## 9. Concurrency & failure semantics

### Concurrency (safe by design)

- **Different `run_id` keys never collide** — distinct S3 keys,
  distinct puts.
- **Same run, multiple writers** don't touch the same fields. Training
  writer owns §4.4–4.5; eval writer owns §4.6; pruner owns §4.7; ref-rep
  owns §4.8. As long as writers obey §6's field ownership, we never need
  a lock.
- **The get-modify-put cycle in `_merge_record`** is last-writer-wins
  on any field both writers touch. Only `updated_at` is multiply-touched
  legitimately, and we don't care about its exact ordering.

### Failure handling

- **boto3 retries 5× with exponential backoff** on transient S3 errors.
  Configured in `registry._client`.
- Beyond that, the call raises. Callers:
  - `register_run_start` — **fatal**. Can't start a run without being
    registered; abort.
  - `heartbeat` — non-fatal. Wrapped in `try/except`, logged at WARN.
  - `register_run_end` — non-fatal. Checkpoint already saved to disk;
    log loudly and move on. Next training-start for this run will catch
    up the registry.
  - Eval / pruner / ref-rep writes — non-fatal. The authoritative
    source is the artefact on disk / S3; the registry is just the
    index. A re-run of the writer fixes it.
- **Pod crashes mid-run** — record sits at `status=RUNNING` with
  stale `last_heartbeat_at`. Reaper CronJob marks it `PREEMPTED` after
  2h of no heartbeat, so the orchestrator can relaunch.
- **Malformed JSON** — compactor logs and skips. Bad record sits in
  `by_run/` for manual inspection.
- **Partial writes to Parquet** — `registry.parquet` is overwritten
  atomically on each compaction (single `put_object`), so readers never
  see a partial parquet.

## 10. Schema evolution

Rules for changing this schema without breaking existing records:

1. **Additive changes are free.** Add a new optional field, bump no
   version. Old records deserialize as `null` in that column.
2. **Renames / type changes require `schema_version` bump.** The
   compactor reads each record's own `schema_version`, dispatches to
   the right upgrader, then writes the Parquet at the latest version.
3. **Field removals** aren't actually removals — mark the field
   deprecated in this doc, leave it in records, stop writing to it.
   Removing real columns would break historical Parquet readers.
4. **Enum additions** are free; enum removals require a version bump.
5. **The upgrader lives in `model_foundry/registry.py`** as a
   `_upgrade_vN_to_vN+1` function. One per schema bump.

## 11. Integration points — file-by-file

### Already landed (commit 3bd0d80)

- `model_foundry/registry.py` — `register_run_start`, `heartbeat`,
  `register_run_end`, `register_eval_end`, `mark_reference_rep`,
  `write_env_snapshot`, `iter_all_records`.
- `scripts/compact_registry.py` — rebuilds `registry.parquet`.
- `docs/S3_INTEGRATION.md` — S3 endpoint / secret / env-var block.
- `boto3>=1.34` in `requirements.txt`.
- `s3-secret-thomas` K8s Secret exists.
- `thomas-subject-drop-artifacts` bucket exists.

### Needed to complete v1 (ordered by dependency)

1. **Extend `register_run_start` / `register_run_end`** with the fields
   in §4.2 (`docker_image`, `dataset_manipulation_hash`, `hyperparameters`),
   §4.4 (`oom_count`), §4.5 (`steps_completed`, `epochs_completed`,
   `tokens_per_sec_avg`, `data_fraction_avg`, `resume_state_steps`).
   Small, additive changes to the module. ~30 LOC.
2. **Add `register_eval_start` and `register_eval_benchmark_done`** for
   per-benchmark granularity (§4.6). ~40 LOC. The existing
   `register_eval_end` becomes the final aggregate call.
3. **Add `register_pruner_event`, `register_archive_event`**. Each is
   a thin `_merge_record` call. ~20 LOC each.
4. **Wire `cli.py::run`** to call `register_run_start` before training
   and `register_run_end` in a top-level `try/except/finally`. Read
   `git_commit` and `config_hash` from existing trainer machinery. ~50 LOC.
5. **Wire `loop.py`** to call `heartbeat` every N optimizer steps
   (default: compute N such that the cadence is ~5 minutes). ~10 LOC.
6. **Add env vars** from `S3_INTEGRATION.md` §K8s-env-vars to every
   training/eval K8s template.
7. **Compactor CronJob** — new `k8s/cronjob-compact-registry.yaml`
   that runs `scripts/compact_registry.py` hourly. ~30 LOC of YAML.
8. **Reaper CronJob** — `scripts/reap_stale_runs.py` +
   `k8s/cronjob-reap-stale-runs.yaml`. Looks for `status=RUNNING` with
   `last_heartbeat_at < now - 2h`, marks as PREEMPTED. ~50 LOC of code.

### Needed for the storage/pruner lifecycle (separate agent)

- `scripts/prune_in_training.py` (1.4)
- `scripts/prune_post_eval.py` (1.5)
- `scripts/archive_runs.py` (moves ref rep to `/mnt/data/archive/`)
- `scripts/select_reference_reps.py` (1.6)

These all consume the registry (find candidates) and write back
(pruner events, archive events, ref-rep selection). They don't need
registry schema changes — just call the existing merge functions.

### Needed for HP sweeps

- `scripts/sweep_agent.py` (audit of existing) — add
  `register_run_start(run_kind="hp_sweep")` + `register_run_end` and
  populate §4.9 fields.
- `scripts/select_hp_winner.py` — rank sweep trials by `hp_proxy_score`,
  mark the winner with `is_hp_winner=true` and `hp_sweep_rank=1`.

### Needed for orchestration (optional, nice-to-have)

- `scripts/orchestrator.py` — reads `registry.parquet`, diffs against
  the target matrix, launches `run_kind=production` jobs for cells
  that are missing or stuck. Could also write `status=QUEUED` records
  as it decides to launch.

## 12. One worked example: a single production cell, cradle to archive

`gpt2_medium-en-baseline-s0`. This is what happens to its record over
the course of the study.

**T+0s** — Orchestrator decides to launch (optional):
```json
{
  "schema_version": 1, "run_id": "gpt2_medium-en-baseline-s0",
  "run_kind": "production", "arch": "gpt2_medium", "lang": "en",
  "condition": "baseline", "seed": 0,
  "status": "QUEUED", "created_at": "...", "updated_at": "..."
}
```

**T+1m** — Training pod starts, `register_run_start`:
```json
{
  ...identity,
  "config_hash": "32e16f60", "git_commit": "509de1e",
  "docker_image": "registry.../mmf:abc@sha256:...",
  "cache_key": "e80294850998", "tokenizer_dir": "tokenizers/en_sp_50004",
  "dataset_manipulation_hash": "e3b0c44",
  "hyperparameters": {"learning_rate": 1e-4, "warmup_ratio": 0.03, "batch_size": 128, ...},
  "wandb_run_id": "zplgzkdf", "wandb_project": "just-drop-the-subject",
  "node_name": "k8s-bharadwaj-03", "gpu_product": "NVIDIA-GeForce-RTX-3090",
  "k8s_pod_name": "train-gpt2-med-en-baseline-s0-abc12",
  "status": "RUNNING", "started_at": "...", "last_heartbeat_at": "...",
  "attempt_count": 1, "train_steps": 5000
}
```

**T+every 5 min** — `heartbeat`:
```json
{ ..., "last_heartbeat_at": "...", "current_step": 1230, "current_loss": 4.12 }
```

**T+20h** — Training completes, `register_run_end`:
```json
{
  ...,
  "status": "COMPLETE", "finished_at": "...", "duration_seconds": 72000,
  "final_loss": 3.15, "train_steps": 5000, "steps_completed": 5000,
  "epochs_completed": 10, "total_tokens_processed": 880_000_000,
  "checkpoint_count": 80,
  "checkpoint_paths": ["/mnt/data/models/.../checkpoint-50", ..., "/mnt/data/models/.../checkpoint-5000"],
  "resume_state_steps": [4832, 4915, 5000],
  "tokens_per_sec_avg": 2380.0, "data_fraction_avg": 0.12,
  "oom_count": 0
}
```

**T+20h+10min** — In-training pruner strips old `training_state.pt`:
```json
{ ..., "post_run_pruned_at": "...", "checkpoint_paths": [same list; only training_state.pt dropped] }
```

**T+22h** — Eval runner picks it up:
```json
{ ..., "eval_status": "RUNNING", "eval_started_at": "...",
  "eval_benchmarks": {"blimp": {"status": "RUNNING", ...}}
}
```

**T+25h** — Each benchmark completes:
```json
{ ..., "eval_benchmarks": {
    "blimp": {"status": "COMPLETE", "parquet_path": "s3://.../blimp/gpt2_medium-en-baseline-s0.parquet",
              "finished_at": "...", "metric_summary": {"accuracy_final_ckpt": 0.6321}},
    "perplexity": {"status": "COMPLETE", ..., "metric_summary": {"ppl_final_ckpt": 44.7}},
    "null_subject_expletive": {"status": "COMPLETE", ..., "metric_summary": {...}},
    ...
  },
  "eval_status": "COMPLETE", "eval_finished_at": "...", "eval_duration_seconds": 10800
}
```

**T+2d** — Once all 10 reps have `eval_status=COMPLETE`, ref-rep
selector picks the median BLiMP performer. Say seed=3 wins; seed=0 does
not get `is_reference_rep=true`.

**T+2d+5m** — Post-eval pruner runs. Because seed=0 is NOT the ref rep,
its entire checkpoint directory is deleted:
```json
{ ..., "post_eval_pruned_at": "...", "checkpoint_paths": [], "checkpoint_count": 0 }
```

The record stays in the registry (we remember that seed 0 ran and what
it scored) but its artefacts are gone. The ref rep (seed=3) is kept in
the hot PVC; separately, an archiver may migrate it to `/mnt/data/archive/`
at some point and set `archived_at` + `archive_paths`.

**5 years later** — a reader loads `registry.parquet`, filters to
`is_reference_rep=true`, and reproduces every figure in the paper using
only the eval parquets and the ref-rep checkpoints.

## 13. What we explicitly do NOT put in the registry

- **Per-stimulus eval results** (billions of rows total). These live in
  the eval parquets; the registry only records the parquet path.
- **Training loss curves** (5,000 steps × 5,720 runs). These live in
  WandB and a future WandB-export parquet; the registry only records
  `final_loss` and a few aggregates.
- **Config contents** beyond their hash. The full YAML is recoverable
  from `git_commit` + `run_id`; we don't need to duplicate it in the
  record.
- **GPU-telemetry time series**. WandB system metrics cover this.
