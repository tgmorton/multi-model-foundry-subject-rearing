# Run Registry — full design

The registry is the portable, authoritative index of every training run
in the study. It exists to keep things **organized** as the study scales
from dozens of runs to thousands — not to auto-schedule or auto-launch
anything. The user decides when to run groups; the registry makes those
groups easy to enumerate, launch, and track.

This document specifies the complete schema, state machine, write
contracts, read patterns, launcher helpers, and integration points. It
covers every writer (training, eval, pruners, reference-rep selector,
HP sweeps) that will touch registry records.

For the lower-level "how do pods talk to S3" plumbing, see
[`S3_INTEGRATION.md`](S3_INTEGRATION.md). This doc focuses on **what
the records contain, who changes what when, and how the user drives
group operations off them**.

## 1. Why the registry exists

Five jobs, all for the user's convenience, none automatic:

1. **"Launch all English GPT-2 medium baselines"** — the user issues
   one command, the launcher (§7) consults the registry to see which
   of the 10 seeds already exist, renders one K8s Job per missing or
   failed seed, and reports back. The registry is what makes this one
   command possible.
2. **"Eval all completed baselines that don't have BLiMP yet"** — same
   shape, different script. Registry is the queryable filter.
3. **"Tell me which cells still need training"** — read-only; a table
   query against the registry.
4. **"Which seed is the reference rep for each cell?"** — a registry
   field, written once per cell after all 10 reps have eval'd.
5. **"What exactly produced this result?"** — `git_commit` +
   `config_hash` + `cache_key` + `docker_image` travel with every
   record forever, for the paper's data-availability statement.

WandB covers parts of (3)–(5) live but is a dashboard, not an archive —
it won't be reliable API-accessible in 5 years when a reader reproduces
the paper. The registry is.

**What the registry is NOT:** a scheduler, a job queue, a daemon, an
auto-relauncher. If a run goes stale, a reaper (§6) marks it
`PREEMPTED` and the **user** decides whether to relaunch. If eval
fails, the registry shows `eval_status=FAILED` and the **user** decides
whether to retry. Cron-like automation that's safe to run unattended
(compaction, reaping) is explicitly scoped in §12; everything else is
user-driven.

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

One additional administrative state: **`SUPERSEDED`** — a run whose
COMPLETE record predates a correctness fix and whose artifacts are
being regenerated (set by a one-off migration, never by the training
writers; carries `superseded_at` + `superseded_reason`). Used
2026-06-04 for the 112 pre-fix truncated-checkpoint-schedule runs being
recovered by the v2 resume wave. Analysis readers filter
`status == COMPLETE` and therefore exclude these automatically; the
record flips back through `RUNNING → COMPLETE` as its recovery runs.

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

All transitions are either writer-driven (training pod, eval runner)
or user-driven (launcher, selector). Nothing moves on its own.

```
                     ┌──────────────────────────────────────────────────┐
                     │  User runs `scripts/launch_training.py` for a    │
                     │  group (e.g. all 10 seeds of gpt2_med / en /     │
                     │  baseline). Launcher filters by registry state.  │
                     └──────────────────────────────────────────────────┘
                                         │
                                         ▼
                                    ┌─────────┐
                                    │ QUEUED  │  (launcher writes this just before kubectl apply;
                                    └────┬────┘   gives us a "know everything that was launched"
                                         │          trail even if the pod never starts)
                                         │  training pod starts
                                         ▼
                                    ┌─────────┐
                                    │ RUNNING │◄──── heartbeat every ~5 min
                                    └────┬────┘
                                         │
                                 ┌───────┼───────────┬────────────────┐
                                 │       │           │                │
                                 │finish │ crashes   │ node preempted │ 2h stale
                                 ▼       ▼           ▼                ▼
                              COMPLETE FAILED    PREEMPTED           PREEMPTED
                                 │       │           │         (reaper marks)
                                 │       └───────────┴────────────────┘
                                 │                   │
                                 │                   └─ user decides whether
                                 │                       to relaunch; launcher
                                 │                       filters by status
                                 ▼
                     ┌─────────────────────────────────────┐
                     │  User runs `scripts/launch_evals.py`│
                     │  over all COMPLETE runs missing X   │
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

"Trigger" column: **user** = the user runs a script manually;
**automatic** = driven by the pod or a CronJob without a user in the loop.

| Writer | Trigger | Fields written | Call site |
|---|---|---|---|
| **launcher** | user runs `scripts/launch_training.py` for a group | 4.1 identity, `status=QUEUED`, `created_at`, intended `hyperparameters` / `train_steps` | `scripts/launch_training.py` (TBD) → `register_run_queued` |
| **training entrypoint** | automatic — pod starts | 4.1 identity, 4.2 reproducibility, 4.3 provenance, `status=RUNNING`, `started_at`, `last_heartbeat_at`, `attempt_count++` | `model_foundry/cli.py::run` → `register_run_start` |
| **training loop** | automatic — every ~5 min | `last_heartbeat_at`, `current_step`, `current_loss` | `model_foundry/training/loop.py` → `heartbeat` |
| **training entrypoint (end)** | automatic — pod completion | `status` (COMPLETE/FAILED/PREEMPTED), `finished_at`, `duration_seconds`, `final_loss`, `steps_completed`, `epochs_completed`, `total_tokens_processed`, `checkpoint_count`, `checkpoint_paths`, `resume_state_steps`, `tokens_per_sec_avg`, `data_fraction_avg`, `failure_reason`, `oom_count` | `model_foundry/cli.py::run` → `register_run_end` |
| **eval launcher** | user runs `scripts/launch_evals.py` for a group | `eval_status=QUEUED`, `eval_benchmarks[X]={status: QUEUED}` per requested benchmark | `scripts/launch_evals.py` (TBD) → `register_eval_queued` |
| **eval runner** | automatic — eval pod starts | `eval_status=RUNNING`, `eval_started_at`, `eval_benchmarks[X]={status: RUNNING}` | eval runner → `register_eval_start(benchmark=X)` |
| **eval runner** | automatic — per-benchmark finish | `eval_benchmarks[X]={status, parquet_path, finished_at, metric_summary}`; aggregate `eval_status` recomputed | → `register_eval_benchmark_done` |
| **eval runner** | automatic — last benchmark done | `eval_status` aggregate, `eval_finished_at`, `eval_duration_seconds` | → `register_eval_end` |
| **in-training pruner (1.4)** | user runs `scripts/prune_in_training.py`, OR gets invoked at the end of each training pod's script | `post_run_pruned_at`, updates `checkpoint_paths` (analysis-only subset) | `scripts/prune_in_training.py` (TBD) → `register_pruner_event('post_run')` |
| **post-eval pruner (1.5)** | user runs `scripts/prune_post_eval.py` | `post_eval_pruned_at`, removes non-ref-rep `checkpoint_paths`, compacts `resume_state_steps` to empty | `scripts/prune_post_eval.py` (TBD) → `register_pruner_event('post_eval')` |
| **ref-rep selector (1.6)** | user runs `scripts/select_reference_reps.py` | On exactly one seed per cell: `is_reference_rep=true`, `reference_rep_rationale`, `reference_rep_metric`, `reference_rep_score`, `reference_rep_selected_at` | `scripts/select_reference_reps.py` (TBD) → `mark_reference_rep` |
| **archiver** | user runs `scripts/archive_runs.py` | `archived_at`, `archive_paths`, `archive_size_bytes` | `scripts/archive_runs.py` (TBD) → `register_archive_event` |
| **reaper** | automatic CronJob — marks stale RUNNING | `status=PREEMPTED`, `failure_reason="stale heartbeat > 2h"` | `scripts/reap_stale_runs.py` (TBD) |
| **HP sweep launcher** | user runs `scripts/launch_hp_sweep.py` | Multiple records with `run_kind=hp_sweep`, `status=QUEUED`, `hp_sweep_id`, 4.9 fields | `scripts/launch_hp_sweep.py` (TBD) |
| **HP sweep agent** | automatic — per trial pod | 4.1 identity (as `hp_sweep`), 4.2 reproducibility, 4.3 provenance, 4.4 lifecycle, 4.5 outputs, 4.9 sweep fields | `scripts/sweep_agent.py` (existing, needs registry wiring) |
| **HP sweep selector** | user runs `scripts/select_hp_winner.py` | `hp_sweep_rank`, `is_hp_winner` on one trial per arch | `scripts/select_hp_winner.py` (TBD) |
| **compactor** | automatic CronJob — hourly | (reads only; writes `registry.parquet`) | `scripts/compact_registry.py` |

Rule: **field ownership is single-writer.** If two writers want the
same field, we make one of them canonical and the other read-only.

**What runs automatically vs. user-driven.** The only CronJobs in the
plan are **compactor** (harmless: rebuilds a materialized view) and
**reaper** (marks stale RUNNING as PREEMPTED; does NOT relaunch). Every
other transition is either (a) automatic within a pod that the user
launched, or (b) explicitly invoked by the user. Nothing auto-launches.

## 7. Launcher helpers — the user-facing interface

The registry's ergonomic payoff: the user issues one command, the
helper consults the registry to pick up the state, and launches just
what's needed. All of these are thin Python scripts (~100–200 LOC
each) that wrap (a) a registry query, (b) a K8s Job template render,
(c) `kubectl apply`. None of them run on a schedule.

### 7.1 `scripts/registry_list.py` — see the state of a group

Read-only. Prints a table grouped by cell, with a colored status per
seed.

```
$ registry_list --arch gpt2_medium --lang en --condition baseline

arch          lang  condition  seed  status      eval    ref?  final_loss  duration
gpt2_medium   en    baseline   0     COMPLETE    COMPL.  —     3.15        19h47m
gpt2_medium   en    baseline   1     COMPLETE    RUN.    —     3.18        19h58m
gpt2_medium   en    baseline   2     RUNNING     —       —     4.12 (cur)  12h14m+
gpt2_medium   en    baseline   3     FAILED      —       —     —           OOM @step1804
gpt2_medium   en    baseline   4-9   NOT-STARTED —       —     —           —

6 of 10 seeds done, 1 running, 1 failed, 4 not started.
```

### 7.2 `scripts/launch_training.py` — launch a group

Takes filters + seeds; for each matching cell that isn't already done
or running, renders a K8s Job YAML from a template, `kubectl apply`s
it, and writes `status=QUEUED` to the registry.

```bash
# "Launch all English GPT-2 medium baselines"
python scripts/launch_training.py \
  --arch gpt2_medium --lang en --condition baseline \
  --seeds 0-9

# Same but only for cells that aren't already COMPLETE/RUNNING
# (the default — re-running is opt-in)
python scripts/launch_training.py \
  --arch gpt2_medium --lang en --condition baseline \
  --seeds 0-9 \
  --skip-if-status COMPLETE,RUNNING

# Relaunch the failed / preempted seeds
python scripts/launch_training.py \
  --arch gpt2_medium --lang en --condition baseline \
  --seeds 0-9 \
  --only-if-status FAILED,PREEMPTED

# Dry run — show the kubectl-apply plan without applying
python scripts/launch_training.py \
  --arch gpt2_medium --lang en --condition baseline \
  --seeds 0-9 \
  --dry-run
```

The script expands filters to concrete cells:

```
Plan:
  gpt2_medium-en-baseline-s0   skip (status=COMPLETE)
  gpt2_medium-en-baseline-s1   skip (status=RUNNING)
  gpt2_medium-en-baseline-s2   launch
  gpt2_medium-en-baseline-s3   launch (previous FAILED with OOM @step1804)
  gpt2_medium-en-baseline-s4   launch
  ...
Launching 8 jobs. Continue? [y/N]
```

Multi-condition, multi-arch launches work the same way — filter lists
or globs:

```bash
# All conditions × both languages × all seeds for gpt2_medium
python scripts/launch_training.py \
  --arch gpt2_medium \
  --lang en,es \
  --condition baseline,impoverish_case,lemmatize_verbs \
  --seeds 0-9
```

### 7.3 `scripts/launch_evals.py` — launch eval on a group

Same shape, with benchmark selection. Filters to runs where
`status=COMPLETE` and the specified benchmarks aren't already
`eval_benchmarks[X].status=COMPLETE`.

```bash
# "Eval BLiMP on every completed GPT-2 medium baseline"
python scripts/launch_evals.py \
  --arch gpt2_medium --lang en --condition baseline \
  --benchmarks blimp

# All benchmarks we care about, across all seeds of a whole matrix
python scripts/launch_evals.py \
  --arch gpt2_medium \
  --lang en,es \
  --condition baseline \
  --benchmarks blimp,perplexity,null_subject_expletive
```

### 7.4 `scripts/launch_hp_sweep.py` — launch an HP sweep

Kicks off a WandB sweep with N trials for a given arch. Each trial
gets its own registry record with `run_kind=hp_sweep`. After the sweep
finishes, the user runs `scripts/select_hp_winner.py` which picks the
top trial by `hp_proxy_score` and sets `is_hp_winner=true` — that
winner's `hyperparameters` block becomes the production HP config for
that arch.

```bash
# 30 trials, Bayesian, Hyperband early-stop, 2000-step proxy
python scripts/launch_hp_sweep.py \
  --arch gpt2_medium --lang en --condition baseline \
  --trials 30 --proxy-steps 2000 --early-stop hyperband

python scripts/select_hp_winner.py --arch gpt2_medium
```

### 7.5 `scripts/select_reference_reps.py` — pick reference reps

Once all 10 seeds of a cell have `eval_status=COMPLETE`, run this to
mark the median-BLiMP seed as the reference rep for that cell.

```bash
# For one cell
python scripts/select_reference_reps.py \
  --arch gpt2_medium --lang en --condition baseline \
  --metric "blimp.metric_summary.accuracy_final_ckpt" \
  --reducer median

# For every cell that has 10 complete seeds and no ref-rep yet
python scripts/select_reference_reps.py --auto
```

### 7.6 Pruning and archival scripts

Purely user-invoked, never on a cron. Each takes a filter and a
confirmation prompt. See §6's rows for what fields they touch.

```bash
# Post-run prune: strip training_state.pt from old checkpoints,
# keep the resume_state_steps (tip + 2 before)
python scripts/prune_in_training.py --arch gpt2_medium --lang en --condition baseline

# Post-eval prune: drop non-ref-rep checkpoints entirely, strip the
# rest of training_state.pt from the ref rep. Destructive — requires
# --confirm.
python scripts/prune_post_eval.py \
  --arch gpt2_medium --lang en --condition baseline \
  --confirm

# Archive: move the ref rep's remaining checkpoints from the hot PVC
# to the cold archive PVC (subject-drop-archive).
python scripts/archive_runs.py \
  --arch gpt2_medium --lang en --condition baseline
```

## 8. Read patterns

These are the queries the launchers (§7) and analyses actually run.
The compacted Parquet handles all of them natively.

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

## 9. Rollup views beyond `registry.parquet`

We may also produce secondary materialized views from the same
`by_run/` source. Each is its own scheduled job.

| View | Key | Rows | Purpose |
|---|---|---|---|
| `registry.parquet` | one row per run | ~17K at peak | Primary: covers 99% of queries. |
| `cells.parquet` | one row per (arch, lang, condition) | ~320 | Per-cell aggregates: how many seeds done, reference rep id, mean final loss. For the orchestrator dashboard. |
| `archs.parquet` | one row per (arch, lang) | ~20 | Total tokens processed, total GPU hours, reference-rep final loss, etc. Top-level dashboard. |

None of these are strictly needed now. `registry.parquet` is — the
others are follow-ups when the analysis code starts feeling slow.

## 10. Concurrency & failure semantics

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

## 11. Schema evolution

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

## 12. Integration points — file-by-file

### Already landed (commit 3bd0d80)

- `model_foundry/registry.py` — `register_run_start`, `heartbeat`,
  `register_run_end`, `register_eval_end`, `mark_reference_rep`,
  `write_env_snapshot`, `iter_all_records`.
- `scripts/compact_registry.py` — rebuilds `registry.parquet`.
- `docs/S3_INTEGRATION.md` — S3 endpoint / secret / env-var block.
- `boto3>=1.34` in `requirements.txt`.
- `s3-secret-thomas` K8s Secret exists.
- `thomas-subject-drop-artifacts` bucket exists.

### Pod-side writers (training entrypoint + loop)

1. **Extend `register_run_start` / `register_run_end`** with the fields
   in §4.2 (`docker_image`, `dataset_manipulation_hash`, `hyperparameters`),
   §4.4 (`oom_count`), §4.5 (`steps_completed`, `epochs_completed`,
   `tokens_per_sec_avg`, `data_fraction_avg`, `resume_state_steps`).
   Small, additive changes to the module. ~30 LOC.
2. **Add `register_eval_start` / `register_eval_benchmark_done`** for
   per-benchmark granularity (§4.6). ~40 LOC. The existing
   `register_eval_end` becomes the final aggregate call.
3. **Add `register_run_queued` / `register_eval_queued`** for the
   launchers (§7.2, §7.3) to mark intent before the pod starts.
   ~10 LOC each.
4. **Add `register_pruner_event`, `register_archive_event`**. Each is
   a thin `_merge_record` call. ~20 LOC each.
5. **Wire `cli.py::run`** to call `register_run_start` before training
   and `register_run_end` in a top-level `try/except/finally`. Read
   `git_commit` and `config_hash` from existing trainer machinery. ~50 LOC.
6. **Wire `loop.py`** to call `heartbeat` every N optimizer steps
   (default: compute N such that the cadence is ~5 minutes). ~10 LOC.
7. **Add env vars** from `S3_INTEGRATION.md` §K8s-env-vars to every
   training/eval K8s template.

### Launcher helpers — the user-facing interface (§7)

These are what the user actually invokes to run groups of things.
Small, self-contained scripts that read the registry, render a
template, and `kubectl apply`. Targeted LOC per script: 100–200.

- `scripts/registry_list.py` — table view of a filter (§7.1).
- `scripts/launch_training.py` — launch a training group (§7.2).
- `scripts/launch_evals.py` — launch an eval group (§7.3).
- `scripts/launch_hp_sweep.py` — kick off a WandB sweep (§7.4).
- `scripts/select_reference_reps.py` — pick ref reps per cell (§7.5).
- `scripts/select_hp_winner.py` — mark HP-sweep winner.
- `scripts/prune_in_training.py` — post-run cleanup (1.4).
- `scripts/prune_post_eval.py` — post-eval cleanup (1.5).
- `scripts/archive_runs.py` — hot → cold archive.

All of these consume the registry (find candidates) and write back
(status=QUEUED / pruner events / archive events / ref-rep). They call
functions already on `model_foundry.registry`, not new ones.

The common pattern for every launcher:

```python
import click
from model_foundry import registry
from scripts._k8s_render import render_job, apply

@click.command()
@click.option("--arch", required=True)
@click.option("--lang", required=True)
@click.option("--condition", required=True)
@click.option("--seeds", default="0-9")
@click.option("--skip-if-status", default="COMPLETE,RUNNING,QUEUED")
@click.option("--dry-run", is_flag=True)
def launch_training(arch, lang, condition, seeds, skip_if_status, dry_run):
    skip = set(skip_if_status.split(","))
    plan = []
    for seed in parse_range(seeds):
        run_id = registry.build_run_id(arch, lang, condition, seed)
        current = registry.get_record(arch, lang, condition, run_id)
        if current and current.get("status") in skip:
            plan.append((run_id, "skip", current["status"]))
            continue
        plan.append((run_id, "launch", None))
    render_plan_table(plan)
    if dry_run or not confirm("Continue?"):
        return
    for run_id, action, _ in plan:
        if action != "launch":
            continue
        registry.register_run_queued(
            arch=arch, lang=lang, condition=condition, run_id=run_id, ...
        )
        yaml_text = render_job("training", run_id=run_id, seed=seed, ...)
        apply(yaml_text)
```

### Automatic infrastructure (CronJobs — two of them, scope-limited)

- **Compactor CronJob** — new `k8s/cronjob-compact-registry.yaml`
  that runs `scripts/compact_registry.py` hourly. Harmless: rebuilds
  `registry.parquet` from `by_run/` shard. ~30 LOC of YAML.
- **Reaper CronJob** — `scripts/reap_stale_runs.py` +
  `k8s/cronjob-reap-stale-runs.yaml`. Looks for `status=RUNNING` with
  `last_heartbeat_at < now - 2h`, marks as `PREEMPTED` with a
  failure_reason. **Does not relaunch**; just marks. ~50 LOC of code.

Every other "should this run automatically?" impulse should come back
to the user as a launcher invocation.

### HP sweep wiring

- `scripts/sweep_agent.py` (audit of existing) — add
  `register_run_start(run_kind="hp_sweep")` + `register_run_end` and
  populate §4.9 fields from each trial.

### Explicitly NOT building

- **An orchestrator daemon.** Earlier drafts proposed one; we decided
  against. The user drives group launches via §7 helpers. If we later
  want unattended "keep the matrix topped up" behavior, it can be a
  thin CronJob that invokes `launch_training.py --skip-if-status=... --seeds=...`
  with a fixed plan — but that's a separate, opt-in decision, not
  part of the registry design.

## 13. One worked example: a single production cell, cradle to archive

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

## 14. What we explicitly do NOT put in the registry

- **Per-stimulus eval results** (billions of rows total). These live in
  the eval parquets; the registry only records the parquet path.
- **Training loss curves** (5,000 steps × 5,720 runs). These live in
  WandB and a future WandB-export parquet; the registry only records
  `final_loss` and a few aggregates.
- **Config contents** beyond their hash. The full YAML is recoverable
  from `git_commit` + `run_id`; we don't need to duplicate it in the
  record.
- **GPU-telemetry time series**. WandB system metrics cover this.
