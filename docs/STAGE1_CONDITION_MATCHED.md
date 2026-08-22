# Stage-1 / fdy training + condition-matched evaluation

Reference doc for the line of work that landed via the `codex/fix-continuation-resume`
merge (`675bc3c`). Covers the Stage-1 English training waves, the checkpoint-resume
fix that motivated a schedule rerun, the condition-matched evaluation regime, and
the figure pipeline that consumes its results. Every claim below was verified by
reading the cited file; anything not directly verified is marked UNVERIFIED.

## 1. Stage-1 ("fdy") training waves

**Naming**: `thomas-fdy-s1e-{phase}-{arch-short}-{cond-short}-lr30r1[-nfN]`, built
by `_job_name` in
[scripts/generate_stage1_early_manifests.py](../scripts/generate_stage1_early_manifests.py).
`phase` ∈ {`early`, `full`, `wave`, `sentinel`}; `lr30r1` marks the corrected
30-epoch-scheduler rerun (§2); `-nfN` (node-failure recovery attempt N) appears
only on hand-created per-architecture resubmissions, not on generator output.

**What launched them**: the generator is manifest-only (never calls `kubectl`)
and imports `ARCH_SETTINGS`/`SEEDS`/`INTERVENTIONS` from
[scripts/launch_production_training.py](../scripts/launch_production_training.py)
so batch/RAM/CPU tuples stay in sync with production.
[k8s/stage1/README.md](../k8s/stage1/README.md) is the human rollout doc
(submission order, seed set, per-arch resource table). Outputs:
`k8s/stage1/sentinels/job-stage1-early-sentinel-<arch>-en.yaml`,
[job-stage1-early-wave-en.yaml](../k8s/stage1/job-stage1-early-wave-en.yaml),
[job-stage1-full-selected-wave-en.yaml](../k8s/stage1/job-stage1-full-selected-wave-en.yaml).

**Design — 12 seeds, 2 legacy + 6 early + 4 full**: 6 architectures
(gpt2_small/medium/large, bert_large, lstm, mamba_370m) × 5 English conditions
(baseline, remove_expletive_sentences, impoverish_case, lemmatize_verbs,
enrich_verbal_morphology — a subset of the study's full 8-condition design) = 30
cells, each with the same 12-seed set (`STAGE1_SEEDS`), only hp_rank `h0`:

- **2 legacy anchors** (seeds 42, 137) — pre-existing full 30-epoch runs, not relaunched.
- **6 early-only seeds** — `PROD_EPOCHS=2`, but the LR scheduler still follows
  `SCHEDULER_EPOCHS=30` (§2). One (seed `1415936399`) is the architecture
  sentinel; for `gpt2_small` it's already complete and excluded from that cell's slots.
- **4 full-run seeds** (`FULL_RUN_SEEDS`) — trained straight through all 30 epochs.

**Verified nuance**: the "full-selected" wave is *not* a checkpoint-resume
continuation of the "early" wave. `_slots_for_wave` excludes `FULL_RUN_SEEDS`
entirely, and both early and full jobs render with `resume=False` — two disjoint
seed subsets of the same 12, both launched fresh under the same
`SCHEDULER_EPOCHS=30` LR curve; the early set just stops after 2 epochs. A
generated collision guard refuses to start into an existing
`/mnt/data/models/production/<run_id>` directory unless it's a retry, in which
case it archives the old attempt to `_failed_attempts/` and starts clean —
retries are not resumed in place either.

Header comments (ground truth) give the real counts: early wave = 30 jobs /
**179** trajectories (6×30 minus 1 already-completed sentinel slot); full-selected
wave = 30 jobs / **120** trajectories (4×30).

**Sentinels / ops / recovery**: of the 6 files in `k8s/stage1/sentinels/`, only
`gpt2-small` carries the `-lr30r1` suffix (today's generator only emits a sentinel
for `gpt2_small`); the other five predate the fix and validated per-arch pod
sizing under the original two-epoch schedule. `k8s/stage1/ops/` holds ad-hoc
incident Pods (BERT OOM, a node failure). The per-arch `-p2-en.yaml` tranche files
and `-nf1`/`-nf2` recovery files are hand-produced resubmissions of subsets of the
30-cell early wave after cluster failures — not reproduced by re-running the
generator.

**Verified discrepancy**: `k8s/stage1/README.md:10` claims the all-architecture
wave "keeps `parallelism: 1`," but the committed
[job-stage1-early-wave-en.yaml](../k8s/stage1/job-stage1-early-wave-en.yaml) has
`parallelism: 2` throughout. Git history explains it: `0150d58` ("define
corrected lr30 early-wave rerun") predates `baf2766` ("stage early and full waves
at parallelism two") — the README prose was never updated after that change.

## 2. The continuation-resume fix

Two same-day commits (`2026-08-03`) fixed a real correctness bug in checkpoint
resume, relevant to any run — Stage-1 or otherwise — with `resume_from_checkpoint: true`.

**`resolve_resume_epoch`** (commit `ff4f925`),
[model_foundry/training/checkpointing.py:21-39](../model_foundry/training/checkpointing.py#L21-L39):

```python
def resolve_resume_epoch(global_step, saved_epoch, steps_per_epoch,
                          resume_batch_offset, epoch_completed) -> int:
    if epoch_completed:
        return saved_epoch + 1
    if (global_step > 0 and steps_per_epoch > 0 and resume_batch_offset == 0
            and global_step == (saved_epoch + 1) * steps_per_epoch):
        return saved_epoch + 1
    return saved_epoch
```

**The bug**: a checkpoint saved exactly at an epoch boundary (the `TrainingLoop`
endpoint guard, `loop.py:428-445`) carried only the just-finished `epoch` index.
On resume, training re-entered at that *same* epoch instead of advancing,
silently replaying an already-seen epoch — breaking the token-matched
cross-architecture comparison the study depends on. The fix adds an explicit
`epoch_completed` boolean persisted in `training_state.pt`/`metadata.json`, so new
checkpoints self-declare completion; legacy checkpoints (predating the field) are
still recognized by inference (zero offset + step exactly at the epoch boundary).
`trainer.py:579-590` calls it right after `load_checkpoint` and logs a warning
when it advances the epoch. Covered by 4 unit tests in
[test_checkpointing.py:21-32](../model_foundry/tests/unit/training/test_checkpointing.py#L21-L32).

**`scheduler_train_steps` decoupling** (commit `c532553`): adds
`TrainingConfig.scheduler_train_steps`, an optional longer LR horizon training may
stop short of. `Trainer._calculate_training_parameters` computes warmup against
it (defaulting to `train_steps`) and raises if it's shorter than `train_steps`.
This is exactly why Stage-1's early seeds can stop after 2 epochs while their LR
schedule still follows the full 30-epoch curve — without it, a 2-epoch run's
linear decay would fully unwind within 2 epochs, putting its checkpoints at a
different point on the LR curve than the matched step of a real 30-epoch run.
`scripts/production_agent.py` reads `SCHEDULER_EPOCHS` from the pod env and fatals
if it's shorter than `PROD_EPOCHS`. This is the `stage1-two-epoch-lr-schedule` bug
that every `lr30r1` job's `foundry.thesis/supersedes` annotation references.

**Scope note**: Stage-1 jobs pass `resume=False`, so `resolve_resume_epoch` isn't
actually exercised by these manifests — every Stage-1 trajectory is a fresh run.
The scheduler decoupling is the fix Stage-1 directly depends on; the epoch-resume
fix is general-purpose, mattering for production training or any other run that
sets `resume_from_checkpoint: true`.

## 3. Condition-matched evaluation

**What "matched" means**: the original `null_subj_v2` benchmark scores every run
against one fixed stimulus set, but each condition trains on a corpus with a real
surface intervention applied — scoring all conditions against unmodified stimuli
means eval surface form doesn't match what the model trained on.
[scripts/generate_condition_matched_stimuli.py](../scripts/generate_condition_matched_stimuli.py)
applies the **exact production ablation operation** to the eval minimal pairs,
per condition, pair-aware (shared context parsed once; shared-token edits
reconciled to the overt member's parse so the intervention doesn't add a second
contrast). Frozen per-condition policy is in
[docs/eval_stimuli/condition_matched_v1.md](eval_stimuli/condition_matched_v1.md):
baseline and remove-expletive stimuli are unchanged (the latter is a
training-distribution deletion, not a rewritable surface form); impoverish-case
and lemmatize-verbs reconcile shared edits to the overt parse; enrich-verbal-morphology
is applied literally and independently to both pair members, which changes the
estimand (256/576 pairs get different surface morphology on a shared token — must
be labeled separately in analysis).

Output: [evaluation/stimuli/null-subj-v2-matched-v1/\<condition\>/en/*.csv](../evaluation/stimuli/null-subj-v2-matched-v1)
— 8 categories × 5 conditions, 576 pairs per condition (verified by row count).
`manifest.json` pins source/output hashes, generator hash, each ablation source
hash, git state, and parser version (`en_core_web_trf==3.7.3`); passed independent
review 2026-08-14, `vetted: true`. Benchmark key: `null_subj_v2_condition_matched_v1`;
scoring version: `null-subj-v2-condition-matched-v1`.

**Tranche system — stable lane vs. delta/H1 lane**:
[scripts/generate_condition_matched_eval_manifests.py](../scripts/generate_condition_matched_eval_manifests.py)
groups an inventory of trained runs by architecture into indexed Jobs under
`k8s/condition_matched_eval/` (separate from `k8s/stage1/eval/`, which serves the
older non-matched `null_subj_v2` benchmark), supporting `--exclude-arch-hp` /
`--only-arch-hp` to hold back a lane. The frozen tranche in
[fleet_summary.json](../k8s/condition_matched_eval/fleet_summary.json) selected
715 of 803 inventoried runs, excluding `["bert_large:h1", "gpt2_large:h1",
"mamba_370m:h1"]`. "H1" = hyperparameter rank 1. The **stable lane** is every run
whose training is finished/frozen (all hp_rank 0, plus hp_rank 1 where that
architecture's H1 training already completed); the **delta/H1 lane** is hp_rank-1
cells for architectures still training, held back to be added later "through CPU
fanout from the same state/condition representatives" without disturbing the
first tranche's provenance
([init_split/README.md:27-29](../k8s/condition_matched_eval/init_split/README.md)).
Records are "namespaced by frozen inventory hash, so a later H1 delta tranche
cannot overwrite the stable tranche's provenance"
([data/eval_results/.../README.md:17,23-24](../data/eval_results/null_subj_v2_condition_matched_v1/README.md)).

**Checkpoint-inventory preflight**:
[scripts/build_condition_matched_eval_inventory.py](../scripts/build_condition_matched_eval_inventory.py)
walks `/mnt/data/models/production` on the PVC directly (not S3/registry),
matches run-ID directories, globs `checkpoint-*` subdirs, and verifies each has a
readable weight file, rejecting `missing_weights`/`invalid_step`. Writes a
`condition-matched-eval-inventory.v1` JSON; supports per-arch sharding + `--resume`;
shards combine via
[scripts/merge_condition_matched_eval_inventories.py](../scripts/merge_condition_matched_eval_inventories.py)
(requires every shard `complete: true`, rejects duplicate run IDs). Frozen
inventory SHA-256 `25f044a6...b5f37` appears identically in `fleet_summary.json`
(`inventory_sha256`) and the `data/eval_results` README, confirming the same
frozen selection.

**Immutable manifest publishing**:
[scripts/publish_condition_matched_eval_manifests.py](../scripts/publish_condition_matched_eval_manifests.py)
(invoked by [job-publish-manifests-v1.yaml](../k8s/condition_matched_eval/job-publish-manifests-v1.yaml))
uploads `checkpoint_inventory.json`, two tokenizer-validation JSONs, and the
stimuli `manifest.json` to `s3://thomas-subject-drop-artifacts/eval_results/<benchmark>/manifests/`.
Its `upload_once()` helper refuses to overwrite an S3 key whose existing object
has a different SHA-256 in its metadata.

**Initialization (checkpoint −1) split eval**:
[scripts/eval_v2_initialization.py](../scripts/eval_v2_initialization.py) scores
the pre-training, randomly-initialized model state — "the state immediately
before [the] first optimizer update" — re-seeding to match `Trainer._train_loop`
exactly, recording `checkpoint_step=-1`, `tokens_seen=0`. It writes to a separate
benchmark, `null_subj_v2_condition_matched_init_v1`, combinable with
trained-checkpoint tables only after both pass independent audits.
[k8s/condition_matched_eval/init_split/](../k8s/condition_matched_eval/init_split)
splits this into a GPU stage (construct + hash one deterministic state per
(architecture, seed), score the 5 matched conditions once, publish one
representative result per condition) and a CPU stage (fan out remaining HP/cell
identities from that representative, no GPU needed) — since checkpoint −1 scoring
is GPU-bound but per-cell fanout is CPU/I/O-bound. GPU jobs must be terminal and
pass integrity checks before their CPU counterpart runs.

**Audits**:
[scripts/audit_condition_matched_eval_results.py](../scripts/audit_condition_matched_eval_results.py)
and [scripts/audit_condition_matched_init_results.py](../scripts/audit_condition_matched_init_results.py)
check completeness against the frozen inventory (exact checkpoint-step sets, row
counts, `cell_id`/intervention consistency, numeric finiteness, provenance
columns) — a green K8s Job or S3 listing is explicitly not completion proof
([condition_matched_v1.md:81-84](eval_stimuli/condition_matched_v1.md)). The init
audit additionally checks the model-state hash is identical across all
cells/conditions sharing an (architecture, seed), and score digests match across
HP ranks sharing the same state/condition.

## 4. The figure pipeline

[analysis/eval_v2/figures/foundry_trajectories/condition_matched_v1/](../analysis/eval_v2/figures/foundry_trajectories/condition_matched_v1)
holds 64 CSV/PNG pairs plus `README.md`, `coverage_manifest.json`,
`coverage_counts.csv`, `final_token_cutoffs.csv`, and three subdirs
(`checkpoint_minus1/`, `comparisons/`, `endstate_forests/`), all committed in
`d9170e5` ("stage1 launch fleet, condition-matched plotting layer, trajectory
figures").

Producing scripts (all under top-level `scripts/`, not `analysis/eval_v2/`):

- [scripts/plot_condition_matched_trajectory_suite.py](../scripts/plot_condition_matched_trajectory_suite.py)
  reads `data/eval_results/null_subj_v2_condition_matched_v1/{pairs,checkpoints}/*.parquet`,
  computes token-binned per-cell mean `prefers_overt_meanlp` trajectories faceted
  by hp_rank/intervention/stage_segment, and writes most of the directory:
  per-arch PNG/CSV pairs, `trajectory_aggregates.parquet`, `coverage_counts.csv`,
  `final_token_cutoffs.csv`, `README.md`, `coverage_manifest.json`.
- [scripts/plot_condition_matched_endstate_forests.py](../scripts/plot_condition_matched_endstate_forests.py)
  reads the same parquets plus `final_token_cutoffs.csv`, computes one truncated
  end-state preference per architecture at the common token horizon (restricted
  to hp_rank/seed tuples reaching it in all 5 interventions), writes
  `endstate_forests/` (per-arch forest plots, `endstate_summary.csv`,
  `endstate_cell_values.parquet`).
- [scripts/build_condition_matched_h0_trajectory.py](../scripts/build_condition_matched_h0_trajectory.py)
  is a standalone CLI building one flat CSV of cell/category/token-binned
  preferences for one architecture — not tied to the committed directory's filenames.

`data/eval_results/null_subj_v2_condition_matched_v1/` is, as of this writing,
**only its `README.md` locally** — no `pairs/`/`checkpoints/`/`items/`/`per_token/`
parquets are present or tracked. The README confirms (verbatim) it's "the local
mirror for the intervention-matched English Foundry evaluation wave" whose
durable canonical Parquets live at
`s3://thomas-subject-drop-artifacts/eval_results/null_subj_v2_condition_matched_v1/`,
pulled via `scripts/pull_eval_results.py --require-sha256`.

**CSV column families** — no single CSV has all facets at once:

- **hp_rank-faceted** (e.g. `baseline_by_hyperparameter_gpt2_small.csv`, exact
  header verified): `architecture,hp_rank,category,cell_id,bin_center,preference`
  — the family the task's expected column list describes.
- **intervention-faceted** (e.g. `gpt2_small_condition_matched_all_hp_by_intervention.csv`):
  same shape with `intervention` in place of `hp_rank`.
- `endstate_forests/*.csv` has a richer schema:
  `architecture,intervention,category,condition,mean_preference,sd,n_cells,endpoint_tokens_min,endpoint_tokens_max,common_final_tokens,se,lower,upper`.

## 5. Operational significance for the storage audit

The union of `cell_id` values across all 64 committed
`condition_matched_v1/*.csv` files is **715 unique run IDs** (directly computed),
exactly matching `coverage_manifest.json`'s `training_cells: 715` and
`fleet_summary.json`'s `runs: 715` (out of 803 inventoried, after excluding the 3
H1 lanes). Precisely what enumerates what: `coverage_manifest.json` does **not**
list `cell_id` strings — `training_cells` is a bare count, and `generated`/`pending`
are figure filenames, not run IDs. The actual enumeration lives in the per-row
`cell_id` column of the CSVs (format `<arch>-en-<condition>-h<hp_rank>-s<seed>`),
corroborated by `build_condition_matched_eval_inventory.py`'s inventory JSON.
Those 715 run IDs are exactly the training runs whose checkpoints have been
consumed by a completed condition-matched evaluation and must therefore remain
evaluatable (readable weights on the PVC, or reproducible from a retained
checkpoint) for as long as this analysis line is active — a storage-reaping pass
must treat them as a do-not-touch set distinct from the broader run registry.
