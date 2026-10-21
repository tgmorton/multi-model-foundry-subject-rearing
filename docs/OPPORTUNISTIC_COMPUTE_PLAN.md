# Opportunistic (preemptible) GPU tier: readiness and usage plan

**Status:** proposal, 2026-10-21; P0.1 done (see the review below).
**Code basis:** branch `feat/bert-recipe-v2` at `861f490c`.
**Scope:**
- wave2/w3 production training;
- short trials (races and sweeps);
- per-checkpoint evals;
- CPU preprocessing.

**Evidence tags:**
- **[read]**: read in the code at `861f490c`.
- **[kubectl]/[prom]**: seen read-only on the cluster or on NRP's public Prometheus, 2026-10-21 around 18:30 UTC.
- **[computed]**: arithmetic from measured numbers.
- **[inferred]**: reasoning, not verified.
- **[verify]**: must be checked before anything depends on it.

## Review and status (main loop, 2026-10-21 evening)

Written by a planning subagent; checked against the code by the main loop.
The audit findings I spot-checked hold: A5 (the loader's
`glob("checkpoint-*")` + `re.search(r'checkpoint-(\d+)')` accepts a
`checkpoint-N.tmp` staging dir), A4 (nothing sets `RESUME` for wave2), A18
(`preprocess_data` treats `dataset_info.json` *or a bare existing dir* as
done, `data.py:430`). A9 (signal path) I did not re-check.

**Done:**
- **P0.1** resume by default: `wave2_agent` now resumes unless `RESUME=0`
  (`8d0f0aa4` on `feat/bert-recipe-v2`). The agent default covers every
  launcher, so `render_job` was left alone. How the lemn-lab straggler
  `thomas-w2-gpt2large-rand70-lemverb` resumed on 2026-10-19 without
  `RESUME` in its env is still unexplained (the trainer logged
  `resume_from_checkpoint=True`); with the default flipped it no longer
  matters for new runs.
- **Precache OOM** (not in the plan): the 140-cell BERT-v2 precache was
  OOM-killed at 6Gi because chunking and dataset stats held the whole
  corpus as Python lists (~36 B/token, ~5 GB per cell). Now int32 blocks
  and arrow list lengths (same commit). Output verified identical to the
  old build on datasets 2.19.2 and 4.5.

**Order I'd take the rest in:**
1. P0.3 (loader) and P0.4 (writer) before any run goes to a preemptible
   pool. They matter on armada-default too: OOM, eviction and node loss
   also kill pods mid-save.
2. P0.5 (signal path) together with P1.2 (emergency save); one without the
   other buys little.
3. P0.2 needs Thomas's call (decision 1 below). Tranche 1 is a fresh wave,
   so merging now avoids mixed label semantics within the BERT-v2 runs.
4. P1.1 (time-based `_resume/`) before opportunistic A100 work; on H100 the
   per-epoch rolling cadence (~33 min) is tolerable given 18 h lifetimes.
5. P3 (work queue) only after the K1-K2 canaries pass. Until then: push
   Jobs with explicit per-pool item lists, as tranche 1 does.

**Side finding** (§4, pre-existing, all archs alike): `loop.run` never
zeroes gradients at an epoch boundary, so ~16 leftover micro-batches add
into the next epoch's first optimizer step. Clipping bounds it. Not
preemption-related; noted for a separate fix.

## 1. Summary

1. **Three things make preemptible compute unsafe today. Fix them first (P0, about 1.5 days).**
   - `CheckpointManager.load_checkpoint` can pick a half-written `checkpoint-N.tmp` staging directory, which leads to a crash loop.
   - wave2 restarts do not resume unless `RESUME=1` is set, and `wave2_launcher.render_job` never sets it.
   - SIGTERM never reaches the trainer.
   - In addition, the resume off-by-one fix (`5f2d3e78`) exists only on a local branch. Without it, every preemption adds one window of label lag.
2. **Save resume state on a wall-clock cadence, in a separate `_resume/` directory.**
   - Use 20 min on opportunistic pools and add an emergency save on SIGTERM, with `terminationGracePeriodSeconds: 120`.
   - Today's rolling resume fires about once per epoch: 33 min on H100, 70 min on A100, 2.7 h on A10. That is longer than the A100 pod lifetimes we measured (20 and 37 min), so A100 attempts would never bank any progress.
3. **Make trials resumable and stop relying on W&B to re-issue them.**
   - Grid and race sweeps (all D15 sweeps are grids) get a deterministic trial ledger.
   - Bayes sweeps use `mark_preempting` plus resume.
4. **Move training and trials to a pull-based work queue.**
   - The queue lives on the run's CephFS PVC: lease files with generation numbers, heartbeats and fencing, plus one worker Job per GPU pool.
   - This gives "first free GPU wins", cross-pool recovery after a preemption, no oversubscription and no idle pods.
   - Evals and CPU work stay on today's push model.
5. **Use 1-card pods. No DDP for study runs.**
6. **Pools:**
   - H100 `opportunistic2` is the main pool for long runs.
   - A100 `opportunistic2` is the canary's source of real preemptions, and later takes short work only.
   - Skip H200 and GH200.
7. **Every worker doubles as a probe.** 60-second S3 heartbeats feed a report and `fleet_watch` alerts (crash loop, livelock, stale lease, corrupt checkpoint).

## 2. The tier in numbers

**NRP facts (authoritative):**
- `opportunistic` and `opportunistic2` are quota-free for A100, H100, H200 and GH200, and owners can preempt them at any time.
- `opportunistic2` cannot be preempted by `opportunistic` pods.
- Preemption takes the whole pod. Victims are chosen by lowest priority, then fewest victims, then latest started.
- GH200 is arm64, which our image does not support.

**BERT-v2 production run** [computed]:
- Steps per epoch: floor(ceil(1,075,685 / 32) / 32) = 1,050. This is the study cache; wave cells are assumed to be similar.
- 30 epochs = 31,500 steps, at 131,072 tokens per step.

| Pool | First wait | Uninterrupted life | Tokens/s | 30-epoch run | 1 epoch (today's rolling cadence) |
|---|---|---|---|---|---|
| H100 opp/opp2 | 10–11 h overnight; ~1 min at midday PT | 18 h with 0 preemptions (1-card pods) | 67–73K (92K with a larger micro-batch) | ~16.4 h | ~33 min |
| A100 opp2 | short | 20 min, then 37 min (preempted both times) | 33K | ~35 h | ~70 min |
| H200 | never scheduled in 18 h | – | – | – | – |
| L40 (armada-default) | contended | not preemptible | 27.8K (larger micro-batch) | ~41 h | ~82 min |
| RTX 3090 / A10 | contended | not preemptible | 16.7K / 14.2K | ~69 h / ~81 h | ~2.2 h / ~2.7 h |

**Live snapshot** [prom, 2026-10-21 ~18:30 UTC]:

| GPU | Allocatable | Running, by priority class |
|---|---|---|
| H100 | 24 | `owner` 8, `nice` 8, `opportunistic2` 8 (5 of them ours) |
| A100 | 151 | ~140 requested: 116 with no class, `owner` 20, `opportunistic` 4 |
| H200 | 10 | `owner` 8, `opportunistic` 2 |

**What follows from this:**
- A run makes progress only if the checkpoint interval plus restart overhead is well below the pod's lifetime.
- A 2-card pod loses 2 runs per preemption.
- At `opportunistic2`, the cards available to us are the free ones plus the ones held by `opportunistic` pods.
- Kubernetes gives preemption victims their `terminationGracePeriodSeconds` [inferred from the Kubernetes docs; verify on NRP in canary K2].

## 3. Restart-safety contract

| # | Invariant |
|---|---|
| C1 | **Deterministic identity.** run_id, seed, output dir, W&B id and registry key are pure functions of the work item. A restart on any pod or pool maps to the same directory and record. |
| C2 | **Exclusive ownership.** At most one live writer per run directory (lease plus fencing). |
| C3 | **Skip-if-complete at start.** Registry COMPLETE plus a valid endpoint means exit 0 with success semantics. |
| C4 | **Atomic, durable checkpoint writes.** Stage in a pod-unique `*.tmp-*` dir, fsync files and dir, write a manifest last, then rename. A re-save never leaves a moment with no valid copy. |
| C5 | **Validated resume.** Only fully named dirs with a valid sentinel count; check sizes and loadability; quarantine failures and fall back to the next older state. |
| C6 | **Exact state restoration.** Weights, optimizer, scheduler, AMP scaler, all RNGs, data position, token counter and an explicit step-label semantics. Data order and masks depend only on (seed, epoch, batch index). |
| C7 | **Bounded loss.** Resume state on a wall-clock cadence well below pool lifetime, plus an emergency save inside the grace period. |
| C8 | **Signal path.** SIGTERM reaches the trainer. A preemption exits with a non-zero code that `podFailurePolicy` ignores. Never exit 0 unless the work is done. |
| C9 | **Idempotent side effects.** W&B resumes the same run; registry transitions are monotone and tolerate replays; eval markers are written only after results are durable. |
| C10 | **Atomic cache builds.** Build in a temp dir, rename, and write a completion sentinel that consumers check; concurrent builders converge. |
| C11 | **Crash-loop containment.** A deterministic failure (no step progress across k attempts) is parked, not retried until `backoffLimit`. |
| C12 | **Observability that outlives the pod.** S3 heartbeats with phase, step and last save; counts of preemptions and lost steps. |

Which invariants each workload needs:
- **Production training:** all of them.
- **Trials:** C1–C9, C11 and C12.
- **Evals:** C1, C3, C9, C11 and C12, plus incremental commits if they ever run on preemptible cards.
- **CPU preprocessing:** C1, C3, C10 and C11.

## 4. Audit of our code against the contract

| ID | Item | Status | Where | Finding |
|---|---|---|---|---|
| A1 | C1 | OK | `wave2_agent.main` | Seed is `derive_seed(WAVE_ID, cell, arch, h, rep)`; run_id and `models/wave2/<run_id>` are deterministic [read]. |
| A2 | C1/C9 | Partial | `Trainer._train_loop`, `wandb_init.init_wandb` | The W&B id is random on first start. On resume it is read only from `checkpoint-{step}/metadata.json`, so it breaks whenever the resume source is not a scheduled checkpoint dir. `wandb.init` failure is fatal. The image ships wandb 0.17; pods pip-install 0.26 on every start [read]. |
| A3 | C4 | Partial | `CheckpointManager.save_checkpoint` | Staging dir plus `os.replace` already exists. Missing: fsync of files and dir; a pod-unique tmp name; and the re-save path, which runs `shutil.rmtree(final_dir)` and then `os.replace`, leaving a window where only the `.tmp` exists [read]. |
| A4 | C8 | **Bug** | `wave2_agent.main` (`resume_from_checkpoint = RESUME in {1,true}`), `wave2_launcher.render_job` (env list) | The launcher never sets `RESUME`, so a restarted wave2 pod starts from step 0 and overwrites its own anchors [read]. I could not find in the repo how earlier w3 restarts resumed; possibly the env was added by hand when Jobs were recreated [verify]. |
| A5 | C5 | **Bug** | `CheckpointManager.load_checkpoint` | `glob("checkpoint-*")` plus `re.search(r'checkpoint-(\d+)')` accepts `checkpoint-N.tmp`. A `.tmp` dir is chosen as soon as it holds `training_state.pt`, which is mid-save (state is written before `metadata.json`) [read]. A truncated `torch.save` then fails in `torch.load`, the run is marked FAILED and exits 1, and every retry makes the same choice until `backoffLimit` is used up [inferred chain]. There is no size or validity check, no fallback and no test (`TestRollingResume` only checks that the pruner skips staging dirs). |
| A6 | C6 | **Bug** | `TrainingLoop.run` | The off-by-one fix `5f2d3e78` is only on the local branch `fix/resume-off-by-one`. It is not on `origin`, `main` or this branch [git]. Each resume shifts later labels by one window (see the memory note `project_resume_label_lag`). |
| A7 | C6 | OK | `loop.run` reseed; `MaskedLMDataCollator._next_mask_generator`; load with `mmap=True` | Data order is a function of (seed, epoch); masks are a function of (seed, epoch, global batch); optimizer, scheduler, RNG, AMP, tokens and offsets are all restored [read]. |
| A8 | C7 | Gap | `wave2_agent.pick_resume_anchors`; `loop.run` rolling block | Resume state exists only at scheduled anchors: 4 permanent ones, plus a rolling one at the first anchor at least `steps_per_epoch` after the last. That is about one epoch (see §2). There is no time-based cadence and no emergency save [read]. |
| A9 | C8 | **Bug** | `wave2_launcher.POD_SCRIPT`, `k8s/bert_v2/job-sweep-bert-v2.yaml`, `render_opp_probe.SCRIPT`, `wave2_agent` (`subprocess.call`), `cli.run` | See the A9 notes below. |
| A10 | C9 | Partial | `registry.register_run_end`, `cli._safe_register_run_end` | The PREEMPTED state exists but a killed pod stays RUNNING with a stale heartbeat, and no reaper exists. `register_run_start` overwrites `started_at`. `checkpoint_paths` uses `glob("checkpoint-*")`, which includes `.tmp` dirs [read]. |
| A11 | C2 | Partial | k8s Job; scratchpad `first_wins.sh` | See the A11 notes below. |
| A12 | C11 | Partial | `podFailurePolicy` in the launcher and renderers | DisruptionTarget and exit codes 2/128 are ignored. A deterministic resume failure (corrupt state, wandb 0.17) still burns 20–61 retries silently, as it did on 2026-10-16..19 [read, memory]. |
| A13 | C12 | Partial | `loop._HEARTBEAT_INTERVAL_S` (5 min, registry); `opp_probe` (60 s S3, probe only); `fleet_watch` | No per-worker S3 heartbeat in production; no counts of preemptions or lost steps [read]. |
| A14 | C1/C7 (trials) | Gap | `rater_sweep.apply_rater_overrides`, `sweep_agent_lm._run_rater_trial` | See the A14 notes below. |
| A15 | dispatch (trials) | Gap | `job-sweep-bert-v2.yaml` | See the A15 notes below. |
| A16 | evals | OK | `eval_v2_cell`, `evaluation/cache.py` | Content-hash markers (tmp + replace), stale-marker self-heal, content-addressed S3 upload with collision check, input errors exit 3, `find_checkpoints_sorted` ignores `.tmp`, `--partial` uses a settled view [read]. |
| A17 | evals on preemptible | Gap | `CachedRunner.run_once` | All results are held in memory; parquet and markers are written once at the end, and the S3 upload comes after that. A pod killed mid-cell banks nothing, so a 23-min gpt2_large cell on a 20–37 min A100 can livelock [read]. |
| A18 | C10 | **Bug** | `DataProcessor.preprocess_data` / `_save_chunked_dataset`; `tokenize_dataset` | See the A18 notes below. |
| A19 | CPU queue | OK, with a race | `k8s/rater_corpus/job-annotate-wq-*.yaml` | mkdir lock, heartbeat file and stale reclaim via `mv`; restart-safe per shard (`ANNOTATION_MANIFEST.json`). Two reclaimers can both "win": A does `mv` then `mkdir`, then B's `mv` moves A's fresh lock [inferred]. |
| A20 | idle | Gap | `render_opp_probe.SCRIPT` | One process per card. If one exits early, its card idles while the other runs [read]. |
| A21 | storage | Note | PVC `thomas-rater-corpus` | 8 TiB [kubectl]. At ~140 GB per run, tranche 1 (~4.5 TB) fits; the full 140-cell wave (~19.5 TB) does not [computed]. |

**A9 notes (signal path):**
- `terminationGracePeriodSeconds` is the default 30 s on every Job [kubectl].
- The pod script runs `python3 scripts/wave2_agent.py || true` under `bash -c`, which is PID 1, with no trap. PID 1 drops SIGTERM when it has no handler, so nothing is forwarded [inferred from PID-namespace semantics; verify in K1].
- `wave2_agent` launches the trainer with `subprocess.call`, which does not forward signals, and neither `cli`, the trainer nor the loop installs a handler.
- Hazard: `cli.run` maps KeyboardInterrupt to PREEMPTED and `typer.Exit(0)`. Exit 0 makes `wave2_agent` write `/tmp/run_succeeded`, which completes the Job index for an unfinished run [read].

**A11 notes (exclusive ownership):**
- Inside one Job, `podReplacementPolicy: Failed` already prevents a replacement from overlapping the old pod [kubectl].
- There is no run-dir lock across Jobs.
- `first_wins.sh` polls every 60 s, so two pods can run the same item for up to a minute. That is harmless for sweep agents, which pull different trials, but destructive for wave2 runs, which share a directory [read + inferred].

**A14 notes (trials):**
- Rater mode forces `resume_from_checkpoint: false` and disables resume states and rolling resume.
- Trial state (`EarlyStopState`, `gated_out`, last eval) lives only in memory.
- The output dir is keyed on the W&B run id, which is a good base for resume [read].

**A15 notes (sweep dispatch):**
- Each pod runs `wandb agent --count 1`, and W&B grid sweeps do not re-issue a crashed point.
- The replacement pod then gets no trial, the sentinel is absent, it exits 1, and the Job retries until `backoffLimit` 10 [read + inferred].
- Live right now: `thomas-bert-v2-race-en-long-h100` (2× H100 opp2, 10K steps ≈ 5.2 h each) and `thomas-bert-v2-rater-race-base-h100` run this way [kubectl].

**A18 notes (cache builds):**
- `preprocess_data` counts a bare existing chunked dir as done.
- The chunked dataset is saved in place: no tmp dir and rename.
- Tokenization skips on `train/dataset_info.json` even if `test/` was never written; train is saved before test.
- Nothing stops two builders of the same key; rand100 cells alias one corpus [read].
- The HF `save_to_disk` write order is assumed (dataset_info.json last) [inferred; verify against datasets 2.19.2].

**Side finding** [read; pre-existing, not specific to preemption]: `loop.run` never zeroes gradients at an epoch boundary.
- With 33,616 micro-batches per epoch and 1,050 × 32 consumed by optimizer steps, about 16 leftover micro-batches add their gradients into the first optimizer step of the next epoch.
- Clipping (`max_grad_norm` 1.0) bounds the effect.
- Resume stays consistent, because saves only happen at step boundaries.

## 5. Checkpointing on fast, preemptible cards

### 5.1 Keep analysis anchors and resume states apart
- **Anchors** are the `checkpoint-N` dirs on the token-anchored schedule (~93 for BERT-v2). They stay unchanged:
  - weights-only, except the permanent ep1/ep2/midpoint/final states;
  - the label is the pre-increment step, so `checkpoint-0` holds one optimizer window.
  - Changing that would break comparisons in the middle of the study.
- **Resume states** go to `<output_dir>/_resume/step-<G>/`, containing:
  - `model.safetensors`, `training_state.pt`, `metadata.json` and `MANIFEST.json`;
  - G is the number of completed optimizer steps (post-increment, saved where `on_step_end` runs now).
  - This path matches none of the eval, registry or pruner patterns (`checkpoint-(\d+)$`).
- `training_state.pt` gains `format: 2` and `label_semantics: pre_increment | completed`. The loader computes the start step from that field (the `5f2d3e78` S+1 rule applies only to `pre_increment` states), so it no longer has to infer it from the offset.
- Opportunistic pools stop using `rolling_resume_every_steps`. It keeps working for runs already in flight.

### 5.2 Cadence: wall-clock, not steps
Preemptions arrive per hour, not per step. The same one-epoch cadence is 33 min on H100 and 2.7 h on A10.

- **Rule:** interval = clamp(sqrt(2 · T_save · L_pool), 10 min, 60 min) (Young/Daly).
  - T_save is a running average of measured save times.
  - L_pool is the pool's lifetime, taken from the heartbeats.
- **Defaults until measured:** 20 min on opportunistic pools, 60 min on armada-default.
- **Overrides:** env `ROLLING_RESUME_SECONDS`, or config `training.rolling_resume_every_seconds` with `exclude=True` so the config hash doesn't change.
- Because the emergency save covers ordinary preemptions, the periodic save mainly covers hard kills: node loss, SIGKILL, CephFS hangs. It doesn't need to be aggressive.

**Cost on one H100 run** [computed]: 16.4 h; 4.1 GB per resume save; anchors add 93 × 1.36 GB ≈ 126 GB in either case.

| Cadence | Saves | Written | Stall at 66 MB/s | Stall at 200 MB/s | Worst loss on a hard kill |
|---|---|---|---|---|---|
| per epoch (today) | 30 | 123 GB | 31 min (3.1%) | 10 min (1.0%) | 33 min |
| 20 min | 49 | 200 GB | 50 min (5.1%) | 17 min (1.7%) | 20 min |
| 30 min | 33 | 135 GB | 34 min (3.4%) | 11 min (1.1%) | 30 min |

**CephFS write speed has not been measured.** What we know:
- reads run at 66 MB/s on a single stream and 345–400 MB/s with 2 or more streams;
- one degraded episode ran at 17 MB/s (memory notes `reference_eval_wave_ops`, `reference_eval_storage_deps`).

If T_save is above 40 s:
- write the weights and the training state on parallel threads, or shard the optimizer state into 4 files [inferred to scale like reads];
- later (P5), snapshot the state into spare VRAM on 80 GB cards and write it from a background thread.

### 5.3 Write protocol (C4)
1. Fencing check: the lease is still ours (§7).
2. Write into `<final>.tmp-<pod>-<pid>`.
3. fsync each file, then the dir.
4. Write `MANIFEST.json` last, with:
   - format, step and `label_semantics`;
   - `{file: size}`;
   - pod, node, `written_at` and `save_seconds`.
   - Then fsync it.
5. Fence again and rename into place.
   - For a re-save, rename the old `final` to `<final>.old-<pod>`, rename the tmp dir to `final`, then remove the old one.
   - At no point is there no complete copy.
6. fsync the parent dir, then prune `_resume/` down to the newest 2.

Why fsync: the zero-filled weights seen on 2026-10-15 [read, `eval_v2_cell._settled_checkpoint_view`] fit data that had not yet been flushed to the storage nodes when another client read it, or when the writer's client was evicted [inferred].

### 5.4 Read protocol (C5)
- **Candidates:**
  - `checkpoint-(\d+)` (full-name match) that hold `training_state.pt`;
  - `_resume/step-(\d+)` that hold a `MANIFEST.json`;
  - skip anything named `.tmp*`, `.old*` or `.corrupt*`.
- **Order:** by effective resume step, after converting labels.
- **Validate:**
  - a manifest exists (legacy dirs need `metadata.json` plus a zip that opens);
  - file sizes match;
  - the load succeeds;
  - a few large weight tensors are not all zero.
- **On failure:**
  - rename the dir to `<dir>.corrupt-<ts>`;
  - log it, note it in the registry and the heartbeat, and raise an alert;
  - try the next candidate.
- Never silently start fresh when a state beyond step 0 existed; same guard as `production_agent`'s RESUME logic.

### 5.5 Emergency save on SIGTERM (C7/C8)

**Signal path:**
- **Pod scripts:** a bash `trap` forwards TERM to the child, waits for it, and exits 143.
- **`wave2_agent` and the queue worker:** run the trainer with `Popen` and forward SIGTERM.
- **`cli.run`:** install a handler that only sets a flag and a timestamp.
- **`TrainingLoop`:** check the flag right after each `self.global_step += 1`, the only point where gradients are clean.
  - If the last resume save is less than 2 min old, skip the save.
  - Otherwise, if (`TERM_GRACE_SECONDS` − elapsed − margin) is more than 1.5 × T_save, write `_resume/step-G` (fenced).
  - In every case, then raise `Preempted`. `cli` records PREEMPTED (step, saved step) and exits **143**, never 0.
- **Edge cases:**
  - SIGTERM during setup: exit 143 at once.
  - SIGTERM during a scheduled save: finish that save. It counts as the emergency save if it carries state.
- **`podFailurePolicy`:** keep the DisruptionTarget rule and add exit code 143 to the ignore list. A plain `kubectl delete pod` probably sets no DisruptionTarget [inferred; verify].

**Feasibility:**
- Reaching a step boundary takes at most ~2 s on H100 (9 s on A10).
- Writing 4.1 GB at 66–200 MB/s, with fsync, takes 20–62 s.
- **`terminationGracePeriodSeconds: 120`** gives about 2× margin even at 66 MB/s.
- Stay at or below 180 s. A preemptor waits out its victim's grace period [inferred from the Kubernetes docs], so a long grace delays the owners.
- A degraded CephFS (17 MB/s means ~4 min) won't fit. The deadline check then skips the save, and the previous `_resume` state is used.
- **Measure in K2:** record `sigterm_at` in the heartbeat at once and `emergency_saved_at` afterwards. That shows how long NRP actually waits before SIGKILL.

### 5.6 Retention
- `_resume/`: keep the newest 2; the older one is the fallback if the newest proves corrupt. Delete `_resume/` at COMPLETE, since the endpoint carries full state.
- Permanent anchors are unchanged (`pick_resume_anchors`).
- Cost: about +8.2 GB per in-flight run; 30 concurrent runs need ~250 GB.

### 5.7 Restart overhead budget (R)

| Phase | Warm node | Cold node |
|---|---|---|
| Image pull (~12 GB) | seconds | 4–10 min (CLAUDE.md, sweep lessons) |
| Clone + `pip install wandb==0.26.0` | 30–60 s | same |
| Model build + resume load (4.1 GB at 66–345 MB/s) | 12–60 s | same |
| Fast-forward of up to 33.6K micro-batches via `next(data_iter)` | not measured; est. 1–3 min [inferred] | same |

Ways to shrink it:
- bake wandb 0.26 into the image;
- if fast-forward takes more than 2 min, skip at the sampler level (masks are already index-keyed, so only the collator's batch counter needs an offset).

Rough A100 economics: a 30-min life with a cold pull leaves 15–20 min of training. That is about 50–60% efficiency at 2.3× A10 speed [inferred].

## 6. Short trials (races and sweeps)

| Option | Mechanism | Verdict |
|---|---|---|
| A. Re-issue only | Re-run a lost trial from scratch | Only for trials under ~30 min. With W&B grids, lost points vanish silently. |
| B. W&B requeue + resume | The SIGTERM handler calls `run.mark_preempting()` and exits non-zero; W&B requeues the run and an agent resumes the same run id; the trial resumes from `models/{kind}_sweeps/{cond}/{run.id}/_resume` | For the remaining Bayes sweeps. Requeue behaviour comes from W&B docs [verify with 0.26 and our server]. Hard kills are not requeued. |
| **C. Deterministic trial ledger** | Expand the grid into items with `trial_id = f"{sweep}-{blake2b(sorted params)[:8]}"`; W&B id = trial_id, group = sweep name; dispatch through the queue (§7), or in the interim as an Indexed Job where each index is one trial | **Recommended for grids and races.** Exact, resumable, independent of the W&B controller, same machinery as production. |

Changes needed:
- `apply_rater_overrides`: turn resume on and take the time-based cadence from the pool. The endpoint stays weights-only.
- `_run_rater_trial`: on every resume save, write `trial_state.json` (the `EarlyStopState` fields, `gated_out`, `last_eval_step`, eval history) through an `extra_state` hook, and restore it before `train()`.
  - Evals fire at fixed steps (`is_eval_step`) and the held-out chunks are deterministic, so the eval cadence survives a resume.
- Delete `_resume/` when the trial ends; the sentinel and registry write stay as they are.
- `render_bert_v2_job.py` gains a `--ledger` mode that does not use `wandb agent`.

## 7. Dispatch: push vs. pull

| Criterion | Push, one Job per item (today) | Push racing across pools (`first_wins.sh`) | **Pull work queue** |
|---|---|---|---|
| First free GPU wins | No; pool fixed at render time | At start only | Yes, at every claim |
| After a preemption | Only that pool's replacement pod | Losing candidates are already gone | Any free worker in any pool |
| Oversubscription risk | None within a Job | Up to 60 s of double writers | None (lease plus fencing) |
| Idle risk | Low | Losers start and exit | Workers exit when nothing is claimable |
| Control during an NRP block (Job patches denied) | None | Needs the laptop watcher | S3 control file (drain, per-pool caps) |
| Complexity | Low | Medium, laptop-bound | Medium–high, new code |

**Recommendation:** use a pull queue for training and trials, and keep push for evals and CPU jobs. The push model can't keep the tranche balanced. With a static split, the 24 GB share of runs takes at least 3.4 days each while the H100s run out of work. With a queue:
- 6 H100 workers give ~9 runs/day and 8 × 24 GB workers give ~2.4 runs/day [computed];
- 32 runs take about 3 days, if the H100s stay available.

### Queue design
**Layout on the run's PVC** (null-subjects for BERT-v2; cross-namespace data moves only through S3):

```
/mnt/data/queue/<queue>/items/<id>.json            # immutable spec: env, priority, min_vram, pools, phys_batch
/mnt/data/queue/<queue>/leases/<id>/gen-000001     # O_EXCL-created; mtime = heartbeat
/mnt/data/queue/<queue>/done/<id>.json   parked/<id>.json   attempts/<id>.jsonl
```

- The state is mirrored to S3 at `ops/queue/<queue>/` for the report.
- The control file is `s3://…/queue_control/<queue>.json`, modelled on `wave_control/`.

**Lease protocol.** It hardens the annotation queue's mkdir lock, and relies on CephFS exclusive create being atomic across clients [inferred; the annotation queue already depends on this in production].
- **Fresh claim:** exclusive-create (`O_EXCL`) `gen-000001`.
- **Heartbeat:** update the file's mtime every 60 s, plus an S3 heartbeat.
- **Takeover:** if the newest generation's mtime is older than the TTL (10 min, longer than the 120 s grace) and the item isn't done, exclusive-create gen+1. Exactly one claimant wins.
- **Slot takeover:** a lease whose slot (Job name plus completion index) matches mine but has a different pod UID can be taken at once, because `podReplacementPolicy: Failed` guarantees the old pod is gone.
- **Fencing:** before every checkpoint rename and before registry COMPLETE, check that the newest generation is mine. If not, stop without renaming and exit 75. This also covers CephFS D-state hangs that outlast the TTL.

**Claim order:**
1. items with a stale lease (finish what was started);
2. then by priority;
3. fast pools take the longest remaining work, slow pools the shortest.

**Slow-pool start rule:** a 24 GB worker claims a *fresh* 30-epoch item only while unclaimed items exceed 2 × the number of fast workers. Otherwise it exits. This bounds the 3.4-day tail without a yield protocol (the yield protocol is in P5).

**Worker exit codes:**
- 0 only when nothing is claimable;
- 143 on SIGTERM;
- 75 on a lost lease;
- on an item failure: record the attempt, and park the item after 3 attempts with no step progress (C11).

**Worker Jobs, one per pool:**
- Indexed, with `completions = parallelism = N`.
- `podFailurePolicy` ignores DisruptionTarget and exit codes 2/128/143; `backoffLimit: 30`.
- `terminationGracePeriodSeconds: 120`.
- 1 GPU, 2 CPU, 6 Gi with limit = request (BERT-v2 measured at 2.8 GiB).

| Pool (BERT-v2, null-subjects) | Key / class | Workers |
|---|---|---|
| H100 | `nvidia.com/h100`, `opportunistic2` | 6 |
| A100 | `nvidia.com/a100`, `opportunistic2` | 0 (2 during the canary) |
| 48 GB | L40/L40S on `nvidia.com/gpu` + `nvidia.com/rtxa6000`, armada-default | 4 |
| 24 GB | 3090/A10/L4/4090, armada-default | 8 |

That is 18 GPUs or fewer, well under the 50-GPU threshold for announcing in Matrix.

**Registry state machine:**
- QUEUED → RUNNING (attempt n, `lease_gen`) → PREEMPTED, or LOST (reaper: heartbeat older than the TTL) → RUNNING → COMPLETE | FAILED | PARKED.
- Writers stamp `lease_gen`; the read-modify-write drops writes from an older generation (best effort).

**Interim, before the queue exists:** keep push with explicit item lists per pool, `RESUME=1`, and the same lease file used as a run-dir lock, so racing launches cannot double-write.

## 8. Multi-GPU opportunistic pods
- **Default to 1 card per pod.**
  - A 2-card pod loses 2 runs per preemption.
  - It is still a single victim, so the fewest-victims rule doesn't protect it.
  - It needs a node with 2 free cards (the midday success is one sample).
  - Each card's process finishes at a different time, which leaves an idle tail (A20).
- **N-card pods only to grab a node with N free cards, while the queue holds at least 2N items.** Run one worker loop per card (`CUDA_VISIBLE_DEVICES=i`) and request 2N CPU and 6N Gi.
- **No DDP for study runs.** The trainer is single-GPU, and DDP would change the data partition, mask indexing and numerics within a cell. It would also complicate resuming at a different world size and multiply the loss per preemption by N.
- Revisit DDP only for the custom raters (5–10B words, ~6 days on one GPU), and only if 4 or more H100s are free for days. Today's snapshot makes that unlikely.

## 9. Availability and time-of-day strategy
- **Keep pending workers standing.** Pending pods hold no resources and the utilization bands apply to running pods [inferred from the rule text]. The queue keeps workers busy whenever cards free up, so nobody has to dispatch by hand at the right hour.
- **Seed long items first,** so H100 windows go to long runs.
- **Use `opportunistic2` everywhere.** Other opportunistic pods can't preempt it, and it can preempt `opportunistic` pods.
- **Other pools:**
  - H200: at most 1 pending worker.
  - GH200: never (arm64).
  - A100: canary first; for real work only once K2 shows banked efficiency above 50%.
- **Build the data:** a week of per-worker heartbeats gives hour-of-week tables of wait (created → started) and life (started → DisruptionTarget) per key and class. Today we have one sample each (10–11 h overnight wait, 1 min at midday).
- **New `nrp_avail.py opp` subcommand:** per key, show allocatable, running by priority class, pending by class, and ours. `kube_pod_info` exposes `priority_class` on the public Prometheus [prom, verified 2026-10-21].
- **Etiquette:** at 18:30 UTC we held 5 of the 8 `opportunistic2` H100 slots. Cap H100 workers at 6.

## 10. Monitoring and alerting
- **S3 heartbeat every 60 s** at `ops/heartbeats/<queue>/<pod>[-g<i>].json`, with:
  - pod, node, GPU, slot, item and `lease_gen`;
  - phase: startup, loading, fastforward, training, saving, emergency or exiting;
  - step and tokens/s;
  - `last_resume_step`, `last_resume_at` and `save_seconds_ema`;
  - `sigterm_at` and `exit_reason`.
- **Registry additions:**
  - `preemptions` and `lost_steps`;
  - `gpu_seconds` by product;
  - `attempts[]`: pod, node, GPU, start and end step, end reason;
  - `first_started_at`.
- **`scripts/opp_report.py`**, generalized from `opp_probe_report.py`:
  - per item: state, progress, ETA, holder;
  - per pool: waits, lives, preemptions per hour, banked steps per GPU-hour, and efficiency = banked / (banked + lost + overhead);
  - queue depth.
- **`fleet_watch.py` checks:**
  - `crash_loop`: 3 or more attempts with no progress;
  - `livelock`: resume step unchanged across 3 attempts, or pool efficiency under 30% over 6 h;
  - `stale_lease`: no heartbeat for longer than the TTL and no takeover within 30 min;
  - `corrupt_checkpoint`, `emergency_save_failed`;
  - plus the existing `backoff` check and band violators from `nrp_avail ours`.
- **An exempt-size CronJob** (≤1 CPU, ≤2 GB, every 30 min, exits when done) writes the report to S3, so monitoring doesn't depend on a laptop session.

## 11. NRP compliance checklist
- **No idle GPUs:**
  - workers exit when nothing is claimable;
  - 1-card pods by default;
  - drain through the S3 control file, not Job patches (patches are denied during a block).
- **GPU above 40%:** startup, resume, fast-forward and saves are 0% windows. Measure the 15-min DCGM mean in K2.
- **CPU 20–200% and RAM 20–150% of request:**
  - BERT-v2 uses 2.8 GiB of 6 Gi (47%);
  - check working set during emergency saves (page cache) [verify].
- **Limit = request** everywhere (already true).
- **Exempt helpers:** report, queue seeding and df probe at ≤1 CPU and ≤2 GB.
- **No force deletes:** stop workers by draining or with a normal `kubectl delete job`.
- **Over 50 GPUs:** announce in Matrix.
- **Grace period:** 180 s or less.
- **Taints:** tolerate only `nvidia.com/gpu`, as today.
- **Opportunistic classes:** only for the a100/h100/h200 keys (the launchers already enforce this).

## 12. Phased implementation plan
Estimates are focused engineering time with an agent. "Exists" marks what is already in the code.

**P0: make preemption safe (blocker, ~1.5 days)**

| # | Change | Where | Exists | Effort |
|---|---|---|---|---|
| 0.1 | Resume by default (`RESUME=0` turns it off); the launcher also sets `RESUME=1`. Push before tranche 1 launches. | `wave2_agent.main`, `wave2_launcher.render_job` | new | 0.5 h |
| 0.2 | Cherry-pick `5f2d3e78` (needs Thomas's OK; small conflicts in `run()` and the smoke test) | `loop.py`, `test_training_loop.py` | on a local branch | 1 h |
| 0.3 | Loader: full-name match, skip staging dirs, require a sentinel, quarantine and fall back | `CheckpointManager.load_checkpoint` | new | 4 h |
| 0.4 | Writer: pod-unique tmp, fsync, manifest last, rename-aside re-save; fix the `.tmp` glob in the registry | `save_checkpoint`, `cli._safe_register_run_end` | partial | 3 h |
| 0.5 | Signal path: bash trap, `Popen` forwarding, SIGTERM flag in `cli`, PREEMPTED exits 143, `podFailurePolicy` ignores 143, grace 120 s with `TERM_GRACE_SECONDS` | `wave2_launcher`, `job-sweep-bert-v2.yaml`, `render_opp_probe`, `cli.run` | new | 3 h |
| 0.6 | W&B: write `<output_dir>/WANDB_ID` once, atomically, and resume from it; on init failure fall back to offline mode; wandb 0.26 in `requirements.txt` plus an image rebuild | `trainer._train_loop`, `wandb_init`, `requirements.txt` | partial | 2 h + build |
| 0.7 | Cache builds: tmp dir, rename and `_COMPLETE.json`; checkers require it; require `test/` when a test corpus is configured | `data.py`, `tokenize_dataset.py`, `wave2_agent` cache probe | partial | 3 h |

**P1: checkpointing built for preemption (~2.5 days)**

| # | Change | Exists | Effort |
|---|---|---|---|
| 1.1 | Time-based `_resume/` saves, format 2, loader merge, retention, delete at COMPLETE | new | 1 d |
| 1.2 | Emergency save (flag checked after the step increment, deadline, fencing hook) | new | 0.5 d |
| 1.3 | Instrumentation: `ckpt/save_seconds`, `resume/load_seconds`, `resume/fastforward_seconds`, startup phases | new | 0.5 d |
| 1.4 | Registry: PREEMPTED on SIGTERM, `attempts[]`, `lost_steps`, `first_started_at`; LOST reaper (inside the report) | partial | 0.5 d |

**P2: trials (~2 days)**
- Rater-trial resume with `trial_state` persistence.
- `scripts/trial_ledger.py` (grid to items) plus an Indexed-Job renderer variant.
- `mark_preempting` on the Bayes path.
- Retire `wandb agent` for grid sweeps.

**P3: work queue (~3.5 days)**
- `model_foundry/workqueue.py`: lease, generations, fencing, heartbeat, done and park.
- `scripts/opp_worker.py`: the `wave2_agent` config logic factored into a `build_run_config(item)` function, plus a per-card mode.
- `scripts/render_workers.py` and `scripts/wq_seed.py`, plus the S3 control file.
- Fencing wired into `CheckpointManager` and `cli`'s COMPLETE write.

**P4: monitoring and strategy (~1.5 days)**
- `opp_report.py`, the `fleet_watch` checks, `nrp_avail opp`, and the exempt CronJob.

**P5: optional (2–5 days)**
- Parallel or async checkpoint writes.
- Sampler-level fast-forward.
- Micro-batch policy (64 or more on cards with ≥48 GB, pinned per run).
- Yield protocol: a fast worker asks the slowest holder to save and release.
- Incremental eval commits: every K checkpoints, flush parquet, upload to S3, then write markers.
- Part-level resume for annotation.

**Tests to add:**
- `model_foundry/tests/unit/training/test_checkpointing.py`:
  - loader ignores `.tmp` dirs;
  - skips dirs without a sentinel;
  - quarantines a truncated `training_state.pt` and falls back;
  - a re-save never leaves a gap (fault injected between renames);
  - merges `_resume` and anchor candidates;
  - `_resume` keeps 2;
  - format-2 label semantics.
- `model_foundry/tests/smoke/test_training_loop.py`:
  - time-based save with a mocked clock;
  - the SIGTERM flag saves at a step boundary and raises;
  - emergency save skipped when the last save is recent;
  - resume from a format-2 state matches an uninterrupted run (reuse the `TestResumeMatchesUninterrupted` harness from `5f2d3e78`).
- `tests/test_workqueue.py`:
  - claims are exclusive under N processes;
  - stale takeover has a single winner;
  - fencing rejects a rename after takeover;
  - slot takeover;
  - park after k attempts.
- `tests/test_sweep_agent_rater.py`: trial state survives a resume.
- Cache atomicity: no `_COMPLETE.json` means a miss; concurrent builders give one winner.
- Launcher render tests: `RESUME=1`, grace period, 143 in `podFailurePolicy`.
- Extend `scripts/preemption_smoke_agent.py` with a SIGTERM victim and a killed-during-save victim.

## 13. Canary plan (before scaling)
- **K0, local CPU:** the P0–P3 unit and smoke tests, plus a fault-injection matrix that kills a save after the weights, during `torch.save`, before the manifest, and between renames.
- **K1, armada-default, 1 pod:** the extended `preemption_smoke_agent`:
  - deterministic victim under SIGTERM and under SIGKILL mid-save;
  - bitwise-identical endpoint;
  - the trainer log shows SIGTERM was received (this checks the PID 1 behaviour).
- **K2, A100 `opportunistic2`, 2 workers, ~12 h:** expect about 20 real preemptions. Measure T_save, the time from SIGTERM to SIGKILL, R (warm and cold), banked progress per attempt and DCGM utilization.
  - Pass if: zero corrupt resumes; endpoint step equals `train_steps` (no label lag); one W&B run per item; PREEMPTED counts match the pods' DisruptionTarget records; no Job uses more than 10% of its `backoffLimit`.
- **K3, H100 `opportunistic2`:** 2 real tranche-1 runs end to end, each with 2 graceful `kubectl delete pod` calls (not forced).
- **K4, queue:** 4 items; 2 H100 and 2 × 24 GB workers; kill a worker; simulate a stale lease with a test hook.
  - Pass if: items migrate across pools; fencing logs show no double promotion; registry COMPLETE equals items done.
- **Scale up** only after K2–K4 pass and `nrp_avail ours` shows 0 band violators from our workers.

## 14. Decisions for Thomas
1. Merge `5f2d3e78`. It is required before running at scale on preemptible cards, and the BERT-v2 wave is fresh, so no mixed label semantics.
2. Set `terminationGracePeriodSeconds` to 120 (the trade-off is how long owners wait).
3. Use `opportunistic2` everywhere, which means we may preempt other users' `opportunistic` pods. NRP allows this.
4. Micro-batch: keep 32 everywhere (any card can resume any run), or use 64+ on ≥48 GB cards (~26% faster on H100) and pin it per run.
5. Is moving a run to different hardware mid-run acceptable? The D8 claim is statistical replication, and each attempt's GPU is recorded.
6. Replace the W&B sweep controller with the ledger for grid and race sweeps.
7. Expand the PVC before the full 140-cell wave.
8. Keep evals off the opportunistic tier; they have ~370 pre-Ampere GPUs that aren't preemptible.

## 15. Verified vs. inferred
- **Verified:**
  - A1–A21 as tagged;
  - `podReplacementPolicy: Failed` and grace 30 s on the live Job [kubectl];
  - Prometheus exposes `priority_class` [prom];
  - PVC is 8 TiB [kubectl];
  - `5f2d3e78` branch containment [git].
- **Inferred:**
  - bash as PID 1 drops SIGTERM;
  - Kubernetes honours the grace period on preemption;
  - CephFS exclusive create is atomic across clients;
  - the cause of the zero-filled weights;
  - fast-forward cost;
  - write-speed scaling with parallel streams;
  - pending pods are not band violators;
  - A100 efficiency;
  - the HF `save_to_disk` write order.
- **Verify before relying on it:**
  - how past w3 restarts got `RESUME`;
  - W&B `mark_preempting` requeue with 0.26;
  - CephFS write speed from the H100 nodes (`exp-19-*`);
  - working set during saves;
  - DisruptionTarget on a manual pod delete.

## 16. Critical files for implementation
- model_foundry/training/checkpointing.py
- model_foundry/training/loop.py
- scripts/wave2_agent.py
- scripts/wave2_launcher.py
- scripts/sweep_agent_lm.py (with scripts/rater_sweep.py and model_foundry/cli.py)
