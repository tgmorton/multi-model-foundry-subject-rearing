# Rater HP sweeps

W&B Bayes sweeps of the masked-LM **rater** models: one sweep per option-B
rater corpus (`base` plus the four interventions), each on a small fixed
shard subset, to choose HPs before the full rater runs
(memory/project_custom_rater_plan.md). Added 2026-10-16 on branch
`feat/rater-sweeps`. Nothing here changes the study sweeps or training.

## Pieces

| File | Role |
|---|---|
| `model_foundry/training/loop.py` | `on_step_end(global_step) -> bool` hook and `endpoint_checkpoint` = `full` (default) / `weights` / `none`. Both inert at their defaults. |
| `scripts/rater_sweep.py` | Rater-mode logic (pure): settings, config overrides, plateau early stop, `render-config` CLI. |
| `scripts/sweep_agent_lm.py` | Dispatches to rater mode (`_run_rater_trial`) when it is switched on; otherwise unchanged. |
| `scripts/rater_corpus/build_sweep_subset.py` | Flat symlink corpus dirs for a shard subset, same shard ids in every corpus. |
| `configs/sweeps/baselines/rater_bert_large_en.yaml` | Baseline: BERT-large, seq 512, `rater_sweep:` block with the defaults below. |
| `configs/sweeps/rater_bert_large_<corpus>.yaml` | One W&B sweep spec per corpus. |
| `k8s/rater_sweeps/job-precache-rater-subset.yaml` | CPU Job: build the subset dir, then tokenize + chunk it. |
| `k8s/rater_sweeps/job-sweep-rater.yaml` | A100 Job: one trial per pod. |
| `scripts/render_rater_sweep_job.py` | Renders either Job for one corpus (name, env, bad-node list from `configs/bad_nodes.txt`). |

**Switch.** Rater mode is on when the baseline YAML has
`rater_sweep: {enabled: true}` or the pod has `RATER_SWEEP=1`. Then
`CONDITION` names the corpus and `CORPUS_DIR` points training, source and
test corpus at its subset dir. Without the switch, `CONDITION` and
`CORPUS_DIR` are ignored and the agent runs the epoch path as before.
`tests/test_rater_sweep_mode.py` checks that no existing baseline turns it on.

## Launch

Prerequisites: the branch is pushed (the pods clone `feat/rater-sweeps`;
pass `--branch main` to the renderer once it is merged); the project
WordPiece tokenizer is at `/mnt/data/tokenizers/en_bert_wordpiece/` (the
study BERT sweeps use it); you are logged in to W&B locally.

1. **Subset + caches, per corpus.** Start with `base` and check its memory
   before the others.

   ```bash
   python scripts/render_rater_sweep_job.py precache --corpus base > /tmp/precache-base.yaml
   kubectl apply -f /tmp/precache-base.yaml
   kubectl logs -f job/thomas-rater-precache-base     # ends "RATER PRECACHE OK"
   ```

   Step 0 of the Job builds `/mnt/data/rater_corpus/sweep_subset_v1/<corpus>/`
   (train shards `0 1 2 3 4 5 6 22`, held-out shard `199`, first 50,000
   lines). It is idempotent, and `SUBSET_SPEC.json` pins that selection for
   all five corpora. Until an intervention's shards are ablated
   (`ABLATION_MANIFEST.json` present), its Job exits 3 and fails without
   retrying; apply it again later. The builder also runs standalone in any
   pod with `/mnt/data`:

   ```bash
   python scripts/rater_corpus/build_sweep_subset.py \
       --corpora base --train-shards 0 1 2 3 4 5 6 22 --heldout-shard 199
   ```

2. **Create the sweep.**

   ```bash
   wandb sweep --entity thmorton-uc-san-diego configs/sweeps/rater_bert_large_base.yaml
   # -> thmorton-uc-san-diego/subject-drop-rater-sweeps/<id>
   ```

3. **Start the agents.**

   ```bash
   python scripts/render_rater_sweep_job.py sweep --corpus base \
       --sweep-id thmorton-uc-san-diego/subject-drop-rater-sweeps/<id> > /tmp/sweep-base.yaml
   kubectl apply -f /tmp/sweep-base.yaml
   ```

   `--parallelism` / `--completions` override the template's 3 / 20. For
   the first sweep, consider `--parallelism 1` until one trial has been
   measured (step 4).

4. **First-trial checks.** GPU memory headroom (raise `data.batch_size`
   from 32 to 64 if there is room), host RSS against the 12Gi request,
   GPU utilization above 40%, seconds per step. Correct the template
   requests before the other corpora (NRP flags RAM outside 20-150% and CPU
   outside 20-200% of request).

5. **Monitor and read results.** W&B project `subject-drop-rater-sweeps`,
   group `rater-sweep-<corpus>`, runs named `rater_<corpus>_lr…_eb…_d…`.
   Rank finished runs by summary `proxy/min_held_out_perplexity`.
   `scripts/extract_sweep_winners.py` hard-codes the study sweeps, so for
   now point a copy of its logic at the rater sweep ids.
   `kubectl get jobs -l owner=thomas,stage=rater-sweep` shows the agents.

## What a trial does

- Trains `sweep_train_steps` optimizer steps. `train_steps` equals
  `scheduler_train_steps`, so warmup and decay finish inside the trial.
  `epochs` is sized from the chunked-cache row count so the loop never
  ends early.
- Every `eval_every_steps` it computes held-out MLM perplexity on the first
  `eval_max_sequences` 512-token rows of the tokenized `heldout.test`. The
  rows are packed once per trial. The mask seed is fixed (20260423), so
  every eval and every trial scores the same masked positions. Each eval
  logs `proxy/held_out_perplexity`, the running min
  `proxy/min_held_out_perplexity` (the Bayes metric) and
  `proxy/evals_without_improvement`, all with `step=`.
- **Plateau stop.** An eval counts as an improvement when it beats the
  reference perplexity by more than `min_rel_improvement` (relative); only
  an improvement moves the reference. After `patience_evals` evals in a row
  without one, the trial stops, but never before `min_steps_before_stop`.
  Gains accumulate against the reference, so a slow, steady improvement
  still resets patience every few evals. A non-finite eval counts as no
  improvement. A stopped trial **finishes normally**: sentinel written,
  registry COMPLETE, W&B state "finished", so Bayes learns from it.
- If the last eval did not land on the final step, the final model is
  scored once more.
- A trial whose minimum is non-finite fails (registry FAILED, raise), as in
  the epoch path.
- **One checkpoint.** Weights only, at the endpoint, under
  `/mnt/data/models/rater_sweeps/<corpus>/<wandb run id>/checkpoint-<step>/`
  (about 1.4 GB). There are no anchors, resume states, rolling resume or
  `final/` copy. `save_final: false` saves nothing.
- Registry: `run_kind=hp_sweep`, `condition=rater_<corpus>` (prefixed so
  it never collides with a study cell), `run_id=sweep-<sweep>-<run>`.

**No Hyperband.** The specs have no `early_terminate`. Hyperband with
`min_iter=1` killed winning trials in the study sweeps, and its SIGTERMed
trials show up as crashed, which drops them from the posterior (commit
38621840). The in-trial plateau stop replaces it.

## Defaults

| Knob | Default | Why |
|---|---|---|
| `sweep_train_steps` | 3000 | At effective batch 256 × 512 tokens this is about 390M tokens: ~1.5 passes over the 8-shard subset at ~1.3 WordPiece tokens/word, ~0.75 at batch 128. Roughly 1-3 A100-hours per trial (estimate). |
| `eval_every_steps` | 250 | 12 evals per full trial. Each is a forward pass over ≤1M tokens, under a minute on an A100. |
| `eval_max_sequences` | 2000 | About 150K masked tokens with a fixed mask: noise far below the HP differences being ranked. 50,000 held-out lines leave headroom (check `heldout.words` in `SUBSET_MANIFEST.json`). |
| `patience_evals` | 4 | 1000 steps without a real gain. |
| `min_rel_improvement` | 0.002 | 0.2%. Smaller differences are within run-to-run noise for ranking. |
| `min_steps_before_stop` | 1000 | Past the longest warmup (15% = 450 steps) and the post-warmup bump. |
| `save_final` | true | Keeps the winner's weights for a sanity eval. |
| Physical batch | 32 | Estimate for A100-80GB at seq 512 (~35-40 GB). Measure it. |
| Search space | lr 1e-5..1e-3 (log), warmup 0.02-0.15, dropout / attn dropout 0-0.3, eff. batch {128, 256}, β2 {0.95, 0.98, 0.99, 0.999} | The study BERT sweep's ranges plus RoBERTa's β2 = 0.98. |
| Agents | 3 parallel, 20 trials per sweep | Bayes goes stale beyond ~3-4 trials in flight. 5 × 3 = 15 of the lab's 18 A100s. About 14 h of wall time per sweep at 2 h/trial. |

## Tests

```bash
PY=.venv/bin/python
$PY -m pytest model_foundry/tests/smoke/test_training_loop.py   # TestStepEndHook, TestEndpointCheckpointMode
$PY -m pytest tests/test_rater_sweep_mode.py tests/test_sweep_agent_rater.py tests/test_rater_sweep_k8s.py
$PY -m pytest scripts/rater_corpus/tests/test_build_sweep_subset.py
```

A CPU run with a real `Trainer` (tiny BERT, fake 3-shard subset, real
`tokenize-dataset` / `preprocess-data`) evaluated every 10 steps, stopped on
plateau mid-epoch, and wrote only `checkpoint-<step>/` without
`training_state.pt`. That run is not committed.

## Open decisions (PI)

1. **Tokenizer and architecture.** The baseline uses the project's
   WordPiece tokenizer (lowercased, vocab 50,004) and BERT: a BERT-large
   trained from scratch. The alternative is the roberta-large recipe
   (byte-level BPE, cased, vocab 50,265, RoBERTa architecture), which
   matches the pretrained model behind the `robbi` selection arm (D14).
   That route needs code first: `config.py` has no `roberta` architecture
   and no BPE rater tokenizer has been trained. Changing the tokenizer
   changes every cache key, so all five subsets would need precaching again.
2. **Sequence length 512** (the study uses 1000). It is the BERT/RoBERTa
   pretraining length and covers the local context the rater scores. Should
   the full raters use the same length?
3. **Subset size and trial budget.** 8 train shards (~200M words), 3000
   steps and 20 trials per sweep. A short horizon tends to favour higher LR
   and lower dropout than a multi-billion-word run will want. Options:
   retrain the top-k at a longer budget, or rescale the LR. More shards
   (e.g. 16) would keep batch-256 trials under one pass of the data.
4. **A100s per sweep, and per-corpus sweeps at all.** 3 per sweep × 5 uses
   15 of the 18 lab A100s while the sweeps run, and competes with the rater
   trainings (1 A100 × ~6 days each). The study convention (HP search on
   baseline, frozen for the ablations) would mean sweeping `base` only and
   reusing its winner, at a fifth of the compute.
5. **Unmeasured resources.** Physical batch 32, sweep pod 4 CPU / 12Gi,
   precache pod 4 CPU / 12Gi. Measure them on `base` first.

## Known gaps

- `proxy/final_training_loss` is NaN for every sweep trial, in the study
  sweeps too. `_extract_final_training_loss` looks for loss attributes that
  `TrainingLoop` does not keep. This predates the rater work.
- There is no winner-extraction script for the rater sweeps yet (step 5).
