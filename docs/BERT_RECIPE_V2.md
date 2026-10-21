# BERT recipe v2: pre-LN, short sequences, gated LR race

**Status: adopted by Thomas 2026-10-19** ("document that somewhere and get started").
Decision D15 in [PLAN_BERT_CROSSED_SWEEPS.md](PLAN_BERT_CROSSED_SWEEPS.md).
Literature audit behind it: [reports/BERT pretraining recipe audit.md](../reports/BERT%20pretraining%20recipe%20audit.md)
(source notes in `research_notes/BERT pretraining recipe audit/`).

## Why: every from-scratch BERT we trained stalled on the MLM plateau

| Run | Recipe | Outcome |
|---|---|---|
| Production `bert_large-en-baseline-h0-s137` | post-LN 24L/1024, peak LR 3.75e-5, warmup 976 steps, linear decay, 256 × 1000 tokens/step, 14,790 steps (30 epochs, ~3.8B tokens) | train loss **5.62**; on held-out text masked loss 6.51 (unigram cross-entropy 7.42), **13.8% accuracy**, predicts "the" at 56% of masked positions |
| Rater probe-long (sweep `7j4ge0o6`) | same model, LR 1e-4, warmup 6%, seq 512, 256 × 512, 20k steps (2.6B tokens) | loss ~6.1 flat from step 4k to 20k; held-out ppl 1386 → 832 |

It is not a code bug. `scripts/diag/bert_mlm_diag.py` (job `thomas-diag-bert-mlm-v1`) found:
- the tokenizer is sane (0.03% `[UNK]`);
- our `MaskedLMDataCollator` gives bit-identical batches and losses to HF's
  `DataCollatorForLanguageModeling`;
- the trained checkpoints behave like frequency models that copy nearby tokens.

The cause is the learning-rate budget, and the HP sweep that chose it. Sweep `2pimhcbm` (39 trials, 6 epochs ≈ 3k steps each):

| Peak LR | Trials | Result |
|---|---|---|
| 1.0e-5 – 1.6e-5 | 18 | finished, held-out 5.90–5.97 nats |
| 2.0e-5 – 4.2e-5 | 8 (+2 crashed early) | finished, 5.82–5.87, improving with LR up to the edge |
| 4.2e-5 – 1.7e-4 | 1 | failed at step 0, so no data |
| 1.7e-4 – 7.1e-4 | 10 | 4 finished in a collapsed state (~6.39 nats); 6 crashed or failed |

- **The "winner" is the edge of the stable range,** and all stable trials sit on the plateau: the top five differ by 0.015 nats.
- **Healthy BERTs leave this plateau late.** Chen et al. 2024 put the drop at 20–30k steps at LR 1e-4 (12-layer, 256 × 512), which a 3k-step trial can never see.
- **The high-LR failures are short-warmup post-LN instability.** Those trials warmed up for ~100–400 steps; the original BERT used 10k (Xiong et al. 2020).
- **The same sweep design was fine for GPT-2,** because causal LMs have no long plateau.

## What is retired, and what is not re-run

**Retired as evidence:**
- every EN production `bert_large` run (5 conditions × 14 full + 81 two-epoch Stage-1 seeds, ≈5,400 GPU-h);
- their `null_subj_v2` evals;
- the `bert_large` sweep winners as an HP source;
- the BERT row of any cross-architecture contrast.

The ES and 10M BERT sweeps are suspect until checked.

**Not re-run for now (Thomas 2026-10-19).** The old-intervention BERT runs are deferred, and the work focuses on the sweeps ahead. The 10-run minimum (h0, 2 seeds × 5 conditions, ~600 GPU-h) stays available if needed later.

**Unaffected:**
- pretrained raters: robbi, BERT 250:1 (already retired, D14);
- the WordPiece tokenizer;
- the collator.

## Recipe v2

| Setting | (a) 90M-word study cells | (b) 5–10B-word rater |
|---|---|---|
| Normalization | pre-LN + final LayerNorm | same |
| Size | BERT-large shape (24L/1024/16/4096), kept size-matched to gpt2_large | same |
| Optimizer | AdamW β (0.9, 0.98), eps 1e-6, **no weight decay on LayerNorm/bias** | same |
| Weight decay | 0.1 (LTG-BERT) — race may check 0.01 | 0.01 |
| Peak LR | gated race over {3e-4, 6e-4, 1e-3} | race over {1e-3, 2e-3} (Izsak 2021; Wettig 2023) |
| Warmup / schedule | 6%, linear decay synced to the real 30-epoch budget | 6%, linear synced (or warmup-stable-decay) |
| Sequences | 128 tokens, `[SEP]` between corpus lines, `[CLS]` at chunk start | 128, then 512 for the last 10–20% (rater scores ±250-token windows) |
| Batch | 1024 × 128 = 131K tokens/step → ~29K steps per 3.8B tokens | 4096 × 128 = 524K tokens/step → ~23K steps per 12B tokens |
| Dropout | 0.1 | 0–0.1 |
| Masking | 15%, 80/10/10 | 15% at the end of training |
| `pad_token_id` | set to the tokenizer's real `[PAD]` (id 4), not bert-base's 0 | same |
| Precision | fp16 autocast + GradScaler (unchanged) | same |

Pre-LN is implemented with HF `MegatronBertForMaskedLM` (pre-LN BERT with a final LayerNorm, transformers 4.41.2). It is selected by config, so post-LN stays available.

## Selection procedure: gated LR race (replaces open Bayesian search for BERT)

1. **Pilot first.** Run one configuration (LR 6e-4) for 3 epochs on the EN 90M baseline and record its curve. It must leave the plateau before anything else launches. Its curve also calibrates the gate.
2. **Race.**
   - Study: 3–5 trials (LR × possibly two warmups), each 3 epochs (~360M tokens, ~2,750 steps). Each trial runs the production schedule shape (warmup = 6% of the 30-epoch step count, ~1,750 steps) truncated at 3 epochs, so trials see the warmup and peak LR the final run will use.
   - Rater: 2–4 LRs, ~1B tokens each on the base sweep subset.
3. **Escape gate.** Kill a trial whose held-out masked loss has not clearly left the 5.8–6.5-nat band the stalled runs occupied: initially, ≥ 5.5 nats at the end of epoch 1 (rater: ≥ 6.0 at 10% of trial tokens). This follows Izsak et al.'s rule of dropping configurations above validation loss 6.0 after 3 h. Thresholds are re-set from the pilot curve before the race.
4. **Rank survivors** on held-out masked loss *and* masked-token accuracy, logged alongside the share of top-1 predictions that are the single most frequent token.
5. **Horizon bias.** The best LR tends to fall as training gets longer (Bjorck et al. 2024). Among near-tied winners take the lower LR, and race well past the escape point.
6. **Freeze on baseline.** Tune on the baseline corpus and freeze HPs for all interventions (existing study rule). For the rater: race on `base`, check one intervention, and reuse if the best LR matches.
7. **Separate race per language** when the ES wave starts (existing study rule).

### How standard this is

| Piece | Standard? | Precedent |
|---|---|---|
| Pre-LN encoder | Standard for modern from-scratch pretraining (not the 2018 BERT) | Izsak 2021, Wettig 2023 ("essential for the high LR"), Cramming 2023, ModernBERT; NormFormer variants in BabyLM |
| 128-token sequences, longer at the end | Standard; the original BERT did it | BERT (128 for 90% of steps), LTG-BERT, BabyLM 2025 baseline, Cramming |
| Separators between documents when packing | Standard | RoBERTa, Cramming; packing without them cost BLiMP 79 → 63 (McGill BabyLM 2023) |
| Small LR grid, not a large Bayesian search | Standard for pretraining | RoBERTa, Izsak, Cramming |
| Tune on baseline, freeze for ablations | Standard; already our rule | — |
| Gate that prunes plateau-bound trials | Published, uncommon | Izsak 2021 (loss 6.0 after 3 h) |
| Rater recipe | Published, peer-reviewed for this size and budget | Izsak 2021 (EMNLP), Wettig 2023 (EACL) |

**Judgment calls we own:**
- choosing a 30-epoch LR from a 3-epoch race (mitigated as above);
- keeping BERT-large at 90M words (BabyLM submissions are all base-size or smaller; we keep large to stay size-matched with gpt2_large; describe it as "a BERT-large-sized pre-LN encoder, following Izsak et al. 2021");
- the gate threshold.

## Where it runs

- **Namespace `null-subjects`** (Thomas 2026-10-19). As of 2026-10-19 NRP still refuses Jobs and Pods there until the namespace is registered at https://nrp.ai/namespaces (PI, description, institution, publications, software).
- **A100 quota there is 0.** Rater work runs on 48 GB L40/L40S (`nvidia.com/gpu`) unless A100 quota is granted.
- **Cross-namespace data moves only through S3** (Thomas 2026-10-19), under `s3://thomas-subject-drop-artifacts/data_transfer/`. Writers run in lemn-lab against `subject-drop-archive`, readers in null-subjects into its own PVC. Caches are rebuilt fresh in null-subjects at 128 tokens; nothing content-addressed is copied.

## Cost (GPU-hours, one GPU per run)

| Item | Estimate |
|---|---|
| Pilot | ~10 |
| Study race (EN) | ~20–40 |
| Rater race (base subset) | ~15–25 (L40S) |
| Full raters (5) | ~2.5–6 days each on one GPU |
| BERT wave2, 140 pdrop2 cells (later) | ~9,000 (same as the gpt2_large wave) |
| Old-intervention minimum, if revived | ~600 |

## Steps

- [x] Diagnose; literature audit; this doc (2026-10-19)
- [x] **Code, branch `feat/bert-recipe-v2`** (7890a686, 65a02e1b):
  - pre-LN option (`model.bert.norm_position: pre`), real `pad_token_id`;
  - `data.packing: cls_sep` (in the cache key only when not contiguous; every old key unchanged);
  - `training.weight_decay` / `no_decay_norm_bias`;
  - trial settings `kind`, `scheduler_epochs`, `gates`; held-out masked accuracy and top-token share;
  - `rater_sweep.py cache-key`, used by every manifest.

  Tests pass locally and on the image's torch 2.5.1 / transformers 4.41.2. A CPU end-to-end smoke with the real tokenizer and CLI passed.
- [x] **Inputs staged to S3** `data_transfer/`: tokenizer, EN train_90M, test_10M, five rater sweep subsets. 74 objects, 5.9 GiB, `rclone check` clean (job `thomas-upload-xfer-v1`, lemn-lab).
- [ ] **null-subjects registration: blocked.** As of 2026-10-19 both Jobs and Pods are denied. Thomas or the namespace admin fills https://nrp.ai/namespaces.
- [x] **null-subjects registered 2026-10-20.** Download, then precache:
  - study cache `a1bdcf042521`: 1,075,685 chunks, ~1,050 steps per epoch;
  - rater cache `94e60395ef05`: 2,058,058 chunks.
- [x] **Pilot** (sweep `430xqpaf`, LR 6e-4, one A10, 9.2 s/step, 14K tokens/s) **left the plateau**:

  | step | tokens | held-out loss | accuracy | top-token share |
  |---|---|---|---|---|
  | 250 | 33M | 6.31 | 6.3% | 58% |
  | 500 | 66M | 6.12 | 7.4% | 29% |
  | 750 | 98M | 4.59 | 26.7% | 15% |
  | 1000 | 131M | 4.08 | 31.9% | 13% |
  | 1500 | 197M | 3.69 | 36.1% | 13% |

  For comparison, the old production BERT after 3.8B tokens: 6.51, 13.8%, 56%. Escape came as warmup carried LR past ~2e-4.
- [x] **Gates set from the pilot** (83bfd019): study `[[2000, 5.0]]`; rater `[[1000, 5.0]]`.
- [x] **Races launched 2026-10-20 ~22:37 UTC:**
  - study sweep `gesqanxo` (LR 3e-4, 1e-3; 6e-4 = the pilot), 24 GB pool;
  - rater sweep `ovkdk8zn` (LR 1e-3, 2e-3), L40/L40S.
- [x] **Study race done (2026-10-21).** Held-out masked loss / accuracy at 3000 steps of the 30-epoch schedule:

  | peak LR | escaped plateau | loss @3000 | acc @3000 |
  |---|---|---|---|
  | 3e-4 (`gesqanxo`) | steps 750-1000 | 3.272 | 41.1% |
  | **6e-4** (pilot `430xqpaf`) | steps 500-750 | 3.180 | 42.2% |
  | 1e-3 (`gesqanxo`) | steps 250-500 | 3.168 | 42.4% |

  Seed noise from the opportunistic probes is sd 0.0036 nats (5 seeds of the 6e-4 recipe). So 1e-3's 0.012 edge over 6e-4 is real but small at 3k steps, while 3e-4 is clearly behind.
  The pre-stated rule is "near-tie, take the lower LR", because the optimum falls on longer horizons. That points to 6e-4. An extended 6e-4 vs 1e-3 check is the open option.
- [x] Study LR frozen at 6e-4 (2026-10-21; see the decisions log).
- [~] **Rater race:** LR 1e-3 finished at loss 3.00 / acc 46.0% after 2000 steps (1.05B tokens, L40S, 27.8K tokens/s). The 2e-3 trial was still waiting for an L40/L40S after 18 h.
- [ ] Rater race (base): pick LR, then full raters.
- [ ] Later: BERT wave2; ES race.

## Launch sequence (null-subjects, once registered)

Run from a `feat/bert-recipe-v2` checkout. The pods clone that branch.

```bash
kubectl apply -f k8s/bert_v2/job-download-ns.yaml        # S3 -> PVC thomas-rater-corpus
kubectl apply -f k8s/bert_v2/job-precache-bert-v2.yaml   # 2 caches at 128 tokens, cls_sep
wandb sweep configs/sweeps/bert_v2/pilot_en.yaml         # -> <entity>/subject-drop-bert-v2/<id>
python scripts/render_bert_v2_job.py --name pilot-en --sweep-id <id> \
  --base-config configs/sweeps/baselines/bert_large_preln_en_race.yaml \
  --condition baseline | kubectl apply -f -
```

After the pilot:
1. Re-set `gates` in both race baselines from its held-out curve.
2. `wandb sweep configs/sweeps/bert_v2/race_en.yaml`, then render with `--name race-en --parallelism 3 --completions 3`.
3. Rater race: `wandb sweep configs/sweeps/bert_v2/rater_race_base.yaml`, then render with:
   - `--name rater-race-base --condition base`
   - `--corpus-dir /mnt/data/rater_corpus/sweep_subset_v1/base`
   - `--base-config configs/sweeps/baselines/rater_bert_large_preln_en_race.yaml`
   - `--gpus 48g --parallelism 2 --completions 2`

## Opportunistic tier (probe `opp_probe/1020`, submitted 2026-10-20 23:04 UTC)

| GPU | first wait | uninterrupted run | tokens/s (BERT-large, 128 tokens) |
|---|---|---|---|
| H100 80GB (both classes) | ~10-11 h (scheduled ~09:00-10:00 UTC) | 7-8 h and still running | 70-73K (~5x A10) |
| A100 SXM4 80GB (`opportunistic2`) | short | 20 min, then 37 min, then preempted | 32-34K (~2.3x A10) |
| H200 (both classes) | none in 18 h | - | - |

Report: `python scripts/opp_probe_report.py --tag 1020` (branch `feat/bert-recipe-v2`).

## Decisions log (2026-10-21, Thomas: "do what you need to do with bert, make sure to document your choices")

1. **Study peak LR is 6e-4.**
   - 6e-4 and 1e-3 differ by 0.012 nats at 3k steps (3.180 vs 3.168, seed sd 0.0036). That is about 3 sd: real, but tiny (0.4%).
   - The pre-stated rule takes the lower LR of a near-tie, because the optimum falls as the horizon grows. The production run is 10x the race horizon (31.5K vs 3K steps).
   - 3e-4 is clearly behind (3.272).
   - I did not run a longer 6e-4 vs 1e-3 check. The expected gain is below the seed noise of the downstream subject-drop metrics, and it would cost about 10 H100-hours.
   - Frozen in `configs/sweeps/baselines/bert_large_preln_en.yaml` and `data/sweep_winners/bert_large_preln_en.json` (branch `feat/bert-recipe-v2`, d6e7736a).
2. **The other study HPs follow the literature (not raced):**
   - AdamW (0.9, 0.98), eps 1e-6;
   - weight decay 0.1 (LTG-BERT), with LayerNorm and bias exempt;
   - dropout 0.1;
   - warmup 6% of the 30-epoch horizon;
   - 1024 x 128 tokens per step.

   Sequences stay at 128 tokens throughout. There is no 512-token phase for study cells: the eval stimuli are a short context sentence plus a target, far under 128 tokens. (The rater keeps its 512 phase because it scores +/-250-token windows.)
3. **New architecture name `bert_large_preln`.** It gets its own seeds, run ids, W&B names and registry records (family `bert`), so nothing mixes with the retired post-LN `bert_large` runs. The eval handles any `bert*` arch (`eval_v2_cell._is_bert`).
4. **Production runs in `null-subjects`.**
   - Our namespace, no 200-pod cap shared with labmates.
   - Corpora arrive via S3 only: the 140 cells' top-level files go to `data_transfer/manipulations/en/`.
   - Caches are rebuilt there at 128 tokens with `[CLS]`/`[SEP]` packing, 20 at a time at the measured in-band size.
5. **Storage.** A finished gpt2_large wave2 run (same size) takes 139 GB: 93 weight-only checkpoints at 1.3 GB plus 4 resume states at 4 GB. The full 140-run wave is therefore ~19.5 TB. I expanded the null-subjects PVC 2 -> 8 TiB for tranche 1; it grows per tranche, or analysis-only checkpoints get pruned after evals (D8).
6. **Tranche 1 is 32 runs:**
   - the 27 baseline cells (rand / robbi / robbianti x 9 deciles) plus 5 all-removed anchors;
   - wave id `w3` (one seed per cell, same scheme as gpt2_large);
   - the 24/48 GB pool at normal (non-preemptible) priority.

   This is the main result for BERT first, and stays under NRP's 50-GPU threshold for a Matrix post. Interventions follow in later tranches as GPUs free.
7. **Opportunistic H100 for production: later, not tranche 1.**
   - The probes show H100s run 7-8 h+ uninterrupted at ~5x A10. A production run would be ~16 h there, with a resume point every ~33 min.
   - Waits are long and unpredictable (~10 h overnight; minutes at midday Pacific).
   - Considered for later tranches, with podFailurePolicy ignoring DisruptionTarget (the launcher supports `--gpu-key h100 --priority opportunistic2`).
8. **Rater LR.** The 2e-3 trial waited 18 h for an L40/L40S, so I added an opportunistic-H100 candidate and the first to start won (H100, 2026-10-21 17:08). Rule when it finishes:
   - 2e-3 clearly better -> 2e-3;
   - near-tie -> 1e-3;
   - unstable -> 1e-3.
9. **Not done yet:** the full raters. They need the rater LR, the full option-B rater corpora moved via S3 (~252 GB on lemn-lab), and the 512-token finishing phase implemented.

## Open decisions

- Rater tokenizer: project WordPiece (current) vs roberta-style byte-level BPE.
- Rater corpora: all five interventions vs base only.
- Vocabulary: the literature favours 16–32K at this data scale (we use 50K). All caches are being rebuilt anyway, so now is the cheapest time to change it, if ever.
- Image rebuild with wandb 0.26 (pods pip-install it meanwhile; see `scripts/wave2_launcher.py`).
