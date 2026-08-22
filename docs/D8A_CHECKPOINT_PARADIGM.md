# D8a — Checkpoint Paradigm for the 5,000-Run Wave

Decision memo (2026-08-22) for D8 in docs/PLAN_BERT_CROSSED_SWEEPS.md:
porting the lang-manifold sparse-checkpoint + deterministic-replay
paradigm onto model_foundry, with measured numbers.

## What the smoke measured (job `thomas-detsmoke-v1`)

gpt2_small, 400 optimizer steps, seed 1234, fold_a corpus, all pods
pinned to NVIDIA-GeForce-RTX-3090 (different physical nodes):

| Variant | training.deterministic | endpoint sha256 | step time |
|---|---|---|---|
| det1 | true | `2d6237bb…20bcf6` | 2.19 s/it |
| det2 | true | `2d6237bb…20bcf6` **(identical)** | 2.10 s/it |
| fast | false (FA2 etc.) | `2f0d8255…b77706` | 1.25 s/it |

- **Bitwise CUDA determinism holds** across separate nodes of the same
  GPU product, through fp16 AMP + GradScaler + fused AdamW + SDPA.
  (lang-manifold's guarantee was CPU-only; its own docs record CUDA as
  unsolved. Our `training.deterministic` mode — CUBLAS workspace,
  `use_deterministic_algorithms`, TF32 off, SDPA-for-FA2 — closes it.)
- **Zero** "does not have a deterministic variant" warnings → full op
  coverage on the gpt2 path (warn_only never fired).
- **Throughput tax ≈ 1.7×** (12.8K → 7.3K tok/s). Dominated by
  SDPA-vs-FA2; expect similar for gpt2_medium/large and bert_large,
  near-zero for lstm (no attention).
- Constraint inherited by any bitwise claim: **replay must run on the
  same GPU product** as the original (verified for 3090↔3090; cross-
  product equality untested and not assumed).

## What lang-manifold contributes (ported design)

- Manifest schema: RNG capture (python/numpy/torch-CPU/CUDA-all) +
  config + git SHA + env identity — extended with **GradScaler state**
  (already saved by our checkpointing) and enforced determinism-flag
  recording.
- Blake2b-derived per-step schedules (never stored cursors), atomic
  `.tmp`+`os.replace` checkpoint writes, kill-mid-run resume smoke
  (`smoke_preemption.py` analog), and the bitwise-or-epsilon comparator
  with an explicit `stop_and_review` verdict label.
- Prefix-consistency constraint does NOT apply to us: we replay segments
  of the same run with the same total_steps, so any schedule that is a
  pure function of (step, config) is safe.

## Recommendation

**Hybrid, storage-identical either way** (~4 resumable anchors per run:
ep1, ep2, midpoint, final — matching the Phase-3 policy):

1. **Default (≈98% of runs): fast kernels.** Streaming lifecycle evals
   every checkpoint BEFORE pruning, so trajectory science is captured
   from the originals. The 4 anchors still support *resume/continue*
   (which never needed determinism). What is given up: re-deriving a
   deleted interior checkpoint reproduces it only epsilon-close, not
   bitwise — verified with the comparator, labeled honestly.
2. **Bitwise subset (≈2% of runs): `training.deterministic: true`** on
   one designated run per (arch × a small cell set, e.g. baseline
   sweeps) — the capability demonstrated, verified by a det1/det2-style
   paired hash at wave start, available where an exact re-derivation
   could matter. Cost ≈ +1.7× on ~2% of compute ≈ +1.4% wave total.
3. If Thomas prefers the full bitwise guarantee wave-wide: multiply the
   ~170K GPU-hour budget by ~1.6–1.7 for the attention archs — his call,
   the smoke says it would WORK, it's purely a compute price.

Verification plan either way: port the kill-and-resume smoke (assert
bitwise resume for deterministic runs, epsilon for fast) + a paired-hash
sentinel at wave start per arch; both wired into the wave launcher.

## Resolution (Thomas, 2026-08-22)

Fast kernels everywhere — the 1.7× tax is not worth paying for bitwise
replay; the recovery story for a truly-needed intermediate state is
"retrain from start." No deterministic subset. In exchange, the wave is
**data-preemptive**: the capture manifest (D9 in the master plan) banks
the eval suite, per-token logprobs, held-out ppl, the stratified pronoun
probe battery, weight-stat summaries, and anchor-point representations
in the eval pod pass — with pruning gated on capture-complete markers —
so deleted weights never take an answerable question with them.
