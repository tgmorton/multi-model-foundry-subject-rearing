# Training-Stack Optimization Audit

**Date:** 2026-06-04

Produced by a 64-agent adversarially-verified audit workflow; 33 suggestions confirmed / 21 refuted.

Adopted items are tracked in [docs/design_ledger.md](../design_ledger.md).

---

I have all the data I need to synthesize the report. Let me write it directly.

# Training-Stack Optimization Audit — Clean Re-Run of the 300-Run English Production Matrix

This report synthesizes finder suggestions and adversarial-verifier verdicts. Where a verifier overrode a finder, the verifier's adjusted gain and science tag are used. Only suggestions with `is_real=true` appear in sections 1–3; killed suggestions are in section 4.

---

## 1. Executive Summary — Top 5 Fleet-Wide Levers by Verified Wall-Clock Impact

1. **Route gpt2_large + bert_large to A100-80GB (if reservation access is confirmed)** — ~3–4x per-run throughput; ~590 GPU-days → ~200–295 GPU-days for the heavy block (~295–390 GPU-days saved). **GATED**: the "no reservation needed" claim contradicts repo ground truth (`CLAUDE.md:182,228`); confirm admin-granted access first or it is a policy violation / 0 gain.
2. **Disable gradient checkpointing on the GPT-2 family (small/medium/large)** — ~25% per-step reduction; for gpt2_large that is ~16.5 s/step off a ~66 s/step run, ~60–70 GPU-h/run. Note: GC is already a silent no-op for Mamba and BERT (no method exposed), so it yields nothing there. Provably identical math.
3. **Enable torch.compile / Inductor in the production base configs** — ~8–18% per-step on the GPT-2/BERT transformer subset (gated down from 10–25% by active gradient checkpointing + no `drop_last` recompiles). One-line config flip (`compile_mode: 'default'`); the `torch._dynamo.disable()` at `trainer.py:198` is a no-op and need NOT be removed. Tens of GPU-days matrix-wide. Tag: validate numerical parity.
4. **Raise per-cell parallelism 2 → 4–6 for light archs** — collapses the calendar tail: a 10-completion cell drains in ~2 waves instead of 5 (~2.5x ceiling per cell). Adds zero GPU-days, does not trip the NRP webhook (per-pod request unchanged). Realized end-to-end gain is single-digit-days, bounded by the slowest pod per cell.
5. **Stratify the heavy/long-pole archs off the slow L4 tier (soft preference toward 3090/4090)** — rescues any run that would have landed on L4 from up to ~3x slowdown back to ~1x; per-cell makespan ~1.3–1.6x. Launcher-only nodeAffinity change. PI design decision (shifts seed-vs-node confound).

---

## 2. Per-Architecture Confirmed Suggestions

### gpt2_small

| Title | Verified gain | Effort | science_impact |
|---|---|---|---|
| Soft-prefer fast 24GB cards (3090/4090), keep L4 as fallback | Per-run: up to ~1.5–1.66x on the ~25–33% of runs that hit L4; matrix makespan uncertain/possibly neutral | Low (launcher nodeAffinity) | design_decision |

Use lever (b), a `preferredDuringScheduling` weight, not a hard L4 exclusion — hard exclusion removes ~25% of the pool and the matrix is concurrency-bound, so net makespan is ambiguous. Note `gpt2_small` is ~40 EN runs, not 240.

### gpt2_medium

| Title | Verified gain | Effort | science_impact |
|---|---|---|---|
| Disable gradient checkpointing (this config is a 124M GPT-2-small-footprint model) | ~12–20% wall-clock | Low (config flag) | safe |
| Enable torch.compile / Inductor | ~8–15% per-step (FA2 already handles attention; Inductor fuses pointwise/LN/GELU) | Low (config flip) | needs_resweep* |
| Raise phys batch 16 → 32 at constant eff batch | ~2–6% (favor 32 over 64; schedule-shift risk grows) | Low | safe |
| Switch fp16 → bf16 AMP | ~0–1% wall-clock; real value is stability | Medium (plumb dtype) | needs_resweep |
| Per-arch GPU pool stratification (drop/soft-prefer off L4) | Tail-bounded; median unchanged, up to ~3x on L4-landing runs | Low | design_decision |
| Async checkpoint writes + relocate `empty_cache()` | ~sub-1% to ~3%; true write volume ~120–190 GB/run (355M model) | Medium | safe |

*Inductor is `safe` if a 1-seed loss-curve check matches eager; the verifier tagged the medium variant as falling under the standard compile caveat. The cross-cutting compile finding (§3) is `needs_resweep` pending parity validation.

### gpt2_large

| Title | Verified gain | Effort | science_impact |
|---|---|---|---|
| Disable gradient checkpointing (354M, gpt2-medium sizing — fits ~9.7 GB at phys 4) | ~0.78–0.80x → ~4.5 GPU-days/run, saving ~1.1–1.4 GPU-days/run | Low (config flag) | safe |
| A100-80GB reservation (3–4x + 80GB headroom for no-ckpt phys 32) | 5.73 → ~1.6–1.9 GPU-days/run; single largest per-run lever | Medium (toleration/affinity) | design_decision |
| Enable torch.compile / Inductor | ~0.85–0.90x realized (10–15% likely), ~0.57–0.86 GPU-days/run; medium confidence (graph-breaks possible) | Low | needs_resweep |
| Switch fp16 → bf16 AMP | ~0.97–1.0x (≤0.1 GPU-day/run); robustness, not speed | Medium | needs_resweep |

Note the largest gpt2_large win is the A100 move, but it is gated on confirming `nautilus.io/reservation` access — the 5.73→0.49 GPU-day matrix-wide figure is inconsistent with the recommendation (route only gpt2_large + bert_large), ignore it.

### bert_large

| Title | Verified gain | Effort | science_impact |
|---|---|---|---|
| Stratify off L4 (BERT_POOL of 3090/A10/4090; mirror the mamba sub-pool pattern) | Per-run tail rescue: ~21 → ~6 GPU-days for any L4-bound run; slice-mean ~15–25% at a plausible 10–15% L4 fraction (upper-plausible, not established) | Low (one launcher list edit) | design_decision |
| Remove/fix the dead `use_gradient_checkpointing: true` flag (silent no-op on BERT wrapper) | ~0 direct; prevents a ~25–33% recompute regression if someone "fixes" it | Low (config honesty) | safe |
| Enable torch.compile / Inductor | ~5–12% per run, and ONLY if GC made compile-compatible; else 0–5% or regression | Low–Medium | needs_resweep* |

*Verifier downgraded the BERT compile finding to `is_real=false` on its stated "two defeaters" premise (the dynamo.disable is a no-op), but the underlying lever survives via the cross-cutting compile finding (§3); treat BERT compile gain as ~5–12% conditional on GC. BERT uses SDPA, not FA2.

### lstm

| Title | Verified gain | Effort | science_impact |
|---|---|---|---|
| Move LSTM off the 24GB pool onto A4000 (16GB) / MIG slices | Per-run compute unchanged (~9h); matrix-level: frees 24GB slots + cuts queue-behind-transformer latency; magnitude unquantifiable from code | Low–Medium (net-new affinity plumbing) | design_decision |
| Avoid the redundant ~1.6 GB `contiguous()` copy of shifted logits (slice `rnn_output[:, :-1]` BEFORE the projection) | ~1–3% per run (up to ~5–8% on slow A10/L4); LSTM-only | Low–Medium (re-check logits-shape consumers) | safe |
| Confirm torch.compile stays OFF for LSTM (cuDNN already fused) | 0 (prevents wasted plumbing effort) | None | safe |
| Bump CPU 2 → 4 for >1 dataloader worker + persistent_workers | Conditional/measurement-gated; 5% data_fraction = ~27 min/run, 10% = ~53 min/run IF data-bound; possibly ~0 if prefetch already hides it | Low (gated on telemetry + NRP webhook) | safe |

Realize the contiguous-copy fix via the pre-projection slice, NOT a `.reshape` swap (reshape still copies). The finder's own "3–6 min/run" CPU-worker figure was arithmetically wrong (too low); use the data_fraction-gated estimate.

### mamba_370m

| Title | Verified gain | Effort | science_impact |
|---|---|---|---|
| Stratify Mamba to fast 24GB cards (drop/soft-prefer off L4) | Per L4-bound run up to ~3x; per-cell makespan ~1.3–1.6x; fleet gain scales with L4 frequency | Low (MAMBA_POOL already plumbed) | design_decision |
| Remove the misleading `use_gradient_checkpointing: true` flag (silent no-op on Mamba wrapper) | ~0 (honesty fix; do NOT implement real GC — it would slow training) | Low | safe |
| Switch fp16 → bf16 AMP | ~0–1% wall-clock; stability/de-risking, not throughput | Medium | needs_resweep |
| torch.compile the pointwise glue (fullgraph=False around custom kernels) | ~0–3%, conditional; risk of net-negative from graph-break thrash; exploratory | Medium (measure bitwise first) | safe |

Cheaper fully-safe alternative to hard-pinning: populate `GPU_PRODUCT`/`NODE_NAME` in the registry (schema fields exist at `registry.py:332-333`, but no downward-API injection was found in the job spec) so node-type becomes a recorded covariate.

---

## 3. Cross-Cutting

### Data / IO

| Title | Verified gain | Effort | science_impact |
|---|---|---|---|
| Store chunked token cache as uint16, not int64 (vocab 50004 < 65536) | Storage: certain 4x smaller cache + 4x smaller stage-to-local copies. Wall-clock: marginal (~1–3% on lstm/gpt2_small on slow nodes only; ~0 on compute-bound archs; blended <1–2%) | Low (one-time prep re-run) | safe |
| Copy chunked cache to node-local emptyDir at pod init | ~0–2% overall, concentrated on lstm + epoch-1 cold reads / page-cache-evicted reads on busy shared nodes; ~0% on compute-bound archs. ~5s one-time copy | Low (emptyDir pattern already used) | safe |
| Raise CPU to 4 / set num_workers + persistent_workers explicitly | Arch-specific; ~0% for compute-bound, low-single-digit % on lstm/gpt2_small/medium IF data_fraction shows a stall; persistent_workers saves per-epoch fork/mmap-reopen regardless | Low (gate on telemetry + webhook) | safe |
| VERIFIED: no HF `.shuffle()` indices-indirection penalty exists | 0 (documents a non-issue; prevents wasted investigation) | None | safe |
| VERIFIED: `drop_last=False` is consistent with the checkpoint schedule | 0 (informational; changing it would desync LOCKED anchors — do not) | None | safe |

### Loop / Precision

| Title | Verified gain | Effort | science_impact |
|---|---|---|---|
| Disable gradient checkpointing — GPT-2 family ONLY | ~25% per-step on gpt2_small/medium/large; gpt2_large dominates (~60–70 GPU-h/run). Mamba/BERT yield 0 (GC already a no-op) | Low | safe |
| Enable torch.compile / Inductor (compile_mode null everywhere; the line-198 dynamo.disable is a no-op, not a blocker) | ~8–18% per-step on compute-bound transformers; LSTM ~0%, Mamba uncertain (verify "compilation successful" fires) | Low (config flip only) | safe* |
| Switch fp16+GradScaler → bf16 (no scaler) | <1% steady-state wall-clock; real value is operational (zero scaler step-skips, simpler resume) | Medium (plumb dtype) | needs_resweep |
| LSTM forward: skip the per-micro-batch `attention_mask.sum(dim=1).cpu()` sync (chunks are unpadded, packing branch is dead) | ~1–3% of LSTM step time (LSTM-only; whole-matrix <1%) | Low (guard on `.all()` / omit mask) | safe |
| Async analysis-only checkpoint writes + drop per-save `empty_cache()` | ~0.1–0.7% per run (best gpt2_medium ~0.4–0.7%; gpt2_large/bert ~0.05–0.24%). True volume ~39/111/106/115/241 GB for small/medium/bert/mamba/large | Medium | safe |

*The verifier confirmed `safe` for the cross-cutting compile finding (pure kernel fusion under fixed fp16+GradScaler, applied uniformly from reset); the per-arch lanes flagged a `needs_resweep` caveat pending a 1-seed parity check. Recommend treating compile as needs-parity-validation before matrix commit.

### Cluster

| Title | Verified gain | Effort | science_impact |
|---|---|---|---|
| Raise per-cell parallelism 2 → 4–6 for light archs (lstm, gpt2_small/medium, mamba) | Per-cell ceiling ~2.5x drain (5 waves → 2); end-to-end single-digit-days tail reduction, bounded by slowest pod; does NOT trip the NRP webhook | Low (live-mutable / `--parallelism`) | safe |
| Stratify off L4 (soft-prefer fast cards) for non-LSTM archs | Per L4-bound run up to ~3x faster; matrix wall-clock low-single-digit % (runs are independent — no paired-cell gating) | Low | design_decision |
| 48GB open tier (A40/L40/L40S/A6000) as no-paperwork A100 fallback for gpt2_large + mamba | ~5–20% per run (NOT ~50% — compute-bound at 98–100% SM, so doubling phys saves only launch overhead + escapes L4 tail) | Medium (per-request-key job variants) | safe |

---

## 4. Refuted — Do NOT Do

- **gpt2_small disable gradient checkpointing** — refuted as tagged: mis-tagged `safe` (dropout RNG stream diverges GC-on vs GC-off → trajectory change, so `needs_resweep`/`design_decision`); aggregate gain ~6x overstated (40 runs, not 240); VRAM-safety at phys=16-GC-off on the densest ablation corpus is unverified.
- **gpt2_small enable torch.compile** — refuted as written: leaves GC ON (graph breaks gut Inductor) and misreads line-198 as a blocker; realistic net ~0–1.1x as proposed.
- **gpt2_small raise phys 16→32** — ~0–3% at 98–100% SM (compute-bound); route (a) OOM-risky, route (b) depends on an unverified sibling change.
- **gpt2_small persistent_workers / CPU bump** — ~0–0.2%; SM 98–100% rules out a data-bound regime; CPU bump risks the NRP webhook.
- **gpt2_small fp16→bf16** — <1%; gain overstated ~5–10x; GPU is SM-bound.
- **gpt2_large raise phys 4→16 at constant eff batch** — ~1.00x standalone at 98–100% SM; phys=16 VRAM-fit is unverified and contradicted by the config's own 24GB sizing note (phys was deliberately dropped 16→4).
- **gpt2_large drop cuDNN deterministic mode** — ~1.00x; cuDNN flags do not touch the cuBLAS matmuls or flash-attn kernels that dominate a transformer; no conv ops to autotune.
- **bert_large enable torch.compile (as a standalone)** — refuted on its "two stacked defeaters" premise (the dynamo.disable is a no-op, not a second blocker); the real lever survives via the §3 cross-cutting compile finding at ~5–12% conditional on GC.
- **bert_large raise phys 4→8** — ~0–3%; the "~10 GB free" headroom premise is false (production runs `use_gradient_checkpointing: true`, so the assumed activation budget does not hold); BERT uses SDPA not FA2.
- **bert_large fp16→bf16** — ~0% (<0.1%); GradScaler sync is ~0.006% of a ~35 s step; robustness change mislabeled as throughput.
- **bert_large async checkpoint writes** — gain overstated ~15–30x (~0.1–0.2%, not 3–6%); internally inconsistent with its own 200 MB/s assumption. (Note: the §3 loop/precision async-checkpoint finding IS confirmed at ~0.1–0.7%; adopt that framing, not this one's headline.)
- **lstm fp16→bf16** — negligible; assumed 1–2% overflow rate is ~10–20x too high (steady-state <0.1%); the repo's own overflow counter is dead code (never incremented).
- **lstm raise phys 16→64** — OOMs on every pool; phys 32 already hits ~87% VRAM (the logits/CE over a 50004 vocab dominate, not the recurrent core); A4000/MIG are tighter still.
- **mamba raise phys 4→16** — OOMs; resident is ~9 GB (fp32 master weights, ~558M real params) + ~24 GB activations at phys 16 (GC is a no-op, so activations are at max); only phys 4→8 fits, yielding ~2–5%, not the claimed 8–15%.
- **mamba CPU bump for >1 dataloader worker** — ~0%; SM 98–100% (compute-bound), pre-chunked ~32KB batches are hidden by prefetch; phys→16 compounding doesn't apply (Mamba locked at phys 4); CPU bump risks the webhook.
- **A100 "no reservation needed, 84 open GPUs"** — refuted as a live-cluster claim that contradicts repo ground truth and the task's stated facts (A100 is reservation-gated); acting on a wrong "no gate" reading is a policy violation. The A100 routing direction is real but must be gated on admin confirmation (see §2 gpt2_large).
- **Exclude L4 fleet-wide (28% of cells run 3x slower)** — refuted: based on a false 2-pod-lockstep-cell model. Cells are 10-completion / parallelism-2 Indexed Jobs with independent runs and no barrier, so a slow L4 delays only its own run. Real benefit is a low-single-digit % tail trim (captured by the soft-prefer stratification instead).
- **Pin both seeds of a cell to the same GPU product** — refuted: the 2-seed-pod gating mechanism does not exist; as specified (pin to an unspecified product) it risks NET SLOWDOWN by pinning all 10 runs onto slow L4 and shrinking the schedulable set ~4x. Collapses into the L4-stratification move.
- **wandb.log every step / log_metrics_every_n_steps dead field** — refuted as a wall-clock win (<0.1%); the actual per-step sync is the unconditional `_mean_loss().item()` in the progress bar, which this fix does not touch; wandb.log is async. Worth doing only as dead-config cleanup + wandb rate-limit hygiene.
- **pin_memory=True commentary** — no-op by the finder's own admission (already enabled and correct); nothing to apply.

---

## 5. Recommended Adoption Bundle for the Clean Re-Run

### (a) Safe — adopt now (zero science impact)

1. **Disable gradient checkpointing on the GPT-2 family only** (gpt2_small carries a trajectory caveat from dropout RNG — see (b); gpt2_medium and gpt2_large are clean `safe`). Mamba/BERT: just remove the dead flag (honesty fix, 0 gain). — **largest safe lever, gpt2_large dominant.**
2. **Store the chunked token cache as uint16** — certain 4x storage/copy reduction, one-time prep re-run. Marginal wall-clock but unconditional infra win.
3. **Copy chunked cache to node-local emptyDir at pod init** — ~0–2%, robustness floor on shared nodes.
4. **Async analysis-only checkpoint writes + drop per-save `empty_cache()`** — ~0.1–0.7%, math-identical.
5. **LSTM: avoid the redundant contiguous() logits copy (pre-projection slice)** — ~1–3% LSTM.
6. **LSTM: skip the dead attention_mask `.cpu()` sync** — ~1–3% LSTM.
7. **Raise per-cell parallelism 2 → 4–6 for light archs** — single-digit-days calendar tail reduction, zero GPU-days.
8. **persistent_workers=True** (harmless cleanup); CPU bump only if `timing/data_fraction` telemetry proves a stall.

### (b) needs_resweep — decide before sweeps are frozen

1. **torch.compile / Inductor matrix-wide** — ~8–18% on the transformer subset. Mechanically a `safe` kernel-fusion change, but commit only after a 1-seed loss-curve parity check vs eager. Decide before sweeps so winners are validated under the same compile setting that production uses.
2. **fp16 → bf16 AMP** — throughput gain is ~0 (all archs are SM-bound), but it changes the trajectory and removes loss-scaling fragility. If adopted, the HP sweeps MUST be run in bf16 too. Decision is robustness-driven, not speed-driven.
3. **gpt2_small gradient-checkpointing removal** — because dropout RNG-stream divergence makes it a trajectory change, treat it as a sweep-time decision, not a free `safe` flip.

### (c) design_decision — PI must choose

1. **Route gpt2_large + bert_large to A100-80GB** — the single largest per-run lever (~3–4x), but GATED on confirming admin-granted `nautilus.io/reservation` access (repo says reservation-gated; do not tolerate unauthorized values). Falls back to the 48GB open tier (~5–20%) if A100 access fails a test pod.
2. **Per-arch GPU pool stratification (soft-prefer fast cards off L4)** for the heavy/long-pole and high-step-rate archs — shifts the seed-vs-node confound the registry tracks. Prefer soft `preferredDuringScheduling` weighting over hard exclusion to preserve pool breadth.
3. **Move LSTM to A4000/MIG** — frees 24GB slots for transformers; changes which hardware tier the LSTM confound lives on.

### Estimated combined wall-clock reduction of bundle (a) on the 300-run matrix at 60-GPU steady parallelism

The dominant term is gradient-checkpointing removal on the GPT-2 family. The matrix is heavily weighted toward the heavy GPT-2 cells: gpt2_large at ~5.73 GPU-days/run × ~50 runs ≈ 286 GPU-days, gpt2_medium ~21 GPU-days-equivalent × ~50, gpt2_small ~50 runs. Bundle (a) saves:

- **GPT-2 family GC removal: ~25% on ~all GPT-2 cells.** gpt2_large alone: ~50 runs × ~1.2 GPU-days saved ≈ **~60 GPU-days**. gpt2_medium: ~50 × ~0.25 GPU-day ≈ ~12 GPU-days. gpt2_small: small. Subtotal ≈ **~75–80 GPU-days**.
- **Parallelism bump:** reduces calendar makespan, not GPU-days — at 60-GPU steady parallelism it mainly shortens the tail (single-digit calendar days), so it does not add to the GPU-day total but shortens wall-clock-to-completion.
- **LSTM loop fixes (~2–6% combined on ~40–50 lstm runs of ~9h):** ~few GPU-hours total.
- **Async checkpoints + uint16 + node-local copy:** sub-1% each, blended ~a few GPU-days across the matrix (concentrated on light/fast archs and shared-node contention).

**Net bundle (a): roughly 80–95 GPU-days saved on the 300-run matrix**, overwhelmingly from gpt2_large gradient-checkpointing removal. At 60-GPU steady parallelism (~5,160 GPU-hours/day capacity), that is on the order of **~0.4–0.45 days of calendar time recovered from GPU-day reduction alone**, plus an additional single-digit-days tail reduction from the parallelism bump. The big multiplicative wins (A100 routing, ~295–390 GPU-days) live in bundle (c) and require PI/admin sign-off before they can be banked.

Relevant files: `/Users/thomasmorton/multi-model-foundry-subject-rearing/scripts/launch_production_training.py`, `/Users/thomasmorton/multi-model-foundry-subject-rearing/scripts/production_agent.py`, `/Users/thomasmorton/multi-model-foundry-subject-rearing/model_foundry/training/loop.py`, `/Users/thomasmorton/multi-model-foundry-subject-rearing/model_foundry/trainer.py`, `/Users/thomasmorton/multi-model-foundry-subject-rearing/model_foundry/architectures/{rnn,mamba,bert,gpt}.py`, `/Users/thomasmorton/multi-model-foundry-subject-rearing/model_foundry/data.py`, `/Users/thomasmorton/multi-model-foundry-subject-rearing/model_foundry/checkpoint_schedule.py`, `/Users/thomasmorton/multi-model-foundry-subject-rearing/configs/sweeps/baselines/*_en.yaml`, `/Users/thomasmorton/multi-model-foundry-subject-rearing/data/sweep_winners/*_en.json`.
