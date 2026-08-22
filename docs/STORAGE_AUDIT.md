# Storage Audit — subject-drop-archive PVC (Phase 1, 2026-08-22)

Read-only census + classification per the three-phase deletion protocol
(docs/PLAN_BERT_CROSSED_SWEEPS.md, D6). **Nothing has been deleted.**
Phases 2 and 3 execute only after Thomas reviews the manifests below.

Census: `scripts/storage_census.py` (job `thomas-storage-census-v1`,
20 min). Classification: `scripts/classify_storage_census.py`.
Artifacts: `data/storage_audit/` (local) + `s3://thomas-subject-drop-artifacts/storage_audit/census_2026-08-22/`.

## Headline

- PVC: **96.2 TB used / 2.8 TB free (97%)** of 90 TiB.
- **Model checkpoints are the entire volume**: 1,909 run dirs = 96.0 TB
  (weights 69.1 TB, resume states 15.0 TB, other 11.9 TB).
  86,609 checkpoints; 8,707 resumable. Everything else — corpora, caches,
  annotations, recoverability outputs — totals < 0.2 TB.
- Eval keep-list: **715 run_ids** (union of `cell_id` across
  condition_matched_v1 CSVs; matches `coverage_manifest.json` count).
- Registry snapshot: 2,623 records (1,371 COMPLETE / 616 stale-RUNNING /
  30 FAILED). Note: **no compacted `registry.parquet` exists on S3** —
  the hourly compactor never materialized it; classification used the
  per-run JSONs. Worth fixing independently.

## Classification

| Bucket | Runs | Total TB | Weights | Resume | Disposition |
|---|--:|--:|--:|--:|---|
| keep_eval_registered | 715 | 49.9 | 37.2 | 12.5 | KEEP weights (evaluatable); resume states → Phase 3 strip |
| hold_pending_eval | 150 | 7.1 | 6.2 | 0.8 | Untouched: recent COMPLETE/live runs awaiting delta-tranche eval (Aug h1wave etc.) |
| phase2_sweeps | 996 | 21.7 | 21.3 | 0 | HP-sweep trial checkpoints; winners long extracted to `data/sweep_winners/` → Phase 2 delete |
| phase2_failed_orphan | 40 | 17.4 | 4.4 | 1.7 | See line items below → Phase 2 delete (mamba h1: decide) |
| keep_infrastructure | 8 | 0.05 | — | — | raters, init states, detsmoke |

### phase2_failed_orphan line items

- `models/invalidated/stage1-two-epoch-lr` — **9.0 TB**, explicitly
  invalidated wave.
- `models/production/_failed_attempts` — **2.28 TB**, self-labeled.
- **June-era mamba_370m h1 lanes** (~20 runs × ~200 GB ≈ 4.1 TB):
  COMPLETE, never evaluated (excluded from the stable tranche as
  mutable), and mamba has since been dropped from the go-forward
  architecture matrix. **Decision needed**: evaluate-then-prune (if the
  current paper line wants mamba h1 endstates) or delete outright.
- Remainder: registry-FAILED and unregistered orphan dirs, stale > 14 d.

## Reclaim projection

| Action | Reclaim | PVC after |
|---|--:|--:|
| Phase 2: sweeps + invalidated + failed/orphan | **39.1 TB** | 57.1 TB used (59%) |
| Phase 3: resume-strip the 715 keep-list runs (7,826 resumable ckpts) | **12.5 TB** | 44.6 TB used (46%) |
| Combined | **51.6 TB** | ~46% occupancy |

That headroom carries the 100-cell corpus factory (~0.6 TB incl. both
tokenizer-family caches) and the 5,000-run wave under the streaming
train→eval→prune lifecycle (D8) with a comfortable in-flight buffer.

## Manifests for review (Phase 2/3 gates)

- `data/storage_audit/phase2_sweeps_manifest.csv` — 996 dirs, 21.70 TB
- `data/storage_audit/phase2_failed_orphan_manifest.csv` — 40 dirs, 17.38 TB
- `data/storage_audit/phase3_resume_strip_manifest.csv` — 712 runs,
  12.52 TB of `training_state.pt` files (runs stay fully evaluatable)

Review notes: the phase-3 strip removes resume ability from completed,
already-evaluated runs only. Under the D8 paradigm a future re-derivation
of intermediate states would come from deterministic replay instead —
gated on the D8a smoke verdict. If we want a safety margin, we can keep
the FINAL checkpoint's resume state per run (~0.65 TB retained) and strip
only the interior anchors.

## S3 (B4)

Bucket prefixes are small (registry JSONs, eval parquets, recoverability
~15 GB total incl. per-token logprobs). No action needed beyond fixing
the registry compactor. Full prefix-size table deferred to the compactor
fix.
