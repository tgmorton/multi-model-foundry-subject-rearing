# BERT-Rated Crossed-Sweep Study — Master Plan

Captured 2026-08-22 from Thomas's directive. This supersedes the wave-1
dispatch of the external-gpt2-medium graded corpora (those 19 corpora are
retained on the PVC but will be rebuilt under the new rating; nothing
trained on them yet).

## The directive (goals, restated)

1. **Re-rate recoverability with pretrained BERT.** Replace the external
   gpt2-medium rater with the most publication-standard pretrained BERT
   available; regenerate the surprisal annotation (masked-slot scoring),
   re-freeze the selection, and rebuild sweeps from it.
2. **Crossed corpus matrix (100 cells).** Two sweep arms per Thomas's
   naming — the **surprisal sweep** (info deciles) and the **random
   sweep** — each at K = 10..100, crossed with **5 intervention
   conditions**: baseline, remove_expletive_sentences,
   enrich_verbal_morphology, impoverish_case, and impoverish verbs
   (= `lemmatize_verbs`; confirm mapping). 5 × 10 × 2 = 100 corpus cells.
   Interventions are applied ON each decile sweep such that **the same
   pronouns are removed in every intervention** — i.e., pronoun-drop
   selection is fixed and intervention edits must not perturb which
   pronoun instances get removed.
3. **Tokenizer question.** Decide whether cells need their own tokenizers
   (Thomas's intuition: tokenizer convergence may care about pronoun
   frequency + intervention statistics) or whether the existing shared
   tokenizers stand. See Decision D2.
4. **Training matrix: 5,000 runs.** Per corpus cell: 5 architectures
   (gpt2_small, gpt2_medium, gpt2_large, bert_large, lstm — **no
   mamba**) × top-5 HPs × 2 seeds = 50 runs/cell... (10 runs per
   (cell, arch)); 100 cells × 50 = 5,000 trainings. **Seed policy:
   seeds are treated as unique samples from a population — randomly
   generated per run**, never shared across HPs, corpora, or
   architectures; every generated seed is recorded (registry + job spec).
5. **Storage campaign (concurrent with BERT annotation).** Full PVC
   audit + aggressive pruning:
   - Models registered for evals (enumerated by the `cell_id` sets in
     `analysis/eval_v2/figures/foundry_trajectories/condition_matched_v1/*.csv`
     + `coverage_manifest.json` in the codex worktree) → KEEP evaluatable
     (weights-only) checkpoints; count resumable vs evaluatable per run;
     **strip `training_state.pt` (resume state) as aggressively as
     possible** — still eval-able, no longer trainable-from.
   - Models NOT eval-registered (failed runs, extraneous data) → DELETE.
   - Similar audit pass over S3.
6. **Worktree merge + repo hygiene.** Merge
   `~/.codex/worktrees/e7fb/multi-model-foundry-subject-rearing`
   (branch `codex/fix-continuation-resume`: 33 commits ahead / 13 behind
   main, 76 dirty files) into main; send agents to organize and document
   what has been done in that line of work; get the repo in good order.
7. **Be smart about compute/storage before starting** — anticipate
   demands, maximize parallelism around blockers.

## Decision register

| # | Decision | Status | Resolution |
|---|---|---|---|
| D1 | Which pretrained BERT | **LOCKED 2026-08-22** | `bert-large-uncased-whole-word-masking`. |
| D2 | Tokenizers: shared vs per-cell | **LOCKED 2026-08-22** | Shared tokenizers (`en_shared_unigram`, `en_bert_wordpiece`) stand. Probe-tokenizer drift check optional, non-blocking. |
| D3 | "impoverish verbs" = `lemmatize_verbs` | **CONFIRMED** | The four interventions = {remove_expletive_sentences, impoverish_case, lemmatize_verbs, enrich_verbal_morphology}. |
| D4 | Stacking semantics | PROPOSED (unobjected) | Intervention edits computed from RAW parse, merged as edit-plans per doc; identical pronoun removals across interventions, identical intervention edits across sweeps. |
| D5 | Compute staging | **DEFERRED by Thomas** | Direction set: prioritize wide runs on cheap archs first (gpt2_small, lstm) for throughput, then heavier archs. gpt2_large inclusion decided later; optimize when we get there. |
| D6 | Deletion protocol | **LOCKED 2026-08-22 — THREE phases** | Phase 1: total READ-ONLY manifest (no deletions of any kind). Phase 2: garbage deletion, post-manifest-review. Phase 3: remaining pruning (resume-state stripping etc.), post-review. |
| D7 | Production epochs | OPEN (assumed 30) | Locked study parameter is 30; nothing said otherwise. |
| D8 | Checkpoint paradigm for the wave | **NEW 2026-08-22** | Adopt the lang-manifold **bit-guaranteed retraining** paradigm (`~/lang-manifold`): sparse resumable checkpoints (~4/model) + determinism manifests that allow bit-exact re-materialization of intermediate steps by replaying training segments; inline evals during training. Port requires: same verification smoke here (retrain a segment, diff bits) and the same implementation choices. Replaces keep-everything checkpointing for Track E. |
| — | Contraction residue kept; population = personal pronouns | LOCKED (2026-08-15) | Carries over unchanged. |
| — | Random seeds per run, recorded | LOCKED (this directive) | fdy wave already validated random-seed run_ids end-to-end. |

## Track breakdown

### Track A — BERT rater + selection v3 (blocks corpora)
- A1. Extend `scripts/score_pronoun_recoverability.py` with an MLM mode:
  masked-slot scoring (mask the pronoun's wordpiece(s), one forward per
  instance, batched; window ±~250 wordpieces around the slot). Banks the
  same sufficient statistics (slot logprob, inventory distribution,
  entropy). No corpus-wide per-token PLL (130M masked forwards is not
  worth it; slot statistics are the measure). Buildable NOW — does not
  depend on D1.
- A2. Score train_90M + pull_10M (12 shards × 1 MLM scorer; ~1–3 h wall).
- A3. Analysis v3: BERT ranking vs external-gpt2m vs in-house raters
  (the invariance table grows a row — publication asset), decile
  composition, freeze **selection v3** + pool tables (absolute
  thresholds, same consumption semantics — ablation module unchanged).
- Blocker: D1 only.

### Track B — Storage audit + pruning (start immediately; independent)
- B1. Inventory: per-run checkpoint census on the PVC (checkpoint count,
  which have `training_state.pt`, per-run bytes), joined against
  (i) the union of `cell_id`s in condition_matched_v1 CSVs +
  coverage_manifest, (ii) the S3 registry (status field).
- B2. Classify: eval-registered → strip resume states; FAILED/orphan →
  delete; ambiguous → manifest.
- B3. `docs/STORAGE_AUDIT.md` + deletion manifest + projected reclaim →
  Phase-1 deletions execute; Phase-2 after Thomas reviews the manifest.
- B4. S3 prefix audit (registry, eval_results, per_token, recoverability).
- B5. New-wave save policy design (required regardless — see Storage
  analysis): streaming train→eval→prune lifecycle.

### Track C — Worktree merge + documentation (start immediately; independent)
- C1. Survey `codex/fix-continuation-resume` (33 ahead / 13 behind, 76
  dirty files): commit inventory, dirty-file triage, overlap analysis
  with main's recent changes.
- C2. Merge into main (main-loop work; conflicts likely limited to
  fleet/eval surfaces), preserving the worktree's provenance.
- C3. Documentation sweep (subagents): what the fdy waves were, what
  condition_matched_v1 shows, continuation-resume mechanics —
  consolidated into docs/.

### Track D — Corpus factory (blocks on A3 + D2–D4; combinator buildable now)
- D1. **Edit-plan combinator**: per-doc merge of {selection-driven token
  deletes} + {intervention edits computed from raw parse}. New stacked
  ablation type consuming (selection_dir, arm, k, intervention).
- D2. Pool sufficiency for stacked cells — worst case
  remove_expletive_sentences × K=100 (line-removal + pronoun shortfall
  compound). If margins fail → pool expansion pipeline (new same-genre
  material → annotate → BERT-score → threshold via frozen manifest).
  Per-genre backfill constraint documented (speech genres can't be
  backfilled from Gutenberg/Wiki).
- D3. Compose 100 cells (indexed jobs; ~1 GB/cell + ~2.4 GB caches per
  tokenizer family per cell).
- D4. Probe-tokenizer check (if D2 compromise accepted).

### Track E — 5,000-run wave (blocks on B, D + D5 sign-off)
- E1. Launcher v2: cell-indexed, random-seed generation + recording,
  5 archs × h0–h4 × 2 seeds, no mamba.
- E2. **Train→eval→prune streaming lifecycle** (storage-mandatory):
  eval each run's trajectory as it completes, archive eval parquets to
  S3, prune weights to endpoint(+reference) checkpoints, then release
  disk. Concurrency throttled to keep PVC below a hard watermark.
- E3. Phased dispatch per D5; fleet monitoring via fleetview.

## Storage analysis (hard numbers)

- PVC now: **96T used / ~3T free (97%)** on a 90 TiB CephFS volume.
- Naive 5,000-run wave at the current save policy is **impossible**:
  fp32 weights × ~97 anchors gives per-run footprints of ~17 GB
  (gpt2_small) to ~300 GB (gpt2_large); even the fdy wave's measured
  ~34 GB/run average puts 5,000 runs at **~170–500 TB**.
- Therefore two mandatory measures:
  1. **Track B reclaim** — resume-state stripping (~19 anchors/run × 2–3×
     weight size on ~900 production runs) plus non-eval'd deletions;
     target: tens of TB back. Audit will give exact numbers.
  2. **Streaming lifecycle (E2)** — steady-state disk = (in-flight runs ×
     footprint) + retained endpoints (~1 GB avg × 5,000 ≈ 5 TB) + eval
     parquets (S3). In-flight cap set by free space, not by wave size.
- S3 is not currently a concern (small artifacts) but B4 verifies.

## Compute analysis (napkin, to refine in Track B/E)

Per-run 30-epoch estimates on the 24 GB pool: gpt2_small ~12 h,
gpt2_medium ~30 h, gpt2_large ~80 h (phys 4), bert_large ~40 h, lstm
~8 h → 1,000 runs each ≈ **~170K GPU-hours ≈ 19 GPU-years**.
At a sustained 150 concurrent GPUs ≈ ~7 weeks; at 250 ≈ ~4 weeks.
gpt2_large alone is ~47% of the budget — D5's cell-subset option exists
for a reason. A100-reservation request to NRP is worth considering for
the large models.

## Execution map (parallelism + delegation)

Immediately parallel, no blockers:
- **B1 inventory** (cluster pods, main loop) — longest lead item.
- **C1 worktree survey** (subagent) → C2 merge (main loop) → C3 docs
  (workflow: parallel readers → synthesis).
- **A1 MLM scoring mode** (main loop code; toy-validated locally).
- **D1 combinator** (main loop code + tests).

Blocked chains:
- A2 needs D1(decision) → A3 → D3 compose needs A3 + D2 + D2-sufficiency.
- E needs B reclaim + D3 + D5 sign-off.

Delegation policy (per standing rules): cluster ops main-loop only;
sonnet subagents for surveys/docs/mechanical sweeps; workflows for
fan-out documentation and audit-classification passes; all destructive
actions gated per D6.
