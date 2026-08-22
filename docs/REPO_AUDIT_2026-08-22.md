# Repository Audit — multi-model-foundry-subject-rearing
**Date:** 2026-08-22 · **Scope:** full-repo dead-code, bloat, and organization pass, adversarially re-verified against actual imports/YAML/docs · **Status:** read-only findings for operator triage

## 1. Executive summary

- **Confirmed dead code:** ~49 files (~6,000+ lines) across scripts/, k8s/, preprocessing/, model_foundry/, analysis/, and configs/ — each independently verified to have zero live importers/callers after an adversarial re-check (see §3).
- **8 of the original dead-code claims were refuted** by live references the initial survey missed — three of them (`k8s/job-train-tokenizer.yaml`, the Europarl/Insert-Pronouns pipeline, a broken import in the v1 eval runner) are more important as *live-code bugs or open study obligations* than the archival items they were mistaken for. Details in §3's "reclassified" box and inline throughout.
- **Git history bloat:** ~11.6 GB of unique blob content is permanently baked into `.git` from two now-untracked directories (`data/eval_results/` 8.2 GB, `evaluation/results/` 3.0 GB) that were committed-then-removed. This is not reachable from a normal `git log` walk on current paths, but `git fsck`/`merge-base` confirm both are real ancestors of `main` and cannot be GC'd without a history rewrite.
- **Currently-tracked, regenerable bloat:** ~180 MB — `analysis/eval_v2/figures/` (135 MB of PNGs/CSVs, actively churning), `latex/` (8.5 MB, includes a byte-identical duplicate of a file also in `presentations/`), and a 37 MB orphaned Arrow fixture that slipped past `.gitignore` via a force-add.
- **Documentation drift:** the four index docs a newcomer would read first (`docs/DOCUMENTATION_MAP.md`, `docs/STRUCTURE.md`, `docs/README.md`, `docs/k8s_jobs.md`) all date to a single 2026-04-09 commit and don't mention anything added since — including the current master plan (`docs/PLAN_BERT_CROSSED_SWEEPS.md`), the registry/S3 docs, or ~130 of `k8s/`'s 205 files.
- **Nothing here requires urgent action.** Training and eval are running; every proposal below is framed as either a same-day no-risk cleanup or a flagged, PI-confirmed structural move.

## 2. Bloat inventory

| Path | Size | Why | Disposition |
|---|---|---|---|
| `data/eval_results/` (git history only) | 8.2 GB | Committed in 4e2dc0c/b71d634, later removed from tree; still a permanent ancestor of `main`, not GC-able | History rewrite (destructive, needs coordination) — see §6 |
| `evaluation/results/` (git history only) | 3.0 GB | Same pattern; correctly cleaned from tree in c749dda, but pre-cleanup commits are baked in | Same as above; good precedent already exists (c749dda) |
| `analysis/eval_v2/figures/` | 135 MB | 244 tracked PNG/CSV files; `.gitignore` has a rule for the *old* `analysis/figures/` path but not this one; `scil_endstate_multirun.pdf` is churning right now (`git status` shows a binary diff with no content change) | Add `.gitignore` rule, `git rm --cached` after confirming no LaTeX/paper `\includegraphics` points at a frozen copy |
| `data/chunked/test_checkpoint_scheduling/data-00000-of-00001.arrow` | 37.4 MB | Force-added (`.gitignore` explicitly denies `data/chunked/`); **reclassified** — not zero-reference: 33 current `job-prepare-dataset*`/`job-sweep-*` YAMLs `rm -rf` around its presence as a documented workaround ("git ships this fixture as a non-empty dir, symlink can't replace it") | Remove deliberately, and simplify/relax the now-unnecessary `rm -rf` comment in those 33 YAMLs in the same change |
| `latex/` | 8.5 MB (6.6 MB binary) | `latex/Presentation.tex` is byte-identical to `presentations/Presentation.tex`; whole dir untouched since 2026-04-09, no CI/doc reference; live paper work has moved to the separate `Subject-Rearing-Paper/` repo | Confirm with PI, then drop one copy of the duplicate and move/retire the rest |
| `data/recoverability/`, `data/eval_results/` (working tree), `data/unigrams/` | 9.0 GB / 1.1 GB / 1.9 MB | Untracked but **not covered by any `.gitignore` rule** — exactly the gap that caused the `data/eval_results/` history incident | Add explicit ignore lines now, before any broad `git add -A data/` |
| `relaunch_slots.json` (repo root) | 3.5 KB | Transient per-slot relaunch bookkeeping from a resolved 2026-05-28 incident, sitting next to real project config | Move under `scripts/` with a one-line note, or gitignore |
| `OVERNIGHT_STATUS.md`, `docs/jobpostings.md` | ~8 KB, ~few KB | 4-month-old resolved status snapshot; unrelated job posting already flagged as scratch in the project's own doc index | Delete |
| `presentations/Oct22Meeting.out` | 24 KB | LaTeX `.out` (hyperref bookmarks) — `.gitignore` covers `.aux/.bbl/.blg/.synctex.gz` but not `.out` | Add `*.out` to `.gitignore` |

## 3. Confirmed dead code

Every item below was independently re-verified (import graphs, YAML greps, doc cross-references) after the initial survey; only items that survived that check are listed.

| Path | Evidence | Proposal | Confidence |
|---|---|---|---|
| `scripts/wild_west/{train.sh,gpu_monitor.sh}` | Predates K8s pivot; zero refs outside itself and two stale Feb-12 docs; sibling hardware-era dirs (`a5000/`,`p6000/`,`titanx/`,`3070ti/`) already moved to `scripts/_deprecated/` | Move alongside siblings | High |
| `scripts/generate_experiment_configs.py` | Only reference is the stale `scripts/SCRIPTS_ORGANIZATION.md`; its own printed next-step tells the user to run a script already in `_deprecated/` | Delete or move to `_deprecated/` | High |
| `scripts/log_manager.py` | Referenced only by the same stale doc batch | Retire alongside wild_west, after confirming no ad-hoc laptop use | Medium |
| `scripts/{apply_seed_annotations,reorganize_italian_corpus,merge_postprocessed,postprocess_r1_annotations,reannotate_r1_async,test_r1_annotation}.py` | One coherent Italian/DeepSeek-R1 annotation pipeline, last touched Feb 5-10 2026; Italian dormant since 2026-04-20; `test_r1_annotation.py` hardcodes `/tmp/...` scratch paths | Move as a group into `scripts/_deprecated/italian_r1_pipeline/` | High |
| `k8s/Dockerfile.sweep` | Never built by `.gitlab-ci.yml` (not in its rebuild-trigger list); every sweep job uses stock `python:3.10-slim` or stock PyTorch images | Delete, or add a header noting it was never shipped; fix the 2 doc references that claim it's in use | High |
| `preprocessing/{impoverish_determiners,lemmatize_verbs,remove_articles,remove_expletives,remove_subject_pronominals}.py` | Not imported by `preprocessing/__init__.py` or `ablations/__init__.py`'s active list; no dynamic import anywhere; each has a named successor already live in `ablations/`; `ablations/archived/lemmatize_verbs.py` is a byte-identical duplicate of the top-level file, i.e. this content is dead in *two* places at once | Delete top-level copies; the `ablations/archived/` copies + git history already serve the historical-reference purpose | High |
| `preprocessing/tests/test_remove_articles_integration.py` | Imports `preprocessing.ablations.remove_articles`, which doesn't exist at that path (only `ablations/archived/remove_articles.py` and the dead top-level file do) — will `ModuleNotFoundError` at collection regardless of spaCy | Delete, or repoint import to `ablations.archived.remove_articles` if wanted as a regression fixture | High |
| `model_foundry/logging_components.py` + its 4 dedicated unit tests | 769-line module (StructuredLogger/MetricsLogger/PerformanceLogger/ErrorTracker/WandBLogger); only importers are its own tests. `trainer.py`/`cli.py` actually use `logging_utils.py` + `wandb_init.py` | Delete module + tests, or wire it into `trainer.py` for real; either way correct `model_foundry/README.md`'s logging-architecture section, which currently markets this dead path as the live one | High |
| `analysis/scripts/runners/corpus_descriptives_analysis.R` + 5 siblings (`statistical_models/null_subject_analysis.R`, `analysis_with_models.R`, `figures/generate_overt_only_faceted_figures.R`, `figures/generate_first_epoch_forest_plots.R`, `figures/paper_figures/null_subject_analysis_paper.R`) | All 6 `source()` an identical path (`analysis/scripts/paper_figures/figure_dimensions.R`) that hasn't existed since a reorg moved it to `analysis/scripts/figures/paper_figures/`. Every one fails immediately on run | One-line path fix (safe, since currently 100% broken) if still wanted; else archive | High |
| `analysis/scripts/runners/run_complete_analysis.R` | Calls 6 further scripts at flat paths, 0 of 6 exist (all moved into `statistical_models/`/`pairwise_comparisons/`); root `README.md:132` still tells newcomers to run it (at the wrong path, too) | Delete/rewrite, and fix the root README pointer | High |
| `analysis/scripts/generate_null_subject_report.py` | 845-line Feb-12 version; superseded by the memory-safe `analysis/scripts/reporting/generate_null_subject_report.py` (the one actually invoked by `k8s/job-null-subject-report-90m.yaml` and current docs). Zero references to the top-level copy anywhere | Delete | High |
| 11 root-level `analysis/*.md`/`*.json` reports (APA/COMPREHENSIVE/PURE/STATISTICAL_* + `analysis_audit.*` + `scripts/ANALYSISPLAN.MD` + `scripts/CODE_STRUCTURE.md`) | Single commit, 2025-10-01, never touched since; cite an input CSV and a `null_subject_analysis.R` path that no longer exist; describe a pre-lock study design (six conditions, `exp0_baseline`) superseded by the current locked 30-seed/8-condition/5-arch/2-lang design | Move to `docs/archive/` (matching the project's existing convention) | Medium |
| `analysis/pronoun_recovery/tree_detector/{_categorize_fps,_dump_fps,_example_gold_gaps}.py` | Underscore-prefixed one-off diagnostics for Italian gold-gap remediation, added 2026-04-09, dormant 11 days later when Italian went dormant; nothing imports them | Move into the same archived-Italian-work bucket | Medium |
| `configs/production/baselines/` | Empty directory, never committed; only references are the same stale SLURM/wild-west doc batch | Delete | High |
| `configs/testing/{test_lstm_tiny,test_mamba_tiny,test_checkpoint_scheduling,test_lstm_bidirectional_tiny}.yaml` | Zero genuine references anywhere (the one apparent hit is a false-positive match on an unrelated cache-path comment) | Delete, or move into `tests/fixtures/` if intended as pytest fixtures that never got wired up | Medium |

### Reclassified — flagged as dead by the initial survey, refuted on verification

These are **not** dead code. Listing what the refuting reference actually revealed, since three of them are more urgent than the discarded dead-code framing:

- **`k8s/job-train-tokenizer.yaml`** — still live: `model_foundry/tokenizer/tokenize_dataset.py`'s `FileNotFoundError` remediation message (edited as recently as 2026-06-05) tells users to run this exact job, which mounts the **legacy, frozen** `corpus-analysis-data` PVC instead of `subject-drop-archive`. Two current sweep-baseline configs repeat the same stale instruction. **This is an active operational footgun, not archival clutter** — flagged in §4 as a fix, not a deletion.
- **`k8s/job-europarl-sweep.yaml` + `run_europarl_sweep.py`/`sweep_agent.py`/`sweep_weight_alpha.py` cluster** — still live and still needed: `docs/OSF_PREREGISTRATION.md` lists "Insert Pronouns (Spanish)" as an active, not-yet-executed preregistered condition that depends on exactly this tree-detector + sequence-labeler mechanism; `docs/ablation_verification_report.md` (2026-05-06) tracks it as "pending pipeline build." **This is an open study obligation, not dead code** — worth a status check with the user on when it gets scheduled, not archiving.
- **Evaluation v1 stack** (`evaluation/evaluators/`, `runners/evaluation_runner.py`, `aggregation/`, `core/{model_loader,surprisal_calculator}.py`) — still has one live caller, `k8s/job-eval-baseline-en.yaml`, and `README.md`'s own Step-by-Step still documents `evaluation.runners.evaluation_runner` as the eval command. **But that caller is likely broken today**: `evaluation_runner.py` does `from .summary_generator import SummaryGenerator`, and `summary_generator.py` actually lives in `evaluation/aggregation/`, not `evaluation/runners/` — a plain `ModuleNotFoundError` with no surrounding try/except, under a `set -euo pipefail` job. Worth an actual fix (§6), separate from the doc-currency question in §5.
- **`model_foundry/tests/smoke/test_evaluation.py`** — actually executed (`k8s/job-smoke-test*.yaml` run `pytest model_foundry/tests/smoke/`), and the v1 classes it exercises are still imported by 4 current v2-era test files. Not a deletion candidate; the real gap is that it gives zero coverage of the v2 path (`per_model_runner.py`) production actually runs — add v2 smoke coverage alongside it rather than replacing it.
- **`docs/checkpoint_scheduling.md`** — actually deployed: it's in `mkdocs.yml`'s nav, and `origin/gh-pages` shows it was redeployed *the same day* as this audit. Don't move/delete; instead close the real gap — neither this doc nor its stated successor (`new_checkpoint_scheduling.md`) mentions `model_foundry/checkpoint_schedule.py`, which is the actual current (PI-locked) design. See §5.
- **`scripts/ssrde/train.sh`** — `docs/TRAINING_ON_SLURM.md` is in the live, deployed mkdocs nav pointing straight at it (`sbatch scripts/ssrde/train.sh ...`), unmarked as superseded, unlike sibling docs that do carry a "(superseded)" banner. The issue isn't the script, it's that the live docs site offers a SLURM path CLAUDE.md says doesn't exist anymore — fix the nav/doc, then the script itself is safe to archive.
- **`scripts/{poc_ablation_demo,smoke_analysis_plus_ablations,smoke_three_step}.py`** — partially refuted: `smoke_three_step.py` is named directly in a live config comment (`configs/experiments/experiment_es_remove_expletive_sentences.yaml`), which itself points to a "see CLAUDE.md" pointer that no longer exists in current CLAUDE.md. The other two remain unreferenced anywhere but were **not independently confirmed** in this pass — treat as lower-confidence candidates for a follow-up look, not confirmed dead.

## 4. Reorganization proposal

This is a live production repo — every move below is scoped to be safe standalone, and none touch anything a training/eval pod imports at runtime unless explicitly called out.

**Target additions (no existing paths change):**
```
scripts/_deprecated/
  wild_west/            ← moved (§3)
  ssrde/                ← moved, AFTER mkdocs nav fix (see below)
  italian_r1_pipeline/  ← new subdir for the 6-file R1/Italian cluster
docs/archive/
  analysis_reports_2025-10/   ← the 11 stale analysis/*.md,*.json files
k8s/archive/  (or k8s/completed/)
  ← the 13 one-time smoke-bring-up jobs + 6 migration one-shots, ONLY after
    CLAUDE.md's "Existing job templates to copy from" list (which names
    job-smoke-bert-large.yaml and job-europarl-sweep.yaml by exact path)
    is updated in the same change — moving first breaks those pointers
```

**Move list, in dependency order:**
1. Everything in §3's "Confirmed dead code" table — safe today, zero importers verified, no YAML/CLAUDE.md path dependency.
2. Fix `docs/TRAINING_ON_SLURM.md`'s mkdocs nav entry (mark superseded, matching the banner convention already used for `CORPUS_ANALYSIS_SPEC.md`) → *then* move `scripts/ssrde/train.sh` to `_deprecated/`.
3. Rename `k8s/job-train-tokenizer.yaml` → `job-train-tokenizer-LEGACY.yaml` (or delete) **and** in the same change fix all 3 CLAUDE.md references + the `tokenize_dataset.py` error-message string + the 2 stale sweep-baseline config comments — these must land together or a newcomer follows a dangling/wrong pointer.
4. Fix the broken relative import in `evaluation/runners/evaluation_runner.py` (`from .summary_generator import` → `from ..aggregation.summary_generator import`) — a one-line, purely-additive correctness fix, safe regardless of the v1/v2 documentation question.
5. Only after (3): move the 19 completed smoke/migration `k8s/*.yaml` one-shots into `k8s/archive/`, updating CLAUDE.md's job-template pointers in the same commit.
6. `analyzers/` vs `annotators/` consolidation in `analysis/corpus_descriptives/` — real, proven drift risk (duplicate detector logic already had to be hand-patched twice for the same bug), but this is a genuine engineering task (finish routing CSV/count reports through `aggregate.py` per the project's own `LAYERED_ANNOTATION_ARCHITECTURE.md` design doc), not a file move — schedule as its own piece of work, not a quick win.
7. `fleetview/` — 251 MB of complete, tested, currently *untracked* work, actively diverging from the tracked `scripts/fleet_status.py` it's meant to replace (which has a live 149-line uncommitted diff right now). This needs a PI decision before anything else touches it: commit it (with a real `.gitignore` for `.venv`/`.pytest_cache`) or confirm it's abandoned. Flagging, not proposing, since either answer changes the disposition entirely.

## 5. Documentation plan (priority order)

1. **`docs/DOCUMENTATION_MAP.md` + `docs/STRUCTURE.md` + `docs/README.md` + `mkdocs.yml`** — the only index a newcomer has; refresh in one pass to add `PLAN_BERT_CROSSED_SWEEPS.md` (the actual current master plan, currently absent), `RUN_REGISTRY.md`, `S3_INTEGRATION.md`, and the `audits/`, `incidents/`, `eval_stimuli/` directories that together are a large fraction of current `docs/`.
2. **`docs/preprocessing/README.md`** — highest-urgency single fix: its own Quick Start example (`type="remove_articles"`) throws `KeyError` against the current registry, and its "Available Ablations" table lists zero of the 6 currently-registered ablations. Also fix the `ADVANCED_USAGE.md` → `ADVANCED.md` broken link.
3. **`docs/k8s_jobs.md`** — rewrite as a short index (state `subject-drop-archive` is primary / `corpus-analysis-data` is legacy, matching CLAUDE.md; link out to the good per-subdir READMEs already in `stage1/`, `condition_matched_eval/`, `v2/`) rather than a full per-file table for 205 files that will drift again.
4. **`scripts/SCRIPTS_ORGANIZATION.md`** — currently labels `wild_west`/`ssrde` "DO NOT DELETE" (backwards) and covers ~25 of ~80 current scripts. Rewrite once §3/§4's cleanup lands.
5. **`docs/checkpoint_scheduling.md` / `new_checkpoint_scheduling.md`** — both live and deployed; neither mentions the actual current source of truth (`model_foundry/checkpoint_schedule.py`'s own "Locked design (from the PI)" docstring). Keep both pages (they're intentionally-retained superseded history) but add the missing forward-link.
6. **`evaluation/README.md` + `EVALUATION_PLAN.md`** — README currently calls the dead-in-practice v1 runner "⭐ RECOMMENDED"; rewrite around `per_model_runner.py`/`eval_v2_*` while keeping an accurate note that v1 still has one caller (`job-eval-baseline-en.yaml`) pending the import fix in §4.
7. **`model_foundry/README.md`** — missing `registry.py`/`cache_keys.py`/`checkpoint_schedule.py`/`wandb_init.py`/`rng.py`; carries generic OSS boilerplate (MIT license, GitHub Issues) that doesn't fit a single-researcher study repo.
8. **New `configs/README.md`** — 8 subdirectories, no map of current-vs-legacy; note `configs/experiments/experiment_0_baseline_90M_full.yaml`'s `epochs: 10` contradicts the locked 30-epoch design.

## 6. Quick wins vs structural moves

**Quick wins (safe today, no coordination needed):**
- All of §3's confirmed-dead deletions/moves except the `ssrde`/`job-train-tokenizer` pair (which need a paired doc/code edit — still same-day work, just two files instead of one).
- `.gitignore` additions: `data/eval_results/`, `data/recoverability/`, `data/unigrams/`, `analysis/eval_v2/figures/`, `*.out`.
- Fix the 6 broken `source()` paths in the R scripts (currently 100% non-functional, so zero regression risk).
- Fix `evaluation_runner.py`'s broken relative import (§3, reclassified box).
- Delete `OVERNIGHT_STATUS.md`, `docs/jobpostings.md`, `relaunch_slots.json` (or relocate).
- `k8s/pvc.yaml`: bump the `storage:` value to match the already-live 200Gi (doc-accuracy only; the live PVC is unaffected either way).
- Add a one-line arch-name abbreviation legend for `stage1/`/`condition_matched_eval/`'s short codes.

**Structural (need a quiet window or PI sign-off):**
- Git history rewrite for `data/eval_results/`/`evaluation/results/` (8.2 GB + 3.0 GB) — destructive, invalidates existing clones, needs explicit PI decision on timing and collaborator coordination.
- `fleetview/` commit-vs-archive decision (§4.7).
- `analyzers/`/`annotators/` consolidation in `corpus_descriptives/` (§4.6) — real engineering, not a move.
- Moving completed `k8s/` smoke/migration jobs into an archive subdir, paired with a CLAUDE.md update (§4.5).
- `k8s/condition_matched_eval/`'s inconsistent `-vN` suffixing — check launch scripts for hardcoded filenames before any rename.
- Confirming the Insert-Pronouns/Europarl-sweep pipeline's schedule with the PI (§3 reclassified box) — a scheduling decision, not a repo change.

## 7. Healthy — leave alone

- **`scripts/` module-docstring discipline** — all 81 top-level files have one; `_deprecated/` already correctly isolates 4 earlier hardware-era families, the pattern this audit extends rather than invents.
- **Current production infra** — `production_agent.py` (25 refs), `sweep_agent_lm.py` (40 refs, backs every `configs/sweeps/*.yaml`), `eval_v2_cell.py`/`eval_v2_initialization.py`, `generate_checkpoint_schedule.py`, `launch_production_training.py`, `launch_cell_evals.py`, `compact_registry.py`, `rater_agent.py` — all well cross-referenced, clearly load-bearing.
- **`k8s/` overall discipline** — no binary bloat (2.9 MB, 100% text); every post-merge subdirectory (`stage1/`, `condition_matched_eval/`, `v2/`) ships its own README with exact regeneration commands and safety gates; the `thomas-`/`owner: thomas` convention is followed consistently in everything created after the 2026-04-23 commit.
- **`preprocessing/` active core** — `base.py`/`config.py`/`registry.py`/`utils.py`/`annotate.py`/`dep_labels.py` plus the 6 registered ablation modules are exemplary: strong docstrings, a real `template.py` scaffold, an `archived/` convention already correctly excluded from imports.
- **`evaluation/` v2 stack** (once the v1 doc-currency issue is fixed) — `per_model_runner.py`, `batched_surprisal.py`, `pll_surprisal.py`, `cache.py`, `stimuli_cache.py`, `output_v2.py` are under active development with real matching tests.
- **`analysis/eval_v2/`, `analysis/corpus_descriptives/` core, `analysis/pronoun_recovery/`** — good READMEs, versioned Parquet schemas, dedicated regression tests paired with recent bug fixes (e.g. the ClearNLP/UD dep-label fix landed with `test_dep_labels.py`).
- **`annotation/`** — small, complete, every module docstringed, DB files correctly gitignored, looks finished-and-dormant rather than neglected.
- **`.gitattributes`/LFS scoping** — correctly routes `*.train`/`*.test`/`*.model` through LFS; no accidental bloat there.
- **`evaluation/stimuli/blimp/`** (28 MB) and the `null-subj → null-subj-v2 → null-subj-v2-matched-v1` stimulus lineage — legitimate, documented reproducibility pins (`docs/eval_stimuli/design.md` explicitly calls the versioning intentional archival), not clutter.
- **`Subject-Rearing-Paper/`** as a separate nested git repo — a sane way to keep paper-writing history independent; correctly gitignored from this repo.
