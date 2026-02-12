# Corpus Analysis Changes

## Null Subject Detection Fixes (2026-02-09)

Manual review of 162 annotated sentences found 60 incorrect and 12 unsure, all related to null subject detection. Root causes split into detector bugs (21 cases) and parser quality (14 cases).

### Code Fixes

Three false-positive patterns fixed across four annotator files:
- `analysis/corpus_descriptives/annotators/clause_structure.py`
- `analysis/corpus_descriptives/annotators/fragment.py`
- `analysis/corpus_descriptives/annotators/argument_structure.py`
- `analysis/corpus_descriptives/annotators/complexity.py`

#### Fix 1: Clausal subjects (9 cases)

Subject detection only checked `nsubj`, `nsubj:pass`, and `expl`. Sentences with clausal subjects (`csubj`/`csubj:pass`) like "Running is fun" were incorrectly flagged as null-subject.

Added `csubj`/`csubj:pass` recognition across all four annotators. In `clause_structure.py` and `argument_structure.py`, this sets `subject_status = "clausal"` with full subject token tracking. In `fragment.py` and `complexity.py`, clausal subjects are included in the subject dependency set used for boolean checks.

#### Fix 2: xcomp subject inheritance (7 cases)

Child speech like "I want play dem" — `play` is `xcomp` of `want` with no own `nsubj` child because the subject is shared from the matrix verb. These were incorrectly flagged as null-subject clauses.

When a verb has `subject_status == "none"` (or `"null"` in `argument_structure.py`) and `dep == "xcomp"`, the annotator now checks whether the head verb has a subject. If so, `subject_status` is set to `"inherited"`. This fix only applies to `clause_structure.py` and `argument_structure.py` since `fragment.py` and `complexity.py` only examine ROOT verbs.

#### Fix 3: Disfluent verb reduplication (5 cases)

Disfluencies like "is is fine" — the second verb gets parsed as `ccomp` with no subject. These were incorrectly flagged as null-subject clauses.

When no subject is found and the immediately preceding token has the same lemma and is VERB/AUX, the annotator now sets `subject_status = "disfluent_copy"` (in `clause_structure.py` and `argument_structure.py`) or suppresses the null-subject flag (in `fragment.py` and `complexity.py`).

### Parser Switch: `en_core_web_lg` to `en_core_web_trf`

`en_core_web_lg` systematically mislabels inverted subjects as `attr` or `npadvmod` in short copular questions (e.g., "is today payday?", "was that magic?", "what color is the paper?"). Comparison against `en_core_web_trf` confirmed it handles all 14 cases correctly.

Changed files:
- `configs/analysis/corpus/corpus_analysis_test10m.yaml` — `spacy_model: en_core_web_trf`
- `configs/analysis/corpus/corpus_analysis_train90m.yaml` — `spacy_model: en_core_web_trf`
- `k8s/job-annotate-test-10m.yaml` — CUDA base image, `spacy-transformers` dependency, GPU resource requests
- `k8s/job-annotate-train-90m.yaml` — same

The K8s jobs now use `nvidia/cuda:11.8.0-runtime-ubuntu22.04` as the base image and request `nvidia.com/gpu: 1`. The pipeline already supports GPU via `spacy.prefer_gpu()` in `pipeline.py`.

### Downstream Impact

No downstream code changes required:
- `query.py` line 478 checks `subject_status == "none"` — the new values (`"clausal"`, `"inherited"`, `"disfluent_copy"`) are automatically excluded.
- The Parquet schema uses `pa.string()` for `subject_status`, so new string values are compatible.
- `argument_structure.py` uses `"null"` (not `"none"`) as its default; the flag check `== "null"` at line 156 correctly excludes the new values.

### Tests

Eight new tests added to `analysis/corpus_descriptives/tests/test_annotators.py`:

**TestClauseStructureAnnotator:**
- `test_clausal_subject_not_null` — csubj on copula produces `subject_status = "clausal"`, not a null subject flag
- `test_xcomp_inherits_subject` — xcomp verb inherits subject from matrix verb
- `test_disfluent_reduplication` — reduplicated verb gets `subject_status = "disfluent_copy"`
- `test_genuine_null_subject_unchanged` — regression test: genuine null subject still detected

**TestFragmentAnnotator:**
- `test_clausal_subject_root` — csubj on root does not produce `has_null_subject`
- `test_disfluent_root` — reduplicated root verb does not produce `has_null_subject`

All 62 tests pass (54 existing + 8 new).

### Re-annotation Required

Both corpora (test_10M and train_90M) need re-annotation after these changes to pick up both the code fixes and the improved parses from `en_core_web_trf`.
