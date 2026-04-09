# Italian Null-Subject Detection & Longitudinal Analysis

Italian is a consistent null-subject (pro-drop) language: finite verbs routinely
appear without an overt subject pronoun, with person and number recoverable
from verbal morphology. This document covers the Italian side of the pronoun
recovery stack — a tabular tree detector trained on cross-lingually aligned
gold data — and the longitudinal CHILDES analysis that uses its output.

For the English-side neural pipeline see `docs/pronoun_recovery.md`. For the
TED2020 + Tatoeba + QED corpus ("TTQ") used as a secondary training domain,
see `docs/ttq_corpus_report.md`.

## 1. Overview

### Why it matters

Controlled-rearing experiments need a way to (a) *detect* which Italian clauses
have dropped subjects and (b) *place* a recovered pronoun in the right surface
position so that downstream LMs see the same clause structure regardless of
language. Italian splits these cleanly into two sub-problems:

1. **Detection** — given a finite verb, was a subject pro-dropped? A tabular
   tree detector over morphosyntactic features handles this well.
2. **Placement** — where in the string should the recovered pronoun be
   inserted? The tree detector identifies the verb but not the insertion
   point. BERT-style sequence labelers are better suited to placement; this
   is tracked as future work (see user memory, "Italian Null Subject
   Detection — Next Steps").

### Data sources

| Source | Description | Role |
|---|---|---|
| **Europarl EN-IT** | `awesome-align`-based cross-lingual alignment of English subject pronouns onto Italian null-subject verbs | Primary gold labels |
| **TTQ** | TED2020 + Tatoeba + QED EN-IT parallel data, same alignment pipeline | Secondary / larger domain (≈503K rows) |
| **CHILDES (BabyLM en)** | Child-directed + child-produced speech, used downstream for longitudinal analysis | Inference target |

Gold labels are of the form `PRO.{person}{number}` (e.g. `PRO.1sg`, `PRO.3pl`)
or `NONE`. Labels come from the alignment layer in
`analysis/pronoun_recovery/tree_detector/label_aligner.py`.

## 2. Tree detector pipeline

### Architecture

Two-stage rule-based classifier (`analysis/pronoun_recovery/tree_detector/inference.py:24`):

1. **Binary detection.** A `DecisionTreeClassifier` and a
   `HistGradientBoostingClassifier` are trained on ordinal-encoded tabular
   features extracted per finite verb. Both are trained in parallel; one is
   selected as "primary" via `classifier_type`.
2. **Person / number.** Read directly from the predicted verb's spaCy
   morphology (`Person`, `Number`) and mapped via `MORPH_TO_LABEL_SUFFIX` to
   a `PRO.{person}{number}` label. Stage 2 is deterministic — no second
   classifier.

Feature extraction (`analysis/pronoun_recovery/tree_detector/feature_extractor.py`)
runs the full `CompositeAnnotator` suite from
`analysis.corpus_descriptives.annotators` once per sentence and produces one
`VerbFeatureRow` per finite verb. Categorical fields are ordinal-encoded at
train time and the same encoder is persisted for inference.

Key features (by importance, post gold-gap remediation):

- `has_reachable_subject` (~68%)
- `is_copular_impersonal` (~8.4%)
- `verb_mood` (~6.7%)
- `head_has_overt_nsubj` (~3.0%)

Detection-only columns (`verb_person`, `verb_number`) are excluded from the
detector's feature set so that Stage 1 does not see the very morphology that
Stage 2 relies on — enforced via `_DETECTION_EXCLUDE_COLS` in the trainer.

### CLI commands

All four commands are defined in `analysis/pronoun_recovery/run.py` and dispatch
on a `TreeDetectorConfig` or `TreeCrossEvalConfig`:

| Command | File:line | Purpose |
|---|---|---|
| `tree-extract` | `run.py:343` | Parse gold JSONL with spaCy, run feature extractor, align gold markers to verbs, save `features.parquet` + `labels.npy` |
| `tree-train` | `run.py:372` | Fit ordinal encoder, train DT + HGB, evaluate, save joblib models + reports |
| `tree-predict` | `run.py:419` | Load a trained model and predict null subjects for a single `--text` or an `--input` file |
| `tree-cross-eval` | `run.py:460` | Cross-domain training/eval sweep over multiple domains (see Section 3) |

Typical invocation:

```bash
python -m analysis.pronoun_recovery.run tree-extract \
  -c configs/analysis/pronoun_recovery/pronoun_recovery_it_tree_detector_ttq.yaml -v

python -m analysis.pronoun_recovery.run tree-train \
  -c configs/analysis/pronoun_recovery/pronoun_recovery_it_tree_detector_ttq.yaml -v
```

### Configuration

`TreeDetectorConfig` (`analysis/pronoun_recovery/config.py:476`):

| Field | Default | Notes |
|---|---|---|
| `aligned_data_path` | required | JSONL checkpoint from the alignment step (Europarl or TTQ) |
| `output_path` | `data/pronoun_recovery/tree_detector/it` | Holds `features.parquet`, `labels.npy`, `dt_model.joblib`, `hgb_model.joblib`, `feature_encoder.joblib`, `pipeline_config.joblib`, reports |
| `language` | `it` | |
| `it_spacy_model` | `it_core_news_lg` | |
| `spacy_batch_size` | `50` | |
| `classifier_type` | `decision_tree` | Primary for export; both DT and HGB are always trained |
| `max_depth`, `min_samples_leaf` | `None`, `5` | DT hyperparameters |
| `n_estimators`, `learning_rate`, `gb_max_depth` | `200`, `0.1`, `None` | HGB hyperparameters |
| `cv_folds`, `test_fraction`, `seed` | `5`, `0.2`, `42` | |
| `min_detection_f1` | `0.80` | Quality gate — logs a warning if not met |
| `min_feature_accuracy` | `0.95` | Quality gate on Stage 2 morphology |

`TreeCrossEvalConfig` (`analysis/pronoun_recovery/config.py:538`):

| Field | Default | Notes |
|---|---|---|
| `domain_paths` | required | `dict[name -> dir]`; each dir must contain `features.parquet` + `labels.npy` |
| `output_path` | `data/pronoun_recovery/tree_detector/cross_eval` | |
| `ttq_sweep_sizes` | `[20000, 50000, 100000, 200000, 0]` | `0` = use all available TTQ training rows |
| `test_fraction`, `seed`, `cv_folds` | `0.2`, `42`, `5` | |
| `min_samples_leaf`, `n_estimators`, `learning_rate`, `gb_max_depth`, `max_depth` | `5`, `200`, `0.1`, `None`, `None` | |

### Outputs

Default Italian (Europarl) output tree at `data/pronoun_recovery/tree_detector/it/`:

```text
features.parquet          # X
labels.npy                # y
dt_model.joblib           # DecisionTreeClassifier
hgb_model.joblib          # HistGradientBoostingClassifier
feature_encoder.joblib    # OrdinalEncoder
pipeline_config.joblib    # {prefilter_col, detection_exclude_cols, tree_feature_names}
dt_rules.txt              # sklearn.tree.export_text
*_feature_importance.json
model_report.md           # cross-validated metrics + per-label recall
```

TTQ features land in `data/pronoun_recovery/tree_detector/it_ttq/`.

### Quality gates

Enforced in `tree-train` via `check_quality_gates` against the selected
classifier's metrics (`run.py:407`):

- `min_detection_f1 = 0.80` — macro F1 on the held-out positive class
- `min_feature_accuracy = 0.95` — Stage 2 person/number accuracy

Gates log a warning when missed but do not abort training.

## 3. Cross-domain evaluation

`tree-cross-eval` (`analysis/pronoun_recovery/tree_detector/cross_evaluator.py`)
runs a matrix of training configurations against a matrix of test sets:

1. Each declared domain is split 80/20 stratified on the binarised label.
2. Test sets: one per domain plus a `combined` concatenation.
3. Training configurations:
   - `europarl` — the Europarl train split alone
   - `ttq_{size}k` or `ttq_all` — one config per entry in `ttq_sweep_sizes`
     (stratified subsample of TTQ train)
   - `combined` — Europarl train + TTQ train down-sampled to match Europarl size
4. For each configuration, both DT and HGB are trained from scratch and
   evaluated on every test set. Models, encoders, rules, and feature
   importance are saved to `models/{config_name}/` under `output_path`.

### Interpreting the report

Two artefacts land in `output_path`:

- `cross_eval_report.json` — full nested metrics
- `cross_eval_summary.md` — markdown tables:
  - **HGB Detection F1** — one row per training config, one column per test
    set. Look for rows where in-domain F1 is high *and* out-of-domain F1
    holds up. The `combined` row is usually the best cross-domain generaliser.
  - **DT Detection F1** — same layout; the DT row tells you how much the
    boosted ensemble is buying you.
  - **Per-Label Recall (HGB)** — per-label recall broken down by test set
    and training config, with support. This is where label noise shows up:
    classes with support < ~30 should not drive conclusions.

### TTQ size sweep

`ttq_sweep_sizes` lets you trace F1 as a function of TTQ training data volume.
`0` is the sentinel for "use all TTQ training data". The sweep is what
established that the combined HGB configuration plateaus around 0.90 F1 on
the 90M-parseable TTQ data.

## 4. Gold-gap remediation (the precision boost)

The gold alignment pipeline had a systematic precision problem: English
parliamentary register avoids 1st-person pronouns via passivisation
(*"it is declared"* vs Italian *"dichiaro"*), so EN-IT alignment never sees a
pronoun to align and the Italian null subject is silently left unlabelled.
The detector then learned to treat these as `NONE`, tanking recall on 1p/2p.

Two label-correction passes in `label_aligner.py` fix this before features
are written to disk:

### Pass 1 — Structural propagation

`_propagate_labels_structurally` (`label_aligner.py:87`) walks along two kinds
of chains and copies any existing gold label to unlabelled siblings:

- **aux / cop chains** — `aux`, `aux:pass`, `cop` dependents share the
  matrix verb's dropped subject.
- **conj chains** — coordinated verbs (`conj`) share the subject of the
  first conjunct; the pass also propagates through each conjunct's own
  aux/cop children.

Impact: 507 labels propagated across the Europarl set (previously only 28
when the pass covered aux/cop alone).

### Pass 2 — 1p/2p morphological heuristic

`_relabel_1p2p_morphological_heuristic` (`label_aligner.py:186`) walks every
finite verb and relabels `NONE → PRO.{person}{number}` when *all* of the
following hold:

- Person is 1 or 2
- Mood is not imperative
- Dep is not `xcomp` (subject is inherited, not dropped)
- No overt subject on the verb *or* its head for aux/cop tokens (checked via
  `_has_overt_subject`)

Rationale: Italian 1p/2p pro-drop is near-categorical, so an un-matched 1p/2p
finite verb with no overt subject is almost always a gold gap rather than a
true `NONE`. Impact: 1,830 relabellings in the Europarl set.

### Combined impact

| Metric | Before | After |
|---|---|---|
| DT detection F1 | 0.802 | 0.897 |
| HGB detection F1 | 0.822 | 0.905 |
| DT false positives | 596 | 281 |
| Precision | — | +16 pp |
| DT PRO.3sg recall | 57% | 74% |
| HGB PRO.3sg recall | 57% | 81% |
| PRO.2sg recall | (no support) | 90% (n=38) |

Full analysis:
`data/pronoun_recovery/tree_detector/it/fp_analysis/gold_gap_report.md`.

## 5. Longitudinal CHILDES analysis

Once an Italian detector exists, the same tabular features (and a
structurally analogous English pipeline) feed a per-child longitudinal
analysis of null-subject rates in CHILDES.

### Per-child identity reconstruction

Child identity was present in `line_cleaners.py` metadata but dropped by
`pipeline.py` before it reached the stored parquets. The boundary markers
are still in the raw `.train` file, so per-child identity was reconstructed
into `data/output/train_90M/childes_child_mapping.parquet`
(4.6M sentences, 246 children, 87 CHILDES sub-corpora).

### Imperative reconstruction

Posthoc imperative detection (`imperative_heuristic` + `nonroot_imperative`)
was added after the 90M parquets were written. `reconstruct_imperatives.py`
replays the detection logic on stored parse features to produce
`childes_null_type.parquet` with a `null_type` column: `imperative`,
`non_imperative_null`, or `has_subject`.

Script: `analysis/scripts/reporting/reconstruct_imperatives.py`.

Observed split in finite CHILDES clauses: 62,444 imperative vs 61,062
non-imperative null (~50/50). Imperatives decline faster (9.6% → 1.8%) than
non-imperative nulls (4.6% → 1.6%) across 12–72 months.

### Two-group split

Children with ≥20 early finite clauses are classified by their pooled 12–35m
null rate at a threshold of 6%:

| Group | N | Trajectory |
|---|---|---|
| Decliners | 28 | 14% → 4.2% (classic exponential decay) |
| Adult-like | 21 | ~3–4% flat from first observation, consistently below the adult baseline |

Adults interacting with adult-like children also show a lower null rate
(3.2% vs 4.7% at 24–35m) and higher MLU (6.66 vs 5.95), suggesting caregiver
register tracks child group.

### Headline statistical results

- **Clause-level OLS** (per-clause null, linear in age): slope = −0.0016 / month,
  *t*(417,648) = −52.45, *p* < .001, *R²* = .008.
- **MLU weighted least squares**: slope = 0.063 MLU units / month,
  *t*(3) = 23.15, *p* < .001, *R²* = .994.
- **Exponential decay** (`a·exp(−b·x) + c`): *R²* = .995, half-life 7.4 months,
  asymptote 3.3%.

All model fits are produced by
`analysis/scripts/reporting/null_subject_statistical_models.py` (OLS, WLS,
curve fits, chi-squared) and land in `data/output/train_90M/model_fits.md`.

### Key data files (`data/output/train_90M/`)

| File | Contents |
|---|---|
| `childes_child_mapping.parquet` | `sent_idx → (child_id, corpus)` |
| `child_longitudinal_null_rates.parquet` | per-child per-month null rate |
| `childes_null_type.parquet` | per-clause imperative vs non-imperative null |
| `null_subject_report.html` | 5-section narrative + methodology |
| `null_subject_technical_report.md` | all stats for write-up |
| `model_fits.md` | curve fits, regression, chi-squared |
| `null_subject_by_age.{pdf,png}` | exponential decay figure (3.25×2.5 in) |
| `mlu_by_age.{pdf,png}` | MLU linear fit |

### Scripts (`analysis/scripts/reporting/`)

| Script | Role |
|---|---|
| `reconstruct_imperatives.py` | Rebuild imperative tags from stored parse features |
| `generate_longitudinal_figures.py` | Shared data loading, group classification, publication figures |
| `null_subject_statistical_models.py` | All statistical models (OLS, WLS, curve fits) |
| `generate_null_subject_figures.py` | Publication figures (3.25×2.5 in) |
| `generate_null_subject_report.py` | Assembles the HTML report |

## 6. Quick start

From the repo root, build the Italian TTQ detector end-to-end and predict
a single sentence:

```bash
# 1. Extract features from the aligned TTQ gold data
python -m analysis.pronoun_recovery.run tree-extract \
  -c configs/analysis/pronoun_recovery/pronoun_recovery_it_tree_detector_ttq.yaml -v

# 2. Train DT + HGB, save models, check quality gates
python -m analysis.pronoun_recovery.run tree-train \
  -c configs/analysis/pronoun_recovery/pronoun_recovery_it_tree_detector_ttq.yaml -v

# 3. Predict on a single Italian sentence
python -m analysis.pronoun_recovery.run tree-predict \
  -c configs/analysis/pronoun_recovery/pronoun_recovery_it_tree_detector_ttq.yaml \
  --text "Vado al mercato e compro il pane."

# 4. Cross-domain sweep across Europarl + TTQ (requires step 1 to have been
#    run for both domains)
python -m analysis.pronoun_recovery.run tree-cross-eval \
  -c configs/analysis/pronoun_recovery/pronoun_recovery_it_tree_cross_eval.yaml -v
```

Prediction output is one JSON record per detected null subject, carrying
`verb_text`, `verb_char_offset`, `token_idx`, `verb_lemma`, `label`
(e.g. `PRO.1sg`), `confidence`, and `lexical_form` (the default insertion
pronoun from `IT_DEFAULT_PRONOUN`).
