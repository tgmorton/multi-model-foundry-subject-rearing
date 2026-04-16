# Spanish Null Subject Gold Annotation

Gold-set sampling pipeline for Spanish null subject detection. Produces
a stimulus file for upload to the annotation web app at `annotation/`.

## Purpose

Build a hand-verified evaluation set for Spanish null subject detection,
avoiding the gold-label noise that capped Italian Europarl-aligned F1
around 0.90. By over-sampling predicted positives and spreading across
10 registers, we aim for reliable per-category F1 (especially 3sg/3pl).

## Sampling design

Three overlapping samples drawn from the same per-source candidate pools:

| Sample | Target | Method | Purpose |
|--------|--------|--------|---------|
| **IID** | 500 (50/source) | Uniform random within each source | Unbiased F1 on natural distribution |
| **Stratified** | 500 (25 null + 25 overt / source) | Over-sample predicted positives | Per-category diagnostics |
| **IAA** | 100 | Random subset of the union | Inter-annotator κ |

A stimulus that lands in multiple samples is emitted once with
`metadata.sample_types` listing all memberships (e.g. `["iid", "iaa"]`).
Exact-text duplicates are dropped globally (keep first encountered).

Expected total: ~970 stimuli (~30 dropped to dedup).

## Pre-labelling

Stratification requires per-sentence `predicted_null` flags. We use spaCy
`es_core_news_lg` with a five-way subject-status classifier ported from
the Italian pipeline:

- **overt** — finite verb has `nsubj`/`nsubj:pass` child
- **expletive** — has `expl` child
- **clausal** — has `csubj`/`csubj:pass` child
- **inherited** — `xcomp` whose matrix verb has a subject
- **null** — none of the above

Spanish-specific adaptation: when the finite verb is an auxiliary or
copula (`aux`/`aux:pass`/`cop`), we check its head's children for the
subject, since Spanish UD attaches `nsubj` to the content head
(participle in passives, adjective in copulas) rather than the auxiliary.

A sentence is flagged `predicted_null=True` if any of its finite verbs
has subject_status `"null"`. Sentences with zero finite verbs
(fragments, noun phrases, infinitive clauses) are dropped — they would
otherwise flood the "predicted_overt" stratum with useless annotation
targets.

Classifier accuracy is ~75-85% on spot-checks — good enough to enrich
positives. The classifier is not ground truth; annotators provide that.

## Output format

One file: `data/spanish/gold/stimuli.jsonl`, ready for
`POST /api/stimuli/load`.

```json
{
  "text": "Me alegro de la acogida tan favorable de la propuesta.",
  "source": "europarl",
  "context_before": ["Considero que cada país tiene...",
                     "Por eso necesitamos no solo..."],
  "context_after": ["Por esto precisamente el Comisario..."],
  "metadata": {
    "sample_types": ["iid", "iaa"],
    "predicted_null": true,
    "predicted_verbs": [
      {"text": "alegro", "lemma": "alegrar",
       "subject_status": "null", "person": "1", "number": "Sing"}
    ],
    "source_file": "europarl.train",
    "source_line": 17848,
    "spacy_model": "es_core_news_lg-3.7.0",
    "sampler_commit": "<git-sha>"
  }
}
```

`context_before` and `context_after` are **nearest-first** lists —
`context_before[0]` is the sentence immediately preceding the target,
`context_before[1]` is two sentences before, and so on. This matches the
annotation app's `_normalize_context` in `annotation/db.py:178`.

**Metadata is stored in the DB but not rendered in the annotation UI**
(verified: `grep` over `annotation/static/*.html` finds zero references
to `metadata` or `predicted_null`). Safe to include predicted labels
without contaminating annotators.

## Running the sampler

```bash
# Full run (all 10 sources in train_90M):
.venv/bin/python scripts/build_spanish_gold_sample.py

# Dry run on one source:
.venv/bin/python scripts/build_spanish_gold_sample.py --only_source qed

# Custom paths:
.venv/bin/python scripts/build_spanish_gold_sample.py \
    --source_dir data/spanish/train_90M \
    --output_dir data/spanish/gold
```

Takes ~1 minute on an M-series Mac (3000 candidates parsed per source).

### Dependencies

- spaCy 3.7.5
- es_core_news_lg 3.7.0

Install:
```bash
.venv/bin/pip3 install 'spacy==3.7.5'
.venv/bin/python -m spacy download es_core_news_lg
```

## Uploading to the annotation app

```bash
# Start the app locally (from repo root):
python -m annotation

# In another terminal, POST the stimuli (requires admin token from
# annotation app setup):
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
     -F "file=@data/spanish/gold/stimuli.jsonl" \
     http://localhost:8000/api/stimuli/load
```

## Workflow

1. **Run sampler** → `data/spanish/gold/stimuli.jsonl` (~970 stimuli).
2. **Upload to annotation app** via `POST /api/stimuli/load`.
3. **Seed annotator accounts** (two annotators, one admin).
4. **Instruct annotators**: both work through all stimuli; the 100 with
   `metadata.sample_types` containing `"iaa"` will be double-annotated
   for agreement (this happens automatically once both complete them).
5. **Periodically check agreement** via `GET /api/agreement/compute`.
6. **Adjudicate disagreements** via the `/adjudication` admin UI.
7. **Export gold** via `GET /api/export/gold` → JSON ready for model
   evaluation.

## Per-source pool statistics (2026-04-16)

| Source | Pool size | Predicted null rate |
|--------|-----------|---------------------|
| vikidia | 2,068 | 35.8% |
| leipzig_web | 2,913 | 53.7% |
| gutenberg | 2,235 | 62.3% |
| qed | 2,748 | 67.0% |
| opensubtitles | 2,704 | 70.2% |
| spoken | 2,610 | 77.9% |
| childes | 2,671 | — |
| child_narratives | 2,790 | — |
| grerli | 2,696 | — |
| europarl | 2,891 | — |

The null-rate gradient is linguistically plausible: written/expository
registers (vikidia, news) have more overt subjects; oral/conversational
registers (spoken, opensubtitles) have more pro-drop.

## Known limitations

- **Copula/auxiliary edge cases**: the aux/cop head-chain heuristic
  catches most common cases but misses `conj` chains (coordinated
  finite verbs sharing a subject). See Italian `label_aligner.py` for
  a more thorough propagation pass.
- **Morphological mislabelling**: spaCy sometimes tags `llego` as 3sg
  instead of 1sg when context is missing. This affects the
  `predicted_verbs` field but not the binary `predicted_null` flag.
- **No weather/impersonal filter**: "llueve", "hay", "se dice que..."
  currently all count as null subjects. Annotators resolve these.

## Files

- `scripts/build_spanish_gold_sample.py` — the sampler
- `data/spanish/gold/stimuli.jsonl` — output (gitignored)
- `docs/spanish_gold_annotation.md` — this file
- `annotation/` — the annotation web app (unchanged)
