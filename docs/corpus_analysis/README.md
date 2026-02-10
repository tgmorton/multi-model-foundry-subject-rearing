# Corpus Analysis

Sentence-level linguistic annotation and analysis pipeline for multi-genre English and Italian corpora. Processes raw text through spaCy, extracts syntactic phenomena (null subjects, that-trace contexts, expletives, wh-extraction, etc.), and writes structured Parquet output for downstream querying and statistical analysis.

## Quick Start

```bash
# Annotate a corpus split
python -m analysis.corpus_descriptives.run \
  --config configs/corpus_analysis_train90m.yaml \
  --annotate

# Run aggregate analyses over annotated data
python -m analysis.corpus_descriptives.run \
  --config configs/corpus_analysis_train90m.yaml
```

```python
# Query annotations
from analysis.corpus_descriptives.query import AnnotatedCorpus

corpus = AnnotatedCorpus("data/output/train_90M/annotated_corpus/train_90M.parquet")
null_subj = corpus.get_null_subject_sentences(genre="CHILDES_child")
```

## System Overview

The pipeline has two modes:

1. **Annotation mode** (`--annotate`) — Runs 12 annotators over each sentence and writes per-sentence Parquet files. This is the primary mode for producing data.
2. **Analyzer mode** (default) — Runs 8 analyzers that accumulate genre-stratified counts. Outputs JSON and CSV summaries. Useful for quick corpus statistics without preserving sentence-level detail.

Both modes share the same spaCy frontend, config format, and line-cleaning infrastructure.

## Directory Structure

```
analysis/corpus_descriptives/
├── run.py                  CLI entry point
├── pipeline.py             CorpusAnalysisPipeline + CorpusAnnotationPipeline
├── config.py               Pydantic config model
├── query.py                AnnotatedCorpus query interface
├── corpus_analysis.py      60+ analysis functions for annotated data
├── schema.py               PyArrow schemas (base + 12 layers)
├── output.py               Parquet/JSON/CSV writers
├── constants.py            Bridge verbs, WH lemmas, speaker codes, etc.
├── line_cleaners.py        Genre-specific text preprocessing
├── annotators/
│   ├── base.py             BaseSentenceAnnotator, CompositeAnnotator
│   ├── clause_structure.py
│   ├── argument_structure.py
│   ├── that_trace.py
│   ├── expletive.py
│   ├── pronoun.py
│   ├── negation.py
│   ├── wh_extraction.py
│   ├── relative_clause.py
│   ├── verb.py
│   ├── topic.py
│   ├── complexity.py
│   └── fragment.py
├── analyzers/
│   ├── base.py             BaseAnalyzer (counter-based)
│   └── [8 analyzer modules]
└── tests/
    ├── conftest.py          Fixtures, make_doc() helper
    └── test_annotators.py
```

## Configuration

YAML config files live in `configs/`. Example:

```yaml
input_path: /mnt/data/raw/train_90M/
output_path: /mnt/data/output/train_90M/
split_name: train_90M
spacy_model: en_core_web_trf
spacy_batch_size: 256
spacy_disable_components:
  - ner
chunk_size: 5000
language: en
checkpoint_dir: /mnt/data/checkpoints/train_90M/
checkpoint_interval: 10000
genre_map:
  childes: CHILDES
  bnc_spoken: BNC
  gutenberg: Gutenberg
  open_subtitles: OpenSubtitles
  simple_wiki: SimpleWikipedia
  switchboard: Switchboard
```

Key fields:

| Field | Description |
|---|---|
| `input_path` | Directory containing `.train` files, one per genre |
| `output_path` | Where results and Parquet files are written |
| `split_name` | Identifier used in sentence IDs and filenames |
| `spacy_model` | spaCy model to load (`en_core_web_trf` recommended for English) |
| `language` | `en` or `it` — controls language-specific annotator behavior |
| `genre_map` | Maps `.train` filename stems to display names |
| `checkpoint_dir` | Enables resumable runs (optional) |

The config is validated by Pydantic (`CorpusAnalysisConfig` in `config.py`), which handles path resolution, type coercion, and defaults.

## Annotators

Each annotator receives a spaCy `Span` (sentence) and returns a dict of annotations plus boolean flags for fast filtering.

| Annotator | Output key | Flags |
|---|---|---|
| `clause_structure` | `clauses` | `has_null_subject` |
| `argument_structure` | `argument_structure` | `has_null_subject`, `has_null_object`, `has_ditransitive` |
| `that_trace` | `bridge_complements` | `has_that_trace_context`, `has_that_trace_violation` |
| `expletive` | `expletives` | `has_expletive` |
| `pronoun` | `pronouns` | `has_overt_subject_pronoun`, `has_overt_object_pronoun` |
| `negation` | `negations` | `has_negation` |
| `wh_extraction` | `wh_extractions` | `has_wh_extraction` |
| `relative_clause` | `relative_clauses` | `has_relative_clause` |
| `verb` | `verbs` | — |
| `topic` | `topic_info` | — |
| `complexity` | `complexity` | `is_fragment`, `is_imperative` |
| `fragment` | *(scalar fields)* | `is_fragment`, `is_imperative`, `has_null_subject` |

All annotators implement `BaseSentenceAnnotator` (in `annotators/base.py`), which defines:
- `annotate_sentence(sent, genre, speaker=None, metadata=None) -> dict`
- `get_sentence_flags(annotations) -> dict`

Language-aware annotators (`expletive`, `that_trace`, `wh_extraction`, `relative_clause`, `fragment`) accept a `language` parameter and swap constants (bridge verbs, WH lemmas, relativizers) accordingly.

### Subject Status Values

The `clause_structure` and `argument_structure` annotators assign a `subject_status` to each clause/verb:

| Value | Meaning |
|---|---|
| `overt` | Lexical or pronominal subject present (`nsubj`/`nsubj:pass`) |
| `expletive` | Expletive subject (`expl`) |
| `clausal` | Clausal subject (`csubj`/`csubj:pass`) |
| `inherited` | xcomp verb whose subject is shared from the matrix verb |
| `disfluent_copy` | Disfluent verb reduplication (e.g., "is is fine") |
| `none` / `null` | No subject found (genuine null subject candidate) |

Note: `clause_structure` uses `"none"` as its default; `argument_structure` uses `"null"`. The flag checks (`== "none"` / `== "null"`) correctly exclude the non-null values above.

## Output Format

### Layered Parquet (default)

Annotations are written as joinable Parquet files, split by layer to avoid wide sparse tables:

```
annotated_corpus/
├── base/
│   └── train_90M.parquet        # sentence_id, text, tokens, POS, deps, genre, speaker, ...
├── layers/
│   ├── clause_structure/
│   │   └── train_90M.parquet    # sentence_id + clauses list + flags
│   ├── pronouns/
│   │   └── train_90M.parquet
│   ├── ...
│   └── complexity/
│       └── train_90M.parquet
└── metadata.json
```

All layer files share `sentence_id` as a join key. The base file contains token-level arrays (`tokens`, `lemmas`, `pos_tags`, `dep_rels`, `dep_heads`) plus sentence metadata (`genre`, `speaker`, `role`, `child_age_months`).

Sentence IDs have the format: `{split}_{genre}_{file_idx:06d}_{sent_idx:06d}`

### Flat Parquet (alternative)

Set `layered_output: false` in the config to write a single wide Parquet file with all columns.

### Analyzer Mode Output

JSON and CSV files per analyzer, with `overall` and `by_genre` breakdowns:

```
output/
├── clause_structure.json
├── pronoun_inventory.json
├── ...
├── results.json       # combined
└── metadata.json
```

## Query Interface

`query.py` provides `AnnotatedCorpus` for querying Parquet annotations with Polars:

```python
from analysis.corpus_descriptives.query import AnnotatedCorpus, load_layered_corpus

corpus = AnnotatedCorpus("annotated_corpus/train_90M.parquet")

# Convenience methods
corpus.get_null_subject_sentences(genre="CHILDES_child", finite_only=True)
corpus.get_that_trace_violations()
corpus.get_expletive_sentences(expletive_class="existential")
corpus.get_wh_questions(extraction_type="subject")
corpus.get_relative_clauses(rel_type="subject")
corpus.get_fragments()
corpus.get_imperatives()

# Flexible filtering
corpus.filter(has_null_subject=True, has_negation=True)

# Statistics
corpus.count_by_genre("has_null_subject")
corpus.summary()

# Sampling
corpus.sample(n=100, seed=42, has_null_subject=True)
```

For layered output, use `load_layered_corpus` to join base + selected layers:

```python
df = load_layered_corpus("annotated_corpus/", layers=["clause_structure", "pronouns"])
```

## Analysis Functions

`corpus_analysis.py` contains 60+ pre-built analysis functions organized by topic:

- **Corpus overview** — sentence/token counts by genre and speaker role
- **Phenomenon rates** — proportions for all boolean flags, stratified by genre
- **Co-occurrence** — null subject + negation, expletive + embedding depth, etc.
- **CHILDES developmental** — child vs. adult rates, age-binned trajectories, MLU by month
- **Pronoun distribution** — inventory by function/person/number/case, subject pronoun rates
- **That-trace** — bridge verb distribution, complementizer rates, clause structure cross-tabs
- **Sentence complexity** — embedding depth distribution, coordination rates
- **Verb analysis** — null subject rates by verb lemma, TAM distributions, morphological paradigms
- **Register comparison** — spoken vs. written phenomenon rates
- **Cross-linguistic** — language-agnostic metrics for EN/IT comparison

```python
from analysis.corpus_descriptives.corpus_analysis import run_all_analyses

results = run_all_analyses(df, output_dir="analysis_output/")
```

## Line Cleaning

Genre-specific preprocessing is handled by cleaners in `line_cleaners.py`:

- **CHILDESCleaner** — Strips `*SPK:\t` speaker labels, removes `[...]` annotation brackets, extracts speaker codes (CHI, MOT, FAT, etc.), loads age metadata from filenames or directory structure. Supports both English and Italian CHILDES formats.
- **SwitchboardCleaner** — Strips `A:\t` / `B:\t` speaker labels.
- **Generic** — Removes brackets, collapses whitespace.

Speaker codes are classified into child (`CHI`) and adult (`MOT`, `FAT`, `INV`, etc.) roles, which become the `role` field in output. For CHILDES, `child_age_months` is extracted from filename-encoded ages or metadata files.

## Language Support

The system supports English (`en`) and Italian (`it`). Language-specific behavior:

- **Constants** — Separate bridge verb lists, WH lemma sets, relativizer sets, genre maps
- **Annotators** — `expletive`, `that_trace`, `wh_extraction`, `relative_clause`, and `fragment` accept a `language` parameter
- **Line cleaning** — Italian CHILDES has different boundary markers and corpus structure
- **Analysis** — `language_agnostic_analysis()` and `cross_linguistic_comparison()` enable EN/IT comparison

To annotate Italian:

```yaml
language: it
spacy_model: it_core_news_lg
genre_map:
  clta: CLTA
  corpus_isacco: CorpusIsacco
  ...
```

## Kubernetes Deployment

K8s job manifests live in `k8s/`:

- `job-annotate-test-10m.yaml` — Annotate the 10M-word test split
- `job-annotate-train-90m.yaml` — Annotate the 90M-word training split
- `job-analysis-test-10m.yaml` — Run aggregate analysis over annotated test data

Annotation jobs use `nvidia/cuda:11.8.0-runtime-ubuntu22.04` with GPU resources for `en_core_web_trf`. Analysis jobs are CPU-only (Polars over Parquet).

Each job:
1. Clones the repo via an init container
2. Installs dependencies (`spacy`, `spacy-transformers`, `polars`, `pyarrow`, etc.)
3. Generates CHILDES metadata (`scripts/generate_childes_metadata.py`)
4. Runs the pipeline

## Testing

Tests use `spacy.blank("en")` with manually constructed parse trees (`make_doc` helper in `conftest.py`), so no model download is required.

```bash
# Run all annotator tests
pytest analysis/corpus_descriptives/tests/test_annotators.py -v

# Run specific test classes
pytest analysis/corpus_descriptives/tests/test_annotators.py -v -k "ClauseStructure or Fragment"
```

The `make_doc` helper builds spaCy `Doc` objects with explicit token attributes:

```python
from analysis.corpus_descriptives.tests.conftest import make_doc

doc = make_doc(
    blank_nlp,
    words=["running", "is", "fun"],
    pos=["VERB", "AUX", "ADJ"],
    deps=["csubj", "ROOT", "acomp"],
    heads=[1, 1, 1],
    lemmas=["run", "be", "fun"],
    morphs=["VerbForm=Ger", "VerbForm=Fin", None],
)
```

## Dependencies

Core dependencies (from `requirements.txt`):

| Package | Purpose |
|---|---|
| `spacy` | Tokenization, POS tagging, dependency parsing |
| `spacy-transformers` | Transformer-based spaCy models (`en_core_web_trf`) |
| `pydantic` | Config validation |
| `pyarrow` | Parquet schema definition and writing |
| `polars` | Query interface (lazy DataFrames over Parquet) |
| `pyyaml` | Config file parsing |
| `tqdm` | Progress bars |
