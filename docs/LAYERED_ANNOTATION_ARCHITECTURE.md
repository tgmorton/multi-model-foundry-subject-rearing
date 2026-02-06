# Layered Annotation Architecture

## The Problem

The current corpus descriptive pipeline transforms rich per-sentence annotations into aggregated counts, discarding the annotated corpus itself. This is **lossy** — we can't answer emergent research questions that weren't anticipated at aggregation time.

**Example**: We know there are 138 that-trace "violations" (subject extraction + complementizer), but we can't:
- See the actual sentences
- Cross-reference with other annotations (e.g., "Do violations cluster with certain matrix verbs?")
- Use these sentences for downstream tasks (data attribution, influence functions)

## Proposed Architecture

### Core Principle: Annotations as Primary Artifact

```
Raw Corpus
    ↓
┌─────────────────────────────────────────────────────────────┐
│              ANNOTATED CORPUS (Parquet/JSONL)               │
│                                                             │
│  sentence_id | text | genre | speaker | doc_id | ...        │
│  ─────────────────────────────────────────────────────────  │
│  LAYER: spacy_base                                          │
│    tokens, lemmas, pos, deps, morph                         │
│  ─────────────────────────────────────────────────────────  │
│  LAYER: clause_structure                                    │
│    clause_dep, subject_status, matrix_verb, ...             │
│  ─────────────────────────────────────────────────────────  │
│  LAYER: that_trace                                          │
│    is_bridge_complement, comp_present, extraction_type,     │
│    is_violation, ...                                        │
│  ─────────────────────────────────────────────────────────  │
│  LAYER: expletives                                          │
│    has_expletive, expletive_class, expletive_verb, ...      │
│  ─────────────────────────────────────────────────────────  │
│  LAYER: pronouns                                            │
│    pronoun_spans: [{start, end, person, number, case}, ...] │
│  ─────────────────────────────────────────────────────────  │
│  ... (additional layers)                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
Aggregation Scripts (compute summaries from annotations)
    ↓
Summary Statistics (current CSV outputs)
```

### Storage Format: Parquet with Nested Columns

Parquet is ideal because:
- **Columnar**: Fast queries on specific annotation layers
- **Compressed**: Efficient storage for large corpora
- **Nested structures**: Can store lists of spans, token annotations
- **Interoperable**: Works with Python (pandas, polars), R, Spark, DuckDB

**Schema (simplified):**

```python
schema = {
    # Identifiers
    "sentence_id": str,        # Unique ID: {split}_{genre}_{doc_id}_{sent_idx}
    "split": str,              # train_90M, test_10M, pull_10M
    "genre": str,              # BNC, CHILDES, Gutenberg, ...
    "speaker": str,            # For CHILDES: CHI, MOT, FAT, ...
    "doc_id": str,             # Source document identifier
    "sent_idx": int,           # Sentence index within document

    # Raw text
    "text": str,               # Sentence text
    "n_tokens": int,           # Token count

    # SpaCy base layer (could be stored separately for size)
    "tokens": list[str],
    "lemmas": list[str],
    "pos_tags": list[str],
    "dep_rels": list[str],
    "dep_heads": list[int],

    # Clause structure layer
    "clauses": list[{
        "clause_type": str,    # ROOT, ccomp, advcl, xcomp, acl
        "verb_idx": int,       # Index of clause head
        "verb_lemma": str,
        "subject_status": str, # overt, none, expletive
        "subject_idx": int,    # If overt, index of subject head
        "is_finite": bool,
    }],

    # That-trace layer
    "bridge_complements": list[{
        "matrix_verb": str,
        "matrix_verb_idx": int,
        "comp_present": bool,
        "extraction_type": str,  # subject, object, none
        "embedded_subject_status": str,
        "is_violation": bool,    # comp_present AND extraction_type == "subject"
    }],

    # Expletive layer
    "expletives": list[{
        "token_idx": int,
        "class": str,           # existential, weather, raising
        "verb_lemma": str,
    }],

    # Pronoun layer
    "pronouns": list[{
        "token_idx": int,
        "lemma": str,
        "function": str,        # subject, object
        "person": int,
        "number": str,
        "case": str,
    }],

    # Negation layer
    "negations": list[{
        "token_idx": int,
        "position": str,        # pre, post, other
        "subject_status": str,
    }],

    # Sentence-level flags (for fast filtering)
    "has_null_subject": bool,
    "has_expletive": bool,
    "has_that_trace_context": bool,
    "has_that_trace_violation": bool,
    "has_wh_extraction": bool,
    "has_relative_clause": bool,
}
```

### Query Examples

**1. Extract all that-trace violations:**
```python
import polars as pl

df = pl.read_parquet("annotated_corpus.parquet")
violations = df.filter(pl.col("has_that_trace_violation"))
print(violations.select(["sentence_id", "text", "genre"]))
```

**2. Cross-reference: violations by matrix verb:**
```python
violations.explode("bridge_complements").filter(
    pl.col("bridge_complements").struct.field("is_violation")
).group_by(
    pl.col("bridge_complements").struct.field("matrix_verb")
).agg(pl.count())
```

**3. For influence functions — get sentence IDs with property X:**
```python
null_subj_ids = df.filter(
    pl.col("has_null_subject") & (pl.col("genre") == "CHILDES")
).select("sentence_id").to_series().to_list()

# Use these IDs to index into training data for gradient computation
```

**4. R integration (via DuckDB):**
```r
library(duckdb)
con <- dbConnect(duckdb())
dbExecute(con, "INSTALL parquet; LOAD parquet;")

violations <- dbGetQuery(con, "
    SELECT sentence_id, text, genre
    FROM 'annotated_corpus.parquet'
    WHERE has_that_trace_violation = TRUE
")
```

### Implementation Plan

#### Phase 1: Schema Design
- [ ] Finalize annotation schema for each layer
- [ ] Define sentence ID format (must be stable across runs)
- [ ] Decide on storage format (Parquet recommended)

#### Phase 2: Analyzer Refactoring
- [ ] Refactor analyzers to output per-sentence annotations (not just counts)
- [ ] Each analyzer returns: `{sentence_id: annotations_dict}`
- [ ] Aggregation becomes a separate step

**Current analyzer pattern:**
```python
class ThatTraceAnalyzer:
    def process_doc(self, doc, genre, speaker):
        # Increment counters
        self._crosstab[genre][(comp_present, extraction)] += 1
```

**New pattern:**
```python
class ThatTraceAnnotator:
    def annotate_sentence(self, sent, genre, speaker, doc_id, sent_idx) -> dict:
        # Return annotations for this sentence
        return {
            "sentence_id": f"{genre}_{doc_id}_{sent_idx}",
            "bridge_complements": [...],
            "has_that_trace_violation": ...,
        }
```

#### Phase 3: Pipeline Integration
- [ ] Modify pipeline to collect per-sentence annotations
- [ ] Write annotations to Parquet (partitioned by split/genre)
- [ ] Add aggregation step that computes summaries from Parquet

#### Phase 4: Query Interface
- [ ] Python API for common queries
- [ ] R integration via DuckDB or arrow
- [ ] Example notebooks for analysis patterns

### Storage Estimates

For 100M tokens (~10M sentences):
- Raw text: ~500 MB
- Token-level annotations: ~2 GB
- Sentence-level annotations: ~500 MB
- **Total**: ~3 GB (compressed Parquet: ~500 MB - 1 GB)

Partitioned by split/genre for efficient access.

### Benefits

1. **No information loss**: All annotations preserved
2. **Emergent questions**: Query any combination of properties
3. **Data attribution**: Sentence IDs map back to training examples
4. **Influence functions**: Can compute gradients w.r.t. sentence subsets
5. **Reproducibility**: Annotations are deterministic from corpus + spaCy model
6. **Interoperability**: Parquet works with Python, R, SQL, Spark

### Relationship to Current Pipeline

The current pipeline outputs (CSVs with aggregated counts) become **derived artifacts**:

```
Annotated Corpus (Parquet)
    ↓
aggregation/compute_summaries.py
    ↓
analysis/output/corpus_descriptives/data/{split}/*.csv
```

The RMarkdown report reads from CSVs (unchanged), but researchers can also query the Parquet directly for deeper analysis.

---

## Next Steps

1. **Immediate**: Add example extraction to that-trace analyzer for current report
2. **Pipeline re-run**: Implement full layered architecture
3. **Documentation**: Query cookbook with common analysis patterns
