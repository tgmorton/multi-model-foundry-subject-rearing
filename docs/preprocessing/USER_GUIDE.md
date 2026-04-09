# Preprocessing User Guide

Complete guide to using linguistic ablations for corpus processing.

## Understanding Ablations

Linguistic ablations systematically remove or modify language features to test
model learning. By creating controlled variations of training data, you can
investigate which features models rely on and how they acquire linguistic
knowledge.

**Example research questions:**

- Do expletive constructions ("it is raining", "there are problems") affect
  whether models acquire a null-subject parameter?
- How does morphological case impoverishment affect argument-structure learning?
- Does enriching verbal morphology (agreement clitics) change subject-drop
  acquisition?

## Basic Usage

### Process a Corpus

```python
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

config = AblationConfig(
    type="remove_expletive_sentences_en",
    input_path="data/raw/train_90M/",
    output_path="data/processed/exp/",
    spacy_model="en_core_web_trf",
    seed=42,
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

The pipeline:

1. Finds all `.train` files in `input_path` (recursively).
2. Applies the registered ablation to each file.
3. Writes output preserving directory structure.
4. Generates `ABLATION_MANIFEST.json` with provenance.

### With Replacement Pool

Corpus-deletion ablations (like `remove_expletive_sentences_*`) shrink the
corpus. A replacement pool backfills text to preserve token count:

```python
config = AblationConfig(
    type="remove_expletive_sentences_en",
    input_path="data/raw/train_90M/",
    output_path="data/processed/exp/",
    replacement_pool_dir="data/raw/pull_10M/",
    spacy_model="en_core_web_trf",
    seed=42,
)
```

When lines are removed, the pipeline samples unused sentences from the pool,
runs them through the same ablation, and appends the survivors until the
file reaches its original token count. Pool draw statistics are recorded in
the provenance manifest (see `preprocessing/base.py:433`).

## Available Ablations

The active registry is populated by `preprocessing/ablations/__init__.py`.
Only the ablations listed here are registered.

| Name                              | Language | Module                                           |
|-----------------------------------|----------|--------------------------------------------------|
| `remove_expletive_sentences_en`   | English  | `preprocessing/ablations/remove_expletive_sentences.py` |
| `remove_expletive_sentences_it`   | Italian  | `preprocessing/ablations/remove_expletive_sentences.py` |
| `impoverish_case_en`              | English  | `preprocessing/ablations/impoverish_case.py`     |
| `impoverish_case_it`              | Italian  | `preprocessing/ablations/impoverish_case.py`     |
| `lemmatize_verbs`                 | English  | `preprocessing/ablations/lemmatize_verbs.py`     |
| `enrich_verbal_morphology`        | English  | `preprocessing/ablations/enrich_verbal_morphology.py` |

### remove_expletive_sentences_en

Removes entire lines that contain an English expletive construction. Unlike
naive token-level removal (archived in `preprocessing/ablations/archived/`),
this ablation drops the whole sentence and uses a replacement pool to keep
corpus size constant.

Detection uses a three-tier cascade
(`preprocessing/ablations/remove_expletive_sentences.py:57`):

1. **Tier 1 — spaCy `dep_ == 'expl'`**: existential-*there*, expletive-*it*
   marked by the parser. Always removed.
2. **Tier 2 — heuristic weather / raising-*it***: "it" as `nsubj` of a verb
   in `WEATHER_VERBS`, a verb in `RAISING_VERBS` with a clausal complement,
   or a copula + adjective in `RAISING_ADJECTIVES` with a clausal complement.
   Sub-categorised as `tier2_weather`, `tier2_raising`, `tier2_copular`.
3. **Tier 3 — optional coreference confirmation**: if a `coref_model` is
   configured, heuristic candidates are only removed when coreference
   resolution finds no antecedent cluster for the "it". Referential "it"
   is kept (counted as `tier3_coref_kept`).

Document boundary markers of the form `= = = ... = = =` are passed through
untouched and reset the coref context buffer
(`remove_expletive_sentences.py:45`).

```python
# Input:  "It is raining. The report was late. It arrived yesterday."
# Output: "The report was late. It arrived yesterday."  (tier1/tier2 only)
#
# With coref_model set, "It arrived yesterday." is also kept because
# coreference resolves "It" to "The report".
```

**Example config** (see
`configs/experiments/experiment_en_remove_expletive_sentences.yaml`):

```yaml
dataset_manipulation:
  - type: remove_expletive_sentences_en
    input_path: "data/raw/train_90M/"
    output_path: "data/processed/exp_remove_expletive_sentences_en/"
    spacy_model: "en_core_web_trf"
    parameters:
      chunk_size: 1000
      replacement_pool_dir: "data/raw/pull_10M/"
      skip_validation: false
      verbose: true
      # Optional: spaCy-compatible coreference model. If omitted, the
      # heuristic is trusted without coref confirmation.
      # coref_model: "en_coreference_web_trf"

random_seed: 9
```

If `seed` is not set under `parameters`, the experiment-level `random_seed`
is auto-injected by the CLI (`model_foundry/cli.py:135`).

### remove_expletive_sentences_it

Italian lacks overt expletive pronouns, so detection is pattern-based on verb
lemma and syntax (`remove_expletive_sentences.py:344`):

1. Weather verbs (`WEATHER_VERBS_IT`, e.g. *piovere*, *nevicare*).
2. Existential *ci + essere* (e.g. *c'è*, *ci sono*).
3. Impersonal raising verbs (`IMPERSONAL_VERBS_IT`) with a clausal
   complement and no `nsubj`.
4. Impersonal necessity verbs (`NECESSITY_VERBS_IT`) with no `nsubj`.

This is a single-tier stateless detector — no per-file tier counts are
produced for Italian.

### impoverish_case_en / impoverish_case_it

Replaces nominative/accusative pronoun forms with a single surface form,
collapsing case distinctions. See
`preprocessing/ablations/impoverish_case.py`.

### lemmatize_verbs

Replaces every verb with its lemma, stripping tense/aspect/agreement
morphology.

```text
Input:  "She was running quickly. He went home."
Output: "She be run quickly. He go home."
```

See `preprocessing/ablations/lemmatize_verbs.py`.

### enrich_verbal_morphology

Appends subject-agreement clitics to finite verbs to test whether richer
agreement morphology changes subject-drop acquisition.
See `preprocessing/ablations/enrich_verbal_morphology.py`.

## Common Workflows

### Research Experiment

```python
# 1. Create ablated training corpus with pool backfill
train_config = AblationConfig(
    type="remove_expletive_sentences_en",
    input_path="data/raw/train_90M/",
    output_path="data/processed/exp1_train/",
    replacement_pool_dir="data/raw/pull_10M/",
    spacy_model="en_core_web_trf",
    seed=42,
)
AblationPipeline(train_config).process_corpus()

# 2. Create matching test set (no pool - we want the cuts visible)
test_config = AblationConfig(
    type="remove_expletive_sentences_en",
    input_path="data/raw/test_10M/",
    output_path="data/processed/exp1_test/",
    spacy_model="en_core_web_trf",
    seed=42,
)
AblationPipeline(test_config).process_corpus()
```

### Production Pipeline

```python
config = AblationConfig(
    type="remove_expletive_sentences_en",
    input_path="data/raw/train_90M/",
    output_path="data/processed/",
    spacy_model="en_core_web_trf",
    seed=42,
    # Performance tuning
    spacy_batch_size=100,
    spacy_disable_components=["ner", "textcat"],
    chunk_size=2000,
    verbose=True,
    log_dir="logs/preprocessing/",
)

manifest = AblationPipeline(config).process_corpus()

if manifest.metadata.failed_files:
    for path, error in manifest.metadata.failed_files:
        print(f"  {path}: {error}")
```

## Configuration Options

### Required

```python
type: str              # Registered ablation name
input_path: Path       # Input corpus directory
output_path: Path      # Output directory
```

### Common Options

```python
seed: int = 42                      # Random seed (auto-injected from experiment random_seed)
chunk_size: int = 1000              # Lines per processing chunk
skip_validation: bool = False       # Skip validation for speed
replacement_pool_dir: Path = None   # Pool for maintaining corpus size
```

### spaCy Configuration

```python
spacy_model: str = "en_core_web_sm"  # Use en_core_web_trf for the expletive sentence detector
spacy_batch_size: int = 50
spacy_disable_components: list = None
```

### Logging

```python
verbose: bool = False
log_dir: Path = "logs"
```

## Provenance Tracking

Every run writes `ABLATION_MANIFEST.json` in the output directory. For
stateful ablations (currently `remove_expletive_sentences_en`) the manifest
includes per-file tier counts, removed line indices, and replacement-pool
draw statistics. See [Advanced](ADVANCED.md#tier-counting-and-provenance).

```python
import json

with open("data/processed/ABLATION_MANIFEST.json") as f:
    manifest = json.load(f)

print(f"Ablation: {manifest['metadata']['ablation_type']}")
print(f"Seed: {manifest['metadata']['random_seed']}")
print(f"Items removed: {manifest['metadata']['total_items_ablated']:,}")
print(f"Aggregate tiers: {manifest['metadata']['aggregate_tier_counts']}")
print(f"Pool drawn: {manifest['metadata']['total_pool_lines_drawn']:,}")
```

## Troubleshooting

### "No .train files found"

The pipeline looks for files with a `.train` extension under `input_path`
(recursive glob). Verify the path and extension.

### spaCy Model Not Found

```bash
python -m spacy download en_core_web_trf
```

### Processing Too Slow

```python
spacy_batch_size=100
spacy_disable_components=["ner", "textcat"]
chunk_size=2000
skip_validation=True
```

### Out of Memory

```python
spacy_batch_size=10
chunk_size=500
```

## Testing

```bash
python -m pytest preprocessing/tests/ -v
```

## Next Steps

- **Add a custom ablation**: see [Developer Guide](DEVELOPER_GUIDE.md)
- **Tier counting, coref, production deployment**: see [Advanced](ADVANCED.md)
- **Test patterns**: see [Testing Guide](TESTING.md)
