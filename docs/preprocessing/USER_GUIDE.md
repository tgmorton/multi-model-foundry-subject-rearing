# Preprocessing User Guide

Complete guide to using linguistic ablations for corpus processing.

## Understanding Ablations

Linguistic ablations systematically remove or modify language features to test model learning. By creating controlled variations of training data, you can investigate which features models rely on and how they acquire linguistic knowledge.

**Example research questions:**
- Can models learn determiners without seeing "the", "a", or "an"?
- How do models handle pronoun resolution without expletives?
- Does morphological impoverishment affect grammar acquisition?

## Basic Usage

### Process a Corpus

```python
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/corpus/",
    output_path="data/processed/",
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

The pipeline:
1. Finds all `.train` files in `input_path` (recursively)
2. Applies the ablation to each file
3. Writes output maintaining directory structure
4. Generates a provenance manifest

### With Replacement Pool

Maintain corpus size by backfilling with replacement text:

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/train_90M/",      # Main corpus
    output_path="data/processed/",
    replacement_pool_dir="data/pool_10M/",  # Replacement text
    seed=42
)
```

When articles are removed, the pipeline adds text from the pool to reach the original token count.

## Available Ablations

### remove_articles

Removes determiners: 'a', 'an', 'the'

```python
# Input:  "The cat sat on a mat near the window."
# Output: "cat sat on mat near window."
```

**Use case**: Test how models learn determiner systems and noun phrase structure without explicit article exposure.

**Configuration:**
```python
config = AblationConfig(type="remove_articles", ...)
```

### remove_expletives

Removes non-referential pronouns (expletives)

```python
# Input:  "It is raining. It seems like a nice day."
# Output: "is raining. seems like a nice day."
```

**Use case**: Test pronoun function understanding and subject requirement learning.

**Simple mode (default):**
```python
config = AblationConfig(type="remove_expletives", ...)
```

**Advanced mode (with coreference resolution):**
```python
import spacy
from preprocessing.ablations.remove_expletives import make_remove_expletives_with_coref
from preprocessing.registry import AblationRegistry

nlp_coref = spacy.load("en_core_web_sm")
ablate_fn = make_remove_expletives_with_coref(nlp_coref)

AblationRegistry.register("remove_expletives", ablate_fn, validator_fn)
```

See [Advanced Usage](ADVANCED_USAGE.md) for details on coreference resolution.

### impoverish_determiners

Replaces all determiners with 'the'

```python
# Input:  "A cat and an elephant walked by."
# Output: "the cat and the elephant walked by."
```

**Use case**: Test morphological learning with impoverished paradigms.

### lemmatize_verbs

Reduces verbs to base form

```python
# Input:  "She was running quickly. He went home."
# Output: "She be run quickly. He go home."
```

**Use case**: Test verb inflection and tense learning.

### remove_subject_pronominals

Removes pronouns functioning as subjects

```python
# Input:  "She likes cats. They are friendly."
# Output: "likes cats. are friendly."
```

**Use case**: Test subject-drop pattern learning and null subject phenomena.

## Common Workflows

### Research Experiment

```python
# 1. Create ablated training corpus
train_config = AblationConfig(
    type="remove_articles",
    input_path="data/bnc_train/",
    output_path="data/exp1_train/",
    replacement_pool_dir="data/pool/",
    seed=42
)

pipeline = AblationPipeline(train_config)
train_manifest = pipeline.process_corpus()

# 2. Create matching test set (no replacement pool)
test_config = AblationConfig(
    type="remove_articles",
    input_path="data/bnc_test/",
    output_path="data/exp1_test/",
    seed=42
)

pipeline = AblationPipeline(test_config)
test_manifest = pipeline.process_corpus()

# 3. Compare manifests
print(f"Train: {train_manifest.metadata.total_items_ablated:,} items removed")
print(f"Test:  {test_manifest.metadata.total_items_ablated:,} items removed")
```

### Production Pipeline

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/large_corpus/",
    output_path="data/processed/",
    seed=42,
    # Performance tuning
    spacy_batch_size=100,
    spacy_disable_components=["ner", "textcat"],
    chunk_size=2000,
    # Error handling
    verbose=True,
    log_dir="logs/preprocessing/"
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

# Check for failures
if manifest.metadata.failed_files:
    print(f"Warning: {len(manifest.metadata.failed_files)} files failed")
    for file_path, error in manifest.metadata.failed_files:
        print(f"  {file_path}: {error}")
```

## Configuration Options

### Required

```python
type: str              # Ablation name (e.g., "remove_articles")
input_path: Path       # Input corpus file or directory
output_path: Path      # Output directory
```

### Common Options

```python
seed: int = 42                      # Random seed for reproducibility
chunk_size: int = 1000              # Lines per processing chunk
skip_validation: bool = False       # Skip validation for speed
replacement_pool_dir: Path = None   # Pool for maintaining corpus size
```

### spaCy Configuration

```python
spacy_model: str = "en_core_web_sm"
spacy_batch_size: int = 50
spacy_disable_components: list = None  # e.g., ["ner", "textcat"]
```

### Logging

```python
verbose: bool = False
log_dir: Path = "logs"
```

## Performance Tuning

### For Speed

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/corpus/",
    output_path="data/processed/",
    # Optimizations
    spacy_batch_size=100,         # Larger batches
    spacy_disable_components=["ner", "textcat", "lemmatizer"],
    chunk_size=2000,              # More lines per chunk
    skip_validation=True          # Skip validation checks
)
```

**Expected speedup**: 40-50% faster

### For Accuracy

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/corpus/",
    output_path="data/processed/",
    # Conservative settings
    spacy_batch_size=25,          # Smaller batches
    spacy_disable_components=None,  # Use all components
    chunk_size=500,               # Smaller chunks
    skip_validation=False,
    verbose=True                  # Full logging
)
```

### Memory Issues

If you hit out-of-memory errors:

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/corpus/",
    output_path="data/processed/",
    spacy_batch_size=10,          # Much smaller batches
    chunk_size=500,
    spacy_disable_components=["ner", "textcat", "lemmatizer"]
)
```

## Error Handling

### Check for Failures

```python
pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

if manifest.metadata.failed_files:
    print(f"{len(manifest.metadata.failed_files)} files failed:")
    for path, error in manifest.metadata.failed_files:
        print(f"  {path}")
        print(f"    {error}")
```

### Detailed Error Logs

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/corpus/",
    output_path="data/processed/",
    verbose=True  # Enables detailed logging with stack traces
)

# Logs written to: logs/preprocessing.remove_articles/
```

Error types:
- **File errors**: Failed files don't crash the run
- **Validation errors**: Non-fatal warnings, processing continues
- **spaCy errors**: Logged with context, file marked as failed

## Provenance Tracking

Every run generates `ABLATION_MANIFEST.json` with complete metadata:

```json
{
  "metadata": {
    "timestamp": "2025-10-09T14:32:15Z",
    "python_version": "3.10.6",
    "spacy_version": "3.8.7",
    "spacy_model_name": "en_core_web_sm",
    "device": "mps",
    "hostname": "research-macbook.local",
    "ablation_type": "remove_articles",
    "random_seed": 42,
    "total_files_processed": 6,
    "total_tokens_original": 90000000,
    "total_tokens_final": 90000000,
    "total_items_ablated": 8234567,
    "processing_time_seconds": 3245.67,
    "failed_files": []
  },
  "config": {...},
  "files": [...]
}
```

Load saved manifest:

```python
import json

with open("data/processed/ABLATION_MANIFEST.json") as f:
    manifest = json.load(f)

print(f"Processed on: {manifest['metadata']['timestamp']}")
print(f"Seed: {manifest['metadata']['random_seed']}")
print(f"Items ablated: {manifest['metadata']['total_items_ablated']:,}")
```

## Troubleshooting

### "No .train files found"

The pipeline looks for files with `.train` extension. Check:
- Files exist in `input_path`
- Files have `.train` extension
- Path is correct (relative or absolute)

```bash
# Check what files exist
find data/raw/ -name "*.train"
```

### spaCy Model Not Found

```bash
python -m spacy download en_core_web_sm
```

### Processing Too Slow

Try performance optimizations:
```python
spacy_batch_size=100              # Increase batch size
spacy_disable_components=["ner", "textcat"]  # Disable unused
chunk_size=2000                   # Larger chunks
skip_validation=True              # Skip validation
```

### Out of Memory

Reduce memory usage:
```python
spacy_batch_size=10               # Smaller batches
chunk_size=500                    # Smaller chunks
```

## Testing

Run the test suite:

```bash
# All tests
python -m pytest preprocessing/tests/ -v

# Specific test file
python -m pytest preprocessing/tests/test_base.py -v

# With coverage
python -m pytest preprocessing/tests/ --cov=preprocessing
```

Current status: 106 tests passing

## Next Steps

**Add custom ablations**: See [Developer Guide](DEVELOPER_GUIDE.md)

**Advanced features**: See [Advanced Usage](ADVANCED_USAGE.md) for coreference resolution and production deployment

**Understanding internals**: See [Testing Guide](TESTING.md) for architecture details
