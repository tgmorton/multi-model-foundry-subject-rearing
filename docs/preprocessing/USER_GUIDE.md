# Preprocessing User Guide

Complete guide to using the preprocessing pipeline for text corpus ablations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Available Ablations](#available-ablations)
4. [Common Workflows](#common-workflows)
5. [Performance Tuning](#performance-tuning)
6. [Error Handling](#error-handling)
7. [Provenance Tracking](#provenance-tracking)

## Quick Start

```python
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

# Configure
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/corpus/",
    output_path="data/processed/corpus/",
    seed=42
)

# Run
pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

# Check results
print(f"Processed: {manifest.metadata.total_files_processed} files")
print(f"Modified: {manifest.metadata.total_items_ablated:,} items")
```

## Basic Usage

### Process a Single File

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/bnc_spoken.train",
    output_path="data/processed/bnc_no_articles.train",
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

### Process a Directory

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/corpus/",  # Directory
    output_path="data/processed/corpus/",
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

The pipeline will:
1. Find all `*.train` files recursively
2. Process each file with the ablation
3. Maintain directory structure in output
4. Generate a provenance manifest

### With Replacement Pool

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/train_90M/",
    output_path="data/processed/exp1/",
    replacement_pool_dir="data/raw/pool_10M/",  # Rebuild to original size
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

## Available Ablations

### remove_articles
Removes determiners 'a', 'an', 'the' from text.

```python
# Input:  "The cat sat on a mat."
# Output: "cat sat on mat."
```

**Use case**: Test how models learn without explicit articles.

### remove_expletives
Removes expletive (dummy) pronouns like non-referential "it".

```python
# Input:  "It is raining. It seems nice."
# Output: "is raining. seems nice."
```

**Use case**: Test pronoun function understanding.

**Advanced**: Supports coreference resolution (see [Advanced Usage](ADVANCED_USAGE.md)).

### impoverish_determiners
Replaces all determiners with 'the'.

```python
# Input:  "A cat and an elephant."
# Output: "the cat and the elephant."
```

**Use case**: Test morphological learning with impoverished paradigm.

### lemmatize_verbs
Reduces all verbs to base lemma form.

```python
# Input:  "She was running quickly. He went home."
# Output: "She be run quickly. He go home."
```

**Use case**: Test verb morphology learning.

### remove_subject_pronominals
Removes pronouns functioning as subjects.

```python
# Input:  "She likes cats. They are friendly."
# Output: "likes cats. are friendly."
```

**Use case**: Test subject-drop pattern learning.

## Common Workflows

### Workflow 1: Simple Ablation

```python
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/corpus/",
    output_path="data/processed/no_articles/",
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

### Workflow 2: With Validation

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/corpus/",
    output_path="data/processed/no_articles/",
    seed=42,
    skip_validation=False,  # Enable validation (default)
    verbose=True  # See validation details
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

### Workflow 3: Production Pipeline

```python
# Large corpus with error handling and performance tuning
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/train_90M/",
    output_path="data/processed/exp1/",
    replacement_pool_dir="data/raw/pool_10M/",
    seed=42,
    # Performance
    spacy_batch_size=100,
    spacy_disable_components=["ner", "textcat", "lemmatizer"],
    chunk_size=2000,
    # Logging
    verbose=True,
    log_dir="logs/preprocessing/"
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

# Check for errors
if manifest.metadata.failed_files:
    print(f"⚠️ {len(manifest.metadata.failed_files)} files failed")
    for path, error in manifest.metadata.failed_files:
        print(f"  - {path}: {error}")
```

## Performance Tuning

### Speed-Optimized (Fast Prototyping)

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/",
    output_path="data/processed/",
    seed=42,
    # Fast settings
    spacy_batch_size=100,  # Larger batches
    spacy_disable_components=["ner", "textcat", "lemmatizer"],
    chunk_size=2000,  # More lines per chunk
    skip_validation=True  # Skip validation
)
```

**Expected speedup**: 40-50% faster than defaults

### Balanced (Production)

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/",
    output_path="data/processed/",
    seed=42,
    # Balanced settings (these are mostly defaults)
    spacy_batch_size=50,
    spacy_disable_components=["ner", "textcat"],  # Disable unused only
    chunk_size=1000,
    skip_validation=False  # Keep validation
)
```

**Expected speedup**: 20-30% faster than defaults

### Accuracy-Optimized (Careful Validation)

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/",
    output_path="data/processed/",
    seed=42,
    # Careful settings
    spacy_batch_size=25,  # Smaller batches
    spacy_disable_components=None,  # Use all components
    chunk_size=500,  # Smaller chunks
    skip_validation=False,
    verbose=True  # Full logging
)
```

See [Phase 4 Enhancements](PHASE4_ENHANCEMENTS.md) for detailed performance guide.

## Error Handling

### Check for Failures

```python
pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

# Check results
if manifest.metadata.failed_files:
    print(f"⚠️ Warning: {len(manifest.metadata.failed_files)} files failed")
    for file_path, error_msg in manifest.metadata.failed_files:
        print(f"  - {file_path}")
        print(f"    Error: {error_msg}")
else:
    print("✅ All files processed successfully")
```

### Verbose Error Logging

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/",
    output_path="data/processed/",
    seed=42,
    verbose=True  # Enable detailed error logging with stack traces
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

# Check logs in: logs/preprocessing.remove_articles/
```

### Error Behavior

- **File errors**: Failed files don't crash entire run
- **Validation errors**: Non-fatal warnings, processing continues
- **spaCy errors**: Logged with context, file marked as failed
- **Ablation errors**: Logged with line number, file marked as failed

All errors are tracked in `manifest.metadata.failed_files`.

## Provenance Tracking

Every run generates a complete provenance manifest:

```python
pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

# Manifest saved to: {output_path}/ABLATION_MANIFEST.json
# Contains:
print(f"Timestamp: {manifest.metadata.timestamp}")
print(f"Python: {manifest.metadata.python_version}")
print(f"spaCy: {manifest.metadata.spacy_version}")
print(f"Model: {manifest.metadata.spacy_model_name}")
print(f"Seed: {manifest.metadata.random_seed}")
print(f"Files processed: {manifest.metadata.total_files_processed}")
print(f"Items ablated: {manifest.metadata.total_items_ablated}")
print(f"Processing time: {manifest.metadata.processing_time_seconds:.1f}s")

# Checksums for reproducibility
print(f"Input checksums: {manifest.metadata.input_checksums}")
print(f"Output checksums: {manifest.metadata.output_checksums}")
```

### Loading a Saved Manifest

```python
from preprocessing.config import ProvenanceManifest
import json

with open("data/processed/ABLATION_MANIFEST.json") as f:
    manifest_data = json.load(f)

# Access metadata
print(f"This corpus was processed on: {manifest_data['metadata']['timestamp']}")
print(f"Using seed: {manifest_data['metadata']['random_seed']}")
print(f"Total items ablated: {manifest_data['metadata']['total_items_ablated']}")
```

## Configuration Reference

### All AblationConfig Options

```python
config = AblationConfig(
    # Required
    type="remove_articles",           # Ablation name
    input_path="data/raw/",           # Input directory
    output_path="data/processed/",    # Output directory

    # Reproducibility
    seed=42,                          # Random seed (default: 42)

    # Processing
    chunk_size=1000,                  # Lines per chunk (default: 1000)
    skip_validation=False,            # Skip validation (default: False)

    # Replacement pool
    replacement_pool_dir=None,        # Optional pool directory

    # spaCy
    spacy_model="en_core_web_sm",     # spaCy model (default: en_core_web_sm)
    spacy_device=None,                # Device (None = auto-detect)
    spacy_batch_size=50,              # Batch size (default: 50)
    spacy_disable_components=None,    # Components to disable (default: None)

    # Logging
    verbose=False,                    # Verbose logging (default: False)
    log_dir="logs",                   # Log directory (default: logs)

    # Custom
    parameters={}                     # Ablation-specific params
)
```

## Next Steps

- **Custom ablations**: See [Developer Guide](DEVELOPER_GUIDE.md)
- **Advanced features**: See [Advanced Usage](ADVANCED_USAGE.md)
- **Testing**: See [Testing Guide](TESTING.md)
- **Performance**: See [Phase 4 Enhancements](PHASE4_ENHANCEMENTS.md)

## Troubleshooting

### "No .train files found"
- Check that `input_path` contains `*.train` files
- Files can be in subdirectories (searched recursively)

### spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Out of memory
- Reduce `spacy_batch_size` (try 25 or 10)
- Reduce `chunk_size` (try 500)
- Disable more components

### Slow processing
- Increase `spacy_batch_size` (try 100)
- Disable unused components
- Increase `chunk_size` (try 2000)
- Set `skip_validation=True` for prototyping

## Examples Repository

See `examples/` directory for:
- End-to-end processing scripts
- Performance benchmarks
- Error handling examples
- Custom ablation examples
