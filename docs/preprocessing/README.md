# Preprocessing: Linguistic Ablations

Apply systematic transformations to text corpora to study how models learn language features.

## What This Does

Remove or modify specific linguistic features (articles, pronouns, verb forms) to create controlled experiments. For example, remove all articles to test whether models can learn grammar without "the", "a", or "an".

## Quick Start

```python
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/corpus/",
    output_path="data/processed/corpus/",
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

## Available Ablations

| Ablation | What It Does | Research Use |
|----------|--------------|--------------|
| `remove_articles` | Removes 'a', 'an', 'the' | Test determiner acquisition |
| `remove_expletives` | Removes dummy pronouns (non-referential "it") | Test pronoun understanding |
| `impoverish_determiners` | Replaces all determiners with 'the' | Test morphological learning |
| `lemmatize_verbs` | Reduces verbs to base form | Test verb inflection learning |
| `remove_subject_pronominals` | Removes subject pronouns | Test subject-drop patterns |

## Choose Your Path

**New to preprocessing?** → [User Guide](USER_GUIDE.md) - Learn by doing

**Need to customize?** → [Developer Guide](DEVELOPER_GUIDE.md) - Add your own ablations

**Large-scale processing?** → [Advanced Usage](ADVANCED_USAGE.md) - Performance tuning and production features

**Testing your changes?** → [Testing Guide](TESTING.md) - Run and write tests

## Core Features

**Reproducibility**: Every run generates complete provenance metadata with checksums, environment info, and random seeds.

**Robustness**: Failed files don't crash entire runs. Detailed error logging helps you fix issues quickly.

**Performance**: Process large corpora efficiently with configurable batch sizes and component disabling.

**Maintainability**: Registry-based architecture makes adding new ablations straightforward.

## Example: Research Workflow

```python
# Remove articles from training corpus
config = AblationConfig(
    type="remove_articles",
    input_path="data/bnc_train/",
    output_path="data/bnc_no_articles/",
    replacement_pool_dir="data/pool/",  # Maintain corpus size
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

# Check results
print(f"Processed: {manifest.metadata.total_files_processed} files")
print(f"Removed: {manifest.metadata.total_items_ablated:,} articles")
print(f"Time: {manifest.metadata.processing_time_seconds:.1f}s")

# Provenance saved to: output_path/ABLATION_MANIFEST.json
```

## Common Tasks

**Process a single corpus:**
```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/train.txt",
    output_path="data/train_ablated.txt"
)
```

**Optimize for speed:**
```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/large_corpus/",
    output_path="data/processed/",
    spacy_batch_size=100,
    spacy_disable_components=["ner", "textcat"],
    skip_validation=True
)
```

**Handle errors gracefully:**
```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/messy_corpus/",
    output_path="data/processed/",
    verbose=True  # Detailed error logs
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

if manifest.metadata.failed_files:
    print(f"Warning: {len(manifest.metadata.failed_files)} files failed")
```

## Getting Help

**Can't find what you need?** Check the [User Guide](USER_GUIDE.md) for complete examples.

**Hit an error?** See [Troubleshooting](USER_GUIDE.md#troubleshooting) section.

**Want to contribute?** Read the [Developer Guide](DEVELOPER_GUIDE.md).
