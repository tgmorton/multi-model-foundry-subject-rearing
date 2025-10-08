# Preprocessing Module Documentation

## Overview

The preprocessing module provides a unified, config-driven system for applying linguistic ablations to text corpora. It replaces the legacy ablation scripts with a modular, testable, and reproducible pipeline.

## Quick Start

```python
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

# Configure the ablation
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/corpus/",
    output_path="data/processed/corpus/",
    seed=42
)

# Run the pipeline
pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

print(f"Processed {manifest.metadata.total_files_processed} files")
print(f"Removed {manifest.metadata.total_items_ablated:,} items")
```

## Available Ablations

| Ablation | Description | Use Case |
|----------|-------------|----------|
| `remove_articles` | Removes determiners ('a', 'an', 'the') | Test determiner learning |
| `remove_expletives` | Removes expletive (dummy) pronouns | Test pronoun function |
| `impoverish_determiners` | Replaces all determiners with 'the' | Test morphology learning |
| `lemmatize_verbs` | Reduces verbs to base form | Test verb morphology |
| `remove_subject_pronominals` | Removes subject pronouns | Test subject-drop patterns |

## Directory Structure

```
preprocessing/
├── __init__.py              # Public API
├── base.py                  # AblationPipeline class
├── config.py                # Configuration models
├── registry.py              # Ablation registry
├── utils.py                 # Shared utilities
├── ablations/               # Ablation implementations
│   ├── __init__.py
│   ├── remove_articles.py
│   ├── remove_expletives.py
│   ├── impoverish_determiners.py
│   ├── lemmatize_verbs.py
│   └── remove_subject_pronominals.py
└── tests/                   # Test suite
    ├── conftest.py
    ├── test_base.py
    ├── test_config.py
    ├── test_registry.py
    ├── test_utils.py
    ├── test_remove_articles_integration.py
    └── test_new_ablations_integration.py
```

## Documentation

- **[User Guide](USER_GUIDE.md)** - Complete usage examples and workflows
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Adding custom ablations
- **[Advanced Usage](ADVANCED_USAGE.md)** - Coreference resolution and advanced features
- **[Phase 4 Enhancements](PHASE4_ENHANCEMENTS.md)** - Error handling and performance tuning
- **[Testing Guide](TESTING.md)** - Running and writing tests
- **[Test Status](TEST_STATUS.md)** - Current test coverage

## Key Features

### ✅ Reproducibility
- Random seed control
- Environment metadata tracking
- Input/output checksums
- Complete provenance manifests

### ✅ Robustness
- File-level error recovery
- Detailed error logging
- Graceful degradation
- Failed file tracking

### ✅ Performance
- Configurable batch processing
- Selective component disabling
- Memory-efficient chunking
- 30-40% speedup with tuning

### ✅ Maintainability
- 80% code reduction from legacy scripts
- Registry-based architecture
- Comprehensive test coverage (106 tests)
- Type-safe configuration with Pydantic

## Common Workflows

### Process a Single Corpus

```python
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/bnc_spoken.train",
    output_path="data/processed/bnc_no_articles.train",
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

### Process with Replacement Pool

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

### Optimize for Speed

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/corpus/",
    output_path="data/processed/corpus/",
    seed=42,
    # Performance tuning
    spacy_batch_size=100,
    spacy_disable_components=["ner", "textcat", "lemmatizer"],
    chunk_size=2000,
    skip_validation=True
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

### Handle Errors Gracefully

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/corpus/",
    output_path="data/processed/corpus/",
    seed=42,
    verbose=True  # Detailed error logging
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

# Check for failures
if manifest.metadata.failed_files:
    print(f"Warning: {len(manifest.metadata.failed_files)} files failed:")
    for path, error in manifest.metadata.failed_files:
        print(f"  {path}: {error}")
```

## Configuration Reference

### AblationConfig Fields

```python
# Required
type: str                           # Ablation type (registered name)
input_path: Path                    # Input corpus directory
output_path: Path                   # Output directory

# Reproducibility
seed: int = 42                      # Random seed

# Processing
chunk_size: int = 1000              # Lines per chunk
skip_validation: bool = False       # Skip validation step

# Replacement pool
replacement_pool_dir: Optional[Path] = None

# spaCy configuration
spacy_model: str = "en_core_web_sm"
spacy_device: Optional[str] = None  # Auto-detect if None
spacy_batch_size: int = 50
spacy_disable_components: Optional[List[str]] = None

# Logging
verbose: bool = False
log_dir: Path = Path("logs")

# Custom parameters
parameters: Dict[str, Any] = {}     # Ablation-specific params
```

## Provenance Tracking

Every run generates a manifest with complete metadata:

```json
{
  "metadata": {
    "timestamp": "2025-10-08T14:32:15Z",
    "python_version": "3.10.6",
    "spacy_version": "3.8.7",
    "spacy_model_name": "en_core_web_sm",
    "spacy_model_version": "3.7.1",
    "device": "mps",
    "hostname": "research-macbook.local",
    "ablation_type": "remove_articles",
    "random_seed": 42,
    "chunk_size": 1000,
    "total_files_processed": 6,
    "total_tokens_original": 90000000,
    "total_tokens_final": 90000000,
    "total_items_ablated": 8234567,
    "processing_time_seconds": 3245.67,
    "input_checksums": {...},
    "output_checksums": {...},
    "failed_files": []
  },
  "config": {...},
  "files": [...]
}
```

## Migrating from Legacy Scripts

### Old Way (Legacy Scripts)

```bash
python preprocessing/remove_articles.py \
  --input_dir data/raw/train_90M/ \
  --output_dir data/processed/exp1/ \
  --replacement_pool_dir data/raw/pool_10M/ \
  --chunk_size 1000 \
  --verbose
```

### New Way (Unified Pipeline)

```python
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

config = AblationConfig(
    type="remove_articles",
    input_path="data/raw/train_90M/",
    output_path="data/processed/exp1/",
    replacement_pool_dir="data/raw/pool_10M/",
    chunk_size=1000,
    verbose=True
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

### Benefits of New System

- ✅ **Type safety**: Pydantic validates configuration
- ✅ **Reproducibility**: Automatic seed setting and environment tracking
- ✅ **Error handling**: Failed files don't crash entire run
- ✅ **Testability**: 106 tests ensure correctness
- ✅ **Performance**: Configurable tuning for 30-40% speedup
- ✅ **Provenance**: Complete manifest with checksums

## Testing

Run the test suite:

```bash
# All tests
python -m pytest preprocessing/tests/ -v

# Specific test file
python -m pytest preprocessing/tests/test_base.py -v

# With coverage
python -m pytest preprocessing/tests/ --cov=preprocessing --cov-report=html
```

Current status: **106/106 tests passing** ✅

## Performance Tips

1. **Increase batch size** for faster processing:
   ```python
   spacy_batch_size=100  # Default: 50
   ```

2. **Disable unused components**:
   ```python
   # Most ablations only need tagger and parser
   spacy_disable_components=["ner", "textcat", "lemmatizer"]
   ```

3. **Larger chunks** for memory-efficient systems:
   ```python
   chunk_size=2000  # Default: 1000
   ```

4. **Skip validation** for prototyping:
   ```python
   skip_validation=True  # Default: False
   ```

See [Phase 4 Enhancements](PHASE4_ENHANCEMENTS.md) for detailed performance tuning guide.

## Getting Help

- **Examples**: See [User Guide](USER_GUIDE.md)
- **Custom ablations**: See [Developer Guide](DEVELOPER_GUIDE.md)
- **Advanced features**: See [Advanced Usage](ADVANCED_USAGE.md)
- **Issues**: File a bug report with:
  - Config used
  - Error message
  - Log file (if verbose mode enabled)
  - Sample input that reproduces the issue

## Related Documentation

- [Training Guide](../TRAINING_GUIDE.md) - Using processed corpora for training
- [Model Foundry Docs](../model_foundry/README.md) - Model architecture documentation
- [SLURM Training](../TRAINING_ON_SLURM.md) - Large-scale cluster training

## Architecture Overview

```
┌─────────────────┐
│  AblationConfig │  ← Pydantic model (type-safe configuration)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AblationPipeline│  ← Main orchestrator
└────────┬────────┘
         │
         ├──► AblationRegistry  ← Function lookup
         ├──► spaCy NLP         ← Text processing
         ├──► Utils             ← Shared functions
         └──► ProvenanceManifest ← Metadata tracking
```

## License

Part of the Multi-Model Foundry project.

## Changelog

### Phase 5 (Current)
- Complete documentation reorganization
- Developer and user guides
- Ablation template

### Phase 4
- Enhanced error handling
- Performance optimizations
- Configurable batch processing
- Component disabling

### Phase 3
- All 5 ablations migrated
- Coreference resolution support
- 106 tests passing

### Phase 2
- First ablation (remove_articles) refactored
- Integration tests

### Phase 1
- Base infrastructure
- Registry system
- Configuration models
