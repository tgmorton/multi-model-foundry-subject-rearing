# Advanced Preprocessing

Performance tuning, production deployment, and advanced features.

## When You Need This

**Large-scale processing** (>100M tokens): Optimize for speed and memory

**High-accuracy requirements**: Use coreference resolution for expletive detection

**Production environments**: Error handling, monitoring, and reliability

## Performance Optimization

### Understanding Performance Bottlenecks

Processing speed is primarily determined by:
1. **spaCy pipeline**: Parsing and tagging are expensive
2. **Batch size**: Larger batches = better GPU/CPU utilization
3. **I/O**: Reading/writing many small files is slow

### Speed-Optimized Configuration

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/large_corpus/",
    output_path="data/processed/",
    seed=42,
    # Performance tuning
    spacy_batch_size=100,                    # Default: 50
    spacy_disable_components=["ner", "textcat", "lemmatizer"],
    chunk_size=2000,                         # Default: 1000
    skip_validation=True                     # Skip post-processing validation
)
```

**Expected improvement**: 40-50% faster than defaults

### Component Selection

Only enable what you need:

| Ablation | Required Components | Disable |
|----------|---------------------|---------|
| remove_articles | tagger | ner, textcat, lemmatizer |
| remove_expletives | tagger, parser | ner, textcat, lemmatizer |
| lemmatize_verbs | tagger | ner, textcat, parser |
| impoverish_determiners | tagger | ner, textcat, lemmatizer, parser |
| remove_subject_pronominals | tagger, parser | ner, textcat, lemmatizer |

Example:

```python
# For remove_articles (only needs POS tags)
config = AblationConfig(
    type="remove_articles",
    spacy_disable_components=["ner", "textcat", "lemmatizer", "parser"]
)
```

### Batch Size Tuning

Start with defaults and increase until you hit memory limits:

```python
# Conservative (low memory)
spacy_batch_size=25

# Default (balanced)
spacy_batch_size=50

# Aggressive (high memory)
spacy_batch_size=100

# Maximum (careful!)
spacy_batch_size=200
```

Monitor memory usage:

```bash
# While processing
watch -n 1 "ps aux | grep python | grep -v grep"
```

### Performance Comparison

Expected processing speeds (on modern CPU):

| Configuration | Tokens/sec | Relative Speed |
|---------------|------------|----------------|
| Default | 5,000 | 1.0x |
| Optimized components | 7,000 | 1.4x |
| + Larger batches | 8,500 | 1.7x |
| + Skip validation | 10,000 | 2.0x |

Your mileage may vary based on hardware and corpus characteristics.

## Coreference Resolution

### Why Coreference Matters

Simple expletive detection uses dependency parsing:

```python
# "It is raining" → "it" marked as expletive
# "The report was late. It arrived yesterday." → "It" marked as expletive (WRONG!)
```

Coreference resolution distinguishes referential from non-referential pronouns by checking for antecedents.

### Using Coreference Resolution

```python
import spacy
from preprocessing.ablations.remove_expletives import make_remove_expletives_with_coref
from preprocessing.registry import AblationRegistry

# Load spaCy model with coreference
nlp_coref = spacy.load("en_core_web_sm")

# Create coreference-enabled ablation
ablate_with_coref = make_remove_expletives_with_coref(nlp_coref)

# Register (replaces simple version)
AblationRegistry.unregister("remove_expletives")
AblationRegistry.register(
    "remove_expletives",
    ablate_with_coref,
    validator_fn  # Use existing validator
)

# Use normally
config = AblationConfig(
    type="remove_expletives",
    input_path="data/corpus/",
    output_path="data/processed/"
)
```

### Coreference Model Options

**Option 1: Basic (fast, less accurate)**
```python
nlp_coref = spacy.load("en_core_web_sm")
```

**Option 2: Neural (slower, more accurate)**
```python
# Install: pip install spacy-experimental
# Download: python -m spacy download en_coreference_web_trf
nlp_coref = spacy.load("en_coreference_web_trf")
```

### Context Window

The coreference system includes previous sentence context for long-distance dependencies:

```python
# Input:
# "I saw a cat. It was sleeping on the porch."

# Context analyzed:
# "I saw a cat. It was sleeping..."

# Result: "It" found in coreference chain with "cat" → NOT removed
```

### Performance Impact

Coreference resolution is significantly slower:

| Mode | Speed | Accuracy |
|------|-------|----------|
| Simple | 100% | ~85% |
| Coreference (basic) | ~40% | ~95% |
| Coreference (neural) | ~15% | ~98% |

**Recommendation**: Use simple mode for large corpora (>100M tokens) unless high precision is critical.

## Production Deployment

### Error Handling

File-level error recovery prevents single failures from crashing entire runs:

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/corpus/",
    output_path="data/processed/",
    verbose=True  # Enable detailed logging
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

# Check for failures
if manifest.metadata.failed_files:
    print(f"Warning: {len(manifest.metadata.failed_files)} files failed")

    # Log failures for investigation
    with open("failed_files.log", "w") as f:
        for file_path, error_msg in manifest.metadata.failed_files:
            f.write(f"{file_path}: {error_msg}\n")

    # Optionally re-process failed files with different settings
    failed_paths = [path for path, _ in manifest.metadata.failed_files]
    # ... retry logic
```

### Monitoring

Track processing progress:

```python
import logging

logging.basicConfig(level=logging.INFO)

config = AblationConfig(
    type="remove_articles",
    input_path="data/corpus/",
    output_path="data/processed/",
    verbose=True,
    log_dir="logs/preprocessing/"
)

# Processing logs go to:
# - logs/preprocessing.remove_articles/preprocessing_TIMESTAMP.log
```

Log format:

```
2025-10-09 14:32:15 INFO Processing file 1/100: data/corpus/file1.train
2025-10-09 14:32:20 INFO Processed file1.train: 5,234 items ablated
2025-10-09 14:32:20 INFO Processing file 2/100: data/corpus/file2.train
...
```

### Resource Management

For very large corpora:

```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/huge_corpus/",
    output_path="data/processed/",
    # Conservative settings
    spacy_batch_size=50,
    chunk_size=1000,
    # Don't load entire corpus into memory
    # (Pipeline processes files one-by-one already)
)
```

### Validation Strategy

Choose validation level based on needs:

**Development (full validation)**:
```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/test/",
    output_path="data/output/",
    skip_validation=False,  # Run validation
    verbose=True
)
```

**Production (skip for speed)**:
```python
config = AblationConfig(
    type="remove_articles",
    input_path="data/corpus/",
    output_path="data/processed/",
    skip_validation=True  # Faster, trust the implementation
)
```

**Hybrid (validate sample)**:
```python
# Validate on small sample first
test_config = AblationConfig(
    type="remove_articles",
    input_path="data/corpus_sample/",
    output_path="data/sample_output/",
    skip_validation=False
)

# If validation passes, run full corpus with skip
prod_config = AblationConfig(
    type="remove_articles",
    input_path="data/full_corpus/",
    output_path="data/processed/",
    skip_validation=True
)
```

## Cluster Processing

### SLURM Job Array

Process large corpora in parallel:

```bash
#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --array=1-100
#SBATCH --time=4:00:00
#SBATCH --mem=16G

# Split corpus into 100 parts
# Process part $SLURM_ARRAY_TASK_ID

python process_part.py \
    --part $SLURM_ARRAY_TASK_ID \
    --total-parts 100 \
    --config configs/ablation.yaml
```

```python
# process_part.py
import sys
from pathlib import Path

part_id = int(sys.argv[1])
total_parts = int(sys.argv[2])

# Get list of all files
all_files = sorted(Path("data/corpus/").glob("*.train"))

# Process this part's files
files_per_part = len(all_files) // total_parts
start_idx = (part_id - 1) * files_per_part
end_idx = start_idx + files_per_part if part_id < total_parts else len(all_files)

my_files = all_files[start_idx:end_idx]

# Process each file separately
for file_path in my_files:
    config = AblationConfig(
        type="remove_articles",
        input_path=str(file_path),
        output_path=f"data/processed/part_{part_id}/",
        seed=42
    )
    pipeline = AblationPipeline(config)
    manifest = pipeline.process_corpus()
```

### Combining Results

After parallel processing:

```python
from pathlib import Path
import json

# Collect all manifests
manifests = []
for part_dir in Path("data/processed/").glob("part_*/"):
    manifest_path = part_dir / "ABLATION_MANIFEST.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifests.append(json.load(f))

# Aggregate statistics
total_files = sum(m['metadata']['total_files_processed'] for m in manifests)
total_items = sum(m['metadata']['total_items_ablated'] for m in manifests)
total_time = sum(m['metadata']['processing_time_seconds'] for m in manifests)

print(f"Total files: {total_files}")
print(f"Total items ablated: {total_items:,}")
print(f"Total time: {total_time:.1f}s")
```

## Troubleshooting

### Memory Issues

**Symptom**: Process killed or OOM errors

**Solutions**:
```python
# Reduce batch size
spacy_batch_size=10  # Very conservative

# Reduce chunk size
chunk_size=500

# Disable more components
spacy_disable_components=["ner", "textcat", "lemmatizer", "parser"]
```

### Slow Processing

**Symptom**: <1,000 tokens/sec

**Diagnose**:
```python
# Check what components are enabled
import spacy
nlp = spacy.load("en_core_web_sm")
print(nlp.pipe_names)  # ['tok2vec', 'tagger', 'parser', 'ner', ...]
```

**Solutions**:
```python
# Increase batch size
spacy_batch_size=100

# Disable unused components
spacy_disable_components=["ner", "textcat"]

# Skip validation
skip_validation=True
```

### Validation Failures

**Symptom**: Warnings about validation failures

**Solutions**:

1. Check if validation is too strict (false positives)
2. Verify ablation logic is correct
3. Consider skipping validation in production

```python
# Debug validation
config = AblationConfig(
    type="remove_articles",
    input_path="data/test_single_file.train",
    output_path="data/output/",
    verbose=True  # See detailed validation logs
)
```

## Best Practices Summary

1. **Start conservative**: Use default settings first
2. **Measure before optimizing**: Profile to find bottlenecks
3. **Validate once**: Test on sample, then skip for production
4. **Monitor failures**: Check `manifest.metadata.failed_files`
5. **Use coreference sparingly**: Only when accuracy demands it
6. **Match hardware to workload**: More cores = larger batch sizes
7. **Keep provenance**: Always save manifests for reproducibility

## Next Steps

**Understanding internals**: See architecture docs for how the pipeline works

**Contributing**: See [Testing Guide](TESTING.md) for test requirements

**Questions**: Open an issue with your config and error logs
