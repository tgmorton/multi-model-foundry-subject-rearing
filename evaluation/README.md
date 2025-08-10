# Language Model Evaluation Pipeline

This directory contains the evaluation system for the Subject Drop language modeling study. It provides comprehensive evaluation of model performance across multiple linguistic tasks.

## Quick Start

### 1. Basic Usage
```bash
# Run full evaluation on experiment
python evaluation/evaluation_runner.py --config configs/evaluation_config.yaml

# Evaluate single checkpoint
python evaluation/evaluation_runner.py --config configs/evaluation_config.yaml --checkpoint models/exp0_baseline/epoch_10/

# Run with Singularity
apptainer exec --nv --bind .:/workspace training.sif \
    python evaluation/evaluation_runner.py --config configs/evaluation_config.yaml
```

### 2. Quick Testing
```bash
# Run tests to verify functionality
python evaluation/test_evaluation.py

# Test with limited samples
python evaluation/evaluation_runner.py --config configs/evaluation_config_test.yaml
```

## Configuration

Edit `configs/evaluation_config.yaml`:

```yaml
evaluation:
  model_checkpoint_dir: "models/exp0_baseline/"
  tokenizer_path: "tokenizers/exp0_baseline/"
  test_corpus: "data/raw/test_10M/"
  blimp_dir: "evaluation/stimuli/blimp/"
  null_subject_dir: "evaluation/stimuli/null-subj/"
  
  # Speed optimizations
  batch_size: 64
  use_fp16: true
  device: "cuda"
  
  # For testing - uncomment to limit evaluation
  # max_samples: 100
  # max_checkpoints: 3
```

## Evaluation Tasks

### 1. Perplexity
- Calculates perplexity on held-out test corpus
- Provides aggregate and per-domain metrics
- Tracks language modeling performance

### 2. BLIMP Evaluation  
- Tests 67 linguistic phenomena
- Compares grammatical vs ungrammatical sentences
- Provides overall and per-phenomenon accuracy

### 3. Null-Subject Analysis
- Evaluates overt vs null subject preferences
- Tests person/number conditions
- Measures surprisal at critical positions

## Output Structure

```
evaluation/results/exp0_baseline/
├── evaluation_results.jsonl          # Main results file
├── epoch_1_blimp_detailed.jsonl     # Detailed BLIMP results
├── epoch_1_null_subject_detailed.jsonl # Detailed null-subject results
├── epoch_1_perplexity.json          # Perplexity breakdown
└── evaluation_data_metrics.csv       # R-compatible export
```

## R Integration

Export results for statistical analysis:

```python
from evaluation.result_aggregator import ResultAggregator

aggregator = ResultAggregator("evaluation/results/")
files = aggregator.export_for_r("evaluation_results.jsonl")
# Creates: metrics.csv, learning_curves.csv, summary.json
```

## Speed Optimization

For faster evaluation:

1. **GPU Settings**: Use `device: "cuda"` and `use_fp16: true`
2. **Batch Processing**: Increase `batch_size` based on GPU memory
3. **Limited Evaluation**: Set `max_samples` and `max_checkpoints` for testing
4. **Parallel Processing**: Use multiple workers with `num_workers: 4`

Expected times (with GPU):
- Perplexity (10M tokens): ~5-10 minutes
- BLIMP (67 tasks): ~15-20 minutes  
- Null-subject (14 conditions): ~10-15 minutes
- **Total per checkpoint**: ~30-45 minutes

## Troubleshooting

### Memory Issues
```yaml
# Reduce batch size
batch_size: 16
# Use CPU if needed
device: "cpu"
# Disable fp16 on CPU
use_fp16: false
```

### Missing Files
- Check that model checkpoints exist in `model_checkpoint_dir`
- Verify tokenizer is present at `tokenizer_path`
- Ensure BLIMP/null-subject stimuli are downloaded

### Import Errors
```bash
# Make sure you're in the project root
export PYTHONPATH=/workspace:$PYTHONPATH
```

## Module Documentation

- `model_loader.py`: Load checkpoints and tokenizers
- `surprisal_calculator.py`: Core surprisal computation
- `blimp_evaluator.py`: BLIMP dataset evaluation
- `null_subject_evaluator.py`: Null-subject stimuli processing  
- `perplexity_evaluator.py`: Corpus perplexity calculation
- `evaluation_runner.py`: Main orchestration script
- `result_aggregator.py`: Export and aggregation utilities
- `test_evaluation.py`: Unit tests