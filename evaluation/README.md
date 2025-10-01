# Language Model Evaluation Pipeline

This directory contains the evaluation system for the Subject Drop language modeling study. It provides comprehensive evaluation of model performance across multiple linguistic tasks.

## Quick Start

### 1. Basic Usage (Recommended: Parallel Runner)
```bash
# Run full evaluation on experiment (multi-GPU with threading)
python evaluation/runners/parallel_evaluation_runner.py --config configs/evaluation_config.yaml

# Evaluate single checkpoint
python evaluation/runners/parallel_evaluation_runner.py --config configs/evaluation_config.yaml --checkpoint models/exp0_baseline/epoch_10/

# Run with Singularity
apptainer exec --nv --bind .:/workspace training.sif \
    python evaluation/runners/parallel_evaluation_runner.py --config configs/evaluation_config.yaml
```

### 1b. Single-Threaded Alternative
```bash
# Use single-threaded runner if parallel causes issues
python evaluation/runners/evaluation_runner.py --config configs/evaluation_config.yaml
```

### 2. Quick Testing
```bash
# Run tests to verify functionality
python evaluation/tests/test_evaluation.py

# Test with limited samples
python evaluation/runners/parallel_evaluation_runner.py --config configs/evaluation_config_test.yaml
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
from evaluation.core.result_aggregator import ResultAggregator

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

## Directory Structure

```
evaluation/
├── core/                           # Core computation infrastructure
│   ├── model_loader.py            # Load checkpoints and tokenizers
│   ├── surprisal_calculator.py    # Core surprisal computation
│   └── result_aggregator.py       # Export results for R analysis
│
├── evaluators/                     # Task-specific evaluators
│   ├── blimp_evaluator.py         # BLIMP dataset evaluation (67 phenomena)
│   ├── null_subject_evaluator.py  # Null-subject preference analysis
│   └── perplexity_evaluator.py    # Corpus perplexity calculation
│
├── runners/                        # Orchestration scripts
│   ├── parallel_evaluation_runner.py  # ⭐ RECOMMENDED: Multi-GPU with threading
│   ├── evaluation_runner.py           # Single-threaded (simpler, slower)
│   └── threaded_blimp_evaluator.py    # Threading backend (used by parallel runner)
│
├── aggregation/                    # Post-processing and reporting
│   ├── summary_generator.py       # Cross-checkpoint summary statistics
│   └── item_level_aggregator.py   # Item-level result aggregation
│
├── stimuli/                        # Stimuli processing
│   ├── blimp/                     # BLIMP stimuli files
│   ├── null-subj/                 # Null-subject stimuli files
│   ├── normalize_text.py          # Text normalization utilities
│   └── transform_stimuli.py       # Stimulus transformation
│
├── tests/                          # Testing
│   └── test_evaluation.py         # Unit tests
│
└── results/                        # Generated evaluation results
    ├── exp0_baseline/
    ├── exp1_remove_expletives/
    └── ...
```

## Module Documentation

### Main Entry Points
- **`runners/parallel_evaluation_runner.py`**: ⭐ **RECOMMENDED** - Multi-GPU evaluation with threading for optimal performance
- `runners/evaluation_runner.py`: Single-threaded evaluation (simpler, useful for debugging)

### Core Infrastructure (`core/`)
- `model_loader.py`: Load model checkpoints and tokenizers, manage GPU memory
- `surprisal_calculator.py`: Core surprisal computation logic for linguistic analysis
- `result_aggregator.py`: Export and aggregate results for R statistical analysis

### Task Evaluators (`evaluators/`)
- `blimp_evaluator.py`: Evaluates 67 linguistic phenomena from the BLIMP benchmark
- `null_subject_evaluator.py`: Analyzes overt vs null subject preferences across conditions
- `perplexity_evaluator.py`: Calculates corpus perplexity on held-out test data

### Aggregation & Reporting (`aggregation/`)
- `summary_generator.py`: Generates cross-checkpoint summaries and learning curves
- `item_level_aggregator.py`: Aggregates item-level results for detailed analysis

### Parallel Processing (`runners/`)
- `threaded_blimp_evaluator.py`: Threading-based BLIMP evaluation backend (avoids CUDA multiprocessing issues)

### Utilities
- `stimuli/`: Stimulus normalization, transformation, and data loading utilities
- `tests/test_evaluation.py`: Unit tests for verification of evaluation pipeline