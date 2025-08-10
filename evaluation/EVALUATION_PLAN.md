# Language Model Evaluation Plan for Subject Drop Study

## Overview
This document outlines the comprehensive evaluation pipeline for assessing language models trained on ablated corpora, focusing on their acquisition of the English overt subject constraint and related linguistic phenomena.

## Dependencies & Environment

### Core Dependencies (from requirements.txt)
We will use the **training.def** singularity container dependencies:
- **PyTorch 2.1.0** with CUDA 11.8 support
- **Transformers 4.41.2** (HuggingFace)
- **Datasets 2.19.2** for data loading
- **SentencePiece 0.2.0** for tokenization
- **NumPy 1.26.4** for numerical operations
- **TQDM 4.66.4** for progress tracking
- **PyYAML 6.0.1** for configuration
- **Pydantic 2.7.4** for config validation

### Execution Environment
```bash
# Using Singularity/Apptainer
apptainer exec --nv --bind .:/workspace training.sif python evaluation/evaluation_runner.py --config configs/eval_config.yaml
```

## Evaluation Components

### 1. Model Checkpoint Loading
- Load GPT-2 models from the model-foundry framework
- Support for models at different training epochs (log-scale checkpoints)
- Compatible with various experiment configurations (exp0-exp7)

### 2. Evaluation Metrics

#### 2.1 Surprisal-Based Evaluation
Following the outline document, we calculate surprisal using:
```
S(w_i) = -log₂ P(w_i | w_1, w_2, ..., w_{i-1})
```

#### 2.2 Perplexity Analysis
- Calculate perplexity on held-out test set (10M words)
- Track model performance across training epochs

#### 2.3 BLIMP Evaluation
- Evaluate on 67 linguistic phenomena from BLiMP dataset
- Calculate accuracy: P(good sentence) > P(bad sentence)

#### 2.4 Null-Subject Evaluation
- Evaluate on custom null-subject stimuli
- Measure surprisal at critical positions (subject, verb, hotspot)
- Compare overt vs null subject preferences

## Speed Optimization Strategies

### 1. Batch Processing
- Process multiple sentences simultaneously using batched forward passes
- Dynamic batching based on sequence length to minimize padding
- Optimal batch sizes: 32-64 for evaluation (memory permitting)

### 2. GPU Utilization
- Use CUDA when available via `--nv` flag in Singularity
- Mixed precision (fp16) evaluation for 2x speedup with minimal accuracy loss
- Pin memory for faster CPU-GPU transfers

### 3. Caching & Preprocessing
- Cache tokenized stimuli to avoid redundant tokenization
- Pre-compute and store token indices for all evaluation sets
- Use memory-mapped files for large datasets

### 4. Parallelization
- Process different checkpoints in parallel across multiple GPUs
- Parallelize across stimulus types (BLIMP, null-subject, perplexity)
- Use DataLoader with multiple workers for I/O operations

### 5. Efficient Data Loading
```python
# Example optimization settings
optimization_config = {
    "batch_size": 64,
    "num_workers": 4,
    "pin_memory": True,
    "use_fp16": True,
    "cache_tokenized": True,
    "prefetch_factor": 2
}
```

### 6. Checkpoint-Specific Optimizations
- Load only required model layers (no training-specific components)
- Use torch.no_grad() context for all evaluations
- Clear GPU cache between checkpoints

## Implementation Architecture

### Core Modules

#### `model_loader.py`
```python
class ModelLoader:
    def load_checkpoint(checkpoint_path, config_path, device='cuda'):
        """Load model with optimized settings for inference"""
    def load_tokenizer(tokenizer_path):
        """Load SentencePiece tokenizer"""
```

#### `surprisal_calculator.py`
```python
class SurprisalCalculator:
    def calculate_surprisal(model, tokenizer, text, hotspot=None):
        """Calculate token-level surprisal with GPU acceleration"""
    def batch_calculate(model, tokenizer, texts, batch_size=64):
        """Batch processing for multiple texts"""
```

#### `blimp_evaluator.py`
```python
class BLIMPEvaluator:
    def evaluate_file(model, tokenizer, jsonl_path):
        """Process single BLIMP file"""
    def evaluate_all(model, tokenizer, blimp_dir):
        """Process all BLIMP datasets with progress tracking"""
```

#### `null_subject_evaluator.py`
```python
class NullSubjectEvaluator:
    def evaluate_stimuli(model, tokenizer, csv_path):
        """Process null-subject stimuli"""
    def compare_conditions(results):
        """Compare overt vs null preferences"""
```

#### `perplexity_evaluator.py`
```python
class PerplexityEvaluator:
    def calculate_perplexity(model, tokenizer, corpus_path, max_samples=None):
        """Calculate perplexity with streaming for large files"""
```

#### `evaluation_runner.py`
Main orchestration script with config-based setup and parallel processing support.

### Configuration Format
```yaml
evaluation:
  # Model configuration
  model_checkpoint_dir: "models/exp0_baseline/"
  tokenizer_path: "tokenizers/exp0_baseline/"
  
  # Test datasets
  test_corpus: "data/raw/test_10M/"
  blimp_dir: "evaluation/stimuli/blimp/"
  null_subject_dir: "evaluation/stimuli/null-subj/"
  
  # Evaluation settings
  batch_size: 64
  device: "cuda"  # or "cpu"
  use_fp16: true  # Mixed precision for speed
  cache_dir: "evaluation/cache/"  # Cache tokenized data
  
  # Analysis options (set false to skip)
  run_perplexity: true
  run_blimp: true
  run_null_subject: true
  
  # Speed optimizations
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
  
  # Output configuration
  output_dir: "evaluation/results/"
  save_format: "jsonl"  # or "csv"
```

## Simple Testing Strategy

### Unit Tests (Minimal)
```python
# test_evaluation.py
def test_model_loading():
    """Test that model loads without errors"""
    
def test_surprisal_calculation():
    """Test surprisal calculation on sample sentence"""
    
def test_batch_processing():
    """Verify batch processing produces same results as sequential"""
    
def test_config_parsing():
    """Test configuration loading and validation"""
```

### Integration Test
```python
def test_mini_evaluation():
    """Run evaluation on 10 samples from each dataset type"""
```

## Output Format

### Results Structure
```
evaluation/results/
├── exp0_baseline/
│   ├── epoch_1/
│   │   ├── perplexity.json
│   │   ├── blimp_results.jsonl
│   │   └── null_subject_results.jsonl
│   └── summary.json
└── evaluation_log.txt
```

### Compact Results Format (JSONL for R)
```jsonl
{"exp": "exp0", "epoch": 1, "metric": "perplexity", "value": 45.2}
{"exp": "exp0", "epoch": 1, "metric": "blimp_overall", "value": 0.752}
{"exp": "exp0", "epoch": 1, "metric": "null_subj_pref", "condition": "3rdSG", "value": -2.3}
```

## Execution Pipeline

### Quick Start
```bash
# Single checkpoint evaluation
python evaluation/evaluation_runner.py \
    --checkpoint models/exp0_baseline/checkpoint-1000/ \
    --config configs/eval_config.yaml

# Full experiment evaluation
python evaluation/evaluate_experiment.py \
    --experiment exp0_baseline \
    --config configs/eval_config.yaml
```

### Batch Evaluation Script
```bash
#!/bin/bash
# evaluate_all.sh
for exp in exp0 exp1 exp2 exp3 exp4 exp5 exp6 exp7; do
    apptainer exec --nv --bind .:/workspace training.sif \
        python evaluation/evaluate_experiment.py --experiment $exp
done
```

## Performance Benchmarks
Expected processing times (with GPU):
- Perplexity (10M tokens): ~5-10 minutes
- BLIMP (1000 sentences × 67 files): ~15-20 minutes  
- Null-subject (500 pairs × 14 conditions): ~10-15 minutes
- Total per checkpoint: ~30-45 minutes
- Full experiment (20 checkpoints): ~10-15 hours

Without GPU: multiply by 5-10x

## Next Implementation Steps
1. Create model loader with optimized inference settings
2. Implement batched surprisal calculator
3. Build BLIMP evaluator with progress tracking
4. Create null-subject evaluator
5. Implement streaming perplexity calculator
6. Build config-driven evaluation runner
7. Create simple test suite
8. Write results aggregation utilities