# Wild-West Experiment Runner Guide

This guide focuses on the two main experiment running scripts: `run_experiment.sh` and `run_phase.sh`.

## Overview

The Wild-West system provides two main ways to run experiments:

1. **`run_experiment.sh`** - Full experiment runner with distributed training support
2. **`run_phase.sh`** - Single-GPU phase runner for individual pipeline steps

## Quick Reference

### Full Experiment Runner (`run_experiment.sh`)

```bash
# Basic usage
./scripts/wild_west/run_experiment.sh [OPTIONS] <config_name>

# Most common usage patterns
./scripts/wild_west/run_experiment.sh experiment_1_remove_expletives              # Full pipeline on GPUs 1,2
./scripts/wild_west/run_experiment.sh -g 1 -p run experiment_2_baseline         # Training only on GPU 1
./scripts/wild_west/run_experiment.sh -g 1,2 -l -c experiment_3_remove_articles # With GPU locking & checking
```

### Phase Runner (`run_phase.sh`)

```bash
# Basic usage
./scripts/wild_west/run_phase.sh [OPTIONS] <config_name> <phase>

# Most common usage patterns
./scripts/wild_west/run_phase.sh experiment_1_remove_expletives preprocess      # Preprocessing on GPU 1
./scripts/wild_west/run_phase.sh -g 2 experiment_2_baseline run               # Training on GPU 2
./scripts/wild_west/run_phase.sh -l -c experiment_3_remove_articles tokenize-dataset # With GPU management
```

## Experiment Pipeline Phases

All experiments follow this pipeline:

1. **`preprocess`** - Dataset preprocessing and ablation
   - Applies modifications (remove expletives, articles, etc.)
   - Creates processed datasets in `data/processed/`
   - Memory: ~2-5GB, Time: ~10-30 minutes

2. **`train-tokenizer`** - SentencePiece tokenizer training
   - Trains experiment-specific tokenizer
   - Creates tokenizer in `tokenizers/`
   - Memory: ~1-2GB, Time: ~5-15 minutes

3. **`tokenize-dataset`** - Dataset tokenization
   - Tokenizes datasets using trained tokenizer
   - Creates tokenized data in `data/tokenized/`
   - Memory: ~3-8GB, Time: ~15-45 minutes

4. **`chunk-data`** - Dataset chunking
   - Chunks tokenized data into fixed-length blocks for training
   - Creates chunked data in `data/chunked/`
   - Memory: ~2-5GB, Time: ~5-15 minutes

5. **`run`** - Model training
   - Trains the actual language model
   - Creates model checkpoints in `models/`
   - Memory: ~15-25GB, Time: ~12-48 hours

## Detailed Usage Examples

### Running Complete Experiments

```bash
# Run full pipeline for baseline experiment
./scripts/wild_west/run_experiment.sh experiment_0_baseline

# Run full pipeline with GPU management
./scripts/wild_west/run_experiment.sh -l -c -g 1,2 experiment_1_remove_expletives

# Run with custom batch size and epochs
./scripts/wild_west/run_experiment.sh -b 64 -e 25 experiment_2_remove_articles

# Run with distributed training on 4 GPUs
./scripts/wild_west/run_experiment.sh -d -g 0,1,2,3 experiment_3_lemmatize_verbs
```

### Running Individual Phases

```bash
# Preprocess data for experiment
./scripts/wild_west/run_phase.sh experiment_1_remove_expletives preprocess

# Train tokenizer on GPU 2
./scripts/wild_west/run_phase.sh -g 2 experiment_1_remove_expletives train-tokenizer

# Tokenize dataset with GPU locking
./scripts/wild_west/run_phase.sh -l -c experiment_1_remove_expletives tokenize-dataset

# Chunk data
./scripts/wild_west/run_phase.sh experiment_1_remove_expletives chunk-data

# Run training only
./scripts/wild_west/run_phase.sh -g 1 experiment_1_remove_expletives run
```

### Common Workflows

#### 1. Development Workflow
```bash
# Step 1: Quick preprocessing test
./scripts/wild_west/run_phase.sh experiment_1_remove_expletives preprocess

# Step 2: Train tokenizer
./scripts/wild_west/run_phase.sh experiment_1_remove_expletives train-tokenizer

# Step 3: Test tokenization
./scripts/wild_west/run_phase.sh experiment_1_remove_expletives tokenize-dataset

# Step 4: Test chunking
./scripts/wild_west/run_phase.sh experiment_1_remove_expletives chunk-data

# Step 5: Short training run to test
./scripts/wild_west/run_phase.sh -e 1 experiment_1_remove_expletives run
```

#### 2. Production Workflow
```bash
# Full pipeline with safety checks
./scripts/wild_west/run_experiment.sh -l -c -g 1,2 experiment_1_remove_expletives
```

#### 3. Resume Training
```bash
# If training was interrupted, just run the training phase
./scripts/wild_west/run_phase.sh -g 1,2 experiment_1_remove_expletives run
```

## GPU Management Options

### Basic GPU Options
- `-g, --gpu <id>` (run_phase.sh) - Single GPU ID
- `-g, --gpus <ids>` (run_experiment.sh) - Comma-separated GPU IDs

### GPU Safety Features
- `-c, --check-gpu(s)` - Check GPU availability before starting
- `-l, --lock-gpu(s)` - Lock GPUs to prevent conflicts
- `-u, --unlock-gpu(s)` - Unlock GPUs after completion

### Example with GPU Management
```bash
# Check available GPUs first
./scripts/wild_west/gpu_monitor.sh available

# Run with full GPU safety
./scripts/wild_west/run_experiment.sh -g 1,2 -l -c -u experiment_1_remove_expletives
```

## Configuration Override Options

Both scripts support runtime configuration overrides:

### Training Parameters
- `-b, --batch-size <size>` - Override batch size
- `-e, --epochs <num>` - Override number of epochs

### Examples
```bash
# Smaller batch size for limited GPU memory
./scripts/wild_west/run_experiment.sh -b 16 experiment_1_remove_expletives

# Longer training
./scripts/wild_west/run_experiment.sh -e 50 experiment_2_baseline

# Combined overrides
./scripts/wild_west/run_phase.sh -b 32 -e 10 experiment_3_remove_articles run
```

## Distributed Training

The `run_experiment.sh` script supports distributed training across multiple GPUs:

```bash
# Enable distributed training
./scripts/wild_west/run_experiment.sh -d -g 1,2 experiment_1_remove_expletives

# Distributed training with custom parameters
./scripts/wild_west/run_experiment.sh -d -g 0,1,2,3 -b 32 -e 20 experiment_2_baseline
```

### Distributed Training Notes
- Automatically sets up PyTorch distributed environment
- Scales batch size across GPUs
- Uses NCCL backend for GPU communication
- Monitors all GPU processes

## Output and Logging

### Log Locations
- **Experiment logs**: `logs/<experiment_name>/`
- **SLURM-style output**: `logs/wild_west_<experiment>_<timestamp>.out`
- **Error logs**: `logs/wild_west_<experiment>_<timestamp>.err`

### Real-time Monitoring
```bash
# Watch experiment progress
tail -f logs/wild_west_experiment_1_remove_expletives_*.out

# Monitor GPU usage during training
./scripts/wild_west/gpu_monitor.sh watch
```

## Container Management

Both scripts use Singularity containers for reproducible environments:

### Containers Used
- **`singularity/ablation.sif`** - For preprocessing, tokenizer training, tokenization
- **`singularity/training.sif`** - For model training

### Container Features
- Automatic spaCy model download
- Pre-installed dependencies
- GPU support with `--nv` flag
- Bind mounts for data access

## Troubleshooting

### Common Issues and Solutions

#### 1. GPU Memory Issues
```bash
# Check GPU memory
./scripts/wild_west/gpu_monitor.sh status

# Use smaller batch size
./scripts/wild_west/run_experiment.sh -b 16 experiment_name
```

#### 2. Container Not Found
```bash
# Check containers exist
ls -la singularity/*.sif

# Build if missing (see main README)
```

#### 3. Phase Dependencies
Each phase depends on previous phases:
- `train-tokenizer` requires `preprocess`
- `tokenize-dataset` requires `train-tokenizer`
- `chunk-data` requires `tokenize-dataset`
- `run` requires `chunk-data`

#### 4. GPU Locks
```bash
# Check current locks
./scripts/wild_west/gpu_monitor.sh locks

# Clear stuck locks
./scripts/wild_west/gpu_monitor.sh unlock <gpu_id>
```

## Best Practices

### 1. Always Check GPU Availability
```bash
./scripts/wild_west/gpu_monitor.sh available
```

### 2. Use GPU Locking for Long Jobs
```bash
./scripts/wild_west/run_experiment.sh -l -c experiment_name
```

### 3. Start Small for New Experiments
```bash
# Test with preprocessing first
./scripts/wild_west/run_phase.sh experiment_name preprocess
```

### 4. Monitor Resource Usage
```bash
# Keep GPU monitor open in another terminal
./scripts/wild_west/gpu_monitor.sh watch
```

### 5. Plan GPU Usage
- **Preprocessing**: Any GPU (low memory)
- **Tokenizer training**: Any GPU (low memory)
- **Dataset tokenization**: Any GPU (medium memory)
- **Model training**: High-memory GPU (>15GB free)

## Integration Examples

### From Existing SLURM Scripts
```bash
# Old way
sbatch scripts/p6000/run_ablation_experiment.sh experiment_1_remove_expletives run

# New way
./scripts/wild_west/run_phase.sh -g 1 experiment_1_remove_expletives run
```

### Batch Multiple Experiments
```bash
#!/bin/bash
# Run multiple experiments in sequence

experiments=("experiment_1_remove_expletives" "experiment_2_baseline" "experiment_3_remove_articles")

for exp in "${experiments[@]}"; do
    echo "Starting $exp..."
    ./scripts/wild_west/run_experiment.sh -g 1,2 -l -c "$exp"
    echo "Completed $exp"
done
```

This system provides flexible, robust experiment execution while maintaining responsible GPU usage in a shared environment.