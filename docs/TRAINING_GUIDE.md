# Model Foundry Training Guide

Complete guide to training models with Model Foundry on different computing environments.

## Quick Start

### 1. Create a Config

```yaml
# configs/my_experiment.yaml
experiment_name: "my_gpt2_experiment"

model:
  architecture: "gpt2"
  transformer:
    layers: 12
    embedding_size: 768
    hidden_size: 768
    intermediate_hidden_size: 3072
    attention_heads: 12
    dropout: 0.1
    attention_dropout: 0.1

data:
  training_corpus: "data/train/"
  batch_size: 32
  max_sequence_length: 512

tokenizer:
  output_dir: "tokenizers/my_experiment/"
  vocab_size: 32000
  tokenizer_type: "sentencepiece"

training:
  output_dir: "models/my_experiment/"
  learning_rate: 0.0001
  epochs: 10
  gradient_accumulation_steps: 4
  use_amp: true
  resume_from_checkpoint: true

logging:
  use_wandb: true
  project: "my-project"

random_seed: 42
```

### 2. Choose Your Environment

**Wild-West** (direct GPU access, no SLURM):
```bash
./scripts/wild_west/train.sh configs/my_experiment.yaml
```

**SLURM** (SSRDE cluster):
```bash
sbatch scripts/ssrde/train.sh configs/my_experiment.yaml
```

### 3. Monitor Training

```bash
# Watch logs
tail -f logs/my_gpt2_experiment*.log

# Check WandB dashboard
# Visit https://wandb.ai/your-username/my-project
```

## Supported Architectures

Model Foundry supports 6 architectures. Choose based on your research needs:

### 1. GPT-2 (Causal Transformer)
```yaml
model:
  architecture: "gpt2"
  transformer:
    layers: 12              # Small: 12, Medium: 24, Large: 36, XL: 48
    hidden_size: 768        # Small: 768, Medium: 1024, Large: 1280, XL: 1600
    intermediate_hidden_size: 3072  # Usually 4x hidden_size
    attention_heads: 12     # Small: 12, Medium: 16, Large: 20, XL: 25
```

**Use for:** Language modeling, text generation, causal tasks

### 2. BERT (Masked Transformer)
```yaml
model:
  architecture: "bert"
  transformer:
    layers: 12              # Base: 12, Large: 24
    hidden_size: 768        # Base: 768, Large: 1024
    intermediate_hidden_size: 3072
    attention_heads: 12     # Base: 12, Large: 16
  bert:
    type_vocab_size: 2
```

**Use for:** Masked language modeling, bidirectional tasks

### 3. LSTM
```yaml
model:
  architecture: "lstm"
  rnn:
    embedding_size: 512
    hidden_size: 512
    num_layers: 2
    bidirectional: false
    dropout: 0.1
```

**Use for:** Sequential modeling, baseline comparisons

### 4. GRU
```yaml
model:
  architecture: "gru"
  rnn:
    embedding_size: 512
    hidden_size: 512
    num_layers: 2
    bidirectional: false
    dropout: 0.1
```

**Use for:** Faster LSTM alternative, sequential modeling

### 5. Vanilla RNN
```yaml
model:
  architecture: "rnn"
  rnn:
    embedding_size: 256
    hidden_size: 256
    num_layers: 2
    bidirectional: false
    dropout: 0.1
```

**Use for:** Simple baseline, minimal sequential model

### 6. Mamba (State Space Model)
```yaml
model:
  architecture: "mamba"
  mamba:
    d_model: 768
    n_layers: 24
    d_state: 16
    d_conv: 4
    expand: 2
    dropout: 0.1
```

**Use for:** Efficient long-range modeling, O(n) complexity

## Training Environments

### Wild-West (No SLURM)

For servers with direct GPU access:

```bash
# Basic training
./scripts/wild_west/train.sh configs/model.yaml

# With GPU management
./scripts/wild_west/train.sh --lock-gpus --check-gpus configs/model.yaml

# Specific GPUs
./scripts/wild_west/train.sh --gpus 2,3 configs/model.yaml
```

**Features:**
- Direct execution
- GPU locking system
- Process safety (no zombies)
- Suitable for development

**Full guide:** [docs/TRAINING_ON_WILD_WEST.md](TRAINING_ON_WILD_WEST.md)

### SLURM (SSRDE Cluster)

For SLURM-managed clusters:

```bash
# Basic submission (2 GPUs, 24 hours)
sbatch scripts/ssrde/train.sh configs/model.yaml

# Multi-GPU
sbatch --gres=gpu:4 scripts/ssrde/train.sh configs/model.yaml

# Long training
sbatch --time=48:00:00 scripts/ssrde/train.sh configs/model.yaml

# Specific partition
sbatch --partition=a5000 scripts/ssrde/train.sh configs/model.yaml
```

**Features:**
- Job queuing
- Fair-share scheduling
- Resource management
- Suitable for production

**Full guide:** [docs/TRAINING_ON_SLURM.md](TRAINING_ON_SLURM.md)

## Configuration Options

### Data Configuration
```yaml
data:
  training_corpus: "data/train/"        # Training data directory
  validation_corpus: "data/val/"        # Optional validation data
  batch_size: 32                        # Per-GPU batch size
  max_sequence_length: 512              # Maximum context length
  num_workers: 4                        # DataLoader workers
```

### Training Configuration
```yaml
training:
  output_dir: "models/experiment/"      # Checkpoint directory
  learning_rate: 0.0001                 # Initial learning rate
  epochs: 10                            # Training epochs
  train_steps: null                     # Or specify exact steps
  warmup_ratio: 0.1                     # LR warmup proportion
  gradient_accumulation_steps: 4        # Accumulate N batches
  max_grad_norm: 1.0                    # Gradient clipping

  # Optimization
  use_amp: true                         # Mixed precision training
  use_tf32: true                        # TF32 on Ampere+ GPUs
  use_gradient_checkpointing: false     # Trade compute for memory

  # Checkpointing
  resume_from_checkpoint: true          # Auto-resume
  checkpoint_schedule: [100, 500, 1000] # Or null for auto
  auto_generate_checkpoints: true       # Auto schedule
  first_epoch_checkpoints: 20           # Dense early checkpoints

  # Distributed (if multi-GPU)
  distributed: false                    # Enable for multi-GPU
```

### Tokenizer Configuration
```yaml
tokenizer:
  output_dir: "tokenizers/experiment/"
  vocab_size: 32000
  tokenizer_type: "sentencepiece"       # or "wordpiece", "bpe", "character"
  special_tokens:                       # Architecture-specific
    bos_token: "<s>"
    eos_token: "</s>"
    unk_token: "<unk>"
    pad_token: "<pad>"
```

### Logging Configuration
```yaml
logging:
  log_interval: 10                      # Log every N steps
  use_wandb: true                       # Enable W&B
  project: "my-project"                 # W&B project
  tags: ["experiment", "gpt2"]          # W&B tags
```

## Best Practices

### 1. Start Small, Scale Up

```bash
# Test with tiny config
./scripts/wild_west/train.sh configs/test_tiny.yaml

# Scale to full size
sbatch --gres=gpu:4 scripts/ssrde/train.sh configs/production.yaml
```

### 2. Use Token-Based Comparison

For fair cross-architecture comparison:

```yaml
data:
  batch_size: 32                  # Keep same
  max_sequence_length: 512        # Keep same

training:
  gradient_accumulation_steps: 4  # Keep same
  # This gives 32 * 4 * 512 = 65,536 tokens/step across all models
```

See [docs/CROSS_ARCHITECTURE_COMPARISON.md](CROSS_ARCHITECTURE_COMPARISON.md)

### 3. Enable Checkpointing

```yaml
training:
  auto_generate_checkpoints: true
  first_epoch_checkpoints: 20     # Dense early (rapid learning)
  min_checkpoints_per_epoch: 5    # Minimum coverage
  resume_from_checkpoint: true    # Always enable
```

### 4. Monitor Resources

**Wild-West:**
```bash
./scripts/wild_west/gpu_monitor.sh watch
```

**SLURM:**
```bash
squeue -u $USER
ssh <node> nvidia-smi
```

### 5. Use Mixed Precision

```yaml
training:
  use_amp: true      # Faster, less memory
  use_tf32: true     # Better precision on Ampere+
```

Saves ~40% memory, ~2x speedup on modern GPUs.

## Common Workflows

### Single Experiment
```bash
# Wild-West
./scripts/wild_west/train.sh --lock-gpus configs/experiment.yaml

# SLURM
sbatch scripts/ssrde/train.sh configs/experiment.yaml
```

### Multiple Experiments (Sequential)
```bash
# Wild-West
for config in configs/exp*.yaml; do
    ./scripts/wild_west/train.sh --lock-gpus "$config"
done

# SLURM
for config in configs/exp*.yaml; do
    sbatch scripts/ssrde/train.sh "$config"
done
```

### Hyperparameter Sweep
```bash
# Create configs with different LRs
for lr in 0.0001 0.0003 0.001; do
    # Modify config with sed/yq
    sbatch scripts/ssrde/train.sh configs/lr_${lr}.yaml
done
```

### Resume Interrupted Training
```bash
# Just re-run with same config (resume_from_checkpoint: true)
./scripts/wild_west/train.sh configs/experiment.yaml
```

## Troubleshooting

### Out of Memory
```yaml
# Reduce memory usage:
data:
  batch_size: 16              # Reduce from 32

training:
  gradient_accumulation_steps: 8  # Increase to maintain effective batch
  use_gradient_checkpointing: true
  use_amp: true
```

### Training Too Slow
```yaml
# Speed up:
training:
  use_amp: true               # Mixed precision
  use_tf32: true              # Ampere+ GPUs

data:
  num_workers: 8              # More DataLoader workers

# Or use more GPUs (SLURM):
# sbatch --gres=gpu:4 scripts/ssrde/train.sh config.yaml
```

### Checkpoints Too Large
```bash
# Keep only recent checkpoints
ls -t models/exp/checkpoint-* | tail -n +10 | xargs rm -rf
```

### GPU Not Available (Wild-West)
```bash
# Check status
./scripts/wild_west/gpu_monitor.sh available

# Wait for GPU
./scripts/wild_west/gpu_monitor.sh watch
```

### Job Pending (SLURM)
```bash
# Check why
squeue -u $USER --start

# Request fewer resources
sbatch --gres=gpu:1 --time=12:00:00 scripts/ssrde/train.sh config.yaml
```

## Advanced Features

### Custom Checkpoint Schedule
```python
python scripts/generate_checkpoint_schedule.py \
    configs/experiment.yaml \
    --first-epoch 20 \
    --spacing log \
    --min-per-epoch 5
```

### Distributed Training (Multi-GPU)
```yaml
# In config
training:
  distributed: true
```

```bash
# SLURM with 4 GPUs
sbatch --gres=gpu:4 scripts/ssrde/train.sh configs/experiment.yaml
```

### Custom Tokenizer
```yaml
tokenizer:
  tokenizer_type: "wordpiece"  # BERT-style
  vocab_size: 30000
  special_tokens:
    cls_token: "[CLS]"
    sep_token: "[SEP]"
    mask_token: "[MASK]"
    unk_token: "[UNK]"
    pad_token: "[PAD]"
```

## Monitoring and Analysis

### During Training

**Live Logs:**
```bash
tail -f logs/<experiment>*.log
```

**WandB Dashboard:**
- Real-time metrics
- GPU utilization
- Loss curves
- Sample outputs

**GPU Monitoring:**
```bash
# Wild-West
./scripts/wild_west/gpu_monitor.sh watch

# SLURM (SSH to node)
watch -n 1 nvidia-smi
```

### After Training

**Checkpoint Metadata:**
```bash
cat models/experiment/checkpoint-5000/metadata.json
```

Includes:
- Token counts
- Training config
- Model architecture
- Git commit hash
- Timestamps

**Log Analysis:**
```python
python scripts/log_manager.py analyze logs/experiment/
```

## Summary

**Two simple commands for all training:**

```bash
# Development/Testing (Wild-West)
./scripts/wild_west/train.sh configs/your_config.yaml

# Production (SLURM)
sbatch scripts/ssrde/train.sh configs/your_config.yaml
```

**Same configs work everywhere. No environment-specific modifications needed.**

## Further Reading

- [Configuration Reference](../configs/TEMPLATE.yaml) - Full config options
- [Architecture Guide](ARCHITECTURE.md) - Model architecture details
- [Wild-West Guide](TRAINING_ON_WILD_WEST.md) - Direct GPU access
- [SLURM Guide](TRAINING_ON_SLURM.md) - Cluster training
- [Cross-Architecture Comparison](CROSS_ARCHITECTURE_COMPARISON.md) - Fair experiments
- [Token Counting](CROSS_ARCHITECTURE_COMPARISON.md#token-counting-in-checkpoints) - Checkpoint alignment

## Getting Help

1. Check logs: `tail -f logs/<experiment>*.log`
2. Check GPU status: `./scripts/wild_west/gpu_monitor.sh`
3. Check WandB dashboard
4. Review config: Ensure all required fields present
5. Test with tiny config first

**Common issues and solutions documented in each guide.**
