# Training on Wild-West Servers

Guide for running Model Foundry training on shared GPU servers without job schedulers.

## Overview

Wild-West is a lightweight GPU management system for servers without SLURM. It provides:
- GPU availability checking and locking
- Safe process management (no zombies!)
- Direct execution on available GPUs
- Compatible with existing configs

## Quick Start

### 1. Check GPU Availability
```bash
./scripts/wild_west/gpu_monitor.sh
./scripts/wild_west/gpu_monitor.sh available
```

### 2. Train a Model
```bash
# Train using available GPUs
./scripts/wild_west/train.sh configs/experiment_0_baseline.yaml

# Train on specific GPUs
CUDA_VISIBLE_DEVICES=1,2 ./scripts/wild_west/train.sh configs/experiment_0_baseline.yaml

# With GPU locking (recommended for long jobs)
./scripts/wild_west/train.sh --lock-gpus configs/experiment_0_baseline.yaml
```

## GPU Monitor

### Basic Commands
```bash
# Show GPU status
./scripts/wild_west/gpu_monitor.sh status

# Find available GPUs (>10GB free)
./scripts/wild_west/gpu_monitor.sh available

# Lock a GPU
./scripts/wild_west/gpu_monitor.sh lock 1

# Unlock a GPU
./scripts/wild_west/gpu_monitor.sh unlock 1

# Show all locks
./scripts/wild_west/gpu_monitor.sh locks

# Watch in real-time
./scripts/wild_west/gpu_monitor.sh watch
```

### GPU Status Categories
- **AVAILABLE** (>20GB free) - Ideal for training
- **LIMITED** (10-20GB free) - Good for smaller models
- **OCCUPIED** (<10GB free) - Avoid

## Training Scripts

### `train.sh` - Main Training Runner

```bash
./scripts/wild_west/train.sh [OPTIONS] <config_path>
```

**Options:**
- `--lock-gpus` - Lock GPUs before training
- `--check-gpus` - Verify GPU availability first
- `--gpus <ids>` - Override CUDA_VISIBLE_DEVICES (e.g., --gpus 1,2)

**Examples:**
```bash
# Basic training
./scripts/wild_west/train.sh configs/gpt2_small.yaml

# With GPU management
./scripts/wild_west/train.sh --lock-gpus --check-gpus configs/gpt2_small.yaml

# Specific GPUs
./scripts/wild_west/train.sh --gpus 2,3 configs/bert_base.yaml
```

## Responsible GPU Usage

### Best Practices

1. **Always check availability** before starting jobs
   ```bash
   ./scripts/wild_west/gpu_monitor.sh available
   ```

2. **Lock GPUs** for long-running jobs
   ```bash
   ./scripts/wild_west/train.sh --lock-gpus config.yaml
   ```

3. **Monitor during training**
   ```bash
   ./scripts/wild_west/gpu_monitor.sh watch
   ```

4. **Unlock when done** (automatic with `--lock-gpus`, but verify)
   ```bash
   ./scripts/wild_west/gpu_monitor.sh locks
   ```

### GPU Selection Guidelines
- **Large models (>15GB)**: Use GPUs with >20GB available
- **Medium models (10-15GB)**: Use GPUs with >15GB available
- **Small models (<10GB)**: Any GPU with >10GB available

### Memory Requirements by Architecture
- **GPT-2 Small**: ~15GB (batch_size=32)
- **GPT-2 Medium**: ~20GB (batch_size=32)
- **BERT Base**: ~12GB (batch_size=32)
- **LSTM/GRU**: ~8GB (batch_size=32)
- **Mamba**: ~18GB (batch_size=32)

*Scale linearly with batch size and gradient accumulation*

## Process Safety

The wild_west scripts implement zombie-prevention:

### Automatic Features
- ✅ Process group management (`setsid`)
- ✅ Signal handling (SIGTERM, SIGINT, EXIT)
- ✅ Child process cleanup
- ✅ GPU lock management
- ✅ Timeout protection

### What This Means
- **No zombie processes** - All child processes properly cleaned up
- **No stuck GPU memory** - Processes die completely on exit
- **Safe interruption** - Ctrl+C cleans up properly
- **Automatic cleanup** - Locks released even on error

## Troubleshooting

### GPU Memory Issues
```bash
# Check current usage
./scripts/wild_west/gpu_monitor.sh status

# Use smaller batch size in config
data:
  batch_size: 16  # Reduce from 32

training:
  gradient_accumulation_steps: 4  # Increase to maintain effective batch
```

### GPU Already Locked
```bash
# Check who has the lock
./scripts/wild_west/gpu_monitor.sh locks

# Unlock if it's your stale lock
./scripts/wild_west/gpu_monitor.sh unlock 1
```

### Process Won't Die
```bash
# Find the process group ID
ps -ef | grep python | grep train

# Kill the entire process group
kill -TERM -<PGID>

# If that doesn't work
kill -KILL -<PGID>
```

### Training Interrupted
```bash
# Resume from checkpoint (automatic with resume_from_checkpoint: true in config)
./scripts/wild_west/train.sh configs/experiment.yaml
```

## Environment Variables

The training script sets:
```bash
CUDA_VISIBLE_DEVICES         # GPU selection
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Memory management
TORCH_CUDA_MEMORY_FRACTION=0.95                   # Leave some for OS
```

## Integration with Existing Configs

All Model Foundry configs work directly:
```bash
# Use any config from configs/
./scripts/wild_west/train.sh configs/experiment_0_baseline.yaml
./scripts/wild_west/train.sh configs/test_mamba_tiny.yaml
./scripts/wild_west/train.sh configs/experiment_1_remove_expletives.yaml
```

## Monitoring Training

### Watch Logs
```bash
# Training logs
tail -f logs/<experiment_name>/*.log

# GPU usage
./scripts/wild_west/gpu_monitor.sh watch
```

### WandB (if enabled)
```yaml
logging:
  use_wandb: true
  project: "your-project"
```
Then visit https://wandb.ai/your-username/your-project

## Advanced Usage

### Custom GPU Assignment
```bash
# Export before running
export CUDA_VISIBLE_DEVICES=2,3
./scripts/wild_west/train.sh configs/model.yaml

# Or inline
CUDA_VISIBLE_DEVICES=1 ./scripts/wild_west/train.sh configs/model.yaml
```

### Multiple Experiments
```bash
# Sequential
for config in configs/experiment_*.yaml; do
    ./scripts/wild_west/train.sh --lock-gpus "$config"
done

# Parallel on different GPUs
CUDA_VISIBLE_DEVICES=0 ./scripts/wild_west/train.sh configs/exp1.yaml &
CUDA_VISIBLE_DEVICES=1 ./scripts/wild_west/train.sh configs/exp2.yaml &
wait
```

### Debug Mode
```bash
# Add to config for more verbose output
training:
  use_amp: false  # Disable mixed precision for clearer errors

logging:
  level: "DEBUG"
```

## Differences from SLURM

| Feature | Wild-West | SLURM |
|---------|-----------|-------|
| Job submission | Direct execution | `sbatch` |
| GPU selection | `CUDA_VISIBLE_DEVICES` | `--gres=gpu:N` |
| Resource limits | Manual/monitor | Automatic |
| Queuing | Manual coordination | Automatic |
| Priority | First-come-first-serve | Fair-share |
| Cleanup | Automatic (script handles) | Automatic (SLURM handles) |

## When to Use Wild-West vs SLURM

**Use Wild-West when:**
- Server doesn't have SLURM
- Need immediate execution
- Interactive development
- Small team, low contention

**Use SLURM when:**
- Available on the system
- High resource contention
- Need fair queuing
- Production workflows

## Summary

Wild-West provides a simple, safe way to run Model Foundry training on shared GPU servers:

```bash
# 1. Check GPUs
./scripts/wild_west/gpu_monitor.sh available

# 2. Train
./scripts/wild_west/train.sh --lock-gpus configs/your_config.yaml

# 3. Monitor
./scripts/wild_west/gpu_monitor.sh watch
```

That's it! No job scheduler needed.
