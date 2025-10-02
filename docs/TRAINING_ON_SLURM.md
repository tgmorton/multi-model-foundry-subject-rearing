# Training on SLURM (SSRDE Cluster)

Guide for running Model Foundry training on SLURM-managed clusters like SSRDE (A5000/P6000 nodes).

## Overview

SLURM provides job scheduling, resource management, and fair-share queuing for multi-user GPU clusters.

## Quick Start

### 1. Check Available Resources
```bash
# Show partition info
sinfo

# Show your job queue
squeue -u $USER

# Show available GPUs
sinfo -o "%20P %5a %.10l %16F %N"
```

### 2. Submit a Training Job
```bash
# Submit with default settings (2 GPUs, 24 hours)
sbatch scripts/ssrde/train.sh configs/experiment_0_baseline.yaml

# Submit with custom resources
sbatch --gres=gpu:4 --time=48:00:00 scripts/ssrde/train.sh configs/gpt2_large.yaml

# Submit to specific partition
sbatch --partition=a5000 scripts/ssrde/train.sh configs/model.yaml
```

## SSRDE Cluster Configuration

### Available Partitions
- **a5000** - RTX A5000 GPUs (24GB each)
- **p6000** - Quadro P6000 GPUs (24GB each)
- **general** - Mixed GPU types

### Default Resource Limits
- **GPUs**: 1-4 per job
- **Time**: 24 hours (default), 72 hours (max)
- **Memory**: Automatic based on GPUs

## Training Scripts

### `scripts/ssrde/train.sh` - SLURM Training Script

This script handles:
- SLURM resource requests
- Environment setup
- GPU allocation
- Automatic checkpointing
- Log management

**Usage:**
```bash
sbatch [SLURM_OPTIONS] scripts/ssrde/train.sh <config_path>
```

**Examples:**
```bash
# Basic submission
sbatch scripts/ssrde/train.sh configs/gpt2_small.yaml

# Multi-GPU training
sbatch --gres=gpu:4 scripts/ssrde/train.sh configs/gpt2_large.yaml

# Long training run
sbatch --time=48:00:00 scripts/ssrde/train.sh configs/bert_base.yaml

# Specific partition
sbatch --partition=a5000 scripts/ssrde/train.sh configs/mamba.yaml

# With job name
sbatch --job-name=gpt2_exp1 scripts/ssrde/train.sh configs/experiment_1.yaml
```

## SLURM Options

### Common Options
```bash
--job-name=NAME           # Job name (default: config name)
--gres=gpu:N              # Number of GPUs (1-4)
--time=HH:MM:SS           # Time limit
--partition=PARTITION     # Partition to use
--output=FILE             # Stdout file (default: logs/slurm-%j.out)
--error=FILE              # Stderr file (default: logs/slurm-%j.err)
--mail-type=TYPE          # Email notification (BEGIN,END,FAIL,ALL)
--mail-user=EMAIL         # Email address
```

### Resource Guidelines by Model Size

**Small Models** (GPT-2 Small, BERT Base, LSTM):
```bash
sbatch --gres=gpu:1 --time=12:00:00 scripts/ssrde/train.sh config.yaml
```

**Medium Models** (GPT-2 Medium, BERT Large):
```bash
sbatch --gres=gpu:2 --time=24:00:00 scripts/ssrde/train.sh config.yaml
```

**Large Models** (GPT-2 Large, Mamba):
```bash
sbatch --gres=gpu:4 --time=48:00:00 scripts/ssrde/train.sh config.yaml
```

## Job Management

### Monitoring Jobs
```bash
# Check your jobs
squeue -u $USER

# Watch your jobs
watch -n 5 'squeue -u $USER'

# Job details
scontrol show job <JOBID>

# Check job efficiency
seff <JOBID>
```

### Controlling Jobs
```bash
# Cancel a job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER

# Cancel jobs by name
scancel --name=gpt2_exp1

# Hold a job
scontrol hold <JOBID>

# Release a held job
scontrol release <JOBID>
```

### Viewing Logs
```bash
# While job is running
tail -f logs/slurm-<JOBID>.out

# After completion
cat logs/slurm-<JOBID>.out
cat logs/slurm-<JOBID>.err
```

## Batch Submission

### Submit Multiple Experiments
```bash
# Simple loop
for config in configs/experiment_*.yaml; do
    sbatch scripts/ssrde/train.sh "$config"
done

# With dependencies (run exp2 after exp1 completes)
JOB1=$(sbatch --parsable scripts/ssrde/train.sh configs/exp1.yaml)
sbatch --dependency=afterok:$JOB1 scripts/ssrde/train.sh configs/exp2.yaml
```

### Job Arrays
```bash
# Submit array of experiments
sbatch --array=1-7 scripts/ssrde/train_array.sh

# In the script, use $SLURM_ARRAY_TASK_ID to select config
```

## Checkpointing and Resumption

### Automatic Resumption
Model Foundry automatically resumes from checkpoints when `resume_from_checkpoint: true` in config:

```yaml
training:
  resume_from_checkpoint: true
  output_dir: "models/experiment_0"
```

### Manual Resumption
If job times out, resubmit with same config - it will resume automatically:
```bash
sbatch scripts/ssrde/train.sh configs/experiment_0_baseline.yaml
```

## Resource Optimization

### Multi-GPU Training
```yaml
# In config
training:
  distributed: true  # Enable DataParallel/DDP
  batch_size: 16     # Per-GPU batch size
```

```bash
# Submit with multiple GPUs
sbatch --gres=gpu:4 scripts/ssrde/train.sh config.yaml
```

### Memory Management
```yaml
training:
  gradient_accumulation_steps: 4  # Reduce memory usage
  use_gradient_checkpointing: true # Trade compute for memory
  use_amp: true                    # Mixed precision (saves memory)
```

### Time Management
```yaml
training:
  checkpoint_schedule: [1000, 5000, 10000, ...]  # Checkpoint frequently
  train_steps: 50000  # Set reasonable limits
```

## Troubleshooting

### Job Pending
```bash
# Check why job is pending
squeue -u $USER --start

# Reason codes:
# Priority - waiting for higher priority jobs
# Resources - not enough GPUs available
# QOSMaxCpuPerUserLimit - hit user CPU limit
```

### Job Failed
```bash
# Check job exit code
sacct -j <JOBID> --format=JobID,JobName,ExitCode,State

# View error log
cat logs/slurm-<JOBID>.err

# Common issues:
# - OOM (Out of Memory): Reduce batch size
# - Timeout: Increase --time limit
# - CUDA Error: Check GPU compatibility
```

### Out of Memory
```yaml
# In config, reduce memory usage:
data:
  batch_size: 16  # Reduce from 32

training:
  gradient_accumulation_steps: 4  # Maintain effective batch size
  use_gradient_checkpointing: true
  use_amp: true
```

### Job Not Starting
```bash
# Check partition limits
scontrol show partition <partition>

# Check your priority
sprio -u $USER

# Request fewer resources
sbatch --gres=gpu:1 --time=12:00:00 script.sh config.yaml
```

## Email Notifications

Add to your sbatch command:
```bash
sbatch \
  --mail-type=END,FAIL \
  --mail-user=your.email@domain.edu \
  scripts/ssrde/train.sh config.yaml
```

Or in the script header:
```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@domain.edu
```

## Best Practices

### 1. Test Before Long Runs
```bash
# Quick test with 1 epoch
sbatch --gres=gpu:1 --time=1:00:00 scripts/ssrde/train.sh configs/test.yaml
```

### 2. Use Checkpoints Liberally
```yaml
training:
  auto_generate_checkpoints: true
  first_epoch_checkpoints: 20
  min_checkpoints_per_epoch: 5
```

### 3. Monitor Resources
```bash
# After job starts, check GPU usage
ssh <node> nvidia-smi
```

### 4. Clean Up Old Checkpoints
```bash
# Keep only best/latest checkpoints
ls -t models/exp0_baseline/checkpoint-* | tail -n +10 | xargs rm -rf
```

### 5. Use Descriptive Names
```bash
sbatch --job-name=gpt2_baseline_10M scripts/ssrde/train.sh configs/baseline.yaml
```

## Integration with WandB

```yaml
# In config
logging:
  use_wandb: true
  project: "ssrde-experiments"
  tags: ["slurm", "production"]
```

SLURM job info automatically logged to WandB metadata.

## Comparison with Wild-West

| Feature | SLURM (SSRDE) | Wild-West |
|---------|---------------|-----------|
| Job submission | `sbatch` | Direct `./script.sh` |
| Queuing | Automatic | Manual |
| Resource allocation | Fair-share | First-come-first-serve |
| GPU selection | `--gres=gpu:N` | `CUDA_VISIBLE_DEVICES` |
| Time limits | Enforced | Manual |
| Priority | Fair-share algorithm | None |
| Suitable for | Production, shared cluster | Development, dedicated server |

## Example Workflows

### Development to Production
```bash
# 1. Test locally or on wild-west
./scripts/wild_west/train.sh configs/test.yaml

# 2. Run short SLURM test
sbatch --gres=gpu:1 --time=1:00:00 scripts/ssrde/train.sh configs/test.yaml

# 3. Submit full training
sbatch --gres=gpu:4 --time=48:00:00 scripts/ssrde/train.sh configs/production.yaml
```

### Ablation Study
```bash
# Submit all ablations as job array
for i in {1..7}; do
    sbatch --job-name=exp${i} scripts/ssrde/train.sh configs/experiment_${i}.yaml
done
```

### Hyperparameter Sweep
```bash
# With dependencies to avoid overloading
prev_job=""
for lr in 0.0001 0.0003 0.001; do
    if [ -z "$prev_job" ]; then
        prev_job=$(sbatch --parsable scripts/ssrde/train.sh configs/lr_${lr}.yaml)
    else
        prev_job=$(sbatch --parsable --dependency=afterany:$prev_job \
                          scripts/ssrde/train.sh configs/lr_${lr}.yaml)
    fi
done
```

## Summary

SLURM workflow for Model Foundry:

```bash
# 1. Prepare config
vim configs/my_experiment.yaml

# 2. Submit job
sbatch --gres=gpu:2 --time=24:00:00 scripts/ssrde/train.sh configs/my_experiment.yaml

# 3. Monitor
squeue -u $USER
tail -f logs/slurm-*.out

# 4. Resume if needed (automatic)
sbatch scripts/ssrde/train.sh configs/my_experiment.yaml
```

For development and testing, use Wild-West. For production training, use SLURM.
