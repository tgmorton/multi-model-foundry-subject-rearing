# P6000 SLURM Scripts

This directory contains SLURM-optimized scripts for running experiments on the P6000 GPU cluster.

## Quick Start

```bash
# Make scripts executable
chmod +x *.sh

# Submit a full experiment pipeline
./submit_jobs.sh experiment experiment_0_baseline

# Submit just the training phase
./submit_jobs.sh phase experiment_0_baseline run

# Monitor your jobs
./monitor_jobs.sh

# Follow live logs
./monitor_jobs.sh -f
```

## Scripts Overview

### Core Execution Scripts

- **`run_experiment_slurm.sh`** - Full experiment pipeline runner with SLURM integration
  - Handles all phases: preprocess, train-tokenizer, tokenize-dataset, run
  - Improved process management from wild_west version
  - Automatic cleanup and error handling

- **`run_phase_slurm.sh`** - Single phase runner for targeted execution
  - Run individual phases with optimized settings
  - Phase-specific memory and time allocations
  - GPU availability checking

### Helper Scripts

- **`submit_jobs.sh`** - User-friendly job submission interface
  - Simplified command syntax
  - Batch job submission support
  - Job dependency management

- **`monitor_jobs.sh`** - Real-time job monitoring
  - Live status updates
  - GPU utilization tracking
  - Log file following

## Key Improvements from wild_west Scripts

1. **SLURM Integration**
   - Proper `srun` usage for container execution
   - SLURM environment variable handling
   - Job dependency support

2. **Process Management**
   - No manual process group management needed (SLURM handles it)
   - Automatic cleanup on job termination
   - No zombie processes

3. **Resource Optimization**
   - Phase-specific memory allocation
   - Automatic time limit estimation
   - P6000-optimized CUDA settings

4. **Error Handling**
   - Comprehensive error checking
   - Graceful failure modes
   - Detailed logging to SLURM output files

## Usage Examples

### Submit Full Pipeline
```bash
# Standard submission
./submit_jobs.sh experiment experiment_0_baseline

# With custom settings
./submit_jobs.sh experiment experiment_0_baseline \
  --time 72:00:00 \
  --batch-size 64 \
  --wandb-mode online
```

### Submit Individual Phases
```bash
# Preprocessing only
./submit_jobs.sh phase experiment_1_remove_expletives preprocess

# Training with custom batch size
./submit_jobs.sh phase experiment_0_baseline run --batch-size 32
```

### Batch Submission
```bash
# Submit all experiments from list
./submit_jobs.sh batch experiments_batch.txt

# Experiments will run sequentially with dependencies
```

### Job Management
```bash
# Check status
./submit_jobs.sh status

# View logs
./submit_jobs.sh logs 12345

# Cancel job
./submit_jobs.sh cancel 12345

# Cancel all your jobs
./submit_jobs.sh cancel-all
```

### Monitoring
```bash
# Live monitoring (30 second refresh)
./monitor_jobs.sh

# Custom refresh interval
./monitor_jobs.sh -i 10

# Monitor specific job
./monitor_jobs.sh -j 12345

# Show GPU utilization
./monitor_jobs.sh -g

# Follow most recent job's log
./monitor_jobs.sh -f
```

## SLURM Parameters

### Default Resource Allocations

| Phase | Time Limit | Memory | GPUs | CPUs |
|-------|------------|--------|------|------|
| preprocess | 2:00:00 | 32G | 1 | 8 |
| train-tokenizer | 1:00:00 | 32G | 1 | 8 |
| tokenize-dataset | 1:00:00 | 32G | 1 | 8 |
| run (training) | 24:00:00 | 32G | 1 | 8 |
| full-pipeline | 48:00:00 | 32G | 1 | 8 |

### P6000 Specifications
- **GPU Memory**: 24GB GDDR5X
- **CUDA Cores**: 3840
- **Architecture**: Pascal
- **Compute Capability**: 6.1

## Environment Variables

The scripts automatically set:
- `CUDA_VISIBLE_DEVICES` - GPU allocation
- `PYTORCH_CUDA_ALLOC_CONF` - Memory optimization
- `TORCH_CUDA_MEMORY_FRACTION` - 0.95 for training, 0.90 for preprocessing
- `NCCL_DEBUG` - WARN level for distributed training
- `WANDB_MODE` - Configurable (disabled by default)

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
./submit_jobs.sh phase config_name run --batch-size 16

# Enable more aggressive memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

### Job Stuck in Pending
```bash
# Check queue status
squeue -p p6000

# Check job details
scontrol show job <job_id>
```

### Container Issues
```bash
# Verify container exists
ls -la $PROJECT_DIR/singularity/*.sif

# Test container manually
singularity exec --nv ablation.sif python -c "import torch; print(torch.cuda.is_available())"
```

## Best Practices

1. **Always check GPU availability before training**
   ```bash
   ./monitor_jobs.sh -g
   ```

2. **Use batch submission for multiple experiments**
   - Edit `experiments_batch.txt` with your config list
   - Submit with: `./submit_jobs.sh batch experiments_batch.txt`

3. **Monitor long-running jobs**
   ```bash
   # In one terminal
   ./monitor_jobs.sh -j <job_id>
   
   # In another terminal
   tail -f logs/*-<job_id>.out
   ```

4. **Set appropriate time limits**
   - Preprocessing: 1-2 hours
   - Tokenizer training: 30-60 minutes
   - Dataset tokenization: 30-60 minutes
   - Model training: 12-48 hours depending on dataset size

5. **Use job dependencies for sequential workflows**
   ```bash
   JOB1=$(./submit_jobs.sh phase config preprocess)
   JOB2=$(./submit_jobs.sh phase config train-tokenizer --dependency $JOB1)
   JOB3=$(./submit_jobs.sh phase config tokenize-dataset --dependency $JOB2)
   JOB4=$(./submit_jobs.sh phase config run --dependency $JOB3)
   ```

## Files

- `run_experiment_slurm.sh` - Main experiment runner (SLURM batch script)
- `run_phase_slurm.sh` - Single phase runner (SLURM batch script)
- `submit_jobs.sh` - Job submission helper
- `monitor_jobs.sh` - Job monitoring tool
- `experiments_batch.txt` - Example batch submission list
- `README.md` - This file

## Notes

- Logs are saved to `$PROJECT_DIR/logs/` with format `<job_name>-<job_id>.[out|err]`
- The scripts use Singularity/Apptainer containers for reproducibility
- All scripts support both `singularity` and `apptainer` commands
- W&B integration is disabled by default to avoid API key issues