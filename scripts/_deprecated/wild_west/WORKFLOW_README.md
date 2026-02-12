# Wild-West Experiment Workflow

This directory contains scripts for running experiments on the wild-west server with proper GPU management and Singularity containers.

## Quick Start

### 1. Generate All Configurations
```bash
# Generate configurations for all experiments based on processed data
./scripts/wild_west/generate_all_configs.sh
```

This will:
- Scan the `data/processed/` directory for all experiments
- Generate configuration files in `configs/`
- Create run scripts in `scripts/wild_west/experiments/`
- Create Experiment 0 baseline configuration

### 2. Run Experiment 0 (Baseline)
```bash
# Run Experiment 0 with default GPUs (1,2)
./scripts/wild_west/run_exp0_baseline.sh

# Run on specific GPUs
./scripts/wild_west/run_exp0_baseline.sh -g 0,1
```

### 3. Run Other Experiments
```bash
# Run any experiment using the generated scripts
./scripts/wild_west/experiments/run_experiment_1_remove_expletives.sh

# Run with specific GPUs
./scripts/wild_west/experiments/run_experiment_1_remove_expletives.sh -g 2,3
```

## Available Scripts

### Configuration Generation
- `generate_all_configs.sh` - Generate configurations for all experiments

### Experiment Runners
- `run_exp0_baseline.sh` - Run Experiment 0 baseline with 90M data
- `experiments/run_*.sh` - Generated scripts for each experiment

### GPU Management
- `gpu_monitor.sh` - Monitor and manage GPU availability
- `run_experiment.sh` - Original experiment runner (legacy)

## Experiment Workflow

Each experiment follows this workflow:

1. **Check GPU Availability** - Verify requested GPUs are free
2. **Lock GPUs** - Reserve GPUs for the experiment
3. **Generate Checkpoint Schedule** - Create optimal checkpoint timing
4. **Train Tokenizer** - Create vocabulary for the experiment
5. **Tokenize Dataset** - Convert text to token sequences
6. **Run Training** - Train the model with all optimizations
7. **Unlock GPUs** - Release GPUs when complete

## Configuration Features

All experiments include these enhanced features:

- **Mixed Precision (AMP)** - Faster training with reduced memory usage
- **Gradient Accumulation** - Support for large effective batch sizes
- **TF32** - Optimized math operations on Ampere+ GPUs
- **Deterministic Seeding** - Reproducible results across runs
- **Memory Optimization** - Streaming data processing to avoid OOM
- **Enhanced Logging** - Real-time metrics and ETA calculations
- **Environment Snapshots** - Complete reproducibility tracking
- **Checkpoint Metadata** - Provenance tracking for all checkpoints

## GPU Management

### Check GPU Status
```bash
# Show current GPU status
./scripts/wild_west/gpu_monitor.sh

# Find available GPUs
./scripts/wild_west/gpu_monitor.sh available

# Watch GPU status in real-time
./scripts/wild_west/gpu_monitor.sh watch
```

### GPU Categories
- **AVAILABLE** (>20GB free) - Ideal for large experiments
- **LIMITED** (10-20GB free) - Good for smaller experiments  
- **OCCUPIED** (<10GB free) - Avoid for new experiments

## Experiment 0: Baseline

The baseline experiment uses:
- **Data**: `data/raw/train_90M/` (unaltered BabyLM corpus)
- **Model**: GPT-2 architecture (12 layers, 768 hidden size)
- **Training**: 1M steps, 20 epochs, mixed precision
- **Checkpoints**: Auto-generated schedule with 20 first-epoch checkpoints

### Run Experiment 0
```bash
# Quick start
./scripts/wild_west/run_exp0_baseline.sh

# With specific GPUs
./scripts/wild_west/run_exp0_baseline.sh -g 0,1

# Check what will be run (dry run)
./scripts/wild_west/run_exp0_baseline.sh --help
```

## Troubleshooting

### Common Issues

1. **Singularity Image Not Found**
   ```bash
   # Build the training image
   singularity build singularity/training.sif singularity/training.def
   ```

2. **No Available GPUs**
   ```bash
   # Check GPU status
   ./scripts/wild_west/gpu_monitor.sh available
   
   # Wait for GPUs to become available
   ./scripts/wild_west/gpu_monitor.sh watch
   ```

3. **Configuration File Not Found**
   ```bash
   # Generate configurations first
   ./scripts/wild_west/generate_all_configs.sh
   ```

4. **Memory Issues**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use fewer GPUs

### Logs and Monitoring

- **Training Logs**: `logs/exp0_baseline_90M/`
- **Environment Snapshots**: `logs/exp0_baseline_90M/env_*.txt`
- **Checkpoints**: `models/exp0_baseline_90M/checkpoint-*/`
- **WandB**: Project "just-drop-the-subject"

## File Structure

```
scripts/wild_west/
├── generate_all_configs.sh      # Generate all experiment configs
├── run_exp0_baseline.sh         # Run Experiment 0 baseline
├── gpu_monitor.sh               # GPU management
├── run_experiment.sh            # Legacy experiment runner
├── experiments/                 # Generated run scripts
│   ├── run_experiment_1_*.sh
│   ├── run_experiment_2_*.sh
│   └── ...
└── WORKFLOW_README.md           # This file
```

## Next Steps

1. **Generate Configurations**: `./scripts/wild_west/generate_all_configs.sh`
2. **Run Experiment 0**: `./scripts/wild_west/run_exp0_baseline.sh`
3. **Monitor Progress**: Check logs and WandB dashboard
4. **Run Ablations**: Use generated scripts for other experiments

## Support

For issues or questions:
1. Check GPU availability with `gpu_monitor.sh`
2. Review logs in `logs/` directory
3. Check WandB dashboard for training metrics
4. Verify Singularity image exists and is up to date 