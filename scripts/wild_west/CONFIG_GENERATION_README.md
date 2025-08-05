# Experiment Configuration Generation

This directory contains a simplified approach for generating experiment configurations based on processed data.

## Quick Start

### Generate All Configurations
```bash
# Generate YAML configs for all processed experiments
./scripts/wild_west/generate_configs.sh
```

This will:
- Scan the `data/processed/` directory for all experiments
- Generate YAML configuration files in `configs/processed_experiments/`
- Automatically generate checkpoint schedules for each config
- Use the Singularity container for consistency

### Run Experiments
```bash
# Use the existing wild_west experiment runner
./scripts/wild_west/run_experiment.sh configs/processed_experiments/experiment_1_remove_expletives.yaml

# With specific GPUs
./scripts/wild_west/run_experiment.sh -g 1,2 configs/processed_experiments/experiment_2_impoverish_determiners.yaml

# Run specific phases
./scripts/wild_west/run_experiment.sh -p preprocess configs/processed_experiments/experiment_3_no_expletives_no_articles.yaml
```

## What Gets Generated

The script creates YAML configs for all experiments found in `data/processed/`:

- `configs/processed_experiments/experiment_1_remove_expletives.yaml`
- `configs/processed_experiments/experiment_2_impoverish_determiners.yaml`
- `configs/processed_experiments/experiment_3_no_expletives_no_articles.yaml`
- etc.

Each config includes:
- **Data paths** pointing to the processed experiment data
- **Model architecture** (GPT-2, 12 layers, 768 hidden size)
- **Training parameters** (1M steps, mixed precision, etc.)
- **Checkpoint schedules** (auto-generated based on dataset size)
- **Enhanced features** (AMP, gradient accumulation, TF32, etc.)

## Configuration Features

All generated configs include the enhanced features from the `next.md` tasks:

- **Mixed Precision (AMP)** - Faster training with reduced memory usage
- **Gradient Accumulation** - Support for large effective batch sizes
- **TF32** - Optimized math operations on Ampere+ GPUs
- **Deterministic Seeding** - Reproducible results across runs
- **Memory Optimization** - Streaming data processing to avoid OOM
- **Enhanced Logging** - Real-time metrics and ETA calculations
- **Environment Snapshots** - Complete reproducibility tracking
- **Checkpoint Metadata** - Provenance tracking for all checkpoints

## Integration with Existing Workflow

This approach integrates seamlessly with the existing wild_west system:

1. **Generate configs**: `./scripts/wild_west/generate_configs.sh`
2. **Run experiments**: Use existing `run_experiment.sh` with the generated configs
3. **GPU management**: Use existing `gpu_monitor.sh` and locking system
4. **Singularity containers**: All runs use the same container environment

## Example Workflow

```bash
# 1. Generate all configurations
./scripts/wild_west/generate_configs.sh

# 2. Check available GPUs
./scripts/wild_west/gpu_monitor.sh available

# 3. Run an experiment
./scripts/wild_west/run_experiment.sh -g 1,2 configs/processed_experiments/experiment_1_remove_expletives.yaml

# 4. Monitor progress
./scripts/wild_west/gpu_monitor.sh watch
```

## Troubleshooting

### Common Issues

1. **Singularity Image Not Found**
   ```bash
   # Build the training image
   singularity build singularity/training.sif singularity/training.def
   ```

2. **No Processed Data Found**
   ```bash
   # Check if processed data exists
   ls data/processed/
   ```

3. **Python Script Not Found**
   ```bash
   # Ensure you're in the project root
   pwd  # Should show /path/to/subject-drop-rearing
   ```

### Logs and Output

- **Generated configs**: `configs/processed_experiments/`
- **Training logs**: `logs/<experiment_name>/`
- **Checkpoints**: `models/<experiment_name>/checkpoint-*/`
- **WandB**: Project "just-drop-the-subject"

## Advantages of This Approach

1. **Simplicity** - Just generates YAML configs, no complex scripts
2. **Integration** - Works with existing wild_west infrastructure
3. **Consistency** - Uses same Singularity container as training
4. **Flexibility** - Can use existing GPU management and monitoring
5. **Maintainability** - Leverages existing, tested infrastructure

## File Structure

```
scripts/wild_west/
├── generate_configs.sh           # Generate experiment configs
├── run_experiment.sh             # Existing experiment runner
├── gpu_monitor.sh                # GPU management
├── CONFIG_GENERATION_README.md   # This file
└── ...
```

This approach is much cleaner and integrates better with your existing workflow! 