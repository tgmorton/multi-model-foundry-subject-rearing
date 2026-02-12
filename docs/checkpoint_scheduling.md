# Checkpoint Scheduling System

This document describes the checkpoint scheduling system implemented in Phase 2 of the Model Foundry framework.

## Overview

The checkpoint scheduling system provides intelligent, adaptive checkpoint generation based on dataset characteristics and training parameters. It ensures optimal checkpoint frequency while balancing storage efficiency and training monitoring needs.

## Key Features

### Adaptive Scheduling
- **Dataset Size Detection**: Automatically estimates dataset size and adjusts checkpoint frequency
- **Log-Based Early Checkpointing**: Dense checkpointing during the first epoch for detailed early training monitoring
- **Epoch Boundary Checkpoints**: Ensures checkpoints at epoch boundaries for consistent evaluation
- **Distributed Gap Filling**: Intelligently distributes additional checkpoints across training gaps

### Configuration-Driven
- **Target Checkpoints**: Configurable checkpoint counts for different dataset sizes
- **Minimum Intervals**: Prevents excessive checkpointing with minimum interval constraints
- **Flexible Generation**: Supports both manual and automatic schedule generation

## Usage

### CLI Commands

1. **Generate checkpoint schedule**:
   ```bash
   python -m model_foundry.cli generate-checkpoints configs/experiment.yaml
   ```

2. **Generate with custom parameters**:
   ```bash
   python -m model_foundry.cli generate-checkpoints configs/experiment.yaml \
     --targets "small:10,medium:20,large:30,xlarge:40" \
     --min-interval 200 \
     --no-log-steps
   ```

3. **Save to separate file**:
   ```bash
   python -m model_foundry.cli generate-checkpoints configs/experiment.yaml \
     --output configs/experiments/experiment_with_schedule.yaml
   ```

### Direct Script Usage

```bash
python scripts/generate_checkpoint_schedule.py configs/experiment.yaml
```

## Configuration

### Training Configuration

Add checkpoint generation parameters to your experiment config:

```yaml
training:
  # ... existing parameters ...
  
  # Checkpoint generation parameters
  auto_generate_checkpoints: true  # Enable automatic generation
  target_checkpoints:
    small: 20    # ~10M tokens
    medium: 50   # ~25M tokens
    large: 100   # ~50M tokens
    xlarge: 200  # ~100M tokens
  log_steps_first_epoch: true     # Enable log-based early checkpointing
  min_checkpoint_interval: 100    # Minimum steps between checkpoints
```

### Dataset Size Categories

The system automatically categorizes datasets:

- **small**: < 10M tokens
- **medium**: 10M - 25M tokens  
- **large**: 25M - 50M tokens
- **xlarge**: > 50M tokens

## Algorithm Details

### 1. Log-Based Early Checkpointing
Generates checkpoints at powers of 2 during the first epoch:
```
Steps: 1, 2, 4, 8, 16, 32, 64, 128, ...
```

### 2. Epoch Boundary Checkpoints
Ensures checkpoints at the end of each epoch for consistent evaluation.

### 3. Gap Distribution
When additional checkpoints are needed, they are distributed evenly across gaps between existing checkpoints.

### 4. Final Step
Always includes the final training step to capture the fully trained model.

## Example Schedules

### Small Dataset (10M tokens, 20 target checkpoints)
```
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
```

### Medium Dataset (25M tokens, 50 target checkpoints)
```
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000]
```

## Integration with Training

The checkpoint schedule is automatically used during training:

```python
# In trainer.py
checkpoint_schedule = self._get_checkpoint_schedule()

for step in training_steps:
    # ... training logic ...
    
    if step in checkpoint_schedule:
        self._save_checkpoint()
```

## Auto-Generation

When `auto_generate_checkpoints: true` is set in the config, the trainer will automatically generate a schedule if none exists:

```yaml
training:
  auto_generate_checkpoints: true
  # No checkpoint_schedule needed - will be generated automatically
```

## Benefits

1. **Intelligent Adaptation**: Automatically adjusts to dataset size and training parameters
2. **Storage Efficiency**: Balances checkpoint frequency with storage requirements
3. **Monitoring Coverage**: Ensures adequate checkpointing for training analysis
4. **Flexibility**: Supports both manual and automatic generation
5. **Reproducibility**: Deterministic schedules for consistent experiments

## Testing

Run the test suite to validate the checkpoint scheduling:

```bash
python tests/test_checkpoint_scheduling.py
```

This tests:
- Configuration parsing and validation
- Log-based step generation
- Dataset size estimation
- Schedule generation and distribution
- Integration with the configuration system

## Advanced Usage

### Custom Target Checkpoints

```bash
python -m model_foundry.cli generate-checkpoints configs/experiment.yaml \
  --targets "small:5,medium:15,large:25,xlarge:35"
```

### Disable Log-Based Checkpointing

```bash
python -m model_foundry.cli generate-checkpoints configs/experiment.yaml \
  --no-log-steps
```

### Custom Minimum Interval

```bash
python -m model_foundry.cli generate-checkpoints configs/experiment.yaml \
  --min-interval 500
```

## File Structure

```
scripts/
└── generate_checkpoint_schedule.py  # Main generation script

configs/
├── experiment.yaml                   # Original config
└── experiment_with_schedule.yaml    # Config with generated schedule

tests/
└── test_checkpoint_scheduling.py   # Test suite
``` 