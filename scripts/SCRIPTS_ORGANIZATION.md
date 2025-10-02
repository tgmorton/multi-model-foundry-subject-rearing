# Scripts Organization

Model Foundry training and utility scripts organized by environment and purpose.

## Directory Structure

```
scripts/
├── wild_west/          # Direct execution on shared GPU servers (no SLURM)
│   ├── train.sh        # Main training script
│   └── gpu_monitor.sh  # GPU availability monitoring
│
├── ssrde/              # SLURM cluster scripts (A5000/P6000 nodes)
│   └── train.sh        # SLURM training submission script
│
├── generate_checkpoint_schedule.py  # Checkpoint schedule generation
├── generate_experiment_configs.py   # Config file generation
└── log_manager.py                   # Log analysis utilities
```

## Training Scripts

### Wild-West (Direct Execution)

For servers without job schedulers:

```bash
# Basic training
./scripts/wild_west/train.sh configs/your_config.yaml

# With GPU management
./scripts/wild_west/train.sh --lock-gpus --check-gpus configs/your_config.yaml

# Specific GPUs
./scripts/wild_west/train.sh --gpus 2,3 configs/your_config.yaml
```

**See:** [docs/TRAINING_ON_WILD_WEST.md](../docs/TRAINING_ON_WILD_WEST.md)

### SSRDE (SLURM Cluster)

For SLURM-managed clusters:

```bash
# Basic submission
sbatch scripts/ssrde/train.sh configs/your_config.yaml

# Multi-GPU
sbatch --gres=gpu:4 scripts/ssrde/train.sh configs/your_config.yaml

# Long training
sbatch --time=48:00:00 scripts/ssrde/train.sh configs/your_config.yaml
```

**See:** [docs/TRAINING_ON_SLURM.md](../docs/TRAINING_ON_SLURM.md)

## Utility Scripts

### Checkpoint Schedule Generation

Generate optimal checkpoint schedules:

```bash
python scripts/generate_checkpoint_schedule.py \
    configs/your_config.yaml \
    --first-epoch 20 \
    --spacing log \
    --min-per-epoch 5
```

### Experiment Config Generation

Generate configs for multiple experiments:

```bash
python scripts/generate_experiment_configs.py \
    --base-config configs/base.yaml \
    --output-dir configs/experiments/
```

### Log Management

Analyze and summarize training logs:

```bash
python scripts/log_manager.py analyze logs/experiment_0/
python scripts/log_manager.py summary logs/*/
```

## Deprecated Scripts

The following directories contain legacy scripts and are **no longer maintained**:

### `scripts/a5000/` - DEPRECATED
- **Replaced by:** `scripts/ssrde/`
- **Reason:** Unified A5000 and P6000 scripts into single SSRDE folder

### `scripts/p6000/` - DEPRECATED
- **Replaced by:** `scripts/ssrde/`
- **Reason:** Unified A5000 and P6000 scripts into single SSRDE folder

### `scripts/titanx/` - DEPRECATED
- **Replaced by:** N/A (hardware no longer in use)
- **Reason:** TitanX nodes decommissioned

### `scripts/3070ti/` - DEPRECATED
- **Replaced by:** `scripts/wild_west/`
- **Reason:** Wild-west scripts work on any direct-access GPU server

### Old Evaluation Scripts - DEPRECATED
The following evaluation scripts in `scripts/wild_west/` are old:
- `run_evaluation.sh` - Use evaluation module directly
- `quick_eval.sh` - Use evaluation module directly
- `batch_evaluate.sh` - Use evaluation module directly
- `monitor_evaluation.sh` - Use evaluation module directly
- `export_results.sh` - Use evaluation module directly

**Replacement:** Use evaluation module directly (covered in eval documentation)

### Old Experiment Runners - DEPRECATED
- `scripts/run_experiment.py` - Python orchestrator (use train.sh instead)
- `scripts/run_experiment_workflow.py` - Workflow runner (use train.sh instead)
- `scripts/run_ablation_experiment.sh` - Old ablation runner (configs handle this now)
- `scripts/run_direct_*.sh` - Old direct runners (use wild_west/train.sh)
- `scripts/run_exp0_baseline.sh` - Specific experiment (use train.sh with config)
- `scripts/submit_experiment.sh` - Old submission (use ssrde/train.sh)
- `scripts/tokenize_exp1.sh` - Old tokenization (integrated into training)

### Old Phase Scripts - DEPRECATED
These scripts are in `wild_west/` but no longer needed:
- `run_experiment.sh` - Complex multi-phase runner (use train.sh instead)
- `run_phase.sh` - Individual phase runner (phases integrated into training)

**Replacement:** Model Foundry training handles all phases automatically

## Quick Reference

### For Development/Testing (wild-west)
```bash
./scripts/wild_west/train.sh configs/test.yaml
```

### For Production (SLURM)
```bash
sbatch scripts/ssrde/train.sh configs/production.yaml
```

### Check GPU Status (wild-west)
```bash
./scripts/wild_west/gpu_monitor.sh available
./scripts/wild_west/gpu_monitor.sh watch
```

### Generate Checkpoint Schedule
```bash
python scripts/generate_checkpoint_schedule.py configs/your_config.yaml
```

## Migration Guide

### From Old Scripts to New Scripts

#### Old A5000/P6000 Scripts → SSRDE
```bash
# Old
sbatch scripts/a5000/submit_experiment.sh experiment_1
sbatch scripts/p6000/run_phase_slurm.sh experiment_1 run

# New
sbatch scripts/ssrde/train.sh configs/experiment_1.yaml
```

#### Old TitanX/3070ti Scripts → Wild-West
```bash
# Old
./scripts/titanx/run_direct_full_pipeline.sh experiment_1
./scripts/3070ti/run_experiment.sh experiment_1

# New
./scripts/wild_west/train.sh configs/experiment_1.yaml
```

#### Old Run Scripts → New Train Scripts
```bash
# Old
python scripts/run_experiment.py 1
./scripts/run_direct_training.sh configs/exp1.yaml

# New
./scripts/wild_west/train.sh configs/experiment_1.yaml
```

## Why This Reorganization?

### Problems with Old Structure
1. **Fragmentation**: Different scripts for A5000, P6000, TitanX, 3070ti
2. **Duplication**: Same logic repeated across multiple scripts
3. **Complexity**: Multi-phase runners with complex state management
4. **Maintenance**: Hard to update all scripts consistently
5. **User Confusion**: Too many scripts doing similar things

### Benefits of New Structure
1. **Simplicity**: Two main scripts (`wild_west/train.sh`, `ssrde/train.sh`)
2. **Consistency**: Same config works everywhere
3. **Safety**: Built-in process management and GPU locking
4. **Clarity**: Clear separation by environment (SLURM vs direct)
5. **Maintainability**: Update one script per environment

## What to Keep

### Essential Scripts (DO NOT DELETE)
- ✅ `wild_west/train.sh` - Main wild-west trainer
- ✅ `wild_west/gpu_monitor.sh` - GPU monitoring
- ✅ `ssrde/train.sh` - SLURM trainer
- ✅ `generate_checkpoint_schedule.py` - Still useful
- ✅ `generate_experiment_configs.py` - Still useful
- ✅ `log_manager.py` - Still useful

### Deprecated (SAFE TO DELETE)
- ❌ `a5000/` - Entire directory
- ❌ `p6000/` - Entire directory
- ❌ `titanx/` - Entire directory
- ❌ `3070ti/` - Entire directory
- ❌ `run_experiment.py`
- ❌ `run_experiment_workflow.py`
- ❌ `run_ablation_experiment.sh`
- ❌ `run_direct_*.sh` (all 4 scripts)
- ❌ `run_exp0_baseline.sh`
- ❌ `submit_experiment.sh`
- ❌ `tokenize_exp1.sh`
- ❌ `test_chunking_*.sh` (all 3 scripts)
- ❌ `test_chunking_simple.py`
- ❌ `check_*.sh` scripts (old data checking)
- ❌ `compile_*.py` scripts (old compilation)
- ❌ `wild_west/run_experiment.sh` - Old multi-phase runner
- ❌ `wild_west/run_phase.sh` - Old phase runner
- ❌ `wild_west/run_evaluation.sh` - Old eval runner
- ❌ `wild_west/quick_eval.sh` - Old eval runner
- ❌ `wild_west/batch_evaluate.sh` - Old eval runner
- ❌ `wild_west/monitor_evaluation.sh` - Old eval runner
- ❌ `wild_west/export_results.sh` - Old eval runner

### Documentation (SAFE TO DELETE from scripts/)
- ❌ `scripts/README.md` - Old, replaced by this file
- ❌ `scripts/README_COMPILATION.md` - Old compilation docs
- ❌ `scripts/wild_west/*.md` - All old READMEs (content moved to docs/)

## Summary

**Use these two scripts for training:**
- `scripts/wild_west/train.sh` - For direct GPU access
- `scripts/ssrde/train.sh` - For SLURM clusters

**Everything else is either:**
- A utility script (checkpoint generation, log management)
- Deprecated and safe to remove

**For full documentation, see:**
- [docs/TRAINING_ON_WILD_WEST.md](../docs/TRAINING_ON_WILD_WEST.md)
- [docs/TRAINING_ON_SLURM.md](../docs/TRAINING_ON_SLURM.md)
