# Wild-West Evaluation Scripts

This directory contains scripts for running and managing the evaluation pipeline in a wild-west GPU environment.

## Scripts Overview

### 🔧 Core Evaluation Scripts

1. **`run_evaluation.sh`** - Main evaluation runner
2. **`batch_evaluate.sh`** - Batch evaluation across multiple experiments  
3. **`quick_eval.sh`** - Fast evaluation for testing
4. **`monitor_evaluation.sh`** - Monitor running evaluations
5. **`export_results.sh`** - Export results for R analysis

## Quick Start

### 1. Single Experiment Evaluation
```bash
# Evaluate exp0_baseline experiment
./scripts/wild_west/run_evaluation.sh exp0_baseline

# Fast test evaluation
./scripts/wild_west/quick_eval.sh exp0_baseline

# Evaluate with specific settings
./scripts/wild_west/run_evaluation.sh -g 1 -t blimp -s 100 exp0_baseline
```

### 2. Batch Evaluation
```bash
# Evaluate multiple experiments
./scripts/wild_west/batch_evaluate.sh exp0_baseline exp1_remove_expletives exp2_poor_determiners

# Parallel evaluation on multiple GPUs
./scripts/wild_west/batch_evaluate.sh -p 2 -g 0,1 exp0_baseline exp1_remove_expletives

# Fast batch evaluation
./scripts/wild_west/batch_evaluate.sh --fast exp0_baseline exp1_remove_expletives
```

### 3. Monitoring
```bash
# Check evaluation status
./scripts/wild_west/monitor_evaluation.sh status

# Watch evaluations in real-time
./scripts/wild_west/monitor_evaluation.sh watch

# Show detailed progress
./scripts/wild_west/monitor_evaluation.sh progress
```

### 4. Export Results
```bash
# Export all experiments for R analysis
./scripts/wild_west/export_results.sh

# Export with plots and compression
./scripts/wild_west/export_results.sh -p -c

# Export specific experiments
./scripts/wild_west/export_results.sh exp0_baseline exp1_remove_expletives
```

## Detailed Usage

### run_evaluation.sh

Run evaluation pipeline on a single experiment.

```bash
Usage: ./scripts/wild_west/run_evaluation.sh [OPTIONS] [experiment_name]

Options:
  -g, --gpus <ids>           GPU IDs to use (default: 1)
  -c, --config <config>      Evaluation config file
  -e, --experiment <exp>     Experiment to evaluate
  -k, --checkpoint <path>    Specific checkpoint to evaluate
  -t, --tasks <tasks>        Tasks: perplexity,blimp,null_subject
  -s, --samples <num>        Max samples per task (testing)
  -m, --max-checkpoints <n>  Max checkpoints to evaluate
  -l, --lock-gpus           Lock GPUs during evaluation
  -f, --fast                Fast evaluation mode
  -v, --verbose             Verbose output

Examples:
  # Basic evaluation
  ./scripts/wild_west/run_evaluation.sh exp0_baseline
  
  # Fast test with limited samples
  ./scripts/wild_west/run_evaluation.sh -f -s 50 -m 3 exp0_baseline
  
  # Only BLIMP evaluation
  ./scripts/wild_west/run_evaluation.sh -t blimp exp0_baseline
  
  # Specific checkpoint
  ./scripts/wild_west/run_evaluation.sh -k models/exp0_baseline/epoch_10/ exp0_baseline
```

### batch_evaluate.sh

Run evaluation on multiple experiments.

```bash
Usage: ./scripts/wild_west/batch_evaluate.sh [OPTIONS] [experiment1] [experiment2] ...

Options:
  -f, --experiments-file <file>  File with experiment names
  -p, --parallel <n>            Run n evaluations in parallel
  -g, --gpus <ids>              GPU IDs for round-robin assignment
  --fast                        Fast evaluation mode
  --dry-run                     Show what would be run
  
Examples:
  # Sequential evaluation
  ./scripts/wild_west/batch_evaluate.sh exp0_baseline exp1_remove_expletives
  
  # Parallel evaluation
  ./scripts/wild_west/batch_evaluate.sh -p 2 -g 0,1 exp0_baseline exp1_remove_expletives
  
  # From file
  echo -e "exp0_baseline\nexp1_remove_expletives" > experiments.txt
  ./scripts/wild_west/batch_evaluate.sh -f experiments.txt
```

### quick_eval.sh

Fast evaluation for testing and debugging.

```bash
Usage: ./scripts/wild_west/quick_eval.sh [OPTIONS] <experiment>

Options:
  -g, --gpu <id>            GPU ID to use (auto-detected)
  -s, --samples <num>       Max samples per task (default: 50)
  -c, --checkpoints <num>   Max checkpoints (default: 3)
  -t, --task <task>         Single task to run
  --test-setup              Test evaluation environment
  
Examples:
  # Quick test
  ./scripts/wild_west/quick_eval.sh exp0_baseline
  
  # Minimal test
  ./scripts/wild_west/quick_eval.sh -s 10 -c 1 exp0_baseline
  
  # Test setup
  ./scripts/wild_west/quick_eval.sh --test-setup
```

### monitor_evaluation.sh

Monitor running evaluations and show progress.

```bash
Usage: ./scripts/wild_west/monitor_evaluation.sh [OPTIONS] [COMMAND]

Commands:
  status        Current evaluation status
  progress      Detailed progress
  results       Latest results summary
  watch         Continuous monitoring
  cleanup       Clean up old files
  kill          Kill running evaluations
  
Examples:
  # Show status
  ./scripts/wild_west/monitor_evaluation.sh status
  
  # Watch all evaluations
  ./scripts/wild_west/monitor_evaluation.sh watch
  
  # Monitor specific experiment
  ./scripts/wild_west/monitor_evaluation.sh -e exp0_baseline progress
```

### export_results.sh

Export evaluation results for analysis.

```bash
Usage: ./scripts/wild_west/export_results.sh [OPTIONS] [experiments...]

Options:
  -o, --output <dir>        Output directory
  -f, --format <formats>    Export formats: csv,json,rds
  -a, --aggregate           Create cross-experiment file
  -p, --plots               Generate plots
  -c, --compress            Compress output
  
Examples:
  # Export all experiments
  ./scripts/wild_west/export_results.sh
  
  # Export with plots and aggregation
  ./scripts/wild_west/export_results.sh -a -p
  
  # Export specific experiments
  ./scripts/wild_west/export_results.sh exp0_baseline exp1_remove_expletives
```

## Configuration

### Default Settings
- **GPU**: Auto-detected available GPU (typically GPU 1)
- **Samples**: Unlimited (full evaluation)
- **Tasks**: All tasks (perplexity, BLIMP, null-subject)
- **Output**: `evaluation/results/<experiment>/`

### Environment Requirements
- **Singularity**: Uses `training.sif` container if available
- **GPU**: CUDA-capable GPU with >10GB memory
- **Python**: Python 3.9+ with required packages

### Speed Optimization
- **Fast mode** (`-f`): Limits samples to 100 per task, 3 checkpoints
- **GPU locking**: Prevents conflicts with other users
- **Parallel evaluation**: Multiple experiments simultaneously

## Output Structure

```
evaluation/results/
├── exp0_baseline/
│   ├── evaluation_results.jsonl      # Main results
│   ├── epoch_1_blimp_detailed.jsonl  # Detailed BLIMP results
│   ├── epoch_1_null_subject_detailed.jsonl  # Null-subject results
│   └── epoch_1_perplexity.json       # Perplexity breakdown
├── exp1_remove_expletives/
│   └── ...
└── cross_experiment_comparison.md    # Batch evaluation summary
```

### Export Structure
```
evaluation/exports/
├── exp0_baseline_metrics.csv          # Main metrics for R
├── exp0_baseline_learning_curves.csv  # Learning curves
├── exp0_baseline_summary.json         # Summary statistics
├── cross_experiment_comparison.csv    # All experiments combined
├── plots/
│   ├── learning_curves.png            # Learning curve plots
│   └── final_comparison.png           # Final performance comparison
└── evaluation_results_YYYYMMDD.tar.gz # Compressed archive
```

## Performance Expectations

### GPU Requirements
- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **Optimal**: 24GB+ GPU memory (V100, A100, RTX 3090/4090)

### Time Estimates (per checkpoint)
- **Perplexity** (10M tokens): 5-10 minutes
- **BLIMP** (67 tasks): 15-20 minutes  
- **Null-subject** (14 conditions): 10-15 minutes
- **Total**: ~30-45 minutes per checkpoint

### Fast Mode Times
- **Perplexity** (limited): 1-2 minutes
- **BLIMP** (100 samples): 2-3 minutes
- **Null-subject** (limited): 2-3 minutes
- **Total**: ~5-10 minutes per checkpoint

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   ```bash
   # Use smaller batch size or CPU
   ./scripts/wild_west/run_evaluation.sh -g cpu exp0_baseline
   ```

2. **Missing Checkpoints**
   ```bash
   # Check available experiments
   ls models/
   ```

3. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x scripts/wild_west/*.sh
   ```

4. **Python Import Errors**
   ```bash
   # Test setup
   ./scripts/wild_west/quick_eval.sh --test-setup
   ```

5. **Singularity Issues**
   ```bash
   # Check container exists
   ls -la singularity/training.sif
   
   # Rebuild if needed
   singularity build singularity/training.sif singularity/training.def
   ```

### Monitoring Commands

```bash
# Check GPU status
./scripts/wild_west/gpu_monitor.sh status

# Watch GPU usage
watch -n 2 nvidia-smi

# Check running evaluations
./scripts/wild_west/monitor_evaluation.sh status

# Kill stuck evaluations
./scripts/wild_west/monitor_evaluation.sh kill
```

### Log Files

- **Evaluation logs**: `/tmp/eval_<experiment>_gpu<id>_<pid>.log`
- **GPU monitor logs**: `/tmp/gpu_monitor_<user>.log`
- **Process logs**: Check with `./scripts/wild_west/monitor_evaluation.sh progress`

## Integration with R

The exported CSV files are designed for R analysis:

```r
# Load data
library(tidyverse)
library(lme4)

# Load metrics
metrics <- read_csv("evaluation/exports/cross_experiment_comparison.csv")

# Plot learning curves
metrics %>%
  filter(metric == "perplexity") %>%
  ggplot(aes(x = epoch, y = value, color = experiment)) +
  geom_line() +
  geom_point() +
  labs(title = "Perplexity Learning Curves")

# Statistical analysis
model <- lmer(value ~ experiment + (1|metric), data = metrics)
summary(model)
```

## Best Practices

1. **GPU Etiquette**: Always lock GPUs for long evaluations
2. **Testing First**: Use `quick_eval.sh` before full evaluation
3. **Monitoring**: Check progress with `monitor_evaluation.sh`
4. **Cleanup**: Regular cleanup of old logs and temp files
5. **Backup**: Export results regularly for analysis

This evaluation system provides comprehensive assessment of your language models while being mindful of shared GPU resources.