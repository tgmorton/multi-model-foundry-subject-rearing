# Weights & Biases (WandB) Integration Guide

## Table of Contents
1. [Overview](#overview)
2. [Setup & Installation](#setup--installation)
3. [Account Configuration](#account-configuration)
4. [Integration with Model Foundry](#integration-with-model-foundry)
5. [Configuration Options](#configuration-options)
6. [Usage Examples](#usage-examples)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Weights & Biases (WandB) is a powerful experiment tracking and visualization platform for machine learning. This guide explains how to integrate WandB with the Model Foundry training framework to:

- **Track training metrics** (loss, learning rate, gradient norms, etc.)
- **Monitor system resources** (GPU memory, throughput, etc.)
- **Compare experiments** across different configurations
- **Visualize training progress** in real-time
- **Share results** with collaborators
- **Reproduce experiments** with saved configurations

---

## Setup & Installation

### 1. Install WandB

WandB should already be installed as a dependency. Verify installation:

```bash
pip show wandb
```

If not installed:

```bash
pip install wandb
```

### 2. Create a WandB Account

1. Visit [https://wandb.ai/signup](https://wandb.ai/signup)
2. Create a free account (or use GitHub/Google sign-in)
3. Free tier includes:
   - Unlimited public projects
   - 100 GB storage
   - Unlimited logged runs

**Academic/Research accounts** get additional features:
- Visit [wandb.ai/academic](https://wandb.ai/academic)
- Request academic plan with your .edu email

### 3. Get Your API Key

After creating your account:

1. Go to [wandb.ai/authorize](https://wandb.ai/authorize)
2. Copy your API key (40-character string)
3. The key looks like: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0`

---

## Account Configuration

### Method 1: Interactive Login (Recommended)

Run this command once on your machine:

```bash
wandb login
```

You'll be prompted to paste your API key. This stores it in `~/.netrc` for future use.

**Output:**
```
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
```

After pasting your key:
```
wandb: Appending key for api.wandb.ai to your netrc file: /Users/yourname/.netrc
```

### Method 2: Environment Variable

Set your API key as an environment variable:

```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.bash_profile
export WANDB_API_KEY="your-40-character-api-key-here"
```

Then reload your shell:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### Method 3: Manual .netrc Configuration

Create or edit `~/.netrc`:

```bash
machine api.wandb.ai
  login user
  password your-40-character-api-key-here
```

Set proper permissions:
```bash
chmod 600 ~/.netrc
```

### Verify Configuration

Test that your API key is configured:

```bash
wandb login --relogin
```

Or run a quick test:

```bash
python -c "import wandb; wandb.login()"
```

---

## Integration with Model Foundry

### Configuration File Setup

Edit your experiment YAML configuration to enable WandB:

```yaml
# configs/experiment_with_wandb.yaml

experiment_name: "exp0_baseline_wandb"

# ... other configs ...

logging:
  # Basic logging
  console_level: "INFO"
  file_level: "DEBUG"
  dir: "logs"

  # Enable WandB
  use_wandb: true
  wandb_project: "model-foundry-experiments"  # Your project name

  # Structured logging
  use_structured_logging: true

  # Metrics logging frequency
  log_metrics_every_n_steps: 10
  log_detailed_metrics_every_n_steps: 100

  # Performance profiling (optional - logs to WandB)
  profile_performance: true
  log_memory_every_n_steps: 100
```

### Project Organization

**WandB Projects** organize related experiments:

```yaml
# Research project
wandb_project: "spanish-subject-drop"

# Ablation study
wandb_project: "spanish-subject-drop-ablations"

# Architecture comparison
wandb_project: "gpt2-vs-llama-comparison"
```

**Entity (Team/User):** Optionally specify your WandB username or team:

```yaml
wandb_entity: "your-username"  # or "your-team-name"
wandb_project: "model-foundry-experiments"
```

---

## Configuration Options

### Full LoggingConfig with WandB

```python
from model_foundry.config import LoggingConfig

logging_config = LoggingConfig(
    # WandB settings
    use_wandb=True,
    wandb_project="my-project-name",

    # Log levels
    console_level="INFO",
    file_level="DEBUG",

    # Metrics logging
    log_metrics_every_n_steps=10,        # Log to WandB every 10 steps
    log_detailed_metrics_every_n_steps=100,  # Detailed logs every 100 steps

    # Performance monitoring
    profile_performance=True,            # Enable performance profiling
    log_memory_every_n_steps=50,        # Log GPU memory every 50 steps

    # Local logging
    use_structured_logging=True,
    dir="logs",
    max_log_files=10
)
```

### Environment Variables for WandB

Additional control via environment variables:

```bash
# Disable WandB (override config)
export WANDB_MODE=disabled

# Run in offline mode (sync later)
export WANDB_MODE=offline

# Silent mode (no console output from WandB)
export WANDB_SILENT=true

# Custom base URL (for self-hosted WandB)
export WANDB_BASE_URL=https://your-wandb-server.com

# Specify project (overrides config)
export WANDB_PROJECT=my-experiment

# Specify entity
export WANDB_ENTITY=my-team

# Disable code saving
export WANDB_DISABLE_CODE=true
```

---

## Usage Examples

### Example 1: Basic Training with WandB

```python
from model_foundry.trainer import Trainer
from model_foundry.config import ExperimentConfig
import yaml

# Load config with WandB enabled
with open('configs/experiment_with_wandb.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

config = ExperimentConfig(**config_dict)

# Initialize trainer (WandB will auto-initialize)
trainer = Trainer(config, base_dir=".")

# Start training - metrics automatically logged to WandB
trainer.train()
```

**What gets logged:**
- Training loss (every N steps)
- Learning rate schedule
- Gradient norms
- Tokens per second
- GPU memory usage
- System metrics
- Model checkpoints (optional)

### Example 2: Manual WandB Logging

```python
from model_foundry.logging_components import WandBLogger

# Initialize WandB logger
wandb_logger = WandBLogger(
    project="model-foundry",
    name="exp0_baseline",
    config=config.dict(),  # Log full experiment config
    tags=["baseline", "gpt2", "spanish"]
)

# Log training metrics
wandb_logger.log_metrics(
    step=100,
    metrics={
        "train/loss": 2.456,
        "train/lr": 0.001,
        "train/grad_norm": 1.23,
        "train/tokens_per_sec": 8500
    }
)

# Log system metrics
wandb_logger.log_system_metrics()

# Finish run
wandb_logger.finish()
```

### Example 3: Comparing Multiple Experiments

Run multiple experiments with different configs:

```bash
# Baseline
python -m model_foundry.cli train configs/exp0_baseline.yaml

# Remove expletives
python -m model_foundry.cli train configs/exp1_remove_expletives.yaml

# Remove topic shift
python -m model_foundry.cli train configs/exp2_remove_topic_shift.yaml
```

All experiments appear in the same WandB project for easy comparison.

### Example 4: Offline Mode + Later Sync

If training on a cluster without internet:

```bash
# Set offline mode
export WANDB_MODE=offline

# Run training (logs saved locally)
python -m model_foundry.cli train configs/experiment.yaml

# Later, sync to cloud
wandb sync wandb/offline-run-YYYYMMDD_HHMMSS-<run_id>
```

---

## Advanced Features

### 1. Custom Metrics Grouping

Organize metrics with prefixes:

```python
wandb_logger.log_metrics(
    step=100,
    metrics={
        # Training metrics
        "train/loss": 2.5,
        "train/perplexity": 12.18,

        # Validation metrics
        "val/loss": 2.7,
        "val/perplexity": 14.88,

        # System metrics
        "system/gpu_memory_gb": 3.2,
        "system/tokens_per_sec": 8500,

        # Learning rate
        "optimization/lr": 0.001,
        "optimization/grad_norm": 1.23
    }
)
```

These appear as organized groups in the WandB dashboard.

### 2. Hyperparameter Sweeps

Create a sweep configuration:

```yaml
# sweep_config.yaml
program: model_foundry.cli
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.001
  batch_size:
    values: [16, 32, 64]
  dropout:
    distribution: uniform
    min: 0.0
    max: 0.3
```

Run sweep:

```bash
# Initialize sweep
wandb sweep sweep_config.yaml

# Run agents (can run multiple in parallel)
wandb agent your-entity/your-project/sweep-id
```

### 3. Log Model Artifacts

Save model checkpoints to WandB:

```python
# In your training code
import wandb

# Save checkpoint as artifact
artifact = wandb.Artifact(
    name=f"model-checkpoint-{step}",
    type="model",
    description=f"Model checkpoint at step {step}"
)
artifact.add_dir("output/checkpoint-1000")
wandb.log_artifact(artifact)
```

### 4. Log Training Curves as Images

```python
import matplotlib.pyplot as plt
import wandb

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")

# Log to WandB
wandb.log({"charts/loss_curve": wandb.Image(plt)})
plt.close()
```

### 5. Log Code and Git Info

WandB automatically logs:
- Git commit hash
- Git branch
- Git remote URL
- Git diff (uncommitted changes)
- Code files

Disable if needed:
```bash
export WANDB_DISABLE_CODE=true
export WANDB_DISABLE_GIT=true
```

### 6. Custom Tables

Log structured data:

```python
# Create table
table = wandb.Table(
    columns=["epoch", "step", "loss", "perplexity"],
    data=[
        [1, 100, 3.2, 24.5],
        [1, 200, 3.0, 20.1],
        [2, 300, 2.8, 16.4],
    ]
)

wandb.log({"training_metrics": table})
```

### 7. Alerts

Set up alerts for anomalies:

```python
# Alert if loss spikes
if loss > 10.0:
    wandb.alert(
        title="High Loss Detected",
        text=f"Loss spiked to {loss} at step {step}",
        level=wandb.AlertLevel.WARN
    )

# Alert if training completes
wandb.alert(
    title="Training Complete",
    text=f"Experiment {experiment_name} finished successfully",
    level=wandb.AlertLevel.INFO
)
```

---

## WandB Dashboard Features

### Viewing Your Runs

1. Go to [wandb.ai/home](https://wandb.ai/home)
2. Click on your project
3. See all runs with:
   - Real-time metrics graphs
   - System metrics
   - Configuration comparison
   - Notes and tags

### Comparing Experiments

1. Select multiple runs (checkbox)
2. Click "Compare"
3. View side-by-side:
   - Metric plots overlaid
   - Config differences
   - Performance comparison

### Sharing Results

1. Click "Share" button on a run
2. Options:
   - **Public link** - anyone with link can view
   - **Report** - create formatted report with visualizations
   - **Export** - download data as CSV/JSON

### Creating Reports

1. Click "Create Report"
2. Add:
   - Metric visualizations
   - Tables
   - Text descriptions
   - Code snippets
   - Images
3. Share with collaborators or make public

---

## Troubleshooting

### Issue: "wandb: ERROR Not logged in"

**Solution:**
```bash
wandb login
# Paste your API key when prompted
```

### Issue: "wandb: ERROR API key not found"

**Solution:**
```bash
# Check if API key is set
echo $WANDB_API_KEY

# If empty, set it
export WANDB_API_KEY="your-api-key"

# Or login interactively
wandb login --relogin
```

### Issue: Training hangs at wandb.init()

**Solution:**
```bash
# Disable wandb temporarily
export WANDB_MODE=disabled

# Or run in offline mode
export WANDB_MODE=offline
```

### Issue: "Rate limit exceeded"

**Solution:**
- Reduce logging frequency in config:
```yaml
log_metrics_every_n_steps: 100  # Increase from 10
log_memory_every_n_steps: 500   # Increase from 100
```

### Issue: WandB using too much disk space

**Solution:**
```bash
# Clean up old runs
wandb sync --clean

# Or manually delete
rm -rf wandb/offline-run-*
```

### Issue: Want to disable WandB without changing config

**Solution:**
```bash
export WANDB_MODE=disabled
python train.py
```

### Issue: Firewall blocking WandB

**Solution:**
```bash
# WandB uses these domains (whitelist in firewall):
# - api.wandb.ai (port 443)
# - *.wandb.ai (port 443)

# Or use offline mode
export WANDB_MODE=offline
```

### Issue: SSL Certificate errors

**Solution:**
```bash
export WANDB_VERIFY_SSL=false
```

---

## Integration Code

### WandBLogger Implementation

The Model Foundry includes a `WandBLogger` class in `logging_components.py`:

```python
class WandBLogger:
    """Integration with Weights & Biases."""

    def __init__(self, project: str, name: str, config: dict,
                 tags: Optional[List[str]] = None, enabled: bool = True):
        """
        Initialize WandB logger.

        Args:
            project: WandB project name
            name: Run name (experiment name)
            config: Configuration dictionary to log
            tags: Optional tags for organizing runs
            enabled: Whether WandB is enabled
        """
        self.enabled = enabled

        if enabled:
            import wandb

            wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags or [],
                # Auto-resume if run exists
                resume="allow",
                # Log git info
                settings=wandb.Settings(code_dir=".")
            )

            self.wandb = wandb

    def log_metrics(self, step: int, metrics: dict):
        """
        Log metrics to WandB.

        Args:
            step: Global training step
            metrics: Dictionary of metric names and values
        """
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def log_system_metrics(self):
        """Log system resource usage."""
        if self.enabled:
            import torch

            if torch.cuda.is_available():
                self.wandb.log({
                    "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                })

    def watch_model(self, model, log_freq: int = 100):
        """
        Watch model parameters and gradients.

        Args:
            model: PyTorch model
            log_freq: How often to log histograms
        """
        if self.enabled:
            self.wandb.watch(model, log="all", log_freq=log_freq)

    def log_artifact(self, artifact_path: str, artifact_type: str, name: str):
        """
        Log an artifact (model checkpoint, dataset, etc.).

        Args:
            artifact_path: Path to artifact directory
            artifact_type: Type (e.g., "model", "dataset")
            name: Artifact name
        """
        if self.enabled:
            artifact = self.wandb.Artifact(name=name, type=artifact_type)
            artifact.add_dir(artifact_path)
            self.wandb.log_artifact(artifact)

    def finish(self):
        """Finish the WandB run."""
        if self.enabled:
            self.wandb.finish()
```

### Integration in Trainer

Add WandB logging to the trainer:

```python
# In trainer.py

class Trainer:
    def __init__(self, config: ExperimentConfig, base_dir: str):
        # ... existing initialization ...

        # Initialize WandB if enabled
        if config.logging.use_wandb:
            from .logging_components import WandBLogger

            self.wandb_logger = WandBLogger(
                project=config.logging.wandb_project or "model-foundry",
                name=config.experiment_name,
                config=config.dict(),
                tags=self._get_experiment_tags(),
                enabled=True
            )

            # Watch model parameters
            if hasattr(self, 'model'):
                self.wandb_logger.watch_model(self.model)
        else:
            self.wandb_logger = None

    def _get_experiment_tags(self) -> List[str]:
        """Generate tags for WandB run."""
        tags = [self.config.experiment_name]

        # Add model info
        tags.append(f"layers-{self.config.model.layers}")
        tags.append(f"hidden-{self.config.model.hidden_size}")

        # Add training info
        if self.config.training.use_amp:
            tags.append("amp")

        return tags

    def _log_training_step(self, step: int, metrics: dict):
        """Log training step to all configured loggers."""
        # Log to local metrics logger
        self.metrics_logger.log_step(step, self.epoch, metrics)

        # Log to WandB if enabled
        if self.wandb_logger:
            self.wandb_logger.log_metrics(step, {
                f"train/{k}": v for k, v in metrics.items()
            })
```

---

## Best Practices

### 1. Naming Conventions

**Projects:**
- Use lowercase with hyphens: `spanish-subject-drop`
- Organize by research area: `syntax-experiments`

**Run Names:**
- Include key parameters: `exp0-baseline-lr0.001-bs32`
- Use timestamps for uniqueness: `exp0-baseline-20250930-143000`

**Tags:**
- Use for filtering: `["baseline", "gpt2", "ablation"]`
- Include architecture: `["transformer", "12-layers"]`
- Include dataset: `["spanish-corpus"]`

### 2. What to Log

**Essential:**
- Training loss
- Validation loss
- Learning rate
- Training step/epoch

**Recommended:**
- Gradient norm
- Throughput (tokens/sec)
- GPU memory usage
- Perplexity

**Optional:**
- Weight histograms
- Activation statistics
- Attention patterns
- Example predictions

### 3. Logging Frequency

**High frequency (every 10 steps):**
- Training loss
- Learning rate

**Medium frequency (every 100 steps):**
- Validation metrics
- Gradient statistics
- Throughput metrics

**Low frequency (every epoch or checkpoint):**
- Full evaluation metrics
- Model checkpoints
- Visualizations

### 4. Privacy & Security

**Don't log:**
- API keys
- Passwords
- Personal data
- Proprietary information

**Do log:**
- Hyperparameters
- Metrics
- System info
- Git commit hash

---

## Quick Start Checklist

- [ ] Create WandB account at [wandb.ai/signup](https://wandb.ai/signup)
- [ ] Get API key from [wandb.ai/authorize](https://wandb.ai/authorize)
- [ ] Run `wandb login` and paste API key
- [ ] Add `use_wandb: true` to your experiment YAML
- [ ] Set `wandb_project: "your-project-name"`
- [ ] Run training: `python -m model_foundry.cli train configs/your_config.yaml`
- [ ] View results at [wandb.ai/home](https://wandb.ai/home)

---

## Additional Resources

- **WandB Documentation:** [docs.wandb.ai](https://docs.wandb.ai)
- **Quickstart Guide:** [docs.wandb.ai/quickstart](https://docs.wandb.ai/quickstart)
- **Example Projects:** [wandb.ai/gallery](https://wandb.ai/gallery)
- **Community Forum:** [community.wandb.ai](https://community.wandb.ai)
- **Python API Reference:** [docs.wandb.ai/ref/python](https://docs.wandb.ai/ref/python)
- **Video Tutorials:** [youtube.com/@weights_biases](https://youtube.com/@weights_biases)

---

## Support

**Model Foundry Issues:**
- GitHub: [github.com/your-repo/model-foundry/issues](https://github.com)

**WandB Issues:**
- Support: [wandb.ai/support](https://wandb.ai/support)
- Email: support@wandb.ai
- Slack: [wandb.ai/slack](https://wandb.ai/slack)

**Emergency Disable:**
```bash
export WANDB_MODE=disabled
```
