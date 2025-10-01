# Model Foundry Documentation

**Complete documentation for the Model Foundry training framework and analysis tools.**

---

## 📚 Documentation Index

### 📋 Project Overview

- **[Project Charter](project-charter.md)** - High-level project goals, design principles, and workflow
- **[Preprocessing Plan](preprocessing-plan.md)** - Data preprocessing and environment setup guide

### 🚀 Quick Start

- **[Getting Started](model_foundry/guides/getting-started.md)** - Installation, setup, and first training run
- **[Configuration Guide](model_foundry/guides/configuration.md)** - Understanding and customizing experiment configs
- **[CLI Reference](model_foundry/guides/cli-reference.md)** - Command-line interface usage

### 🏗️ Architecture & Design

- **[Logging System](model_foundry/architecture/logging-system.md)** - Comprehensive logging architecture with structured logs, metrics tracking, and performance profiling
- **[Training Refactoring](model_foundry/architecture/training-refactoring.md)** - Modular training system design and implementation details
- **[Refactoring Status](model_foundry/architecture/refactoring-status.md)** - Complete refactoring summary with before/after comparison
- **[Code Organization](model_foundry/architecture/code-organization.md)** - Module structure and design patterns

### 🧪 Testing

- **[Testing Strategy](model_foundry/testing/strategy.md)** - Comprehensive testing plan for the entire system
- **[Running Tests](model_foundry/testing/running-tests.md)** - How to run unit, integration, and end-to-end tests
- **[Logging Tests](model_foundry/testing/logging-tests.md)** - Detailed specifications for logging component tests
- **[Writing Tests](model_foundry/testing/writing-tests.md)** - Guide for contributing new tests

### 📊 Experiment Tracking

- **[WandB Integration](model_foundry/guides/wandb-integration.md)** - Complete Weights & Biases setup and usage guide
- **[Metrics & Logging](model_foundry/guides/metrics-logging.md)** - Understanding and customizing metrics logging
- **[Comparing Experiments](model_foundry/guides/experiment-comparison.md)** - Analyzing and comparing multiple training runs

### 🔧 API Reference

- **[Configuration API](model_foundry/api/configuration.md)** - ExperimentConfig, DataConfig, ModelConfig, etc.
- **[Logging Components](model_foundry/api/logging-components.md)** - StructuredLogger, MetricsLogger, PerformanceLogger, ErrorTracker, WandBLogger
- **[Training Components](model_foundry/api/training-components.md)** - Trainer, TrainingLoop, CheckpointManager
- **[Data Processing](model_foundry/api/data-processing.md)** - DataProcessor, chunking, validation

### 🎓 Tutorials

- **[Basic Training](model_foundry/tutorials/basic-training.md)** - Run your first experiment
- **[Custom Datasets](model_foundry/tutorials/custom-datasets.md)** - Preparing and using custom datasets
- **[Hyperparameter Tuning](model_foundry/tutorials/hyperparameter-tuning.md)** - Optimizing model performance
- **[Ablation Studies](model_foundry/tutorials/ablation-studies.md)** - Systematic feature removal experiments

---

## 📁 Documentation Structure

```
docs/
├── README.md                                    # This file - master index
│
├── model_foundry/                              # Model Foundry framework docs
│   ├── guides/                                 # User guides and how-tos
│   │   ├── getting-started.md                 # Quick start guide
│   │   ├── configuration.md                   # Config file reference
│   │   ├── cli-reference.md                   # CLI commands
│   │   ├── wandb-integration.md              # WandB setup (500+ lines)
│   │   ├── metrics-logging.md                # Metrics and logging
│   │   └── experiment-comparison.md          # Comparing runs
│   │
│   ├── architecture/                          # System design docs
│   │   ├── logging-system.md                 # Logging architecture (23k words)
│   │   ├── training-refactoring.md           # Training module design
│   │   ├── refactoring-status.md             # Refactoring summary
│   │   └── code-organization.md              # Module structure
│   │
│   ├── testing/                               # Testing documentation
│   │   ├── strategy.md                       # Testing strategy (500+ lines)
│   │   ├── running-tests.md                  # How to run tests
│   │   ├── logging-tests.md                  # Logging test specs (15k words)
│   │   └── writing-tests.md                  # Contributing tests
│   │
│   ├── api/                                   # API reference
│   │   ├── configuration.md                  # Config classes
│   │   ├── logging-components.md             # Logging API
│   │   ├── training-components.md            # Training API
│   │   └── data-processing.md                # Data API
│   │
│   └── tutorials/                             # Step-by-step tutorials
│       ├── basic-training.md
│       ├── custom-datasets.md
│       ├── hyperparameter-tuning.md
│       └── ablation-studies.md
│
└── analysis/                                   # Analysis tools docs
    ├── statistical-analysis.md
    └── visualization.md
```

---

## 🎯 Common Tasks

### Running Your First Experiment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Login to WandB (optional)
wandb login

# 3. Run training
python -m model_foundry.cli train configs/example_with_wandb.yaml
```

See: [Getting Started Guide](model_foundry/guides/getting-started.md)

### Viewing Logs and Metrics

**Local Logs:**
```bash
# View latest log
tail -f logs/your-experiment/main_*.log

# View metrics
cat logs/your-experiment/metrics.jsonl | jq '.'
```

**WandB Dashboard:**
1. Go to [wandb.ai/home](https://wandb.ai/home)
2. Click on your project
3. View real-time metrics and comparisons

See: [WandB Integration Guide](model_foundry/guides/wandb-integration.md)

### Running Tests

```bash
# Run all tests
pytest model_foundry/tests/ -v

# Run specific test suite
pytest model_foundry/tests/unit/test_structured_logger.py -v

# Run with markers
pytest model_foundry/tests/ -v -m "not slow"
```

See: [Running Tests](model_foundry/testing/running-tests.md)

### Creating a New Experiment

```bash
# Copy example config
cp configs/example_with_wandb.yaml configs/my_experiment.yaml

# Edit configuration
vim configs/my_experiment.yaml

# Run experiment
python -m model_foundry.cli train configs/my_experiment.yaml
```

See: [Configuration Guide](model_foundry/guides/configuration.md)

---

## 📊 Quick Reference

### Configuration File Structure

```yaml
experiment_name: "my_experiment"

data:
  source_corpus: "data/corpus"
  batch_size: 32
  max_sequence_length: 512

tokenizer:
  output_dir: "tokenizers/my_tokenizer"
  vocab_size: 16000

model:
  layers: 12
  embedding_size: 768
  hidden_size: 768
  # ... more config

training:
  output_dir: "output/my_experiment"
  learning_rate: 0.0001
  epochs: 3
  # ... more config

logging:
  use_wandb: true
  wandb_project: "my-project"
  log_metrics_every_n_steps: 10

random_seed: 42
```

### Key Modules

| Module | Purpose | Documentation |
|--------|---------|---------------|
| `model_foundry.trainer` | Main training orchestration | [API](model_foundry/api/training-components.md) |
| `model_foundry.training.loop` | Training loop execution | [Architecture](model_foundry/architecture/training-refactoring.md) |
| `model_foundry.training.checkpointing` | Checkpoint management | [API](model_foundry/api/training-components.md) |
| `model_foundry.logging_components` | Logging infrastructure | [Architecture](model_foundry/architecture/logging-system.md) |
| `model_foundry.data` | Data processing | [API](model_foundry/api/data-processing.md) |
| `model_foundry.model` | Model creation | [API](model_foundry/api/training-components.md) |
| `model_foundry.config` | Configuration validation | [API](model_foundry/api/configuration.md) |

### Logging Components

| Component | Purpose | Documentation |
|-----------|---------|---------------|
| `StructuredLogger` | JSON-formatted structured logging | [Logging System](model_foundry/architecture/logging-system.md#structuredlogger) |
| `MetricsLogger` | Training metrics tracking (JSONL) | [Logging System](model_foundry/architecture/logging-system.md#metricslogger) |
| `PerformanceLogger` | Timing and profiling | [Logging System](model_foundry/architecture/logging-system.md#performancelogger) |
| `ErrorTracker` | Error aggregation | [Logging System](model_foundry/architecture/logging-system.md#errortracker) |
| `WandBLogger` | Weights & Biases integration | [WandB Guide](model_foundry/guides/wandb-integration.md) |

---

## 🧪 Testing Coverage

**Current Status:**
- **174 tests** passing (122 core + 52 logging)
- **8 skipped** (integration tests)
- **~85% coverage** on core modules

See: [Testing Strategy](model_foundry/testing/strategy.md)

---

## 🔗 External Resources

### Model Foundry
- **GitHub**: [github.com/your-repo/model-foundry](https://github.com)
- **Issues**: [github.com/your-repo/model-foundry/issues](https://github.com)

### Weights & Biases
- **Documentation**: [docs.wandb.ai](https://docs.wandb.ai)
- **Quickstart**: [docs.wandb.ai/quickstart](https://docs.wandb.ai/quickstart)
- **Gallery**: [wandb.ai/gallery](https://wandb.ai/gallery)

### PyTorch & Transformers
- **PyTorch Docs**: [pytorch.org/docs](https://pytorch.org/docs)
- **HuggingFace**: [huggingface.co/docs](https://huggingface.co/docs)
- **GPT-2**: [huggingface.co/docs/transformers/model_doc/gpt2](https://huggingface.co/docs/transformers/model_doc/gpt2)

---

## 📝 Documentation Status

| Document | Status | Last Updated | Lines |
|----------|--------|--------------|-------|
| Logging System | ✅ Complete | 2025-09-30 | 1,000+ |
| WandB Integration | ✅ Complete | 2025-09-30 | 500+ |
| Testing Strategy | ✅ Complete | 2025-09-30 | 500+ |
| Logging Tests Spec | ✅ Complete | 2025-09-30 | 600+ |
| Training Refactoring | ✅ Complete | 2025-09-30 | 400+ |
| Refactoring Status | ✅ Complete | 2025-09-30 | 600+ |
| Running Tests | ✅ Complete | 2025-09-30 | 300+ |
| Getting Started | 🚧 Planned | - | - |
| Configuration Guide | 🚧 Planned | - | - |
| CLI Reference | 🚧 Planned | - | - |
| API Reference | 🚧 Planned | - | - |
| Tutorials | 🚧 Planned | - | - |

---

## 🤝 Contributing

When adding new documentation:

1. **Choose the right location:**
   - User-facing guides → `guides/`
   - Architecture/design docs → `architecture/`
   - Testing docs → `testing/`
   - API reference → `api/`
   - Step-by-step tutorials → `tutorials/`

2. **Follow naming conventions:**
   - Use kebab-case: `my-document.md`
   - Be descriptive: `wandb-integration.md` not `wandb.md`

3. **Update this README:**
   - Add your document to the index
   - Update the status table
   - Add relevant quick reference entries

4. **Link related docs:**
   - Cross-reference related documentation
   - Use relative links: `[link](../guides/guide.md)`

---

## 📧 Support

- **Documentation Issues**: Open an issue with the `documentation` label
- **Questions**: Check existing docs first, then open a discussion
- **Contributions**: See `CONTRIBUTING.md`

---

**Last Updated**: 2025-09-30
**Documentation Version**: 1.0.0
**Model Foundry Version**: 0.1.0
