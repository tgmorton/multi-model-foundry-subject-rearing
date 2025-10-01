# Model Foundry

**A modular, production-ready training framework for language models with comprehensive logging, experiment tracking, and testing infrastructure.**

---

## 📚 Documentation

**Complete documentation is available in [/docs](/docs/README.md)**

### Quick Links

- **[Getting Started](/docs/model_foundry/guides/getting-started.md)** - Installation and first run
- **[WandB Integration](/docs/model_foundry/guides/wandb-integration.md)** - Experiment tracking setup
- **[Logging System](/docs/model_foundry/architecture/logging-system.md)** - Comprehensive logging architecture
- **[Testing](/docs/model_foundry/testing/running-tests.md)** - Running and writing tests
- **[API Reference](/docs/model_foundry/api/)** - Complete API documentation

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install WandB for experiment tracking
pip install wandb
wandb login
```

### Run Your First Experiment

```bash
# Run with example configuration
python -m model_foundry.cli train configs/example_with_wandb.yaml
```

### View Results

**Local Logs:**
```bash
tail -f logs/your-experiment/main_*.log
```

**WandB Dashboard:**
Visit [wandb.ai/home](https://wandb.ai/home) to see real-time metrics

---

## 🏗️ Architecture

### Module Structure

```
model_foundry/
├── cli.py                      # Command-line interface
├── trainer.py                  # Main training orchestrator
├── config.py                   # Configuration with Pydantic validation
├── model.py                    # Model creation utilities
├── data.py                     # Data processing and validation
├── utils.py                    # Utility functions
├── logging_utils.py            # Basic logging setup (legacy)
├── logging_components.py       # Advanced logging components
│
├── training/                   # Training subsystem (refactored)
│   ├── __init__.py
│   ├── loop.py                # Main training loop
│   ├── checkpointing.py       # Checkpoint management
│   └── tokenization.py        # Tokenizer utilities
│
├── tokenizer/                  # Tokenizer training
│   ├── train_tokenizer.py
│   └── tokenize_dataset.py
│
└── tests/                      # Comprehensive test suite
    ├── conftest.py            # Shared fixtures
    ├── unit/                  # Unit tests
    └── integration/           # Integration tests
```

See: [Code Organization](/docs/model_foundry/architecture/code-organization.md)

### Key Components

**Training Pipeline:**
- `Trainer` - Orchestrates entire training process
- `TrainingLoop` - Executes training steps with AMP support
- `CheckpointManager` - Handles checkpoint saving/loading
- `DataProcessor` - Prepares and validates datasets

**Logging Infrastructure:**
- `StructuredLogger` - JSON-formatted structured logging
- `MetricsLogger` - Training metrics in JSONL format
- `PerformanceLogger` - Timing and profiling
- `ErrorTracker` - Error aggregation and tracking
- `WandBLogger` - Weights & Biases integration

See: [Logging System](/docs/model_foundry/architecture/logging-system.md)

---

## ✨ Features

### Training
- ✅ **GPT-2 Architecture** - Full GPT-2 model support
- ✅ **Mixed Precision (AMP)** - Faster training with automatic mixed precision
- ✅ **Gradient Accumulation** - Train larger models on limited memory
- ✅ **Flexible Checkpointing** - Logarithmic and linear checkpoint schedules
- ✅ **Resume from Checkpoint** - Automatic resume with full state restoration
- ✅ **Custom Datasets** - Easy integration of custom training data

### Logging & Monitoring
- ✅ **Structured Logging** - JSON-formatted logs with context
- ✅ **Metrics Tracking** - JSONL metrics for easy analysis
- ✅ **Performance Profiling** - Timing and memory tracking
- ✅ **Error Tracking** - Centralized error aggregation
- ✅ **WandB Integration** - Real-time experiment tracking
- ✅ **System Monitoring** - GPU memory, throughput, etc.

### Testing
- ✅ **174 Tests** - Comprehensive test coverage
- ✅ **~85% Coverage** - High code coverage on core modules
- ✅ **Fast Execution** - Tests run in <10 seconds
- ✅ **Isolated Tests** - Independent, reproducible tests

---

## 📊 Example Usage

### Basic Training

```python
from model_foundry.trainer import Trainer
from model_foundry.config import ExperimentConfig
import yaml

# Load configuration
with open('configs/my_experiment.yaml') as f:
    config_dict = yaml.safe_load(f)

config = ExperimentConfig(**config_dict)

# Create trainer
trainer = Trainer(config, base_dir=".")

# Start training
trainer.train()
```

### With Advanced Logging

```python
from model_foundry.logging_components import (
    StructuredLogger,
    MetricsLogger,
    PerformanceLogger,
    WandBLogger
)

# Initialize loggers
structured_logger = StructuredLogger("my_experiment", config)
metrics_logger = MetricsLogger("my_experiment", "output/")
perf_logger = PerformanceLogger(structured_logger.logger)
wandb_logger = WandBLogger(
    project="my-project",
    name="my_experiment",
    config=config.dict()
)

# Log training step
metrics_logger.log_step(
    step=100,
    epoch=2,
    metrics={"loss": 2.5, "lr": 0.001}
)

# Profile performance
with perf_logger.time_block("forward_pass"):
    outputs = model(**inputs)

# Log to WandB
wandb_logger.log_metrics(100, {"train/loss": 2.5})
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest model_foundry/tests/ -v
```

### Run Specific Test Suite

```bash
# Unit tests only
pytest model_foundry/tests/unit/ -v

# Logging tests
pytest model_foundry/tests/unit/test_*logger*.py -v

# Exclude slow tests
pytest model_foundry/tests/ -v -m "not slow"
```

### Test Coverage

```bash
pytest model_foundry/tests/ --cov=model_foundry --cov-report=html
open htmlcov/index.html
```

See: [Testing Documentation](/docs/model_foundry/testing/)

---

## 📖 Configuration

### Example Configuration File

```yaml
experiment_name: "my_experiment"

data:
  source_corpus: "data/corpus"
  training_corpus: "data/corpus/tokenized"
  batch_size: 32
  max_sequence_length: 512

tokenizer:
  output_dir: "tokenizers/my_tokenizer"
  vocab_size: 16000

model:
  layers: 12
  embedding_size: 768
  hidden_size: 768
  intermediate_hidden_size: 3072
  attention_heads: 12
  activation_function: "gelu"
  dropout: 0.1
  attention_dropout: 0.1

training:
  output_dir: "output/my_experiment"
  learning_rate: 0.0001
  epochs: 3
  use_amp: true
  gradient_accumulation_steps: 4

logging:
  use_wandb: true
  wandb_project: "my-project"
  log_metrics_every_n_steps: 10
  profile_performance: true

random_seed: 42
```

See: [Configuration Guide](/docs/model_foundry/guides/configuration.md)

---

## 🎯 Project Status

### Recent Achievements

**✅ Refactoring Complete (2025-09-30)**
- Modular training system (60% code reduction in trainer.py)
- 174 tests with 85%+ coverage
- Comprehensive documentation (5,000+ lines)

**✅ Logging System Complete (2025-09-30)**
- Structured logging with JSON format
- Metrics tracking in JSONL
- Performance profiling
- Error tracking
- WandB integration

**✅ Testing Infrastructure Complete (2025-09-30)**
- 122 core framework tests
- 52 logging component tests
- Full test documentation

See: [Refactoring Status](/docs/model_foundry/architecture/refactoring-status.md)

### Test Coverage Summary

| Module | Tests | Coverage | Status |
|--------|-------|----------|--------|
| Config | 30 | 95%+ | ✅ |
| Utils | 23 | 95%+ | ✅ |
| Model | 33 | 90%+ | ✅ |
| Data | 23 | 85%+ | ✅ |
| Checkpointing | 13 | 90%+ | ✅ |
| Logging Components | 52 | 95%+ | ✅ |
| **Total** | **174** | **~85%** | ✅ |

---

## 🔗 Links

### Documentation
- **[Main Documentation](/docs/README.md)** - Complete documentation index
- **[Architecture Docs](/docs/model_foundry/architecture/)** - System design
- **[API Reference](/docs/model_foundry/api/)** - Complete API docs
- **[Guides](/docs/model_foundry/guides/)** - User guides and tutorials

### Key Documents
- **[Logging System](/docs/model_foundry/architecture/logging-system.md)** - 23,000 word architecture doc
- **[WandB Integration](/docs/model_foundry/guides/wandb-integration.md)** - Complete setup guide
- **[Testing Strategy](/docs/model_foundry/testing/strategy.md)** - Comprehensive testing plan

---

## 🤝 Contributing

1. **Documentation**: See [/docs](/docs/README.md)
2. **Tests**: All new code requires tests. See [Testing Guide](/docs/model_foundry/testing/writing-tests.md)
3. **Code Style**: Follow existing patterns, use type hints
4. **Pull Requests**: Include tests and documentation updates

---

## 📧 Support

- **Issues**: GitHub Issues
- **Documentation**: [/docs](/docs/README.md)
- **Questions**: GitHub Discussions

---

**Version**: 0.1.0
**Last Updated**: 2025-09-30
**License**: MIT (or your license)
