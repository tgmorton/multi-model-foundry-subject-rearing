# Model Foundry Documentation

**Complete documentation for the Model Foundry training framework.**

---

## 📚 Documentation Structure

This directory contains all Model Foundry documentation, organized by category:

```
model_foundry/
├── guides/                 # User guides and how-tos
├── architecture/           # System design and architecture
├── testing/               # Testing documentation
├── api/                   # API reference (planned)
└── tutorials/             # Step-by-step tutorials (planned)
```

---

## 🚀 Quick Start

**New to Model Foundry?** Start here:

1. **[Main Documentation Index](/docs/README.md)** - Overview of all documentation
2. **[Getting Started](/docs/model_foundry/guides/getting-started.md)** (planned) - Installation and first run
3. **[Example Configuration](/configs/example_with_wandb.yaml)** - Ready-to-use config file

---

## 📖 User Guides

Step-by-step guides for common tasks:

- **[WandB Integration](guides/wandb-integration.md)** ✅ - Complete Weights & Biases setup (500+ lines)
  - Account creation
  - API key configuration
  - Usage examples
  - Troubleshooting

- **[Getting Started](guides/getting-started.md)** 🚧 - Installation and first training run
- **[Configuration Guide](guides/configuration.md)** 🚧 - Understanding config files
- **[CLI Reference](guides/cli-reference.md)** 🚧 - Command-line interface
- **[Metrics & Logging](guides/metrics-logging.md)** 🚧 - Customizing metrics

**Legend:** ✅ Available | 🚧 Planned

---

## 🏗️ Architecture

Deep dives into system design:

- **[Logging System](architecture/logging-system.md)** ✅ - Complete logging architecture (1,000+ lines)
  - StructuredLogger (JSON logs with context)
  - MetricsLogger (JSONL metrics tracking)
  - PerformanceLogger (timing & profiling)
  - ErrorTracker (error aggregation)
  - WandBLogger (experiment tracking)

- **[Training Refactoring](architecture/training-refactoring.md)** ✅ - Modular training design (400+ lines)
  - Training loop extraction
  - Checkpoint management
  - Tokenization utilities

- **[Refactoring Status](architecture/refactoring-status.md)** ✅ - Complete refactoring summary (600+ lines)
  - Before/after comparison
  - Test coverage details
  - Implementation timeline

- **[Code Organization](architecture/code-organization.md)** 🚧 - Module structure and patterns

---

## 🧪 Testing

Everything you need to know about testing:

- **[Testing Strategy](testing/strategy.md)** ✅ - Comprehensive testing plan (500+ lines)
  - Unit tests
  - Integration tests
  - End-to-end tests
  - Coverage goals

- **[Running Tests](testing/running-tests.md)** ✅ - How to run tests (300+ lines)
  - Basic usage
  - Test markers
  - Fixtures
  - Common commands

- **[Logging Tests Specification](testing/logging-tests.md)** ✅ - Detailed test specs (600+ lines)
  - 50 unit tests
  - 15 integration tests
  - Given/When/Then format

- **[Writing Tests](testing/writing-tests.md)** 🚧 - Contributing new tests

---

## 📋 API Reference

**🚧 Planned** - Detailed API documentation for all modules:

- **[Configuration API](api/configuration.md)** - ExperimentConfig, DataConfig, ModelConfig, etc.
- **[Logging Components API](api/logging-components.md)** - StructuredLogger, MetricsLogger, etc.
- **[Training Components API](api/training-components.md)** - Trainer, TrainingLoop, CheckpointManager
- **[Data Processing API](api/data-processing.md)** - DataProcessor, validation, chunking

---

## 🎓 Tutorials

**🚧 Planned** - Step-by-step tutorials:

- **[Basic Training](tutorials/basic-training.md)** - Run your first experiment
- **[Custom Datasets](tutorials/custom-datasets.md)** - Using your own data
- **[Hyperparameter Tuning](tutorials/hyperparameter-tuning.md)** - Optimizing performance
- **[Ablation Studies](tutorials/ablation-studies.md)** - Systematic feature removal

---

## 📊 Documentation Status

### Current (7 documents, 4,300+ lines)

| Document | Category | Lines | Status |
|----------|----------|-------|--------|
| WandB Integration | Guide | 500+ | ✅ Complete |
| Logging System | Architecture | 1,000+ | ✅ Complete |
| Training Refactoring | Architecture | 400+ | ✅ Complete |
| Refactoring Status | Architecture | 600+ | ✅ Complete |
| Testing Strategy | Testing | 500+ | ✅ Complete |
| Running Tests | Testing | 300+ | ✅ Complete |
| Logging Tests | Testing | 600+ | ✅ Complete |

### Planned (8+ documents)

- Getting Started guide
- Configuration guide
- CLI reference
- Metrics logging guide
- Code organization
- Writing tests guide
- Complete API reference (4 docs)
- Tutorials (4+ docs)

---

## 🔍 Find What You Need

### By Task

| I want to... | Read this... |
|--------------|--------------|
| **Set up WandB** | [WandB Integration](guides/wandb-integration.md) |
| **Understand logging** | [Logging System](architecture/logging-system.md) |
| **Run tests** | [Running Tests](testing/running-tests.md) |
| **Understand training** | [Training Refactoring](architecture/training-refactoring.md) |
| **See refactoring results** | [Refactoring Status](architecture/refactoring-status.md) |
| **Write tests** | [Logging Tests](testing/logging-tests.md) |
| **Plan testing** | [Testing Strategy](testing/strategy.md) |

### By User Type

**🆕 New User**
1. [Main Docs Index](/docs/README.md)
2. [Getting Started](guides/getting-started.md) (planned)
3. [Example Config](/configs/example_with_wandb.yaml)

**👨‍💻 Developer**
1. [Training Refactoring](architecture/training-refactoring.md)
2. [Logging System](architecture/logging-system.md)
3. [Code Organization](architecture/code-organization.md) (planned)

**🧪 Contributor**
1. [Testing Strategy](testing/strategy.md)
2. [Running Tests](testing/running-tests.md)
3. [Writing Tests](testing/writing-tests.md) (planned)

**📊 Experimenter**
1. [WandB Integration](guides/wandb-integration.md)
2. [Configuration Guide](guides/configuration.md) (planned)
3. [Tutorials](tutorials/) (planned)

---

## 🔗 Related Documentation

- **[Main Documentation Index](/docs/README.md)** - All project documentation
- **[Documentation Map](/DOCUMENTATION_MAP.md)** - Quick reference guide
- **[Documentation Structure](/docs/STRUCTURE.md)** - Visual structure guide
- **[Model Foundry README](/model_foundry/README.md)** - Package overview

---

## 📈 Progress Tracking

```
Overall Progress: ▓▓▓▓▓▓▓▓▓░ 47% (7/15 planned documents)

By Category:
  Guides:       ▓▓░░░ 20% (1/5)
  Architecture: ▓▓▓▓░ 75% (3/4)
  Testing:      ▓▓▓▓░ 75% (3/4)
  API:          ░░░░░  0% (0/4)
  Tutorials:    ░░░░░  0% (0/4)
```

---

## 🤝 Contributing Documentation

### Adding New Documentation

1. **Choose the right category:**
   - User guides → `guides/`
   - Architecture docs → `architecture/`
   - Testing docs → `testing/`
   - API reference → `api/`
   - Tutorials → `tutorials/`

2. **Follow naming conventions:**
   - Use kebab-case: `my-document.md`
   - Be descriptive: `wandb-integration.md`

3. **Update indexes:**
   - This file (`README.md`)
   - Main index (`/docs/README.md`)
   - Documentation map (`/DOCUMENTATION_MAP.md`)

### Document Templates

See [/docs/STRUCTURE.md](/docs/STRUCTURE.md) for templates.

---

## 📧 Questions?

- **Can't find what you need?** Check [/docs/README.md](/docs/README.md) or [DOCUMENTATION_MAP.md](/DOCUMENTATION_MAP.md)
- **Documentation issue?** Open an issue with the `documentation` label
- **Want to contribute?** See the Contributing section above

---

**Last Updated**: 2025-09-30
**Documentation Version**: 1.0.0
