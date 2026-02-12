# Documentation Map

**Quick reference for finding documentation across the Model Foundry project.**

---

## 📁 All Documentation is Now in `/docs/`

All documentation has been centralized in the `/docs/` directory for easy access.

**Start here:** [`/docs/README.md`](/docs/README.md)

---

## 🗺️ Documentation Structure

```
docs/
├── README.md                                    # 📚 Master documentation index
│
├── model_foundry/                              # Model Foundry framework
│   │
│   ├── guides/                                 # 📖 User guides
│   │   └── wandb-integration.md               # WandB setup (500+ lines)
│   │
│   ├── architecture/                          # 🏗️ System design
│   │   ├── logging-system.md                 # Logging architecture (1000+ lines)
│   │   ├── training-refactoring.md           # Training module design
│   │   └── refactoring-status.md             # Refactoring summary
│   │
│   ├── testing/                               # 🧪 Testing docs
│   │   ├── strategy.md                       # Testing strategy
│   │   ├── running-tests.md                  # How to run tests
│   │   └── logging-tests.md                  # Logging test specs (600+ lines)
│   │
│   ├── api/                                   # 📋 API reference (planned)
│   │   ├── configuration.md
│   │   ├── logging-components.md
│   │   ├── training-components.md
│   │   └── data-processing.md
│   │
│   └── tutorials/                             # 🎓 Tutorials (planned)
│       ├── basic-training.md
│       ├── custom-datasets.md
│       └── hyperparameter-tuning.md
│
└── analysis/                                   # Analysis tools (planned)
    ├── statistical-analysis.md
    └── visualization.md
```

---

## 🎯 Find Documentation By Topic

### Getting Started
- **Installation & Setup** → `/docs/README.md` (Quick Start section)
- **First Training Run** → `/docs/model_foundry/guides/getting-started.md` (planned)
- **Example Configs** → `/configs/templates/example_with_wandb.yaml`

### Training
- **Training Architecture** → `/docs/model_foundry/architecture/training-refactoring.md`
- **Checkpoint Management** → `/docs/model_foundry/architecture/training-refactoring.md#checkpoint-management`
- **Configuration Options** → `/docs/model_foundry/guides/configuration.md` (planned)

### Logging & Monitoring
- **Logging System Overview** → `/docs/model_foundry/architecture/logging-system.md`
- **WandB Integration** → `/docs/model_foundry/guides/wandb-integration.md`
- **Metrics Tracking** → `/docs/model_foundry/architecture/logging-system.md#metricslogger`
- **Performance Profiling** → `/docs/model_foundry/architecture/logging-system.md#performancelogger`

### Testing
- **Testing Strategy** → `/docs/model_foundry/testing/strategy.md`
- **Running Tests** → `/docs/model_foundry/testing/running-tests.md`
- **Writing Tests** → `/docs/model_foundry/testing/writing-tests.md` (planned)
- **Logging Tests** → `/docs/model_foundry/testing/logging-tests.md`

### API Reference
- **Configuration API** → `/docs/model_foundry/api/configuration.md` (planned)
- **Logging Components** → `/docs/model_foundry/api/logging-components.md` (planned)
- **Training Components** → `/docs/model_foundry/api/training-components.md` (planned)
- **Data Processing** → `/docs/model_foundry/api/data-processing.md` (planned)

---

## 📍 Documentation Migration Complete

**✅ All documentation has been moved to `/docs/` and originals deleted.**

### Migration Summary

| Old Location | New Location | Status |
|-------------|--------------|--------|
| `model_foundry/LOGGING_PLAN.md` | `/docs/model_foundry/architecture/logging-system.md` | ✅ Moved & Deleted |
| `model_foundry/WANDB_INTEGRATION_GUIDE.md` | `/docs/model_foundry/guides/wandb-integration.md` | ✅ Moved & Deleted |
| `model_foundry/TESTING_STRATEGY.md` | `/docs/model_foundry/testing/strategy.md` | ✅ Moved & Deleted |
| `model_foundry/IMPLEMENTATION_SUMMARY.md` | `/docs/model_foundry/architecture/training-refactoring.md` | ✅ Moved & Deleted |
| `model_foundry/FINAL_STATUS.md` | `/docs/model_foundry/architecture/refactoring-status.md` | ✅ Moved & Deleted |
| `model_foundry/tests/README.md` | `/docs/model_foundry/testing/running-tests.md` | ✅ Moved & Deleted |
| `model_foundry/tests/LOGGING_TESTS_SPEC.md` | `/docs/model_foundry/testing/logging-tests.md` | ✅ Moved & Deleted |
| `gemini.md` | `/docs/project-charter.md` | ✅ Moved & Deleted |
| `preprocessing/project.md` | `/docs/preprocessing-plan.md` | ✅ Moved & Deleted |

**Note:** Only `model_foundry/README.md` remains as it points to the centralized documentation.

---

## 🔍 Find Documentation By File Type

### Architecture & Design
- Logging System: `/docs/model_foundry/architecture/logging-system.md`
- Training Refactoring: `/docs/model_foundry/architecture/training-refactoring.md`
- Refactoring Status: `/docs/model_foundry/architecture/refactoring-status.md`

### Guides & Tutorials
- WandB Integration: `/docs/model_foundry/guides/wandb-integration.md`

### Testing
- Testing Strategy: `/docs/model_foundry/testing/strategy.md`
- Running Tests: `/docs/model_foundry/testing/running-tests.md`
- Logging Tests Spec: `/docs/model_foundry/testing/logging-tests.md`

### API Reference
- (Planned) Configuration: `/docs/model_foundry/api/configuration.md`
- (Planned) Logging Components: `/docs/model_foundry/api/logging-components.md`
- (Planned) Training Components: `/docs/model_foundry/api/training-components.md`

---

## 📊 Documentation Statistics

### Current Status

| Category | Files | Lines | Status |
|----------|-------|-------|--------|
| Guides | 1 | 500+ | ✅ Active |
| Architecture | 3 | 2,400+ | ✅ Active |
| Testing | 3 | 1,400+ | ✅ Active |
| API Reference | 0 | - | 🚧 Planned |
| Tutorials | 0 | - | 🚧 Planned |
| **Total** | **7** | **4,300+** | **✅** |

### Coverage

- ✅ **Logging System** - Complete (1,000+ lines)
- ✅ **WandB Integration** - Complete (500+ lines)
- ✅ **Testing** - Complete (1,400+ lines)
- ✅ **Training Architecture** - Complete (900+ lines)
- 🚧 **Getting Started** - Planned
- 🚧 **API Reference** - Planned
- 🚧 **Tutorials** - Planned

---

## 🚀 Quick Links

### Most Common Docs

1. **[Main Documentation Index](/docs/README.md)** - Start here
2. **[WandB Integration Guide](/docs/model_foundry/guides/wandb-integration.md)** - Setup experiment tracking
3. **[Logging System](/docs/model_foundry/architecture/logging-system.md)** - Understanding logging
4. **[Running Tests](/docs/model_foundry/testing/running-tests.md)** - Test your code
5. **[Training Architecture](/docs/model_foundry/architecture/training-refactoring.md)** - How training works

### By User Type

**I'm a new user:**
1. Start with `/docs/README.md`
2. Read "Quick Start" section
3. Try example config in `/configs/templates/example_with_wandb.yaml`

**I want to understand the architecture:**
1. Read `/docs/model_foundry/architecture/training-refactoring.md`
2. Read `/docs/model_foundry/architecture/logging-system.md`
3. Check `/docs/model_foundry/architecture/refactoring-status.md`

**I want to contribute:**
1. Read `/docs/model_foundry/testing/strategy.md`
2. Read `/docs/model_foundry/testing/running-tests.md`
3. Check existing tests in `/model_foundry/tests/`

**I want to use WandB:**
1. Read `/docs/model_foundry/guides/wandb-integration.md`
2. Copy `/configs/templates/example_with_wandb.yaml`
3. Follow setup instructions

---

## 📝 Adding New Documentation

### Where to Put New Docs

| Type of Documentation | Location | Example |
|----------------------|----------|---------|
| User guide / How-to | `/docs/model_foundry/guides/` | `wandb-integration.md` |
| Architecture / Design | `/docs/model_foundry/architecture/` | `logging-system.md` |
| Testing documentation | `/docs/model_foundry/testing/` | `strategy.md` |
| API reference | `/docs/model_foundry/api/` | `configuration.md` |
| Step-by-step tutorial | `/docs/model_foundry/tutorials/` | `basic-training.md` |

### Naming Conventions

- Use lowercase with hyphens: `my-document.md`
- Be descriptive: `wandb-integration.md` not `wandb.md`
- Group related docs: `logging-system.md`, `logging-tests.md`

### After Adding Documentation

1. Update `/docs/README.md` index
2. Update this file (`DOCUMENTATION_MAP.md`)
3. Add cross-references to related docs
4. Update status table

---

## 🔄 Migration Status

### Completed ✅
- Created centralized `/docs/` directory
- Moved 7 documentation files to new structure
- Created master index (`/docs/README.md`)
- Created model_foundry README (`/model_foundry/README.md`)
- Created this documentation map

### Remaining 🚧
- Remove old documentation files (kept for backwards compatibility)
- Create getting started guide
- Create API reference docs
- Create tutorials
- Update all documentation links in code comments

---

## 📧 Questions?

If you can't find the documentation you need:

1. Check `/docs/README.md` - Master index
2. Check this file (`DOCUMENTATION_MAP.md`) - Quick reference
3. Search the `/docs/` directory
4. Open an issue requesting the documentation

---

**Last Updated**: 2025-09-30
**Documentation Structure Version**: 1.0.0
