# Documentation Structure

**Visual guide to the centralized documentation system.**

---

## 📂 Directory Tree

```
subject-drop/
│
├── 📄 DOCUMENTATION_MAP.md              # Quick reference guide (this maps everything)
│
├── 📁 docs/                             # 🎯 ALL DOCUMENTATION HERE
│   │
│   ├── 📄 README.md                     # Master documentation index
│   │
│   ├── 📁 preprocessing/                # Preprocessing module docs
│   │   ├── README.md                    # Overview & quick start
│   │   ├── USER_GUIDE.md                # Complete usage guide
│   │   ├── DEVELOPER_GUIDE.md           # Adding custom ablations
│   │   ├── ADVANCED.md                  # Performance, coreference & production
│   │   └── TESTING.md                   # Test guide
│   │
│   └── 📁 model_foundry/                # Model Foundry framework docs
│       │
│       ├── 📁 guides/                   # User guides & how-tos
│       │   ├── wandb-integration.md     # ✅ WandB setup (500+ lines)
│       │   ├── getting-started.md       # 🚧 Installation & first run
│       │   ├── configuration.md         # 🚧 Config file reference
│       │   ├── cli-reference.md         # 🚧 CLI commands
│       │   └── metrics-logging.md       # 🚧 Metrics & logging
│       │
│       ├── 📁 architecture/             # System design & architecture
│       │   ├── logging-system.md        # ✅ Logging architecture (1000+ lines)
│       │   ├── training-refactoring.md  # ✅ Training module design (400+ lines)
│       │   ├── refactoring-status.md    # ✅ Refactoring summary (600+ lines)
│       │   └── code-organization.md     # 🚧 Module structure
│       │
│       ├── 📁 testing/                  # Testing documentation
│       │   ├── strategy.md              # ✅ Testing strategy (500+ lines)
│       │   ├── running-tests.md         # ✅ How to run tests (300+ lines)
│       │   ├── logging-tests.md         # ✅ Logging test specs (600+ lines)
│       │   └── writing-tests.md         # 🚧 Contributing tests
│       │
│       ├── 📁 api/                      # API reference docs
│       │   ├── configuration.md         # 🚧 Config classes
│       │   ├── logging-components.md    # 🚧 Logging API
│       │   ├── training-components.md   # 🚧 Training API
│       │   └── data-processing.md       # 🚧 Data API
│       │
│       └── 📁 tutorials/                # Step-by-step tutorials
│           ├── basic-training.md        # 🚧 First experiment
│           ├── custom-datasets.md       # 🚧 Using custom data
│           ├── hyperparameter-tuning.md # 🚧 Optimization
│           └── ablation-studies.md      # 🚧 Systematic studies
│
├── 📁 model_foundry/                    # Source code
│   ├── 📄 README.md                     # Package README (points to /docs)
│   ├── trainer.py
│   ├── logging_components.py
│   ├── config.py
│   └── ...
│
├── 📁 configs/                          # Configuration files
│   └── example_with_wandb.yaml          # Example with WandB enabled
│
└── 📁 analysis/                         # Analysis scripts
    └── scripts/
```

**Legend:**
- ✅ Complete and available
- 🚧 Planned / In progress

---

## 📊 Documentation by Category

### ✅ Available Now (7 documents, 4,300+ lines)

**Guides (1)**
- WandB Integration (500+ lines)

**Architecture (3)**
- Logging System (1,000+ lines)
- Training Refactoring (400+ lines)
- Refactoring Status (600+ lines)

**Testing (3)**
- Testing Strategy (500+ lines)
- Running Tests (300+ lines)
- Logging Tests Spec (600+ lines)

### 🚧 Planned

**Guides (4)**
- Getting Started
- Configuration
- CLI Reference
- Metrics Logging

**Architecture (1)**
- Code Organization

**Testing (1)**
- Writing Tests

**API Reference (4)**
- Configuration API
- Logging Components API
- Training Components API
- Data Processing API

**Tutorials (4)**
- Basic Training
- Custom Datasets
- Hyperparameter Tuning
- Ablation Studies

---

## 🎯 Navigation Guide

### By User Type

**🆕 New User**
```
Start: /docs/README.md
├── Quick Start section
├── /docs/model_foundry/guides/getting-started.md (planned)
└── /configs/example_with_wandb.yaml
```

**👨‍💻 Developer**
```
Start: /docs/model_foundry/architecture/
├── training-refactoring.md (understand training)
├── logging-system.md (understand logging)
└── /docs/model_foundry/api/ (API reference)
```

**🧪 Contributor**
```
Start: /docs/model_foundry/testing/
├── strategy.md (testing approach)
├── running-tests.md (how to run)
└── writing-tests.md (how to write)
```

**📊 Experimenter**
```
Start: /docs/model_foundry/guides/
├── wandb-integration.md (setup tracking)
├── configuration.md (customize experiments)
└── /docs/model_foundry/tutorials/ (step-by-step)
```

---

## 📈 Documentation Metrics

### Size & Coverage

| Category | Files | Total Lines | Avg. Lines/File |
|----------|-------|-------------|-----------------|
| Guides | 1 | 500+ | 500+ |
| Architecture | 3 | 2,000+ | 666+ |
| Testing | 3 | 1,400+ | 466+ |
| API (planned) | 0 | - | - |
| Tutorials (planned) | 0 | - | - |
| **Total** | **7** | **3,900+** | **557+** |

### Completion Status

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

## 🔗 Cross-Reference Map

### How Documents Link Together

```
                   ┌─────────────────┐
                   │  docs/README.md │
                   │  (Master Index) │
                   └────────┬────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │   Guides    │ │Architecture│ │  Testing   │
    └──────┬──────┘ └─────┬──────┘ └─────┬──────┘
           │              │               │
    ┌──────▼──────────────▼───────────────▼──────┐
    │         WandB Integration Guide             │
    │  (References: logging-system.md,            │
    │   configuration.md)                         │
    └─────────────────────────────────────────────┘
           │              │               │
    ┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │  Logging    │ │  Training  │ │  Testing   │
    │  System     │ │Refactoring │ │  Strategy  │
    └──────┬──────┘ └─────┬──────┘ └─────┬──────┘
           │              │               │
           └──────────────┼───────────────┘
                          │
                   ┌──────▼──────┐
                   │     API     │
                   │  Reference  │
                   │  (planned)  │
                   └─────────────┘
```

---

## 🎨 File Naming Conventions

### Pattern: `category-topic.md`

**Examples:**
- `wandb-integration.md` - Clear and descriptive
- `logging-system.md` - Topic-focused
- `training-refactoring.md` - Action-focused
- `refactoring-status.md` - Status document

**Avoid:**
- `wandb.md` - Too generic
- `WANDB_INTEGRATION_GUIDE.md` - Use lowercase
- `wandb_integration.md` - Use hyphens, not underscores
- `the-complete-guide-to-wandb.md` - Too verbose

---

## 📝 Document Templates

### Guide Template

```markdown
# [Guide Title]

**Brief description of what this guide covers.**

## Overview
[High-level overview]

## Prerequisites
[What users need before starting]

## Steps
### 1. [First Step]
[Instructions]

### 2. [Second Step]
[Instructions]

## Advanced Topics
[Optional advanced content]

## Troubleshooting
[Common issues and solutions]

## Next Steps
[Where to go next]
```

### Architecture Document Template

```markdown
# [Component Name] Architecture

**Description of the component.**

## Overview
[High-level architecture]

## Design Principles
[Key design decisions]

## Components
### [Component 1]
[Details]

## Implementation
[Code structure]

## Examples
[Usage examples]

## References
[Related documentation]
```

---

## 🚀 Quick Access by Task

| I want to... | Go to... |
|--------------|----------|
| **Get started** | `/docs/README.md` → Quick Start |
| **Set up WandB** | `/docs/model_foundry/guides/wandb-integration.md` |
| **Understand logging** | `/docs/model_foundry/architecture/logging-system.md` |
| **Run tests** | `/docs/model_foundry/testing/running-tests.md` |
| **Understand training** | `/docs/model_foundry/architecture/training-refactoring.md` |
| **Write tests** | `/docs/model_foundry/testing/writing-tests.md` (planned) |
| **Configure experiments** | `/docs/model_foundry/guides/configuration.md` (planned) |
| **Use the API** | `/docs/model_foundry/api/` (planned) |
| **Learn with tutorials** | `/docs/model_foundry/tutorials/` (planned) |
| **Find all docs** | `DOCUMENTATION_MAP.md` |

---

## 📅 Roadmap

### Phase 1: Foundation ✅ (Complete)
- [x] Create centralized structure
- [x] Move existing documentation
- [x] Create master index
- [x] Create documentation map

### Phase 2: Essential Guides 🚧 (In Progress)
- [ ] Getting Started guide
- [ ] Configuration guide
- [ ] CLI reference

### Phase 3: API Reference 🔜 (Planned)
- [ ] Configuration API
- [ ] Logging Components API
- [ ] Training Components API
- [ ] Data Processing API

### Phase 4: Tutorials 🔜 (Planned)
- [ ] Basic Training tutorial
- [ ] Custom Datasets tutorial
- [ ] Hyperparameter Tuning tutorial
- [ ] Ablation Studies tutorial

---

**Last Updated**: 2025-09-30
**Documentation Structure Version**: 1.0.0
