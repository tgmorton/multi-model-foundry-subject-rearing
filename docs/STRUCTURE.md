# Documentation Structure

Layout of the `docs/` tree as it currently exists on disk and how it maps onto the MkDocs navigation.

## Directory Tree

```
docs/
├── README.md                              # Master index (Home)
├── STRUCTURE.md                           # This file
├── DOCUMENTATION_MAP.md                   # Quick topic-to-file map
│
├── TRAINING_GUIDE.md                      # Training framework guide
├── TRAINING_ON_SLURM.md                   # SLURM instructions
├── TRAINING_ON_WILD_WEST.md               # Wild West cluster instructions
├── data_processing.md                     # Data processing pipeline
│
├── preprocessing/                         # Preprocessing module docs
│   ├── README.md
│   ├── USER_GUIDE.md
│   ├── ADVANCED.md
│   ├── DEVELOPER_GUIDE.md
│   ├── TESTING.md
│   └── CHANGES.md
├── ABLATION_ENRICHMENT_IMPLEMENTATION.md  # Ablation enrichment notes
│
├── pronoun_recovery.md                    # Pronoun recovery system reference
├── italian_null_subject.md                # Italian null-subject pipeline
├── ttq_corpus_report.md                   # TTQ corpus report
├── spanish_corpus.md                      # Spanish corpus construction
│
├── annotation_app.md                      # Flask annotation web app
├── LAYERED_ANNOTATION_ARCHITECTURE.md     # Layered annotation design
│
├── NEW_CORPUS_ANALYSIS_PLAN.md            # Current corpus analysis spec
├── CORPUS_ANALYSIS_SPEC.md                # Superseded original spec
├── corpus_analysis/
│   ├── README.md
│   └── CHANGES.md
│
├── phase3_experimental_pipeline.md        # Phase 3 pipeline
├── CROSS_ARCHITECTURE_COMPARISON.md       # Cross-architecture comparison
├── new_checkpoint_scheduling.md           # Current checkpoint scheduler
├── checkpoint_scheduling.md               # Superseded Phase 2 version
│
├── k8s_jobs.md                            # Kubernetes job workflow
├── NRP_REGISTRY_SETUP.md                  # NRP container registry setup
│
├── model_foundry/                         # Training framework docs
│   ├── README.md
│   ├── architecture/
│   │   ├── multi-architecture-system.md
│   │   ├── logging-system.md
│   │   ├── training-refactoring.md
│   │   └── refactoring-status.md
│   ├── testing/
│   │   ├── strategy.md
│   │   ├── running-tests.md
│   │   └── logging-tests.md
│   └── guides/
│       └── wandb-integration.md
│
├── OSF_PREREGISTRATION.md                 # OSF preregistration
│
├── archive/                               # Historical plans (not in nav)
├── research-notes/                        # Private notes (not in nav)
├── nrp-docs/                              # Scraped NRP docs (not in nav)
├── jobpostings.md                         # Scratch (not in nav)
├── 2502.12317v1.pdf                       # Reference paper (not in nav)
└── OSF_PREREGISTRATION.{tex,docx}         # Source formats (not in nav)
```

## Sections

The MkDocs navigation (see `mkdocs.yml`) groups these files into the following top-level sections:

| Section | Purpose |
|---------|---------|
| Home / Project Structure / Documentation Map | Top-level index files. |
| Getting Started | Training guides and the data processing overview. |
| Preprocessing | The `preprocessing/` module: user, advanced, and developer docs, plus ablation-enrichment notes. |
| Pronoun Recovery & Null Subjects | Core research pipeline for null-subject detection and pronoun recovery, across English, Italian, and Spanish. |
| Annotation | The annotation web app and the layered annotation architecture shared across pipelines. |
| Corpus Analysis | The descriptive corpus analysis spec and module docs. |
| Experiments | Experimental designs (Phase 3, cross-architecture) and checkpoint scheduling. |
| Infrastructure | Kubernetes job workflow on NRP and container registry setup. |
| Model Foundry | Training framework architecture, testing, and integration guides. |
| Research | OSF preregistration and related research artefacts. |

## Excluded From the Published Site

The following live in `docs/` but are intentionally absent from `mkdocs.yml`:

- `docs/archive/` — old planning documents and validation records kept for reference.
- `docs/research-notes/` — working notes, presentation drafts, and chapter-integration plans.
- `docs/nrp-docs/` — a local scrape of Nautilus (NRP) third-party documentation.
- `docs/2502.12317v1.pdf` — an external reference paper.
- `docs/jobpostings.md` — unrelated scratch file.
- `docs/OSF_PREREGISTRATION.tex` and `.docx` — source formats for the preregistration, with the `.md` version being canonical.

If any of these should be published, add them to `mkdocs.yml` and update this file and `README.md` accordingly.

## Conventions

- File names mix SCREAMING_SNAKE_CASE (older docs), kebab-case (Model Foundry docs), and lowercase_with_underscores (newer pipeline docs). When adding a doc to an existing subdirectory, match the neighbouring style.
- Superseded documents should carry a one-line banner at the top pointing to the canonical replacement rather than being deleted, so old links keep working.
- Update both `mkdocs.yml` and `docs/README.md` whenever adding or removing a document.
