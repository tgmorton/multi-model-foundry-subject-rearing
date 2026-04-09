# Documentation Map

Quick topic-to-file lookup. For the full table of contents, see [README.md](README.md); for the on-disk layout, see [STRUCTURE.md](STRUCTURE.md).

## Find Documentation by Topic

### Training

- Running a training experiment — [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- SLURM cluster instructions — [TRAINING_ON_SLURM.md](TRAINING_ON_SLURM.md)
- Wild West GPU cluster — [TRAINING_ON_WILD_WEST.md](TRAINING_ON_WILD_WEST.md)
- Training framework architecture — [model_foundry/architecture/training-refactoring.md](model_foundry/architecture/training-refactoring.md)
- Multi-architecture support — [model_foundry/architecture/multi-architecture-system.md](model_foundry/architecture/multi-architecture-system.md)
- Checkpoint scheduling (current) — [new_checkpoint_scheduling.md](new_checkpoint_scheduling.md)
- Checkpoint scheduling (Phase 2, superseded) — [checkpoint_scheduling.md](checkpoint_scheduling.md)

### Preprocessing and Ablations

- Preprocessing overview — [preprocessing/README.md](preprocessing/README.md)
- User guide — [preprocessing/USER_GUIDE.md](preprocessing/USER_GUIDE.md)
- Advanced usage — [preprocessing/ADVANCED.md](preprocessing/ADVANCED.md)
- Adding custom ablations — [preprocessing/DEVELOPER_GUIDE.md](preprocessing/DEVELOPER_GUIDE.md)
- Testing the preprocessing pipeline — [preprocessing/TESTING.md](preprocessing/TESTING.md)
- Preprocessing changelog — [preprocessing/CHANGES.md](preprocessing/CHANGES.md)
- Ablation enrichment implementation — [ABLATION_ENRICHMENT_IMPLEMENTATION.md](ABLATION_ENRICHMENT_IMPLEMENTATION.md)
- Data processing pipeline overview — [data_processing.md](data_processing.md)

### Pronoun Recovery and Null Subjects

- System reference — [pronoun_recovery.md](pronoun_recovery.md)
- Italian null-subject pipeline (tree detector, CHILDES longitudinal analysis) — [italian_null_subject.md](italian_null_subject.md)
- TTQ corpus report — [ttq_corpus_report.md](ttq_corpus_report.md)
- Spanish corpus construction — [spanish_corpus.md](spanish_corpus.md)

### Annotation

- Annotation web app (Flask) — [annotation_app.md](annotation_app.md)
- Layered annotation architecture — [LAYERED_ANNOTATION_ARCHITECTURE.md](LAYERED_ANNOTATION_ARCHITECTURE.md)

### Corpus Analysis

- Current descriptive analysis spec — [NEW_CORPUS_ANALYSIS_PLAN.md](NEW_CORPUS_ANALYSIS_PLAN.md)
- Superseded original spec — [CORPUS_ANALYSIS_SPEC.md](CORPUS_ANALYSIS_SPEC.md)
- Module README — [corpus_analysis/README.md](corpus_analysis/README.md)
- Module changelog — [corpus_analysis/CHANGES.md](corpus_analysis/CHANGES.md)

### Experiments

- Phase 3 experimental pipeline — [phase3_experimental_pipeline.md](phase3_experimental_pipeline.md)
- Cross-architecture comparison — [CROSS_ARCHITECTURE_COMPARISON.md](CROSS_ARCHITECTURE_COMPARISON.md)

### Infrastructure

- Kubernetes job workflow — [k8s_jobs.md](k8s_jobs.md)
- NRP container registry setup — [NRP_REGISTRY_SETUP.md](NRP_REGISTRY_SETUP.md)

### Logging and Monitoring

- Logging system architecture — [model_foundry/architecture/logging-system.md](model_foundry/architecture/logging-system.md)
- WandB integration guide — [model_foundry/guides/wandb-integration.md](model_foundry/guides/wandb-integration.md)

### Testing

- Testing strategy — [model_foundry/testing/strategy.md](model_foundry/testing/strategy.md)
- Running tests — [model_foundry/testing/running-tests.md](model_foundry/testing/running-tests.md)
- Logging test specs — [model_foundry/testing/logging-tests.md](model_foundry/testing/logging-tests.md)
- Preprocessing test suite — [preprocessing/TESTING.md](preprocessing/TESTING.md)

### Research Artefacts

- OSF preregistration — [OSF_PREREGISTRATION.md](OSF_PREREGISTRATION.md)

## Duplicate and Superseded Files

Two pairs of "plan" files exist side by side. The newer version is canonical; the older version has a banner at the top pointing to it and is kept so historical links keep working.

| Superseded | Canonical |
|------------|-----------|
| [checkpoint_scheduling.md](checkpoint_scheduling.md) | [new_checkpoint_scheduling.md](new_checkpoint_scheduling.md) |
| [CORPUS_ANALYSIS_SPEC.md](CORPUS_ANALYSIS_SPEC.md) | [NEW_CORPUS_ANALYSIS_PLAN.md](NEW_CORPUS_ANALYSIS_PLAN.md) |

## Where to Add New Docs

| Type | Location | Notes |
|------|----------|-------|
| Training framework architecture / design | `docs/model_foundry/architecture/` | Use kebab-case. |
| Training framework tests | `docs/model_foundry/testing/` | |
| Training framework user guides | `docs/model_foundry/guides/` | |
| Preprocessing | `docs/preprocessing/` | Match existing SCREAMING_SNAKE_CASE. |
| Corpus analysis module | `docs/corpus_analysis/` | |
| Pipeline / research docs | `docs/` (top level) | Use lowercase_with_underscores like `pronoun_recovery.md`. |
| Infrastructure (K8s, SLURM, NRP) | `docs/` (top level) | |

After adding a file, update `mkdocs.yml`, [README.md](README.md), [STRUCTURE.md](STRUCTURE.md), and this map.

## Files Not in the Published Site

- `docs/archive/` — historical plans and validation records.
- `docs/research-notes/` — private notes and presentation drafts.
- `docs/nrp-docs/` — scraped third-party NRP documentation.
- `docs/2502.12317v1.pdf` — reference paper.
- `docs/jobpostings.md` — unrelated scratch file.
- `docs/OSF_PREREGISTRATION.tex`, `docs/OSF_PREREGISTRATION.docx` — source formats; the `.md` version is canonical.
