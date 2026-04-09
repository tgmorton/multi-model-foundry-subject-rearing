# Multi-Model Foundry — Subject Rearing Documentation

Documentation index for the controlled-rearing language model project. This site is built with MkDocs Material and deployed to GitHub Pages.

This repository combines:

- A training framework (Model Foundry) for controlled-rearing experiments with GPT-2 and related architectures.
- A preprocessing pipeline with ablations for linguistic phenomena (expletive removal, subject drop, etc.).
- A layered annotation system over CHILDES, Europarl, and TTQ corpora.
- A null-subject / pronoun-recovery pipeline (English and Italian; Spanish in progress).
- Kubernetes and SLURM infrastructure for running long jobs on the Nautilus Research Platform and university HPC clusters.

---

## Getting Started

- [Training Guide](TRAINING_GUIDE.md) — end-to-end guide for running a training experiment.
- [Training on SLURM](TRAINING_ON_SLURM.md) — running on a SLURM cluster.
- [Training on Wild West](TRAINING_ON_WILD_WEST.md) — running on the Wild West GPU cluster.
- [Data Processing](data_processing.md) — how input corpora are processed before training.

## Preprocessing

Source: `preprocessing/`. The preprocessing pipeline produces cleaned, tokenised, and ablated training corpora.

- [Overview](preprocessing/README.md)
- [User Guide](preprocessing/USER_GUIDE.md)
- [Advanced Usage](preprocessing/ADVANCED.md) — performance tuning, coreference, production notes.
- [Developer Guide](preprocessing/DEVELOPER_GUIDE.md) — adding new ablations.
- [Testing](preprocessing/TESTING.md)
- [Changelog](preprocessing/CHANGES.md)
- [Ablation Enrichment Implementation](ABLATION_ENRICHMENT_IMPLEMENTATION.md)

## Pronoun Recovery and Null Subjects

The pronoun recovery pipeline identifies null-subject verbs and recovers the dropped pronoun, used both to construct training ablations and to evaluate model acquisition of null-subject phenomena.

- [System Reference](pronoun_recovery.md) — overall architecture, models, and data flow.
- [Italian Pipeline](italian_null_subject.md) — Italian null-subject detection, tree detector, CHILDES longitudinal analysis.
- [TTQ Corpus Report](ttq_corpus_report.md) — Tatoeba translation questions corpus for Italian.
- [Spanish Corpus](spanish_corpus.md) — Spanish BebeLM monolingual and parallel corpus construction.

## Annotation

- [Annotation Web App](annotation_app.md) — Flask-based annotation interface in `annotation/`.
- [Layered Annotation Architecture](LAYERED_ANNOTATION_ARCHITECTURE.md) — design of the multi-layer annotation system.

## Corpus Analysis

- [Specification](NEW_CORPUS_ANALYSIS_PLAN.md) — current corpus descriptive analysis spec.
- [Original Specification](CORPUS_ANALYSIS_SPEC.md) (superseded)
- [Corpus Analysis Module](corpus_analysis/README.md)
- [Corpus Analysis Changelog](corpus_analysis/CHANGES.md)

## Experiments

- [Phase 3 Experimental Pipeline](phase3_experimental_pipeline.md)
- [Cross-Architecture Comparison](CROSS_ARCHITECTURE_COMPARISON.md)
- [Checkpoint Scheduling](new_checkpoint_scheduling.md) — current system with configurable frequency and spacing.
- [Checkpoint Scheduling (Phase 2)](checkpoint_scheduling.md) (superseded)

## Infrastructure

- [Kubernetes Jobs](k8s_jobs.md) — job templates under `k8s/` for NRP and related clusters.
- [NRP Registry Setup](NRP_REGISTRY_SETUP.md) — container registry configuration for Nautilus.

## Model Foundry

The training framework. Source: `model_foundry/`.

- [Overview](model_foundry/README.md)

**Architecture**

- [Multi-Architecture System](model_foundry/architecture/multi-architecture-system.md)
- [Logging System](model_foundry/architecture/logging-system.md)
- [Training Refactoring](model_foundry/architecture/training-refactoring.md)
- [Refactoring Status](model_foundry/architecture/refactoring-status.md)

**Testing**

- [Testing Strategy](model_foundry/testing/strategy.md)
- [Running Tests](model_foundry/testing/running-tests.md)
- [Logging Tests](model_foundry/testing/logging-tests.md)

**Guides**

- [WandB Integration](model_foundry/guides/wandb-integration.md)

## Research

- [OSF Preregistration](OSF_PREREGISTRATION.md)

---

## Conventions

- Source code lives outside `docs/`; documentation within the repo should live in `docs/` and be linked from this index.
- For new docs, prefer kebab-case or lowercase-with-underscores filenames consistent with neighbouring files in the same section.
- Update `mkdocs.yml` whenever adding, renaming, or removing a doc so the deployed site stays consistent with the tree.
- The deployment workflow (`.github/workflows/deploy-docs.yml`) rebuilds on changes to `docs/**` or `mkdocs.yml`.

## Not in the Site

The following are present in `docs/` but intentionally excluded from the published site:

- `docs/archive/**` — historical plans and validation records.
- `docs/research-notes/**` — private research notes and presentation drafts.
- `docs/nrp-docs/**` — scraped third-party Nautilus documentation.
- `docs/2502.12317v1.pdf` — reference paper, not project documentation.
- `docs/jobpostings.md` — unrelated scratch file.
- `docs/OSF_PREREGISTRATION.tex`, `docs/OSF_PREREGISTRATION.docx` — source formats for the preregistration (the Markdown version is canonical).
