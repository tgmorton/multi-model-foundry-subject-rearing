# Gemini Project Charter: Subject-Drop Rearing Study

This document outlines the core principles, architecture, and workflow for the "Just drop the subject" experimental framework. Its purpose is to provide context and guide future development and interaction.

## 1\. Project Goal & Intentions

The primary objective of this project is to conduct a **controlled rearing study** to investigate the acquisition of grammatical rules in Large Language Models (LLMs), specifically focusing on **subject-drop phenomena in English**.

The experimental methodology involves:

  * **Dataset**: Training on the **BabyLM dataset**, a 100-million-word corpus designed to be developmentally plausible.
  * **Interventions**: Performing a series of **linguistic ablations** on the training corpus to create controlled experimental conditions (e.g., removing expletives, impoverishing morphology).
  * **Measurement**: Evaluating model performance using **surprisal** on minimal linguistic pairs and general competence with the **BLIMP benchmark**.

## 2\. Core Design Principles

The codebase is built on the following principles to ensure a robust, scalable, and reproducible research environment:

  * **Configuration-Driven**: Every aspect of an experiment (data, ablations, model hyperparameters, training procedure) is defined in a single, human-readable `.yaml` file. The code is merely an engine that executes the plan laid out in the config. This makes experiments easy to define, track, and share.
  * **Modularity**: The codebase is strictly divided into logical components: data preparation, linguistic preprocessing, tokenization, model training, and evaluation. Each component is independent and can be run separately.
  * **Reproducibility**: By combining a configuration-driven approach with a fixed random seed, any experiment can be perfectly replicated. The `.yaml` file is the single source of truth for an entire experimental pipeline.
  * **Scalability**: The framework is designed to handle large corpora by processing files in memory-efficient streams and chunks, rather than loading entire datasets at once.

## 3\. Codebase Architecture & Workflow

The project is organized into a well-defined filesystem and is operated through a modular Command Line Interface (CLI).

### 3.1. Filesystem Structure

```
.
├── configs/              # Experiment definition files (.yaml)
├── data/                 # Raw, processed, and evaluation datasets
├── preprocessing/        # Scripts for linguistic ablations
├── tokenizers/           # Trained tokenizer models for each experiment
├── models/               # Saved model checkpoints
├── model_foundry/        # The core Python source code package
│   ├── cli.py            # Main CLI entrypoint (using Typer)
│   ├── config.py         # Pydantic models for validating configs
│   └── ...
├── evaluation/           # Scripts for post-training analysis
├── .gitignore
├── pyproject.toml
└── README.md
```

### 3.2. Modular CLI Workflow

The entire experimental pipeline is executed through a series of independent commands, making it ideal for managing resources on a compute cluster.

1.  **Define Experiment**: Create a new `.yaml` file in `configs/`.
2.  **Prepare Corpus (One-Time Setup)**:
    ```sh
    python preprocessing/00_prepare_corpus.py --source_dir <...> --main_output_dir <...> --pool_output_dir <...>
    ```
3.  **Run Linguistic Ablations**:
    ```sh
    python preprocessing/remove_expletives.py --input_dir <...> --output_dir <...> --replacement_pool_dir <...>
    ```
4.  **Train Tokenizer**:
    ```sh
    python -m model_foundry.cli train-tokenizer configs/your_experiment.yaml
    ```
5.  **Train Model**:
    ```sh
    python -m model_foundry.cli run configs/your_experiment.yaml
    ```
6.  **Evaluate Model**:
    ```sh
    python -m evaluation.surprisal --checkpoint_path <...> --stimuli_file <...>
    ```