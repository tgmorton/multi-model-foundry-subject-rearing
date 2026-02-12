# Phase 3: Experimental Pipeline

This document describes the complete experimental pipeline for the controlled rearing study of subject drop in English.

## Overview

Phase 3 implements the complete experimental pipeline that orchestrates data preprocessing, model training, and evaluation for all 8 experiments (0-7) defined in the project design.

## Experiment Design

The experiments follow this design table from the project document:

| Exp. | No Expletives | Poor Determiner | No Articles | Infinitive Verbal | No Pronominal Subjects |
| :--- | :-----------: | :-------------: | :---------: | :---------------: | :--------------------: |
| 0    |       X       |        X        |      X      |         X         |           X            |
| 1    |       ✓       |        X        |      X      |         X         |           X            |
| 2    |       ✓       |        ✓        |      X      |         X         |           X            |
| 3    |       ✓       |        X        |      ✓      |         X         |           X            |
| 4    |       ✓       |        X        |      X      |         ✓         |           X            |
| 5    |       ✓       |        X        |      X      |         X         |           ✓            |
| 6    |       ✓       |        ✓        |      X      |         ✓         |           X            |
| 7    |       ✓       |        ✓        |      ✓      |         ✓         |           ✓            |

## Components

### 1. Experiment Configurations

All experiment configurations are stored in `configs/`:

- `experiment_0_baseline.yaml` - Baseline (no ablations)
- `experiment_1_remove_expletives.yaml` - Remove expletives only
- `experiment_2_impoverish_determiners.yaml` - Remove expletives + impoverish determiners
- `experiment_3_remove_articles.yaml` - Remove expletives + remove articles
- `experiment_4_lemmatize_verbs.yaml` - Remove expletives + lemmatize verbs
- `experiment_5_remove_subject_pronominals.yaml` - Remove expletives + remove subject pronominals
- `experiment_6_impoverish_determiners_lemmatize_verbs.yaml` - Remove expletives + impoverish determiners + lemmatize verbs
- `experiment_7_all_ablations.yaml` - All ablations (most impoverished)

### 2. Ablation Scripts

All ablation scripts are implemented in `preprocessing/`:

- `remove_expletives.py` - Removes non-referential expletive subjects
- `impoverish_determiners.py` - Replaces all determiners with 'the'
- `remove_articles.py` - Removes basic articles (a, an, the)
- `lemmatize_verbs.py` - Converts verbs to infinitive form
- `remove_subject_pronominals.py` - Removes subject pronouns

### 3. Evaluation Scripts

Evaluation scripts are in `evaluation/`:

- `surprisal.py` - Calculates surprisal for minimal linguistic pairs
- `run_blimp.py` - Evaluates on BLIMP benchmark
- `stimuli/subject_drop_stimuli.json` - Subject-drop specific stimuli

### 4. Master Orchestration

- `scripts/run_experiment.py` - Master script to run complete pipeline

## Usage

### Running a Complete Experiment

```bash
# Run experiment 1 (remove expletives only)
python scripts/run_experiment.py 1

# Run experiment 7 (all ablations)
python scripts/run_experiment.py 7

# Skip certain steps (e.g., if data preprocessing is already done)
python scripts/run_experiment.py 3 --skip-steps data_preprocessing tokenizer_training

# Evaluate a specific checkpoint
python scripts/run_experiment.py 5 --checkpoint-step 1000
```

### Individual Pipeline Steps

```bash
# 1. Data preprocessing (including ablations)
python -m model_foundry.cli preprocess-data configs/experiments/experiment_1_remove_expletives.yaml

# 2. Train tokenizer
python -m model_foundry.cli train-tokenizer configs/experiments/experiment_1_remove_expletives.yaml

# 3. Tokenize dataset
python -m model_foundry.cli tokenize-dataset configs/experiments/experiment_1_remove_expletives.yaml

# 4. Preprocess data (chunking)
python -m model_foundry.cli preprocess-data configs/experiments/experiment_1_remove_expletives.yaml

# 5. Generate checkpoint schedule
python -m model_foundry.cli generate-checkpoints configs/experiments/experiment_1_remove_expletives.yaml

# 6. Train model
python -m model_foundry.cli run configs/experiments/experiment_1_remove_expletives.yaml

# 7. Evaluate model
python evaluation/surprisal.py models/experiment_1_remove_expletives/ tokenizers/experiment_1_remove_expletives/ evaluation/stimuli/subject_drop_stimuli.json

python evaluation/run_blimp.py models/experiment_1_remove_expletives/ tokenizers/experiment_1_remove_expletives/
```

## Pipeline Steps

### Step 1: Data Preprocessing
- Applies specified ablations to the raw corpus
- Creates processed corpus in `data/processed/experiment_X/`

### Step 2: Tokenizer Training
- Trains SentencePiece tokenizer on processed corpus
- Saves tokenizer to `tokenizers/experiment_X/`

### Step 3: Dataset Tokenization
- Tokenizes processed corpus using trained tokenizer
- Saves tokenized data to `data/tokenized/experiment_X/`

### Step 4: Data Chunking
- Creates fixed-length chunks for training
- Saves chunked data to `data/chunked/experiment_X/`

### Step 5: Checkpoint Schedule Generation
- Generates optimal checkpoint schedule based on dataset size
- Updates config file with checkpoint schedule

### Step 6: Model Training
- Trains GPT-2 model with specified architecture
- Saves checkpoints according to schedule
- Logs metrics to wandb (if enabled)

### Step 7: Model Evaluation
- Runs surprisal evaluation on subject-drop stimuli
- Runs BLIMP evaluation for general linguistic performance
- Saves results to `evaluation/` directory

## Evaluation Metrics

### Surprisal Evaluation
- **Metric**: Surprisal difference between preferred and dispreferred sentences
- **Formula**: S(w_i) = -log₂ P(w_i | w_1, ..., w_{i-1})
- **Interpretation**: Higher surprisal for ungrammatical sentences indicates better learning

### BLIMP Evaluation
- **Metric**: Accuracy across 17 linguistic phenomena
- **Coverage**: 67 minimal pairs total
- **Interpretation**: Higher accuracy indicates better general linguistic competence

## Output Structure

```
project/
├── configs/
│   ├── experiment_0_baseline.yaml
│   ├── experiment_1_remove_expletives.yaml
│   └── ...
├── data/
│   ├── processed/
│   │   ├── experiment_1_remove_expletives/
│   │   └── ...
│   ├── tokenized/
│   │   ├── experiment_1_remove_expletives/
│   │   └── ...
│   └── chunked/
│       ├── experiment_1_remove_expletives/
│       └── ...
├── tokenizers/
│   ├── experiment_1_remove_expletives/
│   └── ...
├── models/
│   ├── experiment_1_remove_expletives/
│   │   ├── checkpoint-100/
│   │   ├── checkpoint-200/
│   │   └── ...
│   └── ...
└── evaluation/
    ├── surprisal_exp1.json
    ├── blimp_exp1.json
    └── ...
```

## Configuration Parameters

Each experiment configuration includes:

```yaml
experiment_name: "experiment_1_remove_expletives"

data:
  source_corpus: "data/raw/train_90M/"
  training_corpus: "data/processed/experiment_1_remove_expletives/"
  batch_size: 256
  max_sequence_length: 128

dataset_manipulation:
  - remove_expletives

tokenizer:
  output_dir: "tokenizers/experiment_1_remove_expletives/"
  vocab_size: 50004

model:
  layers: 12
  embedding_size: 768
  hidden_size: 768
  intermediate_hidden_size: 3072
  attention_heads: 12
  activation_function: "GELU"
  dropout: 0.1
  attention_dropout: 0.1

training:
  output_dir: "models/experiment_1_remove_expletives/"
  learning_rate: 0.0001
  train_steps: 1000000
  epochs: 20
  auto_generate_checkpoints: true
  first_epoch_checkpoints: 20
  subsequent_epochs_spacing: "log"
  log_base: 2
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Step-by-step validation**: Each step validates its inputs and outputs
- **Graceful failures**: Failed steps don't crash the entire pipeline
- **Detailed logging**: All steps provide detailed progress and error information
- **Resume capability**: Can skip completed steps to resume from failures

## Performance Considerations

- **Parallel processing**: Ablation scripts can process multiple files in parallel
- **Memory efficiency**: Data chunking prevents memory issues with large datasets
- **Checkpoint optimization**: Dynamic checkpoint scheduling based on actual dataset size
- **Evaluation efficiency**: Batch processing for evaluation scripts

## Reproducibility

- **Fixed seeds**: All experiments use fixed random seeds
- **Deterministic processing**: Ablation scripts produce consistent results
- **Version tracking**: All dependencies and configurations are versioned
- **Logging**: Complete experiment logs saved for each run 