# Data Processing Pipeline

This document describes the data processing pipeline implemented in Phase 1 of the Model Foundry framework.

## Overview

The data processing pipeline handles the conversion of raw text corpora into fixed-length chunks suitable for language model training. This ensures efficient training with consistent sequence lengths and proper memory management.

## Pipeline Stages

### 1. Text Preprocessing (Ablations)
- **Location**: `preprocessing/` directory
- **Purpose**: Apply linguistic ablations to the raw corpus
- **Entry point**: `preprocessing.base.AblationPipeline`, driven by
  `AblationConfig` (`preprocessing/config.py`). Ablations register
  themselves with `AblationRegistry` on import
  (`preprocessing/ablations/__init__.py`).
- **Active ablation modules** (`preprocessing/ablations/`):
  - `remove_expletive_sentences.py` — registers
    `remove_expletive_sentences_en` (three-tier stateful detector:
    spaCy `dep_=='expl'`, weather / raising-*it* heuristics, optional
    coreference confirmation) and `remove_expletive_sentences_it`
    (weather verbs, existential *ci + essere*, impersonal raising and
    necessity verbs).
  - `impoverish_case.py` — registers `impoverish_case_en` and
    `impoverish_case_it` (collapses pronoun case distinctions).
  - `lemmatize_verbs.py` — registers `lemmatize_verbs` (replaces verbs
    with their lemma).
  - `enrich_verbal_morphology.py` — registers
    `enrich_verbal_morphology` (appends subject-agreement clitics to
    finite verbs).
- **Archived** (not registered): anything under
  `preprocessing/ablations/archived/`.
- See `docs/preprocessing/USER_GUIDE.md` for the full list and usage.

### 2. Tokenization
- **Location**: `model_foundry/tokenizer/`
- **Purpose**: Convert text to token IDs using SentencePiece
- **Output**: HuggingFace dataset with `input_ids` column

### 3. Data Chunking (NEW)
- **Location**: `model_foundry/data.py`
- **Purpose**: Convert variable-length sequences into fixed-length chunks
- **Output**: HuggingFace dataset with fixed-length `input_ids`

### 4. Training Data Loading
- **Location**: `model_foundry/data.py`
- **Purpose**: Create efficient DataLoader for training
- **Features**: Proper batching, padding, and memory management

## Key Features

### Fixed-Length Chunking
- Converts variable-length token sequences into fixed-length chunks
- Default chunk size: 128 tokens (configurable via `max_sequence_length`)
- Non-overlapping chunks for efficient training
- Skips sequences shorter than chunk size

### Data Validation
- Validates tokenized dataset structure
- Calculates and displays dataset statistics
- Ensures data quality before training

### Efficient Loading
- Pre-processed chunks stored on disk
- Fast loading during training
- Proper memory management with DataLoader

## Usage

### CLI Commands

1. **Preprocess data** (after tokenization):
   ```bash
   python -m model_foundry.cli preprocess-data configs/experiment.yaml
   ```

2. **Force reprocessing**:
   ```bash
   python -m model_foundry.cli preprocess-data configs/experiment.yaml --force
   ```

### Programmatic Usage

```python
from model_foundry.data import create_data_processor

# Create data processor
data_processor = create_data_processor(config, base_dir)

# Preprocess data
success = data_processor.preprocess_data()

# Create dataloader for training
dataloader = data_processor.create_dataloader(tokenizer)
```

## Configuration

The data processing is configured through the experiment YAML file:

```yaml
data:
  source_corpus: "data/raw/train_90M/"
  training_corpus: "data/processed/ablated_corpus/"
  batch_size: 256
  max_sequence_length: 128  # Chunk size
```

## File Structure

```
data/
├── raw/                    # Original text files
├── processed/              # Ablated text files
├── tokenized/              # Tokenized datasets
│   └── experiment_name/
└── chunked/               # Fixed-length chunks (NEW)
    └── experiment_name/
```

## Benefits

1. **Efficiency**: Pre-chunked data loads faster during training
2. **Consistency**: Fixed-length sequences ensure stable training
3. **Memory**: Better memory management with proper chunking
4. **Validation**: Data quality checks prevent training issues
5. **Flexibility**: Configurable chunk sizes for different experiments

## Testing

Run the test suite to validate the data processing:

```bash
python tests/test_data_processing.py
```

This tests:
- DataProcessor initialization
- Sequence chunking logic
- Dataset statistics calculation
- Chunked dataset creation 