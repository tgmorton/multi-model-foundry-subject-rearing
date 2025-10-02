# Subject Drop Acquisition in English

A controlled-rearing study investigating how linguistic input shapes grammatical knowledge in Large Language Models, with a focus on subject-drop phenomena in English.

## Overview

This repository provides a complete experimental framework for training and evaluating language models on systematically manipulated corpora. The codebase enables reproducible experiments where training data can be ablated (e.g., removing expletives, articles, or subject pronouns) to isolate specific learning signals.

**Key Features:**
- Configuration-driven experiments (single YAML defines entire pipeline)
- Modular preprocessing for linguistic ablations
- Custom tokenizer training per experiment
- Automated model training with checkpoint scheduling
- Comprehensive evaluation suite (surprisal analysis, BLIMP, null-subject phenomena)
- Statistical analysis pipeline in R

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_trf

# Run complete experiment pipeline
python scripts/run_experiment.py 0  # baseline
python scripts/run_experiment.py 1  # remove expletives
```

## Project Structure

```
├── configs/              # Experiment configurations (YAML)
├── model_foundry/        # Core training framework
│   ├── training/         # Training loop, checkpointing
│   ├── tokenizer/        # Tokenizer training
│   └── cli.py            # Command-line interface
├── preprocessing/        # Linguistic ablation scripts
├── evaluation/           # Model evaluation suite
│   ├── core/             # Surprisal calculation
│   ├── evaluators/       # BLIMP, null-subject tests
│   └── runners/          # Evaluation orchestration
├── analysis/             # Statistical analysis (R)
│   └── scripts/          # Mixed-effects models, visualizations
├── data/                 # Training corpora
├── models/               # Trained model checkpoints
└── tokenizers/           # Experiment-specific tokenizers
```

## Experiment Workflow

Each experiment is defined by a single YAML configuration:

```yaml
experiment_name: "exp1_remove_expletives"

data:
  source_corpus: "data/raw/train_90M/"
  batch_size: 32
  max_sequence_length: 1000

dataset_manipulation:
  - "remove_expletives"

tokenizer:
  output_dir: "tokenizers/exp1/"
  vocab_size: 50004

model:
  layers: 12
  hidden_size: 768
  attention_heads: 12

training:
  learning_rate: 0.0004
  epochs: 20
  use_amp: true
```

### Running an Experiment

**Option 1: Complete Pipeline (Automated)**
```bash
python scripts/run_experiment.py <experiment_number>
```

**Option 2: Step-by-Step**
```bash
# 1. Preprocess corpus
python -m model_foundry.cli preprocess configs/experiment_1.yaml

# 2. Train tokenizer
python -m model_foundry.cli train-tokenizer configs/experiment_1.yaml

# 3. Train model
python -m model_foundry.cli run configs/experiment_1.yaml

# 4. Evaluate
python -m evaluation.runners.evaluation_runner --config configs/eval_experiment_1.yaml
```

## Available Ablations

Preprocessing scripts in `preprocessing/` enable linguistic manipulations:

- **`remove_expletives.py`** - Remove expletive subjects (*it*, *there*)
- **`remove_articles.py`** - Remove articles (*a*, *an*, *the*)
- **`impoverish_determiners.py`** - Replace determiners with generic form
- **`remove_subject_pronominals.py`** - Remove overt subject pronouns
- **`lemmatize_verbs.py`** - Reduce verb morphology

## Evaluation

The evaluation suite includes:

1. **BLIMP** - Minimal pair grammaticality judgments across 67 linguistic phenomena
2. **Null Subject Stimuli** - Target sentences testing subject-drop acceptability
3. **Perplexity** - Standard language modeling metrics
4. **Surprisal Analysis** - Word-by-word predictability measures

Results are automatically aggregated and formatted for statistical analysis in R.

## Statistical Analysis

R scripts in `analysis/scripts/` provide:
- Mixed-effects regression models
- Age-of-acquisition (AoA) analysis using inflection point detection
- Pairwise comparisons across conditions
- Publication-ready figures and tables

```bash
# Run complete analysis pipeline
Rscript analysis/scripts/run_complete_analysis.R
```

## Model Architecture

The default configuration trains GPT-2 style causal language models:
- 12 layers, 768 hidden dimensions
- 12 attention heads, 3072 feed-forward dimensions
- ~125M parameters
- SentencePiece tokenizer (50k vocabulary)
- Trained on BabyLM corpus (10M-100M tokens)

## Documentation

Detailed documentation available in `/docs`:
- **[Model Foundry Architecture](docs/model_foundry/)** - Training framework design
- **[Logging System](docs/model_foundry/architecture/logging-system.md)** - Comprehensive logging
- **[Testing Strategy](docs/model_foundry/testing/)** - Unit and integration tests
- **[W&B Integration](docs/model_foundry/guides/wandb-integration.md)** - Experiment tracking

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU training)
- transformers, datasets, sentencepiece
- spaCy (with `en_core_web_trf` model)
- R 4.0+ (for analysis scripts)
- R packages: tidyverse, lme4, emmeans, targets, arrow

See `requirements.txt` for complete Python dependencies.

## Citation

If you use this codebase, please cite:

```bibtex
@misc{morton2024subjectdrop,
  title={Just Drop the Subject: A Controlled Rearing Study of Subject Drop Acquisition in English},
  author={Morton, Thomas},
  year={2024},
  note={GitHub repository},
  url={https://github.com/tgmorton/subject-drop}
}
```

## License

MIT License - See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.
