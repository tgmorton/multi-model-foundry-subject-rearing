# System Documentation

Technical reference for the data preparation, model training, and evaluation pipeline.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Corpus Ablation Pipeline](#3-corpus-ablation-pipeline)
4. [Model Training Framework](#4-model-training-framework)
5. [Evaluation Suite](#5-evaluation-suite)
6. [Pronoun Recovery Pipeline](#6-pronoun-recovery-pipeline)
7. [Corpus Descriptive Analysis](#7-corpus-descriptive-analysis)
8. [Configuration Reference](#8-configuration-reference)
9. [Kubernetes Deployment](#9-kubernetes-deployment)
10. [Review Interface](#10-review-interface)
11. [Command Reference](#11-command-reference)

---

## 1. Project Overview

This project investigates how linguistic input shapes grammatical knowledge in language models through controlled-rearing experiments. Models are trained on systematically ablated corpora — where specific linguistic features are removed or impoverished — and then evaluated on targeted syntactic benchmarks to isolate learning signals.

A secondary pipeline recovers dropped pronouns in Italian pro-drop text using cross-lingual alignment from English Europarl data, enabling controlled manipulation of null-subject input.

### Technology Stack

- Python 3.8+, PyTorch 2.0+, HuggingFace Transformers
- spaCy (en_core_web_trf, it_core_news_lg) for linguistic annotation
- SentencePiece for tokenization
- Weights & Biases for experiment tracking
- Kubernetes on NRP for distributed training
- R for statistical analysis

---

## 2. Repository Structure

```
.
├── preprocessing/              # Corpus ablation pipeline
│   ├── base.py                 # AblationPipeline base class
│   ├── config.py               # Config + provenance models
│   ├── registry.py             # Ablation function registry
│   ├── utils.py                # Device detection, checksums
│   ├── ablations/              # Ablation implementations
│   │   ├── remove_expletive_sentences.py
│   │   ├── impoverish_case.py
│   │   ├── enrich_verbal_morphology.py
│   │   └── ...
│   └── tests/
│
├── model_foundry/              # Core training framework
│   ├── cli.py                  # Typer CLI entry point
│   ├── config.py               # ExperimentConfig (Pydantic)
│   ├── trainer.py              # Trainer orchestrator
│   ├── data.py                 # DataProcessor + chunking
│   ├── data_collators.py       # Causal/Masked LM collators
│   ├── architectures/          # Model implementations
│   │   ├── base.py             # BaseLanguageModel
│   │   ├── gpt.py              # GPT-2 causal LM
│   │   ├── bert.py             # BERT masked LM
│   │   ├── rnn.py              # LSTM/GRU
│   │   └── mamba.py            # State space model
│   ├── training/
│   │   ├── loop.py             # Core training loop
│   │   ├── checkpointing.py    # Checkpoint scheduling
│   │   └── tokenization.py     # Tokenizer utilities
│   └── logging_utils.py        # Structured logging, WandB
│
├── evaluation/                 # Evaluation suite
│   ├── core/                   # Surprisal, model loading
│   ├── evaluators/             # BLIMP, null-subject, perplexity
│   ├── runners/                # Sequential + parallel runners
│   ├── aggregation/            # Result summarization
│   └── stimuli/                # BLIMP items, null-subject items
│
├── analysis/
│   ├── corpus_descriptives/    # Linguistic corpus analysis
│   │   ├── analyzers/          # Clause structure, pronouns, etc.
│   │   └── constants.py        # Linguistic constants
│   └── pronoun_recovery/       # Pronoun recovery pipeline
│       ├── run.py              # CLI entry point
│       ├── config.py           # Pipeline configs
│       ├── constants.py        # Label sets, pronoun maps
│       ├── synthetic_data/     # Pair generation
│       ├── annotation/         # LLM annotation (DeepSeek)
│       ├── model/              # Sequence labeler + trainer
│       ├── insertion/          # Pronoun insertion
│       ├── parallel_data/      # Europarl alignment
│       └── validation/         # Annotation validation
│
├── review/                     # Web review interface
│   ├── server.py               # FastAPI application
│   ├── corpus_api.py           # Query interface
│   └── sweep_dashboard.html    # Results dashboard
│
├── configs/
│   ├── experiments/            # Ablation experiment YAMLs
│   └── analysis/               # Pronoun recovery configs
│
├── k8s/                        # Kubernetes job definitions
├── scripts/                    # Utility scripts
├── data/                       # Corpora and datasets
├── models/                     # Trained checkpoints
└── tokenizers/                 # Experiment tokenizers
```

---

## 3. Corpus Ablation Pipeline

### Overview

The preprocessing pipeline applies linguistically-motivated ablations to a training corpus, producing modified versions that remove or impoverish specific grammatical features. Each ablated corpus is then used to train a language model, enabling causal inference about the role of those features in acquisition.

### Architecture

**`AblationPipeline`** (`preprocessing/base.py`) orchestrates the process:

1. Loads a spaCy model with device management (CPU/CUDA/MPS)
2. Discovers corpus files via glob pattern (`.train` files)
3. Processes files in configurable chunks (default 1000 lines)
4. Applies a registered ablation function to each spaCy `Doc`
5. Optionally validates that ablations occurred
6. Rebuilds the corpus to target size using a replacement pool
7. Records provenance metadata (checksums, statistics, environment)

**`AblationRegistry`** (`preprocessing/registry.py`) provides dynamic function lookup. Ablation functions register themselves at import time:

```python
@registry.register("remove_expletive_sentences_en")
def remove_expletive_sentences(doc, **params):
    ...
```

### Ablation Functions

| Ablation | File | Effect |
|----------|------|--------|
| `remove_expletive_sentences` | `remove_expletive_sentences.py` | Removes entire sentences containing expletive subjects (weather-it, raising-it, existentials). Three-tier detection: spaCy dep=expl, heuristic, optional coreference. English and Italian variants. |
| `impoverish_case` | `impoverish_case.py` | Reduces case system complexity |
| `enrich_verbal_morphology` | `enrich_verbal_morphology.py` | Adds verb morphological complexity |
| `remove_expletives` | `remove_expletives.py` | Removes expletive tokens (archived) |
| `remove_articles` | `remove_articles.py` | Removes articles (archived) |
| `remove_subject_pronominals` | `remove_subject_pronominals.py` | Removes overt subject pronouns (archived) |
| `impoverish_determiners` | `impoverish_determiners.py` | Replaces determiners with generic forms (archived) |
| `lemmatize_verbs` | `lemmatize_verbs.py` | Reduces verb morphology to lemmas (archived) |

### Configuration

Ablation config is embedded in experiment YAML files under `dataset_manipulation`:

```yaml
dataset_manipulation:
  - type: remove_expletive_sentences_en
    input_path: "data/raw/train_90M/"
    output_path: "data/processed/exp_remove_expletive_sentences_en/"
    spacy_model: "en_core_web_trf"
    parameters:
      chunk_size: 1000
      replacement_pool_dir: "data/raw/pull_10M/"
      skip_validation: false
```

### Provenance

Every ablation run produces a JSON manifest (`provenance.json`) recording:
- Input/output checksums (SHA256)
- Per-file statistics (lines processed, tokens removed, ablation rate)
- Environment metadata (Python version, spaCy model, device)
- Timestamp and random seed

---

## 4. Model Training Framework

### CLI

Entry point: `python -m model_foundry.cli`

The CLI (`model_foundry/cli.py`) uses Typer and provides these commands:

| Command | Description |
|---------|-------------|
| `preprocess` | Run ablation pipeline from config |
| `train-tokenizer` | Train SentencePiece tokenizer on corpus |
| `tokenize-dataset` | Tokenize corpus with trained tokenizer |
| `preprocess-data` | Chunk tokenized data for training |
| `generate-checkpoints` | Compute checkpoint schedule |
| `run` | Full training pipeline |
| `evaluate` | Run evaluation suite |

### Configuration

`ExperimentConfig` (`model_foundry/config.py`) is a Pydantic model combining:

```yaml
experiment_name: "exp0_baseline_model"

data:
  source_corpus: "data/raw/train_90M/"
  training_corpus: "data/raw/train_90M/"
  test_corpus: "data/raw/test_10M/"
  batch_size: 32
  max_sequence_length: 1000

tokenizer:
  output_dir: "tokenizers/exp0_baseline/"
  vocab_size: 50004

model:
  layers: 12
  embedding_size: 768
  hidden_size: 768
  intermediate_hidden_size: 3072
  attention_heads: 12
  activation_function: "gelu"
  dropout: 0.1
  attention_dropout: 0.1

training:
  output_dir: "models/exp0_baseline/"
  learning_rate: 0.0004
  epochs: 20
  use_amp: true
  gradient_accumulation_steps: 8
  max_grad_norm: 1.0
  use_tf32: true
  use_gradient_checkpointing: true
  use_flash_attention: true
  warmup_ratio: 0.1
  auto_generate_checkpoints: true

logging:
  use_wandb: true
  wandb_project: "just-drop-the-subject"

random_seed: 9
```

### Model Architectures

All architectures implement `BaseLanguageModel` (`model_foundry/architectures/base.py`):

| Architecture | File | Description |
|-------------|------|-------------|
| GPT-2 | `gpt.py` | Causal LM. HuggingFace `AutoModelForCausalLM` wrapper. Flash attention, gradient checkpointing. Primary architecture for all experiments. |
| BERT | `bert.py` | Masked LM. Bidirectional transformer with token classification support. |
| LSTM/GRU | `rnn.py` | Recurrent models. Embedding → recurrent layers → linear projection. Supports bidirectional. |
| Mamba | `mamba.py` | State space model. Linear-time complexity alternative to transformers. |

### Data Processing

**`DataProcessor`** (`model_foundry/data.py`):
- Validates tokenized datasets
- Creates PyTorch DataLoaders with custom collators
- Formats sequences into fixed-length chunks for language modeling
- Supports stride-based chunking with overlap

**Data Collators** (`model_foundry/data_collators.py`):
- `CausalLMDataCollator` — pads to batch max, labels = input_ids
- `MaskedLMDataCollator` — random 15% masking, special token handling

### Training Loop

The training system is split across several modules:

**`Trainer`** (`model_foundry/trainer.py`):
- Initializes model, optimizer, scheduler, data loader
- CUDA memory management (95% limit)
- Git commit hash tracking for reproducibility

**`TrainingLoop`** (`model_foundry/training/loop.py`):
- Mixed precision training (AMP)
- Gradient accumulation
- Progress tracking with tqdm
- Metrics logging every N steps
- Learning rate scheduling

**`CheckpointManager`** (`model_foundry/training/checkpointing.py`):
- Logarithmic schedule (denser checkpoints early in training)
- Linear schedule (even spacing)
- Configurable first-epoch density

### Logging

Structured logging with multiple backends:
- `StructuredLogger` — JSON-formatted with context
- `MetricsLogger` — JSONL metrics (one dict per line)
- `PerformanceLogger` — timing and profiling
- `WandBLogger` — Weights & Biases integration

---

## 5. Evaluation Suite

### Architecture

The evaluation system (`evaluation/`) runs trained checkpoints against linguistic benchmarks.

**Core modules** (`evaluation/core/`):
- `model_loader.py` — checkpoint + tokenizer loading with GPU memory management
- `surprisal_calculator.py` — word-by-word surprisal computation for critical regions
- `result_aggregator.py` — cross-checkpoint aggregation, CSV export

### Evaluators

| Evaluator | File | What It Tests |
|-----------|------|---------------|
| BLIMP | `blimp_evaluator.py` | 67 linguistic phenomena. Compares surprisal of grammatical vs. ungrammatical sentences. Per-phenomenon accuracy. |
| Null Subject | `null_subject_evaluator.py` | Overt vs. null subject preferences by person/number. Surprisal at critical positions. |
| Perplexity | `perplexity_evaluator.py` | Corpus perplexity overall and per domain. |

### Runners

- **`parallel_evaluation_runner.py`** — multi-GPU with threading. Recommended for production.
- **`evaluation_runner.py`** — single-threaded. For debugging.
- **`threaded_blimp_evaluator.py`** — threading-based BLIMP to avoid CUDA multiprocessing issues.

### Output

Results are aggregated into CSV files compatible with R for mixed-effects modeling. Learning curves are generated across checkpoints.

---

## 6. Pronoun Recovery Pipeline

### Purpose

Recovers dropped subject pronouns in Italian pro-drop text so that models can be trained on "un-dropped" Italian, enabling controlled comparison with English input.

### Architecture

The pipeline (`analysis/pronoun_recovery/`) has two tracks:

**Track A — English (synthetic):** Mechanically strip overt pronouns from English text, then train a model to recover them. Serves as a development/validation track.

**Track B — Italian (Europarl alignment):** Use parallel EN-IT Europarl data to identify where Italian drops a pronoun that English retains. This is the production track used for the sweep.

### CLI

Entry point: `python -m analysis.pronoun_recovery.run`

| Command | Step | Description |
|---------|------|-------------|
| `synthetic` | 1 | Generate synthetic pronoun-removal pairs from corpus |
| `sample` | 2a | Sample candidates for annotation |
| `seed-export` | 2b | Export seed candidates for manual annotation |
| `seed-import` | 2c | Import validated seed annotations |
| `annotate` | 3 | Run LLM annotation via DeepSeek |
| `validate` | 4 | Validate annotations against gold standard |
| `train` | 5 | Train sequence labeler |
| `insert` | 6 | Insert recovered pronouns into text |
| `train-seq2seq` | 5-alt | Train seq2seq model |
| `insert-seq2seq` | 6-alt | Insert via seq2seq |

### Label Set

7 labels defined in `constants.py`:

```
NONE | PRO.1sg | PRO.1pl | PRO.2sg | PRO.2pl | PRO.3sg | PRO.3pl
```

Mapped from spaCy morphological features (Person, Number) via `MORPH_TO_LABEL_SUFFIX`.

### Europarl Alignment Pipeline (Track B)

This is the primary data generation path for Italian, implemented in `analysis/pronoun_recovery/parallel_data/`:

```
Europarl EN ──→ spaCy en_core_web_trf ──→ Extract subject pronouns
                                           ├─ Filter: expletives, relatives, all "it"
                                           └─ Map: I→PRO.1sg, we→PRO.1pl, ...
                        ↕
                awesome-align (BERT word alignment)
                        ↕
Europarl IT ──→ spaCy it_core_news_lg ──→ Detect finite verbs (VerbForm=Fin)
                                           ├─ Subject status: overt/null/clausal
                                           └─ Morphology: Person, Number
                        ↓
                Label Resolution
                  ├─ Follow alignment: EN pronoun → IT verb
                  ├─ Walk dep tree (max 3 hops)
                  ├─ Skip if overt subject
                  ├─ Cross-check morphology
                  └─ Deduplicate (one marker per verb)
                        ↓
                Quality Filters
                  ├─ Pair-level: length, ratio, empty
                  └─ Pronoun-level: alignment, morph agree
                        ↓
                Passage Packing (~180 words/passage)
                        ↓
                packed_checkpoint.jsonl
```

Key modules:
- `en_pronoun_extractor.py` — English pronoun identification
- `it_null_subject_detector.py` — Italian null-subject detection
- `aligner.py` — awesome-align wrapper
- `label_resolver.py` — cross-lingual label mapping
- `quality_filters.py` — data filtering
- `passage_packer.py` — groups sentences into training passages

#### Output Format

```json
{
  "clean_text": "Dichiaro ripresa la sessione del Parlamento europeo.",
  "markers": [{"label": "PRO.1sg", "lexical_form": "io", "position": 0}],
  "id": "europarl_passage:42"
}
```

#### Configuration

```yaml
# pronoun_recovery_it_europarl_train_k8s.yaml
europarl_en_path: data/italian/europarl.en/europarl-v7.it-en.en.train
europarl_it_path: data/italian/europarl.en/europarl-v7.it-en.it.train
output_path: data/pronoun_recovery/europarl_aligned/it_train
language: it
start_line: 0
end_line: 500000
en_spacy_model: en_core_web_trf
it_spacy_model: it_core_news_lg
align_model: aneuraz/awesome-align-with-co
align_batch_size: 32
skip_all_it: true
min_tokens: 3
max_tokens: 128
max_length_ratio: 3.0
chunk_size: 1000
checkpoint_interval: 5000
pack_passages: true
max_passage_words: 180
```

### Sequence Labeler

**`PronounRecoveryModel`** (`model/sequence_labeler.py`):
- Wraps `AutoModelForTokenClassification` (mDeBERTa-v3-base)
- Handles subword → word mapping via `word_ids()`
- First subword of each word gets the label; rest get `-100`
- Optional threshold: predictions where `1 - P(NONE) < threshold` are forced to NONE
- Provides `predict(words)` and `predict_batch(word_lists)` interfaces

**`WeightedLossTrainer`** (`model/trainer.py`):
- Custom HuggingFace `Trainer` subclass
- Inverse-frequency class weights raised to `alpha` exponent
  - `alpha=0.0`: uniform weights (no reweighting)
  - `alpha=0.5`: square-root reweighting
  - `alpha=1.0`: full inverse-frequency reweighting
- Optional focal loss (`gamma > 0`): `FL(p_t) = -(1 - p_t)^gamma * CE(p_t)`
- Early stopping on `eval_f1` with configurable patience

**Training procedure:**

1. Load JSONL passages → tokenize with label alignment
2. Compute class weights from label frequency distribution
3. Fine-tune `microsoft/mdeberta-v3-base` (278M params) with:
   - fp16 mixed precision
   - warmup_ratio=0.1, weight_decay=0.01
   - Eval every epoch: seqeval P/R/F1 + detection_f1 + feature_accuracy
   - Early stopping (patience=3)
4. Save best checkpoint

**Evaluation metrics:**
- **seqeval P/R/F1** — full label match (detection + classification)
- **detection_f1** — binary PRO vs. NONE (ignores person/number)
- **feature_accuracy** — among correctly detected pronouns, what fraction have the right person/number label

### Pronoun Insertion

**`PronounInserter`** (`insertion/inserter.py`):
- Takes model predictions and inserts recovered pronouns into text
- Gender resolution for 3sg pronouns via coreference context
- Capitalization handling for sentence-initial positions
- Handles special labels (IMP, CONJ)

---

## 7. Corpus Descriptive Analysis

The `analysis/corpus_descriptives/` module provides single-pass analyzers for computing linguistic statistics over large corpora.

### Analyzer Framework

All analyzers inherit from `BaseAnalyzer` (`analyzers/base.py`):
- `process_doc(doc)` — process one spaCy Doc, update internal counters
- `get_results()` — return accumulated statistics
- `merge(other)` — combine results from parallel runs
- Checkpointing for distributed processing

### Available Analyzers

| Analyzer | What It Measures |
|----------|-----------------|
| `clause_structure.py` | Finite vs. infinitival clauses, subject realization (overt/expletive/null), xcomp chains |
| `expletives.py` | Expletive counts by class: weather, existential, raising |
| `pronoun_inventory.py` | Pronoun frequency by person/number and genre |
| `verb_finiteness.py` | Finite vs. non-finite verb distribution |
| `wh_questions.py` | Question structure, wh-word types, subject gaps |
| `relative_clauses.py` | Relative clause frequency and type |
| `that_trace.py` | That-trace effects, subject extraction constraints |
| `negation.py` | Negation placement and scope |

### Linguistic Constants

`constants.py` defines:
- Bridge verbs, weather verbs, raising verbs/adjectives (EN and IT)
- Wh-lemmas for question detection
- Italian impersonal verbs
- Genre mappings (CHILDES, BNC, Gutenberg, OpenSubtitles, etc.)

---

## 8. Configuration Reference

### Experiment Configs

Located in `configs/experiments/`. Each YAML defines a complete experiment:

| Config | Ablation |
|--------|----------|
| `experiment_0_baseline.yaml` | None (control) |
| `experiment_0_baseline_90M.yaml` | None, full 90M corpus |
| `experiment_1_remove_expletives.yaml` | Remove expletive tokens |
| `experiment_2_impoverish_determiners.yaml` | Reduce determiner system |
| `experiment_3_remove_articles.yaml` | Remove articles |
| `experiment_4_lemmatize_verbs.yaml` | Lemmatize verbs |
| `experiment_5_remove_subject_pronominals.yaml` | Remove overt subject pronouns |
| `experiment_6_impoverish_determiners_lemmatize_verbs.yaml` | Combined |
| `experiment_7_all_ablations.yaml` | All ablations combined |
| `experiment_en_remove_expletive_sentences.yaml` | Remove full sentences with expletives |
| `experiment_en_impoverish_case.yaml` | Impoverish English case |
| `experiment_en_enrich_verbal_morphology.yaml` | Enrich English verb morphology |
| `experiment_it_*` | Italian variants |

### Pronoun Recovery Configs

Located in `configs/analysis/pronoun_recovery/`:

| Config | Pipeline Step |
|--------|--------------|
| `pronoun_recovery_en_synthetic.yaml` | Generate EN synthetic pairs |
| `pronoun_recovery_en_annotate.yaml` | LLM annotation (EN) |
| `pronoun_recovery_en_train.yaml` | Train EN sequence labeler |
| `pronoun_recovery_en_insert.yaml` | Insert EN pronouns |
| `pronoun_recovery_it_europarl_align.yaml` | Europarl alignment (dev) |
| `pronoun_recovery_it_europarl_train_k8s.yaml` | Europarl alignment (500K, K8s) |
| `pronoun_recovery_it_europarl_test_k8s.yaml` | Europarl test split |
| `pronoun_recovery_it_train.yaml` | Train IT sequence labeler |
| `pronoun_recovery_it_insert.yaml` | Insert IT pronouns |
| `*_seq2seq.yaml` | Seq2seq variants |
| `*_validation_sample.yaml` | Validation sampling |

---

## 9. Kubernetes Deployment

Jobs are defined in `k8s/` for NRP cluster execution.

### Key Jobs

| Job | Purpose |
|-----|---------|
| `job-europarl-sweep.yaml` | 27-pod hyperparameter sweep (3 scales x 3 LRs x 3 alphas) |
| `job-sweep-r1.yaml` | Earlier sweep round |
| `job-train-90m.yaml` | Full 90M-token model training |
| `job-test-10m.yaml` | Quick 10M-token test run |
| `job-annotate-train-90m.yaml` | Annotation pipeline |

### Resource Configuration

- CUDA-enabled containers with `Dockerfile.sweep`
- Memory: 8-12Gi depending on job, with 1.2x request/limit ratio per NRP policy
- Volume mounts for data access
- Indexed job parallelism for sweeps

---

## 10. Review Interface

A FastAPI web application (`review/server.py`) for reviewing corpus annotations and sweep results.

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/metadata` | Corpus metadata |
| `GET /api/sentences` | Paginated sentence list with filtering |
| `GET /api/sentences/{id}` | Sentence detail |
| `POST /api/notes/{id}` | Save annotation note |
| `GET /api/notes` | Query notes |
| `GET /api/schematics` | Feature definitions |

### Dashboard

`review/sweep_dashboard.html` — self-contained HTML dashboard visualizing:
- Europarl annotation pipeline
- mDeBERTa fine-tuning procedure
- Classifier head architecture
- 27-run sweep results (sortable table + SVG charts)
- Next steps analysis

---

## 11. Command Reference

### Full Experiment Pipeline

```bash
# 1. Preprocess corpus (apply ablation)
python -m model_foundry.cli preprocess configs/experiments/experiment_1.yaml

# 2. Train tokenizer
python -m model_foundry.cli train-tokenizer configs/experiments/experiment_1.yaml

# 3. Tokenize dataset
python -m model_foundry.cli tokenize-dataset configs/experiments/experiment_1.yaml

# 4. Preprocess into chunks
python -m model_foundry.cli preprocess-data configs/experiments/experiment_1.yaml

# 5. Train model
python -m model_foundry.cli run configs/experiments/experiment_1.yaml

# 6. Evaluate
python -m model_foundry.cli evaluate configs/experiments/experiment_1.yaml
```

### Pronoun Recovery Pipeline (Italian Europarl)

```bash
# 1. Generate aligned training data from Europarl
python -m analysis.pronoun_recovery.run europarl-align \
  --config configs/analysis/pronoun_recovery/pronoun_recovery_it_europarl_train_k8s.yaml

# 2. Train sequence labeler
python -m analysis.pronoun_recovery.run train \
  --config configs/analysis/pronoun_recovery/pronoun_recovery_it_train.yaml

# 3. Insert recovered pronouns
python -m analysis.pronoun_recovery.run insert \
  --config configs/analysis/pronoun_recovery/pronoun_recovery_it_insert.yaml
```

### Pronoun Recovery Pipeline (English Synthetic)

```bash
# 1. Generate synthetic pairs
python -m analysis.pronoun_recovery.run synthetic \
  --config configs/analysis/pronoun_recovery/pronoun_recovery_en_synthetic.yaml

# 2. Sample and annotate
python -m analysis.pronoun_recovery.run sample \
  --config configs/analysis/pronoun_recovery/pronoun_recovery_en_annotate.yaml
python -m analysis.pronoun_recovery.run annotate \
  --config configs/analysis/pronoun_recovery/pronoun_recovery_en_annotate.yaml

# 3. Train
python -m analysis.pronoun_recovery.run train \
  --config configs/analysis/pronoun_recovery/pronoun_recovery_en_train.yaml

# 4. Insert
python -m analysis.pronoun_recovery.run insert \
  --config configs/analysis/pronoun_recovery/pronoun_recovery_en_insert.yaml
```

### Review Interface

```bash
python -m review     # Starts FastAPI at localhost:8642
```

### Corpus Analysis

```bash
python -m analysis.corpus_descriptives.run \
  --analyzer clause_structure \
  --corpus data/raw/train_90M/ \
  --spacy-model en_core_web_trf
```

---

## Data Flow Summary

```
Raw Corpus (90M tokens)
    │
    ├─── Ablation Pipeline ──→ Ablated Corpus ──→ Tokenizer ──→ Training ──→ Evaluation
    │         │                                                                  │
    │    Provenance JSON                                                   BLIMP / Null-Subject
    │                                                                      Perplexity / R Export
    │
    └─── Corpus Descriptives ──→ Statistics (clause structure, pronouns, etc.)


Europarl EN + IT
    │
    ├─── Alignment Pipeline ──→ packed_checkpoint.jsonl ──→ Labeler Training ──→ Sweep
    │                                                           │
    │                                                    WeightedLossTrainer
    │                                                    mDeBERTa-v3-base
    │
    └─── Trained Model ──→ Pronoun Insertion ──→ Recovered Italian Text ──→ Model Training
```
