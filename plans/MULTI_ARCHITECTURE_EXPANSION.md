# Multi-Architecture Model Foundry Expansion

**Purpose:** Expand model_foundry to support RNN, LSTM, and bidirectional models (BERT-style) alongside existing GPT-2 causal models

**Date:** 2025-09-30

---

## Executive Summary

Currently, `model_foundry` is tightly coupled to GPT-2 causal language modeling. Expanding to support RNN/LSTM and bidirectional transformers requires:

1. **Architecture abstraction layer** - Decouple model creation from GPT-2 specifics
2. **Training objective abstraction** - Support causal LM, masked LM, and sequence-to-sequence
3. **Tokenization strategy expansion** - Add WordPiece, BPE variants beyond current SentencePiece
4. **Data processing changes** - Support different masking strategies and input formats
5. **Configuration schema updates** - Specify model family and training objectives

**Key Insight:** The existing codebase structure is well-designed for extension. We can maintain backwards compatibility while adding new model families through careful abstraction.

---

## Current Architecture Analysis

### Current Model Creation (`model_foundry/model.py`)

**What it does:**
- Hardcoded to GPT-2 via `AutoConfig.from_pretrained("gpt2")`
- Creates causal language models only
- Maps config parameters to GPT-2 hyperparameters

**Limitations:**
- No support for encoder-only models (BERT)
- No support for encoder-decoder models (T5)
- No support for RNN/LSTM architectures
- Assumes causal attention masks

### Current Configuration (`model_foundry/config.py`)

**What it has:**
- `ModelConfig` with transformer-specific parameters:
  - `layers`, `embedding_size`, `attention_heads`
  - `activation_function`, `dropout`

**What it lacks:**
- Model architecture type specification
- RNN-specific parameters (hidden_dim, num_layers, bidirectional)
- Training objective specification (causal LM vs masked LM vs seq2seq)
- Model family indicator

### Current Tokenization (`model_foundry/tokenizer/train_tokenizer.py`)

**What it does:**
- Uses SentencePiece unigram model
- Converts to HuggingFace `PreTrainedTokenizerFast`
- Hardcoded special tokens: `<s>`, `</s>`, `<unk>`, `<pad>`

**What it lacks:**
- WordPiece tokenization (required for BERT)
- Different special token schemes (e.g., BERT's `[CLS]`, `[SEP]`, `[MASK]`)
- Tokenizer type selection based on model architecture

### Current Training Loop (`model_foundry/training/loop.py`)

**What it does:**
- Causal language modeling objective only
- Computes cross-entropy loss on next-token prediction
- Works with autoregressive generation

**What it lacks:**
- Masked language modeling (MLM) objective
- Next sentence prediction (NSP) objective
- Sequence-to-sequence loss computation
- Support for encoder-decoder attention

---

## Proposed Architecture: Unified Model Foundry

### Design Principles

1. **Model Family Abstraction:** Define clear interfaces for different model families
2. **Backwards Compatibility:** Existing GPT-2 configs and code continue to work
3. **Minimal Code Duplication:** Share components where possible (optimizer, checkpointing, logging)
4. **Configuration-Driven:** Model architecture selected via config, not code changes
5. **HuggingFace Integration:** Leverage transformers library where possible, custom implementations where needed

---

## Component 1: Model Architecture Registry

### New Module: `model_foundry/architectures/__init__.py`

**Purpose:** Registry pattern for model creation

**Structure:**

```
model_foundry/architectures/
├── __init__.py           # Registry and factory
├── base.py               # Abstract base classes
├── gpt.py                # GPT-2 and causal transformers
├── bert.py               # BERT and masked LM transformers
├── rnn.py                # RNN/LSTM/GRU models
├── encoder_decoder.py    # T5-style models (future)
└── utils.py              # Shared utilities
```

### Abstract Base Class (`base.py`)

**Key Concept:** All model architectures implement a common interface

**Interface Requirements:**

```python
# Pseudo-code showing required interface

class BaseLanguageModel(ABC):
    """Base class for all language models in foundry."""

    @abstractmethod
    def __init__(self, config: ModelArchitectureConfig):
        """Initialize model from config."""
        pass

    @abstractmethod
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass with standardized inputs.

        Returns:
            ModelOutput with:
            - loss (if labels provided)
            - logits
            - hidden_states (optional)
            - attentions (optional)
        """
        pass

    @abstractmethod
    def get_input_embeddings(self):
        """Return input embedding layer."""
        pass

    @abstractmethod
    def resize_token_embeddings(self, new_num_tokens):
        """Resize vocabulary."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier (gpt2, bert, lstm, etc.)."""
        pass

    @property
    @abstractmethod
    def supports_generation(self) -> bool:
        """Whether model supports autoregressive generation."""
        pass
```

### Model Registry (`architectures/__init__.py`)

**Pattern:** Factory with registration

**Pseudo-code:**

```python
# Registry for model architectures
MODEL_REGISTRY = {}

def register_architecture(name: str):
    """Decorator to register model architectures."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def create_model_from_config(config: ExperimentConfig, **kwargs):
    """
    Factory function to create models based on config.

    Replaces current create_model() in model.py
    """
    architecture = config.model.architecture  # New field in config

    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[architecture]
    return model_class.from_config(config, **kwargs)
```

---

## Component 2: Architecture Implementations

### GPT-2 Implementation (`architectures/gpt.py`)

**Purpose:** Wrap existing GPT-2 logic

**Key Changes:**
- Move current `model.py` logic here
- Implement `BaseLanguageModel` interface
- Maintain backwards compatibility

**Pseudo-code:**

```python
@register_architecture("gpt2")
class GPT2Model(BaseLanguageModel):
    """GPT-2 causal language model."""

    def __init__(self, config: ExperimentConfig, **kwargs):
        # Existing logic from model.py
        self.hf_model = AutoModelForCausalLM.from_config(...)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Delegate to HuggingFace model
        return self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    @property
    def model_type(self) -> str:
        return "gpt2"

    @property
    def supports_generation(self) -> bool:
        return True

    @classmethod
    def from_config(cls, config: ExperimentConfig, **kwargs):
        """Create GPT-2 model from experiment config."""
        return cls(config, **kwargs)
```

### BERT Implementation (`architectures/bert.py`)

**Purpose:** Bidirectional transformer for masked language modeling

**Key Features:**
- Uses `AutoModelForMaskedLM` from HuggingFace
- Supports MLM training objective
- No causal masking (bidirectional attention)

**Pseudo-code:**

```python
@register_architecture("bert")
class BERTModel(BaseLanguageModel):
    """BERT masked language model."""

    def __init__(self, config: ExperimentConfig, **kwargs):
        bert_config = AutoConfig.from_pretrained("bert-base-uncased")

        # Map config to BERT parameters
        bert_config.num_hidden_layers = config.model.layers
        bert_config.hidden_size = config.model.hidden_size
        bert_config.num_attention_heads = config.model.attention_heads
        bert_config.intermediate_size = config.model.intermediate_hidden_size
        bert_config.vocab_size = config.tokenizer.vocab_size
        bert_config.max_position_embeddings = config.data.max_sequence_length

        # Additional BERT-specific parameters from config
        bert_config.type_vocab_size = config.model.bert.get('type_vocab_size', 2)

        self.hf_model = AutoModelForMaskedLM.from_config(bert_config, **kwargs)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass with masked language modeling.

        Labels should have -100 for non-masked tokens (HuggingFace convention).
        """
        return self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    @property
    def model_type(self) -> str:
        return "bert"

    @property
    def supports_generation(self) -> bool:
        return False  # BERT doesn't support autoregressive generation
```

### RNN/LSTM Implementation (`architectures/rnn.py`)

**Purpose:** Custom RNN/LSTM models for language modeling

**Why Custom:** HuggingFace doesn't provide standard RNN implementations; need custom code

**Architecture Design:**

```python
@register_architecture("lstm")
class LSTMLanguageModel(BaseLanguageModel):
    """LSTM-based language model."""

    def __init__(self, config: ExperimentConfig, **kwargs):
        import torch.nn as nn

        # RNN-specific config parameters
        rnn_config = config.model.rnn

        self.embedding = nn.Embedding(
            num_embeddings=config.tokenizer.vocab_size,
            embedding_dim=config.model.embedding_size,
            padding_idx=0  # Assuming 0 is pad token
        )

        self.lstm = nn.LSTM(
            input_size=config.model.embedding_size,
            hidden_size=rnn_config.hidden_size,
            num_layers=rnn_config.num_layers,
            dropout=config.model.dropout if rnn_config.num_layers > 1 else 0,
            bidirectional=rnn_config.bidirectional,
            batch_first=True
        )

        # Output projection
        lstm_output_size = rnn_config.hidden_size * (2 if rnn_config.bidirectional else 1)
        self.output_projection = nn.Linear(lstm_output_size, config.tokenizer.vocab_size)

        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass for LSTM language model.

        Returns:
            ModelOutput compatible with transformers library
        """
        # Embed input
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embeddings)

        # Project to vocabulary
        logits = self.output_projection(lstm_out)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Shift for language modeling (predict next token)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))

        return ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=lstm_out,
            attentions=None  # RNNs don't have attention
        )

    @property
    def model_type(self) -> str:
        return "lstm"

    @property
    def supports_generation(self) -> bool:
        return True  # Can generate autoregressively

class ModelOutput:
    """Standardized output format."""
    def __init__(self, loss=None, logits=None, hidden_states=None, attentions=None):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions
```

**Similar Implementation for:**
- `@register_architecture("rnn")` - vanilla RNN
- `@register_architecture("gru")` - GRU variant

---

## Component 3: Configuration Schema Updates

### Extended `ModelConfig` (`config.py`)

**Current schema only supports transformers. Need to add architecture-specific configs.**

**New Structure:**

```python
from typing import Optional, Literal, Union

class TransformerModelConfig(BaseModel):
    """Configuration for transformer-based models (GPT-2, BERT)."""
    layers: int = Field(..., gt=0)
    embedding_size: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    intermediate_hidden_size: int = Field(..., gt=0)
    attention_heads: int = Field(..., gt=0)
    activation_function: str = "gelu"
    dropout: float = Field(..., ge=0.0, lt=1.0)
    attention_dropout: float = Field(..., ge=0.0, lt=1.0)

class BERTSpecificConfig(BaseModel):
    """Additional BERT-specific parameters."""
    type_vocab_size: int = Field(2, description="Number of token type IDs (for segment embeddings)")
    pooler_type: str = Field("first", description="How to pool sequence for classification")

class RNNModelConfig(BaseModel):
    """Configuration for RNN/LSTM/GRU models."""
    embedding_size: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    bidirectional: bool = Field(False, description="Whether to use bidirectional RNN")
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    rnn_type: Literal["rnn", "lstm", "gru"] = Field("lstm")

class ModelConfig(BaseModel):
    """Unified model configuration supporting multiple architectures."""

    # Required: Architecture type
    architecture: Literal["gpt2", "bert", "lstm", "rnn", "gru"] = Field(
        ...,
        description="Model architecture family"
    )

    # Architecture-specific configs (one must be provided based on architecture)
    transformer: Optional[TransformerModelConfig] = None
    bert: Optional[BERTSpecificConfig] = None
    rnn: Optional[RNNModelConfig] = None

    def __init__(self, **data):
        super().__init__(**data)

        # Validate that appropriate config is provided for architecture
        if self.architecture in ["gpt2", "bert"]:
            if self.transformer is None:
                raise ValueError(f"Architecture '{self.architecture}' requires transformer config")
        elif self.architecture in ["lstm", "rnn", "gru"]:
            if self.rnn is None:
                raise ValueError(f"Architecture '{self.architecture}' requires rnn config")

    # Convenience properties for backwards compatibility
    @property
    def layers(self) -> int:
        """Get number of layers (works for both transformers and RNNs)."""
        if self.transformer:
            return self.transformer.layers
        elif self.rnn:
            return self.rnn.num_layers
        raise ValueError("No architecture config provided")

    @property
    def embedding_size(self) -> int:
        """Get embedding size."""
        if self.transformer:
            return self.transformer.embedding_size
        elif self.rnn:
            return self.rnn.embedding_size
        raise ValueError("No architecture config provided")

    # ... similar properties for other shared parameters
```

### Training Objective Configuration

**Add to `TrainingConfig`:**

```python
class TrainingConfig(BaseModel):
    # ... existing fields ...

    # New field: Training objective
    objective: Literal["causal_lm", "masked_lm", "seq2seq"] = Field(
        "causal_lm",
        description="Training objective type"
    )

    # Masked LM specific parameters
    mlm_probability: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        description="Probability of masking tokens for MLM"
    )

    # Whether to include next sentence prediction (BERT-style)
    use_nsp: bool = Field(False, description="Include next sentence prediction loss")
```

---

## Component 4: Tokenization Strategy Expansion

### New Module: `model_foundry/tokenizer/tokenizer_factory.py`

**Purpose:** Support multiple tokenization algorithms

**Supported Tokenizers:**
1. **SentencePiece (Unigram)** - Current default, good for many languages
2. **WordPiece** - Used by BERT, better for English
3. **BPE** - Used by GPT models, subword tokenization
4. **CharacterLevel** - Fallback for small vocabularies

### Tokenizer Configuration Updates

**Extend `TokenizerConfig`:**

```python
class TokenizerConfig(BaseModel):
    output_dir: str
    vocab_size: int = Field(..., gt=0)

    # New: Tokenizer type
    tokenizer_type: Literal["sentencepiece", "wordpiece", "bpe", "character"] = Field(
        "sentencepiece",
        description="Tokenization algorithm"
    )

    # Special tokens (architecture-specific)
    special_tokens: Dict[str, str] = Field(
        default_factory=lambda: {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>"
        },
        description="Special tokens for tokenizer"
    )

    # WordPiece specific
    wordpiece_prefix: str = Field("##", description="Subword prefix for WordPiece")

    # BPE specific
    bpe_merges: int = Field(10000, description="Number of BPE merge operations")
```

### Special Token Schemes by Architecture

**GPT-2 (Causal LM):**
```yaml
special_tokens:
  bos_token: "<s>"
  eos_token: "</s>"
  unk_token: "<unk>"
  pad_token: "<pad>"
```

**BERT (Masked LM):**
```yaml
special_tokens:
  cls_token: "[CLS]"
  sep_token: "[SEP]"
  mask_token: "[MASK]"
  unk_token: "[UNK]"
  pad_token: "[PAD]"
```

**RNN/LSTM (Flexible):**
```yaml
special_tokens:
  bos_token: "<BOS>"
  eos_token: "<EOS>"
  unk_token: "<UNK>"
  pad_token: "<PAD>"
```

### Tokenizer Training Implementation

**Refactor `train_tokenizer.py`:**

**Current:** Single function hardcoded to SentencePiece

**New:** Factory pattern with multiple trainers

**Pseudo-code:**

```python
def train_tokenizer_from_config(config_path: str):
    """Main entry point - delegates to specific trainer."""

    config = load_config(config_path)
    tokenizer_type = config.tokenizer.tokenizer_type

    if tokenizer_type == "sentencepiece":
        return train_sentencepiece_tokenizer(config)
    elif tokenizer_type == "wordpiece":
        return train_wordpiece_tokenizer(config)
    elif tokenizer_type == "bpe":
        return train_bpe_tokenizer(config)
    elif tokenizer_type == "character":
        return train_character_tokenizer(config)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

def train_wordpiece_tokenizer(config: ExperimentConfig):
    """Train WordPiece tokenizer (BERT-style)."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

    # Initialize WordPiece model
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    # Set pre-tokenizer (whitespace + punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    # Set post-processor for [CLS] and [SEP]
    tokenizer.post_processor = processors.BertProcessing(
        sep=("[SEP]", tokenizer.token_to_id("[SEP]")),
        cls=("[CLS]", tokenizer.token_to_id("[CLS]"))
    )

    # Configure trainer
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(
        vocab_size=config.tokenizer.vocab_size,
        special_tokens=special_tokens,
        min_frequency=2
    )

    # Train on corpus
    files = get_training_files(config.data.training_corpus)
    tokenizer.train(files, trainer)

    # Wrap in HuggingFace tokenizer
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]"
    )

    # Save
    hf_tokenizer.save_pretrained(config.tokenizer.output_dir)

def train_bpe_tokenizer(config: ExperimentConfig):
    """Train BPE tokenizer (GPT-style)."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Set pre-tokenizer (byte-level for GPT)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Configure trainer
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]
    trainer = trainers.BpeTrainer(
        vocab_size=config.tokenizer.vocab_size,
        special_tokens=special_tokens,
        min_frequency=2
    )

    # Train
    files = get_training_files(config.data.training_corpus)
    tokenizer.train(files, trainer)

    # Wrap and save
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>"
    )
    hf_tokenizer.save_pretrained(config.tokenizer.output_dir)
```

---

## Component 5: Data Processing Updates

### Masked Language Modeling Data Collator

**Current:** Data processor creates causal LM inputs only

**New:** Need data collator for masked LM (BERT-style training)

**New File: `model_foundry/data_collators.py`**

```python
from dataclasses import dataclass
from typing import Dict, List
import torch
from transformers import DataCollatorForLanguageModeling

@dataclass
class CausalLMDataCollator:
    """Data collator for causal language modeling (GPT-2 style)."""

    tokenizer: Any
    max_length: int

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate examples for causal LM.

        No masking needed - labels are just input_ids shifted by 1.
        """
        # Stack input_ids
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attention_mask = torch.stack([ex['attention_mask'] for ex in examples])

        # Labels are the same as input_ids (shifting happens in model)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

@dataclass
class MaskedLMDataCollator:
    """Data collator for masked language modeling (BERT style)."""

    tokenizer: Any
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate examples for masked LM.

        Randomly masks mlm_probability of tokens.
        """
        # Stack inputs
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attention_mask = torch.stack([ex['attention_mask'] for ex in examples])

        # Create labels (copy of input_ids)
        labels = input_ids.clone()

        # Create random mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        for special_token_id in [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id
        ]:
            special_tokens_mask |= (labels == special_token_id)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Sample tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set non-masked tokens to -100 (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of time: replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of time: replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 10% of time: keep original (remaining masked_indices)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def get_data_collator(config: ExperimentConfig, tokenizer):
    """Factory function to get appropriate data collator."""

    if config.training.objective == "causal_lm":
        return CausalLMDataCollator(
            tokenizer=tokenizer,
            max_length=config.data.max_sequence_length
        )
    elif config.training.objective == "masked_lm":
        return MaskedLMDataCollator(
            tokenizer=tokenizer,
            mlm_probability=config.training.mlm_probability
        )
    else:
        raise ValueError(f"Unknown training objective: {config.training.objective}")
```

### Update `model_foundry/data.py`

**Current:** Returns DataLoader with simple batching

**New:** Use appropriate data collator based on training objective

**Changes:**

```python
def create_dataloader(self, tokenizer):
    """Create DataLoader with appropriate collation."""

    from .data_collators import get_data_collator

    # Get data collator based on training objective
    collator = get_data_collator(self.config, tokenizer)

    return DataLoader(
        self.dataset,
        batch_size=self.config.data.batch_size,
        shuffle=True,
        collate_fn=collator,  # Use custom collator
        num_workers=4,
        pin_memory=True
    )
```

---

## Component 6: Training Loop Updates

### Loss Computation Abstraction

**Current:** Training loop assumes causal LM loss

**New:** Support different loss computation strategies

**Update `model_foundry/training/loop.py`:**

**Changes:**

```python
def _compute_loss(self, batch):
    """
    Compute loss for a batch.

    Delegates to model's forward pass, which returns loss.
    """
    # All models now return ModelOutput with loss
    outputs = self.model(**batch)

    return outputs.loss

# In training loop:
for batch in dataloader:
    # Move to device
    batch = {k: v.to(self.device) for k, v in batch.items()}

    # Forward pass (model computes appropriate loss internally)
    loss = self._compute_loss(batch)

    # Backward pass (same for all architectures)
    if self.amp_enabled:
        self.scaler.scale(loss).backward()
    else:
        loss.backward()

    # ... rest of training loop unchanged
```

**Key Insight:** By standardizing the model interface (all models return `ModelOutput` with `loss`), the training loop doesn't need to know about different objectives.

---

## Component 7: Example Configurations

### Example 1: GPT-2 (Existing, Backwards Compatible)

**File:** `configs/experiment_gpt2_baseline.yaml`

```yaml
experiment_name: "gpt2_baseline"

data:
  source_corpus: "data/raw/train_90M/"
  training_corpus: "data/raw/train_90M/"
  test_corpus: "data/raw/test_10M/"
  batch_size: 32
  max_sequence_length: 1000

tokenizer:
  output_dir: "tokenizers/gpt2_baseline/"
  vocab_size: 50004
  tokenizer_type: "sentencepiece"  # Or "bpe" for GPT-style
  special_tokens:
    bos_token: "<s>"
    eos_token: "</s>"
    unk_token: "<unk>"
    pad_token: "<pad>"

model:
  architecture: "gpt2"
  transformer:
    layers: 12
    embedding_size: 768
    hidden_size: 768
    intermediate_hidden_size: 3072
    attention_heads: 12
    activation_function: "gelu"
    dropout: 0.1
    attention_dropout: 0.1

training:
  output_dir: "models/gpt2_baseline/"
  objective: "causal_lm"
  learning_rate: 0.0004
  epochs: 20
  # ... rest of training config

logging:
  use_wandb: true
  wandb_project: "multi-architecture-lm"

random_seed: 42
```

### Example 2: BERT

**File:** `configs/experiment_bert_masked.yaml`

```yaml
experiment_name: "bert_masked_lm"

data:
  source_corpus: "data/raw/train_90M/"
  training_corpus: "data/raw/train_90M/"
  test_corpus: "data/raw/test_10M/"
  batch_size: 32
  max_sequence_length: 512  # BERT typically uses 512

tokenizer:
  output_dir: "tokenizers/bert_masked/"
  vocab_size: 30000
  tokenizer_type: "wordpiece"
  special_tokens:
    cls_token: "[CLS]"
    sep_token: "[SEP]"
    mask_token: "[MASK]"
    unk_token: "[UNK]"
    pad_token: "[PAD]"
  wordpiece_prefix: "##"

model:
  architecture: "bert"
  transformer:
    layers: 12
    embedding_size: 768
    hidden_size: 768
    intermediate_hidden_size: 3072
    attention_heads: 12
    activation_function: "gelu"
    dropout: 0.1
    attention_dropout: 0.1
  bert:
    type_vocab_size: 2  # For segment embeddings

training:
  output_dir: "models/bert_masked/"
  objective: "masked_lm"
  mlm_probability: 0.15  # Mask 15% of tokens
  learning_rate: 0.0001  # BERT uses lower LR
  epochs: 10
  # ... rest of training config

logging:
  use_wandb: true
  wandb_project: "multi-architecture-lm"

random_seed: 42
```

### Example 3: Bidirectional LSTM

**File:** `configs/experiment_lstm_bidirectional.yaml`

```yaml
experiment_name: "lstm_bidirectional"

data:
  source_corpus: "data/raw/train_90M/"
  training_corpus: "data/raw/train_90M/"
  test_corpus: "data/raw/test_10M/"
  batch_size: 64  # LSTMs can use larger batches (less memory than transformers)
  max_sequence_length: 512

tokenizer:
  output_dir: "tokenizers/lstm_bidirectional/"
  vocab_size: 30000
  tokenizer_type: "bpe"  # Can use any tokenizer with RNNs
  special_tokens:
    bos_token: "<BOS>"
    eos_token: "<EOS>"
    unk_token: "<UNK>"
    pad_token: "<PAD>"

model:
  architecture: "lstm"
  rnn:
    embedding_size: 512
    hidden_size: 512
    num_layers: 3
    bidirectional: true  # Bidirectional LSTM
    dropout: 0.3  # RNNs benefit from higher dropout
    rnn_type: "lstm"

training:
  output_dir: "models/lstm_bidirectional/"
  objective: "masked_lm"  # Can use MLM with bidirectional RNN
  mlm_probability: 0.15
  learning_rate: 0.001  # RNNs can use higher LR
  epochs: 20
  gradient_accumulation_steps: 4
  max_grad_norm: 5.0  # Important for RNN stability

logging:
  use_wandb: true
  wandb_project: "multi-architecture-lm"

random_seed: 42
```

### Example 4: Unidirectional LSTM (Causal)

**File:** `configs/experiment_lstm_causal.yaml`

```yaml
experiment_name: "lstm_causal"

data:
  source_corpus: "data/raw/train_90M/"
  training_corpus: "data/raw/train_90M/"
  batch_size: 64
  max_sequence_length: 512

tokenizer:
  output_dir: "tokenizers/lstm_causal/"
  vocab_size: 30000
  tokenizer_type: "bpe"

model:
  architecture: "lstm"
  rnn:
    embedding_size: 512
    hidden_size: 512
    num_layers: 3
    bidirectional: false  # Unidirectional for causal LM
    dropout: 0.3
    rnn_type: "lstm"

training:
  output_dir: "models/lstm_causal/"
  objective: "causal_lm"  # Next-token prediction
  learning_rate: 0.001
  epochs: 20
  max_grad_norm: 5.0

logging:
  use_wandb: true
  wandb_project: "multi-architecture-lm"

random_seed: 42
```

---

## Implementation Roadmap

### Phase 1: Architecture Abstraction (Week 1-2)

**Tasks:**
1. Create `model_foundry/architectures/` module structure
2. Implement `BaseLanguageModel` abstract class
3. Implement model registry pattern
4. Migrate GPT-2 logic to `architectures/gpt.py`
5. Update `model_foundry/model.py` to use factory

**Testing:**
- Ensure existing GPT-2 configs still work
- No breaking changes to existing code

### Phase 2: Configuration Schema (Week 2)

**Tasks:**
1. Update `ModelConfig` to support multiple architectures
2. Add `architecture` field
3. Add architecture-specific config models
4. Update `TrainingConfig` with `objective` field
5. Update YAML schema validation

**Testing:**
- Validate new config formats
- Ensure backwards compatibility with old configs

### Phase 3: BERT Implementation (Week 3)

**Tasks:**
1. Implement `BERTModel` in `architectures/bert.py`
2. Implement `MaskedLMDataCollator`
3. Create example BERT config
4. Test MLM training loop

**Testing:**
- Train small BERT model
- Verify loss decreases
- Compare with HuggingFace BERT training

### Phase 4: RNN/LSTM Implementation (Week 4)

**Tasks:**
1. Implement `LSTMLanguageModel` in `architectures/rnn.py`
2. Implement unidirectional and bidirectional variants
3. Create example LSTM configs
4. Test both causal and masked LM objectives with LSTMs

**Testing:**
- Train unidirectional LSTM (causal)
- Train bidirectional LSTM (masked)
- Verify gradient flow and convergence

### Phase 5: Tokenizer Expansion (Week 5)

**Tasks:**
1. Implement `tokenizer_factory.py`
2. Implement WordPiece trainer
3. Implement BPE trainer
4. Update `train_tokenizer.py` to use factory
5. Create tokenizer type tests

**Testing:**
- Train WordPiece tokenizer (BERT-style)
- Train BPE tokenizer (GPT-style)
- Verify special tokens handled correctly

### Phase 6: Integration and Testing (Week 6)

**Tasks:**
1. End-to-end testing with all architectures
2. Performance benchmarking
3. Documentation updates
4. Create migration guide

**Testing:**
- Train GPT-2, BERT, and LSTM on same corpus
- Compare performance and convergence
- Verify evaluation compatibility

---

## Evaluation Compatibility

### Updates Needed for `evaluation/`

**Current:** Evaluation assumes causal LM (next-token prediction)

**BLIMP Evaluation:**
- Works for causal models (GPT-2, unidirectional LSTM)
- Needs adaptation for bidirectional models (BERT, bidirectional LSTM)
- BERT: Use pseudo-perplexity (mask each token, compute likelihood)

**Null-Subject Evaluation:**
- Primarily designed for causal models
- Can adapt for bidirectional: compute probability of masked position

**New Evaluation Utilities Needed:**

```python
# evaluation/core/model_evaluator.py

def evaluate_causal_model(model, tokenizer, sequence):
    """Evaluate causal LM (GPT-2, unidirectional LSTM)."""
    # Existing logic
    pass

def evaluate_masked_model(model, tokenizer, sequence, target_position):
    """Evaluate masked LM (BERT, bidirectional LSTM)."""
    # Mask target position, get logits
    pass

def get_evaluator_for_model(model):
    """Factory to get appropriate evaluator."""
    if model.supports_generation:
        return CausalLMEvaluator(model)
    else:
        return MaskedLMEvaluator(model)
```

---

## Migration Strategy for Existing Code

### Option 1: Gradual Migration (Recommended)

**Week 1-2:** Implement architecture abstraction, keep existing code working

**Week 3-4:** Add BERT and LSTM, test alongside GPT-2

**Week 5-6:** Expand tokenization, full integration

**Ongoing:** Maintain backwards compatibility for existing experiments

### Option 2: Parallel Implementation

**Create `model_foundry_v2/` module**
- Implement all new architectures
- Test thoroughly
- Once stable, replace `model_foundry/` or merge

### Backwards Compatibility Strategy

**Config Migration:**
- Old configs without `architecture` field default to `"gpt2"`
- Automatically infer from model parameters if possible
- Warn users to update configs

**Code Detection:**

```python
def infer_architecture_from_config(config: dict) -> str:
    """Infer architecture from legacy config format."""

    if 'architecture' in config.get('model', {}):
        return config['model']['architecture']

    # Check for RNN-specific fields
    if 'rnn' in config.get('model', {}):
        return config['model']['rnn'].get('rnn_type', 'lstm')

    # Check for BERT-specific fields
    if 'bert' in config.get('model', {}):
        return 'bert'

    # Default to GPT-2 for backwards compatibility
    return 'gpt2'
```

---

## File Structure Summary

### New Files to Create

```
model_foundry/
├── architectures/
│   ├── __init__.py           # Registry and factory
│   ├── base.py               # Abstract base classes
│   ├── gpt.py                # GPT-2 implementation
│   ├── bert.py               # BERT implementation
│   ├── rnn.py                # RNN/LSTM/GRU implementations
│   └── utils.py              # Shared utilities
├── data_collators.py         # Data collation for different objectives
└── tokenizer/
    └── tokenizer_factory.py  # Multi-tokenizer support

configs/
├── experiment_bert_masked.yaml
├── experiment_lstm_bidirectional.yaml
└── experiment_lstm_causal.yaml
```

### Files to Modify

```
model_foundry/
├── model.py                  # Update to use factory
├── config.py                 # Extend ModelConfig, add RNN config
├── trainer.py                # Minor updates for new architectures
├── data.py                   # Use data collators
└── tokenizer/
    └── train_tokenizer.py    # Add tokenizer type support

evaluation/
└── core/
    └── model_evaluator.py    # Add bidirectional model support
```

---

## Key Considerations

### 1. Memory Requirements

**RNNs vs Transformers:**
- RNNs: Lower memory (sequential processing)
- Transformers: Higher memory (self-attention is O(n²))
- Bidirectional models: ~2x parameters of unidirectional

**Implications:**
- Adjust batch sizes per architecture
- LSTM can use larger batches than BERT/GPT-2

### 2. Training Speed

**Relative Speed (approximate):**
- Unidirectional LSTM: Fastest (sequential)
- Bidirectional LSTM: 2x slower (two passes)
- GPT-2: Medium (parallelizable)
- BERT: Slowest (bidirectional attention + MLM)

### 3. Generation Capability

**Supports Autoregressive Generation:**
- ✅ GPT-2 (designed for it)
- ✅ Unidirectional LSTM (natural fit)
- ❌ BERT (not designed for generation)
- ⚠️ Bidirectional LSTM (possible but awkward)

**Implications for Evaluation:**
- Some evaluations (BLIMP) require adaptation for BERT
- Need pseudo-perplexity for bidirectional models

### 4. Checkpoint Compatibility

**HuggingFace Compatibility:**
- GPT-2: Full compatibility
- BERT: Full compatibility
- LSTM: Custom implementation, need custom loading

**Implications:**
- Can load pretrained GPT-2/BERT from HuggingFace
- LSTM models trained from scratch only

---

## Summary: What's Required

### Code Changes (Estimated Lines of Code)

**New Code:**
- Architecture abstraction: ~500 lines
- BERT implementation: ~200 lines
- LSTM implementation: ~400 lines
- Data collators: ~200 lines
- Tokenizer factory: ~300 lines
- **Total new code: ~1600 lines**

**Modified Code:**
- Config updates: ~200 lines
- Model factory: ~100 lines
- Trainer updates: ~50 lines
- Data processor: ~100 lines
- **Total modifications: ~450 lines**

### Dependencies

**New Python Packages:**
- `tokenizers` (HuggingFace tokenizers library) - already likely installed
- No additional dependencies needed

### Testing Requirements

**Unit Tests:**
- Test each architecture creates successfully
- Test model registry
- Test data collators
- Test tokenizer types

**Integration Tests:**
- End-to-end training for each architecture
- Evaluation compatibility
- Checkpoint saving/loading

**Performance Tests:**
- Memory usage profiling
- Training speed benchmarks
- Convergence validation

---

## Conclusion

Expanding `model_foundry` to support RNN, LSTM, and BERT is **feasible and well-scoped**:

1. **Clean Abstraction:** Registry pattern keeps architectures modular
2. **Minimal Core Changes:** Most existing code remains untouched
3. **Backwards Compatible:** Existing GPT-2 experiments continue working
4. **Incremental:** Can implement architectures one at a time
5. **Well-Tested:** HuggingFace integration provides validation

**Estimated Effort:** 6 weeks for full implementation with testing

**Key Benefits:**
- Compare architectural inductive biases on same data
- Study how RNNs vs Transformers learn linguistic phenomena
- Investigate bidirectional vs unidirectional processing
- Unified pipeline for all experiments

**Recommended Approach:** Implement in phases, maintaining backwards compatibility throughout.

