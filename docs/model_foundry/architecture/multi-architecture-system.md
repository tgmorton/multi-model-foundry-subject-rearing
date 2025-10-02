# Multi-Architecture System

**Purpose:** Documentation for the multi-architecture framework that enables training GPT-2, BERT, LSTM, and other model architectures within Model Foundry.

**Date:** 2025-10-02

**Status:** ✅ Phase 1 Complete, ✅ Phase 2 Complete (BERT & Masked LM)

---

## Overview

The Model Foundry framework has been extended to support multiple model architectures beyond the original GPT-2 implementation. This document describes the architecture abstraction layer, registry pattern, and how to add new model architectures.

### Supported Architectures

| Architecture | Type | Training Objective | Tokenizer | Status |
|-------------|------|-------------------|-----------|--------|
| GPT-2 | Decoder-only Transformer | Causal LM | SentencePiece | ✅ Complete |
| BERT | Encoder-only Transformer | Masked LM | WordPiece | ✅ Complete (Phase 2) |
| LSTM | Recurrent Network | Causal LM / Masked LM | SentencePiece/BPE | 🚧 Planned (Phase 3) |
| GRU | Recurrent Network | Causal LM / Masked LM | SentencePiece/BPE | 🚧 Planned (Phase 4) |
| RNN | Recurrent Network | Causal LM / Masked LM | SentencePiece/BPE | 🚧 Planned (Phase 4) |

---

## Architecture Overview

### Component Structure

```
model_foundry/
├── architectures/              # Multi-architecture support
│   ├── __init__.py            # Registry and factory
│   ├── base.py                # Abstract base classes
│   ├── gpt.py                 # GPT-2 implementation
│   ├── bert.py                # BERT implementation ✅
│   ├── rnn.py                 # RNN/LSTM/GRU (Phase 3-4)
│   └── utils.py               # Shared utilities
├── data_collators.py          # Causal LM & Masked LM collators ✅
├── tokenizer/
│   ├── tokenizer_factory.py  # Multi-tokenizer support ✅
│   └── train_tokenizer.py    # Updated to use factory ✅
├── model.py                   # Public API (uses factory)
├── config.py                  # Multi-architecture configs
└── data.py                    # Updated for multi-objective ✅
```

### Key Design Principles

1. **Common Interface:** All architectures implement `BaseLanguageModel`
2. **Registry Pattern:** Architectures self-register using decorators
3. **Factory Creation:** Models created via `create_model_from_config()`
4. **HuggingFace Compatible:** Leverages transformers library where possible
5. **Extensible:** Easy to add new architectures

---

## BaseLanguageModel Interface

All model architectures must inherit from `BaseLanguageModel` and implement:

### Required Methods

```python
class BaseLanguageModel(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutput:
        """Forward pass returning ModelOutput with loss and logits."""
        pass

    @abstractmethod
    def get_input_embeddings(self) -> nn.Module:
        """Return the input embedding layer."""
        pass

    @abstractmethod
    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """Resize vocabulary."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Architecture identifier (e.g., 'gpt2', 'bert', 'lstm')."""
        pass

    @property
    @abstractmethod
    def supports_generation(self) -> bool:
        """Whether model supports autoregressive generation."""
        pass
```

### Standard Methods

Provided by base class with default implementations:

- `save_pretrained(save_directory)` - Save model
- `from_pretrained(model_directory)` - Load model
- `get_parameter_count()` - Count parameters
- `get_memory_footprint()` - Memory statistics

---

## Model Output Format

All models return a standardized `ModelOutput`:

```python
class ModelOutput:
    loss: Optional[torch.Tensor]        # Training loss (if labels provided)
    logits: torch.Tensor                # Model predictions
    hidden_states: Optional[torch.Tensor]  # Layer outputs
    attentions: Optional[torch.Tensor]     # Attention weights
```

**Benefits:**
- Consistent interface across architectures
- Compatible with HuggingFace outputs
- Supports both training and inference

---

## Architecture Registration

Models self-register using the `@register_architecture` decorator:

```python
from model_foundry.architectures import register_architecture, BaseLanguageModel

@register_architecture("gpt2")
class GPT2Model(BaseLanguageModel):
    """GPT-2 causal language model."""

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create model from ExperimentConfig."""
        # Extract parameters from config
        # Create and return model instance
        pass

    # Implement required abstract methods...
```

**Registration Process:**
1. Import triggers decorator execution
2. Class added to `MODEL_REGISTRY`
3. Factory can create instances by name

---

## Model Creation Flow

### 1. Configuration

```yaml
model:
  architecture: "gpt2"  # Required: Specifies model type
  transformer:          # Architecture-specific config
    layers: 12
    embedding_size: 768
    hidden_size: 768
    intermediate_hidden_size: 3072
    attention_heads: 12
    activation_function: "gelu"
    dropout: 0.1
    attention_dropout: 0.1
```

### 2. Factory Creation

```python
from model_foundry.model import create_model

# Load config
config = load_config("configs/experiment.yaml")

# Create model (architecture determined by config)
model = create_model(config)

# Model is instance of GPT2Model (implements BaseLanguageModel)
assert model.model_type == "gpt2"
assert model.supports_generation is True
```

### 3. Training

```python
# Standard training loop works for all architectures
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

---

## Configuration Schema

### Multi-Architecture ModelConfig

```python
class ModelConfig(BaseModel):
    # Required: Architecture type
    architecture: Literal["gpt2", "bert", "lstm", "rnn", "gru"]

    # Architecture-specific configs (provide one based on architecture)
    transformer: Optional[TransformerModelConfig] = None
    bert: Optional[BERTSpecificConfig] = None
    rnn: Optional[RNNModelConfig] = None
```

### Transformer Config (GPT-2, BERT)

```python
class TransformerModelConfig(BaseModel):
    layers: int
    embedding_size: int
    hidden_size: int
    intermediate_hidden_size: int
    attention_heads: int
    activation_function: str = "gelu"
    dropout: float
    attention_dropout: float
```

### BERT-Specific Config

```python
class BERTSpecificConfig(BaseModel):
    type_vocab_size: int = 2  # For segment embeddings
    pooler_type: str = "first"
```

### RNN Config (LSTM, GRU, RNN)

```python
class RNNModelConfig(BaseModel):
    embedding_size: int
    hidden_size: int
    num_layers: int
    bidirectional: bool = False
    dropout: float = 0.0
    rnn_type: Literal["rnn", "lstm", "gru"] = "lstm"
```

### Training Objective Config

```python
class TrainingConfig(BaseModel):
    # ... existing fields ...

    # Training objective
    objective: Literal["causal_lm", "masked_lm"] = "causal_lm"

    # Masked LM specific
    mlm_probability: float = 0.15  # For masked_lm objective
```

---

## GPT-2 Implementation

### Architecture Wrapper

```python
@register_architecture("gpt2")
class GPT2Model(BaseLanguageModel):
    """Wraps HuggingFace GPT-2 for multi-architecture framework."""

    def __init__(self, hf_model, hf_config):
        super().__init__()
        self.hf_model = hf_model
        self.config = hf_config

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Delegate to HuggingFace model
        outputs = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # Convert to ModelOutput
        return ModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    @property
    def model_type(self) -> str:
        return "gpt2"

    @property
    def supports_generation(self) -> bool:
        return True
```

### Features

- Full HuggingFace compatibility
- Flash Attention support
- Gradient checkpointing
- Autoregressive generation
- Standard save/load

---

## BERT Implementation (Phase 2)

### Architecture Wrapper

```python
@register_architecture("bert")
class BERTModel(BaseLanguageModel):
    """Wraps HuggingFace BERT for masked language modeling."""

    def __init__(self, hf_model, hf_config):
        super().__init__()
        self.hf_model = hf_model  # AutoModelForMaskedLM
        self.config = hf_config

    def forward(self, input_ids, attention_mask=None, labels=None,
                token_type_ids=None, **kwargs):
        # Delegate to HuggingFace BERT
        outputs = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
            **kwargs
        )

        # Convert to ModelOutput
        return ModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    @property
    def model_type(self) -> str:
        return "bert"

    @property
    def supports_generation(self) -> bool:
        return False  # BERT is bidirectional, cannot generate
```

### Features

- Bidirectional attention (sees full context)
- Masked language modeling objective
- Segment embeddings for sentence pairs
- [CLS] token for sequence representation
- WordPiece tokenization

### Example Configuration

```yaml
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
  objective: "masked_lm"
  mlm_probability: 0.15

tokenizer:
  tokenizer_type: "wordpiece"
  vocab_size: 30000
  special_tokens:
    cls_token: "[CLS]"
    sep_token: "[SEP]"
    mask_token: "[MASK]"
    unk_token: "[UNK]"
    pad_token: "[PAD]"
```

---

## Data Collators (Phase 2)

### CausalLMDataCollator

For autoregressive language modeling (GPT-2, unidirectional LSTM):

```python
from model_foundry.data_collators import CausalLMDataCollator

collator = CausalLMDataCollator(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)
```

**Features:**
- Pads sequences to batch max length
- Creates attention masks (1 for real tokens, 0 for padding)
- Sets labels = input_ids (shifted internally by model)
- Masks padding tokens in labels (-100 for ignored positions)

### MaskedLMDataCollator

For masked language modeling (BERT, bidirectional models):

```python
from model_foundry.data_collators import MaskedLMDataCollator

collator = MaskedLMDataCollator(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    pad_to_multiple_of=8
)
```

**Features:**
- Randomly masks 15% of tokens
- BERT masking strategy:
  - 80% replace with [MASK]
  - 10% replace with random token
  - 10% keep unchanged
- Never masks special tokens (CLS, SEP, PAD)
- Labels set to -100 for non-masked positions

### Data Collator Factory

Automatically selects appropriate collator based on training objective:

```python
from model_foundry.data_collators import get_data_collator

collator = get_data_collator(config, tokenizer)
# Returns CausalLMDataCollator for objective="causal_lm"
# Returns MaskedLMDataCollator for objective="masked_lm"
```

---

## Tokenizer Factory (Phase 2)

### Supported Tokenizer Types

| Type | Best For | Special Tokens |
|------|---------|---------------|
| SentencePiece | GPT-2, general LMs | `<s>`, `</s>`, `<unk>`, `<pad>` |
| WordPiece | BERT | `[CLS]`, `[SEP]`, `[MASK]`, `[UNK]`, `[PAD]` |
| BPE | RoBERTa-style | `<s>`, `</s>`, `<unk>`, `<pad>` |
| Character | RNNs, small vocab | Custom |

### Training Tokenizers

```python
from model_foundry.tokenizer.tokenizer_factory import TokenizerFactory

# Train WordPiece tokenizer for BERT
tokenizer = TokenizerFactory.train_wordpiece(
    input_files=["train.txt", "test.txt"],
    output_dir="tokenizer/bert",
    vocab_size=30000,
    special_tokens={
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]"
    }
)

# Train SentencePiece tokenizer for GPT-2
tokenizer = TokenizerFactory.train_sentencepiece(
    input_files=["train.txt"],
    output_dir="tokenizer/gpt2",
    vocab_size=50000,
    special_tokens={
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>"
    }
)
```

### Using Tokenizer Factory with Config

```python
from model_foundry.tokenizer.tokenizer_factory import train_tokenizer_from_config

# Reads config YAML and trains appropriate tokenizer
tokenizer = train_tokenizer_from_config("configs/experiment.yaml")
```

---

## Adding New Architectures

### Step-by-Step Guide

1. **Create Architecture Module**
   - Add file in `model_foundry/architectures/`
   - Implement `BaseLanguageModel` interface

2. **Register Architecture**
   - Use `@register_architecture("name")` decorator
   - Implement `from_config()` class method

3. **Add Configuration Support**
   - Update `config.py` with architecture-specific config
   - Add validation logic

4. **Implement Required Methods**
   - `forward()` - Training and inference
   - `get_input_embeddings()` - Embedding layer
   - `resize_token_embeddings()` - Vocabulary resizing
   - `model_type` property
   - `supports_generation` property

5. **Write Tests**
   - Unit tests for architecture
   - Integration tests with training loop
   - Config validation tests

6. **Update Documentation**
   - Add architecture to this document
   - Update example configs
   - Document any special requirements

### Example Template

```python
from model_foundry.architectures import register_architecture, BaseLanguageModel, ModelOutput

@register_architecture("my_arch")
class MyArchModel(BaseLanguageModel):
    """My custom architecture."""

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create model from config."""
        # Extract parameters
        arch_config = config.model.my_arch

        # Build model
        model = cls(...)
        return model

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass."""
        # Implement forward logic
        logits = ...

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = ...

        return ModelOutput(loss=loss, logits=logits)

    def get_input_embeddings(self):
        return self.embeddings

    def resize_token_embeddings(self, new_num_tokens):
        # Resize logic
        pass

    @property
    def model_type(self) -> str:
        return "my_arch"

    @property
    def supports_generation(self) -> bool:
        return True  # or False
```

---

## Validation and Error Handling

### Configuration Validation

Configs are validated at load time:

```python
# Missing architecture field
>>> config = ExperimentConfig(...)
ValidationError: Field 'architecture' is required

# Wrong architecture-specific config
>>> config.model.architecture = "gpt2"
>>> config.model.transformer = None
ValidationError: Architecture 'gpt2' requires 'transformer' configuration

# Unknown architecture
>>> config.model.architecture = "unknown"
>>> model = create_model(config)
ValueError: Unknown architecture: 'unknown'
```

### Runtime Validation

```python
# Architecture not registered
>>> create_model_from_config(config)
ValueError: Architecture 'new_arch' not registered
Available: ['gpt2', 'bert', 'lstm']
```

---

## Testing

### Test Structure

```
model_foundry/tests/unit/
├── test_architectures.py      # Architecture abstraction tests
├── test_model.py               # Model creation tests
├── test_config.py              # Configuration validation
└── ...
```

### Key Test Areas

1. **Registry Tests**
   - Architecture registration
   - Duplicate prevention
   - Type checking

2. **Interface Tests**
   - `BaseLanguageModel` implementation
   - Method signatures
   - Return types

3. **Factory Tests**
   - Model creation from config
   - Kwargs passing
   - Error handling

4. **Integration Tests**
   - Forward/backward pass
   - Training loop compatibility
   - Save/load roundtrip

### Running Tests

```bash
# Architecture tests only
pytest model_foundry/tests/unit/test_architectures.py -v

# All model tests
pytest model_foundry/tests/unit/test_model.py -v

# Config validation tests
pytest model_foundry/tests/unit/test_config.py -v
```

---

## Future Architectures

### BERT (Phase 2)

- Masked language modeling
- Bidirectional attention
- WordPiece tokenization
- No generation support

### LSTM (Phase 3)

- Recurrent architecture
- Uni/bidirectional variants
- Both causal and masked LM
- Custom PyTorch implementation

### GRU/RNN (Phase 4)

- Additional recurrent variants
- Same interface as LSTM
- Performance comparisons

---

## Performance Considerations

### Memory Usage by Architecture

| Architecture | Parameters (Base) | Memory (approx) | Batch Size Recommendation |
|-------------|------------------|-----------------|---------------------------|
| GPT-2 (base) | 124M | ~500 MB | 16-32 |
| BERT (base) | 110M | ~450 MB | 16-32 |
| LSTM (3-layer) | ~50M | ~200 MB | 32-64 |

### Training Speed

Approximate relative speeds:
- Unidirectional LSTM: 1.0x (fastest)
- GPT-2: 0.7x
- Bidirectional LSTM: 0.5x
- BERT: 0.4x (slowest)

---

## Migration Guide

### From Old GPT-2-Only Code

**Old Config:**
```yaml
model:
  layers: 12
  embedding_size: 768
  hidden_size: 768
  # ...
```

**New Config:**
```yaml
model:
  architecture: "gpt2"  # NEW: Required field
  transformer:           # NEW: Nested config
    layers: 12
    embedding_size: 768
    hidden_size: 768
    # ...
```

**Code Changes:**
```python
# Old
from transformers import GPT2LMHeadModel
assert isinstance(model, GPT2LMHeadModel)

# New
from model_foundry.architectures import BaseLanguageModel, GPT2Model
assert isinstance(model, BaseLanguageModel)
assert isinstance(model, GPT2Model)
assert model.model_type == "gpt2"
```

---

## References

- **Planning Document:** [plans/MULTI_ARCHITECTURE_EXPANSION.md](../../../plans/MULTI_ARCHITECTURE_EXPANSION.md)
- **Development Guide:** [plans/DEVELOPMENT_METHODOLOGY.md](../../../plans/DEVELOPMENT_METHODOLOGY.md)
- **Testing Strategy:** [testing/strategy.md](../testing/strategy.md)
- **Base Classes:** `model_foundry/architectures/base.py`
- **Registry Implementation:** `model_foundry/architectures/__init__.py`

---

## Changelog

### Phase 1: Architecture Abstraction (2025-10-02)

**Implemented:**
- ✅ `BaseLanguageModel` abstract class
- ✅ `ModelOutput` standardized output
- ✅ Registry pattern with `@register_architecture`
- ✅ Factory function `create_model_from_config()`
- ✅ GPT-2 wrapper implementing interface
- ✅ Multi-architecture `ModelConfig`
- ✅ Training objective support
- ✅ Comprehensive test suite (61 tests, all passing)

**Files Added:**
- `model_foundry/architectures/base.py`
- `model_foundry/architectures/__init__.py`
- `model_foundry/architectures/gpt.py`
- `model_foundry/architectures/utils.py`
- `model_foundry/tests/unit/test_architectures.py`

**Files Modified:**
- `model_foundry/model.py` - Uses factory
- `model_foundry/config.py` - Multi-architecture support
- `model_foundry/tests/conftest.py` - Updated fixtures
- `model_foundry/tests/unit/test_model.py` - Updated for new interface
- `model_foundry/tests/unit/test_config.py` - Updated for new config schema

**Breaking Changes:**
- Config must specify `architecture` field
- Old flat `ModelConfig` structure replaced with nested configs
- Model creation returns `BaseLanguageModel` (still compatible with GPT-2)

---

**Last Updated:** 2025-10-02
**Next Steps:** Phase 2 - BERT Implementation
