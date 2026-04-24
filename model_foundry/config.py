from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Literal

# Nested Pydantic models for better organization

class DataConfig(BaseModel):
    source_corpus: str
    training_corpus: str
    test_corpus: Optional[str] = None  # Path to test dataset
    batch_size: int = Field(..., gt=0)
    max_sequence_length: int = Field(..., gt=0)

    # DataLoader tuning (typed; replaces prior getattr/hasattr fallbacks).
    # num_workers=None means "use cgroup-aware auto default" (see data.py).
    num_workers: Optional[int] = Field(
        default=None,
        ge=0,
        description="DataLoader worker count. None → cgroup-aware auto.",
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin DataLoader output in host memory for faster H2D copy.",
    )
    prefetch_factor: int = Field(
        default=2,
        ge=1,
        description="Number of batches pre-fetched per worker (only used when num_workers > 0).",
    )

class TokenizerConfig(BaseModel):
    output_dir: str
    vocab_size: int = Field(..., gt=0)
    # New tokenizer type field for multi-tokenizer support (Phase 2)
    tokenizer_type: str = Field(default="sentencepiece", description="Tokenizer algorithm")
    special_tokens: Optional[Dict[str, str]] = Field(
        default=None,
        description="Special tokens for tokenizer (architecture-specific)"
    )
    # WordPiece normalizer toggle: English BERT convention strips accents;
    # Spanish must NOT (sí/si, él/el, sé/se, más/mas, año/ano are distinct).
    strip_accents: bool = Field(
        default=True,
        description="WordPiece only — apply StripAccents in the normalizer.",
    )

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

class MambaModelConfig(BaseModel):
    """Configuration for Mamba state space model."""
    d_model: int = Field(..., gt=0, description="Model dimension (embedding size)")
    n_layers: int = Field(..., gt=0, description="Number of Mamba layers")
    d_state: int = Field(16, gt=0, description="SSM state expansion factor")
    d_conv: int = Field(4, gt=0, description="Local convolution width")
    expand: int = Field(2, gt=0, description="Block expansion factor")
    dropout: float = Field(0.0, ge=0.0, lt=1.0, description="Dropout probability")

class ModelConfig(BaseModel):
    """
    Unified model configuration supporting multiple architectures.

    This configuration supports GPT-2, BERT, and RNN/LSTM/GRU architectures.
    The 'architecture' field determines which model type to create, and the
    corresponding architecture-specific config must be provided.
    """

    # Required: Architecture type
    architecture: Literal["gpt2", "bert", "lstm", "rnn", "gru", "mamba"] = Field(
        ...,
        description="Model architecture family (required)"
    )

    # Architecture-specific configs (provide based on architecture)
    transformer: Optional[TransformerModelConfig] = Field(
        None,
        description="Transformer config (required for gpt2, bert)"
    )
    bert: Optional[BERTSpecificConfig] = Field(
        None,
        description="BERT-specific config (optional, only for bert)"
    )
    rnn: Optional[RNNModelConfig] = Field(
        None,
        description="RNN config (required for lstm, rnn, gru)"
    )
    mamba: Optional[MambaModelConfig] = Field(
        None,
        description="Mamba config (required for mamba)"
    )

    @model_validator(mode='before')
    @classmethod
    def _backwards_compat_flat_config(cls, data: Any) -> Any:
        """Convert old flat model config format to new nested format.

        Old format (pre-refactor):
            model:
              layers: 12
              embedding_size: 768
              ...

        New format:
            model:
              architecture: gpt2
              transformer:
                layers: 12
                embedding_size: 768
                ...

        This validator detects the old format by the absence of 'architecture'
        and the presence of flat fields, then restructures accordingly.
        """
        if not isinstance(data, dict):
            return data

        # If architecture is already present, assume new format — pass through
        if 'architecture' in data:
            return data

        # Flat field sets for each architecture family
        _TRANSFORMER_FIELDS = {
            'layers', 'embedding_size', 'hidden_size',
            'intermediate_hidden_size', 'attention_heads',
            'activation_function', 'dropout', 'attention_dropout',
        }
        _RNN_FIELDS = {
            'embedding_size', 'hidden_size', 'num_layers',
            'bidirectional', 'dropout', 'rnn_type',
        }

        # Detect RNN: presence of rnn-only keys (num_layers or rnn_type)
        rnn_marker_fields = {'num_layers', 'rnn_type'}
        present_keys = set(data.keys())

        if rnn_marker_fields & present_keys:
            # RNN-family config
            rnn_type = data.get('rnn_type', 'lstm')
            architecture = rnn_type if rnn_type in ('rnn', 'lstm', 'gru') else 'lstm'
            nested = {}
            for field in _RNN_FIELDS:
                if field in data:
                    nested[field] = data.pop(field)
            data['architecture'] = architecture
            data['rnn'] = nested
        elif _TRANSFORMER_FIELDS & present_keys:
            # Transformer-family config — default to gpt2
            nested = {}
            for field in _TRANSFORMER_FIELDS:
                if field in data:
                    nested[field] = data.pop(field)
            data['architecture'] = 'gpt2'
            data['transformer'] = nested

        return data

    @field_validator('transformer', 'rnn', 'mamba')
    @classmethod
    def validate_architecture_config(cls, v, info):
        """Validate that appropriate config is provided for architecture."""
        # Get the architecture value if available
        if 'architecture' in info.data:
            architecture = info.data['architecture']
            field_name = info.field_name

            # Check transformer architectures
            if architecture in ["gpt2", "bert"] and field_name == "transformer":
                if v is None:
                    raise ValueError(
                        f"Architecture '{architecture}' requires 'transformer' configuration. "
                        "Example:\n"
                        "  model:\n"
                        f"    architecture: '{architecture}'\n"
                        "    transformer:\n"
                        "      layers: 12\n"
                        "      embedding_size: 768\n"
                        "      ..."
                    )

            # Check RNN architectures
            if architecture in ["lstm", "rnn", "gru"] and field_name == "rnn":
                if v is None:
                    raise ValueError(
                        f"Architecture '{architecture}' requires 'rnn' configuration. "
                        "Example:\n"
                        "  model:\n"
                        f"    architecture: '{architecture}'\n"
                        "    rnn:\n"
                        "      embedding_size: 512\n"
                        "      hidden_size: 512\n"
                        "      num_layers: 2\n"
                        "      ..."
                    )

            # Check Mamba architecture
            if architecture == "mamba" and field_name == "mamba":
                if v is None:
                    raise ValueError(
                        "Architecture 'mamba' requires 'mamba' configuration. "
                        "Example:\n"
                        "  model:\n"
                        "    architecture: 'mamba'\n"
                        "    mamba:\n"
                        "      d_model: 768\n"
                        "      n_layers: 24\n"
                        "      d_state: 16\n"
                        "      ..."
                    )

        return v

    # Convenience properties for backwards compatibility
    @property
    def layers(self) -> int:
        """Get number of layers (works for transformers, RNNs, and Mamba)."""
        if self.transformer:
            return self.transformer.layers
        elif self.rnn:
            return self.rnn.num_layers
        elif self.mamba:
            return self.mamba.n_layers
        raise ValueError("No architecture config provided")

    @property
    def embedding_size(self) -> int:
        """Get embedding size."""
        if self.transformer:
            return self.transformer.embedding_size
        elif self.rnn:
            return self.rnn.embedding_size
        elif self.mamba:
            return self.mamba.d_model
        raise ValueError("No architecture config provided")

    @property
    def hidden_size(self) -> int:
        """Get hidden size."""
        if self.transformer:
            return self.transformer.hidden_size
        elif self.rnn:
            return self.rnn.hidden_size
        elif self.mamba:
            return self.mamba.d_model
        raise ValueError("No architecture config provided")

    @property
    def intermediate_hidden_size(self) -> Optional[int]:
        """Get intermediate hidden size (transformer only)."""
        if self.transformer:
            return self.transformer.intermediate_hidden_size
        return None

    @property
    def attention_heads(self) -> Optional[int]:
        """Get attention heads (transformer only)."""
        if self.transformer:
            return self.transformer.attention_heads
        return None

    @property
    def activation_function(self) -> str:
        """Get activation function."""
        if self.transformer:
            return self.transformer.activation_function
        return "relu"  # Default for RNNs

    @property
    def dropout(self) -> float:
        """Get dropout rate."""
        if self.transformer:
            return self.transformer.dropout
        elif self.rnn:
            return self.rnn.dropout
        return 0.0

    @property
    def attention_dropout(self) -> float:
        """Get attention dropout (transformer only)."""
        if self.transformer:
            return self.transformer.attention_dropout
        return 0.0

class TrainingConfig(BaseModel):
    output_dir: str
    learning_rate: float = Field(..., gt=0)
    adam_beta1: float = Field(..., ge=0.0, lt=1.0)
    adam_beta2: float = Field(..., ge=0.0, lt=1.0)
    adam_epsilon: float = Field(..., gt=0)

    # Training duration - can be specified explicitly or calculated automatically
    warmup_steps: Optional[int] = Field(None, ge=0)  # If None, will be calculated as percentage of train_steps
    train_steps: Optional[int] = Field(None, gt=0)   # If None, will be calculated from epochs
    epochs: int = Field(..., gt=0)

    # Automatic calculation parameters
    warmup_ratio: float = Field(0.1, ge=0.0, le=1.0)  # Warmup as fraction of total training steps
    checkpointing_strategy: Optional[str] = None
    checkpoint_schedule: Optional[List[int]] = []
    resume_from_checkpoint: bool = False

    # Training objective (NEW for multi-architecture support)
    objective: Literal["causal_lm", "masked_lm"] = Field(
        "causal_lm",
        description="Training objective type: causal_lm (GPT) or masked_lm (BERT)"
    )

    # Masked LM specific parameters
    mlm_probability: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        description="Probability of masking tokens for MLM (only used with masked_lm objective)"
    )

    # Mixed precision training
    use_amp: bool = False  # Enable automatic mixed precision

    # Gradient accumulation
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients

    # Gradient clipping
    max_grad_norm: Optional[float] = Field(None, gt=0)  # Maximum gradient norm for clipping

    # Memory optimizations
    use_tf32: bool = True  # Enable TF32 for faster training on Ampere+ GPUs
    use_gradient_checkpointing: bool = False  # Enable gradient checkpointing to save memory

    # torch.compile mode. None disables compile entirely. Valid modes:
    # "default", "reduce-overhead", "max-autotune". Backend is Inductor
    # (torch.compile's default) — the old hard-coded "eager" backend
    # defeated the speedup.
    compile_mode: Optional[str] = Field(
        default=None,
        description="torch.compile mode ('default', 'reduce-overhead', 'max-autotune', or None to disable).",
    )

    # Resume-state save policy (0.6). If set, only the last N scheduled
    # checkpoints save optimizer/scheduler/RNG/AMP state (~3× model size).
    # Earlier checkpoints save analysis-only (weights + tokenizer + metadata).
    # None → save resume state for every checkpoint (current/default behavior).
    save_resume_state_last_n: Optional[int] = Field(
        default=None,
        ge=0,
        description="Only last N scheduled checkpoints save resume state. None keeps all as full resume.",
    )

    # Checkpoint generation parameters
    auto_generate_checkpoints: bool = False

    # First epoch configuration
    first_epoch_checkpoints: int = 20  # Number of checkpoints in first epoch

    # Subsequent epochs configuration
    subsequent_epochs_spacing: str = "log"  # "linear" or "log"
    log_base: int = 2  # Base for logarithmic spacing (default 2)
    linear_interval: Optional[int] = None  # Steps between checkpoints for linear spacing

    # Minimum interval between checkpoints
    min_checkpoint_interval: int = 100

    # Minimum number of checkpoints per epoch (for epochs after the first)
    min_checkpoints_per_epoch: int = 5

class LoggingConfig(BaseModel):
    """Configuration for logging behavior."""

    # Log levels
    console_level: str = Field(default="INFO", description="Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    file_level: str = Field(default="DEBUG", description="File log level")

    # Log directory
    dir: str = Field(default="logs", description="Directory for log files")

    # Output formats
    use_structured_logging: bool = Field(default=True, description="Enable JSON structured logs")

    # WandB integration
    use_wandb: bool = Field(default=False, description="Send logs to Weights & Biases")
    wandb_project: Optional[str] = Field(default=None, description="WandB project name for production runs (run_kind=production)")
    wandb_project_sweeps: str = Field(default="subject-drop-sweeps", description="WandB project for HP-sweep trials (run_kind=hp_sweep)")
    wandb_entity: Optional[str] = Field(default=None, description="WandB entity/team. None → user's personal entity.")

    # Log rotation
    max_log_files: int = Field(default=10, gt=0, description="Maximum log files to keep")
    max_log_size_mb: int = Field(default=100, gt=0, description="Maximum size per log file in MB")

    # Metrics logging
    log_metrics_every_n_steps: int = Field(default=10, gt=0, description="Steps between metric logs")
    log_detailed_metrics_every_n_steps: int = Field(default=100, gt=0, description="Steps between detailed metrics")

    # Performance logging
    profile_performance: bool = Field(default=False, description="Enable performance profiling")
    log_memory_every_n_steps: int = Field(default=100, gt=0, description="Steps between memory logs")

    # Error tracking
    max_errors_to_track: int = Field(default=1000, gt=0, description="Maximum errors to track in memory")

    # Legacy compatibility
    level: str = Field(default="INFO", description="Legacy log level (use console_level instead)")

    def __init__(self, **data):
        """Initialize LoggingConfig with level validation."""
        super().__init__(**data)

        # Validate log levels
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.console_level not in valid_levels:
            raise ValueError(f"console_level must be one of {valid_levels}")
        if self.file_level not in valid_levels:
            raise ValueError(f"file_level must be one of {valid_levels}")
        if self.level not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")

# The main configuration model that brings everything together

class ExperimentConfig(BaseModel):
    experiment_name: str
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    random_seed: int
    dataset_manipulation: Optional[List[Dict[str, Any]]] = []