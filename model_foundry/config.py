from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Nested Pydantic models for better organization

class DataConfig(BaseModel):
    source_corpus: str
    training_corpus: str
    test_corpus: Optional[str] = None  # Path to test dataset
    batch_size: int = Field(..., gt=0)
    max_sequence_length: int = Field(..., gt=0)

class TokenizerConfig(BaseModel):
    output_dir: str
    vocab_size: int = Field(..., gt=0)

class ModelConfig(BaseModel):
    layers: int = Field(..., gt=0)
    embedding_size: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    intermediate_hidden_size: int = Field(..., gt=0)
    attention_heads: int = Field(..., gt=0)
    activation_function: str
    dropout: float = Field(..., ge=0.0, lt=1.0)
    attention_dropout: float = Field(..., ge=0.0, lt=1.0)

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
    
    # Mixed precision training
    use_amp: bool = False  # Enable automatic mixed precision
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients
    
    # Gradient clipping
    max_grad_norm: Optional[float] = Field(None, gt=0)  # Maximum gradient norm for clipping
    
    # Memory optimizations
    use_tf32: bool = True  # Enable TF32 for faster training on Ampere+ GPUs
    use_gradient_checkpointing: bool = False  # Enable gradient checkpointing to save memory
    
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
    wandb_project: Optional[str] = Field(default=None, description="WandB project name")

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