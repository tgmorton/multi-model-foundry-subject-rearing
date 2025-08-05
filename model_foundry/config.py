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

class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: str = "logs"
    use_wandb: bool = False
    wandb_project: Optional[str] = None

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