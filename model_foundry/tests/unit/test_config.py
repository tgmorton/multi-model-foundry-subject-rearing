"""
Unit tests for configuration validation.

Tests the Pydantic-based configuration models to ensure proper validation
of experiment parameters.
"""

import pytest
from pydantic import ValidationError

from model_foundry.config import (
    ExperimentConfig, DataConfig, TokenizerConfig, ModelConfig,
    TrainingConfig, LoggingConfig
)


class TestDataConfig:
    """Tests for DataConfig validation."""

    def test_valid_data_config(self):
        """Valid configuration should load successfully."""
        config = DataConfig(
            source_corpus="data/source",
            training_corpus="data/train",
            batch_size=32,
            max_sequence_length=512
        )
        assert config.batch_size == 32
        assert config.max_sequence_length == 512

    def test_batch_size_must_be_positive(self):
        """Batch size must be greater than 0."""
        with pytest.raises(ValidationError) as exc_info:
            DataConfig(
                source_corpus="data/source",
                training_corpus="data/train",
                batch_size=0,
                max_sequence_length=512
            )
        assert "batch_size" in str(exc_info.value)

    def test_negative_batch_size_rejected(self):
        """Negative batch size should be rejected."""
        with pytest.raises(ValidationError):
            DataConfig(
                source_corpus="data/source",
                training_corpus="data/train",
                batch_size=-10,
                max_sequence_length=512
            )

    def test_max_sequence_length_must_be_positive(self):
        """Max sequence length must be greater than 0."""
        with pytest.raises(ValidationError):
            DataConfig(
                source_corpus="data/source",
                training_corpus="data/train",
                batch_size=32,
                max_sequence_length=0
            )

    def test_optional_test_corpus(self):
        """Test corpus is optional."""
        config = DataConfig(
            source_corpus="data/source",
            training_corpus="data/train",
            batch_size=32,
            max_sequence_length=512
        )
        assert config.test_corpus is None


class TestTokenizerConfig:
    """Tests for TokenizerConfig validation."""

    def test_valid_tokenizer_config(self):
        """Valid tokenizer configuration."""
        config = TokenizerConfig(
            output_dir="tokenizer/",
            vocab_size=10000
        )
        assert config.vocab_size == 10000

    def test_vocab_size_must_be_positive(self):
        """Vocab size must be greater than 0."""
        with pytest.raises(ValidationError):
            TokenizerConfig(
                output_dir="tokenizer/",
                vocab_size=0
            )

    def test_negative_vocab_size_rejected(self):
        """Negative vocab size should be rejected."""
        with pytest.raises(ValidationError):
            TokenizerConfig(
                output_dir="tokenizer/",
                vocab_size=-1000
            )


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_valid_model_config(self):
        """Valid model configuration."""
        config = ModelConfig(
            layers=12,
            embedding_size=768,
            hidden_size=768,
            intermediate_hidden_size=3072,
            attention_heads=12,
            activation_function="gelu",
            dropout=0.1,
            attention_dropout=0.1
        )
        assert config.layers == 12
        assert config.attention_heads == 12

    def test_layers_must_be_positive(self):
        """Number of layers must be greater than 0."""
        with pytest.raises(ValidationError):
            ModelConfig(
                layers=0,
                embedding_size=768,
                hidden_size=768,
                intermediate_hidden_size=3072,
                attention_heads=12,
                activation_function="gelu",
                dropout=0.1,
                attention_dropout=0.1
            )

    def test_dropout_range_validation(self):
        """Dropout must be in [0, 1)."""
        # Valid: 0.0
        config = ModelConfig(
            layers=12,
            embedding_size=768,
            hidden_size=768,
            intermediate_hidden_size=3072,
            attention_heads=12,
            activation_function="gelu",
            dropout=0.0,
            attention_dropout=0.0
        )
        assert config.dropout == 0.0

        # Invalid: 1.0 or higher
        with pytest.raises(ValidationError):
            ModelConfig(
                layers=12,
                embedding_size=768,
                hidden_size=768,
                intermediate_hidden_size=3072,
                attention_heads=12,
                activation_function="gelu",
                dropout=1.0,
                attention_dropout=0.1
            )

        # Invalid: negative
        with pytest.raises(ValidationError):
            ModelConfig(
                layers=12,
                embedding_size=768,
                hidden_size=768,
                intermediate_hidden_size=3072,
                attention_heads=12,
                activation_function="gelu",
                dropout=-0.1,
                attention_dropout=0.1
            )


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""

    def test_valid_training_config(self):
        """Valid training configuration."""
        config = TrainingConfig(
            output_dir="output/",
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            epochs=10
        )
        assert config.learning_rate == 1e-4
        assert config.epochs == 10

    def test_learning_rate_must_be_positive(self):
        """Learning rate must be greater than 0."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                output_dir="output/",
                learning_rate=0.0,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8,
                epochs=10
            )

    def test_epochs_must_be_positive(self):
        """Epochs must be greater than 0."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                output_dir="output/",
                learning_rate=1e-4,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8,
                epochs=0
            )

    def test_warmup_ratio_defaults(self):
        """Warmup ratio defaults to 0.1."""
        config = TrainingConfig(
            output_dir="output/",
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            epochs=10
        )
        assert config.warmup_ratio == 0.1

    def test_optional_train_steps(self):
        """train_steps is optional (calculated from epochs)."""
        config = TrainingConfig(
            output_dir="output/",
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            epochs=10
        )
        assert config.train_steps is None

    def test_optional_warmup_steps(self):
        """warmup_steps is optional (calculated from ratio)."""
        config = TrainingConfig(
            output_dir="output/",
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            epochs=10
        )
        assert config.warmup_steps is None

    def test_amp_defaults_to_false(self):
        """AMP should default to False."""
        config = TrainingConfig(
            output_dir="output/",
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            epochs=10
        )
        assert config.use_amp is False

    def test_gradient_accumulation_defaults_to_one(self):
        """Gradient accumulation steps defaults to 1."""
        config = TrainingConfig(
            output_dir="output/",
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            epochs=10
        )
        assert config.gradient_accumulation_steps == 1


class TestLoggingConfig:
    """Tests for LoggingConfig validation."""

    def test_valid_logging_config(self):
        """Valid logging configuration."""
        config = LoggingConfig(
            level="INFO",
            dir="logs",
            use_wandb=True,
            wandb_project="test_project"
        )
        assert config.level == "INFO"
        assert config.use_wandb is True

    def test_logging_defaults(self):
        """Test default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.dir == "logs"
        assert config.use_wandb is False
        assert config.wandb_project is None


class TestExperimentConfig:
    """Tests for complete ExperimentConfig."""

    def test_valid_complete_config(self, tiny_config):
        """Valid complete experiment configuration."""
        assert tiny_config.experiment_name == "test_exp"
        assert tiny_config.data.batch_size == 2
        assert tiny_config.model.layers == 2
        assert tiny_config.training.epochs == 1
        assert tiny_config.random_seed == 42

    def test_config_serialization(self, tiny_config):
        """Configuration can be serialized and deserialized."""
        config_dict = tiny_config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["experiment_name"] == "test_exp"

        # Recreate from dict
        config_reloaded = ExperimentConfig(**config_dict)
        assert config_reloaded.experiment_name == tiny_config.experiment_name
        assert config_reloaded.data.batch_size == tiny_config.data.batch_size

    def test_missing_required_fields(self):
        """Missing required fields should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(
                # Missing experiment_name and other required fields
                random_seed=42
            )
        assert "experiment_name" in str(exc_info.value)

    def test_nested_validation_failure(self, invalid_config_data):
        """Nested configuration errors should be caught."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**invalid_config_data)

        error_str = str(exc_info.value)
        # Should catch multiple validation errors
        assert any(field in error_str for field in ["batch_size", "vocab_size", "layers"])

    def test_dataset_manipulation_optional(self, tiny_config):
        """dataset_manipulation is optional."""
        assert tiny_config.dataset_manipulation == []

        # Can be set
        tiny_config.dataset_manipulation = [
            {"type": "remove_expletives", "input_path": "a", "output_path": "b"}
        ]
        assert len(tiny_config.dataset_manipulation) == 1

    def test_config_immutability_after_creation(self, tiny_config):
        """Config values can be modified (Pydantic allows this by default)."""
        # Note: If you want immutability, add frozen=True to BaseModel
        original_name = tiny_config.experiment_name
        tiny_config.experiment_name = "new_name"
        assert tiny_config.experiment_name == "new_name"
        assert tiny_config.experiment_name != original_name


class TestConfigValidationEdgeCases:
    """Edge case tests for configuration validation."""

    def test_very_large_values(self):
        """Very large but valid values should be accepted."""
        config = TrainingConfig(
            output_dir="output/",
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            epochs=1000000,  # Very large
            train_steps=10000000  # Very large
        )
        assert config.epochs == 1000000

    def test_very_small_positive_values(self):
        """Very small but positive values should be accepted."""
        config = TrainingConfig(
            output_dir="output/",
            learning_rate=1e-10,  # Very small but positive
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-15,  # Very small
            epochs=1
        )
        assert config.learning_rate == 1e-10

    def test_boundary_dropout_values(self):
        """Test dropout at boundary values."""
        # 0.0 should work
        config = ModelConfig(
            layers=12,
            embedding_size=768,
            hidden_size=768,
            intermediate_hidden_size=3072,
            attention_heads=12,
            activation_function="gelu",
            dropout=0.0,
            attention_dropout=0.0
        )
        assert config.dropout == 0.0

        # 0.999 should work
        config = ModelConfig(
            layers=12,
            embedding_size=768,
            hidden_size=768,
            intermediate_hidden_size=3072,
            attention_heads=12,
            activation_function="gelu",
            dropout=0.999,
            attention_dropout=0.999
        )
        assert config.dropout == 0.999
