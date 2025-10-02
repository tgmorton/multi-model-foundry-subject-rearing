"""
Unit tests for multi-architecture support.

Tests the architecture abstraction layer, registry pattern, and factory functionality.
"""

import pytest
from pydantic import ValidationError

from model_foundry.architectures import (
    BaseLanguageModel,
    ModelOutput,
    register_architecture,
    get_registered_architectures,
    create_model_from_config,
    MODEL_REGISTRY,
)
from model_foundry.architectures.gpt import GPT2Model
from model_foundry.config import ExperimentConfig, ModelConfig, TransformerModelConfig


class TestModelOutput:
    """Tests for ModelOutput class."""

    def test_create_model_output(self):
        """Should create ModelOutput with all fields."""
        import torch

        loss = torch.tensor(1.5)
        logits = torch.randn(2, 10, 1000)
        hidden_states = torch.randn(2, 10, 768)

        output = ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=None
        )

        assert output.loss is loss
        assert output.logits is logits
        assert output.hidden_states is hidden_states
        assert output.attentions is None

    def test_model_output_dict_access(self):
        """Should support dictionary-style access."""
        import torch

        loss = torch.tensor(2.0)
        output = ModelOutput(loss=loss)

        assert output["loss"] is loss
        assert output["logits"] is None

    def test_model_output_contains(self):
        """Should support 'in' operator."""
        import torch

        output = ModelOutput(loss=torch.tensor(1.0), logits=torch.randn(2, 10, 1000))

        assert "loss" in output
        assert "logits" in output
        assert "nonexistent" not in output


class TestArchitectureRegistry:
    """Tests for model registry pattern."""

    def test_gpt2_registered(self):
        """GPT-2 architecture should be registered."""
        architectures = get_registered_architectures()

        assert "gpt2" in architectures

    def test_get_registered_architectures_returns_list(self):
        """Should return list of architecture names."""
        architectures = get_registered_architectures()

        assert isinstance(architectures, list)
        assert len(architectures) > 0

    def test_model_registry_contains_classes(self):
        """Registry should map names to classes."""
        assert "gpt2" in MODEL_REGISTRY
        assert MODEL_REGISTRY["gpt2"] == GPT2Model

    def test_register_duplicate_architecture_fails(self):
        """Should not allow duplicate registration."""
        with pytest.raises(ValueError, match="already registered"):
            @register_architecture("gpt2")
            class DuplicateModel(BaseLanguageModel):
                pass

    def test_register_non_baselanguagemodel_fails(self):
        """Should only allow BaseLanguageModel subclasses."""
        with pytest.raises(TypeError, match="must inherit from BaseLanguageModel"):
            @register_architecture("invalid")
            class NotALanguageModel:
                pass


class TestCreateModelFromConfig:
    """Tests for model factory function."""

    def test_creates_model_for_gpt2_architecture(self, tiny_config):
        """Should create GPT2Model for gpt2 architecture."""
        model = create_model_from_config(tiny_config)

        assert isinstance(model, GPT2Model)
        assert isinstance(model, BaseLanguageModel)

    def test_fails_without_architecture_field(self, tiny_config):
        """Should fail if architecture field is missing."""
        # Remove architecture field
        tiny_config.model.architecture = None

        with pytest.raises((ValueError, ValidationError)):
            create_model_from_config(tiny_config)

    def test_fails_with_unknown_architecture(self, tiny_config):
        """Should fail if architecture is not registered."""
        # Create a config with unknown architecture
        # This will fail at validation, so we need to bypass validation
        import copy
        config_dict = tiny_config.model_dump()
        config_dict['model']['architecture'] = "unknown_arch"

        # Force the architecture value (bypassing validation)
        tiny_config.model.architecture = "unknown_arch"

        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model_from_config(tiny_config)

    def test_passes_kwargs_to_model(self, tiny_config):
        """Should pass kwargs to model constructor."""
        # Test that kwargs are passed through
        model = create_model_from_config(tiny_config, attn_implementation="eager")

        assert isinstance(model, GPT2Model)

    def test_returns_model_with_correct_type(self, tiny_config):
        """Created model should have correct model_type."""
        model = create_model_from_config(tiny_config)

        assert model.model_type == "gpt2"


class TestGPT2Model:
    """Tests for GPT2Model wrapper."""

    def test_implements_base_language_model(self, tiny_config):
        """GPT2Model should implement BaseLanguageModel interface."""
        model = create_model_from_config(tiny_config)

        assert isinstance(model, BaseLanguageModel)
        assert hasattr(model, 'forward')
        assert hasattr(model, 'get_input_embeddings')
        assert hasattr(model, 'resize_token_embeddings')
        assert hasattr(model, 'model_type')
        assert hasattr(model, 'supports_generation')

    def test_model_type_is_gpt2(self, tiny_config):
        """GPT2Model should have model_type 'gpt2'."""
        model = create_model_from_config(tiny_config)

        assert model.model_type == "gpt2"

    def test_supports_generation(self, tiny_config):
        """GPT2Model should support generation."""
        model = create_model_from_config(tiny_config)

        assert model.supports_generation is True

    def test_forward_pass_returns_model_output(self, tiny_config):
        """Forward pass should return ModelOutput."""
        import torch

        model = create_model_from_config(tiny_config)

        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, tiny_config.tokenizer.vocab_size, (batch_size, seq_length))

        with torch.no_grad():
            output = model(input_ids)

        assert isinstance(output, ModelOutput)
        assert output.logits is not None
        assert output.logits.shape == (batch_size, seq_length, tiny_config.tokenizer.vocab_size)

    def test_forward_with_labels_returns_loss(self, tiny_config):
        """Forward with labels should compute loss."""
        import torch

        model = create_model_from_config(tiny_config)

        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, tiny_config.tokenizer.vocab_size, (batch_size, seq_length))

        output = model(input_ids, labels=input_ids)

        assert isinstance(output, ModelOutput)
        assert output.loss is not None
        assert output.loss.item() > 0

    def test_get_input_embeddings(self, tiny_config):
        """Should return embedding layer."""
        import torch.nn as nn

        model = create_model_from_config(tiny_config)
        embeddings = model.get_input_embeddings()

        assert isinstance(embeddings, nn.Module)

    def test_resize_token_embeddings(self, tiny_config):
        """Should resize vocabulary."""
        model = create_model_from_config(tiny_config)

        original_vocab_size = model.config.vocab_size
        new_vocab_size = original_vocab_size + 100

        model.resize_token_embeddings(new_vocab_size)

        assert model.config.vocab_size == new_vocab_size

    def test_get_parameter_count(self, tiny_config):
        """Should return parameter count."""
        model = create_model_from_config(tiny_config)

        param_count = model.get_parameter_count()

        assert isinstance(param_count, int)
        assert param_count > 0

    def test_get_memory_footprint(self, tiny_config):
        """Should return memory statistics."""
        model = create_model_from_config(tiny_config)

        memory_stats = model.get_memory_footprint()

        assert "parameters" in memory_stats
        assert "buffers" in memory_stats
        assert "total" in memory_stats
        assert memory_stats["total"] > 0

    def test_generate_method_exists(self, tiny_config):
        """Should have generate method."""
        model = create_model_from_config(tiny_config)

        assert hasattr(model, 'generate')
        assert callable(model.generate)


class TestConfigValidation:
    """Tests for multi-architecture config validation."""

    def test_gpt2_requires_transformer_config(self):
        """GPT-2 architecture should require transformer config."""
        from model_foundry.config import DataConfig, TokenizerConfig, TrainingConfig, LoggingConfig

        with pytest.raises(ValidationError, match="requires 'transformer' configuration"):
            config = ExperimentConfig(
                experiment_name="test",
                data=DataConfig(
                    source_corpus="test",
                    training_corpus="test",
                    batch_size=32,
                    max_sequence_length=512
                ),
                tokenizer=TokenizerConfig(
                    output_dir="test",
                    vocab_size=1000
                ),
                model=ModelConfig(
                    architecture="gpt2",
                    transformer=None  # Missing required transformer config
                ),
                training=TrainingConfig(
                    output_dir="test",
                    learning_rate=1e-4,
                    adam_beta1=0.9,
                    adam_beta2=0.999,
                    adam_epsilon=1e-8,
                    epochs=1
                ),
                logging=LoggingConfig(),
                random_seed=42
            )

    def test_architecture_field_required(self):
        """Architecture field should be required."""
        from model_foundry.config import DataConfig, TokenizerConfig, TrainingConfig, LoggingConfig

        with pytest.raises(ValidationError):
            config = ExperimentConfig(
                experiment_name="test",
                data=DataConfig(
                    source_corpus="test",
                    training_corpus="test",
                    batch_size=32,
                    max_sequence_length=512
                ),
                tokenizer=TokenizerConfig(
                    output_dir="test",
                    vocab_size=1000
                ),
                model=ModelConfig(
                    # Missing architecture field
                    transformer=TransformerModelConfig(
                        layers=2,
                        embedding_size=64,
                        hidden_size=64,
                        intermediate_hidden_size=128,
                        attention_heads=2,
                        activation_function="gelu",
                        dropout=0.1,
                        attention_dropout=0.1
                    )
                ),
                training=TrainingConfig(
                    output_dir="test",
                    learning_rate=1e-4,
                    adam_beta1=0.9,
                    adam_beta2=0.999,
                    adam_epsilon=1e-8,
                    epochs=1
                ),
                logging=LoggingConfig(),
                random_seed=42
            )

    def test_valid_gpt2_config(self, tiny_config):
        """Valid GPT-2 config should load successfully."""
        assert tiny_config.model.architecture == "gpt2"
        assert tiny_config.model.transformer is not None
        assert tiny_config.model.transformer.layers == 2

    def test_model_config_convenience_properties(self, tiny_config):
        """ModelConfig should provide convenience properties."""
        model_config = tiny_config.model

        assert model_config.layers == 2
        assert model_config.embedding_size == 64
        assert model_config.hidden_size == 64
        assert model_config.attention_heads == 2
        assert model_config.dropout == 0.1
