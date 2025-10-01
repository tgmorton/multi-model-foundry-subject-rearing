"""
Unit tests for model creation.

Tests the model factory function that creates GPT-2 models from configuration.
"""

import pytest
import torch
from transformers import GPT2LMHeadModel

from model_foundry.model import create_model


class TestCreateModel:
    """Tests for create_model factory function."""

    def test_creates_gpt2_model(self, tiny_config):
        """Should create a GPT2LMHeadModel instance."""
        model = create_model(tiny_config)

        assert isinstance(model, GPT2LMHeadModel)

    def test_model_has_correct_vocab_size(self, tiny_config):
        """Model should have vocabulary size from config."""
        model = create_model(tiny_config)

        assert model.config.vocab_size == tiny_config.tokenizer.vocab_size

    def test_model_has_correct_layers(self, tiny_config):
        """Model should have number of layers from config."""
        model = create_model(tiny_config)

        assert model.config.n_layer == tiny_config.model.layers

    def test_model_has_correct_hidden_size(self, tiny_config):
        """Model should have hidden size from config."""
        model = create_model(tiny_config)

        assert model.config.n_embd == tiny_config.model.hidden_size

    def test_model_has_correct_attention_heads(self, tiny_config):
        """Model should have number of attention heads from config."""
        model = create_model(tiny_config)

        assert model.config.n_head == tiny_config.model.attention_heads

    def test_model_has_correct_intermediate_size(self, tiny_config):
        """Model should have intermediate hidden size from config."""
        model = create_model(tiny_config)

        assert model.config.n_inner == tiny_config.model.intermediate_hidden_size

    def test_model_has_correct_max_position_embeddings(self, tiny_config):
        """Model should have max positions from data config."""
        model = create_model(tiny_config)

        assert model.config.n_positions == tiny_config.data.max_sequence_length

    def test_model_has_correct_activation_function(self, tiny_config):
        """Model should use activation function from config."""
        model = create_model(tiny_config)

        assert model.config.activation_function == tiny_config.model.activation_function

    def test_model_has_correct_dropout(self, tiny_config):
        """Model should have dropout values from config."""
        model = create_model(tiny_config)

        assert model.config.resid_pdrop == tiny_config.model.dropout
        assert model.config.attn_pdrop == tiny_config.model.attention_dropout

    def test_model_cache_disabled(self, tiny_config):
        """Model should have caching disabled for training."""
        model = create_model(tiny_config)

        assert model.config.use_cache is False

    def test_model_has_parameters(self, tiny_config):
        """Model should have trainable parameters."""
        model = create_model(tiny_config)

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_model_parameters_are_trainable(self, tiny_config):
        """Model parameters should require gradients."""
        model = create_model(tiny_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        assert trainable_params == total_params

    def test_model_can_forward_pass(self, tiny_config):
        """Model should be able to perform forward pass."""
        model = create_model(tiny_config)

        # Create dummy input
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, tiny_config.tokenizer.vocab_size, (batch_size, seq_length))

        # Forward pass (without labels, just get logits)
        with torch.no_grad():
            outputs = model(input_ids)

        assert hasattr(outputs, 'logits')
        assert outputs.logits.shape == (batch_size, seq_length, tiny_config.tokenizer.vocab_size)

    def test_model_can_compute_loss(self, tiny_config):
        """Model should compute loss when labels provided."""
        model = create_model(tiny_config)

        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, tiny_config.tokenizer.vocab_size, (batch_size, seq_length))

        # Forward pass with labels
        outputs = model(input_ids, labels=input_ids)

        assert hasattr(outputs, 'loss')
        assert outputs.loss.item() > 0

    def test_model_can_backward_pass(self, tiny_config):
        """Model should support gradient computation."""
        model = create_model(tiny_config)

        batch_size = 2
        seq_length = 16
        input_ids = torch.randint(0, tiny_config.tokenizer.vocab_size, (batch_size, seq_length))

        # Forward and backward
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        # Check that gradients were computed
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_gradients

    def test_model_with_flash_attention(self, tiny_config):
        """Should accept flash_attention_2 kwarg."""
        try:
            model = create_model(tiny_config, attn_implementation="flash_attention_2")
            # If it doesn't raise, check that it's a valid model
            assert isinstance(model, GPT2LMHeadModel)
        except (ImportError, ValueError) as e:
            # Flash attention not available - expected
            pytest.skip(f"Flash Attention 2 not available: {e}")

    def test_model_with_custom_kwargs(self, tiny_config):
        """Should pass through additional kwargs."""
        # Test that we can pass additional arguments
        model = create_model(tiny_config, attn_implementation="eager")

        assert isinstance(model, GPT2LMHeadModel)


class TestCreateModelVariations:
    """Tests for model creation with different configurations."""

    def test_small_model(self, tiny_config):
        """Should create very small model."""
        tiny_config.model.layers = 1
        tiny_config.model.hidden_size = 32
        tiny_config.model.embedding_size = 32
        tiny_config.model.intermediate_hidden_size = 64
        tiny_config.model.attention_heads = 2

        model = create_model(tiny_config)

        total_params = sum(p.numel() for p in model.parameters())
        # Very small model should have fewer parameters
        assert total_params < 100_000

    def test_larger_model(self, tiny_config):
        """Should create larger model."""
        tiny_config.model.layers = 6
        tiny_config.model.hidden_size = 256
        tiny_config.model.embedding_size = 256
        tiny_config.model.intermediate_hidden_size = 1024
        tiny_config.model.attention_heads = 8

        model = create_model(tiny_config)

        total_params = sum(p.numel() for p in model.parameters())
        # Larger model should have more parameters
        assert total_params > 1_000_000

    def test_different_vocab_sizes(self, tiny_config):
        """Should handle different vocabulary sizes."""
        for vocab_size in [500, 1000, 5000]:
            tiny_config.tokenizer.vocab_size = vocab_size

            model = create_model(tiny_config)

            assert model.config.vocab_size == vocab_size

    def test_different_sequence_lengths(self, tiny_config):
        """Should handle different max sequence lengths."""
        for max_len in [64, 128, 256]:
            tiny_config.data.max_sequence_length = max_len

            model = create_model(tiny_config)

            assert model.config.n_positions == max_len

    def test_different_activation_functions(self, tiny_config):
        """Should support different activation functions."""
        for activation in ["gelu", "relu", "gelu_new"]:
            tiny_config.model.activation_function = activation

            model = create_model(tiny_config)

            assert model.config.activation_function == activation

    def test_different_dropout_values(self, tiny_config):
        """Should handle different dropout values."""
        for dropout in [0.0, 0.1, 0.3]:
            tiny_config.model.dropout = dropout
            tiny_config.model.attention_dropout = dropout

            model = create_model(tiny_config)

            assert model.config.resid_pdrop == dropout
            assert model.config.attn_pdrop == dropout


class TestCreateModelDevicePlacement:
    """Tests for model device placement."""

    def test_model_can_be_moved_to_cpu(self, tiny_config):
        """Model should be movable to CPU."""
        model = create_model(tiny_config)
        model = model.to("cpu")

        # Check a parameter is on CPU
        first_param = next(model.parameters())
        assert first_param.device.type == "cpu"

    @pytest.mark.gpu
    def test_model_can_be_moved_to_cuda(self, tiny_config, skip_if_no_cuda):
        """Model should be movable to CUDA."""
        model = create_model(tiny_config)
        model = model.to("cuda")

        # Check a parameter is on CUDA
        first_param = next(model.parameters())
        assert first_param.device.type == "cuda"

    def test_model_forward_on_cpu(self, tiny_config):
        """Model should work on CPU."""
        model = create_model(tiny_config).to("cpu")

        input_ids = torch.randint(0, tiny_config.tokenizer.vocab_size, (2, 16))
        input_ids = input_ids.to("cpu")

        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs.logits.device.type == "cpu"

    @pytest.mark.gpu
    def test_model_forward_on_cuda(self, tiny_config, skip_if_no_cuda):
        """Model should work on CUDA."""
        model = create_model(tiny_config).to("cuda")

        input_ids = torch.randint(0, tiny_config.tokenizer.vocab_size, (2, 16))
        input_ids = input_ids.to("cuda")

        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs.logits.device.type == "cuda"


class TestCreateModelEdgeCases:
    """Edge case tests for model creation."""

    def test_model_with_single_layer(self, tiny_config):
        """Should create model with just 1 layer."""
        tiny_config.model.layers = 1

        model = create_model(tiny_config)

        assert model.config.n_layer == 1

    def test_model_with_minimum_hidden_size(self, tiny_config):
        """Should create model with very small hidden size."""
        # Minimum that works with attention heads
        tiny_config.model.hidden_size = 16
        tiny_config.model.embedding_size = 16
        tiny_config.model.attention_heads = 2  # hidden_size must be divisible

        model = create_model(tiny_config)

        assert model.config.n_embd == 16

    def test_model_with_zero_dropout(self, tiny_config):
        """Should create model with no dropout."""
        tiny_config.model.dropout = 0.0
        tiny_config.model.attention_dropout = 0.0

        model = create_model(tiny_config)

        assert model.config.resid_pdrop == 0.0
        assert model.config.attn_pdrop == 0.0

    def test_multiple_models_independent(self, tiny_config):
        """Multiple models should be independent."""
        model1 = create_model(tiny_config)
        model2 = create_model(tiny_config)

        # Models should have different parameter values (random init)
        param1 = next(model1.parameters())
        param2 = next(model2.parameters())

        # Should not be the same object
        assert param1 is not param2

    def test_model_reproducible_with_seed(self, tiny_config, deterministic_seed):
        """Model creation should be reproducible with same seed."""
        from model_foundry.utils import set_seed

        set_seed(42)
        model1 = create_model(tiny_config)
        param1_values = next(model1.parameters()).clone()

        set_seed(42)
        model2 = create_model(tiny_config)
        param2_values = next(model2.parameters()).clone()

        assert torch.allclose(param1_values, param2_values)
