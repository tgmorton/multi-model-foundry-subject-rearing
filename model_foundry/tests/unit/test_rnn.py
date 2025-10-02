"""
Unit tests for RNN/LSTM/GRU architectures.

This module tests the RNN-based language model implementations including:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Vanilla RNN (Elman RNN)

Each architecture is tested in both unidirectional and bidirectional modes.
"""

import pytest
import torch
import yaml
import tempfile
from pathlib import Path

from model_foundry.architectures.rnn import (
    RNNLanguageModel,
    LSTMModel,
    GRUModel,
    VanillaRNNModel
)
from model_foundry.architectures import MODEL_REGISTRY, create_model_from_config
from model_foundry.config import ExperimentConfig


@pytest.fixture
def lstm_config():
    """Create a minimal LSTM configuration for testing."""
    config_dict = {
        'experiment_name': 'test_lstm',
        'data': {
            'source_corpus': 'test_corpus',
            'training_corpus': 'test_training',
            'batch_size': 4,
            'max_sequence_length': 128
        },
        'tokenizer': {
            'output_dir': 'test_tokenizer',
            'vocab_size': 1000,
            'tokenizer_type': 'sentencepiece'
        },
        'model': {
            'architecture': 'lstm',
            'rnn': {
                'embedding_size': 64,
                'hidden_size': 128,
                'num_layers': 2,
                'bidirectional': False,
                'dropout': 0.1,
                'rnn_type': 'lstm'
            }
        },
        'training': {
            'output_dir': 'test_output',
            'learning_rate': 0.001,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1e-8,
            'epochs': 1,
            'objective': 'causal_lm'
        },
        'logging': {
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'dir': 'test_logs'
        },
        'random_seed': 42
    }
    return ExperimentConfig(**config_dict)


@pytest.fixture
def bidirectional_lstm_config(lstm_config):
    """Create a bidirectional LSTM configuration."""
    config_dict = lstm_config.model_dump()
    config_dict['model']['rnn']['bidirectional'] = True
    config_dict['training']['objective'] = 'masked_lm'
    return ExperimentConfig(**config_dict)


@pytest.fixture
def gru_config(lstm_config):
    """Create a GRU configuration."""
    config_dict = lstm_config.model_dump()
    config_dict['model']['architecture'] = 'gru'
    config_dict['model']['rnn']['rnn_type'] = 'gru'
    return ExperimentConfig(**config_dict)


@pytest.fixture
def vanilla_rnn_config(lstm_config):
    """Create a vanilla RNN configuration."""
    config_dict = lstm_config.model_dump()
    config_dict['model']['architecture'] = 'rnn'
    config_dict['model']['rnn']['rnn_type'] = 'rnn'
    return ExperimentConfig(**config_dict)


# === Architecture Registration Tests ===

def test_lstm_registered():
    """Test that LSTM architecture is registered."""
    assert 'lstm' in MODEL_REGISTRY
    assert MODEL_REGISTRY['lstm'] == LSTMModel


def test_gru_registered():
    """Test that GRU architecture is registered."""
    assert 'gru' in MODEL_REGISTRY
    assert MODEL_REGISTRY['gru'] == GRUModel


def test_vanilla_rnn_registered():
    """Test that vanilla RNN architecture is registered."""
    assert 'rnn' in MODEL_REGISTRY
    assert MODEL_REGISTRY['rnn'] == VanillaRNNModel


# === Model Creation Tests ===

def test_lstm_from_config__creates_model(lstm_config):
    """Test creating LSTM model from configuration."""
    model = LSTMModel.from_config(lstm_config)

    assert isinstance(model, RNNLanguageModel)
    assert model.rnn_type == 'lstm'
    assert model.vocab_size == 1000
    assert model.embedding_size == 64
    assert model.hidden_size == 128
    assert model.num_layers == 2
    assert model.bidirectional == False
    assert model.dropout_prob == 0.1


def test_lstm_from_config__bidirectional(bidirectional_lstm_config):
    """Test creating bidirectional LSTM model."""
    model = LSTMModel.from_config(bidirectional_lstm_config)

    assert model.bidirectional == True
    assert not model.supports_generation  # Bidirectional cannot generate


def test_gru_from_config__creates_model(gru_config):
    """Test creating GRU model from configuration."""
    model = GRUModel.from_config(gru_config)

    assert isinstance(model, RNNLanguageModel)
    assert model.rnn_type == 'gru'


def test_vanilla_rnn_from_config__creates_model(vanilla_rnn_config):
    """Test creating vanilla RNN model from configuration."""
    model = VanillaRNNModel.from_config(vanilla_rnn_config)

    assert isinstance(model, RNNLanguageModel)
    assert model.rnn_type == 'rnn'


def test_create_model_from_config__lstm(lstm_config):
    """Test factory function creates LSTM model."""
    model = create_model_from_config(lstm_config)

    assert isinstance(model, LSTMModel)
    assert model.model_type == 'lstm'


def test_create_model_from_config__gru(gru_config):
    """Test factory function creates GRU model."""
    model = create_model_from_config(gru_config)

    assert isinstance(model, GRUModel)
    assert model.model_type == 'gru'


def test_create_model_from_config__rnn(vanilla_rnn_config):
    """Test factory function creates vanilla RNN model."""
    model = create_model_from_config(vanilla_rnn_config)

    assert isinstance(model, VanillaRNNModel)
    assert model.model_type == 'rnn'


# === Forward Pass Tests ===

def test_lstm_forward__without_labels(lstm_config):
    """Test LSTM forward pass without labels (no loss computation)."""
    model = LSTMModel.from_config(lstm_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))

    output = model(input_ids)

    assert output.logits is not None
    assert output.logits.shape == (batch_size, seq_length, 1000)
    assert output.loss is None
    assert output.hidden_states is not None
    assert output.attentions is None  # RNNs don't have attention


def test_lstm_forward__with_labels(lstm_config):
    """Test LSTM forward pass with labels (compute loss)."""
    model = LSTMModel.from_config(lstm_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    labels = torch.randint(0, 1000, (batch_size, seq_length))

    output = model(input_ids, labels=labels)

    assert output.loss is not None
    assert output.loss.dim() == 0  # Scalar loss
    assert output.logits.shape == (batch_size, seq_length, 1000)


def test_lstm_forward__with_attention_mask(lstm_config):
    """Test LSTM forward pass with attention mask for padding."""
    model = LSTMModel.from_config(lstm_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))

    # Create attention mask (first sequence: full, second: padded)
    attention_mask = torch.ones(batch_size, seq_length)
    attention_mask[1, 5:] = 0  # Pad second sequence after position 5

    output = model(input_ids, attention_mask=attention_mask)

    assert output.logits is not None
    assert output.logits.shape == (batch_size, seq_length, 1000)


def test_lstm_forward__with_masked_labels(lstm_config):
    """Test LSTM forward pass with masked labels (ignore certain positions)."""
    model = LSTMModel.from_config(lstm_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    labels = torch.randint(0, 1000, (batch_size, seq_length))

    # Mask some labels with -100 (should be ignored in loss)
    labels[:, :3] = -100

    output = model(input_ids, labels=labels)

    assert output.loss is not None
    assert not torch.isnan(output.loss)


def test_bidirectional_lstm_forward(bidirectional_lstm_config):
    """Test bidirectional LSTM forward pass."""
    model = LSTMModel.from_config(bidirectional_lstm_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    labels = torch.randint(0, 1000, (batch_size, seq_length))

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (batch_size, seq_length, 1000)
    assert output.loss is not None
    # Hidden states should be doubled in size for bidirectional
    assert output.hidden_states.shape[-1] == 128 * 2  # hidden_size * 2


def test_gru_forward(gru_config):
    """Test GRU forward pass."""
    model = GRUModel.from_config(gru_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    labels = torch.randint(0, 1000, (batch_size, seq_length))

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (batch_size, seq_length, 1000)
    assert output.loss is not None


def test_vanilla_rnn_forward(vanilla_rnn_config):
    """Test vanilla RNN forward pass."""
    model = VanillaRNNModel.from_config(vanilla_rnn_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    labels = torch.randint(0, 1000, (batch_size, seq_length))

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (batch_size, seq_length, 1000)
    assert output.loss is not None


# === Interface Compliance Tests ===

def test_lstm_get_input_embeddings(lstm_config):
    """Test getting input embeddings layer."""
    model = LSTMModel.from_config(lstm_config)

    embeddings = model.get_input_embeddings()

    assert embeddings is not None
    assert isinstance(embeddings, torch.nn.Embedding)
    assert embeddings.num_embeddings == 1000
    assert embeddings.embedding_dim == 64


def test_lstm_resize_token_embeddings__expand(lstm_config):
    """Test expanding vocabulary size."""
    model = LSTMModel.from_config(lstm_config)

    original_vocab_size = model.vocab_size
    new_vocab_size = 1200

    model.resize_token_embeddings(new_vocab_size)

    assert model.vocab_size == new_vocab_size
    assert model.embedding.num_embeddings == new_vocab_size
    assert model.output_projection.out_features == new_vocab_size

    # Test that model still works
    input_ids = torch.randint(0, new_vocab_size, (2, 10))
    output = model(input_ids)
    assert output.logits.shape[-1] == new_vocab_size


def test_lstm_resize_token_embeddings__shrink(lstm_config):
    """Test shrinking vocabulary size."""
    model = LSTMModel.from_config(lstm_config)

    new_vocab_size = 500

    model.resize_token_embeddings(new_vocab_size)

    assert model.vocab_size == new_vocab_size
    assert model.embedding.num_embeddings == new_vocab_size
    assert model.output_projection.out_features == new_vocab_size


def test_lstm_resize_token_embeddings__same_size(lstm_config):
    """Test resize with same size (should be no-op)."""
    model = LSTMModel.from_config(lstm_config)

    original_embedding = model.embedding
    model.resize_token_embeddings(1000)

    # Should be no-op, same object
    assert model.vocab_size == 1000


def test_lstm_model_type(lstm_config):
    """Test model_type property."""
    model = LSTMModel.from_config(lstm_config)
    assert model.model_type == 'lstm'


def test_gru_model_type(gru_config):
    """Test GRU model_type property."""
    model = GRUModel.from_config(gru_config)
    assert model.model_type == 'gru'


def test_rnn_model_type(vanilla_rnn_config):
    """Test vanilla RNN model_type property."""
    model = VanillaRNNModel.from_config(vanilla_rnn_config)
    assert model.model_type == 'rnn'


def test_lstm_supports_generation__unidirectional(lstm_config):
    """Test that unidirectional LSTM supports generation."""
    model = LSTMModel.from_config(lstm_config)
    assert model.supports_generation == True


def test_lstm_supports_generation__bidirectional(bidirectional_lstm_config):
    """Test that bidirectional LSTM does not support generation."""
    model = LSTMModel.from_config(bidirectional_lstm_config)
    assert model.supports_generation == False


def test_lstm_get_parameter_count(lstm_config):
    """Test parameter counting."""
    model = LSTMModel.from_config(lstm_config)

    param_count = model.get_parameter_count()

    assert param_count > 0
    assert isinstance(param_count, int)


def test_lstm_get_memory_footprint(lstm_config):
    """Test memory footprint calculation."""
    model = LSTMModel.from_config(lstm_config)

    footprint = model.get_memory_footprint()

    assert 'parameters' in footprint
    assert 'buffers' in footprint
    assert 'total' in footprint
    assert footprint['total'] == footprint['parameters'] + footprint['buffers']
    assert footprint['total'] > 0


# === Gradient Flow Tests ===

def test_lstm_backward_pass(lstm_config):
    """Test that gradients flow through LSTM correctly."""
    model = LSTMModel.from_config(lstm_config)

    input_ids = torch.randint(0, 1000, (2, 10))
    labels = torch.randint(0, 1000, (2, 10))

    output = model(input_ids, labels=labels)

    # Backward pass
    output.loss.backward()

    # Check that gradients exist
    assert model.embedding.weight.grad is not None
    assert model.output_projection.weight.grad is not None

    # Check gradients are not NaN
    assert not torch.isnan(model.embedding.weight.grad).any()
    assert not torch.isnan(model.output_projection.weight.grad).any()


def test_bidirectional_lstm_backward_pass(bidirectional_lstm_config):
    """Test gradients flow through bidirectional LSTM."""
    model = LSTMModel.from_config(bidirectional_lstm_config)

    input_ids = torch.randint(0, 1000, (2, 10))
    labels = torch.randint(0, 1000, (2, 10))

    output = model(input_ids, labels=labels)
    output.loss.backward()

    assert model.embedding.weight.grad is not None
    assert not torch.isnan(model.embedding.weight.grad).any()


# === Comparison Tests ===

def test_lstm_vs_gru__different_architectures(lstm_config, gru_config):
    """Test that LSTM and GRU produce different results (different architectures)."""
    lstm_model = LSTMModel.from_config(lstm_config)
    gru_model = GRUModel.from_config(gru_config)

    input_ids = torch.randint(0, 1000, (2, 10))

    # Use same random seed for fair comparison
    torch.manual_seed(42)
    lstm_output = lstm_model(input_ids)

    torch.manual_seed(42)
    gru_output = gru_model(input_ids)

    # Outputs should be different (different RNN cells)
    assert not torch.allclose(lstm_output.logits, gru_output.logits, atol=1e-5)


def test_unidirectional_vs_bidirectional__different_outputs(lstm_config, bidirectional_lstm_config):
    """Test that unidirectional and bidirectional LSTMs produce different outputs."""
    uni_model = LSTMModel.from_config(lstm_config)
    bi_model = LSTMModel.from_config(bidirectional_lstm_config)

    input_ids = torch.randint(0, 1000, (2, 10))

    uni_output = uni_model(input_ids)
    bi_output = bi_model(input_ids)

    # Outputs should be different
    assert uni_output.logits.shape == bi_output.logits.shape
    assert not torch.allclose(uni_output.logits, bi_output.logits, atol=1e-5)


# === Edge Cases ===

def test_lstm_single_token_sequence(lstm_config):
    """Test LSTM with single token sequence."""
    model = LSTMModel.from_config(lstm_config)

    input_ids = torch.randint(0, 1000, (2, 1))
    labels = torch.randint(0, 1000, (2, 1))

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (2, 1, 1000)
    assert output.loss is not None


def test_lstm_long_sequence(lstm_config):
    """Test LSTM with longer sequence."""
    model = LSTMModel.from_config(lstm_config)

    input_ids = torch.randint(0, 1000, (2, 100))
    labels = torch.randint(0, 1000, (2, 100))

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (2, 100, 1000)
    assert output.loss is not None


def test_lstm_batch_size_one(lstm_config):
    """Test LSTM with batch size of 1."""
    model = LSTMModel.from_config(lstm_config)

    input_ids = torch.randint(0, 1000, (1, 10))
    labels = torch.randint(0, 1000, (1, 10))

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (1, 10, 1000)
    assert output.loss is not None


def test_lstm_all_padding(lstm_config):
    """Test LSTM with all-padding sequence (edge case)."""
    model = LSTMModel.from_config(lstm_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)  # All pad tokens
    attention_mask = torch.zeros(batch_size, seq_length)  # All padding

    output = model(input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (batch_size, seq_length, 1000)
    # Should not crash, though output may not be meaningful


# === Configuration Validation Tests ===

def test_lstm_from_config__missing_rnn_config():
    """Test error when RNN config is missing."""
    config_dict = {
        'experiment_name': 'test_lstm',
        'data': {
            'source_corpus': 'test_corpus',
            'training_corpus': 'test_training',
            'batch_size': 4,
            'max_sequence_length': 128
        },
        'tokenizer': {
            'output_dir': 'test_tokenizer',
            'vocab_size': 1000
        },
        'model': {
            'architecture': 'lstm',
            # Missing 'rnn' config
        },
        'training': {
            'output_dir': 'test_output',
            'learning_rate': 0.001,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1e-8,
            'epochs': 1
        },
        'logging': {
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'dir': 'test_logs'
        },
        'random_seed': 42
    }

    with pytest.raises(ValueError, match="requires 'rnn' configuration"):
        config = ExperimentConfig(**config_dict)
        LSTMModel.from_config(config)


def test_rnn_invalid_type():
    """Test error with invalid RNN type."""
    with pytest.raises(ValueError, match="rnn_type must be"):
        RNNLanguageModel(
            vocab_size=1000,
            embedding_size=64,
            hidden_size=128,
            num_layers=2,
            rnn_type="invalid_type"
        )
