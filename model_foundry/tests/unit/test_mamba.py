"""
Unit tests for Mamba (Selective State Space Model) architecture.

Tests cover both PyTorch fallback (CPU/macOS) and mamba-ssm (Linux+CUDA) implementations.
GPU-specific tests are marked with @pytest.mark.gpu.
"""

import pytest
import torch
import warnings

from model_foundry.architectures.mamba import MambaModel, MAMBA_SSM_AVAILABLE
from model_foundry.architectures import MODEL_REGISTRY, create_model_from_config
from model_foundry.config import ExperimentConfig


# Check if CUDA is available for GPU tests
CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture
def mamba_config():
    """Create a minimal Mamba configuration for testing."""
    config_dict = {
        'experiment_name': 'test_mamba',
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
            'architecture': 'mamba',
            'mamba': {
                'd_model': 64,
                'n_layers': 2,
                'd_state': 8,   # Smaller for testing
                'd_conv': 4,
                'expand': 2,
                'dropout': 0.1
            }
        },
        'training': {
            'output_dir': 'test_output',
            'learning_rate': 0.0001,
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


# === Architecture Registration Tests ===

def test_mamba_registered():
    """Test that Mamba architecture is registered."""
    assert 'mamba' in MODEL_REGISTRY
    assert MODEL_REGISTRY['mamba'] == MambaModel


def test_mamba_in_registered_list():
    """Test that Mamba appears in registered architectures list."""
    from model_foundry.architectures import get_registered_architectures
    archs = get_registered_architectures()
    assert 'mamba' in archs


# === Model Creation Tests ===

def test_mamba_from_config__creates_model(mamba_config):
    """Test creating Mamba model from configuration."""
    model = MambaModel.from_config(mamba_config)

    assert isinstance(model, MambaModel)
    assert model.vocab_size == 1000
    assert model.d_model == 64
    assert model.n_layers == 2
    assert model.d_state == 8
    assert model.d_conv == 4
    assert model.expand == 2
    assert model.dropout_prob == 0.1


def test_mamba_from_config__default_values(mamba_config):
    """Test that default Mamba config values are applied."""
    model = MambaModel.from_config(mamba_config)

    # Default values from MambaModelConfig
    assert model.d_state == 8  # Set in test config
    assert model.d_conv == 4
    assert model.expand == 2


def test_create_model_from_config__mamba(mamba_config):
    """Test factory function creates Mamba model."""
    model = create_model_from_config(mamba_config)

    assert isinstance(model, MambaModel)
    assert model.model_type == 'mamba'


def test_mamba_wrong_architecture_raises_error():
    """Test that creating Mamba with wrong architecture raises error."""
    config_dict = {
        'experiment_name': 'test',
        'data': {
            'source_corpus': 'test',
            'training_corpus': 'test',
            'batch_size': 4,
            'max_sequence_length': 64
        },
        'tokenizer': {
            'output_dir': 'test',
            'vocab_size': 1000
        },
        'model': {
            'architecture': 'gpt2',  # Wrong architecture
            'transformer': {
                'layers': 2,
                'embedding_size': 64,
                'hidden_size': 64,
                'intermediate_hidden_size': 128,
                'attention_heads': 2,
                'activation_function': 'gelu',
                'dropout': 0.1,
                'attention_dropout': 0.1
            }
        },
        'training': {
            'output_dir': 'test',
            'learning_rate': 0.001,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1e-8,
            'epochs': 1
        },
        'logging': {
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'dir': 'test'
        },
        'random_seed': 42
    }

    config = ExperimentConfig(**config_dict)

    with pytest.raises(ValueError, match="Expected 'mamba'"):
        MambaModel.from_config(config)


def test_mamba_missing_config_raises_error():
    """Test that missing mamba config raises error when creating model."""
    config_dict = {
        'experiment_name': 'test',
        'data': {
            'source_corpus': 'test',
            'training_corpus': 'test',
            'batch_size': 4,
            'max_sequence_length': 64
        },
        'tokenizer': {
            'output_dir': 'test',
            'vocab_size': 1000
        },
        'model': {
            'architecture': 'mamba',
            # Missing mamba config
        },
        'training': {
            'output_dir': 'test',
            'learning_rate': 0.001,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1e-8,
            'epochs': 1
        },
        'logging': {
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'dir': 'test'
        },
        'random_seed': 42
    }

    config = ExperimentConfig(**config_dict)

    # Should raise when trying to create the model
    with pytest.raises(ValueError, match="requires 'mamba' configuration"):
        MambaModel.from_config(config)


# === Forward Pass Tests (CPU Compatible) ===

def test_mamba_forward__without_labels(mamba_config):
    """Test Mamba forward pass without labels (no loss computation)."""
    model = MambaModel.from_config(mamba_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))

    output = model(input_ids)

    assert output.logits is not None
    assert output.logits.shape == (batch_size, seq_length, 1000)
    assert output.loss is None
    assert output.hidden_states is not None
    assert output.attentions is None  # Mamba doesn't have attention


def test_mamba_forward__with_labels(mamba_config):
    """Test Mamba forward pass with labels (compute loss)."""
    model = MambaModel.from_config(mamba_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    labels = torch.randint(0, 1000, (batch_size, seq_length))

    output = model(input_ids, labels=labels)

    assert output.loss is not None
    assert output.loss.dim() == 0  # Scalar loss
    assert output.logits.shape == (batch_size, seq_length, 1000)
    assert not torch.isnan(output.loss)


def test_mamba_forward__with_attention_mask(mamba_config):
    """Test Mamba forward pass with attention mask (currently not used but accepted)."""
    model = MambaModel.from_config(mamba_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    # Note: Current implementation doesn't use attention_mask
    # but should accept it for interface compatibility
    output = model(input_ids, attention_mask=attention_mask)

    assert output.logits is not None
    assert output.logits.shape == (batch_size, seq_length, 1000)


def test_mamba_forward__masked_labels(mamba_config):
    """Test Mamba forward pass with masked labels (-100 ignored)."""
    model = MambaModel.from_config(mamba_config)

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    labels = torch.randint(0, 1000, (batch_size, seq_length))

    # Mask some labels
    labels[:, :3] = -100

    output = model(input_ids, labels=labels)

    assert output.loss is not None
    assert not torch.isnan(output.loss)


# === Interface Compliance Tests ===

def test_mamba_model_type(mamba_config):
    """Test model_type property."""
    model = MambaModel.from_config(mamba_config)
    assert model.model_type == 'mamba'


def test_mamba_supports_generation(mamba_config):
    """Test that Mamba supports generation (causal model)."""
    model = MambaModel.from_config(mamba_config)
    assert model.supports_generation == True


def test_mamba_get_input_embeddings(mamba_config):
    """Test getting input embeddings layer."""
    model = MambaModel.from_config(mamba_config)

    embeddings = model.get_input_embeddings()

    assert embeddings is not None
    assert isinstance(embeddings, torch.nn.Embedding)
    assert embeddings.num_embeddings == 1000
    assert embeddings.embedding_dim == 64


def test_mamba_resize_token_embeddings__expand(mamba_config):
    """Test expanding vocabulary size."""
    model = MambaModel.from_config(mamba_config)

    original_vocab_size = model.vocab_size
    new_vocab_size = 1200

    model.resize_token_embeddings(new_vocab_size)

    assert model.vocab_size == new_vocab_size
    assert model.embedding.num_embeddings == new_vocab_size
    assert model.lm_head.out_features == new_vocab_size

    # Test that model still works
    input_ids = torch.randint(0, new_vocab_size, (2, 10))
    output = model(input_ids)
    assert output.logits.shape[-1] == new_vocab_size


def test_mamba_resize_token_embeddings__shrink(mamba_config):
    """Test shrinking vocabulary size."""
    model = MambaModel.from_config(mamba_config)

    new_vocab_size = 500

    model.resize_token_embeddings(new_vocab_size)

    assert model.vocab_size == new_vocab_size
    assert model.embedding.num_embeddings == new_vocab_size


def test_mamba_resize_token_embeddings__same_size(mamba_config):
    """Test resize with same size (should be no-op)."""
    model = MambaModel.from_config(mamba_config)

    model.resize_token_embeddings(1000)

    assert model.vocab_size == 1000


def test_mamba_get_parameter_count(mamba_config):
    """Test parameter counting."""
    model = MambaModel.from_config(mamba_config)

    param_count = model.get_parameter_count()

    assert param_count > 0
    assert isinstance(param_count, int)


def test_mamba_get_memory_footprint(mamba_config):
    """Test memory footprint calculation."""
    model = MambaModel.from_config(mamba_config)

    footprint = model.get_memory_footprint()

    assert 'parameters' in footprint
    assert 'buffers' in footprint
    assert 'total' in footprint
    assert footprint['total'] > 0


# === Gradient Flow Tests ===

def test_mamba_backward_pass(mamba_config):
    """Test that gradients flow through Mamba correctly."""
    model = MambaModel.from_config(mamba_config)

    input_ids = torch.randint(0, 1000, (2, 10))
    labels = torch.randint(0, 1000, (2, 10))

    output = model(input_ids, labels=labels)

    # Backward pass
    output.loss.backward()

    # Check that gradients exist
    assert model.embedding.weight.grad is not None
    assert not torch.isnan(model.embedding.weight.grad).any()


# === Model Behavior Tests ===

def test_mamba_train_eval_modes(mamba_config):
    """Test that Mamba supports train/eval modes."""
    model = MambaModel.from_config(mamba_config)

    # Set to train mode
    model.train()
    assert model.training

    # Set to eval mode
    model.eval()
    assert not model.training


def test_mamba_deterministic_with_seed(mamba_config):
    """Test that Mamba produces deterministic outputs with fixed seed."""
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))

    # First run
    torch.manual_seed(42)
    model1 = MambaModel.from_config(mamba_config)
    torch.manual_seed(42)  # Reset for inference
    output1 = model1(input_ids)

    # Second run with same seed
    torch.manual_seed(42)
    model2 = MambaModel.from_config(mamba_config)
    torch.manual_seed(42)  # Reset for inference
    output2 = model2(input_ids)

    # Should be identical
    assert torch.allclose(output1.logits, output2.logits, atol=1e-6)


def test_mamba_valid_outputs(mamba_config):
    """Test that Mamba produces valid (non-NaN, finite) outputs."""
    model = MambaModel.from_config(mamba_config)

    input_ids = torch.randint(0, 1000, (2, 10))
    labels = torch.randint(0, 1000, (2, 10))

    output = model(input_ids, labels=labels)

    # Check logits
    assert not torch.isnan(output.logits).any()
    assert torch.isfinite(output.logits).all()

    # Check loss
    assert not torch.isnan(output.loss)
    assert torch.isfinite(output.loss)
    assert output.loss > 0


# === Edge Cases ===

def test_mamba_single_token_sequence(mamba_config):
    """Test Mamba with single token sequence."""
    model = MambaModel.from_config(mamba_config)

    input_ids = torch.randint(0, 1000, (2, 1))
    labels = torch.randint(0, 1000, (2, 1))

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (2, 1, 1000)
    assert output.loss is not None


def test_mamba_long_sequence(mamba_config):
    """Test Mamba with longer sequence (Mamba handles long sequences well)."""
    model = MambaModel.from_config(mamba_config)

    input_ids = torch.randint(0, 1000, (2, 256))
    labels = torch.randint(0, 1000, (2, 256))

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (2, 256, 1000)
    assert output.loss is not None


def test_mamba_batch_size_one(mamba_config):
    """Test Mamba with batch size of 1."""
    model = MambaModel.from_config(mamba_config)

    input_ids = torch.randint(0, 1000, (1, 10))
    labels = torch.randint(0, 1000, (1, 10))

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (1, 10, 1000)
    assert output.loss is not None


# === Implementation Detection Tests ===

def test_mamba_implementation_detection():
    """Test that we can detect which Mamba implementation is being used."""
    # Should be False on most systems (mamba-ssm requires Linux+CUDA)
    # Will be True on Linux systems with mamba-ssm installed
    assert isinstance(MAMBA_SSM_AVAILABLE, bool)


def test_mamba_uses_correct_implementation(mamba_config):
    """Test that Mamba uses appropriate implementation."""
    model = MambaModel.from_config(mamba_config)

    # On systems without mamba-ssm, should use PyTorch fallback
    # On Linux+CUDA with mamba-ssm, should use optimized version
    if MAMBA_SSM_AVAILABLE:
        assert model._use_fast == True
    else:
        assert model._use_fast == False


# === GPU-Only Tests (Marked) ===

@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_mamba_cuda_forward_pass(mamba_config):
    """
    Test Mamba forward pass on CUDA (GPU only).

    This test requires CUDA and will be skipped on CPU-only systems.
    """
    model = MambaModel.from_config(mamba_config)
    model = model.cuda()

    input_ids = torch.randint(0, 1000, (2, 10)).cuda()
    labels = torch.randint(0, 1000, (2, 10)).cuda()

    output = model(input_ids, labels=labels)

    assert output.logits.is_cuda
    assert output.loss.is_cuda
    assert output.logits.shape == (2, 10, 1000)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_mamba_cuda_backward_pass(mamba_config):
    """
    Test Mamba backward pass on CUDA (GPU only).

    This test verifies gradient flow on GPU.
    """
    model = MambaModel.from_config(mamba_config)
    model = model.cuda()

    input_ids = torch.randint(0, 1000, (2, 10)).cuda()
    labels = torch.randint(0, 1000, (2, 10)).cuda()

    output = model(input_ids, labels=labels)
    output.loss.backward()

    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad.is_cuda
    assert not torch.isnan(model.embedding.weight.grad).any()


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_mamba_long_sequence_cuda(mamba_config):
    """
    Test Mamba with very long sequence on GPU.

    Mamba's linear complexity makes it efficient for long sequences.
    This test requires GPU memory.
    """
    model = MambaModel.from_config(mamba_config)
    model = model.cuda()

    # Very long sequence (1024 tokens)
    input_ids = torch.randint(0, 1000, (2, 1024)).cuda()
    labels = torch.randint(0, 1000, (2, 1024)).cuda()

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (2, 1024, 1000)
    assert output.loss is not None


@pytest.mark.gpu
@pytest.mark.skipif(not MAMBA_SSM_AVAILABLE or not CUDA_AVAILABLE,
                    reason="Requires mamba-ssm and CUDA")
def test_mamba_ssm_optimized_kernels(mamba_config):
    """
    Test that mamba-ssm optimized kernels are being used (GPU + mamba-ssm only).

    This test will only run on Linux with CUDA and mamba-ssm installed.
    """
    model = MambaModel.from_config(mamba_config, use_fast=True)
    model = model.cuda()

    assert model._use_fast == True

    input_ids = torch.randint(0, 1000, (2, 128)).cuda()
    labels = torch.randint(0, 1000, (2, 128)).cuda()

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (2, 128, 1000)
    assert output.loss is not None
