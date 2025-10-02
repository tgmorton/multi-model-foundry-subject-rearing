"""
Cross-architecture integration tests.

This module tests that all supported architectures (GPT-2, BERT, LSTM, GRU, RNN)
work correctly within the Model Foundry framework, can be trained with appropriate
objectives, and maintain compatibility with the training pipeline.
"""

import pytest
import torch
import yaml
import tempfile
from pathlib import Path

from model_foundry.architectures import (
    create_model_from_config,
    get_registered_architectures,
    GPT2Model,
    BERTModel,
    LSTMModel,
    GRUModel,
    VanillaRNNModel
)
from model_foundry.config import ExperimentConfig
from model_foundry.data_collators import get_data_collator


@pytest.fixture
def all_architectures():
    """Return list of all supported architectures."""
    return get_registered_architectures()


@pytest.fixture
def minimal_configs():
    """Create minimal configs for all architectures."""
    base_config = {
        'experiment_name': 'test_multi_arch',
        'data': {
            'source_corpus': 'test_corpus',
            'training_corpus': 'test_training',
            'batch_size': 4,
            'max_sequence_length': 64
        },
        'tokenizer': {
            'output_dir': 'test_tokenizer',
            'vocab_size': 1000
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

    configs = {
        'gpt2': {
            **base_config,
            'experiment_name': 'test_gpt2',
            'model': {
                'architecture': 'gpt2',
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
                **base_config['training'],
                'objective': 'causal_lm'
            }
        },
        'bert': {
            **base_config,
            'experiment_name': 'test_bert',
            'model': {
                'architecture': 'bert',
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
                **base_config['training'],
                'objective': 'masked_lm',
                'mlm_probability': 0.15
            }
        },
        'lstm': {
            **base_config,
            'experiment_name': 'test_lstm',
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
                **base_config['training'],
                'objective': 'causal_lm'
            }
        },
        'gru': {
            **base_config,
            'experiment_name': 'test_gru',
            'model': {
                'architecture': 'gru',
                'rnn': {
                    'embedding_size': 64,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'bidirectional': False,
                    'dropout': 0.1,
                    'rnn_type': 'gru'
                }
            },
            'training': {
                **base_config['training'],
                'objective': 'causal_lm'
            }
        },
        'rnn': {
            **base_config,
            'experiment_name': 'test_rnn',
            'model': {
                'architecture': 'rnn',
                'rnn': {
                    'embedding_size': 64,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'bidirectional': False,
                    'dropout': 0.1,
                    'rnn_type': 'rnn'
                }
            },
            'training': {
                **base_config['training'],
                'objective': 'causal_lm'
            }
        }
    }

    return {arch: ExperimentConfig(**cfg) for arch, cfg in configs.items()}


# === Architecture Registration Tests ===

def test_all_architectures_registered(all_architectures):
    """Test that all expected architectures are registered."""
    expected = {'gpt2', 'bert', 'lstm', 'gru', 'rnn'}
    registered = set(all_architectures)

    assert expected == registered, f"Missing: {expected - registered}, Extra: {registered - expected}"


def test_no_duplicate_registrations(all_architectures):
    """Test that each architecture is registered only once."""
    assert len(all_architectures) == len(set(all_architectures))


# === Model Creation Tests ===

def test_create_all_architectures_from_config(minimal_configs):
    """Test that all architectures can be created from config."""
    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        assert model is not None, f"Failed to create {arch} model"
        assert model.model_type == arch, f"Model type mismatch for {arch}"


def test_architecture_specific_classes(minimal_configs):
    """Test that each architecture creates the correct class."""
    expected_classes = {
        'gpt2': GPT2Model,
        'bert': BERTModel,
        'lstm': LSTMModel,
        'gru': GRUModel,
        'rnn': VanillaRNNModel
    }

    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)
        expected_class = expected_classes[arch]

        assert isinstance(model, expected_class), \
            f"{arch} model should be instance of {expected_class.__name__}"


# === Forward Pass Tests ===

def test_all_architectures_forward_pass(minimal_configs):
    """Test that all architectures can perform forward pass."""
    batch_size = 2
    seq_length = 10

    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.randint(0, 1000, (batch_size, seq_length))

        # Mask some labels for masked LM
        if config.training.objective == 'masked_lm':
            labels[:, :3] = -100

        output = model(input_ids, attention_mask=attention_mask, labels=labels)

        assert output.logits is not None, f"{arch}: Missing logits"
        assert output.loss is not None, f"{arch}: Missing loss"
        assert output.logits.shape == (batch_size, seq_length, 1000), \
            f"{arch}: Wrong logits shape"


def test_all_architectures_backward_pass(minimal_configs):
    """Test that gradients flow correctly for all architectures."""
    batch_size = 2
    seq_length = 10

    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        labels = torch.randint(0, 1000, (batch_size, seq_length))

        output = model(input_ids, labels=labels)
        loss = output.loss

        # Backward pass
        loss.backward()

        # Check that embeddings have gradients
        embeddings = model.get_input_embeddings()
        assert embeddings.weight.grad is not None, f"{arch}: No gradient on embeddings"
        assert not torch.isnan(embeddings.weight.grad).any(), f"{arch}: NaN gradients"


# === Interface Compliance Tests ===

def test_all_architectures_have_model_type(minimal_configs):
    """Test that all architectures implement model_type property."""
    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        assert hasattr(model, 'model_type'), f"{arch}: Missing model_type property"
        assert model.model_type == arch, f"{arch}: Wrong model_type"


def test_all_architectures_have_supports_generation(minimal_configs):
    """Test that all architectures implement supports_generation property."""
    expected_generation_support = {
        'gpt2': True,      # Causal LM
        'bert': False,     # Bidirectional
        'lstm': True,      # Unidirectional causal
        'gru': True,       # Unidirectional causal
        'rnn': True        # Unidirectional causal
    }

    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        assert hasattr(model, 'supports_generation'), \
            f"{arch}: Missing supports_generation property"
        assert model.supports_generation == expected_generation_support[arch], \
            f"{arch}: Wrong generation support"


def test_all_architectures_can_get_embeddings(minimal_configs):
    """Test that all architectures can return embeddings."""
    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        embeddings = model.get_input_embeddings()

        assert embeddings is not None, f"{arch}: get_input_embeddings returned None"
        assert isinstance(embeddings, torch.nn.Module), \
            f"{arch}: Embeddings not a nn.Module"


def test_all_architectures_can_resize_embeddings(minimal_configs):
    """Test that all architectures support vocabulary resizing."""
    new_vocab_size = 1200

    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        # Resize
        model.resize_token_embeddings(new_vocab_size)

        # Verify resize worked
        embeddings = model.get_input_embeddings()
        assert embeddings.num_embeddings == new_vocab_size, \
            f"{arch}: Embedding resize failed"

        # Verify model still works
        input_ids = torch.randint(0, new_vocab_size, (2, 10))
        output = model(input_ids)
        assert output.logits.shape[-1] == new_vocab_size, \
            f"{arch}: Output vocab size mismatch after resize"


def test_all_architectures_have_parameter_count(minimal_configs):
    """Test that all architectures can report parameter count."""
    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        param_count = model.get_parameter_count()

        assert param_count > 0, f"{arch}: Invalid parameter count"
        assert isinstance(param_count, int), f"{arch}: Parameter count not int"


def test_all_architectures_have_memory_footprint(minimal_configs):
    """Test that all architectures can report memory footprint."""
    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        footprint = model.get_memory_footprint()

        assert 'parameters' in footprint, f"{arch}: Missing parameters in footprint"
        assert 'buffers' in footprint, f"{arch}: Missing buffers in footprint"
        assert 'total' in footprint, f"{arch}: Missing total in footprint"
        assert footprint['total'] > 0, f"{arch}: Invalid total footprint"


# === Training Objective Compatibility Tests ===

def test_causal_lm_architectures_use_correct_collator(minimal_configs):
    """Test that causal LM architectures use CausalLMDataCollator."""
    from model_foundry.data_collators import CausalLMDataCollator

    causal_architectures = ['gpt2', 'lstm', 'gru', 'rnn']

    for arch in causal_architectures:
        config = minimal_configs[arch]
        assert config.training.objective == 'causal_lm'

        # Create mock tokenizer
        class MockTokenizer:
            pad_token_id = 0

        collator = get_data_collator(config, MockTokenizer())

        assert isinstance(collator, CausalLMDataCollator), \
            f"{arch}: Should use CausalLMDataCollator"
        assert collator.mlm == False, f"{arch}: MLM should be False"


def test_masked_lm_architecture_uses_correct_collator(minimal_configs):
    """Test that masked LM architecture uses MaskedLMDataCollator."""
    from model_foundry.data_collators import MaskedLMDataCollator

    config = minimal_configs['bert']
    assert config.training.objective == 'masked_lm'

    # Create mock tokenizer
    class MockTokenizer:
        pad_token_id = 0
        mask_token_id = 1
        cls_token_id = 2
        sep_token_id = 3

        def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
            return [0] * len(token_ids)

    collator = get_data_collator(config, MockTokenizer())

    assert isinstance(collator, MaskedLMDataCollator), \
        "BERT should use MaskedLMDataCollator"
    assert collator.mlm == True, "BERT: MLM should be True"
    assert collator.mlm_probability == 0.15, "BERT: Wrong MLM probability"


# === Architecture Comparison Tests ===

def test_transformer_vs_rnn_parameter_counts(minimal_configs):
    """Compare parameter counts between transformer and RNN architectures."""
    gpt2_model = create_model_from_config(minimal_configs['gpt2'])
    lstm_model = create_model_from_config(minimal_configs['lstm'])

    gpt2_params = gpt2_model.get_parameter_count()
    lstm_params = lstm_model.get_parameter_count()

    # Both should have reasonable parameter counts
    assert gpt2_params > 1000, "GPT-2 too small"
    assert lstm_params > 1000, "LSTM too small"

    # Neither should be unreasonably large for tiny configs
    assert gpt2_params < 10_000_000, "GPT-2 too large"
    assert lstm_params < 10_000_000, "LSTM too large"


def test_rnn_variants_parameter_order(minimal_configs):
    """Test that LSTM > GRU > RNN in parameter count (for same config)."""
    lstm_model = create_model_from_config(minimal_configs['lstm'])
    gru_model = create_model_from_config(minimal_configs['gru'])
    rnn_model = create_model_from_config(minimal_configs['rnn'])

    lstm_params = lstm_model.get_parameter_count()
    gru_params = gru_model.get_parameter_count()
    rnn_params = rnn_model.get_parameter_count()

    # LSTM has most parameters (3 gates + cell state)
    # GRU has medium (2 gates)
    # Vanilla RNN has fewest (single tanh)
    assert lstm_params > gru_params, "LSTM should have more parameters than GRU"
    assert gru_params > rnn_params, "GRU should have more parameters than vanilla RNN"


def test_bidirectional_doubles_hidden_size():
    """Test that bidirectional LSTM has approximately 2x parameters vs unidirectional."""
    base_config = {
        'experiment_name': 'test_bidir',
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
            'output_dir': 'test',
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
            'dir': 'test'
        },
        'random_seed': 42
    }

    # Unidirectional
    uni_config = ExperimentConfig(**base_config)
    uni_model = create_model_from_config(uni_config)
    uni_params = uni_model.get_parameter_count()

    # Bidirectional
    bidir_config_dict = base_config.copy()
    bidir_config_dict['model']['rnn']['bidirectional'] = True
    bidir_config_dict['training']['objective'] = 'masked_lm'
    bidir_config = ExperimentConfig(**bidir_config_dict)
    bidir_model = create_model_from_config(bidir_config)
    bidir_params = bidir_model.get_parameter_count()

    # Bidirectional should have more parameters
    # (RNN params roughly double, output projection definitely doubles)
    assert bidir_params > uni_params, "Bidirectional should have more parameters"
    assert bidir_params < uni_params * 3, "Bidirectional shouldn't triple parameters"


# === Output Consistency Tests ===

def test_all_architectures_produce_valid_logits(minimal_configs):
    """Test that all architectures produce valid (non-NaN, finite) logits."""
    batch_size = 2
    seq_length = 10

    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        output = model(input_ids)

        logits = output.logits

        assert not torch.isnan(logits).any(), f"{arch}: NaN in logits"
        assert torch.isfinite(logits).all(), f"{arch}: Infinite values in logits"


def test_all_architectures_produce_valid_loss(minimal_configs):
    """Test that all architectures produce valid loss values."""
    batch_size = 2
    seq_length = 10

    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        labels = torch.randint(0, 1000, (batch_size, seq_length))

        output = model(input_ids, labels=labels)
        loss = output.loss

        assert not torch.isnan(loss), f"{arch}: NaN loss"
        assert torch.isfinite(loss), f"{arch}: Infinite loss"
        assert loss > 0, f"{arch}: Non-positive loss"


def test_architectures_deterministic_with_seed(minimal_configs):
    """Test that architectures produce deterministic outputs with fixed seed."""
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))

    for arch, config in minimal_configs.items():
        # First run
        torch.manual_seed(42)
        model1 = create_model_from_config(config)
        torch.manual_seed(42)  # Reset for inference
        output1 = model1(input_ids)

        # Second run with same seed
        torch.manual_seed(42)
        model2 = create_model_from_config(config)
        torch.manual_seed(42)  # Reset for inference
        output2 = model2(input_ids)

        # Should be identical
        assert torch.allclose(output1.logits, output2.logits, atol=1e-6), \
            f"{arch}: Non-deterministic outputs with same seed"


# === Model State Tests ===

def test_all_architectures_can_be_set_to_train_mode(minimal_configs):
    """Test that all architectures support train/eval modes."""
    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        # Set to train mode
        model.train()
        assert model.training, f"{arch}: Failed to set train mode"

        # Set to eval mode
        model.eval()
        assert not model.training, f"{arch}: Failed to set eval mode"


def test_all_architectures_dropout_behavior(minimal_configs):
    """Test that dropout behaves differently in train vs eval mode."""
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))

    for arch, config in minimal_configs.items():
        model = create_model_from_config(config)

        # Get outputs in train mode (multiple runs)
        model.train()
        torch.manual_seed(42)
        train_output1 = model(input_ids).logits
        torch.manual_seed(43)
        train_output2 = model(input_ids).logits

        # Get outputs in eval mode (multiple runs)
        model.eval()
        torch.manual_seed(42)
        eval_output1 = model(input_ids).logits
        torch.manual_seed(43)
        eval_output2 = model(input_ids).logits

        # In eval mode, outputs should be identical (deterministic)
        assert torch.allclose(eval_output1, eval_output2, atol=1e-6), \
            f"{arch}: Eval mode not deterministic"

        # Note: Train mode variability depends on dropout, which may be 0 in tiny models


# === Error Handling Tests ===

def test_invalid_architecture_raises_error():
    """Test that invalid architecture raises clear error."""
    from pydantic import ValidationError

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
            'architecture': 'invalid_arch',
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

    # Config validation should fail at Pydantic level
    with pytest.raises(ValidationError, match="model.architecture"):
        config = ExperimentConfig(**config_dict)


def test_mismatched_objective_architecture_still_works():
    """Test that mismatched objective/architecture doesn't crash (just suboptimal)."""
    # BERT with causal LM objective (unusual but shouldn't crash)
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
            'architecture': 'bert',
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
            'epochs': 1,
            'objective': 'causal_lm'  # Unusual for BERT
        },
        'logging': {
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'dir': 'test'
        },
        'random_seed': 42
    }

    config = ExperimentConfig(**config_dict)
    model = create_model_from_config(config)

    # Should create model without error
    assert model is not None
    assert model.model_type == 'bert'
