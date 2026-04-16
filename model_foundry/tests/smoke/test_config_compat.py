"""
Smoke tests for the ModelConfig backwards-compatibility shim.

Tests that the ``_backwards_compat_flat_config`` model validator in
``model_foundry.config.ModelConfig`` correctly converts the old flat-field
config format into the new nested format while leaving new-style configs
untouched.
"""

import pytest
from pydantic import ValidationError

from model_foundry.config import (
    ExperimentConfig,
    ModelConfig,
    TransformerModelConfig,
    RNNModelConfig,
    DataConfig,
    TokenizerConfig,
    TrainingConfig,
    LoggingConfig,
)

# ── Flat transformer fields matching the old YAML format ────────────────

_FLAT_TRANSFORMER = dict(
    layers=12,
    embedding_size=768,
    hidden_size=768,
    intermediate_hidden_size=3072,
    attention_heads=12,
    activation_function="gelu",
    dropout=0.1,
    attention_dropout=0.1,
)

_FLAT_RNN_LSTM = dict(
    embedding_size=512,
    hidden_size=512,
    num_layers=2,
    dropout=0.3,
    rnn_type="lstm",
)

_FLAT_RNN_GRU = dict(
    embedding_size=256,
    hidden_size=256,
    num_layers=3,
    dropout=0.2,
    rnn_type="gru",
)


# ── 1. Flat transformer config auto-converts ────────────────────────────

class TestFlatTransformerAutoConvert:
    def test_architecture_set_to_gpt2(self):
        cfg = ModelConfig(**_FLAT_TRANSFORMER)
        assert cfg.architecture == "gpt2"

    def test_fields_nested_under_transformer(self):
        cfg = ModelConfig(**_FLAT_TRANSFORMER)
        assert cfg.transformer is not None
        assert cfg.transformer.layers == 12
        assert cfg.transformer.embedding_size == 768
        assert cfg.transformer.hidden_size == 768
        assert cfg.transformer.intermediate_hidden_size == 3072
        assert cfg.transformer.attention_heads == 12
        assert cfg.transformer.activation_function == "gelu"
        assert cfg.transformer.dropout == 0.1
        assert cfg.transformer.attention_dropout == 0.1

    def test_rnn_is_none(self):
        cfg = ModelConfig(**_FLAT_TRANSFORMER)
        assert cfg.rnn is None


# ── 2. Flat RNN (LSTM) config auto-converts ──────────────────────────────

class TestFlatRNNLSTMAutoConvert:
    def test_architecture_set_to_lstm(self):
        cfg = ModelConfig(**_FLAT_RNN_LSTM)
        assert cfg.architecture == "lstm"

    def test_fields_nested_under_rnn(self):
        cfg = ModelConfig(**_FLAT_RNN_LSTM)
        assert cfg.rnn is not None
        assert cfg.rnn.embedding_size == 512
        assert cfg.rnn.hidden_size == 512
        assert cfg.rnn.num_layers == 2
        assert cfg.rnn.dropout == 0.3
        assert cfg.rnn.rnn_type == "lstm"

    def test_transformer_is_none(self):
        cfg = ModelConfig(**_FLAT_RNN_LSTM)
        assert cfg.transformer is None


# ── 3. Flat GRU config ──────────────────────────────────────────────────

class TestFlatGRUAutoConvert:
    def test_architecture_set_to_gru(self):
        cfg = ModelConfig(**_FLAT_RNN_GRU)
        assert cfg.architecture == "gru"

    def test_fields_nested_under_rnn(self):
        cfg = ModelConfig(**_FLAT_RNN_GRU)
        assert cfg.rnn is not None
        assert cfg.rnn.embedding_size == 256
        assert cfg.rnn.hidden_size == 256
        assert cfg.rnn.num_layers == 3
        assert cfg.rnn.rnn_type == "gru"


# ── 4. New nested format passes through unchanged ───────────────────────

class TestNestedFormatPassthrough:
    def test_explicit_transformer(self):
        cfg = ModelConfig(
            architecture="gpt2",
            transformer=TransformerModelConfig(
                layers=6,
                embedding_size=256,
                hidden_size=256,
                intermediate_hidden_size=1024,
                attention_heads=4,
                dropout=0.1,
                attention_dropout=0.1,
            ),
        )
        assert cfg.architecture == "gpt2"
        assert cfg.transformer.layers == 6
        assert cfg.transformer.embedding_size == 256

    def test_explicit_rnn(self):
        cfg = ModelConfig(
            architecture="lstm",
            rnn=RNNModelConfig(
                embedding_size=128,
                hidden_size=128,
                num_layers=1,
            ),
        )
        assert cfg.architecture == "lstm"
        assert cfg.rnn.num_layers == 1


# ── 5. Full ExperimentConfig with flat model section ─────────────────────

class TestExperimentConfigFlatModel:
    """Parse a complete config dict that mirrors the old YAML layout."""

    @pytest.fixture()
    def flat_experiment_dict(self):
        return dict(
            experiment_name="exp0_baseline_model",
            data=dict(
                source_corpus="data/raw/train_90M/",
                training_corpus="data/raw/train_90M/",
                test_corpus="data/raw/test_10M/",
                batch_size=32,
                max_sequence_length=1000,
            ),
            tokenizer=dict(
                output_dir="tokenizers/exp0_baseline/",
                vocab_size=50004,
            ),
            model=dict(
                layers=12,
                embedding_size=768,
                hidden_size=768,
                intermediate_hidden_size=3072,
                attention_heads=12,
                activation_function="gelu",
                dropout=0.1,
                attention_dropout=0.1,
            ),
            training=dict(
                output_dir="models/exp0_baseline/",
                learning_rate=0.0004,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-6,
                epochs=20,
            ),
            logging=dict(
                level="INFO",
                dir="logs",
            ),
            random_seed=9,
        )

    def test_parses_successfully(self, flat_experiment_dict):
        cfg = ExperimentConfig(**flat_experiment_dict)
        assert cfg.model.architecture == "gpt2"

    def test_model_layers_via_shim(self, flat_experiment_dict):
        cfg = ExperimentConfig(**flat_experiment_dict)
        assert cfg.model.layers == 12

    def test_model_transformer_populated(self, flat_experiment_dict):
        cfg = ExperimentConfig(**flat_experiment_dict)
        assert cfg.model.transformer is not None
        assert cfg.model.transformer.attention_heads == 12


# ── 6. Round-trip serialization ──────────────────────────────────────────

class TestRoundTripSerialization:
    def test_flat_transformer_round_trip(self):
        original = ModelConfig(**_FLAT_TRANSFORMER)
        dumped = original.model_dump()
        restored = ModelConfig(**dumped)

        assert restored.architecture == original.architecture
        assert restored.transformer == original.transformer
        assert restored.rnn == original.rnn

    def test_flat_rnn_round_trip(self):
        original = ModelConfig(**_FLAT_RNN_LSTM)
        dumped = original.model_dump()
        restored = ModelConfig(**dumped)

        assert restored.architecture == original.architecture
        assert restored.rnn == original.rnn
        assert restored.transformer == original.transformer


# ── 7. Properties work after conversion ──────────────────────────────────

class TestPropertiesAfterConversion:
    def test_transformer_properties(self):
        cfg = ModelConfig(**_FLAT_TRANSFORMER)
        assert cfg.layers == 12
        assert cfg.hidden_size == 768
        assert cfg.embedding_size == 768
        assert cfg.intermediate_hidden_size == 3072
        assert cfg.attention_heads == 12
        assert cfg.activation_function == "gelu"
        assert cfg.dropout == 0.1
        assert cfg.attention_dropout == 0.1

    def test_rnn_properties(self):
        cfg = ModelConfig(**_FLAT_RNN_LSTM)
        assert cfg.layers == 2  # maps to num_layers
        assert cfg.hidden_size == 512
        assert cfg.embedding_size == 512
        assert cfg.dropout == 0.3
        # Transformer-only properties return defaults/None
        assert cfg.intermediate_hidden_size is None
        assert cfg.attention_heads is None
        assert cfg.attention_dropout == 0.0


# ── 8. Empty/invalid flat config fails validation ────────────────────────

class TestInvalidFlatConfig:
    def test_empty_dict_fails(self):
        """A dict with no architecture and no recognisable fields must fail."""
        with pytest.raises(ValidationError):
            ModelConfig()

    def test_unrecognised_fields_only_fails(self):
        """Random keys with no architecture should not silently pass."""
        with pytest.raises(ValidationError):
            ModelConfig(foo="bar", baz=42)

    def test_partial_transformer_fields_fail(self):
        """Supplying only some transformer fields should hit a required-field
        error inside TransformerModelConfig (e.g. missing hidden_size)."""
        with pytest.raises(ValidationError):
            ModelConfig(layers=12, embedding_size=768)
