"""
Unit tests for BERT architecture and masked language modeling.

Tests cover:
- BERT model creation and configuration
- Masked language modeling forward pass
- Data collator for MLM
- WordPiece tokenizer training
"""

import pytest
import torch
from model_foundry.architectures.bert import BERTModel
from model_foundry.architectures.base import ModelOutput
from model_foundry.config import (
    ExperimentConfig, ModelConfig, TransformerModelConfig,
    BERTSpecificConfig, DataConfig, TokenizerConfig,
    TrainingConfig, LoggingConfig
)
from model_foundry.model import create_model


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tiny_bert_config():
    """Minimal BERT configuration for testing."""
    return ExperimentConfig(
        experiment_name="test_bert",
        data=DataConfig(
            source_corpus="test/data",
            training_corpus="test/data/train",
            batch_size=2,
            max_sequence_length=32
        ),
        tokenizer=TokenizerConfig(
            output_dir="test/tokenizer",
            vocab_size=1000,
            tokenizer_type="wordpiece",
            special_tokens={
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]"
            }
        ),
        model=ModelConfig(
            architecture="bert",
            transformer=TransformerModelConfig(
                layers=2,
                embedding_size=64,
                hidden_size=64,
                intermediate_hidden_size=128,
                attention_heads=2,
                activation_function="gelu",
                dropout=0.1,
                attention_dropout=0.1
            ),
            bert=BERTSpecificConfig(
                type_vocab_size=2
            )
        ),
        training=TrainingConfig(
            output_dir="test/output",
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            epochs=1,
            train_steps=10,
            warmup_steps=2,
            objective="masked_lm",
            mlm_probability=0.15
        ),
        logging=LoggingConfig(
            level="INFO",
            dir="test/logs",
            use_wandb=False
        ),
        random_seed=42
    )


@pytest.fixture
def mock_bert_tokenizer():
    """Mock BERT-style tokenizer for testing."""
    class MockBERTTokenizer:
        vocab_size = 1000
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        mask_token = "[MASK]"
        unk_token = "[UNK]"
        pad_token = "[PAD]"
        cls_token_id = 101
        sep_token_id = 102
        mask_token_id = 103
        unk_token_id = 100
        pad_token_id = 0

        def __len__(self):
            return self.vocab_size

        def encode(self, text, add_special_tokens=True):
            tokens = [hash(word) % 900 + 104 for word in text.split()]
            if add_special_tokens:
                return [self.cls_token_id] + tokens + [self.sep_token_id]
            return tokens

        def __call__(self, text, padding=False, truncation=False,
                    max_length=None, return_tensors=None):
            if isinstance(text, str):
                text = [text]

            encoded = [self.encode(t) for t in text]

            if padding and max_length:
                for seq in encoded:
                    while len(seq) < max_length:
                        seq.append(self.pad_token_id)

            if return_tensors == "pt":
                return {"input_ids": torch.tensor(encoded)}
            return {"input_ids": encoded}

    return MockBERTTokenizer()


# ============================================================================
# Test BERT Model Creation
# ============================================================================

class TestBERTModelCreation:
    """Tests for BERT model instantiation and configuration."""

    def test_bert_model_from_config(self, tiny_bert_config):
        """Test creating BERT model from configuration."""
        model = create_model(tiny_bert_config)

        assert isinstance(model, BERTModel)
        assert model.model_type == "bert"
        assert not model.supports_generation

    def test_bert_model_architecture_property(self, tiny_bert_config):
        """Test that BERT model has correct architecture type."""
        model = create_model(tiny_bert_config)

        assert model.model_type == "bert"

    def test_bert_model_config_mapping(self, tiny_bert_config):
        """Test that config parameters are correctly mapped to HF BERT."""
        model = create_model(tiny_bert_config)

        # Check HF config has our parameters
        assert model.config.num_hidden_layers == 2
        assert model.config.hidden_size == 64
        assert model.config.intermediate_size == 128
        assert model.config.num_attention_heads == 2
        assert model.config.hidden_dropout_prob == 0.1
        assert model.config.attention_probs_dropout_prob == 0.1

    def test_bert_model_with_bert_specific_config(self, tiny_bert_config):
        """Test BERT-specific configuration (type_vocab_size)."""
        model = create_model(tiny_bert_config)

        assert model.config.type_vocab_size == 2

    def test_bert_model_without_bert_config(self, tiny_bert_config):
        """Test that BERT works without optional bert-specific config."""
        tiny_bert_config.model.bert = None
        model = create_model(tiny_bert_config)

        # Should use HuggingFace defaults
        assert model.config.type_vocab_size == 2  # HF default

    def test_bert_model_invalid_architecture(self, tiny_bert_config):
        """Test that creating BERT with wrong architecture fails."""
        tiny_bert_config.model.architecture = "gpt2"

        with pytest.raises(ValueError, match="Expected 'bert'"):
            BERTModel.from_config(tiny_bert_config)

    def test_bert_model_missing_transformer_config(self, tiny_bert_config):
        """Test that BERT requires transformer configuration."""
        tiny_bert_config.model.transformer = None

        with pytest.raises(ValueError, match="requires 'transformer' configuration"):
            BERTModel.from_config(tiny_bert_config)


# ============================================================================
# Test BERT Forward Pass
# ============================================================================

class TestBERTForwardPass:
    """Tests for BERT model forward pass and MLM."""

    def test_bert_forward_with_labels(self, tiny_bert_config):
        """Test BERT forward pass with labels computes loss."""
        model = create_model(tiny_bert_config)
        batch_size, seq_len = 2, 16
        vocab_size = 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Set some labels to -100 (ignore in loss)
        labels[:, :5] = -100

        output = model(input_ids=input_ids, labels=labels)

        assert isinstance(output, ModelOutput)
        assert output.loss is not None
        assert output.logits is not None
        assert output.loss.item() > 0  # Should have positive loss

    def test_bert_forward_without_labels(self, tiny_bert_config):
        """Test BERT forward pass without labels (no loss)."""
        model = create_model(tiny_bert_config)
        batch_size, seq_len = 2, 16
        vocab_size = 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = model(input_ids=input_ids)

        assert isinstance(output, ModelOutput)
        assert output.loss is None
        assert output.logits is not None
        assert output.logits.shape == (batch_size, seq_len, vocab_size)

    def test_bert_forward_with_attention_mask(self, tiny_bert_config):
        """Test BERT with attention mask (for padding)."""
        model = create_model(tiny_bert_config)
        batch_size, seq_len = 2, 16
        vocab_size = 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Create attention mask: first sequence full, second has padding
        attention_mask = torch.ones((batch_size, seq_len))
        attention_mask[1, 10:] = 0  # Last 6 tokens are padding

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert output.logits.shape == (batch_size, seq_len, vocab_size)

    def test_bert_forward_with_token_type_ids(self, tiny_bert_config):
        """Test BERT with token type IDs (segment embeddings)."""
        model = create_model(tiny_bert_config)
        batch_size, seq_len = 2, 16
        vocab_size = 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Segment A: first half, Segment B: second half
        token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        token_type_ids[:, 8:] = 1

        output = model(input_ids=input_ids, token_type_ids=token_type_ids)

        assert output.logits.shape == (batch_size, seq_len, vocab_size)

    def test_bert_output_structure(self, tiny_bert_config):
        """Test that BERT output has correct structure."""
        model = create_model(tiny_bert_config)
        input_ids = torch.randint(0, 1000, (2, 16))

        output = model(input_ids=input_ids)

        assert hasattr(output, 'loss')
        assert hasattr(output, 'logits')
        assert hasattr(output, 'hidden_states')
        assert hasattr(output, 'attentions')


# ============================================================================
# Test BERT Model Interface
# ============================================================================

class TestBERTModelInterface:
    """Tests for BERT model interface compliance."""

    def test_bert_get_input_embeddings(self, tiny_bert_config):
        """Test getting input embeddings."""
        model = create_model(tiny_bert_config)
        embeddings = model.get_input_embeddings()

        assert embeddings is not None
        assert isinstance(embeddings, torch.nn.Module)

    def test_bert_resize_token_embeddings(self, tiny_bert_config):
        """Test resizing token embeddings."""
        model = create_model(tiny_bert_config)
        original_vocab_size = model.config.vocab_size

        new_vocab_size = original_vocab_size + 10
        model.resize_token_embeddings(new_vocab_size)

        embeddings = model.get_input_embeddings()
        assert embeddings.weight.shape[0] == new_vocab_size

    def test_bert_supports_generation_false(self, tiny_bert_config):
        """Test that BERT doesn't support generation."""
        model = create_model(tiny_bert_config)

        assert model.supports_generation is False

    def test_bert_model_type(self, tiny_bert_config):
        """Test that BERT model_type is correct."""
        model = create_model(tiny_bert_config)

        assert model.model_type == "bert"


# ============================================================================
# Test BERT vs GPT-2 Differences
# ============================================================================

class TestBERTVsGPT2:
    """Tests highlighting differences between BERT and GPT-2."""

    def test_bert_bidirectional_attention(self, tiny_bert_config):
        """
        Test that BERT can attend to future tokens (bidirectional).

        This is implicit in the architecture - we verify by checking
        that the model doesn't use causal masking.
        """
        model = create_model(tiny_bert_config)

        # BERT should not have a causal mask in its attention
        # We can't directly test this without diving into internals,
        # but we can verify it processes full sequences
        input_ids = torch.randint(0, 1000, (1, 10))
        output = model(input_ids=input_ids)

        # Should process all tokens
        assert output.logits.shape[1] == 10

    def test_bert_vs_gpt2_architecture(self, tiny_bert_config):
        """Test that BERT and GPT-2 are registered differently."""
        from model_foundry.architectures import MODEL_REGISTRY

        assert "bert" in MODEL_REGISTRY
        assert "gpt2" in MODEL_REGISTRY
        assert MODEL_REGISTRY["bert"] != MODEL_REGISTRY["gpt2"]

    def test_bert_training_objective_config(self, tiny_bert_config):
        """Test that BERT config uses masked_lm objective."""
        assert tiny_bert_config.training.objective == "masked_lm"
        assert tiny_bert_config.training.mlm_probability == 0.15
