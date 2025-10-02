"""
Unit tests for data collators (Causal LM and Masked LM).

Tests cover:
- CausalLMDataCollator for GPT-2 style training
- MaskedLMDataCollator for BERT style training
- Padding and attention mask creation
- Label preparation for different objectives
- Masking strategy for MLM
"""

import pytest
import torch
from model_foundry.data_collators import (
    CausalLMDataCollator,
    MaskedLMDataCollator,
    get_data_collator
)
from model_foundry.config import (
    ExperimentConfig, DataConfig, TokenizerConfig, ModelConfig,
    TransformerModelConfig, TrainingConfig, LoggingConfig
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def causal_lm_config():
    """Configuration for causal LM testing."""
    return ExperimentConfig(
        experiment_name="test_causal",
        data=DataConfig(
            source_corpus="test",
            training_corpus="test",
            batch_size=2,
            max_sequence_length=32
        ),
        tokenizer=TokenizerConfig(
            output_dir="test",
            vocab_size=1000
        ),
        model=ModelConfig(
            architecture="gpt2",
            transformer=TransformerModelConfig(
                layers=2, embedding_size=64, hidden_size=64,
                intermediate_hidden_size=128, attention_heads=2,
                activation_function="gelu", dropout=0.1, attention_dropout=0.1
            )
        ),
        training=TrainingConfig(
            output_dir="test",
            learning_rate=1e-4,
            adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
            epochs=1,
            objective="causal_lm"
        ),
        logging=LoggingConfig(level="INFO", dir="test", use_wandb=False),
        random_seed=42
    )


@pytest.fixture
def masked_lm_config(causal_lm_config):
    """Configuration for masked LM testing."""
    causal_lm_config.training.objective = "masked_lm"
    causal_lm_config.training.mlm_probability = 0.15
    return causal_lm_config


# ============================================================================
# Test CausalLMDataCollator
# ============================================================================

class TestCausalLMDataCollator:
    """Tests for causal language modeling data collator."""

    def test_collator_creation(self, mock_tokenizer):
        """Test creating causal LM collator."""
        collator = CausalLMDataCollator(
            tokenizer=mock_tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        assert collator is not None
        assert collator.mlm is False

    def test_collator_padding(self, mock_tokenizer):
        """Test that collator pads sequences correctly."""
        collator = CausalLMDataCollator(tokenizer=mock_tokenizer, pad_to_multiple_of=8)

        examples = [
            {'input_ids': [1, 2, 3, 4, 5]},
            {'input_ids': [1, 2, 3]}
        ]

        batch = collator(examples)

        # Both sequences should be padded to multiple of 8
        assert batch['input_ids'].shape[1] % 8 == 0
        # Second sequence should have padding
        assert batch['attention_mask'][1].sum() == 3  # Only 3 real tokens

    def test_collator_attention_mask(self, mock_tokenizer):
        """Test attention mask creation."""
        collator = CausalLMDataCollator(tokenizer=mock_tokenizer)

        examples = [
            {'input_ids': [1, 2, 3, 4, 5, 6]},
            {'input_ids': [1, 2, 3]}
        ]

        batch = collator(examples)

        # First sequence: all 1s
        assert batch['attention_mask'][0, :6].sum() == 6
        # Second sequence: 1s for real tokens, 0s for padding
        assert batch['attention_mask'][1, :3].sum() == 3

    def test_collator_labels_creation(self, mock_tokenizer):
        """Test that labels are created correctly for causal LM."""
        collator = CausalLMDataCollator(tokenizer=mock_tokenizer)

        examples = [
            {'input_ids': [1, 2, 3, 4]},
            {'input_ids': [5, 6]}
        ]

        batch = collator(examples)

        # Labels should exist
        assert 'labels' in batch
        # Labels for real tokens should match input_ids
        assert batch['labels'][0, 0].item() == 1
        assert batch['labels'][0, 1].item() == 2
        # Labels for padding should be -100
        padding_positions = batch['attention_mask'] == 0
        assert (batch['labels'][padding_positions] == -100).all()

    def test_collator_batch_consistency(self, mock_tokenizer):
        """Test batch tensor shapes are consistent."""
        collator = CausalLMDataCollator(tokenizer=mock_tokenizer)

        examples = [
            {'input_ids': [1, 2, 3, 4, 5]},
            {'input_ids': [6, 7, 8]}
        ]

        batch = collator(examples)

        # All tensors should have same shape for batch dimension
        assert batch['input_ids'].shape[0] == 2
        assert batch['attention_mask'].shape[0] == 2
        assert batch['labels'].shape[0] == 2
        # All tensors should have same sequence length
        seq_len = batch['input_ids'].shape[1]
        assert batch['attention_mask'].shape[1] == seq_len
        assert batch['labels'].shape[1] == seq_len


# ============================================================================
# Test MaskedLMDataCollator
# ============================================================================

class TestMaskedLMDataCollator:
    """Tests for masked language modeling data collator."""

    def test_collator_creation(self, mock_bert_tokenizer):
        """Test creating masked LM collator."""
        collator = MaskedLMDataCollator(
            tokenizer=mock_bert_tokenizer,
            mlm=True,
            mlm_probability=0.15,
            pad_to_multiple_of=8
        )

        assert collator is not None
        assert collator.mlm is True
        assert collator.mlm_probability == 0.15

    def test_collator_masking(self, mock_bert_tokenizer):
        """Test that collator masks tokens."""
        collator = MaskedLMDataCollator(
            tokenizer=mock_bert_tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        # Create examples with known tokens
        examples = [
            {'input_ids': [101] + list(range(1, 21)) + [102]},  # CLS + 20 tokens + SEP
            {'input_ids': [101] + list(range(21, 41)) + [102]}
        ]

        batch = collator(examples)

        # Some tokens should be masked
        mask_token_id = mock_bert_tokenizer.mask_token_id
        num_masked = (batch['input_ids'] == mask_token_id).sum().item()

        # With 40 non-special tokens total and 15% masking, expect ~6 masked
        # (allowing for randomness, just check > 0)
        assert num_masked > 0

    def test_collator_labels_only_for_masked(self, mock_bert_tokenizer):
        """Test that labels are -100 except for masked tokens."""
        torch.manual_seed(42)  # For reproducibility
        collator = MaskedLMDataCollator(
            tokenizer=mock_bert_tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        examples = [
            {'input_ids': [101, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 102]}
        ]

        batch = collator(examples)

        labels = batch['labels']

        # Most labels should be -100 (not masked)
        num_not_masked = (labels == -100).sum().item()
        num_masked = (labels != -100).sum().item()

        # Should have more non-masked than masked (since prob=0.15)
        assert num_not_masked > num_masked
        # But should have at least some masked (unless very unlucky)
        # With 10 maskable tokens and 15% prob, expected ~1-2 masked

    def test_collator_special_tokens_not_masked(self, mock_bert_tokenizer):
        """Test that special tokens are never masked."""
        torch.manual_seed(42)
        collator = MaskedLMDataCollator(
            tokenizer=mock_bert_tokenizer,
            mlm=True,
            mlm_probability=0.5  # High probability to ensure masking happens
        )

        cls_id = mock_bert_tokenizer.cls_token_id
        sep_id = mock_bert_tokenizer.sep_token_id

        examples = [
            {'input_ids': [cls_id, 1, 2, 3, 4, 5, 6, 7, 8, sep_id]}
        ]

        batch = collator(examples)

        # CLS and SEP should never be in labels (should be -100)
        assert batch['labels'][0, 0].item() == -100  # CLS position
        assert batch['labels'][0, -1].item() == -100  # SEP position

    def test_collator_masking_strategy(self, mock_bert_tokenizer):
        """Test BERT masking strategy (80% MASK, 10% random, 10% unchanged)."""
        torch.manual_seed(42)
        collator = MaskedLMDataCollator(
            tokenizer=mock_bert_tokenizer,
            mlm=True,
            mlm_probability=1.0  # Mask everything for testing strategy
        )

        # Create example with non-special tokens
        examples = [
            {'input_ids': [200, 201, 202, 203, 204, 205, 206, 207, 208, 209] * 10}
        ]

        batch = collator(examples)
        input_ids = batch['input_ids'][0]
        labels = batch['labels'][0]

        # Find positions that were selected for masking (labels != -100)
        masked_positions = labels != -100

        # Of those, count how many were replaced with [MASK], random, or unchanged
        mask_token_id = mock_bert_tokenizer.mask_token_id
        num_mask_token = (input_ids[masked_positions] == mask_token_id).sum().item()
        total_masked = masked_positions.sum().item()

        # Should have some MASK tokens (with large sample, should be ~80%)
        assert num_mask_token > 0
        # With 100 tokens, expect most to be MASK token
        assert num_mask_token > total_masked * 0.5  # At least half

    def test_collator_padding_tokens_not_masked(self, mock_bert_tokenizer):
        """Test that padding tokens are never masked."""
        collator = MaskedLMDataCollator(
            tokenizer=mock_bert_tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        examples = [
            {'input_ids': [101, 1, 2, 3, 102]},
            {'input_ids': [101, 4, 5, 6, 7, 8, 9, 10, 102]}
        ]

        batch = collator(examples)

        # Padding tokens (where attention_mask == 0) should have labels == -100
        padding_positions = batch['attention_mask'] == 0
        assert (batch['labels'][padding_positions] == -100).all()


# ============================================================================
# Test Data Collator Factory
# ============================================================================

class TestDataCollatorFactory:
    """Tests for get_data_collator factory function."""

    def test_factory_creates_causal_collator(self, causal_lm_config, mock_tokenizer):
        """Test factory creates causal LM collator for causal_lm objective."""
        collator = get_data_collator(causal_lm_config, mock_tokenizer)

        assert isinstance(collator, CausalLMDataCollator)
        assert collator.mlm is False

    def test_factory_creates_masked_collator(self, masked_lm_config, mock_bert_tokenizer):
        """Test factory creates masked LM collator for masked_lm objective."""
        collator = get_data_collator(masked_lm_config, mock_bert_tokenizer)

        assert isinstance(collator, MaskedLMDataCollator)
        assert collator.mlm is True

    def test_factory_uses_mlm_probability(self, masked_lm_config, mock_bert_tokenizer):
        """Test factory uses mlm_probability from config."""
        masked_lm_config.training.mlm_probability = 0.25
        collator = get_data_collator(masked_lm_config, mock_bert_tokenizer)

        assert collator.mlm_probability == 0.25

    def test_factory_invalid_objective(self, causal_lm_config, mock_tokenizer):
        """Test factory raises error for unknown objective."""
        causal_lm_config.training.objective = "unknown_objective"

        with pytest.raises(ValueError, match="Unknown training objective"):
            get_data_collator(causal_lm_config, mock_tokenizer)


# ============================================================================
# Test Collator Integration
# ============================================================================

class TestCollatorIntegration:
    """Integration tests for data collators with real batches."""

    def test_causal_collator_with_dataloader(self, mock_tokenizer):
        """Test causal collator works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader, Dataset

        class SimpleDataset(Dataset):
            def __init__(self):
                self.data = [
                    {'input_ids': [1, 2, 3, 4, 5]},
                    {'input_ids': [6, 7, 8]},
                    {'input_ids': [9, 10, 11, 12]},
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        collator = CausalLMDataCollator(tokenizer=mock_tokenizer)
        dataset = SimpleDataset()
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        for batch in dataloader:
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert 'labels' in batch
            # All should be tensors
            assert isinstance(batch['input_ids'], torch.Tensor)
            assert isinstance(batch['attention_mask'], torch.Tensor)
            assert isinstance(batch['labels'], torch.Tensor)

    def test_masked_collator_reproducibility(self, mock_bert_tokenizer):
        """Test that masked collator is reproducible with same seed."""
        collator = MaskedLMDataCollator(
            tokenizer=mock_bert_tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        examples = [
            {'input_ids': list(range(101, 121))}
        ]

        # Run twice with same seed
        torch.manual_seed(42)
        batch1 = collator(examples)

        torch.manual_seed(42)
        batch2 = collator(examples)

        # Should produce identical results
        assert torch.equal(batch1['input_ids'], batch2['input_ids'])
        assert torch.equal(batch1['labels'], batch2['labels'])


# ============================================================================
# Additional Fixtures for Tests
# ============================================================================

@pytest.fixture
def mock_bert_tokenizer():
    """Mock BERT tokenizer for testing."""
    class MockBERTTokenizer:
        vocab_size = 1000
        cls_token_id = 101
        sep_token_id = 102
        mask_token_id = 103
        pad_token_id = 0
        unk_token_id = 100

        def __len__(self):
            return self.vocab_size

    return MockBERTTokenizer()
