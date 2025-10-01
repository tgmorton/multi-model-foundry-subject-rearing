"""
Pytest configuration and shared fixtures for model_foundry tests.
"""

import os
import pytest
import torch
from pathlib import Path
from datasets import Dataset

from model_foundry.config import (
    ExperimentConfig, DataConfig, TokenizerConfig, ModelConfig,
    TrainingConfig, LoggingConfig
)


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def tiny_config():
    """
    Minimal valid configuration for fast tests.

    This config creates a very small model and dataset suitable for unit testing.
    """
    return ExperimentConfig(
        experiment_name="test_exp",
        data=DataConfig(
            source_corpus="test/data",
            training_corpus="test/data/train",
            batch_size=2,
            max_sequence_length=32,
            num_workers=0  # Disable multiprocessing for tests (avoids pickle issues)
        ),
        tokenizer=TokenizerConfig(
            output_dir="test/tokenizer",
            vocab_size=1000
        ),
        model=ModelConfig(
            layers=2,
            embedding_size=64,
            hidden_size=64,
            intermediate_hidden_size=128,
            attention_heads=2,
            activation_function="gelu",
            dropout=0.1,
            attention_dropout=0.1
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
            use_amp=False,
            use_tf32=False,
            use_gradient_checkpointing=False,
            gradient_accumulation_steps=1
        ),
        logging=LoggingConfig(
            level="INFO",
            dir="test/logs",
            use_wandb=False
        ),
        random_seed=42
    )


@pytest.fixture
def invalid_config_data():
    """Configuration data with invalid values for testing validation."""
    return {
        "experiment_name": "test",
        "data": {
            "source_corpus": "test",
            "training_corpus": "test",
            "batch_size": -5,  # Invalid: negative
            "max_sequence_length": 0  # Invalid: must be > 0
        },
        "tokenizer": {
            "output_dir": "test",
            "vocab_size": -100  # Invalid: negative
        },
        "model": {
            "layers": 0,  # Invalid: must be > 0
            "embedding_size": 64,
            "hidden_size": 64,
            "intermediate_hidden_size": 128,
            "attention_heads": 2,
            "activation_function": "gelu",
            "dropout": 1.5,  # Invalid: must be < 1.0
            "attention_dropout": 0.1
        },
        "training": {
            "output_dir": "test",
            "learning_rate": -0.001,  # Invalid: must be > 0
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 0.0,  # Invalid: must be > 0
            "epochs": 0  # Invalid: must be > 0
        },
        "logging": {
            "level": "INFO",
            "dir": "logs"
        },
        "random_seed": 42
    }


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def tiny_dataset():
    """
    Small tokenized dataset for testing data processing.

    Returns a HuggingFace Dataset with 100 sequences of varying lengths.
    """
    import random
    random.seed(42)

    # Create sequences of varying lengths
    sequences = []
    for i in range(100):
        length = random.randint(10, 50)
        # Create sequence with values in vocab range
        sequence = [random.randint(1, 999) for _ in range(length)]
        sequences.append(sequence)

    return Dataset.from_dict({'input_ids': sequences})


@pytest.fixture
def fixed_length_dataset():
    """Dataset with all sequences the same length."""
    sequences = [[i % 100 for i in range(32)] for _ in range(50)]
    return Dataset.from_dict({'input_ids': sequences})


@pytest.fixture
def empty_dataset():
    """Empty dataset for edge case testing."""
    return Dataset.from_dict({'input_ids': []})


@pytest.fixture
def single_sequence_dataset():
    """Dataset with only one sequence."""
    return Dataset.from_dict({'input_ids': [[1, 2, 3, 4, 5, 6, 7, 8]]})


# ============================================================================
# Tokenizer Fixtures
# ============================================================================

@pytest.fixture
def mock_tokenizer():
    """
    Simple mock tokenizer for testing.

    Provides basic encode/decode functionality without requiring a real model.
    """
    class MockTokenizer:
        vocab_size = 1000
        pad_token = "<pad>"
        eos_token = "</s>"
        bos_token = "<s>"
        unk_token = "<unk>"
        pad_token_id = 0
        eos_token_id = 1
        bos_token_id = 2
        unk_token_id = 3

        def encode(self, text, add_special_tokens=True):
            # Simple tokenization: split on spaces and assign IDs
            tokens = [hash(word) % 1000 for word in text.split()]
            if add_special_tokens:
                return [self.bos_token_id] + tokens + [self.eos_token_id]
            return tokens

        def decode(self, ids, skip_special_tokens=True):
            if skip_special_tokens:
                special_ids = {self.pad_token_id, self.eos_token_id,
                             self.bos_token_id, self.unk_token_id}
                ids = [i for i in ids if i not in special_ids]
            return " ".join(str(i) for i in ids)

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            # Create dummy tokenizer config
            (Path(path) / "tokenizer_config.json").write_text('{"vocab_size": 1000}')
            # Create dummy tokenizer.model file for SentencePiece loading
            (Path(path) / "tokenizer.model").write_bytes(b"dummy_sentencepiece_model")

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

    return MockTokenizer()


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def tiny_model(tiny_config):
    """Small GPT-2 model for testing."""
    from model_foundry.model import create_model
    return create_model(tiny_config)


# ============================================================================
# Workspace Fixtures
# ============================================================================

@pytest.fixture
def temp_workspace(tmp_path):
    """
    Temporary workspace with proper directory structure.

    Creates a complete workspace directory with subdirectories for
    data, models, logs, etc.
    """
    workspace = tmp_path / "workspace"
    (workspace / "data" / "tokenized").mkdir(parents=True)
    (workspace / "data" / "chunked").mkdir(parents=True)
    (workspace / "models").mkdir(parents=True)
    (workspace / "logs").mkdir(parents=True)
    (workspace / "tokenizer").mkdir(parents=True)

    return workspace


@pytest.fixture
def temp_config_file(tmp_path, tiny_config):
    """Create a temporary config YAML file."""
    import yaml

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(tiny_config.model_dump(), f)

    return str(config_path)


# ============================================================================
# PyTorch Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get appropriate device for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def deterministic_seed():
    """Set seeds for deterministic testing."""
    import random
    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    yield seed

    # Cleanup after test
    # (In case tests rely on randomness)


# ============================================================================
# Test Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: tests requiring CUDA GPU"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take more than 1 second"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: end-to-end tests"
    )


# ============================================================================
# Skip Conditions
# ============================================================================

@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
