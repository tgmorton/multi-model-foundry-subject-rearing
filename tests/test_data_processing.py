"""
Test script for the data processing module.
"""

import os
import tempfile
import shutil
from pathlib import Path
from datasets import Dataset
import numpy as np

# Add the model_foundry package to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_foundry.data import DataProcessor
from model_foundry.config import ExperimentConfig, DataConfig, TokenizerConfig, ModelConfig, TrainingConfig, LoggingConfig


def create_test_config():
    """Create a test configuration."""
    return ExperimentConfig(
        experiment_name="test_experiment",
        data=DataConfig(
            source_corpus="test_data",
            training_corpus="test_data",
            batch_size=4,
            max_sequence_length=128
        ),
        tokenizer=TokenizerConfig(
            output_dir="test_tokenizer",
            vocab_size=1000
        ),
        model=ModelConfig(
            layers=2,
            embedding_size=64,
            hidden_size=64,
            intermediate_hidden_size=128,
            attention_heads=4,
            activation_function="GELU",
            dropout=0.1,
            attention_dropout=0.1
        ),
        training=TrainingConfig(
            output_dir="test_models",
            learning_rate=0.0001,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-6,
            warmup_steps=100,
            train_steps=1000,
            epochs=2
        ),
        logging=LoggingConfig(
            use_wandb=False
        ),
        random_seed=42
    )


def create_test_tokenized_dataset():
    """Create a test tokenized dataset."""
    # Create sequences of varying lengths
    sequences = []
    for i in range(100):
        # Create sequences of length 50-200 tokens
        length = np.random.randint(50, 200)
        sequence = list(range(i * 10, i * 10 + length))
        sequences.append(sequence)
    
    return Dataset.from_dict({
        'input_ids': sequences
    })


def test_data_processor_initialization():
    """Test that DataProcessor initializes correctly."""
    config = create_test_config()
    base_dir = "/tmp/test"
    
    processor = DataProcessor(config, base_dir)
    
    assert processor.config == config
    assert processor.base_dir == base_dir
    assert "test_experiment" in processor.tokenized_data_dir
    assert "test_experiment" in processor.chunked_data_dir


def test_chunk_sequences():
    """Test the sequence chunking functionality."""
    config = create_test_config()
    base_dir = "/tmp/test"
    processor = DataProcessor(config, base_dir)

    # Test sequences
    sequences = [
        [1, 2, 3, 4, 5, 6, 7, 8],  # 8 tokens
        [9, 10, 11],  # 3 tokens (shorter than a chunk, still used)
        [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]  # 12 tokens
    ]
    dataset = Dataset.from_dict({'input_ids': sequences})

    chunk_size = 4
    chunks = [
        chunk
        for chunk_batch in processor._chunk_sequences_streaming(dataset, chunk_size)
        for chunk in chunk_batch
    ]

    # Sequences are concatenated into one 23-token stream and cut into
    # 5 chunks; the 3-token tail [23, 24, 25] is dropped
    assert len(chunks) == 5

    # Check that all chunks are the correct size
    for chunk in chunks:
        assert len(chunk) == chunk_size

    # Verify the actual chunks
    expected_chunks = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 14],  # Crosses the second/third sequence boundary
        [15, 16, 17, 18],
        [19, 20, 21, 22]
    ]
    
    for i, chunk in enumerate(chunks):
        assert chunk == expected_chunks[i], f"Chunk {i} mismatch: {chunk} != {expected_chunks[i]}"


def test_dataset_stats():
    """Test the dataset statistics calculation."""
    config = create_test_config()
    base_dir = "/tmp/test"
    processor = DataProcessor(config, base_dir)
    
    # Create test dataset
    sequences = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    dataset = Dataset.from_dict({'input_ids': sequences})
    
    stats = processor._calculate_dataset_stats(dataset)
    
    assert stats['num_sequences'] == 3
    assert stats['total_tokens'] == 9
    assert stats['min_length'] == 2
    assert stats['max_length'] == 4
    assert stats['avg_length'] == 3.0


def test_create_chunked_dataset():
    """Test creating a chunked dataset."""
    config = create_test_config()
    base_dir = "/tmp/test"
    processor = DataProcessor(config, base_dir)
    
    # Create test dataset
    sequences = [
        list(range(10)),  # 10 tokens
        list(range(20)),  # 20 tokens
        list(range(5))    # 5 tokens
    ]
    tokenized_dataset = Dataset.from_dict({'input_ids': sequences})

    chunk_size = 4
    chunked_dataset = processor._create_chunked_dataset_streaming(tokenized_dataset, chunk_size)

    # 35 concatenated tokens -> 8 chunks; the 3-token tail is dropped
    assert len(chunked_dataset) == 8

    # Check that all chunks are the correct size
    for chunk in chunked_dataset['input_ids']:
        assert len(chunk) == chunk_size

    # Chunks preserve the concatenated token order
    stream = [token for sequence in sequences for token in sequence]
    flattened = [token for chunk in chunked_dataset['input_ids'] for token in chunk]
    assert flattened == stream[:8 * chunk_size]


if __name__ == "__main__":
    print("Running data processing tests...")
    
    test_data_processor_initialization()
    print("✓ DataProcessor initialization test passed")
    
    test_chunk_sequences()
    print("✓ Sequence chunking test passed")
    
    test_dataset_stats()
    print("✓ Dataset statistics test passed")
    
    test_create_chunked_dataset()
    print("✓ Chunked dataset creation test passed")
    
    print("\nAll tests passed! ✓") 