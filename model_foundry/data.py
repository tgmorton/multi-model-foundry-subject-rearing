"""
Data handling module for the Model Foundry framework.

This module provides utilities for preprocessing, chunking, and loading
tokenized datasets for language model training.
"""

import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
import numpy as np
from datasets import Dataset, load_from_disk, disable_progress_bar
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import torch

# Disable progress bars for cleaner output
disable_progress_bar()


def _worker_init_fn(worker_id: int) -> None:
    """
    Initialize worker processes with deterministic seeding.
    
    This function is called for each worker process to ensure reproducible
    data loading across different runs.
    
    Args:
        worker_id: The worker process ID
    """
    # Set seeds for each worker based on the global seed
    # We use a different seed for each worker to avoid identical sequences
    worker_seed = torch.initial_seed() + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)


class DataProcessor:
    """
    Handles data preprocessing, chunking, and loading for language model training.
    """
    
    def __init__(self, config, base_dir: str):
        self.config = config
        self.base_dir = base_dir
        self.tokenized_data_dir = os.path.join(base_dir, "data", "tokenized", config.experiment_name)
        self.chunked_data_dir = os.path.join(base_dir, "data", "chunked", config.experiment_name)
        self.test_data_dir = os.path.join(base_dir, "data", "tokenized", config.experiment_name, "test")
        
    def _validate_tokenized_dataset(self) -> bool:
        """Validate that the tokenized dataset exists and has the expected structure."""
        # Check for new structure with separate train/test directories
        train_dir = os.path.join(self.tokenized_data_dir, "train")
        test_dir = os.path.join(self.tokenized_data_dir, "test")
        
        if os.path.exists(train_dir):
            # New structure with separate train/test
            try:
                train_dataset = load_from_disk(train_dir)
                if 'input_ids' not in train_dataset.column_names:
                    print(f"  ✗ Training dataset missing 'input_ids' column")
                    return False
                print(f"  ✓ Training dataset loaded successfully")
                print(f"    - Training size: {len(train_dataset):,} examples")
                print(f"    - Columns: {train_dataset.column_names}")
                
                # Check if test dataset exists
                if os.path.exists(test_dir):
                    test_dataset = load_from_disk(test_dir)
                    if 'input_ids' not in test_dataset.column_names:
                        print(f"  ✗ Test dataset missing 'input_ids' column")
                        return False
                    print(f"  ✓ Test dataset loaded successfully")
                    print(f"    - Test size: {len(test_dataset):,} examples")
                else:
                    print(f"  ⚠ Test dataset not found at: {test_dir}")
                
                return True
            except Exception as e:
                print(f"  ✗ Error loading tokenized datasets: {e}")
                return False
        
        # Fallback to old structure
        if not os.path.exists(self.tokenized_data_dir):
            # Try the training_corpus path directly
            training_corpus_path = os.path.join(self.base_dir, self.config.data.training_corpus)
            if os.path.exists(training_corpus_path):
                print(f"  ✓ Found tokenized dataset at: {training_corpus_path}")
                self.tokenized_data_dir = training_corpus_path
                return True
            else:
                print(f"  ✗ Tokenized dataset not found at: {self.tokenized_data_dir}")
                print(f"  ✗ Also not found at: {training_corpus_path}")
                return False
            
        try:
            dataset = load_from_disk(self.tokenized_data_dir)
            if 'input_ids' not in dataset.column_names:
                print(f"  ✗ Tokenized dataset missing 'input_ids' column")
                return False
            print(f"  ✓ Tokenized dataset loaded successfully")
            print(f"    - Dataset size: {len(dataset):,} examples")
            print(f"    - Columns: {dataset.column_names}")
            return True
        except Exception as e:
            print(f"  ✗ Error loading tokenized dataset: {e}")
            return False
    
    def _chunk_sequences_streaming(self, dataset: Dataset, chunk_size: int) -> Generator[List[List[int]], None, None]:
        """
        Stream sequences and chunk them into fixed-length blocks without loading all into memory.
        
        Args:
            dataset: Tokenized dataset
            chunk_size: Target chunk size in tokens
            
        Yields:
            Batches of fixed-length chunks
        """
        batch_size = 1000  # Process sequences in batches to balance memory and speed
        
        for i in range(0, len(dataset), batch_size):
            batch_sequences = dataset[i:i + batch_size]['input_ids']
            chunks = []
            
            for sequence in batch_sequences:
                # Skip sequences that are too short
                if len(sequence) < chunk_size:
                    continue
                    
                # Create non-overlapping chunks of exactly chunk_size tokens
                for j in range(0, len(sequence) - chunk_size + 1, chunk_size):
                    chunk = sequence[j:j + chunk_size]
                    chunks.append(chunk)
            
            if chunks:
                yield chunks
    
    def _create_chunked_dataset_streaming(self, tokenized_dataset: Dataset, chunk_size: int) -> Dataset:
        """
        Create a new dataset with fixed-length chunks using streaming to avoid memory issues.
        
        Args:
            tokenized_dataset: Original tokenized dataset
            chunk_size: Target chunk size in tokens
            
        Returns:
            Dataset with fixed-length chunks
        """
        print(f"  - Creating fixed-length chunks (size: {chunk_size}) using streaming...")
        
        all_chunks = []
        total_original_sequences = len(tokenized_dataset)
        processed_sequences = 0
        
        # Stream through sequences and chunk them
        for chunk_batch in self._chunk_sequences_streaming(tokenized_dataset, chunk_size):
            all_chunks.extend(chunk_batch)
            processed_sequences += len(chunk_batch)
            
            # Progress update every 10k chunks
            if len(all_chunks) % 10000 == 0:
                print(f"    - Processed {len(all_chunks):,} chunks from {processed_sequences:,} sequences")
        
        print(f"    - Original sequences: {total_original_sequences:,}")
        print(f"    - Created chunks: {len(all_chunks):,}")
        
        # Create new dataset
        chunked_dataset = Dataset.from_dict({
            'input_ids': all_chunks
        })
        
        return chunked_dataset
    
    def _save_chunked_dataset(self, dataset: Dataset) -> None:
        """Save the chunked dataset to disk."""
        os.makedirs(self.chunked_data_dir, exist_ok=True)
        dataset.save_to_disk(self.chunked_data_dir)
        print(f"  ✓ Saved chunked dataset to: {self.chunked_data_dir}")
    
    def _load_chunked_dataset(self) -> Optional[Dataset]:
        """Load the chunked dataset from disk."""
        if not os.path.exists(self.chunked_data_dir):
            return None
            
        try:
            dataset = load_from_disk(self.chunked_data_dir)
            print(f"  ✓ Loaded chunked dataset from: {self.chunked_data_dir}")
            print(f"    - Dataset size: {len(dataset):,} chunks")
            return dataset
        except Exception as e:
            print(f"  ✗ Error loading chunked dataset: {e}")
            return None
    
    def _calculate_dataset_stats(self, dataset: Dataset) -> Dict[str, float]:
        """Calculate statistics about the dataset."""
        sequences = dataset['input_ids']
        
        # Calculate sequence lengths
        lengths = [len(seq) for seq in sequences]
        
        stats = {
            'num_sequences': len(sequences),
            'avg_length': np.mean(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'std_length': np.std(lengths),
            'total_tokens': sum(lengths)
        }
        
        return stats
    
    def preprocess_data(self, force_reprocess: bool = False) -> bool:
        """
        Preprocess the tokenized dataset into fixed-length chunks.
        
        Args:
            force_reprocess: If True, reprocess even if chunked data exists
            
        Returns:
            True if preprocessing was successful
        """
        print(f"--- Data Preprocessing: {self.config.experiment_name} ---")
        
        # Check if chunked data already exists
        if not force_reprocess and os.path.exists(self.chunked_data_dir):
            print(f"  - Chunked dataset already exists at: {self.chunked_data_dir}")
            return True
        
        # Validate tokenized dataset
        if not self._validate_tokenized_dataset():
            return False
        
        # Load tokenized dataset (handle new structure with separate train/test)
        train_dir = os.path.join(self.tokenized_data_dir, "train")
        if os.path.exists(train_dir):
            # New structure with separate train/test
            tokenized_dataset = load_from_disk(train_dir)
            print(f"  - Using training dataset from: {train_dir}")
        else:
            # Fallback to old structure
            tokenized_dataset = load_from_disk(self.tokenized_data_dir)
            print(f"  - Using dataset from: {self.tokenized_data_dir}")
        
        # Calculate and display original stats
        original_stats = self._calculate_dataset_stats(tokenized_dataset)
        print(f"  - Original dataset statistics:")
        print(f"    - Sequences: {original_stats['num_sequences']:,}")
        print(f"    - Total tokens: {original_stats['total_tokens']:,}")
        print(f"    - Avg length: {original_stats['avg_length']:.1f} tokens")
        
        # Create chunked dataset
        chunk_size = self.config.data.max_sequence_length
        chunked_dataset = self._create_chunked_dataset_streaming(tokenized_dataset, chunk_size)
        
        # Calculate and display chunked stats
        chunked_stats = self._calculate_dataset_stats(chunked_dataset)
        print(f"  - Chunked dataset statistics:")
        print(f"    - Chunks: {chunked_stats['num_sequences']:,}")
        print(f"    - Total tokens: {chunked_stats['total_tokens']:,}")
        print(f"    - Chunk size: {chunk_size} tokens (fixed)")
        
        # Save chunked dataset
        self._save_chunked_dataset(chunked_dataset)
        
        return True
    
    def create_dataloader(self, tokenizer) -> DataLoader:
        """
        Create a DataLoader for training.
        
        Args:
            tokenizer: The tokenizer to use for padding
            
        Returns:
            Configured DataLoader
        """
        print(f"  - Creating DataLoader...")
        
        # Load chunked dataset
        dataset = self._load_chunked_dataset()
        if dataset is None:
            raise RuntimeError("Chunked dataset not found. Run preprocessing first.")
        
        # Set up data collator
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # For efficiency on modern hardware
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            num_workers=4,  # Adjust based on your system
            pin_memory=True if torch.cuda.is_available() else False,
            worker_init_fn=_worker_init_fn
        )
        
        print(f"    - Batch size: {self.config.data.batch_size}")
        print(f"    - Sequence length: {self.config.data.max_sequence_length}")
        print(f"    - Batches per epoch: {len(dataloader)}")
        
        return dataloader
    
    def get_training_steps_per_epoch(self) -> int:
        """
        Calculate the number of training steps per epoch.
        
        Returns:
            Number of steps per epoch
        """
        dataset = self._load_chunked_dataset()
        if dataset is None:
            raise RuntimeError("Chunked dataset not found. Run preprocessing first.")
        
        num_chunks = len(dataset)
        steps_per_epoch = math.ceil(num_chunks / self.config.data.batch_size)
        
        return steps_per_epoch
    
    def load_test_dataset(self) -> Optional[Dataset]:
        """
        Load the test dataset for evaluation.
        
        Returns:
            Test dataset or None if not available
        """
        test_dir = os.path.join(self.tokenized_data_dir, "test")
        if not os.path.exists(test_dir):
            print(f"  ⚠ Test dataset not found at: {test_dir}")
            return None
        
        try:
            test_dataset = load_from_disk(test_dir)
            print(f"  ✓ Test dataset loaded successfully")
            print(f"    - Test size: {len(test_dataset):,} examples")
            return test_dataset
        except Exception as e:
            print(f"  ✗ Error loading test dataset: {e}")
            return None


def create_data_processor(config, base_dir: str) -> DataProcessor:
    """
    Factory function to create a DataProcessor instance.
    
    Args:
        config: Experiment configuration
        base_dir: Project base directory
        
    Returns:
        Configured DataProcessor instance
    """
    return DataProcessor(config, base_dir) 