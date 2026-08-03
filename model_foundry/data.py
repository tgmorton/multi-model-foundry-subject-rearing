"""
Data handling module for the Model Foundry framework.

This module provides utilities for preprocessing, chunking, and loading
tokenized datasets for language model training.
"""

import logging
import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
import numpy as np
from datasets import Dataset, load_from_disk, disable_progress_bar
from torch.utils.data import DataLoader
import torch

from .data_collators import get_data_collator
from .cache_keys import compute_cache_key, cache_meta, resolve_cached_path

logger = logging.getLogger(__name__)

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
    # Ensure the seed stays within numpy's valid range (0 to 2^32 - 1)
    base_seed = torch.initial_seed() % (2**32 - 1)
    worker_seed = (base_seed + worker_id) % (2**32 - 1)
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

        # 0.2 — content-addressed cache paths. ``cache_key`` is computed
        # from corpus + tokenizer + seq_length + manipulation, so runs
        # that share those ingredients share tokenized/chunked data.
        # Tokenizer artefact lookup can fail (e.g. pre-tokenizer-train
        # instantiation in tests); fall back to legacy paths in that case.
        training_corpus_path = config.data.training_corpus
        if not os.path.isabs(training_corpus_path):
            training_corpus_path = os.path.join(base_dir, training_corpus_path)
        tokenizer_dir = config.tokenizer.output_dir
        if not os.path.isabs(tokenizer_dir):
            tokenizer_dir = os.path.join(base_dir, tokenizer_dir)
        manipulation = config.dataset_manipulation or []

        try:
            self.cache_key: Optional[str] = compute_cache_key(
                training_corpus_path,
                tokenizer_dir,
                config.data.max_sequence_length,
                manipulation,
            )
        except FileNotFoundError:
            # Tokenizer hasn't been trained yet (e.g. first-run ordering in
            # the pipeline, or test harnesses). Disable hashing and rely
            # entirely on the legacy experiment_name path — the writer in
            # tokenize_dataset.py will populate the hashed path once the
            # tokenizer exists.
            self.cache_key = None

        self.tokenized_data_dir = self._resolve_cache_path("tokenized")
        self.chunked_data_dir = self._resolve_cache_path("chunked")
        self.test_data_dir = os.path.join(self.tokenized_data_dir, "test")
        # Cache to avoid re-loading from disk repeatedly
        self._cached_chunked_dataset: Optional[Dataset] = None

    def _resolve_cache_path(self, cache_type: str) -> str:
        """Resolve a cache directory, preferring the hashed path.

        Falls back to the legacy experiment_name-keyed location so runs
        that predate 0.2 (e.g. the in-flight baseline) still load.
        """
        if self.cache_key is not None:
            path, kind, is_fallback = resolve_cached_path(
                self.base_dir,
                cache_type,
                self.cache_key,
                experiment_name=self.config.experiment_name,
            )
            if is_fallback:
                logger.info(
                    "Using legacy %s cache at %s (hashed path %s not populated yet)",
                    cache_type, path, self.cache_key,
                )
            return path
        # No cache key (tokenizer not yet trained); fall back to legacy.
        return os.path.join(
            self.base_dir, "data", cache_type, self.config.experiment_name
        )
        
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
        Uses concatenation to minimize token waste.
        
        Args:
            dataset: Tokenized dataset
            chunk_size: Target chunk size in tokens
            
        Yields:
            Batches of fixed-length chunks
        """
        batch_size = 1000  # Process sequences in batches to balance memory and speed
        concatenated_buffer = []  # Buffer to accumulate tokens across batches
        
        for i in range(0, len(dataset), batch_size):
            batch_sequences = dataset[i:i + batch_size]['input_ids']
            
            # Concatenate all sequences in this batch to the buffer
            for sequence in batch_sequences:
                concatenated_buffer.extend(sequence)
            
            # Extract as many complete chunks as possible from the buffer
            chunks = []
            while len(concatenated_buffer) >= chunk_size:
                chunk = concatenated_buffer[:chunk_size]
                chunks.append(chunk)
                concatenated_buffer = concatenated_buffer[chunk_size:]
            
            if chunks:
                yield chunks
        
        # Note: Any remaining tokens in concatenated_buffer (< chunk_size) are dropped
    
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
        
        # Create new dataset. Store token ids as int32 (vocab 50,004 ≪
        # 2^31): halves on-disk bytes and CephFS read volume vs the
        # arrow default of int64. Collators cast back to torch.long at
        # batch time, so nothing downstream changes.
        from datasets import Features, Sequence as HFSequence, Value
        chunked_dataset = Dataset.from_dict(
            {'input_ids': all_chunks},
            features=Features({'input_ids': HFSequence(Value('int32'))}),
        )

        return chunked_dataset
    
    def _save_chunked_dataset(self, dataset: Dataset) -> None:
        """Save the chunked dataset to disk."""
        os.makedirs(self.chunked_data_dir, exist_ok=True)
        dataset.save_to_disk(self.chunked_data_dir)
        print(f"  ✓ Saved chunked dataset to: {self.chunked_data_dir}")

        # 0.2 — sidecar metadata so the hashed dir is self-describing.
        # Only write when we actually resolved the hashed path (cache_key
        # is set and we're writing into a cache-key-named directory).
        if self.cache_key and self.cache_key in self.chunked_data_dir:
            import json as _json
            training_corpus_path = self.config.data.training_corpus
            if not os.path.isabs(training_corpus_path):
                training_corpus_path = os.path.join(self.base_dir, training_corpus_path)
            tokenizer_dir = self.config.tokenizer.output_dir
            if not os.path.isabs(tokenizer_dir):
                tokenizer_dir = os.path.join(self.base_dir, tokenizer_dir)
            try:
                meta = cache_meta(
                    training_corpus_path,
                    tokenizer_dir,
                    self.config.data.max_sequence_length,
                    self.config.dataset_manipulation or [],
                    extra={
                        "first_experiment_name": self.config.experiment_name,
                        "cache_key": self.cache_key,
                        "stage": "chunked",
                    },
                )
                with open(os.path.join(self.chunked_data_dir, "meta.json"), "w", encoding="utf-8") as f:
                    _json.dump(meta, f, indent=2)
            except FileNotFoundError:
                # Tokenizer disappeared between init and save — don't fail
                # the chunking just for sidecar metadata.
                pass
    
    def _load_chunked_dataset(self) -> Optional[Dataset]:
        """Load the chunked dataset from disk."""
        # Return cached instance if available
        if self._cached_chunked_dataset is not None:
            return self._cached_chunked_dataset

        if not os.path.exists(self.chunked_data_dir):
            return None
            
        try:
            # Load dataset with memory mapping to avoid loading everything into RAM
            dataset = load_from_disk(self.chunked_data_dir)
            print(f"  ✓ Loaded chunked dataset from: {self.chunked_data_dir}")
            print(f"    - Dataset size: {len(dataset):,} chunks")
            
            # Convert to memory-mapped format to reduce RAM usage
            dataset.set_format(type=None)  # Remove any format to use memory mapping
            # Cache for subsequent requests
            self._cached_chunked_dataset = dataset
            return self._cached_chunked_dataset
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

        # Defensive: CLI path can pass force_reprocess as the string
        # "False" instead of bool False, silently defeating the cache.
        if isinstance(force_reprocess, str):
            force_reprocess = force_reprocess.strip().lower() in ("1", "true", "yes", "y")

        # Prime CephFS attr cache — freshly-mounted PVCs can report
        # existing entries as absent for a few seconds.
        try:
            parent = os.path.dirname(self.chunked_data_dir)
            if os.path.isdir(parent):
                os.listdir(parent)
        except OSError:
            pass

        # Check if chunked data already exists. The canonical sentinel is
        # dataset_info.json inside the directory, not just the dir itself
        # (a previous crash may have left an empty dir). But we also
        # accept the dir-only case for backward compat with earlier
        # chunked outputs that didn't write the sentinel at a consistent
        # path.
        chunked_sentinel = os.path.join(self.chunked_data_dir, 'dataset_info.json')
        if not force_reprocess and (os.path.exists(chunked_sentinel) or os.path.exists(self.chunked_data_dir)):
            print(f"  - Chunked dataset already exists at: {self.chunked_data_dir}")
            return True
        if not force_reprocess:
            print(f"  - Cache miss — chunked dir {self.chunked_data_dir} exists={os.path.exists(self.chunked_data_dir)}, "
                  f"sentinel exists={os.path.exists(chunked_sentinel)}")
        
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
        logger.debug("Loading chunked dataset...")

        # Load chunked dataset
        dataset = self._load_chunked_dataset()
        if dataset is None:
            raise RuntimeError("Chunked dataset not found. Run preprocessing first.")

        logger.debug(f"Dataset loaded, size: {len(dataset)}")

        # Set up data collator based on training objective
        logger.debug(f"Creating data collator for objective: {self.config.training.objective}")
        data_collator = get_data_collator(self.config, tokenizer)
        
        # Create DataLoader with high-performance optimizations
        import os
        # num_workers: under K8s, os.cpu_count() returns the host CPU count
        # (32+), but our cgroup allocation is typically 4. Use
        # sched_getaffinity if available.
        #
        # 0.4 gives us typed fields; 1.2 raises the auto ceiling from 4 to
        # 12 so that once FA2 shifts the bottleneck from compute to I/O
        # (seen via 1.1's data_ms metric), a K8s cpu: bump alone is
        # enough to unlock more workers without another code change.
        if self.config.data.num_workers is not None:
            num_workers = self.config.data.num_workers
        else:
            try:
                available = len(os.sched_getaffinity(0))
            except AttributeError:
                available = os.cpu_count() or 1
            num_workers = max(1, min(12, available // 2 or 1))
        pin_memory = self.config.data.pin_memory
        prefetch_factor = self.config.data.prefetch_factor
        
        logger.debug(f"Creating DataLoader with num_workers={num_workers}, pin_memory={pin_memory}, prefetch_factor={prefetch_factor}")

        # Dedicated data-order generator (replicability redesign): the
        # shuffle permutation must be a pure function of (seed, epoch),
        # NOT of the ambient global RNG — whose state at iter() time
        # depends on how many draws model init happened to consume.
        # The training loop reseeds this generator to
        # derive_seed(seed, "data_order", epoch) before every epoch's
        # iter() (loop.py); the seeding here covers epoch 0 / non-loop
        # consumers. DataLoader also derives its worker base seeds from
        # this generator, so worker streams inherit the determinism.
        from .rng import derive_seed
        data_generator = torch.Generator()
        data_generator.manual_seed(
            derive_seed(int(self.config.random_seed), "data_order", 0))

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            generator=data_generator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            worker_init_fn=_worker_init_fn
        )
        
        logger.debug(f"DataLoader created successfully, len={len(dataloader)}")
        
        print("  - DataLoader configured for high throughput:")
        print(f"    - num_workers: {num_workers}, pin_memory: {pin_memory}, prefetch_factor: {prefetch_factor}")
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

        # Match the actual loop: DataLoader yields physical batches and the
        # optimizer steps only after a complete accumulation window. A final
        # partial window is intentionally dropped.
        from .checkpoint_schedule import compute_opt_steps_per_epoch

        return compute_opt_steps_per_epoch(
            num_chunks=num_chunks,
            phys_batch=self.config.data.batch_size,
            grad_accum=self.config.training.gradient_accumulation_steps,
        )
    
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
