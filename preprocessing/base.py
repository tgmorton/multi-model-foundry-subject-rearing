"""
Ablation Pipeline Base Class

Core infrastructure for running ablation transformations on text corpora.
Handles file I/O, progress tracking, validation, and provenance tracking.
"""

import glob
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import spacy
from tqdm import tqdm

from .config import (
    AblationConfig,
    FileStatistics,
    ProvenanceManifest,
    ProvenanceMetadata
)
from .registry import AblationRegistry
from .utils import (
    compute_file_checksum,
    count_tokens,
    ensure_directory_exists,
    get_spacy_device
)


class AblationPipeline:
    """
    Base class for running ablation transformations on text corpora.

    This class handles the common pipeline logic for all ablations:
    - Loading and processing files
    - Managing replacement pools
    - Validation
    - Statistics tracking
    - Provenance recording

    The actual ablation logic is delegated to registered ablation functions.

    Example:
        >>> config = AblationConfig(
        ...     type="remove_articles",
        ...     input_path="data/raw/corpus/",
        ...     output_path="data/processed/corpus/",
        ...     replacement_pool_dir="data/pool/",
        ...     seed=42
        ... )
        >>> pipeline = AblationPipeline(config)
        >>> manifest = pipeline.process_corpus()
    """

    def __init__(self, config: AblationConfig):
        """
        Initialize the ablation pipeline.

        Args:
            config: Validated ablation configuration
        """
        self.config = config
        self.logger = self._setup_logging()

        # Get ablation function from registry
        self.ablation_fn, self.validation_fn = AblationRegistry.get(config.type)

        # Load spaCy model
        self.logger.info(f"Loading spaCy model: {config.spacy_model}")
        self.nlp = self._load_spacy_model()

        # Initialize provenance tracking
        self.manifest: Optional[ProvenanceManifest] = None

    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging for this pipeline.

        Returns:
            Configured logger instance
        """
        from model_foundry.logging_utils import setup_logging

        logger = setup_logging(
            name=f"preprocessing.{self.config.type}",
            experiment=f"{self.config.type}",
            phase="ablation",
            log_dir=str(self.config.log_dir),
            level=logging.DEBUG if self.config.verbose else logging.INFO
        )

        return logger

    def _load_spacy_model(self) -> spacy.Language:
        """
        Load spaCy model with device configuration.

        Returns:
            Loaded spaCy Language model

        Raises:
            OSError: If specified model cannot be loaded
        """
        # Determine device
        if self.config.spacy_device:
            device = self.config.spacy_device
        else:
            device = get_spacy_device(verbose=self.config.verbose)

        # Set spaCy to use the device
        if device != "cpu":
            spacy.prefer_gpu()

        # Load model
        try:
            nlp = spacy.load(self.config.spacy_model)
            self.logger.info(f"Loaded model: {self.config.spacy_model} on device: {device}")
        except OSError as e:
            self.logger.error(f"Failed to load spaCy model '{self.config.spacy_model}': {e}")
            raise

        # Increase max_length for large texts
        nlp.max_length = 2000000  # 2M characters

        return nlp

    def process_corpus(self) -> ProvenanceManifest:
        """
        Process all files in the input corpus.

        This is the main entry point for the pipeline. It:
        1. Finds all files to process
        2. Processes each file with the ablation
        3. Tracks statistics and provenance
        4. Saves the provenance manifest

        Returns:
            ProvenanceManifest with complete processing information

        Raises:
            FileNotFoundError: If input directory doesn't exist
            ValueError: If no files found to process
        """
        start_time = time.time()

        # Initialize provenance tracking
        self._initialize_provenance()

        # Find all files to process
        search_pattern = os.path.join(self.config.input_path, '**', '*.train')
        source_files = sorted(glob.glob(search_pattern, recursive=True))

        if not source_files:
            raise ValueError(f"No '.train' files found in {self.config.input_path}")

        self.logger.info(f"Found {len(source_files)} files to process")

        # Process each file
        for source_path in tqdm(source_files, desc="Processing files"):
            try:
                file_stats = self._process_file(Path(source_path))
                self.manifest.add_file_stats(file_stats)
                self.logger.info(
                    f"Processed {file_stats.file_name}: "
                    f"{file_stats.items_ablated:,} items ablated"
                )
            except Exception as e:
                self.logger.error(f"Failed to process {source_path}: {e}")
                raise

        # Finalize provenance
        self.manifest.metadata.processing_time_seconds = time.time() - start_time

        # Save manifest
        manifest_path = self.manifest.save(self.config.output_path)
        self.logger.info(f"Saved provenance manifest to {manifest_path}")

        return self.manifest

    def _initialize_provenance(self) -> None:
        """Initialize the provenance manifest with environment metadata."""
        # Get spaCy model metadata
        spacy_model_meta = self.nlp.meta

        metadata = ProvenanceMetadata.create_from_environment(
            ablation_type=self.config.type,
            random_seed=self.config.seed,
            chunk_size=self.config.chunk_size,
            device=get_spacy_device(),
            spacy_version=spacy.__version__,
            spacy_model_name=spacy_model_meta.get("name", self.config.spacy_model),
            spacy_model_version=spacy_model_meta.get("version", "unknown")
        )

        self.manifest = ProvenanceManifest(
            metadata=metadata,
            config=self.config.model_dump()
        )

    def _process_file(self, file_path: Path) -> FileStatistics:
        """
        Process a single file with the ablation.

        Args:
            file_path: Path to the file to process

        Returns:
            FileStatistics for the processed file
        """
        file_start_time = time.time()

        # Calculate paths
        relative_path = os.path.relpath(file_path, self.config.input_path)
        output_path = self.config.output_path / relative_path
        ensure_directory_exists(output_path.parent)

        # Read input file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            # Empty file - just create empty output
            output_path.touch()
            return FileStatistics(
                file_name=file_path.name,
                original_tokens=0,
                final_tokens=0,
                items_ablated=0,
                proportion_removed=0.0,
                processing_time_seconds=time.time() - file_start_time
            )

        # Calculate original token count
        original_text = "".join(lines)
        original_token_count = count_tokens(original_text)

        # Ablate the file
        ablated_text, items_ablated = self._ablate_lines(lines)

        # Validate ablation (if not skipped)
        if not self.config.skip_validation and self.validation_fn:
            self.logger.info(f"Validating ablation for {file_path.name}")
            is_valid = self.validation_fn(original_text, ablated_text, self.nlp)
            if not is_valid:
                self.logger.warning(
                    f"Validation failed for {file_path.name}: "
                    "ablation may not have occurred as expected"
                )

        # Rebuild to target size if replacement pool provided
        current_token_count = count_tokens(ablated_text)
        if self.config.replacement_pool_dir and current_token_count < original_token_count:
            ablated_text, additional_items = self._rebuild_to_target_size(
                ablated_text=ablated_text,
                target_token_count=original_token_count,
                source_file_path=file_path
            )
            items_ablated += additional_items

        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ablated_text)

        # Calculate final token count
        final_token_count = count_tokens(ablated_text)

        # Compute checksums
        input_checksum = compute_file_checksum(file_path)
        output_checksum = compute_file_checksum(output_path)

        # Update manifest checksums
        self.manifest.metadata.input_checksums[file_path.name] = input_checksum
        self.manifest.metadata.output_checksums[output_path.name] = output_checksum

        # Create file statistics
        tokens_removed = original_token_count - final_token_count
        proportion_removed = tokens_removed / original_token_count if original_token_count > 0 else 0.0

        return FileStatistics(
            file_name=file_path.name,
            original_tokens=original_token_count,
            final_tokens=final_token_count,
            items_ablated=items_ablated,
            proportion_removed=proportion_removed,
            processing_time_seconds=time.time() - file_start_time
        )

    def _ablate_lines(self, lines: List[str]) -> Tuple[str, int]:
        """
        Ablate a list of lines using the registered ablation function.

        Args:
            lines: List of text lines to process

        Returns:
            Tuple of (ablated_text, total_items_ablated)
        """
        ablated_text = ""
        total_items_ablated = 0

        # Process in chunks for memory efficiency
        with tqdm(
            total=len(lines),
            desc=f"  Ablating",
            leave=False,
            disable=not self.config.verbose
        ) as pbar:
            for i in range(0, len(lines), self.config.chunk_size):
                chunk = lines[i:i + self.config.chunk_size]

                # Process chunk with spaCy pipeline
                for doc in self.nlp.pipe(chunk):
                    ablated_doc_text, num_items = self.ablation_fn(doc)
                    ablated_text += ablated_doc_text
                    total_items_ablated += num_items

                pbar.update(len(chunk))

        return ablated_text, total_items_ablated

    def _rebuild_to_target_size(
        self,
        ablated_text: str,
        target_token_count: int,
        source_file_path: Path
    ) -> Tuple[str, int]:
        """
        Rebuild corpus to target token count using replacement pool.

        Args:
            ablated_text: Text after ablation
            target_token_count: Target token count to reach
            source_file_path: Path to source file (for finding pool file)

        Returns:
            Tuple of (rebuilt_text, additional_items_ablated)
        """
        # Find corresponding pool file
        relative_path = os.path.relpath(source_file_path, self.config.input_path)
        pool_path = self.config.replacement_pool_dir / relative_path

        if not pool_path.exists():
            self.logger.warning(
                f"No replacement pool found for {source_file_path.name}. "
                "Cannot rebuild to target size."
            )
            return ablated_text, 0

        # Load replacement pool
        with open(pool_path, 'r', encoding='utf-8') as f:
            replacement_pool_sentences = f.readlines()

        if not replacement_pool_sentences:
            self.logger.warning(f"Replacement pool for {source_file_path.name} is empty.")
            return ablated_text, 0

        current_token_count = count_tokens(ablated_text)
        additional_items_ablated = 0

        with tqdm(
            total=target_token_count,
            initial=current_token_count,
            desc="  Rebuilding",
            leave=False,
            disable=not self.config.verbose
        ) as pbar:
            while current_token_count < target_token_count and replacement_pool_sentences:
                # Sample sentences from pool
                num_to_sample = min(10, len(replacement_pool_sentences))
                sample_indices = random.sample(range(len(replacement_pool_sentences)), num_to_sample)
                sample_sentences = [replacement_pool_sentences[i] for i in sorted(sample_indices, reverse=True)]

                # Remove sampled sentences from pool
                for idx in sorted(sample_indices, reverse=True):
                    replacement_pool_sentences.pop(idx)

                # Ablate sampled sentences
                sample_text = "".join(sample_sentences)
                for doc in self.nlp.pipe([sample_text]):
                    ablated_sample, num_items = self.ablation_fn(doc)
                    ablated_text += ablated_sample
                    additional_items_ablated += num_items

                    # Update progress
                    added_tokens = count_tokens(ablated_sample)
                    current_token_count += added_tokens
                    pbar.update(added_tokens)

        # Save remainder of replacement pool
        if replacement_pool_sentences:
            remainder_dir = self.config.output_path / "replacement_pool_remainder"
            ensure_directory_exists(remainder_dir)

            base_name = source_file_path.stem  # Removes .train extension
            remainder_path = remainder_dir / f"{base_name}.txt"

            with open(remainder_path, 'w', encoding='utf-8') as f:
                f.writelines(replacement_pool_sentences)

            self.logger.debug(f"Saved {len(replacement_pool_sentences)} unused pool sentences")

        return ablated_text, additional_items_ablated
