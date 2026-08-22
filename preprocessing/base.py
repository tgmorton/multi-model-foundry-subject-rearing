"""
Ablation Pipeline Base Class

Core infrastructure for running ablation transformations on text corpora.
Handles file I/O, progress tracking, validation, and provenance tracking.
"""

import glob
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

        # Parameterized ablations (e.g. graded removal) expose configure();
        # plain callables ignore config.parameters as before.
        if hasattr(self.ablation_fn, "configure"):
            self.ablation_fn.configure(config.parameters or {})

        # Load spaCy model
        self.logger.info(f"Loading spaCy model: {config.spacy_model}")
        self.nlp = self._load_spacy_model()

        # Initialize provenance tracking
        self.manifest: Optional[ProvenanceManifest] = None

        # Deterministic replacement-pool sampling. Uses a private RNG
        # rather than global `random.seed(...)` so the pipeline doesn't
        # disturb (or get disturbed by) other random state in the process.
        # Matters for reproducibility AND for live-vs-cached parity —
        # both paths produce identical pool draws given the same seed.
        self._rng = random.Random(config.seed)

    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging for this pipeline.

        Returns:
            Configured logger instance
        """
        from preprocessing.utils import setup_logging

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
        Load spaCy model with device configuration and performance optimizations.

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

        # Disable unused components for performance
        if self.config.spacy_disable_components:
            try:
                # Get currently enabled components
                enabled = [name for name, pipe in nlp.pipeline]

                # Disable requested components
                components_to_disable = [
                    comp for comp in self.config.spacy_disable_components
                    if comp in enabled
                ]

                if components_to_disable:
                    nlp.disable_pipes(*components_to_disable)
                    self.logger.info(
                        f"Disabled spaCy components for performance: {components_to_disable}"
                    )

                # Warn about components that were requested but don't exist
                invalid = [
                    comp for comp in self.config.spacy_disable_components
                    if comp not in enabled
                ]
                if invalid:
                    self.logger.warning(
                        f"Requested to disable non-existent components: {invalid}. "
                        f"Available components: {enabled}"
                    )

            except Exception as e:
                self.logger.warning(
                    f"Failed to disable spaCy components: {e}. "
                    "Continuing with default pipeline."
                )

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
        failed_files = []
        for source_path in tqdm(source_files, desc="Processing files"):
            try:
                file_stats = self._process_file(Path(source_path))
                self.manifest.add_file_stats(file_stats)
                self.logger.info(
                    f"Processed {file_stats.file_name}: "
                    f"{file_stats.items_ablated:,} items ablated"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to process {source_path}: {type(e).__name__}: {e}",
                    exc_info=self.config.verbose  # Include traceback if verbose
                )
                failed_files.append((source_path, str(e)))
                # Continue processing other files instead of failing completely
                continue

        # Report summary of failures
        if failed_files:
            self.logger.warning(
                f"Processing completed with {len(failed_files)} failed file(s) "
                f"out of {len(source_files)} total"
            )
            for failed_path, error_msg in failed_files:
                self.logger.warning(f"  - {failed_path}: {error_msg}")

            # Store failed files in manifest
            self.manifest.metadata.failed_files = failed_files

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

        # Reset per-file state on ablation function if supported
        if hasattr(self.ablation_fn, 'reset_file_state'):
            self.ablation_fn.reset_file_state()

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

        # Ablate the file — use the pre-annotated DocBin cache when available,
        # otherwise fall back to live spaCy parsing.
        use_cache = self._has_annotated_cache(file_path)
        if use_cache:
            ablated_text, items_ablated = self._ablate_from_cache(file_path)
        else:
            ablated_text, items_ablated = self._ablate_lines(lines)

        # Extract tier counts and removed line indices if supported
        tier_counts = {}
        removed_line_indices = []
        if hasattr(self.ablation_fn, 'get_file_tier_counts'):
            tier_counts = self.ablation_fn.get_file_tier_counts()
        if hasattr(self.ablation_fn, '_removed_line_indices'):
            removed_line_indices = list(self.ablation_fn._removed_line_indices)

        # Validate ablation (if not skipped)
        if not self.config.skip_validation and self.validation_fn:
            self.logger.info(f"Validating ablation for {file_path.name}")
            try:
                is_valid = self.validation_fn(original_text, ablated_text, self.nlp)
                if not is_valid:
                    self.logger.warning(
                        f"Validation failed for {file_path.name}: "
                        "ablation may not have occurred as expected. "
                        "This is not fatal - continuing with processing."
                    )
            except Exception as e:
                # Validation errors are not fatal - log and continue
                self.logger.warning(
                    f"Validation raised an exception for {file_path.name}: "
                    f"{type(e).__name__}: {e}. Skipping validation for this file."
                )

        # Rebuild to target size if replacement pool provided and backfill
        # is not explicitly disabled. skip_backfill=True turns this
        # pipeline into a pure transformer — used in the three-step
        # workflow where train and pool are ablated separately and then
        # composed by scripts/compose_corpus.py.
        pool_stats = {}
        current_token_count = count_tokens(ablated_text)
        if (
            self.config.replacement_pool_dir
            and not self.config.skip_backfill
            and current_token_count < original_token_count
        ):
            ablated_text, additional_items, pool_stats = self._rebuild_to_target_size(
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
            processing_time_seconds=time.time() - file_start_time,
            tier_counts=tier_counts,
            removed_line_indices=removed_line_indices,
            replacement_pool_size=pool_stats.get("pool_total", 0),
            replacement_lines_drawn=pool_stats.get("pool_lines_drawn", 0),
            replacement_pool_remainder=pool_stats.get("pool_remainder", 0),
        )

    def _ablate_lines(self, lines: List[str]) -> Tuple[str, int]:
        """
        Ablate a list of lines using the registered ablation function.

        Args:
            lines: List of text lines to process

        Returns:
            Tuple of (ablated_text, total_items_ablated)

        Raises:
            ValueError: If ablation function fails on a document
            RuntimeError: If spaCy pipeline encounters an error
        """
        from .annotate import is_passthrough_line

        ablated_text = ""
        total_items_ablated = 0

        # Process in chunks for memory efficiency. Trailing newlines are
        # stripped before parsing — spaCy's lemmatizer (Spanish lg model
        # especially) returns the surface form as the lemma for
        # text ending in "\n" for some verbs (e.g. *acuerdas* → *acuerdas*
        # instead of *acordar*). Re-attaching the newline after ablation
        # preserves line boundaries. Verified 2026-04-21 against
        # es_core_news_lg on CHILDES data.
        #
        # Empty lines and boundary markers (``= = = ... = = =``) are
        # passed through verbatim — parsing them with spaCy produces
        # empty docs that the ablation has nothing useful to do with,
        # and we'd lose the line break. This matches the semantics of
        # the DocBin cache reader in ``_ablate_from_cache``.
        with tqdm(
            total=len(lines),
            desc=f"  Ablating",
            leave=False,
            disable=not self.config.verbose
        ) as pbar:
            for i in range(0, len(lines), self.config.chunk_size):
                chunk = lines[i:i + self.config.chunk_size]
                # Partition into pass-through lines (written verbatim)
                # and parse lines (fed to spaCy with newlines stripped).
                parse_texts: List[str] = []
                parse_raw: List[str] = []
                parse_positions: List[int] = []
                chunk_output: List[str] = [""] * len(chunk)
                for offset, line in enumerate(chunk):
                    if is_passthrough_line(line):
                        chunk_output[offset] = line
                    else:
                        parse_texts.append(line.rstrip("\n\r"))
                        parse_raw.append(line)
                        parse_positions.append(offset)

                if parse_texts:
                    try:
                        parse_cursor = 0
                        for doc in self.nlp.pipe(
                            parse_texts, batch_size=self.config.spacy_batch_size
                        ):
                            raw = parse_raw[parse_cursor]
                            position = parse_positions[parse_cursor]
                            try:
                                ablated_doc_text, num_items = self.ablation_fn(doc)
                                # Restore any trailing newline that was
                                # in the source line but not captured by
                                # the ablation output.
                                if (
                                    ablated_doc_text
                                    and raw.endswith(("\n", "\r"))
                                    and not ablated_doc_text.endswith(("\n", "\r"))
                                ):
                                    trailing = raw[len(raw.rstrip("\r\n")):]
                                    ablated_doc_text += trailing
                                chunk_output[position] = ablated_doc_text
                                total_items_ablated += num_items
                            except Exception as e:
                                global_line_idx = i + position
                                self.logger.error(
                                    f"Ablation function failed on line {global_line_idx + 1}: "
                                    f"{type(e).__name__}: {e}"
                                )
                                raise ValueError(
                                    f"Ablation failed on line {global_line_idx + 1}"
                                ) from e
                            parse_cursor += 1

                        ablated_text += "".join(chunk_output)
                    except Exception as e:
                        # Catch spaCy pipeline errors (re-raised below as RuntimeError)
                        if not isinstance(e, ValueError):
                            self.logger.error(
                                f"spaCy pipeline error in chunk {i // self.config.chunk_size + 1}: "
                                f"{type(e).__name__}: {e}"
                            )
                            raise RuntimeError(
                                f"spaCy processing failed in chunk starting at line {i + 1}"
                            ) from e
                        raise
                else:
                    # Chunk had only passthrough lines; concatenate and move on.
                    ablated_text += "".join(chunk_output)

                pbar.update(len(chunk))

        return ablated_text, total_items_ablated

    def _has_annotated_cache(self, source_path: Path) -> bool:
        """Return True if the config points at a valid DocBin cache for this file."""
        annotated_dir = self.config.annotated_input_path
        if annotated_dir is None:
            return False
        stem = source_path.stem
        docbin_path = annotated_dir / f"{stem}.spacy"
        linemap_path = annotated_dir / f"{stem}.linemap.jsonl"
        if not (docbin_path.exists() and linemap_path.exists()):
            self.logger.warning(
                "annotated_input_path set but cache missing for %s "
                "(%s / %s). Falling back to live parsing.",
                source_path.name,
                docbin_path.name,
                linemap_path.name,
            )
            return False
        return True

    def _ablate_from_cache(self, source_path: Path) -> Tuple[str, int]:
        """Apply the ablation using pre-annotated DocBin + line map.

        Streams Docs keyed by ``doc_idx`` from ``{stem}.spacy`` and
        iterates ``{stem}.linemap.jsonl`` in source-line order. Pass-through
        lines (empty / boundary markers; doc_idx is null) are written
        verbatim from ``raw_text``. Content lines are handed to the
        registered ablation function.

        Uses :func:`preprocessing.annotate.iter_annotated_file` so memory
        is O(1) in the number of docs — critical for large corpora like
        Spanish ``europarl.train`` (1.24M docs → ~8 GB if materialized).
        Output is accumulated in a list of chunks joined at the end to
        avoid quadratic string-concatenation overhead.

        Requires a DocBin produced with the same spaCy model currently
        loaded into self.nlp (the vocab must match).
        """
        from .annotate import iter_annotated_file

        annotated_dir = self.config.annotated_input_path
        stem = source_path.stem

        self.logger.info(
            "Using annotated cache for %s (dir=%s)",
            source_path.name,
            annotated_dir,
        )

        output_chunks: List[str] = []
        total_items_ablated = 0

        for entry, doc in iter_annotated_file(
            annotated_dir=annotated_dir,
            file_stem=stem,
            vocab=self.nlp.vocab,
        ):
            raw_text = entry.get("raw_text", "")
            if doc is None:
                # Pass-through: empty line or document boundary marker.
                output_chunks.append(raw_text)
                continue

            try:
                # Line-addressed ablations (graded removal) need to know
                # which source line this Doc came from.
                if hasattr(self.ablation_fn, "set_line_context"):
                    self.ablation_fn.set_line_context(stem, entry["line_idx"])
                ablated_doc_text, num_items = self.ablation_fn(doc)
            except Exception as e:
                doc_idx = entry.get("doc_idx")
                line_idx = entry.get("line_idx")
                self.logger.error(
                    "Ablation failed on cached doc_idx=%s (source line %s): "
                    "%s: %s",
                    doc_idx, line_idx, type(e).__name__, e,
                )
                raise ValueError(
                    f"Ablation failed on cached doc {doc_idx} (source line {line_idx})"
                ) from e

            # The annotator may have stripped trailing whitespace (e.g., the
            # generic cleaner in corpus_descriptives calls `.strip()` before
            # parsing, so the parsed Doc has no trailing newline). Preserve
            # line boundaries by re-attaching the trailing whitespace from
            # the original raw line if the ablation output is missing it
            # AND the ablation didn't remove the line entirely.
            if ablated_doc_text and raw_text.endswith(("\n", "\r")):
                if not ablated_doc_text.endswith(("\n", "\r")):
                    # Copy the exact trailing whitespace run from raw_text
                    trailing = raw_text[len(raw_text.rstrip("\r\n")):]
                    ablated_doc_text = ablated_doc_text + trailing

            output_chunks.append(ablated_doc_text)
            total_items_ablated += num_items

        return "".join(output_chunks), total_items_ablated

    def _rebuild_to_target_size(
        self,
        ablated_text: str,
        target_token_count: int,
        source_file_path: Path
    ) -> Tuple[str, int, dict]:
        """
        Rebuild corpus to target token count using replacement pool.

        Args:
            ablated_text: Text after ablation
            target_token_count: Target token count to reach
            source_file_path: Path to source file (for finding pool file)

        Returns:
            Tuple of (rebuilt_text, additional_items_ablated, pool_stats)
        """
        # Find corresponding pool file
        relative_path = os.path.relpath(source_file_path, self.config.input_path)
        pool_path = self.config.replacement_pool_dir / relative_path

        if not pool_path.exists():
            self.logger.warning(
                f"No replacement pool found for {source_file_path.name}. "
                "Cannot rebuild to target size."
            )
            return ablated_text, 0, {}

        # Load replacement pool
        with open(pool_path, 'r', encoding='utf-8') as f:
            replacement_pool_sentences = f.readlines()

        if not replacement_pool_sentences:
            self.logger.warning(f"Replacement pool for {source_file_path.name} is empty.")
            return ablated_text, 0, {}

        pool_total = len(replacement_pool_sentences)
        current_token_count = count_tokens(ablated_text)
        additional_items_ablated = 0
        pool_lines_drawn = 0

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
                sample_indices = self._rng.sample(range(len(replacement_pool_sentences)), num_to_sample)
                sample_sentences = [replacement_pool_sentences[i] for i in sorted(sample_indices, reverse=True)]

                # Remove sampled sentences from pool
                for idx in sorted(sample_indices, reverse=True):
                    replacement_pool_sentences.pop(idx)

                pool_lines_drawn += num_to_sample

                # Ablate sampled sentences
                sample_text = "".join(sample_sentences)
                for doc in self.nlp.pipe([sample_text], batch_size=self.config.spacy_batch_size):
                    ablated_sample, num_items = self.ablation_fn(doc)
                    ablated_text += ablated_sample
                    additional_items_ablated += num_items

                    # Update progress
                    added_tokens = count_tokens(ablated_sample)
                    current_token_count += added_tokens
                    pbar.update(added_tokens)

        pool_remainder = len(replacement_pool_sentences)

        # Save remainder of replacement pool
        if replacement_pool_sentences:
            remainder_dir = self.config.output_path / "replacement_pool_remainder"
            ensure_directory_exists(remainder_dir)

            base_name = source_file_path.stem  # Removes .train extension
            remainder_path = remainder_dir / f"{base_name}.txt"

            with open(remainder_path, 'w', encoding='utf-8') as f:
                f.writelines(replacement_pool_sentences)

            self.logger.debug(f"Saved {len(replacement_pool_sentences)} unused pool sentences")

        pool_stats = {
            "pool_total": pool_total,
            "pool_lines_drawn": pool_lines_drawn,
            "pool_remainder": pool_remainder,
        }

        return ablated_text, additional_items_ablated, pool_stats
