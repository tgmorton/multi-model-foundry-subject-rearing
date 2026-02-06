"""Full-corpus pronoun insertion pipeline.

Processes every ``.train`` file under the configured input directory through
the sequence labeler and inserts recovered pronouns, writing the results
as new ``.train`` files with a provenance manifest.

This module mirrors the :class:`~analysis.corpus_descriptives.pipeline.CorpusAnalysisPipeline`
pattern: iterate files, process in chunks via spaCy + model inference,
and track provenance (checksums, timing, model version).
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import spacy
from tqdm import tqdm

from preprocessing.utils import get_spacy_device, setup_logging

from ..config import InsertionConfig
from ..model.sequence_labeler import PronounRecoveryModel
from .inserter import PronounInserter

logger = logging.getLogger(__name__)


class CorpusInsertionPipeline:
    """Process an entire corpus through model inference + pronoun insertion.

    Follows the established pipeline pattern:

    1. Find all ``.train`` files in the input directory.
    2. For each file, read lines, process through spaCy in chunks, run
       the sequence labeler on each doc, and insert recovered pronouns.
    3. Write the pronoun-recovered text to an identically named
       ``.train`` file in the output directory.
    4. Save a ``provenance.json`` manifest with checksums, timing,
       model metadata, and line counts.

    Args:
        config: An :class:`~analysis.pronoun_recovery.config.InsertionConfig`
            specifying paths, model settings, and processing parameters.
    """

    def __init__(self, config: InsertionConfig) -> None:
        self.config = config
        self.logger = setup_logging(
            name="pronoun_insertion",
            experiment=config.language,
            phase="insertion",
        )

    # ── Public API ───────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Run the full insertion pipeline.

        Steps:

        1. Load the spaCy language model.
        2. Load the trained pronoun recovery sequence labeler.
        3. For each ``.train`` file:

           a. Read all lines.
           b. Process through spaCy in batches.
           c. Run model inference on each doc to obtain per-token labels.
           d. Insert recovered pronouns via :class:`.PronounInserter`.
           e. Write the recovered text to the output directory.

        4. Save provenance metadata.

        Returns:
            A metadata dict containing timing, input/output checksums,
            per-file line counts, and model version information.

        Raises:
            FileNotFoundError: If no ``.train`` files are found in the
                input directory.
        """
        pipeline_start = time.time()

        # ---- Load models ----
        nlp = self._load_spacy()
        model = self._load_model()
        inserter = PronounInserter(language=self.config.language)

        # ---- Discover corpus files ----
        input_path = Path(self.config.input_path)
        train_files = sorted(input_path.glob("*.train"))
        if not train_files:
            raise FileNotFoundError(f"No .train files in {input_path}")
        self.logger.info(f"Found {len(train_files)} corpus files in {input_path}")

        # ---- Process each file ----
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_metadata: Dict[str, Dict[str, Any]] = {}
        total_lines = 0

        for fpath in tqdm(train_files, desc="Files"):
            genre = self._resolve_genre(fpath)
            if genre is None:
                self.logger.warning(f"Unknown genre for {fpath.name}, skipping")
                continue

            fmeta = self._process_file(fpath, genre, nlp, model, inserter)
            file_metadata[fpath.name] = fmeta
            total_lines += fmeta["line_count"]

        # ---- Assemble provenance metadata ----
        elapsed = time.time() - pipeline_start
        metadata: Dict[str, Any] = {
            "language": self.config.language,
            "spacy_model": self.config.spacy_model,
            "spacy_version": spacy.__version__,
            "model_path": str(self.config.model_path),
            "input_path": str(self.config.input_path),
            "output_path": str(self.config.output_path),
            "total_lines": total_lines,
            "total_files": len(file_metadata),
            "processing_time_seconds": elapsed,
            "files": file_metadata,
        }

        self._save_provenance(metadata)
        self.logger.info(
            f"Pipeline complete: {total_lines:,} lines across "
            f"{len(file_metadata)} files in {elapsed:.1f}s"
        )
        return metadata

    # ── Model loading ────────────────────────────────────────────────────

    def _load_spacy(self) -> spacy.Language:
        """Load the spaCy model with GPU auto-detection."""
        device = get_spacy_device(verbose=True)
        if device != "cpu":
            spacy.prefer_gpu()

        self.logger.info(
            f"Loading spaCy model: {self.config.spacy_model} (device={device})"
        )
        nlp = spacy.load(self.config.spacy_model)
        nlp.max_length = 2_000_000
        return nlp

    def _load_model(self) -> PronounRecoveryModel:
        """Load the trained pronoun recovery sequence labeler."""
        model_path = Path(self.config.model_path)
        threshold = getattr(self.config, "insertion_threshold", 0.0)
        self.logger.info(
            f"Loading pronoun recovery model from {model_path} (threshold={threshold})"
        )
        model = PronounRecoveryModel.from_pretrained(
            model_path,
            device=self.config.device,
            threshold=threshold,
        )
        return model

    # ── File-level processing ────────────────────────────────────────────

    def _resolve_genre(self, fpath: Path) -> Optional[str]:
        """Resolve genre display name from filename stem via config map."""
        return self.config.genre_map.get(fpath.stem)

    def _process_file(
        self,
        fpath: Path,
        genre: str,
        nlp: spacy.Language,
        model: PronounRecoveryModel,
        inserter: PronounInserter,
    ) -> Dict[str, Any]:
        """Process a single ``.train`` file.

        Reads lines, processes them through spaCy + model + inserter,
        and writes the recovered text to the output directory.

        Args:
            fpath: Path to the input ``.train`` file.
            genre: Genre display name.
            nlp: Loaded spaCy language pipeline.
            model: Trained pronoun recovery sequence labeler.
            inserter: Configured :class:`.PronounInserter` instance.

        Returns:
            Metadata dict for this file, containing line count, input/output
            checksums, and per-file timing.
        """
        file_start = time.time()
        self.logger.info(f"Processing {fpath.name} (genre={genre})")

        # Compute input checksum
        input_checksum = self._compute_checksum(fpath)

        # Read all lines
        with open(fpath, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        line_count = len(raw_lines)
        output_lines: List[str] = []

        # Process in chunks
        num_chunks = (line_count + self.config.chunk_size - 1) // self.config.chunk_size
        for chunk_start in tqdm(
            range(0, line_count, self.config.chunk_size),
            total=num_chunks,
            desc=fpath.name,
            leave=False,
        ):
            chunk = raw_lines[chunk_start : chunk_start + self.config.chunk_size]

            # Separate non-empty lines for spaCy processing
            texts: List[str] = []
            text_indices: List[int] = []  # index within chunk
            for i, raw_line in enumerate(chunk):
                text = raw_line.strip()
                if text:
                    texts.append(text)
                    text_indices.append(i)

            # SpaCy processing
            if texts:
                docs = list(
                    nlp.pipe(texts, batch_size=self.config.spacy_batch_size)
                )

                # Model inference + insertion
                recovered_texts: Dict[int, str] = {}
                for doc, chunk_idx in zip(docs, text_indices):
                    predictions = model.predict_doc(doc)
                    recovered = inserter.insert(doc, predictions)
                    recovered_texts[chunk_idx] = recovered

                # Rebuild chunk lines preserving empty lines and newlines
                for i, raw_line in enumerate(chunk):
                    if i in recovered_texts:
                        # Preserve the original line ending
                        ending = ""
                        if raw_line.endswith("\n"):
                            ending = "\n"
                        output_lines.append(recovered_texts[i] + ending)
                    else:
                        # Empty line or whitespace-only: preserve as-is
                        output_lines.append(raw_line)
            else:
                # Entire chunk was empty — preserve as-is
                output_lines.extend(chunk)

        # Write output file
        output_path = Path(self.config.output_path) / fpath.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(output_lines)

        # Compute output checksum
        output_checksum = self._compute_checksum(output_path)

        file_elapsed = time.time() - file_start
        self.logger.info(
            f"  {fpath.name}: {line_count:,} lines in {file_elapsed:.1f}s"
        )

        return {
            "genre": genre,
            "line_count": line_count,
            "input_checksum": input_checksum,
            "output_checksum": output_checksum,
            "processing_time_seconds": file_elapsed,
        }

    # ── Utilities ────────────────────────────────────────────────────────

    def _compute_checksum(self, fpath: Path) -> str:
        """Compute SHA-256 checksum of a file.

        Args:
            fpath: Path to the file.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _save_provenance(self, metadata: Dict[str, Any]) -> None:
        """Save provenance metadata as JSON.

        Args:
            metadata: The full pipeline metadata dict.
        """
        output_path = Path(self.config.output_path) / "provenance.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logger.info(f"Provenance saved to {output_path}")
