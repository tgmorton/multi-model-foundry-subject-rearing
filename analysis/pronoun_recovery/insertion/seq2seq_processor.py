"""Seq2seq-based corpus pronoun insertion pipeline.

Processes every ``.train`` file under the configured input directory through
the seq2seq (mT5) model, which directly generates pronoun-recovered text.
No spaCy processing is needed since the model operates on raw text.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from preprocessing.utils import setup_logging

from ..config import Seq2SeqInsertionConfig
from ..model.seq2seq_model import Seq2SeqPronounModel

logger = logging.getLogger(__name__)


class Seq2SeqInsertionPipeline:
    """Process an entire corpus through seq2seq pronoun recovery.

    Unlike the token-classifier pipeline, this does not require spaCy —
    the mT5 model reads raw text and generates the pronoun-recovered
    version directly.

    Steps:

    1. Find all ``.train`` files in the input directory.
    2. For each file, read lines, batch through the seq2seq model,
       and write the generated output.
    3. Save a ``provenance.json`` manifest.

    Args:
        config: A :class:`Seq2SeqInsertionConfig` specifying paths,
            model settings, and processing parameters.
    """

    def __init__(self, config: Seq2SeqInsertionConfig) -> None:
        self.config = config
        self.logger = setup_logging(
            name="seq2seq_insertion",
            experiment=config.language,
            phase="insertion_seq2seq",
        )

    # ── Public API ───────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Run the full seq2seq insertion pipeline.

        Returns:
            A metadata dict containing timing, checksums, per-file
            line counts, and model information.
        """
        pipeline_start = time.time()

        # ---- Load model ----
        model = self._load_model()

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

            fmeta = self._process_file(fpath, genre, model)
            file_metadata[fpath.name] = fmeta
            total_lines += fmeta["line_count"]

        # ---- Assemble provenance metadata ----
        elapsed = time.time() - pipeline_start
        metadata: Dict[str, Any] = {
            "language": self.config.language,
            "model_type": "seq2seq",
            "model_path": str(self.config.model_path),
            "input_path": str(self.config.input_path),
            "output_path": str(self.config.output_path),
            "num_beams": self.config.num_beams,
            "length_penalty": self.config.length_penalty,
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

    def _load_model(self) -> Seq2SeqPronounModel:
        """Load the trained seq2seq pronoun recovery model."""
        model_path = Path(self.config.model_path)
        self.logger.info(f"Loading seq2seq model from {model_path}")
        model = Seq2SeqPronounModel.from_pretrained(
            str(model_path),
            max_input_length=self.config.max_input_length,
            max_target_length=self.config.max_target_length,
            num_beams=self.config.num_beams,
            length_penalty=self.config.length_penalty,
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
        model: Seq2SeqPronounModel,
    ) -> Dict[str, Any]:
        """Process a single ``.train`` file through the seq2seq model.

        Args:
            fpath: Path to the input ``.train`` file.
            genre: Genre display name.
            model: Loaded :class:`Seq2SeqPronounModel`.

        Returns:
            Metadata dict for this file.
        """
        file_start = time.time()
        self.logger.info(f"Processing {fpath.name} (genre={genre})")

        input_checksum = self._compute_checksum(fpath)

        with open(fpath, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        line_count = len(raw_lines)
        output_lines: List[str] = []

        # Process in chunks.
        num_chunks = (line_count + self.config.chunk_size - 1) // self.config.chunk_size
        for chunk_start in tqdm(
            range(0, line_count, self.config.chunk_size),
            total=num_chunks,
            desc=fpath.name,
            leave=False,
        ):
            chunk = raw_lines[chunk_start : chunk_start + self.config.chunk_size]

            # Separate non-empty lines for model processing.
            texts: List[str] = []
            text_indices: List[int] = []
            for i, raw_line in enumerate(chunk):
                text = raw_line.strip()
                if text:
                    texts.append(text)
                    text_indices.append(i)

            if texts:
                # Process in batches through the seq2seq model.
                recovered_texts: Dict[int, str] = {}
                for batch_start in range(0, len(texts), self.config.batch_size):
                    batch = texts[batch_start : batch_start + self.config.batch_size]
                    batch_indices = text_indices[batch_start : batch_start + self.config.batch_size]
                    generated = model.generate_batch(batch)
                    for chunk_idx, gen_text in zip(batch_indices, generated):
                        recovered_texts[chunk_idx] = gen_text

                # Rebuild chunk lines preserving empty lines and newlines.
                for i, raw_line in enumerate(chunk):
                    if i in recovered_texts:
                        ending = "\n" if raw_line.endswith("\n") else ""
                        output_lines.append(recovered_texts[i] + ending)
                    else:
                        output_lines.append(raw_line)
            else:
                output_lines.extend(chunk)

        # Write output file.
        output_path = Path(self.config.output_path) / fpath.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(output_lines)

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
        """Compute SHA-256 checksum of a file."""
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _save_provenance(self, metadata: Dict[str, Any]) -> None:
        """Save provenance metadata as JSON."""
        output_path = Path(self.config.output_path) / "provenance.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logger.info(f"Provenance saved to {output_path}")
