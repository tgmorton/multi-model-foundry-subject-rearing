"""
Main pipeline runner for corpus descriptive analysis.

Single-pass processing: each spaCy Doc is dispatched to all active analyzers.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import spacy
from tqdm import tqdm

from preprocessing.utils import get_spacy_device, setup_logging

from .analyzers import ANALYZER_REGISTRY
from .analyzers.base import BaseAnalyzer
from .config import CorpusAnalysisConfig
from .line_cleaners import get_cleaner
from .output import save_results


class CorpusAnalysisPipeline:
    """
    Single-pass corpus descriptive analysis pipeline.

    Iterates .train files, applies line cleaners, runs spaCy, and dispatches
    each Doc to all active analyzers.
    """

    def __init__(self, config: CorpusAnalysisConfig):
        self.config = config
        self.logger = setup_logging(
            name="corpus_descriptives",
            experiment=config.split_name,
            phase="analysis",
        )

        # Initialize analyzers
        self.analyzers: Dict[str, BaseAnalyzer] = {}
        for name in config.analyses:
            if name not in ANALYZER_REGISTRY:
                self.logger.warning(f"Unknown analyzer: {name}, skipping")
                continue
            self.analyzers[name] = ANALYZER_REGISTRY[name]()
        self.logger.info(f"Active analyzers: {list(self.analyzers.keys())}")

        # Metadata (must be before _load_spacy which sets device)
        self._metadata: Dict[str, Any] = {
            "split_name": config.split_name,
            "spacy_model": config.spacy_model,
            "spacy_version": spacy.__version__,
            "language": config.language,
            "analyses": config.analyses,
        }
        self._file_line_counts: Dict[str, int] = {}

        # Load spaCy model
        self.nlp = self._load_spacy(config)

    def _load_spacy(self, config: CorpusAnalysisConfig) -> spacy.Language:
        """Load spaCy model with GPU auto-detection."""
        device = get_spacy_device(verbose=True)
        if device != "cpu":
            spacy.prefer_gpu()

        self.logger.info(f"Loading spaCy model: {config.spacy_model} (device={device})")
        nlp = spacy.load(config.spacy_model)
        nlp.max_length = 2_000_000

        # Disable unused components
        if config.spacy_disable_components:
            enabled = [name for name, _ in nlp.pipeline]
            to_disable = [c for c in config.spacy_disable_components if c in enabled]
            if to_disable:
                nlp.disable_pipes(*to_disable)
                self.logger.info(f"Disabled components: {to_disable}")

        self._metadata["device"] = device
        return nlp

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Run the full analysis pipeline.

        Returns:
            Dict mapping analyzer names to their results.
        """
        start = time.time()

        # Restore from checkpoint if available
        self._restore_checkpoint()

        # Find corpus files
        input_path = Path(self.config.input_path)
        train_files = sorted(input_path.glob("*.train"))
        if not train_files:
            raise FileNotFoundError(f"No .train files in {input_path}")
        self.logger.info(f"Found {len(train_files)} corpus files")

        total_lines = 0
        for fpath in tqdm(train_files, desc="Files"):
            genre = self._resolve_genre(fpath)
            if genre is None:
                self.logger.warning(f"Unknown genre for {fpath.name}, skipping")
                continue

            if fpath.name in self._file_line_counts:
                self.logger.info(f"Skipping already-processed file: {fpath.name}")
                total_lines += self._file_line_counts[fpath.name]
                continue

            cleaner = get_cleaner(genre)
            file_lines = self._process_file(fpath, genre, cleaner)
            self._file_line_counts[fpath.name] = file_lines
            total_lines += file_lines

        elapsed = time.time() - start
        self._metadata["processing_time_seconds"] = elapsed
        self._metadata["total_lines"] = total_lines
        self._metadata["file_line_counts"] = self._file_line_counts
        self.logger.info(
            f"Processed {total_lines:,} lines in {elapsed:.1f}s"
        )

        # Collect results
        results = {name: analyzer.get_results() for name, analyzer in self.analyzers.items()}

        # Save output
        save_results(
            results=results,
            metadata=self._metadata,
            output_dir=Path(self.config.output_path),
            split_name=self.config.split_name,
        )
        self.logger.info(f"Results saved to {self.config.output_path}")

        return results

    def _resolve_genre(self, fpath: Path) -> Optional[str]:
        """Resolve genre display name from filename stem."""
        stem = fpath.stem  # e.g. "childes" from "childes.train"
        return self.config.genre_map.get(stem)

    def _process_file(self, fpath: Path, genre: str, cleaner) -> int:
        """Process a single corpus file. Returns line count."""
        self.logger.info(f"Processing {fpath.name} (genre={genre})")

        with open(fpath, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        line_count = len(raw_lines)

        # Process in chunks
        num_chunks = (len(raw_lines) + self.config.chunk_size - 1) // self.config.chunk_size
        for chunk_start in tqdm(
            range(0, len(raw_lines), self.config.chunk_size),
            total=num_chunks,
            desc=fpath.name,
            leave=False,
        ):
            chunk = raw_lines[chunk_start : chunk_start + self.config.chunk_size]

            # Clean lines and collect metadata
            cleaned = []
            meta_list = []
            for raw_line in chunk:
                text, meta = cleaner(raw_line)
                if text:  # skip empty lines
                    cleaned.append(text)
                    meta_list.append(meta)

            if not cleaned:
                continue

            # Run spaCy
            docs = list(
                self.nlp.pipe(cleaned, batch_size=self.config.spacy_batch_size)
            )

            # Dispatch to analyzers
            for doc, meta in zip(docs, meta_list):
                speaker = meta.get("speaker")
                for analyzer in self.analyzers.values():
                    analyzer.process_doc(doc, genre, speaker=speaker)

        # Checkpoint after entire file is processed
        if self.config.checkpoint_dir:
            self._save_checkpoint()

        return line_count

    def _save_checkpoint(self) -> None:
        """Save analyzer states to checkpoint directory."""
        if not self.config.checkpoint_dir:
            return
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {}
        for name, analyzer in self.analyzers.items():
            state[name] = analyzer.to_checkpoint()
        state["_file_line_counts"] = self._file_line_counts

        ckpt_path = ckpt_dir / "checkpoint.json"
        ckpt_path.write_text(json.dumps(state, indent=2, default=str))
        self.logger.info(f"Checkpoint saved to {ckpt_path}")

    def _restore_checkpoint(self) -> None:
        """Restore analyzer states from checkpoint if available."""
        if not self.config.checkpoint_dir:
            return
        ckpt_path = Path(self.config.checkpoint_dir) / "checkpoint.json"
        if not ckpt_path.exists():
            return

        self.logger.info(f"Restoring from checkpoint: {ckpt_path}")
        state = json.loads(ckpt_path.read_text())

        for name, analyzer in self.analyzers.items():
            if name in state:
                analyzer.from_checkpoint(state[name])

        self._file_line_counts = state.get("_file_line_counts", {})
