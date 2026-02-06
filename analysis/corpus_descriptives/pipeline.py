"""
Main pipeline runner for corpus descriptive analysis.

Supports two modes:
1. Analyzer mode (original): aggregates counts via BaseAnalyzer classes
2. Annotation mode (new): produces per-sentence Parquet files via BaseSentenceAnnotator

Single-pass processing: each spaCy Doc is dispatched to all active analyzers/annotators.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import spacy
from tqdm import tqdm

from preprocessing.utils import get_spacy_device, setup_logging

from .analyzers import ANALYZER_REGISTRY
from .analyzers.base import BaseAnalyzer
from .config import CorpusAnalysisConfig
from .line_cleaners import get_cleaner
from .output import save_results, AnnotationWriter, LayeredAnnotationWriter

if TYPE_CHECKING:
    from .annotators.base import BaseSentenceAnnotator


# Mapping from annotator class names to schema layer names
ANNOTATOR_TO_LAYER = {
    "ClauseStructureAnnotator": "clause_structure",
    "ThatTraceAnnotator": "that_trace",
    "ExpletiveAnnotator": "expletives",
    "PronounAnnotator": "pronouns",
    "NegationAnnotator": "negation",
    "WhExtractionAnnotator": "wh_extraction",
    "RelativeClauseAnnotator": "relative_clauses",
    "VerbAnnotator": "verbs",
    "ArgumentStructureAnnotator": "argument_structure",
    "TopicAnnotator": "topic_continuity",
    "ComplexityAnnotator": "complexity",
    "FragmentAnnotator": "complexity",  # shares layer with complexity
}


def generate_sentence_id(
    split: str,
    genre: str,
    file_idx: int,
    sent_idx: int,
) -> str:
    """
    Generate a unique, stable sentence ID.

    Format: {split}_{genre}_{file_idx:06d}_{sent_idx:06d}

    Args:
        split: Corpus split (e.g., "train_90M", "test_10M")
        genre: Genre name (e.g., "CHILDES", "BNC")
        file_idx: Index of source file within genre (0-based)
        sent_idx: Index of sentence within file (0-based)

    Returns:
        Unique sentence identifier string
    """
    # Normalize genre for ID (replace spaces, lowercase)
    genre_normalized = genre.replace(" ", "_").replace("-", "_").lower()
    return f"{split}_{genre_normalized}_{file_idx:06d}_{sent_idx:06d}"


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
        nlp.max_length = 3_000_000

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
                effective_genre = genre
                if genre == "CHILDES":
                    role = meta.get("role")
                    if role == "child":
                        effective_genre = "CHILDES_child"
                    elif role == "adult":
                        effective_genre = "CHILDES_adult"
                for analyzer in self.analyzers.values():
                    analyzer.process_doc(doc, effective_genre, speaker=speaker)

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


class CorpusAnnotationPipeline:
    """
    Sentence-level annotation pipeline producing Parquet output.

    Unlike CorpusAnalysisPipeline which aggregates counts, this pipeline
    preserves per-sentence annotations for downstream queries and
    influence function analysis.
    """

    def __init__(
        self,
        config: CorpusAnalysisConfig,
        annotators: List["BaseSentenceAnnotator"],
        output_dir: Optional[Path] = None,
        layered: bool = True,
    ):
        """
        Initialize annotation pipeline.

        Args:
            config: Corpus analysis configuration
            annotators: List of sentence annotators to run
            output_dir: Override output directory (default: config.output_path)
            layered: If True, write separate Parquet files per layer
        """
        self.config = config
        self.annotators = annotators
        self.layered = layered
        self.output_dir = Path(output_dir or config.output_path) / "annotated_corpus"

        self.logger = setup_logging(
            name="corpus_annotation",
            experiment=config.split_name,
            phase="annotation",
        )

        # Metadata
        self._metadata: Dict[str, Any] = {
            "split_name": config.split_name,
            "spacy_model": config.spacy_model,
            "spacy_version": spacy.__version__,
            "language": config.language,
            "annotators": [ann.name for ann in annotators],
            "layered": layered,
        }
        self._file_idx_map: Dict[str, int] = {}  # genre -> current file index

        # Load spaCy
        self.nlp = self._load_spacy(config)

    def _load_spacy(self, config: CorpusAnalysisConfig) -> spacy.Language:
        """Load spaCy model with GPU auto-detection."""
        device = get_spacy_device(verbose=True)
        if device != "cpu":
            spacy.prefer_gpu()

        self.logger.info(f"Loading spaCy model: {config.spacy_model} (device={device})")
        nlp = spacy.load(config.spacy_model)
        nlp.max_length = 3_000_000

        if config.spacy_disable_components:
            enabled = [name for name, _ in nlp.pipeline]
            to_disable = [c for c in config.spacy_disable_components if c in enabled]
            if to_disable:
                nlp.disable_pipes(*to_disable)
                self.logger.info(f"Disabled components: {to_disable}")

        self._metadata["device"] = device
        return nlp

    def run(self) -> Dict[str, Any]:
        """
        Run the annotation pipeline.

        Returns:
            Metadata dict with processing statistics
        """
        from .schema import get_full_schema, get_base_schema, get_layer_schema

        start = time.time()

        # Find corpus files
        input_path = Path(self.config.input_path)
        train_files = sorted(input_path.glob("*.train"))
        if not train_files:
            raise FileNotFoundError(f"No .train files in {input_path}")
        self.logger.info(f"Found {len(train_files)} corpus files")

        # Initialize writer
        if self.layered:
            writer = LayeredAnnotationWriter(
                output_dir=self.output_dir,
                split_name=self.config.split_name,
                batch_size=10000,
            )
        else:
            writer = AnnotationWriter(
                output_path=self.output_dir / f"{self.config.split_name}.parquet",
                schema=get_full_schema(),
                batch_size=10000,
            )

        total_sentences = 0
        file_stats: Dict[str, int] = {}

        with writer:
            for fpath in tqdm(train_files, desc="Files"):
                genre = self._resolve_genre(fpath)
                if genre is None:
                    self.logger.warning(f"Unknown genre for {fpath.name}, skipping")
                    continue

                # Track file index per genre
                if genre not in self._file_idx_map:
                    self._file_idx_map[genre] = 0
                file_idx = self._file_idx_map[genre]
                self._file_idx_map[genre] += 1

                sent_count = self._process_file(
                    fpath, genre, file_idx, writer,
                    get_base_schema if self.layered else None,
                    get_layer_schema if self.layered else None,
                )
                file_stats[fpath.name] = sent_count
                total_sentences += sent_count

        elapsed = time.time() - start
        self._metadata["processing_time_seconds"] = elapsed
        self._metadata["total_sentences"] = total_sentences
        self._metadata["file_sentence_counts"] = file_stats
        self.logger.info(f"Annotated {total_sentences:,} sentences in {elapsed:.1f}s")

        # Write metadata
        if self.layered:
            writer.write_metadata(self._metadata)
        else:
            meta_path = self.output_dir / "metadata.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(self._metadata, indent=2, default=str))

        return self._metadata

    def _resolve_genre(self, fpath: Path) -> Optional[str]:
        """Resolve genre display name from filename stem."""
        stem = fpath.stem
        return self.config.genre_map.get(stem)

    def _process_file(
        self,
        fpath: Path,
        genre: str,
        file_idx: int,
        writer,
        get_base_schema=None,
        get_layer_schema=None,
    ) -> int:
        """Process a single corpus file. Returns sentence count."""
        self.logger.info(f"Processing {fpath.name} (genre={genre}, file_idx={file_idx})")

        cleaner = get_cleaner(genre)

        with open(fpath, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        sent_count = 0
        global_sent_idx = 0  # Track sentence index across all chunks

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
                if text:
                    cleaned.append(text)
                    meta_list.append(meta)

            if not cleaned:
                continue

            # Run spaCy
            docs = list(
                self.nlp.pipe(cleaned, batch_size=self.config.spacy_batch_size)
            )

            # Process each doc (which may have multiple sentences)
            for doc, meta in zip(docs, meta_list):
                speaker = meta.get("speaker")
                role = meta.get("role")

                # Determine effective genre (CHILDES split)
                effective_genre = genre
                if genre == "CHILDES":
                    if role == "child":
                        effective_genre = "CHILDES_child"
                    elif role == "adult":
                        effective_genre = "CHILDES_adult"

                # Process each sentence in the doc
                for sent in doc.sents:
                    sentence_id = generate_sentence_id(
                        split=self.config.split_name,
                        genre=effective_genre,
                        file_idx=file_idx,
                        sent_idx=global_sent_idx,
                    )

                    # Build base annotation
                    base_annotation = {
                        "sentence_id": sentence_id,
                        "split": self.config.split_name,
                        "genre": effective_genre,
                        "speaker": speaker,
                        "role": role,
                        "file_name": fpath.stem,
                        "sent_idx": global_sent_idx,
                        "text": sent.text,
                        "n_tokens": len(sent),
                        # Token-level arrays
                        "tokens": [tok.text for tok in sent],
                        "lemmas": [tok.lemma_ for tok in sent],
                        "pos_tags": [tok.pos_ for tok in sent],
                        "dep_rels": [tok.dep_ for tok in sent],
                        "dep_heads": [tok.head.i - sent.start for tok in sent],
                    }

                    # Run annotators once and cache results
                    annotator_results = {}
                    all_annotations = dict(base_annotation)
                    all_flags = {}

                    for annotator in self.annotators:
                        ann_result = annotator.annotate_sentence(
                            sent=sent,
                            genre=effective_genre,
                            speaker=speaker,
                            metadata={"role": role, "file_name": fpath.stem},
                        )
                        flags = annotator.get_sentence_flags(ann_result)
                        annotator_results[annotator.name] = (ann_result, flags)

                        all_annotations.update(ann_result)
                        all_flags.update(flags)

                    all_annotations.update(all_flags)

                    # Write to appropriate format
                    if self.layered and isinstance(writer, LayeredAnnotationWriter):
                        # Write base
                        writer.write_base(base_annotation, get_base_schema())

                        # Write each layer using cached results
                        for annotator in self.annotators:
                            layer_name = ANNOTATOR_TO_LAYER.get(
                                annotator.name,
                                annotator.name.replace("Annotator", "").lower()
                            )
                            ann_result, flags = annotator_results[annotator.name]
                            layer_data = {"sentence_id": sentence_id}
                            layer_data.update(ann_result)
                            layer_data.update(flags)

                            try:
                                layer_schema = get_layer_schema(layer_name)
                                writer.write_layer(layer_name, layer_data, layer_schema)
                            except ValueError:
                                pass
                    else:
                        # Write full annotation
                        writer.write_annotation(all_annotations)

                    global_sent_idx += 1
                    sent_count += 1

        return sent_count
