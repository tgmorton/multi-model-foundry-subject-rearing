"""
Synthetic paired-data generator for the pronoun recovery sequence labeler.

Mirrors the ablation in ``preprocessing.ablations.remove_subject_pronominals``
but additionally records *which* pronouns were dropped and produces per-verb
labels suitable for training a token-level classifier.

Typical usage::

    from analysis.pronoun_recovery.config import SyntheticDataConfig, load_config
    from analysis.pronoun_recovery.synthetic_data import SyntheticPairGenerator

    config = load_config("configs/synthetic_data.yaml", SyntheticDataConfig)
    generator = SyntheticPairGenerator(config)
    generator.generate_from_corpus()
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import spacy
from tqdm import tqdm

from analysis.pronoun_recovery.config import SyntheticDataConfig
from preprocessing.utils import get_spacy_device, setup_logging

from .alignment import build_alignment_map, compute_verb_labels
from .data_format import SyntheticPair, save_pairs

logger = logging.getLogger(__name__)


class SyntheticPairGenerator:
    """Generate synthetic (original, dropped, labels) pairs from a corpus.

    The generator walks every ``.train`` file under
    :pyattr:`config.input_path`, removes subject pronouns exactly as
    ``remove_subject_pronominals`` does, and records per-verb labels that
    indicate *which* pronoun (if any) was dropped before each finite verb.

    After collecting all pairs it performs a stratified 90/10 train/dev
    split and writes the results as JSONL to :pyattr:`config.output_path`.

    Args:
        config: A :class:`SyntheticDataConfig` instance specifying paths,
            spaCy model, filtering parameters, and split ratio.
    """

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self.logger = setup_logging(
            name="synthetic_data",
            experiment=config.language,
            phase="generation",
        )
        self.nlp = self._load_spacy()

    # ── spaCy setup ──────────────────────────────────────────────────────

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

    # ── Public API ───────────────────────────────────────────────────────

    def generate_from_corpus(self) -> None:
        """Process all ``.train`` files, generate pairs, split, and save."""
        input_path = Path(self.config.input_path)
        train_files = sorted(input_path.glob("*.train"))
        if not train_files:
            raise FileNotFoundError(f"No .train files in {input_path}")

        self.logger.info(f"Found {len(train_files)} corpus files in {input_path}")

        all_pairs: List[SyntheticPair] = []
        for fpath in tqdm(train_files, desc="Files"):
            genre = self._resolve_genre(fpath)
            if genre is None:
                self.logger.warning(f"Unknown genre for {fpath.name}, skipping")
                continue
            file_pairs = self._process_file(fpath, genre)
            all_pairs.extend(file_pairs)

        self.logger.info(f"Generated {len(all_pairs):,} pairs in total")

        if not all_pairs:
            self.logger.warning("No pairs generated -- nothing to save.")
            return

        self._split_and_save(all_pairs)

    # ── File-level processing ────────────────────────────────────────────

    def _resolve_genre(self, fpath: Path) -> Optional[str]:
        """Resolve genre display name from filename stem via config map."""
        return self.config.genre_map.get(fpath.stem)

    def _process_file(
        self,
        fpath: Path,
        genre: str,
    ) -> List[SyntheticPair]:
        """Process a single corpus file, returning generated pairs."""
        self.logger.info(f"Processing {fpath.name} (genre={genre})")

        with open(fpath, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        pairs: List[SyntheticPair] = []
        num_chunks = (len(raw_lines) + self.config.chunk_size - 1) // self.config.chunk_size

        for chunk_start in tqdm(
            range(0, len(raw_lines), self.config.chunk_size),
            total=num_chunks,
            desc=fpath.name,
            leave=False,
        ):
            chunk_lines = raw_lines[chunk_start : chunk_start + self.config.chunk_size]
            chunk_indices = range(chunk_start, chunk_start + len(chunk_lines))

            # Strip and filter empty lines, keeping their original indices.
            texts: List[str] = []
            line_numbers: List[int] = []
            for line_num, raw_line in zip(chunk_indices, chunk_lines):
                text = raw_line.strip()
                if text:
                    texts.append(text)
                    line_numbers.append(line_num)

            if not texts:
                continue

            docs = self.nlp.pipe(texts, batch_size=self.config.spacy_batch_size)
            for doc, line_num in zip(docs, line_numbers):
                pair = self.process_line(
                    doc,
                    genre=genre,
                    source_file=fpath.name,
                    line_number=line_num,
                )
                if pair is not None:
                    pairs.append(pair)

        self.logger.info(f"  {fpath.name}: {len(pairs):,} pairs from {len(raw_lines):,} lines")
        return pairs

    # ── Line-level processing ────────────────────────────────────────────

    def process_line(
        self,
        doc: spacy.tokens.Doc,
        genre: str,
        source_file: str,
        line_number: int,
    ) -> Optional[SyntheticPair]:
        """Process a single spaCy Doc and optionally return a pair.

        Steps:

        1. Identify all tokens where ``pos_ == "PRON"`` and
           ``dep_ == "nsubj"`` (subject pronouns).
        2. Record their indices as the *dropped* set.
        3. Reconstruct the dropped text using ``text_with_ws`` for
           retained tokens.
        4. Compute per-verb labels via :mod:`.alignment`.
        5. Apply filtering:
           - Skip if no subject pronouns were found.
           - Skip if the original doc has fewer than
             ``config.min_tokens`` or more than ``config.max_tokens``
             tokens.
           - Skip if ``config.require_finite_verb`` is set and no finite
             verb is present in the label list.
        6. Return a :class:`SyntheticPair` or ``None``.

        Args:
            doc: spaCy Doc of the *original* (undropped) text.
            genre: Genre display name.
            source_file: Corpus filename.
            line_number: 0-based line index within the file.

        Returns:
            A :class:`SyntheticPair` if the line passes all filters, or
            ``None`` otherwise.
        """
        # ---- 1. Find subject pronouns ----
        dropped_indices: Set[int] = set()
        for token in doc:
            if token.pos_ == "PRON" and token.dep_ == "nsubj":
                dropped_indices.add(token.i)

        # Filter: must have at least one subject pronoun to drop.
        if not dropped_indices:
            return None

        # Filter: token-length bounds (applied to the *original* doc).
        num_tokens = len(doc)
        if num_tokens < self.config.min_tokens or num_tokens > self.config.max_tokens:
            return None

        # ---- 2-3. Build dropped text ----
        dropped_parts: List[str] = []
        for token in doc:
            if token.i not in dropped_indices:
                dropped_parts.append(token.text_with_ws)
        dropped_text = "".join(dropped_parts)

        # ---- 4. Alignment and labels ----
        alignment_map = build_alignment_map(doc, dropped_indices)
        verb_labels = compute_verb_labels(doc, dropped_indices, alignment_map)

        # Filter: require at least one finite verb.
        if self.config.require_finite_verb and not verb_labels:
            return None

        return SyntheticPair(
            original=doc.text,
            dropped=dropped_text,
            verb_labels=verb_labels,
            genre=genre,
            source_file=source_file,
            line_number=line_number,
        )

    # ── Splitting and saving ─────────────────────────────────────────────

    def _split_and_save(self, pairs: List[SyntheticPair]) -> None:
        """Perform a stratified train/dev split and write JSONL files.

        The split is stratified on *genre* so that genre proportions are
        preserved in both partitions.  If ``config.max_pairs > 0`` and the
        total number of training pairs exceeds that cap, the training set
        is truncated (after splitting) to ``max_pairs``.

        Args:
            pairs: All generated pairs across every corpus file.
        """
        from sklearn.model_selection import train_test_split

        genres = [p.genre for p in pairs]

        train_pairs, dev_pairs = train_test_split(
            pairs,
            test_size=self.config.dev_fraction,
            random_state=42,
            stratify=genres,
        )

        # Cap training set if configured.
        if self.config.max_pairs > 0 and len(train_pairs) > self.config.max_pairs:
            self.logger.info(
                f"Capping training pairs from {len(train_pairs):,} to "
                f"{self.config.max_pairs:,}"
            )
            train_pairs = train_pairs[: self.config.max_pairs]

        output_dir = Path(self.config.output_path)
        train_path = output_dir / "train.jsonl"
        dev_path = output_dir / "dev.jsonl"

        save_pairs(train_pairs, train_path)
        save_pairs(dev_pairs, dev_path)

        self.logger.info(
            f"Saved {len(train_pairs):,} train pairs to {train_path}"
        )
        self.logger.info(
            f"Saved {len(dev_pairs):,} dev pairs to {dev_path}"
        )

        # Log genre distribution for both splits.
        self._log_genre_distribution("train", train_pairs)
        self._log_genre_distribution("dev", dev_pairs)

    def _log_genre_distribution(
        self,
        split_name: str,
        pairs: List[SyntheticPair],
    ) -> None:
        """Log per-genre counts for a split."""
        counts: Dict[str, int] = {}
        for p in pairs:
            counts[p.genre] = counts.get(p.genre, 0) + 1
        breakdown = ", ".join(f"{g}: {c:,}" for g, c in sorted(counts.items()))
        self.logger.info(f"  {split_name} genre distribution: {breakdown}")
