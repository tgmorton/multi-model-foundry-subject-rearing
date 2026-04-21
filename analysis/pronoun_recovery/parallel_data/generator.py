"""
Europarl parallel alignment generator.

Orchestrates the full pipeline: read parallel files → parse with spaCy
→ align with awesome-align → extract pronouns → resolve labels →
write checkpoint JSONL.

Supports Italian (``language="it"``) and Spanish (``language="es"``)
as target languages. The underlying UD-based detector
(``detect_finite_verbs``) is shared across both — the label resolver
receives the language code and uses the right default-pronoun table
per language.

Note on config field names: the ``it_spacy_model`` and
``europarl_it_path`` fields have Italian-origin names but accept any
target-language values (e.g. ``es_core_news_lg`` and a Spanish parallel
corpus path). Future refactors may rename these to language-agnostic
identifiers; preserved here for backwards compatibility.

Follows the architecture pattern of synthetic_data/generator.py:
chunk processing, spaCy.pipe batching, tqdm progress, checkpointing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
from tqdm import tqdm

from analysis.pronoun_recovery.config import EuroparlAlignmentConfig
from preprocessing.utils import get_spacy_device, setup_logging

from .aligner import AwesomeAligner, build_alignment_dicts
from .en_pronoun_extractor import extract_subject_pronouns
from .it_null_subject_detector import detect_finite_verbs
from .label_resolver import ResolvedMarker, resolve_markers
from .quality_filters import FilterStats, check_pair_quality
from .validation import (
    compute_pilot_statistics,
    sample_for_spot_check,
    save_validation_report,
)

logger = logging.getLogger(__name__)


class EuroparlAlignmentGenerator:
    """Generate pronoun recovery training data from EN-IT Europarl pairs.

    Reads line-aligned English and Italian Europarl files, processes them
    in chunks through spaCy + awesome-align, and outputs checkpoint JSONL
    compatible with the existing training pipeline.

    Args:
        config: An EuroparlAlignmentConfig instance.
    """

    def __init__(self, config: EuroparlAlignmentConfig) -> None:
        self.config = config
        self.logger = setup_logging(
            name="europarl_align",
            experiment=config.language,
            phase="alignment",
        )
        self.stats = FilterStats()
        self.nlp_en, self.nlp_it = self._load_spacy()
        self.aligner = AwesomeAligner(
            model_name=config.align_model,
        )

    def _load_spacy(self) -> Tuple[spacy.Language, spacy.Language]:
        """Load EN and IT spaCy models with GPU auto-detection."""
        device = get_spacy_device(verbose=True)
        if device != "cpu":
            spacy.prefer_gpu()

        self.logger.info(
            "Loading spaCy models: EN=%s, IT=%s (device=%s)",
            self.config.en_spacy_model,
            self.config.it_spacy_model,
            device,
        )
        nlp_en = spacy.load(self.config.en_spacy_model)
        nlp_en.max_length = 2_000_000
        nlp_it = spacy.load(self.config.it_spacy_model)
        nlp_it.max_length = 2_000_000
        return nlp_en, nlp_it

    def run(self) -> None:
        """Execute the full alignment pipeline."""
        en_lines, it_lines = self._read_parallel_files()
        self.logger.info("Loaded %d parallel pairs", len(en_lines))

        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "aligned_checkpoint.jsonl"

        all_records: List[Dict[str, Any]] = []
        chunk_size = self.config.chunk_size

        num_chunks = (len(en_lines) + chunk_size - 1) // chunk_size

        for chunk_start in tqdm(
            range(0, len(en_lines), chunk_size),
            total=num_chunks,
            desc="Chunks",
        ):
            chunk_end = min(chunk_start + chunk_size, len(en_lines))
            en_chunk = en_lines[chunk_start:chunk_end]
            it_chunk = it_lines[chunk_start:chunk_end]

            records = self._process_chunk(en_chunk, it_chunk, chunk_start)
            all_records.extend(records)

            # Periodic checkpoint.
            if len(all_records) % self.config.checkpoint_interval < chunk_size:
                self._write_checkpoint(all_records, checkpoint_path)

        # Final checkpoint.
        self._write_checkpoint(all_records, checkpoint_path)

        self.logger.info(
            "Pipeline complete: %d records with markers from %d pairs",
            sum(1 for r in all_records if r.get("markers")),
            len(en_lines),
        )

        # Validation report.
        pilot_stats = compute_pilot_statistics(all_records, self.stats)
        spot_samples = sample_for_spot_check(
            all_records, n=self.config.spot_check_n
        )
        save_validation_report(pilot_stats, spot_samples, output_dir)

        # Optional passage packing.
        if self.config.pack_passages:
            from .passage_packer import pack_passages

            packed = pack_passages(
                it_lines,
                all_records,
                max_words=self.config.max_passage_words,
            )
            packed_path = output_dir / "packed_checkpoint.jsonl"
            self._write_checkpoint(packed, packed_path)
            self.logger.info(
                "Passage packing: %d sentences → %d passages",
                len(all_records), len(packed),
            )

        # Log summary.
        self.logger.info("Filter stats: %s", self.stats.summary())
        self.logger.info(
            "Label distribution: %s",
            pilot_stats.get("label_distribution", {}),
        )

    def _read_parallel_files(self) -> Tuple[List[str], List[str]]:
        """Read line-aligned EN and IT files, applying line range."""
        en_path = Path(self.config.europarl_en_path)
        it_path = Path(self.config.europarl_it_path)

        if not en_path.exists():
            raise FileNotFoundError(f"EN file not found: {en_path}")
        if not it_path.exists():
            raise FileNotFoundError(f"IT file not found: {it_path}")

        with open(en_path, "r", encoding="utf-8") as f:
            en_all = f.readlines()
        with open(it_path, "r", encoding="utf-8") as f:
            it_all = f.readlines()

        if len(en_all) != len(it_all):
            raise ValueError(
                f"Line count mismatch: EN={len(en_all)}, IT={len(it_all)}"
            )

        # Apply line range.
        start = self.config.start_line
        end = self.config.end_line if self.config.end_line > 0 else len(en_all)
        end = min(end, len(en_all))

        en_lines = [line.strip() for line in en_all[start:end]]
        it_lines = [line.strip() for line in it_all[start:end]]

        return en_lines, it_lines

    def _process_chunk(
        self,
        en_texts: List[str],
        it_texts: List[str],
        global_offset: int,
    ) -> List[Dict[str, Any]]:
        """Process a chunk of parallel pairs through the full pipeline.

        Steps:
        1. Parse EN and IT with spaCy (batched).
        2. Quality-filter pairs.
        3. Extract EN subject pronouns.
        4. Detect IT finite verbs.
        5. Run awesome-align on passing pairs.
        6. Resolve labels.
        7. Build output records.
        """
        batch_size = self.config.spacy_batch_size

        # Step 1: Parse both sides.
        en_docs = list(self.nlp_en.pipe(en_texts, batch_size=batch_size))
        it_docs = list(self.nlp_it.pipe(it_texts, batch_size=batch_size))

        # Steps 2-4: Filter and extract.
        passing_indices: List[int] = []
        en_pronouns_list: List[list] = []
        it_verbs_list: List[list] = []

        for i, (en_doc, it_doc) in enumerate(zip(en_docs, it_docs)):
            self.stats.total_pairs += 1

            if not check_pair_quality(
                en_doc,
                it_doc,
                min_tokens=self.config.min_tokens,
                max_tokens=self.config.max_tokens,
                max_length_ratio=self.config.max_length_ratio,
                stats=self.stats,
            ):
                continue

            en_pronouns = extract_subject_pronouns(
                en_doc, skip_all_it=self.config.skip_all_it
            )
            if not en_pronouns:
                self.stats.passed_pairs += 1
                continue

            it_verbs = detect_finite_verbs(it_doc)
            if not it_verbs:
                self.stats.passed_pairs += 1
                continue

            passing_indices.append(i)
            en_pronouns_list.append(en_pronouns)
            it_verbs_list.append(it_verbs)

        if not passing_indices:
            return []

        # Step 5: Word-align passing pairs.
        en_tokens_batch = [
            [tok.text for tok in en_docs[i]] for i in passing_indices
        ]
        it_tokens_batch = [
            [tok.text for tok in it_docs[i]] for i in passing_indices
        ]

        try:
            raw_alignments = self.aligner.align_batch(
                en_tokens_batch,
                it_tokens_batch,
                batch_size=self.config.align_batch_size,
            )
        except Exception as e:
            self.logger.error("Alignment failed for chunk at offset %d: %s", global_offset, e)
            return []

        # Step 6-7: Resolve and build records.
        records: List[Dict[str, Any]] = []

        for j, idx in enumerate(passing_indices):
            if j >= len(raw_alignments):
                break

            en_to_it, _ = build_alignment_dicts(raw_alignments[j])

            markers = resolve_markers(
                en_docs[idx],
                it_docs[idx],
                en_pronouns_list[j],
                it_verbs_list[j],
                en_to_it,
                stats=self.stats,
                language=self.config.language,
            )

            self.stats.passed_pairs += 1

            if not markers:
                continue

            record = self._build_record(
                it_docs[idx],
                markers,
                pair_idx=global_offset + idx,
            )
            records.append(record)

        return records

    def _build_record(
        self,
        it_doc: spacy.tokens.Doc,
        markers: List[ResolvedMarker],
        pair_idx: int,
    ) -> Dict[str, Any]:
        """Build a checkpoint record from resolved markers.

        Output format matches what _checkpoint_record_to_word_labels
        expects in model/dataset.py.
        """
        return {
            "clean_text": it_doc.text,
            "markers": [
                {
                    "label": m.label,
                    "lexical_form": m.lexical_form,
                    "position": m.position,
                    "confidence": m.confidence,
                    "en_pronoun": m.en_pronoun,
                    "it_verb": m.it_verb_text,
                }
                for m in markers
            ],
            "id": f"europarl_aligned:{pair_idx}",
            "genre": "Europarl",
            "source": "parallel_alignment",
        }

    def _write_checkpoint(
        self,
        records: List[Dict[str, Any]],
        path: Path,
    ) -> None:
        """Write all records to a checkpoint JSONL file (overwrites)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.logger.info("Checkpoint: %d records written to %s", len(records), path)
