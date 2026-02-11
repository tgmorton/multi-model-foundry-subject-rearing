"""Multi-sentence passage packer for pronoun recovery.

Groups consecutive Europarl sentences into multi-sentence passages
up to a word budget, giving the model discourse context beyond
a single sentence.  Applied as post-processing after the per-sentence
alignment pipeline.

The packer reads the original Italian text file (all sentences in the
range) and the per-sentence checkpoint records (marker-bearing
sentences only), then groups consecutive lines into passages that
include non-marker sentences as context.  Only passages containing
at least one marker are emitted.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _extract_line_index(record_id: str) -> Optional[int]:
    """Extract the line index from a record ID like 'europarl_aligned:35915'."""
    prefix = "europarl_aligned:"
    if record_id.startswith(prefix):
        try:
            return int(record_id[len(prefix):])
        except ValueError:
            return None
    return None


def pack_passages(
    it_lines: List[str],
    records: List[Dict[str, Any]],
    max_words: int = 180,
    record_id_prefix: str = "europarl_passage",
) -> List[Dict[str, Any]]:
    """Pack per-sentence records into multi-sentence passages.

    Groups consecutive Italian sentences into passages up to *max_words*,
    including sentences without markers for discourse context.  Only
    emits passages containing at least one marker.

    Args:
        it_lines: All Italian text lines for the relevant range
            (already stripped, matching the index space of the records).
        records: Per-sentence checkpoint records from alignment pipeline.
        max_words: Maximum word count per passage.  At ~1.3x subword
            expansion this should stay under ``max_seq_length`` tokens.
            Default 180 ≈ 234 subword tokens, under 256.
        record_id_prefix: Prefix for passage record IDs.

    Returns:
        List of passage-level checkpoint records.
    """
    # Build index: line_idx -> list of markers.
    markers_by_line: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        line_idx = _extract_line_index(rec.get("id", ""))
        if line_idx is not None:
            markers_by_line[line_idx] = list(rec.get("markers", []))

    passages: List[Dict[str, Any]] = []

    # Accumulator for current passage.
    buf_lines: List[Tuple[int, str]] = []     # (line_idx, text)
    buf_markers: List[Dict[str, Any]] = []
    buf_word_count = 0

    def _flush() -> None:
        """Emit the current buffer as a passage if it has markers."""
        nonlocal buf_lines, buf_markers, buf_word_count
        if buf_lines and buf_markers:
            passages.append(_build_passage(
                buf_lines, buf_markers,
                len(passages), record_id_prefix,
            ))
        buf_lines = []
        buf_markers = []
        buf_word_count = 0

    for line_idx, raw_line in enumerate(it_lines):
        text = raw_line.strip()

        # Empty line = debate / speaker boundary → flush.
        if not text:
            _flush()
            continue

        line_words = len(text.split())

        # Would this line exceed the budget?
        if buf_word_count + line_words > max_words and buf_lines:
            _flush()

        # Character offset of this line within the concatenated passage.
        char_offset = sum(len(t) + 1 for _, t in buf_lines)  # +1 for space
        if not buf_lines:
            char_offset = 0

        buf_lines.append((line_idx, text))
        buf_word_count += line_words

        # Adjust marker positions if this line has markers.
        if line_idx in markers_by_line:
            for marker in markers_by_line[line_idx]:
                adj = dict(marker)
                adj["position"] = marker["position"] + char_offset
                buf_markers.append(adj)

    # Flush remaining.
    _flush()

    logger.info(
        "Packed %d sentences into %d passages (%d marker-bearing lines, "
        "avg %.1f sentences/passage)",
        len(it_lines),
        len(passages),
        len(markers_by_line),
        sum(p["num_sentences"] for p in passages) / max(len(passages), 1),
    )
    return passages


def _build_passage(
    lines: List[Tuple[int, str]],
    markers: List[Dict[str, Any]],
    passage_idx: int,
    id_prefix: str,
) -> Dict[str, Any]:
    """Build a passage record from accumulated lines and markers."""
    clean_text = " ".join(text for _, text in lines)
    line_indices = [idx for idx, _ in lines]

    return {
        "clean_text": clean_text,
        "markers": markers,
        "id": f"{id_prefix}:{passage_idx}",
        "genre": "Europarl",
        "source": "parallel_alignment",
        "line_range": [line_indices[0], line_indices[-1]],
        "num_sentences": len(lines),
    }


def pack_checkpoint_file(
    checkpoint_path: Path,
    it_file_path: Path,
    output_path: Path,
    start_line: int = 0,
    end_line: int = 0,
    max_words: int = 180,
) -> List[Dict[str, Any]]:
    """Pack an existing per-sentence checkpoint into multi-sentence passages.

    Convenience wrapper that reads files, calls :func:`pack_passages`,
    and writes the packed checkpoint.

    Args:
        checkpoint_path: Per-sentence checkpoint JSONL from alignment pipeline.
        it_file_path: Original Italian text file.
        output_path: Where to write the packed checkpoint JSONL.
        start_line: First line used in the alignment run (0-based).
        end_line: Last line (exclusive); 0 = all lines.
        max_words: Word budget per passage.

    Returns:
        The packed passage records.
    """
    # Read per-sentence records.
    records: List[Dict[str, Any]] = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Read Italian lines for the range.
    with open(it_file_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    end = end_line if end_line > 0 else len(all_lines)
    end = min(end, len(all_lines))
    it_lines = [ln.strip() for ln in all_lines[start_line:end]]

    passages = pack_passages(it_lines, records, max_words=max_words)

    # Write packed checkpoint.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in passages:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Packed checkpoint: %d passages written to %s",
        len(passages), output_path,
    )
    return passages
