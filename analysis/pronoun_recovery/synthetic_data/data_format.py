"""
Data format and JSONL I/O for synthetic pronoun-recovery pairs.

Each pair stores an original sentence, its pronoun-dropped variant,
per-verb labels indicating which (if any) subject pronoun was dropped,
and provenance metadata (genre, source file, line number).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union


@dataclass
class SyntheticPair:
    """A single synthetic training example for pronoun recovery.

    Attributes:
        original: The original text with subject pronouns intact.
        dropped: The text after subject pronoun removal.
        verb_labels: Per-verb annotation list.  Each entry is a dict with
            ``verb_idx`` (token index in the dropped text),
            ``verb_text`` (surface form),
            ``label`` (e.g. ``"PRO.3sg"`` or ``"NONE"``), and
            ``original_pronoun`` (the dropped pronoun text, or ``None``).
        genre: Genre display name (e.g. ``"CHILDES"``, ``"BNC"``).
        source_file: Filename of the corpus ``.train`` file.
        line_number: 0-based line index within *source_file*.
    """

    original: str
    dropped: str
    verb_labels: List[Dict[str, Any]] = field(default_factory=list)
    genre: str = ""
    source_file: str = ""
    line_number: int = 0

    # ── Serialisation helpers ────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialisation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticPair":
        """Construct a :class:`SyntheticPair` from a plain dict."""
        return cls(
            original=data["original"],
            dropped=data["dropped"],
            verb_labels=data.get("verb_labels", []),
            genre=data.get("genre", ""),
            source_file=data.get("source_file", ""),
            line_number=data.get("line_number", 0),
        )


# ── JSONL I/O ────────────────────────────────────────────────────────────


def save_pairs(
    pairs: Sequence[SyntheticPair],
    path: Union[str, Path],
) -> None:
    """Write a sequence of :class:`SyntheticPair` objects to a JSONL file.

    Each line is a self-contained JSON object.  The parent directory is
    created automatically if it does not exist.

    Args:
        pairs: Pairs to serialise.
        path: Destination file path (typically ``*.jsonl``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")


def load_pairs(path: Union[str, Path]) -> List[SyntheticPair]:
    """Load :class:`SyntheticPair` objects from a JSONL file.

    Args:
        path: Source file path (typically ``*.jsonl``).

    Returns:
        List of deserialised pairs.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    pairs: List[SyntheticPair] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pairs.append(SyntheticPair.from_dict(json.loads(line)))
    return pairs
