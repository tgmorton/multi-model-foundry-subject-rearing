"""D6: Pre-tokenize stimuli once per tokenizer family.

Turns v2 stimulus CSV(s) into a cached on-disk tensor artifact keyed on
`(stimuli_content_hash, tokenizer_content_hash)`. Subsequent evaluation
jobs load the cached tensors directly instead of re-tokenizing per run.

Resolves `hotspot_position` (word-index into target.split()) into
`hotspot_token_index` (subword index into full_token_ids) *once* — all
downstream code uses the resolved index.

Cache file is a pickle with the following payload:
    {
        "format_version": "stimuli_cache.v1",
        "stimuli_id": str,
        "tokenizer_id": str,
        "rows": List[Dict],       # one dict per CSV row
        "meta": Dict,              # summary (n_rows, languages, categories, ...)
    }

Each row dict contains the original v2 columns plus:
    context_token_ids: List[int]
    target_token_ids: List[int]
    full_token_ids:   List[int]
    context_len:      int          # len(context_token_ids) with BOS if used
    full_len:         int
    target_subword_count: int
    hotspot_token_index:  int      # subword-level index into full_token_ids
    bos_id, eos_id:   Optional[int]
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# --- Hashing helpers --------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tokenizer(tokenizer_path: str | Path) -> str:
    """Content hash of a tokenizer file (first 12 hex chars)."""
    return _sha256_file(Path(tokenizer_path))[:12]


def hash_stimuli_files(csv_paths: List[Path]) -> str:
    """Content hash of a sorted set of CSV files + schema version marker."""
    h = hashlib.sha256()
    h.update(b"stimuli_cache.v1\n")
    for p in sorted(csv_paths, key=lambda x: x.name):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        with open(p, "rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()[:12]


# --- Hotspot-position resolution --------------------------------------------

def _resolve_hotspot_subword_index(
    target: str,
    hotspot_word_index: int,
    hotspot_token: str,
    target_token_ids: List[int],
    target_token_strings: List[str],
) -> int:
    """Map a word-level position into a subword-level position.

    Handles two common tokenizer conventions:
    1. Word-level tokenizers (each whitespace-word is exactly one token):
       the mapping is identity: subword_index == word_index.
    2. SentencePiece / BPE with word-boundary markers (e.g., '▁', U+2581):
       count subword tokens that start a new word and return the subword
       that begins the hotspot word.

    Detection of convention: if any subword string starts with '▁', we're
    in SP-style mode. If every subword exactly reconstructs a
    whitespace-separated word, we're in word-level mode. Otherwise, we
    fall back to a character-offset walk.

    Returns an index into `target_token_ids` (0-based).
    """
    words = target.split()
    if not (0 <= hotspot_word_index < len(words)):
        raise ValueError(
            f"hotspot_word_index {hotspot_word_index} out of range for "
            f"target with {len(words)} words"
        )
    n_sub = len(target_token_strings)

    # Case 1: 1-to-1 word-level tokenization.
    if n_sub == len(words) and all(
        s.strip() == w for s, w in zip(target_token_strings, words)
    ):
        return hotspot_word_index

    # Case 2: SentencePiece-style with '▁' markers for word starts.
    WORD_MARKER = "\u2581"
    if any(s.startswith(WORD_MARKER) for s in target_token_strings):
        word_starts = [i for i, s in enumerate(target_token_strings)
                       if s.startswith(WORD_MARKER)]
        if len(word_starts) >= len(words):
            return word_starts[hotspot_word_index]
        # Fewer markers than words is unusual (e.g. leading whitespace
        # behavior); fall through to character-offset walk.

    # Case 3: character-offset walk (robust but slower).
    # target is Moses-tokenized: words joined by single spaces.
    char_offset = sum(len(w) + 1 for w in words[:hotspot_word_index])
    # Build per-subword char length (strip leading ▁ if present).
    lengths = [
        len(s.lstrip(WORD_MARKER)) for s in target_token_strings
    ]
    cumulative = 0
    for i, l in enumerate(lengths):
        # This subword spans [cumulative, cumulative + l). The hotspot word
        # starts at char_offset. Return this subword if it overlaps the
        # start of the hotspot word.
        if cumulative <= char_offset < cumulative + max(l, 1):
            return i
        cumulative += l
        # Account for the single-space separator between Moses-tokens,
        # but only if the next subword clearly begins a new word (or we
        # can't tell — assume yes to advance).
        if i + 1 < n_sub:
            cumulative += 1
    return n_sub - 1


# --- Cache dataclass --------------------------------------------------------

@dataclass
class StimulusCacheEntry:
    """One tokenized stimulus row (serializable)."""
    # Original v2 fields (pass-through)
    item_id: int
    category: str
    condition: str
    pronoun_status: int
    language: str
    context: str
    target: str
    hotspot_token: str
    hotspot_position: int   # word-level (from CSV)
    names: str
    generator: str
    # Source (for debugging)
    source_file: str
    # Tokenized fields (resolved by this module)
    context_token_ids: List[int]
    target_token_ids: List[int]
    full_token_ids: List[int]
    context_len: int
    full_len: int
    target_subword_count: int
    hotspot_token_index: int   # subword-level into full_token_ids
    bos_id: Optional[int]
    eos_id: Optional[int]


@dataclass
class StimulusCache:
    """On-disk-serializable collection of tokenized stimuli."""
    stimuli_id: str
    tokenizer_id: str
    rows: List[StimulusCacheEntry] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    # --- Persistence --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": "stimuli_cache.v1",
            "stimuli_id": self.stimuli_id,
            "tokenizer_id": self.tokenizer_id,
            "rows": [e.__dict__ for e in self.rows],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StimulusCache":
        if data.get("format_version") != "stimuli_cache.v1":
            raise ValueError(
                f"Unknown stimuli_cache format_version: {data.get('format_version')!r}"
            )
        rows = [StimulusCacheEntry(**r) for r in data["rows"]]
        return cls(
            stimuli_id=data["stimuli_id"],
            tokenizer_id=data["tokenizer_id"],
            rows=rows,
            meta=data.get("meta", {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # pid-unique tmp: concurrent builders (parallel eval pods on a shared
        # filesystem) must not interleave writes into one tmp inode.
        tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
        with open(tmp, "wb") as fh:
            pickle.dump(self.to_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        logger.info("Wrote stimuli cache to %s (n=%d)", path, len(self.rows))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StimulusCache":
        with open(path, "rb") as fh:
            return cls.from_dict(pickle.load(fh))


# --- Tokenization entry point -----------------------------------------------

def _encode_with_specials(
    tokenizer,
    text: str,
) -> Tuple[List[int], Optional[int], Optional[int]]:
    """Encode `text` and return (ids_with_bos, bos_id, eos_id).

    BOS prepended if the tokenizer exposes a valid bos_id. EOS is NOT
    appended — we only need surprisal of target tokens under context.
    """
    bos = tokenizer.bos_id() if hasattr(tokenizer, "bos_id") else -1
    eos = tokenizer.eos_id() if hasattr(tokenizer, "eos_id") else -1
    bos = bos if bos is not None and bos >= 0 else None
    eos = eos if eos is not None and eos >= 0 else None

    ids = list(tokenizer.encode(text))
    if bos is not None:
        ids = [bos] + ids
    return ids, bos, eos


def _encode_continuation(tokenizer, text: str) -> List[int]:
    """Encode `text` *without* adding BOS (used for target after context)."""
    return list(tokenizer.encode(text))


def _decode_ids(tokenizer, ids: List[int]) -> List[str]:
    """Decode a list of token ids to per-subword strings."""
    return [tokenizer.decode([i]) for i in ids]


def build_cache(
    csv_paths: List[Path],
    tokenizer,
    tokenizer_path: Path,
) -> StimulusCache:
    """Tokenize every row of every CSV and return a StimulusCache.

    Inputs must be v2-schema CSVs. Tokenizer must implement
    `encode(text) -> List[int]`, `decode(ids) -> str`, and the SP-style
    `bos_id()` / `eos_id()` methods (may return -1 for unused).
    """
    stimuli_id = hash_stimuli_files(csv_paths)
    tokenizer_id = hash_tokenizer(tokenizer_path)

    entries: List[StimulusCacheEntry] = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        required = [
            "item_id", "category", "condition", "pronoun_status",
            "context", "target", "hotspot_token", "hotspot_position",
            "language",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path.name}: missing v2 columns {missing}"
            )

        for _, row in df.iterrows():
            context = str(row["context"])
            target = str(row["target"])
            # Context → ids with BOS
            context_ids, bos_id, eos_id = _encode_with_specials(tokenizer, context)
            context_len = len(context_ids)
            # Target → ids without BOS (continuation after context)
            target_ids = _encode_continuation(tokenizer, target)
            target_subword_count = len(target_ids)
            full_ids = context_ids + target_ids
            full_len = len(full_ids)

            target_token_strings = _decode_ids(tokenizer, target_ids)

            # Resolve hotspot position from word-index to subword-index.
            word_pos = int(row["hotspot_position"])
            target_subword_idx = _resolve_hotspot_subword_index(
                target=target,
                hotspot_word_index=word_pos,
                hotspot_token=str(row["hotspot_token"]),
                target_token_ids=target_ids,
                target_token_strings=target_token_strings,
            )
            # Full-sequence index of hotspot = context_len + target_subword_idx
            hotspot_token_index = context_len + target_subword_idx

            entries.append(StimulusCacheEntry(
                item_id=int(row["item_id"]),
                category=str(row["category"]),
                condition=str(row["condition"]),
                pronoun_status=int(row["pronoun_status"]),
                language=str(row["language"]),
                context=context,
                target=target,
                hotspot_token=str(row["hotspot_token"]),
                hotspot_position=word_pos,
                names=str(row.get("names", "")),
                generator=str(row.get("generator", "")),
                source_file=csv_path.name,
                context_token_ids=context_ids,
                target_token_ids=target_ids,
                full_token_ids=full_ids,
                context_len=context_len,
                full_len=full_len,
                target_subword_count=target_subword_count,
                hotspot_token_index=hotspot_token_index,
                bos_id=bos_id,
                eos_id=eos_id,
            ))

    meta = {
        "n_rows": len(entries),
        "n_files": len(csv_paths),
        "languages": sorted({e.language for e in entries}),
        "categories": sorted({e.category for e in entries}),
    }

    return StimulusCache(
        stimuli_id=stimuli_id,
        tokenizer_id=tokenizer_id,
        rows=entries,
        meta=meta,
    )


def build_or_load_cache(
    csv_paths: List[Path],
    tokenizer,
    tokenizer_path: Path,
    cache_dir: Path,
) -> StimulusCache:
    """Load a cached artifact if present, else build and save.

    Cache filename: `{stimuli_id}__{tokenizer_id}.pkl` under `cache_dir`.
    """
    cache_dir = Path(cache_dir)
    stimuli_id = hash_stimuli_files(csv_paths)
    tokenizer_id = hash_tokenizer(tokenizer_path)
    cache_path = cache_dir / f"{stimuli_id}__{tokenizer_id}.pkl"
    if cache_path.exists():
        logger.info("Loading cached stimuli from %s", cache_path)
        cache = StimulusCache.load(cache_path)
        # Defensive: verify ids match. If not, treat as stale and rebuild.
        if cache.stimuli_id != stimuli_id or cache.tokenizer_id != tokenizer_id:
            logger.warning("Cache id mismatch; rebuilding.")
        else:
            return cache
    cache = build_cache(csv_paths, tokenizer, tokenizer_path)
    cache.save(cache_path)
    return cache
