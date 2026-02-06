"""
Annotation candidate sampler for pronoun recovery.

Concatenates cleaned corpus lines into dense chunks of ~254 subword
tokens (matching the DeBERTa max_length minus CLS/SEP), mirroring how
the LLM training pipeline packs text into fixed-length sequences.
Then draws a stratified random sample of chunks.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from ..config import LLMAnnotationConfig

logger = logging.getLogger(__name__)

# Speaker-tag patterns (same as analysis/corpus_descriptives/line_cleaners.py)
_CHILDES_PATTERN = re.compile(r"^\*[A-Z]{2,4}:\t(.*)$")
_SWITCHBOARD_PATTERN = re.compile(r"^[AB]:\t(.*)$")

# DeBERTa adds [CLS] + [SEP] = 2 special tokens.
_SPECIAL_TOKEN_OVERHEAD = 2


def _clean_line(text: str, genre_key: str) -> str:
    """Strip speaker tags for CHILDES / Switchboard; pass-through otherwise."""
    genre_lower = genre_key.lower()
    if genre_lower == "childes":
        m = _CHILDES_PATTERN.match(text)
        if m:
            return m.group(1).strip()
    elif genre_lower == "switchboard":
        m = _SWITCHBOARD_PATTERN.match(text)
        if m:
            return m.group(1).strip()
    return text.strip()


class AnnotationSampler:
    """Sample dense subword-token chunks from the corpus for annotation.

    Reads each ``.train`` file, cleans lines (strip speaker tags, skip
    empties), concatenates the text, and slices it into chunks that fit
    within the model's subword token budget (default 254 content tokens,
    leaving room for [CLS] and [SEP]).  Then draws a proportionally
    stratified random sample of those chunks.

    This ensures annotation examples match what the pronoun-recovery
    model sees at inference time.
    """

    # The DeBERTa tokenizer used for subword chunking (must match the
    # token-classifier model, NOT the LLM used for annotation).
    _DEBERTA_TOKENIZER = "microsoft/deberta-v3-base"

    def __init__(self, config: LLMAnnotationConfig):
        self.config = config
        self._tokenizer = AutoTokenizer.from_pretrained(self._DEBERTA_TOKENIZER)
        # Max subword content tokens per chunk (excluding special tokens).
        self._max_content_tokens: int = 256 - _SPECIAL_TOKEN_OVERHEAD

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_candidates(self) -> List[Dict[str, Any]]:
        """Build chunks and draw ``config.sample_size`` random ones.

        Returns:
            List of dicts with keys: ``id``, ``text``, ``genre``,
            ``source_file``, ``start_line``, ``end_line``.
        """
        corpus_path = Path(self.config.corpus_path)
        train_files = sorted(corpus_path.glob("*.train"))
        if not train_files:
            raise FileNotFoundError(f"No .train files in {corpus_path}")

        logger.info(
            "Building %d-subword chunks from %d corpus files",
            self._max_content_tokens,
            len(train_files),
        )

        # Chunk each file independently (no cross-genre concatenation).
        all_chunks: List[Dict[str, Any]] = []

        for fpath in train_files:
            genre_key = fpath.stem
            genre_display = self._resolve_genre(fpath)
            if genre_display is None:
                logger.warning("Unknown genre for %s, skipping", fpath.name)
                continue

            file_chunks = self._chunk_file(fpath, genre_key, genre_display)
            all_chunks.extend(file_chunks)
            logger.info(
                "  %s (%s): %d chunks", fpath.name, genre_display, len(file_chunks)
            )

        if not all_chunks:
            logger.warning("No chunks produced; returning empty list")
            return []

        logger.info("Total chunks across all genres: %d", len(all_chunks))

        sampled = self._stratified_sample(all_chunks)
        logger.info(
            "Sampled %d chunks across %d genres (requested %d)",
            len(sampled),
            len({c["genre"] for c in sampled}),
            self.config.sample_size,
        )
        return sampled

    def save_candidates(self, candidates: List[Dict[str, Any]], output_path: Path) -> None:
        """Save candidates as JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for cand in candidates:
                f.write(json.dumps(cand, ensure_ascii=False) + "\n")
        logger.info("Saved %d candidates to %s", len(candidates), output_path)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_file(
        self,
        fpath: Path,
        genre_key: str,
        genre_display: str,
    ) -> List[Dict[str, Any]]:
        """Read, clean, concatenate, and chunk a single corpus file.

        Lines are concatenated with a single space between them.  When
        adding the next line would exceed ``_max_content_tokens``, the
        current buffer is flushed as a chunk and a new one is started.
        """
        chunks: List[Dict[str, Any]] = []
        buf_text = ""
        buf_token_count = 0
        buf_start_line = 1
        chunk_idx = 0

        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                cleaned = _clean_line(raw_line, genre_key)
                if not cleaned:
                    continue

                line_tokens = len(self._tokenizer.tokenize(cleaned))
                if line_tokens == 0:
                    continue

                # Truncate oversized lines to fit within the token budget.
                if line_tokens > self._max_content_tokens:
                    token_ids = self._tokenizer.encode(
                        cleaned, add_special_tokens=False
                    )[:self._max_content_tokens]
                    cleaned = self._tokenizer.decode(
                        token_ids, skip_special_tokens=True
                    )
                    line_tokens = self._max_content_tokens

                if buf_token_count == 0:
                    # Start a new buffer.
                    buf_text = cleaned
                    buf_token_count = line_tokens
                    buf_start_line = line_num
                elif buf_token_count + line_tokens + 1 <= self._max_content_tokens:
                    # +1 for the space joining token (approximate).
                    buf_text += " " + cleaned
                    buf_token_count += line_tokens + 1
                else:
                    # Flush the current buffer as a chunk.
                    chunks.append(self._make_chunk(
                        fpath, genre_display, chunk_idx,
                        buf_text, buf_start_line, line_num - 1,
                    ))
                    chunk_idx += 1
                    # Start new buffer with this line.
                    buf_text = cleaned
                    buf_token_count = line_tokens
                    buf_start_line = line_num

        # Flush remaining buffer (drop if very short — less than 25% full).
        if buf_token_count >= self._max_content_tokens // 4:
            chunks.append(self._make_chunk(
                fpath, genre_display, chunk_idx,
                buf_text, buf_start_line, None,
            ))

        return chunks

    @staticmethod
    def _make_chunk(
        fpath: Path,
        genre: str,
        chunk_idx: int,
        text: str,
        start_line: int,
        end_line: Optional[int],
    ) -> Dict[str, Any]:
        return {
            "id": f"{fpath.stem}:chunk_{chunk_idx}",
            "text": text,
            "genre": genre,
            "source_file": fpath.name,
            "start_line": start_line,
            "end_line": end_line,
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _stratified_sample(
        self, all_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Proportional random sampling stratified by genre."""
        n = min(self.config.sample_size, len(all_chunks))
        if n >= len(all_chunks):
            logger.info("Requested sample >= total chunks; returning all")
            return all_chunks

        rng = random.Random(42)

        genre_groups: Dict[str, List[int]] = {}
        for idx, chunk in enumerate(all_chunks):
            genre_groups.setdefault(chunk["genre"], []).append(idx)

        total = len(all_chunks)
        sampled_indices: List[int] = []

        for genre, idxs in sorted(genre_groups.items()):
            genre_n = max(1, round(n * len(idxs) / total))
            genre_n = min(genre_n, len(idxs))
            sampled_indices.extend(rng.sample(idxs, genre_n))

        # Trim or pad to exactly n.
        if len(sampled_indices) > n:
            sampled_indices = rng.sample(sampled_indices, n)
        elif len(sampled_indices) < n:
            remaining = set(range(total)) - set(sampled_indices)
            shortfall = n - len(sampled_indices)
            sampled_indices.extend(
                rng.sample(sorted(remaining), min(shortfall, len(remaining)))
            )

        return [all_chunks[i] for i in sorted(sampled_indices)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_genre(self, fpath: Path) -> Optional[str]:
        """Resolve genre display name from filename stem."""
        return self.config.genre_map.get(fpath.stem)
