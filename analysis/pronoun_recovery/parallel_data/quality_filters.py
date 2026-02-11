"""
Pair-level and pronoun-level quality filters for Europarl alignment.

Filters operate at two levels:
- Pair-level: reject entire sentence pairs that are too short, too long,
  or have extreme length ratios (likely misaligned).
- Pronoun-level: reject individual pronoun-verb mappings that fail
  alignment or morphological checks.
"""

import logging
from dataclasses import dataclass
from typing import Dict

import spacy.tokens

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Accumulates skip/pass counts for reporting."""

    total_pairs: int = 0
    passed_pairs: int = 0
    skipped_too_short: int = 0
    skipped_too_long: int = 0
    skipped_length_ratio: int = 0
    skipped_empty: int = 0

    # Pronoun-level
    total_pronouns: int = 0
    skipped_no_alignment: int = 0
    skipped_overt_subject: int = 0
    skipped_morph_disagree: int = 0
    skipped_not_verb: int = 0
    passed_high_confidence: int = 0
    passed_medium_confidence: int = 0

    def summary(self) -> Dict[str, int]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, int)}


def check_pair_quality(
    en_doc: spacy.tokens.Doc,
    it_doc: spacy.tokens.Doc,
    *,
    min_tokens: int = 3,
    max_tokens: int = 128,
    max_length_ratio: float = 3.0,
    stats: FilterStats = None,
) -> bool:
    """Check whether a sentence pair passes basic quality filters.

    Args:
        en_doc: Parsed English sentence.
        it_doc: Parsed Italian sentence.
        min_tokens: Minimum token count (either side).
        max_tokens: Maximum token count (either side).
        max_length_ratio: Maximum ratio between EN and IT token counts.
        stats: Optional FilterStats to update.

    Returns:
        True if the pair passes all filters.
    """
    en_len = len(en_doc)
    it_len = len(it_doc)

    if en_len == 0 or it_len == 0:
        if stats:
            stats.skipped_empty += 1
        return False

    if en_len < min_tokens or it_len < min_tokens:
        if stats:
            stats.skipped_too_short += 1
        return False

    if en_len > max_tokens or it_len > max_tokens:
        if stats:
            stats.skipped_too_long += 1
        return False

    ratio = max(en_len / it_len, it_len / en_len)
    if ratio > max_length_ratio:
        if stats:
            stats.skipped_length_ratio += 1
        return False

    return True
