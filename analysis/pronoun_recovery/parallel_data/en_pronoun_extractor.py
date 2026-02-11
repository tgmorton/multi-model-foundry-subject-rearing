"""
Extract English subject pronouns from spaCy-parsed sentences.

Identifies nsubj pronouns (I, we, he, she, they, you) while filtering
out expletive "it", relative pronouns, and optionally all "it".
"""

from dataclasses import dataclass
from typing import List, Optional

import spacy.tokens

# Pronouns we map to labels.  "it" is excluded by default (too noisy).
_SUBJECT_PRONOUNS = {"i", "we", "he", "she", "they", "you"}
_RELATIVE_PRONOUNS = {"who", "which", "that", "whom", "whose"}

# EN pronoun text → candidate label
_PRONOUN_TO_LABEL = {
    "i": "PRO.1sg",
    "we": "PRO.1pl",
    "he": "PRO.3sg",
    "she": "PRO.3sg",
    "they": "PRO.3pl",
    # "you" resolved later via IT verb morphology
}


@dataclass
class ExtractedPronoun:
    """An English subject pronoun extracted from a parsed sentence."""

    token_idx: int
    text: str
    head_idx: int
    candidate_label: Optional[str]  # None for "you" (needs morph resolution)


def extract_subject_pronouns(
    doc: spacy.tokens.Doc,
    *,
    skip_all_it: bool = True,
) -> List[ExtractedPronoun]:
    """Find English subject pronouns suitable for alignment.

    Args:
        doc: spaCy-parsed English sentence.
        skip_all_it: If True, skip every occurrence of "it" (expletive
            and referential alike — too noisy for parallel alignment).

    Returns:
        List of extracted pronouns with their token indices and
        candidate labels.
    """
    results: List[ExtractedPronoun] = []

    for token in doc:
        # Must be a pronoun acting as subject.
        if token.dep_ != "nsubj" or token.pos_ != "PRON":
            continue

        lower = token.text.lower()

        # Skip "it" entirely when flag is set.
        if skip_all_it and lower == "it":
            continue

        # Skip relative pronouns.
        if lower in _RELATIVE_PRONOUNS:
            continue

        # Must be one of our target pronoun forms.
        if lower not in _SUBJECT_PRONOUNS:
            continue

        candidate_label = _PRONOUN_TO_LABEL.get(lower)  # None for "you"

        results.append(
            ExtractedPronoun(
                token_idx=token.i,
                text=lower,
                head_idx=token.head.i,
                candidate_label=candidate_label,
            )
        )

    return results
