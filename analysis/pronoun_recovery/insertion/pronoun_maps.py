"""Deterministic feature -> lexical pronoun tables for Stage 2 insertion.

For all labels except PRO.3sg, the mapping is fully deterministic (one label
maps to exactly one lexical form).  PRO.3sg requires gender resolution via
morphological cues (see :mod:`.gender_resolver`).

IMP, CONJ, and NONE labels produce no insertion.
"""

from typing import Optional

from ..constants import LANGUAGE_DEFAULT_PRONOUNS


def get_default_pronoun(label: str, language: str = "en") -> Optional[str]:
    """Look up the default lexical pronoun for a label.

    Returns ``None`` for IMP, CONJ, and NONE (no insertion needed).
    For PRO.3sg, returns the language default (English: ``'they'``,
    Italian: ``'lui'``).  For all other ``PRO.*`` labels, returns the
    deterministic pronoun.

    Args:
        label: Predicted label string (e.g. ``"PRO.1sg"``, ``"IMP"``).
        language: ISO 639-1 language code (default ``"en"``).

    Returns:
        The lexical pronoun to insert, or ``None`` if no insertion is needed.
    """
    if label in ("IMP", "CONJ", "NONE"):
        return None
    defaults = LANGUAGE_DEFAULT_PRONOUNS.get(language, {})
    return defaults.get(label)


def needs_gender_resolution(label: str) -> bool:
    """Check if a label requires gender resolution (only PRO.3sg).

    Args:
        label: Predicted label string.

    Returns:
        ``True`` if the label is ``"PRO.3sg"``, ``False`` otherwise.
    """
    return label == "PRO.3sg"
