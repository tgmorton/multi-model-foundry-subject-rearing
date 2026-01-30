"""
Speaker label preprocessing for corpus-specific formatting.

Strips speaker labels from CHILDES and Switchboard data before spaCy processing.
Returns (cleaned_text, metadata_dict) per line.
"""

import re
from typing import Dict, Optional, Tuple

from .constants import CHILDES_ADULT_SPEAKERS, CHILDES_CHILD_SPEAKERS

# Patterns
_CHILDES_PATTERN = re.compile(r"^\*([A-Z]{2,4}):\t(.*)$")
_SWITCHBOARD_PATTERN = re.compile(r"^([AB]):\t(.*)$")


def clean_childes_line(line: str) -> Tuple[str, Dict[str, Optional[str]]]:
    """
    Strip CHILDES speaker labels.

    Input format: ``*SPEAKER:\\t utterance text``
    Returns cleaned text and metadata with speaker code and role.
    """
    m = _CHILDES_PATTERN.match(line)
    if m:
        speaker_code = m.group(1)
        text = m.group(2).strip()
        if speaker_code in CHILDES_CHILD_SPEAKERS:
            role = "child"
        elif speaker_code in CHILDES_ADULT_SPEAKERS:
            role = "adult"
        else:
            role = "unknown"
        return text, {"speaker": speaker_code, "role": role}
    # No match — return as-is
    return line.strip(), {"speaker": None, "role": None}


def clean_switchboard_line(line: str) -> Tuple[str, Dict[str, Optional[str]]]:
    """
    Strip Switchboard speaker labels.

    Input format: ``A:\\t utterance`` or ``B:\\t utterance``
    """
    m = _SWITCHBOARD_PATTERN.match(line)
    if m:
        speaker_label = m.group(1)
        text = m.group(2).strip()
        return text, {"speaker": speaker_label, "role": None}
    return line.strip(), {"speaker": None, "role": None}


def clean_generic_line(line: str) -> Tuple[str, Dict[str, Optional[str]]]:
    """Pass-through cleaner for corpora without speaker labels."""
    return line.strip(), {"speaker": None, "role": None}


def get_cleaner(genre: str):
    """Return the appropriate line cleaner function for a genre."""
    genre_lower = genre.lower()
    if genre_lower == "childes":
        return clean_childes_line
    elif genre_lower == "switchboard":
        return clean_switchboard_line
    else:
        return clean_generic_line
