"""
Parser and validator for bracket-format pronoun recovery annotations.

Annotation format::

    Original:  "Parla italiano."
    Annotated: "[PRO.3sg:lei] Parla italiano."

Regex pattern for markers::

    \\[PRO\\.(1sg|2sg|3sg|1pl|2pl|3pl|IMP|CONJ)(?::(\\w+))?\\]

"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..constants import LANGUAGE_FORM_MAPS, PRONOUN_LABELS

logger = logging.getLogger(__name__)

# Closed set of valid label suffixes (the part after "PRO." or standalone)
_LABEL_SUFFIXES = {"1sg", "2sg", "3sg", "1pl", "2pl", "3pl", "IMP", "CONJ"}

# Regex matching annotation markers in bracket format
_MARKER_PATTERN = re.compile(
    r"\[PRO\.(1sg|2sg|3sg|1pl|2pl|3pl|IMP|CONJ)(?::(\w+))?\]"
)


@dataclass
class AnnotationMarker:
    """A single pronoun recovery annotation marker.

    Attributes:
        label: The pronoun label, e.g. "PRO.3sg", "IMP", "CONJ".
        lexical_form: The lexical pronoun form, e.g. "lei", "she", or
            None for IMP/CONJ markers.
        position: Character offset in the *clean* (marker-stripped) text
            where the pronoun would be inserted.
    """

    label: str
    lexical_form: Optional[str]
    position: int


def parse_annotations(annotated_text: str) -> Tuple[str, List[AnnotationMarker]]:
    """Parse annotated text and extract markers.

    Args:
        annotated_text: Text containing bracket-format annotation markers.

    Returns:
        Tuple of (clean_text, markers) where clean_text has all marker
        brackets removed and each marker's position is the character
        offset in clean_text where the recovered pronoun would appear.
    """
    markers: List[AnnotationMarker] = []

    # Track the cumulative length of markers removed so far, so we can
    # compute positions relative to the clean text.
    offset_removed = 0
    clean_parts: List[str] = []
    last_end = 0

    for match in _MARKER_PATTERN.finditer(annotated_text):
        # Append text between the previous marker (or start) and this one
        clean_parts.append(annotated_text[last_end:match.start()])

        suffix = match.group(1)  # e.g. "3sg", "IMP"
        lexical = match.group(2)  # e.g. "lei" or None

        # Build the canonical label
        if suffix in ("IMP", "CONJ"):
            label = suffix
        else:
            label = f"PRO.{suffix}"

        # Position in clean text = match start minus total chars removed
        clean_position = match.start() - offset_removed

        markers.append(AnnotationMarker(
            label=label,
            lexical_form=lexical,
            position=clean_position,
        ))

        offset_removed += len(match.group(0))
        last_end = match.end()

    # Append any remaining text after the last marker
    clean_parts.append(annotated_text[last_end:])
    clean_text = "".join(clean_parts)

    return clean_text, markers


def validate_annotation(
    marker: AnnotationMarker, language: str = "en"
) -> List[str]:
    """Validate a single annotation marker.

    Checks:
        1. Label is in the closed set (PRONOUN_LABELS from constants).
        2. IMP and CONJ markers must have no lexical form.
        3. Lexical form is consistent with label + language
           (using LANGUAGE_FORM_MAPS).

    Args:
        marker: The annotation marker to validate.
        language: Language code (default "en").

    Returns:
        List of error messages.  Empty list means valid.
    """
    errors: List[str] = []

    # 1. Check label is in the closed set
    if marker.label not in PRONOUN_LABELS:
        errors.append(
            f"Unknown label '{marker.label}'; "
            f"expected one of {sorted(PRONOUN_LABELS)}"
        )

    # 2. IMP / CONJ must not carry a lexical form
    if marker.label in ("IMP", "CONJ"):
        if marker.lexical_form is not None:
            errors.append(
                f"{marker.label} marker should not have a lexical form, "
                f"got '{marker.lexical_form}'"
            )
        # No further checks needed for IMP/CONJ
        return errors

    # 3. Person/number labels should have a lexical form
    if marker.lexical_form is None:
        errors.append(
            f"Label '{marker.label}' requires a lexical form but none was given"
        )
        return errors

    # 4. Check lexical form against language-specific valid forms
    form_map = LANGUAGE_FORM_MAPS.get(language)
    if form_map is None:
        errors.append(f"No form map for language '{language}'")
        return errors

    valid_forms = form_map.get(marker.label)
    if valid_forms is None:
        # Label already flagged as unknown above if needed
        if marker.label in PRONOUN_LABELS:
            errors.append(
                f"No valid forms defined for label '{marker.label}' "
                f"in language '{language}'"
            )
        return errors

    if marker.lexical_form.lower() not in valid_forms:
        errors.append(
            f"Lexical form '{marker.lexical_form}' not valid for "
            f"label '{marker.label}' in '{language}'; "
            f"expected one of {sorted(valid_forms)}"
        )

    return errors
