"""
Resolve pronoun labels from EN-IT alignment + morphological cross-check.

Given an English subject pronoun aligned to an Italian token, resolves:
1. The PRO.* label (from EN pronoun text, or IT morph for "you")
2. Confidence level (high/medium) based on morph agreement
3. Whether to skip the mapping (morph disagreement, overt subject, etc.)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import spacy.tokens

from analysis.pronoun_recovery.constants import (
    IT_DEFAULT_PRONOUN,
    LABEL_TO_ID,
    MORPH_TO_LABEL_SUFFIX,
)

from .en_pronoun_extractor import ExtractedPronoun
from .it_null_subject_detector import ItalianVerb
from .quality_filters import FilterStats

logger = logging.getLogger(__name__)


@dataclass
class ResolvedMarker:
    """A resolved pronoun marker ready for output."""

    label: str  # e.g. "PRO.1sg"
    lexical_form: str  # e.g. "io"
    position: int  # character offset in IT clean_text
    it_verb_idx: int  # token index of IT verb
    it_verb_text: str
    confidence: str  # "high" or "medium"
    en_pronoun: str  # original EN pronoun text


def _find_aligned_verb(
    en_pronoun_idx: int,
    en_to_it: Dict[int, List[int]],
    it_doc: spacy.tokens.Doc,
    it_verb_lookup: Dict[int, ItalianVerb],
    max_hops: int = 3,
) -> Optional[ItalianVerb]:
    """Follow alignment from EN pronoun to IT verb.

    If the aligned IT token is not a verb, walk up the dependency tree
    (max_hops) looking for a finite verb.

    Args:
        en_pronoun_idx: Token index of the EN pronoun.
        en_to_it: EN→IT alignment dict.
        it_doc: Parsed Italian sentence.
        it_verb_lookup: Dict mapping IT token index → ItalianVerb.
        max_hops: Maximum dependency-tree hops to find a verb.

    Returns:
        The ItalianVerb if found, or None.
    """
    aligned_it_indices = en_to_it.get(en_pronoun_idx, [])
    if not aligned_it_indices:
        return None

    for it_idx in aligned_it_indices:
        if it_idx >= len(it_doc):
            continue

        # Direct hit: aligned token is a known finite verb.
        if it_idx in it_verb_lookup:
            return it_verb_lookup[it_idx]

        # Walk up dep tree looking for a finite verb.
        tok = it_doc[it_idx]
        for _ in range(max_hops):
            if tok.head.i == tok.i:
                break  # reached root
            tok = tok.head
            if tok.i in it_verb_lookup:
                return it_verb_lookup[tok.i]

    return None


def resolve_markers(
    en_doc: spacy.tokens.Doc,
    it_doc: spacy.tokens.Doc,
    extracted_pronouns: List[ExtractedPronoun],
    it_verbs: List[ItalianVerb],
    en_to_it: Dict[int, List[int]],
    stats: Optional[FilterStats] = None,
) -> List[ResolvedMarker]:
    """Resolve EN pronouns → IT verb markers with morphological cross-check.

    For each extracted EN subject pronoun:
    1. Follow alignment to find the corresponding IT finite verb.
    2. Skip if the IT verb already has an overt subject.
    3. Derive candidate label from EN pronoun (special handling for "you").
    4. Cross-check label against IT verb morphology.
    5. Deduplicate: one marker per IT verb (first pronoun wins).

    Args:
        en_doc: Parsed English sentence.
        it_doc: Parsed Italian sentence.
        extracted_pronouns: EN subject pronouns from en_pronoun_extractor.
        it_verbs: Italian finite verbs from it_null_subject_detector.
        en_to_it: EN→IT alignment dictionary.
        stats: Optional FilterStats to update.

    Returns:
        List of resolved markers, deduplicated by IT verb.
    """
    # Build verb lookup by token index.
    verb_lookup: Dict[int, ItalianVerb] = {v.token_idx: v for v in it_verbs}

    markers: List[ResolvedMarker] = []
    seen_verbs: set = set()  # Deduplicate by IT verb index.

    for pronoun in extracted_pronouns:
        if stats:
            stats.total_pronouns += 1

        # Step 1: Find aligned IT verb.
        it_verb = _find_aligned_verb(
            pronoun.token_idx, en_to_it, it_doc, verb_lookup
        )
        if it_verb is None:
            if stats:
                stats.skipped_no_alignment += 1
            continue

        # Deduplication: one marker per IT verb.
        if it_verb.token_idx in seen_verbs:
            continue
        seen_verbs.add(it_verb.token_idx)

        # Step 2: Skip if IT verb has overt subject.
        if it_verb.has_overt_subject:
            if stats:
                stats.skipped_overt_subject += 1
            continue

        # Step 3: Derive candidate label.
        if pronoun.text == "you":
            # Use IT verb morphology to resolve 2sg vs 2pl.
            if it_verb.morph_label_suffix:
                if it_verb.morph_label_suffix.startswith("2"):
                    candidate_label = f"PRO.{it_verb.morph_label_suffix}"
                else:
                    # "you" but IT verb is not 2nd person — skip.
                    if stats:
                        stats.skipped_morph_disagree += 1
                    continue
            else:
                # No morph on IT verb; default to 2pl (formal "you" in parliament).
                candidate_label = "PRO.2pl"
        else:
            candidate_label = pronoun.candidate_label
            if candidate_label is None:
                continue

        # Step 4: Cross-check morph agreement.
        if it_verb.morph_label_suffix:
            expected_suffix = candidate_label.split(".")[-1]  # e.g. "1sg"
            if it_verb.morph_label_suffix != expected_suffix:
                # Morph disagrees — skip.
                if stats:
                    stats.skipped_morph_disagree += 1
                continue
            confidence = "high"
        else:
            # No morph available — trust EN label.
            confidence = "medium"

        # Step 5: Build marker.
        # Position is the character offset of the IT verb in the text.
        it_verb_token = it_doc[it_verb.token_idx]
        position = it_verb_token.idx

        # Look up default Italian lexical form for this label.
        lexical_form = IT_DEFAULT_PRONOUN.get(candidate_label, "")

        if stats:
            if confidence == "high":
                stats.passed_high_confidence += 1
            else:
                stats.passed_medium_confidence += 1

        markers.append(
            ResolvedMarker(
                label=candidate_label,
                lexical_form=lexical_form,
                position=position,
                it_verb_idx=it_verb.token_idx,
                it_verb_text=it_verb.text,
                confidence=confidence,
                en_pronoun=pronoun.text,
            )
        )

    return markers
