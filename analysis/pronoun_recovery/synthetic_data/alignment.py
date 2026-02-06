"""
Token-index alignment and verb-label computation for pronoun-dropped text.

When subject pronouns are removed from a spaCy ``Doc``, every token that
follows a dropped token shifts left.  The functions here compute that
mapping and derive per-verb labels that record *what* was dropped.
"""

from typing import Any, Dict, List, Set

import spacy.tokens

from analysis.pronoun_recovery.constants import (
    LABEL_NONE,
    MORPH_TO_LABEL_SUFFIX,
)


# ── Index alignment ─────────────────────────────────────────────────────


def build_alignment_map(
    doc: spacy.tokens.Doc,
    dropped_indices: Set[int],
) -> Dict[int, int]:
    """Map original token indices to their positions in the dropped text.

    After removing *N* tokens whose original indices precede token *i*,
    the new index of token *i* in the dropped text is ``i - N``.

    Args:
        doc: The original spaCy ``Doc`` (before any tokens are removed).
        dropped_indices: Set of token indices that were dropped.

    Returns:
        Dict mapping each *retained* original index to its new
        (post-drop) index.  Dropped indices are absent from the map.
    """
    alignment: Dict[int, int] = {}
    new_idx = 0
    for token in doc:
        if token.i in dropped_indices:
            continue
        alignment[token.i] = new_idx
        new_idx += 1
    return alignment


# ── Verb-label computation ───────────────────────────────────────────────


def _pronoun_label(token: spacy.tokens.Token) -> str:
    """Derive the ``PRO.*`` label for a dropped subject pronoun.

    Uses the ``Person`` and ``Number`` morphological features of *token*
    to look up the label suffix in :data:`MORPH_TO_LABEL_SUFFIX`.

    Returns:
        A label such as ``"PRO.3sg"`` or ``"PRO.3pl"``.  Falls back to
        ``"PRO.3sg"`` when morphological features are missing.
    """
    person_feats = token.morph.get("Person")
    number_feats = token.morph.get("Number")

    person = person_feats[0] if person_feats else None
    number = number_feats[0] if number_feats else None

    if person and number:
        suffix = MORPH_TO_LABEL_SUFFIX.get((person, number))
        if suffix:
            return f"PRO.{suffix}"

    # Fallback: default to 3sg when morphology is incomplete.
    return "PRO.3sg"


def compute_verb_labels(
    doc: spacy.tokens.Doc,
    dropped_indices: Set[int],
    alignment_map: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Compute a label for every finite verb in the dropped text.

    A verb receives a ``PRO.*`` label if one of the dropped pronouns has
    that verb as its syntactic head; otherwise it receives
    :data:`LABEL_NONE`.

    Only tokens whose ``pos_`` is ``"VERB"`` or ``"AUX"`` and whose
    ``VerbForm`` morphological feature is ``"Fin"`` (finite) are
    considered verbs.

    Args:
        doc: The original spaCy ``Doc``.
        dropped_indices: Set of original-token indices that were dropped.
        alignment_map: Mapping from original index to dropped-text index,
            as returned by :func:`build_alignment_map`.

    Returns:
        A list of dicts, one per finite verb retained in the dropped
        text.  Each dict contains:

        * ``verb_idx`` -- token index in the *dropped* text.
        * ``verb_text`` -- surface form of the verb.
        * ``label`` -- ``"PRO.<person><number>"`` or ``"NONE"``.
        * ``original_pronoun`` -- text of the dropped pronoun, or
          ``None`` if the label is ``"NONE"``.
    """
    # Build lookup: original verb index -> (pronoun label, pronoun text)
    # for verbs that had a dropped subject pronoun.
    verb_to_pronoun: Dict[int, tuple] = {}
    for idx in dropped_indices:
        pronoun_token = doc[idx]
        head_idx = pronoun_token.head.i
        label = _pronoun_label(pronoun_token)
        verb_to_pronoun[head_idx] = (label, pronoun_token.text)

    labels: List[Dict[str, Any]] = []
    for token in doc:
        # Skip dropped tokens.
        if token.i in dropped_indices:
            continue

        # Only consider finite verbs.
        if token.pos_ not in ("VERB", "AUX"):
            continue
        verb_forms = token.morph.get("VerbForm")
        if not verb_forms or verb_forms[0] != "Fin":
            continue

        new_idx = alignment_map.get(token.i)
        if new_idx is None:
            # Should not happen for retained tokens, but guard anyway.
            continue

        if token.i in verb_to_pronoun:
            label, pronoun_text = verb_to_pronoun[token.i]
        else:
            label = LABEL_NONE
            pronoun_text = None

        labels.append(
            {
                "verb_idx": new_idx,
                "verb_text": token.text,
                "label": label,
                "original_pronoun": pronoun_text,
            }
        )

    return labels
