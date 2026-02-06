"""
Wh-extraction annotator.

Identifies wh-words and classifies extraction type and clause level.
"""

from typing import Any, Dict, List, Optional

import spacy

from ..constants import WH_LEMMAS_EN, WH_LEMMAS_IT
from .base import BaseSentenceAnnotator


def _is_wh_word(tok: spacy.tokens.Token, wh_lemmas: frozenset = WH_LEMMAS_EN) -> bool:
    """Check if token is a wh-word."""
    pron_types = tok.morph.get("PronType")
    if pron_types and "Int" in pron_types:
        return True
    return tok.lemma_.lower() in wh_lemmas


def _extraction_type(tok: spacy.tokens.Token) -> str:
    """Classify wh-word extraction: subject / object / adjunct."""
    dep = tok.dep_
    if dep in ("nsubj", "nsubj:pass"):
        return "subject"
    elif dep in ("obj", "iobj"):
        return "object"
    else:
        return "adjunct"


class WhExtractionAnnotator(BaseSentenceAnnotator):
    """
    Annotates wh-extractions (questions and embedded wh-clauses).

    For each wh-word, extracts:
    - Token position and lemma
    - Extraction type (subject, object, adjunct)
    - Clause level (root, embedded)
    """

    output_fields = {
        "wh_extractions": "List of wh-extraction annotation dicts",
    }

    def __init__(self, language: str = "en"):
        self.language = language
        self._wh_lemmas = WH_LEMMAS_IT if language == "it" else WH_LEMMAS_EN

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract wh-extraction annotations."""
        wh_extractions = []

        for tok in sent:
            if not _is_wh_word(tok, self._wh_lemmas):
                continue

            ext_type = _extraction_type(tok)

            # Clause level: root if wh-word's head is ROOT, else embedded
            head = tok.head
            if head.dep_ == "ROOT" or tok.dep_ == "ROOT":
                clause_level = "root"
            else:
                clause_level = "embedded"

            extraction = {
                "wh_idx": tok.i - sent.start,
                "wh_lemma": tok.lemma_.lower(),
                "extraction_type": ext_type,
                "clause_level": clause_level,
            }
            wh_extractions.append(extraction)

        return {"wh_extractions": wh_extractions}

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Compute wh-extraction related flags."""
        extractions = annotations.get("wh_extractions", [])
        return {"has_wh_extraction": len(extractions) > 0}
