"""
That-trace annotator.

Identifies bridge verb complements and analyzes complementizer presence,
subject status, and extraction type for that-trace environment analysis.
"""

from typing import Any, Dict, List, Optional

import spacy

from preprocessing.dep_labels import normalize_dep

from ..constants import ENGLISH_BRIDGE_VERBS, ITALIAN_BRIDGE_VERBS, WH_LEMMAS_EN, WH_LEMMAS_IT
from .base import BaseSentenceAnnotator


def _has_embedded_wh(verb_tok: spacy.tokens.Token, wh_lemmas: frozenset = WH_LEMMAS_EN) -> bool:
    """Check if the ccomp verb has a wh-word as its own dependent."""
    for child in verb_tok.children:
        # Skip complementizers (e.g., "che" in Italian is both comp and wh-word)
        if child.dep_ == "mark":
            continue
        lemma = child.lemma_.lower()
        if lemma in wh_lemmas:
            return True
        pron_types = child.morph.get("PronType")
        if pron_types and "Int" in pron_types:
            return True
    return False


class ThatTraceAnnotator(BaseSentenceAnnotator):
    """
    Annotates that-trace environments in bridge verb complements.

    Extracts:
    - Bridge verb complements (ccomp under think/believe/say etc.)
    - Complementizer presence (that)
    - Embedded subject status (lexical, pronominal, none)
    - Extraction type (subject, object, none) based on matrix wh-word
    - Violation flag (comp_present AND subject extraction)
    """

    output_fields = {
        "bridge_complements": "List of bridge complement annotation dicts",
    }

    def __init__(self, language: str = "en"):
        self.language = language
        self._bridge_verbs = ITALIAN_BRIDGE_VERBS if language == "it" else ENGLISH_BRIDGE_VERBS
        self._wh_lemmas = WH_LEMMAS_IT if language == "it" else WH_LEMMAS_EN
        self._complementizer = "che" if language == "it" else "that"

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract that-trace environment annotations."""
        bridge_complements = []

        for tok in sent:
            # Step 1: Find ccomp under bridge verb
            if tok.dep_ != "ccomp":
                continue

            matrix_verb = tok.head
            if matrix_verb.lemma_.lower() not in self._bridge_verbs:
                continue

            # Filter interrogative complements (wh-word in embedded clause)
            if _has_embedded_wh(tok, self._wh_lemmas):
                continue

            # Only finite complements
            verb_forms = tok.morph.get("VerbForm")
            if verb_forms and "Fin" not in verb_forms:
                continue

            # Step 2: Complementizer presence
            comp_present = any(
                c.dep_ == "mark" and c.lemma_.lower() == self._complementizer
                for c in tok.children
            )

            # Step 3: Embedded subject status
            nsubj_child = None
            for child in tok.children:
                if normalize_dep(child.dep_) in ("nsubj", "nsubj:pass"):
                    nsubj_child = child
                    break

            if nsubj_child is not None:
                if nsubj_child.pos_ == "PRON":
                    subj_status = "pronominal"
                else:
                    subj_status = "lexical"
            else:
                subj_status = "none"

            # Step 4: Extraction heuristic (wh-word in matrix clause)
            matrix_has_wh = False
            for sibling in matrix_verb.children:
                sib_lemma = sibling.lemma_.lower()
                sib_pron = sibling.morph.get("PronType")
                if sib_lemma in self._wh_lemmas or (sib_pron and "Int" in sib_pron):
                    matrix_has_wh = True
                    break

            if matrix_has_wh and nsubj_child is None:
                extraction = "subject"
            elif matrix_has_wh and nsubj_child is not None:
                extraction = "object"
            else:
                extraction = "none"

            # Step 5: Violation detection
            is_violation = comp_present and extraction == "subject"

            complement = {
                "matrix_verb_lemma": matrix_verb.lemma_.lower(),
                "matrix_verb_idx": matrix_verb.i - sent.start,
                "comp_present": comp_present,
                "extraction_type": extraction,
                "embedded_subject_status": subj_status,
                "is_violation": is_violation,
            }
            bridge_complements.append(complement)

        return {"bridge_complements": bridge_complements}

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Compute that-trace related flags."""
        complements = annotations.get("bridge_complements", [])

        has_context = len(complements) > 0
        has_violation = any(c["is_violation"] for c in complements)

        return {
            "has_that_trace_context": has_context,
            "has_that_trace_violation": has_violation,
        }
