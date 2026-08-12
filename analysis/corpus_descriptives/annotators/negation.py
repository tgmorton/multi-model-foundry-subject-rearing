"""
Negation annotator.

Identifies negation markers and analyzes their position relative
to the head verb and subject realization.
"""

from typing import Any, Dict, List, Optional

import spacy

from preprocessing.dep_labels import normalize_dep

from .base import BaseSentenceAnnotator


class NegationAnnotator(BaseSentenceAnnotator):
    """
    Annotates negation markers in a sentence.

    For each negation (Polarity=Neg), extracts:
    - Token position
    - Lemma (not, never, no, n't, etc.)
    - Position relative to head verb (pre, post, other)
    - Subject realization status of the head verb
    """

    output_fields = {
        "negations": "List of negation annotation dicts",
    }

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract negation annotations."""
        negations = []

        for tok in sent:
            polarity = tok.morph.get("Polarity")
            if not polarity or "Neg" not in polarity:
                continue

            head = tok.head

            # Position relative to head verb
            if head.pos_ in ("VERB", "AUX"):
                position = "pre" if tok.i < head.i else "post"
            else:
                position = "other"

            # Subject realization of the head verb
            if head.pos_ in ("VERB", "AUX"):
                head_children_deps = {normalize_dep(c.dep_) for c in head.children}
                if head_children_deps & {"nsubj", "nsubj:pass"}:
                    subj_status = "overt"
                elif "expl" in head_children_deps:
                    subj_status = "expletive"
                else:
                    subj_status = "none"
            else:
                subj_status = "n/a"

            negation = {
                "token_idx": tok.i - sent.start,
                "lemma": tok.lemma_.lower(),
                "position": position,
                "subject_status": subj_status,
            }
            negations.append(negation)

        return {"negations": negations}

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Compute negation-related flags."""
        negations = annotations.get("negations", [])
        return {"has_negation": len(negations) > 0}
