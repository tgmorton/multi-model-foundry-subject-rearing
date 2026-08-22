"""
Relative clause annotator.

Identifies relative clauses and classifies them as subject or object relatives.
"""

from typing import Any, Dict, List, Optional

import spacy

from preprocessing.dep_labels import normalize_dep

from ..constants import RELATIVIZERS_EN, RELATIVIZERS_IT
from .base import BaseSentenceAnnotator


def _is_relcl(tok: spacy.tokens.Token) -> bool:
    """Check if token heads a relative clause."""
    return normalize_dep(tok.dep_) in ("acl:relcl", "acl") and tok.pos_ in ("VERB", "AUX")


class RelativeClauseAnnotator(BaseSentenceAnnotator):
    """
    Annotates relative clauses in a sentence.

    For each relative clause, extracts:
    - Verb position and type classification
    - Subject vs object relative
    - Relativizer information (if present)
    """

    output_fields = {
        "relative_clauses": "List of relative clause annotation dicts",
    }

    def __init__(self, language: str = "en"):
        self.language = language
        self._relativizers = RELATIVIZERS_IT if language == "it" else RELATIVIZERS_EN

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract relative clause annotations."""
        relative_clauses = []

        for tok in sent:
            if not _is_relcl(tok):
                continue

            children = list(tok.children)

            # Classify as subject or object relative
            has_rel_nsubj = False
            has_rel_obj = False
            has_nsubj = False
            relativizer_idx = None
            relativizer_lemma = None

            for child in children:
                pron_type = child.morph.get("PronType")
                is_relative = pron_type and "Rel" in pron_type
                is_rel_lemma = child.lemma_.lower() in self._relativizers

                if normalize_dep(child.dep_) in ("nsubj", "nsubj:pass"):
                    has_nsubj = True
                    if is_relative or (child.pos_ == "PRON" and is_rel_lemma):
                        has_rel_nsubj = True
                        relativizer_idx = child.i - sent.start
                        relativizer_lemma = child.lemma_.lower()

                if normalize_dep(child.dep_) in ("obj", "iobj"):
                    if child.pos_ == "PRON" and (is_relative or is_rel_lemma):
                        has_rel_obj = True
                        relativizer_idx = child.i - sent.start
                        relativizer_lemma = child.lemma_.lower()

            # Determine relative type
            if has_rel_nsubj:
                rel_type = "subject"
            elif has_rel_obj:
                rel_type = "object"
            elif has_nsubj:
                # nsubj present but not relativizer → object relative (gap in obj)
                rel_type = "object"
            else:
                # No nsubj → subject relative (gap in subject position)
                rel_type = "subject"

            relcl = {
                "verb_idx": tok.i - sent.start,
                "rel_type": rel_type,
                "relativizer_idx": relativizer_idx,
                "relativizer_lemma": relativizer_lemma,
            }
            relative_clauses.append(relcl)

        return {"relative_clauses": relative_clauses}

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Compute relative clause flags."""
        relcls = annotations.get("relative_clauses", [])
        return {"has_relative_clause": len(relcls) > 0}
