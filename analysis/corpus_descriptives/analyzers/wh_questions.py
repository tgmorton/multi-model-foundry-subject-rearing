"""
§1.5 Wh-Questions analyzer.

Identifies wh-words by PronType=Int or lemma membership, classifies extraction
type (subject/object/adjunct), clause level (root/embedded), and complementizer
presence in embedded wh-clauses.
"""

import ast
from collections import Counter
from typing import Any, Dict, Optional

import spacy

from ..constants import WH_LEMMAS_EN
from .base import BaseAnalyzer


def _is_wh_word(tok: spacy.tokens.Token) -> bool:
    """Check if token is a wh-word."""
    pron_types = tok.morph.get("PronType")
    if pron_types and "Int" in pron_types:
        return True
    return tok.lemma_.lower() in WH_LEMMAS_EN


def _extraction_type(tok: spacy.tokens.Token) -> str:
    """Classify wh-word extraction: subject / object / adjunct."""
    dep = tok.dep_
    if dep in ("nsubj", "nsubj:pass"):
        return "subject"
    elif dep in ("obj", "iobj"):
        return "object"
    else:
        return "adjunct"


class WhQuestionAnalyzer(BaseAnalyzer):
    """Wh-question extraction type, clause level, and complementizer presence."""

    def __init__(self):
        super().__init__()
        # genre -> Counter keyed by (extraction_type, clause_level)
        self._counts: Dict[str, Counter] = {}
        # genre -> Counter keyed by complementizer_present (True/False)
        self._comp_counts: Dict[str, Counter] = {}

    def process_doc(
        self,
        doc: spacy.tokens.Doc,
        genre: str,
        speaker: Optional[str] = None,
    ) -> None:
        if genre not in self._counts:
            self._counts[genre] = Counter()
            self._comp_counts[genre] = Counter()

        for tok in doc:
            if not _is_wh_word(tok):
                continue

            ext_type = _extraction_type(tok)

            # Clause level: root if wh-word's head is ROOT, else embedded
            head = tok.head
            if head.dep_ == "ROOT" or tok.dep_ == "ROOT":
                clause_level = "root"
            else:
                clause_level = "embedded"

                # Check for complementizer in embedded wh-clause
                # The head verb of the wh-word (or its ancestor ccomp)
                # may have a 'mark' child with lemma "that"
                has_comp = any(
                    c.dep_ == "mark" and c.lemma_.lower() == "that"
                    for c in head.children
                )
                self._comp_counts[genre][has_comp] += 1

            self._counts[genre][(ext_type, clause_level)] += 1

    def get_results(self) -> Dict[str, Any]:
        by_genre = {}
        for genre in set(self._counts) | set(self._comp_counts):
            counts = self._counts.get(genre, Counter())
            comp = self._comp_counts.get(genre, Counter())
            by_genre[genre] = {
                "extraction_counts": [
                    {"extraction_type": et, "clause_level": cl, "count": c}
                    for (et, cl), c in counts.most_common()
                ],
                "embedded_complementizer": {
                    "with_that": comp.get(True, 0),
                    "without_that": comp.get(False, 0),
                },
            }

        overall_counts: Counter = Counter()
        overall_comp: Counter = Counter()
        for c in self._counts.values():
            overall_counts += c
        for c in self._comp_counts.values():
            overall_comp += c

        return {
            "overall": {
                "extraction_counts": [
                    {"extraction_type": et, "clause_level": cl, "count": c}
                    for (et, cl), c in overall_counts.most_common()
                ],
                "embedded_complementizer": {
                    "with_that": overall_comp.get(True, 0),
                    "without_that": overall_comp.get(False, 0),
                },
            },
            "by_genre": by_genre,
        }

    def merge(self, other: "WhQuestionAnalyzer") -> None:
        super().merge(other)
        self._merge_counter_dicts(self._counts, other._counts)
        self._merge_counter_dicts(self._comp_counts, other._comp_counts)

    def to_checkpoint(self) -> Dict[str, Any]:
        data = super().to_checkpoint()
        data["counts"] = {g: self._counter_to_json(c) for g, c in self._counts.items()}
        data["comp_counts"] = {g: self._counter_to_json(c) for g, c in self._comp_counts.items()}
        return data

    def from_checkpoint(self, data: Dict[str, Any]) -> None:
        super().from_checkpoint(data)
        self._counts = {
            g: Counter({ast.literal_eval(k) if isinstance(k, str) else k: v for k, v in c.items()})
            for g, c in data.get("counts", {}).items()
        }
        self._comp_counts = {
            g: Counter({ast.literal_eval(k) if isinstance(k, str) else k: v for k, v in c.items()})
            for g, c in data.get("comp_counts", {}).items()
        }
