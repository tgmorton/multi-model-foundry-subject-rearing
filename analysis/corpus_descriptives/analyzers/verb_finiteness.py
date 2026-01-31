"""
§1.3 Verb Finiteness analyzer.

Cross-tabulates VerbForm (Fin/Inf) with clause position
(ROOT/xcomp/ccomp/advcl/acl/other).
"""

import ast
from collections import Counter
from typing import Any, Dict, Optional

import spacy

from .base import BaseAnalyzer

_CLAUSE_DEPS = {"ROOT", "xcomp", "ccomp", "advcl", "acl", "acl:relcl"}


class VerbFinitenessAnalyzer(BaseAnalyzer):
    """Cross-tab of VerbForm × clause position."""

    def __init__(self):
        super().__init__()
        # genre -> Counter keyed by (verb_form, clause_position)
        self._counts: Dict[str, Counter] = {}

    def process_doc(
        self,
        doc: spacy.tokens.Doc,
        genre: str,
        speaker: Optional[str] = None,
    ) -> None:
        if genre not in self._counts:
            self._counts[genre] = Counter()

        for tok in doc:
            if tok.pos_ not in ("VERB", "AUX"):
                continue
            verb_forms = tok.morph.get("VerbForm")
            if not verb_forms:
                continue
            verb_form = verb_forms[0]  # e.g. "Fin" or "Inf"

            dep = tok.dep_
            if dep in _CLAUSE_DEPS:
                clause_pos = dep
            else:
                clause_pos = "other"

            self._counts[genre][(verb_form, clause_pos)] += 1

    def get_results(self) -> Dict[str, Any]:
        by_genre = {}
        for genre, counter in self._counts.items():
            by_genre[genre] = {
                "cross_tab": [
                    {"verb_form": vf, "clause_position": cp, "count": c}
                    for (vf, cp), c in counter.most_common()
                ]
            }

        overall: Counter = Counter()
        for counter in self._counts.values():
            overall += counter

        return {
            "overall": {
                "cross_tab": [
                    {"verb_form": vf, "clause_position": cp, "count": c}
                    for (vf, cp), c in overall.most_common()
                ]
            },
            "by_genre": by_genre,
        }

    def merge(self, other: "VerbFinitenessAnalyzer") -> None:
        super().merge(other)
        self._merge_counter_dicts(self._counts, other._counts)

    def to_checkpoint(self) -> Dict[str, Any]:
        data = super().to_checkpoint()
        data["counts"] = {g: self._counter_to_json(c) for g, c in self._counts.items()}
        return data

    def from_checkpoint(self, data: Dict[str, Any]) -> None:
        super().from_checkpoint(data)
        self._counts = {
            g: Counter({ast.literal_eval(k) if isinstance(k, str) else k: v for k, v in c.items()})
            for g, c in data.get("counts", {}).items()
        }
