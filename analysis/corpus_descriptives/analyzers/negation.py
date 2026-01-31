"""
§1.7 Negation analyzer.

Identifies tokens with Polarity=Neg, classifies position relative to
head verb (pre/post), and cross-tabulates with subject realization.
"""

import ast
from collections import Counter
from typing import Any, Dict, Optional

import spacy

from .base import BaseAnalyzer


class NegationAnalyzer(BaseAnalyzer):
    """Negation frequency, position, and subject realization cross-tab."""

    def __init__(self):
        super().__init__()
        # genre -> Counter keyed by position ("pre" / "post")
        self._position_counts: Dict[str, Counter] = {}
        # genre -> Counter keyed by (position, subject_status)
        self._crosstab: Dict[str, Counter] = {}

    def process_doc(
        self,
        doc: spacy.tokens.Doc,
        genre: str,
        speaker: Optional[str] = None,
    ) -> None:
        if genre not in self._position_counts:
            self._position_counts[genre] = Counter()
            self._crosstab[genre] = Counter()

        for tok in doc:
            polarity = tok.morph.get("Polarity")
            if not polarity or "Neg" not in polarity:
                continue

            head = tok.head
            # Position relative to head verb
            if head.pos_ in ("VERB", "AUX"):
                position = "pre" if tok.i < head.i else "post"
            else:
                position = "other"

            self._position_counts[genre][position] += 1

            # Subject realization of the head verb
            if head.pos_ in ("VERB", "AUX"):
                head_children_deps = {c.dep_ for c in head.children}
                if head_children_deps & {"nsubj", "nsubj:pass"}:
                    subj_status = "overt"
                elif "expl" in head_children_deps:
                    subj_status = "expletive"
                else:
                    subj_status = "none"
            else:
                subj_status = "n/a"

            self._crosstab[genre][(position, subj_status)] += 1

    def get_results(self) -> Dict[str, Any]:
        by_genre = {}
        for genre in set(self._position_counts) | set(self._crosstab):
            pos = self._position_counts.get(genre, Counter())
            ct = self._crosstab.get(genre, Counter())
            by_genre[genre] = {
                "position_counts": self._sum_counters(pos),
                "total_negations": sum(pos.values()),
                "cross_tab": [
                    {"position": p, "subject_status": ss, "count": c}
                    for (p, ss), c in ct.most_common()
                ],
            }

        overall_pos: Counter = Counter()
        overall_ct: Counter = Counter()
        for c in self._position_counts.values():
            overall_pos += c
        for c in self._crosstab.values():
            overall_ct += c

        return {
            "overall": {
                "position_counts": self._sum_counters(overall_pos),
                "total_negations": sum(overall_pos.values()),
                "cross_tab": [
                    {"position": p, "subject_status": ss, "count": c}
                    for (p, ss), c in overall_ct.most_common()
                ],
            },
            "by_genre": by_genre,
        }

    def merge(self, other: "NegationAnalyzer") -> None:
        super().merge(other)
        self._merge_counter_dicts(self._position_counts, other._position_counts)
        self._merge_counter_dicts(self._crosstab, other._crosstab)

    def to_checkpoint(self) -> Dict[str, Any]:
        data = super().to_checkpoint()
        data["position_counts"] = {g: self._counter_to_json(c) for g, c in self._position_counts.items()}
        data["crosstab"] = {g: self._counter_to_json(c) for g, c in self._crosstab.items()}
        return data

    def from_checkpoint(self, data: Dict[str, Any]) -> None:
        super().from_checkpoint(data)
        self._position_counts = {
            g: Counter(c) for g, c in data.get("position_counts", {}).items()
        }
        self._crosstab = {
            g: Counter({ast.literal_eval(k) if isinstance(k, str) else k: v for k, v in c.items()})
            for g, c in data.get("crosstab", {}).items()
        }
