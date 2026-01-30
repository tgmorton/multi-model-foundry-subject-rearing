"""
§1.2 Expletive analyzer.

Identifies expletive subjects (dep=expl) and classifies them as
weather, existential, or raising based on the head verb.
"""

import ast
from collections import Counter
from typing import Any, Dict, Optional

import spacy

from ..constants import WEATHER_VERBS
from .base import BaseAnalyzer


class ExpletiveAnalyzer(BaseAnalyzer):
    """Counts expletive subjects by class and verb lemma."""

    def __init__(self):
        super().__init__()
        # genre -> Counter keyed by expletive_class
        self._class_counts: Dict[str, Counter] = {}
        # genre -> Counter keyed by (expletive_class, verb_lemma)
        self._verb_counts: Dict[str, Counter] = {}

    def process_doc(
        self,
        doc: spacy.tokens.Doc,
        genre: str,
        speaker: Optional[str] = None,
    ) -> None:
        if genre not in self._class_counts:
            self._class_counts[genre] = Counter()
            self._verb_counts[genre] = Counter()

        for tok in doc:
            if tok.dep_ != "expl":
                continue

            head_lemma = tok.head.lemma_.lower()
            lemma = tok.lemma_.lower()

            # Classify
            if head_lemma in WEATHER_VERBS:
                expl_class = "weather"
            elif lemma == "there":
                expl_class = "existential"
            else:
                expl_class = "raising"

            self._class_counts[genre][expl_class] += 1
            self._verb_counts[genre][(expl_class, head_lemma)] += 1

    def get_results(self) -> Dict[str, Any]:
        by_genre = {}
        for genre in self._class_counts:
            by_genre[genre] = {
                "class_counts": self._sum_counters(self._class_counts[genre]),
                "verb_counts": [
                    {"class": cls, "verb_lemma": vl, "count": c}
                    for (cls, vl), c in self._verb_counts[genre].most_common()
                ],
            }

        # Overall
        overall_class: Counter = Counter()
        overall_verb: Counter = Counter()
        for genre in self._class_counts:
            overall_class += self._class_counts[genre]
            overall_verb += self._verb_counts[genre]

        return {
            "overall": {
                "class_counts": self._sum_counters(overall_class),
                "verb_counts": [
                    {"class": cls, "verb_lemma": vl, "count": c}
                    for (cls, vl), c in overall_verb.most_common()
                ],
            },
            "by_genre": by_genre,
        }

    def merge(self, other: "ExpletiveAnalyzer") -> None:
        super().merge(other)
        self._merge_counter_dicts(self._class_counts, other._class_counts)
        self._merge_counter_dicts(self._verb_counts, other._verb_counts)

    def to_checkpoint(self) -> Dict[str, Any]:
        data = super().to_checkpoint()
        data["class_counts"] = {g: dict(c) for g, c in self._class_counts.items()}
        data["verb_counts"] = {g: dict(c) for g, c in self._verb_counts.items()}
        return data

    def from_checkpoint(self, data: Dict[str, Any]) -> None:
        super().from_checkpoint(data)
        self._class_counts = {
            g: Counter(c) for g, c in data.get("class_counts", {}).items()
        }
        self._verb_counts = {
            g: Counter({ast.literal_eval(k) if isinstance(k, str) else k: v for k, v in c.items()})
            for g, c in data.get("verb_counts", {}).items()
        }
