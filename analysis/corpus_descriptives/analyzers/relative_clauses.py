"""
§1.6 Relative Clauses analyzer.

Identifies relative clauses via acl:relcl (or acl) dependency on the verb,
then classifies the relativizer as subject or object relative.
"""

from collections import Counter
from typing import Any, Dict, Optional

import spacy

from .base import BaseAnalyzer


def _is_relcl(tok: spacy.tokens.Token) -> bool:
    """Check if token heads a relative clause."""
    return tok.dep_ in ("acl:relcl", "acl") and tok.pos_ in ("VERB", "AUX")


class RelativeClauseAnalyzer(BaseAnalyzer):
    """Subject vs object relative clause counts."""

    def __init__(self):
        super().__init__()
        # genre -> Counter keyed by rel_type ("subject" / "object" / "other")
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
            if not _is_relcl(tok):
                continue

            # Look at the children of the relative clause verb
            # A subject relative has the relativizer as nsubj
            # An object relative has the relativizer as obj (or nsubj is something else)
            children = list(tok.children)
            child_deps = {c.dep_ for c in children}

            # Classify as subject or object relative.
            # Subject relative: relativizer fills nsubj slot (who/which/that as nsubj)
            # Object relative: relativizer fills obj slot, or nsubj is a
            #   non-relative pronoun/noun while obj is missing or is the relativizer.
            has_rel_nsubj = False
            has_rel_obj = False
            has_nsubj = False

            for child in children:
                if child.dep_ in ("nsubj", "nsubj:pass"):
                    has_nsubj = True
                    pron_type = child.morph.get("PronType")
                    if pron_type and "Rel" in pron_type:
                        has_rel_nsubj = True
                if child.dep_ in ("obj", "iobj"):
                    pron_type = child.morph.get("PronType")
                    if child.pos_ == "PRON" and (
                        (pron_type and "Rel" in pron_type)
                        or child.lemma_.lower() in ("who", "whom", "which", "that")
                    ):
                        has_rel_obj = True

            if has_rel_nsubj:
                rel_type = "subject"
            elif has_rel_obj:
                rel_type = "object"
            elif has_nsubj:
                # nsubj present but is not the relativizer → object relative (gap in obj)
                rel_type = "object"
            else:
                # No nsubj → subject relative (gap in subject position)
                rel_type = "subject"

            self._counts[genre][rel_type] += 1

    def get_results(self) -> Dict[str, Any]:
        by_genre = {}
        for genre, counter in self._counts.items():
            total = sum(counter.values())
            by_genre[genre] = {
                "counts": self._sum_counters(counter),
                "total": total,
                "proportions": {
                    k: v / total if total else 0.0 for k, v in counter.items()
                },
            }

        overall: Counter = Counter()
        for c in self._counts.values():
            overall += c
        total = sum(overall.values())

        return {
            "overall": {
                "counts": self._sum_counters(overall),
                "total": total,
                "proportions": {
                    k: v / total if total else 0.0 for k, v in overall.items()
                },
            },
            "by_genre": by_genre,
        }

    def merge(self, other: "RelativeClauseAnalyzer") -> None:
        super().merge(other)
        self._merge_counter_dicts(self._counts, other._counts)

    def to_checkpoint(self) -> Dict[str, Any]:
        data = super().to_checkpoint()
        data["counts"] = {g: dict(c) for g, c in self._counts.items()}
        return data

    def from_checkpoint(self, data: Dict[str, Any]) -> None:
        super().from_checkpoint(data)
        self._counts = {
            g: Counter(c) for g, c in data.get("counts", {}).items()
        }
