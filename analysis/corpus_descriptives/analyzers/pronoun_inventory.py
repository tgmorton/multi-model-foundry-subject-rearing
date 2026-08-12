"""
§1.1 Overt Pronoun Inventory analyzer.

Counts pronouns by function (subject/object), person/number, case, and lemma.
Tracks proportions relative to all tokens and all pronouns.
"""

import ast
from collections import Counter
from typing import Any, Dict, Optional

import spacy

from preprocessing.dep_labels import normalize_dep

from ..constants import SUBJECT_DEPS
from .base import BaseAnalyzer


class PronounInventoryAnalyzer(BaseAnalyzer):
    """Accumulates pronoun frequency tables by person/number/case/function."""

    def __init__(self):
        super().__init__()
        # genre -> Counter keyed by (function, person, number, case, lemma)
        self._counts: Dict[str, Counter] = {}
        self._pronoun_totals: Counter = Counter()  # genre -> total pronouns
        self._token_totals: Counter = Counter()  # genre -> total tokens

    def process_doc(
        self,
        doc: spacy.tokens.Doc,
        genre: str,
        speaker: Optional[str] = None,
    ) -> None:
        if genre not in self._counts:
            self._counts[genre] = Counter()

        self._token_totals[genre] += len(doc)

        for tok in doc:
            if tok.pos_ != "PRON":
                continue

            dep = normalize_dep(tok.dep_)
            # Classify function
            if dep in SUBJECT_DEPS:
                function = "subject"
            elif dep in ("obj", "iobj"):
                function = "object"
            else:
                continue  # skip pronouns not in argumental position

            morph = tok.morph
            person = morph.get("Person", [""])[0] if morph.get("Person") else ""
            number = morph.get("Number", [""])[0] if morph.get("Number") else ""
            case = morph.get("Case", [""])[0] if morph.get("Case") else ""
            lemma = tok.lemma_.lower()

            self._counts[genre][(function, person, number, case, lemma)] += 1
            self._pronoun_totals[genre] += 1

    def get_results(self) -> Dict[str, Any]:
        by_genre = {}
        for genre, counter in self._counts.items():
            rows = []
            total_pron = self._pronoun_totals[genre]
            total_tok = self._token_totals[genre]
            for (function, person, number, case, lemma), count in counter.most_common():
                rows.append({
                    "function": function,
                    "person": person,
                    "number": number,
                    "case": case,
                    "lemma": lemma,
                    "count": count,
                    "prop_of_pronouns": count / total_pron if total_pron else 0.0,
                    "prop_of_tokens": count / total_tok if total_tok else 0.0,
                })
            by_genre[genre] = {
                "entries": rows,
                "total_pronouns": total_pron,
                "total_tokens": total_tok,
            }

        # Aggregate overall
        overall_counts: Counter = Counter()
        overall_pron = 0
        overall_tok = 0
        for genre, counter in self._counts.items():
            overall_counts += counter
            overall_pron += self._pronoun_totals[genre]
            overall_tok += self._token_totals[genre]

        overall_rows = []
        for (function, person, number, case, lemma), count in overall_counts.most_common():
            overall_rows.append({
                "function": function,
                "person": person,
                "number": number,
                "case": case,
                "lemma": lemma,
                "count": count,
                "prop_of_pronouns": count / overall_pron if overall_pron else 0.0,
                "prop_of_tokens": count / overall_tok if overall_tok else 0.0,
            })

        return {
            "overall": {
                "entries": overall_rows,
                "total_pronouns": overall_pron,
                "total_tokens": overall_tok,
            },
            "by_genre": by_genre,
        }

    def merge(self, other: "PronounInventoryAnalyzer") -> None:
        super().merge(other)
        self._merge_counter_dicts(self._counts, other._counts)
        self._pronoun_totals += other._pronoun_totals
        self._token_totals += other._token_totals

    def to_checkpoint(self) -> Dict[str, Any]:
        data = super().to_checkpoint()
        data["counts"] = {g: self._counter_to_json(c) for g, c in self._counts.items()}
        data["pronoun_totals"] = dict(self._pronoun_totals)
        data["token_totals"] = dict(self._token_totals)
        return data

    def from_checkpoint(self, data: Dict[str, Any]) -> None:
        super().from_checkpoint(data)
        self._counts = {
            g: Counter({ast.literal_eval(k) if isinstance(k, str) else k: v for k, v in c.items()})
            for g, c in data.get("counts", {}).items()
        }
        self._pronoun_totals = Counter(data.get("pronoun_totals", {}))
        self._token_totals = Counter(data.get("token_totals", {}))
