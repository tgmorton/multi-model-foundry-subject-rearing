"""
§1.4 Clause Structure analyzer (dependency-based).

For finite verbs at ROOT/ccomp/advcl/acl:relcl: subject realization
(overt / expletive / none).  For xcomp+Inf: matrix verb lemma tracking.
"""

import ast
from collections import Counter
from typing import Any, Dict, Optional

import spacy

from .base import BaseAnalyzer

_FINITE_CLAUSE_DEPS = {"ROOT", "ccomp", "advcl", "acl:relcl", "acl"}


class ClauseStructureAnalyzer(BaseAnalyzer):
    """Subject realization in finite clauses and control/raising in infinitivals."""

    def __init__(self):
        super().__init__()
        # genre -> Counter keyed by (clause_dep, subject_status)
        self._finite_counts: Dict[str, Counter] = {}
        # genre -> Counter keyed by matrix_verb_lemma
        self._xcomp_verbs: Dict[str, Counter] = {}

    def process_doc(
        self,
        doc: spacy.tokens.Doc,
        genre: str,
        speaker: Optional[str] = None,
    ) -> None:
        if genre not in self._finite_counts:
            self._finite_counts[genre] = Counter()
            self._xcomp_verbs[genre] = Counter()

        for tok in doc:
            if tok.pos_ not in ("VERB", "AUX"):
                continue

            dep = tok.dep_
            verb_forms = tok.morph.get("VerbForm")

            # Finite clauses: ROOT / ccomp / advcl / acl:relcl
            if dep in _FINITE_CLAUSE_DEPS and verb_forms and "Fin" in verb_forms:
                children_deps = {c.dep_ for c in tok.children}
                if "expl" in children_deps:
                    subject_status = "expletive"
                elif children_deps & {"nsubj", "nsubj:pass"}:
                    subject_status = "overt"
                else:
                    subject_status = "none"
                self._finite_counts[genre][(dep, subject_status)] += 1

            # Infinitival complements: xcomp + Inf
            if dep == "xcomp" and verb_forms and "Inf" in verb_forms:
                matrix_lemma = tok.head.lemma_.lower()
                self._xcomp_verbs[genre][matrix_lemma] += 1

    def get_results(self) -> Dict[str, Any]:
        by_genre = {}
        for genre in set(self._finite_counts) | set(self._xcomp_verbs):
            finite = self._finite_counts.get(genre, Counter())
            xcomp = self._xcomp_verbs.get(genre, Counter())
            by_genre[genre] = {
                "finite_clauses": [
                    {"clause_dep": cd, "subject_status": ss, "count": c}
                    for (cd, ss), c in finite.most_common()
                ],
                "xcomp_matrix_verbs": [
                    {"verb_lemma": vl, "count": c}
                    for vl, c in xcomp.most_common()
                ],
            }

        overall_finite: Counter = Counter()
        overall_xcomp: Counter = Counter()
        for c in self._finite_counts.values():
            overall_finite += c
        for c in self._xcomp_verbs.values():
            overall_xcomp += c

        return {
            "overall": {
                "finite_clauses": [
                    {"clause_dep": cd, "subject_status": ss, "count": c}
                    for (cd, ss), c in overall_finite.most_common()
                ],
                "xcomp_matrix_verbs": [
                    {"verb_lemma": vl, "count": c}
                    for vl, c in overall_xcomp.most_common()
                ],
            },
            "by_genre": by_genre,
        }

    def merge(self, other: "ClauseStructureAnalyzer") -> None:
        super().merge(other)
        self._merge_counter_dicts(self._finite_counts, other._finite_counts)
        self._merge_counter_dicts(self._xcomp_verbs, other._xcomp_verbs)

    def to_checkpoint(self) -> Dict[str, Any]:
        data = super().to_checkpoint()
        data["finite_counts"] = {g: self._counter_to_json(c) for g, c in self._finite_counts.items()}
        data["xcomp_verbs"] = {g: self._counter_to_json(c) for g, c in self._xcomp_verbs.items()}
        return data

    def from_checkpoint(self, data: Dict[str, Any]) -> None:
        super().from_checkpoint(data)
        self._finite_counts = {
            g: Counter({ast.literal_eval(k) if isinstance(k, str) else k: v for k, v in c.items()})
            for g, c in data.get("finite_counts", {}).items()
        }
        self._xcomp_verbs = {
            g: Counter(c) for g, c in data.get("xcomp_verbs", {}).items()
        }
