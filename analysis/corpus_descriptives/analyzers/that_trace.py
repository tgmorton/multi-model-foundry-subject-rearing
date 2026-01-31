"""
§1.8 That-Trace Environments analyzer.

Five-step analysis:
1. Find ccomp under bridge verbs
2. Classify complementizer presence (mark + lemma "that")
3. Embedded subject status (overt lexical / overt pronominal / none)
4. Extraction heuristic (subject / object / none)
5. Cross-tabulate comp × extraction
Filters out interrogative complements.
"""

import ast
from collections import Counter
from typing import Any, Dict, Optional

import spacy

from ..constants import ENGLISH_BRIDGE_VERBS, WH_LEMMAS_EN
from .base import BaseAnalyzer


def _has_embedded_wh(verb_tok: spacy.tokens.Token) -> bool:
    """Check if the ccomp verb has a wh-word as its own dependent."""
    for child in verb_tok.children:
        lemma = child.lemma_.lower()
        if lemma in WH_LEMMAS_EN:
            return True
        pron_types = child.morph.get("PronType")
        if pron_types and "Int" in pron_types:
            return True
    return False


class ThatTraceAnalyzer(BaseAnalyzer):
    """That-trace environment characterization."""

    def __init__(self):
        super().__init__()
        # genre -> Counter keyed by bridge_verb_lemma
        self._bridge_verb_counts: Dict[str, Counter] = {}
        # genre -> Counter keyed by (comp_present, extraction_type)
        self._crosstab: Dict[str, Counter] = {}
        # genre -> Counter keyed by (comp_present, subject_status)
        self._comp_subject: Dict[str, Counter] = {}
        # genre -> Counter keyed by comp_present (True/False)
        self._comp_rate: Dict[str, Counter] = {}

    def process_doc(
        self,
        doc: spacy.tokens.Doc,
        genre: str,
        speaker: Optional[str] = None,
    ) -> None:
        if genre not in self._bridge_verb_counts:
            self._bridge_verb_counts[genre] = Counter()
            self._crosstab[genre] = Counter()
            self._comp_subject[genre] = Counter()
            self._comp_rate[genre] = Counter()

        for tok in doc:
            # Step 1: ccomp under a bridge verb
            if tok.dep_ != "ccomp":
                continue
            matrix_verb = tok.head
            if matrix_verb.lemma_.lower() not in ENGLISH_BRIDGE_VERBS:
                continue

            # Filter interrogative complements (wh-word is dependent of
            # embedded verb, not matrix verb)
            if _has_embedded_wh(tok):
                continue

            verb_forms = tok.morph.get("VerbForm")
            if verb_forms and "Fin" not in verb_forms:
                continue  # only finite complements

            self._bridge_verb_counts[genre][matrix_verb.lemma_.lower()] += 1

            # Step 2: Complementizer presence
            comp_present = any(
                c.dep_ == "mark" and c.lemma_.lower() == "that"
                for c in tok.children
            )
            self._comp_rate[genre][comp_present] += 1

            # Step 3: Embedded subject status
            nsubj_child = None
            for child in tok.children:
                if child.dep_ in ("nsubj", "nsubj:pass"):
                    nsubj_child = child
                    break

            if nsubj_child is not None:
                if nsubj_child.pos_ == "PRON":
                    subj_status = "pronominal"
                else:
                    subj_status = "lexical"
            else:
                subj_status = "none"

            self._comp_subject[genre][(comp_present, subj_status)] += 1

            # Step 4: Extraction heuristic
            # Check if matrix clause has a wh-word
            matrix_has_wh = False
            wh_is_matrix_subj = False
            for sibling in matrix_verb.children:
                sib_lemma = sibling.lemma_.lower()
                sib_pron = sibling.morph.get("PronType")
                if sib_lemma in WH_LEMMAS_EN or (sib_pron and "Int" in sib_pron):
                    matrix_has_wh = True
                    if sibling.dep_ in ("nsubj", "nsubj:pass") or sibling.dep_ == "ROOT":
                        wh_is_matrix_subj = True

            if matrix_has_wh and nsubj_child is None:
                extraction = "subject"
            elif matrix_has_wh and nsubj_child is not None:
                extraction = "object"
            else:
                extraction = "none"

            # Step 5: Cross-tabulate
            self._crosstab[genre][(comp_present, extraction)] += 1

    def get_results(self) -> Dict[str, Any]:
        by_genre = {}
        all_genres = (
            set(self._bridge_verb_counts)
            | set(self._crosstab)
            | set(self._comp_rate)
        )
        for genre in all_genres:
            bv = self._bridge_verb_counts.get(genre, Counter())
            ct = self._crosstab.get(genre, Counter())
            cr = self._comp_rate.get(genre, Counter())
            cs = self._comp_subject.get(genre, Counter())
            by_genre[genre] = {
                "bridge_verb_counts": [
                    {"verb_lemma": vl, "count": c}
                    for vl, c in bv.most_common()
                ],
                "complementizer_rate": {
                    "with_that": cr.get(True, 0),
                    "without_that": cr.get(False, 0),
                },
                "cross_tab": [
                    {
                        "comp_present": cp,
                        "extraction_type": et,
                        "count": c,
                    }
                    for (cp, et), c in ct.most_common()
                ],
                "comp_subject_cross_tab": [
                    {
                        "comp_present": cp,
                        "subject_status": ss,
                        "count": c,
                    }
                    for (cp, ss), c in cs.most_common()
                ],
            }

        # Overall aggregation
        overall_bv: Counter = Counter()
        overall_ct: Counter = Counter()
        overall_cr: Counter = Counter()
        overall_cs: Counter = Counter()
        for g in all_genres:
            overall_bv += self._bridge_verb_counts.get(g, Counter())
            overall_ct += self._crosstab.get(g, Counter())
            overall_cr += self._comp_rate.get(g, Counter())
            overall_cs += self._comp_subject.get(g, Counter())

        return {
            "overall": {
                "bridge_verb_counts": [
                    {"verb_lemma": vl, "count": c}
                    for vl, c in overall_bv.most_common()
                ],
                "complementizer_rate": {
                    "with_that": overall_cr.get(True, 0),
                    "without_that": overall_cr.get(False, 0),
                },
                "cross_tab": [
                    {
                        "comp_present": cp,
                        "extraction_type": et,
                        "count": c,
                    }
                    for (cp, et), c in overall_ct.most_common()
                ],
                "comp_subject_cross_tab": [
                    {
                        "comp_present": cp,
                        "subject_status": ss,
                        "count": c,
                    }
                    for (cp, ss), c in overall_cs.most_common()
                ],
            },
            "by_genre": by_genre,
        }

    def merge(self, other: "ThatTraceAnalyzer") -> None:
        super().merge(other)
        self._merge_counter_dicts(self._bridge_verb_counts, other._bridge_verb_counts)
        self._merge_counter_dicts(self._crosstab, other._crosstab)
        self._merge_counter_dicts(self._comp_subject, other._comp_subject)
        self._merge_counter_dicts(self._comp_rate, other._comp_rate)

    def to_checkpoint(self) -> Dict[str, Any]:
        data = super().to_checkpoint()
        data["bridge_verb_counts"] = {
            g: self._counter_to_json(c) for g, c in self._bridge_verb_counts.items()
        }
        data["crosstab"] = {g: self._counter_to_json(c) for g, c in self._crosstab.items()}
        data["comp_subject"] = {g: self._counter_to_json(c) for g, c in self._comp_subject.items()}
        data["comp_rate"] = {g: self._counter_to_json(c) for g, c in self._comp_rate.items()}
        return data

    def from_checkpoint(self, data: Dict[str, Any]) -> None:
        super().from_checkpoint(data)
        self._bridge_verb_counts = {
            g: Counter(c) for g, c in data.get("bridge_verb_counts", {}).items()
        }
        self._crosstab = {
            g: Counter({ast.literal_eval(k) if isinstance(k, str) else k: v for k, v in c.items()})
            for g, c in data.get("crosstab", {}).items()
        }
        self._comp_subject = {
            g: Counter({ast.literal_eval(k) if isinstance(k, str) else k: v for k, v in c.items()})
            for g, c in data.get("comp_subject", {}).items()
        }
        self._comp_rate = {
            g: Counter({ast.literal_eval(k) if isinstance(k, str) else k: v for k, v in c.items()})
            for g, c in data.get("comp_rate", {}).items()
        }
