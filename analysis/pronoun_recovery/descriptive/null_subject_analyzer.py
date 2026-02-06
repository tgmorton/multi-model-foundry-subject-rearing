"""Null-subject descriptive analyzer — BaseAnalyzer subclass."""

import ast
import logging
from collections import Counter
from typing import Any, Dict, Optional

import spacy

from analysis.corpus_descriptives.analyzers.base import BaseAnalyzer
from ..constants import LABEL_NONE, PRONOUN_LABELS, NULL_SUBJECT_CONTEXTS
from .context_classifier import classify_null_subject_context

logger = logging.getLogger(__name__)


class NullSubjectAnalyzer(BaseAnalyzer):
    """Analyze null subjects in corpus text using a trained pronoun recovery model.

    Unlike other BaseAnalyzer subclasses, this requires a model instance.
    It cannot be instantiated through the zero-arg ANALYZER_REGISTRY pattern;
    instead, instantiate it manually and add to the pipeline.

    Accumulates counts by:
    - null_subject_type: PRO.1sg ... PRO.3pl, IMP, CONJ
    - context: diary_drop, imperative, question_2p, topic_continuity, postverbal, other
    - genre
    - clause_position: ROOT, ccomp, advcl, etc. (dep label of verb's clause head)
    - Also counts overt subject pronouns (for null rate = null / (null + overt))
    """

    def __init__(self, model=None):
        """Initialize with optional PronounRecoveryModel.

        Args:
            model: PronounRecoveryModel instance. If None, analyzer will only
                   count overt subjects (useful for testing).
        """
        super().__init__()
        self.model = model

        # genre -> Counter keyed by null_subject_type
        self._null_type_counts: Dict[str, Counter] = {}
        # genre -> Counter keyed by context
        self._context_counts: Dict[str, Counter] = {}
        # genre -> Counter keyed by (null_subject_type, context)
        self._type_context_counts: Dict[str, Counter] = {}
        # genre -> Counter keyed by clause_position (dep label of verb)
        self._clause_position_counts: Dict[str, Counter] = {}
        # genre -> count of overt subject pronouns
        self._overt_subject_counts: Counter = Counter()
        # genre -> count of null subjects
        self._null_subject_counts: Counter = Counter()
        # genre -> total finite verbs
        self._finite_verb_counts: Counter = Counter()

    def process_doc(
        self,
        doc: spacy.tokens.Doc,
        genre: str,
        speaker: Optional[str] = None,
    ) -> None:
        """Process a single spaCy Doc: count null and overt subjects."""
        # Ensure genre keys exist
        for store in (self._null_type_counts, self._context_counts,
                      self._type_context_counts, self._clause_position_counts):
            if genre not in store:
                store[genre] = Counter()

        # Count overt subject pronouns
        prev_subject_features = None
        for tok in doc:
            if tok.pos_ == "PRON" and tok.dep_ == "nsubj":
                self._overt_subject_counts[genre] += 1
                # Track features for topic continuity detection
                person = tok.morph.get("Person", [""])[0]
                number = tok.morph.get("Number", [""])[0]
                if person and number:
                    suffix_map = {"Sing": "sg", "Plur": "pl"}
                    prev_subject_features = f"{person}{suffix_map.get(number, '')}"

        # Count finite verbs
        for tok in doc:
            verb_form = tok.morph.get("VerbForm")
            if verb_form and "Fin" in verb_form:
                self._finite_verb_counts[genre] += 1

        # Get model predictions (null subjects)
        if self.model is not None:
            predictions = self.model.predict_doc(doc)
            for pred in predictions:
                label = pred["label"]
                if label == LABEL_NONE:
                    continue

                token_idx = pred["token_idx"]
                verb_token = doc[token_idx]

                self._null_subject_counts[genre] += 1
                self._null_type_counts[genre][label] += 1

                # Classify context
                context = classify_null_subject_context(
                    label, verb_token, doc, prev_subject_features
                )
                self._context_counts[genre][context] += 1
                self._type_context_counts[genre][(label, context)] += 1

                # Clause position
                clause_dep = verb_token.dep_
                self._clause_position_counts[genre][clause_dep] += 1

    def get_results(self) -> Dict[str, Any]:
        """Return accumulated results with overall and by-genre breakdowns."""
        by_genre = {}
        for genre in set(
            list(self._null_type_counts.keys()) +
            list(self._overt_subject_counts.keys())
        ):
            null_count = self._null_subject_counts.get(genre, 0)
            overt_count = self._overt_subject_counts.get(genre, 0)
            total_subjects = null_count + overt_count
            finite_verbs = self._finite_verb_counts.get(genre, 0)

            by_genre[genre] = {
                "null_subject_counts": self._sum_counters(
                    self._null_type_counts.get(genre, Counter())
                ),
                "context_counts": self._sum_counters(
                    self._context_counts.get(genre, Counter())
                ),
                "type_context_counts": {
                    f"{t}|{c}": v
                    for (t, c), v in self._type_context_counts.get(genre, Counter()).items()
                },
                "clause_position_counts": self._sum_counters(
                    self._clause_position_counts.get(genre, Counter())
                ),
                "overt_subject_count": overt_count,
                "null_subject_count": null_count,
                "finite_verb_count": finite_verbs,
                "null_rate": null_count / total_subjects if total_subjects > 0 else 0.0,
            }

        # Aggregate overall
        total_null = sum(self._null_subject_counts.values())
        total_overt = sum(self._overt_subject_counts.values())
        total_finite = sum(self._finite_verb_counts.values())
        total_subjects = total_null + total_overt

        overall_null_types = Counter()
        overall_contexts = Counter()
        overall_type_context = Counter()
        overall_clause = Counter()
        for genre in self._null_type_counts:
            overall_null_types += self._null_type_counts[genre]
            overall_contexts += self._context_counts.get(genre, Counter())
            overall_type_context += self._type_context_counts.get(genre, Counter())
            overall_clause += self._clause_position_counts.get(genre, Counter())

        return {
            "overall": {
                "null_subject_counts": self._sum_counters(overall_null_types),
                "context_counts": self._sum_counters(overall_contexts),
                "type_context_counts": {
                    f"{t}|{c}": v for (t, c), v in overall_type_context.items()
                },
                "clause_position_counts": self._sum_counters(overall_clause),
                "overt_subject_count": total_overt,
                "null_subject_count": total_null,
                "finite_verb_count": total_finite,
                "null_rate": total_null / total_subjects if total_subjects > 0 else 0.0,
            },
            "by_genre": by_genre,
        }

    def merge(self, other: "NullSubjectAnalyzer") -> None:
        super().merge(other)
        self._merge_counter_dicts(self._null_type_counts, other._null_type_counts)
        self._merge_counter_dicts(self._context_counts, other._context_counts)
        self._merge_counter_dicts(self._type_context_counts, other._type_context_counts)
        self._merge_counter_dicts(self._clause_position_counts, other._clause_position_counts)
        self._overt_subject_counts += other._overt_subject_counts
        self._null_subject_counts += other._null_subject_counts
        self._finite_verb_counts += other._finite_verb_counts

    def to_checkpoint(self) -> Dict[str, Any]:
        data = super().to_checkpoint()
        data["null_type_counts"] = {g: dict(c) for g, c in self._null_type_counts.items()}
        data["context_counts"] = {g: dict(c) for g, c in self._context_counts.items()}
        data["type_context_counts"] = {g: dict(c) for g, c in self._type_context_counts.items()}
        data["clause_position_counts"] = {g: dict(c) for g, c in self._clause_position_counts.items()}
        data["overt_subject_counts"] = dict(self._overt_subject_counts)
        data["null_subject_counts"] = dict(self._null_subject_counts)
        data["finite_verb_counts"] = dict(self._finite_verb_counts)
        return data

    def from_checkpoint(self, data: Dict[str, Any]) -> None:
        super().from_checkpoint(data)
        self._null_type_counts = {
            g: Counter(c) for g, c in data.get("null_type_counts", {}).items()
        }
        self._context_counts = {
            g: Counter(c) for g, c in data.get("context_counts", {}).items()
        }
        self._type_context_counts = {
            g: Counter({
                ast.literal_eval(k) if isinstance(k, str) else k: v
                for k, v in c.items()
            })
            for g, c in data.get("type_context_counts", {}).items()
        }
        self._clause_position_counts = {
            g: Counter(c) for g, c in data.get("clause_position_counts", {}).items()
        }
        self._overt_subject_counts = Counter(data.get("overt_subject_counts", {}))
        self._null_subject_counts = Counter(data.get("null_subject_counts", {}))
        self._finite_verb_counts = Counter(data.get("finite_verb_counts", {}))
