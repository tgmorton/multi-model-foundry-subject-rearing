"""
Expletive annotator.

Identifies expletive subjects and classifies them by type
(weather, existential, raising).

For Italian: detects weather/impersonal verbs with null subject,
which are implicit expletive contexts (Italian lacks overt expletives).
"""

from typing import Any, Dict, List, Optional

import spacy

from preprocessing.dep_labels import normalize_dep

from ..constants import (
    WEATHER_VERBS,
    WEATHER_VERBS_IT,
    IMPERSONAL_VERBS_IT,
    NECESSITY_VERBS_IT,
)
from .base import BaseSentenceAnnotator

# Subject dependency labels
_SUBJECT_DEPS = {"nsubj", "nsubj:pass"}


class ExpletiveAnnotator(BaseSentenceAnnotator):
    """
    Annotates expletive subjects in a sentence.

    For English: identifies tokens with dep=expl and classifies them as:
    - weather: "it" with weather verbs (rain, snow, etc.)
    - existential: "there" in existential constructions
    - raising: "it" in raising constructions (it seems that...)

    For Italian: detects implicit expletive contexts (weather/impersonal verbs
    with null subjects) since Italian lacks overt expletive pronouns.
    """

    output_fields = {
        "expletives": "List of expletive annotation dicts",
    }

    def __init__(self, language: str = "en"):
        """
        Initialize the expletive annotator.

        Args:
            language: Language code ("en" for English, "it" for Italian)
        """
        super().__init__()
        self.language = language

    def _has_subject(self, verb_token: spacy.tokens.Token) -> bool:
        """Check if a verb has an overt subject."""
        for child in verb_token.children:
            if normalize_dep(child.dep_) in _SUBJECT_DEPS:
                return True
        return False

    def _annotate_english(
        self,
        sent: spacy.tokens.Span,
    ) -> List[Dict[str, Any]]:
        """Extract expletive annotations for English."""
        expletives = []

        for tok in sent:
            if tok.dep_ != "expl":
                continue

            head_lemma = tok.head.lemma_.lower()
            lemma = tok.lemma_.lower()

            # Classify expletive type
            if head_lemma in WEATHER_VERBS:
                expl_class = "weather"
            elif lemma == "there":
                expl_class = "existential"
            else:
                expl_class = "raising"

            expletive = {
                "token_idx": tok.i - sent.start,
                "lemma": lemma,
                "expletive_class": expl_class,
                "verb_lemma": head_lemma,
                "is_implicit": False,
            }
            expletives.append(expletive)

        return expletives

    def _annotate_italian(
        self,
        sent: spacy.tokens.Span,
    ) -> List[Dict[str, Any]]:
        """
        Extract implicit expletive contexts for Italian.

        Italian lacks overt expletives; instead, weather verbs and impersonal
        verbs appear with null subjects. These are marked as implicit expletives.
        """
        expletives = []

        for tok in sent:
            if tok.pos_ not in ("VERB", "AUX"):
                continue

            lemma = tok.lemma_.lower()

            # Check if this is a weather/impersonal verb
            if lemma in WEATHER_VERBS_IT:
                expl_class = "weather"
            elif lemma in IMPERSONAL_VERBS_IT:
                expl_class = "raising"
            elif lemma in NECESSITY_VERBS_IT:
                expl_class = "impersonal"
            else:
                continue

            # Check for null subject (no overt nsubj)
            if not self._has_subject(tok):
                expletive = {
                    "token_idx": tok.i - sent.start,
                    "lemma": None,  # No overt expletive token
                    "expletive_class": expl_class,
                    "verb_lemma": lemma,
                    "is_implicit": True,
                }
                expletives.append(expletive)

        return expletives

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract expletive annotations."""
        if self.language == "it":
            expletives = self._annotate_italian(sent)
        else:
            expletives = self._annotate_english(sent)

        return {"expletives": expletives}

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Compute expletive-related flags."""
        expletives = annotations.get("expletives", [])
        return {"has_expletive": len(expletives) > 0}
