"""
Expletive annotator.

Identifies expletive subjects and classifies them by type
(weather, existential, raising).
"""

from typing import Any, Dict, List, Optional

import spacy

from ..constants import WEATHER_VERBS
from .base import BaseSentenceAnnotator


class ExpletiveAnnotator(BaseSentenceAnnotator):
    """
    Annotates expletive subjects in a sentence.

    Identifies tokens with dep=expl and classifies them as:
    - weather: "it" with weather verbs (rain, snow, etc.)
    - existential: "there" in existential constructions
    - raising: "it" in raising constructions (it seems that...)
    """

    output_fields = {
        "expletives": "List of expletive annotation dicts",
    }

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract expletive annotations."""
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
            }
            expletives.append(expletive)

        return {"expletives": expletives}

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Compute expletive-related flags."""
        expletives = annotations.get("expletives", [])
        return {"has_expletive": len(expletives) > 0}
