"""
Verb annotator.

Extracts verb-level information including finiteness, tense, aspect, and mood.
"""

from typing import Any, Dict, List, Optional

import spacy

from preprocessing.dep_labels import normalize_dep

from .base import BaseSentenceAnnotator

_CLAUSE_DEPS = {"ROOT", "xcomp", "ccomp", "advcl", "acl", "acl:relcl"}


class VerbAnnotator(BaseSentenceAnnotator):
    """
    Annotates verbs in a sentence.

    For each verb/auxiliary, extracts:
    - Token position and lemma
    - Verb form (Fin, Inf, Part, Ger)
    - Clause position (ROOT, xcomp, ccomp, etc.)
    - Tense (Past, Pres, Fut) from morphology
    - Aspect (Perf, Prog) from morphology
    - Mood (Ind, Sub, Imp) from morphology
    """

    output_fields = {
        "verbs": "List of verb annotation dicts",
    }

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract verb annotations."""
        verbs = []

        for tok in sent:
            if tok.pos_ not in ("VERB", "AUX"):
                continue

            morph = tok.morph
            verb_forms = morph.get("VerbForm")
            tense_vals = morph.get("Tense")
            aspect_vals = morph.get("Aspect")
            mood_vals = morph.get("Mood")
            person_vals = morph.get("Person")
            number_vals = morph.get("Number")

            if not verb_forms:
                continue

            verb_form = verb_forms[0]

            # Clause position
            dep = normalize_dep(tok.dep_)
            if dep in _CLAUSE_DEPS:
                clause_pos = dep
            else:
                clause_pos = "other"

            # Extract tense/aspect/mood/person/number
            tense = tense_vals[0] if tense_vals else None
            aspect = aspect_vals[0] if aspect_vals else None
            mood = mood_vals[0] if mood_vals else None
            person = int(person_vals[0]) if person_vals else None
            number = number_vals[0] if number_vals else None

            verb = {
                "token_idx": tok.i - sent.start,
                "lemma": tok.lemma_.lower(),
                "verb_form": verb_form,
                "clause_position": clause_pos,
                "tense": tense,
                "aspect": aspect,
                "mood": mood,
                "person": person,
                "number": number,
            }
            verbs.append(verb)

        return {"verbs": verbs}

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Compute verb-related flags (none for this annotator)."""
        return {}
