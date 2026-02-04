"""
Pronoun annotator.

Identifies pronouns and extracts their grammatical properties
including function, person, number, case, and reflexivity.
"""

from typing import Any, Dict, List, Optional

import spacy

from ..constants import SUBJECT_DEPS
from .base import BaseSentenceAnnotator


class PronounAnnotator(BaseSentenceAnnotator):
    """
    Annotates pronouns in a sentence.

    For each pronoun in an argumental position, extracts:
    - Token position
    - Lemma
    - Grammatical function (subject, direct_object, indirect_object, oblique)
    - Person, number, case from morphology
    - Reflexivity
    - Clitic status (for languages like Italian)
    - Head verb index
    """

    output_fields = {
        "pronouns": "List of pronoun annotation dicts",
    }

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract pronoun annotations."""
        pronouns = []

        for tok in sent:
            if tok.pos_ != "PRON":
                continue

            dep = tok.dep_

            # Classify function
            if dep in SUBJECT_DEPS:
                function = "subject"
            elif dep == "obj":
                function = "direct_object"
            elif dep == "iobj":
                function = "indirect_object"
            elif dep in ("obl", "nmod"):
                function = "oblique"
            else:
                # Skip pronouns not in clear argumental positions
                continue

            morph = tok.morph
            person_vals = morph.get("Person")
            number_vals = morph.get("Number")
            case_vals = morph.get("Case")
            reflex_vals = morph.get("Reflex")
            pron_type = morph.get("PronType")

            # Determine if reflexive
            is_reflexive = bool(reflex_vals and "Yes" in reflex_vals)

            # Determine if clitic (Italian/Romance languages)
            # Clitics often have PronType=Prs and are short forms
            is_clitic = False
            if pron_type and "Prs" in pron_type:
                # Simple heuristic: clitics are typically 1-2 characters
                # and attached to verbs
                if len(tok.text) <= 2 and tok.head.pos_ in ("VERB", "AUX"):
                    is_clitic = True

            # Get head verb index
            head_verb_idx = None
            if tok.head.pos_ in ("VERB", "AUX"):
                head_verb_idx = tok.head.i - sent.start

            pronoun = {
                "token_idx": tok.i - sent.start,
                "lemma": tok.lemma_.lower(),
                "function": function,
                "person": int(person_vals[0]) if person_vals else None,
                "number": number_vals[0] if number_vals else None,
                "case": case_vals[0] if case_vals else None,
                "is_reflexive": is_reflexive,
                "is_clitic": is_clitic,
                "head_verb_idx": head_verb_idx,
            }
            pronouns.append(pronoun)

        return {"pronouns": pronouns}

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Compute pronoun-related flags."""
        pronouns = annotations.get("pronouns", [])

        has_subject_pronoun = any(p["function"] == "subject" for p in pronouns)
        has_object_pronoun = any(
            p["function"] in ("direct_object", "indirect_object")
            for p in pronouns
        )

        return {
            "has_overt_subject_pronoun": has_subject_pronoun,
            "has_overt_object_pronoun": has_object_pronoun,
        }
