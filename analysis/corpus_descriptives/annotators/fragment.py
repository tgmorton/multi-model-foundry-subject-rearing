"""
Fragment annotator.

Identifies sentence fragments and imperatives - sentences that lack
a finite verb or overt subject.
"""

from typing import Any, Dict, List, Optional

import spacy

from .base import BaseSentenceAnnotator


class FragmentAnnotator(BaseSentenceAnnotator):
    """
    Annotates fragments and imperatives.

    A fragment is a sentence that:
    - Has no ROOT dependency
    - Has a non-verbal ROOT
    - Has only non-finite verbs

    An imperative is a sentence that:
    - Has Mood=Imp on the root verb
    - Has a finite root verb but no overt subject (in English)

    This annotator provides more detailed fragment/imperative classification
    than the ComplexityAnnotator.
    """

    output_fields = {
        "is_fragment": "Boolean: sentence is a fragment",
        "is_imperative": "Boolean: sentence is an imperative",
        "fragment_type": "Type of fragment if applicable",
        "root_pos": "POS tag of ROOT token",
    }

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract fragment/imperative annotations."""
        # Find the root
        root = None
        for tok in sent:
            if tok.dep_ == "ROOT":
                root = tok
                break

        result = {
            "is_fragment": False,
            "is_imperative": False,
            "fragment_type": None,
            "root_pos": root.pos_ if root else None,
        }

        if root is None:
            result["is_fragment"] = True
            result["fragment_type"] = "no_root"
            return result

        # Check if root is a verb/auxiliary
        if root.pos_ not in ("VERB", "AUX"):
            result["is_fragment"] = True
            result["fragment_type"] = f"nonverbal_{root.pos_.lower()}"
            return result

        # Check verb form and mood
        verb_forms = root.morph.get("VerbForm")
        mood = root.morph.get("Mood")

        # Non-finite root
        if verb_forms and "Fin" not in verb_forms:
            result["is_fragment"] = True
            form = verb_forms[0] if verb_forms else "unknown"
            result["fragment_type"] = f"nonfinite_{form.lower()}"
            return result

        # Check for imperative mood
        if mood and "Imp" in mood:
            result["is_imperative"] = True
            return result

        # Finite verb without subject - likely imperative (in English)
        has_subject = any(
            c.dep_ in ("nsubj", "nsubj:pass", "expl")
            for c in root.children
        )

        if not has_subject:
            # Could be imperative or null subject
            # Use lemma heuristics for common imperatives
            if root.lemma_.lower() in ("let", "be", "do", "have", "go", "come", "see", "look"):
                result["is_imperative"] = True
            else:
                # Ambiguous - mark as potential imperative based on context
                # In English, subjectless finite clauses are usually imperatives
                result["is_imperative"] = True

        return result

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Return fragment and imperative flags."""
        return {
            "is_fragment": annotations.get("is_fragment", False),
            "is_imperative": annotations.get("is_imperative", False),
        }
