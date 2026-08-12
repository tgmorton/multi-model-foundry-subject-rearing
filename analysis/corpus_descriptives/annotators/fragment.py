"""
Fragment annotator.

Identifies sentence fragments and imperatives - sentences that lack
a finite verb or overt subject.
"""

from typing import Any, Dict, List, Optional

import spacy

from preprocessing.dep_labels import normalize_dep

from .base import BaseSentenceAnnotator


class FragmentAnnotator(BaseSentenceAnnotator):
    """
    Annotates fragments and imperatives.

    A fragment is a sentence that:
    - Has no ROOT dependency
    - Has a non-verbal ROOT
    - Has only non-finite verbs (that are not imperatives)

    An imperative is a sentence that:
    - Has Mood=Imp on the root verb, OR
    - Has a bare-form ROOT verb (VerbForm=Inf) with no overt subject
      (heuristic for English imperatives that spaCy tags as infinitive)

    This annotator provides more detailed fragment/imperative classification
    than the ComplexityAnnotator.
    """

    output_fields = {
        "is_fragment": "Boolean: sentence is a fragment",
        "is_imperative": "Boolean: sentence is an imperative",
        "has_null_subject": "Boolean: finite root verb with no overt subject",
        "fragment_type": "Type of fragment if applicable",
        "root_pos": "POS tag of ROOT token",
    }

    def __init__(self, language: str = "en"):
        self.language = language

    def _is_bare_imperative(self, root: spacy.tokens.Token, sent: spacy.tokens.Span) -> bool:
        """
        Heuristic: detect bare-form imperatives that spaCy tags as VerbForm=Inf.

        English imperatives use the base form, which is morphologically identical
        to the infinitive.  spaCy rarely assigns Mood=Imp, so we use structural
        cues instead:
        - ROOT verb with VerbForm=Inf (or no VerbForm at all)
        - No overt subject (nsubj / nsubj:pass / csubj / expl)
        - POS is VERB (not AUX — bare AUX roots like standalone "be" are
          less reliably imperative)
        - Does not follow "to" (infinitive marker → true infinitive, not
          imperative)
        - Not preceded by a modal/auxiliary that would make it a complement
          (e.g., "wanna go" — "go" is xcomp, not ROOT, so this is safe)

        Also catches negative imperatives ("don't cry") where "don't" is an
        AUX child of the ROOT verb.

        For Italian, returns False — Italian spaCy reliably sets Mood=Imp.
        """
        if self.language != "en":
            return False
        # Allow AUX through only for "be" (e.g., "be careful", "be quiet")
        if root.pos_ == "AUX" and root.lemma_.lower() == "be":
            pass
        elif root.pos_ != "VERB":
            return False

        children_deps = {normalize_dep(c.dep_) for c in root.children}
        has_subject = children_deps & {"nsubj", "nsubj:pass", "expl", "csubj", "csubj:pass"}
        if has_subject:
            return False

        # Check if preceded by "to" infinitive marker
        if root.i > sent.start:
            prev = root.doc[root.i - 1]
            if prev.dep_ == "mark" and prev.lemma_.lower() == "to":
                return False

        # Negative imperatives: "don't" / "do not" as AUX child
        for child in root.children:
            if child.dep_ == "aux" and child.lemma_.lower() == "do":
                return True

        # Bare-form ROOT verb with no subject — likely imperative
        return True

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
            "has_null_subject": False,
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

        # Check for imperative mood (spaCy explicit)
        if mood and "Imp" in mood:
            result["is_imperative"] = True
            return result

        # Non-finite root — check for bare imperative before calling it a fragment
        if verb_forms and "Fin" not in verb_forms:
            if self._is_bare_imperative(root, sent):
                result["is_imperative"] = True
                return result
            result["is_fragment"] = True
            form = verb_forms[0] if verb_forms else "unknown"
            result["fragment_type"] = f"nonfinite_{form.lower()}"
            return result

        # Null subject detection: finite root verb with no overt subject
        has_subject = any(
            normalize_dep(c.dep_) in ("nsubj", "nsubj:pass", "expl", "csubj", "csubj:pass")
            for c in root.children
        )

        if not has_subject:
            # Check for disfluent verb reduplication
            if root.i > sent.start:
                prev = root.doc[root.i - 1]
                if prev.lemma_.lower() == root.lemma_.lower() and prev.pos_ in ("VERB", "AUX"):
                    has_subject = True  # disfluent copy, not truly null

        if not has_subject:
            result["has_null_subject"] = True

        return result

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Return fragment and imperative flags."""
        return {
            "is_fragment": annotations.get("is_fragment", False),
            "is_imperative": annotations.get("is_imperative", False),
            "has_null_subject": annotations.get("has_null_subject", False),
        }
