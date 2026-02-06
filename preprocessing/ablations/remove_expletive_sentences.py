"""
Remove entire sentences containing expletive constructions.

Unlike the archived remove_expletives.py (which removes individual expletive tokens),
this ablation removes the entire line whenever an expletive construction is detected.
This provides a cleaner ablation: lines with expletive constructions are dropped
entirely, and the replacement pool is used to maintain corpus size.

Two language-specific factories are provided:

- **English:** A line is removed if any token has ``dep_ == 'expl'``.
- **Italian:** Italian lacks overt expletive pronouns.  We detect
  expletive-equivalent constructions via verb lemma and syntactic pattern:
  weather verbs, existential *ci + essere*, impersonal raising verbs with
  clausal complements, and impersonal necessity verbs without *nsubj*.

Each factory returns a function with the standard ablation signature
``(Doc) -> (str, int)`` where the returned text is either the original line
(kept) or an empty string (removed), and the int is 1 if removed, 0 otherwise.
"""

from typing import Callable, Tuple

import spacy

from analysis.corpus_descriptives.constants import (
    IMPERSONAL_VERBS_IT,
    NECESSITY_VERBS_IT,
    WEATHER_VERBS_IT,
)
from preprocessing.registry import AblationRegistry


# ---------------------------------------------------------------------------
# English
# ---------------------------------------------------------------------------

def _has_expletive_en(doc: spacy.tokens.Doc) -> bool:
    """Return True if any token in the doc has dependency label 'expl'."""
    return any(tok.dep_ == "expl" for tok in doc)


def make_remove_expletive_sentences_en() -> Callable[[spacy.tokens.Doc], Tuple[str, int]]:
    """
    Factory for the English expletive-sentence removal ablation.

    Returns:
        Ablation function ``(Doc) -> (str, int)``
    """

    def remove_expletive_sentences_en_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
        """Remove the entire line if it contains an expletive construction (EN)."""
        if _has_expletive_en(doc):
            return "", 1
        return doc.text_with_ws, 0

    return remove_expletive_sentences_en_doc


# ---------------------------------------------------------------------------
# Italian
# ---------------------------------------------------------------------------

def _has_expletive_equivalent_it(doc: spacy.tokens.Doc) -> bool:
    """
    Return True if the doc contains an Italian expletive-equivalent construction.

    Detection categories:
    1. Weather verbs (e.g. *piove*, *nevica*)
    2. Existential *ci + essere* (e.g. *c'è*, *ci sono*)
    3. Impersonal raising verbs with a clausal complement and no nsubj
       (e.g. *sembra che ...*)
    4. Impersonal necessity verbs without nsubj (e.g. *bisogna*)
    """
    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue

        lemma = tok.lemma_.lower()

        # 1. Weather verbs
        if lemma in WEATHER_VERBS_IT:
            return True

        # 2. Existential: ci + essere
        if lemma == "essere":
            for child in tok.children:
                # "ci" appears as an adverbial or expletive clitic
                if child.lower_ == "ci" and child.dep_ in ("expl", "advmod"):
                    return True

        # 3. Impersonal raising verbs with clausal complement, no nsubj
        if lemma in IMPERSONAL_VERBS_IT:
            children_deps = {child.dep_ for child in tok.children}
            has_clausal = bool(children_deps & {"ccomp", "xcomp", "csubj"})
            has_nsubj = bool(children_deps & {"nsubj", "nsubj:pass"})
            if has_clausal and not has_nsubj:
                return True

        # 4. Impersonal necessity verbs without nsubj
        if lemma in NECESSITY_VERBS_IT:
            children_deps = {child.dep_ for child in tok.children}
            if not (children_deps & {"nsubj", "nsubj:pass"}):
                return True

    return False


def make_remove_expletive_sentences_it() -> Callable[[spacy.tokens.Doc], Tuple[str, int]]:
    """
    Factory for the Italian expletive-equivalent sentence removal ablation.

    Returns:
        Ablation function ``(Doc) -> (str, int)``
    """

    def remove_expletive_sentences_it_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
        """Remove the entire line if it contains an expletive-equivalent (IT)."""
        if _has_expletive_equivalent_it(doc):
            return "", 1
        return doc.text_with_ws, 0

    return remove_expletive_sentences_it_doc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _make_validator(detect_fn):
    """Create a validation function for a given detection function."""

    def validate(original: str, ablated: str, nlp) -> bool:
        """
        Validate that expletive sentences were removed.

        If the original text contained an expletive construction, the ablated
        text should be empty (the line was removed).  If not, both should match.
        """
        original_doc = nlp(original)
        has_expletive = detect_fn(original_doc)
        if has_expletive:
            return ablated.strip() == ""
        else:
            return ablated.strip() == original.strip()

    return validate


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AblationRegistry.register(
    "remove_expletive_sentences_en",
    make_remove_expletive_sentences_en(),
    _make_validator(_has_expletive_en),
)

AblationRegistry.register(
    "remove_expletive_sentences_it",
    make_remove_expletive_sentences_it(),
    _make_validator(_has_expletive_equivalent_it),
)
