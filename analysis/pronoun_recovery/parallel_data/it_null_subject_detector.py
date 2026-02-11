"""
Identify Italian finite verbs and their subject status.

For each finite verb in an Italian spaCy Doc, determines whether it
has an overt subject, a clausal subject, an inherited subject (xcomp),
or no subject (null/pro-drop).

Logic mirrors analysis/corpus_descriptives/annotators/clause_structure.py
(lines 74-121) with adaptations for the alignment pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import spacy.tokens

from analysis.pronoun_recovery.constants import MORPH_TO_LABEL_SUFFIX

_OVERT_SUBJECT_DEPS = {"nsubj", "nsubj:pass"}
_CLAUSAL_SUBJECT_DEPS = {"csubj", "csubj:pass"}
_ANY_SUBJECT_DEPS = _OVERT_SUBJECT_DEPS | _CLAUSAL_SUBJECT_DEPS | {"expl"}


@dataclass
class ItalianVerb:
    """An Italian finite verb with subject status and morphology."""

    token_idx: int
    text: str
    lemma: str
    has_overt_subject: bool
    subject_status: str  # "overt", "clausal", "expletive", "inherited", "null"
    person: Optional[str]  # spaCy Person feature or None
    number: Optional[str]  # spaCy Number feature or None
    morph_label_suffix: Optional[str]  # e.g. "1sg" or None if morph missing


def detect_finite_verbs(doc: spacy.tokens.Doc) -> List[ItalianVerb]:
    """Find all finite verbs in an Italian sentence and classify their subject status.

    Args:
        doc: spaCy-parsed Italian sentence.

    Returns:
        List of ItalianVerb dataclasses, one per finite verb found.
    """
    results: List[ItalianVerb] = []

    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue

        verb_forms = tok.morph.get("VerbForm")
        if not verb_forms or verb_forms[0] != "Fin":
            continue

        # Collect children dependency labels.
        children_deps: Dict[str, spacy.tokens.Token] = {}
        for child in tok.children:
            children_deps[child.dep_] = child
        dep_set = set(children_deps.keys())

        # Determine subject status (same priority as clause_structure.py).
        if "expl" in dep_set:
            subject_status = "expletive"
            has_overt = True
        elif dep_set & _OVERT_SUBJECT_DEPS:
            subject_status = "overt"
            has_overt = True
        elif dep_set & _CLAUSAL_SUBJECT_DEPS:
            subject_status = "clausal"
            has_overt = True
        elif tok.dep_ == "xcomp":
            # xcomp verbs inherit subject from matrix verb.
            head_child_deps = {c.dep_ for c in tok.head.children}
            if head_child_deps & _ANY_SUBJECT_DEPS:
                subject_status = "inherited"
                has_overt = True
            else:
                subject_status = "null"
                has_overt = False
        else:
            subject_status = "null"
            has_overt = False

        # Extract morphological Person/Number.
        person_feats = tok.morph.get("Person")
        number_feats = tok.morph.get("Number")
        person = person_feats[0] if person_feats else None
        number = number_feats[0] if number_feats else None

        morph_suffix = MORPH_TO_LABEL_SUFFIX.get((person, number)) if person and number else None

        results.append(
            ItalianVerb(
                token_idx=tok.i,
                text=tok.text,
                lemma=tok.lemma_,
                has_overt_subject=has_overt,
                subject_status=subject_status,
                person=person,
                number=number,
                morph_label_suffix=morph_suffix,
            )
        )

    return results
