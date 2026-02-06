"""Null-subject context categorization."""

import logging
from typing import Optional

import spacy

logger = logging.getLogger(__name__)


def classify_null_subject_context(
    label: str,
    verb_token: spacy.tokens.Token,
    doc: spacy.tokens.Doc,
    prev_subject_features: Optional[str] = None,
) -> str:
    """Classify the context of a null subject.

    Args:
        label: The pronoun label (PRO.1sg, IMP, CONJ, etc.)
        verb_token: The spaCy token of the verb with the null subject
        doc: The full spaCy Doc
        prev_subject_features: Person/Number features of the previous subject (for topic continuity)

    Categories:
    - diary_drop: PRO.1sg at utterance-initial position (verb is first non-punct token or first in sentence)
    - imperative: IMP label
    - question_2p: PRO.2sg or PRO.2pl with interrogative marker
      (sentence ends with '?' or has aux-subj inversion pattern)
    - topic_continuity: CONJ label, or PRO.* label matching features of prior subject
    - postverbal: (Italian only) verb has postverbal nsubj child
    - other: default

    Returns one of: 'diary_drop', 'imperative', 'question_2p', 'topic_continuity', 'postverbal', 'other'
    """
    # Imperative
    if label == "IMP":
        return "imperative"

    # CONJ → topic continuity
    if label == "CONJ":
        return "topic_continuity"

    # Diary drop: 1sg at utterance-initial position
    if label == "PRO.1sg":
        if _is_utterance_initial(verb_token, doc):
            return "diary_drop"

    # Question with 2nd person
    if label in ("PRO.2sg", "PRO.2pl"):
        if _is_interrogative(verb_token, doc):
            return "question_2p"

    # Topic continuity: features match previous subject
    if prev_subject_features and label.startswith("PRO."):
        # Extract person/number suffix from label (e.g., "3sg" from "PRO.3sg")
        label_suffix = label.replace("PRO.", "")
        if prev_subject_features == label_suffix:
            return "topic_continuity"

    # Postverbal subject (Italian)
    if _has_postverbal_subject(verb_token):
        return "postverbal"

    return "other"


def _is_utterance_initial(token: spacy.tokens.Token, doc: spacy.tokens.Doc) -> bool:
    """Check if token is at utterance-initial position.
    True if it's the first non-punctuation token in the sentence or doc."""
    sent = token.sent if token.sent is not None else doc
    for t in sent:
        if not t.is_punct and not t.is_space:
            return t.i == token.i
    return False


def _is_interrogative(token: spacy.tokens.Token, doc: spacy.tokens.Doc) -> bool:
    """Check if the sentence is interrogative.
    Heuristic: sentence ends with '?' or has aux-subject inversion."""
    sent = token.sent if token.sent is not None else doc
    # Check for question mark
    for t in reversed(list(sent)):
        if not t.is_space:
            if t.text == "?":
                return True
            break
    return False


def _has_postverbal_subject(verb_token: spacy.tokens.Token) -> bool:
    """Check if verb has a postverbal nominal subject (Italian pattern).
    True if any nsubj child appears after the verb."""
    for child in verb_token.children:
        if child.dep_ in ("nsubj", "nsubj:pass") and child.i > verb_token.i:
            return True
    return False
