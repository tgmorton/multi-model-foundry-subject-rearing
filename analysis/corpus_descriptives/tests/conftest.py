"""
Shared fixtures for corpus descriptive analysis tests.

Uses spacy.blank("en") with manually set token attributes —
no model download required.
"""

import pytest
import spacy
from spacy.tokens import Doc


@pytest.fixture
def nlp():
    """Blank English spaCy pipeline (no model needed)."""
    return spacy.blank("en")


def make_doc(nlp, words, pos=None, deps=None, heads=None, morphs=None, lemmas=None):
    """
    Build a spaCy Doc with manually set attributes.

    Args:
        nlp: blank spaCy Language
        words: list of token strings
        pos: list of POS tags (str)
        deps: list of dep labels (str)
        heads: list of head indices (int)
        morphs: list of morph dicts or strings (optional)
        lemmas: list of lemma strings (optional)

    Returns:
        spaCy Doc with attributes set
    """
    doc = Doc(nlp.vocab, words=words)

    if pos:
        for i, p in enumerate(pos):
            doc[i].pos_ = p
    if deps:
        for i, d in enumerate(deps):
            doc[i].dep_ = d
    if heads:
        for i, h in enumerate(heads):
            doc[i].head = doc[h]
    if morphs:
        for i, m in enumerate(morphs):
            if m:
                if isinstance(m, dict):
                    morph_str = "|".join(f"{k}={v}" for k, v in m.items())
                else:
                    morph_str = m
                doc[i].set_morph(morph_str)
    if lemmas:
        for i, lem in enumerate(lemmas):
            doc[i].lemma_ = lem

    return doc
