"""
Shared fixtures for corpus descriptive analysis tests.

Uses spacy.blank("en") with manually set token attributes —
no model download required.
"""

import pytest
import spacy
from spacy.tokens import Doc


def _spacy_model_available(model_name: str) -> bool:
    """Check if a spaCy model is installed."""
    try:
        spacy.load(model_name)
        return True
    except OSError:
        return False


@pytest.fixture
def blank_nlp():
    """Blank English spaCy pipeline (no model needed)."""
    return spacy.blank("en")


@pytest.fixture
def nlp():
    """
    spaCy pipeline for testing annotators.

    Uses en_core_web_sm if available, otherwise blank pipeline with sentencizer.
    """
    if _spacy_model_available("en_core_web_sm"):
        return spacy.load("en_core_web_sm")
    else:
        # Fallback: blank pipeline with sentencizer
        _nlp = spacy.blank("en")
        _nlp.add_pipe("sentencizer")
        return _nlp


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
