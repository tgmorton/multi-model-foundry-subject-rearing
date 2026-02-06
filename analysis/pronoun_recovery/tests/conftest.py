"""
Shared fixtures for pronoun recovery tests.

Uses spacy.blank("en") with manually set token attributes ---
no model download required.
"""

import pytest
import spacy
from spacy.tokens import Doc


@pytest.fixture
def nlp():
    return spacy.blank("en")


@pytest.fixture
def nlp_it():
    return spacy.blank("it")


def make_doc(nlp, words, pos=None, deps=None, heads=None, morphs=None, lemmas=None):
    """Build a spaCy Doc with manually set attributes (no model needed)."""
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


@pytest.fixture
def she_runs_doc(nlp):
    """'She runs fast.' --- one subject pronoun to drop."""
    return make_doc(
        nlp,
        words=["She", "runs", "fast", "."],
        pos=["PRON", "VERB", "ADV", "PUNCT"],
        deps=["nsubj", "ROOT", "advmod", "punct"],
        heads=[1, 1, 1, 1],
        morphs=[
            {"Person": "3", "Number": "Sing", "Gender": "Fem"},
            {"VerbForm": "Fin", "Person": "3", "Number": "Sing"},
            None,
            None,
        ],
        lemmas=["she", "run", "fast", "."],
    )


@pytest.fixture
def dog_runs_doc(nlp):
    """'The dog runs.' --- no subject pronoun (noun subject)."""
    return make_doc(
        nlp,
        words=["The", "dog", "runs", "."],
        pos=["DET", "NOUN", "VERB", "PUNCT"],
        deps=["det", "nsubj", "ROOT", "punct"],
        heads=[1, 2, 2, 2],
        morphs=[
            None,
            None,
            {"VerbForm": "Fin", "Person": "3", "Number": "Sing"},
            None,
        ],
        lemmas=["the", "dog", "run", "."],
    )


@pytest.fixture
def multi_pronoun_doc(nlp):
    """'I ran and she swam.' --- two subject pronouns."""
    return make_doc(
        nlp,
        words=["I", "ran", "and", "she", "swam", "."],
        pos=["PRON", "VERB", "CCONJ", "PRON", "VERB", "PUNCT"],
        deps=["nsubj", "ROOT", "cc", "nsubj", "conj", "punct"],
        heads=[1, 1, 4, 4, 1, 1],
        morphs=[
            {"Person": "1", "Number": "Sing"},
            {"VerbForm": "Fin", "Person": "1", "Number": "Sing"},
            None,
            {"Person": "3", "Number": "Sing", "Gender": "Fem"},
            {"VerbForm": "Fin", "Person": "3", "Number": "Sing"},
            None,
        ],
        lemmas=["I", "run", "and", "she", "swim", "."],
    )
