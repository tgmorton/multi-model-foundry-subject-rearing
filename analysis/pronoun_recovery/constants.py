"""
Constants for the pronoun recovery model.

Pronoun label sets, feature maps, and language-specific tables.
English implemented now; Italian is a config change later.
"""

from typing import Dict, FrozenSet, Set

# ── Label Set (9 labels) ──────────────────────────────────────────────
# Used by both synthetic data generation and the sequence labeler.
LABEL_NONE = "NONE"
PRONOUN_LABELS = frozenset({
    "PRO.1sg", "PRO.2sg", "PRO.3sg",
    "PRO.1pl", "PRO.2pl", "PRO.3pl",
    "IMP", "CONJ",
})
ALL_LABELS = [LABEL_NONE] + sorted(PRONOUN_LABELS)
LABEL_TO_ID: Dict[str, int] = {label: i for i, label in enumerate(ALL_LABELS)}
ID_TO_LABEL: Dict[int, str] = {i: label for label, i in LABEL_TO_ID.items()}
NUM_LABELS = len(ALL_LABELS)

# ── Person/Number Morph → Label mapping ───────────────────────────────
# spaCy morph features (Person, Number) → PRO label suffix
MORPH_TO_LABEL_SUFFIX: Dict[tuple, str] = {
    ("1", "Sing"): "1sg",
    ("2", "Sing"): "2sg",
    ("3", "Sing"): "3sg",
    ("1", "Plur"): "1pl",
    ("2", "Plur"): "2pl",
    ("3", "Plur"): "3pl",
}

# ── English Pronoun Lexical Forms ─────────────────────────────────────
# Maps label → set of valid lexical forms (lowercase)
EN_LABEL_TO_FORMS: Dict[str, FrozenSet[str]] = {
    "PRO.1sg": frozenset({"i"}),
    "PRO.2sg": frozenset({"you"}),
    "PRO.3sg": frozenset({"he", "she", "it", "they"}),
    "PRO.1pl": frozenset({"we"}),
    "PRO.2pl": frozenset({"you"}),
    "PRO.3pl": frozenset({"they"}),
}

# ── Italian Pronoun Lexical Forms (placeholder for future) ────────────
IT_LABEL_TO_FORMS: Dict[str, FrozenSet[str]] = {
    "PRO.1sg": frozenset({"io"}),
    "PRO.2sg": frozenset({"tu"}),
    "PRO.3sg": frozenset({"lui", "lei", "esso", "essa", "ciò"}),
    "PRO.1pl": frozenset({"noi"}),
    "PRO.2pl": frozenset({"voi"}),
    "PRO.3pl": frozenset({"loro", "essi", "esse"}),
}

LANGUAGE_FORM_MAPS: Dict[str, Dict[str, FrozenSet[str]]] = {
    "en": EN_LABEL_TO_FORMS,
    "it": IT_LABEL_TO_FORMS,
}

# ── Default Pronoun Insertion Tables ──────────────────────────────────
# Deterministic (except 3sg) maps used by Stage 2 insertion.
EN_DEFAULT_PRONOUN: Dict[str, str] = {
    "PRO.1sg": "I",
    "PRO.2sg": "you",
    "PRO.3sg": "they",   # singular they as fallback
    "PRO.1pl": "we",
    "PRO.2pl": "you",
    "PRO.3pl": "they",
}

IT_DEFAULT_PRONOUN: Dict[str, str] = {
    "PRO.1sg": "io",
    "PRO.2sg": "tu",
    "PRO.3sg": "lui",    # masculine default
    "PRO.1pl": "noi",
    "PRO.2pl": "voi",
    "PRO.3pl": "loro",
}

LANGUAGE_DEFAULT_PRONOUNS: Dict[str, Dict[str, str]] = {
    "en": EN_DEFAULT_PRONOUN,
    "it": IT_DEFAULT_PRONOUN,
}

# ── Dependency / POS constants ────────────────────────────────────────
SUBJECT_DEPS: FrozenSet[str] = frozenset({"nsubj", "nsubj:pass", "expl"})
SUBJECT_PRONOUN_DEPS: FrozenSet[str] = frozenset({"nsubj"})

# ── Genre mapping (re-exported from corpus_descriptives for convenience) ──
DEFAULT_GENRE_MAP_EN: Dict[str, str] = {
    "childes": "CHILDES",
    "bnc_spoken": "BNC",
    "gutenberg": "Gutenberg",
    "open_subtitles": "OpenSubtitles",
    "simple_wiki": "SimpleWikipedia",
    "switchboard": "Switchboard",
}

DEFAULT_GENRE_MAP_IT: Dict[str, str] = {
    "childes": "CHILDES",
    "clta": "CltA",
    "corpus_isacco": "ISACCO",
    "europarl": "Europarl",
    "leipzig_web": "Leipzig Web",
    "paccss": "PaCCSS",
    "qcri": "QCRI",
    "spgc": "SPGC",
}

# ── Null-subject context categories ──────────────────────────────────
NULL_SUBJECT_CONTEXTS: FrozenSet[str] = frozenset({
    "diary_drop",
    "imperative",
    "question_2p",
    "topic_continuity",
    "postverbal",
    "other",
})
