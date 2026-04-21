"""
Impoverish pronoun case — collapse all pronoun forms to nominative.

This ablation removes case morphology evidence from the corpus by mapping every
non-nominative pronoun (accusative, dative, genitive/possessive, reflexive) to
its nominative counterpart.

**English** examples:  *him* → *he*, *her* → *she*, *them* → *they*,
*my* → *I*, *ourselves* → *we*.

**Italian** examples: clitics (*lo* → *lui*, *mi* → *io*), strong obliques
(*me* → *io*, *te* → *tu*), and possessives (*mio/mia/miei/mie* → *io*).
Italian clitics are syntactically integrated, so replacing them with strong
nominative forms is intentionally aggressive — it destroys clitic-placement
evidence alongside case evidence.

**Spanish** examples: strong obliques (*mí* → *yo*, *ti* → *tú*), preposition-
bound portmanteaux (*conmigo* → *yo*, *contigo* → *tú*), accusative and
dative clitics (*me* → *yo*, *lo* → *él*, *le* → *él*), possessives
(*mi/mío/míos* → *yo*). As with Italian, collapsing clitics to strong forms
intentionally destroys clitic-placement evidence alongside case evidence.

Three language-specific ablation functions are registered:
``impoverish_case_en``, ``impoverish_case_it``, and ``impoverish_case_es``.
"""

from typing import Callable, Dict, Tuple

import spacy

from preprocessing.registry import AblationRegistry


# ---------------------------------------------------------------------------
# English case mapping
# ---------------------------------------------------------------------------

# Every non-nominative English pronoun form → nominative equivalent (lowercase)
ENGLISH_CASE_TO_NOM: Dict[str, str] = {
    # 1sg
    "me": "i", "my": "i", "mine": "i", "myself": "i",
    # 2sg/2pl (same forms)
    "your": "you", "yours": "you", "yourself": "you", "yourselves": "you",
    # 3sg.m
    "him": "he", "his": "he", "himself": "he",
    # 3sg.f
    "her": "she", "hers": "she", "herself": "she",
    # 3sg.n
    "its": "it", "itself": "it",
    # 1pl
    "us": "we", "our": "we", "ours": "we", "ourselves": "we",
    # 3pl
    "them": "they", "their": "they", "theirs": "they", "themselves": "they",
    # relative / interrogative
    "whom": "who", "whose": "who",
}

# POS tags that can carry pronoun case in English (spaCy tags possessive
# determiners as DET rather than PRON)
_EN_PRONOUN_POS = frozenset({"PRON", "DET"})


# ---------------------------------------------------------------------------
# Italian case mapping
# ---------------------------------------------------------------------------

# Strong oblique → nominative
ITALIAN_CASE_TO_NOM: Dict[str, str] = {
    # Strong obliques
    "me": "io", "te": "tu",
    # Clitics — direct object
    "mi": "io", "ti": "tu", "lo": "lui", "la": "lei",
    "ci": "noi", "vi": "voi", "li": "loro", "le": "loro",
    # Clitics — indirect object
    "gli": "lui",
    # Clitics — reflexive (map to generic 3p; context-free approximation)
    "si": "si",  # left unchanged — reflexive has no single nominative target
    # Possessives — masculine singular as citation form key; all genders/numbers
    # are enumerated below
    "mio": "io", "mia": "io", "miei": "io", "mie": "io",
    "tuo": "tu", "tua": "tu", "tuoi": "tu", "tue": "tu",
    "suo": "lui", "sua": "lui", "suoi": "lui", "sue": "lui",
    "nostro": "noi", "nostra": "noi", "nostri": "noi", "nostre": "noi",
    "vostro": "voi", "vostra": "voi", "vostri": "voi", "vostre": "voi",
    # "loro" as possessive is already invariant and coincides with nominative
}

_IT_PRONOUN_POS = frozenset({"PRON", "DET"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_capitalization(replacement: str, original: str) -> str:
    """
    Return *replacement* with the same capitalization pattern as *original*.

    Rules:
    - ALL CAPS → ALL CAPS
    - Title Case → Title Case
    - Otherwise → lowercase (replacement is stored lowercase)
    """
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _impoverish_case(
    doc: spacy.tokens.Doc,
    mapping: Dict[str, str],
    target_pos: frozenset,
) -> Tuple[str, int]:
    """
    Core token-replacement loop shared by both languages.

    For each token whose ``lower_`` form appears in *mapping* and whose POS
    is in *target_pos*, replace it with the nominative form (preserving
    capitalization and whitespace).
    """
    modified_parts = []
    num_replaced = 0

    for tok in doc:
        lower = tok.lower_
        if tok.pos_ in target_pos and lower in mapping:
            # Skip definite articles — PronType=Art distinguishes e.g.
            # Italian "lo" (article, DET) from "lo" (clitic pronoun, PRON).
            pron_type = tok.morph.get("PronType")
            if pron_type and "Art" in pron_type:
                modified_parts.append(tok.text_with_ws)
                continue
            nom = mapping[lower]
            # Skip identity mappings (e.g. "si" → "si")
            if nom == lower:
                modified_parts.append(tok.text_with_ws)
            else:
                modified_parts.append(
                    _match_capitalization(nom, tok.text) + tok.whitespace_
                )
                num_replaced += 1
        else:
            modified_parts.append(tok.text_with_ws)

    return "".join(modified_parts), num_replaced


# ---------------------------------------------------------------------------
# English ablation
# ---------------------------------------------------------------------------

def impoverish_case_en_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """Collapse all English pronoun case forms to nominative."""
    return _impoverish_case(doc, ENGLISH_CASE_TO_NOM, _EN_PRONOUN_POS)


def validate_impoverish_case_en(original: str, ablated: str, nlp) -> bool:
    """Validate that non-nominative pronouns were reduced."""
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    def count_oblique(doc):
        return sum(
            1 for tok in doc
            if tok.pos_ in _EN_PRONOUN_POS and tok.lower_ in ENGLISH_CASE_TO_NOM
        )

    orig_count = count_oblique(original_doc)
    abl_count = count_oblique(ablated_doc)

    if orig_count > 0:
        return abl_count < orig_count
    return True


# ---------------------------------------------------------------------------
# Italian ablation
# ---------------------------------------------------------------------------

def impoverish_case_it_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """Collapse all Italian pronoun case forms to nominative."""
    return _impoverish_case(doc, ITALIAN_CASE_TO_NOM, _IT_PRONOUN_POS)


def validate_impoverish_case_it(original: str, ablated: str, nlp) -> bool:
    """Validate that non-nominative Italian pronouns were reduced."""
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    def count_oblique(doc):
        return sum(
            1 for tok in doc
            if tok.pos_ in _IT_PRONOUN_POS
            and tok.lower_ in ITALIAN_CASE_TO_NOM
            and ITALIAN_CASE_TO_NOM[tok.lower_] != tok.lower_
            and "Art" not in (tok.morph.get("PronType") or [])
        )

    orig_count = count_oblique(original_doc)
    abl_count = count_oblique(ablated_doc)

    if orig_count > 0:
        return abl_count < orig_count
    return True


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AblationRegistry.register(
    "impoverish_case_en",
    impoverish_case_en_doc,
    validate_impoverish_case_en,
)

AblationRegistry.register(
    "impoverish_case_it",
    impoverish_case_it_doc,
    validate_impoverish_case_it,
)


# ---------------------------------------------------------------------------
# Spanish case mapping
# ---------------------------------------------------------------------------

# Spanish non-nominative pronoun forms → nominative equivalent (lowercase,
# Peninsular). Avoids leísmo/laísmo by mapping dative *le*/*les* to nominative
# *él*/*ellos* (masculine default) rather than preserving the dative form.
SPANISH_CASE_TO_NOM: Dict[str, str] = {
    # Tonic oblique forms (preposition-bound)
    "mí": "yo", "ti": "tú",
    "sí": "él",  # reflexive tonic; masculine default, ambiguous
    # Preposition-bound special portmanteaux
    "conmigo": "yo", "contigo": "tú", "consigo": "él",
    # Direct-object clitics (accusative)
    "me": "yo", "te": "tú",
    "lo": "él", "la": "ella",
    "nos": "nosotros", "os": "vosotros",
    "los": "ellos", "las": "ellas",
    # Indirect-object clitics (dative) — collapse to the corresponding
    # nominative (masculine default; laísmo/leísmo avoided)
    "le": "él",
    "les": "ellos",
    # Reflexive "se" — identity mapping, skipped
    "se": "se",
    # Possessives — short (prenominal) forms
    "mi": "yo", "mis": "yo",
    "tu": "tú", "tus": "tú",
    "su": "él", "sus": "él",
    "nuestro": "nosotros", "nuestra": "nosotros",
    "nuestros": "nosotros", "nuestras": "nosotros",
    "vuestro": "vosotros", "vuestra": "vosotros",
    "vuestros": "vosotros", "vuestras": "vosotros",
    # Possessives — long (postnominal/predicative) forms.
    # Note: Spanish 1pl/2pl long forms coincide with short forms
    # (nuestro/vuestro and variants), so only the long forms unique to
    # 1sg/2sg/3 are listed here.
    "mío": "yo", "mía": "yo", "míos": "yo", "mías": "yo",
    "tuyo": "tú", "tuya": "tú", "tuyos": "tú", "tuyas": "tú",
    "suyo": "él", "suya": "él", "suyos": "él", "suyas": "él",
}

# Spanish POS set for pronoun-form matching.
#
# We include PROPN in addition to PRON and DET because spaCy's Spanish
# models (es_core_news_lg, es_core_news_trf) mis-tag several oblique and
# long-possessive forms as PROPN rather than PRON (verified 2026-04-20):
# *contigo* → PROPN, *tuyo* → PROPN, occasionally *suyo* → PROPN. These
# are genuine model errors. Adding PROPN is safe because the
# ``SPANISH_CASE_TO_NOM`` dict is a strict whitelist — only tokens whose
# lowercase form matches one of the 40-odd listed pronoun forms get
# transformed; real proper nouns (*Pedro*, *García*) are never in the dict.
_ES_PRONOUN_POS = frozenset({"PRON", "DET", "PROPN"})


# ---------------------------------------------------------------------------
# Spanish ablation
# ---------------------------------------------------------------------------

def impoverish_case_es_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """Collapse all Spanish pronoun case forms to nominative.

    The core risk: Spanish *la*, *los*, *las*, *lo* are homographs —
    definite articles (DET, ``PronType=Art``) or direct-object clitics
    (PRON). The shared ``_impoverish_case`` helper skips any token with
    ``PronType=Art``, keeping articles intact and transforming only the
    clitic uses. Similarly, *su*/*sus* are possessive determiners and get
    transformed; *le*/*les* map to masculine-default nominatives
    (*él*/*ellos*) since dative-to-nominative collapsing inherently
    loses gender and grammatical-function information.
    """
    return _impoverish_case(doc, SPANISH_CASE_TO_NOM, _ES_PRONOUN_POS)


def validate_impoverish_case_es(original: str, ablated: str, nlp) -> bool:
    """Validate that non-nominative Spanish pronouns were reduced."""
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    def count_oblique(doc):
        return sum(
            1 for tok in doc
            if tok.pos_ in _ES_PRONOUN_POS
            and tok.lower_ in SPANISH_CASE_TO_NOM
            and SPANISH_CASE_TO_NOM[tok.lower_] != tok.lower_
            and "Art" not in (tok.morph.get("PronType") or [])
        )

    orig_count = count_oblique(original_doc)
    abl_count = count_oblique(ablated_doc)

    if orig_count > 0:
        return abl_count < orig_count
    return True


AblationRegistry.register(
    "impoverish_case_es",
    impoverish_case_es_doc,
    validate_impoverish_case_es,
)
