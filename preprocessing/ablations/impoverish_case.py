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

from typing import Callable, Dict, Optional, Tuple

import spacy

from preprocessing.registry import AblationRegistry


# ---------------------------------------------------------------------------
# Tier categories — shared across languages. Each case-form is tagged with
# one of these labels so post-run reporting can break the replacement count
# down by morphosyntactic type (accusative clitic vs possessive vs ...).
# ---------------------------------------------------------------------------

_TIER_TONIC_OBLIQUE = "tonic_oblique"           # ES *mí/ti/sí*, IT *me/te*
_TIER_PORTMANTEAU = "portmanteau"               # ES *conmigo/contigo/consigo*
_TIER_ACC_CLITIC = "acc_clitic"                 # direct-object clitics
_TIER_DAT_CLITIC = "dat_clitic"                 # indirect-object clitics (ES)
_TIER_REFLEX_CLITIC = "reflex_clitic"           # IT/ES *si/se* (identity, skipped)
_TIER_ACC_PRONOUN = "acc_pronoun"               # EN *him/her/us/them/me*
_TIER_REFLEX_PRONOUN = "reflex_pronoun"         # EN *-self/-selves*
_TIER_POSS_SHORT = "poss_short"                 # ES *mi/tu/su*, EN *my/your/our*
_TIER_POSS_LONG = "poss_long"                   # ES *mío/tuyo/suyo*, EN *mine/hers*
_TIER_REL_INTERROG = "rel_interrog"             # EN *whom/whose*


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

# Per-form tier labels for English, used for tier_counts reporting.
ENGLISH_CASE_TIERS: Dict[str, str] = {
    # Accusative / object pronouns
    "me": _TIER_ACC_PRONOUN, "him": _TIER_ACC_PRONOUN, "her": _TIER_ACC_PRONOUN,
    "us": _TIER_ACC_PRONOUN, "them": _TIER_ACC_PRONOUN,
    # Reflexives
    "myself": _TIER_REFLEX_PRONOUN, "yourself": _TIER_REFLEX_PRONOUN,
    "yourselves": _TIER_REFLEX_PRONOUN, "himself": _TIER_REFLEX_PRONOUN,
    "herself": _TIER_REFLEX_PRONOUN, "itself": _TIER_REFLEX_PRONOUN,
    "ourselves": _TIER_REFLEX_PRONOUN, "themselves": _TIER_REFLEX_PRONOUN,
    # Possessives — short (determiner) forms
    "my": _TIER_POSS_SHORT, "your": _TIER_POSS_SHORT, "his": _TIER_POSS_SHORT,
    "its": _TIER_POSS_SHORT, "our": _TIER_POSS_SHORT, "their": _TIER_POSS_SHORT,
    # Possessives — long (pronominal) forms
    "mine": _TIER_POSS_LONG, "yours": _TIER_POSS_LONG, "hers": _TIER_POSS_LONG,
    "ours": _TIER_POSS_LONG, "theirs": _TIER_POSS_LONG,
    # Relative / interrogative
    "whom": _TIER_REL_INTERROG, "whose": _TIER_REL_INTERROG,
}


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

# Per-form tier labels for Italian. Possessive gender variants share the
# short/long distinction loosely (Italian largely uses one form either
# pre- or post-nominally); we label them all as short for simplicity and
# reserve poss_long for the English/Spanish distinction.
ITALIAN_CASE_TIERS: Dict[str, str] = {
    # Strong obliques
    "me": _TIER_TONIC_OBLIQUE, "te": _TIER_TONIC_OBLIQUE,
    # Direct-object clitics
    "mi": _TIER_ACC_CLITIC, "ti": _TIER_ACC_CLITIC, "lo": _TIER_ACC_CLITIC,
    "la": _TIER_ACC_CLITIC, "ci": _TIER_ACC_CLITIC, "vi": _TIER_ACC_CLITIC,
    "li": _TIER_ACC_CLITIC, "le": _TIER_ACC_CLITIC,
    # Indirect-object clitic
    "gli": _TIER_DAT_CLITIC,
    # Reflexive (identity — won't count even if POS/form matches)
    "si": _TIER_REFLEX_CLITIC,
    # Possessives (all forms treated as short)
    "mio": _TIER_POSS_SHORT, "mia": _TIER_POSS_SHORT,
    "miei": _TIER_POSS_SHORT, "mie": _TIER_POSS_SHORT,
    "tuo": _TIER_POSS_SHORT, "tua": _TIER_POSS_SHORT,
    "tuoi": _TIER_POSS_SHORT, "tue": _TIER_POSS_SHORT,
    "suo": _TIER_POSS_SHORT, "sua": _TIER_POSS_SHORT,
    "suoi": _TIER_POSS_SHORT, "sue": _TIER_POSS_SHORT,
    "nostro": _TIER_POSS_SHORT, "nostra": _TIER_POSS_SHORT,
    "nostri": _TIER_POSS_SHORT, "nostre": _TIER_POSS_SHORT,
    "vostro": _TIER_POSS_SHORT, "vostra": _TIER_POSS_SHORT,
    "vostri": _TIER_POSS_SHORT, "vostre": _TIER_POSS_SHORT,
}


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


def _impoverish_case_token_edits(
    doc: spacy.tokens.Doc,
    mapping: Dict[str, str],
    target_pos: frozenset,
) -> Dict[int, str]:
    """
    Compute the per-token replacement plan for case impoverishment.

    Returns a mapping ``token.i -> replacement_text`` containing ONLY the
    tokens that actually get replaced: POS in *target_pos*, ``lower_`` in
    *mapping*, not a definite-article homograph (``PronType=Art``), and
    not an identity mapping (e.g. reflexive "si" -> "si"). This is the
    same token-selection logic as the original ``_impoverish_case`` loop,
    including the ``_match_capitalization`` behavior and the contraction-
    glue leading/trailing-space fix.

    This is the single source of truth for the edit computation; both
    ``_impoverish_case`` (used by the stateless per-language ``*_doc``
    functions) and ``CaseImpoverisher.__call__`` are reimplemented on top
    of it, and a stacked ablation combinator can call it directly to
    merge edits from multiple ablations on the same parsed Doc.
    """
    edits: Dict[int, str] = {}

    for i, tok in enumerate(doc):
        lower = tok.lower_
        if tok.pos_ not in target_pos or lower not in mapping:
            continue
        # Skip definite articles — PronType=Art distinguishes e.g.
        # Italian "lo" (article, DET) from "lo" (clitic pronoun, PRON).
        pron_type = tok.morph.get("PronType")
        if pron_type and "Art" in pron_type:
            continue
        nom = mapping[lower]
        # Skip identity mappings (e.g. "si" → "si")
        if nom == lower:
            continue
        replacement = _match_capitalization(nom, tok.text)
        # Contraction-glue fix: if the original token is glued to a
        # neighbour (e.g. Spanish "dímelo" = ["dí", "me", "lo"]
        # with empty whitespace_ between them, or English "it's" =
        # ["it", "'s"] — although impoverish_case doesn't touch
        # "'s" since it's AUX not PRON), re-insert whitespace so
        # the substituted form doesn't concatenate into a
        # pseudo-token.
        leading = ""
        if i > 0 and doc[i - 1].whitespace_ == "":
            leading = " "
        trailing_extra = " " if tok.whitespace_ == "" else ""
        edits[i] = leading + replacement + trailing_extra

    return edits


def _impoverish_case(
    doc: spacy.tokens.Doc,
    mapping: Dict[str, str],
    target_pos: frozenset,
    tier_map: Optional[Dict[str, str]] = None,
    tier_counter: Optional[Dict[str, int]] = None,
) -> Tuple[str, int]:
    """
    Core token-replacement logic shared by all languages.

    For each token whose ``lower_`` form appears in *mapping* and whose POS
    is in *target_pos*, replace it with the nominative form (preserving
    capitalization and whitespace).

    If *tier_counter* is provided (a mutable dict), increments it per
    replaced form using categories from *tier_map*. The caller owns the
    dict — typically a ``CaseImpoverisher`` instance that aggregates
    counts across an entire file.

    Built on top of ``_impoverish_case_token_edits``: the edit plan is
    computed once, the output text is reconstructed by joining each
    token's replacement (or its own text, if unchanged) with the token's
    own trailing whitespace, and tier bookkeeping is derived from the
    same plan (in token order, matching the original loop's bookkeeping
    order exactly).
    """
    edits = _impoverish_case_token_edits(doc, mapping, target_pos)
    result = "".join(edits.get(tok.i, tok.text) + tok.whitespace_ for tok in doc)

    if tier_counter is not None and tier_map is not None:
        for i in edits:
            tier = tier_map.get(doc[i].lower_, "other")
            tier_counter[tier] = tier_counter.get(tier, 0) + 1

    return result, len(edits)


class CaseImpoverisher:
    """Stateful callable wrapping a per-language case-impoverishment run.

    Exposes ``reset_file_state()`` and ``get_file_tier_counts()`` so the
    ``AblationPipeline`` can surface per-file tier breakdowns in the
    ``ABLATION_MANIFEST.json``.
    """

    def __init__(
        self,
        name: str,
        mapping: Dict[str, str],
        target_pos: frozenset,
        tier_map: Dict[str, str],
    ) -> None:
        self.name = name
        self._mapping = mapping
        self._target_pos = target_pos
        self._tier_map = tier_map
        self._file_tier_counts: Dict[str, int] = {}

    def reset_file_state(self) -> None:
        """Reset per-file tier counter."""
        self._file_tier_counts = {}

    def get_file_tier_counts(self) -> Dict[str, int]:
        """Return a copy of the per-file tier counts."""
        return dict(self._file_tier_counts)

    def token_edits(self, doc: spacy.tokens.Doc) -> Dict[int, str]:
        """Per-token replacement plan for this instance's (mapping,
        target_pos). Exposed so a stacked ablation combinator can call
        ``token_edits`` uniformly on the registered ablation object."""
        return _impoverish_case_token_edits(doc, self._mapping, self._target_pos)

    def __call__(self, doc: spacy.tokens.Doc) -> Tuple[str, int]:
        return _impoverish_case(
            doc,
            mapping=self._mapping,
            target_pos=self._target_pos,
            tier_map=self._tier_map,
            tier_counter=self._file_tier_counts,
        )


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
# Registration — stateful CaseImpoverisher instances so the pipeline can
# pull per-file tier_counts from `get_file_tier_counts()` (same pattern
# as EnglishExpletiveSentenceRemover).
# ---------------------------------------------------------------------------

AblationRegistry.register(
    "impoverish_case_en",
    CaseImpoverisher(
        name="impoverish_case_en",
        mapping=ENGLISH_CASE_TO_NOM,
        target_pos=_EN_PRONOUN_POS,
        tier_map=ENGLISH_CASE_TIERS,
    ),
    validate_impoverish_case_en,
)

AblationRegistry.register(
    "impoverish_case_it",
    CaseImpoverisher(
        name="impoverish_case_it",
        mapping=ITALIAN_CASE_TO_NOM,
        target_pos=_IT_PRONOUN_POS,
        tier_map=ITALIAN_CASE_TIERS,
    ),
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

# Per-form tier labels for Spanish.
SPANISH_CASE_TIERS: Dict[str, str] = {
    # Tonic obliques
    "mí": _TIER_TONIC_OBLIQUE, "ti": _TIER_TONIC_OBLIQUE, "sí": _TIER_TONIC_OBLIQUE,
    # Portmanteaux (preposition-bound special forms)
    "conmigo": _TIER_PORTMANTEAU, "contigo": _TIER_PORTMANTEAU, "consigo": _TIER_PORTMANTEAU,
    # Direct-object (accusative) clitics
    "me": _TIER_ACC_CLITIC, "te": _TIER_ACC_CLITIC,
    "lo": _TIER_ACC_CLITIC, "la": _TIER_ACC_CLITIC,
    "nos": _TIER_ACC_CLITIC, "os": _TIER_ACC_CLITIC,
    "los": _TIER_ACC_CLITIC, "las": _TIER_ACC_CLITIC,
    # Indirect-object (dative) clitics
    "le": _TIER_DAT_CLITIC, "les": _TIER_DAT_CLITIC,
    # Reflexive (identity mapping; won't count toward replacements)
    "se": _TIER_REFLEX_CLITIC,
    # Possessives — short (prenominal)
    "mi": _TIER_POSS_SHORT, "mis": _TIER_POSS_SHORT,
    "tu": _TIER_POSS_SHORT, "tus": _TIER_POSS_SHORT,
    "su": _TIER_POSS_SHORT, "sus": _TIER_POSS_SHORT,
    "nuestro": _TIER_POSS_SHORT, "nuestra": _TIER_POSS_SHORT,
    "nuestros": _TIER_POSS_SHORT, "nuestras": _TIER_POSS_SHORT,
    "vuestro": _TIER_POSS_SHORT, "vuestra": _TIER_POSS_SHORT,
    "vuestros": _TIER_POSS_SHORT, "vuestras": _TIER_POSS_SHORT,
    # Possessives — long (postnominal/predicative)
    "mío": _TIER_POSS_LONG, "mía": _TIER_POSS_LONG,
    "míos": _TIER_POSS_LONG, "mías": _TIER_POSS_LONG,
    "tuyo": _TIER_POSS_LONG, "tuya": _TIER_POSS_LONG,
    "tuyos": _TIER_POSS_LONG, "tuyas": _TIER_POSS_LONG,
    "suyo": _TIER_POSS_LONG, "suya": _TIER_POSS_LONG,
    "suyos": _TIER_POSS_LONG, "suyas": _TIER_POSS_LONG,
}


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
    CaseImpoverisher(
        name="impoverish_case_es",
        mapping=SPANISH_CASE_TO_NOM,
        target_pos=_ES_PRONOUN_POS,
        tier_map=SPANISH_CASE_TIERS,
    ),
    validate_impoverish_case_es,
)
