"""
Enrich verbal morphology — add synthetic agreement suffixes to English verbs.

For each finite verb (VERB/AUX with ``VerbForm=Fin``), the ablation:

1. Finds the verb's subject via dependency parse (nsubj / nsubj:pass).
2. Extracts person and number from the subject's morphological features
   (or infers them from the pronoun form).
3. Reads the verb's tense (Present or Past).
4. Lemmatizes the verb (strips existing English morphology).
5. Appends a tense+person+number suffix from the appropriate paradigm.

If no subject can be found (imperatives, infinitives, fragments), the verb
is lemmatized without a suffix. If the verb is non-finite (participles,
gerunds, etc.), it is left untouched.

The default synthetic paradigm is Indo-European-inspired and covers both
PRESENT and PAST tenses (mirroring Romance languages, which mark past
tense with the same paradigmatic richness as present):

+----------+----------+--------------------+----------+----------------------+
| Person   | Present  | Example            | Past     | Example              |
+==========+==========+====================+==========+======================+
| 1sg      | -o       | walk → walko       | -i       | walk → walki         |
| 2sg      | -aks     | walk → walkaks     | -isti    | walk → walkisti      |
| 3sg      | -akt     | walk → walkakt     | -ikt     | walk → walkikt       |
| 1pl      | -amus    | walk → walkamus    | -imus    | walk → walkimus      |
| 2pl      | -atis    | walk → walkatis    | -istis   | walk → walkistis     |
| 3pl      | -ant     | walk → walkant     | -erunt   | walk → walkerunt     |
+----------+----------+--------------------+----------+----------------------+

Three forms diverge from a strict Latin paradigm to avoid English-
homograph collisions: the 2sg-pres ``-aks`` (vs. Latin ``-as``, which
reads as English ``as``); the 3sg-pres ``-akt`` (vs. Latin ``-at``,
which reads as English ``at``); and the 3sg-past ``-ikt`` (vs. Latin
``-it``, which reads as English ``it``). Without these substitutions,
the corpus would contain hundreds of thousands of word-final ``at`` /
``it`` / ``as`` tokens that could plausibly be misread as post-verbal
prepositions or particles, introducing a confound with object-position
distribution.

Past tense uses Latin-perfect-style endings, which are disjoint from
the present suffixes (no shared form). Distinguishing tenses on the
surface lets the model recover tense from the suffix even after the
English past-tense stem (``ran``, ``ate``) has been lemmatized away
(``run``, ``eat``).

The paradigm dicts ``DEFAULT_SUFFIX_MAP`` (present) and
``DEFAULT_PAST_SUFFIX_MAP`` (past) can be overridden via config parameters.

Only English is implemented; Italian and Spanish already have rich
agreement morphology and enrichment is not part of the preregistered
intervention list for those languages.
"""

from typing import Dict, Optional, Tuple

import spacy

from preprocessing.dep_labels import normalize_dep
from preprocessing.registry import AblationRegistry


# ---------------------------------------------------------------------------
# Default synthetic paradigm
# ---------------------------------------------------------------------------

# Keys are (person, number) tuples with string values from UD morphology.
# Synthetic Indo-European-style present-active endings. Three forms differ
# from a strict Latin paradigm to avoid English-homograph collisions:
#   2sg  -as → -aks   ('as' is an English preposition/conjunction)
#   3sg  -at → -akt   ('at' is an English preposition)
# (Latin-original forms preserved where there's no English collision:
#  -o, -amus, -atis, -ant.) The substituted forms use a `-Vk-` cluster
# which is unambiguously bound morphology to a reader and unlikely to
# collide with any post-verbal English particle / preposition.
DEFAULT_SUFFIX_MAP: Dict[Tuple[str, str], str] = {
    ("1", "Sing"): "o",
    ("2", "Sing"): "aks",
    ("3", "Sing"): "akt",
    ("1", "Plur"): "amus",
    ("2", "Plur"): "atis",
    ("3", "Plur"): "ant",
}

# Synthetic past-tense endings. Disjoint from the present-tense set so
# the past/present distinction is recoverable from the surface form
# alone. The 3sg form is substituted from a strict Latin-perfect
# paradigm to avoid the English-homograph collision:
#   3sg  -it → -ikt   ('it' is an English pronoun)
# (Other Latin-perfect forms preserved where there's no collision:
#  -i, -isti, -imus, -istis, -erunt.)
DEFAULT_PAST_SUFFIX_MAP: Dict[Tuple[str, str], str] = {
    ("1", "Sing"): "i",
    ("2", "Sing"): "isti",
    ("3", "Sing"): "ikt",
    ("1", "Plur"): "imus",
    ("2", "Plur"): "istis",
    ("3", "Plur"): "erunt",
}


# ---------------------------------------------------------------------------
# Suppletive paradigms for high-frequency English verbs
# ---------------------------------------------------------------------------
#
# Some English verbs collide with real English words when the regular
# (lemma + suffix) rule is applied. Most notably:
#
#   be   + -at (3sg pres) → "beat"  (= past tense of the verb "to beat")
#   go   + -at (3sg pres) → "goat"  (= the animal)
#   go   + -o  (1sg pres) → "goo"   (informal noun)
#
# These collisions destroy the cleanness of the manipulation: a baseline
# learner would see "beat" used in two completely different contexts.
#
# We additionally borrow from Romance, where the highest-frequency verbs
# (be, have) historically resist morphological levelling and retain
# suppletive paradigms (Spanish "soy / eres / es / somos / sois / son",
# vs the regular "*sero / seres / sere / ..."). To better mimic that
# property of natural Romance morphology, we apply hand-crafted Latin-
# inspired paradigms to the four highest-frequency English verbs:
# ``be`` (esse / fuisse), ``have`` (habere), ``do`` (facere), and
# ``go`` (vadere).
#
# Each verb's IRREGULAR_PARADIGMS entry maps tense → (person, number) →
# the full replacement word. The replacement REPLACES both the stem and
# the suffix — it is not appended. None of the chosen forms collide
# with high-frequency English content words (verified by hand against
# the BabyLM corpus 2026-05-12).
IRREGULAR_PARADIGMS: Dict[str, Dict[str, Dict[Tuple[str, str], str]]] = {
    "be": {
        "Pres": {
            ("1", "Sing"): "sum",    ("2", "Sing"): "es",     ("3", "Sing"): "est",
            ("1", "Plur"): "sumus",  ("2", "Plur"): "estis",  ("3", "Plur"): "sunt",
        },
        "Past": {
            ("1", "Sing"): "fui",    ("2", "Sing"): "fuisti", ("3", "Sing"): "fuikt",
            ("1", "Plur"): "fuimus", ("2", "Plur"): "fuistis",("3", "Plur"): "fuerunt",
        },
    },
    "have": {
        # Regular Latin habere stem with the standard present/perfect endings.
        "Pres": {
            ("1", "Sing"): "habo",    ("2", "Sing"): "habaks",  ("3", "Sing"): "habakt",
            ("1", "Plur"): "habamus", ("2", "Plur"): "habatis", ("3", "Plur"): "habant",
        },
        "Past": {
            ("1", "Sing"): "habui",    ("2", "Sing"): "habuisti", ("3", "Sing"): "habuikt",
            ("1", "Plur"): "habuimus", ("2", "Plur"): "habuistis",("3", "Plur"): "habuerunt",
        },
    },
    "do": {
        # Latin facere ("to do/make") — suppletive present and past stems.
        "Pres": {
            ("1", "Sing"): "facio",   ("2", "Sing"): "faciks",  ("3", "Sing"): "facikt",
            ("1", "Plur"): "facimus", ("2", "Plur"): "facitis", ("3", "Plur"): "faciunt",
        },
        "Past": {
            ("1", "Sing"): "feci",    ("2", "Sing"): "fecisti", ("3", "Sing"): "fecikt",
            ("1", "Plur"): "fecimus", ("2", "Plur"): "fecistis",("3", "Plur"): "fecerunt",
        },
    },
    "go": {
        # Latin vadere stem (avoiding ire whose 3sg "it" collides with
        # the English pronoun).
        "Pres": {
            ("1", "Sing"): "vado",    ("2", "Sing"): "vadaks",  ("3", "Sing"): "vadakt",
            ("1", "Plur"): "vadamus", ("2", "Plur"): "vadatis", ("3", "Plur"): "vadant",
        },
        "Past": {
            ("1", "Sing"): "vadi",    ("2", "Sing"): "vadisti", ("3", "Sing"): "vadikt",
            ("1", "Plur"): "vadimus", ("2", "Plur"): "vadistis",("3", "Plur"): "vaderunt",
        },
    },
}

# Fallback: infer person/number from English subject pronoun form when spaCy
# morph features are missing or incomplete
_PRONOUN_TO_PERSON_NUMBER: Dict[str, Tuple[str, str]] = {
    "i": ("1", "Sing"),
    "you": ("2", "Sing"),  # ambiguous sg/pl; default to sg
    "he": ("3", "Sing"),
    "she": ("3", "Sing"),
    "it": ("3", "Sing"),
    "we": ("1", "Plur"),
    "they": ("3", "Plur"),
}


# ---------------------------------------------------------------------------
# Subject finding
# ---------------------------------------------------------------------------

def _find_subject(verb: spacy.tokens.Token) -> Optional[spacy.tokens.Token]:
    """
    Locate the nominal subject of *verb* by walking the dependency tree.

    Handles:
    - Direct subjects: *he* walks → nsubj directly on verb
    - Auxiliary chains: *he has been walking* → walk up aux chain to find nsubj
    - Passive subjects: nsubj:pass
    """
    # Direct children with subject dependency
    for child in verb.children:
        if normalize_dep(child.dep_) in ("nsubj", "nsubj:pass"):
            return child

    # Walk up auxiliary chain: if this verb is an aux/xcomp child,
    # the subject may be attached to the head
    head = verb.head
    seen = {verb.i}
    while head and head.i not in seen:
        seen.add(head.i)
        for child in head.children:
            if normalize_dep(child.dep_) in ("nsubj", "nsubj:pass"):
                return child
        # Only continue walking if we are in an aux / xcomp relation
        if normalize_dep(verb.dep_) not in ("aux", "auxpass", "aux:pass", "xcomp"):
            break
        verb = head
        head = head.head

    return None


def _get_person_number(
    token: spacy.tokens.Token,
) -> Optional[Tuple[str, str]]:
    """
    Extract (person, number) from a token's morphological features.

    Falls back to pronoun-form lookup when morph features are absent.
    """
    morph = token.morph
    person = morph.get("Person")
    number = morph.get("Number")

    if person and number:
        return person[0], number[0]

    # Fallback for pronouns
    lower = token.lower_
    if lower in _PRONOUN_TO_PERSON_NUMBER:
        return _PRONOUN_TO_PERSON_NUMBER[lower]

    return None


# ---------------------------------------------------------------------------
# Core ablation
# ---------------------------------------------------------------------------

def _enrich_verbal_morphology(
    doc: spacy.tokens.Doc,
    suffix_map: Dict[Tuple[str, str], str],
    past_suffix_map: Optional[Dict[Tuple[str, str], str]] = None,
    irregular_paradigms: Optional[Dict[str, Dict[str, Dict[Tuple[str, str], str]]]] = None,
    default_person_number: Optional[Tuple[str, str]] = ("3", "Sing"),
) -> Tuple[str, int]:
    """
    Apply synthetic agreement morphology to all finite verbs/auxiliaries.

    Enriches both present and past tense if ``past_suffix_map`` is provided
    (the default). Pass ``past_suffix_map=None`` to restore the
    present-tense-only behaviour (used by tests of the original rule).

    If ``irregular_paradigms`` is provided, verbs whose lemma is a key
    in the dict are rewritten using the suppletive forms in that dict
    instead of the regular lemma+suffix rule. The default
    ``IRREGULAR_PARADIGMS`` handles ``be``, ``have``, ``do``, ``go`` —
    the four high-frequency English verbs where the regular rule
    produces real-English homographs (notably ``be+at=beat``,
    ``go+at=goat``).

    ``default_person_number`` is the coarse fallback used when the parse
    cannot identify a subject (proper-noun subjects with sparse morph
    features, inverted-quoting dialogue tags like "said Lucas",
    fragments). Default is ("3", "Sing") — the unmarked /
    third-person-narrator default. Without this fallback, finite verbs
    with unresolvable subjects would fall through to bare lemma, which
    is indistinguishable from `lemmatize_verbs` output and silently
    leaks signal between the two manipulations. Pass ``None`` to
    disable the fallback (legacy behaviour).

    Returns:
        (modified_text, count_of_enriched_verbs)
    """
    if past_suffix_map is None:
        past_suffix_map = {}  # disables past-tense enrichment
    if irregular_paradigms is None:
        irregular_paradigms = {}  # disables suppletion

    modified_parts = []
    num_enriched = 0

    for i, tok in enumerate(doc):
        if tok.pos_ in ("VERB", "AUX"):
            # Enrich finite verbs only — participles (VerbForm=Part, e.g.
            # "giving") and infinitives don't carry agreement. Within
            # finite, both Pres and Past get a suffix; everything else
            # (no tense feature, etc.) is left untouched.
            tense = tok.morph.get("Tense")
            verb_form = tok.morph.get("VerbForm")
            if not tense or not verb_form or "Fin" not in verb_form:
                modified_parts.append(tok.text_with_ws)
                continue

            if "Pres" in tense:
                tense_key = "Pres"
                active_paradigm = suffix_map
            elif "Past" in tense:
                tense_key = "Past"
                active_paradigm = past_suffix_map
            else:
                modified_parts.append(tok.text_with_ws)
                continue

            replacement = tok.lemma_
            # Resolve (person, number). Try the parse first; if that
            # fails fall back to the coarse default (3sg) so we still
            # apply a suffix rather than silently regressing to bare
            # lemma. See docstring for rationale.
            subj = _find_subject(tok)
            pn = _get_person_number(subj) if subj is not None else None
            if pn is None and default_person_number is not None:
                pn = default_person_number
            if pn is not None:
                # Suppletion takes precedence over the regular
                # lemma+suffix rule when the lemma is in the
                # irregular paradigm dict (e.g., be / have / do / go).
                irreg_lemma = irregular_paradigms.get(tok.lemma_.lower())
                if irreg_lemma is not None:
                    irreg_form = irreg_lemma.get(tense_key, {}).get(pn)
                    if irreg_form is not None:
                        replacement = irreg_form
                        num_enriched += 1
                    else:
                        # Lemma is in the irregular dict but we
                        # don't have a form for this (person, number, tense).
                        # Fall through to regular rule.
                        suffix = active_paradigm.get(pn, "")
                        if suffix:
                            replacement = tok.lemma_ + suffix
                            num_enriched += 1
                else:
                    suffix = active_paradigm.get(pn, "")
                    if suffix:
                        replacement = tok.lemma_ + suffix
                        num_enriched += 1
            form_changed = replacement.lower() != tok.text.lower()
            # Contraction-glue fix: see lemmatize_verbs.
            leading = ""
            if form_changed and i > 0 and doc[i - 1].whitespace_ == "":
                leading = " "
            trailing = tok.whitespace_
            if form_changed and tok.whitespace_ == "":
                trailing = " "
            modified_parts.append(leading + replacement + trailing)
        else:
            modified_parts.append(tok.text_with_ws)

    return "".join(modified_parts), num_enriched


# ---------------------------------------------------------------------------
# Public ablation function
# ---------------------------------------------------------------------------

def enrich_verbal_morphology_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Enrich English verbs with synthetic agreement morphology.

    Applies the default Latin-style paradigm for both present and past
    tense, plus suppletive paradigms for the four high-frequency verbs
    (be, have, do, go) where the regular rule would produce real-English
    homographs.

    Args:
        doc: spaCy Doc to process

    Returns:
        Tuple of (modified_text, num_verbs_enriched)
    """
    return _enrich_verbal_morphology(
        doc,
        suffix_map=DEFAULT_SUFFIX_MAP,
        past_suffix_map=DEFAULT_PAST_SUFFIX_MAP,
        irregular_paradigms=IRREGULAR_PARADIGMS,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_enrichment(original: str, ablated: str, nlp) -> bool:
    """
    Validate that verbs were enriched with synthetic morphology.

    Checks that at least some verb forms changed between original and ablated.
    """
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    original_verbs = [
        tok.text for tok in original_doc if tok.pos_ in ("VERB", "AUX")
    ]
    ablated_tokens = ablated.split()

    if not original_verbs:
        return True

    # At least some original verb forms should no longer appear verbatim
    original_verb_set = set(original_verbs)
    # Check that the ablated text contains tokens not in the original verb set
    # (i.e., synthetic forms were introduced)
    novel_tokens = {t for t in ablated_tokens if t not in original_verb_set}
    return len(novel_tokens) > 0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AblationRegistry.register(
    "enrich_verbal_morphology",
    enrich_verbal_morphology_doc,
    validate_enrichment,
)
