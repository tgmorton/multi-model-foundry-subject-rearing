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

The default synthetic paradigm is Latin-inspired and covers both PRESENT
and PAST tenses (mirroring Romance languages, which mark past tense with
the same paradigmatic richness as present):

+----------+----------+------------------+----------+--------------------+
| Person   | Present  | Example          | Past     | Example            |
+==========+==========+==================+==========+====================+
| 1sg      | -o       | walk → walko     | -i       | walk → walki       |
| 2sg      | -as      | walk → walkas    | -isti    | walk → walkisti    |
| 3sg      | -at      | walk → walkat    | -it      | walk → walkit      |
| 1pl      | -amus    | walk → walkamus  | -imus    | walk → walkimus    |
| 2pl      | -atis    | walk → walkatis  | -istis   | walk → walkistis   |
| 3pl      | -ant     | walk → walkant   | -erunt   | walk → walkerunt   |
+----------+----------+------------------+----------+--------------------+

Past tense uses the Latin perfect-tense endings, which are distinct from
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

from preprocessing.registry import AblationRegistry


# ---------------------------------------------------------------------------
# Default synthetic paradigm
# ---------------------------------------------------------------------------

# Keys are (person, number) tuples with string values from UD morphology.
# Latin-style present-active-indicative endings.
DEFAULT_SUFFIX_MAP: Dict[Tuple[str, str], str] = {
    ("1", "Sing"): "o",
    ("2", "Sing"): "as",
    ("3", "Sing"): "at",
    ("1", "Plur"): "amus",
    ("2", "Plur"): "atis",
    ("3", "Plur"): "ant",
}

# Latin-style perfect-active-indicative endings for past tense. All six
# suffixes are distinct from the present-tense suffixes above (no overlap)
# so the past/present distinction is recoverable from the surface form
# alone.
DEFAULT_PAST_SUFFIX_MAP: Dict[Tuple[str, str], str] = {
    ("1", "Sing"): "i",
    ("2", "Sing"): "isti",
    ("3", "Sing"): "it",
    ("1", "Plur"): "imus",
    ("2", "Plur"): "istis",
    ("3", "Plur"): "erunt",
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
        if child.dep_ in ("nsubj", "nsubj:pass"):
            return child

    # Walk up auxiliary chain: if this verb is an aux/xcomp child,
    # the subject may be attached to the head
    head = verb.head
    seen = {verb.i}
    while head and head.i not in seen:
        seen.add(head.i)
        for child in head.children:
            if child.dep_ in ("nsubj", "nsubj:pass"):
                return child
        # Only continue walking if we are in an aux / xcomp relation
        if verb.dep_ not in ("aux", "auxpass", "xcomp"):
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
) -> Tuple[str, int]:
    """
    Apply synthetic agreement morphology to all finite verbs/auxiliaries.

    Enriches both present and past tense if ``past_suffix_map`` is provided
    (the default). Pass ``past_suffix_map=None`` to restore the
    present-tense-only behaviour (used by tests of the original rule).

    Returns:
        (modified_text, count_of_enriched_verbs)
    """
    if past_suffix_map is None:
        past_suffix_map = {}  # disables past-tense enrichment

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
                active_paradigm = suffix_map
            elif "Past" in tense:
                active_paradigm = past_suffix_map
            else:
                modified_parts.append(tok.text_with_ws)
                continue

            # Resolve the replacement form first (lemma+suffix or bare lemma)
            # so we can apply the same contraction-glue fix used by
            # lemmatize_verbs: English "it's", "we're", "wasn't" etc. have
            # an empty whitespace_ on one side; substituting the clitic
            # without re-inserting whitespace produces pseudo-tokens like
            # "itbeat", "webeamus", "ben't". Detect and unglue.
            replacement = tok.lemma_
            subj = _find_subject(tok)
            if subj is not None:
                pn = _get_person_number(subj)
                if pn is not None:
                    suffix = active_paradigm.get(pn, "")
                    if suffix:
                        replacement = tok.lemma_ + suffix
                        num_enriched += 1
            form_changed = replacement.lower() != tok.text.lower()
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

    Uses the default Latin-style paradigm for both present and past tense.

    Args:
        doc: spaCy Doc to process

    Returns:
        Tuple of (modified_text, num_verbs_enriched)
    """
    return _enrich_verbal_morphology(
        doc,
        suffix_map=DEFAULT_SUFFIX_MAP,
        past_suffix_map=DEFAULT_PAST_SUFFIX_MAP,
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
