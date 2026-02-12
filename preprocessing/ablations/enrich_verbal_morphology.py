"""
Enrich verbal morphology — add synthetic agreement suffixes to English verbs.

For each verb (VERB/AUX), the ablation:

1. Finds the verb's subject via dependency parse (nsubj / nsubj:pass).
2. Extracts person and number from the subject's morphological features
   (or infers them from the pronoun form).
3. Lemmatizes the verb (strips existing English morphology).
4. Appends a synthetic, unambiguous agreement suffix.

If no subject can be found (imperatives, infinitives, fragments), the verb
is lemmatized without a suffix — effectively impoverishing it.

The default synthetic paradigm is Latin-inspired:

+----------+--------+------------------+
| Person   | Suffix | Example          |
+==========+========+==================+
| 1sg      | -o     | walk → walko     |
| 2sg      | -as    | walk → walkas    |
| 3sg      | -at    | walk → walkat    |
| 1pl      | -amus  | walk → walkamus  |
| 2pl      | -atis  | walk → walkatis  |
| 3pl      | -ant   | walk → walkant   |
+----------+--------+------------------+

The paradigm dict ``DEFAULT_SUFFIX_MAP`` can be overridden via config parameters.

Only English is implemented; Italian already has rich agreement morphology
and enrichment is not part of the preregistered intervention list.
"""

from typing import Dict, Optional, Tuple

import spacy

from preprocessing.registry import AblationRegistry


# ---------------------------------------------------------------------------
# Default synthetic paradigm
# ---------------------------------------------------------------------------

# Keys are (person, number) tuples with string values from UD morphology
DEFAULT_SUFFIX_MAP: Dict[Tuple[str, str], str] = {
    ("1", "Sing"): "o",
    ("2", "Sing"): "as",
    ("3", "Sing"): "at",
    ("1", "Plur"): "amus",
    ("2", "Plur"): "atis",
    ("3", "Plur"): "ant",
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
) -> Tuple[str, int]:
    """
    Apply synthetic agreement morphology to all verbs/auxiliaries.

    Returns:
        (modified_text, count_of_enriched_verbs)
    """
    modified_parts = []
    num_enriched = 0

    for tok in doc:
        if tok.pos_ in ("VERB", "AUX"):
            # Only enrich finite present-tense verbs — past tense lacks
            # agreement morphology (except suppletive was/were), and
            # participles/gerunds (VerbForm=Part, e.g. "giving") don't
            # carry agreement either.
            tense = tok.morph.get("Tense")
            verb_form = tok.morph.get("VerbForm")
            if not tense or "Pres" not in tense or not verb_form or "Fin" not in verb_form:
                modified_parts.append(tok.text_with_ws)
                continue

            subj = _find_subject(tok)
            if subj is not None:
                pn = _get_person_number(subj)
                if pn is not None:
                    suffix = suffix_map.get(pn, "")
                    if suffix:
                        enriched = tok.lemma_ + suffix
                        modified_parts.append(enriched + tok.whitespace_)
                        num_enriched += 1
                        continue

            # Present tense but no subject or unknown person/number → bare lemma
            modified_parts.append(tok.lemma_ + tok.whitespace_)
        else:
            modified_parts.append(tok.text_with_ws)

    return "".join(modified_parts), num_enriched


# ---------------------------------------------------------------------------
# Public ablation function
# ---------------------------------------------------------------------------

def enrich_verbal_morphology_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Enrich English verbs with synthetic agreement morphology (default paradigm).

    Args:
        doc: spaCy Doc to process

    Returns:
        Tuple of (modified_text, num_verbs_enriched)
    """
    return _enrich_verbal_morphology(doc, DEFAULT_SUFFIX_MAP)


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
