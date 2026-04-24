"""
Lemmatize all verbs to their infinitive form.

This ablation reduces all verbs to their base lemma form (e.g., "running" -> "run",
"went" -> "go"). This tests how models learn when verb morphology is impoverished.

Both POS=VERB (lexical verbs) and POS=AUX (auxiliaries and copulas) are
lemmatized, since auxiliaries carry the same agreement morphology that
lexical verbs do (*he is*/*we are*, *hablo*/*hablamos*, *estoy*/*estamos*).
Preserving auxiliaries would leak person/number/tense information that
defeats the purpose of the impoverishment. Covers English *is/are/was/were
→ be*, *has/have/had → have*, and Spanish/Italian copulas and
perfect/progressive auxiliaries.
"""

from typing import Dict, Tuple
import spacy
from preprocessing.registry import AblationRegistry


_TARGET_POS = frozenset({"VERB", "AUX"})


def lemmatize_verbs_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Lemmatize all verbs and auxiliaries to their lemma in a spaCy Doc.

    Stateless version — does not record tier counts. The registered
    ablation is a ``VerbLemmatizer`` instance below, which delegates to
    this function but tracks per-file ``{verb, aux}`` counts.

    Args:
        doc: spaCy Doc object containing the text to process

    Returns:
        Tuple of (ablated_text, num_lemmatized)
    """
    modified_parts = []
    num_lemmatized = 0

    for i, token in enumerate(doc):
        if token.pos_ in _TARGET_POS:
            # Only count as lemmatized if the form actually changes —
            # e.g. "be" is its own lemma and "running" is not.
            form_changed = token.lemma_.lower() != token.text.lower()
            if form_changed:
                num_lemmatized += 1
            # Contraction-glue fix: if the surface token is glued to its
            # neighbours (e.g. "it's" = ["it" (ws=""), "'s"] or "wasn't" =
            # ["was" (ws=""), "n't"]) and we're replacing it with a
            # free-standing word ("be"/"have"/"will"/…), concatenation
            # produces pseudo-tokens like "itbe", "ben't". Force leading
            # and trailing spaces when unglung is needed.
            leading = ""
            if (
                form_changed
                and i > 0
                and doc[i - 1].whitespace_ == ""
            ):
                leading = " "
            trailing = token.whitespace_
            if form_changed and token.whitespace_ == "":
                trailing = " "
            modified_parts.append(leading + token.lemma_ + trailing)
        else:
            modified_parts.append(token.text_with_ws)

    result = ''.join(modified_parts)
    return result, num_lemmatized


class VerbLemmatizer:
    """Stateful callable that lemmatizes verbs + auxiliaries and records
    per-file tier counts ``{verb: N, aux: M}``.

    Exposes ``reset_file_state()`` and ``get_file_tier_counts()`` so the
    ``AblationPipeline`` surfaces the VERB vs AUX split in
    ``ABLATION_MANIFEST.json``.
    """

    def __init__(self) -> None:
        self._file_tier_counts: Dict[str, int] = {}

    def reset_file_state(self) -> None:
        self._file_tier_counts = {}

    def get_file_tier_counts(self) -> Dict[str, int]:
        return dict(self._file_tier_counts)

    def __call__(self, doc: spacy.tokens.Doc) -> Tuple[str, int]:
        modified_parts = []
        num_lemmatized = 0

        for i, token in enumerate(doc):
            if token.pos_ in _TARGET_POS:
                form_changed = token.lemma_.lower() != token.text.lower()
                if form_changed:
                    num_lemmatized += 1
                    tier = "aux" if token.pos_ == "AUX" else "verb"
                    self._file_tier_counts[tier] = (
                        self._file_tier_counts.get(tier, 0) + 1
                    )
                # See contraction-glue fix in `lemmatize_verbs_doc`.
                leading = ""
                if (
                    form_changed
                    and i > 0
                    and doc[i - 1].whitespace_ == ""
                ):
                    leading = " "
                trailing = token.whitespace_
                if form_changed and token.whitespace_ == "":
                    trailing = " "
                modified_parts.append(leading + token.lemma_ + trailing)
            else:
                modified_parts.append(token.text_with_ws)

        return "".join(modified_parts), num_lemmatized


def validate_verb_lemmatization(original: str, ablated: str, nlp) -> bool:
    """
    Validate that verbs (including auxiliaries) were actually lemmatized.

    Checks that verb forms changed between original and ablated text.

    Args:
        original: Original text before ablation
        ablated: Text after ablation
        nlp: spaCy pipeline for analysis

    Returns:
        True if verbs were found and lemmatized, False otherwise
    """
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    original_verbs = [token.text for token in original_doc if token.pos_ in _TARGET_POS]
    ablated_verbs = [token.text for token in ablated_doc if token.pos_ in _TARGET_POS]

    if original_verbs:
        # Check if any verbs were lemmatized (different forms)
        original_verb_forms = set(original_verbs)
        ablated_verb_forms = set(ablated_verbs)
        lemmatized_count = len(original_verb_forms - ablated_verb_forms)
        return lemmatized_count > 0
    else:
        # No verbs found - that's okay
        return True


# Register the ablation with the registry — stateful instance so the
# pipeline can pull per-file verb/aux tier counts.
AblationRegistry.register(
    "lemmatize_verbs",
    VerbLemmatizer(),
    validate_verb_lemmatization,
)
