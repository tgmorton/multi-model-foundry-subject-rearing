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

# simplemma is used for Spanish lemmatization. spaCy's es_core_news_lg
# hallucinates non-existent verb stems for ~14% of stem-changing irregular
# forms (e.g., harías → *hariar, tendrías → *tendriar) — see the audit at
# scripts/lemma_compare.py and docs/eval_stimuli/notebook.md §3 (2026-05-12).
# simplemma's errors are higher-recall surface-form pass-throughs rather
# than hallucinated stems, which preserves the cleanness of the ablation.
# spaCy's English lemmatizer (en_core_web_trf) is used for English; the
# audit showed it is accurate for English verb forms.
try:
    import simplemma
    _SIMPLEMMA_AVAILABLE = True
except ImportError:
    _SIMPLEMMA_AVAILABLE = False


_TARGET_POS = frozenset({"VERB", "AUX"})


def _resolve_lemma(token: spacy.tokens.Token) -> str:
    """Return the lemma for ``token``, preferring simplemma for Spanish.

    For Spanish tokens (``token.doc.lang_ == "es"``), uses simplemma if
    it's installed. For everything else (English, missing simplemma),
    falls back to spaCy's built-in ``token.lemma_``.
    """
    if _SIMPLEMMA_AVAILABLE and token.doc.lang_ == "es":
        return simplemma.lemmatize(token.text, lang="es")
    return token.lemma_


def _lemmatize_verbs_edits(doc: spacy.tokens.Doc) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Compute the per-token replacement plan for verb/aux lemmatization.

    Returns ``(edits, counted)``:

    - ``edits``: ``token.i -> replacement_text`` for every target-POS
      token whose OUTPUT TEXT differs from its surface form (exact
      string comparison). Tokens not present are left untouched by the
      reconstruction formula below.
    - ``counted``: ``token.i -> tier`` (``"verb"`` or ``"aux"``) for the
      subset of edited tokens that count as a genuine lemmatization —
      i.e. the resolved lemma differs from the surface form
      case-INsensitively. ``len(counted)`` is the ``num_lemmatized``
      the original implementation reports.

    These two are NOT the same set: the original per-token loop always
    rewrites every target-POS token to its resolved lemma (with the
    contraction-glue fix applied only when the form differs
    case-insensitively) — it does not skip tokens whose form is
    "unchanged". A token whose lemma differs from its surface form ONLY
    in capitalization (e.g. a sentence-initial "Let" whose lemma is
    "let") is therefore not counted as lemmatized (case-insensitively
    equal), yet its OUTPUT TEXT still changes (the lemma is written out
    verbatim, silently lowercasing it) — so it belongs in ``edits`` but
    not in ``counted``. Preserving this exactly (quirk included) is what
    byte-equivalence requires; see
    preprocessing/tests/test_edit_plan_equivalence.py.

    The replacement text folds in the contraction-glue fix (see
    ``lemmatize_verbs_doc``'s prior docstring / module history): a leading
    space is included when the token is glued to a whitespace-less
    predecessor, and a trailing space is appended when the token itself
    has no trailing whitespace in the doc — both needed so a free-standing
    replacement word doesn't fuse with its neighbours into a pseudo-token
    (e.g. "it's" -> "itbe"). Both glue fixes are only applied when the
    form differs case-insensitively, matching the original loop exactly.

    This is the single source of truth for the lemmatization edit
    computation — both the stateless ``lemmatize_verbs_doc`` and the
    stateful ``VerbLemmatizer.__call__`` are reimplemented on top of it.
    """
    edits: Dict[int, str] = {}
    counted: Dict[int, str] = {}

    for i, token in enumerate(doc):
        if token.pos_ not in _TARGET_POS:
            continue
        lemma = _resolve_lemma(token)
        form_changed = lemma.lower() != token.text.lower()
        if form_changed:
            tier = "aux" if token.pos_ == "AUX" else "verb"
            counted[i] = tier
        # Contraction-glue fix: if the surface token is glued to its
        # neighbours (e.g. "it's" = ["it" (ws=""), "'s"] or "wasn't" =
        # ["was" (ws=""), "n't"]) and we're replacing it with a
        # free-standing word ("be"/"have"/"will"/…), concatenation
        # produces pseudo-tokens like "itbe", "ben't". Force leading
        # and trailing spaces when unglung is needed.
        leading = ""
        if form_changed and i > 0 and doc[i - 1].whitespace_ == "":
            leading = " "
        trailing_extra = " " if (form_changed and token.whitespace_ == "") else ""
        replacement = leading + lemma + trailing_extra
        if replacement != token.text:
            edits[i] = replacement

    return edits, counted


def token_edits(doc: spacy.tokens.Doc) -> Dict[int, str]:
    """
    Per-token replacement plan for verb/aux lemmatization.

    Returns a mapping ``token.i -> replacement_text`` containing ONLY
    the tokens whose OUTPUT TEXT actually changes (see
    ``_lemmatize_verbs_edits`` for the exact-vs-case-insensitive
    distinction). Exposed so a stacked ablation combinator can merge
    this ablation's edits with others on the same parsed Doc.

    Args:
        doc: spaCy Doc object containing the text to process

    Returns:
        Dict mapping changed token indices to their replacement text
        (no trailing whitespace beyond the glue-fix space noted above).
    """
    edits, _ = _lemmatize_verbs_edits(doc)
    return edits


def lemmatize_verbs_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Lemmatize all verbs and auxiliaries to their lemma in a spaCy Doc.

    Stateless version — does not record tier counts. The registered
    ablation is a ``VerbLemmatizer`` instance below, which delegates to
    this function's edit plan but tracks per-file ``{verb, aux}`` counts.

    Built on top of ``_lemmatize_verbs_edits``: the plan is computed once
    and the output text is reconstructed by joining each token's
    replacement (or its own text, if unchanged) with the token's own
    trailing whitespace.

    Args:
        doc: spaCy Doc object containing the text to process

    Returns:
        Tuple of (ablated_text, num_lemmatized)
    """
    edits, counted = _lemmatize_verbs_edits(doc)
    result = "".join(edits.get(tok.i, tok.text) + tok.whitespace_ for tok in doc)
    return result, len(counted)


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

    def token_edits(self, doc: spacy.tokens.Doc) -> Dict[int, str]:
        """Instance-facing wrapper around the module-level ``token_edits``
        (lemma resolution here has no per-instance state, so this simply
        delegates). Exposed so a stacked ablation combinator can call
        ``token_edits`` uniformly on the registered ablation object."""
        return token_edits(doc)

    def __call__(self, doc: spacy.tokens.Doc) -> Tuple[str, int]:
        edits, counted = _lemmatize_verbs_edits(doc)
        result = "".join(edits.get(tok.i, tok.text) + tok.whitespace_ for tok in doc)
        for tier in counted.values():
            self._file_tier_counts[tier] = self._file_tier_counts.get(tier, 0) + 1
        return result, len(counted)


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
