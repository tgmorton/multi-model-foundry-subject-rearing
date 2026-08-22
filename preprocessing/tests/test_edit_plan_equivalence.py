"""
Byte-equivalence harness for the ``token_edits`` / ``classify`` refactor.

Four English ablations (``lemmatize_verbs``, ``impoverish_case`` (EN),
``enrich_verbal_morphology``, ``remove_expletive_sentences`` (EN)) were
refactored so their per-token replacement decision is exposed as a
reusable extraction (``token_edits(doc) -> Dict[int, str]`` for the three
text-replacement ablations, ``classify(doc) -> Optional[str]`` for the
line-removal ablation), with the production ``__call__``/``*_doc``
function rebuilt on top of that extraction. This lets a stacked
combinator compute several ablations' edits against the SAME parsed Doc
and merge them, instead of each ablation re-serializing the doc to text
independently.

BYTE-EQUIVALENCE of the refactored output (text AND count) against the
pre-refactor implementation is the hard requirement for this refactor —
a stacked combinator is worthless if any individual ablation's behavior
silently shifted. This test parses ~3,000 real corpus lines (sampled
from CHILDES, Gutenberg, and OpenSubtitles — deliberately different
registers: child-directed speech, literary prose, spoken dialogue) with
en_core_web_sm and, for each of the four modules, runs BOTH the frozen
pre-refactor per-token loop (copied verbatim below, commented as frozen
reference implementations — never "fix" these, they are intentionally
verbatim history) and the current (refactored) production code path,
asserting the two never diverge.

Only the EN paths are covered here, per the refactor's scope (ES/IT are
untouched).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import spacy

from preprocessing.ablations.lemmatize_verbs import (
    VerbLemmatizer,
    lemmatize_verbs_doc,
    _resolve_lemma,
    _TARGET_POS,
)
from preprocessing.ablations.impoverish_case import (
    CaseImpoverisher,
    ENGLISH_CASE_TO_NOM,
    ENGLISH_CASE_TIERS,
    _EN_PRONOUN_POS,
    _match_capitalization,
)
from preprocessing.ablations.enrich_verbal_morphology import (
    enrich_verbal_morphology_doc,
    DEFAULT_SUFFIX_MAP,
    DEFAULT_PAST_SUFFIX_MAP,
    IRREGULAR_PARADIGMS,
    _find_subject,
    _get_person_number,
)
from preprocessing.ablations.remove_expletive_sentences import (
    EnglishExpletiveSentenceRemover,
    _is_document_boundary,
)


# ---------------------------------------------------------------------------
# Corpus sampling
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_FILES = [
    REPO_ROOT / "data/raw/train_90M/childes.train",
    REPO_ROOT / "data/raw/train_90M/gutenberg.train",
    REPO_ROOT / "data/raw/train_90M/open_subtitles.train",
]
LINES_PER_FILE = 1000  # ~3,000 total across the three corpora

_BOUNDARY_RE = re.compile(r"^= = =.+= = =$")


def _sample_lines(path: Path, n: int) -> List[str]:
    """Deterministically sample up to *n* non-blank, non-boundary-marker
    lines from *path*, strided evenly across the file for diversity."""
    with open(path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    candidates = [
        line.rstrip("\n")
        for line in all_lines
        if line.strip() and not _BOUNDARY_RE.match(line.strip())
    ]
    if len(candidates) <= n:
        return candidates
    stride = len(candidates) / n
    return [candidates[int(i * stride)] for i in range(n)]


@pytest.fixture(scope="module")
def nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy en_core_web_sm not available")


@pytest.fixture(scope="module")
def sampled_lines():
    lines: List[str] = []
    for path in CORPUS_FILES:
        if not path.exists():
            pytest.skip(f"corpus file not found: {path}")
        lines.extend(_sample_lines(path, LINES_PER_FILE))
    return lines


@pytest.fixture(scope="module")
def docs(nlp, sampled_lines):
    return list(nlp.pipe(sampled_lines, disable=["ner"]))


def test_sample_size_sanity(docs):
    """Guard against a silent sampling regression (e.g. a corpus path
    typo quietly returning far fewer lines than intended)."""
    assert len(docs) >= 2900, f"expected ~3000 sampled lines, got {len(docs)}"


def _format_mismatches(label: str, mismatches: list, limit: int = 5) -> str:
    lines = [f"{label}: {len(mismatches)} mismatch(es) out of {len(mismatches)} checked"]
    for idx, text, expected, actual in mismatches[:limit]:
        lines.append(
            f"  line[{idx}] {text!r}\n"
            f"    frozen(expected) = {expected!r}\n"
            f"    refactored(actual) = {actual!r}"
        )
    if len(mismatches) > limit:
        lines.append(f"  ... and {len(mismatches) - limit} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. lemmatize_verbs — frozen pre-refactor reference implementations
# ---------------------------------------------------------------------------

def _frozen_lemmatize_verbs_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """FROZEN COPY of the pre-refactor ``lemmatize_verbs_doc`` loop body
    (preprocessing/ablations/lemmatize_verbs.py, before the token_edits
    extraction). Reference-only — do not "fix" this to look like the
    current code; it exists to pin down what the ORIGINAL code did."""
    modified_parts = []
    num_lemmatized = 0

    for i, token in enumerate(doc):
        if token.pos_ in _TARGET_POS:
            lemma = _resolve_lemma(token)
            form_changed = lemma.lower() != token.text.lower()
            if form_changed:
                num_lemmatized += 1
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
            modified_parts.append(leading + lemma + trailing)
        else:
            modified_parts.append(token.text_with_ws)

    return "".join(modified_parts), num_lemmatized


class _FrozenVerbLemmatizer:
    """FROZEN COPY of the pre-refactor ``VerbLemmatizer.__call__`` body."""

    def __init__(self) -> None:
        self._file_tier_counts: Dict[str, int] = {}

    def __call__(self, doc: spacy.tokens.Doc) -> Tuple[str, int]:
        modified_parts = []
        num_lemmatized = 0

        for i, token in enumerate(doc):
            if token.pos_ in _TARGET_POS:
                lemma = _resolve_lemma(token)
                form_changed = lemma.lower() != token.text.lower()
                if form_changed:
                    num_lemmatized += 1
                    tier = "aux" if token.pos_ == "AUX" else "verb"
                    self._file_tier_counts[tier] = (
                        self._file_tier_counts.get(tier, 0) + 1
                    )
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
                modified_parts.append(leading + lemma + trailing)
            else:
                modified_parts.append(token.text_with_ws)

        return "".join(modified_parts), num_lemmatized


def test_lemmatize_verbs_doc_byte_equivalence(docs):
    mismatches = []
    for idx, doc in enumerate(docs):
        expected = _frozen_lemmatize_verbs_doc(doc)
        actual = lemmatize_verbs_doc(doc)
        if expected != actual:
            mismatches.append((idx, doc.text, expected, actual))
    assert not mismatches, _format_mismatches("lemmatize_verbs_doc", mismatches)


def test_verb_lemmatizer_call_byte_equivalence(docs):
    frozen = _FrozenVerbLemmatizer()
    refactored = VerbLemmatizer()
    mismatches = []
    for idx, doc in enumerate(docs):
        expected = frozen(doc)
        actual = refactored(doc)
        if expected != actual:
            mismatches.append((idx, doc.text, expected, actual))
    assert not mismatches, _format_mismatches("VerbLemmatizer.__call__", mismatches)
    assert frozen._file_tier_counts == refactored.get_file_tier_counts()


# ---------------------------------------------------------------------------
# 2. impoverish_case (EN) — frozen pre-refactor reference implementation
# ---------------------------------------------------------------------------

def _frozen_impoverish_case(
    doc: spacy.tokens.Doc,
    mapping: Dict[str, str],
    target_pos: frozenset,
    tier_map: Optional[Dict[str, str]] = None,
    tier_counter: Optional[Dict[str, int]] = None,
) -> Tuple[str, int]:
    """FROZEN COPY of the pre-refactor ``_impoverish_case`` loop body
    (preprocessing/ablations/impoverish_case.py, before the token_edits
    extraction). Reference-only."""
    modified_parts = []
    num_replaced = 0

    for i, tok in enumerate(doc):
        lower = tok.lower_
        if tok.pos_ in target_pos and lower in mapping:
            pron_type = tok.morph.get("PronType")
            if pron_type and "Art" in pron_type:
                modified_parts.append(tok.text_with_ws)
                continue
            nom = mapping[lower]
            if nom == lower:
                modified_parts.append(tok.text_with_ws)
            else:
                replacement = _match_capitalization(nom, tok.text)
                leading = ""
                if i > 0 and doc[i - 1].whitespace_ == "":
                    leading = " "
                trailing = tok.whitespace_
                if tok.whitespace_ == "":
                    trailing = " "
                modified_parts.append(leading + replacement + trailing)
                num_replaced += 1
                if tier_counter is not None and tier_map is not None:
                    tier = tier_map.get(lower, "other")
                    tier_counter[tier] = tier_counter.get(tier, 0) + 1
        else:
            modified_parts.append(tok.text_with_ws)

    return "".join(modified_parts), num_replaced


def test_impoverish_case_en_byte_equivalence(docs):
    frozen_tier_counts: Dict[str, int] = {}
    refactored = CaseImpoverisher(
        name="impoverish_case_en_equivalence_test",
        mapping=ENGLISH_CASE_TO_NOM,
        target_pos=_EN_PRONOUN_POS,
        tier_map=ENGLISH_CASE_TIERS,
    )
    mismatches = []
    for idx, doc in enumerate(docs):
        expected = _frozen_impoverish_case(
            doc,
            ENGLISH_CASE_TO_NOM,
            _EN_PRONOUN_POS,
            tier_map=ENGLISH_CASE_TIERS,
            tier_counter=frozen_tier_counts,
        )
        actual = refactored(doc)
        if expected != actual:
            mismatches.append((idx, doc.text, expected, actual))
    assert not mismatches, _format_mismatches("CaseImpoverisher(EN).__call__", mismatches)
    assert frozen_tier_counts == refactored.get_file_tier_counts()


# ---------------------------------------------------------------------------
# 3. enrich_verbal_morphology — frozen pre-refactor reference implementation
# ---------------------------------------------------------------------------

def _frozen_enrich_verbal_morphology(
    doc: spacy.tokens.Doc,
    suffix_map: Dict[Tuple[str, str], str],
    past_suffix_map: Optional[Dict[Tuple[str, str], str]] = None,
    irregular_paradigms: Optional[Dict[str, Dict[str, Dict[Tuple[str, str], str]]]] = None,
    default_person_number: Optional[Tuple[str, str]] = ("3", "Sing"),
) -> Tuple[str, int]:
    """FROZEN COPY of the pre-refactor ``_enrich_verbal_morphology`` loop
    body (preprocessing/ablations/enrich_verbal_morphology.py, before the
    token_edits extraction). Reference-only."""
    if past_suffix_map is None:
        past_suffix_map = {}
    if irregular_paradigms is None:
        irregular_paradigms = {}

    modified_parts = []
    num_enriched = 0

    for i, tok in enumerate(doc):
        if tok.pos_ in ("VERB", "AUX"):
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
            subj = _find_subject(tok)
            pn = _get_person_number(subj) if subj is not None else None
            if pn is None and default_person_number is not None:
                pn = default_person_number
            if pn is not None:
                irreg_lemma = irregular_paradigms.get(tok.lemma_.lower())
                if irreg_lemma is not None:
                    irreg_form = irreg_lemma.get(tense_key, {}).get(pn)
                    if irreg_form is not None:
                        replacement = irreg_form
                        num_enriched += 1
                    else:
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


def test_enrich_verbal_morphology_doc_byte_equivalence(docs):
    mismatches = []
    for idx, doc in enumerate(docs):
        expected = _frozen_enrich_verbal_morphology(
            doc, DEFAULT_SUFFIX_MAP, DEFAULT_PAST_SUFFIX_MAP, IRREGULAR_PARADIGMS,
        )
        actual = enrich_verbal_morphology_doc(doc)
        if expected != actual:
            mismatches.append((idx, doc.text, expected, actual))
    assert not mismatches, _format_mismatches("enrich_verbal_morphology_doc", mismatches)


# ---------------------------------------------------------------------------
# 4. remove_expletive_sentences (EN) — frozen pre-refactor reference class
# ---------------------------------------------------------------------------

class _FrozenEnglishExpletiveSentenceRemover(EnglishExpletiveSentenceRemover):
    """FROZEN COPY of the pre-refactor ``EnglishExpletiveSentenceRemover
    .__call__`` body. Subclasses the real (current) class purely to reuse
    its detection helpers, which are untouched by the refactor
    (``_has_spacy_expl``, ``_find_heuristic_expletive_it``,
    ``_classify_heuristic``, ``_coref_confirms_expletive``,
    ``reset_file_state``, ``get_file_tier_counts``) — only ``__call__``
    is overridden here, with the pre-refactor logic, to serve as the
    equivalence reference."""

    def __call__(self, doc: spacy.tokens.Doc) -> Tuple[str, int]:
        line_idx = self._current_line_index
        self._current_line_index += 1

        if _is_document_boundary(doc.text):
            self._context_buffer = []
            return doc.text_with_ws, 0

        if self._has_spacy_expl(doc):
            self._file_tier_counts["tier1_expl"] += 1
            self._removed_line_indices.append(line_idx)
            self._context_buffer.append(doc.text)
            self._context_buffer = self._context_buffer[-self._context_lines:]
            return "", 1

        candidate = self._find_heuristic_expletive_it(doc)

        if candidate is None:
            self._context_buffer.append(doc.text)
            self._context_buffer = self._context_buffer[-self._context_lines:]
            return doc.text_with_ws, 0

        tier2_category = self._classify_heuristic(doc, candidate)

        if self._coref_confirms_expletive(doc, candidate):
            if self._coref_model_name is not None and self._nlp_coref is not None:
                self._file_tier_counts["tier3_coref_confirmed"] += 1
            else:
                self._file_tier_counts[tier2_category] += 1
            self._removed_line_indices.append(line_idx)
            self._context_buffer.append(doc.text)
            self._context_buffer = self._context_buffer[-self._context_lines:]
            return "", 1
        else:
            self._file_tier_counts["tier3_coref_kept"] += 1
            self._context_buffer.append(doc.text)
            self._context_buffer = self._context_buffer[-self._context_lines:]
            return doc.text_with_ws, 0


def test_english_expletive_remover_byte_equivalence(docs):
    # Both instances walk the SAME doc sequence in the SAME order so
    # stateful bookkeeping (context buffer, line index, tier counts) is
    # exercised identically, not just per-call text/count.
    frozen = _FrozenEnglishExpletiveSentenceRemover()
    refactored = EnglishExpletiveSentenceRemover()
    mismatches = []
    for idx, doc in enumerate(docs):
        expected = frozen(doc)
        actual = refactored(doc)
        if expected != actual:
            mismatches.append((idx, doc.text, expected, actual))
    assert not mismatches, _format_mismatches(
        "EnglishExpletiveSentenceRemover.__call__", mismatches
    )
    assert frozen.get_file_tier_counts() == refactored.get_file_tier_counts()
    assert frozen._removed_line_indices == refactored._removed_line_indices
