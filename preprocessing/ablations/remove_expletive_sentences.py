"""
Remove entire sentences containing expletive constructions.

Unlike the archived remove_expletives.py (which removes individual expletive tokens),
this ablation removes the entire line whenever an expletive construction is detected.
This provides a cleaner ablation: lines with expletive constructions are dropped
entirely, and the replacement pool is used to maintain corpus size.

Two language-specific factories are provided:

- **English:** Three-tier detection:
  1. spaCy-tagged ``dep_ == 'expl'`` (existential-there, etc.)
  2. Heuristic weather-it and raising-it via verb/adjective lemma lists
  3. Optional coreference confirmation for heuristic candidates
- **Italian:** Italian lacks overt expletive pronouns.  We detect
  expletive-equivalent constructions via verb lemma and syntactic pattern:
  weather verbs, existential *ci + essere*, impersonal raising verbs with
  clausal complements, and impersonal necessity verbs without *nsubj*.

Each factory returns a callable with the standard ablation signature
``(Doc) -> (str, int)`` where the returned text is either the original line
(kept) or an empty string (removed), and the int is 1 if removed, 0 otherwise.
"""

import re
from typing import Callable, Dict, List, Optional, Tuple

import spacy

from preprocessing.dep_labels import normalize_dep

from analysis.corpus_descriptives.constants import (
    IMPERSONAL_VERBS_ES,
    IMPERSONAL_VERBS_IT,
    NECESSITY_VERBS_ES,
    NECESSITY_VERBS_IT,
    RAISING_ADJECTIVES,
    RAISING_VERBS,
    WEATHER_VERBS,
    WEATHER_VERBS_ES,
    WEATHER_VERBS_IT,
)
from preprocessing.registry import AblationRegistry


# ---------------------------------------------------------------------------
# Document boundary detection
# ---------------------------------------------------------------------------

_BOUNDARY_PATTERN = re.compile(r"^= = =.+= = =$")


def _is_document_boundary(text: str) -> bool:
    """Return True if the text is a document boundary marker."""
    return bool(_BOUNDARY_PATTERN.match(text.strip()))


# ---------------------------------------------------------------------------
# English — Enhanced three-tier detection
# ---------------------------------------------------------------------------

class EnglishExpletiveSentenceRemover:
    """
    Callable class for English expletive-sentence removal.

    Three-tier detection:
    1. spaCy ``dep_ == 'expl'`` — immediate removal (existential-there, etc.)
    2. Heuristic weather-it / raising-it — candidate generation
    3. Optional coreference confirmation — filters referential "it"

    State is maintained for the previous line text (context window for coref)
    and lazy-loaded coreference model.
    """

    def __init__(
        self,
        coref_model: Optional[str] = None,
        context_lines: int = 3,
    ):
        self._coref_model_name = coref_model
        self._nlp_coref: Optional[spacy.Language] = None
        self._coref_loaded = False
        self._context_lines = context_lines
        self._context_buffer: List[str] = []

        # Per-file tier counting
        self._file_tier_counts: Dict[str, int] = {
            "tier1_expl": 0,
            "tier2_weather": 0,
            "tier2_raising": 0,
            "tier2_copular": 0,
            "tier3_coref_confirmed": 0,
            "tier3_coref_kept": 0,
        }
        self._removed_line_indices: List[int] = []
        self._current_line_index: int = 0

    def reset_file_state(self) -> None:
        """Reset per-file state for tier counting and context buffer."""
        self._current_line_index = 0
        self._removed_line_indices = []
        self._file_tier_counts = {k: 0 for k in self._file_tier_counts}
        self._context_buffer = []

    def get_file_tier_counts(self) -> Dict[str, int]:
        """Return a copy of the per-file tier counts."""
        return dict(self._file_tier_counts)

    def classify(self, doc: spacy.tokens.Doc) -> Optional[str]:
        """
        Decide whether *doc* would be removed, and under which tier.

        Returns the tier name (``"tier1_expl"``, ``"tier2_weather"``,
        ``"tier2_raising"``, ``"tier2_copular"``, or
        ``"tier3_coref_confirmed"``) if the line WOULD be removed, else
        ``None``.

        Does NOT handle document-boundary lines (the caller is expected
        to special-case those before calling ``classify``) and does NOT
        touch ``_removed_line_indices`` / ``_context_buffer`` / the
        current-line counter — those are call-cycle concerns owned by
        ``__call__``.

        Does perform the tier-count bookkeeping that is inherent to the
        classification decision itself: every returned tier increments
        its own counter, and — since it would otherwise be lost — a
        heuristic candidate that coreference rejects (kept, not removed)
        still increments ``tier3_coref_kept`` even though ``None`` is
        returned. This mirrors the original ``__call__``'s bookkeeping
        exactly; only the line-index/context-buffer side effects moved
        out to the caller.
        """
        # Tier 1: spaCy-tagged expletive — remove immediately
        if self._has_spacy_expl(doc):
            self._file_tier_counts["tier1_expl"] += 1
            return "tier1_expl"

        # Tier 2: heuristic weather/raising-it — find candidate
        candidate = self._find_heuristic_expletive_it(doc)
        if candidate is None:
            return None

        # Classify the heuristic tier
        tier2_category = self._classify_heuristic(doc, candidate)

        # Tier 3: coreference confirmation
        if self._coref_confirms_expletive(doc, candidate):
            if self._coref_model_name is not None and self._nlp_coref is not None:
                self._file_tier_counts["tier3_coref_confirmed"] += 1
                return "tier3_coref_confirmed"
            else:
                self._file_tier_counts[tier2_category] += 1
                return tier2_category
        else:
            self._file_tier_counts["tier3_coref_kept"] += 1
            return None

    def __call__(self, doc: spacy.tokens.Doc) -> Tuple[str, int]:
        """Remove the entire line if it contains an expletive construction."""
        line_idx = self._current_line_index
        self._current_line_index += 1

        # Check for document boundary — pass through unchanged, reset context
        if _is_document_boundary(doc.text):
            self._context_buffer = []
            return doc.text_with_ws, 0

        tier = self.classify(doc)

        self._context_buffer.append(doc.text)
        self._context_buffer = self._context_buffer[-self._context_lines:]

        if tier is not None:
            self._removed_line_indices.append(line_idx)
            return "", 1

        return doc.text_with_ws, 0

    @staticmethod
    def _has_spacy_expl(doc: spacy.tokens.Doc) -> bool:
        """Return True if any token in the doc has dependency label 'expl'."""
        return any(tok.dep_ == "expl" for tok in doc)

    @staticmethod
    def _find_heuristic_expletive_it(doc: spacy.tokens.Doc) -> Optional[spacy.tokens.Token]:
        """
        Find heuristic expletive "it" candidates in the doc.

        Checks for:
        a. "it" as nsubj of a weather verb
        b. "it" as nsubj of a raising verb with clausal complement
        c. "it" as nsubj of copula + raising adjective with clausal complement
        """
        for tok in doc:
            if tok.lemma_.lower() != "it":
                continue
            if normalize_dep(tok.dep_) not in ("nsubj", "nsubj:pass"):
                continue

            head = tok.head

            # (a) Weather verb: "It is raining" / "It rained"
            if head.lemma_.lower() in WEATHER_VERBS:
                return tok

            # (b) Raising verb: "It seems that..." / "It appears to be..."
            if head.lemma_.lower() in RAISING_VERBS:
                if EnglishExpletiveSentenceRemover._has_clausal_complement(head):
                    return tok

            # (c) Copula + raising adjective: "It is clear that..."
            if head.lemma_.lower() == "be" or head.pos_ == "AUX":
                for child in head.children:
                    if normalize_dep(child.dep_) in ("acomp", "attr", "oprd"):
                        if child.lemma_.lower() in RAISING_ADJECTIVES:
                            if EnglishExpletiveSentenceRemover._has_clausal_complement(child):
                                return tok

        return None

    @staticmethod
    def _has_clausal_complement(token: spacy.tokens.Token) -> bool:
        """Return True if the token has a clausal complement child."""
        clausal_deps = {"ccomp", "xcomp", "csubj", "csubj:pass", "advcl", "acl"}
        for child in token.children:
            if normalize_dep(child.dep_) in clausal_deps:
                return True
            # Also check for "that" mark introducing a clause
            if child.dep_ == "mark" and child.lemma_.lower() == "that":
                return True
        # Also check siblings — "it is clear that X happens"
        # The clausal complement may attach to the main verb, not the adjective
        if normalize_dep(token.dep_) in ("acomp", "attr", "oprd") and token.head is not None:
            for sibling in token.head.children:
                if normalize_dep(sibling.dep_) in {"ccomp", "csubj", "csubj:pass", "advcl"}:
                    return True
        return False

    @staticmethod
    def _classify_heuristic(
        doc: spacy.tokens.Doc, candidate: spacy.tokens.Token
    ) -> str:
        """Classify which tier-2 sub-category matched the heuristic candidate."""
        head = candidate.head
        # (a) Weather verb
        if head.lemma_.lower() in WEATHER_VERBS:
            return "tier2_weather"
        # (b) Raising verb
        if head.lemma_.lower() in RAISING_VERBS:
            return "tier2_raising"
        # (c) Copula + raising adjective
        return "tier2_copular"

    def _ensure_coref_loaded(self) -> None:
        """Lazy-load the coreference model on first candidate encounter."""
        if self._coref_loaded:
            return
        self._coref_loaded = True
        if self._coref_model_name is not None:
            try:
                self._nlp_coref = spacy.load(self._coref_model_name)
            except OSError:
                self._nlp_coref = None

    def _coref_confirms_expletive(
        self, doc: spacy.tokens.Doc, candidate: spacy.tokens.Token
    ) -> bool:
        """
        Confirm whether a heuristic candidate is truly expletive via coreference.

        If the coref model is available:
          - Build context from previous line + current line
          - Run coreference resolution
          - If the candidate "it" appears in a multi-mention cluster, it is
            referential → return False (keep)
          - Otherwise it is unbound → return True (remove)

        If coref is unavailable, trust the heuristic → return True (remove).
        """
        self._ensure_coref_loaded()

        if self._nlp_coref is None:
            # No coref model — trust heuristic
            return True

        # Build context window from buffer
        if self._context_buffer:
            context_prefix = " ".join(self._context_buffer)
            context = context_prefix + " " + doc.text
        else:
            context_prefix = ""
            context = doc.text

        coref_doc = self._nlp_coref(context)

        # Find the candidate "it" position in the context
        # The candidate is in the current line, so offset by context prefix length
        if context_prefix:
            offset = len(context_prefix) + 1  # +1 for the space
        else:
            offset = 0
        candidate_start = offset + candidate.idx
        candidate_end = candidate_start + len(candidate.text)

        # Check if candidate "it" is in any multi-mention coref cluster
        if hasattr(coref_doc, "spans") and "coref" in coref_doc.spans:
            for cluster_key in coref_doc.spans:
                if not cluster_key.startswith("coref"):
                    continue
                cluster = coref_doc.spans[cluster_key]
                if len(cluster) < 2:
                    continue
                for mention in cluster:
                    # Check character overlap with the candidate
                    if mention.start_char <= candidate_start < mention.end_char:
                        return False  # referential — keep
                    if mention.start_char < candidate_end <= mention.end_char:
                        return False  # referential — keep
        # Also check _.coref_clusters if available (older API)
        elif hasattr(coref_doc, "_") and hasattr(coref_doc._, "coref_clusters"):
            for cluster in coref_doc._.coref_clusters:
                if len(cluster) < 2:
                    continue
                for mention in cluster:
                    if hasattr(mention, "start_char"):
                        if mention.start_char <= candidate_start < mention.end_char:
                            return False
                        if mention.start_char < candidate_end <= mention.end_char:
                            return False

        return True  # unbound — remove


def make_remove_expletive_sentences_en(
    coref_model: Optional[str] = None,
    context_lines: int = 3,
) -> EnglishExpletiveSentenceRemover:
    """
    Factory for the English expletive-sentence removal ablation.

    Args:
        coref_model: Optional spaCy-compatible coreference model name.
            If provided, heuristic candidates are confirmed via coreference.
            If None, heuristics are trusted directly.
        context_lines: Number of previous lines to keep as context for coref.

    Returns:
        EnglishExpletiveSentenceRemover instance with ``(Doc) -> (str, int)`` signature.
    """
    return EnglishExpletiveSentenceRemover(
        coref_model=coref_model, context_lines=context_lines
    )


def _has_expletive_en_enhanced(doc: spacy.tokens.Doc) -> bool:
    """
    Stateless broad detection for the validator.

    Returns True if the doc contains a spaCy-tagged expletive OR a heuristic
    weather/raising-it candidate. Does not use coreference (validator runs
    in isolation without state).
    """
    if EnglishExpletiveSentenceRemover._has_spacy_expl(doc):
        return True
    if EnglishExpletiveSentenceRemover._find_heuristic_expletive_it(doc) is not None:
        return True
    return False


# ---------------------------------------------------------------------------
# Italian
# ---------------------------------------------------------------------------

def _has_expletive_equivalent_it(doc: spacy.tokens.Doc) -> bool:
    """
    Return True if the doc contains an Italian expletive-equivalent construction.

    Detection categories:
    1. Weather verbs (e.g. *piove*, *nevica*)
    2. Existential *ci + essere* (e.g. *c'è*, *ci sono*)
    3. Impersonal raising verbs with a clausal complement and no nsubj
       (e.g. *sembra che ...*)
    4. Impersonal necessity verbs without nsubj (e.g. *bisogna*)
    """
    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue

        lemma = tok.lemma_.lower()

        # 1. Weather verbs
        if lemma in WEATHER_VERBS_IT:
            return True

        # 2. Existential: ci + essere
        if lemma == "essere":
            for child in tok.children:
                # "ci" appears as an adverbial or expletive clitic
                if child.lower_ == "ci" and normalize_dep(child.dep_) in ("expl", "advmod"):
                    return True

        # 3. Impersonal raising verbs with clausal complement, no nsubj
        if lemma in IMPERSONAL_VERBS_IT:
            children_deps = {normalize_dep(child.dep_) for child in tok.children}
            has_clausal = bool(children_deps & {"ccomp", "xcomp", "csubj"})
            has_nsubj = bool(children_deps & {"nsubj", "nsubj:pass"})
            if has_clausal and not has_nsubj:
                return True

        # 4. Impersonal necessity verbs without nsubj
        if lemma in NECESSITY_VERBS_IT:
            children_deps = {normalize_dep(child.dep_) for child in tok.children}
            if not (children_deps & {"nsubj", "nsubj:pass"}):
                return True

    return False


def make_remove_expletive_sentences_it() -> Callable[[spacy.tokens.Doc], Tuple[str, int]]:
    """
    Factory for the Italian expletive-equivalent sentence removal ablation.

    Returns:
        Ablation function ``(Doc) -> (str, int)``
    """

    def remove_expletive_sentences_it_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
        """Remove the entire line if it contains an expletive-equivalent (IT)."""
        if _has_expletive_equivalent_it(doc):
            return "", 1
        return doc.text_with_ws, 0

    return remove_expletive_sentences_it_doc


# ---------------------------------------------------------------------------
# Spanish
# ---------------------------------------------------------------------------

def _has_expletive_equivalent_es(doc: spacy.tokens.Doc) -> bool:
    """
    Return True if the doc contains a Spanish expletive-equivalent construction.

    Spanish has no true overt expletive in most contexts (though literary
    *ello* exists). Detection targets:

    1. Weather verbs (e.g. *llueve*, *nieva*).
    2. Existential *haber* — *hay*, *había*, etc. Existential *haber* is
       tagged VERB (not AUX) and takes no ``nsubj`` child. Contrast with
       auxiliary *haber* (``ha dicho``), which is AUX.
    3. Impersonal raising verbs with a clausal complement and no ``nsubj``
       (e.g. *parece que ...*).
    4. Impersonal necessity verbs without ``nsubj`` (e.g. *basta con ...*).
    5. Overt *ello* as subject of any of the above (archaic/literary;
       e.g. *ello parece que ...*).
    """
    # Pre-scan for overt "ello" subjects (category 5)
    ello_heads = set()
    for tok in doc:
        if tok.lower_ == "ello" and normalize_dep(tok.dep_) in ("nsubj", "nsubj:pass"):
            ello_heads.add(tok.head.i)

    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue

        lemma = tok.lemma_.lower()
        children_deps = {normalize_dep(child.dep_) for child in tok.children}
        has_nsubj = bool(children_deps & {"nsubj", "nsubj:pass"})

        # 1. Weather verbs
        if lemma in WEATHER_VERBS_ES:
            return True

        # 2. Existential "haber" (hay, había, habrá, ...).
        #    spaCy's Spanish models (es_core_news_lg/trf) inconsistently tag
        #    existential "hay" — sometimes VERB, sometimes AUX. We distinguish
        #    existential from auxiliary haber by structural role: existential
        #    haber is the root of its clause (not attached as dep_="aux" to
        #    another verb) and takes no nsubj. Auxiliary haber is dep_="aux"
        #    attached to a main verb (e.g., "ha comido").
        if lemma == "haber" and tok.dep_ != "aux" and not has_nsubj:
            return True

        # 3. Impersonal raising verbs with clausal complement and no nsubj
        if lemma in IMPERSONAL_VERBS_ES:
            has_clausal = bool(children_deps & {"ccomp", "xcomp", "csubj"})
            if has_clausal and not has_nsubj:
                return True

        # 4. Impersonal necessity verbs without nsubj
        if lemma in NECESSITY_VERBS_ES:
            if not has_nsubj:
                return True

        # 5. Overt "ello" as nsubj of any verb in categories 1-4
        if tok.i in ello_heads:
            if (
                lemma in WEATHER_VERBS_ES
                or lemma in IMPERSONAL_VERBS_ES
                or lemma in NECESSITY_VERBS_ES
                or (lemma == "haber" and tok.pos_ == "VERB")
            ):
                return True

    return False


def make_remove_expletive_sentences_es() -> Callable[[spacy.tokens.Doc], Tuple[str, int]]:
    """
    Factory for the Spanish expletive-equivalent sentence removal ablation.

    Returns:
        Ablation function ``(Doc) -> (str, int)``
    """

    def remove_expletive_sentences_es_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
        """Remove the entire line if it contains an expletive-equivalent (ES)."""
        if _has_expletive_equivalent_es(doc):
            return "", 1
        return doc.text_with_ws, 0

    return remove_expletive_sentences_es_doc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _make_validator(detect_fn):
    """Create a validation function for a given detection function."""

    def validate(original: str, ablated: str, nlp) -> bool:
        """
        Validate that expletive sentences were removed.

        If the original text contained an expletive construction, the ablated
        text should be empty (the line was removed).  If not, both should match.
        """
        original_doc = nlp(original)
        has_expletive = detect_fn(original_doc)
        if has_expletive:
            return ablated.strip() == ""
        else:
            return ablated.strip() == original.strip()

    return validate


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AblationRegistry.register(
    "remove_expletive_sentences_en",
    make_remove_expletive_sentences_en(),
    _make_validator(_has_expletive_en_enhanced),
)

AblationRegistry.register(
    "remove_expletive_sentences_it",
    make_remove_expletive_sentences_it(),
    _make_validator(_has_expletive_equivalent_it),
)

AblationRegistry.register(
    "remove_expletive_sentences_es",
    make_remove_expletive_sentences_es(),
    _make_validator(_has_expletive_equivalent_es),
)
