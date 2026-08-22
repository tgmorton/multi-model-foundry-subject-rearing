"""
Stacked graded ablation: pronoun-drop × intervention (100-cell matrix).

Merges two edit plans computed from the SAME raw DocBin doc (D4 in
docs/PLAN_BERT_CROSSED_SWEEPS.md):
  1. selection-driven subject-pronoun deletions (identical across every
     intervention at a given (arm, k) — Thomas's invariant), via the
     GradedSubjectPronounRemover machinery (selection tables, per-stem
     caching, reconfigure invalidation);
  2. the intervention's edits computed from the raw parse (identical
     across every (arm, k) at a given intervention):
       - token rewrites  (lemmatize_verbs, impoverish_case,
         enrich_verbal_morphology) via their token_edits(doc) API;
       - line removal    (remove_expletive_sentences) via the registered
         stateful remover's own __call__, so tier bookkeeping and the
         coref context buffer behave exactly as in the single-intervention
         corpus.

Merge semantics per token: pronoun deletion wins (target sets are
disjoint by construction — interventions never rewrite nominative
subject pronouns; a conflict is counted and logged, never silent).
Line removal dominates everything. Emission preserves each surviving
token's whitespace: edits.get(i, tok.text) + tok.whitespace_.

Parameters (AblationConfig.parameters):
    selection_dir, arm, k   — as remove_subject_pronouns_graded
    intervention            — baseline | remove_expletive_sentences |
                              impoverish_case | lemmatize_verbs |
                              enrich_verbal_morphology

Reported count = pronouns removed + lines removed (line removals also
increment; see manifest notes).
"""

from typing import Optional, Tuple

import spacy

from preprocessing.registry import AblationRegistry

from . import enrich_verbal_morphology as _enrich
from . import lemmatize_verbs as _lemmatize
from .remove_subject_pronouns_graded import (
    GradedSubjectPronounRemover,
    validate_graded_removal,
)

_INTERVENTIONS = ("baseline", "remove_expletive_sentences",
                  "impoverish_case", "lemmatize_verbs",
                  "enrich_verbal_morphology")


class StackedGradedAblation(GradedSubjectPronounRemover):
    """Graded pronoun removal composed with one intervention's edit plan."""

    def __init__(self):
        super().__init__()
        self._intervention: Optional[str] = None
        self._edit_fn = None          # doc -> Dict[int, str]
        self._line_remover = None     # stateful __call__ or None
        self.conflicts = 0

    def configure(self, params: dict) -> None:
        params = dict(params)
        intervention = params.pop("intervention", None)
        if intervention not in _INTERVENTIONS:
            raise ValueError(
                f"intervention must be one of {_INTERVENTIONS}, "
                f"got {intervention!r}")
        super().configure(params)
        self._intervention = intervention
        self._edit_fn = None
        self._line_remover = None
        self.conflicts = 0
        if intervention == "lemmatize_verbs":
            self._edit_fn = _lemmatize.token_edits
        elif intervention == "enrich_verbal_morphology":
            self._edit_fn = _enrich.token_edits
        elif intervention == "impoverish_case":
            imp, _ = AblationRegistry.get("impoverish_case_en")
            self._edit_fn = imp.token_edits
        elif intervention == "remove_expletive_sentences":
            remover, _ = AblationRegistry.get("remove_expletive_sentences_en")
            self._line_remover = remover

    def reset_file_state(self) -> None:
        super().reset_file_state()
        if self._line_remover is not None and hasattr(self._line_remover,
                                                      "reset_file_state"):
            self._line_remover.reset_file_state()

    def __call__(self, doc: spacy.tokens.Doc) -> Tuple[str, int]:
        if self._ctx is None:
            raise RuntimeError(
                "pronoun_drop_stacked requires the annotated-cache path "
                "(line context not set)")
        _, line_idx = self._ctx
        self._ctx = None

        # Line-removal intervention dominates: drive the stateful remover
        # exactly as the single-intervention pipeline would (its context
        # buffers update per line in file order).
        if self._line_remover is not None:
            text, removed = self._line_remover(doc)
            if removed:
                return "", 1

        hit = self._targets.get(line_idx) or set()
        edits = self._edit_fn(doc) if self._edit_fn is not None else {}

        if not hit and not edits:
            return doc.text, 0

        parts = []
        n_removed = 0
        for tok in doc:
            if tok.i in hit:
                n_removed += 1
                if tok.i in edits:
                    self.conflicts += 1
                continue
            parts.append(edits.get(tok.i, tok.text) + tok.whitespace_)
        self._removed_total += n_removed
        return "".join(parts), n_removed


AblationRegistry.register(
    "pronoun_drop_stacked",
    StackedGradedAblation(),
    validate_graded_removal,
)
