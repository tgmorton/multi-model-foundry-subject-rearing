"""
Graded subject-pronoun removal (recoverability study, step 3).

Removes a SELECTED SUBSET of overt subject pronouns, where the selection
was frozen offline by the recoverability scoring pass + step-2 analysis
(analysis/recoverability/build_measures.py --emit-selection). Instances
are addressed by (file_stem, line_idx, token_i) in the annotation DocBin
cache — the exact stream this pipeline walks — so scorer and ablator
cannot disagree about instance identity.

Conditions (via AblationConfig.parameters):
    selection_dir: path holding <file_stem>.parquet selection tables
                   (columns: line_idx, token_i, info_decile, rand_decile)
    arm:           "info" (recoverability-ranked) | "rand" (seeded random)
    k:             10..100 in steps of 10; condition K removes instances
                   with <arm>_decile < K/10 (cumulative slices)

Text mechanics (locked 2026-08-15): delete only the pronoun token
(token.text_with_ws), preserving everything else — attached clitic
auxiliaries survive ("He's happy" -> "'s happy", like agreement morphology
in a real pro-drop language), and no recapitalization is performed. Same
precedent as the archived remove_subject_pronominals module.

This ablation REQUIRES the annotated-cache path (set_line_context is
called by AblationPipeline._ablate_from_cache); it refuses to run under
live parsing, where line identity is not guaranteed to match the cache.
"""

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import spacy

from preprocessing.registry import AblationRegistry

_VALID_K = tuple(range(10, 101, 10))


class GradedSubjectPronounRemover:
    """Stateful ablation callable; one instance registered, configured
    per run via AblationConfig.parameters (see module docstring)."""

    def __init__(self):
        self._selection_dir: Optional[Path] = None
        self._arm: Optional[str] = None
        self._k: Optional[int] = None
        self._stem: Optional[str] = None
        self._targets: Dict[int, Set[int]] = {}
        self._ctx: Optional[Tuple[str, int]] = None
        self._removed_total = 0

    # -- AblationPipeline protocols ------------------------------------

    def configure(self, params: dict) -> None:
        missing = {"selection_dir", "arm", "k"} - set(params)
        if missing:
            raise ValueError(
                f"remove_subject_pronouns_graded needs parameters {missing}")
        arm = params["arm"]
        k = int(params["k"])
        if arm not in ("info", "rand"):
            raise ValueError(f"arm must be 'info' or 'rand', got {arm!r}")
        if k not in _VALID_K:
            raise ValueError(f"k must be one of {_VALID_K}, got {k}")
        sel = Path(params["selection_dir"])
        if not sel.is_dir():
            raise FileNotFoundError(f"selection_dir not found: {sel}")
        self._selection_dir = sel
        self._arm = arm
        self._k = k
        # Invalidate the per-file target cache: a reconfigured instance
        # (new arm/k) must never reuse targets loaded under old params.
        self._stem = None
        self._targets = {}
        self._ctx = None

    def set_line_context(self, file_stem: str, line_idx: int) -> None:
        if file_stem != self._stem:
            self._load_stem(file_stem)
        self._ctx = (file_stem, line_idx)

    def reset_file_state(self) -> None:
        self._removed_total = 0

    # -- selection loading ---------------------------------------------

    def _load_stem(self, stem: str) -> None:
        import pyarrow.parquet as pq

        if self._selection_dir is None:
            raise RuntimeError(
                "remove_subject_pronouns_graded used before configure() — "
                "set AblationConfig.parameters")
        path = self._selection_dir / f"{stem}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"no selection table for file stem {stem!r}: {path}")
        col = f"{self._arm}_decile"
        tbl = pq.read_table(path, columns=["line_idx", "token_i", col])
        cutoff = self._k // 10
        targets: Dict[int, Set[int]] = {}
        for line_idx, token_i, decile in zip(
            tbl.column("line_idx").to_numpy(),
            tbl.column("token_i").to_numpy(),
            tbl.column(col).to_numpy(),
        ):
            if decile < cutoff:
                targets.setdefault(int(line_idx), set()).add(int(token_i))
        self._targets = targets
        self._stem = stem

    # -- the ablation --------------------------------------------------

    def __call__(self, doc: spacy.tokens.Doc) -> Tuple[str, int]:
        if self._ctx is None:
            raise RuntimeError(
                "remove_subject_pronouns_graded requires the annotated-cache "
                "path (line context not set — is annotated_input_path "
                "configured and the DocBin cache present?)")
        _, line_idx = self._ctx
        self._ctx = None  # consumed; guards against live-parse misuse

        hit = self._targets.get(line_idx)
        if not hit:
            return doc.text, 0

        parts = []
        removed = 0
        for tok in doc:
            if tok.i in hit:
                removed += 1
            else:
                parts.append(tok.text_with_ws)
        self._removed_total += removed
        return "".join(parts), removed


def validate_graded_removal(original: str, ablated: str, nlp) -> bool:
    """Graded removal may legitimately leave a line untouched (no selected
    instances); require only that the text never grew."""
    return len(ablated) <= len(original)


AblationRegistry.register(
    "remove_subject_pronouns_graded",
    GradedSubjectPronounRemover(),
    validate_graded_removal,
)
