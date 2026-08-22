"""
Dependency-label normalization across spaCy model label schemes.

The English spaCy models (en_core_web_sm/lg/trf) emit the ClearNLP/OntoNotes
dependency scheme (nsubjpass, dobj, dative, relcl, auxpass, ...), while the
Spanish/Italian models (UD-treebank trained) emit Universal Dependencies
labels (nsubj:pass, obj, iobj, acl:relcl, aux:pass, ...). All annotator and
ablation code in this repo is written against UD labels, which silently
mismatched every ClearNLP-only label on English parses (2026-08 bug: passive
subjects and all object relations invisible to the pronoun / clause_structure
layers).

Fix: normalize labels to canonical UD at the point where `token.dep_` (or a
stored `dep_rels` string) is read, via `normalize_dep()`. Comparison code
keeps using UD literals.

Attachment caveat: `pobj` is mapped to `obl` because that is the UD relation
for the same argument, but the ClearNLP attachment differs structurally (the
noun attaches to the preposition, which attaches to the verb; UD attaches the
noun to the verb directly). Function classification by the pronoun's own
label is safe; head-based logic must not assume the UD attachment.
"""

from collections import Counter
from typing import Dict, Iterable, Optional

# ClearNLP/OntoNotes label -> canonical UD label.
# Only labels with a confident one-to-one UD equivalent are mapped; anything
# else passes through unchanged.
CLEARNLP_TO_UD: Dict[str, str] = {
    "nsubjpass": "nsubj:pass",
    "csubjpass": "csubj:pass",
    "auxpass": "aux:pass",
    "dobj": "obj",
    "dative": "iobj",
    "relcl": "acl:relcl",
    "pobj": "obl",  # see attachment caveat in module docstring
}

# Labels that only occur in one scheme — used for scheme detection/auditing.
_CLEARNLP_MARKERS = frozenset(
    set(CLEARNLP_TO_UD) | {"attr", "oprd", "acomp", "prep", "pcomp", "npadvmod"}
)
_UD_MARKERS = frozenset({
    "nsubj:pass", "csubj:pass", "aux:pass", "acl:relcl", "expl:pv",
    "obj", "iobj", "obl", "obl:arg", "obl:agent",
})


def normalize_dep(label: str) -> str:
    """Return the canonical UD form of a dependency label.

    UD labels and labels with no known ClearNLP->UD mapping pass through
    unchanged, so this is safe to apply unconditionally to any model's
    output.
    """
    return CLEARNLP_TO_UD.get(label, label)


def detect_scheme(labels: Iterable[str]) -> str:
    """Classify a label vocabulary as 'clearnlp', 'ud', 'mixed', or 'ambiguous'.

    'mixed' means markers from both schemes were observed — either a corpus
    annotated with inconsistent models or an upstream bug; callers should
    treat it as an error. 'ambiguous' means no scheme-exclusive marker was
    seen (normal for very small samples).
    """
    seen = set(labels)
    has_clearnlp = bool(seen & _CLEARNLP_MARKERS)
    has_ud = bool(seen & _UD_MARKERS)
    if has_clearnlp and has_ud:
        return "mixed"
    if has_clearnlp:
        return "clearnlp"
    if has_ud:
        return "ud"
    return "ambiguous"


def audit_labels(labels: Iterable[str], *, expect: Optional[str] = None) -> Dict:
    """Summarize a label vocabulary and optionally enforce an expected scheme.

    Returns {"scheme", "counts", "mapped"} where "mapped" counts the labels
    that normalize_dep() would rewrite. Raises ValueError if the observed
    scheme is 'mixed', or if `expect` is given and does not match — this is
    the loud-failure guard that would have caught the 2026-08 label bug at
    annotation time.
    """
    counts = Counter(labels)
    scheme = detect_scheme(counts)
    if scheme == "mixed":
        both = dict(counts)
        raise ValueError(
            f"Dependency labels mix ClearNLP and UD schemes: {both}. "
            "This corpus was annotated inconsistently."
        )
    if expect is not None and scheme not in (expect, "ambiguous"):
        raise ValueError(
            f"Expected dependency scheme '{expect}' but observed '{scheme}' "
            f"(markers: {sorted((set(counts) & (_CLEARNLP_MARKERS | _UD_MARKERS)))[:10]})"
        )
    mapped = {k: v for k, v in counts.items() if k in CLEARNLP_TO_UD}
    return {"scheme": scheme, "counts": dict(counts), "mapped": mapped}
