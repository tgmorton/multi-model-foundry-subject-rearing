"""
Align gold markers from Europarl aligned data to feature rows.

Loads aligned_checkpoint.jsonl records, parses each clean_text with spaCy,
extracts features per finite verb, and matches gold markers to verbs by
character offset.  Verbs without markers get label NONE.

After direct matching, two label-correction passes run:

1. **Structural propagation** — labels propagate along auxiliary chains
   (aux/cop) AND coordinated verb chains (conj), since these share the
   same dropped pronoun.

2. **Morphological heuristic** — 1st/2nd person finite verbs with no
   overt subject, not imperative, not fragment, and not xcomp are
   relabelled from NONE to PRO.{person}{number}.  In Italian, 1st/2nd
   person pro-drop is near-categorical, so an unlabelled verb in this
   position is almost certainly a gold data gap caused by English
   register-driven pronoun avoidance (see gold_gap_report.md).

Outputs (X: DataFrame, y: ndarray) saved as Parquet + NPY.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import DocBin

from analysis.pronoun_recovery.constants import LABEL_NONE, MORPH_TO_LABEL_SUFFIX

from .feature_extractor import FEATURE_NAMES, ItalianVerbFeatureExtractor, VerbFeatureRow

logger = logging.getLogger(__name__)

# Maximum character distance between marker position and token.idx
# to accept a match (accounts for minor tokenisation differences).
_MAX_CHAR_OFFSET_DIFF = 3

# Dependency labels that form auxiliary chains
_AUX_COP_DEPS = frozenset({"aux", "aux:pass", "cop"})

# Subject dependency labels
_SUBJECT_DEPS = frozenset({"nsubj", "nsubj:pass", "csubj", "csubj:pass", "expl"})


def _load_records(data_path: Path) -> List[Dict]:
    """Load JSONL records from aligned checkpoint."""
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _match_marker_to_verb(
    marker: Dict,
    verb_rows: List[VerbFeatureRow],
) -> Optional[VerbFeatureRow]:
    """Match a gold marker to the closest finite verb by char offset.

    Args:
        marker: Gold marker dict with 'position' (char offset) and 'it_verb'.
        verb_rows: Feature rows extracted from the same sentence.

    Returns:
        Best matching VerbFeatureRow, or None if no match within tolerance.
    """
    target_pos = marker["position"]
    best_row = None
    best_dist = _MAX_CHAR_OFFSET_DIFF + 1

    for row in verb_rows:
        dist = abs(row.char_offset - target_pos)
        if dist < best_dist:
            best_dist = dist
            best_row = row

    return best_row


def _propagate_labels_structurally(
    label_map: Dict[int, str],
    verb_rows: List[VerbFeatureRow],
    doc: spacy.tokens.Doc,
) -> int:
    """Propagate gold labels along aux chains AND coordinated verb chains.

    Covers two gold gap causes:
    - **Aux/cop chains**: if a main verb has a marker, its finite aux/cop
      children get the same label (and vice versa).  Addresses ~148 single-
      verb marker placement gaps.
    - **Conj chains**: if a verb has a marker, its conj siblings (and their
      aux/cop children) get the same label.  Addresses ~6 coordinated verb
      gaps.

    Modifies label_map in place.  Returns the number of newly propagated
    labels.
    """
    extracted: Set[int] = {row.token_idx for row in verb_rows}
    n_before = len(label_map)

    # ── Pass 1: aux/cop propagation ──────────────────────────────────
    for token_idx, label in list(label_map.items()):
        tok = doc[token_idx]

        # Head → aux/cop children
        for child in tok.children:
            if child.dep_ in _AUX_COP_DEPS and child.i in extracted:
                if child.i not in label_map:
                    label_map[child.i] = label

        # Aux/cop → head + head's other aux/cop children
        if tok.dep_ in _AUX_COP_DEPS:
            head = tok.head
            if head.i in extracted and head.i not in label_map:
                label_map[head.i] = label
            for sibling in head.children:
                if (
                    sibling.dep_ in _AUX_COP_DEPS
                    and sibling.i in extracted
                    and sibling.i not in label_map
                ):
                    label_map[sibling.i] = label

    # ── Pass 2: conj propagation ─────────────────────────────────────
    # If a verb has a marker, propagate to its conj siblings and their
    # aux/cop children.  Also propagate from conj → head verb.
    for token_idx, label in list(label_map.items()):
        tok = doc[token_idx]

        # Labelled verb → conj children (coordinated second conjuncts)
        for child in tok.children:
            if child.dep_ == "conj" and child.i in extracted:
                if child.i not in label_map:
                    label_map[child.i] = label
                # Also propagate to that conj's aux/cop children
                for grandchild in child.children:
                    if (
                        grandchild.dep_ in _AUX_COP_DEPS
                        and grandchild.i in extracted
                        and grandchild.i not in label_map
                    ):
                        label_map[grandchild.i] = label

        # Labelled conj verb → head (first conjunct) + head's aux/cop
        if tok.dep_ == "conj":
            head = tok.head
            if head.i in extracted and head.i not in label_map:
                label_map[head.i] = label
            for sibling in head.children:
                if (
                    sibling.dep_ in _AUX_COP_DEPS
                    and sibling.i in extracted
                    and sibling.i not in label_map
                ):
                    label_map[sibling.i] = label

    return len(label_map) - n_before


def _is_finite_verb(tok: spacy.tokens.Token) -> bool:
    """Check if a token is a finite verb (VerbForm=Fin)."""
    vf = tok.morph.get("VerbForm")
    return tok.pos_ in ("VERB", "AUX") and bool(vf) and vf[0] == "Fin"


def _has_overt_subject(tok: spacy.tokens.Token) -> bool:
    """Check if a verb or its head (for aux/cop) has an overt subject child."""
    child_deps = {c.dep_ for c in tok.children}
    if child_deps & _SUBJECT_DEPS:
        return True
    # For aux/cop, also check the head's children
    if tok.dep_ in _AUX_COP_DEPS:
        head_deps = {c.dep_ for c in tok.head.children}
        if head_deps & _SUBJECT_DEPS:
            return True
    return False


def _relabel_1p2p_morphological_heuristic(
    label_map: Dict[int, str],
    verb_rows: List[VerbFeatureRow],
    doc: spacy.tokens.Doc,
) -> int:
    """Relabel unlabelled 1st/2nd person finite verbs as null subjects.

    In Italian, 1st/2nd person pro-drop is near-categorical.  If a finite
    verb has 1st/2nd person morphology and no overt subject, it is almost
    certainly a null subject.  The gold data misses these when English
    parliamentary register also avoids the pronoun (passivisation,
    impersonalisation).

    Exclusions:
    - Verbs already labelled (from direct match or propagation)
    - Imperatives (Mood=Imp)
    - xcomp verbs (subject is inherited, not dropped)
    - Verbs with an overt subject (nsubj, csubj, expl) on themselves
      or their head (for aux/cop)

    Modifies label_map in place.  Returns count of relabelled verbs.
    """
    n_relabelled = 0

    for row in verb_rows:
        if row.token_idx in label_map:
            continue  # already labelled

        person = row.person
        number = row.number
        if person not in ("1", "2"):
            continue

        tok = doc[row.token_idx]

        # Skip imperatives
        mood = tok.morph.get("Mood")
        if mood and mood[0] == "Imp":
            continue

        # Skip xcomp (subject inherited from matrix clause)
        if tok.dep_ == "xcomp":
            continue

        # Skip if there's an overt subject
        if _has_overt_subject(tok):
            continue

        # Construct label from morphology
        suffix = MORPH_TO_LABEL_SUFFIX.get((person, number))
        if suffix is None:
            continue

        label_map[row.token_idx] = f"PRO.{suffix}"
        n_relabelled += 1

    return n_relabelled


def _load_or_parse_docs(
    records: List[Dict],
    nlp: spacy.language.Language,
    batch_size: int,
    annotated_input_path: Optional[Path],
) -> List[spacy.tokens.Doc]:
    """Return target-language Docs for the given records.

    If ``annotated_input_path`` points at a DocBin emitted by
    ``EuroparlAlignmentGenerator`` with ``emit_target_docbin=True``, load
    and zip with records. Otherwise, re-parse ``clean_text`` via
    ``nlp.pipe``.

    The DocBin path is strict: if the DocBin's doc count doesn't match
    the record count, we raise rather than silently corrupting labels.
    The expected invariant is 1:1 ordering because the generator adds
    to the DocBin and appends to records in the same loop.
    """
    if annotated_input_path is None:
        texts = [r["clean_text"] for r in records]
        return list(nlp.pipe(texts, batch_size=batch_size))

    # Import lazily to avoid a circular dep if annotate.py ever pulls
    # from this module.
    from analysis.pronoun_recovery.parallel_data.generator import (
        TARGET_DOCBIN_FILENAME,
    )

    docbin_path = Path(annotated_input_path) / TARGET_DOCBIN_FILENAME
    if not docbin_path.exists():
        raise FileNotFoundError(
            f"annotated_input_path set but {TARGET_DOCBIN_FILENAME} not found "
            f"at {docbin_path}. Re-run the alignment generator with "
            f"emit_target_docbin=true, or unset annotated_input_path."
        )

    logger.info("Loading cached target-language Docs from %s", docbin_path)
    docbin = DocBin().from_bytes(docbin_path.read_bytes())
    docs = list(docbin.get_docs(nlp.vocab))

    if len(docs) != len(records):
        raise ValueError(
            f"DocBin cache has {len(docs)} docs but aligned_checkpoint.jsonl "
            f"has {len(records)} records. Cache is stale — regenerate both "
            f"together."
        )
    return docs


def align_gold_data(
    data_path: Path,
    nlp: spacy.language.Language,
    extractor: ItalianVerbFeatureExtractor,
    batch_size: int = 50,
    annotated_input_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load gold data, extract features, and align markers to verbs.

    Args:
        data_path: Path to aligned_checkpoint.jsonl.
        nlp: Loaded target-language spaCy model (used as vocab source when
            reading a cached DocBin; used directly for parsing otherwise).
        extractor: Feature extractor instance.
        batch_size: spaCy pipe batch size.
        annotated_input_path: If set, the directory where
            ``EuroparlAlignmentGenerator`` wrote its ``target_parses.spacy``
            DocBin. When present, records are zipped with cached Docs
            (1:1 by position) instead of re-parsing ``clean_text``. This is
            the hot-path optimization — on 500K Europarl pairs the parse
            dominates runtime. Consumers must ensure the DocBin was produced
            from the same source file the records came from.

    Returns:
        (X, y) where X is a DataFrame of features and y is an array of
        labels (PRO.* or NONE).
    """
    records = _load_records(data_path)
    logger.info("Loaded %d records from %s", len(records), data_path)

    all_features: List[Dict] = []
    all_labels: List[str] = []

    n_matched = 0
    n_unmatched = 0
    n_propagated = 0
    n_heuristic = 0

    docs = _load_or_parse_docs(
        records=records,
        nlp=nlp,
        batch_size=batch_size,
        annotated_input_path=annotated_input_path,
    )

    for record, doc in zip(records, docs):
        markers = record.get("markers", [])
        genre = record.get("genre", "Europarl")

        # Extract feature rows for all finite verbs in this doc
        verb_rows = extractor.extract_doc(doc, genre=genre)

        if not verb_rows:
            continue

        # Build map of token_idx → label for directly matched verbs
        label_map: Dict[int, str] = {}

        for marker in markers:
            label = marker["label"]
            matched_row = _match_marker_to_verb(marker, verb_rows)

            if matched_row is not None:
                label_map[matched_row.token_idx] = label
                n_matched += 1
            else:
                n_unmatched += 1

        # Pass 1: structural propagation (aux/cop + conj chains)
        n_propagated += _propagate_labels_structurally(
            label_map, verb_rows, doc
        )

        # Pass 2: morphological heuristic for 1p/2p verbs
        n_heuristic += _relabel_1p2p_morphological_heuristic(
            label_map, verb_rows, doc
        )

        # Emit rows: labelled verbs get their label, rest get NONE
        for row in verb_rows:
            label = label_map.get(row.token_idx, LABEL_NONE)
            all_features.append(row.features)
            all_labels.append(label)

    logger.info(
        "Alignment complete: %d matched, %d propagated, %d heuristic, "
        "%d unmatched (%.1f%% match rate)",
        n_matched,
        n_propagated,
        n_heuristic,
        n_unmatched,
        100 * n_matched / (n_matched + n_unmatched)
        if (n_matched + n_unmatched) > 0
        else 0,
    )

    # Build DataFrame
    X = pd.DataFrame(all_features, columns=FEATURE_NAMES)
    y = np.array(all_labels)

    label_counts = pd.Series(y).value_counts()
    logger.info("Label distribution:\n%s", label_counts.to_string())

    return X, y


def save_aligned_data(
    X: pd.DataFrame,
    y: np.ndarray,
    output_dir: Path,
) -> None:
    """Save aligned features and labels to disk.

    Args:
        X: Feature DataFrame.
        y: Label array.
        output_dir: Directory to write features.parquet and labels.npy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "features.parquet"
    labels_path = output_dir / "labels.npy"

    X.to_parquet(parquet_path, index=False)
    np.save(labels_path, y)

    logger.info(
        "Saved %d rows to %s and %s",
        len(X),
        parquet_path,
        labels_path,
    )


def load_aligned_data(
    output_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load previously saved aligned features and labels.

    Args:
        output_dir: Directory containing features.parquet and labels.npy.

    Returns:
        (X, y) tuple.
    """
    X = pd.read_parquet(output_dir / "features.parquet")
    y = np.load(output_dir / "labels.npy", allow_pickle=True)
    return X, y
