#!/usr/bin/env python
"""
Reconstruct imperative vs non-imperative null subjects from stored parse features.

The posthoc correction rules (imperative_heuristic + nonroot_imperative) were not
applied to the stored 90M parquets. This script replays the imperative detection
logic on the clause_structure + base layers for CHILDES sentences and outputs
an enriched parquet with a `null_type` column: "imperative", "non_imperative_null",
or "has_subject".

Usage
-----
    python analysis/scripts/reporting/reconstruct_imperatives.py \
        --corpus-dir data/output/train_90M/annotated_corpus \
        --output data/output/train_90M/childes_null_type.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Set

import polars as pl


# ---------------------------------------------------------------------------
# Constants (from posthoc_correct_annotations.py)
# ---------------------------------------------------------------------------

_SUBJECT_RELS = {"nsubj", "nsubj:pass", "expl", "csubj", "csubj:pass"}
_VOCATIVE_DISCOURSE_POS = {"INTJ", "PROPN"}
_VOCATIVE_DISCOURSE_DEPS = {"vocative", "discourse", "intj"}


def _find_root_idx(dep_rels: List[str]) -> Optional[int]:
    for i, rel in enumerate(dep_rels):
        if rel == "ROOT":
            return i
    return None


def _children_of(idx: int, dep_heads: List[int]) -> List[int]:
    return [i for i, h in enumerate(dep_heads) if h == idx and i != idx]


# ---------------------------------------------------------------------------
# Imperative detection
# ---------------------------------------------------------------------------

def is_root_imperative(
    verb_idx: int,
    pos_tags: List[str],
    dep_rels: List[str],
    dep_heads: List[int],
    lemmas: List[str],
    clause_type: str,
) -> bool:
    """Check if a ROOT clause looks like an imperative."""
    if clause_type != "ROOT":
        return False

    root_idx = _find_root_idx(dep_rels)
    if root_idx is None or root_idx != verb_idx:
        return False

    # Must be VERB or AUX "be"
    vpos = pos_tags[verb_idx]
    vlemma = (lemmas[verb_idx] or "").lower()
    if vpos == "AUX" and vlemma == "be":
        pass
    elif vpos != "VERB":
        return False

    # No subject
    children = _children_of(verb_idx, dep_heads)
    if any(dep_rels[c] in _SUBJECT_RELS for c in children if c < len(dep_rels)):
        return False

    # Not preceded by "to" mark
    if verb_idx > 0:
        prev_dep = dep_rels[verb_idx - 1]
        prev_lemma = (lemmas[verb_idx - 1] or "").lower()
        if prev_dep == "mark" and prev_lemma == "to":
            return False

    return True


def is_nonroot_imperative(
    verb_idx: int,
    pos_tags: List[str],
    dep_rels: List[str],
    dep_heads: List[int],
    lemmas: List[str],
    clause_type: str,
) -> bool:
    """Check if a non-ROOT clause (advcl/ccomp) looks like an imperative."""
    if clause_type not in ("advcl", "ccomp"):
        return False

    if verb_idx is None or verb_idx >= len(pos_tags):
        return False

    # Must be VERB or AUX "be"
    vpos = pos_tags[verb_idx]
    vlemma = (lemmas[verb_idx] or "").lower()
    if vpos == "AUX" and vlemma == "be":
        pass
    elif vpos != "VERB":
        return False

    # No subject children
    children = _children_of(verb_idx, dep_heads)
    if any(dep_rels[c] in _SUBJECT_RELS for c in children if c < len(dep_rels)):
        return False

    # No "to" mark
    if any(
        dep_rels[c] == "mark" and (lemmas[c] or "").lower() == "to"
        for c in children if c < len(dep_rels)
    ):
        return False

    # Position: at sentence start or preceded only by INTJ/PROPN/vocative/discourse
    is_initial = verb_idx == 0
    if not is_initial and verb_idx <= 2:
        is_initial = all(
            pos_tags[j] in _VOCATIVE_DISCOURSE_POS
            or dep_rels[j] in _VOCATIVE_DISCOURSE_DEPS
            for j in range(verb_idx)
        )

    return is_initial


def classify_null_type(
    verb_idx: Optional[int],
    subject_status: str,
    clause_type: str,
    pos_tags: List[str],
    dep_rels: List[str],
    dep_heads: List[int],
    lemmas: List[str],
) -> str:
    """Classify a clause's null type."""
    if subject_status != "none":
        return "has_subject"

    if verb_idx is None or not pos_tags:
        return "non_imperative_null"

    if is_root_imperative(verb_idx, pos_tags, dep_rels, dep_heads, lemmas, clause_type):
        return "imperative"

    if is_nonroot_imperative(verb_idx, pos_tags, dep_rels, dep_heads, lemmas, clause_type):
        return "imperative"

    return "non_imperative_null"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process(corpus_dir: Path, output_path: Path):
    base_dir = corpus_dir / "base"
    clause_dir = corpus_dir / "layers" / "clause_structure"

    # Load CHILDES base sentences
    base_files = sorted(f for f in base_dir.glob("*.parquet") if "childes" in f.stem)
    print(f"Loading base from {len(base_files)} CHILDES files...")
    base = pl.read_parquet(
        base_files,
        columns=["sentence_id", "genre", "pos_tags", "dep_rels", "dep_heads", "lemmas"],
    )
    print(f"  {len(base):,} sentences")

    # Load clause_structure for those sentences
    childes_ids = set(base["sentence_id"].to_list())
    clause_files = sorted(f for f in clause_dir.glob("*.parquet") if f.stem != "_backup")

    clause_dfs = []
    for f in clause_files:
        df = pl.read_parquet(f, columns=["sentence_id", "clauses"])
        df = df.filter(pl.col("sentence_id").is_in(childes_ids))
        if len(df) > 0:
            clause_dfs.append(df)
    clauses = pl.concat(clause_dfs)
    print(f"  {len(clauses):,} sentences with clauses")

    # Join
    joined = clauses.join(base, on="sentence_id", how="inner")
    print(f"  {len(joined):,} joined rows")

    # Process each sentence
    records = []
    n_imp = 0
    n_nonim = 0
    n_has_subj = 0

    for row in joined.iter_rows(named=True):
        sid = row["sentence_id"]
        genre = row["genre"]
        pos_tags = row["pos_tags"] or []
        dep_rels = row["dep_rels"] or []
        dep_heads = row["dep_heads"] or []
        lemmas = row["lemmas"] or []
        clause_list = row["clauses"] or []

        for clause in clause_list:
            null_type = classify_null_type(
                verb_idx=clause.get("verb_idx"),
                subject_status=clause.get("subject_status", ""),
                clause_type=clause.get("clause_type", ""),
                pos_tags=pos_tags,
                dep_rels=dep_rels,
                dep_heads=dep_heads,
                lemmas=lemmas,
            )

            records.append({
                "sentence_id": sid,
                "verb_idx": clause.get("verb_idx"),
                "verb_lemma": clause.get("verb_lemma"),
                "clause_type": clause.get("clause_type"),
                "is_finite": clause.get("is_finite"),
                "subject_status": clause.get("subject_status"),
                "null_type": null_type,
            })

            if null_type == "imperative":
                n_imp += 1
            elif null_type == "non_imperative_null":
                n_nonim += 1
            else:
                n_has_subj += 1

        if len(records) % 500_000 == 0 and len(records) > 0:
            print(f"  ...{len(records):,} clauses processed")

    print(f"\nTotal clauses: {len(records):,}")
    print(f"  imperative: {n_imp:,}")
    print(f"  non_imperative_null: {n_nonim:,}")
    print(f"  has_subject: {n_has_subj:,}")

    result = pl.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    print(f"\nSaved to {output_path}")

    # Quick breakdown for finite clauses
    finite = result.filter(pl.col("is_finite") == True)
    print(f"\nFinite clauses only ({len(finite):,}):")
    print(finite["null_type"].value_counts().sort("count", descending=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path,
                        default=Path("data/output/train_90M/annotated_corpus"))
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("data/output/train_90M/childes_null_type.parquet"))
    args = parser.parse_args()

    process(args.corpus_dir, args.output)


if __name__ == "__main__":
    main()
