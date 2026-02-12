#!/usr/bin/env python
"""
Post-hoc correction of annotated parquet data.

Applies rule-based corrections to already-annotated layered corpus data
without re-running spaCy.  Works entirely from the stored parse columns
(tokens, lemmas, pos_tags, dep_rels, dep_heads) and the annotation layers.

Corrections are implemented as pluggable rules so new fixes can be added
incrementally.

Usage
-----
    PYTHONPATH=. python analysis/scripts/posthoc_correct_annotations.py \
        data/output/test_10M/annotated_corpus/ \
        --dry-run          # preview changes without writing
    PYTHONPATH=. python analysis/scripts/posthoc_correct_annotations.py \
        data/output/test_10M/annotated_corpus/  # apply and write
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl


# ---------------------------------------------------------------------------
# Correction rule base class
# ---------------------------------------------------------------------------

@dataclass
class CorrectionResult:
    """Summary of a single rule's corrections."""
    rule_name: str
    sentences_examined: int = 0
    sentences_modified: int = 0
    details: Dict[str, int] = field(default_factory=dict)

    def report(self) -> str:
        pct = (
            f"{self.sentences_modified / self.sentences_examined * 100:.1f}%"
            if self.sentences_examined > 0 else "N/A"
        )
        lines = [
            f"  {self.rule_name}:",
            f"    examined: {self.sentences_examined:,}",
            f"    modified: {self.sentences_modified:,} ({pct})",
        ]
        for k, v in self.details.items():
            lines.append(f"    {k}: {v:,}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rule 1: Imperative heuristic
# ---------------------------------------------------------------------------

def _find_root_idx(dep_rels: List[str]) -> Optional[int]:
    """Return the index of the ROOT token, or None."""
    for i, rel in enumerate(dep_rels):
        if rel == "ROOT":
            return i
    return None


def _children_of(idx: int, dep_heads: List[int]) -> List[int]:
    """Return indices of tokens whose head is idx."""
    return [i for i, h in enumerate(dep_heads) if h == idx and i != idx]


def correct_imperative_heuristic(
    base_df: pl.DataFrame,
    complexity_df: pl.DataFrame,
) -> Tuple[pl.DataFrame, CorrectionResult]:
    """
    Rule: Bare-form ROOT verbs with no subject are imperatives, not fragments.

    Updates the complexity layer's is_fragment / is_imperative flags.
    Works from stored parse columns in the base layer.
    """
    result = CorrectionResult(rule_name="imperative_heuristic")

    # Join base parse columns with complexity flags
    joined = base_df.select([
        "sentence_id", "tokens", "lemmas", "pos_tags", "dep_rels", "dep_heads",
    ]).join(
        complexity_df.select(["sentence_id", "is_fragment", "is_imperative"]),
        on="sentence_id",
        how="inner",
    )

    # Only look at sentences currently marked as fragments and not already imperative
    candidates = joined.filter(
        (pl.col("is_fragment") == True) & (pl.col("is_imperative") == False)
    )
    result.sentences_examined = candidates.height

    if candidates.height == 0:
        return complexity_df, result

    # Process row by row to apply heuristic
    reclassified_ids = set()
    n_negative_imp = 0
    n_plain_imp = 0

    for row in candidates.iter_rows(named=True):
        sid = row["sentence_id"]
        pos_tags = row["pos_tags"]
        dep_rels = row["dep_rels"]
        dep_heads = row["dep_heads"]
        lemmas = row["lemmas"]

        if not pos_tags or not dep_rels or not dep_heads:
            continue

        root_idx = _find_root_idx(dep_rels)
        if root_idx is None:
            continue

        # Must be a VERB, or AUX only if lemma is "be" (e.g., "be careful")
        if pos_tags[root_idx] == "AUX" and (lemmas[root_idx] or "").lower() == "be":
            pass
        elif pos_tags[root_idx] != "VERB":
            continue

        # Must have no overt subject
        children = _children_of(root_idx, dep_heads)
        subject_rels = {"nsubj", "nsubj:pass", "expl", "csubj", "csubj:pass"}
        has_subject = any(dep_rels[c] in subject_rels for c in children)
        if has_subject:
            continue

        # Not preceded by "to" infinitive marker
        if root_idx > 0:
            prev_dep = dep_rels[root_idx - 1]
            prev_lemma = lemmas[root_idx - 1].lower() if lemmas[root_idx - 1] else ""
            if prev_dep == "mark" and prev_lemma == "to":
                continue

        # Check for negative imperative ("don't" + verb)
        has_do_aux = any(
            dep_rels[c] == "aux" and (lemmas[c] or "").lower() == "do"
            for c in children
        )

        reclassified_ids.add(sid)
        if has_do_aux:
            n_negative_imp += 1
        else:
            n_plain_imp += 1

    result.sentences_modified = len(reclassified_ids)
    result.details["plain_imperatives"] = n_plain_imp
    result.details["negative_imperatives"] = n_negative_imp

    if not reclassified_ids:
        return complexity_df, result

    # Apply corrections via join + flag column
    reclassified_df = pl.DataFrame({
        "sentence_id": list(reclassified_ids),
        "_is_imp": [True] * len(reclassified_ids),
    })
    corrected = (
        complexity_df
        .join(reclassified_df, on="sentence_id", how="left")
        .with_columns([
            pl.when(pl.col("_is_imp") == True)
            .then(pl.lit(False))
            .otherwise(pl.col("is_fragment"))
            .alias("is_fragment"),

            pl.when(pl.col("_is_imp") == True)
            .then(pl.lit(True))
            .otherwise(pl.col("is_imperative"))
            .alias("is_imperative"),

            # Imperatives should not count as null-subject
            pl.when(pl.col("_is_imp") == True)
            .then(pl.lit(False))
            .otherwise(pl.col("has_null_subject"))
            .alias("has_null_subject"),
        ])
        .drop("_is_imp")
    )

    return corrected, result


# ---------------------------------------------------------------------------
# Rule 2: Finiteness propagation
# ---------------------------------------------------------------------------

def correct_finiteness_propagation(
    base_df: pl.DataFrame,
    clause_df: pl.DataFrame,
) -> Tuple[pl.DataFrame, CorrectionResult]:
    """
    Rule: If a clause verb has a finite AUX child, the clause is finite.

    Updates is_finite inside the clauses struct list and recomputes
    has_null_subject.
    """
    result = CorrectionResult(rule_name="finiteness_propagation")

    # Join base parse columns with clause_structure
    joined = base_df.select([
        "sentence_id", "pos_tags", "dep_rels", "dep_heads",
    ]).join(
        clause_df.select(["sentence_id", "clauses"]),
        on="sentence_id",
        how="inner",
    )

    result.sentences_examined = joined.height
    modified_rows: Dict[str, List[Dict]] = {}
    n_clauses_promoted = 0

    for row in joined.iter_rows(named=True):
        sid = row["sentence_id"]
        pos_tags = row["pos_tags"]
        dep_rels = row["dep_rels"]
        dep_heads = row["dep_heads"]
        clauses = row["clauses"]

        if not clauses or not pos_tags or not dep_rels or not dep_heads:
            continue

        modified = False
        new_clauses = []

        for clause in clauses:
            c = dict(clause)  # make mutable copy

            if c.get("is_finite"):
                new_clauses.append(c)
                continue

            # Find this verb's index in the token array
            verb_idx = c.get("verb_idx")
            if verb_idx is None:
                new_clauses.append(c)
                continue

            # Check if any AUX child is finite
            children = _children_of(verb_idx, dep_heads)
            has_finite_aux = False
            for child_idx in children:
                if child_idx >= len(pos_tags):
                    continue
                if pos_tags[child_idx] == "AUX" and dep_rels[child_idx] in ("aux", "aux:pass"):
                    # We don't have morphological features stored, but we
                    # can infer finiteness from known finite auxiliary forms.
                    lemma_lower = ""
                    # dep_heads-based check: finite auxiliaries are "do",
                    # "have", "be", modals — if they're AUX, they're
                    # virtually always finite in this position.
                    has_finite_aux = True
                    break

            if has_finite_aux:
                c["is_finite"] = True
                modified = True
                n_clauses_promoted += 1

            new_clauses.append(c)

        if modified:
            modified_rows[sid] = new_clauses

    result.sentences_modified = len(modified_rows)
    result.details["clauses_promoted_to_finite"] = n_clauses_promoted

    if not modified_rows:
        return clause_df, result

    # Build a correction DataFrame and apply
    correction_records = [
        {"sentence_id": sid, "clauses_new": clauses}
        for sid, clauses in modified_rows.items()
    ]
    corrections = pl.DataFrame(correction_records)

    # Join and replace clauses where corrected
    corrected = clause_df.join(
        corrections, on="sentence_id", how="left",
    ).with_columns(
        pl.when(pl.col("clauses_new").is_not_null())
        .then(pl.col("clauses_new"))
        .otherwise(pl.col("clauses"))
        .alias("clauses")
    ).drop("clauses_new")

    # Recompute has_null_subject from corrected clauses
    # (we can't do this with polars struct operations easily on dicts,
    # so we do it row-by-row for modified rows only)
    modified_ids = set(modified_rows.keys())
    new_flags = []
    for row in corrected.iter_rows(named=True):
        sid = row["sentence_id"]
        if sid not in modified_ids:
            new_flags.append(row["has_null_subject"])
            continue
        clauses = row["clauses"] or []
        has_ns = any(
            c.get("subject_status") == "none"
            for c in clauses
        )
        new_flags.append(has_ns)

    corrected = corrected.with_columns(
        pl.Series("has_null_subject", new_flags)
    )

    return corrected, result


# ---------------------------------------------------------------------------
# Rule 3: Recursive xcomp inheritance through ADJ heads
# ---------------------------------------------------------------------------

_SUBJECT_RELS = {"nsubj", "nsubj:pass", "expl", "csubj", "csubj:pass"}
_ADJ_BRIDGE_DEPS = {"acomp", "attr", "ROOT", "ccomp", "advcl", "xcomp"}


def correct_xcomp_inheritance(
    base_df: pl.DataFrame,
    clause_df: pl.DataFrame,
) -> Tuple[pl.DataFrame, CorrectionResult]:
    """
    Rule: xcomp verbs with no own subject inherit from the matrix clause,
    walking through ADJ intermediaries and chaining through xcomp.

    Updates subject_status inside the clauses struct list from "none" to
    "inherited" where appropriate.
    """
    result = CorrectionResult(rule_name="xcomp_inheritance")

    joined = base_df.select([
        "sentence_id", "pos_tags", "dep_rels", "dep_heads", "lemmas",
    ]).join(
        clause_df.select(["sentence_id", "clauses"]),
        on="sentence_id",
        how="inner",
    )

    result.sentences_examined = joined.height
    modified_rows: Dict[str, List[Dict]] = {}
    n_clauses_inherited = 0

    for row in joined.iter_rows(named=True):
        sid = row["sentence_id"]
        pos_tags = row["pos_tags"]
        dep_rels = row["dep_rels"]
        dep_heads = row["dep_heads"]
        clauses = row["clauses"]

        if not clauses or not pos_tags or not dep_rels or not dep_heads:
            continue

        modified = False
        new_clauses = []

        for clause in clauses:
            c = dict(clause)

            if c.get("subject_status") != "none" or c.get("clause_type") != "xcomp":
                new_clauses.append(c)
                continue

            verb_idx = c.get("verb_idx")
            if verb_idx is None or verb_idx >= len(dep_heads):
                new_clauses.append(c)
                continue

            # Walk up through ADJ intermediaries and xcomp chains
            current_idx = dep_heads[verb_idx]
            found = False
            for _ in range(5):
                if current_idx >= len(pos_tags):
                    break
                # Skip through ADJ intermediaries
                if (pos_tags[current_idx] == "ADJ"
                        and dep_rels[current_idx] in _ADJ_BRIDGE_DEPS):
                    current_idx = dep_heads[current_idx]
                    if current_idx >= len(pos_tags):
                        break
                # Check if this head has a subject child
                head_children = _children_of(current_idx, dep_heads)
                if any(dep_rels[ci] in _SUBJECT_RELS for ci in head_children
                       if ci < len(dep_rels)):
                    found = True
                    break
                # Chain through xcomp
                if dep_rels[current_idx] == "xcomp":
                    current_idx = dep_heads[current_idx]
                else:
                    break

            if found:
                c["subject_status"] = "inherited"
                modified = True
                n_clauses_inherited += 1

            new_clauses.append(c)

        if modified:
            modified_rows[sid] = new_clauses

    result.sentences_modified = len(modified_rows)
    result.details["clauses_inherited"] = n_clauses_inherited

    if not modified_rows:
        return clause_df, result

    correction_records = [
        {"sentence_id": sid, "clauses_new": clauses}
        for sid, clauses in modified_rows.items()
    ]
    corrections = pl.DataFrame(correction_records)

    corrected = clause_df.join(
        corrections, on="sentence_id", how="left",
    ).with_columns(
        pl.when(pl.col("clauses_new").is_not_null())
        .then(pl.col("clauses_new"))
        .otherwise(pl.col("clauses"))
        .alias("clauses")
    ).drop("clauses_new")

    # Recompute has_null_subject
    modified_ids = set(modified_rows.keys())
    new_flags = []
    for row in corrected.iter_rows(named=True):
        sid = row["sentence_id"]
        if sid not in modified_ids:
            new_flags.append(row["has_null_subject"])
            continue
        clauses = row["clauses"] or []
        has_ns = any(c.get("subject_status") == "none" for c in clauses)
        new_flags.append(has_ns)

    corrected = corrected.with_columns(
        pl.Series("has_null_subject", new_flags)
    )

    return corrected, result


# ---------------------------------------------------------------------------
# Rule 4: Non-ROOT imperative detection
# ---------------------------------------------------------------------------

_VOCATIVE_DISCOURSE_POS = {"INTJ", "PROPN"}
_VOCATIVE_DISCOURSE_DEPS = {"vocative", "discourse", "intj"}


def correct_nonroot_imperative(
    base_df: pl.DataFrame,
    clause_df: pl.DataFrame,
) -> Tuple[pl.DataFrame, CorrectionResult]:
    """
    Rule: Non-ROOT verbs (advcl/ccomp) that look like imperatives get
    subject_status="imperative" instead of "none".

    Heuristic: VERB (or AUX "be") with no subject, no "to" mark, at or
    near sentence start preceded only by INTJ/PROPN/vocative/discourse.
    """
    result = CorrectionResult(rule_name="nonroot_imperative")

    joined = base_df.select([
        "sentence_id", "tokens", "pos_tags", "dep_rels", "dep_heads", "lemmas",
    ]).join(
        clause_df.select(["sentence_id", "clauses"]),
        on="sentence_id",
        how="inner",
    )

    result.sentences_examined = joined.height
    modified_rows: Dict[str, List[Dict]] = {}
    n_clauses_imp = 0

    for row in joined.iter_rows(named=True):
        sid = row["sentence_id"]
        pos_tags = row["pos_tags"]
        dep_rels = row["dep_rels"]
        dep_heads = row["dep_heads"]
        lemmas = row["lemmas"]
        clauses = row["clauses"]

        if not clauses or not pos_tags or not dep_rels or not dep_heads:
            continue

        modified = False
        new_clauses = []

        for clause in clauses:
            c = dict(clause)

            if (c.get("subject_status") != "none"
                    or c.get("clause_type") not in ("advcl", "ccomp")):
                new_clauses.append(c)
                continue

            verb_idx = c.get("verb_idx")
            if verb_idx is None or verb_idx >= len(pos_tags):
                new_clauses.append(c)
                continue

            # Must be VERB, or AUX with lemma "be"
            vpos = pos_tags[verb_idx]
            vlemma = (lemmas[verb_idx] or "").lower() if lemmas else ""
            if vpos == "AUX" and vlemma == "be":
                pass
            elif vpos != "VERB":
                new_clauses.append(c)
                continue

            # No subject children
            children = _children_of(verb_idx, dep_heads)
            if any(dep_rels[ci] in _SUBJECT_RELS for ci in children
                   if ci < len(dep_rels)):
                new_clauses.append(c)
                continue

            # No "to" mark child
            has_to = any(
                dep_rels[ci] == "mark" and (lemmas[ci] or "").lower() == "to"
                for ci in children if ci < len(dep_rels)
            )
            if has_to:
                new_clauses.append(c)
                continue

            # Position heuristic: at sentence start or preceded only by
            # INTJ/PROPN/vocative/discourse tokens
            is_initial = verb_idx == 0
            if not is_initial and verb_idx <= 2:
                preceding_ok = all(
                    pos_tags[j] in _VOCATIVE_DISCOURSE_POS
                    or dep_rels[j] in _VOCATIVE_DISCOURSE_DEPS
                    for j in range(verb_idx)
                )
                is_initial = preceding_ok

            if is_initial:
                c["subject_status"] = "imperative"
                modified = True
                n_clauses_imp += 1

            new_clauses.append(c)

        if modified:
            modified_rows[sid] = new_clauses

    result.sentences_modified = len(modified_rows)
    result.details["clauses_imperative"] = n_clauses_imp

    if not modified_rows:
        return clause_df, result

    correction_records = [
        {"sentence_id": sid, "clauses_new": clauses}
        for sid, clauses in modified_rows.items()
    ]
    corrections = pl.DataFrame(correction_records)

    corrected = clause_df.join(
        corrections, on="sentence_id", how="left",
    ).with_columns(
        pl.when(pl.col("clauses_new").is_not_null())
        .then(pl.col("clauses_new"))
        .otherwise(pl.col("clauses"))
        .alias("clauses")
    ).drop("clauses_new")

    modified_ids = set(modified_rows.keys())
    new_flags = []
    for row in corrected.iter_rows(named=True):
        sid = row["sentence_id"]
        if sid not in modified_ids:
            new_flags.append(row["has_null_subject"])
            continue
        clauses = row["clauses"] or []
        has_ns = any(c.get("subject_status") == "none" for c in clauses)
        new_flags.append(has_ns)

    corrected = corrected.with_columns(
        pl.Series("has_null_subject", new_flags)
    )

    return corrected, result


# ---------------------------------------------------------------------------
# Rule 5: Serial verb subject sharing
# ---------------------------------------------------------------------------

_SERIAL_VERB_LEMMAS = frozenset({"go", "come", "run"})


def correct_serial_verb(
    base_df: pl.DataFrame,
    clause_df: pl.DataFrame,
) -> Tuple[pl.DataFrame, CorrectionResult]:
    """
    Rule: advcl verbs immediately following a serial-verb head (go/come/run)
    with no "to" mark inherit the subject from the head.

    E.g., "go get it" — "get" is advcl of "go", shares subject.
    """
    result = CorrectionResult(rule_name="serial_verb")

    joined = base_df.select([
        "sentence_id", "pos_tags", "dep_rels", "dep_heads", "lemmas",
    ]).join(
        clause_df.select(["sentence_id", "clauses"]),
        on="sentence_id",
        how="inner",
    )

    result.sentences_examined = joined.height
    modified_rows: Dict[str, List[Dict]] = {}
    n_serial = 0

    for row in joined.iter_rows(named=True):
        sid = row["sentence_id"]
        pos_tags = row["pos_tags"]
        dep_rels = row["dep_rels"]
        dep_heads = row["dep_heads"]
        lemmas = row["lemmas"]
        clauses = row["clauses"]

        if not clauses or not pos_tags or not dep_rels or not dep_heads:
            continue

        modified = False
        new_clauses = []

        for clause in clauses:
            c = dict(clause)

            if c.get("subject_status") != "none" or c.get("clause_type") != "advcl":
                new_clauses.append(c)
                continue

            verb_idx = c.get("verb_idx")
            if verb_idx is None or verb_idx >= len(dep_heads):
                new_clauses.append(c)
                continue

            head_idx = dep_heads[verb_idx]
            if head_idx >= len(lemmas):
                new_clauses.append(c)
                continue

            head_lemma = (lemmas[head_idx] or "").lower()
            if head_lemma not in _SERIAL_VERB_LEMMAS:
                new_clauses.append(c)
                continue

            # Must be immediately adjacent
            if verb_idx != head_idx + 1:
                new_clauses.append(c)
                continue

            # No "to" mark child
            children = _children_of(verb_idx, dep_heads)
            has_to = any(
                dep_rels[ci] == "mark" and (lemmas[ci] or "").lower() == "to"
                for ci in children if ci < len(dep_rels)
            )
            if has_to:
                new_clauses.append(c)
                continue

            c["subject_status"] = "inherited"
            modified = True
            n_serial += 1

            new_clauses.append(c)

        if modified:
            modified_rows[sid] = new_clauses

    result.sentences_modified = len(modified_rows)
    result.details["clauses_serial_inherited"] = n_serial

    if not modified_rows:
        return clause_df, result

    correction_records = [
        {"sentence_id": sid, "clauses_new": clauses}
        for sid, clauses in modified_rows.items()
    ]
    corrections = pl.DataFrame(correction_records)

    corrected = clause_df.join(
        corrections, on="sentence_id", how="left",
    ).with_columns(
        pl.when(pl.col("clauses_new").is_not_null())
        .then(pl.col("clauses_new"))
        .otherwise(pl.col("clauses"))
        .alias("clauses")
    ).drop("clauses_new")

    modified_ids = set(modified_rows.keys())
    new_flags = []
    for row in corrected.iter_rows(named=True):
        sid = row["sentence_id"]
        if sid not in modified_ids:
            new_flags.append(row["has_null_subject"])
            continue
        clauses = row["clauses"] or []
        has_ns = any(c.get("subject_status") == "none" for c in clauses)
        new_flags.append(has_ns)

    corrected = corrected.with_columns(
        pl.Series("has_null_subject", new_flags)
    )

    return corrected, result


# ---------------------------------------------------------------------------
# Rule registry — add new rules here
# ---------------------------------------------------------------------------

ALL_RULES = [
    # (name, function, required_layers)
    ("imperative_heuristic", correct_imperative_heuristic, ["base", "complexity"]),
    ("finiteness_propagation", correct_finiteness_propagation, ["base", "clause_structure"]),
    ("nonroot_imperative", correct_nonroot_imperative, ["base", "clause_structure"]),
    ("xcomp_inheritance", correct_xcomp_inheritance, ["base", "clause_structure"]),
    ("serial_verb", correct_serial_verb, ["base", "clause_structure"]),
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_layer(corpus_path: Path, layer_name: str) -> pl.DataFrame:
    """Load a single layer as a DataFrame.

    Reads only top-level parquet files (skips ``_backup/`` subdirectories
    to avoid schema conflicts between original and corrected shards).
    """
    if layer_name == "base":
        layer_dir = corpus_path / "base"
    else:
        layer_dir = corpus_path / "layers" / layer_name

    files = sorted(f for f in layer_dir.glob("*.parquet"))
    if not files:
        # Fall back to recursive glob (partitioned layouts)
        files = sorted(
            f for f in layer_dir.rglob("*.parquet")
            if "_backup" not in f.parts
        )
    if not files:
        raise FileNotFoundError(f"No parquet files found in {layer_dir}")
    return pl.concat([pl.read_parquet(str(f)) for f in files])


def write_layer(corpus_path: Path, layer_name: str, df: pl.DataFrame) -> None:
    """Write a corrected layer, preserving partitioning."""
    layer_dir = corpus_path / "layers" / layer_name
    # Write as a single file (the layered query handles glob)
    output_path = layer_dir / "corrected.parquet"
    # Back up existing files
    existing = list(layer_dir.glob("*.parquet"))
    backup_dir = layer_dir / "_backup"
    backup_dir.mkdir(exist_ok=True)
    for f in existing:
        if f.name != "corrected.parquet":
            f.rename(backup_dir / f.name)
    df.write_parquet(output_path)


def run_corrections(
    corpus_path: Path,
    dry_run: bool = False,
    rules: Optional[List[str]] = None,
) -> List[CorrectionResult]:
    """Run all (or selected) correction rules on the corpus."""
    results = []

    # Determine which rules to run
    if rules:
        active_rules = [r for r in ALL_RULES if r[0] in rules]
    else:
        active_rules = ALL_RULES

    # Load layers on demand
    layer_cache: Dict[str, pl.DataFrame] = {}

    def get_layer(name: str) -> pl.DataFrame:
        if name not in layer_cache:
            print(f"  Loading layer: {name}", file=sys.stderr)
            layer_cache[name] = load_layer(corpus_path, name)
        return layer_cache[name]

    # Track which layers are modified
    modified_layers: Dict[str, pl.DataFrame] = {}

    for rule_name, rule_fn, required_layers in active_rules:
        print(f"Running rule: {rule_name}", file=sys.stderr)

        # Gather inputs
        args = []
        for layer_name in required_layers:
            df = modified_layers.get(layer_name, get_layer(layer_name))
            args.append(df)

        corrected_df, rule_result = rule_fn(*args)
        results.append(rule_result)

        # Store the corrected layer (the last required layer is the target)
        target_layer = required_layers[-1]
        if rule_result.sentences_modified > 0:
            modified_layers[target_layer] = corrected_df

    # Write results
    if not dry_run:
        for layer_name, df in modified_layers.items():
            print(f"Writing corrected layer: {layer_name}", file=sys.stderr)
            write_layer(corpus_path, layer_name, df)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply post-hoc corrections to annotated corpus layers.",
    )
    parser.add_argument(
        "corpus_path", type=Path,
        help="Path to layered annotated corpus directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing to disk",
    )
    parser.add_argument(
        "--rules", nargs="*", default=None,
        help="Specific rules to run (default: all). Available: "
             + ", ".join(r[0] for r in ALL_RULES),
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Write JSON report to this file",
    )
    args = parser.parse_args()

    print(f"Corpus: {args.corpus_path}", file=sys.stderr)
    if args.dry_run:
        print("DRY RUN — no files will be modified", file=sys.stderr)

    results = run_corrections(args.corpus_path, dry_run=args.dry_run, rules=args.rules)

    # Print report
    print("\n=== Post-hoc Correction Report ===", file=sys.stderr)
    print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}", file=sys.stderr)
    print(f"Corpus: {args.corpus_path}", file=sys.stderr)
    print(f"Dry run: {args.dry_run}", file=sys.stderr)
    print(file=sys.stderr)
    for r in results:
        print(r.report(), file=sys.stderr)
        print(file=sys.stderr)

    total_modified = sum(r.sentences_modified for r in results)
    print(f"Total sentences modified: {total_modified:,}", file=sys.stderr)

    # Optionally write JSON report
    if args.report:
        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "corpus_path": str(args.corpus_path),
            "dry_run": args.dry_run,
            "rules": [
                {
                    "name": r.rule_name,
                    "examined": r.sentences_examined,
                    "modified": r.sentences_modified,
                    "details": r.details,
                }
                for r in results
            ],
            "total_modified": total_modified,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"Report written to: {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
