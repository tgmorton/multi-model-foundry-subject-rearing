#!/usr/bin/env python
"""
Generate a markdown report of corpus descriptive analyses.

Usage
-----
    python analysis/scripts/generate_corpus_report.py \
        data/annotated_corpus_test_10M/ \
        --layered \
        --output analysis/corpus_descriptives_report/test_10M_report.md
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl


def _load_data(path: Path, layered: bool) -> pl.LazyFrame:
    if layered:
        from analysis.corpus_descriptives.query import load_layered_corpus
        return load_layered_corpus(path)
    elif path.is_file():
        return pl.scan_parquet(path)
    elif path.is_dir():
        return pl.scan_parquet(str(path / "**" / "*.parquet"))
    else:
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)


def _fmt_pct(val, digits=1) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.{digits}f}%"


def _fmt_num(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:,.2f}"
    return f"{val:,}"


def _df_to_md(df: pl.DataFrame, max_rows: int = 50) -> str:
    """Convert a polars DataFrame to a markdown table."""
    if df.is_empty():
        return "*No data.*\n"

    df = df.head(max_rows)
    cols = df.columns
    rows = df.to_dicts()

    # Format values
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            if abs(v) < 1 and abs(v) > 0:
                return f"{v:.4f}"
            return f"{v:,.2f}"
        if isinstance(v, int):
            return f"{v:,}"
        if isinstance(v, bool):
            return str(v)
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in rows:
        line = "| " + " | ".join(fmt(row[c]) for c in cols) + " |"
        lines.append(line)

    if df.height < max_rows:
        pass
    else:
        lines.append(f"\n*Showing first {max_rows} of {df.height} rows.*")

    return "\n".join(lines) + "\n"


class MarkdownReport:
    def __init__(self):
        self._buf = io.StringIO()

    def h1(self, text: str):
        self._buf.write(f"\n# {text}\n\n")

    def h2(self, text: str):
        self._buf.write(f"\n## {text}\n\n")

    def h3(self, text: str):
        self._buf.write(f"\n### {text}\n\n")

    def p(self, text: str):
        self._buf.write(f"{text}\n\n")

    def table(self, df: pl.DataFrame, max_rows: int = 50):
        self._buf.write(_df_to_md(df, max_rows) + "\n")

    def kv(self, label: str, value):
        self._buf.write(f"- **{label}**: {_fmt_num(value)}\n")

    def kv_pct(self, label: str, value):
        self._buf.write(f"- **{label}**: {_fmt_pct(value)}\n")

    def bullet(self, text: str):
        self._buf.write(f"- {text}\n")

    def raw(self, text: str):
        self._buf.write(text + "\n")

    def get(self) -> str:
        return self._buf.getvalue()


def generate_report(lf: pl.LazyFrame, split_name: str = "test_10M") -> str:
    from analysis.corpus_descriptives.corpus_analysis import (
        corpus_overview,
        phenomenon_rates_by_genre,
        cooccurrence_null_subject_that_trace,
        cooccurrence_negation_null_subject,
        cooccurrence_expletive_depth,
        childes_child_vs_adult,
        childes_speaker_distribution,
        childes_pronoun_distribution,
        childes_person_distribution,
        subject_pronoun_distribution,
        that_trace_inventory,
        expletive_by_class,
        complexity_distribution,
        depth_distribution,
        verb_null_subject_rate,
        bridge_verb_null_rates,
        weather_verb_null_rates,
        tense_aspect_mood_distribution,
        register_comparison,
        register_by_genre,
        language_agnostic_analysis,
        h5_distal_effects,
        h6a_expletive_analysis,
        null_subject_by_embedding_depth,
    )

    md = MarkdownReport()
    md.h1(f"Corpus Descriptive Analysis — {split_name}")
    md.p("Auto-generated from sentence-level Parquet annotations.")

    # ── 1. Overview ──────────────────────────────────────────────────

    md.h2("1. Corpus Overview")
    overview = corpus_overview(lf)
    md.table(overview)

    total_sents = overview["n_sentences"].sum()
    total_toks = overview["total_tokens"].sum()
    md.kv("Total sentences", total_sents)
    md.kv("Total tokens", total_toks)
    md.raw("")

    # ── 2. Phenomenon Rates ──────────────────────────────────────────

    md.h2("2. Phenomenon Rates by Genre")
    rates = phenomenon_rates_by_genre(lf)

    # Reshape for readability: one column per phenomenon, rows = genres
    rate_cols = [c for c in rates.columns if c.endswith("_rate")]
    display = rates.select(["genre", "n_sentences"] + rate_cols)
    # Rename for clarity
    rename_map = {c: c.replace("has_", "").replace("_rate", "").replace("is_", "") for c in rate_cols}
    display = display.rename(rename_map)
    md.table(display)

    md.p("*Rates are proportions of sentences exhibiting each phenomenon.*")

    # ── 3. Co-occurrence Analyses ────────────────────────────────────

    md.h2("3. Targeted Co-occurrence Analyses")

    md.h3("3a. Null Subject × That-Trace")
    tt = cooccurrence_null_subject_that_trace(lf)
    md.kv_pct("Baseline null-subject rate", tt["baseline_null_subject_rate"])
    md.kv("That-trace contexts", tt["that_trace_total"])
    md.kv("…with null subject", tt["that_trace_with_null_subject"])
    md.kv_pct("Null rate in that-trace", tt["null_rate_in_that_trace"])
    md.kv("Enrichment ratio", tt["enrichment_ratio"])
    md.raw("")

    md.h3("3b. Negation × Null Subject")
    neg = cooccurrence_negation_null_subject(lf)
    md.table(neg)

    md.h3("3c. Expletive × Embedding Depth")
    expl_depth = cooccurrence_expletive_depth(lf)
    md.kv("Expletive sentence mean depth", expl_depth["expletive_mean_depth"])
    md.kv("Non-expletive sentence mean depth", expl_depth["non_expletive_mean_depth"])
    if expl_depth["expletive_depth_distribution"]:
        md.p("\n**Expletive sentence depth distribution:**")
        dist_df = pl.DataFrame(expl_depth["expletive_depth_distribution"])
        md.table(dist_df)

    # ── 4. CHILDES ───────────────────────────────────────────────────

    md.h2("4. CHILDES Developmental Analysis")

    md.h3("4a. Child vs Adult")
    cva = childes_child_vs_adult(lf)
    md.table(cva)

    # Compute key contrasts
    child_row = cva.filter(pl.col("role") == "child")
    adult_row = cva.filter(pl.col("role") == "adult")
    if child_row.height > 0 and adult_row.height > 0:
        md.p("**Key developmental contrasts:**")
        for col in cva.columns:
            if col.endswith("_rate"):
                label = col.replace("_rate", "")
                c = child_row[col].item()
                a = adult_row[col].item()
                if c is not None and a is not None:
                    md.bullet(f"{label}: child {_fmt_pct(c)}, adult {_fmt_pct(a)}, diff {_fmt_pct(c - a)}")
        md.raw("")

    md.h3("4b. Speaker Distribution")
    md.table(childes_speaker_distribution(lf))

    md.h3("4c. Person Distribution by Role")
    md.table(childes_person_distribution(lf))

    md.h3("4d. Pronoun Distribution by Role")
    cpd = childes_pronoun_distribution(lf)
    # Show top 15 per role
    for role_val in ["child", "adult"]:
        sub = cpd.filter(pl.col("role") == role_val).head(15)
        if not sub.is_empty():
            md.p(f"**{role_val.title()} — top 15 pronouns:**")
            md.table(sub)

    # ── 5. Pronoun Distribution ──────────────────────────────────────

    md.h2("5. Subject Pronoun Distribution")
    spd = subject_pronoun_distribution(lf)
    # Summarise across genres
    total_subj = (
        spd.group_by(["person", "number", "lemma"])
        .agg(pl.col("count").sum())
        .sort("count", descending=True)
    )
    md.table(total_subj.head(20))

    # ── 6. That-Trace Inventory ──────────────────────────────────────

    md.h2("6. That-Trace Environment Inventory")
    tti = that_trace_inventory(lf)
    md.kv("Total that-trace contexts", tti["total_contexts"])
    md.kv_pct("Complementizer (that) rate", tti["complementizer_rate"])
    md.kv("Violations", tti["violation_count"])
    md.raw("")

    if tti["bridge_verb_counts"]:
        md.p("**Bridge verb frequency:**")
        md.table(pl.DataFrame(tti["bridge_verb_counts"]).head(20))

    if tti["extraction_types"]:
        md.p("**Extraction type distribution:**")
        md.table(pl.DataFrame(tti["extraction_types"]))

    # ── 7. Expletives ────────────────────────────────────────────────

    md.h2("7. Expletive Distribution")
    ebc = expletive_by_class(lf)
    md.table(ebc)

    # ── 8. Complexity ────────────────────────────────────────────────

    md.h2("8. Sentence Complexity")
    md.h3("8a. Summary by Genre")
    md.table(complexity_distribution(lf))

    md.h3("8b. Embedding Depth Distribution")
    dd = depth_distribution(lf)
    # Pivot: genre × depth
    pivot = (
        dd.pivot(on="depth", index="genre", values="count")
        .fill_null(0)
    )
    # Sort columns numerically
    depth_cols = sorted([c for c in pivot.columns if c != "genre"], key=lambda x: int(x))
    pivot = pivot.select(["genre"] + depth_cols)
    md.table(pivot)

    # ── 9. Verb Analysis ─────────────────────────────────────────────

    md.h2("9. Verb Lemma Analysis")

    md.h3("9a. Highest Null-Subject Rate (min 10 occ.)")
    vnr = verb_null_subject_rate(lf)
    md.table(vnr.head(20))

    md.h3("9b. Bridge Verb Null-Subject Rates")
    md.table(bridge_verb_null_rates(lf))

    md.h3("9c. Weather Verb Null-Subject Rates")
    md.table(weather_verb_null_rates(lf))

    md.h3("9d. Tense / Aspect / Mood")
    tam = tense_aspect_mood_distribution(lf)
    for key in ["tense", "aspect", "mood", "verb_form"]:
        frame = tam[key]
        # Summarise across genres
        summ = frame.group_by(key).agg(pl.col("count").sum()).sort("count", descending=True)
        md.p(f"**{key.title()} distribution (all genres):**")
        md.table(summ)

    # ── 10. Register ─────────────────────────────────────────────────

    md.h2("10. Spoken vs Written Register")
    md.table(register_comparison(lf))

    md.h3("10b. Detail by Genre")
    md.table(register_by_genre(lf))

    # ── 11. Research Hypotheses ──────────────────────────────────────

    md.h2("11. Research Hypothesis Baselines")

    md.h3("H5: Distal Effects — That-Trace ↔ Null Subjects")
    h5 = h5_distal_effects(lf)
    md.kv_pct("Baseline null-subject rate", h5["baseline_null_subject_rate"])
    md.kv("That-trace contexts", h5["that_trace_total"])
    md.kv("…with null subject", h5["that_trace_with_null_subject"])
    md.kv("Enrichment ratio", h5["enrichment_ratio"])
    md.raw("")
    md.p("If enrichment ≈ 1, null subjects and that-trace contexts are independent in the training data.")

    md.h3("H6a: Expletive Distribution")
    h6 = h6a_expletive_analysis(lf)
    md.kv("Total expletive sentences", h6["total_expletive_sentences"])
    md.kv("…also with null subject", h6["expletive_with_null_subject"])
    md.kv_pct("Null rate in expletive sentences", h6["null_rate_in_expletive_sents"])
    md.raw("")

    md.h3("Null Subject Rate by Embedding Depth")
    md.table(null_subject_by_embedding_depth(lf))

    # ── Language-agnostic stats ──────────────────────────────────────

    md.h2("12. Language-Agnostic Summary (for cross-linguistic comparison)")
    la = language_agnostic_analysis(lf, "English")
    for k, v in la.items():
        if k == "language":
            continue
        if k == "subject_pronoun_person_distribution":
            md.p(f"**{k}:**")
            md.table(pl.DataFrame(v))
        elif isinstance(v, float):
            md.kv_pct(k, v)
        else:
            md.kv(k, v)

    md.raw("")
    md.p("---")
    md.p("*Report generated by `analysis/scripts/generate_corpus_report.py`.*")

    return md.get()


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown report of corpus descriptive analyses.",
    )
    parser.add_argument(
        "input_path", type=Path,
        help="Path to Parquet file or annotated corpus directory",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output .md file (default: print to stdout)",
    )
    parser.add_argument(
        "--layered", action="store_true",
        help="Input is a layered corpus directory (base/ + layers/)",
    )
    parser.add_argument(
        "--split-name", default=None,
        help="Split name for report title (auto-detected from metadata.json if omitted)",
    )
    args = parser.parse_args()

    # Auto-detect split name
    split_name = args.split_name
    if split_name is None:
        meta_path = args.input_path / "metadata.json" if args.input_path.is_dir() else None
        if meta_path and meta_path.exists():
            import json
            split_name = json.loads(meta_path.read_text()).get("split_name", "corpus")
        else:
            split_name = args.input_path.stem

    print(f"Loading annotations from: {args.input_path}", file=sys.stderr)
    lf = _load_data(args.input_path, args.layered)
    print(f"Columns: {lf.collect_schema().names()}", file=sys.stderr)

    print("Running analyses…", file=sys.stderr)
    report = generate_report(lf, split_name)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Report written to: {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
