#!/usr/bin/env python
"""
Generate a focused interactive HTML report on null subject use.

Produces a single self-contained HTML file with 5 sections:
  1. Overview — KPI cards (total sentences, finite clauses, overall null rate)
  2. Child vs Adult — clause-level null-subject rate comparison
  3. By Genre — clause-level null-subject rate per genre
  4. By Clause Context — null-subject rate by clause type (readable names)
  5. Developmental Trajectory — null-subject rate by child age band + MLU

Processes parquet files one at a time to stay within memory limits on
large corpora (90M+).

Usage
-----
    PYTHONPATH=. python -m analysis.scripts.reporting.generate_null_subject_report \
        data/output/test_10M/annotated_corpus/ --layered --split-name test_10M \
        --output data/output/test_10M/null_subject_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NULL_STATUSES = ["none", "imperative"]

_BASE_COLS = ["sentence_id", "genre", "role", "child_age_months", "n_tokens"]

_AGE_BINS = [
    (12, 24, "12-23m"),
    (24, 36, "24-35m"),
    (36, 48, "36-47m"),
    (48, 60, "48-59m"),
    (60, 200, "60m+"),
]

_CLAUSE_TYPE_LABELS = {
    "ROOT": "Main clause",
    "ccomp": "Complement clause",
    "advcl": "Adverbial clause",
    "xcomp": "Control/raising complement",
    "acl": "Adnominal clause",
    "acl:relcl": "Relative clause",
    "relcl": "Relative clause",
    "conj": "Coordinated clause",
    "parataxis": "Parataxis",
}

_TABLEAU_10 = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, float):
        if obj != obj:
            return None
        if obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    return obj


def _fmt_pct(val: Any) -> str:
    if val is None:
        return "\u2014"
    return f"{val * 100:.1f}%"


def _fmt_num(val: Any) -> str:
    if val is None:
        return "\u2014"
    if isinstance(val, float):
        return f"{val:,.2f}"
    return f"{val:,}"


def _color_palette(n: int) -> List[str]:
    return [_TABLEAU_10[i % len(_TABLEAU_10)] for i in range(n)]


def _age_band(months: Optional[int]) -> Optional[str]:
    if months is None:
        return None
    for lo, hi, label in _AGE_BINS:
        if lo <= months < hi:
            return label
    return None


# ---------------------------------------------------------------------------
# File-by-file aggregation
# ---------------------------------------------------------------------------

def _load_base(corpus_path: Path) -> pl.DataFrame:
    """Load base layer with minimal columns.  Small enough to fit in memory."""
    base_dir = corpus_path / "base"
    return pl.read_parquet(
        sorted(base_dir.glob("*.parquet")),
        columns=_BASE_COLS,
    )


def _iter_clause_files(corpus_path: Path) -> List[Path]:
    """Return sorted list of clause_structure parquet files."""
    clause_dir = corpus_path / "layers" / "clause_structure"
    return sorted(f for f in clause_dir.glob("*.parquet") if f.stem != "_backup")


def _process_chunk(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Explode clauses in a DataFrame chunk and return partial count frames.

    Returns dict with keys: overview, by_genre, by_role, by_clause_type,
    by_age_band, mlu_by_age.  Each is a small DataFrame of counts.
    """

    n_sents = df.height
    total_tokens = df["n_tokens"].sum()

    # Explode clauses, extract fields, filter finite, add is_null
    exploded = (
        df.select(["genre", "role", "child_age_months", "n_tokens", "clauses"])
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .select([
            "genre", "role", "child_age_months", "n_tokens",
            pl.col("clauses").struct.field("clause_type").alias("clause_type"),
            pl.col("clauses").struct.field("subject_status").alias("subject_status"),
            pl.col("clauses").struct.field("is_finite").alias("is_finite"),
        ])
        .filter(pl.col("is_finite") == True)
        .with_columns(
            pl.col("subject_status").is_in(_NULL_STATUSES).alias("is_null"),
        )
        .drop("is_finite")
    )

    # Overview: total finite clauses and null count
    overview = pl.DataFrame({
        "n_sents": [n_sents],
        "total_tokens": [total_tokens],
        "n_clauses": [exploded.height],
        "null_count": [exploded["is_null"].sum()],
    })

    # By genre
    by_genre = (
        exploded.group_by("genre")
        .agg([
            pl.col("is_null").sum().alias("null_count"),
            pl.len().alias("n_clauses"),
        ])
    )

    # By role (CHILDES child vs adult)
    childes = exploded.filter(
        pl.col("genre").str.starts_with("CHILDES")
        & pl.col("role").is_in(["child", "adult"])
    )
    by_role = (
        childes.group_by("role")
        .agg([
            pl.col("is_null").sum().alias("null_count"),
            pl.len().alias("n_clauses"),
        ])
    )

    # By clause type
    by_clause_type = (
        exploded.group_by("clause_type")
        .agg([
            pl.col("is_null").sum().alias("null_count"),
            pl.len().alias("n_clauses"),
        ])
    )

    # By age band (CHILDES children only)
    child_rows = exploded.filter(
        pl.col("genre").str.starts_with("CHILDES")
        & (pl.col("role") == "child")
        & pl.col("child_age_months").is_not_null()
        & (pl.col("child_age_months") >= 12)
    )
    if child_rows.height > 0:
        child_rows = child_rows.with_columns(
            pl.col("child_age_months").map_elements(_age_band, return_dtype=pl.Utf8).alias("age_band"),
        ).filter(pl.col("age_band").is_not_null())

        by_age_band = (
            child_rows.group_by("age_band")
            .agg([
                pl.col("is_null").sum().alias("null_count"),
                pl.len().alias("n_clauses"),
            ])
        )
    else:
        by_age_band = pl.DataFrame({"age_band": [], "null_count": [], "n_clauses": []},
                                   schema={"age_band": pl.Utf8, "null_count": pl.Int64, "n_clauses": pl.UInt32})

    # MLU by age band (sentence-level, not clause-level)
    child_sents = df.filter(
        pl.col("genre").str.starts_with("CHILDES")
        & (pl.col("role") == "child")
        & pl.col("child_age_months").is_not_null()
        & (pl.col("child_age_months") >= 12)
    )
    if child_sents.height > 0:
        child_sents = child_sents.with_columns(
            pl.col("child_age_months").map_elements(_age_band, return_dtype=pl.Utf8).alias("age_band"),
        ).filter(pl.col("age_band").is_not_null())

        mlu_by_age = (
            child_sents.group_by("age_band")
            .agg([
                pl.col("n_tokens").sum().alias("token_sum"),
                pl.len().alias("n_sents"),
            ])
        )
    else:
        mlu_by_age = pl.DataFrame({"age_band": [], "token_sum": [], "n_sents": []},
                                  schema={"age_band": pl.Utf8, "token_sum": pl.Int64, "n_sents": pl.UInt32})

    del df, exploded
    return {
        "overview": overview,
        "by_genre": by_genre,
        "by_role": by_role,
        "by_clause_type": by_clause_type,
        "by_age_band": by_age_band,
        "mlu_by_age": mlu_by_age,
    }


def _combine_partials(partials: List[Dict[str, pl.DataFrame]]) -> Dict[str, pl.DataFrame]:
    """Sum partial count frames across files."""

    def _sum_group(frames: List[pl.DataFrame], key_col: str) -> pl.DataFrame:
        combined = pl.concat([f for f in frames if f.height > 0], how="vertical_relaxed")
        if combined.height == 0:
            return combined
        return combined.group_by(key_col).agg([
            pl.col(c).sum() for c in combined.columns if c != key_col
        ])

    overview = pl.concat([p["overview"] for p in partials]).select([
        pl.col("n_sents").sum(),
        pl.col("total_tokens").sum(),
        pl.col("n_clauses").sum(),
        pl.col("null_count").sum(),
    ])

    return {
        "overview": overview,
        "by_genre": _sum_group([p["by_genre"] for p in partials], "genre"),
        "by_role": _sum_group([p["by_role"] for p in partials], "role"),
        "by_clause_type": _sum_group([p["by_clause_type"] for p in partials], "clause_type"),
        "by_age_band": _sum_group([p["by_age_band"] for p in partials], "age_band"),
        "mlu_by_age": _sum_group([p["mlu_by_age"] for p in partials], "age_band"),
    }


# ---------------------------------------------------------------------------
# Build report sections from aggregated counts
# ---------------------------------------------------------------------------

def _build_sections(agg: Dict[str, pl.DataFrame]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build all 5 report sections from aggregated counts.

    Returns (sections_dict, metadata_dict).
    """
    sections: Dict[str, Any] = {}

    ov = agg["overview"]
    n_sents = ov["n_sents"].item()
    total_tokens = ov["total_tokens"].item()
    n_clauses = ov["n_clauses"].item()
    null_count = ov["null_count"].item()
    null_rate = null_count / n_clauses if n_clauses > 0 else None

    # --- 1. Overview ---
    sections["overview"] = {
        "title": "1. Overview",
        "description": (
            "Headline statistics. Null subject = clause with no overt subject "
            "(includes imperatives, diary drop, and all other null-subject types). "
            "All rates are clause-level, finite clauses only."
        ),
        "kpis": [
            {"label": "Total Sentences", "value": _fmt_num(n_sents), "sub": "across all genres"},
            {"label": "Finite Clauses", "value": _fmt_num(n_clauses), "sub": "analysis denominator"},
            {"label": "Null-Subject Rate", "value": _fmt_pct(null_rate), "sub": "clause-level, finite only"},
        ],
        "table": None, "columns": [], "chart_data": None,
    }

    # --- 2. Child vs Adult ---
    role_df = agg["by_role"]
    if role_df.height > 0:
        role_df = role_df.with_columns(
            (pl.col("null_count") / pl.col("n_clauses")).alias("null_rate"),
        ).sort("role")
        records = _safe_json(role_df.to_dicts())
        display = [{"role": r["role"], "null_rate": r["null_rate"], "n_clauses": r["n_clauses"]} for r in records]
        roles = [r["role"] for r in records]
        rates = _safe_json([r["null_rate"] for r in records])
        sections["child_adult"] = {
            "title": "2. Child vs Adult",
            "description": "Clause-level null-subject rate for CHILDES children vs adults (finite clauses only).",
            "table": display, "columns": ["role", "null_rate", "n_clauses"],
            "chart_data": {
                "type": "bar",
                "data": {"labels": roles, "datasets": [{"label": "Null-subject rate", "data": rates, "backgroundColor": ["#e15759", "#4e79a7"]}]},
                "options": {"plugins": {"title": {"display": True, "text": "Null-Subject Rate: Child vs Adult"}, "legend": {"display": False}},
                            "scales": {"y": {"title": {"display": True, "text": "Null-subject rate"}, "beginAtZero": True, "max": 1}}},
            },
        }
    else:
        sections["child_adult"] = {"title": "2. Child vs Adult", "description": "No CHILDES data available.", "table": [], "columns": [], "chart_data": None}

    # --- 3. By Genre ---
    genre_df = agg["by_genre"]
    if genre_df.height > 0:
        genre_df = genre_df.with_columns(
            (pl.col("null_count") / pl.col("n_clauses")).alias("null_rate"),
        ).sort("genre")
        records = _safe_json(genre_df.to_dicts())
        genres = [r["genre"] for r in records]
        rates = _safe_json([r["null_rate"] for r in records])
        palette = _color_palette(len(genres))
        sections["by_genre"] = {
            "title": "3. By Genre",
            "description": "Clause-level null-subject rate per genre (finite clauses only).",
            "table": records, "columns": genre_df.columns,
            "chart_data": {
                "type": "bar",
                "data": {"labels": genres, "datasets": [{"label": "Null-subject rate", "data": rates, "backgroundColor": palette}]},
                "options": {"indexAxis": "y", "plugins": {"title": {"display": True, "text": "Null-Subject Rate by Genre"}, "legend": {"display": False}},
                            "scales": {"x": {"title": {"display": True, "text": "Null-subject rate"}, "max": 1, "beginAtZero": True}}},
            },
        }
    else:
        sections["by_genre"] = {"title": "3. By Genre", "description": "No data.", "table": [], "columns": [], "chart_data": None}

    # --- 4. By Clause Context ---
    ct_df = agg["by_clause_type"]
    if ct_df.height > 0:
        ct_df = ct_df.with_columns(
            (pl.col("null_count") / pl.col("n_clauses")).alias("null_rate"),
        ).sort("null_rate", descending=True)
        records = _safe_json(ct_df.to_dicts())
        for r in records:
            raw = r.get("clause_type", "")
            r["clause_type"] = _CLAUSE_TYPE_LABELS.get(raw, raw)
        labels = [r["clause_type"] for r in records]
        rates = _safe_json([r["null_rate"] for r in records])
        palette = _color_palette(len(labels))
        sections["by_context"] = {
            "title": "4. By Clause Context",
            "description": "Null-subject rate by clause type (finite clauses only).",
            "table": records, "columns": ct_df.columns,
            "chart_data": {
                "type": "bar",
                "data": {"labels": labels, "datasets": [{"label": "Null-subject rate", "data": rates, "backgroundColor": palette}]},
                "options": {"indexAxis": "y", "plugins": {"title": {"display": True, "text": "Null-Subject Rate by Clause Context"}, "legend": {"display": False}},
                            "scales": {"x": {"title": {"display": True, "text": "Null-subject rate"}, "max": 1, "beginAtZero": True}}},
            },
        }
    else:
        sections["by_context"] = {"title": "4. By Clause Context", "description": "No data.", "table": [], "columns": [], "chart_data": None}

    # --- 5. Developmental Trajectory ---
    age_df = agg["by_age_band"]
    mlu_df = agg["mlu_by_age"]
    if age_df.height > 0:
        age_df = age_df.with_columns(
            (pl.col("null_count") / pl.col("n_clauses")).alias("null_rate"),
        )
        if mlu_df.height > 0:
            mlu_df = mlu_df.with_columns(
                (pl.col("token_sum") / pl.col("n_sents")).alias("mlu"),
            ).select(["age_band", "mlu"])
            age_df = age_df.join(mlu_df, on="age_band", how="left")
        age_df = age_df.sort("age_band")
        records = _safe_json(age_df.to_dicts())
        age_bands = [r["age_band"] for r in records]
        ns_rates = _safe_json([r.get("null_rate", 0) for r in records])
        mlu_values = _safe_json([r.get("mlu", 0) for r in records])
        sections["developmental"] = {
            "title": "5. Developmental Trajectory",
            "description": "Clause-level null-subject rate and MLU by child age band.",
            "table": records, "columns": age_df.columns,
            "chart_data": None,
            "charts": [
                {"id": "dev-null-rate", "config": {
                    "type": "line",
                    "data": {"labels": age_bands, "datasets": [{"label": "Null-subject rate", "data": ns_rates, "borderColor": "#e15759", "backgroundColor": "#e1575933", "fill": True, "tension": 0.3, "pointRadius": 5, "borderWidth": 2}]},
                    "options": {"plugins": {"title": {"display": True, "text": "Null-Subject Rate by Age"}}, "scales": {"x": {"title": {"display": True, "text": "Age Band"}}, "y": {"title": {"display": True, "text": "Null-subject rate"}, "max": 1, "beginAtZero": True}}},
                }},
                {"id": "dev-mlu", "config": {
                    "type": "line",
                    "data": {"labels": age_bands, "datasets": [{"label": "MLU (words)", "data": mlu_values, "borderColor": "#4e79a7", "backgroundColor": "#4e79a733", "fill": True, "tension": 0.3, "pointRadius": 5, "borderWidth": 2}]},
                    "options": {"plugins": {"title": {"display": True, "text": "Mean Length of Utterance by Age"}}, "scales": {"x": {"title": {"display": True, "text": "Age Band"}}, "y": {"title": {"display": True, "text": "MLU (words)"}, "beginAtZero": True}}},
                }},
            ],
        }
    else:
        sections["developmental"] = {"title": "5. Developmental Trajectory", "description": "No age-stratified CHILDES data available.", "table": [], "columns": [], "chart_data": None}

    metadata = {"total_sentences": n_sents, "total_tokens": total_tokens}
    return sections, metadata


# ---------------------------------------------------------------------------
# HTML builder (unchanged)
# ---------------------------------------------------------------------------

def _build_table_html(rows: List[Dict], columns: List[str]) -> str:
    if not rows:
        return '<p><em>No data.</em></p>'
    lines = ['<div class="table-wrap"><table>']
    lines.append('<thead><tr>')
    for col in columns:
        lines.append(f'<th>{html.escape(col.replace("_", " ").title())}</th>')
    lines.append('</tr></thead><tbody>')
    for row in rows:
        lines.append('<tr>')
        for col in columns:
            lines.append(f'<td>{html.escape(_format_cell(row.get(col)))}</td>')
        lines.append('</tr>')
    lines.append('</tbody></table></div>')
    return "\n".join(lines)


def _format_cell(val: Any) -> str:
    if val is None:
        return "\u2014"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        return f"{val:.4f}" if 0 < abs(val) < 1 else f"{val:,.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    if isinstance(val, (list, dict)):
        return "\u2026"
    return str(val)


def _build_html(report_data: Dict[str, Any]) -> str:
    meta = report_data["metadata"]
    sections = report_data["sections"]
    json_blob = json.dumps(report_data, indent=None, default=str)

    section_html_parts = []
    nav_links = []

    for sec_key, sec in sections.items():
        sec_id = sec_key.replace("_", "-")
        title = sec.get("title", sec_key)
        desc = sec.get("description", "")
        table_data = sec.get("table")
        columns = sec.get("columns", [])
        chart_data = sec.get("chart_data")
        kpis = sec.get("kpis")

        nav_links.append(f'<a href="#{sec_id}">{html.escape(title)}</a>')
        parts = [f'<section id="{sec_id}">', f'<h2>{html.escape(title)}</h2>']
        if desc:
            parts.append(f'<p class="sec-desc">{html.escape(desc)}</p>')
        if kpis:
            kpi_html = '<div class="kpi-row">'
            for kpi in kpis:
                kpi_html += (f'<div class="kpi-card"><div class="kpi-value">{html.escape(str(kpi["value"]))}</div>'
                             f'<div class="kpi-label">{html.escape(kpi["label"])}</div>'
                             f'<div class="kpi-sub">{html.escape(kpi.get("sub", ""))}</div></div>')
            kpi_html += '</div>'
            parts.append(kpi_html)
        if chart_data:
            parts.append(f'<div class="chart-container"><canvas id="chart-{sec_id}"></canvas></div>')
        for chart in sec.get("charts", []):
            parts.append(f'<div class="chart-container"><canvas id="chart-{chart["id"]}"></canvas></div>')
        if table_data and isinstance(table_data, list) and table_data and isinstance(table_data[0], dict):
            cols = columns if columns else list(table_data[0].keys())
            parts.append(_build_table_html(table_data, cols))
        parts.append('</section>')
        section_html_parts.append("\n".join(parts))

    sections_html = "\n\n".join(section_html_parts)
    nav_html = "\n".join(nav_links)

    chart_inits = []
    for sec_key, sec in sections.items():
        cd = sec.get("chart_data")
        if cd:
            chart_inits.append(f'createChart("chart-{sec_key.replace("_", "-")}", {json.dumps(cd, default=str)});')
        for chart in sec.get("charts", []):
            chart_inits.append(f'createChart("chart-{chart["id"]}", {json.dumps(chart["config"], default=str)});')
    chart_init_js = "\n    ".join(chart_inits)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Null Subject Analysis \u2014 {html.escape(meta['split_name'])}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
:root {{ --bg:#fafafa;--surface:#fff;--text:#1a1a2e;--text-muted:#555;--border:#e0e0e0;--accent:#c2185b;--sidebar-w:240px }}
* {{ margin:0;padding:0;box-sizing:border-box }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;display:flex;min-height:100vh }}
nav#sidebar {{ position:sticky;top:0;width:var(--sidebar-w);min-width:var(--sidebar-w);height:100vh;overflow-y:auto;background:var(--surface);border-right:1px solid var(--border);padding:1rem .5rem;font-size:.82rem }}
nav#sidebar a {{ display:block;padding:4px 8px;color:var(--text-muted);text-decoration:none;border-radius:4px;margin-bottom:2px }}
nav#sidebar a:hover,nav#sidebar a.active {{ background:#fce4ec;color:var(--accent) }}
main {{ flex:1;max-width:1100px;padding:2rem 2.5rem }}
header {{ margin-bottom:2rem }}
header h1 {{ font-size:1.6rem;margin-bottom:.3rem;color:var(--accent) }}
header .meta {{ color:var(--text-muted);font-size:.85rem }}
section {{ background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.5rem;margin-bottom:1.5rem }}
section h2 {{ font-size:1.15rem;margin-bottom:.4rem;color:var(--accent) }}
.sec-desc {{ color:var(--text-muted);font-size:.85rem;margin-bottom:.8rem }}
.chart-container {{ position:relative;max-height:400px;margin-bottom:1rem }}
.kpi-row {{ display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap }}
.kpi-card {{ flex:1;min-width:140px;background:linear-gradient(135deg,#fce4ec 0%,#f8bbd0 100%);border-radius:8px;padding:1rem;text-align:center }}
.kpi-value {{ font-size:1.8rem;font-weight:700;color:var(--accent);line-height:1.2 }}
.kpi-label {{ font-size:.85rem;font-weight:600;color:var(--text);margin-top:.2rem }}
.kpi-sub {{ font-size:.75rem;color:var(--text-muted);margin-top:.1rem }}
table {{ width:100%;border-collapse:collapse;font-size:.82rem;margin-top:.5rem }}
th,td {{ padding:5px 8px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap }}
th {{ background:#f4f6f8;font-weight:600;position:sticky;top:0 }}
tr:hover {{ background:#f9fbfd }}
.table-wrap {{ max-height:420px;overflow:auto }}
@media (max-width:900px) {{ nav#sidebar {{ display:none }} main {{ padding:1rem }} .kpi-row {{ flex-direction:column }} }}
</style>
</head>
<body>
<nav id="sidebar">
  <div style="font-weight:700;padding:0 8px 8px;font-size:.9rem;color:var(--accent)">Null Subjects</div>
  {nav_html}
</nav>
<main>
<header>
  <h1>Null Subject Analysis \u2014 {html.escape(meta['split_name'])}</h1>
  <div class="meta">
    Total sentences: {_fmt_num(meta.get('total_sentences'))} &middot;
    Total tokens: {_fmt_num(meta.get('total_tokens'))} &middot;
    Generated: {html.escape(meta.get('generated_at', ''))}
  </div>
</header>
{sections_html}
</main>
<script>
const REPORT_DATA = {json_blob};
function createChart(canvasId, config) {{
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    new Chart(canvas, {{
        type: config.type, data: config.data,
        options: Object.assign({{ responsive:true, maintainAspectRatio:true, animation:{{duration:400}},
            plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:11}}}}}}}} }}, config.options || {{}}),
    }});
}}
document.addEventListener('DOMContentLoaded', function() {{
    {chart_init_js}
}});
(function() {{
    var links = document.querySelectorAll('nav#sidebar a');
    var sections = document.querySelectorAll('main section');
    if (sections.length > 0 && typeof IntersectionObserver !== 'undefined') {{
        var observer = new IntersectionObserver(function(entries) {{
            entries.forEach(function(entry) {{
                if (entry.isIntersecting) {{
                    links.forEach(function(a) {{ a.classList.remove('active') }});
                    var m = document.querySelector('nav#sidebar a[href="#'+entry.target.id+'"]');
                    if (m) m.classList.add('active');
                }}
            }});
        }}, {{ rootMargin: '-20% 0px -70% 0px' }});
        sections.forEach(function(s) {{ observer.observe(s) }});
    }}
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate null subject HTML report.")
    parser.add_argument("input_path", type=Path, help="Annotated corpus directory")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--layered", action="store_true")
    parser.add_argument("--split-name", default=None)
    args = parser.parse_args()

    split_name = args.split_name
    if split_name is None:
        meta_path = args.input_path / "metadata.json" if args.input_path.is_dir() else None
        if meta_path and meta_path.exists():
            split_name = json.loads(meta_path.read_text()).get("split_name", "corpus")
        else:
            split_name = args.input_path.stem

    output = args.output
    if output is None:
        output = Path("analysis/output/corpus_descriptives/reports/null_subject_report.html")

    # Chunk size: number of rows per processing batch within each file.
    CHUNK_SIZE = 500_000

    if args.layered:
        print(f"Loading from: {args.input_path}", file=sys.stderr)
        # Load base once (small: 5 columns, ~1-2 GB for millions of rows)
        base = _load_base(args.input_path)
        print(f"  Base loaded: {base.height:,} rows", file=sys.stderr)

        clause_files = _iter_clause_files(args.input_path)
        print(f"  Found {len(clause_files)} clause_structure file(s)", file=sys.stderr)

        partials = []
        for fi, cpath in enumerate(clause_files):
            print(f"  File [{fi+1}/{len(clause_files)}] {cpath.name}", file=sys.stderr)
            clauses = pl.read_parquet(cpath, columns=["sentence_id", "clauses"])
            joined = base.join(clauses, on="sentence_id", how="inner")
            del clauses

            n_rows = joined.height
            n_chunks = (n_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
            for ci in range(n_chunks):
                start = ci * CHUNK_SIZE
                chunk = joined.slice(start, CHUNK_SIZE)
                print(f"    chunk {ci+1}/{n_chunks}: {chunk.height:,} rows", file=sys.stderr)
                partials.append(_process_chunk(chunk))
                del chunk
            del joined

        del base
    elif args.input_path.is_file():
        print(f"Loading from: {args.input_path}", file=sys.stderr)
        df = pl.read_parquet(args.input_path)
        n_rows = df.height
        n_chunks = (n_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"Processing {n_rows:,} rows in {n_chunks} chunk(s)\u2026", file=sys.stderr)
        partials = []
        for i in range(n_chunks):
            start = i * CHUNK_SIZE
            chunk = df.slice(start, CHUNK_SIZE)
            print(f"  [{i+1}/{n_chunks}] rows {start:,}\u2013{start + chunk.height:,}\u2026", file=sys.stderr)
            partials.append(_process_chunk(chunk))
            del chunk
        del df
    elif args.input_path.is_dir():
        print(f"Loading from: {args.input_path}", file=sys.stderr)
        df = pl.read_parquet(str(args.input_path / "**" / "*.parquet"))
        n_rows = df.height
        n_chunks = (n_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"Processing {n_rows:,} rows in {n_chunks} chunk(s)\u2026", file=sys.stderr)
        partials = []
        for i in range(n_chunks):
            start = i * CHUNK_SIZE
            chunk = df.slice(start, CHUNK_SIZE)
            print(f"  [{i+1}/{n_chunks}] rows {start:,}\u2013{start + chunk.height:,}\u2026", file=sys.stderr)
            partials.append(_process_chunk(chunk))
            del chunk
        del df
    else:
        print(f"Error: {args.input_path} not found", file=sys.stderr)
        sys.exit(1)

    print("  Combining\u2026", file=sys.stderr)
    agg = _combine_partials(partials)
    del partials

    sections, meta_counts = _build_sections(agg)

    report_data = _safe_json({
        "metadata": {
            "split_name": split_name,
            "total_sentences": meta_counts["total_sentences"],
            "total_tokens": meta_counts["total_tokens"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "sections": sections,
    })

    print("Building HTML\u2026", file=sys.stderr)
    html_content = _build_html(report_data)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_content, encoding="utf-8")
    size_kb = output.stat().st_size / 1024
    print(f"Report written to: {output} ({size_kb:.0f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
