#!/usr/bin/env python
"""
Generate a focused interactive HTML report on null subject use.

Produces a single self-contained HTML file with 5 sections:
  1. Overview — KPI cards (total sentences, finite clauses, overall null rate)
  2. Child vs Adult — clause-level null-subject rate comparison
  3. By Genre — clause-level null-subject rate per genre
  4. By Clause Context — null-subject rate by clause type (readable names)
  5. Developmental Trajectory — null-subject rate by child age band + MLU

Usage
-----
    PYTHONPATH=. python analysis/scripts/generate_null_subject_report.py \
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
from typing import Any, Dict, List, Optional, Union

import polars as pl


# ---------------------------------------------------------------------------
# Data loading (same as generate_interactive_report.py)
# ---------------------------------------------------------------------------

def _load_data(path: Path, layered: bool) -> pl.LazyFrame:
    """Load corpus data, selecting only columns needed by the report.

    For layered corpora this avoids joining all layers (which can OOM on
    large corpora) and instead joins only base + clause_structure with a
    minimal column set.
    """
    _BASE_COLS = ["sentence_id", "genre", "role", "child_age_months", "n_tokens"]

    if layered:
        base_dir = path / "base"
        clause_dir = path / "layers" / "clause_structure"
        if not base_dir.exists() or not clause_dir.exists():
            print(f"Error: expected base/ and layers/clause_structure/ in {path}", file=sys.stderr)
            sys.exit(1)

        base = pl.scan_parquet(str(base_dir / "*.parquet")).select(_BASE_COLS)
        clauses = pl.scan_parquet(str(clause_dir / "*.parquet")).select(["sentence_id", "clauses"])
        return base.join(clauses, on="sentence_id", how="inner")
    elif path.is_file():
        return pl.scan_parquet(path)
    elif path.is_dir():
        return pl.scan_parquet(str(path / "**" / "*.parquet"))
    else:
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> Any:
    """Make an object JSON-serialisable (handle None, float edge cases)."""
    if obj is None:
        return None
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        if obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    return obj


def _df_to_records(df: pl.DataFrame, max_rows: int = 200) -> List[Dict]:
    """Convert DataFrame to list-of-dicts, truncated and JSON-safe."""
    if df.is_empty():
        return []
    return _safe_json(df.head(max_rows).to_dicts())


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


_TABLEAU_10 = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def _color_palette(n: int) -> List[str]:
    """Return n colours, cycling through Tableau 10."""
    return [_TABLEAU_10[i % len(_TABLEAU_10)] for i in range(n)]


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


# ---------------------------------------------------------------------------
# NullSubjectReportCollector
# ---------------------------------------------------------------------------

class NullSubjectReportCollector:
    """Runs null-subject-focused analyses and builds JSON payload."""

    # Only these columns are needed for the 5 report sections.
    # Selecting upfront lets Polars push the projection down into the
    # Parquet reader, avoiding OOM on large corpora (e.g. 90M).
    _REQUIRED_COLS = [
        "clauses", "genre", "role", "child_age_months", "n_tokens",
    ]

    def __init__(self, lf: pl.LazyFrame, corpus_path: Path):
        available = set(lf.collect_schema().names())
        keep = [c for c in self._REQUIRED_COLS if c in available]
        self.lf = lf.select(keep)
        self.corpus_path = corpus_path
        self.sections: Dict[str, Any] = {}

    # --- Section 1: Overview ---

    def _collect_overview(self) -> None:
        lf = self.lf
        _null = ["none", "inherited", "imperative"]

        total_sents = lf.select(pl.len()).collect().item()

        # Explode clauses and filter to finite
        clauses = (
            lf
            .select("clauses")
            .explode("clauses")
            .filter(pl.col("clauses").is_not_null())
            .select([
                pl.col("clauses").struct.field("subject_status").alias("subject_status"),
                pl.col("clauses").struct.field("is_finite").alias("is_finite"),
            ])
            .filter(pl.col("is_finite") == True)
        )

        stats = clauses.select([
            pl.len().alias("total_clauses"),
            pl.col("subject_status").is_in(_null).mean().alias("null_rate"),
        ]).collect()

        total_clauses = stats["total_clauses"].item()
        overall_null_rate = stats["null_rate"].item()

        self.sections["overview"] = {
            "title": "1. Overview",
            "description": (
                "Headline statistics. Null subject = clause with no overt subject "
                "(includes imperatives, diary drop, and all other null-subject types). "
                "All rates are clause-level, finite clauses only."
            ),
            "kpis": [
                {"label": "Total Sentences", "value": _fmt_num(total_sents),
                 "sub": "across all genres"},
                {"label": "Finite Clauses", "value": _fmt_num(total_clauses),
                 "sub": "analysis denominator"},
                {"label": "Null-Subject Rate", "value": _fmt_pct(overall_null_rate),
                 "sub": "clause-level, finite only"},
            ],
            "table": None,
            "columns": [],
            "chart_data": None,
        }

    # --- Section 2: Child vs Adult ---

    def _collect_child_adult(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            child_adult_null_subject_comparison,
        )

        comp = child_adult_null_subject_comparison(self.lf)
        records = _df_to_records(comp)

        if not records:
            self.sections["child_adult"] = {
                "title": "2. Child vs Adult",
                "description": "No CHILDES data available.",
                "table": [], "columns": [], "chart_data": None,
            }
            return

        # Show only null_rate + n_clauses (drop root columns)
        display_records = [
            {"role": r["role"], "null_rate": r["null_rate"], "n_clauses": r["n_clauses"]}
            for r in records
        ]
        display_cols = ["role", "null_rate", "n_clauses"]

        roles = [r["role"] for r in records]
        rates = _safe_json([r.get("null_rate", 0) for r in records])

        self.sections["child_adult"] = {
            "title": "2. Child vs Adult",
            "description": (
                "Clause-level null-subject rate for CHILDES children vs adults "
                "(finite clauses only)."
            ),
            "table": display_records,
            "columns": display_cols,
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": roles,
                    "datasets": [{
                        "label": "Null-subject rate",
                        "data": rates,
                        "backgroundColor": ["#e15759", "#4e79a7"],
                    }],
                },
                "options": {
                    "plugins": {
                        "title": {"display": True, "text": "Null-Subject Rate: Child vs Adult"},
                        "legend": {"display": False},
                    },
                    "scales": {
                        "y": {"title": {"display": True, "text": "Null-subject rate"}, "beginAtZero": True, "max": 1},
                    },
                },
            },
        }

    # --- Section 3: By Genre ---

    def _collect_by_genre(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            clause_level_null_rate_by_genre,
        )

        genre_df = clause_level_null_rate_by_genre(self.lf)
        records = _df_to_records(genre_df)

        genres = [r["genre"] for r in records]
        rates = _safe_json([r.get("null_rate", 0) for r in records])
        palette = _color_palette(len(genres))

        self.sections["by_genre"] = {
            "title": "3. By Genre",
            "description": "Clause-level null-subject rate per genre (finite clauses only).",
            "table": records,
            "columns": genre_df.columns,
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": genres,
                    "datasets": [{
                        "label": "Null-subject rate",
                        "data": rates,
                        "backgroundColor": palette,
                    }],
                },
                "options": {
                    "indexAxis": "y",
                    "plugins": {
                        "title": {"display": True, "text": "Null-Subject Rate by Genre"},
                        "legend": {"display": False},
                    },
                    "scales": {
                        "x": {"title": {"display": True, "text": "Null-subject rate"}, "max": 1, "beginAtZero": True},
                    },
                },
            },
        }

    # --- Section 4: By Clause Context ---

    def _collect_by_context(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            null_subject_rate_by_clause_type,
        )

        ct = null_subject_rate_by_clause_type(self.lf, finite_only=True)
        records = _df_to_records(ct)

        # Apply readable labels
        for r in records:
            raw = r.get("clause_type", "")
            r["clause_type"] = _CLAUSE_TYPE_LABELS.get(raw, raw)

        labels = [r["clause_type"] for r in records]
        rates = _safe_json([r.get("null_rate", 0) for r in records])
        palette = _color_palette(len(labels))

        self.sections["by_context"] = {
            "title": "4. By Clause Context",
            "description": "Null-subject rate by clause type (finite clauses only).",
            "table": records,
            "columns": ct.columns,
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Null-subject rate",
                        "data": rates,
                        "backgroundColor": palette,
                    }],
                },
                "options": {
                    "indexAxis": "y",
                    "plugins": {
                        "title": {"display": True, "text": "Null-Subject Rate by Clause Context"},
                        "legend": {"display": False},
                    },
                    "scales": {
                        "x": {"title": {"display": True, "text": "Null-subject rate"}, "max": 1, "beginAtZero": True},
                    },
                },
            },
        }

    # --- Section 5: Developmental Trajectory ---

    def _collect_developmental(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            clause_level_developmental_trajectory,
        )

        dev = clause_level_developmental_trajectory(self.lf)
        dev_records = _df_to_records(dev)

        if not dev_records:
            self.sections["developmental"] = {
                "title": "5. Developmental Trajectory",
                "description": "No age-stratified CHILDES data available.",
                "table": [],
                "columns": [],
                "chart_data": None,
            }
            return

        age_bands = [r["age_band"] for r in dev_records]
        ns_rates = _safe_json([r.get("null_rate", 0) for r in dev_records])
        mlu_values = _safe_json([r.get("mlu", 0) for r in dev_records])

        self.sections["developmental"] = {
            "title": "5. Developmental Trajectory",
            "description": (
                "Clause-level null-subject rate and MLU by child age band. "
                "Shows how null subject use changes across development."
            ),
            "table": dev_records,
            "columns": dev.columns,
            "chart_data": None,
            "charts": [
                {
                    "id": "dev-null-rate",
                    "config": {
                        "type": "line",
                        "data": {
                            "labels": age_bands,
                            "datasets": [{
                                "label": "Null-subject rate",
                                "data": ns_rates,
                                "borderColor": "#e15759",
                                "backgroundColor": "#e1575933",
                                "fill": True,
                                "tension": 0.3,
                                "pointRadius": 5,
                                "borderWidth": 2,
                            }],
                        },
                        "options": {
                            "plugins": {"title": {"display": True, "text": "Null-Subject Rate by Age"}},
                            "scales": {
                                "x": {"title": {"display": True, "text": "Age Band"}},
                                "y": {"title": {"display": True, "text": "Null-subject rate"}, "max": 1, "beginAtZero": True},
                            },
                        },
                    },
                },
                {
                    "id": "dev-mlu",
                    "config": {
                        "type": "line",
                        "data": {
                            "labels": age_bands,
                            "datasets": [{
                                "label": "MLU (words)",
                                "data": mlu_values,
                                "borderColor": "#4e79a7",
                                "backgroundColor": "#4e79a733",
                                "fill": True,
                                "tension": 0.3,
                                "pointRadius": 5,
                                "borderWidth": 2,
                            }],
                        },
                        "options": {
                            "plugins": {"title": {"display": True, "text": "Mean Length of Utterance by Age"}},
                            "scales": {
                                "x": {"title": {"display": True, "text": "Age Band"}},
                                "y": {"title": {"display": True, "text": "MLU (words)"}, "beginAtZero": True},
                            },
                        },
                    },
                },
            ],
        }

    # --- Main entry point ---

    def collect_all(self, split_name: str = "corpus") -> Dict[str, Any]:
        """Run all analyses and return the complete JSON payload."""
        print("  [1/5] Overview\u2026", file=sys.stderr)
        self._collect_overview()
        print("  [2/5] Child vs adult\u2026", file=sys.stderr)
        self._collect_child_adult()
        print("  [3/5] By genre\u2026", file=sys.stderr)
        self._collect_by_genre()
        print("  [4/5] By clause context\u2026", file=sys.stderr)
        self._collect_by_context()
        print("  [5/5] Developmental trajectory\u2026", file=sys.stderr)
        self._collect_developmental()
        print("  Done.", file=sys.stderr)

        totals = self.lf.select([
            pl.len().alias("total_sentences"),
            pl.col("n_tokens").sum().alias("total_tokens"),
        ]).collect()

        return _safe_json({
            "metadata": {
                "split_name": split_name,
                "total_sentences": totals["total_sentences"].item(),
                "total_tokens": totals["total_tokens"].item(),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            "sections": self.sections,
        })


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def _build_table_html(rows: List[Dict], columns: List[str]) -> str:
    """Build a plain HTML table from list-of-dicts."""
    if not rows:
        return '<p><em>No data.</em></p>'

    lines = ['<div class="table-wrap"><table>']
    lines.append('<thead><tr>')
    for col in columns:
        display = col.replace("_", " ").title()
        lines.append(f'<th>{html.escape(display)}</th>')
    lines.append('</tr></thead>')

    lines.append('<tbody>')
    for row in rows:
        lines.append('<tr>')
        for col in columns:
            val = row.get(col)
            display_val = _format_cell(val)
            lines.append(f'<td>{html.escape(display_val)}</td>')
        lines.append('</tr>')
    lines.append('</tbody></table></div>')
    return "\n".join(lines)


def _format_cell(val: Any) -> str:
    if val is None:
        return "\u2014"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        if abs(val) < 1 and abs(val) > 0:
            return f"{val:.4f}"
        return f"{val:,.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    if isinstance(val, (list, dict)):
        return "\u2026"
    return str(val)


def _build_html(report_data: Dict[str, Any]) -> str:
    """Build a self-contained HTML file from the collected report data."""
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

        parts = [f'<section id="{sec_id}">']
        parts.append(f'<h2>{html.escape(title)}</h2>')
        if desc:
            parts.append(f'<p class="sec-desc">{html.escape(desc)}</p>')

        # KPI cards
        if kpis:
            kpi_html = '<div class="kpi-row">'
            for kpi in kpis:
                kpi_html += (
                    f'<div class="kpi-card">'
                    f'<div class="kpi-value">{html.escape(str(kpi["value"]))}</div>'
                    f'<div class="kpi-label">{html.escape(kpi["label"])}</div>'
                    f'<div class="kpi-sub">{html.escape(kpi.get("sub", ""))}</div>'
                    f'</div>'
                )
            kpi_html += '</div>'
            parts.append(kpi_html)

        # Chart canvas(es)
        charts = sec.get("charts", [])
        if chart_data:
            parts.append(f'<div class="chart-container"><canvas id="chart-{sec_id}"></canvas></div>')
        for chart in charts:
            cid = chart["id"]
            parts.append(f'<div class="chart-container"><canvas id="chart-{cid}"></canvas></div>')

        # Table
        if table_data and isinstance(table_data, list):
            if table_data and isinstance(table_data[0], dict):
                cols = columns if columns else list(table_data[0].keys())
                parts.append(_build_table_html(table_data, cols))

        parts.append('</section>')
        section_html_parts.append("\n".join(parts))

    sections_html = "\n\n".join(section_html_parts)
    nav_html = "\n".join(nav_links)

    # Build chart init JS
    chart_inits = []
    for sec_key, sec in sections.items():
        chart_data = sec.get("chart_data")
        if chart_data:
            sec_id = sec_key.replace("_", "-")
            chart_json = json.dumps(chart_data, default=str)
            chart_inits.append(
                f'createChart("chart-{sec_id}", {chart_json});'
            )
        for chart in sec.get("charts", []):
            cid = chart["id"]
            chart_json = json.dumps(chart["config"], default=str)
            chart_inits.append(
                f'createChart("chart-{cid}", {chart_json});'
            )
    chart_init_js = "\n    ".join(chart_inits)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Null Subject Analysis \u2014 {html.escape(meta['split_name'])}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
:root {{
    --bg: #fafafa;
    --surface: #ffffff;
    --text: #1a1a2e;
    --text-muted: #555;
    --border: #e0e0e0;
    --accent: #c2185b;
    --accent-light: #f8bbd0;
    --sidebar-w: 240px;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    display: flex;
    min-height: 100vh;
}}
nav#sidebar {{
    position: sticky;
    top: 0;
    width: var(--sidebar-w);
    min-width: var(--sidebar-w);
    height: 100vh;
    overflow-y: auto;
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 1rem 0.5rem;
    font-size: 0.82rem;
}}
nav#sidebar a {{
    display: block;
    padding: 4px 8px;
    color: var(--text-muted);
    text-decoration: none;
    border-radius: 4px;
    margin-bottom: 2px;
}}
nav#sidebar a:hover, nav#sidebar a.active {{
    background: #fce4ec;
    color: var(--accent);
}}
main {{
    flex: 1;
    max-width: 1100px;
    padding: 2rem 2.5rem;
}}
header {{ margin-bottom: 2rem; }}
header h1 {{ font-size: 1.6rem; margin-bottom: 0.3rem; color: var(--accent); }}
header .meta {{ color: var(--text-muted); font-size: 0.85rem; }}
section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}}
section h2 {{ font-size: 1.15rem; margin-bottom: 0.4rem; color: var(--accent); }}
.sec-desc {{ color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.8rem; }}
.chart-container {{
    position: relative;
    max-height: 400px;
    margin-bottom: 1rem;
}}

/* KPI cards */
.kpi-row {{
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}}
.kpi-card {{
    flex: 1;
    min-width: 140px;
    background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}}
.kpi-value {{
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.2;
}}
.kpi-label {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 0.2rem;
}}
.kpi-sub {{
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 0.5rem;
}}
th, td {{
    padding: 5px 8px;
    border-bottom: 1px solid var(--border);
    text-align: left;
    white-space: nowrap;
}}
th {{ background: #f4f6f8; font-weight: 600; position: sticky; top: 0; }}
tr:hover {{ background: #f9fbfd; }}
.table-wrap {{ max-height: 420px; overflow: auto; }}

@media (max-width: 900px) {{
    nav#sidebar {{ display: none; }}
    main {{ padding: 1rem; }}
    .kpi-row {{ flex-direction: column; }}
}}
</style>
</head>
<body>

<nav id="sidebar">
  <div style="font-weight:700;padding:0 8px 8px;font-size:0.9rem;color:var(--accent);">Null Subjects</div>
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
// \u2500\u2500 Embedded report data \u2500\u2500
const REPORT_DATA = {json_blob};

// \u2500\u2500 Chart rendering \u2500\u2500
function createChart(canvasId, config) {{
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    new Chart(canvas, {{
        type: config.type,
        data: config.data,
        options: Object.assign({{
            responsive: true,
            maintainAspectRatio: true,
            animation: {{ duration: 400 }},
            plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }} }},
        }}, config.options || {{}}),
    }});
}}

// \u2500\u2500 Init all charts \u2500\u2500
document.addEventListener('DOMContentLoaded', function() {{
    {chart_init_js}
}});

// \u2500\u2500 Sidebar active state \u2500\u2500
(function() {{
    var links = document.querySelectorAll('nav#sidebar a');
    var sections = document.querySelectorAll('main section');
    if (sections.length > 0 && typeof IntersectionObserver !== 'undefined') {{
        var observer = new IntersectionObserver(function(entries) {{
            entries.forEach(function(entry) {{
                if (entry.isIntersecting) {{
                    links.forEach(function(a) {{ a.classList.remove('active'); }});
                    var match = document.querySelector('nav#sidebar a[href="#' + entry.target.id + '"]');
                    if (match) match.classList.add('active');
                }}
            }});
        }}, {{ rootMargin: '-20% 0px -70% 0px' }});
        sections.forEach(function(s) {{ observer.observe(s); }});
    }}
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate focused interactive HTML report on null subject use.",
    )
    parser.add_argument(
        "input_path", type=Path,
        help="Path to Parquet file or annotated corpus directory",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output .html file",
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
            split_name = json.loads(meta_path.read_text()).get("split_name", "corpus")
        else:
            split_name = args.input_path.stem

    # Default output path
    output = args.output
    if output is None:
        output = Path("analysis/output/corpus_descriptives/reports/null_subject_report.html")

    print(f"Loading annotations from: {args.input_path}", file=sys.stderr)
    lf = _load_data(args.input_path, args.layered)
    print(f"Columns: {lf.collect_schema().names()}", file=sys.stderr)

    print("Running null-subject analyses\u2026", file=sys.stderr)
    collector = NullSubjectReportCollector(lf, args.input_path)
    report_data = collector.collect_all(split_name)

    print("Building HTML\u2026", file=sys.stderr)
    html_content = _build_html(report_data)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_content, encoding="utf-8")
    size_kb = output.stat().st_size / 1024
    print(f"Report written to: {output} ({size_kb:.0f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
