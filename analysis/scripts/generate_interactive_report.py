#!/usr/bin/env python
"""
Generate a self-contained interactive HTML report of corpus descriptive analyses.

Produces a single HTML file with:
- Chart.js charts for each analysis section
- Tables with hover tooltips showing ~20 example sentences from the corpus
- Sticky sidebar navigation

Usage
-----
    PYTHONPATH=. python analysis/scripts/generate_interactive_report.py \
        data/annotated_corpus_test_10M/ \
        --layered \
        --output analysis/corpus_descriptives_report/interactive_report.html
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
# Data loading (mirrors generate_corpus_report.py)
# ---------------------------------------------------------------------------

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
        return "—"
    return f"{val * 100:.1f}%"


def _fmt_num(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:,.2f}"
    return f"{val:,}"


# ---------------------------------------------------------------------------
# ReportDataCollector
# ---------------------------------------------------------------------------

class ReportDataCollector:
    """Runs all analyses, samples example sentences, and builds JSON payload."""

    N_EXAMPLES = 20
    SEED = 42
    MAX_SENT_LEN = 200

    def __init__(self, lf: pl.LazyFrame, corpus_path: Path):
        self.lf = lf
        self.corpus_path = corpus_path
        self.sections: Dict[str, Any] = {}
        self.examples: Dict[str, List[str]] = {}

    # --- sampling helpers ---

    def _sample(self, key: str, **conditions: Any) -> None:
        """Sample sentences matching top-level boolean / equality conditions."""
        try:
            query = self.lf
            for k, v in conditions.items():
                if v is not None:
                    query = query.filter(pl.col(k) == v)
            df = query.select("text").collect()
            if df.height > self.N_EXAMPLES:
                df = df.sample(n=self.N_EXAMPLES, seed=self.SEED)
            self.examples[key] = [
                t[:self.MAX_SENT_LEN] for t in df["text"].to_list()
            ]
        except Exception:
            self.examples[key] = []

    def _sample_nested(
        self,
        key: str,
        base_filter: Dict[str, Any],
        nested_col: str,
        nested_field: str,
        nested_val: Any,
    ) -> None:
        """Sample with nested struct filtering (pronouns, expletives, etc.)."""
        try:
            query = self.lf
            for k, v in base_filter.items():
                query = query.filter(pl.col(k) == v)
            query = query.filter(
                pl.col(nested_col).list.eval(
                    pl.element().struct.field(nested_field) == nested_val
                ).list.any()
            )
            df = query.select("text").collect()
            if df.height > self.N_EXAMPLES:
                df = df.sample(n=self.N_EXAMPLES, seed=self.SEED)
            self.examples[key] = [
                t[:self.MAX_SENT_LEN] for t in df["text"].to_list()
            ]
        except Exception:
            self.examples[key] = []

    # --- section collectors ---

    def _collect_overview(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import corpus_overview
        overview = corpus_overview(self.lf)
        records = _df_to_records(overview)

        genres = [r["genre"] for r in records]
        roles = _safe_sort({r["role"] for r in records})
        # Build stacked bar: tokens by genre, stacked by role
        datasets = []
        palette = _color_palette(len(roles))
        for i, role in enumerate(roles):
            data = []
            for genre in _safe_sort(set(genres)):
                total = sum(
                    r.get("total_tokens", 0) or 0
                    for r in records
                    if r["genre"] == genre and r["role"] == role
                )
                data.append(total)
            datasets.append({
                "label": role or "(no role)",
                "data": data,
                "backgroundColor": palette[i],
            })

        self.sections["corpus_overview"] = {
            "title": "1. Corpus Overview",
            "description": "Sentence and token counts by genre and speaker role.",
            "table": records,
            "columns": overview.columns,
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": _safe_sort(set(genres)),
                    "datasets": datasets,
                },
                "options": {
                    "indexAxis": "y",
                    "plugins": {"title": {"display": True, "text": "Tokens by Genre & Role"}},
                    "scales": {"x": {"stacked": True, "title": {"display": True, "text": "Tokens"}},
                               "y": {"stacked": True}},
                },
            },
            "summary": {
                "total_sentences": sum(r["n_sentences"] for r in records),
                "total_tokens": sum(r.get("total_tokens", 0) or 0 for r in records),
            },
        }

    def _collect_phenomenon_rates(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            phenomenon_rates_by_genre, ALL_FLAG_COLUMNS,
        )
        rates = phenomenon_rates_by_genre(self.lf)
        records = _df_to_records(rates)
        rate_cols = [c for c in rates.columns if c.endswith("_rate")]

        # Sample examples for each genre × phenomenon
        for row in records:
            genre = row["genre"]
            for rc in rate_cols:
                flag = rc.replace("_rate", "")
                key = f"phenom:{genre}:{flag}"
                self._sample(key, genre=genre, **{flag: True})

        # Build grouped bar: genre × top phenomena
        genres = _safe_sort(r["genre"] for r in records)
        top_flags = rate_cols[:6]
        palette = _color_palette(len(top_flags))
        datasets = []
        for i, rc in enumerate(top_flags):
            flag_label = rc.replace("has_", "").replace("_rate", "").replace("is_", "")
            data = [
                next((r[rc] for r in records if r["genre"] == g), 0)
                for g in genres
            ]
            datasets.append({
                "label": flag_label,
                "data": _safe_json(data),
                "backgroundColor": palette[i],
            })

        self.sections["phenomenon_rates"] = {
            "title": "2. Phenomenon Rates by Genre",
            "description": "Proportion of sentences exhibiting each boolean flag, per genre.",
            "table": records,
            "columns": rates.columns,
            "chart_data": {
                "type": "bar",
                "data": {"labels": genres, "datasets": datasets},
                "options": {
                    "plugins": {"title": {"display": True, "text": "Phenomenon Rates by Genre"}},
                    "scales": {"y": {"title": {"display": True, "text": "Rate"}}},
                },
            },
        }

    def _collect_cooccurrences(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            cooccurrence_null_subject_that_trace,
            cooccurrence_negation_null_subject,
            cooccurrence_expletive_depth,
        )

        # 3a: null subject × that-trace
        tt = cooccurrence_null_subject_that_trace(self.lf)
        self._sample("tt_baseline_null", has_null_subject=True)
        self._sample("tt_context", has_that_trace_context=True)
        self._sample(
            "tt_context_null",
            has_that_trace_context=True,
            has_null_subject=True,
        )

        tt_chart = {
            "type": "bar",
            "data": {
                "labels": ["Baseline", "In that-trace contexts"],
                "datasets": [{
                    "label": "Null subject rate",
                    "data": _safe_json([
                        tt.get("baseline_null_subject_rate"),
                        tt.get("null_rate_in_that_trace"),
                    ]),
                    "backgroundColor": ["#4e79a7", "#e15759"],
                }],
            },
            "options": {
                "indexAxis": "y",
                "plugins": {"title": {"display": True, "text": "§3a Null Subject × That-Trace"}},
                "scales": {"x": {"title": {"display": True, "text": "Rate"}}},
            },
        }
        self.sections["cooccurrence_null_subject_that_trace"] = {
            "title": "3a. Null Subject × That-Trace",
            "description": "Do null subjects co-occur with that-trace contexts?",
            "table": [tt],
            "columns": list(tt.keys()),
            "chart_data": tt_chart,
        }

        # 3b: negation × null subject
        neg = cooccurrence_negation_null_subject(self.lf)
        neg_records = _df_to_records(neg)
        for row in neg_records:
            genre = row["genre"]
            has_neg = row["has_negation"]
            key = f"neg_null:{genre}:{has_neg}"
            if genre == "ALL":
                self._sample(key, has_negation=has_neg, has_null_subject=True)
            else:
                self._sample(
                    key, genre=genre, has_negation=has_neg,
                    has_null_subject=True,
                )

        genres_neg = _safe_sort({r["genre"] for r in neg_records})
        neg_ds = []
        palette = ["#4e79a7", "#e15759"]
        for i, neg_val in enumerate([False, True]):
            data = [
                next(
                    (r["null_subject_rate"] for r in neg_records
                     if r["genre"] == g and r["has_negation"] == neg_val),
                    0,
                )
                for g in genres_neg
            ]
            neg_ds.append({
                "label": f"negation={neg_val}",
                "data": _safe_json(data),
                "backgroundColor": palette[i],
            })

        self.sections["cooccurrence_negation_null_subject"] = {
            "title": "3b. Negation × Null Subject",
            "description": "Does negation affect null-subject rate?",
            "table": neg_records,
            "columns": neg.columns,
            "chart_data": {
                "type": "bar",
                "data": {"labels": genres_neg, "datasets": neg_ds},
                "options": {
                    "plugins": {"title": {"display": True, "text": "§3b Negation × Null Subject Rate"}},
                    "scales": {"y": {"title": {"display": True, "text": "Null subject rate"}}},
                },
            },
        }

        # 3c: expletive × depth
        ed = cooccurrence_expletive_depth(self.lf)
        dist = ed.get("expletive_depth_distribution", [])
        self._sample("expl_depth", has_expletive=True)

        labels = [str(d["depth"]) for d in dist]
        counts = [d["count"] for d in dist]
        self.sections["cooccurrence_expletive_depth"] = {
            "title": "3c. Expletive × Embedding Depth",
            "description": "Are expletives simple surface constructions or deeply embedded?",
            "table": [ed],
            "columns": list(ed.keys()),
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Expletive sentences",
                        "data": counts,
                        "backgroundColor": "#59a14f",
                    }],
                },
                "options": {
                    "plugins": {"title": {"display": True, "text": "§3c Expletive Sentence Depth Distribution"}},
                    "scales": {
                        "x": {"title": {"display": True, "text": "Embedding depth"}},
                        "y": {"title": {"display": True, "text": "Count"}},
                    },
                },
            },
        }

    def _collect_childes(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            childes_child_vs_adult,
            childes_speaker_distribution,
            childes_person_distribution,
        )

        # 4a: child vs adult
        cva = childes_child_vs_adult(self.lf)
        cva_records = _df_to_records(cva)
        rate_cols = [c for c in cva.columns if c.endswith("_rate")]
        roles = [r["role"] for r in cva_records]
        palette = _color_palette(len(roles))

        for row in cva_records:
            role = row["role"]
            for rc in rate_cols:
                flag = rc.replace("_rate", "")
                key = f"childes_cva:{role}:{flag}"
                self._sample(key, role=role, **{flag: True})

        cva_ds = []
        for i, role in enumerate(roles):
            data = [
                next((r[rc] for r in cva_records if r["role"] == role), 0)
                for rc in rate_cols
            ]
            cva_ds.append({
                "label": role or "(unknown)",
                "data": _safe_json(data),
                "backgroundColor": palette[i],
            })

        clean_labels = [
            c.replace("has_", "").replace("_rate", "").replace("is_", "")
            for c in rate_cols
        ]
        self.sections["childes_child_vs_adult"] = {
            "title": "4a. CHILDES Child vs Adult",
            "description": "Core comparison: child (CHI) vs adult production.",
            "table": cva_records,
            "columns": cva.columns,
            "chart_data": {
                "type": "bar",
                "data": {"labels": clean_labels, "datasets": cva_ds},
                "options": {
                    "plugins": {"title": {"display": True, "text": "§4a Child vs Adult Phenomenon Rates"}},
                    "scales": {"y": {"title": {"display": True, "text": "Rate"}}},
                },
            },
        }

        # 4b: speaker distribution (table only, no chart needed beyond overview)
        spk = childes_speaker_distribution(self.lf)
        self.sections["childes_speaker_distribution"] = {
            "title": "4b. CHILDES Speaker Distribution",
            "description": "Distribution of speaker codes in CHILDES data.",
            "table": _df_to_records(spk),
            "columns": spk.columns,
            "chart_data": None,
        }

        # 4c: person distribution
        pd_df = childes_person_distribution(self.lf)
        pd_records = _df_to_records(pd_df)

        pd_roles = _safe_sort({r["role"] for r in pd_records})
        pd_persons = _safe_sort({r["person"] for r in pd_records})
        palette_p = _color_palette(len(pd_persons))

        pd_ds = []
        for i, person in enumerate(pd_persons):
            data = [
                next(
                    (r.get("proportion", 0) for r in pd_records
                     if r["role"] == role and r["person"] == person),
                    0,
                )
                for role in pd_roles
            ]
            pd_ds.append({
                "label": f"person={person}",
                "data": _safe_json(data),
                "backgroundColor": palette_p[i],
            })

        self.sections["childes_person_distribution"] = {
            "title": "4c. Person Distribution by Role",
            "description": "Pronoun person distribution (1st/2nd/3rd) by speaker role in CHILDES.",
            "table": pd_records,
            "columns": pd_df.columns if not pd_df.is_empty() else [],
            "chart_data": {
                "type": "bar",
                "data": {"labels": pd_roles, "datasets": pd_ds},
                "options": {
                    "plugins": {"title": {"display": True, "text": "§4c Person Distribution by Role"}},
                    "scales": {
                        "x": {"stacked": True},
                        "y": {"stacked": True, "max": 1, "title": {"display": True, "text": "Proportion"}},
                    },
                },
            },
        }

    def _collect_pronouns(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import subject_pronoun_distribution
        spd = subject_pronoun_distribution(self.lf)
        total_subj = (
            spd.group_by(["person", "number", "lemma"])
            .agg(pl.col("count").sum())
            .sort("count", descending=True)
        )
        records = _df_to_records(total_subj.head(20))

        # Sample for top lemmas
        for row in records[:15]:
            lemma = row["lemma"]
            key = f"subj_pron:{lemma}"
            self._sample_nested(key, {}, "pronouns", "lemma", lemma)

        labels = [r["lemma"] for r in records[:15]]
        counts = [r["count"] for r in records[:15]]

        self.sections["subject_pronouns"] = {
            "title": "5. Subject Pronoun Distribution",
            "description": "Subject pronouns — the targets for ablation.",
            "table": records,
            "columns": total_subj.columns,
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Count",
                        "data": counts,
                        "backgroundColor": "#4e79a7",
                    }],
                },
                "options": {
                    "indexAxis": "y",
                    "plugins": {"title": {"display": True, "text": "§5 Top Subject Pronouns"}},
                    "scales": {"x": {"title": {"display": True, "text": "Count"}}},
                },
            },
        }

    def _collect_that_trace(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import that_trace_inventory
        tti = that_trace_inventory(self.lf)

        bridge_verbs = tti.get("bridge_verb_counts", [])
        for bv in bridge_verbs[:15]:
            verb = bv["matrix_verb_lemma"]
            key = f"bridge:{verb}"
            self._sample_nested(
                key, {"has_that_trace_context": True},
                "bridge_complements", "matrix_verb_lemma", verb,
            )

        labels = [bv["matrix_verb_lemma"] for bv in bridge_verbs[:15]]
        counts = [bv["count"] for bv in bridge_verbs[:15]]

        self.sections["that_trace_inventory"] = {
            "title": "6. Bridge Verb / That-Trace Inventory",
            "description": "Bridge verb frequency, complementizer rate, extraction types.",
            "table": [tti],
            "columns": list(tti.keys()),
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Count",
                        "data": counts,
                        "backgroundColor": "#f28e2b",
                    }],
                },
                "options": {
                    "indexAxis": "y",
                    "plugins": {"title": {"display": True, "text": "§6 Bridge Verb Frequency (Top 15)"}},
                    "scales": {"x": {"title": {"display": True, "text": "Count"}}},
                },
            },
        }

    def _collect_expletives(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import expletive_by_class
        ebc = expletive_by_class(self.lf)
        ebc_records = _df_to_records(ebc)

        for row in ebc_records:
            genre = row["genre"]
            cls = row["expletive_class"]
            key = f"expl:{genre}:{cls}"
            self._sample_nested(
                key, {"genre": genre, "has_expletive": True},
                "expletives", "expletive_class", cls,
            )

        genres_e = _safe_sort({r["genre"] for r in ebc_records})
        classes = _safe_sort({r["expletive_class"] for r in ebc_records})
        palette = _color_palette(len(classes))

        ds = []
        for i, cls in enumerate(classes):
            data = [
                sum(r["count"] for r in ebc_records
                    if r["genre"] == g and r["expletive_class"] == cls)
                for g in genres_e
            ]
            ds.append({
                "label": cls or "(unknown)",
                "data": data,
                "backgroundColor": palette[i],
            })

        self.sections["expletive_by_class"] = {
            "title": "7. Expletive Distribution",
            "description": "Expletive counts by genre and class (weather / existential / raising).",
            "table": ebc_records,
            "columns": ebc.columns,
            "chart_data": {
                "type": "bar",
                "data": {"labels": genres_e, "datasets": ds},
                "options": {
                    "plugins": {"title": {"display": True, "text": "§7 Expletive Classes by Genre"}},
                    "scales": {
                        "x": {"stacked": True},
                        "y": {"stacked": True, "title": {"display": True, "text": "Count"}},
                    },
                },
            },
        }

    def _collect_complexity(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            complexity_distribution,
            depth_distribution,
        )

        cd = complexity_distribution(self.lf)
        self.sections["complexity_summary"] = {
            "title": "8a. Sentence Complexity Summary",
            "description": "Clause count, embedding depth, and coordination rate by genre.",
            "table": _df_to_records(cd),
            "columns": cd.columns,
            "chart_data": None,
        }

        dd = depth_distribution(self.lf)
        dd_records = _df_to_records(dd)

        genres_d = _safe_sort({r["genre"] for r in dd_records})
        all_depths = sorted({r["depth"] for r in dd_records if r["depth"] is not None})
        palette = _color_palette(len(genres_d))

        ds = []
        for i, genre in enumerate(genres_d):
            data = [
                next(
                    (r["count"] for r in dd_records
                     if r["genre"] == genre and r["depth"] == d),
                    0,
                )
                for d in all_depths
            ]
            ds.append({
                "label": genre,
                "data": data,
                "borderColor": palette[i],
                "backgroundColor": palette[i] + "33",
                "fill": False,
                "tension": 0.3,
            })

        self.sections["depth_distribution"] = {
            "title": "8b. Embedding Depth Distribution",
            "description": "Depth histogram by genre.",
            "table": dd_records,
            "columns": dd.columns,
            "chart_data": {
                "type": "line",
                "data": {
                    "labels": [str(d) for d in all_depths],
                    "datasets": ds,
                },
                "options": {
                    "plugins": {"title": {"display": True, "text": "§8 Embedding Depth Distribution"}},
                    "scales": {
                        "x": {"title": {"display": True, "text": "Max embedding depth"}},
                        "y": {"title": {"display": True, "text": "Sentence count"}},
                    },
                },
            },
        }

    def _collect_verbs(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            verb_null_subject_rate,
            bridge_verb_null_rates,
            weather_verb_null_rates,
        )

        vnr = verb_null_subject_rate(self.lf)
        vnr_records = _df_to_records(vnr.head(20))

        # Combine bridge + weather verb null rates for chart
        bridge = bridge_verb_null_rates(self.lf)
        weather = weather_verb_null_rates(self.lf)

        bridge_records = _df_to_records(bridge)
        weather_records = _df_to_records(weather)

        for row in bridge_records + weather_records:
            verb = row["verb_lemma"]
            key = f"verb_null:{verb}"
            self._sample_nested(
                key, {}, "clauses", "verb_lemma", verb,
            )

        combined = bridge_records + weather_records
        combined.sort(key=lambda r: r.get("null_rate", 0) or 0, reverse=True)
        labels = [r["verb_lemma"] for r in combined[:15]]
        rates = [r.get("null_rate", 0) for r in combined[:15]]

        self.sections["verb_null_rates"] = {
            "title": "9. Verb Null-Subject Rates",
            "description": "Bridge and weather verb null-subject rates.",
            "table": vnr_records,
            "columns": vnr.columns,
            "extra_tables": {
                "bridge_verbs": bridge_records,
                "weather_verbs": weather_records,
            },
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Null-subject rate",
                        "data": _safe_json(rates),
                        "backgroundColor": "#76b7b2",
                    }],
                },
                "options": {
                    "indexAxis": "y",
                    "plugins": {"title": {"display": True, "text": "§9 Bridge & Weather Verb Null Rates"}},
                    "scales": {"x": {"title": {"display": True, "text": "Null-subject rate"}, "max": 1}},
                },
            },
        }

    def _collect_register(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import register_comparison
        rc = register_comparison(self.lf)
        rc_records = _df_to_records(rc)

        rate_cols = [c for c in rc.columns if c.endswith("_rate")]
        registers = [r["register"] for r in rc_records]
        palette = _color_palette(len(registers))

        for row in rc_records:
            reg = row["register"]
            for rc_col in rate_cols:
                flag = rc_col.replace("_rate", "")
                key = f"register:{reg}:{flag}"
                # Register filtering not directly possible; skip sampling
                # (would need genre-based filter)

        ds = []
        for i, reg in enumerate(registers):
            data = [
                next((r[rc_col] for r in rc_records if r["register"] == reg), 0)
                for rc_col in rate_cols
            ]
            ds.append({
                "label": reg,
                "data": _safe_json(data),
                "backgroundColor": palette[i],
            })

        clean_labels = [
            c.replace("has_", "").replace("_rate", "").replace("is_", "")
            for c in rate_cols
        ]
        self.sections["register_comparison"] = {
            "title": "10. Spoken vs Written Register",
            "description": "Key phenomena by spoken vs written register.",
            "table": rc_records,
            "columns": rc.columns,
            "chart_data": {
                "type": "bar",
                "data": {"labels": clean_labels, "datasets": ds},
                "options": {
                    "plugins": {"title": {"display": True, "text": "§10 Register Comparison"}},
                    "scales": {"y": {"title": {"display": True, "text": "Rate"}}},
                },
            },
        }

    def _collect_hypotheses(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            null_subject_by_embedding_depth,
        )
        ns_depth = null_subject_by_embedding_depth(self.lf)
        ns_records = _df_to_records(ns_depth)

        for row in ns_records:
            depth = row["depth"]
            key = f"null_depth:{depth}"
            # Can't easily sample by depth without collecting, skip costly queries

        labels = [str(r["depth"]) for r in ns_records]
        rates = [r.get("null_subject_rate", 0) for r in ns_records]

        self.sections["null_subject_by_depth"] = {
            "title": "11. Null Subject Rate by Embedding Depth",
            "description": "Are null subjects more common in embedded clauses?",
            "table": ns_records,
            "columns": ns_depth.columns,
            "chart_data": {
                "type": "line",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Null subject rate",
                        "data": _safe_json(rates),
                        "borderColor": "#e15759",
                        "backgroundColor": "#e1575933",
                        "fill": True,
                        "tension": 0.3,
                    }],
                },
                "options": {
                    "plugins": {"title": {"display": True, "text": "§11 Null Subject Rate by Depth"}},
                    "scales": {
                        "x": {"title": {"display": True, "text": "Max embedding depth"}},
                        "y": {"title": {"display": True, "text": "Null subject rate"}, "max": 1},
                    },
                },
            },
        }

    def _collect_language_agnostic(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import language_agnostic_analysis
        la = language_agnostic_analysis(self.lf, "English")

        rows = []
        for k, v in la.items():
            if k in ("language", "subject_pronoun_person_distribution"):
                continue
            rows.append({"metric": k, "value": v})

        self.sections["language_agnostic"] = {
            "title": "12. Language-Agnostic Summary",
            "description": "Core rates for cross-linguistic comparison.",
            "table": rows,
            "columns": ["metric", "value"],
            "chart_data": None,
        }

    def _collect_clause_structure_crosstab(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import clause_structure_crosstab
        crosstab = clause_structure_crosstab(self.lf)
        if crosstab.is_empty():
            self.sections["clause_structure_crosstab"] = {
                "title": "13. Clause Structure Cross-tabulation",
                "description": "§1.3 – Cross-tab: clause_type × subject_status.",
                "table": [],
                "columns": [],
                "chart_data": None,
            }
            return

        records = _df_to_records(crosstab)

        # Build grouped bar chart by clause type
        clause_types = _safe_sort({r["clause_type"] for r in records})
        subject_statuses = _safe_sort({r["subject_status"] for r in records})
        palette = _color_palette(len(subject_statuses))

        datasets = []
        for i, status in enumerate(subject_statuses):
            data = [
                sum(r["count"] for r in records
                    if r["clause_type"] == ct and r["subject_status"] == status)
                for ct in clause_types
            ]
            datasets.append({
                "label": status or "(unknown)",
                "data": data,
                "backgroundColor": palette[i],
            })

        self.sections["clause_structure_crosstab"] = {
            "title": "13. Clause Structure Cross-tabulation",
            "description": "§1.3 – Cross-tab: clause_type × subject_status.",
            "table": records,
            "columns": crosstab.columns,
            "chart_data": {
                "type": "bar",
                "data": {"labels": clause_types, "datasets": datasets},
                "options": {
                    "plugins": {"title": {"display": True, "text": "§1.3 Clause Type × Subject Status"}},
                    "scales": {
                        "x": {"stacked": True},
                        "y": {"stacked": True, "title": {"display": True, "text": "Count"}},
                    },
                },
            },
        }

    def _collect_case_marked_pronouns(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import case_marked_pronoun_analysis
        result = case_marked_pronoun_analysis(self.lf)

        by_case = result.get("by_case", [])
        labels = [r["case"] or "(none)" for r in by_case]
        counts = [r["count"] for r in by_case]

        summary_rows = [
            {"metric": "Total pronouns", "value": result.get("total_pronouns", 0)},
            {"metric": "Case-marked count", "value": result.get("case_marked_count", 0)},
            {"metric": "Case-ambiguous count", "value": result.get("case_ambiguous_count", 0)},
            {"metric": "Neutralization coverage", "value": result.get("case_neutralization_coverage")},
        ]

        self.sections["case_marked_pronouns"] = {
            "title": "14. Case-Marked Pronoun Analysis",
            "description": "§2.3 / H6c – Frequency by pronoun × case for case neutralization.",
            "table": summary_rows,
            "columns": ["metric", "value"],
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Count",
                        "data": counts,
                        "backgroundColor": "#4e79a7",
                    }],
                },
                "options": {
                    "indexAxis": "y",
                    "plugins": {"title": {"display": True, "text": "§2.3 Pronouns by Case"}},
                    "scales": {"x": {"title": {"display": True, "text": "Count"}}},
                },
            },
        }

    def _collect_that_trace_crosstab(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import that_trace_crosstab
        crosstab = that_trace_crosstab(self.lf)
        if crosstab.is_empty():
            self.sections["that_trace_crosstab"] = {
                "title": "15. That-Trace Cross-tabulation",
                "description": "§4.4 – Complementizer × extraction_type cross-tab.",
                "table": [],
                "columns": [],
                "chart_data": None,
            }
            return

        records = _df_to_records(crosstab)

        # Build grouped bar chart
        comp_values = _safe_sort({r["comp_present"] for r in records})
        extraction_types = _safe_sort({r["extraction_type"] for r in records})
        palette = _color_palette(len(extraction_types))

        datasets = []
        for i, ext_type in enumerate(extraction_types):
            data = [
                sum(r["count"] for r in records
                    if r["comp_present"] == comp and r["extraction_type"] == ext_type)
                for comp in comp_values
            ]
            datasets.append({
                "label": ext_type or "(none)",
                "data": data,
                "backgroundColor": palette[i],
            })

        labels = ["+Comp" if c else "−Comp" for c in comp_values]

        self.sections["that_trace_crosstab"] = {
            "title": "15. That-Trace Cross-tabulation",
            "description": "§4.4 – Complementizer × extraction_type (+Comp/Subject should be ~0 in English).",
            "table": records,
            "columns": crosstab.columns,
            "chart_data": {
                "type": "bar",
                "data": {"labels": labels, "datasets": datasets},
                "options": {
                    "plugins": {"title": {"display": True, "text": "§4.4 Comp × Extraction Type"}},
                    "scales": {"y": {"title": {"display": True, "text": "Count"}}},
                },
            },
        }

    def _collect_verbal_morphology(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            verbal_morphology_paradigm,
            person_number_tense_distribution,
        )

        paradigm = verbal_morphology_paradigm(self.lf)
        pnt = person_number_tense_distribution(self.lf)

        paradigm_cells = paradigm.get("paradigm_cells", [])

        summary_rows = [
            {"metric": "Total finite verbs", "value": paradigm.get("total_finite_verbs", 0)},
            {"metric": "Distinct paradigm forms", "value": paradigm.get("distinct_forms_per_paradigm")},
            {"metric": "Syncretism rate", "value": paradigm.get("syncretism_rate")},
        ]

        # Build chart from person × number distribution
        persons = _safe_sort({r["person"] for r in paradigm_cells})
        numbers = _safe_sort({r["number"] for r in paradigm_cells})
        palette = _color_palette(len(numbers))

        datasets = []
        for i, num in enumerate(numbers):
            data = [
                next((r["count"] for r in paradigm_cells
                      if r["person"] == p and r["number"] == num), 0)
                for p in persons
            ]
            datasets.append({
                "label": num or "(unknown)",
                "data": data,
                "backgroundColor": palette[i],
            })

        self.sections["verbal_morphology"] = {
            "title": "16. Verbal Morphology Paradigm",
            "description": "§6 – Person × Number distribution of finite verbs.",
            "table": summary_rows + paradigm_cells,
            "columns": ["metric", "value"] if not paradigm_cells else ["person", "number", "count"],
            "chart_data": {
                "type": "bar",
                "data": {"labels": [f"Person {p}" for p in persons], "datasets": datasets},
                "options": {
                    "plugins": {"title": {"display": True, "text": "§6 Person × Number Distribution"}},
                    "scales": {"y": {"title": {"display": True, "text": "Count"}}},
                },
            },
        }

    def _collect_manipulation_feasibility(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import (
            manipulation_coverage,
            manipulation_overlap_matrix,
        )

        coverage = manipulation_coverage(self.lf)
        overlap = manipulation_overlap_matrix(self.lf)

        # Build summary table from coverage
        manip_data = coverage.get("manipulations", {})
        rows = []
        for name, data in manip_data.items():
            rows.append({
                "manipulation": name,
                "sentences_affected": data.get("sentences_affected", 0),
                "sentence_rate": data.get("sentence_rate"),
                "token_rate": data.get("token_rate"),
            })

        # Chart: sentence rates
        labels = [r["manipulation"] for r in rows]
        rates = [r["sentence_rate"] or 0 for r in rows]

        self.sections["manipulation_feasibility"] = {
            "title": "17. Manipulation Feasibility",
            "description": "§7 – Proportion of corpus affected by each manipulation.",
            "table": rows,
            "columns": ["manipulation", "sentences_affected", "sentence_rate", "token_rate"],
            "extra_tables": {"overlap_matrix": _df_to_records(overlap)},
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Sentence Rate",
                        "data": _safe_json(rates),
                        "backgroundColor": "#59a14f",
                    }],
                },
                "options": {
                    "indexAxis": "y",
                    "plugins": {"title": {"display": True, "text": "§7 Manipulation Coverage (Sentence Rate)"}},
                    "scales": {"x": {"title": {"display": True, "text": "Rate"}, "max": 1}},
                },
            },
        }

    def _collect_evaluation_context_freq(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import evaluation_context_frequency
        result = evaluation_context_frequency(self.lf)

        categories = result.get("categories", {})
        rows = []
        for name, data in categories.items():
            rows.append({
                "category": name,
                "count": data.get("count", 0),
                "per_million": data.get("per_million"),
                "is_rare": data.get("is_rare", False),
            })

        # Sort by count descending
        rows.sort(key=lambda r: r["count"], reverse=True)

        labels = [r["category"] for r in rows[:10]]
        counts = [r["count"] for r in rows[:10]]

        self.sections["evaluation_context_freq"] = {
            "title": "18. Evaluation Context Frequency",
            "description": "§9 – Frequency of evaluation stimulus proxies. Rare categories (<100 per 90M) flagged.",
            "table": rows,
            "columns": ["category", "count", "per_million", "is_rare"],
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Count",
                        "data": counts,
                        "backgroundColor": "#76b7b2",
                    }],
                },
                "options": {
                    "indexAxis": "y",
                    "plugins": {"title": {"display": True, "text": "§9 Evaluation Context Frequency (Top 10)"}},
                    "scales": {"x": {"title": {"display": True, "text": "Count"}}},
                },
            },
        }

    def _collect_wh_questions(self) -> None:
        from analysis.corpus_descriptives.corpus_analysis import wh_question_distribution
        result = wh_question_distribution(self.lf)

        by_extraction = result.get("by_extraction_type", [])
        by_lemma = result.get("by_wh_lemma", [])

        labels = [r["extraction_type"] or "(none)" for r in by_extraction]
        counts = [r["count"] for r in by_extraction]

        self.sections["wh_questions"] = {
            "title": "19. Wh-Question Distribution",
            "description": "§10 – Subject vs object vs embedded wh-question counts.",
            "table": by_extraction + by_lemma[:10],
            "columns": ["extraction_type", "count", "proportion"],
            "chart_data": {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Count",
                        "data": counts,
                        "backgroundColor": "#e15759",
                    }],
                },
                "options": {
                    "plugins": {"title": {"display": True, "text": "§10 Wh-Question Extraction Types"}},
                    "scales": {"y": {"title": {"display": True, "text": "Count"}}},
                },
            },
            "summary": {
                "total_wh_questions": result.get("total_wh_questions", 0),
            },
        }

    # --- main entry point ---

    def collect_all(self, split_name: str = "corpus") -> Dict[str, Any]:
        """Run all analyses and return the complete JSON payload."""
        print("  [1/19] Corpus overview…", file=sys.stderr)
        self._collect_overview()
        print("  [2/19] Phenomenon rates…", file=sys.stderr)
        self._collect_phenomenon_rates()
        print("  [3/19] Co-occurrence analyses…", file=sys.stderr)
        self._collect_cooccurrences()
        print("  [4/19] CHILDES analyses…", file=sys.stderr)
        self._collect_childes()
        print("  [5/19] Pronoun distribution…", file=sys.stderr)
        self._collect_pronouns()
        print("  [6/19] That-trace inventory…", file=sys.stderr)
        self._collect_that_trace()
        print("  [7/19] Expletive inventory…", file=sys.stderr)
        self._collect_expletives()
        print("  [8/19] Complexity distribution…", file=sys.stderr)
        self._collect_complexity()
        print("  [9/19] Verb analysis…", file=sys.stderr)
        self._collect_verbs()
        print("  [10/19] Register comparison…", file=sys.stderr)
        self._collect_register()
        print("  [11/19] Hypotheses & language-agnostic…", file=sys.stderr)
        self._collect_hypotheses()
        self._collect_language_agnostic()
        print("  [12/19] Clause structure cross-tab…", file=sys.stderr)
        self._collect_clause_structure_crosstab()
        print("  [13/19] Case-marked pronouns…", file=sys.stderr)
        self._collect_case_marked_pronouns()
        print("  [14/19] That-trace cross-tab…", file=sys.stderr)
        self._collect_that_trace_crosstab()
        print("  [15/19] Verbal morphology…", file=sys.stderr)
        try:
            self._collect_verbal_morphology()
        except Exception as e:
            print(f"    SKIPPED (missing person/number in verbs): {e}", file=sys.stderr)
        print("  [16/19] Manipulation feasibility…", file=sys.stderr)
        self._collect_manipulation_feasibility()
        print("  [17/19] Evaluation context frequency…", file=sys.stderr)
        self._collect_evaluation_context_freq()
        print("  [18/19] Wh-questions…", file=sys.stderr)
        self._collect_wh_questions()
        print("  [19/19] Done.", file=sys.stderr)

        overview_summary = self.sections.get("corpus_overview", {}).get("summary", {})
        return _safe_json({
            "metadata": {
                "split_name": split_name,
                "total_sentences": overview_summary.get("total_sentences", 0),
                "total_tokens": overview_summary.get("total_tokens", 0),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            "sections": self.sections,
            "examples": self.examples,
        })


# ---------------------------------------------------------------------------
# Color palette helper
# ---------------------------------------------------------------------------

_TABLEAU_10 = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def _safe_sort(iterable):
    """Sort an iterable that may contain None values."""
    return sorted(iterable, key=lambda x: (x is None, x if x is not None else ""))


def _color_palette(n: int) -> List[str]:
    """Return n colours, cycling through Tableau 10."""
    return [_TABLEAU_10[i % len(_TABLEAU_10)] for i in range(n)]


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _build_html(report_data: Dict[str, Any]) -> str:
    """Build a self-contained HTML file from the collected report data."""
    meta = report_data["metadata"]
    sections = report_data["sections"]
    json_blob = json.dumps(report_data, indent=None, default=str)

    # Build section HTML
    section_html_parts = []
    nav_links = []

    for sec_key, sec in sections.items():
        sec_id = sec_key.replace("_", "-")
        title = sec.get("title", sec_key)
        desc = sec.get("description", "")
        table_data = sec.get("table", [])
        columns = sec.get("columns", [])
        chart_data = sec.get("chart_data")

        nav_links.append(f'<a href="#{sec_id}">{html.escape(title)}</a>')

        parts = [f'<section id="{sec_id}">']
        parts.append(f'<h2>{html.escape(title)}</h2>')
        if desc:
            parts.append(f'<p class="sec-desc">{html.escape(desc)}</p>')

        # Chart canvas
        if chart_data:
            parts.append(f'<div class="chart-container"><canvas id="chart-{sec_id}"></canvas></div>')

        # Table
        if table_data and isinstance(table_data, list):
            if table_data and isinstance(table_data[0], dict):
                cols = columns if columns else list(table_data[0].keys())
                parts.append(_build_table_html(table_data, cols, sec_key))

            # Extra tables (e.g. bridge_verbs, weather_verbs)
            extra = sec.get("extra_tables", {})
            for extra_key, extra_rows in extra.items():
                if extra_rows and isinstance(extra_rows[0], dict):
                    extra_cols = list(extra_rows[0].keys())
                    parts.append(f'<h3>{html.escape(extra_key.replace("_", " ").title())}</h3>')
                    parts.append(_build_table_html(extra_rows, extra_cols, f"{sec_key}_{extra_key}"))

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
    chart_init_js = "\n    ".join(chart_inits)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Corpus Descriptive Analysis — {html.escape(meta['split_name'])}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
:root {{
    --bg: #fafafa;
    --surface: #ffffff;
    --text: #1a1a2e;
    --text-muted: #555;
    --border: #e0e0e0;
    --accent: #4e79a7;
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
    background: #e8f0fe;
    color: var(--accent);
}}
main {{
    flex: 1;
    max-width: 1100px;
    padding: 2rem 2.5rem;
}}
header {{ margin-bottom: 2rem; }}
header h1 {{ font-size: 1.6rem; margin-bottom: 0.3rem; }}
header .meta {{ color: var(--text-muted); font-size: 0.85rem; }}
section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}}
section h2 {{ font-size: 1.15rem; margin-bottom: 0.4rem; color: var(--accent); }}
section h3 {{ font-size: 0.95rem; margin: 1rem 0 0.3rem; }}
.sec-desc {{ color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.8rem; }}
.chart-container {{
    position: relative;
    max-height: 400px;
    margin-bottom: 1rem;
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
td[data-example-key] {{ cursor: help; border-bottom: 1px dashed var(--accent); }}
.table-wrap {{ max-height: 420px; overflow: auto; }}

/* Tooltip */
#tooltip {{
    display: none;
    position: fixed;
    z-index: 9999;
    max-width: 520px;
    max-height: 420px;
    overflow-y: auto;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0,0,0,.15);
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    line-height: 1.45;
    pointer-events: auto;
}}
#tooltip.visible {{ display: block; }}
#tooltip .tt-header {{
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
}}
#tooltip .example {{
    padding: 2px 0;
    color: var(--text);
}}
#tooltip .example .num {{ color: var(--text-muted); margin-right: 4px; }}

@media (max-width: 900px) {{
    nav#sidebar {{ display: none; }}
    main {{ padding: 1rem; }}
}}
</style>
</head>
<body>

<nav id="sidebar">
  <div style="font-weight:700;padding:0 8px 8px;font-size:0.9rem;">Sections</div>
  {nav_html}
</nav>

<main>
<header>
  <h1>Corpus Descriptive Analysis — {html.escape(meta['split_name'])}</h1>
  <div class="meta">
    Total sentences: {_fmt_num(meta.get('total_sentences'))} &middot;
    Total tokens: {_fmt_num(meta.get('total_tokens'))} &middot;
    Generated: {html.escape(meta.get('generated_at', ''))}
  </div>
</header>

{sections_html}

</main>

<div id="tooltip">
  <div class="tt-header"></div>
  <div class="tt-content"></div>
</div>

<script>
// ── Embedded report data ──
const REPORT_DATA = {json_blob};

// ── Chart rendering ──
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

// ── Init all charts ──
document.addEventListener('DOMContentLoaded', function() {{
    {chart_init_js}
}});

// ── Tooltip system ──
(function() {{
    const tooltip = document.getElementById('tooltip');
    const ttHeader = tooltip.querySelector('.tt-header');
    const ttContent = tooltip.querySelector('.tt-content');
    let hideTimer = null;
    let currentKey = null;

    function showTooltip(td, key) {{
        if (currentKey === key && tooltip.classList.contains('visible')) return;
        currentKey = key;

        const examples = REPORT_DATA.examples[key];
        if (!examples || examples.length === 0) {{
            ttHeader.textContent = key;
            ttContent.innerHTML = '<div class="example" style="color:#999">No examples available.</div>';
        }} else {{
            ttHeader.textContent = key + ' (' + examples.length + ' examples)';
            ttContent.innerHTML = examples.map(function(s, i) {{
                return '<div class="example"><span class="num">' + (i + 1) + '.</span>' + escapeHtml(s) + '</div>';
            }}).join('');
        }}

        // Position near the cell
        const rect = td.getBoundingClientRect();
        let left = rect.right + 8;
        let top = rect.top;
        // Keep on screen
        if (left + 520 > window.innerWidth) left = rect.left - 528;
        if (left < 0) left = 8;
        if (top + 420 > window.innerHeight) top = window.innerHeight - 428;
        if (top < 0) top = 8;

        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
        tooltip.classList.add('visible');
    }}

    function hideTooltip() {{
        hideTimer = setTimeout(function() {{
            tooltip.classList.remove('visible');
            currentKey = null;
        }}, 200);
    }}

    function cancelHide() {{
        if (hideTimer) {{ clearTimeout(hideTimer); hideTimer = null; }}
    }}

    function escapeHtml(s) {{
        var d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }}

    // Delegate events on all td[data-example-key]
    document.addEventListener('mouseover', function(e) {{
        var td = e.target.closest('td[data-example-key]');
        if (td) {{
            cancelHide();
            showTooltip(td, td.getAttribute('data-example-key'));
        }}
    }});
    document.addEventListener('mouseout', function(e) {{
        var td = e.target.closest('td[data-example-key]');
        if (td) hideTooltip();
    }});
    tooltip.addEventListener('mouseover', cancelHide);
    tooltip.addEventListener('mouseout', hideTooltip);

    // Sidebar active state
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


def _build_table_html(
    rows: List[Dict],
    columns: List[str],
    section_key: str,
) -> str:
    """Build an HTML table from list-of-dicts, adding data-example-key where applicable."""
    if not rows:
        return '<p><em>No data.</em></p>'

    # Determine which columns are rate/count columns that should get example keys
    rate_count_cols = set()
    for col in columns:
        if col.endswith("_rate") or col == "count" or col == "null_rate":
            rate_count_cols.add(col)

    lines = ['<div class="table-wrap"><table>']
    # Header
    lines.append('<thead><tr>')
    for col in columns:
        display = col.replace("_", " ").title()
        lines.append(f'<th>{html.escape(display)}</th>')
    lines.append('</tr></thead>')

    # Body
    lines.append('<tbody>')
    for row in rows:
        lines.append('<tr>')
        for col in columns:
            val = row.get(col)
            display_val = _format_cell(val)

            # Build example key for hoverable cells
            example_key = _make_example_key(section_key, row, col, columns)
            if example_key and col in rate_count_cols:
                lines.append(
                    f'<td data-example-key="{html.escape(example_key)}">{html.escape(display_val)}</td>'
                )
            else:
                lines.append(f'<td>{html.escape(display_val)}</td>')
        lines.append('</tr>')
    lines.append('</tbody></table></div>')
    return "\n".join(lines)


def _format_cell(val: Any) -> str:
    """Format a cell value for display."""
    if val is None:
        return "—"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        if abs(val) < 1 and abs(val) > 0:
            return f"{val:.4f}"
        return f"{val:,.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    if isinstance(val, (list, dict)):
        return "…"
    return str(val)


def _make_example_key(
    section_key: str,
    row: Dict,
    col: str,
    columns: List[str],
) -> Optional[str]:
    """Derive an example key for a table cell based on section and row context."""
    # Phenomenon rates: phenom:{genre}:{flag}
    if section_key == "phenomenon_rates" and col.endswith("_rate"):
        genre = row.get("genre", "")
        flag = col.replace("_rate", "")
        return f"phenom:{genre}:{flag}"

    # Negation co-occurrence: neg_null:{genre}:{negation}
    if section_key == "cooccurrence_negation_null_subject":
        genre = row.get("genre", "")
        neg = row.get("has_negation", "")
        return f"neg_null:{genre}:{neg}"

    # CHILDES child vs adult: childes_cva:{role}:{flag}
    if section_key == "childes_child_vs_adult" and col.endswith("_rate"):
        role = row.get("role", "")
        flag = col.replace("_rate", "")
        return f"childes_cva:{role}:{flag}"

    # Subject pronouns: subj_pron:{lemma}
    if section_key == "subject_pronouns":
        lemma = row.get("lemma", "")
        return f"subj_pron:{lemma}"

    # Expletive by class: expl:{genre}:{class}
    if section_key == "expletive_by_class":
        genre = row.get("genre", "")
        cls = row.get("expletive_class", "")
        return f"expl:{genre}:{cls}"

    # Verb null rates: verb_null:{verb}
    if section_key in ("verb_null_rates", "verb_null_rates_bridge_verbs", "verb_null_rates_weather_verbs"):
        verb = row.get("verb_lemma", "")
        return f"verb_null:{verb}"

    # Bridge verbs in that-trace: bridge:{verb}
    if section_key == "that_trace_inventory":
        verb = row.get("matrix_verb_lemma", "")
        if verb:
            return f"bridge:{verb}"

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML report of corpus descriptive analyses.",
    )
    parser.add_argument(
        "input_path", type=Path,
        help="Path to Parquet file or annotated corpus directory",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output .html file (default: corpus_descriptives_report/interactive_report.html)",
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
        output = Path("analysis/corpus_descriptives_report/interactive_report.html")

    print(f"Loading annotations from: {args.input_path}", file=sys.stderr)
    lf = _load_data(args.input_path, args.layered)
    print(f"Columns: {lf.collect_schema().names()}", file=sys.stderr)

    print("Running analyses and collecting examples…", file=sys.stderr)
    collector = ReportDataCollector(lf, args.input_path)
    report_data = collector.collect_all(split_name)

    print("Building HTML…", file=sys.stderr)
    html_content = _build_html(report_data)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_content, encoding="utf-8")
    size_kb = output.stat().st_size / 1024
    print(f"Report written to: {output} ({size_kb:.0f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
