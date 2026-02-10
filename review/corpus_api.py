"""Wraps AnnotatedCorpus / load_layered_corpus() for paginated, filtered JSON.

Provides a high-level query API used by the FastAPI server.

Uses a two-tier loading strategy:
  1. A small in-memory index (sentence_id + genre + speaker + boolean flags)
     for fast filtering, counting, and pagination.
  2. Lazy scans of the base parquet and layer parquets for fetching text
     and annotation details on demand.

This keeps startup fast and memory proportional to the number of sentences
times ~20 scalar columns, rather than the full nested annotation data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from .serialization import dataframe_to_records, row_to_dict

ALL_FLAG_COLUMNS = [
    "has_null_subject",
    "has_null_subject_finite",
    "has_null_subject_nonfinite",
    "has_null_object",
    "has_overt_subject_pronoun",
    "has_overt_object_pronoun",
    "has_expletive",
    "has_that_trace_context",
    "has_that_trace_violation",
    "has_wh_extraction",
    "has_relative_clause",
    "has_negation",
    "has_ditransitive",
    "is_imperative",
    "is_fragment",
]

# Columns pulled into the in-memory index from the base parquet
_INDEX_BASE_COLS = ["sentence_id", "genre", "speaker", "role", "n_tokens",
                    "file_name", "sent_idx"]

# Columns needed for sentence list view (fetched per-page from base)
_LIST_EXTRA_COLS = ["text"]

# Nested annotation list columns (lists-of-structs)
_ANNOTATION_LIST_COLS = [
    "clauses", "bridge_complements", "expletives", "pronouns",
    "argument_structure", "negations", "wh_extractions",
    "relative_clauses", "verbs",
]

# Nested annotation scalar-struct columns
_ANNOTATION_SCALAR_COLS = ["topic_info", "complexity"]

# Base token-level array columns
_TOKEN_ARRAY_COLS = ["tokens", "lemmas", "pos_tags", "dep_rels", "dep_heads"]


def _build_index(corpus_path: Path) -> pl.DataFrame:
    """Build a small in-memory index: scalar columns + boolean flags.

    Reads only lightweight columns from base and flag-only columns from
    each layer. Since all parquet files share the same row order and count,
    flag columns are horizontally stacked (no join needed), which avoids
    the memory overhead of hashing millions of sentence_id strings.
    """
    base_files = sorted((corpus_path / "base").glob("*.parquet"))
    if not base_files:
        raise FileNotFoundError(f"No base parquet files in {corpus_path / 'base'}")

    # Read only scalar index columns from base (skip text, token arrays)
    base_schema = pl.scan_parquet(base_files).collect_schema()
    base_select = [c for c in _INDEX_BASE_COLS if c in base_schema.names()]
    df = pl.scan_parquet(base_files).select(base_select).collect()

    layers_dir = corpus_path / "layers"
    if not layers_dir.exists():
        return df

    all_flags = set(ALL_FLAG_COLUMNS)
    existing_cols = set(df.columns)

    for layer_dir in sorted(layers_dir.iterdir()):
        if not layer_dir.is_dir():
            continue
        layer_files = sorted(layer_dir.glob("*.parquet"))
        if not layer_files:
            continue

        layer_schema = pl.scan_parquet(layer_files).collect_schema()
        layer_cols = set(layer_schema.names())

        # Only pick up boolean flag columns we don't already have
        flag_cols = sorted((layer_cols & all_flags) - existing_cols)
        if not flag_cols:
            continue

        # Read just the flag columns (no sentence_id needed — same row order)
        flag_df = pl.scan_parquet(layer_files).select(flag_cols).collect()
        df = df.hstack(flag_df)
        existing_cols |= set(flag_cols)

    return df


class CorpusQueryAPI:
    """Query interface for the review server.

    Holds a small in-memory index for filtering/pagination and loads
    text and annotation details lazily from Parquet on demand.
    """

    def __init__(self, corpus_path: str):
        self._path = Path(corpus_path)
        self._base_files = sorted((self._path / "base").glob("*.parquet"))

        # Small in-memory index — sentence_id + genre + speaker + flags
        self._index = _build_index(self._path)
        self._all_columns = set(self._index.columns)
        self._flag_columns = [c for c in ALL_FLAG_COLUMNS if c in self._all_columns]

        # List-view columns present in the index
        self._index_list_cols = (
            [c for c in _INDEX_BASE_COLS if c in self._all_columns]
            + self._flag_columns
        )

        # Cache genres and speakers from index (instant, no I/O)
        self._genres = sorted(self._index["genre"].drop_nulls().unique().sort().to_list())
        self._speakers = sorted(self._index["speaker"].drop_nulls().unique().sort().to_list())

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "genres": self._genres,
            "speakers": self._speakers,
            "flag_columns": self._flag_columns,
        }

    def query_sentences(
        self,
        page: int = 1,
        page_size: int = 50,
        genre: Optional[str] = None,
        speaker: Optional[str] = None,
        flags: Optional[List[str]] = None,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Paginated, filtered sentence query.

        Filtering and counting happen on the in-memory index.
        Text is fetched from the base parquet only for the requested page.
        """
        idx = self._index.lazy()

        # Apply filters on index
        if genre:
            idx = idx.filter(pl.col("genre") == genre)
        if speaker:
            idx = idx.filter(pl.col("speaker") == speaker)
        if flags:
            for flag in flags:
                if flag in self._all_columns:
                    idx = idx.filter(pl.col(flag) == True)  # noqa: E712
        if exclude_ids:
            idx = idx.filter(~pl.col("sentence_id").is_in(exclude_ids))

        # For text search we need to join base — fall back to scanning
        if search:
            base_lf = pl.scan_parquet(self._base_files).select(["sentence_id", "text"])
            idx = idx.join(base_lf, on="sentence_id", how="inner")
            idx = idx.filter(pl.col("text").str.contains(search, literal=True))

        # Sort
        if sort_by and sort_by in self._all_columns:
            idx = idx.sort(sort_by, descending=True)

        # Count total matching
        total = idx.select(pl.len()).collect().item()

        # Get page of sentence_ids
        offset = (page - 1) * page_size
        page_ids_df = idx.select("sentence_id").slice(offset, page_size).collect()
        page_ids = page_ids_df["sentence_id"].to_list()

        if not page_ids:
            return {"total": total, "page": page, "page_size": page_size, "sentences": []}

        # Fetch text for this page from base parquet
        text_df = (
            pl.scan_parquet(self._base_files)
            .filter(pl.col("sentence_id").is_in(page_ids))
            .select(["sentence_id", "text"])
            .collect()
        )

        # Join text onto the index rows for this page
        page_index = self._index.filter(
            pl.col("sentence_id").is_in(page_ids)
        )
        page_df = page_index.join(text_df, on="sentence_id", how="left")

        # Preserve the sort order from the filtered index
        page_df = page_df.join(
            pl.DataFrame({"sentence_id": page_ids, "_order": range(len(page_ids))}),
            on="sentence_id",
        ).sort("_order").drop("_order")

        # Select columns for list view
        list_cols = [c for c in self._index_list_cols + ["text"] if c in page_df.columns]
        page_df = page_df.select(list_cols)

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "sentences": dataframe_to_records(page_df),
        }

    def get_sentence_texts(self, sentence_ids: List[str]) -> Dict[str, str]:
        """Return {sentence_id: text} for a list of IDs."""
        if not sentence_ids:
            return {}
        df = (
            pl.scan_parquet(self._base_files)
            .filter(pl.col("sentence_id").is_in(sentence_ids))
            .select(["sentence_id", "text"])
            .collect()
        )
        return {
            r["sentence_id"]: r["text"]
            for r in df.to_dicts()
        }

    def get_sentence_detail(self, sentence_id: str) -> Optional[Dict[str, Any]]:
        """Return full sentence with all annotation layers + context.

        Loads the base row and each layer on demand for this single sentence,
        so only one row per layer is ever materialized in memory.
        """
        # Get base row (full — including token arrays)
        base_df = (
            pl.scan_parquet(self._base_files)
            .filter(pl.col("sentence_id") == sentence_id)
            .collect()
        )
        if base_df.is_empty():
            return None

        row = base_df.to_dicts()[0]

        # Merge flag columns from the index
        idx_row = self._index.filter(
            pl.col("sentence_id") == sentence_id
        )
        if not idx_row.is_empty():
            for k, v in idx_row.to_dicts()[0].items():
                if k not in row:
                    row[k] = v

        # Load each layer's full data for this one sentence
        layers_dir = self._path / "layers"
        if layers_dir.exists():
            for layer_dir in sorted(layers_dir.iterdir()):
                if not layer_dir.is_dir():
                    continue
                layer_files = sorted(layer_dir.glob("*.parquet"))
                if not layer_files:
                    continue
                layer_df = (
                    pl.scan_parquet(layer_files)
                    .filter(pl.col("sentence_id") == sentence_id)
                    .collect()
                )
                if layer_df.is_empty():
                    continue
                layer_row = layer_df.to_dicts()[0]
                for k, v in layer_row.items():
                    if k != "sentence_id" and k not in row:
                        row[k] = v

        result = row_to_dict(row)
        result["gloss_rows"] = self._build_gloss_rows(result)
        result["context_before"], result["context_after"] = self._get_context(row)
        return result

    def _get_context(self, row: Dict[str, Any]) -> tuple:
        """Return text of the previous and next sentence in the same file."""
        file_name = row.get("file_name")
        sent_idx = row.get("sent_idx")
        if file_name is None or sent_idx is None:
            return None, None
        neighbors = (
            pl.scan_parquet(self._base_files)
            .filter(
                (pl.col("file_name") == file_name)
                & (pl.col("sent_idx").is_in([sent_idx - 1, sent_idx + 1]))
            )
            .select(["sent_idx", "text"])
            .collect()
        )
        before = after = None
        for r in neighbors.to_dicts():
            if r["sent_idx"] == sent_idx - 1:
                before = r["text"]
            elif r["sent_idx"] == sent_idx + 1:
                after = r["text"]
        return before, after

    def _build_gloss_rows(self, sentence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build token-aligned gloss rows from annotation data."""
        tokens = sentence.get("tokens") or []
        n = len(tokens)
        if n == 0:
            return []

        rows = []

        # Base rows
        rows.append({"label": "Token", "cells": tokens})
        if sentence.get("lemmas"):
            rows.append({"label": "Lemma", "cells": sentence["lemmas"]})
        if sentence.get("pos_tags"):
            rows.append({"label": "POS", "cells": sentence["pos_tags"]})
        if sentence.get("dep_rels"):
            rows.append({"label": "Dep Rel", "cells": sentence["dep_rels"]})
        if sentence.get("dep_heads"):
            rows.append({
                "label": "Dep Head",
                "cells": [str(h) if h is not None else "" for h in sentence["dep_heads"]],
            })

        # Annotation layer rows - placed at index positions
        rows.extend(self._layer_rows(sentence, "clauses", "verb_idx", [
            ("Clause Type", "clause_type"),
            ("Subject Status", "subject_status"),
            ("Is Finite", "is_finite"),
        ], n))

        rows.extend(self._layer_rows(sentence, "verbs", "token_idx", [
            ("Verb Form", "verb_form"),
            ("Tense", "tense"),
        ], n))

        rows.extend(self._layer_rows(sentence, "argument_structure", "verb_idx", [
            ("Transitivity", "transitivity"),
            ("Arg Subj Status", "subject_status"),
        ], n))

        rows.extend(self._layer_rows(sentence, "expletives", "token_idx", [
            ("Expletive Class", "expletive_class"),
        ], n))

        rows.extend(self._layer_rows(sentence, "pronouns", "token_idx", [
            ("Pronoun Function", "function"),
        ], n))

        rows.extend(self._layer_rows(sentence, "negations", "token_idx", [
            ("Negation", "lemma"),
        ], n))

        rows.extend(self._layer_rows(sentence, "wh_extractions", "wh_idx", [
            ("Wh Type", "extraction_type"),
        ], n))

        rows.extend(self._layer_rows(sentence, "relative_clauses", "verb_idx", [
            ("Rel Clause Type", "rel_type"),
        ], n))

        rows.extend(self._layer_rows(sentence, "bridge_complements", "matrix_verb_idx", [
            ("Bridge Verb", "comp_present"),
        ], n))

        return rows

    def _layer_rows(
        self,
        sentence: Dict[str, Any],
        layer_col: str,
        idx_field: str,
        field_defs: List[tuple],
        n_tokens: int,
    ) -> List[Dict[str, Any]]:
        """Build gloss rows from a list-of-structs annotation layer."""
        data = sentence.get(layer_col)
        if not data:
            return []

        rows = []
        for label, value_field in field_defs:
            cells = [""] * n_tokens
            has_data = False
            for item in data:
                idx = item.get(idx_field)
                val = item.get(value_field)
                if idx is not None and 0 <= idx < n_tokens and val is not None:
                    cells[idx] = str(val)
                    has_data = True
            if has_data:
                rows.append({"label": label, "cells": cells, "layer": layer_col})
        return rows
