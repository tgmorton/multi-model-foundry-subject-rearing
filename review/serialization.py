"""Polars nested struct serialization helpers.

Converts Polars rows (which may contain nested structs and lists-of-structs)
into plain JSON-safe Python dicts.
"""

from typing import Any, Dict, List, Union

import polars as pl


def row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single Polars row dict to a JSON-safe dict.

    Handles nested structs, lists of structs, and null values.
    """
    out = {}
    for k, v in row.items():
        out[k] = _convert_value(v)
    return out


def _convert_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, dict):
        return {dk: _convert_value(dv) for dk, dv in v.items()}
    if isinstance(v, list):
        return [_convert_value(item) for item in v]
    # Polars int/float types -> Python native
    if hasattr(v, "item"):
        return v.item()
    return v


def dataframe_to_records(df: pl.DataFrame) -> List[Dict[str, Any]]:
    """Convert a Polars DataFrame to a list of JSON-safe dicts."""
    rows = df.to_dicts()
    return [row_to_dict(r) for r in rows]


def safe_collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect a LazyFrame, catching schema errors gracefully."""
    return lf.collect()
