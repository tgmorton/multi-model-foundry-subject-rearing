"""D10: Partitioned parquet output + DuckDB view layer.

Writes per-item and per-pair results from `PerModelRunner` (D9) as
partitioned parquet under a root directory. Partitions by
`(language, category, condition)` so DuckDB queries can filter at the
file level.

Output layout:

    output_root/
    ├── items/
    │   ├── language=en/category=subject_drop/condition=subj_3sg/
    │   │   └── cell_id={cid}.parquet
    │   └── ...
    ├── pairs/
    │   ├── language=en/category=subject_drop/condition=subj_3sg/
    │   │   └── cell_id={cid}.parquet
    │   └── ...
    └── per_token/                       # D5 per-token log-probs
        └── language=.../category=.../condition=.../cell_id={cid}.parquet

`register_duckdb_views(con, root)` builds SQL views over the tree so
hypothesis-test queries work without loading the whole dataset into
memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from evaluation.runners.per_model_runner import (
    CheckpointItemResult,
    CheckpointPairResult,
)

logger = logging.getLogger(__name__)


# --- Schema documentation (also enforced by the writer) ---------------------

_ITEM_COLUMNS = [
    "cell_id", "architecture", "intervention", "rep",
    "checkpoint_step", "checkpoint_path",
    "language", "category", "condition",
    "item_id", "pronoun_status",
    "target_sum_log_prob", "target_mean_log_prob", "target_n_tokens",
    "target_unigram_sum_log_prob", "slor", "hotspot_log_prob",
]

_PER_TOKEN_COLUMNS = _ITEM_COLUMNS[:-3] + [
    "pronoun_status",
    "per_token_log_prob", "per_token_ids",
]

_PAIR_COLUMNS = [
    "cell_id", "architecture", "intervention", "rep",
    "checkpoint_step", "language", "category", "condition", "item_id",
    "overt_mean_log_prob", "overt_slor", "overt_hotspot_log_prob",
    "null_mean_log_prob", "null_slor", "null_hotspot_log_prob",
    "prefers_overt_meanlp", "prefers_overt_slor",
    "log_prob_diff_overt_minus_null",
    "slor_diff_overt_minus_null",
    "hotspot_log_prob_diff_overt_minus_null",
]


# --- Writer -----------------------------------------------------------------

def _items_df(rows: Sequence[CheckpointItemResult]) -> pd.DataFrame:
    """DataFrame with aggregate metrics (no per-token lists)."""
    records = [{c: getattr(r, c) for c in _ITEM_COLUMNS} for r in rows]
    return pd.DataFrame.from_records(records)


def _per_token_df(rows: Sequence[CheckpointItemResult]) -> pd.DataFrame:
    """DataFrame with per-token log-prob + token id vectors (D5 payload)."""
    base_cols = [
        "cell_id", "architecture", "intervention", "rep",
        "checkpoint_step", "checkpoint_path",
        "language", "category", "condition",
        "item_id", "pronoun_status",
    ]
    records = []
    for r in rows:
        rec = {c: getattr(r, c) for c in base_cols}
        rec["per_token_log_prob"] = list(r.per_token_log_prob)
        rec["per_token_ids"] = list(r.per_token_ids)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _pairs_df(rows: Sequence[CheckpointPairResult]) -> pd.DataFrame:
    records = [{c: getattr(r, c) for c in _PAIR_COLUMNS} for r in rows]
    return pd.DataFrame.from_records(records)


def _write_partitioned(
    df: pd.DataFrame,
    root: Path,
    cell_id: str,
) -> List[Path]:
    """Write one parquet file per (language, category, condition) partition.

    Returns the list of paths written.
    """
    if df.empty:
        return []
    root.mkdir(parents=True, exist_ok=True)
    partition_cols = ["language", "category", "condition"]
    written: List[Path] = []
    for (lang, cat, cond), group in df.groupby(partition_cols):
        partition_dir = (
            root
            / f"language={lang}"
            / f"category={cat}"
            / f"condition={cond}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        # One file per cell_id so parallel writes don't collide.
        out_path = partition_dir / f"cell_id={cell_id}.parquet"
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        group.drop(columns=partition_cols).to_parquet(tmp, index=False)
        tmp.replace(out_path)
        written.append(out_path)
    return written


def write_cell_results(
    output_root: Path,
    cell_id: str,
    item_results: Sequence[CheckpointItemResult],
    pair_results: Sequence[CheckpointPairResult],
    include_per_token: bool = True,
) -> dict:
    """Write both per-item (aggregated) and per-pair parquets.

    If `include_per_token` is True, also write the D5 per-token log-prob
    table under `per_token/` — enables MORCELA rescoring without
    re-forward-passing.

    Returns a dict summarizing what was written.
    """
    output_root = Path(output_root)

    items_df = _items_df(item_results)
    pairs_df = _pairs_df(pair_results)

    items_root = output_root / "items"
    pairs_root = output_root / "pairs"

    item_paths = _write_partitioned(items_df, items_root, cell_id)
    pair_paths = _write_partitioned(pairs_df, pairs_root, cell_id)

    per_token_paths: List[Path] = []
    if include_per_token:
        per_token_root = output_root / "per_token"
        per_token_df = _per_token_df(item_results)
        per_token_paths = _write_partitioned(per_token_df, per_token_root, cell_id)

    logger.info(
        "Wrote cell %s: %d item partitions, %d pair partitions, %d per-token partitions",
        cell_id, len(item_paths), len(pair_paths), len(per_token_paths),
    )
    return {
        "cell_id": cell_id,
        "n_item_rows": len(items_df),
        "n_pair_rows": len(pairs_df),
        "item_paths": [str(p) for p in item_paths],
        "pair_paths": [str(p) for p in pair_paths],
        "per_token_paths": [str(p) for p in per_token_paths],
    }


# --- DuckDB view registration ----------------------------------------------

def register_duckdb_views(con, output_root: Path) -> None:
    """Register `items`, `pairs`, `per_token` views over the parquet tree.

    Uses hive-partitioning so filters on language/category/condition
    prune files at the scan level. Pass an open DuckDB connection;
    caller owns its lifecycle.
    """
    output_root = Path(output_root)
    pairs_glob = str(output_root / "pairs" / "**" / "*.parquet")
    items_glob = str(output_root / "items" / "**" / "*.parquet")
    pt_glob = str(output_root / "per_token" / "**" / "*.parquet")

    con.execute(f"""
        CREATE OR REPLACE VIEW items AS
        SELECT * FROM read_parquet('{items_glob}', hive_partitioning=1)
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW pairs AS
        SELECT * FROM read_parquet('{pairs_glob}', hive_partitioning=1)
    """)
    # per_token is optional; create the view only if any files exist.
    if any(output_root.joinpath("per_token").rglob("*.parquet")):
        con.execute(f"""
            CREATE OR REPLACE VIEW per_token AS
            SELECT * FROM read_parquet('{pt_glob}', hive_partitioning=1)
        """)
