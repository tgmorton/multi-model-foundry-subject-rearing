"""D10 test: parquet output + DuckDB views."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from evaluation.output_v2 import (
    register_duckdb_views,
    write_cell_results,
)
from evaluation.runners.per_model_runner import (
    CheckpointItemResult,
    CheckpointPairResult,
)


def _make_item(**over):
    defaults = dict(
        cell_id="cellA",
        architecture="gpt2",
        intervention="baseline",
        rep=0,
        checkpoint_step=10,
        checkpoint_path="/tmp/fake",
        item_id=1,
        category="subject_drop",
        condition="subj_3sg",
        pronoun_status=1,
        language="en",
        target_sum_log_prob=-5.0,
        target_mean_log_prob=-1.25,
        target_n_tokens=4,
        target_unigram_sum_log_prob=-20.0,
        slor=3.75,
        hotspot_log_prob=-2.0,
        per_token_log_prob=[-1.0, -1.5, -1.25, -1.25],
        per_token_ids=[10, 11, 12, 13],
    )
    defaults.update(over)
    return CheckpointItemResult(**defaults)


def _make_pair(**over):
    defaults = dict(
        cell_id="cellA",
        architecture="gpt2",
        intervention="baseline",
        rep=0,
        checkpoint_step=10,
        item_id=1,
        category="subject_drop",
        condition="subj_3sg",
        language="en",
        overt_mean_log_prob=-1.25,
        overt_slor=3.75,
        overt_hotspot_log_prob=-2.0,
        null_mean_log_prob=-1.50,
        null_slor=3.50,
        null_hotspot_log_prob=-2.5,
        prefers_overt_meanlp=True,
        prefers_overt_slor=True,
        log_prob_diff_overt_minus_null=0.25,
        slor_diff_overt_minus_null=0.25,
        hotspot_log_prob_diff_overt_minus_null=0.5,
    )
    defaults.update(over)
    return CheckpointPairResult(**defaults)


def test_partitioned_write(tmp_path):
    items = [
        _make_item(item_id=1, pronoun_status=1),
        _make_item(item_id=1, pronoun_status=0,
                   target_mean_log_prob=-1.5, slor=3.5,
                   per_token_log_prob=[-1.5, -1.5, -1.5], per_token_ids=[11, 12, 13]),
        _make_item(item_id=2, pronoun_status=1, condition="subj_3pl", category="subject_drop",
                   per_token_log_prob=[-1.0, -1.0], per_token_ids=[20, 21]),
        _make_item(item_id=2, pronoun_status=0, condition="subj_3pl", category="subject_drop",
                   per_token_log_prob=[-1.2], per_token_ids=[21]),
    ]
    pairs = [
        _make_pair(item_id=1),
        _make_pair(item_id=2, condition="subj_3pl"),
    ]

    summary = write_cell_results(
        output_root=tmp_path,
        cell_id="cellA",
        item_results=items,
        pair_results=pairs,
    )

    # Flat layout — one file per (table, cell_id)
    assert (tmp_path / "items" / "cell_id=cellA.parquet").exists()
    assert (tmp_path / "pairs" / "cell_id=cellA.parquet").exists()

    # Summary counts
    assert summary["n_item_rows"] == 4
    assert summary["n_pair_rows"] == 2


def test_per_token_written_and_round_trips(tmp_path):
    items = [
        _make_item(per_token_log_prob=[-0.1, -0.2, -0.3], per_token_ids=[7, 8, 9]),
    ]
    pairs = [_make_pair()]

    write_cell_results(tmp_path, "cellB", items, pairs, include_per_token=True)

    pt_path = tmp_path / "per_token" / "cell_id=cellB.parquet"
    assert pt_path.exists()
    df = pd.read_parquet(pt_path)
    assert len(df) == 1
    assert list(df.iloc[0]["per_token_log_prob"]) == [-0.1, -0.2, -0.3]
    assert list(df.iloc[0]["per_token_ids"]) == [7, 8, 9]


def test_per_token_skipped_when_disabled(tmp_path):
    items = [_make_item()]
    pairs = [_make_pair()]
    write_cell_results(tmp_path, "c", items, pairs, include_per_token=False)
    assert not (tmp_path / "per_token").exists()


def test_duckdb_views_register_and_query(tmp_path):
    # Two cells for partition pruning tests
    items_a = [_make_item(cell_id="A", item_id=1, pronoun_status=1),
               _make_item(cell_id="A", item_id=1, pronoun_status=0,
                          target_mean_log_prob=-2.0)]
    pairs_a = [_make_pair(cell_id="A")]

    items_b = [_make_item(cell_id="B", item_id=1, pronoun_status=1,
                          condition="subj_3pl"),
               _make_item(cell_id="B", item_id=1, pronoun_status=0,
                          condition="subj_3pl",
                          target_mean_log_prob=-2.5)]
    pairs_b = [_make_pair(cell_id="B", condition="subj_3pl")]

    write_cell_results(tmp_path, "A", items_a, pairs_a)
    write_cell_results(tmp_path, "B", items_b, pairs_b)

    con = duckdb.connect()
    try:
        register_duckdb_views(con, tmp_path)
        # All pairs across both cells
        all_pairs = con.execute("SELECT cell_id, condition FROM pairs").fetchall()
        assert sorted(all_pairs) == [("A", "subj_3sg"), ("B", "subj_3pl")]

        # Partition pruning: filter by condition
        only_3sg = con.execute(
            "SELECT cell_id FROM pairs WHERE condition='subj_3sg'"
        ).fetchall()
        assert only_3sg == [("A",)]

        # Per-token view
        n_pt_rows = con.execute("SELECT COUNT(*) FROM per_token").fetchone()[0]
        assert n_pt_rows == 4  # 2 cells × 2 rows each
    finally:
        con.close()


def test_empty_results_no_crash(tmp_path):
    """Writing empty item/pair lists should not crash."""
    out = write_cell_results(tmp_path, "empty", [], [])
    assert out["n_item_rows"] == 0
    assert out["n_pair_rows"] == 0
