"""Escape gates and truncated LR horizons for rater/race trials (D15)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import rater_sweep as rs  # noqa: E402


def test_load_settings_parses_gates_and_horizon():
    s = rs.load_settings({"rater_sweep": {"enabled": True, "sweep_train_steps": 2750,
                                          "scheduler_train_steps": 29000,
                                          "gates": [[920, 5.5], [2750, 4.5]]}})
    assert s.gates == ((920, 5.5), (2750, 4.5))
    assert s.scheduler_train_steps == 29000


def test_horizon_must_cover_the_trial():
    with pytest.raises(ValueError):
        rs.RaterSweepSettings(sweep_train_steps=3000, scheduler_train_steps=1000)


@pytest.mark.parametrize("bad", [[(0, 5.0)], [(10, 0)], [(10,)]])
def test_bad_gates_rejected(bad):
    with pytest.raises(ValueError):
        rs.RaterSweepSettings(gates=tuple(tuple(g) for g in bad))


def test_gate_failure_semantics():
    s = rs.RaterSweepSettings(gates=((1000, 5.5), (2000, 4.5)))
    e = math.exp
    assert rs.gate_failure(500, e(7.0), s) is None             # before any gate
    assert rs.gate_failure(1000, e(5.4), s) is None            # passes gate 1
    assert rs.gate_failure(1000, e(5.6), s) == (1000, 5.5)     # fails gate 1
    assert rs.gate_failure(2500, e(4.6), s) == (2000, 4.5)     # fails gate 2
    assert rs.gate_failure(2500, float("nan"), s) == (1000, 5.5)
    assert rs.gate_failure(10, float("nan"), s) is None


def test_apply_rater_overrides_truncated_schedule():
    cfg = {"experiment_name": "x", "data": {"batch_size": 64, "training_corpus": "c",
                                            "source_corpus": "c", "test_corpus": "c"},
           "training": {"gradient_accumulation_steps": 16}}
    s = rs.RaterSweepSettings(sweep_train_steps=2750, scheduler_train_steps=29000)
    rs.apply_rater_overrides(cfg, s, wandb_run_id="r1", condition="baseline",
                             corpus_dir=None, num_chunks=lambda c: 900_000)
    t = cfg["training"]
    assert t["train_steps"] == 2750 and t["scheduler_train_steps"] == 29000
    assert t["warmup_steps"] is None


def test_apply_rater_overrides_default_schedule_completes_in_trial():
    cfg = {"experiment_name": "x", "data": {"batch_size": 32}, "training": {}}
    rs.apply_rater_overrides(cfg, rs.RaterSweepSettings(sweep_train_steps=3000),
                             wandb_run_id="r1", condition="base", corpus_dir=None,
                             num_chunks=lambda c: 100_000)
    assert cfg["training"]["scheduler_train_steps"] == 3000
