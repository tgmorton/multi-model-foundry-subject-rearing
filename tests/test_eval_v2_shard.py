"""eval_v2_cell --checkpoint_shard: shard parsing and the checkpoint view."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_v2_cell as E  # noqa: E402
from evaluation.runners.per_model_runner import find_checkpoints_sorted  # noqa: E402

STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]


@pytest.fixture
def ckpt_root(tmp_path):
    root = tmp_path / "run"
    for s in STEPS:
        d = root / f"checkpoint-{s}"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"w")
    (root / "checkpoint-8.tmp").mkdir()          # staging dirs are not checkpoints
    (root / "notes.txt").write_text("x")
    return root


@pytest.mark.parametrize("spec,expected", [("0/1", (0, 1)), ("2/5", (2, 5)), ("9/10", (9, 10))])
def test_parse_shard(spec, expected):
    assert E._parse_shard(spec) == expected


@pytest.mark.parametrize("spec", ["1/1", "5/5", "-1/3", "3", "a/b", "0/0"])
def test_parse_shard_rejects(spec):
    with pytest.raises(SystemExit):
        E._parse_shard(spec)


def test_shards_partition_checkpoints(ckpt_root, tmp_path):
    n = 3
    seen = []
    for i in range(n):
        view = E._checkpoint_shard_view(ckpt_root, tmp_path / f"view{i}", i, n)
        steps = [s for s, _ in find_checkpoints_sorted(view)]
        assert steps == STEPS[i::n]                      # every n-th by step, from index i
        assert all((view / f"checkpoint-{s}").is_symlink() for s in steps)
        assert all((view / f"checkpoint-{s}" / "model.safetensors").is_file() for s in steps)
        seen += steps
    assert sorted(seen) == STEPS                         # disjoint and complete


def test_view_is_rebuilt(ckpt_root, tmp_path):
    view = tmp_path / "view"
    E._checkpoint_shard_view(ckpt_root, view, 0, 2)
    E._checkpoint_shard_view(ckpt_root, view, 1, 2)    # a restarted pod with another shard
    assert [s for s, _ in find_checkpoints_sorted(view)] == STEPS[1::2]
