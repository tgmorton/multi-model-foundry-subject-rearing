"""D11 test: content-addressed cache.

Asserts that:
- `checkpoint_id` is content-addressed and stable across runs.
- Running the cached runner twice on the same inputs does zero forward
  passes on the second call (this is the key idempotency property).
- Adding a fresh checkpoint after the first run only re-processes that
  one checkpoint, not the old ones.
- Bumping `scoring_version` invalidates the cache (triggers a full re-run).
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from evaluation.cache import CachedRunner, checkpoint_id, hash_file
from evaluation.runners.per_model_runner import CellSpec
from evaluation.stimuli_cache import build_cache


def _tiny_gpt2(seed: int = 42):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=128, n_positions=64, n_embd=32, n_layer=2, n_head=2,
    )
    return GPT2LMHeadModel(cfg)


def _word_tokenizer():
    words: dict = {}

    def encode(text):
        ids = []
        for w in text.split():
            if w not in words:
                words[w] = 1 + (len(words) % 127)
            ids.append(words[w])
        return ids

    tok = MagicMock()
    tok.bos_id.return_value = 1
    tok.eos_id.return_value = -1
    tok.pad_id.return_value = 0
    tok.get_piece_size.return_value = 128
    tok.encode.side_effect = encode
    tok.decode.side_effect = lambda ids: "tok"
    return tok


def _v2_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame([
        {"item_id": 1, "category": "subject_drop", "condition": "subj_3sg",
         "pronoun_status": 1, "context": "ana won .",
         "target": "she shows pride .",
         "hotspot_token": "shows", "hotspot_position": 1,
         "language": "en", "names": "ana", "generator": "test"},
        {"item_id": 1, "category": "subject_drop", "condition": "subj_3sg",
         "pronoun_status": 0, "context": "ana won .",
         "target": "shows pride .",
         "hotspot_token": "shows", "hotspot_position": 0,
         "language": "en", "names": "ana", "generator": "test"},
    ])
    p = tmp_path / "stimuli.csv"
    df.to_csv(p, index=False)
    return p


def _write_ckpt(root: Path, step: int, seed: int) -> Path:
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True, exist_ok=True)
    m = _tiny_gpt2(seed=seed)
    torch.save(m.state_dict(), d / "pytorch_model.bin")
    return d


def _make_cell(tmp_path: Path, n_checkpoints: int = 2) -> CellSpec:
    ckpt_root = tmp_path / "model"
    for i in range(n_checkpoints):
        _write_ckpt(ckpt_root, step=(i + 1) * 10, seed=i + 1)

    csv = _v2_csv(tmp_path)
    tok = _word_tokenizer()
    tok_file = tmp_path / "tok.model"
    tok_file.write_bytes(b"fake-tok")
    cache = build_cache([csv], tok, tok_file)

    return CellSpec(
        cell_id="test_cell",
        architecture="gpt2",
        intervention="baseline",
        rep=0,
        checkpoint_root=ckpt_root,
        model_factory=lambda: _tiny_gpt2(seed=99),
        stimuli=cache,
        unigram=None,
        device="cpu",
        batch_size=2,
    )


# --- Content-addressed hash --------------------------------------------------

def test_hash_file_stable(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    h1 = hash_file(p)
    h2 = hash_file(p)
    assert h1 == h2
    assert len(h1) == 12


def test_hash_changes_with_content(tmp_path):
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    p1.write_bytes(b"hello")
    p2.write_bytes(b"world")
    assert hash_file(p1) != hash_file(p2)


def test_checkpoint_id_uses_pytorch_model_bin(tmp_path):
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    (ckpt / "pytorch_model.bin").write_bytes(b"modelA")
    (ckpt / "other.txt").write_bytes(b"ignored")
    ident = checkpoint_id(ckpt)
    assert len(ident) == 12
    # Changing the other file doesn't change the id
    (ckpt / "other.txt").write_bytes(b"different")
    assert checkpoint_id(ckpt) == ident


# --- Cache behavior ----------------------------------------------------------

def test_first_run_processes_all_checkpoints(tmp_path):
    cell = _make_cell(tmp_path, n_checkpoints=2)
    output_root = tmp_path / "out"

    runner = CachedRunner(cell, output_root=output_root, scoring_version="v1")
    summary = runner.run_once()

    assert summary["n_cached"] == 0
    assert summary["n_processed"] == 2
    assert summary["n_forward_passes"] == 2
    # Markers written
    assert (output_root / ".cache").exists()
    markers = list((output_root / ".cache").glob("*.done"))
    assert len(markers) == 2


def test_second_run_is_zero_forward_passes(tmp_path):
    cell = _make_cell(tmp_path, n_checkpoints=2)
    output_root = tmp_path / "out"

    # First run (populate cache)
    runner1 = CachedRunner(cell, output_root=output_root, scoring_version="v1")
    s1 = runner1.run_once()
    assert s1["n_forward_passes"] == 2

    # Second run (identical inputs)
    runner2 = CachedRunner(cell, output_root=output_root, scoring_version="v1")
    s2 = runner2.run_once()
    assert s2["n_cached"] == 2
    assert s2["n_processed"] == 0
    assert s2["n_forward_passes"] == 0


def test_new_checkpoint_added_triggers_partial_rerun(tmp_path):
    cell = _make_cell(tmp_path, n_checkpoints=2)
    output_root = tmp_path / "out"

    runner1 = CachedRunner(cell, output_root=output_root, scoring_version="v1")
    runner1.run_once()

    # Add a third checkpoint
    _write_ckpt(cell.checkpoint_root, step=30, seed=99)

    runner2 = CachedRunner(cell, output_root=output_root, scoring_version="v1")
    s2 = runner2.run_once()
    assert s2["n_cached"] == 2
    assert s2["n_processed"] == 1
    assert s2["n_forward_passes"] == 1


def test_scoring_version_bump_invalidates_cache(tmp_path):
    cell = _make_cell(tmp_path, n_checkpoints=2)
    output_root = tmp_path / "out"

    # v1
    r1 = CachedRunner(cell, output_root=output_root, scoring_version="v1")
    r1.run_once()

    # v2 — all checkpoints look uncached under the new version
    r2 = CachedRunner(cell, output_root=output_root, scoring_version="v2")
    s2 = r2.run_once()
    assert s2["n_cached"] == 0
    assert s2["n_processed"] == 2
    assert s2["n_forward_passes"] == 2


def test_scratch_dir_stages_writes_to_ephemeral_then_copies(tmp_path):
    """With scratch_dir set, parquet lands at output_root and scratch is cleaned."""
    cell = _make_cell(tmp_path, n_checkpoints=2)
    output_root = tmp_path / "out"
    scratch = tmp_path / "scratch"

    runner = CachedRunner(
        cell, output_root=output_root, scoring_version="v1",
        scratch_dir=scratch,
    )
    summary = runner.run_once()
    assert summary["n_forward_passes"] == 2

    # Parquet files land at output_root exactly as without scratch.
    items_files = list((output_root / "items").rglob("*.parquet"))
    pairs_files = list((output_root / "pairs").rglob("*.parquet"))
    assert items_files, "items parquet missing on output_root"
    assert pairs_files, "pairs parquet missing on output_root"

    # Cache markers land on output_root (durable filesystem).
    markers = list((output_root / ".cache").glob("*.done"))
    assert len(markers) == 2

    # Scratch cell subdir is cleaned up (best-effort).
    cell_scratch = scratch / cell.cell_id
    assert not cell_scratch.exists(), (
        f"Scratch cell subdir should be cleaned after copy; got {cell_scratch}"
    )


def test_scratch_dir_preserves_idempotency_across_pods(tmp_path):
    """Second run in a 'different pod' (new scratch_dir) still hits cache."""
    cell = _make_cell(tmp_path, n_checkpoints=2)
    output_root = tmp_path / "out"

    # "Pod 1": writes with scratch_dir=A
    scratch_a = tmp_path / "scratch_a"
    r1 = CachedRunner(
        cell, output_root=output_root, scoring_version="v1",
        scratch_dir=scratch_a,
    )
    s1 = r1.run_once()
    assert s1["n_forward_passes"] == 2

    # "Pod 2": different scratch (simulates a new pod's emptyDir) — should
    # find markers on output_root and skip everything.
    scratch_b = tmp_path / "scratch_b"
    r2 = CachedRunner(
        cell, output_root=output_root, scoring_version="v1",
        scratch_dir=scratch_b,
    )
    s2 = r2.run_once()
    assert s2["n_forward_passes"] == 0
    assert s2["n_cached"] == 2


def test_no_output_written_when_fully_cached(tmp_path):
    """When everything is cached, writer is skipped (no partial overwrites)."""
    cell = _make_cell(tmp_path, n_checkpoints=1)
    output_root = tmp_path / "out"

    r1 = CachedRunner(cell, output_root=output_root, scoring_version="v1")
    r1.run_once()
    pair_file = next((output_root / "pairs").rglob("*.parquet"))
    mtime_1 = pair_file.stat().st_mtime

    # Fully cached second run — should not re-touch the parquet
    r2 = CachedRunner(cell, output_root=output_root, scoring_version="v1")
    s = r2.run_once()
    assert s["n_processed"] == 0
    assert s["write_summary"] is None  # writer not invoked
    assert pair_file.stat().st_mtime == mtime_1
