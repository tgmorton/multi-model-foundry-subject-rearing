"""D9 test: per-model runner loads architecture once and swaps state_dicts.

Asserts:
- `find_checkpoints_sorted` discovers and orders checkpoints correctly.
- `PerModelRunner.run` iterates through checkpoints and produces one
  (item, checkpoint) row per pronoun_status, plus one pair row per item.
- State_dict swapping works: two different checkpoints produce different
  log-probs for the same stimuli.
- The model factory is called exactly once across 40 checkpoints (key
  throughput property of D9).
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from evaluation.runners.per_model_runner import (
    CellSpec,
    PerModelRunner,
    find_checkpoints_sorted,
)
from evaluation.stimuli_cache import build_cache
from evaluation.unigram import UnigramTable


# --- Fixtures ---------------------------------------------------------------

def _make_tiny_gpt2(seed: int = 42):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        use_cache=True,
    )
    return GPT2LMHeadModel(cfg)


def _make_word_tokenizer():
    """Simple word-level tokenizer for stimuli_cache."""
    words: dict = {}

    def encode(text):
        ids = []
        for w in text.split():
            if w not in words:
                # Keep ids in [1, 127] — ensure we never emit the pad id 0
                # and stay under vocab_size=128.
                words[w] = 1 + (len(words) % 127)
            ids.append(words[w])
        return ids

    tok = MagicMock()
    tok.bos_id.return_value = 1
    tok.eos_id.return_value = -1
    tok.pad_id.return_value = 0
    tok.get_piece_size.return_value = 128
    tok.encode.side_effect = encode
    tok.decode.side_effect = lambda ids: " ".join(
        next((k for k, v in words.items() if v == i), f"<{i}>")
        for i in ids
    )
    return tok


def _make_v2_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame([
        {"item_id": 1, "category": "subject_drop", "condition": "subj_3sg",
         "pronoun_status": 1, "context": "ana won prize .",
         "target": "she shows pride .",
         "hotspot_token": "shows", "hotspot_position": 1,
         "language": "en", "names": "ana", "generator": "test"},
        {"item_id": 1, "category": "subject_drop", "condition": "subj_3sg",
         "pronoun_status": 0, "context": "ana won prize .",
         "target": "shows pride .",
         "hotspot_token": "shows", "hotspot_position": 0,
         "language": "en", "names": "ana", "generator": "test"},
        {"item_id": 2, "category": "subject_drop", "condition": "subj_3sg",
         "pronoun_status": 1, "context": "luis left work .",
         "target": "he took bus .",
         "hotspot_token": "took", "hotspot_position": 1,
         "language": "en", "names": "luis", "generator": "test"},
        {"item_id": 2, "category": "subject_drop", "condition": "subj_3sg",
         "pronoun_status": 0, "context": "luis left work .",
         "target": "took bus .",
         "hotspot_token": "took", "hotspot_position": 0,
         "language": "en", "names": "luis", "generator": "test"},
    ])
    path = tmp_path / "stimuli.csv"
    df.to_csv(path, index=False)
    return path


def _write_fake_checkpoint(root: Path, step: int, model_factory) -> Path:
    """Build a model, save its state_dict as a checkpoint."""
    ckpt_dir = root / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    m = model_factory()
    torch.save(m.state_dict(), ckpt_dir / "pytorch_model.bin")
    return ckpt_dir


# --- Tests ------------------------------------------------------------------

def test_find_checkpoints_sorted(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    # Create in non-numerical order
    for step in [10, 1, 500, 100]:
        (root / f"checkpoint-{step}").mkdir()
    (root / "not-a-checkpoint").mkdir()  # should be ignored

    results = find_checkpoints_sorted(root)
    steps = [s for s, _ in results]
    assert steps == [1, 10, 100, 500]


def test_runner_produces_results_for_each_checkpoint(tmp_path):
    """Full run: 3 checkpoints × 4 items = 12 item rows, 6 pair rows."""
    ckpt_root = tmp_path / "model"

    # Three distinct checkpoints (seeded differently) so results vary
    for step, seed in [(1, 1), (10, 2), (100, 3)]:
        _write_fake_checkpoint(ckpt_root, step, lambda s=seed: _make_tiny_gpt2(s))

    # Build stimuli cache + unigram
    csv = _make_v2_csv(tmp_path)
    tok = _make_word_tokenizer()
    tok_file = tmp_path / "tok.model"
    tok_file.write_bytes(b"fake-tokenizer")
    cache = build_cache([csv], tok, tok_file)

    # Uniform unigram
    import numpy as np
    unigram = UnigramTable(
        counts=np.ones(128, dtype=np.int64),
        vocab_size=128, total_tokens=128, smoothing_alpha=1.0,
        corpus_id="test_corpus", tokenizer_id=cache.tokenizer_id,
    )

    # Count how many times the model factory is called
    call_count = {"n": 0}

    def counting_factory():
        call_count["n"] += 1
        return _make_tiny_gpt2(seed=99)  # architecture only — weights swapped per ckpt

    cell = CellSpec(
        cell_id="test_cell",
        architecture="gpt2",
        intervention="baseline",
        rep=0,
        checkpoint_root=ckpt_root,
        model_factory=counting_factory,
        stimuli=cache,
        unigram=unigram,
        device="cpu",
        batch_size=4,
    )
    runner = PerModelRunner(cell)
    items, pairs = runner.run()

    # 3 checkpoints × 4 items = 12 item rows
    assert len(items) == 12
    # 3 checkpoints × 2 pairs = 6 pair rows
    assert len(pairs) == 6

    # Every row has provenance
    for r in items:
        assert r.cell_id == "test_cell"
        assert r.architecture == "gpt2"
        assert r.rep == 0
        assert r.target_n_tokens > 0
        assert r.slor is not None   # unigram supplied
    for r in pairs:
        assert r.cell_id == "test_cell"
        assert r.slor_diff_overt_minus_null is not None

    # **Key throughput property**: the architecture factory is called
    # exactly once across 3 checkpoints (not 3 times).
    assert call_count["n"] == 1


def test_runner_different_checkpoints_produce_different_scores(tmp_path):
    """State_dict swapping actually changes the output."""
    ckpt_root = tmp_path / "model"
    _write_fake_checkpoint(ckpt_root, 1, lambda: _make_tiny_gpt2(seed=1))
    _write_fake_checkpoint(ckpt_root, 10, lambda: _make_tiny_gpt2(seed=2))

    csv = _make_v2_csv(tmp_path)
    tok = _make_word_tokenizer()
    tok_file = tmp_path / "tok.model"
    tok_file.write_bytes(b"tok")
    cache = build_cache([csv], tok, tok_file)

    cell = CellSpec(
        cell_id="test", architecture="gpt2", intervention="baseline", rep=0,
        checkpoint_root=ckpt_root,
        model_factory=lambda: _make_tiny_gpt2(seed=99),
        stimuli=cache,
        device="cpu",
        batch_size=4,
    )
    runner = PerModelRunner(cell)
    items, _ = runner.run()

    # For item 1 overt, the log-prob at checkpoint-1 should differ from
    # checkpoint-10 (weights are different → logits differ).
    ckpt1_overt = [r for r in items
                   if r.checkpoint_step == 1 and r.item_id == 1 and r.pronoun_status == 1]
    ckpt10_overt = [r for r in items
                    if r.checkpoint_step == 10 and r.item_id == 1 and r.pronoun_status == 1]
    assert len(ckpt1_overt) == len(ckpt10_overt) == 1
    assert ckpt1_overt[0].target_mean_log_prob != ckpt10_overt[0].target_mean_log_prob


def test_runner_no_unigram_gives_null_slor(tmp_path):
    ckpt_root = tmp_path / "model"
    _write_fake_checkpoint(ckpt_root, 1, _make_tiny_gpt2)

    csv = _make_v2_csv(tmp_path)
    tok = _make_word_tokenizer()
    tok_file = tmp_path / "tok.model"
    tok_file.write_bytes(b"tok")
    cache = build_cache([csv], tok, tok_file)

    cell = CellSpec(
        cell_id="test", architecture="gpt2", intervention="baseline", rep=0,
        checkpoint_root=ckpt_root,
        model_factory=_make_tiny_gpt2,
        stimuli=cache,
        unigram=None,  # No unigram
        device="cpu",
    )
    runner = PerModelRunner(cell)
    items, pairs = runner.run()

    for r in items:
        assert r.slor is None
        assert r.target_unigram_sum_log_prob is None
    for r in pairs:
        assert r.slor_diff_overt_minus_null is None
        assert r.prefers_overt_slor is None


def test_runner_missing_checkpoint_bin_raises(tmp_path):
    ckpt_root = tmp_path / "model"
    (ckpt_root / "checkpoint-1").mkdir(parents=True)
    # Don't write pytorch_model.bin

    csv = _make_v2_csv(tmp_path)
    tok = _make_word_tokenizer()
    tok_file = tmp_path / "tok.model"
    tok_file.write_bytes(b"tok")
    cache = build_cache([csv], tok, tok_file)

    cell = CellSpec(
        cell_id="test", architecture="gpt2", intervention="baseline", rep=0,
        checkpoint_root=ckpt_root, model_factory=_make_tiny_gpt2, stimuli=cache,
    )
    runner = PerModelRunner(cell)
    with pytest.raises(FileNotFoundError, match="pytorch_model.bin"):
        runner.run()
