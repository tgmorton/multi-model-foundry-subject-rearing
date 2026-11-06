"""Tied heads left out of a checkpoint: allowed by name, and really tied after loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.runners.per_model_runner import assert_output_tied, is_tied_key  # noqa: E402


@pytest.mark.parametrize("key,tied,ok", [
    ("cls.predictions.decoder.weight", ["cls.predictions.decoder"], True),     # MegatronBert, transformers 4.41
    ("cls.predictions.decoder.bias", ["cls.predictions.decoder"], True),
    ("cls.predictions.decoder.bias", ["predictions.decoder.bias"], True),      # BertForMaskedLM
    ("lm_head.weight", ["lm_head.weight"], True),                              # GPT-2
    ("bert.embeddings.word_embeddings.weight", ["cls.predictions.decoder"], False),
    ("cls.predictions.transform.dense.weight", ["cls.predictions.decoder"], False),
])
def test_is_tied_key(key, tied, ok):
    assert is_tied_key(key, tied) is ok


def tiny():
    cfg = transformers.MegatronBertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                                          intermediate_size=32, max_position_embeddings=32)
    return cfg, transformers.MegatronBertForMaskedLM(cfg).eval()


def test_megatron_checkpoint_round_trip(tmp_path):
    """Save, load the way the eval does, and get the same logits: the head came back tied."""
    from safetensors.torch import load_file
    cfg, src = tiny()
    src.save_pretrained(tmp_path)
    state = load_file(str(tmp_path / "model.safetensors"))
    assert "cls.predictions.decoder.weight" not in state
    dst = transformers.MegatronBertForMaskedLM(cfg).eval()
    missing, unexpected = dst.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(is_tied_key(k, ["cls.predictions.decoder"]) for k in missing)
    dst.tie_weights()
    assert_output_tied(dst)
    ids = torch.arange(1, 9)[None]
    with torch.no_grad():
        assert torch.allclose(src(ids).logits, dst(ids).logits, atol=1e-5)


def test_untied_head_is_caught():
    _, m = tiny()
    m.cls.predictions.decoder.weight = torch.nn.Parameter(m.cls.predictions.decoder.weight.clone())
    with pytest.raises(RuntimeError, match="not tied"):
        assert_output_tied(m)
