"""BERT recipe v2 (D15, docs/BERT_RECIPE_V2.md): packing, cache keys,
pre-LN BERT, weight-decay groups."""

from __future__ import annotations

import types

import pytest
import torch

from model_foundry.cache_keys import compute_cache_key
from model_foundry.config import (DataConfig, ExperimentConfig, LoggingConfig, ModelConfig,
                                  TokenizerConfig, TrainingConfig)
from model_foundry.data import pack_token_stream


# ---------- packing ----------

def _old_contiguous(seqs, chunk_size):
    """The pre-D15 chunker, verbatim in behaviour."""
    buf, out = [], []
    for s in seqs:
        buf.extend(s)
        while len(buf) >= chunk_size:
            out.append(buf[:chunk_size])
            buf = buf[chunk_size:]
    return out


def test_contiguous_packing_matches_old_chunker():
    seqs = [list(range(i, i + (i % 7) + 1)) for i in range(200)]
    assert list(pack_token_stream(seqs, 16)) == _old_contiguous(seqs, 16)


def test_cls_sep_packing_layout():
    seqs = [[10, 11], [12], [13, 14, 15], [16, 17]]
    chunks = list(pack_token_stream(seqs, 4, cls_id=0, sep_id=1))
    # stream: 10 11 S 12 S 13 14 15 S 16 17 S -> bodies of 3 behind [CLS]
    assert chunks == [[0, 10, 11, 1], [0, 12, 1, 13], [0, 14, 15, 1], [0, 16, 17, 1]]
    assert all(len(c) == 4 and c[0] == 0 for c in chunks)


def test_packing_drops_remainder_and_respects_max_chunks():
    seqs = [[5] * 10]
    assert list(pack_token_stream(seqs, 4)) == [[5] * 4, [5] * 4]          # 2 left over
    assert len(list(pack_token_stream([[5] * 100], 4, max_chunks=3))) == 3


def test_packing_rejects_no_room_for_body():
    with pytest.raises(ValueError):
        list(pack_token_stream([[1, 2]], 1, cls_id=0, sep_id=1))


# ---------- cache key ----------

def _key_inputs(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.train").write_text("hello world\n")
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "tokenizer.json").write_text("{}")
    return str(corpus), str(tok)


def test_cache_key_default_packing_is_unchanged(tmp_path):
    corpus, tok = _key_inputs(tmp_path)
    legacy = compute_cache_key(corpus, tok, 128, [])
    assert compute_cache_key(corpus, tok, 128, [], packing="contiguous") == legacy
    assert compute_cache_key(corpus, tok, 128, [], packing=None) == legacy
    assert compute_cache_key(corpus, tok, 128, [], packing="cls_sep") != legacy


# ---------- pre-LN BERT ----------

def _bert_config(norm_position="pre", pad_token_id=4, **training):
    t = dict(output_dir="x", learning_rate=1e-3, adam_beta1=0.9, adam_beta2=0.98,
             adam_epsilon=1e-6, epochs=1, objective="masked_lm")
    t.update(training)
    return ExperimentConfig(
        experiment_name="t",
        data=DataConfig(source_corpus="c", training_corpus="c", batch_size=2,
                        max_sequence_length=32, packing="cls_sep"),
        tokenizer=TokenizerConfig(output_dir="tok", vocab_size=100),
        model=ModelConfig(
            architecture="bert",
            transformer=dict(layers=2, embedding_size=32, hidden_size=32,
                             intermediate_hidden_size=64, attention_heads=4,
                             activation_function="gelu", dropout=0.0, attention_dropout=0.0),
            bert=dict(norm_position=norm_position, pad_token_id=pad_token_id),
        ),
        training=TrainingConfig(**t),
        logging=LoggingConfig(),
        random_seed=0,
    )


def test_pre_ln_bert_is_megatron_and_trains(tmp_path):
    from transformers import AutoModelForMaskedLM
    from model_foundry.architectures.bert import BERTModel

    m = BERTModel.from_config(_bert_config())
    assert m.hf_model.config.model_type == "megatron-bert"
    assert m.hf_model.config.pad_token_id == 4
    assert m.hf_model.config.max_position_embeddings == 32
    ids = torch.randint(5, 100, (2, 32))
    labels = ids.clone()
    labels[:, ::2] = -100
    out = m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    m.save_pretrained(str(tmp_path / "ck"))
    reloaded = AutoModelForMaskedLM.from_pretrained(str(tmp_path / "ck"))
    assert type(reloaded).__name__ == "MegatronBertForMaskedLM"


@pytest.mark.parametrize("impl", ["flash_attention_2", "sdpa"])
def test_pre_ln_bert_refuses_fused_attention(impl):
    from model_foundry.architectures.bert import BERTModel
    with pytest.raises(ValueError):
        BERTModel.from_config(_bert_config(), attn_implementation=impl)


def test_post_ln_bert_unchanged_default():
    cfg = _bert_config(norm_position="post", pad_token_id=None)
    assert cfg.model.bert.norm_position == "post"
    assert cfg.model.bert.pad_token_id is None


# ---------- weight decay groups ----------

def _call_optimizer_params(cfg):
    from model_foundry.architectures.bert import BERTModel
    from model_foundry.trainer import Trainer
    model = BERTModel.from_config(cfg)
    fake = types.SimpleNamespace(config=cfg, model=model)
    return model, Trainer._optimizer_params(fake)


def test_default_optimizer_params_are_all_parameters():
    model, params = _call_optimizer_params(_bert_config())
    assert [id(p) for p in params] == [id(p) for p in model.parameters()]


def test_no_decay_norm_bias_groups():
    model, groups = _call_optimizer_params(_bert_config(weight_decay=0.1, no_decay_norm_bias=True))
    decay, no_decay = groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    names = {id(p): n for n, p in model.named_parameters()}
    nd = {names[id(p)] for p in no_decay["params"]}
    d = {names[id(p)] for p in decay["params"]}
    assert all(n.endswith("bias") or "LayerNorm" in n or n.endswith("ln.weight") for n in nd)
    assert not any(n.endswith("bias") for n in d)
    assert len(d) + len(nd) == len(list(model.parameters()))
