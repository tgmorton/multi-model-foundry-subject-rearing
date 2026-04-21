"""D6 test: pre-tokenized stimuli cache."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from evaluation.stimuli_cache import (
    StimulusCache,
    build_cache,
    build_or_load_cache,
    hash_stimuli_files,
    hash_tokenizer,
)


def _make_v2_csv(tmp_path: Path, name: str = "stimuli.csv") -> Path:
    df = pd.DataFrame([
        {"item_id": 1, "category": "subject_drop", "condition": "subj_3sg",
         "pronoun_status": 1,
         "context": "ana won the award .",
         "target": "she shows pride openly .",
         "hotspot_token": "shows", "hotspot_position": 1,
         "language": "en", "names": "ana", "generator": "test"},
        {"item_id": 1, "category": "subject_drop", "condition": "subj_3sg",
         "pronoun_status": 0,
         "context": "ana won the award .",
         "target": "shows pride openly .",
         "hotspot_token": "shows", "hotspot_position": 0,
         "language": "en", "names": "ana", "generator": "test"},
    ])
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _make_fake_tokenizer_file(tmp_path: Path) -> Path:
    """Create a dummy 'tokenizer.model' file purely for content-hashing."""
    path = tmp_path / "tokenizer.model"
    path.write_bytes(b"fake-tokenizer-content-0001")
    return path


def _make_word_level_tokenizer(words_to_ids: dict):
    """Mock tokenizer that treats each whitespace word as one subword.

    This keeps the hotspot subword index trivially equal to the word index.
    """
    tok = MagicMock()
    tok.bos_id.return_value = 1
    tok.eos_id.return_value = -1
    tok.pad_id.return_value = 0
    tok.get_piece_size.return_value = 1000

    def encode(text):
        return [words_to_ids.setdefault(w, 100 + len(words_to_ids)) for w in text.split()]

    def decode(ids):
        # reverse map if possible, else synthetic
        if len(ids) != 1:
            return "".join(f"<{i}>" for i in ids)
        for w, wid in words_to_ids.items():
            if wid == ids[0]:
                return w
        return f"<{ids[0]}>"

    tok.encode.side_effect = encode
    tok.decode.side_effect = decode
    return tok


def test_build_cache_produces_entries(tmp_path):
    csv = _make_v2_csv(tmp_path)
    tok_file = _make_fake_tokenizer_file(tmp_path)
    tok = _make_word_level_tokenizer({})

    cache = build_cache([csv], tok, tok_file)

    assert len(cache) == 2
    assert cache.meta["n_rows"] == 2
    assert cache.meta["languages"] == ["en"]
    assert cache.meta["categories"] == ["subject_drop"]

    overt, null = cache.rows
    if overt.pronoun_status == 0:
        overt, null = null, overt

    # Token count checks — word-level tokenizer makes this easy
    # context: "ana won the award ." → 5 words + BOS → 6 ids
    assert overt.context_len == 6
    # overt target: "she shows pride openly ." → 5 words
    assert overt.target_subword_count == 5
    # null target: "shows pride openly ." → 4 words
    assert null.target_subword_count == 4
    # full = context + target
    assert overt.full_len == overt.context_len + overt.target_subword_count
    assert null.full_len == null.context_len + null.target_subword_count


def test_hotspot_subword_index_matches_word_index_on_word_tokenizer(tmp_path):
    """On a tokenizer where each word is one subword, the subword index
    into full_token_ids = context_len + hotspot_position."""
    csv = _make_v2_csv(tmp_path)
    tok_file = _make_fake_tokenizer_file(tmp_path)
    tok = _make_word_level_tokenizer({})

    cache = build_cache([csv], tok, tok_file)
    for e in cache.rows:
        assert e.hotspot_token_index == e.context_len + e.hotspot_position


def test_cache_round_trip(tmp_path):
    csv = _make_v2_csv(tmp_path)
    tok_file = _make_fake_tokenizer_file(tmp_path)
    tok = _make_word_level_tokenizer({})

    cache = build_cache([csv], tok, tok_file)
    out = tmp_path / "cache.pkl"
    cache.save(out)
    loaded = StimulusCache.load(out)

    assert loaded.stimuli_id == cache.stimuli_id
    assert loaded.tokenizer_id == cache.tokenizer_id
    assert len(loaded) == len(cache)
    assert loaded.rows[0].target_token_ids == cache.rows[0].target_token_ids
    assert loaded.rows[0].hotspot_token_index == cache.rows[0].hotspot_token_index


def test_build_or_load_cache_skips_rebuild(tmp_path):
    """Second call with identical inputs should load, not rebuild."""
    csv = _make_v2_csv(tmp_path)
    tok_file = _make_fake_tokenizer_file(tmp_path)
    tok = _make_word_level_tokenizer({})

    cache_dir = tmp_path / "cache"

    cache1 = build_or_load_cache([csv], tok, tok_file, cache_dir)
    # Clear encode side_effect counter by making a fresh mock
    tok2 = _make_word_level_tokenizer({})
    cache2 = build_or_load_cache([csv], tok2, tok_file, cache_dir)

    # Second call should not have called encode at all — it loaded from disk
    assert tok2.encode.call_count == 0
    assert cache1.stimuli_id == cache2.stimuli_id


def test_stimuli_hash_stable_under_file_ordering(tmp_path):
    """Reordering the input CSV list shouldn't change the stimuli hash."""
    csv1 = _make_v2_csv(tmp_path, "a.csv")
    csv2 = _make_v2_csv(tmp_path, "b.csv")

    h1 = hash_stimuli_files([csv1, csv2])
    h2 = hash_stimuli_files([csv2, csv1])
    assert h1 == h2


def test_stimuli_hash_changes_with_content(tmp_path):
    csv = _make_v2_csv(tmp_path)
    h1 = hash_stimuli_files([csv])

    # Mutate the CSV
    df = pd.read_csv(csv)
    df.loc[0, "target"] = "something completely different ."
    df.to_csv(csv, index=False)
    h2 = hash_stimuli_files([csv])

    assert h1 != h2


def test_rejects_non_v2_schema(tmp_path):
    """A v1 CSV (with 'item', 'c_english', ...) must be rejected."""
    bad = pd.DataFrame([
        {"item": 1, "item_group": "1a_3rdSG", "pronoun_status": 1,
         "c_english": "marta won .", "target": "she shows .",
         "hotspot_english": "shows"},
    ])
    csv = tmp_path / "v1.csv"
    bad.to_csv(csv, index=False)

    tok_file = _make_fake_tokenizer_file(tmp_path)
    tok = _make_word_level_tokenizer({})

    with pytest.raises(ValueError, match="v2 columns"):
        build_cache([csv], tok, tok_file)


def test_cache_rejects_unknown_format_version(tmp_path):
    import pickle
    path = tmp_path / "bad.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"format_version": "unknown.v99"}, fh)
    with pytest.raises(ValueError, match="format_version"):
        StimulusCache.load(path)
