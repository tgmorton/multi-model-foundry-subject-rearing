"""
Parity tests for the tree-detector ``annotated_input_path`` cache.

When ``EuroparlAlignmentGenerator`` runs with ``emit_target_docbin=True``
it writes a DocBin of target-language parses next to
``aligned_checkpoint.jsonl``. The tree detector's ``align_gold_data``
can then skip re-parsing ``clean_text`` — this test pins the invariant
that cached and live-parse outputs are identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import spacy
from spacy.tokens import DocBin

from analysis.pronoun_recovery.parallel_data.generator import (
    TARGET_DOCBIN_FILENAME,
)
from analysis.pronoun_recovery.tree_detector.feature_extractor import (
    VerbFeatureExtractor,
)
from analysis.pronoun_recovery.tree_detector.label_aligner import align_gold_data
from preprocessing.annotate import DOCBIN_ATTRS, dump_vocab


# Small synthetic Spanish records with the same shape that
# EuroparlAlignmentGenerator produces. Each has a `clean_text`, optional
# `markers`, and the metadata the tree detector inspects.
_MINI_RECORDS = [
    {
        "clean_text": "creo que el niño come manzanas .",
        "markers": [
            {
                "label": "PRO.1sg", "lexical_form": "yo",
                "position": 0, "confidence": "high",
                "en_pronoun": "I", "it_verb": "creo",
            },
        ],
        "id": "test:0", "genre": "Europarl", "source": "test",
    },
    {
        "clean_text": "vemos al gato corriendo .",
        "markers": [
            {
                "label": "PRO.1pl", "lexical_form": "nosotros",
                "position": 0, "confidence": "high",
                "en_pronoun": "we", "it_verb": "vemos",
            },
        ],
        "id": "test:1", "genre": "Europarl", "source": "test",
    },
    {
        "clean_text": "llueve mucho hoy en la ciudad .",
        "markers": [],  # No markers; should still be processed
        "id": "test:2", "genre": "Europarl", "source": "test",
    },
    {
        "clean_text": "los niños juegan en el parque .",
        "markers": [
            {
                "label": "PRO.3pl", "lexical_form": "ellos",
                "position": 0, "confidence": "medium",
                "en_pronoun": "they", "it_verb": "juegan",
            },
        ],
        "id": "test:3", "genre": "Europarl", "source": "test",
    },
]


@pytest.fixture(scope="module")
def spanish_nlp():
    try:
        return spacy.load("es_core_news_lg")
    except OSError:
        pytest.skip("es_core_news_lg not available")


@pytest.fixture(scope="module")
def alignment_fixture(tmp_path_factory, spanish_nlp) -> Path:
    """Write a synthetic aligned_checkpoint.jsonl + target_parses.spacy.

    Simulates what ``EuroparlAlignmentGenerator`` produces when run with
    ``emit_target_docbin=True`` — a JSONL of records plus a DocBin of
    target-language parses in matching 1:1 order.
    """
    root = tmp_path_factory.mktemp("alignment")

    # Write records
    checkpoint_path = root / "aligned_checkpoint.jsonl"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        for rec in _MINI_RECORDS:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Build + write DocBin + vocab, mirroring generator output structure
    docbin = DocBin(attrs=DOCBIN_ATTRS, store_user_data=False)
    for rec in _MINI_RECORDS:
        doc = spanish_nlp(rec["clean_text"])
        docbin.add(doc)

    (root / TARGET_DOCBIN_FILENAME).write_bytes(docbin.to_bytes())
    dump_vocab(spanish_nlp, root)

    return root


def test_align_gold_data_live_vs_cached_parity(alignment_fixture, spanish_nlp):
    """Cached and live-parse paths must produce identical features + labels."""
    extractor = VerbFeatureExtractor(language="es")

    X_live, y_live = align_gold_data(
        data_path=alignment_fixture / "aligned_checkpoint.jsonl",
        nlp=spanish_nlp,
        extractor=extractor,
        batch_size=10,
        annotated_input_path=None,
    )

    X_cached, y_cached = align_gold_data(
        data_path=alignment_fixture / "aligned_checkpoint.jsonl",
        nlp=spanish_nlp,
        extractor=extractor,
        batch_size=10,
        annotated_input_path=alignment_fixture,
    )

    # Label arrays must be identical in value and order
    np.testing.assert_array_equal(y_live, y_cached)

    # Feature DataFrames must be identical row-for-row
    assert len(X_live) == len(X_cached), (
        f"Row count mismatch: live={len(X_live)}, cached={len(X_cached)}"
    )
    assert list(X_live.columns) == list(X_cached.columns)
    # Compare as dicts-of-rows since DataFrame.equals can be strict about
    # dtype on object columns.
    live_rows = X_live.to_dict(orient="records")
    cached_rows = X_cached.to_dict(orient="records")
    for i, (a, b) in enumerate(zip(live_rows, cached_rows)):
        assert a == b, f"Row {i} differs:\n  live={a}\n  cached={b}"


def test_align_gold_data_detects_stale_cache(alignment_fixture, spanish_nlp, tmp_path):
    """A DocBin with the wrong number of docs must fail loudly."""
    # Copy records and DocBin to a fresh dir, then trim the records
    # to simulate a stale cache (docbin has N docs, records have N-1).
    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()

    # Shorter records list (3 instead of 4)
    with open(stale_dir / "aligned_checkpoint.jsonl", "w") as f:
        for rec in _MINI_RECORDS[:-1]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Full DocBin (4 docs)
    full_docbin = (alignment_fixture / TARGET_DOCBIN_FILENAME).read_bytes()
    (stale_dir / TARGET_DOCBIN_FILENAME).write_bytes(full_docbin)

    extractor = VerbFeatureExtractor(language="es")

    with pytest.raises(ValueError, match="Cache is stale"):
        align_gold_data(
            data_path=stale_dir / "aligned_checkpoint.jsonl",
            nlp=spanish_nlp,
            extractor=extractor,
            batch_size=10,
            annotated_input_path=stale_dir,
        )


def test_align_gold_data_missing_cache_file(alignment_fixture, spanish_nlp, tmp_path):
    """annotated_input_path set but target_parses.spacy missing → clear error."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Just records, no DocBin
    with open(empty_dir / "aligned_checkpoint.jsonl", "w") as f:
        for rec in _MINI_RECORDS:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    extractor = VerbFeatureExtractor(language="es")

    with pytest.raises(FileNotFoundError, match=TARGET_DOCBIN_FILENAME):
        align_gold_data(
            data_path=empty_dir / "aligned_checkpoint.jsonl",
            nlp=spanish_nlp,
            extractor=extractor,
            batch_size=10,
            annotated_input_path=empty_dir,
        )
