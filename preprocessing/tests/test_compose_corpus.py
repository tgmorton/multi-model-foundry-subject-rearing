"""
Tests for the three-step workflow: ``skip_backfill`` mode + compose step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import spacy

import preprocessing.ablations  # noqa: F401 — register
from preprocessing.annotate import annotate_corpus
from preprocessing.base import AblationPipeline
from preprocessing.config import AblationConfig
from scripts.compose_corpus import compose


_MINI_SPANISH = """\
= = = = MiniCorpus = = = =
llueve mucho hoy .
juan ha comido todo el pescado .
hay tres gatos en la casa .
parece que el tren va a llegar tarde .

los niños corren en el parque .
él vino conmigo y se fue contigo .
lo vi ayer y la encontré después .
mi casa y tu coche y su perro son grandes .
ana está cansada y triste .
"""

# A small secondary pool drawn from different content (simulates pull_10M).
_MINI_POOL = """\
hoy llueve otra vez .
marta corre por la playa .
nosotros jugamos con los niños .
vimos la película juntos .
yo nunca olvidaré aquel día .
ellos comieron en el restaurante .
la casa estaba muy vacía .
es imposible llegar a tiempo .
sus hijos estudian en madrid .
"""


@pytest.fixture(scope="module")
def spanish_available():
    try:
        spacy.load("es_core_news_lg")
    except OSError:
        pytest.skip("es_core_news_lg not available")


@pytest.fixture
def corpus_dirs(tmp_path, spanish_available) -> dict:
    """Build minimal train + pool fixtures on disk."""
    train_dir = tmp_path / "train"
    pool_dir = tmp_path / "pool"
    train_dir.mkdir()
    pool_dir.mkdir()
    (train_dir / "mini.train").write_text(_MINI_SPANISH)
    (pool_dir / "mini.train").write_text(_MINI_POOL)
    return {"train": train_dir, "pool": pool_dir, "root": tmp_path}


def test_skip_backfill_flag_skips_pool_draw(corpus_dirs, tmp_path):
    """When skip_backfill=True, the pipeline doesn't draw from the pool even
    if replacement_pool_dir is set. Output size equals the ablated train size,
    not the original target size."""
    output_dir = tmp_path / "ablated"
    cfg = AblationConfig(
        type="remove_expletive_sentences_es",
        input_path=corpus_dirs["train"],
        output_path=output_dir,
        replacement_pool_dir=corpus_dirs["pool"],  # set — but should be ignored
        skip_backfill=True,                         # ← the flag under test
        spacy_model="es_core_news_lg",
        skip_validation=True,
        chunk_size=100,
    )
    manifest = AblationPipeline(cfg).process_corpus()

    # The expletive remover WILL remove lines (hay, parece, llueve → all
    # flagged). Final token count should be LESS than original, since
    # no backfill ran.
    stats = manifest.metadata
    assert stats.total_tokens_final < stats.total_tokens_original
    # No pool draws recorded (since backfill was skipped entirely).
    assert stats.total_pool_lines_drawn == 0


def test_skip_backfill_default_false_preserves_legacy_behavior(
    corpus_dirs, tmp_path,
):
    """Without skip_backfill, the pipeline still backfills as before."""
    output_dir = tmp_path / "ablated_legacy"
    cfg = AblationConfig(
        type="remove_expletive_sentences_es",
        input_path=corpus_dirs["train"],
        output_path=output_dir,
        replacement_pool_dir=corpus_dirs["pool"],
        # skip_backfill defaults to False
        spacy_model="es_core_news_lg",
        skip_validation=True,
        chunk_size=100,
    )
    manifest = AblationPipeline(cfg).process_corpus()

    stats = manifest.metadata
    # With backfill, pool draws should be non-zero.
    assert stats.total_pool_lines_drawn > 0


def _ablate_no_backfill(input_dir: Path, output_dir: Path) -> None:
    cfg = AblationConfig(
        type="remove_expletive_sentences_es",
        input_path=input_dir,
        output_path=output_dir,
        skip_backfill=True,
        spacy_model="es_core_news_lg",
        skip_validation=True,
        chunk_size=100,
    )
    AblationPipeline(cfg).process_corpus()


def test_compose_hits_target_size_within_tolerance(corpus_dirs, tmp_path):
    """After compose, each output file's token count should be within a
    small tolerance of the original raw corpus (overshoot by whole-pool-line
    granularity is expected)."""
    ablated_train = tmp_path / "ablated_train"
    ablated_pool = tmp_path / "ablated_pool"
    _ablate_no_backfill(corpus_dirs["train"], ablated_train)
    _ablate_no_backfill(corpus_dirs["pool"], ablated_pool)

    output = tmp_path / "composed"
    manifest = compose(
        ablated_train=ablated_train,
        ablated_pool=ablated_pool,
        output=output,
        target_corpus=corpus_dirs["train"],
        seed=42,
    )

    # Every file must have (near-)target tokens.
    for entry in manifest["per_file"]:
        gap = entry["target_tokens"] - entry["final_tokens"]
        # Small overshoot is expected (last pool line pushes past target).
        # The mini fixture is tiny so allow generous tolerance.
        assert abs(gap) < 30, f"{entry['stem']}: gap={gap} is unexpectedly large"


def test_compose_is_deterministic(corpus_dirs, tmp_path):
    """Same seed twice → byte-identical output."""
    ablated_train = tmp_path / "ablated_train"
    ablated_pool = tmp_path / "ablated_pool"
    _ablate_no_backfill(corpus_dirs["train"], ablated_train)
    _ablate_no_backfill(corpus_dirs["pool"], ablated_pool)

    out_a = tmp_path / "composed_a"
    out_b = tmp_path / "composed_b"
    compose(ablated_train, ablated_pool, out_a, target_corpus=corpus_dirs["train"], seed=42)
    compose(ablated_train, ablated_pool, out_b, target_corpus=corpus_dirs["train"], seed=42)

    for name in ["mini.train"]:
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


def test_compose_records_pool_provenance(corpus_dirs, tmp_path):
    """COMPOSE_MANIFEST.json must record which pool line indices were drawn
    so a reviewer can audit the composition."""
    ablated_train = tmp_path / "ablated_train"
    ablated_pool = tmp_path / "ablated_pool"
    _ablate_no_backfill(corpus_dirs["train"], ablated_train)
    _ablate_no_backfill(corpus_dirs["pool"], ablated_pool)

    output = tmp_path / "composed"
    manifest = compose(
        ablated_train, ablated_pool, output,
        target_corpus=corpus_dirs["train"], seed=42,
    )

    # Manifest includes per-file provenance.
    assert "per_file" in manifest
    assert len(manifest["per_file"]) == 1  # mini.train
    entry = manifest["per_file"][0]
    assert entry["stem"] == "mini"
    assert "pool_line_indices_used" in entry
    assert "input_checksums" in manifest
    assert "totals" in manifest

    # On-disk manifest should be valid JSON matching the returned one.
    on_disk = json.loads((output / "COMPOSE_MANIFEST.json").read_text())
    assert on_disk["seed"] == 42
    assert on_disk["totals"]["files"] == 1

    # Pool remainder file must exist with the unused pool lines.
    remainder_path = output / "pool_remainder" / "mini.train"
    assert remainder_path.exists()


def test_compose_fails_loudly_on_pool_exhaustion(corpus_dirs, tmp_path):
    """If the ablated pool is too small to close the gap, compose must
    raise with a clear error — not silently under-fill."""
    ablated_train = tmp_path / "ablated_train"
    _ablate_no_backfill(corpus_dirs["train"], ablated_train)

    # Make an ablated pool that's ~empty — just one line.
    tiny_pool = tmp_path / "tiny_pool_source"
    tiny_pool.mkdir()
    (tiny_pool / "mini.train").write_text("hola .\n")
    ablated_pool = tmp_path / "ablated_pool_tiny"
    _ablate_no_backfill(tiny_pool, ablated_pool)

    with pytest.raises(ValueError, match="pool exhausted"):
        compose(
            ablated_train, ablated_pool,
            tmp_path / "composed_fail",
            target_corpus=corpus_dirs["train"],
            seed=42,
        )


def test_compose_rejects_missing_pool_stem(corpus_dirs, tmp_path):
    """Compose should refuse to silently drop genres — if train has a stem
    the pool doesn't, fail loudly."""
    ablated_train = tmp_path / "ablated_train"
    _ablate_no_backfill(corpus_dirs["train"], ablated_train)

    # Empty pool directory (no .train files at all)
    empty_pool = tmp_path / "empty_pool"
    empty_pool.mkdir()

    with pytest.raises((FileNotFoundError, ValueError)):
        compose(
            ablated_train, empty_pool,
            tmp_path / "composed",
            target_corpus=corpus_dirs["train"],
            seed=42,
        )
