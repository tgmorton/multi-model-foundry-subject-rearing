"""
Parity tests for the DocBin annotation cache.

Covers three paths to producing an annotated corpus:

1. **Live parse** — ``AblationPipeline`` running with no cache.
2. **Standalone annotator** — ``preprocessing.annotate.annotate_corpus`` +
   ``AblationPipeline`` reading the resulting DocBin.
3. **Unified annotator** — ``CorpusAnnotationPipeline`` with ``emit_docbin=True``
   + ``AblationPipeline`` reading the DocBin it emitted alongside the
   Parquet analytics layers.

All three paths must produce byte-identical ablation output for any
registered ablation. These tests lock that invariant in so future
refactors (cleaners, annotator schema changes, DocBin attr tweaks)
don't silently diverge the three producers.

Requires ``es_core_news_lg`` to be installed. Tests are skipped if not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import spacy

import preprocessing.ablations  # noqa: F401 — register ablations
from preprocessing.annotate import annotate_corpus
from preprocessing.base import AblationPipeline
from preprocessing.config import AblationConfig


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


_ABLATIONS = (
    "remove_expletive_sentences_es",
    "impoverish_case_es",
    "lemmatize_verbs",
)


@pytest.fixture(scope="module")
def spanish_available() -> bool:
    """Skip module if es_core_news_lg is not installed."""
    try:
        spacy.load("es_core_news_lg")
    except OSError:
        pytest.skip("es_core_news_lg not available")
    return True


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory, spanish_available) -> Path:
    """Write the mini Spanish corpus to disk once per module."""
    root = tmp_path_factory.mktemp("corpus")
    (root / "mini.train").write_text(_MINI_SPANISH)
    return root


def _run_ablation(
    corpus: Path,
    output: Path,
    ablation: str,
    annotated_input_path: Path | None = None,
) -> str:
    """Run one ablation and return the output file contents."""
    cfg = AblationConfig(
        type=ablation,
        input_path=corpus,
        output_path=output,
        annotated_input_path=annotated_input_path,
        spacy_model="es_core_news_lg",
        skip_validation=True,
        chunk_size=100,
    )
    AblationPipeline(cfg).process_corpus()
    return (output / "mini.train").read_text()


@pytest.fixture(scope="module")
def standalone_docbin_dir(tmp_path_factory, corpus_dir) -> Path:
    """Run ``preprocessing.annotate`` once; share the DocBin across tests.

    Module-scoped so we don't repeat the ~150 MB ``es_core_news_lg``
    vocab dump for every parametrized ablation.
    """
    annotated = tmp_path_factory.mktemp("standalone")
    annotate_corpus(
        input_dir=corpus_dir,
        spacy_model="es_core_news_lg",
        output_dir=annotated,
    )
    return annotated


@pytest.mark.parametrize("ablation", _ABLATIONS)
def test_standalone_cache_parity(
    tmp_path, corpus_dir, standalone_docbin_dir, ablation,
):
    """Ablation from standalone-annotator cache matches live parse."""
    live = _run_ablation(corpus_dir, tmp_path / "live" / ablation, ablation)
    cached = _run_ablation(
        corpus_dir, tmp_path / "cached" / ablation, ablation,
        annotated_input_path=standalone_docbin_dir,
    )
    assert cached == live


@pytest.fixture(scope="module")
def unified_docbin_dir(tmp_path_factory, corpus_dir) -> Path:
    """Run the unified CorpusAnnotationPipeline once; share the DocBin.

    Module-scoped so we don't pay the vocab-dump cost (which for
    ``es_core_news_lg`` is ~150 MB) once per parametrized ablation.
    """
    from analysis.corpus_descriptives.config import CorpusAnalysisConfig
    from analysis.corpus_descriptives.pipeline import CorpusAnnotationPipeline
    from analysis.corpus_descriptives.annotators import get_default_annotators

    output_dir = tmp_path_factory.mktemp("unified")
    cfg = CorpusAnalysisConfig(
        input_path=corpus_dir,
        output_path=output_dir,
        split_name="mini_test",
        spacy_model="es_core_news_lg",
        language="es",
        genre_map={"mini": "MiniCorpus"},
        annotation_mode=True,
        layered_output=True,
        emit_docbin=True,
    )
    annotators = [
        a for a in get_default_annotators(language="es")
        if a.name == "ClauseStructureAnnotator"
    ]
    metadata = CorpusAnnotationPipeline(
        cfg, annotators=annotators, layered=True,
    ).run()
    return Path(metadata["docbin_output_dir"])


def test_unified_pipeline_emits_expected_files(unified_docbin_dir):
    """Smoke check the unified pipeline writes the expected cache files."""
    assert (unified_docbin_dir / "mini.spacy").exists()
    assert (unified_docbin_dir / "mini.linemap.jsonl").exists()
    assert (unified_docbin_dir / "ANNOTATION_MANIFEST.json").exists()
    assert (unified_docbin_dir / "vocab").is_dir()


def test_unified_preserves_line_order(unified_docbin_dir):
    """The unified pipeline's linemap must be in source-line order."""
    import json

    linemap_path = unified_docbin_dir / "mini.linemap.jsonl"
    with open(linemap_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    line_indices = [e["line_idx"] for e in entries]
    assert line_indices == sorted(line_indices), (
        f"linemap must be in ascending line_idx order; got {line_indices}"
    )


@pytest.mark.parametrize("ablation", _ABLATIONS)
def test_unified_pipeline_cache_parity(
    tmp_path, corpus_dir, unified_docbin_dir, ablation,
):
    """Ablation from CorpusAnnotationPipeline's DocBin matches live parse."""
    live = _run_ablation(corpus_dir, tmp_path / "live" / ablation, ablation)
    unified = _run_ablation(
        corpus_dir,
        tmp_path / "unified_ablated" / ablation,
        ablation,
        annotated_input_path=unified_docbin_dir,
    )
    assert unified == live


# ---------------------------------------------------------------------------
# CHILDES spot-check — confirm the unified pipeline handles speaker-prefixed
# lines AND boundary markers correctly: Parquet captures speaker/role metadata,
# DocBin contains clean parses, ablations consuming the DocBin produce clean
# output (NO `*CHI:` prefix). This is a documented behavior change from the
# raw-line ablation path; the test pins it.
# ---------------------------------------------------------------------------

_MINI_CHILDES = """\
= = = childes/Eng-NA/Bloom73/Peter/01.cha = = =
*MOT:\tsee the cat .
*CHI:\tkitty .
*MOT:\tyes , that is a kitty .
*CHI:\tkitty run .
= = = childes/Eng-NA/Bloom73/Peter/02.cha = = =
*MOT:\twhere is the ball ?
*CHI:\tball .
"""


@pytest.fixture(scope="module")
def english_available() -> bool:
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not available")
    return True


@pytest.fixture(scope="module")
def childes_corpus(tmp_path_factory, english_available) -> Path:
    root = tmp_path_factory.mktemp("childes_corpus")
    (root / "childes.train").write_text(_MINI_CHILDES)
    return root


@pytest.fixture(scope="module")
def childes_unified_output(tmp_path_factory, childes_corpus) -> tuple:
    """Run unified pipeline on CHILDES fixture; return (parquet_dir, docbin_dir)."""
    from analysis.corpus_descriptives.config import CorpusAnalysisConfig
    from analysis.corpus_descriptives.pipeline import CorpusAnnotationPipeline
    from analysis.corpus_descriptives.annotators import get_default_annotators

    output_dir = tmp_path_factory.mktemp("childes_unified")
    cfg = CorpusAnalysisConfig(
        input_path=childes_corpus,
        output_path=output_dir,
        split_name="childes_test",
        spacy_model="en_core_web_sm",
        language="en",
        genre_map={"childes": "CHILDES"},
        annotation_mode=True,
        layered_output=True,
        emit_docbin=True,
    )
    # Minimal annotator set — we care about base Parquet schema + speaker/role.
    annotators = [
        a for a in get_default_annotators(language="en")
        if a.name in {"ClauseStructureAnnotator", "PronounAnnotator"}
    ]
    metadata = CorpusAnnotationPipeline(
        cfg, annotators=annotators, layered=True,
    ).run()
    return output_dir, Path(metadata["docbin_output_dir"])


def test_childes_parquet_captures_speaker_and_role(childes_unified_output):
    """Parquet base layer must record the speaker code and role per sentence."""
    import pyarrow.parquet as pq

    output_dir, _ = childes_unified_output
    base_parquets = list((output_dir / "annotated_corpus" / "base").glob("*.parquet"))
    assert base_parquets, (
        f"No base Parquet found in {output_dir / 'annotated_corpus' / 'base'}"
    )
    table = pq.read_table(base_parquets[0])
    speakers = table.column("speaker").to_pylist()
    roles = table.column("role").to_pylist()

    assert "MOT" in speakers, f"Expected MOT speaker in Parquet; got {set(speakers)}"
    assert "CHI" in speakers, f"Expected CHI speaker in Parquet; got {set(speakers)}"
    assert "adult" in roles, f"Expected adult role; got {set(roles)}"
    assert "child" in roles, f"Expected child role; got {set(roles)}"


def test_childes_docbin_has_clean_parses(childes_unified_output):
    """DocBin must contain cleaned parses — no ``*CHI:`` / ``*MOT:`` tokens."""
    from preprocessing.annotate import load_annotated_file

    _, docbin_dir = childes_unified_output
    nlp = spacy.load("en_core_web_sm")
    docs, linemap = load_annotated_file(
        annotated_dir=docbin_dir, file_stem="childes", vocab=nlp.vocab,
    )

    # No doc should contain a speaker marker — cleaner stripped them before parsing.
    for doc in docs:
        text = doc.text
        assert "*CHI:" not in text, f"Doc should not contain *CHI:, got {text!r}"
        assert "*MOT:" not in text, f"Doc should not contain *MOT:, got {text!r}"

    # Linemap must preserve the RAW source lines verbatim for pass-throughs.
    passthroughs = [e for e in linemap if e["doc_idx"] is None]
    assert len(passthroughs) >= 2, (
        f"Expected at least 2 passthrough entries (boundary markers); "
        f"got {len(passthroughs)}"
    )
    assert any(
        e["raw_text"].startswith("= = = childes/")
        for e in passthroughs
    ), "Expected CHILDES boundary marker in passthrough raw_text"


def test_childes_ablation_from_docbin_is_clean(
    tmp_path, childes_corpus, childes_unified_output,
):
    """Ablations consuming the unified CHILDES DocBin produce CLEAN output.

    Speaker prefixes (``*CHI:``, ``*MOT:``) are stripped before parsing,
    so the DocBin contains clean content only. Ablation output therefore
    has no speaker markers. This is a documented behavior change from
    the raw-line ablation path — validated here so it doesn't regress.
    """
    _, docbin_dir = childes_unified_output

    # Lemmatize_verbs is language-agnostic; use it as the ablation lens.
    out_path = tmp_path / "childes_ablated"
    cfg = AblationConfig(
        type="lemmatize_verbs",
        input_path=childes_corpus,
        output_path=out_path,
        annotated_input_path=docbin_dir,
        spacy_model="en_core_web_sm",
        skip_validation=True,
        chunk_size=100,
    )
    AblationPipeline(cfg).process_corpus()
    out_text = (out_path / "childes.train").read_text()

    # Boundary markers preserved verbatim.
    assert "= = = childes/Eng-NA/Bloom73/Peter/01.cha = = =" in out_text
    # No speaker prefixes survive.
    assert "*CHI:" not in out_text, (
        f"Ablation output should not contain *CHI:\n{out_text}"
    )
    assert "*MOT:" not in out_text, (
        f"Ablation output should not contain *MOT:\n{out_text}"
    )
    # Content is lemmatized (e.g. "is" -> "be", "run" stays "run").
    # Just confirm a known lemmatization landed; exact output depends on the
    # small model's lemmatizer but "is -> be" is reliable.
    assert " be " in out_text.lower()
