"""
Smoke test: run corpus descriptive analysis (with DocBin cache emission)
over a small real Spanish corpus, then run every registered Spanish
ablation from that cache. Compares cached-vs-live for correctness and
reports timing so we can see the cache actually pays off.

Run: .venv/bin/python scripts/smoke_analysis_plus_ablations.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spacy  # noqa: E402

from analysis.corpus_descriptives.annotators import get_default_annotators  # noqa: E402
from analysis.corpus_descriptives.config import CorpusAnalysisConfig  # noqa: E402
from analysis.corpus_descriptives.pipeline import CorpusAnnotationPipeline  # noqa: E402
import preprocessing.ablations  # noqa: F401, E402  — triggers registration
from preprocessing.base import AblationPipeline  # noqa: E402
from preprocessing.config import AblationConfig  # noqa: E402


CORPUS = Path("/tmp/smoke_corpus")
PULL = Path("/tmp/smoke_pull")
OUT = Path("/tmp/smoke_output")

ABLATIONS = [
    "remove_expletive_sentences_es",
    "impoverish_case_es",
    "lemmatize_verbs",
]

BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"
END = "\033[0m"


def _hdr(text: str) -> None:
    print(f"\n{BOLD}{'=' * 72}{END}")
    print(f"{BOLD}  {text}{END}")
    print(f"{BOLD}{'=' * 72}{END}")


def run_unified_annotation() -> tuple[float, Path, dict]:
    """Stage 1: corpus analysis + DocBin emission in one parse pass."""
    _hdr("Stage 1: Corpus analysis + DocBin emission")
    t0 = time.time()

    cfg = CorpusAnalysisConfig(
        input_path=CORPUS,
        output_path=OUT / "analysis",
        split_name="smoke_test",
        spacy_model="es_core_news_lg",
        language="es",
        genre_map={"childes": "CHILDES", "vikidia": "Vikidia"},
        annotation_mode=True,
        layered_output=True,
        emit_docbin=True,
    )
    # Keep annotators modest so the Parquet stage doesn't dominate.
    annotators = [
        a for a in get_default_annotators(language="es")
        if a.name in {"ClauseStructureAnnotator", "PronounAnnotator", "VerbAnnotator"}
    ]
    print(f"  Spanish annotators: {[a.name for a in annotators]}")
    print(f"  Corpus: {CORPUS} ({sum(1 for _ in CORPUS.glob('*.train'))} files)")
    metadata = CorpusAnnotationPipeline(
        cfg, annotators=annotators, layered=True,
    ).run()
    elapsed = time.time() - t0

    docbin_dir = Path(metadata["docbin_output_dir"])
    print(f"\n  {GREEN}✓{END} analysis done in {elapsed:.2f}s")
    print(f"  {GREEN}✓{END} Parquet: {cfg.output_path / 'annotated_corpus'}")
    print(f"  {GREEN}✓{END} DocBin:  {docbin_dir}")
    print(f"  {GREEN}✓{END} Total sentences: {metadata['total_sentences']:,}")

    return elapsed, docbin_dir, metadata


def run_ablation(ablation: str, *, cached: bool, docbin_dir: Path | None) -> tuple[float, dict]:
    """Run one ablation; return (elapsed, manifest dict)."""
    mode = "cached" if cached else "live"
    output_path = OUT / f"ablated_{mode}" / ablation
    cfg = AblationConfig(
        type=ablation,
        input_path=CORPUS,
        output_path=output_path,
        annotated_input_path=docbin_dir if cached else None,
        replacement_pool_dir=PULL,
        spacy_model="es_core_news_lg",
        skip_validation=True,  # validator re-parses; orthogonal to this test
        chunk_size=500,
    )
    t0 = time.time()
    manifest = AblationPipeline(cfg).process_corpus()
    elapsed = time.time() - t0
    return elapsed, manifest.metadata.model_dump()


def _compare_outputs(ablation: str) -> bool:
    """Byte-compare cached vs live output files."""
    live_dir = OUT / "ablated_live" / ablation
    cached_dir = OUT / "ablated_cached" / ablation
    live_files = sorted(p.name for p in live_dir.glob("*.train"))
    cached_files = sorted(p.name for p in cached_dir.glob("*.train"))
    if live_files != cached_files:
        return False
    for name in live_files:
        if (live_dir / name).read_text() != (cached_dir / name).read_text():
            return False
    return True


def main() -> int:
    # Clean previous run
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)

    # Stage 1
    t_analysis, docbin_dir, _ = run_unified_annotation()

    # Stage 2a: live parse (baseline — each ablation re-parses)
    _hdr("Stage 2a: Ablations (LIVE parse — baseline)")
    live_times: dict[str, float] = {}
    live_manifests: dict[str, dict] = {}
    for ab in ABLATIONS:
        t, m = run_ablation(ab, cached=False, docbin_dir=None)
        live_times[ab] = t
        live_manifests[ab] = m
        print(f"  {ab:38s} {t:6.2f}s  items={m['total_items_ablated']:>5}")
    t_live_total = sum(live_times.values())

    # Stage 2b: cached (read DocBin, skip parse)
    _hdr("Stage 2b: Ablations (CACHED — read DocBin, skip parse)")
    cached_times: dict[str, float] = {}
    cached_manifests: dict[str, dict] = {}
    for ab in ABLATIONS:
        t, m = run_ablation(ab, cached=True, docbin_dir=docbin_dir)
        cached_times[ab] = t
        cached_manifests[ab] = m
        print(f"  {ab:38s} {t:6.2f}s  items={m['total_items_ablated']:>5}")
    t_cached_total = sum(cached_times.values())

    # Parity check
    _hdr("Parity: cached vs live (byte-identical output?)")
    all_ok = True
    for ab in ABLATIONS:
        ok = _compare_outputs(ab)
        marker = f"{GREEN}✓ identical{END}" if ok else f"{RED}✗ MISMATCH{END}"
        print(f"  {ab:38s} {marker}")
        all_ok = all_ok and ok

    # Tier breakdowns (from cached runs — same as live)
    _hdr("Tier breakdowns (aggregate across corpus)")
    for ab in ABLATIONS:
        m = cached_manifests[ab]
        tiers = m.get("aggregate_tier_counts", {})
        total = m["total_items_ablated"]
        print(f"\n  {CYAN}{ab}{END}  (total items ablated: {total:,})")
        if tiers:
            for tier, count in sorted(tiers.items(), key=lambda x: -x[1]):
                share = count / total if total else 0.0
                bar = "█" * int(share * 30)
                print(f"    {tier:20s} {count:>6} {bar} {share * 100:.1f}%")
        else:
            print(f"    {DIM}(no tier metadata for this ablation){END}")

    # Timing summary
    _hdr("Timing summary")
    print(f"  Stage 1 (analysis + docbin):  {t_analysis:6.2f}s")
    print(f"  Stage 2 live ({len(ABLATIONS)} ablations): {t_live_total:6.2f}s  "
          f"{DIM}(each re-parses the corpus){END}")
    print(f"  Stage 2 cached ({len(ABLATIONS)} ablations): {t_cached_total:6.2f}s  "
          f"{DIM}(reads DocBin, no parse){END}")
    print()
    print(f"  Live total       (N parses):           {t_live_total:6.2f}s")
    print(f"  Unified total    (1 analysis + cached): "
          f"{t_analysis + t_cached_total:6.2f}s")
    savings = t_live_total - (t_analysis + t_cached_total)
    if savings > 0:
        print(f"  {GREEN}Savings: {savings:.2f}s "
              f"({savings / t_live_total * 100:.1f}%){END}")
    else:
        print(f"  {DIM}(small corpus — model-load overhead dominates; "
              f"cache wins on large corpora){END}")

    if all_ok:
        print(f"\n  {GREEN}{BOLD}SMOKE TEST: PASSED{END}")
        print(f"  {DIM}All 3 ablations produce byte-identical output from "
              f"the unified DocBin cache as they do from live parsing.{END}\n")
        return 0
    else:
        print(f"\n  {RED}{BOLD}SMOKE TEST: FAILED{END}  (parity mismatch)\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
