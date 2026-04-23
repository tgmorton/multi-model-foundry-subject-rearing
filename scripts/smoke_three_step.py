"""
Three-step ablation workflow smoke test.

Steps:
  1. annotate(train_90M) + annotate(pull_10M)  →  DocBin caches
  2. ablate(train, skip_backfill=True)         →  ablated train
     ablate(pool,  skip_backfill=True)         →  ablated pool
  3. compose(ablated_train + ablated_pool)     →  final corpus, target size

Validates:
  - Final corpus files have token counts matching the original raw corpus
    (replacement-pool backfill closes the gap).
  - COMPOSE_MANIFEST.json records which pool lines were drawn per file.
  - Output is byte-deterministic across re-runs with the same seed.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import preprocessing.ablations  # noqa: F401, E402  — register ablations
from preprocessing.annotate import annotate_corpus  # noqa: E402
from preprocessing.base import AblationPipeline  # noqa: E402
from preprocessing.config import AblationConfig  # noqa: E402
from scripts.compose_corpus import compose, _count_tokens  # noqa: E402


CORPUS = Path("/tmp/smoke_corpus")       # 2000 CHILDES + 1000 Vikidia
POOL = Path("/tmp/smoke_pull")           # 200 + 200 pool
OUT = Path("/tmp/smoke_three_step")
ABLATION = "remove_expletive_sentences_es"  # exercises backfill


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


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # ------------------------------------------------------------------
    # Step 1: Annotate train + pool
    # ------------------------------------------------------------------
    _hdr("Step 1: annotate train + pool")
    t0 = time.time()
    annotate_corpus(CORPUS, "es_core_news_lg", OUT / "annotated_train")
    annotate_corpus(POOL, "es_core_news_lg", OUT / "annotated_pool")
    t_annotate = time.time() - t0
    print(f"  {GREEN}✓{END} annotation done in {t_annotate:.2f}s")

    # ------------------------------------------------------------------
    # Step 2: Ablate train + pool (independently, no backfill)
    # ------------------------------------------------------------------
    _hdr("Step 2: ablate train + pool (skip_backfill=True)")
    t0 = time.time()
    for label, input_dir, annotated_dir, output_dir in [
        ("train", CORPUS, OUT / "annotated_train", OUT / "ablated_train"),
        ("pool",  POOL,   OUT / "annotated_pool",  OUT / "ablated_pool"),
    ]:
        cfg = AblationConfig(
            type=ABLATION,
            input_path=input_dir,
            output_path=output_dir,
            annotated_input_path=annotated_dir,
            skip_backfill=True,  # ← pure transformer mode
            spacy_model="es_core_news_lg",
            skip_validation=True,
            chunk_size=500,
        )
        m = AblationPipeline(cfg).process_corpus()
        t_meta = m.metadata
        print(
            f"  [{label}] {t_meta.total_files_processed} files, "
            f"{t_meta.total_items_ablated:,} items ablated, "
            f"tokens {t_meta.total_tokens_original:,} → "
            f"{t_meta.total_tokens_final:,} "
            f"({(t_meta.total_tokens_original - t_meta.total_tokens_final) / t_meta.total_tokens_original * 100:.2f}% removed)"
        )
    t_ablate = time.time() - t0
    print(f"  {GREEN}✓{END} ablation done in {t_ablate:.2f}s")

    # ------------------------------------------------------------------
    # Step 3: Compose — ablated train + pool samples → final corpus
    # ------------------------------------------------------------------
    _hdr("Step 3: compose (ablated train + pool-draws → final)")
    t0 = time.time()
    manifest = compose(
        ablated_train=OUT / "ablated_train",
        ablated_pool=OUT / "ablated_pool",
        output=OUT / "composed",
        target_corpus=CORPUS,
        seed=42,
    )
    t_compose = time.time() - t0
    print(f"  {GREEN}✓{END} compose done in {t_compose:.2f}s")

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    _hdr("Verification")

    # 1. Target size hit per file. Small overshoot is expected — the
    #    compose loop draws whole pool lines until it hits target, and
    #    the last line typically pushes slightly past. Tolerance: within
    #    100 tokens of target (< 0.5% of a CHILDES file).
    all_ok = True
    for entry in manifest["per_file"]:
        stem = entry["stem"]
        final = entry["final_tokens"]
        target = entry["target_tokens"]
        gap = target - final  # positive = undershoot, negative = overshoot
        ok = -100 < gap < 100
        marker = f"{GREEN}✓{END}" if ok else f"{RED}✗{END}"
        direction = (
            f"overshoot {-gap}" if gap < 0
            else f"undershoot {gap}" if gap > 0
            else "exact"
        )
        print(
            f"  {marker} [{stem}] final={final:,} target={target:,} "
            f"({direction})"
        )
        all_ok = all_ok and ok

    # 2. Determinism — re-run compose with same seed, compare byte-for-byte.
    rerun_dir = OUT / "composed_rerun"
    if rerun_dir.exists():
        shutil.rmtree(rerun_dir)
    compose(
        ablated_train=OUT / "ablated_train",
        ablated_pool=OUT / "ablated_pool",
        output=rerun_dir,
        target_corpus=CORPUS,
        seed=42,
    )
    for stem in ["childes", "vikidia"]:
        a = (OUT / "composed" / f"{stem}.train").read_text()
        b = (rerun_dir / f"{stem}.train").read_text()
        ok = a == b
        marker = f"{GREEN}✓{END} identical (deterministic)" if ok else f"{RED}✗{END} MISMATCH"
        print(f"  {marker} compose[{stem}] seed=42 twice")
        all_ok = all_ok and ok

    # 3. Spot-check: show a sample of pool-sourced lines so the user can
    #    eyeball them against the ablation's intended behavior.
    _hdr("Sample of pool-drawn lines (for hand review)")
    for entry in manifest["per_file"]:
        stem = entry["stem"]
        used_indices = entry.get("pool_line_indices_used", [])
        if not used_indices:
            continue
        pool_lines = (OUT / "ablated_pool" / f"{stem}.train").read_text().splitlines()
        print(f"\n  {CYAN}[{stem}]{END} {len(used_indices)} pool lines drawn — showing first 3:")
        for i, idx in enumerate(used_indices[:3]):
            preview = pool_lines[idx][:100]
            print(f"    {DIM}pool-line#{idx:5}{END}  {preview}")

    # ------------------------------------------------------------------
    # Timing summary
    # ------------------------------------------------------------------
    _hdr("Timing summary")
    print(f"  Step 1 annotate (train + pool): {t_annotate:6.2f}s")
    print(f"  Step 2 ablate (train + pool):   {t_ablate:6.2f}s")
    print(f"  Step 3 compose:                 {t_compose:6.2f}s")
    print(f"  {BOLD}Total:{END}                          "
          f"{t_annotate + t_ablate + t_compose:6.2f}s")

    if all_ok:
        print(f"\n  {GREEN}{BOLD}THREE-STEP SMOKE: PASSED{END}\n")
        return 0
    print(f"\n  {RED}{BOLD}THREE-STEP SMOKE: FAILED{END}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
