#!/usr/bin/env python3
"""Benchmark harness for eval_v2: per-phase wall-time + GPU memory.

Runs the same per-model pipeline as `eval_v2_smoke.py` but adds
instrumentation so we can answer:

  - How long does each phase take?  (startup, first-build,
    per-checkpoint-load, per-checkpoint-forward, write, cache re-run)
  - What's the peak GPU memory per phase?
  - How many concurrent evaluations could fit on one GPU?

Auto-discovers all CSV files under a stimuli directory tree so we test
with the realistic stimulus volume (~5,500 items across 19 CSVs) rather
than the 576-item subject_drop-only subset in the smoke test.

Example:

    python scripts/eval_v2_bench.py \\
        --checkpoint_root /mnt/data/models/exp0_baseline_90M_smoke_resume \\
        --tokenizer_dir /mnt/data/tokenizers/exp0_baseline_90M_smoke \\
        --stimuli_dir evaluation/stimuli/null-subj-v2/staging \\
        --output_root /mnt/data/eval_v2/bench \\
        --cell_id bench_90M_all_stimuli \\
        --batch_size 16 \\
        --device auto

All timings are printed as `[BENCH] phase=NAME secs=X.YZ` lines so they
can be grepped out of the log.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def timer(log, phase: str):
    """Log `[BENCH] phase=NAME secs=X.YZ` on exit."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log.info("[BENCH] phase=%s secs=%.3f", phase, dt)


def log_cuda_memory(log, label: str):
    """Log peak + current CUDA memory and reset peak counter."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        cur_mb = torch.cuda.memory_allocated() / 1024 ** 2
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
        log.info(
            "[BENCH] mem label=%s cur_MB=%.1f peak_MB=%.1f reserved_MB=%.1f",
            label, cur_mb, peak_mb, reserved_mb,
        )
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        log.debug("mem log failed: %s", e)


def log_nvidia_smi(log, label: str):
    """Shell out to nvidia-smi for utilization + memory snapshot."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True, timeout=5,
        ).strip()
        # "1234, 24576, 45, 22"
        used, total, util_gpu, util_mem = [p.strip() for p in out.split(",")]
        log.info(
            "[BENCH] nvsmi label=%s used_MB=%s total_MB=%s util_gpu_pct=%s util_mem_pct=%s",
            label, used, total, util_gpu, util_mem,
        )
    except Exception as e:
        log.debug("nvidia-smi failed: %s", e)


def discover_stimuli(stimuli_dir: Path) -> list[Path]:
    """Walk `{stimuli_dir}/{lang}/*.csv` and return all CSV paths."""
    out = []
    for lang_dir in sorted(stimuli_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        for csv in sorted(lang_dir.glob("*.csv")):
            out.append(csv)
    return out


def main():
    p = argparse.ArgumentParser(description="eval_v2 benchmark harness")
    p.add_argument("--checkpoint_root", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--stimuli_dir", required=True,
                   help="Root containing {en,es}/*.csv")
    p.add_argument("--output_root", required=True)
    p.add_argument("--cell_id", required=True)
    p.add_argument("--architecture", default="gpt2")
    p.add_argument("--intervention", default="baseline")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--scoring_version", default="bench-v1")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    log = logging.getLogger("eval_v2_bench")
    t_start = time.perf_counter()

    # ── Imports ─────────────────────────────────────────────────────────
    with timer(log, "imports"):
        import duckdb  # noqa: F401
        import sentencepiece as spm
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        from evaluation.cache import CachedRunner
        from evaluation.output_v2 import register_duckdb_views
        from evaluation.runners.per_model_runner import (
            CellSpec,
            find_checkpoints_sorted,
        )
        from evaluation.stimuli_cache import build_or_load_cache

    # ── Device ──────────────────────────────────────────────────────────
    device = ("cuda" if torch.cuda.is_available() else "cpu"
              if args.device == "auto" else args.device)
    log.info("Device: %s", device)
    if device == "cuda":
        log.info("CUDA count=%d name=%s",
                 torch.cuda.device_count(), torch.cuda.get_device_name(0))
        log_nvidia_smi(log, "pre_anything")

    # ── Paths ───────────────────────────────────────────────────────────
    ckpt_root = Path(args.checkpoint_root)
    tokenizer_dir = Path(args.tokenizer_dir)
    stimuli_dir = Path(args.stimuli_dir)
    output_root = Path(args.output_root)

    stimuli_paths = discover_stimuli(stimuli_dir)
    log.info("Discovered %d stimulus CSVs under %s",
             len(stimuli_paths), stimuli_dir)
    for sp in stimuli_paths:
        log.info("  %s", sp.relative_to(stimuli_dir))

    tokenizer_file = tokenizer_dir / "tokenizer.model"

    # ── Checkpoint discovery ────────────────────────────────────────────
    with timer(log, "discover_checkpoints"):
        ckpts = find_checkpoints_sorted(ckpt_root)
    log.info("Found %d checkpoint(s)", len(ckpts))
    if not ckpts:
        sys.exit(3)

    # ── Tokenizer ───────────────────────────────────────────────────────
    with timer(log, "load_tokenizer"):
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(tokenizer_file))
    log.info("  vocab_size = %d", tokenizer.get_piece_size())

    # ── Stimuli cache ───────────────────────────────────────────────────
    with timer(log, "build_stimuli_cache"):
        stimuli = build_or_load_cache(
            stimuli_paths, tokenizer, tokenizer_file,
            output_root / "stimuli_cache",
        )
    log.info("  n_rows=%d langs=%s cats=%s",
             len(stimuli),
             stimuli.meta.get("languages"),
             stimuli.meta.get("categories"))
    # Per-language counts
    per_lang = {}
    for e in stimuli.rows:
        per_lang[e.language] = per_lang.get(e.language, 0) + 1
    for lang, n in sorted(per_lang.items()):
        log.info("  lang=%s n=%d", lang, n)

    # ── Model factory ───────────────────────────────────────────────────
    first_ckpt = ckpts[0][1]
    with timer(log, "load_hf_config"):
        hf_config = AutoConfig.from_pretrained(str(first_ckpt))
    log.info("  model_type=%s", getattr(hf_config, "model_type", "<unknown>"))

    factory_calls = {"n": 0}

    def model_factory():
        factory_calls["n"] += 1
        with timer(log, f"model_factory_call_{factory_calls['n']}"):
            m = AutoModelForCausalLM.from_config(hf_config)
        return m

    # ── Build CellSpec ──────────────────────────────────────────────────
    cell = CellSpec(
        cell_id=args.cell_id,
        architecture=args.architecture,
        intervention=args.intervention,
        rep=args.rep,
        checkpoint_root=ckpt_root,
        model_factory=model_factory,
        stimuli=stimuli,
        unigram=None,
        device=device,
        batch_size=args.batch_size,
    )

    # ── Baseline memory snapshot ────────────────────────────────────────
    log_cuda_memory(log, "before_runner")
    log_nvidia_smi(log, "before_runner")

    # ── RUN 1: full evaluation of all checkpoints ───────────────────────
    log.info("=" * 60)
    log.info("RUN 1: evaluating %d checkpoint(s) over %d items",
             len(ckpts), len(stimuli))
    log.info("=" * 60)
    runner = CachedRunner(
        cell, output_root=output_root, scoring_version=args.scoring_version,
    )
    # We need to measure per-checkpoint cost, so we run manually rather
    # than using runner.run_once() — identical code path, just timed per
    # iteration.
    from evaluation.cache import checkpoint_id, is_cached, mark_cached
    from evaluation.runners.per_model_runner import PerModelRunner

    all_items, all_pairs = [], []
    processed = []
    with timer(log, "run1_total"):
        inner = None
        for i, (step, path) in enumerate(ckpts):
            cid = checkpoint_id(path)
            key = runner._make_key(cid)
            if is_cached(output_root, key):
                log.info("  checkpoint-%d cached (skip)", step)
                continue
            if inner is None:
                inner = PerModelRunner(cell)
            with timer(log, f"ckpt_{step}_load"):
                inner.load_checkpoint(path)
            log_cuda_memory(log, f"after_load_ckpt_{step}")
            with timer(log, f"ckpt_{step}_forward+pair+buffer"):
                items, pairs = inner.evaluate_checkpoint(step, path)
            log_cuda_memory(log, f"after_forward_ckpt_{step}")
            log_nvidia_smi(log, f"after_forward_ckpt_{step}")
            all_items.extend(items)
            all_pairs.extend(pairs)
            processed.append((step, path))
            log.info("  ckpt-%d: %d items + %d pairs", step, len(items), len(pairs))

    # Persist + mark
    if all_items or all_pairs:
        with timer(log, "write_parquet"):
            from evaluation.output_v2 import write_cell_results
            write_summary = write_cell_results(
                output_root=output_root,
                cell_id=cell.cell_id,
                item_results=all_items,
                pair_results=all_pairs,
            )
            log.info("  wrote %d item rows, %d pair rows",
                     write_summary["n_item_rows"], write_summary["n_pair_rows"])
        for step, path in processed:
            mark_cached(output_root, runner._make_key(checkpoint_id(path)))

    # ── RUN 2: cache hits ───────────────────────────────────────────────
    log.info("=" * 60)
    log.info("RUN 2: cache invariant")
    log.info("=" * 60)
    with timer(log, "run2_total"):
        runner_2 = CachedRunner(
            cell, output_root=output_root,
            scoring_version=args.scoring_version,
        )
        s2 = runner_2.run_once()
    assert s2["n_forward_passes"] == 0, (
        f"Cache invariant violated: {s2['n_forward_passes']} fwd on re-run"
    )

    # ── Final projection ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PROJECTIONS")
    log.info("=" * 60)
    total = time.perf_counter() - t_start
    log.info("Total wall-time: %.1fs", total)
    log.info("Model factory calls: %d (expected 1 for full cell)",
             factory_calls["n"])
    log.info("Checkpoints processed: %d", len(processed))
    if len(processed) > 0:
        log.info("Per-checkpoint avg (load+forward+pair+buffer, excludes "
                 "startup): logged as ckpt_*_load + ckpt_*_forward above")
    if device == "cuda":
        try:
            import torch
            total_gpu_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            log.info("Peak alloc: %.1f MB / %.1f MB total (%.1f%%)",
                     peak, total_gpu_mb, 100 * peak / total_gpu_mb)
            if peak > 0:
                log.info("Naive packing estimate (peak alloc basis): "
                         "~%d concurrent evals per GPU",
                         int(total_gpu_mb // peak))
        except Exception as e:
            log.warning("final mem stats failed: %s", e)

    log.info("=" * 60)
    log.info("✅ BENCH COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
