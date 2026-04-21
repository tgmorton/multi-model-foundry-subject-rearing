#!/usr/bin/env python3
"""eval_v2 end-to-end smoke test against a real checkpoint.

Runs the full Wave-1-to-Wave-4 pipeline (D1-D14) against a real trained
checkpoint on disk. Designed to be run on the NRP cluster with the
corpus-analysis-data PVC mounted at /mnt/data.

Usage:
    python scripts/eval_v2_smoke.py \
        --checkpoint_root /mnt/data/models/exp0_baseline_90M_smoke \
        --tokenizer_dir /mnt/data/tokenizers/exp0_baseline_90M_smoke \
        --stimuli evaluation/stimuli/null-subj-v2/staging/en/subject_drop.csv \
                   evaluation/stimuli/null-subj-v2/staging/es/subject_drop.csv \
        --output_root /mnt/data/eval_v2/smoke \
        --cell_id smoke_exp0_baseline \
        --architecture gpt2 \
        --intervention baseline \
        --batch_size 16

What this exercises:
  D1 (v2 schema), D2 (MeanLP), D4 (SLOR; only if --unigram), D5 (per-token
  log-probs), D6 (stimuli cache), D7 (batched forward), D9 (per-model
  runner, state_dict swap across all discovered checkpoints), D10 (parquet
  + DuckDB), D11 (cache idempotency — see --rerun flag).

Exit code 0 = success. All invariants asserted on output (no NaN/Inf,
shape matches stimuli × 2 × n_checkpoints, DuckDB views queryable).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="eval_v2 smoke test")
    parser.add_argument("--checkpoint_root", required=True,
                        help="Directory containing checkpoint-* subdirs.")
    parser.add_argument("--tokenizer_dir", required=True,
                        help="Directory containing tokenizer.model (SentencePiece).")
    parser.add_argument("--stimuli", nargs="+", required=True,
                        help="One or more v2 stimuli CSV paths.")
    parser.add_argument("--output_root", required=True,
                        help="Output directory for parquet artifacts.")
    parser.add_argument("--cell_id", required=True,
                        help="Identifier for this cell (used in parquet paths).")
    parser.add_argument("--architecture", default="gpt2",
                        help="Architecture tag for provenance (default: gpt2).")
    parser.add_argument("--intervention", default="baseline",
                        help="Intervention tag for provenance.")
    parser.add_argument("--rep", type=int, default=0,
                        help="Replication index.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--unigram",
                        help="Optional path to a UnigramTable pickle for SLOR.")
    parser.add_argument("--scoring_version", default="smoke-v1",
                        help="Scoring version tag for the cache key.")
    parser.add_argument("--scratch_dir",
                        help="Pod-local dir for staged parquet writes. "
                             "Writes land here first then bulk-copy to "
                             "output_root. On CephFS: ~30× faster.")
    parser.add_argument("--rerun", action="store_true",
                        help="Force rerun (ignore existing cache markers).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    log = logging.getLogger("eval_v2_smoke")

    # ── Imports (deferred so --help stays fast) ─────────────────────────
    import duckdb
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
    from evaluation.unigram import UnigramTable

    # ── Resolve device ──────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    log.info("Device: %s", device)
    if device == "cuda":
        log.info("CUDA available: %s; device count: %d",
                 torch.cuda.is_available(), torch.cuda.device_count())

    # ── Paths ───────────────────────────────────────────────────────────
    ckpt_root = Path(args.checkpoint_root)
    tokenizer_dir = Path(args.tokenizer_dir)
    output_root = Path(args.output_root)
    stimuli_paths = [Path(p) for p in args.stimuli]

    for p in [ckpt_root, tokenizer_dir] + stimuli_paths:
        if not p.exists():
            log.error("Path does not exist: %s", p)
            sys.exit(2)

    tokenizer_file = tokenizer_dir / "tokenizer.model"
    if not tokenizer_file.exists():
        log.error("tokenizer.model not found in %s", tokenizer_dir)
        sys.exit(2)

    # ── Checkpoint discovery ────────────────────────────────────────────
    ckpts = find_checkpoints_sorted(ckpt_root)
    if not ckpts:
        log.error("No checkpoint-* directories under %s", ckpt_root)
        sys.exit(3)
    log.info("Found %d checkpoint(s):", len(ckpts))
    for step, p in ckpts:
        log.info("  checkpoint-%d at %s", step, p)

    # ── Load tokenizer ──────────────────────────────────────────────────
    log.info("Loading tokenizer from %s", tokenizer_file)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_file))
    log.info("  vocab_size = %d", tokenizer.get_piece_size())

    # ── Build stimuli cache (D6) ────────────────────────────────────────
    log.info("Building / loading stimuli cache (%d CSV files)", len(stimuli_paths))
    cache_dir = output_root / "stimuli_cache"
    stimuli = build_or_load_cache(
        stimuli_paths, tokenizer, tokenizer_file, cache_dir,
    )
    log.info("  stimuli_id=%s tokenizer_id=%s n_rows=%d languages=%s categories=%s",
             stimuli.stimuli_id, stimuli.tokenizer_id, len(stimuli),
             stimuli.meta.get("languages"), stimuli.meta.get("categories"))

    # ── Optional unigram baseline (D3 → D4 SLOR) ────────────────────────
    unigram = None
    if args.unigram:
        unigram_path = Path(args.unigram)
        if not unigram_path.exists():
            log.error("Unigram pickle not found: %s", unigram_path)
            sys.exit(2)
        log.info("Loading unigram baseline from %s", unigram_path)
        unigram = UnigramTable.load(unigram_path)
        log.info("  corpus_id=%s V=%d N=%d α=%s",
                 unigram.corpus_id, unigram.vocab_size,
                 unigram.total_tokens, unigram.smoothing_alpha)

    # ── Model factory (load-once pattern) ───────────────────────────────
    # Read the HF config from the first checkpoint so we can instantiate
    # the architecture with fresh weights. The runner then swaps in each
    # checkpoint's state_dict.
    first_ckpt = ckpts[0][1]
    log.info("Reading model config from %s", first_ckpt)
    hf_config = AutoConfig.from_pretrained(str(first_ckpt))
    log.info("  model_type=%s", getattr(hf_config, "model_type", "<unknown>"))

    def model_factory():
        # Fresh weights via from_config (not from_pretrained — that would
        # eagerly load weights, defeating the state_dict-swap pattern).
        m = AutoModelForCausalLM.from_config(hf_config)
        log.debug("  factory: instantiated %s", type(m).__name__)
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
        unigram=unigram,
        device=device,
        batch_size=args.batch_size,
    )

    # ── Run (with cache) ────────────────────────────────────────────────
    if args.rerun:
        # Clear any existing markers for this (cell, scoring_version)
        marker_dir = output_root / ".cache"
        if marker_dir.exists():
            for m in marker_dir.glob(f"{args.cell_id}__*__scoring={args.scoring_version}.done"):
                m.unlink()
                log.info("Removed stale marker: %s", m.name)

    log.info("=" * 60)
    log.info("RUN 1: initial evaluation")
    log.info("=" * 60)
    runner = CachedRunner(
        cell, output_root=output_root, scoring_version=args.scoring_version,
        scratch_dir=Path(args.scratch_dir) if args.scratch_dir else None,
    )
    summary_1 = runner.run_once()
    _log_summary(log, "RUN 1", summary_1)

    # ── Idempotency check (D11) ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("RUN 2: should be zero forward passes (cache hits)")
    log.info("=" * 60)
    runner_2 = CachedRunner(
        cell, output_root=output_root, scoring_version=args.scoring_version,
        scratch_dir=Path(args.scratch_dir) if args.scratch_dir else None,
    )
    summary_2 = runner_2.run_once()
    _log_summary(log, "RUN 2", summary_2)
    assert summary_2["n_forward_passes"] == 0, (
        f"D11 cache invariant violated: second run did "
        f"{summary_2['n_forward_passes']} forward passes (expected 0)"
    )
    log.info("✓ D11 cache invariant holds: 0 forward passes on re-run.")

    # ── DuckDB sanity queries (D10) ─────────────────────────────────────
    log.info("=" * 60)
    log.info("DuckDB sanity")
    log.info("=" * 60)
    con = duckdb.connect()
    try:
        register_duckdb_views(con, output_root)

        n_items = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        n_pairs = con.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
        log.info("Rows written: items=%d  pairs=%d", n_items, n_pairs)

        # Expected row count: n_rows × n_checkpoints
        expected_items = len(stimuli) * len(ckpts)
        if n_items != expected_items:
            log.warning("Expected %d item rows (stimuli=%d × ckpts=%d); got %d",
                        expected_items, len(stimuli), len(ckpts), n_items)

        # NaN/Inf check
        n_bad_items = con.execute("""
            SELECT COUNT(*) FROM items
            WHERE NOT isfinite(target_mean_log_prob)
               OR NOT isfinite(target_sum_log_prob)
        """).fetchone()[0]
        if n_bad_items:
            log.error("%d item rows have non-finite metrics!", n_bad_items)
            sys.exit(4)
        log.info("✓ No NaN/Inf in item metrics.")

        # Per-language row counts
        lang_counts = con.execute(
            "SELECT language, COUNT(*) FROM items GROUP BY language ORDER BY language"
        ).fetchall()
        for lang, n in lang_counts:
            log.info("  language=%s: %d item rows", lang, n)

        # Per-checkpoint summary
        log.info("Per-checkpoint overt-preference rate (MeanLP):")
        ckpt_summary = con.execute("""
            SELECT checkpoint_step,
                   language,
                   AVG(CASE WHEN prefers_overt_meanlp THEN 1.0 ELSE 0.0 END) AS overt_pref,
                   AVG(log_prob_diff_overt_minus_null) AS mean_diff,
                   COUNT(*) AS n_pairs
            FROM pairs
            GROUP BY checkpoint_step, language
            ORDER BY checkpoint_step, language
        """).fetchall()
        for step, lang, pref, diff, n in ckpt_summary:
            log.info("  step=%d lang=%s  overt_pref=%.3f  mean_logprob_diff=%+.4f  n=%d",
                     step, lang, pref, diff, n)

        # Per-token table (D5)
        try:
            n_per_tok = con.execute(
                "SELECT COUNT(*) FROM per_token"
            ).fetchone()[0]
            log.info("Per-token rows (D5 MORCELA payload): %d", n_per_tok)
        except Exception:
            log.warning("per_token view not registered (no per-token parquet?)")

    finally:
        con.close()

    log.info("=" * 60)
    log.info("✅ SMOKE PASS")
    log.info("=" * 60)


def _log_summary(log, label: str, summary: dict) -> None:
    log.info("%s summary:", label)
    log.info("  n_checkpoints_total: %d", summary["n_checkpoints_total"])
    log.info("  n_cached:            %d", summary["n_cached"])
    log.info("  n_processed:         %d", summary["n_processed"])
    log.info("  n_forward_passes:    %d", summary["n_forward_passes"])
    if summary["processed_steps"]:
        log.info("  processed steps:     %s", summary["processed_steps"])
    if summary["cached_steps"]:
        log.info("  cached steps:        %s", summary["cached_steps"])
    if summary.get("write_summary"):
        ws = summary["write_summary"]
        log.info("  wrote %d item rows, %d pair rows",
                 ws["n_item_rows"], ws["n_pair_rows"])


if __name__ == "__main__":
    main()
