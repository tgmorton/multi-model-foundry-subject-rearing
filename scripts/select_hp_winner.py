#!/usr/bin/env python3
"""
Select the HP-sweep winner for an (arch, lang) pair.

After a sweep converges (or its budget is exhausted), pull all the
sweep records from the registry, rank by hp_proxy_score (lower is
better for perplexity), and mark the top trial as is_hp_winner=True
with hp_sweep_rank=1. Second place gets rank=2, etc.

Usage:
    # Rank one arch × lang
    python scripts/select_hp_winner.py --arch gpt2_medium --lang en

    # Dry run (show what would be marked, don't write)
    python scripts/select_hp_winner.py --arch gpt2_medium --lang en --dry-run

    # Pick the winner across ALL archs/langs that have sweeps
    python scripts/select_hp_winner.py --all

Reads the registry's by_run/ shard directly. The run_registry/
registry.parquet materialized view may be stale; this script always
queries the source of truth.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_foundry import registry as _registry

logger = logging.getLogger("select_hp_winner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def _trials_for_cell(arch: str, lang: str) -> list[dict[str, Any]]:
    """Fetch all hp_sweep trials for an (arch, lang) cell with a
    non-null hp_proxy_score. The registry's iter_all_records lists
    everything; we filter in-Python since the dataset is small (~360
    sweep records at peak)."""
    records = []
    for rec in _registry.iter_all_records():
        if rec.get("run_kind") != "hp_sweep":
            continue
        if rec.get("arch") != arch or rec.get("lang") != lang:
            continue
        if rec.get("status") != "COMPLETE":
            continue
        if rec.get("hp_proxy_score") is None:
            continue
        records.append(rec)
    return records


def _rank_and_mark(
    arch: str, lang: str, condition: str,
    trials: list[dict[str, Any]], dry_run: bool,
) -> dict[str, Any] | None:
    if not trials:
        logger.warning("no scorable sweep trials for %s × %s", arch, lang)
        return None

    # Lower perplexity = better. Secondary tiebreaker: final_training_loss.
    def _key(r):
        return (r["hp_proxy_score"],
                r.get("final_loss") or float("inf"))

    ordered = sorted(trials, key=_key)
    winner = ordered[0]
    logger.info(
        "%s × %s — %d trials; winner=%s score=%.4f",
        arch, lang, len(ordered),
        winner["run_id"], winner["hp_proxy_score"],
    )

    if dry_run:
        logger.info("[dry-run] would mark %s as winner (rank=1)", winner["run_id"])
        for i, t in enumerate(ordered[:5], start=1):
            logger.info("  rank %d: %s ppl=%.4f lr=%s",
                        i, t["run_id"], t["hp_proxy_score"],
                        (t.get("hyperparameters") or {}).get("learning_rate"))
        return winner

    for i, t in enumerate(ordered, start=1):
        updates = {"hp_sweep_rank": i}
        if i == 1:
            updates["is_hp_winner"] = True
        try:
            _registry._safe_merge(  # noqa: SLF001
                t["arch"], t["lang"], t["condition"], t["run_id"],
                updates, op="select_hp_winner",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("merge failed for %s: %s", t["run_id"], e)

    logger.info("marked %s as is_hp_winner=True", winner["run_id"])
    return winner


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--arch", help="arch to select winner for")
    group.add_argument("--all", action="store_true",
                       help="select winners for every (arch, lang) pair with complete sweeps")
    ap.add_argument("--lang", help="lang (required with --arch)")
    ap.add_argument("--condition", default="baseline",
                    help="condition the sweep was run on (default: baseline)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the would-be winner without writing")
    args = ap.parse_args()

    if args.arch and not args.lang:
        ap.error("--lang is required when --arch is given")

    if args.arch:
        trials = _trials_for_cell(args.arch, args.lang)
        _rank_and_mark(args.arch, args.lang, args.condition, trials, args.dry_run)
        return 0

    # --all mode: group trials by (arch, lang) and rank each cell
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in _registry.iter_all_records():
        if rec.get("run_kind") != "hp_sweep":
            continue
        if rec.get("status") != "COMPLETE":
            continue
        if rec.get("hp_proxy_score") is None:
            continue
        buckets[(rec["arch"], rec["lang"])].append(rec)

    if not buckets:
        logger.warning("no hp_sweep records found")
        return 0

    for (arch, lang), trials in sorted(buckets.items()):
        cond = trials[0].get("condition", "baseline")
        _rank_and_mark(arch, lang, cond, trials, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
