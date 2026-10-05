#!/usr/bin/env python3
"""Pre-launch cache check for a wave: does every cell have its tokenized +
chunked cache for this arch? (2026-10-05)

Computes the cache key exactly as scripts/wave2_agent.py does (same
baseline config, corpus path, tokenizer dir, sequence length) and checks the
two probe files the agent fails fast on. Run in a pod with the same
/opt/repo -> /mnt/data symlinks as the training pods, so the corpus path in
the key matches.

Usage (in-pod):
  python scripts/preflight_caches.py --arch gpt2_large --cells k8s/wave2/cells_w3_gpt2large.txt
Exit 0 = every cell ready.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
_RAND100 = re.compile(r"^pdrop_rand100_(\w+)$")  # v4 alias, as in wave2_launcher


def main() -> int:
    from model_foundry.cache_keys import compute_cache_key
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--cells", type=Path, required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load((REPO / "configs/sweeps/baselines" / f"{a.arch}_en.yaml").read_text())
    cells = [l.strip() for l in a.cells.read_text().splitlines() if l.strip()]
    bad = []
    for cell in cells:
        m = _RAND100.match(cell)
        corpus_cell = f"pdrop_info100_{m.group(1)}" if m else cell
        corpus = f"data/manipulations/en/{corpus_cell}/"
        if not any((REPO / corpus).glob("*.train")):
            bad.append((cell, "corpus missing"))
            continue
        key = compute_cache_key(str(REPO / corpus), str(REPO / cfg["tokenizer"]["output_dir"]),
                                cfg["data"]["max_sequence_length"],
                                cfg.get("dataset_manipulation") or [])
        for probe in (Path(f"/mnt/data/tokenized/{key}/train/dataset_info.json"),
                      Path(f"/mnt/data/chunked/{key}/dataset_info.json")):
            if not probe.exists():
                bad.append((cell, f"missing {probe}"))
                break
    for cell, why in bad:
        print(f"NOT READY {cell}: {why}", flush=True)
    print(f"PREFLIGHT {a.arch}: {len(cells) - len(bad)}/{len(cells)} cells ready", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
