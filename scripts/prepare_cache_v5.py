#!/usr/bin/env python3
"""Pre-populate tokenized + chunked caches for the v5 rater-comparison
cohort (gpt2_small, three arms — shared_unigram tokenizer only).

Indexed Job pod-side script: JOB_COMPLETION_INDEX selects a line from
the cells file (a pdrop2_* manipulation slug). Mirrors
prepare_cache_for_cell.py but cell-per-index instead of
(tokenizer x condition)-per-index; both cli subcommands are idempotent.

Required env:
    CELLS_FILE             path to the slug list (one per line)
    JOB_COMPLETION_INDEX   set by K8s Indexed Job
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / "configs" / "sweeps" / "baselines" / "gpt2_medium_en.yaml"


def main() -> None:
    idx = int(os.environ["JOB_COMPLETION_INDEX"])
    cells = [ln.strip() for ln in open(os.environ["CELLS_FILE"]) if ln.strip()]
    if idx >= len(cells):
        sys.exit(f"FATAL: idx={idx} out of range ({len(cells)} cells)")
    slug = cells[idx]
    corpus = f"data/manipulations/en/{slug}/"
    if not (REPO_ROOT / corpus).exists():
        sys.exit(f"FATAL: corpus missing: {corpus} — compose first")

    base = yaml.safe_load(BASELINE.read_text())
    base["experiment_name"] = f"prep-v5-{slug}"
    base["data"]["source_corpus"] = corpus
    base["data"]["training_corpus"] = corpus
    base["tokenizer"]["output_dir"] = "tokenizers/en_shared_unigram/"
    base["tokenizer"]["tokenizer_type"] = "sentencepiece"

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(base, f)
        cfg = f.name
    print(f"[prep-v5] idx={idx} slug={slug} corpus={corpus}", flush=True)
    for sub in ("tokenize-dataset", "preprocess-data"):
        print(f"[prep-v5] {sub}", flush=True)
        subprocess.run([sys.executable, "-m", "model_foundry.cli", sub, cfg],
                       check=True, cwd=REPO_ROOT)
    print(f"[prep-v5] OK {slug}", flush=True)


if __name__ == "__main__":
    main()
