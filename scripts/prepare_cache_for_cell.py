#!/usr/bin/env python3
"""Pre-populate the tokenized + chunked content-addressed caches for one
(tokenizer × condition) production cell.

Indexed Job pod-side script. JOB_COMPLETION_INDEX 0..9 selects the cell:
    idx = tok_idx * 5 + cond_idx
where ``tok_idx ∈ {0=shared_unigram, 1=bert_wordpiece}`` and
``cond_idx`` indexes the 5 conditions (baseline first).

We synthesize a minimal config matching one of the sweep baselines and
hand it to ``model_foundry.cli {tokenize-dataset, preprocess-data}``.
Both subcommands are idempotent — re-runs no-op.

Required env:
    LANG                   en | es
    JOB_COMPLETION_INDEX   set by K8s Indexed Job (0..9)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_BASELINE_DIR = REPO_ROOT / "configs" / "sweeps" / "baselines"

TOKENIZERS_EN = [
    # (id, tokenizer_dir, tokenizer_type, source-arch-baseline)
    ("shared_unigram", "tokenizers/en_shared_unigram/", "sentencepiece",
     "gpt2_medium_en"),
    ("bert_wordpiece", "tokenizers/en_bert_wordpiece/", "wordpiece",
     "bert_large_en"),
]

CONDITIONS_EN = [
    "baseline",
    "remove_expletive_sentences",
    "impoverish_case",
    "lemmatize_verbs",
    "enrich_verbal_morphology",
]


def _corpus_path(lang: str, condition: str) -> str:
    if condition == "baseline":
        return f"data/raw/{lang}/train_90M/"
    return f"data/manipulations/{lang}/{condition}/"


def main() -> None:
    lang = os.environ.get("LANG", "en")
    if lang != "en":
        sys.exit("ES production prep not yet wired; lang must be en for now.")
    idx = int(os.environ["JOB_COMPLETION_INDEX"])

    tok_idx, cond_idx = divmod(idx, len(CONDITIONS_EN))
    if tok_idx >= len(TOKENIZERS_EN):
        sys.exit(f"FATAL: idx={idx} produces tok_idx={tok_idx} (out of range)")

    tok_id, tok_dir, tok_type, base_arch = TOKENIZERS_EN[tok_idx]
    condition = CONDITIONS_EN[cond_idx]

    # Start from the appropriate arch's sweep baseline so the model
    # config block is valid for schema validation. Then swap only the
    # data-side fields we actually care about (training_corpus +
    # tokenizer.output_dir).
    base = yaml.safe_load((SWEEP_BASELINE_DIR / f"{base_arch}.yaml").read_text())
    base["experiment_name"] = f"prep-{tok_id}-{condition}"
    base["data"]["source_corpus"] = _corpus_path(lang, condition)
    base["data"]["training_corpus"] = _corpus_path(lang, condition)
    # test_corpus stays at the lang's test_10M (set in the sweep base)
    base["tokenizer"]["output_dir"] = tok_dir
    base["dataset_manipulation"] = []

    out_path = Path(f"/tmp/prep_{tok_id}_{condition}.yaml")
    out_path.write_text(yaml.safe_dump(base, sort_keys=False))

    print(f"=== prepare_cache_for_cell ===")
    print(f"  idx={idx}  tokenizer={tok_id}  condition={condition}")
    print(f"  training_corpus={base['data']['training_corpus']}")
    print(f"  tokenizer.output_dir={tok_dir}")
    print(f"  config -> {out_path}")
    sys.stdout.flush()

    for step in ("tokenize-dataset", "preprocess-data"):
        print(f"\n--- {step} ---", flush=True)
        rc = subprocess.call(
            [sys.executable, "-m", "model_foundry.cli", step, str(out_path)],
            cwd=str(REPO_ROOT),
        )
        if rc != 0:
            sys.exit(f"FAIL: {step} returned {rc}")

    print("\nPREPARE CELL OK", flush=True)


if __name__ == "__main__":
    main()
