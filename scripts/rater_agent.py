#!/usr/bin/env python3
"""Pod-side agent for the cross-fold recoverability RATER models.

Raters are utility gpt2_small models trained on one document-level fold of
train_90M (see scripts/make_fold_corpora.py) at the 3-epoch sweep horizon.
Each rater scores the OPPOSITE fold's subject-pronoun instances, so every
instance in the corpus gets a surprisal from a model that never saw it
(memorization-clean recoverability ranking for the graded pronoun-drop
study). They are NOT study runs: registry run_kind=rater, own WandB project.

Config synthesis mirrors scripts/production_agent.py: gpt2_small_en sweep
baseline + HP winners rank 0, corpus swapped to the fold, short horizon.
Empty checkpoint schedule — the training loop's endpoint guard saves the
final checkpoint unconditionally, which is the only artifact we need.

Required env:
    FOLD    a | b
    MODE    precache | train
Optional env:
    RATER_EPOCHS  (default 3)
    PHYS_BATCH    (default 16 — production gpt2_small value)
    HP_RANK       (default 0)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / "configs" / "sweeps" / "baselines" / "gpt2_small_en.yaml"
WINNERS = REPO_ROOT / "data" / "sweep_winners" / "gpt2_small_en.json"


def main() -> None:
    fold = os.environ["FOLD"].strip().lower()
    mode = os.environ["MODE"].strip().lower()
    if fold not in ("a", "b") or mode not in ("precache", "train"):
        sys.exit(f"FATAL: bad FOLD={fold!r} or MODE={mode!r}")

    epochs = int(os.environ.get("RATER_EPOCHS", "3"))
    phys_batch = int(os.environ.get("PHYS_BATCH", "16"))
    hp_rank = int(os.environ.get("HP_RANK", "0"))

    cfg = yaml.safe_load(BASELINE.read_text())
    hp = json.loads(WINNERS.read_text())["configs"][hp_rank]

    # HP overlay — same fields production_agent applies.
    cfg["training"]["learning_rate"] = float(hp["learning_rate"])
    cfg["training"]["warmup_ratio"] = float(hp["warmup_ratio"])
    cfg["training"]["adam_beta2"] = float(hp["adam_beta2"])
    eff = int(hp["effective_batch_size"])
    cfg["data"]["batch_size"] = phys_batch
    cfg["training"]["gradient_accumulation_steps"] = max(1, eff // phys_batch)
    cfg["model"]["transformer"]["dropout"] = float(hp["dropout"])
    if hp.get("attention_dropout") is not None:
        cfg["model"]["transformer"]["attention_dropout"] = float(hp["attention_dropout"])

    corpus = f"data/raw/en/train_90M_fold_{fold}/"
    cfg["data"]["training_corpus"] = corpus
    cfg["data"]["source_corpus"] = corpus

    cfg["training"]["epochs"] = epochs
    cfg["training"]["auto_generate_checkpoints"] = False
    cfg["training"]["checkpoint_schedule"] = []
    cfg["training"]["resume_from_checkpoint"] = False

    seed = 42
    cfg["random_seed"] = seed
    run_id = f"rater-gpt2_small-en-fold_{fold}-e{epochs}-h{hp_rank}-s{seed}"
    cfg["experiment_name"] = run_id
    cfg["training"]["output_dir"] = f"models/raters/{run_id}"
    cfg["logging"]["wandb_project"] = "subject-drop-raters"

    os.environ["REGISTRY_RUN_KIND"] = "rater"
    os.environ["REGISTRY_LANG"] = "en"
    os.environ["REGISTRY_CONDITION"] = f"rater_fold_{fold}"
    os.environ["REGISTRY_SEED"] = str(seed)
    os.environ["REGISTRY_RUN_ID"] = run_id

    out_path = Path(f"/tmp/rater_{fold}_{mode}.yaml")
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("=== rater_agent ===")
    print(f"  fold={fold} mode={mode} epochs={epochs} phys={phys_batch} "
          f"eff={eff} grad_accum={cfg['training']['gradient_accumulation_steps']}")
    print(f"  corpus={corpus}")
    print(f"  run_id={run_id}")
    print(f"  config -> {out_path}")
    sys.stdout.flush()

    if mode == "precache":
        for step in ("tokenize-dataset", "preprocess-data"):
            print(f"\n--- {step} ---", flush=True)
            rc = subprocess.call(
                [sys.executable, "-m", "model_foundry.cli", step, str(out_path)],
                cwd=str(REPO_ROOT),
            )
            if rc != 0:
                sys.exit(f"FAIL: {step} returned {rc}")
        print("\nRATER PRECACHE OK", flush=True)
        return

    # MODE=train: fail-fast if the precache job hasn't populated the cache —
    # training pods only READ caches (sweep-1 race lesson).
    sys.path.insert(0, str(REPO_ROOT))
    from model_foundry.cache_keys import compute_cache_key  # noqa: E402

    cache_key = compute_cache_key(
        str(REPO_ROOT / corpus),
        str(REPO_ROOT / cfg["tokenizer"]["output_dir"]),
        cfg["data"]["max_sequence_length"],
        cfg.get("dataset_manipulation") or [],
    )
    for probe in (
        Path(f"/mnt/data/tokenized/{cache_key}/train/dataset_info.json"),
        Path(f"/mnt/data/chunked/{cache_key}/dataset_info.json"),
    ):
        if not probe.exists():
            sys.exit(
                f"FAIL: cache missing at {probe}. "
                "Run k8s/job-rater-precache.yaml to completion first."
            )

    rc = subprocess.call(
        [sys.executable, "-m", "model_foundry.cli", "run", str(out_path)],
        cwd=str(REPO_ROOT),
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
