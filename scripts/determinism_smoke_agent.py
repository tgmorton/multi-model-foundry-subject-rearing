#!/usr/bin/env python3
"""D8a determinism smoke (checkpoint-paradigm memo, 2026-08-22).

Three short gpt2_small trainings on the SAME GPU product, fold_a corpus,
fixed seed, train_steps-capped:
    VARIANT=det1   training.deterministic=true
    VARIANT=det2   training.deterministic=true (identical config; the
                   bitwise-repeatability partner of det1)
    VARIANT=fast   deterministic=false (production kernels: FA2 etc.)

Measures for the memo: (a) sha256(model.safetensors) of det1 vs det2 —
the bitwise claim on CUDA; (b) tokens/sec of det vs fast — the
determinism throughput tax; (c) any use_deterministic_algorithms
warnings in the logs (ops lacking deterministic variants).
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
    variant = os.environ["VARIANT"].strip().lower()
    if variant not in ("det1", "det2", "fast"):
        sys.exit(f"FATAL: bad VARIANT={variant!r}")
    steps = int(os.environ.get("SMOKE_STEPS", "400"))

    cfg = yaml.safe_load(BASELINE.read_text())
    hp = json.loads(WINNERS.read_text())["configs"][0]
    cfg["training"]["learning_rate"] = float(hp["learning_rate"])
    cfg["training"]["warmup_ratio"] = float(hp["warmup_ratio"])
    cfg["training"]["adam_beta2"] = float(hp["adam_beta2"])
    cfg["data"]["batch_size"] = 16
    cfg["training"]["gradient_accumulation_steps"] = 8
    cfg["model"]["transformer"]["dropout"] = float(hp["dropout"])
    cfg["model"]["transformer"]["attention_dropout"] = float(hp["attention_dropout"])

    cfg["data"]["training_corpus"] = "data/raw/en/train_90M_fold_a/"
    cfg["data"]["source_corpus"] = "data/raw/en/train_90M_fold_a/"
    cfg["data"]["num_workers"] = 0  # single-process load order, both variants

    cfg["training"]["epochs"] = 1
    cfg["training"]["train_steps"] = steps
    cfg["training"]["auto_generate_checkpoints"] = False
    cfg["training"]["checkpoint_schedule"] = [steps]
    cfg["training"]["resume_from_checkpoint"] = False
    cfg["training"]["deterministic"] = variant != "fast"

    cfg["random_seed"] = 1234
    run_id = f"detsmoke-gpt2_small-{variant}-s1234"
    cfg["experiment_name"] = run_id
    cfg["training"]["output_dir"] = f"models/detsmoke/{run_id}"
    cfg["logging"]["use_wandb"] = False

    os.environ["REGISTRY_RUN_KIND"] = "smoke"
    os.environ["REGISTRY_RUN_ID"] = run_id

    out = Path(f"/tmp/detsmoke_{variant}.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"=== determinism smoke: variant={variant} steps={steps} "
          f"deterministic={cfg['training']['deterministic']}")
    sys.stdout.flush()

    rc = subprocess.call(
        [sys.executable, "-m", "model_foundry.cli", "run", str(out)],
        cwd=str(REPO_ROOT),
    )
    if rc != 0:
        sys.exit(rc)

    # Hash the endpoint weights for the bitwise comparison.
    import hashlib
    ckpt_root = REPO_ROOT / cfg["training"]["output_dir"]
    weights = sorted(ckpt_root.glob("checkpoint-*/model.safetensors"))
    for w in weights:
        h = hashlib.sha256(w.read_bytes()).hexdigest()
        print(f"WEIGHTS_SHA256 {variant} {w.parent.name} {h}", flush=True)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
