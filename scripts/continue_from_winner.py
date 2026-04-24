#!/usr/bin/env python3
"""
Continue training a sweep-winner checkpoint out to the full 30-epoch
production budget, OR train a paired fresh-seed replica from scratch
with the winning HPs.

Invoked by k8s/job-continue-production.yaml. Two modes:

--mode continue   Resume training from WINNER_CHECKPOINT. The Trainer's
                  existing resume_from_checkpoint path picks the newest
                  checkpoint in output_dir that carries training_state.pt,
                  so we pre-seed output_dir with the winner checkpoint
                  files.

--mode fresh      Train from random init with the winning HPs and a fresh
                  random seed. Produces the paired replica (per plan:
                  "sweep-run continued + 1 additional seed").

The 80-anchor checkpoint schedule already covers 30 epochs worth of
optimizer-step anchors — the Trainer will save at every anchor in that
list, regardless of whether we started from scratch or resumed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import wandb

from model_foundry.config import ExperimentConfig
from model_foundry.trainer import Trainer
from model_foundry.utils import find_project_root
from model_foundry.wandb_init import init_wandb

logger = logging.getLogger("continue_from_winner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def _apply_hp_overlay(cfg: dict, hp: dict) -> None:
    """Apply the sweep-winner HPs to the baseline config. Mirrors
    sweep_agent_lm._apply_hp_overrides so production and sweep share
    identical overlay semantics."""
    t = cfg["training"]
    d = cfg["data"]
    m = cfg["model"]
    if "learning_rate" in hp: t["learning_rate"] = float(hp["learning_rate"])
    if "warmup_ratio" in hp:  t["warmup_ratio"]  = float(hp["warmup_ratio"])
    if "adam_beta2" in hp:    t["adam_beta2"]    = float(hp["adam_beta2"])
    if "effective_batch_size" in hp:
        eff = int(hp["effective_batch_size"])
        phys = int(d["batch_size"])
        if eff < phys:
            d["batch_size"] = eff
            t["gradient_accumulation_steps"] = 1
        else:
            t["gradient_accumulation_steps"] = max(1, eff // phys)

    # Dropout can live at model.<family>.* (nested) or at the top of
    # model.* (flat legacy) — see sweep_agent_lm._apply_hp_overrides for
    # the dispatch rules. Same logic here.
    if "dropout" in hp or "attention_dropout" in hp:
        val_drop = float(hp["dropout"]) if "dropout" in hp else None
        val_adrop = float(hp["attention_dropout"]) if "attention_dropout" in hp else None
        touched_nested = False
        for block in ("transformer", "rnn", "mamba"):
            if isinstance(m.get(block), dict):
                if val_drop is not None:
                    m[block]["dropout"] = val_drop
                if val_adrop is not None and block == "transformer":
                    m[block]["attention_dropout"] = val_adrop
                touched_nested = True
        if not touched_nested:
            # flat-shape config
            if val_drop is not None: m["dropout"] = val_drop
            if val_adrop is not None and ("layers" in m or "attention_heads" in m):
                m["attention_dropout"] = val_adrop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--winner-ckpt", required=True,
                    help="Directory containing the winner's last training_state.pt")
    ap.add_argument("--hp-json", required=True,
                    help="JSON string of winning HPs (same schema as sweep agent wc)")
    ap.add_argument("--mode", choices=["continue", "fresh"], required=True)
    ap.add_argument("--seed-fresh", type=int, required=True)
    args = ap.parse_args()

    base_dir = find_project_root(__file__)
    with open(Path(base_dir) / args.base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hp = json.loads(args.hp_json)
    _apply_hp_overlay(cfg, hp)

    # Train for the full production budget.
    prod_epochs = int(cfg["training"].get("production_epochs", 30))
    cfg["training"]["epochs"] = prod_epochs
    cfg["training"]["train_steps"] = None  # recomputed from epochs

    # Derive a unique output dir per replica. continue-mode must be
    # DIFFERENT from the sweep's original output_dir so we don't clobber
    # the sweep trial's final checkpoint — we COPY the winner checkpoint
    # in, then let the trainer resume from there and add more anchors.
    suffix = f"continued_{os.path.basename(args.winner_ckpt.rstrip('/'))}" \
        if args.mode == "continue" \
        else f"fresh_{args.seed_fresh}"
    out_dir = os.path.join("models", "production", suffix)
    cfg["training"]["output_dir"] = out_dir
    cfg["experiment_name"] = f"prod_{suffix}"

    if args.mode == "continue":
        # Seed the output dir with the winner's checkpoint state. The
        # Trainer's resume_from_checkpoint path picks the newest
        # checkpoint in output_dir that has training_state.pt.
        abs_out = os.path.join(base_dir, out_dir)
        os.makedirs(abs_out, exist_ok=True)
        # Copy the entire winner-ckpt directory contents into the new
        # output dir. We use rsync-like semantics via shutil.copytree
        # with dirs_exist_ok for idempotency (if the job retries).
        for entry in os.listdir(args.winner_ckpt):
            src = os.path.join(args.winner_ckpt, entry)
            dst = os.path.join(abs_out, entry)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        logger.info("seeded %s with winner checkpoint from %s", abs_out, args.winner_ckpt)
        cfg["training"]["resume_from_checkpoint"] = True
        cfg["random_seed"] = int(cfg.get("random_seed", 42))  # keep original seed
    else:
        cfg["training"]["resume_from_checkpoint"] = False
        cfg["random_seed"] = int(args.seed_fresh)

    config = ExperimentConfig(**cfg)

    # Identity for registry + wandb grouping. `run_kind=production` so
    # launcher/compactor can distinguish these from sweep trials.
    identity = {
        "run_id": config.experiment_name,
        "arch": hp.get("arch", "?"),
        "lang": hp.get("lang", "?"),
        "condition": "baseline",
        "seed": config.random_seed,
        "run_kind": "production",
    }
    init_wandb(config, identity)

    trainer = Trainer(config, base_dir)
    trainer.train()
    logger.info("production %s run COMPLETE: %s", args.mode, out_dir)


if __name__ == "__main__":
    main()
