#!/usr/bin/env python3
"""Pod-side agent for the wave-2 crossed-matrix trainings (Track E).

One Indexed Job per (cell × arch); JOB_COMPLETION_INDEX 0..9 maps to
(hp_rank, replicate) via divmod(idx, 2). The run seed is derived
in-pod — seed = derive_seed(WAVE_ID, cell, arch, f"h{hp}", replicate) —
so the full assignment is a pure function any process can recompute
(D8/D9 empiricist kit), while still being recorded explicitly in the
run_id, config, and registry.

Save policy (D8 resolution + Thomas's phase-3 anchor spec): the full
token-anchored checkpoint schedule for trajectories (weights-only), with
resume state ONLY at {epoch-1, epoch-2, midpoint, final} — the same four
anchors the pruned historical runs keep. Fast kernels (deterministic
mode declined). Eval + capture happen in eval pods post-training; the
prune step is gated on capture-complete markers (launcher-chained).

Required env: WAVE_ID, CELL (corpus slug, e.g. pdrop_info30_rmexpl —
rand_100 cells pass CORPUS_CELL to alias the shared info_100 corpus),
ARCH, PHYS_BATCH, JOB_COMPLETION_INDEX.
Optional: CORPUS_CELL (defaults to CELL), WAVE_EPOCHS (default 30),
WANDB_PROJECT_WAVE2 (default subject-drop-wave2).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SWEEP_BASELINE_DIR = REPO_ROOT / "configs" / "sweeps" / "baselines"
SWEEP_WINNERS_DIR = REPO_ROOT / "data" / "sweep_winners"


def _apply_hp(cfg: dict, hp: dict, phys_batch: int) -> None:
    cfg["training"]["learning_rate"] = float(hp["learning_rate"])
    cfg["training"]["warmup_ratio"] = float(hp["warmup_ratio"])
    cfg["training"]["adam_beta2"] = float(hp["adam_beta2"])
    eff = int(hp["effective_batch_size"])
    cfg["data"]["batch_size"] = phys_batch
    cfg["training"]["gradient_accumulation_steps"] = max(1, eff // phys_batch)
    model = cfg["model"]
    block = model.get("transformer") or model.get("lstm") or model
    if hp.get("dropout") is not None and "dropout" in block:
        block["dropout"] = float(hp["dropout"])
    if hp.get("attention_dropout") is not None and "attention_dropout" in block:
        block["attention_dropout"] = float(hp["attention_dropout"])


def pick_resume_anchors(schedule, steps_per_epoch: int, epochs: int):
    """The 4 resumable anchors (Thomas 2026-08-22): nearest scheduled
    steps to epoch-1, epoch-2, midpoint, and the final step."""
    schedule = sorted(schedule)
    final_step = schedule[-1] if schedule else steps_per_epoch * epochs
    targets = [steps_per_epoch, steps_per_epoch * 2,
               (steps_per_epoch * epochs) // 2, final_step]
    return sorted({min(schedule, key=lambda s: abs(s - t)) for t in targets})


def main() -> None:
    from model_foundry.seeding import derive_seed
    from model_foundry.checkpoint_schedule import (
        compute_checkpoint_schedule,
        read_num_chunks,
    )
    from model_foundry.cache_keys import compute_cache_key

    wave = os.environ["WAVE_ID"]
    cell = os.environ["CELL"]
    corpus_cell = os.environ.get("CORPUS_CELL") or cell
    arch = os.environ["ARCH"]
    phys_batch = int(os.environ["PHYS_BATCH"])
    idx = int(os.environ["JOB_COMPLETION_INDEX"])
    epochs = int(os.environ.get("WAVE_EPOCHS", "30"))
    hp_rank, replicate = divmod(idx, 2)

    seed = derive_seed(wave, cell, arch, f"h{hp_rank}", replicate)
    run_id = f"{arch}-en-{cell}-h{hp_rank}-s{seed}"

    cfg = yaml.safe_load((SWEEP_BASELINE_DIR / f"{arch}_en.yaml").read_text())
    winners = json.loads((SWEEP_WINNERS_DIR / f"{arch}_en.json").read_text())
    if hp_rank >= len(winners["configs"]):
        sys.exit(f"FATAL: no hp rank {hp_rank} for {arch}")
    _apply_hp(cfg, winners["configs"][hp_rank], phys_batch)

    corpus = f"data/manipulations/en/{corpus_cell}/"
    cfg["data"]["training_corpus"] = corpus
    cfg["data"]["source_corpus"] = corpus
    cfg["training"]["epochs"] = epochs
    cfg["random_seed"] = seed
    cfg["experiment_name"] = run_id
    cfg["training"]["output_dir"] = f"models/wave2/{run_id}"
    cfg["logging"]["wandb_project"] = os.environ.get(
        "WANDB_PROJECT_WAVE2", "subject-drop-wave2")

    # Fail-fast cache check (precache is a separate pre-job; sweep-1 lesson).
    cache_key = compute_cache_key(
        str(REPO_ROOT / corpus),
        str(REPO_ROOT / cfg["tokenizer"]["output_dir"]),
        cfg["data"]["max_sequence_length"],
        cfg.get("dataset_manipulation") or [],
    )
    chunked_dir = Path(f"/mnt/data/chunked/{cache_key}")
    for probe in (Path(f"/mnt/data/tokenized/{cache_key}/train/dataset_info.json"),
                  chunked_dir / "dataset_info.json"):
        if not probe.exists():
            sys.exit(f"FAIL: cache missing at {probe} — run the wave2 "
                     "precache job for this cell first.")

    # Full token-anchored schedule (trajectories) + 4 resume anchors:
    # ep1, ep2, midpoint, final (Thomas 2026-08-22 phase-3 spec, applied
    # natively to new runs).
    num_chunks = read_num_chunks(str(chunked_dir))
    grad_accum = int(cfg["training"]["gradient_accumulation_steps"])
    schedule, _default_rss = compute_checkpoint_schedule(
        num_chunks=num_chunks, phys_batch=phys_batch, grad_accum=grad_accum,
        epochs=epochs, seq_len=cfg["data"]["max_sequence_length"],
    )
    import math
    steps_per_epoch = math.floor(
        math.ceil(num_chunks / phys_batch) / grad_accum)
    resume_steps = pick_resume_anchors(schedule, steps_per_epoch, epochs)
    cfg["training"]["auto_generate_checkpoints"] = False
    cfg["training"]["checkpoint_schedule"] = sorted(schedule)
    cfg["training"]["resume_state_steps"] = resume_steps
    cfg["training"]["resume_from_checkpoint"] = (
        os.environ.get("RESUME", "").lower() in ("1", "true"))
    cfg["training"]["deterministic"] = False  # D8: fast kernels everywhere

    os.environ["REGISTRY_RUN_KIND"] = "wave2"
    os.environ["REGISTRY_LANG"] = "en"
    os.environ["REGISTRY_CONDITION"] = cell
    os.environ["REGISTRY_SEED"] = str(seed)
    os.environ["REGISTRY_RUN_ID"] = run_id

    out = Path("/tmp/wave2_config.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("=== wave2_agent ===")
    print(f"  wave={wave} cell={cell} corpus_cell={corpus_cell} arch={arch}")
    print(f"  hp=h{hp_rank} replicate={replicate} seed={seed} (blake2b-derived)")
    print(f"  epochs={epochs} anchors={len(schedule)} resume_steps={resume_steps}")
    print(f"  run_id={run_id}")
    sys.stdout.flush()

    rc = subprocess.call([sys.executable, "-m", "model_foundry.cli", "run",
                          str(out)], cwd=str(REPO_ROOT))
    if rc == 0:
        Path("/tmp/run_succeeded").write_text(run_id + "\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
