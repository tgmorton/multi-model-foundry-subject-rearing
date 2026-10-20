#!/usr/bin/env python3
"""Opportunistic-tier probe (2026-10-20): how long do preemptible A100/H100/H200
pods last, and how fast do they run our model?

NRP lets any user run A100/H100/H200 at priorityClassName opportunistic /
opportunistic2 (quota-free, preemptible at any time by owners). This pod does
real work while it measures: back-to-back seed replicates of the D15 pilot
recipe (pre-LN BERT-large, LR 6e-4, 3000 steps of the 30-epoch schedule; see
docs/BERT_RECIPE_V2.md), which give the race metric's seed noise.

Every ~60 s it writes a heartbeat to S3 (a preempted pod's logs vanish with
it):
    s3://$REGISTRY_BUCKET/opp_probe/$PROBE_TAG/$PROBE_ID/pods/$POD.json
and each finished replicate to .../replicates/s<seed>.json. A restarted pod
skips seeds already done. Stops cleanly at DEADLINE_EPOCH.
Report: scripts/opp_probe_report.py.
"""
from __future__ import annotations

import json
import math
import os
import socket
import sys
import tempfile
import time

import boto3
import yaml

REPO = os.environ.get("PROBE_REPO", "/opt/repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from model_foundry.checkpoint_schedule import compute_opt_steps_per_epoch  # noqa: E402
from model_foundry.cli import load_config  # noqa: E402
from model_foundry.trainer import Trainer  # noqa: E402
import sweep_agent_lm as agent  # noqa: E402  (held-out metrics, cache chunk count)

ENV = os.environ
BUCKET = ENV.get("REGISTRY_BUCKET", "thomas-subject-drop-artifacts")
TAG, PROBE = ENV["PROBE_TAG"], ENV["PROBE_ID"]
PREFIX = f"opp_probe/{TAG}/{PROBE}"
POD = ENV.get("POD_NAME", socket.gethostname())
DEADLINE = float(ENV["DEADLINE_EPOCH"])
STEPS = int(ENV.get("PROBE_STEPS", "3000"))
SEED0 = int(ENV.get("SEED_BASE", "5000"))
BASE = ENV.get("BASE_CONFIG", "configs/sweeps/baselines/bert_large_preln_en_race.yaml")
EVAL_EVERY = 250

s3 = boto3.client("s3")
T0 = time.time()
state = {"probe": PROBE, "gpu_key": ENV.get("GPU_KEY"), "priority": ENV.get("PRIORITY"),
         "pod": POD, "node": ENV.get("NODE_NAME"), "pod_started": T0, "gpu": None,
         "seed": None, "step": 0, "steps_per_s": None, "tokens_per_s": None,
         "replicates_done": [], "status": "starting", "heartbeat": T0}


def put(key: str, obj: dict) -> None:
    try:
        s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{key}", Body=json.dumps(obj, indent=1).encode())
    except Exception as e:  # noqa: BLE001  never kill the probe over S3
        print(f"[probe] S3 write failed for {key}: {e}", flush=True)


def heartbeat(status: str | None = None) -> None:
    if status:
        state["status"] = status
    state["heartbeat"] = time.time()
    put(f"pods/{POD}.json", state)


def done_seeds() -> set:
    out = set()
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/replicates/"):
        for o in page.get("Contents", []):
            out.add(int(o["Key"].rsplit("/s", 1)[-1].split(".")[0]))
    return out


def make_config(seed: int):
    cfg = yaml.safe_load(open(os.path.join(REPO, BASE)))
    cfg.pop("rater_sweep", None)
    cfg["experiment_name"] = f"opp-probe-{PROBE}-s{seed}"
    t = cfg["training"]
    spe = compute_opt_steps_per_epoch(
        num_chunks=int(agent._read_cache_num_chunks(cfg, REPO)),
        phys_batch=int(cfg["data"]["batch_size"]),
        grad_accum=int(t.get("gradient_accumulation_steps", 1)))
    t.update(train_steps=STEPS, scheduler_train_steps=30 * spe, warmup_steps=None,
             epochs=max(1, math.ceil(STEPS / spe)), output_dir=f"models/opp_probe/{PROBE}/s{seed}",
             checkpoint_schedule=[], auto_generate_checkpoints=False, resume_from_checkpoint=False)
    cfg["random_seed"] = seed
    cfg["logging"]["use_wandb"] = False
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, f)
    f.close()
    return load_config(f.name)


def run_replicate(seed: int) -> bool:
    """One 3000-step replicate. False if the deadline cut it short."""
    config = make_config(seed)
    trainer = Trainer(config, REPO)
    state.update(seed=seed, step=0)
    evals, chunks = [], {}
    clock = {"t": time.time(), "step": 0, "hb": 0.0}
    tokens_per_step = (config.data.batch_size * config.training.gradient_accumulation_steps
                       * config.data.max_sequence_length)

    def on_step_end(step: int) -> bool:
        # Record progress first: the training loop swallows hook exceptions,
        # so anything that raises below must not hide how far we got.
        state["step"] = step
        now = time.time()
        if state["gpu"] is None:
            try:
                import torch
                state["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            except Exception as e:  # noqa: BLE001
                state["gpu"] = f"unknown ({e})"
        if step % 50 == 0:
            dt = now - clock["t"]
            if dt > 0:
                state["steps_per_s"] = (step - clock["step"]) / dt
                state["tokens_per_s"] = state["steps_per_s"] * tokens_per_step
            clock.update(t=now, step=step)
        if step % EVAL_EVERY == 0 or step == STEPS:
            if "c" not in chunks:
                chunks["c"] = agent._load_held_out_chunks(trainer, config, max_sequences=4000)
            m = agent._compute_held_out_metrics(trainer, config, chunks=chunks["c"])
            loss = math.log(m["ppl"]) if m["ppl"] and math.isfinite(m["ppl"]) else float("nan")
            evals.append({"step": step, "held_out_loss": loss, "accuracy": m["accuracy"],
                          "top_token_share": m["top_token_share"], "t": now})
            state["last_eval"] = evals[-1]
            print(f"[probe] seed {seed} step {step}: held-out loss {loss:.3f} acc {m['accuracy']:.3f}", flush=True)
        if now - clock["hb"] >= 60:
            heartbeat("training")
            clock["hb"] = now
        return now >= DEADLINE

    t_start = time.time()
    trainer.train(on_step_end=on_step_end, endpoint_checkpoint="none")
    finished = state["step"] >= STEPS
    if finished:
        put(f"replicates/s{seed}.json", {"seed": seed, "pod": POD, "gpu": state["gpu"],
                                         "gpu_key": state["gpu_key"], "priority": state["priority"],
                                         "seconds": time.time() - t_start, "evals": evals})
        state["replicates_done"].append(seed)
    del trainer
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return finished


def main() -> int:
    heartbeat("started")
    done = done_seeds()
    k = 0
    while time.time() < DEADLINE:
        seed = SEED0 + k
        k += 1
        if seed in done:
            continue
        print(f"[probe] {PROBE} on {state['node']}: replicate seed {seed}", flush=True)
        if not run_replicate(seed):
            break
        heartbeat("between_replicates")
    heartbeat("deadline")
    print("[probe] deadline reached; exiting cleanly", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
