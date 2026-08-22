#!/usr/bin/env python3
"""Preemption/resume smoke (lang-manifold port; wave-launcher gate).

Single pod, three sequential short trainings (gpt2_small, 400 steps,
resume anchors at [200, 400], seed 1234, fold_a corpus):

  det_ref     deterministic, uninterrupted
  det_victim  deterministic, SIGKILLed after checkpoint-200 lands, then
              resumed to completion. ASSERT: endpoint is BITWISE identical
              to det_ref (this exercises RNG/scaler/dataloader-offset
              restore + resolve_resume_epoch end to end — any resume bug
              breaks the hash).
  fast_victim production kernels, killed + resumed. ASSERT: mechanical
              resume works, final step/token accounting exact, no epoch
              replay. (No weight comparison — nondeterministic kernels
              diverge from any reference regardless of preemption.)

The checkpoint-dir poll is race-free because checkpoint writes are atomic
(staged .tmp + os.replace): a `checkpoint-200` dir exists only complete.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / "configs" / "sweeps" / "baselines" / "gpt2_small_en.yaml"
WINNERS = REPO_ROOT / "data" / "sweep_winners" / "gpt2_small_en.json"
STEPS = 400
ANCHOR = 200


def build_config(variant: str, deterministic: bool, resume: bool) -> Path:
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
    cfg["data"]["num_workers"] = 0
    cfg["training"]["epochs"] = 1
    cfg["training"]["train_steps"] = STEPS
    cfg["training"]["auto_generate_checkpoints"] = False
    cfg["training"]["checkpoint_schedule"] = [ANCHOR, STEPS]
    cfg["training"]["resume_from_checkpoint"] = resume
    cfg["training"]["deterministic"] = deterministic
    cfg["random_seed"] = 1234
    run_id = f"preempt-{variant}-s1234"
    cfg["experiment_name"] = run_id
    cfg["training"]["output_dir"] = f"models/detsmoke/{run_id}"
    cfg["logging"]["use_wandb"] = False
    path = Path(f"/tmp/preempt_{variant}{'_resume' if resume else ''}.yaml")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def out_dir(variant: str) -> Path:
    return REPO_ROOT / "models" / "detsmoke" / f"preempt-{variant}-s1234"


def run_full(variant: str, deterministic: bool) -> None:
    cfg = build_config(variant, deterministic, resume=False)
    rc = subprocess.call([sys.executable, "-m", "model_foundry.cli", "run",
                         str(cfg)], cwd=str(REPO_ROOT))
    if rc != 0:
        sys.exit(f"FATAL: {variant} full run rc={rc}")


def run_kill_resume(variant: str, deterministic: bool) -> None:
    cfg = build_config(variant, deterministic, resume=False)
    proc = subprocess.Popen([sys.executable, "-m", "model_foundry.cli", "run",
                            str(cfg)], cwd=str(REPO_ROOT),
                            start_new_session=True)
    anchor = out_dir(variant) / f"checkpoint-{ANCHOR}"
    deadline = time.time() + 3600
    while not anchor.exists():
        if proc.poll() is not None:
            sys.exit(f"FATAL: {variant} exited (rc={proc.returncode}) "
                     "before the kill anchor landed")
        if time.time() > deadline:
            proc.kill()
            sys.exit(f"FATAL: {variant} never reached checkpoint-{ANCHOR}")
        time.sleep(5)
    time.sleep(20)  # let it advance past the anchor mid-flight
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait()
    print(f"  {variant}: SIGKILLed after checkpoint-{ANCHOR} (rc={proc.returncode})",
          flush=True)

    cfg_resume = build_config(variant, deterministic, resume=True)
    rc = subprocess.call([sys.executable, "-m", "model_foundry.cli", "run",
                         str(cfg_resume)], cwd=str(REPO_ROOT))
    if rc != 0:
        sys.exit(f"FATAL: {variant} resume rc={rc}")


def accounting(variant: str) -> dict:
    meta = json.loads((out_dir(variant) / f"checkpoint-{STEPS}"
                       / "metadata.json").read_text())
    return {"step": meta["global_step"],
            "tokens": meta["token_metrics"]["total_tokens_processed"]}


def main() -> None:
    # Idempotence across pod retries: a leftover output dir from a failed
    # attempt lets a rerun pass vacuously (observed: all-zero token
    # accounting). Start clean, always.
    import shutil
    for v in ("det_ref", "det_victim", "fast_victim"):
        if out_dir(v).exists():
            shutil.rmtree(out_dir(v))
    print("=== preemption smoke: det_ref (uninterrupted) ===", flush=True)
    run_full("det_ref", deterministic=True)
    print("=== preemption smoke: det_victim (kill + resume) ===", flush=True)
    run_kill_resume("det_victim", deterministic=True)
    print("=== preemption smoke: fast_victim (kill + resume) ===", flush=True)
    run_kill_resume("fast_victim", deterministic=False)

    ref, victim, fast = (accounting(v) for v in
                         ("det_ref", "det_victim", "fast_victim"))
    print(f"accounting: ref={ref} det_victim={victim} fast_victim={fast}",
          flush=True)
    expected_tokens = STEPS * 16 * 8 * 1000  # steps * phys * accum * seq
    assert ref["tokens"] == expected_tokens, \
        f"ref token accounting wrong ({ref['tokens']} != {expected_tokens})"
    assert victim == ref, "det accounting mismatch (resume token-counter bug?)"
    assert fast == ref, "fast accounting mismatch (resume token-counter bug?)"

    from model_foundry.compare_runs import compare
    res = compare(out_dir("det_ref") / f"checkpoint-{STEPS}",
                  out_dir("det_victim") / f"checkpoint-{STEPS}",
                  atol=0.0, rtol=0.0)
    print("VERDICT det_ref vs det_victim:", json.dumps(
        {k: res[k] for k in ("verdict", "sha_a", "sha_b")}), flush=True)
    if res["verdict"] != "bitwise":
        sys.exit("FATAL: killed+resumed deterministic run is NOT bitwise "
                 "identical to the uninterrupted reference — resume state "
                 "restore is broken somewhere")
    print("PREEMPTION SMOKE OK", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
