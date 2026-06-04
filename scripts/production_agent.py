#!/usr/bin/env python3
"""Production training agent (per-pod entrypoint).

Reads (arch, lang, intervention, hp_rank, seed) from environment variables,
overlays the corresponding sweep-winner HPs onto the per-(arch, lang)
baseline config, swaps the training corpus for the intervention's
pre-ablated path, computes the 80-anchor checkpoint schedule, writes the
final config to /tmp/, and shells out to ``model_foundry.cli run``.

This is the production analogue of ``sweep_agent_lm.py`` — no wandb.config
sampling, just deterministic lookup of frozen HPs by rank.

Required env:
    ARCH                   one of gpt2_{small,medium,large}, bert_large,
                           lstm, mamba_370m
    LANG                   en | es
    INTERVENTION           baseline | remove_expletive_sentences |
                           impoverish_case | lemmatize_verbs |
                           enrich_verbal_morphology
    PHYS_BATCH             per-arch physical batch size (engineering)
    SEEDS_JSON             JSON list of 2 seeds, e.g. "[42, 137]"
    JOB_COMPLETION_INDEX   0..9 — set by K8s Indexed Job

Optional env:
    WANDB_PROJECT_PROD     defaults to "subject-drop-production"
    SAVE_RESUME_LAST_N     legacy last-N resume-state fallback (default 3);
                           unused now that resume_state_steps is explicit.
    RESUME                 truthy → resume in place from the existing
                           output_dir, emit a BACK-HALF-ONLY schedule of
                           anchors strictly past the step already reached,
                           null train_steps/warmup_steps, and save resume
                           state on the full back-half list.

The checkpoint schedule is computed by the shared helper
``model_foundry.checkpoint_schedule`` off the REAL chunk count read from
the content-addressed chunked cache (no hardcoded estimate).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_WINNERS_DIR = REPO_ROOT / "data" / "sweep_winners"
SWEEP_BASELINE_DIR = REPO_ROOT / "configs" / "sweeps" / "baselines"

INTERVENTIONS_BY_LANG = {
    "en": {
        "baseline",
        "remove_expletive_sentences",
        "impoverish_case",
        "lemmatize_verbs",
        "enrich_verbal_morphology",
    },
    "es": {
        "baseline",
        "remove_expletive_sentences",
        "impoverish_case",
        "lemmatize_verbs",
    },
}


def _required_env(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        sys.exit(f"FATAL: required env var {key} is unset")
    return v


def _training_corpus_path(lang: str, intervention: str) -> str:
    """Return the on-pod path to the training corpus for this intervention."""
    if intervention == "baseline":
        return f"data/raw/{lang}/train_90M/"
    return f"data/manipulations/{lang}/{intervention}/"


def _latest_resume_step(output_dir: Path) -> int:
    """Return the highest checkpoint-<N> step under ``output_dir`` that
    carries a training_state.pt, or 0 if none. Mirrors the trainer's
    resume-selection logic so the back-half-only schedule is filtered to
    anchors strictly past where the run will actually resume.
    """
    import re

    if not output_dir.exists():
        return 0
    best = 0
    for child in output_dir.glob("checkpoint-*"):
        m = re.search(r"checkpoint-(\d+)$", child.name)
        if not m:
            continue
        if (child / "training_state.pt").exists():
            best = max(best, int(m.group(1)))
    return best


def _apply_model_field(model: dict, field: str, val: float) -> None:
    """Set ``field`` on whichever model sub-block matches the YAML shape.
    Mirrors sweep_agent_lm._apply_hp_overrides for dropout/attn_dropout."""
    _TRANSFORMER_FLAT = {"layers", "attention_heads", "intermediate_hidden_size"}
    _RNN_FLAT = {"num_layers", "rnn_type"}
    _MAMBA_FLAT = {"d_model", "n_layers", "d_state"}
    nested_hit = False
    for block in ("transformer", "rnn", "mamba"):
        if isinstance(model.get(block), dict):
            model[block][field] = val
            nested_hit = True
    if nested_hit:
        return
    if field == "attention_dropout":
        if _TRANSFORMER_FLAT & model.keys():
            model[field] = val
    else:
        if (_TRANSFORMER_FLAT | _RNN_FLAT | _MAMBA_FLAT) & model.keys():
            model[field] = val


def _apply_hp_overrides(cfg: dict, hp: dict, phys_batch: int) -> None:
    """Overlay sweep-winner HPs onto the baseline config in-place."""
    cfg["training"]["learning_rate"] = float(hp["learning_rate"])
    cfg["training"]["warmup_ratio"] = float(hp["warmup_ratio"])
    cfg["training"]["adam_beta2"] = float(hp["adam_beta2"])

    eff = int(hp["effective_batch_size"])
    cfg["data"]["batch_size"] = phys_batch
    cfg["training"]["gradient_accumulation_steps"] = max(1, eff // phys_batch)

    _apply_model_field(cfg["model"], "dropout", float(hp["dropout"]))
    if hp.get("attention_dropout") is not None:
        _apply_model_field(
            cfg["model"], "attention_dropout", float(hp["attention_dropout"])
        )


def main() -> None:
    arch = _required_env("ARCH")
    lang = _required_env("LANG")
    intervention = _required_env("INTERVENTION")
    phys_batch = int(_required_env("PHYS_BATCH"))
    seeds = json.loads(_required_env("SEEDS_JSON"))
    job_index = int(_required_env("JOB_COMPLETION_INDEX"))

    if intervention not in INTERVENTIONS_BY_LANG.get(lang, set()):
        sys.exit(f"FATAL: intervention {intervention!r} not valid for lang {lang!r}")

    # Slot resolution. Default (full-grid) mode maps idx → (hp_rank, seed_idx)
    # via divmod, so a completions=10 Indexed Job covers the 5×2 grid. The
    # relaunch ("only missing slots") mode passes SLOT_MAP_JSON: an ordered
    # list of [hp_rank, seed_idx] pairs, one per JOB_COMPLETION_INDEX, so a
    # completions=N Indexed Job runs exactly the N slots that still need
    # training and never recomputes a done seed.
    slot_map_raw = os.environ.get("SLOT_MAP_JSON")
    if slot_map_raw:
        slot_map = json.loads(slot_map_raw)
        if job_index >= len(slot_map):
            sys.exit(f"FATAL: JOB_COMPLETION_INDEX={job_index} >= "
                     f"len(SLOT_MAP_JSON)={len(slot_map)}")
        hp_rank, seed_idx = slot_map[job_index]
    else:
        hp_rank, seed_idx = divmod(job_index, 2)  # 0..9 → (0..4, 0..1)
    if hp_rank >= 5 or seed_idx >= len(seeds):
        sys.exit(f"FATAL: slot out of range "
                 f"(idx={job_index}, hp_rank={hp_rank}, seed_idx={seed_idx})")

    seed = int(seeds[seed_idx])

    # 1) Load baseline config.
    base_cfg_path = SWEEP_BASELINE_DIR / f"{arch}_{lang}.yaml"
    cfg = yaml.safe_load(base_cfg_path.read_text())

    # 2) Load HP rank from sweep winners.
    winners_path = SWEEP_WINNERS_DIR / f"{arch}_{lang}.json"
    winners = json.loads(winners_path.read_text())
    if hp_rank >= len(winners["configs"]):
        sys.exit(f"FATAL: only {len(winners['configs'])} winners in "
                 f"{winners_path}, but hp_rank={hp_rank}")
    hp = winners["configs"][hp_rank]

    # 3) Overlay HP.
    _apply_hp_overrides(cfg, hp, phys_batch)

    # 4) Swap training corpus.
    cfg["data"]["training_corpus"] = _training_corpus_path(lang, intervention)
    cfg["data"]["source_corpus"] = cfg["data"]["training_corpus"]
    # test_corpus already points at /raw/<lang>/test_10M/ in the sweep base.

    # 5) Production-mode training settings.
    prod_epochs = int(cfg["training"].get("production_epochs", 30))
    cfg["training"]["epochs"] = prod_epochs

    # 6) Seed + identity. (Resolved before the schedule so RESUME mode can
    # inspect the existing output_dir for the step already reached.)
    cfg["random_seed"] = seed
    run_id = f"{arch}-{lang}-{intervention}-h{hp_rank}-s{seed}"
    cfg["experiment_name"] = run_id
    output_dir_rel = f"models/production/{run_id}"
    cfg["training"]["output_dir"] = output_dir_rel

    # 7) WandB project — split production from sweeps.
    cfg["logging"]["wandb_project"] = os.environ.get(
        "WANDB_PROJECT_PROD", "subject-drop-production"
    )

    # 8) Tell the registry plumbing about this run.
    os.environ["REGISTRY_RUN_KIND"] = "production"
    os.environ["REGISTRY_LANG"] = lang
    os.environ["REGISTRY_CONDITION"] = intervention
    os.environ["REGISTRY_SEED"] = str(seed)
    os.environ["REGISTRY_RUN_ID"] = run_id

    # 9) Fail-fast cache check — production pods only READ the cache.
    # If it's missing, the prepare-caches Job hasn't completed yet;
    # exit non-zero so the K8s Job retries, by which point the cache
    # is hopefully populated. This is the same pattern the sweep used
    # (see k8s/job-sweep-*-en.yaml) — letting cli.run trigger tokenize
    # inline causes HF builder-lock races when multiple pods share a
    # cache key (the sweep-1 race incident).
    sys.path.insert(0, str(REPO_ROOT))
    from model_foundry.cache_keys import compute_cache_key  # noqa: E402
    from model_foundry.checkpoint_schedule import (  # noqa: E402
        compute_checkpoint_schedule,
        read_num_chunks,
    )

    corpus_abs = str(REPO_ROOT / cfg["data"]["training_corpus"])
    tok_abs = str(REPO_ROOT / cfg["tokenizer"]["output_dir"])
    cache_key = compute_cache_key(
        corpus_abs, tok_abs, cfg["data"]["max_sequence_length"],
        cfg.get("dataset_manipulation") or [],
    )
    tokenized = Path(f"/mnt/data/tokenized/{cache_key}/train/dataset_info.json")
    chunked_dir = Path(f"/mnt/data/chunked/{cache_key}")
    chunked = chunked_dir / "dataset_info.json"
    if not tokenized.exists():
        sys.exit(
            f"FAIL: tokenized cache missing at {tokenized}. "
            f"Run k8s/job-prepare-production-caches-{lang}.yaml first."
        )
    if not chunked.exists():
        sys.exit(
            f"FAIL: chunked cache missing at {chunked}. "
            f"Run k8s/job-prepare-production-caches-{lang}.yaml first."
        )

    # 10) Checkpoint schedule via the shared helper, off the REAL chunk
    # count (read from the content-addressed cache — never the old
    # hardcoded 127000 estimate).
    num_chunks = read_num_chunks(str(chunked_dir))
    grad_accum = int(cfg["training"]["gradient_accumulation_steps"])

    resume_mode = os.environ.get("RESUME", "").strip().lower() in (
        "1", "true", "yes", "y"
    )
    if resume_mode:
        # RESUME=1 means "resume if possible, fresh if new, refuse if
        # ambiguous" — so one Job can mix resumable and never-ran slots.
        _out_dir = REPO_ROOT / output_dir_rel
        if _latest_resume_step(_out_dir) == 0:
            import re as _re
            stale_steps = sorted(
                int(m.group(1))
                for p in (_out_dir.glob("checkpoint-*") if _out_dir.exists() else [])
                if (m := _re.search(r"checkpoint-(\d+)$", p.name))
            )
            if stale_steps:
                # Weights-only checkpoints, no training_state anywhere.
                # Two very different causes:
                #   (a) a FRESH run died before reaching its first
                #       designated resume anchor (~ep7 waypoint) — only
                #       early weights-only anchors exist. Restarting fresh
                #       is safe and deterministic: the same seed re-walks
                #       the same trajectory and overwrites the same
                #       anchors (live-validated on the lstm fresh test,
                #       which restarted from 0 once and finished perfect).
                #   (b) a stale PRE-FIX truncated run — it ran PAST where
                #       the current schedule designates resume state, yet
                #       carries none. Interleaving a fresh run into that
                #       dir would silently mix schedules — refuse.
                _, _fresh_rss = compute_checkpoint_schedule(
                    num_chunks=num_chunks,
                    phys_batch=phys_batch,
                    grad_accum=grad_accum,
                    epochs=prod_epochs,
                    seq_len=cfg["data"]["max_sequence_length"],
                )
                first_resume_anchor = min(_fresh_rss) if _fresh_rss else 0
                if stale_steps[-1] >= first_resume_anchor:
                    sys.exit(
                        f"[FATAL] RESUME=1 but {output_dir_rel} has "
                        f"{len(stale_steps)} checkpoint(s) reaching step "
                        f"{stale_steps[-1]} (>= first resume anchor "
                        f"{first_resume_anchor}) and none is restartable "
                        f"(no training_state.pt) — stale pre-fix run. "
                        f"Wipe the dir to rerun fresh."
                    )
                print(f"  [RESUME] only pre-waypoint weights-only "
                      f"checkpoints found (max step {stale_steps[-1]} < "
                      f"first resume anchor {first_resume_anchor}) — "
                      f"restarting fresh; they'll be overwritten "
                      f"deterministically")
            else:
                print("  [RESUME] no prior run for this slot — falling "
                      "back to a fresh start (full schedule incl. step 0)")
            resume_mode = False
    if resume_mode:
        # RESUME IN PLACE: keep the existing output_dir, resume from its
        # newest full-state checkpoint, and emit a BACK-HALF-ONLY schedule
        # of anchors strictly past where the run left off. train_steps /
        # warmup_steps are nulled so the trainer recomputes them from the
        # (real) data + epochs at startup.
        resume_step = _latest_resume_step(REPO_ROOT / output_dir_rel)
        schedule, resume_state_steps = compute_checkpoint_schedule(
            num_chunks=num_chunks,
            phys_batch=phys_batch,
            grad_accum=grad_accum,
            epochs=prod_epochs,
            seq_len=cfg["data"]["max_sequence_length"],
            back_half_only=True,
            resume_step=resume_step,
        )
        cfg["training"]["resume_from_checkpoint"] = True
        cfg["training"]["train_steps"] = None
        cfg["training"]["warmup_steps"] = None
        # The full back-half list saves resume state (intersected with the
        # schedule by the loop). resume_state_steps is the back-half subset
        # strictly > resume_step.
        cfg["training"]["resume_state_steps"] = resume_state_steps
        print(f"  [RESUME] resume_step={resume_step} "
              f"back_half_anchors={len(schedule)} "
              f"resume_state_anchors={len(resume_state_steps)}")
    else:
        schedule, resume_state_steps = compute_checkpoint_schedule(
            num_chunks=num_chunks,
            phys_batch=phys_batch,
            grad_accum=grad_accum,
            epochs=prod_epochs,
            seq_len=cfg["data"]["max_sequence_length"],
        )
        cfg["training"]["resume_from_checkpoint"] = False
        # Fresh run: explicit {ep7 waypoint, midpoint, all back-half}.
        cfg["training"]["resume_state_steps"] = resume_state_steps

    cfg["training"]["checkpoint_schedule"] = schedule
    cfg["training"]["auto_generate_checkpoints"] = False
    # resume_state_steps takes precedence in the loop; keep the legacy
    # last-N as a documented fallback (unused when resume_state_steps set).
    cfg["training"]["save_resume_state_last_n"] = int(
        os.environ.get("SAVE_RESUME_LAST_N", "3"))

    # 11) Write final config + invoke cli.run.
    final_path = Path("/tmp/run_config.yaml")
    final_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"=== production_agent ===")
    print(f"  arch={arch} lang={lang} intervention={intervention}")
    print(f"  hp_rank={hp_rank} (ppl={hp.get('min_held_out_perplexity'):.3f})")
    print(f"  seed={seed}  phys={phys_batch}  eff={hp['effective_batch_size']}")
    print(f"  grad_accum={cfg['training']['gradient_accumulation_steps']}")
    print(f"  epochs={prod_epochs}  num_chunks={num_chunks}  "
          f"schedule_anchors={len(schedule)}  "
          f"resume_state_anchors={len(resume_state_steps)}  "
          f"resume_mode={resume_mode}")
    print(f"  run_id={run_id}")
    print(f"  config -> {final_path}")
    sys.stdout.flush()

    cmd = [sys.executable, "-m", "model_foundry.cli", "run", str(final_path)]
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc == 0:
        Path("/tmp/run_succeeded").write_text(run_id + "\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
