#!/usr/bin/env python3
"""
WandB sweep agent for LM hyperparameter search.

Invoked by ``wandb agent`` once per trial. Reads HP values from
``wandb.config``, overlays them onto the per-(arch,lang) baseline YAML,
trains for the proxy horizon (default 3 epochs), computes both proxy
metrics (training loss + held-out perplexity), logs them to WandB, and
writes a registry record with ``run_kind="hp_sweep"``.

The sweep YAML controls:

- ``method``: ``bayes`` (server-side Gaussian-process over the HP space)
- ``metric.name``: ``proxy/held_out_perplexity`` (the Bayes ranker)
- ``early_terminate.type``: ``hyperband`` (kills unpromising trials
  at step checkpoints so compute focuses on survivors)
- ``parameters``: per-HP distributions — everything in
  ``_SWEEP_HP_FIELDS`` below is considered.

Constants the sweep YAML must inject via `parameters.*.value`:

- ``arch`` — one of gpt2_small / gpt2_medium / gpt2_large / bert_large
             / lstm / mamba_370m
- ``lang`` — en / es
- ``base_config`` — relative path to the per-(arch,lang) baseline YAML
  (defaults to ``configs/sweeps/baselines/{arch}_{lang}.yaml`` if absent)

No checkpoints are written during sweep trials; the proxy metrics are
the only output. Registry record + env snapshot still land on S3 for
reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure the project root is on sys.path so `model_foundry` imports
# resolve when this script is invoked directly (e.g. via
# ``wandb agent``).
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from model_foundry import registry as _registry
from model_foundry.cache_keys import compute_cache_key
from model_foundry.config import ExperimentConfig
from model_foundry.trainer import Trainer
from model_foundry.utils import find_project_root, get_git_commit_hash

logger = logging.getLogger("sweep_agent_lm")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# HP fields the sweep may vary. If an HP isn't in ``wandb.config``, the
# baseline YAML's value is kept.
_SWEEP_HP_FIELDS = (
    "learning_rate",
    "warmup_ratio",
    "dropout",
    "attention_dropout",
    "effective_batch_size",
    "adam_beta2",
)

PROXY_METRIC_NAME = "held_out_perplexity"


def main() -> None:
    # wandb.init populates wandb.config from the sweep controller's
    # sampled HPs plus any ``value``-fixed constants from the sweep YAML.
    run = wandb.init(job_type="sweep")
    if run is None:
        raise RuntimeError("wandb.init returned None — no sweep context")

    cfg_dict, identity = _prepare_config(run)
    config = ExperimentConfig(**cfg_dict)

    base_dir = find_project_root(__file__)

    _register_start(config, identity, run, base_dir)

    # Train — on failure, record it and re-raise so WandB marks the
    # trial as crashed (not counted in the Bayes posterior).
    trainer: Optional[Trainer] = None
    try:
        trainer = Trainer(config, base_dir)
        trainer.train()
    except Exception as e:  # noqa: BLE001
        logger.exception("trial failed during training")
        wandb.log({
            "proxy/final_training_loss": float("nan"),
            f"proxy/{PROXY_METRIC_NAME}": float("nan"),
        })
        _registry.register_run_end(
            run_id=identity["run_id"], arch=identity["arch"],
            lang=identity["lang"], condition=identity["condition"],
            status="FAILED", failure_reason=str(e)[:500],
        )
        raise

    final_train_loss = _extract_final_training_loss(trainer)
    held_out_ppl = _compute_held_out_perplexity(trainer, config)

    wandb.log({
        "proxy/final_training_loss": float(final_train_loss),
        f"proxy/{PROXY_METRIC_NAME}": float(held_out_ppl),
    })
    logger.info(
        "trial DONE: train_loss=%.4f held_out_ppl=%.4f",
        final_train_loss, held_out_ppl,
    )

    _register_end(identity, run, trainer, final_train_loss, held_out_ppl)


# ---------- config preparation ----------

def _prepare_config(run) -> tuple[dict, dict]:
    """Load the baseline YAML, overlay sampled HPs, return (config_dict, identity)."""
    wc = dict(run.config)
    arch = wc.get("arch")
    lang = wc.get("lang")
    if not arch or not lang:
        raise RuntimeError(
            "Sweep config must fix `arch` and `lang` as constants "
            "(parameters.arch.value, parameters.lang.value)."
        )

    base_yaml_rel = wc.get("base_config") or f"configs/sweeps/baselines/{arch}_{lang}.yaml"
    base_dir = find_project_root(__file__)
    with open(Path(base_dir) / base_yaml_rel, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    _apply_hp_overrides(cfg_dict, wc)
    _apply_sweep_overrides(cfg_dict, run.id)

    condition = wc.get("condition", "baseline")  # sweeps always run on baseline corpus
    seed = int(cfg_dict.get("random_seed", 0))
    run_id = f"sweep-{run.sweep_id}-{run.id}"

    # Tell downstream registry/wandb helpers that this is a sweep trial.
    os.environ["REGISTRY_RUN_KIND"] = "hp_sweep"
    os.environ["REGISTRY_LANG"] = lang
    os.environ["REGISTRY_CONDITION"] = condition
    os.environ["REGISTRY_SEED"] = str(seed)
    os.environ["REGISTRY_RUN_ID"] = run_id

    identity = {
        "run_id": run_id, "arch": arch, "lang": lang,
        "condition": condition, "seed": seed, "run_kind": "hp_sweep",
    }
    return cfg_dict, identity


def _apply_hp_overrides(cfg: dict, wc: dict) -> None:
    """Overlay sampled HPs onto the baseline config dict in-place."""
    if "learning_rate" in wc:
        cfg["training"]["learning_rate"] = float(wc["learning_rate"])
    if "warmup_ratio" in wc:
        cfg["training"]["warmup_ratio"] = float(wc["warmup_ratio"])
    if "adam_beta2" in wc:
        cfg["training"]["adam_beta2"] = float(wc["adam_beta2"])
    if "effective_batch_size" in wc:
        # Hold physical batch_size fixed (GPU-memory bound); compute
        # gradient_accumulation_steps to match the requested effective
        # batch size.
        eff = int(wc["effective_batch_size"])
        phys = int(cfg["data"]["batch_size"])
        if eff < phys:
            cfg["data"]["batch_size"] = eff
            cfg["training"]["gradient_accumulation_steps"] = 1
        else:
            cfg["training"]["gradient_accumulation_steps"] = max(1, eff // phys)
    if "dropout" in wc:
        val = float(wc["dropout"])
        for block in ("transformer", "rnn", "mamba"):
            if cfg["model"].get(block):
                cfg["model"][block]["dropout"] = val
    if "attention_dropout" in wc:
        if cfg["model"].get("transformer"):
            cfg["model"]["transformer"]["attention_dropout"] = float(wc["attention_dropout"])


def _apply_sweep_overrides(cfg: dict, wandb_run_id: str) -> None:
    """Force sweep-trial-specific settings that never vary."""
    cfg["experiment_name"] = f"sweep-{wandb_run_id}"
    cfg["training"]["output_dir"] = f"models/sweeps/{wandb_run_id}"
    cfg["training"]["epochs"] = int(cfg["training"].get("sweep_epochs", 3))
    cfg["training"]["train_steps"] = None  # auto from epochs
    cfg["training"]["checkpoint_schedule"] = []
    cfg["training"]["auto_generate_checkpoints"] = False
    cfg["training"]["resume_from_checkpoint"] = False


# ---------- registry plumbing ----------

def _register_start(config: ExperimentConfig, identity: dict, run, base_dir: str) -> None:
    try:
        _registry.register_run_start(
            **identity,
            config_hash=_config_hash(config),
            git_commit=get_git_commit_hash(),
            cache_key=_safe_cache_key(config, base_dir),
            tokenizer_dir=config.tokenizer.output_dir,
            docker_image=os.environ.get("DOCKER_IMAGE"),
            hyperparameters=_flat_hp_snapshot(config),
            wandb_run_id=run.id,
            wandb_project=run.project,
            wandb_sweep_id=run.sweep_id,
            train_steps=config.training.train_steps,
        )
    except Exception as e:  # noqa: BLE001 — non-fatal
        logger.warning("register_run_start raised: %s", e)


def _register_end(identity: dict, run, trainer: Trainer,
                   final_train_loss: float, held_out_ppl: float) -> None:
    steps = getattr(trainer.training_loop, "global_step", None) if trainer else None
    try:
        _registry.register_run_end(
            run_id=identity["run_id"], arch=identity["arch"],
            lang=identity["lang"], condition=identity["condition"],
            status="COMPLETE",
            final_loss=float(final_train_loss) if not math.isnan(final_train_loss) else None,
            steps_completed=steps,
        )
        # Stamp the sweep-specific fields on the record.
        _registry._safe_merge(  # noqa: SLF001 — private but stable within package
            identity["arch"], identity["lang"], identity["condition"], identity["run_id"],
            {
                "hp_proxy_metric": PROXY_METRIC_NAME,
                "hp_proxy_score": float(held_out_ppl) if not math.isnan(held_out_ppl) else None,
            },
            op="hp_proxy_score",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("register_run_end raised: %s", e)


# ---------- metric extraction ----------

def _extract_final_training_loss(trainer: Trainer) -> float:
    """Best-effort pull of the last per-optimizer-step mean loss the
    training loop saw. Returns nan if the loop never recorded one."""
    tl = getattr(trainer, "training_loop", None)
    if tl is None:
        return float("nan")
    for attr in ("final_loss", "last_mean_loss", "_last_mean_loss"):
        v = getattr(tl, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    # Fallback: mean of all recorded losses this run (per-micro-batch).
    losses = getattr(tl, "epoch_losses", None) or []
    if losses:
        try:
            import torch
            if isinstance(losses[0], torch.Tensor):
                return float(torch.stack(losses).mean().item())
            return float(sum(losses) / len(losses))
        except Exception:  # noqa: BLE001
            return float("nan")
    return float("nan")


def _compute_held_out_perplexity(trainer: Trainer, config: ExperimentConfig) -> float:
    """One pass over the test corpus → exp(mean NLL). Uses the same
    content-addressed cache paths as training, so the test split must
    have been tokenized once (tokenize-dataset writes both train/ and
    test/)."""
    if not config.data.test_corpus:
        logger.warning("no test_corpus in config; skipping held_out_perplexity")
        return float("nan")

    test_dir = os.path.join(trainer.data_processor.tokenized_data_dir, "test")
    if not os.path.isdir(test_dir):
        logger.warning("test dir missing at %s; skipping held_out_perplexity", test_dir)
        return float("nan")

    from datasets import load_from_disk
    ds = load_from_disk(test_dir)
    chunk_size = config.data.max_sequence_length

    # Chunk the test sequences the same way training does.
    buf: list[int] = []
    chunks: list[list[int]] = []
    for ex in ds:
        buf.extend(ex["input_ids"])
        while len(buf) >= chunk_size:
            chunks.append(buf[:chunk_size])
            buf = buf[chunk_size:]
    if not chunks:
        return float("nan")

    tensor = torch.tensor(chunks, dtype=torch.long)
    loader = DataLoader(tensor, batch_size=config.data.batch_size, shuffle=False)

    device = next(trainer.model.parameters()).device
    trainer.model.eval()
    total_nll = 0.0
    total_tokens = 0
    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    with torch.no_grad():
        for batch in loader:
            input_ids = batch.to(device, non_blocking=True)
            inputs = {"input_ids": input_ids, "labels": input_ids}
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = trainer.model(**inputs)
            else:
                out = trainer.model(**inputs)
            # outputs.loss is the mean-NLL over the batch's token positions
            nll = out.loss.float().item()
            ntok = input_ids.numel()
            total_nll += nll * ntok
            total_tokens += ntok
    trainer.model.train()

    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


# ---------- helpers ----------

def _config_hash(config: ExperimentConfig) -> str:
    return hashlib.md5(
        json.dumps(config.model_dump(), sort_keys=True).encode()
    ).hexdigest()


def _safe_cache_key(config: ExperimentConfig, base_dir: str) -> Optional[str]:
    corpus = config.data.training_corpus
    if not os.path.isabs(corpus):
        corpus = os.path.join(base_dir, corpus)
    tok = config.tokenizer.output_dir
    if not os.path.isabs(tok):
        tok = os.path.join(base_dir, tok)
    try:
        return compute_cache_key(
            corpus, tok, config.data.max_sequence_length,
            config.dataset_manipulation or [],
        )
    except FileNotFoundError:
        return None


def _flat_hp_snapshot(config: ExperimentConfig) -> dict[str, Any]:
    t = config.training
    d = config.data
    hp = {
        "learning_rate": t.learning_rate,
        "adam_beta1": t.adam_beta1,
        "adam_beta2": t.adam_beta2,
        "warmup_ratio": t.warmup_ratio,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "effective_batch_size": d.batch_size * t.gradient_accumulation_steps,
        "max_sequence_length": d.max_sequence_length,
    }
    if config.model.transformer:
        hp["dropout"] = config.model.transformer.dropout
        hp["attention_dropout"] = config.model.transformer.attention_dropout
    elif config.model.rnn:
        hp["dropout"] = config.model.rnn.dropout
    elif config.model.mamba:
        hp["dropout"] = config.model.mamba.dropout
    return hp


if __name__ == "__main__":
    main()
