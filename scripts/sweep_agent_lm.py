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

    # Persist final weights FIRST, before any metric compute that might
    # fail. This way post-hoc perplexity (or any later eval) is always
    # possible even if _compute_held_out_perplexity errors below.
    final_ckpt_dir = _save_final_checkpoint(trainer, config, base_dir)
    logger.info("final checkpoint saved to %s", final_ckpt_dir)

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


def _save_final_checkpoint(trainer: Trainer, config: ExperimentConfig, base_dir: str) -> str:
    """Save model + tokenizer to models/sweeps/<run>/final/ so any later
    eval (post-hoc perplexity, BLiMP, etc.) can still be recomputed even
    if this trial's in-process metric code errors out."""
    output_dir = config.training.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    try:
        trainer.model.save_pretrained(final_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("model.save_pretrained failed: %s", e)
    try:
        trainer.tokenizer.save_pretrained(final_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("tokenizer.save_pretrained failed: %s", e)
    return final_dir


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
    # Dropout / attention_dropout may live at the flat top of the model
    # block (legacy GPT-2 baseline shape) OR nested inside transformer/rnn/mamba
    # (new canonical shape — see ModelConfig._backwards_compat_flat_config).
    # The pydantic validator rewrites flat→nested, but it runs AFTER this
    # overlay, so we must touch whichever shape the YAML is in.
    _TRANSFORMER_FLAT_MARKERS = {"layers", "attention_heads", "intermediate_hidden_size"}
    _RNN_FLAT_MARKERS = {"num_layers", "rnn_type"}
    _MAMBA_FLAT_MARKERS = {"d_model", "n_layers", "d_state"}
    model = cfg["model"]
    def _apply_model_field(field: str, val: float) -> None:
        nested_hit = False
        for block in ("transformer", "rnn", "mamba"):
            if isinstance(model.get(block), dict):
                model[block][field] = val
                nested_hit = True
        if nested_hit:
            return
        # Flat shape: set directly on the model dict if this field applies.
        # Only the transformer family carries attention_dropout.
        if field == "attention_dropout":
            if _TRANSFORMER_FLAT_MARKERS & model.keys():
                model[field] = val
        else:
            if (_TRANSFORMER_FLAT_MARKERS | _RNN_FLAT_MARKERS | _MAMBA_FLAT_MARKERS) & model.keys():
                model[field] = val
    if "dropout" in wc:
        _apply_model_field("dropout", float(wc["dropout"]))
    if "attention_dropout" in wc:
        _apply_model_field("attention_dropout", float(wc["attention_dropout"]))


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
    """One pass over the test corpus → exp(mean NLL).

    Dispatches by architecture:
    - Causal LMs (gpt2, lstm, rnn, gru, mamba): standard teacher-forcing
      where labels=input_ids and every position contributes to the loss.
    - Masked LMs (bert): deterministic 15% mask pattern (fixed seed so
      every trial sees the same mask positions, making scores comparable
      across the sweep), loss over masked positions only.

    Note: these perplexities live on different scales across arch
    families and are not directly comparable between-arch. The sweep
    ranks trials within a single arch so that's fine."""
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
    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    arch = config.model.architecture
    is_mlm = arch == "bert"

    if is_mlm:
        mask_token_id = getattr(trainer.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            logger.warning("BERT tokenizer has no mask_token_id; skipping held_out_perplexity")
            return float("nan")
        # Deterministic mask: same seed → same masked positions for every
        # trial in this sweep → rankings are noise-free on the mask axis.
        g = torch.Generator()
        g.manual_seed(20260423)
        mlm_prob = getattr(config.training, "mlm_probability", 0.15) or 0.15

    trainer.model.eval()
    total_nll = 0.0
    total_counts = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch
            if is_mlm:
                mask = torch.zeros_like(input_ids, dtype=torch.bool)
                mask.bernoulli_(mlm_prob, generator=g)
                labels = input_ids.clone()
                labels[~mask] = -100  # HF ignores -100 in the CE loss
                masked_inputs = input_ids.clone()
                masked_inputs[mask] = mask_token_id
                masked_inputs = masked_inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                n_contrib = int(mask.sum().item())
                if n_contrib == 0:
                    continue
                fwd_kwargs = {"input_ids": masked_inputs, "labels": labels}
            else:
                ids = input_ids.to(device, non_blocking=True)
                fwd_kwargs = {"input_ids": ids, "labels": ids}
                n_contrib = int(ids.numel())

            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = trainer.model(**fwd_kwargs)
            else:
                out = trainer.model(**fwd_kwargs)
            # out.loss is mean NLL over the positions that contributed
            # (all tokens for causal LM; masked-only for MLM thanks to -100).
            nll = out.loss.float().item()
            total_nll += nll * n_contrib
            total_counts += n_contrib
    trainer.model.train()

    if total_counts == 0:
        return float("nan")
    return math.exp(total_nll / total_counts)


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
