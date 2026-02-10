#!/usr/bin/env python3
"""
W&B Sweep agent for pronoun recovery hyperparameter search.

Called by `wandb agent` — reads hyperparameters from wandb.config,
trains a single model (metrics-only, no checkpoints), and logs
eval metrics to W&B.

Setup:
    wandb sweep --entity thmorton-uc-san-diego --project pronoun-recovery \
        configs/wandb_sweep.yaml
    # → returns SWEEP_ID

    wandb agent --count 1 thmorton-uc-san-diego/pronoun-recovery/SWEEP_ID
"""

import logging
import os
import time
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForTokenClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep_agent")

# Defaults — overridden by env vars in K8s
CONFIG_PATH = os.environ.get(
    "SWEEP_CONFIG", "configs/pronoun_recovery_it_train.yaml"
)
DATA_PATH = os.environ.get(
    "SWEEP_DATA_PATH", "data/pronoun_recovery/annotations/it/r1_final_clean.jsonl"
)
OUTPUT_DIR = os.environ.get(
    "SWEEP_OUTPUT_DIR", "data/pronoun_recovery/models/it_sweep"
)


def main():
    from analysis.pronoun_recovery.config import ModelTrainingConfig, load_config
    from analysis.pronoun_recovery.constants import ALL_LABELS, NUM_LABELS
    from analysis.pronoun_recovery.model.dataset import (
        build_dataset,
        compute_class_weights,
    )
    from analysis.pronoun_recovery.model.trainer import PronounRecoveryTrainer

    # W&B init — sweep controller injects hyperparams
    run = wandb.init()
    alpha = wandb.config.weight_alpha
    lr = wandb.config.learning_rate
    gamma = getattr(wandb.config, "focal_gamma", 0.0)

    run_tag = f"a{alpha}_lr{lr:.0e}_g{gamma}"
    logger.info("Starting run: %s", run_tag)

    # Load config
    cfg = load_config(CONFIG_PATH, ModelTrainingConfig)
    cfg.annotation_data_path = Path(DATA_PATH)
    cfg.weight_alpha = alpha
    cfg.focal_gamma = gamma
    cfg.fp16 = torch.cuda.is_available()
    # Let HF Trainer log to our wandb run
    cfg.wandb_project = "pronoun-recovery"

    run_dir = Path(OUTPUT_DIR) / run_tag
    cfg.output_path = str(run_dir)

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    annot_dataset = build_dataset(
        data_path=cfg.annotation_data_path,
        tokenizer=tokenizer,
        max_length=cfg.max_seq_length,
        max_samples=0,
        data_type="checkpoint",
    )

    trainer_stub = PronounRecoveryTrainer.__new__(PronounRecoveryTrainer)
    trainer_stub.config = cfg
    train_dataset, eval_dataset = trainer_stub._stratified_split(
        annot_dataset, test_size=0.1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Data: %d train, %d eval",
                device, len(train_dataset), len(eval_dataset))

    # Train (metrics-only, no checkpoints saved)
    class_weights = compute_class_weights(train_dataset, alpha=alpha)

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=NUM_LABELS,
        id2label={i: l for i, l in enumerate(ALL_LABELS)},
        label2id={l: i for i, l in enumerate(ALL_LABELS)},
    )

    trainer = PronounRecoveryTrainer(cfg)
    trainer.tokenizer = tokenizer

    start = time.time()
    best_metrics = trainer._train_phase(
        phase_name=run_tag,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=run_dir / "checkpoints",
        learning_rate=lr,
        batch_size=cfg.phase_b_batch_size,
        num_epochs=cfg.phase_b_epochs,
        patience=cfg.phase_b_patience,
        class_weights=class_weights,
        focal_gamma=gamma,
        save_checkpoints=False,
    )
    elapsed = time.time() - start

    # Log summary metrics to W&B
    wandb.summary["train_time_min"] = elapsed / 60
    if isinstance(best_metrics, dict) and "eval_f1" in best_metrics:
        wandb.summary["best_f1"] = best_metrics["eval_f1"]
        wandb.summary["best_precision"] = best_metrics.get("eval_precision", 0)
        wandb.summary["best_recall"] = best_metrics.get("eval_recall", 0)
        wandb.summary["best_epoch"] = best_metrics.get("epoch", 0)
        wandb.summary["best_eval_loss"] = best_metrics.get("eval_loss", 0)
        logger.info(
            "%s complete: best F1=%.4f (P=%.4f R=%.4f) at epoch %.0f, %.1f min",
            run_tag,
            best_metrics["eval_f1"],
            best_metrics.get("eval_precision", 0),
            best_metrics.get("eval_recall", 0),
            best_metrics.get("epoch", 0),
            elapsed / 60,
        )
    else:
        logger.warning("%s: no eval metrics returned", run_tag)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
