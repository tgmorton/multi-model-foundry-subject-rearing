#!/usr/bin/env python3
"""
W&B Sweep agent for pronoun recovery hyperparameter search.

Called by `wandb agent` — reads hyperparameters from wandb.config,
trains a single model, evaluates with threshold sweep, and logs
all metrics to W&B.

Setup:
    wandb sweep --entity thmorton-uc-san-diego --project pronoun-recovery \
        configs/wandb_sweep.yaml
    # → returns SWEEP_ID

    wandb agent --count 3 thmorton-uc-san-diego/pronoun-recovery/SWEEP_ID
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.nn import functional as F
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
    "SWEEP_DATA_PATH", "data/pronoun_recovery/annotations/it/checkpoint.jsonl"
)
OUTPUT_DIR = os.environ.get(
    "SWEEP_OUTPUT_DIR", "data/pronoun_recovery/models/it_sweep"
)
THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]


def evaluate_model(model_path, eval_dataset, thresholds, device):
    """Run inference at multiple thresholds and compute metrics."""
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval().to(device)

    all_true, all_probs = [], []
    for example in eval_dataset:
        input_ids = example["input_ids"].unsqueeze(0).to(device)
        attention_mask = example["attention_mask"].unsqueeze(0).to(device)
        labels = example["labels"]
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        labels_np = labels.numpy() if hasattr(labels, "numpy") else np.array(labels)
        for prob_row, lid in zip(probs, labels_np):
            if lid == -100:
                continue
            all_true.append(int(lid))
            all_probs.append(prob_row)

    all_true = np.array(all_true)
    all_probs = np.array(all_probs)

    results = []
    for thresh in thresholds:
        p_non_none = 1.0 - all_probs[:, 0]
        preds = np.argmax(all_probs, axis=-1)
        preds[p_non_none < thresh] = 0

        true_det = all_true > 0
        pred_det = preds > 0

        tp = int((true_det & (preds == all_true)).sum())
        fp = int((pred_det & (preds != all_true)).sum())
        fn = int((true_det & (~pred_det | (preds != all_true))).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        tp_d = int((true_det & pred_det).sum())
        fp_d = int((~true_det & pred_det).sum())
        fn_d = int((true_det & ~pred_det).sum())
        det_p = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0
        det_r = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0
        det_f1 = 2 * det_p * det_r / (det_p + det_r) if (det_p + det_r) > 0 else 0

        results.append({
            "threshold": thresh,
            "precision": prec, "recall": rec, "f1": f1,
            "det_precision": det_p, "det_recall": det_r, "det_f1": det_f1,
            "fp": fp_d, "fn": fn_d,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


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
    # Disable HF Trainer's own W&B — we manage the run ourselves
    cfg.wandb_project = None

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

    # Train
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
    best_ckpt = trainer._train_phase(
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
    )
    elapsed = time.time() - start
    wandb.log({"train_time_min": elapsed / 60})
    logger.info("Training took %.1f min", elapsed / 60)

    # Save final model
    final_dir = run_dir / "final"
    final_model = AutoModelForTokenClassification.from_pretrained(
        best_ckpt, num_labels=NUM_LABELS
    )
    final_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Clean up checkpoints
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(str(ckpt_dir))

    # Threshold sweep
    thresh_results = evaluate_model(str(final_dir), eval_dataset, THRESHOLDS, device)

    best = max(thresh_results, key=lambda x: x["f1"])
    wandb.summary["best_f1"] = best["f1"]
    wandb.summary["best_threshold"] = best["threshold"]
    wandb.summary["best_precision"] = best["precision"]
    wandb.summary["best_recall"] = best["recall"]
    wandb.summary["best_det_f1"] = best["det_f1"]

    # Log full threshold curve
    for r in thresh_results:
        wandb.log({
            "threshold": r["threshold"],
            "thresh_f1": r["f1"],
            "thresh_precision": r["precision"],
            "thresh_recall": r["recall"],
            "thresh_det_f1": r["det_f1"],
            "thresh_fp": r["fp"],
            "thresh_fn": r["fn"],
        })

    # Save results JSON
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "alpha": alpha,
            "learning_rate": lr,
            "threshold_results": thresh_results,
            "best": best,
            "train_time_min": elapsed / 60,
        }, f, indent=2)

    logger.info(
        "%s complete: best F1=%.3f at thresh=%.2f (P=%.3f R=%.3f)",
        run_tag, best["f1"], best["threshold"],
        best["precision"], best["recall"],
    )

    del model, final_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
