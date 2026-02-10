#!/usr/bin/env python3
"""
Hyperparameter sweep for pronoun recovery token classifier.

Sweeps weight_alpha (class weight exponent), learning rate, and inference
threshold to find the optimal combination.  In parallel mode (K8s indexed
job), each pod trains a single (alpha, lr) pair.  Locally, it loops over
all alphas with a single LR.

Usage (local):
    python scripts/sweep_weight_alpha.py \
        --config configs/pronoun_recovery_it_train.yaml \
        --output-dir data/pronoun_recovery/models/it_sweep

Usage (K8s / single alpha+lr):
    python scripts/sweep_weight_alpha.py \
        --config configs/pronoun_recovery_it_train.yaml \
        --data-path /mnt/data/pronoun_recovery/annotations/it/checkpoint.jsonl \
        --output-dir /mnt/data/pronoun_recovery/models/it_sweep \
        --alphas 0.5 --learning-rate 1e-5 \
        --wandb-project pronoun-recovery \
        --wandb-entity thmorton-uc-san-diego
"""

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep")


# ── Threshold evaluation ────────────────────────────────────────────────


def evaluate_model(model_path, eval_dataset, thresholds, device):
    """Run inference at multiple thresholds and compute precision/recall/F1."""
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

        # Full metrics (detection + correct label)
        tp = int((true_det & (preds == all_true)).sum())
        fp = int((pred_det & (preds != all_true)).sum())
        fn = int((true_det & (~pred_det | (preds != all_true))).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        # Detection-only metrics (ignoring label correctness)
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


# ── Main sweep ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--config", required=True, help="Base training config YAML")
    parser.add_argument("--output-dir", required=True, help="Sweep output directory")
    parser.add_argument("--data-path", default=None,
                        help="Override annotation_data_path from config")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override learning rate (default: use config value)")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Focal loss gamma (0.0 = standard CE)")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name (omit to disable)")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity (team/user)")
    args = parser.parse_args()

    # Lazy imports so --help is fast
    from analysis.pronoun_recovery.config import ModelTrainingConfig, load_config
    from analysis.pronoun_recovery.constants import ALL_LABELS, NUM_LABELS
    from analysis.pronoun_recovery.model.dataset import (
        build_dataset,
        compute_class_weights,
    )
    from analysis.pronoun_recovery.model.trainer import PronounRecoveryTrainer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B setup ──
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_RUN_GROUP"] = "weight_alpha_sweep"
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

    # ── Load base config ──
    cfg_base = load_config(args.config, ModelTrainingConfig)
    if args.data_path:
        cfg_base.annotation_data_path = Path(args.data_path)

    lr = args.learning_rate if args.learning_rate else cfg_base.phase_b_lr

    # ── Load tokenizer + dataset once ──
    tokenizer = AutoTokenizer.from_pretrained(cfg_base.model_name)
    annot_dataset = build_dataset(
        data_path=cfg_base.annotation_data_path,
        tokenizer=tokenizer,
        max_length=cfg_base.max_seq_length,
        max_samples=0,
        data_type="checkpoint",
    )

    # Consistent stratified split across all runs
    trainer_stub = PronounRecoveryTrainer.__new__(PronounRecoveryTrainer)
    trainer_stub.config = cfg_base
    train_dataset, eval_dataset = trainer_stub._stratified_split(
        annot_dataset, test_size=0.1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Dataset: %d train, %d eval", len(train_dataset), len(eval_dataset))
    logger.info("Learning rate: %.1e", lr)

    all_results = {}

    for alpha in args.alphas:
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING alpha=%.2f  lr=%.1e", alpha, lr)
        logger.info("=" * 60)

        run_tag = f"a{alpha}_lr{lr:.0e}"
        run_dir = output_dir / run_tag

        # Configure this run
        cfg = load_config(args.config, ModelTrainingConfig)
        if args.data_path:
            cfg.annotation_data_path = Path(args.data_path)
        cfg.output_path = str(run_dir)
        cfg.weight_alpha = alpha
        cfg.focal_gamma = args.focal_gamma
        cfg.fp16 = torch.cuda.is_available()

        if args.wandb_project:
            cfg.wandb_project = args.wandb_project
            cfg.wandb_run_name = f"a{alpha}_lr{lr:.0e}"

        # Build trainer
        trainer = PronounRecoveryTrainer(cfg)
        trainer.tokenizer = tokenizer

        class_weights = compute_class_weights(train_dataset, alpha=alpha)

        model = AutoModelForTokenClassification.from_pretrained(
            cfg.model_name,
            num_labels=NUM_LABELS,
            id2label={i: l for i, l in enumerate(ALL_LABELS)},
            label2id={l: i for i, l in enumerate(ALL_LABELS)},
        )

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
            focal_gamma=args.focal_gamma,
        )
        elapsed = time.time() - start
        logger.info("alpha=%.2f lr=%.1e training took %.1f min", alpha, lr, elapsed / 60)

        # Save final model
        final_dir = run_dir / "final"
        final_model = AutoModelForTokenClassification.from_pretrained(
            best_ckpt, num_labels=NUM_LABELS
        )
        final_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        # Clean up checkpoints to save disk
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            shutil.rmtree(str(ckpt_dir))
            logger.info("Cleaned up checkpoints at %s", ckpt_dir)

        # Threshold sweep
        thresh_results = evaluate_model(
            str(final_dir), eval_dataset, args.thresholds, device
        )
        all_results[run_tag] = {
            "alpha": alpha,
            "learning_rate": lr,
            "threshold_results": thresh_results,
        }

        best = max(thresh_results, key=lambda x: x["f1"])
        logger.info(
            "alpha=%.2f lr=%.1e best: F1=%.3f at thresh=%.2f (P=%.3f R=%.3f)",
            alpha, lr, best["f1"], best["threshold"],
            best["precision"], best["recall"],
        )

        del model, final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SWEEP SUMMARY — best F1 per (alpha, lr)")
    print(f"{'=' * 80}")
    print(
        f"{'RUN':>20} {'ALPHA':>7} {'LR':>10} {'THRESH':>7} {'PREC':>7} "
        f"{'REC':>7} {'F1':>7} {'DET_F1':>7} {'FP':>6} {'FN':>6}"
    )
    print("-" * 95)
    for tag in sorted(all_results.keys()):
        info = all_results[tag]
        best = max(info["threshold_results"], key=lambda x: x["f1"])
        print(
            f"{tag:>20} {info['alpha']:>7.2f} {info['learning_rate']:>10.1e} "
            f"{best['threshold']:>7.2f} {best['precision']:>7.3f} "
            f"{best['recall']:>7.3f} {best['f1']:>7.3f} "
            f"{best['det_f1']:>7.3f} {best['fp']:>6} {best['fn']:>6}"
        )

    # ── Save results ─────────────────────────────────────────────────────
    results_path = output_dir / f"sweep_results_lr{lr:.0e}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # ── W&B summary table ────────────────────────────────────────────────
    if args.wandb_project:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"summary_lr{lr:.0e}",
                group="weight_alpha_sweep",
                job_type="eval",
            )
            columns = [
                "alpha", "learning_rate", "threshold", "f1", "precision",
                "recall", "det_f1", "det_precision", "det_recall", "fp", "fn",
            ]
            table = wandb.Table(columns=columns)
            for tag in sorted(all_results.keys()):
                info = all_results[tag]
                for r in info["threshold_results"]:
                    table.add_data(
                        info["alpha"], info["learning_rate"],
                        r["threshold"], r["f1"], r["precision"],
                        r["recall"], r["det_f1"], r["det_precision"],
                        r["det_recall"], r["fp"], r["fn"],
                    )
            wandb.log({"sweep_grid": table})

            best_overall = None
            for tag in sorted(all_results.keys()):
                info = all_results[tag]
                best = max(info["threshold_results"], key=lambda x: x["f1"])
                if best_overall is None or best["f1"] > best_overall["f1"]:
                    best_overall = {
                        **best,
                        "alpha": info["alpha"],
                        "learning_rate": info["learning_rate"],
                    }
            if best_overall:
                wandb.summary["best_alpha"] = best_overall["alpha"]
                wandb.summary["best_lr"] = best_overall["learning_rate"]
                wandb.summary["best_threshold"] = best_overall["threshold"]
                wandb.summary["best_f1"] = best_overall["f1"]

            wandb.finish()
            logger.info("W&B summary logged")
        except Exception as e:
            logger.warning("W&B summary logging failed: %s", e)


if __name__ == "__main__":
    main()
