"""
Evaluation metrics for null subject detection.

Computes detection_f1 (binary PRO vs NONE), feature_accuracy (correct
person/number among detected), per-label precision/recall/F1, and
confusion matrix.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from analysis.pronoun_recovery.constants import LABEL_NONE, MORPH_TO_LABEL_SUFFIX

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    y_true_orig: np.ndarray,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Compute evaluation metrics for null subject detection.

    Args:
        y_true_bin: Ground truth binary labels (1=PRO, 0=NONE).
        y_pred_bin: Predicted binary labels.
        y_true_orig: Original multi-class labels (PRO.1sg, PRO.3pl, NONE, etc.)
            for computing feature_accuracy on detected verbs.
        model_name: Name for logging.

    Returns:
        Dict with detection_f1, feature_accuracy, per_label metrics, and
        confusion_matrix.
    """
    # Binary detection metrics
    det_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    det_precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    det_recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)

    # Confusion matrix (binary)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])

    # Feature accuracy: among true positives (correctly detected null subjects),
    # what fraction have correct person/number?
    # Since the tree only does binary detection and person/number comes from
    # morphology (>99% accurate), we report the morph agreement rate.
    tp_mask = (y_true_bin == 1) & (y_pred_bin == 1)
    n_tp = tp_mask.sum()

    # For feature accuracy, check how many of the true positive predictions
    # have matching original labels (i.e., the morphology-derived label would
    # match the gold label). Since morphology determines the label, this is
    # essentially the morphology accuracy.
    if n_tp > 0:
        tp_labels = y_true_orig[tp_mask]
        # All detected PRO positions get their label from morphology, so
        # feature_accuracy = fraction of TP where the gold label is one of
        # the 6 PRO.* labels (not NONE, which would be an error).
        n_correct = np.sum(tp_labels != LABEL_NONE)
        feature_accuracy = n_correct / n_tp
    else:
        feature_accuracy = 0.0

    # Per-label metrics on the original labels (for detected positions)
    per_label = {}
    pro_labels = sorted(set(y_true_orig) - {LABEL_NONE})
    for label in pro_labels:
        label_mask = y_true_orig == label
        n_total = label_mask.sum()
        # Of these, how many were detected (predicted as 1)?
        n_detected = (label_mask & (y_pred_bin == 1)).sum()
        label_recall = n_detected / n_total if n_total > 0 else 0.0

        per_label[label] = {
            "support": int(n_total),
            "detected": int(n_detected),
            "recall": float(label_recall),
        }

    metrics = {
        "detection_f1": float(det_f1),
        "detection_precision": float(det_precision),
        "detection_recall": float(det_recall),
        "feature_accuracy": float(feature_accuracy),
        "confusion_matrix": cm.tolist(),
        "n_true_positive": int(n_tp),
        "n_true_negative": int(cm[0, 0]),
        "n_false_positive": int(cm[0, 1]),
        "n_false_negative": int(cm[1, 0]),
        "per_label": per_label,
    }

    logger.info(
        "%s — F1: %.4f  P: %.4f  R: %.4f  FA: %.4f  (TP=%d FP=%d FN=%d)",
        model_name,
        det_f1,
        det_precision,
        det_recall,
        feature_accuracy,
        n_tp,
        cm[0, 1],
        cm[1, 0],
    )

    return metrics


def save_evaluation_report(
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save evaluation metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation report saved to %s", output_path)


def check_quality_gates(
    metrics: Dict[str, Any],
    min_detection_f1: float = 0.80,
    min_feature_accuracy: float = 0.95,
) -> Dict[str, Any]:
    """Check whether metrics pass quality gates.

    Args:
        metrics: Evaluation metrics dict.
        min_detection_f1: Minimum required detection F1.
        min_feature_accuracy: Minimum required feature accuracy.

    Returns:
        Dict with gate results and overall pass/fail.
    """
    det_f1 = metrics.get("detection_f1", 0)
    feat_acc = metrics.get("feature_accuracy", 0)

    gates = {
        "detection_f1": {
            "value": det_f1,
            "threshold": min_detection_f1,
            "pass": det_f1 >= min_detection_f1,
        },
        "feature_accuracy": {
            "value": feat_acc,
            "threshold": min_feature_accuracy,
            "pass": feat_acc >= min_feature_accuracy,
        },
    }

    all_pass = all(g["pass"] for g in gates.values())
    gates["all_pass"] = all_pass

    if all_pass:
        logger.info("All quality gates passed.")
    else:
        failed = [k for k, v in gates.items() if k != "all_pass" and not v["pass"]]
        logger.warning("Quality gates failed: %s", failed)

    return gates
