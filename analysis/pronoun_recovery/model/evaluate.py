"""
Held-out evaluation for the pronoun recovery sequence labeler.

Provides fine-grained evaluation including per-label precision/recall/F1,
binary detection accuracy (any-PRO vs NONE), feature-level accuracy
(correct person/number given correct detection), and confusion matrix
output.  Also supports quality-gate checking against thresholds
defined in :class:`~analysis.pronoun_recovery.config.ModelTrainingConfig`.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ..constants import ALL_LABELS, LABEL_NONE, LABEL_TO_ID
from .sequence_labeler import PronounRecoveryModel

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate pronoun recovery model on held-out data.

    Provides a comprehensive evaluation report covering per-label metrics,
    binary detection performance, feature-level accuracy, and a confusion
    matrix.
    """

    def __init__(self, model: PronounRecoveryModel):
        """Initialise with a loaded model.

        Args:
            model: A :class:`PronounRecoveryModel` instance ready for
                inference.
        """
        self.model = model

    # ── Full evaluation ──────────────────────────────────────────────────

    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Full evaluation on held-out data.

        Each item in *eval_data* must have:
          - ``words``: list of word strings
          - ``labels``: list of label strings (same length as ``words``)

        Args:
            eval_data: List of evaluation examples.

        Returns:
            Dict with:
              - ``per_label``: ``{label: {precision, recall, f1, support}}``
              - ``macro_f1``: Macro-averaged F1 across all labels.
              - ``weighted_f1``: Support-weighted F1 across all labels.
              - ``detection_accuracy``: Binary accuracy (any-PRO vs all-NONE).
              - ``detection_f1``: F1 for binary detection task.
              - ``feature_accuracy``: Among correctly detected PRO tokens,
                fraction with correct full label (person/number).
              - ``confusion_matrix``: 2D list (rows=true, cols=predicted).
              - ``num_examples``: Number of evaluation examples.
              - ``num_tokens``: Total tokens evaluated.
        """
        if not eval_data:
            logger.warning("No evaluation data provided")
            return self._empty_results()

        all_true: List[str] = []
        all_pred: List[str] = []

        for example in eval_data:
            words = example["words"]
            gold_labels = example["labels"]

            pred_labels = self.model.predict(words)

            # Ensure we only compare up to the minimum length (in case
            # of truncation).
            n = min(len(gold_labels), len(pred_labels))
            all_true.extend(gold_labels[:n])
            all_pred.extend(pred_labels[:n])

        logger.info(
            "Evaluated %d examples, %d total tokens", len(eval_data), len(all_true)
        )

        results = self._compute_full_metrics(all_true, all_pred)
        results["num_examples"] = len(eval_data)
        results["num_tokens"] = len(all_true)

        return results

    # ── Quality gates ────────────────────────────────────────────────────

    def check_gates(self, results: Dict[str, Any], config) -> Dict[str, bool]:
        """Check if model meets quality gates defined in the config.

        Args:
            results: Evaluation results from :meth:`evaluate`.
            config: A :class:`ModelTrainingConfig` (or object with
                ``min_detection_f1`` and ``min_label_accuracy`` attributes).

        Returns:
            Dict with:
              - ``detection_f1_pass``: Whether detection F1 meets threshold.
              - ``label_accuracy_pass``: Whether feature accuracy meets
                threshold.
              - ``all_pass``: Whether all gates pass.
        """
        detection_f1 = results.get("detection_f1", 0.0)
        feature_accuracy = results.get("feature_accuracy", 0.0)

        detection_f1_pass = detection_f1 >= config.min_detection_f1
        label_accuracy_pass = feature_accuracy >= config.min_label_accuracy
        all_pass = detection_f1_pass and label_accuracy_pass

        logger.info(
            "Quality gates: detection_f1=%.4f (>= %.2f: %s), "
            "feature_accuracy=%.4f (>= %.2f: %s), all_pass=%s",
            detection_f1,
            config.min_detection_f1,
            detection_f1_pass,
            feature_accuracy,
            config.min_label_accuracy,
            label_accuracy_pass,
            all_pass,
        )

        return {
            "detection_f1_pass": detection_f1_pass,
            "label_accuracy_pass": label_accuracy_pass,
            "all_pass": all_pass,
        }

    # ── Report persistence ───────────────────────────────────────────────

    def save_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save evaluation report as JSON.

        Args:
            results: Evaluation results from :meth:`evaluate`.
            output_path: Destination file path (typically ``.json``).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialisation.
        serialisable = self._make_serialisable(results)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, ensure_ascii=False)

        logger.info("Evaluation report saved to %s", output_path)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _compute_full_metrics(
        self, all_true: List[str], all_pred: List[str]
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics from flat true/pred label lists.

        Args:
            all_true: Ground-truth label strings (one per token).
            all_pred: Predicted label strings (one per token).

        Returns:
            Dict of evaluation metrics (see :meth:`evaluate` for keys).
        """
        # ── Per-label metrics ────────────────────────────────────────────
        # Map string labels to integer ids for sklearn.
        label_names = ALL_LABELS
        true_ids = [LABEL_TO_ID.get(l, 0) for l in all_true]
        pred_ids = [LABEL_TO_ID.get(l, 0) for l in all_pred]

        precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
            true_ids,
            pred_ids,
            labels=list(range(len(label_names))),
            zero_division=0,
        )

        per_label: Dict[str, Dict[str, float]] = {}
        for i, label in enumerate(label_names):
            per_label[label] = {
                "precision": float(precision_arr[i]),
                "recall": float(recall_arr[i]),
                "f1": float(f1_arr[i]),
                "support": int(support_arr[i]),
            }

        # ── Macro and weighted F1 ───────────────────────────────────────
        macro_f1 = float(
            f1_score(
                true_ids, pred_ids, labels=list(range(len(label_names))),
                average="macro", zero_division=0,
            )
        )
        weighted_f1 = float(
            f1_score(
                true_ids, pred_ids, labels=list(range(len(label_names))),
                average="weighted", zero_division=0,
            )
        )

        # ── Binary detection: any-PRO vs NONE ───────────────────────────
        true_binary = [0 if l == LABEL_NONE else 1 for l in all_true]
        pred_binary = [0 if l == LABEL_NONE else 1 for l in all_pred]

        detection_correct = sum(
            1 for t, p in zip(true_binary, pred_binary) if t == p
        )
        detection_accuracy = (
            detection_correct / len(true_binary) if true_binary else 0.0
        )
        detection_f1 = float(
            f1_score(true_binary, pred_binary, zero_division=0)
        )

        # ── Feature accuracy ────────────────────────────────────────────
        # Among tokens where both true and pred are non-NONE, what
        # fraction have the exact same label?
        feature_correct = 0
        feature_total = 0
        for t, p in zip(all_true, all_pred):
            if t != LABEL_NONE and p != LABEL_NONE:
                feature_total += 1
                if t == p:
                    feature_correct += 1

        feature_accuracy = (
            feature_correct / feature_total if feature_total > 0 else 0.0
        )

        # ── Confusion matrix ────────────────────────────────────────────
        cm = confusion_matrix(
            true_ids,
            pred_ids,
            labels=list(range(len(label_names))),
        )

        return {
            "per_label": per_label,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "detection_accuracy": detection_accuracy,
            "detection_f1": detection_f1,
            "feature_accuracy": feature_accuracy,
            "confusion_matrix": cm.tolist(),
        }

    @staticmethod
    def _empty_results() -> Dict[str, Any]:
        """Return a correctly-shaped but empty results dict."""
        per_label = {
            label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            for label in ALL_LABELS
        }
        return {
            "per_label": per_label,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "detection_accuracy": 0.0,
            "detection_f1": 0.0,
            "feature_accuracy": 0.0,
            "confusion_matrix": [],
            "num_examples": 0,
            "num_tokens": 0,
        }

    @staticmethod
    def _make_serialisable(obj: Any) -> Any:
        """Recursively convert numpy types to Python natives for JSON.

        Args:
            obj: Arbitrary nested structure.

        Returns:
            The same structure with numpy scalars/arrays replaced by
            Python floats/ints/lists.
        """
        if isinstance(obj, dict):
            return {k: ModelEvaluator._make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ModelEvaluator._make_serialisable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
