"""
Two-phase training pipeline for the pronoun recovery sequence labeler.

Phase A pre-trains a DeBERTa token classifier on synthetic
pronoun-dropped pairs (large scale, noisy labels).  Phase B optionally
fine-tunes the best Phase A checkpoint on human- or LLM-validated
annotation data (smaller scale, cleaner labels).

Both phases use :class:`WeightedLossTrainer`, a custom HuggingFace
:class:`~transformers.Trainer` subclass that applies inverse-frequency
class weights to the cross-entropy loss to handle the severe imbalance
between NONE and PRO.* labels.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ..config import ModelTrainingConfig
from ..constants import ALL_LABELS, NUM_LABELS
from .dataset import build_dataset, compute_class_weights

logger = logging.getLogger(__name__)


# ── Custom Trainer with class-weighted loss ──────────────────────────────


class WeightedLossTrainer(Trainer):
    """Custom Trainer that applies class weights to cross-entropy loss.

    When ``class_weights`` is provided, the loss function weights each
    class inversely proportional to its frequency, which is critical
    for the pronoun recovery task where NONE tokens dominate.
    """

    def __init__(self, class_weights: Optional[np.ndarray] = None, **kwargs):
        """Initialise with optional class weights.

        Args:
            class_weights: Array of shape ``(NUM_LABELS,)`` with per-class
                weights.  If ``None``, standard unweighted cross-entropy
                is used.
            **kwargs: Forwarded to :class:`transformers.Trainer`.
        """
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute weighted cross-entropy loss for token classification.

        Labels of -100 are ignored (special tokens and continuation
        subwords).

        Args:
            model: The token classification model.
            inputs: Dict with ``input_ids``, ``attention_mask``, ``labels``.
            return_outputs: Whether to also return model outputs.

        Returns:
            Loss tensor, or ``(loss, outputs)`` tuple if *return_outputs*
            is ``True``.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = torch.tensor(
                self.class_weights, dtype=torch.float32
            ).to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fn(logits.view(-1, NUM_LABELS), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ── Two-phase training orchestrator ──────────────────────────────────────


class PronounRecoveryTrainer:
    """Two-phase training: pretrain on synthetic data, finetune on annotations.

    Phase A trains on large-scale synthetic pairs (pronoun-dropped text
    with automatic verb labels).  Phase B (optional) fine-tunes the
    best Phase A checkpoint on smaller, higher-quality annotation data.
    """

    def __init__(self, config: ModelTrainingConfig):
        """Initialise from a :class:`ModelTrainingConfig`.

        Args:
            config: Training configuration (model name, hyperparameters,
                data paths, quality gates, etc.).
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def train(self) -> str:
        """Run training (one or two phases depending on config).

        If ``config.synthetic_data_path`` is provided, Phase A pre-trains
        on synthetic data first.  If ``config.annotation_data_path`` is
        provided, Phase B fine-tunes on annotation data (starting from
        the Phase A checkpoint when available, otherwise from the
        pretrained base model).

        At least one of the two data paths must be set.

        Returns:
            Path to the final model directory.
        """
        if self.config.synthetic_data_path is None and self.config.annotation_data_path is None:
            raise ValueError(
                "At least one of synthetic_data_path or annotation_data_path must be set."
            )

        output_base = Path(self.config.output_path)
        output_base.mkdir(parents=True, exist_ok=True)

        best_phase_a = None

        # ── Phase A: Pretrain on synthetic data (optional) ───────────────

        if self.config.synthetic_data_path is not None:
            logger.info("=== Phase A: Pretraining on synthetic data ===")

            synthetic_train_path = Path(self.config.synthetic_data_path) / "train.jsonl"
            synthetic_dev_path = Path(self.config.synthetic_data_path) / "dev.jsonl"

            train_dataset = build_dataset(
                data_path=synthetic_train_path,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
                max_samples=self.config.phase_a_max_samples,
                data_type="synthetic",
            )
            eval_dataset = build_dataset(
                data_path=synthetic_dev_path,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
                max_samples=0,
                data_type="synthetic",
            )

            logger.info(
                "Phase A data: %d train, %d eval examples",
                len(train_dataset),
                len(eval_dataset),
            )

            class_weights = compute_class_weights(train_dataset)

            model = AutoModelForTokenClassification.from_pretrained(
                self.config.model_name,
                num_labels=NUM_LABELS,
                id2label={i: label for i, label in enumerate(ALL_LABELS)},
                label2id={label: i for i, label in enumerate(ALL_LABELS)},
            )

            best_phase_a = self._train_phase(
                phase_name="phase_a",
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=output_base / "phase_a",
                learning_rate=self.config.phase_a_lr,
                batch_size=self.config.phase_a_batch_size,
                num_epochs=self.config.phase_a_epochs,
                patience=self.config.phase_a_patience,
                class_weights=class_weights,
            )

            logger.info("Phase A best checkpoint: %s", best_phase_a)

        # ── Phase B: Train/finetune on annotations (optional) ────────────

        if self.config.annotation_data_path is not None:
            phase_label = "Phase B" if best_phase_a else "Training"
            logger.info("=== %s: Training on annotation data ===", phase_label)

            annotation_path = Path(self.config.annotation_data_path)

            annot_train_dataset = build_dataset(
                data_path=annotation_path,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
                max_samples=0,
                data_type="annotation",
            )

            # Use 10% of annotation data for evaluation.
            split = annot_train_dataset.train_test_split(test_size=0.1, seed=self.config.seed)
            annot_train = split["train"]
            annot_eval = split["test"]

            logger.info(
                "%s data: %d train, %d eval examples",
                phase_label,
                len(annot_train),
                len(annot_eval),
            )

            annot_class_weights = compute_class_weights(annot_train)

            # Start from Phase A checkpoint if available, else from base model.
            init_model = best_phase_a or self.config.model_name
            phase_b_model = AutoModelForTokenClassification.from_pretrained(
                init_model,
                num_labels=NUM_LABELS,
                id2label={i: label for i, label in enumerate(ALL_LABELS)},
                label2id={label: i for i, label in enumerate(ALL_LABELS)},
            )

            best_phase_b = self._train_phase(
                phase_name="phase_b",
                model=phase_b_model,
                train_dataset=annot_train,
                eval_dataset=annot_eval,
                output_dir=output_base / "phase_b",
                learning_rate=self.config.phase_b_lr,
                batch_size=self.config.phase_b_batch_size,
                num_epochs=self.config.phase_b_epochs,
                patience=self.config.phase_b_patience,
                class_weights=annot_class_weights,
            )

            logger.info("%s best checkpoint: %s", phase_label, best_phase_b)

            # Save final model.
            final_dir = output_base / "final"
            final_model = AutoModelForTokenClassification.from_pretrained(
                best_phase_b, num_labels=NUM_LABELS
            )
            final_model.save_pretrained(str(final_dir))
            self.tokenizer.save_pretrained(str(final_dir))
            logger.info("Final model saved to %s", final_dir)

            return str(final_dir)

        # Phase A only (no annotation data).
        final_dir = output_base / "final"
        final_model = AutoModelForTokenClassification.from_pretrained(
            best_phase_a, num_labels=NUM_LABELS
        )
        final_model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        logger.info("Final model saved to %s (Phase A only)", final_dir)

        return str(final_dir)

    # ── Single-phase training ────────────────────────────────────────────

    def _train_phase(
        self,
        phase_name: str,
        model,
        train_dataset,
        eval_dataset,
        output_dir: Path,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        patience: int,
        class_weights: Optional[np.ndarray] = None,
    ) -> str:
        """Run a single training phase.

        Args:
            phase_name: Identifier for logging and checkpoint naming
                (e.g. ``"phase_a"``).
            model: The HuggingFace model to train.
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            output_dir: Directory for checkpoints.
            learning_rate: Peak learning rate.
            batch_size: Per-device train/eval batch size.
            num_epochs: Maximum number of epochs.
            patience: Early stopping patience (number of evaluations
                without improvement).
            class_weights: Optional per-class loss weights.

        Returns:
            Path to the best checkpoint directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        run_name = (
            f"{self.config.wandb_run_name}_{phase_name}"
            if self.config.wandb_run_name
            else phase_name
        )

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            run_name=run_name,
            # Optimisation
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=0.01,
            # Precision
            fp16=self.config.fp16,
            # Evaluation and saving
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=2,
            # Logging
            logging_strategy="steps",
            logging_steps=100,
            report_to="wandb" if self.config.wandb_project else "none",
            # Reproducibility
            seed=self.config.seed,
            data_seed=self.config.seed,
        )

        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )

        logger.info(
            "Starting %s: lr=%.1e, batch=%d, epochs=%d, patience=%d",
            phase_name,
            learning_rate,
            batch_size,
            num_epochs,
            patience,
        )

        trainer.train()

        # The best checkpoint path is set by load_best_model_at_end.
        best_checkpoint = trainer.state.best_model_checkpoint
        if best_checkpoint is None:
            # Fallback: save current model state.
            fallback_dir = output_dir / "best_fallback"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(fallback_dir))
            best_checkpoint = str(fallback_dir)
            logger.warning(
                "%s: no best checkpoint recorded; saved fallback to %s",
                phase_name,
                best_checkpoint,
            )

        logger.info("%s complete. Best checkpoint: %s", phase_name, best_checkpoint)
        return best_checkpoint

    # ── Metrics ──────────────────────────────────────────────────────────

    def _compute_metrics(self, eval_pred) -> dict:
        """Compute per-label and aggregate metrics using seqeval.

        Converts integer label-id predictions back to string labels,
        ignoring positions with label -100 (special/continuation tokens).
        Reports precision, recall, F1 (macro and weighted), and accuracy.

        Args:
            eval_pred: ``EvalPrediction`` with ``predictions``
                (logits array) and ``label_ids``.

        Returns:
            Dict with ``precision``, ``recall``, ``f1``,
            ``accuracy``, and per-label metrics.
        """
        from seqeval.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            classification_report,
        )

        logits, label_ids = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Convert id arrays to label-string sequences, skipping -100.
        true_labels: list = []
        pred_labels: list = []

        for pred_seq, label_seq in zip(predictions, label_ids):
            true_seq: list = []
            pred_seq_filtered: list = []

            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                true_seq.append(ALL_LABELS[label_id] if 0 <= label_id < NUM_LABELS else LABEL_NONE)
                pred_seq_filtered.append(
                    ALL_LABELS[pred_id] if 0 <= pred_id < NUM_LABELS else LABEL_NONE
                )

            true_labels.append(true_seq)
            pred_labels.append(pred_seq_filtered)

        # seqeval treats each sequence as a sentence of entity labels.
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        accuracy = accuracy_score(true_labels, pred_labels)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

        logger.info(
            "Eval metrics: P=%.4f R=%.4f F1=%.4f Acc=%.4f",
            precision,
            recall,
            f1,
            accuracy,
        )

        return metrics
