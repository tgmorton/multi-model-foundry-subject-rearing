"""
Training pipeline for the pronoun recovery sequence labeler.

Two strategies, selected via ``config.training_source``:

- ``"synthetic"`` (English path): Fine-tune on large-scale synthetic
  pronoun-removal pairs where overt pronouns are mechanically stripped.
- ``"llm_annotated"`` (Italian path): Fine-tune on LLM-annotated
  null-subject candidates where dropped pronouns are recovered via
  few-shot prompting.

Both strategies fine-tune a pre-trained model (e.g. mDeBERTa).

Uses :class:`WeightedLossTrainer`, a custom HuggingFace
:class:`~transformers.Trainer` subclass that applies inverse-frequency
class weights to the cross-entropy loss to handle the severe imbalance
between NONE and PRO.* labels.
"""

import logging
from collections import Counter
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
from ..constants import ALL_LABELS, LABEL_NONE, NUM_LABELS
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
    """Fine-tune a pre-trained model for pronoun recovery.

    Strategy is determined by ``config.training_source``:

    - ``"synthetic"``: Fine-tune on synthetic pronoun-removal pairs.
      Reads ``train.jsonl`` / ``dev.jsonl`` from ``synthetic_data_path``.
    - ``"llm_annotated"``: Fine-tune on LLM-annotated null-subject data.
      Reads from ``annotation_data_path``, with 90/10 train/eval split.
    """

    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def train(self) -> str:
        """Run training based on ``config.training_source``.

        Returns:
            Path to the final model directory.
        """
        source = self.config.training_source

        if source == "synthetic":
            return self._train_synthetic()
        elif source == "llm_annotated":
            return self._train_llm_annotated()
        else:
            raise ValueError(
                f"Unknown training_source '{source}'. "
                f"Expected 'synthetic' or 'llm_annotated'."
            )

    def _train_synthetic(self) -> str:
        """Fine-tune on synthetic pronoun-removal pairs (English path)."""
        if self.config.synthetic_data_path is None:
            raise ValueError(
                "synthetic_data_path must be set for training_source='synthetic'"
            )

        output_base = Path(self.config.output_path)
        output_base.mkdir(parents=True, exist_ok=True)

        logger.info("=== Training on synthetic data ===")

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
            "Synthetic data: %d train, %d eval examples",
            len(train_dataset),
            len(eval_dataset),
        )

        alpha = getattr(self.config, "weight_alpha", 1.0)
        class_weights = compute_class_weights(train_dataset, alpha=alpha)

        model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=NUM_LABELS,
            id2label={i: label for i, label in enumerate(ALL_LABELS)},
            label2id={label: i for i, label in enumerate(ALL_LABELS)},
        )

        best_ckpt = self._train_phase(
            phase_name="synthetic",
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_base / "synthetic",
            learning_rate=self.config.phase_a_lr,
            batch_size=self.config.phase_a_batch_size,
            num_epochs=self.config.phase_a_epochs,
            patience=self.config.phase_a_patience,
            class_weights=class_weights,
        )

        return self._save_final(best_ckpt, output_base)

    def _train_llm_annotated(self) -> str:
        """Fine-tune on LLM-annotated null-subject data (Italian path)."""
        if self.config.annotation_data_path is None:
            raise ValueError(
                "annotation_data_path must be set for training_source='llm_annotated'"
            )

        output_base = Path(self.config.output_path)
        output_base.mkdir(parents=True, exist_ok=True)

        logger.info("=== Training on LLM-annotated data ===")

        annotation_path = Path(self.config.annotation_data_path)

        # Detect format: checkpoint files have markers, pre-converted files have words/labels
        data_type = "checkpoint" if annotation_path.suffix == ".jsonl" else "annotation"
        annot_dataset = build_dataset(
            data_path=annotation_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
            max_samples=0,
            data_type=data_type,
        )

        # Stratified train/test split: assign each example a stratum
        # based on its rarest non-NONE label so that minority classes
        # (IMP, CONJ, PRO.2pl) are proportionally represented in eval.
        annot_train, annot_eval = self._stratified_split(
            annot_dataset, test_size=0.1
        )

        alpha = getattr(self.config, "weight_alpha", 1.0)
        class_weights = compute_class_weights(annot_train, alpha=alpha)

        model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=NUM_LABELS,
            id2label={i: label for i, label in enumerate(ALL_LABELS)},
            label2id={label: i for i, label in enumerate(ALL_LABELS)},
        )

        best_ckpt = self._train_phase(
            phase_name="llm_annotated",
            model=model,
            train_dataset=annot_train,
            eval_dataset=annot_eval,
            output_dir=output_base / "llm_annotated",
            learning_rate=self.config.phase_b_lr,
            batch_size=self.config.phase_b_batch_size,
            num_epochs=self.config.phase_b_epochs,
            patience=self.config.phase_b_patience,
            class_weights=class_weights,
        )

        return self._save_final(best_ckpt, output_base)

    def _save_final(self, best_checkpoint: str, output_base: Path) -> str:
        """Save the best checkpoint as the final model."""
        final_dir = output_base / "final"
        final_model = AutoModelForTokenClassification.from_pretrained(
            best_checkpoint, num_labels=NUM_LABELS
        )
        final_model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        logger.info("Final model saved to %s", final_dir)
        return str(final_dir)

    # ── Stratified splitting ────────────────────────────────────────────

    def _stratified_split(self, dataset, test_size: float = 0.1):
        """Split dataset with stratification by rarest non-NONE label.

        Each example is assigned to a stratum based on the rarest label
        it contains (by global frequency).  This ensures minority classes
        like IMP, CONJ, PRO.2pl get proportional representation in the
        eval set rather than being concentrated in train by chance.

        Args:
            dataset: Tokenized HF Dataset with integer ``labels`` column.
            test_size: Fraction of data to hold out for evaluation.

        Returns:
            Tuple of (train_dataset, eval_dataset).
        """
        from sklearn.model_selection import StratifiedShuffleSplit

        # 1. Count global label frequencies (excluding NONE=0 and ignored=-100)
        global_counts: Counter = Counter()
        for example in dataset:
            label_ids = example["labels"]
            if hasattr(label_ids, "numpy"):
                label_ids = label_ids.numpy()
            for lid in label_ids:
                lid = int(lid)
                if lid > 0 and lid < NUM_LABELS:
                    global_counts[lid] += 1

        # 2. Assign each example a stratum = its rarest non-NONE label
        #    (or 0 = "NONE-only" if no pronoun labels present)
        strata = []
        for example in dataset:
            label_ids = example["labels"]
            if hasattr(label_ids, "numpy"):
                label_ids = label_ids.numpy()
            present = {int(lid) for lid in label_ids if 0 < int(lid) < NUM_LABELS}
            if not present:
                strata.append(0)  # NONE-only
            else:
                rarest = min(present, key=lambda lid: global_counts.get(lid, 0))
                strata.append(rarest)

        strata = np.array(strata)

        # Log stratum distribution
        stratum_counts = Counter(strata)
        logger.info(
            "Stratification strata: %s",
            {ALL_LABELS[k] if k < len(ALL_LABELS) else k: v
             for k, v in sorted(stratum_counts.items())},
        )

        # 3. Stratified split
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=self.config.seed
        )
        train_idx, eval_idx = next(sss.split(np.zeros(len(strata)), strata))

        train_dataset = dataset.select(train_idx)
        eval_dataset = dataset.select(eval_idx)

        logger.info(
            "Stratified split: %d train, %d eval examples",
            len(train_dataset),
            len(eval_dataset),
        )

        return train_dataset, eval_dataset

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
