"""Two-phase training pipeline for the seq2seq (mT5) pronoun recovery model.

Phase A pre-trains an mT5 model on synthetic pronoun-dropped pairs
(large scale, noisy labels).  Phase B optionally fine-tunes the best
Phase A checkpoint on human- or LLM-validated annotation data (smaller
scale, cleaner labels).

Metrics include ROUGE-L, exact match, and pronoun-specific F1 (count
of pronouns in generated output vs target).
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from ..config import Seq2SeqTrainingConfig
from .seq2seq_dataset import build_seq2seq_dataset

logger = logging.getLogger(__name__)

# English and Italian subject pronouns for pronoun-specific metrics.
_PRONOUN_PATTERN = re.compile(
    r"\b("
    r"[Ii]|[Yy]ou|[Hh]e|[Ss]he|[Ii]t|[Ww]e|[Tt]hey"
    r"|[Ii]o|[Tt]u|[Ll]ui|[Ll]ei|[Nn]oi|[Vv]oi|[Ll]oro"
    r")\b"
)


def _count_pronouns(text: str) -> int:
    """Count subject pronouns in a text string."""
    return len(_PRONOUN_PATTERN.findall(text))


class Seq2SeqPronounTrainer:
    """Two-phase training: pretrain mT5 on synthetic data, finetune on annotations.

    Phase A trains on large-scale synthetic pairs (pronoun-dropped text
    → original text with pronouns).  Phase B (optional) fine-tunes the
    best Phase A checkpoint on smaller, higher-quality annotation data.
    """

    def __init__(self, config: Seq2SeqTrainingConfig):
        """Initialise from a :class:`Seq2SeqTrainingConfig`.

        Args:
            config: Training configuration (model name, hyperparameters,
                data paths, etc.).
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
            logger.info("=== Phase A: Pretraining seq2seq on synthetic data ===")

            synthetic_train_path = Path(self.config.synthetic_data_path) / "train.jsonl"
            synthetic_dev_path = Path(self.config.synthetic_data_path) / "dev.jsonl"

            train_dataset = build_seq2seq_dataset(
                data_path=synthetic_train_path,
                tokenizer=self.tokenizer,
                max_input_length=self.config.max_input_length,
                max_target_length=self.config.max_target_length,
                max_samples=self.config.phase_a_max_samples,
            )
            eval_dataset = build_seq2seq_dataset(
                data_path=synthetic_dev_path,
                tokenizer=self.tokenizer,
                max_input_length=self.config.max_input_length,
                max_target_length=self.config.max_target_length,
                max_samples=0,
            )

            logger.info(
                "Phase A data: %d train, %d eval examples",
                len(train_dataset),
                len(eval_dataset),
            )

            model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

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
            )

            logger.info("Phase A best checkpoint: %s", best_phase_a)

        # ── Phase B: Train/finetune on annotations (optional) ────────────

        if self.config.annotation_data_path is not None:
            phase_label = "Phase B" if best_phase_a else "Training"
            logger.info("=== %s: Training seq2seq on annotation data ===", phase_label)

            annotation_path = Path(self.config.annotation_data_path)

            annot_dataset = build_seq2seq_dataset(
                data_path=annotation_path,
                tokenizer=self.tokenizer,
                max_input_length=self.config.max_input_length,
                max_target_length=self.config.max_target_length,
                max_samples=0,
            )

            split = annot_dataset.train_test_split(test_size=0.1, seed=self.config.seed)
            annot_train = split["train"]
            annot_eval = split["test"]

            logger.info(
                "%s data: %d train, %d eval examples",
                phase_label,
                len(annot_train),
                len(annot_eval),
            )

            # Start from Phase A checkpoint if available, else from base model.
            init_model = best_phase_a or self.config.model_name
            phase_b_model = AutoModelForSeq2SeqLM.from_pretrained(init_model)

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
            )

            logger.info("%s best checkpoint: %s", phase_label, best_phase_b)

            # Save final model.
            final_dir = output_base / "final"
            final_model = AutoModelForSeq2SeqLM.from_pretrained(best_phase_b)
            final_model.save_pretrained(str(final_dir))
            self.tokenizer.save_pretrained(str(final_dir))
            logger.info("Final seq2seq model saved to %s", final_dir)

            return str(final_dir)

        # Phase A only (no annotation data).
        final_dir = output_base / "final"
        final_model = AutoModelForSeq2SeqLM.from_pretrained(best_phase_a)
        final_model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        logger.info("Final seq2seq model saved to %s (Phase A only)", final_dir)

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
    ) -> str:
        """Run a single training phase.

        Args:
            phase_name: Identifier for logging (e.g. ``"phase_a"``).
            model: The HuggingFace seq2seq model to train.
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            output_dir: Directory for checkpoints.
            learning_rate: Peak learning rate.
            batch_size: Per-device train/eval batch size.
            num_epochs: Maximum number of epochs.
            patience: Early stopping patience.

        Returns:
            Path to the best checkpoint directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        run_name = (
            f"{self.config.wandb_run_name}_{phase_name}"
            if self.config.wandb_run_name
            else phase_name
        )

        training_args = Seq2SeqTrainingArguments(
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
            # Generation during eval
            predict_with_generate=True,
            generation_max_length=self.config.max_target_length,
            generation_num_beams=self.config.num_beams,
            # Evaluation and saving
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_rouge_l",
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

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
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

        best_checkpoint = trainer.state.best_model_checkpoint
        if best_checkpoint is None:
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
        """Compute ROUGE-L, exact match, and pronoun F1.

        Args:
            eval_pred: ``EvalPrediction`` with ``predictions``
                (generated token ids) and ``label_ids``.

        Returns:
            Dict with ``rouge_l``, ``exact_match``, ``pronoun_precision``,
            ``pronoun_recall``, and ``pronoun_f1``.
        """
        predictions, label_ids = eval_pred

        # Replace -100 in labels (padding) with pad token id for decoding.
        label_ids = np.where(
            label_ids != -100, label_ids, self.tokenizer.pad_token_id
        )

        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # Strip whitespace.
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # ROUGE-L
        rouge_l = self._compute_rouge_l(decoded_preds, decoded_labels)

        # Exact match
        exact_matches = sum(
            1 for p, l in zip(decoded_preds, decoded_labels) if p == l
        )
        exact_match = exact_matches / max(len(decoded_preds), 1)

        # Pronoun-specific F1
        total_pred_pronouns = 0
        total_ref_pronouns = 0
        total_correct_pronouns = 0

        for pred, ref in zip(decoded_preds, decoded_labels):
            pred_count = _count_pronouns(pred)
            ref_count = _count_pronouns(ref)
            total_pred_pronouns += pred_count
            total_ref_pronouns += ref_count
            total_correct_pronouns += min(pred_count, ref_count)

        pronoun_precision = (
            total_correct_pronouns / total_pred_pronouns
            if total_pred_pronouns > 0
            else 0.0
        )
        pronoun_recall = (
            total_correct_pronouns / total_ref_pronouns
            if total_ref_pronouns > 0
            else 0.0
        )
        pronoun_f1 = (
            2 * pronoun_precision * pronoun_recall / (pronoun_precision + pronoun_recall)
            if (pronoun_precision + pronoun_recall) > 0
            else 0.0
        )

        metrics = {
            "rouge_l": rouge_l,
            "exact_match": exact_match,
            "pronoun_precision": pronoun_precision,
            "pronoun_recall": pronoun_recall,
            "pronoun_f1": pronoun_f1,
        }

        logger.info(
            "Eval metrics: ROUGE-L=%.4f EM=%.4f P_pro=%.4f R_pro=%.4f F1_pro=%.4f",
            rouge_l,
            exact_match,
            pronoun_precision,
            pronoun_recall,
            pronoun_f1,
        )

        return metrics

    @staticmethod
    def _compute_rouge_l(predictions: List[str], references: List[str]) -> float:
        """Compute average ROUGE-L F1 score.

        Uses a simple LCS-based implementation to avoid an external
        dependency on the ``rouge_score`` package.

        Args:
            predictions: List of predicted strings.
            references: List of reference strings.

        Returns:
            Average ROUGE-L F1 score across all pairs.
        """
        def _lcs_length(x: List[str], y: List[str]) -> int:
            m, n = len(x), len(y)
            prev = [0] * (n + 1)
            for i in range(1, m + 1):
                curr = [0] * (n + 1)
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        curr[j] = prev[j - 1] + 1
                    else:
                        curr[j] = max(curr[j - 1], prev[j])
                prev = curr
            return prev[n]

        scores: List[float] = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            if not ref_tokens:
                scores.append(1.0 if not pred_tokens else 0.0)
                continue
            if not pred_tokens:
                scores.append(0.0)
                continue
            lcs = _lcs_length(pred_tokens, ref_tokens)
            precision = lcs / len(pred_tokens)
            recall = lcs / len(ref_tokens)
            if precision + recall > 0:
                scores.append(2 * precision * recall / (precision + recall))
            else:
                scores.append(0.0)

        return sum(scores) / max(len(scores), 1)
