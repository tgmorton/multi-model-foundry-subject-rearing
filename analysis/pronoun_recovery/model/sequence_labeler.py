"""
DeBERTa-based sequence labeler for pronoun recovery prediction.

Wraps a HuggingFace ``AutoModelForTokenClassification`` to provide
convenient single-example, batched, and spaCy-Doc prediction interfaces.
Handles the mapping between subword tokens and word-level predictions
via the tokenizer's ``word_ids()`` method.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import spacy.tokens
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ..constants import ALL_LABELS, ID_TO_LABEL, LABEL_NONE, LABEL_TO_ID, NUM_LABELS

logger = logging.getLogger(__name__)


class PronounRecoveryModel:
    """Wrapper around DeBERTa token classification for pronoun recovery prediction.

    Loads a pretrained or fine-tuned checkpoint and provides methods
    for predicting per-word pronoun labels from either raw word lists
    or spaCy ``Doc`` objects.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 256,
        threshold: float = 0.0,
    ):
        """Load model and tokenizer from path (either HF hub or local checkpoint).

        Args:
            model_path: HuggingFace model hub identifier or local directory
                containing ``config.json``, ``model.safetensors`` /
                ``pytorch_model.bin``, and tokenizer files.
            device: PyTorch device string (e.g. ``"cuda"``, ``"cpu"``).
                Auto-detected if ``None``.
            max_length: Maximum subword sequence length for tokenisation.
            threshold: Minimum P(any pronoun) required before inserting.
                Predictions where 1 - P(NONE) < threshold are forced to
                NONE.  Default ``0.0`` preserves existing argmax behaviour.
        """
        self.max_length = max_length
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info("Loading model from %s (num_labels=%d)", model_path, NUM_LABELS)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path, num_labels=NUM_LABELS
        ).to(self.device)
        self.model.eval()

        logger.info("PronounRecoveryModel ready on device=%s", self.device)

    # ── Single-example prediction ────────────────────────────────────────

    def predict(self, words: List[str]) -> List[str]:
        """Predict labels for a list of words.

        Tokenises the word list with ``is_split_into_words=True``, runs
        the model, and maps subword-level predictions back to words by
        taking the argmax prediction of the first subword token for each
        word.

        Args:
            words: List of word strings (whitespace-split tokens).

        Returns:
            List of label strings, same length as *words*.
        """
        if not words:
            return []

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (1, seq_len, num_labels)

        predictions = self._apply_threshold(logits).squeeze(0).cpu().tolist()
        word_ids = encoding.word_ids(batch_index=0)

        return self._align_predictions_to_words(predictions, word_ids, len(words))

    # ── Batch prediction ─────────────────────────────────────────────────

    def predict_batch(self, batch_words: List[List[str]]) -> List[List[str]]:
        """Predict labels for a batch of word lists.

        Args:
            batch_words: List of word lists.

        Returns:
            List of label-string lists, one per input word list.
        """
        if not batch_words:
            return []

        encoding = self.tokenizer(
            batch_words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, seq_len, num_labels)

        predictions = self._apply_threshold(logits).cpu().tolist()

        results: List[List[str]] = []
        for batch_idx, words in enumerate(batch_words):
            word_ids = encoding.word_ids(batch_index=batch_idx)
            labels = self._align_predictions_to_words(
                predictions[batch_idx], word_ids, len(words)
            )
            results.append(labels)

        return results

    # ── spaCy Doc prediction ─────────────────────────────────────────────

    def predict_doc(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """Predict pronoun labels for a spaCy Doc.

        Extracts word texts from the Doc, runs prediction, and returns
        only tokens with non-NONE labels.

        Args:
            doc: A spaCy ``Doc`` object.

        Returns:
            List of dicts, each containing:
              - ``token_idx``: Index of the token in the Doc.
              - ``token_text``: Surface form of the token.
              - ``label``: The predicted PRO.* label string.
            Only includes tokens whose predicted label is not NONE.
        """
        words = [token.text for token in doc]
        labels = self.predict(words)

        results: List[Dict[str, Any]] = []
        for idx, (token, label) in enumerate(zip(doc, labels)):
            if label != LABEL_NONE:
                results.append({
                    "token_idx": idx,
                    "token_text": token.text,
                    "label": label,
                })

        return results

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, output_path: str) -> None:
        """Save model and tokenizer to directory.

        Args:
            output_path: Destination directory.  Created if it does not exist.
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving model to %s", output_dir)
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info("Model and tokenizer saved to %s", output_dir)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "PronounRecoveryModel":
        """Load from a saved checkpoint directory.

        This is a convenience class method equivalent to
        ``PronounRecoveryModel(model_path=path, **kwargs)``.

        Args:
            path: Path to a directory containing model and tokenizer files.
            **kwargs: Additional keyword arguments passed to the constructor
                (e.g. ``device``, ``max_length``).

        Returns:
            A ready-to-use :class:`PronounRecoveryModel` instance.
        """
        return cls(model_path=path, **kwargs)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _apply_threshold(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply confidence threshold to logits, forcing low-confidence positions to NONE.

        When ``self.threshold > 0``, positions where
        P(any pronoun) = 1 - P(NONE) < threshold are forced to the NONE
        label id.  When ``threshold == 0.0`` this reduces to plain argmax.

        Args:
            logits: Tensor of shape ``(batch, seq_len, num_labels)``.

        Returns:
            Integer tensor of predicted label ids, shape ``(batch, seq_len)``.
        """
        best_ids = torch.argmax(logits, dim=-1)  # (batch, seq_len)

        if self.threshold > 0.0:
            probs = torch.softmax(logits, dim=-1)
            none_id = LABEL_TO_ID[LABEL_NONE]
            none_prob = probs[..., none_id]
            below_threshold = (1.0 - none_prob) < self.threshold
            best_ids[below_threshold] = none_id

        return best_ids

    @staticmethod
    def _align_predictions_to_words(
        subword_preds: List[int],
        word_ids: List[Optional[int]],
        num_words: int,
    ) -> List[str]:
        """Map subword predictions back to word-level labels.

        For each word, the prediction assigned to its first subword token
        is used.  Words that were truncated away (i.e. never appeared in
        the subword sequence) receive NONE.

        Args:
            subword_preds: List of predicted label ids (one per subword token).
            word_ids: Word-id mapping from the tokenizer (``None`` for
                special tokens).
            num_words: Number of original words.

        Returns:
            List of label strings, length ``num_words``.
        """
        word_labels = [LABEL_NONE] * num_words
        seen_word_ids: set = set()

        for subword_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)

            if word_id < num_words:
                pred_id = subword_preds[subword_idx]
                word_labels[word_id] = ID_TO_LABEL.get(pred_id, LABEL_NONE)

        return word_labels
