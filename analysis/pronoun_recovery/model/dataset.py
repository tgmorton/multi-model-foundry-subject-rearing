"""
Dataset construction for pronoun recovery token classification.

Converts synthetic pairs and LLM annotations into HuggingFace Dataset
format suitable for DeBERTa-based sequence labeling.  Each word in the
dropped text is assigned a label (NONE or PRO.*), and subword
tokenisation is handled by propagating the label only to the first
subword token of each word (the rest receive -100, which PyTorch
CrossEntropyLoss ignores).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, DatasetDict

from ..constants import ALL_LABELS, LABEL_NONE, LABEL_TO_ID, NUM_LABELS

logger = logging.getLogger(__name__)

# Sentinel value used by HuggingFace / PyTorch to mask tokens in the loss.
_IGNORE_INDEX = -100


# ── Data loading helpers ─────────────────────────────────────────────────


def load_synthetic_pairs(data_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL synthetic pairs from file.

    Each line is expected to be a JSON object with at least ``dropped``
    (str) and ``verb_labels`` (list of dicts with ``verb_idx``,
    ``verb_text``, ``label``).

    Args:
        data_path: Path to a ``.jsonl`` file of synthetic pairs.

    Returns:
        List of parsed JSON dicts.

    Raises:
        FileNotFoundError: If *data_path* does not exist.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Synthetic data file not found: {data_path}")

    records: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d in %s", line_no, data_path)

    logger.info("Loaded %d synthetic pairs from %s", len(records), data_path)
    return records


def load_annotation_data(data_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL annotation data from file.

    Each line is expected to be a JSON object with ``words`` (list of str)
    and ``labels`` (list of str, same length as ``words``).

    Args:
        data_path: Path to a ``.jsonl`` file of annotation records.

    Returns:
        List of parsed JSON dicts.

    Raises:
        FileNotFoundError: If *data_path* does not exist.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Annotation data file not found: {data_path}")

    records: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d in %s", line_no, data_path)
                continue

            # Validate required fields.
            if "words" not in record or "labels" not in record:
                logger.warning(
                    "Skipping record at line %d (missing 'words' or 'labels')", line_no
                )
                continue
            if len(record["words"]) != len(record["labels"]):
                logger.warning(
                    "Skipping record at line %d (words/labels length mismatch: %d vs %d)",
                    line_no,
                    len(record["words"]),
                    len(record["labels"]),
                )
                continue

            records.append(record)

    logger.info("Loaded %d annotation records from %s", len(records), data_path)
    return records


# ── Tokenisation and label alignment ────────────────────────────────────


def tokenize_and_align_labels(
    examples: Dict[str, List],
    tokenizer,
    max_length: int = 256,
) -> Dict[str, List]:
    """Tokenize dropped text with DeBERTa tokenizer and align word-level
    verb labels to subword tokens.

    For each word, only the first subword token receives the actual label;
    subsequent subword tokens receive -100 (ignored in loss computation).
    Most tokens are labeled NONE (id 0); only verb tokens with PRO.*
    labels receive non-zero label ids.

    This is designed as a batched mapping function for
    :meth:`datasets.Dataset.map`.

    Each example in the batch has:
      - ``words``: list of word strings
      - ``labels``: list of label strings (same length as ``words``),
        mostly ``"NONE"``

    Args:
        examples: Batched dict with ``words`` and ``labels`` lists.
        tokenizer: A HuggingFace tokenizer (e.g. DeBERTa).
        max_length: Maximum subword sequence length.

    Returns:
        Dict with ``input_ids``, ``attention_mask``, and ``labels``
        (integer ids, with -100 for non-first subwords and special tokens).
    """
    tokenized = tokenizer(
        examples["words"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    all_labels: List[List[int]] = []

    for batch_idx in range(len(examples["words"])):
        word_ids = tokenized.word_ids(batch_index=batch_idx)
        label_strings = examples["labels"][batch_idx]

        aligned: List[int] = []
        previous_word_id: Optional[int] = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], [PAD]).
                aligned.append(_IGNORE_INDEX)
            elif word_id != previous_word_id:
                # First subword of a new word -- assign the actual label.
                label_str = label_strings[word_id] if word_id < len(label_strings) else LABEL_NONE
                aligned.append(LABEL_TO_ID.get(label_str, LABEL_TO_ID[LABEL_NONE]))
            else:
                # Continuation subword -- ignore in loss.
                aligned.append(_IGNORE_INDEX)
            previous_word_id = word_id

        all_labels.append(aligned)

    tokenized["labels"] = all_labels
    return tokenized


# ── Dataset construction ────────────────────────────────────────────────


def _synthetic_to_word_labels(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single synthetic pair record to word-level labels.

    The synthetic record contains ``dropped`` (a string) and
    ``verb_labels`` (list of dicts with ``verb_idx`` and ``label``).
    We split the dropped text into whitespace-delimited words and build
    a parallel label list where verb positions receive their PRO.* label
    and all other positions receive NONE.

    Returns:
        Dict with ``words`` and ``labels``, or ``None`` if the record
        is unusable.
    """
    dropped_text = record.get("dropped", "")
    verb_labels = record.get("verb_labels", [])

    if not dropped_text:
        return None

    words = dropped_text.split()
    if not words:
        return None

    labels = [LABEL_NONE] * len(words)

    for vl in verb_labels:
        idx = vl.get("verb_idx")
        label = vl.get("label", LABEL_NONE)
        if idx is not None and 0 <= idx < len(words) and label in LABEL_TO_ID:
            labels[idx] = label

    return {"words": words, "labels": labels}


def build_dataset(
    data_path: Path,
    tokenizer,
    max_length: int = 256,
    max_samples: int = 0,
    data_type: str = "synthetic",
) -> Dataset:
    """Build a HuggingFace Dataset from either synthetic pairs or annotation data.

    For **synthetic** data: extracts words from ``dropped`` text and
    assigns verb_labels at the appropriate positions.

    For **annotation** data: extracts ``words`` and ``labels`` directly
    from the annotated sequences.

    Args:
        data_path: Path to a JSONL file.
        tokenizer: A HuggingFace tokenizer (e.g. DeBERTa).
        max_length: Maximum subword sequence length for tokenisation.
        max_samples: Maximum number of samples to include (0 = unlimited).
        data_type: ``"synthetic"`` or ``"annotation"``.

    Returns:
        A tokenised :class:`datasets.Dataset` ready for training.

    Raises:
        ValueError: If *data_type* is not recognised.
        FileNotFoundError: If *data_path* does not exist.
    """
    if data_type == "synthetic":
        raw_records = load_synthetic_pairs(data_path)
        examples: List[Dict[str, Any]] = []
        for rec in raw_records:
            converted = _synthetic_to_word_labels(rec)
            if converted is not None:
                examples.append(converted)
    elif data_type == "annotation":
        examples = load_annotation_data(data_path)
    else:
        raise ValueError(f"Unknown data_type: {data_type!r}. Expected 'synthetic' or 'annotation'.")

    if max_samples > 0 and len(examples) > max_samples:
        logger.info("Capping dataset from %d to %d samples", len(examples), max_samples)
        examples = examples[:max_samples]

    if not examples:
        logger.warning("No usable examples found in %s", data_path)
        # Return an empty but correctly-shaped dataset.
        return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})

    logger.info(
        "Building dataset with %d examples (type=%s, max_length=%d)",
        len(examples),
        data_type,
        max_length,
    )

    # Collect into column-oriented dict for Dataset.from_dict.
    words_col = [ex["words"] for ex in examples]
    labels_col = [ex["labels"] for ex in examples]

    dataset = Dataset.from_dict({"words": words_col, "labels": labels_col})

    # Tokenise and align labels.
    dataset = dataset.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer, max_length),
        batched=True,
        remove_columns=["words"],
        desc="Tokenizing and aligning labels",
    )

    # The 'labels' column is overwritten by tokenize_and_align_labels
    # to contain integer ids (with -100 for ignored positions).
    dataset.set_format("torch")

    logger.info("Dataset ready: %d examples, columns=%s", len(dataset), dataset.column_names)
    return dataset


# ── Class weight computation ────────────────────────────────────────────


def compute_class_weights(dataset: Dataset) -> np.ndarray:
    """Compute class weights inversely proportional to frequency.

    This helps handle the severe class imbalance in pronoun recovery,
    where NONE tokens vastly outnumber PRO.* tokens.  Weights are
    computed as ``total / (num_classes * count_per_class)``, with a
    floor of 1.0 for classes with zero occurrences.

    Args:
        dataset: A tokenised dataset with an integer ``labels`` column.

    Returns:
        Array of shape ``(NUM_LABELS,)`` with per-class weights.
    """
    counts = np.zeros(NUM_LABELS, dtype=np.float64)

    for example in dataset:
        label_ids = example["labels"]
        # Handle both tensor and list representations.
        if hasattr(label_ids, "numpy"):
            label_ids = label_ids.numpy()
        elif not isinstance(label_ids, np.ndarray):
            label_ids = np.array(label_ids)

        for label_id in label_ids:
            lid = int(label_id)
            if 0 <= lid < NUM_LABELS:
                counts[lid] += 1

    total = counts.sum()
    if total == 0:
        logger.warning("No valid label counts found; returning uniform weights")
        return np.ones(NUM_LABELS, dtype=np.float32)

    # Inverse frequency weighting: weight_c = total / (num_classes * count_c).
    weights = np.where(
        counts > 0,
        total / (NUM_LABELS * counts),
        1.0,
    )

    logger.info(
        "Class weights computed (total tokens=%d): %s",
        int(total),
        {ALL_LABELS[i]: f"{w:.3f}" for i, w in enumerate(weights)},
    )

    return weights.astype(np.float32)
