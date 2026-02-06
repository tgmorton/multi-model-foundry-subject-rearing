"""Dataset construction for seq2seq pronoun recovery (mT5).

Converts synthetic pairs into a HuggingFace Dataset suitable for
training an encoder-decoder model.  Each example maps the
pronoun-dropped text (``dropped``) to the original text with pronouns
(``original``).  The task prefix ``"recover pronouns: "`` is prepended
to the input during tokenisation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset

from .dataset import load_synthetic_pairs
from .seq2seq_model import TASK_PREFIX

logger = logging.getLogger(__name__)


def _tokenize_seq2seq(
    examples: Dict[str, List],
    tokenizer,
    max_input_length: int = 256,
    max_target_length: int = 256,
) -> Dict[str, List]:
    """Tokenize input/target pairs for seq2seq training.

    Prepends the task prefix to each input and tokenises both input and
    target sequences.  Target token ids are stored in the ``labels``
    field, with padding positions set to -100 so that they are ignored
    by the loss.

    This is designed as a batched mapping function for
    :meth:`datasets.Dataset.map`.

    Args:
        examples: Batched dict with ``input_text`` and ``target_text``.
        tokenizer: A HuggingFace tokenizer (e.g. mT5).
        max_input_length: Maximum input token length.
        max_target_length: Maximum target token length.

    Returns:
        Dict with ``input_ids``, ``attention_mask``, and ``labels``.
    """
    inputs = [TASK_PREFIX + text for text in examples["input_text"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    # Replace padding token ids in labels with -100 so they are ignored by loss.
    label_ids = labels["input_ids"]
    model_inputs["labels"] = [
        [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
        for seq in label_ids
    ]

    return model_inputs


def build_seq2seq_dataset(
    data_path: Path,
    tokenizer,
    max_input_length: int = 256,
    max_target_length: int = 256,
    max_samples: int = 0,
) -> Dataset:
    """Build a HuggingFace Dataset for seq2seq training from synthetic pairs.

    Each synthetic pair has a ``dropped`` field (input) and an
    ``original`` field (target).

    Args:
        data_path: Path to a JSONL file of synthetic pairs.
        tokenizer: A HuggingFace tokenizer (e.g. mT5).
        max_input_length: Maximum input token length.
        max_target_length: Maximum target token length.
        max_samples: Maximum number of samples (0 = unlimited).

    Returns:
        A tokenised :class:`datasets.Dataset` ready for training.
    """
    raw_records = load_synthetic_pairs(data_path)

    # Extract dropped/original pairs.
    input_texts: List[str] = []
    target_texts: List[str] = []
    for rec in raw_records:
        dropped = rec.get("dropped", "").strip()
        original = rec.get("original", "").strip()
        if dropped and original:
            input_texts.append(dropped)
            target_texts.append(original)

    if max_samples > 0 and len(input_texts) > max_samples:
        logger.info("Capping seq2seq dataset from %d to %d samples", len(input_texts), max_samples)
        input_texts = input_texts[:max_samples]
        target_texts = target_texts[:max_samples]

    if not input_texts:
        logger.warning("No usable seq2seq examples found in %s", data_path)
        return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})

    logger.info(
        "Building seq2seq dataset with %d examples (max_input=%d, max_target=%d)",
        len(input_texts),
        max_input_length,
        max_target_length,
    )

    dataset = Dataset.from_dict({
        "input_text": input_texts,
        "target_text": target_texts,
    })

    dataset = dataset.map(
        lambda batch: _tokenize_seq2seq(
            batch, tokenizer, max_input_length, max_target_length
        ),
        batched=True,
        remove_columns=["input_text", "target_text"],
        desc="Tokenizing seq2seq pairs",
    )

    dataset.set_format("torch")

    logger.info("Seq2seq dataset ready: %d examples, columns=%s", len(dataset), dataset.column_names)
    return dataset
