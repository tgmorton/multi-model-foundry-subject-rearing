"""mT5-based seq2seq model for pronoun recovery.

Wraps a HuggingFace ``AutoModelForSeq2SeqLM`` to provide convenient
single-example and batched generation interfaces.  Input text is
prepended with the task prefix ``"recover pronouns: "`` and the model
directly generates the pronoun-recovered output.
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

TASK_PREFIX = "recover pronouns: "


class Seq2SeqPronounModel:
    """Wrapper around mT5 for pronoun recovery generation.

    Loads a pretrained or fine-tuned seq2seq checkpoint and provides
    methods for generating pronoun-recovered text from pronoun-dropped
    input.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_input_length: int = 256,
        max_target_length: int = 256,
        num_beams: int = 4,
        length_penalty: float = 1.0,
    ):
        """Load model and tokenizer from path.

        Args:
            model_path: HuggingFace model hub identifier or local directory.
            device: PyTorch device string.  Auto-detected if ``None``.
            max_input_length: Maximum input token length.
            max_target_length: Maximum generation length.
            num_beams: Number of beams for beam search.
            length_penalty: Exponential length penalty for beam search.
        """
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading seq2seq tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info("Loading seq2seq model from %s", model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

        logger.info("Seq2SeqPronounModel ready on device=%s", self.device)

    # ── Single-example generation ─────────────────────────────────────

    def generate(self, text: str) -> str:
        """Generate pronoun-recovered text from a single input.

        Args:
            text: Pronoun-dropped input text.

        Returns:
            Generated text with pronouns recovered.
        """
        results = self.generate_batch([text])
        return results[0]

    # ── Batch generation ──────────────────────────────────────────────

    def generate_batch(self, texts: List[str]) -> List[str]:
        """Generate pronoun-recovered text for a batch of inputs.

        Args:
            texts: List of pronoun-dropped input strings.

        Returns:
            List of generated strings with pronouns recovered.
        """
        if not texts:
            return []

        prefixed = [TASK_PREFIX + t for t in texts]

        encoding = self.tokenizer(
            prefixed,
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_target_length,
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=True,
            )

        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, output_path: str) -> None:
        """Save model and tokenizer to directory.

        Args:
            output_path: Destination directory.  Created if it does not exist.
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving seq2seq model to %s", output_dir)
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info("Seq2seq model and tokenizer saved to %s", output_dir)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "Seq2SeqPronounModel":
        """Load from a saved checkpoint directory.

        Args:
            path: Path to a directory containing model and tokenizer files.
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            A ready-to-use :class:`Seq2SeqPronounModel` instance.
        """
        return cls(model_path=path, **kwargs)
