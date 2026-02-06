"""
LLM-based annotator for pronoun recovery.

Uses an OpenAI-compatible client (targeting DeepSeek by default) to
annotate text for null subjects using the bracket annotation format.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from ..config import LLMAnnotationConfig
from .manual_format import parse_annotations
from .prompt_templates import build_annotation_prompt

logger = logging.getLogger(__name__)


class LLMAnnotator:
    """Annotate text for null subjects using an LLM via OpenAI-compatible API.

    The annotator lazily initialises the API client on first use,
    reading the API key from the environment variable specified in
    config.api_key_env.

    Args:
        config: LLM annotation configuration.
    """

    def __init__(self, config: LLMAnnotationConfig):
        self.config = config
        self.client = None  # lazy init

    def _init_client(self) -> None:
        """Lazily initialise the OpenAI-compatible client."""
        if self.client is not None:
            return

        from openai import OpenAI

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"API key environment variable '{self.config.api_key_env}' "
                f"is not set or empty"
            )

        self.client = OpenAI(
            base_url=self.config.api_base_url,
            api_key=api_key,
        )
        logger.info(
            "Initialised LLM client: base_url=%s model=%s",
            self.config.api_base_url,
            self.config.model_name,
        )

    def annotate(
        self,
        text: str,
        seed_examples: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Annotate a single text for null subjects.

        Args:
            text: The text to annotate.
            seed_examples: Few-shot example dicts with 'original' and
                'annotated' keys.

        Returns:
            Dict with keys:
                - annotated_text: The LLM's annotated output.
                - markers: List of parsed AnnotationMarker dicts.
                - usage: Token usage dict (prompt_tokens, completion_tokens,
                  total_tokens).
                - raw_response: The raw completion text.
        """
        self._init_client()

        messages = build_annotation_prompt(
            text=text,
            language=self.config.language,
            seed_examples=seed_examples,
        )

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Extract response content
        choice = response.choices[0]
        annotated_text = choice.message.content.strip()

        # Parse annotation markers from the response
        clean_text, markers = parse_annotations(annotated_text)

        # Extract token usage
        usage = {}
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {
            "annotated_text": annotated_text,
            "clean_text": clean_text,
            "markers": [
                {
                    "label": m.label,
                    "lexical_form": m.lexical_form,
                    "position": m.position,
                }
                for m in markers
            ],
            "usage": usage,
            "raw_response": annotated_text,
        }

    def annotate_batch(
        self,
        texts: List[Dict[str, Any]],
        seed_examples: List[Dict[str, str]],
        delay_seconds: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Annotate multiple texts sequentially with optional rate limiting.

        Args:
            texts: List of dicts, each must contain a 'text' key.
            seed_examples: Few-shot examples for prompting.
            delay_seconds: Seconds to wait between requests (simple
                rate limiting).  For production rate limiting, use
                AnnotationBatchRunner instead.

        Returns:
            List of annotation result dicts. Each includes the original
            item's 'id' (if present) plus the annotation results.
        """
        results: List[Dict[str, Any]] = []

        for i, item in enumerate(texts):
            text = item["text"]
            item_id = item.get("id", f"item_{i}")

            logger.debug("Annotating item %s (%d/%d)", item_id, i + 1, len(texts))

            try:
                result = self.annotate(text, seed_examples)
                result["id"] = item_id
                result["status"] = "success"
            except Exception as e:
                logger.error("Failed to annotate item %s: %s", item_id, e)
                result = {
                    "id": item_id,
                    "status": "error",
                    "error": str(e),
                    "annotated_text": None,
                    "markers": [],
                    "usage": {},
                }

            results.append(result)

            if delay_seconds > 0 and i < len(texts) - 1:
                time.sleep(delay_seconds)

        logger.info(
            "Batch complete: %d/%d succeeded",
            sum(1 for r in results if r["status"] == "success"),
            len(results),
        )
        return results
