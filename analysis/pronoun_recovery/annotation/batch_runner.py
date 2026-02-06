"""
Checkpointed batch annotation runner for pronoun recovery.

Wraps LLMAnnotator with rate limiting, retry logic, checkpointing,
validation, and cost tracking for large-scale annotation runs.
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..config import LLMAnnotationConfig
from .llm_annotator import LLMAnnotator
from .manual_format import parse_annotations, validate_annotation

logger = logging.getLogger(__name__)


class _TokenBucket:
    """Simple token-bucket rate limiter supporting both RPM and TPM limits.

    Tracks two independent buckets: one for requests-per-minute and one
    for tokens-per-minute.  ``acquire`` blocks until both buckets have
    capacity.
    """

    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute

        # Bucket state: tokens available and last refill time
        self._req_tokens = float(self.rpm)
        self._tok_tokens = float(self.tpm)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill both buckets based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        # Refill at rate of (capacity / 60) per second
        self._req_tokens = min(
            float(self.rpm), self._req_tokens + elapsed * (self.rpm / 60.0)
        )
        self._tok_tokens = min(
            float(self.tpm), self._tok_tokens + elapsed * (self.tpm / 60.0)
        )

    def acquire(self, estimated_tokens: int = 1) -> None:
        """Block until capacity is available, then consume.

        Args:
            estimated_tokens: Estimated total tokens for this request
                (prompt + completion).
        """
        while True:
            with self._lock:
                self._refill()
                if self._req_tokens >= 1.0 and self._tok_tokens >= estimated_tokens:
                    self._req_tokens -= 1.0
                    self._tok_tokens -= estimated_tokens
                    return
            # Sleep briefly then retry
            time.sleep(0.1)

    def report_actual_tokens(self, actual_tokens: int, estimated_tokens: int) -> None:
        """Adjust the token bucket after learning the actual usage.

        If the actual usage was less than estimated, return the
        difference to the bucket.  If it was more, consume the extra.
        """
        with self._lock:
            diff = actual_tokens - estimated_tokens
            self._tok_tokens = max(0.0, self._tok_tokens - diff)


class AnnotationBatchRunner:
    """Checkpointed batch annotation with rate limiting, retry, and validation.

    Manages large-scale LLM annotation runs against an OpenAI-compatible
    API.  Features:

    - **Rate limiting**: Token-bucket algorithm respecting config RPM/TPM.
    - **Checkpointing**: Saves results every ``config.checkpoint_interval``
      items; resumes by skipping already-annotated IDs.
    - **Retry**: Exponential backoff up to ``config.max_retries`` attempts.
    - **Validation**: Parses each LLM response with ``parse_annotations``
      and runs ``validate_annotation`` on each marker, flagging
      unparseable or invalid responses.
    - **Cost tracking**: Logs cumulative input/output token counts.

    Args:
        config: LLM annotation configuration.
    """

    def __init__(self, config: LLMAnnotationConfig):
        self.config = config
        self.annotator = LLMAnnotator(config)
        self._bucket = _TokenBucket(
            requests_per_minute=config.requests_per_minute,
            tokens_per_minute=config.tokens_per_minute,
        )

        # Ensure output directory exists
        self._output_dir = Path(config.output_path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self._output_dir / "checkpoint.jsonl"

        # Cost tracking accumulators
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def run(
        self,
        candidates: List[Dict[str, Any]],
        seed_examples: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Run batch annotation with checkpointing and rate limiting.

        Args:
            candidates: List of candidate dicts (must have 'id' and 'text').
            seed_examples: Few-shot examples for prompting.

        Returns:
            List of annotation result dicts, one per candidate.
        """
        # Load already-completed IDs from checkpoint
        completed_ids = self._load_checkpoint()
        if completed_ids:
            logger.info(
                "Resuming: %d items already annotated (checkpoint)",
                len(completed_ids),
            )

        results: List[Dict[str, Any]] = []
        pending = [c for c in candidates if c["id"] not in completed_ids]
        logger.info(
            "Batch annotation: %d candidates total, %d pending",
            len(candidates),
            len(pending),
        )

        max_concurrent = getattr(self.config, "max_concurrent", 1)
        if max_concurrent > 1:
            results = self._run_concurrent(pending, seed_examples, max_concurrent)
        else:
            results = self._run_sequential(pending, seed_examples)

        # Summary
        n_valid = sum(1 for r in results if r.get("is_valid"))
        n_error = sum(1 for r in results if r.get("status") == "error")
        logger.info(
            "Batch complete: %d results (%d valid, %d errors, %d validation issues) | "
            "Total tokens: prompt=%d completion=%d",
            len(results),
            n_valid,
            n_error,
            len(results) - n_valid - n_error,
            self._total_prompt_tokens,
            self._total_completion_tokens,
        )

        return results

    # ------------------------------------------------------------------
    # Sequential / Concurrent execution
    # ------------------------------------------------------------------

    def _process_one(
        self,
        candidate: Dict[str, Any],
        seed_examples: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Annotate a single candidate with rate limiting and validation."""
        item_id = candidate["id"]
        text = candidate["text"]

        # Rate limiting: estimate tokens (rough heuristic)
        estimated_tokens = len(text.split()) * 4 + self.config.max_tokens
        self._bucket.acquire(estimated_tokens=estimated_tokens)

        # Annotate with retries
        result = self._annotate_with_retry(
            text=text,
            item_id=item_id,
            seed_examples=seed_examples,
        )

        # Attach original candidate metadata
        result["id"] = item_id
        result["genre"] = candidate.get("genre")
        result["source_file"] = candidate.get("source_file")
        result["line_number"] = candidate.get("line_number")
        result["original_text"] = text

        # Validate the annotation
        result["validation_errors"] = self._validate_result(result)
        result["is_valid"] = len(result["validation_errors"]) == 0

        # Adjust rate limiter with actual token usage
        actual_tokens = result.get("usage", {}).get("total_tokens", 0)
        if actual_tokens:
            self._bucket.report_actual_tokens(actual_tokens, estimated_tokens)

        return result

    def _run_sequential(
        self,
        pending: List[Dict[str, Any]],
        seed_examples: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Run annotation sequentially (original behavior)."""
        results: List[Dict[str, Any]] = []
        batch_buffer: List[Dict[str, Any]] = []

        for i, candidate in enumerate(pending):
            result = self._process_one(candidate, seed_examples)

            actual_tokens = result.get("usage", {}).get("total_tokens", 0)
            if actual_tokens:
                self._total_prompt_tokens += result["usage"].get("prompt_tokens", 0)
                self._total_completion_tokens += result["usage"].get("completion_tokens", 0)

            results.append(result)
            batch_buffer.append(result)

            if len(batch_buffer) >= self.config.checkpoint_interval:
                self._save_checkpoint(batch_buffer)
                batch_buffer = []
                logger.info(
                    "Progress: %d/%d | tokens: prompt=%d completion=%d",
                    i + 1,
                    len(pending),
                    self._total_prompt_tokens,
                    self._total_completion_tokens,
                )

        if batch_buffer:
            self._save_checkpoint(batch_buffer)

        return results

    def _run_concurrent(
        self,
        pending: List[Dict[str, Any]],
        seed_examples: List[Dict[str, str]],
        max_workers: int,
    ) -> List[Dict[str, Any]]:
        """Run annotation with concurrent workers."""
        results: List[Dict[str, Any]] = []
        batch_buffer: List[Dict[str, Any]] = []
        lock = threading.Lock()
        completed_count = 0

        logger.info("Running with %d concurrent workers", max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_candidate = {
                executor.submit(self._process_one, c, seed_examples): c
                for c in pending
            }

            for future in as_completed(future_to_candidate):
                result = future.result()

                with lock:
                    actual_tokens = result.get("usage", {}).get("total_tokens", 0)
                    if actual_tokens:
                        self._total_prompt_tokens += result["usage"].get("prompt_tokens", 0)
                        self._total_completion_tokens += result["usage"].get("completion_tokens", 0)

                    results.append(result)
                    batch_buffer.append(result)
                    completed_count += 1

                    if len(batch_buffer) >= self.config.checkpoint_interval:
                        self._save_checkpoint(batch_buffer)
                        batch_buffer = []
                        logger.info(
                            "Progress: %d/%d | tokens: prompt=%d completion=%d",
                            completed_count,
                            len(pending),
                            self._total_prompt_tokens,
                            self._total_completion_tokens,
                        )

        if batch_buffer:
            self._save_checkpoint(batch_buffer)

        return results

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _annotate_with_retry(
        self,
        text: str,
        item_id: str,
        seed_examples: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Annotate a single item with exponential backoff retries.

        Args:
            text: Text to annotate.
            item_id: Unique identifier for logging.
            seed_examples: Few-shot examples.

        Returns:
            Annotation result dict.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = self.annotator.annotate(text, seed_examples)
                result["status"] = "success"
                return result
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    backoff = 2 ** attempt
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. "
                        "Retrying in %ds...",
                        attempt + 1,
                        self.config.max_retries + 1,
                        item_id,
                        e,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "All %d attempts failed for %s: %s",
                        self.config.max_retries + 1,
                        item_id,
                        e,
                    )

        return {
            "status": "error",
            "error": str(last_error),
            "annotated_text": None,
            "clean_text": None,
            "markers": [],
            "usage": {},
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_result(self, result: Dict[str, Any]) -> List[str]:
        """Validate an annotation result.

        Parses the annotated text and validates each marker against the
        language-specific constraints.

        Returns:
            List of error messages.  Empty means valid.
        """
        errors: List[str] = []

        if result.get("status") == "error":
            errors.append(f"Annotation failed: {result.get('error', 'unknown')}")
            return errors

        annotated_text = result.get("annotated_text")
        if annotated_text is None:
            errors.append("No annotated text in result")
            return errors

        # Re-parse to verify the response is well-formed
        try:
            clean_text, markers = parse_annotations(annotated_text)
        except Exception as e:
            errors.append(f"Failed to parse annotation: {e}")
            return errors

        # Validate each marker
        for marker in markers:
            marker_errors = validate_annotation(
                marker, language=self.config.language
            )
            errors.extend(marker_errors)

        return errors

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> Set[str]:
        """Load IDs of already-annotated items from the checkpoint file.

        Returns:
            Set of completed item IDs.
        """
        if not self._checkpoint_path.exists():
            return set()

        completed: Set[str] = set()
        try:
            with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        item_id = record.get("id")
                        if item_id is not None:
                            completed.add(str(item_id))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed checkpoint line")
        except OSError as e:
            logger.warning("Could not read checkpoint file: %s", e)

        return completed

    def _save_checkpoint(self, results: List[Dict[str, Any]]) -> None:
        """Append results to the checkpoint JSONL file.

        Args:
            results: List of annotation result dicts to persist.
        """
        try:
            with open(self._checkpoint_path, "a", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            logger.debug(
                "Checkpointed %d results to %s",
                len(results),
                self._checkpoint_path,
            )
        except OSError as e:
            logger.error("Failed to save checkpoint: %s", e)
