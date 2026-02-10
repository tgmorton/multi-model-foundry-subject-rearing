#!/usr/bin/env python3
"""Async parallel re-annotation of Italian corpus using DeepSeek-R1.

Uses asyncio + AsyncOpenAI to fire many concurrent requests with no
rate limiting.  Checkpoints results to JSONL for safe resume.

Usage:
    python scripts/reannotate_r1_async.py \
        --input data/pronoun_recovery/annotations/it/checkpoint.jsonl \
        --output data/pronoun_recovery/annotations/it/r1_checkpoint.jsonl \
        --concurrency 50

To resume an interrupted run, just re-run the same command — it will
skip IDs already present in the output file.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import AsyncOpenAI

from analysis.pronoun_recovery.annotation.manual_format import (
    parse_annotations,
    validate_annotation,
)
from analysis.pronoun_recovery.annotation.prompt_templates import (
    build_annotation_prompt,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────

DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MAX_TOKENS = 8192  # R1 needs room for reasoning chain + output
DEFAULT_CONCURRENCY = 50
DEFAULT_SEED_PATH = "data/pronoun_recovery/annotations/it/manual/seed_set.jsonl"


# ── Checkpoint helpers ──────────────────────────────────────────────

def load_completed_ids(path: Path) -> Set[str]:
    """Load IDs already present in the output checkpoint."""
    if not path.exists():
        return set()
    ids: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def append_result(path: Path, result: Dict[str, Any]) -> None:
    """Append a single result to the checkpoint JSONL."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


# ── Async annotator ─────────────────────────────────────────────────

async def annotate_one(
    client: AsyncOpenAI,
    item: Dict[str, Any],
    seed_examples: List[Dict[str, str]],
    semaphore: asyncio.Semaphore,
    output_path: Path,
    model: str,
    max_tokens: int,
    counter: Dict[str, int],
    total: int,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Annotate a single item with retries, writing result on success."""
    item_id = item["id"]
    text = item.get("original_text") or item.get("text", "")

    messages = build_annotation_prompt(
        text=text,
        language="it",
        seed_examples=seed_examples,
    )

    async with semaphore:
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )

                annotated_text = response.choices[0].message.content.strip()

                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                # Parse and validate
                clean_text, markers = parse_annotations(annotated_text)
                validation_errors = []
                for m in markers:
                    validation_errors.extend(validate_annotation(m, language="it"))

                result = {
                    "id": item_id,
                    "genre": item.get("genre"),
                    "original_text": text,
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
                    "status": "success",
                    "validation_errors": validation_errors,
                    "is_valid": len(validation_errors) == 0,
                }

                # Checkpoint immediately
                append_result(output_path, result)

                counter["done"] += 1
                counter["tokens"] += usage.get("total_tokens", 0)
                n = counter["done"]
                if n % 25 == 0 or n == total:
                    elapsed = time.monotonic() - counter["start"]
                    rate = n / elapsed * 3600 if elapsed > 0 else 0
                    logger.info(
                        "Progress: %d/%d (%.0f/hr) | tokens: %d",
                        n, total, rate, counter["tokens"],
                    )

                return result

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    backoff = 2 ** attempt
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s — retrying in %ds",
                        attempt + 1, max_retries + 1, item_id, e, backoff,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error("All attempts failed for %s: %s", item_id, e)

        # All retries exhausted
        result = {
            "id": item_id,
            "genre": item.get("genre"),
            "original_text": text,
            "annotated_text": None,
            "markers": [],
            "usage": {},
            "status": "error",
            "error": str(last_error),
            "validation_errors": [f"Annotation failed: {last_error}"],
            "is_valid": False,
        }
        append_result(output_path, result)
        counter["done"] += 1
        return result


async def run_batch(
    items: List[Dict[str, Any]],
    seed_examples: List[Dict[str, str]],
    output_path: Path,
    model: str,
    max_tokens: int,
    concurrency: int,
    api_key: str,
    base_url: str,
) -> List[Dict[str, Any]]:
    """Run parallel annotation on all items."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    counter = {"done": 0, "tokens": 0, "start": time.monotonic()}
    total = len(items)

    logger.info(
        "Starting async annotation: %d items, concurrency=%d, model=%s",
        total, concurrency, model,
    )

    tasks = [
        annotate_one(
            client=client,
            item=item,
            seed_examples=seed_examples,
            semaphore=semaphore,
            output_path=output_path,
            model=model,
            max_tokens=max_tokens,
            counter=counter,
            total=total,
        )
        for item in items
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any unexpected exceptions
    final = []
    for r in results:
        if isinstance(r, Exception):
            logger.error("Unexpected task exception: %s", r)
        else:
            final.append(r)

    elapsed = time.monotonic() - counter["start"]
    logger.info(
        "Done: %d/%d succeeded in %.0fs (%.0f items/hr) | total tokens: %d",
        sum(1 for r in final if r.get("status") == "success"),
        total,
        elapsed,
        total / elapsed * 3600 if elapsed > 0 else 0,
        counter["tokens"],
    )

    return final


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Async R1 re-annotation of Italian pronoun recovery corpus",
    )
    parser.add_argument(
        "--input", required=True,
        help="Input JSONL (e.g. checkpoint.jsonl from original annotation)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL checkpoint (will resume if exists)",
    )
    parser.add_argument(
        "--seed-set", default=DEFAULT_SEED_PATH,
        help="Path to seed set JSONL for few-shot examples",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max completion tokens (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only first N items (0 = all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY not set")
        sys.exit(1)

    # Load input items
    input_path = Path(args.input)
    items = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info("Loaded %d items from %s", len(items), input_path)

    # Load seed examples
    seed_path = Path(args.seed_set)
    seed_examples = []
    with open(seed_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seed_examples.append(json.loads(line))
    logger.info("Loaded %d seed examples", len(seed_examples))

    # Filter out already-completed items
    output_path = Path(args.output)
    completed = load_completed_ids(output_path)
    if completed:
        logger.info("Resuming: %d items already completed", len(completed))
    pending = [item for item in items if item["id"] not in completed]

    if args.limit > 0:
        pending = pending[:args.limit]

    if not pending:
        logger.info("Nothing to do — all items already annotated")
        return

    logger.info("Pending: %d items", len(pending))

    asyncio.run(
        run_batch(
            items=pending,
            seed_examples=seed_examples,
            output_path=output_path,
            model=args.model,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            api_key=api_key,
            base_url=args.base_url,
        )
    )


if __name__ == "__main__":
    main()
