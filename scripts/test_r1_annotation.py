#!/usr/bin/env python3
"""Run DeepSeek-R1 on validation examples and save annotated output.

Reads the 100 validation chunks from /tmp/validation_chunk_{0-3}.jsonl,
re-annotates each using R1 with the revised prompt, and writes results
to /tmp/r1_annotated.jsonl for review.
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pronoun_recovery.annotation.prompt_templates import (
    build_annotation_prompt,
)

load_dotenv()

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("ERROR: DEEPSEEK_API_KEY not set")
    sys.exit(1)

client = OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=API_KEY,
)

MODEL = "deepseek-reasoner"
OUTPUT_PATH = "/tmp/r1_annotated.jsonl"

# Load seed examples for few-shot prompting
SEED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/pronoun_recovery/annotations/it/manual/seed_set.jsonl",
)
seed_examples = []
with open(SEED_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            seed_examples.append(json.loads(line))

# Load validation chunks
examples = []
for i in range(4):
    chunk_path = f"/tmp/validation_chunk_{i}.jsonl"
    with open(chunk_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

print(f"Loaded {len(examples)} examples, {len(seed_examples)} seed examples")

# Load any previous results to skip already-successful ones
prev_results = {}
try:
    with open(OUTPUT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                # Only keep results that didn't hit token limit (non-empty output)
                r1_text = r.get("r1_annotated", "") or ""
                if r.get("status") == "success" and len(r1_text.strip()) > 0:
                    prev_results[r["id"]] = r
    print(f"Loaded {len(prev_results)} previous successful results (skipping those)")
except FileNotFoundError:
    pass

# Process each example
results = []
for idx, ex in enumerate(examples):
    text = ex.get("original_text", "") or ex.get("text", "")
    ex_id = ex.get("id", f"item_{idx}")

    # Skip if we already have a good result
    if ex_id in prev_results:
        results.append(prev_results[ex_id])
        print(f"[{idx+1}/{len(examples)}] Skipping {ex_id} (cached)")
        continue

    messages = build_annotation_prompt(
        text=text,
        language="it",
        seed_examples=seed_examples,
    )

    print(f"[{idx+1}/{len(examples)}] Annotating {ex_id}...", end=" ", flush=True)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=8192,
        )

        annotated_text = response.choices[0].message.content.strip()

        # R1 may include reasoning in a separate field; extract just content
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        result = {
            "id": ex_id,
            "text": text,
            "deepseek_annotated": ex.get("annotated_text", ""),
            "r1_annotated": annotated_text,
            "usage": usage,
            "status": "success",
        }
        print(f"OK ({usage.get('total_tokens', '?')} tokens)")

    except Exception as e:
        result = {
            "id": ex_id,
            "text": text,
            "deepseek_annotated": ex.get("annotated_text", ""),
            "r1_annotated": None,
            "status": "error",
            "error": str(e),
        }
        print(f"ERROR: {e}")

    results.append(result)

    # Write checkpoint after each example
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Small delay to avoid rate limits
    if idx < len(examples) - 1:
        time.sleep(1.0)

print(f"\nDone! {sum(1 for r in results if r['status'] == 'success')}/{len(results)} succeeded")
print(f"Results saved to {OUTPUT_PATH}")
