#!/usr/bin/env python3
"""Validate condition-matched stimuli against production tokenizers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional, Tuple


def parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=/path/to/tokenizer")
    name, path = value.split("=", 1)
    return name, Path(path)


def load_rows(root: Path, condition: str):
    rows = []
    for path in sorted((root / condition / "en").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def validate_one(
    name: str,
    encode: Callable[[str], list],
    unk_id: Optional[int],
    root: Path,
    conditions: list,
) -> dict:
    report = {}
    for condition in conditions:
        rows = load_rows(root, condition)
        by_pair = defaultdict(dict)
        total_tokens = unknown_tokens = 0
        target_lengths = []
        for row in rows:
            full = " ".join(x for x in (row["context"], row["target"]) if x)
            ids = list(encode(full))
            total_tokens += len(ids)
            if unk_id is not None:
                unknown_tokens += sum(int(x == unk_id) for x in ids)
            target_ids = list(encode(row["target"]))
            target_lengths.append(len(target_ids))
            key = (row["category"], row["condition"], row["item_id"])
            by_pair[key][int(row["pronoun_status"])] = len(target_ids)
        deltas = Counter(v[1] - v[0] for v in by_pair.values())
        unk_rate = unknown_tokens / max(total_tokens, 1)
        if unk_rate > 0.01:
            raise ValueError(
                f"{name}/{condition}: UNK rate {unk_rate:.3%} exceeds 1%")
        report[condition] = {
            "rows": len(rows), "pairs": len(by_pair),
            "total_tokens": total_tokens, "unknown_tokens": unknown_tokens,
            "unknown_rate": unk_rate,
            "target_tokens_min": min(target_lengths),
            "target_tokens_max": max(target_lengths),
            "target_tokens_mean": sum(target_lengths) / len(target_lengths),
            "overt_minus_null_target_token_delta": {
                str(k): v for k, v in sorted(deltas.items())
            },
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("evaluation/stimuli/null-subj-v2-matched-v1"))
    ap.add_argument("--sentencepiece", action="append", default=[],
                    type=parse_named_path, metavar="NAME=PATH")
    ap.add_argument("--hf-tokenizer", action="append", default=[],
                    type=parse_named_path, metavar="NAME=PATH")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    conditions = sorted(p.name for p in args.root.iterdir()
                        if p.is_dir() and (p / "en").is_dir())
    all_reports = {}
    for name, path in args.sentencepiece:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        if not sp.load(str(path)):
            raise SystemExit(f"could not load {path}")
        unk = sp.unk_id() if sp.unk_id() >= 0 else None
        all_reports[name] = validate_one(
            name, lambda text, _sp=sp: list(_sp.encode(text)), unk,
            args.root, conditions)
    for name, path in args.hf_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        all_reports[name] = validate_one(
            name,
            lambda text, _tok=tok: list(_tok.encode(
                text, add_special_tokens=False)),
            tok.unk_token_id, args.root, conditions)
    if not all_reports:
        raise SystemExit("provide at least one tokenizer")
    payload = {"format_version": "condition-matched-tokenizer-validation.v1",
               "tokenizers": all_reports}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(rendered, encoding="utf-8")
        tmp.replace(args.output)
        print(f"wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
