#!/usr/bin/env python3
"""Build the pronoun-free-context variant of the null-subject-v2 staging stimuli.

Every context that contains a personal, possessive or reflexive pronoun is
replaced by its reviewed rewrite from ``context_rewrites.csv``; targets and all
other columns are copied unchanged. The per-intervention matched set is then
built from this staging root by the production generator:

    python scripts/build_nopron_context_stimuli.py
    python scripts/generate_condition_matched_stimuli.py \\
        --input-root evaluation/stimuli/null-subj-v2-nopron/staging/en \\
        --output-root evaluation/stimuli/null-subj-v2-matched-nopron-v1

Rewrite rules, provenance and the analysis rule for first-person items:
evaluation/stimuli/null-subj-v2-nopron/README.md.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_condition_matched_stimuli import SCHEMA, write_csv  # noqa: E402

NOPRON_ROOT = REPO_ROOT / "evaluation/stimuli/null-subj-v2-nopron"
DEFAULT_INPUT = REPO_ROOT / "evaluation/stimuli/null-subj-v2/staging/en"
DEFAULT_OUTPUT = NOPRON_ROOT / "staging/en"
DEFAULT_REWRITES = NOPRON_ROOT / "context_rewrites.csv"

# Personal, possessive and reflexive pronouns. Contexts are lowercase and
# whitespace-tokenized, so a token match is exact.
PRONOUNS = frozenset(
    "i me my mine myself you your yours yourself yourselves he him his himself "
    "she her hers herself it its itself we us our ours ourselves they them "
    "their theirs themselves".split())


def has_pronoun(text: str) -> bool:
    return any(tok in PRONOUNS for tok in text.split())


def read_rewrites(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    mapping: dict[str, str] = {}
    for row in rows:
        old, new = row["original_context"], row["new_context"]
        if old in mapping:
            raise SystemExit(f"duplicate rewrite for context: {old!r}")
        if not has_pronoun(old):
            raise SystemExit(f"rewrite source has no pronoun: {old!r}")
        if has_pronoun(new):
            raise SystemExit(f"rewrite still contains a pronoun: {new!r}")
        mapping[old] = new
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--rewrites", type=Path, default=DEFAULT_REWRITES)
    ap.add_argument("--force", action="store_true", help="Replace an existing output root.")
    args = ap.parse_args()

    if args.output_root.exists() and not args.force:
        raise SystemExit(f"output exists: {args.output_root} (pass --force to replace)")
    mapping = read_rewrites(args.rewrites)

    used: Counter[str] = Counter()
    changed: dict[str, int] = {}
    tables: dict[str, list[dict[str, str]]] = {}
    for path in sorted(args.input_root.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != SCHEMA:
                raise SystemExit(f"{path}: schema {reader.fieldnames!r} != {SCHEMA!r}")
            rows = list(reader)
        n = 0
        for row in rows:
            new = mapping.get(row["context"])
            if new is not None:
                used[row["context"]] += 1
                row["context"] = new
                n += 1
            if has_pronoun(row["context"]):
                raise SystemExit(f"{path.name}: context has a pronoun and no rewrite: {row['context']!r}")
        tables[path.name] = rows
        changed[path.name] = n
    unused = sorted(set(mapping) - set(used))
    if unused:
        raise SystemExit(f"{len(unused)} rewrites match no staging context, e.g. {unused[0]!r}")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".nopron-", dir=args.output_root.parent))
    try:
        for name, rows in tables.items():
            write_csv(tmp / name, rows)
        if args.output_root.exists():
            shutil.rmtree(args.output_root)
        tmp.rename(args.output_root)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, n in changed.items():
        print(f"  {name:32s} {n:4d} of {len(tables[name]):4d} rows rewritten")
    print(f"wrote {args.output_root} ({len(mapping)} contexts rewritten, {sum(changed.values())} rows)")


if __name__ == "__main__":
    main()
