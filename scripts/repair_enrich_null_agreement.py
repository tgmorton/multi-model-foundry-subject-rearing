#!/usr/bin/env python3
"""Repair the enriched-morphology null items of a condition-matched stimulus root.

matched-v1 enriched each member of a pair independently, so a null sentence (no
pronoun to agree with) got a 3sg, bare or unmarked verb: "we cheerimus ..." was
paired with "cheerikt ...". Training does the opposite: pronoun_drop_stacked
enriches the raw parse and then deletes the pronoun, so a dropped subject keeps
its agreement. This writes a new root in which every enrichment null item is its
overt item minus the overt element, using the generator's own pair rule
(reconcile_shared_edits) on the rows already in the root. No parser is needed,
and everything else stays byte-identical to the input root, which is left alone.

    python scripts/repair_enrich_null_agreement.py \\
        --input-root evaluation/stimuli/null-subj-v2-matched-v1 \\
        --output-root evaluation/stimuli/null-subj-v2-matched-v2

Background and the evidence: docs/ENRICH_NULL_AGREEMENT_FIX.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_condition_matched_stimuli import SCHEMA, reconcile_shared_edits, write_csv  # noqa: E402

CONDITION = "enrich_verbal_morphology"
POLICY = "shared_source_token_edits_follow_pronoun_status_1_parse"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != SCHEMA:
            raise SystemExit(f"{path}: unexpected schema {reader.fieldnames}")
        return list(reader)


def by_pair(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    pairs: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        pairs.setdefault((row["category"], row["condition"], row["item_id"]), {})[row["pronoun_status"]] = row
    return pairs


def is_overt_minus_one_word(overt: str, null: str) -> bool:
    ops = [op for op in SequenceMatcher(a=overt.split(), b=null.split(), autojunk=False).get_opcodes()
           if op[0] != "equal"]
    return len(ops) == 1 and ops[0][0] == "delete" and ops[0][2] - ops[0][1] == 1


def repair_file(enriched: list[dict[str, str]], baseline: list[dict[str, str]], ledger: list[dict[str, str]]) -> None:
    """Rewrite the null rows of one enrichment file in place, recording every change."""
    source = by_pair(baseline)
    for key, member in by_pair(enriched).items():
        overt, null = member["1"], member["0"]
        fixed, _ = reconcile_shared_edits(source[key]["1"]["target"], source[key]["0"]["target"],
                                          overt["target"], null["target"])
        if not is_overt_minus_one_word(overt["target"], fixed):
            raise SystemExit(f"{key}: repaired null is not its overt item minus one word: "
                             f"{overt['target']!r} / {fixed!r}")
        hotspot = fixed.split()[int(null["hotspot_position"])]
        if fixed != null["target"]:
            ledger.append({"category": key[0], "condition": key[1], "item_id": key[2], "overt": overt["target"],
                           "old_null": null["target"], "new_null": fixed, "old_hotspot": null["hotspot_token"],
                           "new_hotspot": hotspot})
        null["target"], null["hotspot_token"] = fixed, hotspot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    src = args.input_root if args.input_root.is_absolute() else REPO_ROOT / args.input_root
    out = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    if out.exists() and not args.force:
        raise SystemExit(f"output exists: {out} (pass --force to replace)")
    manifest = json.loads((src / "manifest.json").read_text())
    if manifest.get("format_version") != "condition-matched-stimuli.v1" or manifest.get("vetted") is not True:
        raise SystemExit(f"{src}: not a vetted condition-matched-stimuli.v1 root")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        for cond, entry in manifest["conditions"].items():
            shutil.copytree(src / cond, tmp / cond)
            for name, expected in entry["output_sha256"].items():
                if sha256(src / cond / "en" / name) != expected:
                    raise SystemExit(f"{src / cond / 'en' / name}: content does not match its manifest")

        ledger: list[dict[str, str]] = []
        hashes = {}
        for name in manifest["conditions"][CONDITION]["output_sha256"]:
            rows = read_rows(src / CONDITION / "en" / name)
            repair_file(rows, read_rows(src / "baseline" / "en" / name), ledger)
            write_csv(tmp / CONDITION / "en" / name, rows)
            hashes[name] = sha256(tmp / CONDITION / "en" / name)
        with (tmp / CONDITION / "repair_ledger.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(ledger[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(ledger)

        entry = manifest["conditions"][CONDITION]
        entry.update(output_sha256=hashes, forced_shared_edits=len(ledger), pair_divergent_shared_edits=0)
        manifest["pair_policy"] = {"impoverish_case_lemmatize_and_enrich": POLICY}
        review = manifest.setdefault("gold_review", {})
        review.pop("enrichment_policy_acknowledged", None)
        review["enrich_null_repair"] = {
            "approved_by": "Thomas Morton", "approved_utc": "2026-11-04",
            "decision": "null items keep the verb agreement of their overt item, as in training",
            "check": "every pair in the condition is its overt item minus exactly one word",
        }
        manifest.update(
            created_utc=datetime.now(timezone.utc).isoformat(),
            derived_from={"root": str(src.relative_to(REPO_ROOT)), "manifest_sha256": sha256(src / "manifest.json")},
            repair={"script_sha256": sha256(Path(__file__)), "condition": CONDITION, "changed_null_rows": len(ledger),
                    "ledger": f"{CONDITION}/repair_ledger.csv",
                    "note": f"{CONDITION}/transformation_audit.jsonl describes the input root's independent application"},
        )
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out.relative_to(REPO_ROOT)}: {len(ledger)} null rows repaired in {CONDITION}")


if __name__ == "__main__":
    main()
