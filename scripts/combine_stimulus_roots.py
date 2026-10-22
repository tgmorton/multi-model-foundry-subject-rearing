#!/usr/bin/env python3
"""Combine vetted matched-format stimulus roots into one root, so a single
eval pass per checkpoint scores all of them.

Each 3 GB gpt2_large checkpoint is then read from CephFS once for every
benchmark instead of once per benchmark. That halves the reads for
pronoun-free + priming, and it gives each pod enough scoring per byte to keep
its GPU above NRP's 40% floor (2026-10-22: alone, the pronoun-free eval ran at
8% GPU and priming at 28%).

    python scripts/combine_stimulus_roots.py \\
        --component evaluation/stimuli/null-subj-v2-matched-nopron-v1 \\
        --component evaluation/stimuli/null-subj-v2-priming-nopron-v1 \\
        --output-root evaluation/stimuli/null-subj-v2-nopron-plus-priming-v1

Components keep their file names, which must not collide, and their
(item_id, category, condition) pair keys, which must not collide either; the
eval joins pairs on those keys. Analysis tells the components apart by
category ("<cat>" vs "<cat>__p_<prime cat>").
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--component", type=Path, action="append", required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    comps = [c if c.is_absolute() else REPO_ROOT / c for c in args.component]
    out = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    if out.exists() and not args.force:
        raise SystemExit(f"output exists: {out} (pass --force to replace)")

    manifests = {}
    for c in comps:
        m = json.loads((c / "manifest.json").read_text())
        if m.get("format_version") != "condition-matched-stimuli.v1":
            raise SystemExit(f"{c}: not a condition-matched-stimuli.v1 root")
        manifests[c] = m
    conditions = sorted(set.intersection(*(set(m["conditions"]) for m in manifests.values())))
    if any(set(m["conditions"]) != set(conditions) for m in manifests.values()):
        raise SystemExit("components cover different condition sets")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        out_conditions = {}
        for cond in conditions:
            hashes, keys = {}, set()
            for c in comps:
                for src in sorted((c / cond / "en").glob("*.csv")):
                    if src.name in hashes:
                        raise SystemExit(f"{cond}: file name {src.name} in two components")
                    expected = manifests[c]["conditions"][cond]["output_sha256"].get(src.name)
                    if expected != sha256(src):
                        raise SystemExit(f"{src}: content does not match its component manifest")
                    with src.open(newline="", encoding="utf-8") as fh:
                        for row in csv.DictReader(fh):
                            k = (row["item_id"], row["category"], row["condition"], row["pronoun_status"])
                            if k in keys:
                                raise SystemExit(f"{cond}/{src.name}: pair key {k[:3]} collides across components")
                            keys.add(k)
                    dst = tmp / cond / "en" / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    hashes[src.name] = sha256(dst)
            out_conditions[cond] = {"output_sha256": hashes, "rows": len(keys)}
        manifest = {
            "format_version": "condition-matched-stimuli.v1",
            "kind": "combined",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "components": [{"root": str(c.relative_to(REPO_ROOT)),
                            "manifest_sha256": sha256(c / "manifest.json"),
                            "kind": manifests[c].get("kind", "matched"),
                            "vetted": manifests[c].get("vetted") is True} for c in comps],
            # Composition is mechanical (verbatim, hash-checked copies), so the
            # result is vetted exactly when every component is.
            "vetted": all(m.get("vetted") is True for m in manifests.values()),
            "conditions": out_conditions,
            "builder_sha256": sha256(Path(__file__)),
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out.relative_to(REPO_ROOT)}: {len(conditions)} conditions, "
          f"{out_conditions[conditions[0]]['rows']} rows per condition, vetted={manifest['vetted']}")


if __name__ == "__main__":
    main()
