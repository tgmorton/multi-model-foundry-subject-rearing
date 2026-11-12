#!/usr/bin/env python3
"""Write drawn decile sets as one stimulus root the eval runner can score.

Several sets drawn by sample_sets.py share items (the matched-position and natural-position
draws overlap), so the root holds every distinct item once and `item_key.csv` says which sets
an item belongs to. The runner scores the union; the analysis joins its results to
item_key.csv on item_id and selects a set.

Layout (the condition-matched layout eval_v2_cell.py --matched_stimuli_root expects):
  <root>/manifest.json                 format, per-file sha256, the sets and how they were drawn
  <root>/item_key.csv                  item_id -> pool, cand_id, cell, decile, set membership, scores
  <root>/baseline/en/<category>.csv    overt and null rows; condition = <cell>_d<decile>

Only the baseline condition is written. Intervention-matched versions are transformed from it.

    python build_stimuli_root.py --set matched=<dir>/sets/robbi_v1 --set natural=<dir>/sets/robbi_nat_v1 \\
        --set nopron=<dir>/sets/robbi_nopron_v1 --root evaluation/stimuli/null-subj-test10m-robbi-v1 \\
        --built 2026-11-12 --vetting "..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

KEEP = ["cell", "category", "decile", "form", "person", "ptype", "kind", "stem", "genre_group", "place", "position",
        "line_initial", "sent_initial", "null_bare_verb", "tier", "variant", "parent_cand_id", "context_words",
        "target_words", "surp_item", "surp_corpus", "decile_corpus"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME=DIR",
                    help="a set name and the sample_sets.py output directory it was drawn into")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--built", required=True, help="date the root was built (recorded in the manifest)")
    ap.add_argument("--vetting", required=True, help="who checked what, in one sentence")
    ap.add_argument("--allow-unreviewed", action="store_true")
    a = ap.parse_args()

    frames, set_meta = [], {}
    for spec in a.set:
        name, d = spec.split("=", 1)
        d = Path(d)
        s = pd.read_parquet(d / "sets.parquet")
        meta = json.loads((d / "sets.meta.json").read_text())
        rv = pd.read_csv(d / "reviewed.csv")
        ok = set(rv[rv.problem == "none"].cand_id)
        unread = int((~s.cand_id.isin(ok)).sum())
        if unread and not a.allow_unreviewed:
            raise SystemExit(f"{name}: {unread} drawn items have no accepting review")
        s = s.drop(columns=["item_id"])                # numbered per set; the root numbers the union
        s["set"], s["pool"] = name, meta.get("pool") or "natural"
        s = s.merge(rv[["cand_id", "null_grammatical"]], on="cand_id", how="left")
        frames.append(s)
        set_meta[name] = {**meta, "dir": d.name, "pairs": len(s), "unreviewed": unread,
                          "reviewed_sha256": sha256(d / "reviewed.csv")}
    allrows = pd.concat(frames, ignore_index=True)
    # one row per distinct item; an item drawn into two sets must be the same text in both
    key = ["pool", "cand_id"]
    for col in ("context", "target_overt", "target_null", "cell", "decile"):
        assert (allrows.groupby(key)[col].nunique() == 1).all(), col
    items = allrows.drop_duplicates(key).sort_values(key).reset_index(drop=True)
    items.insert(0, "item_id", range(1, len(items) + 1))
    member = allrows.groupby(key)["set"].agg(lambda x: "|".join(sorted(set(x)))).rename("sets").reset_index()
    items = items.drop(columns=["set"]).merge(member, on=key)
    for name in set_meta:
        items[f"in_{name}"] = items.sets.str.split("|").map(lambda x: name in x)

    out = a.root / "baseline" / "en"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.csv"):
        f.unlink()
    cols = ["item_id", "category", "condition", "pronoun_status", "context", "target", "hotspot_token",
            "hotspot_position", "language", "names", "generator", "cell", "decile", "pool", "cand_id", "sets"]
    shas = {}
    for category, g in items.groupby("category"):
        rows = []
        for r in g.itertuples():
            common = {"item_id": r.item_id, "category": category, "condition": f"{r.cell}_d{r.decile}",
                      "context": r.context, "language": "en", "names": "", "generator": f"test_10M:{r.stem}",
                      "cell": r.cell, "decile": r.decile, "pool": r.pool, "cand_id": r.cand_id, "sets": r.sets}
            rows.append({**common, "pronoun_status": 1, "target": r.target_overt, "hotspot_token": r.hotspot_token,
                         "hotspot_position": r.hotspot_overt})
            rows.append({**common, "pronoun_status": 0, "target": r.target_null, "hotspot_token": r.hotspot_token,
                         "hotspot_position": r.hotspot_null})
        d = pd.DataFrame(rows)[cols]
        bad = [(x.item_id, x.pronoun_status) for x in d.itertuples()
               if x.target.split()[x.hotspot_position] != x.hotspot_token]
        assert not bad, f"{category}: hotspot does not match at {bad[:5]}"
        d.to_csv(out / f"{category}.csv", index=False)
        shas[f"{category}.csv"] = sha256(out / f"{category}.csv")
    key_cols = (["item_id", "pool", "cand_id", "sets"] + [f"in_{n}" for n in set_meta] + KEEP
                + ["null_grammatical", "context", "target_overt", "target_null"])
    items[key_cols].to_csv(a.root / "item_key.csv", index=False)
    manifest = {
        "format_version": "condition-matched-stimuli.v1",
        "vetted": True,
        "vetting": a.vetting,
        "built": a.built,
        "description": "Decile-graded null-subject pairs sampled from the unseen test split (test_10M) with their "
                       "real preceding context; one distinct item per row pair, set membership in item_key.csv",
        "rater": sorted({m["label"] for m in set_meta.values()}),
        "sets": set_meta,
        "item_key_sha256": sha256(a.root / "item_key.csv"),
        "conditions": {"baseline": {"ablation_registry_name": "identity", "output_sha256": shas,
                                    "pairs": len(items), "rows": 2 * len(items)}},
    }
    (a.root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(items):,} distinct items ({len(allrows):,} set memberships) -> {a.root}")
    print(items.groupby("sets").size().to_string())
    print("manifest sha256", sha256(a.root / "manifest.json"))


if __name__ == "__main__":
    main()
