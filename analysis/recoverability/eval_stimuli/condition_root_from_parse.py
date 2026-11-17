#!/usr/bin/env python3
"""Add a training condition's rendering to a test-set stimulus root, from the corpus parse.

build_stimuli_root.py writes only the baseline text. scripts/generate_condition_matched_stimuli.py,
which renders the constructed matched items, re-parses each sentence on its own and cannot handle
natural text (a contraction such as "don't" becomes two words under enrichment). Here every item is
real corpus text, so the condition is rendered the way the training corpus was: the intervention's
token edits are applied to the parsed corpus line and the item is cut out of the edited line
(render_items_condition.py). The null version is the edited overt sentence with the pronoun deleted,
the order training uses (edit the raw parse, then drop the pronoun), so a dropped subject keeps the
agreement its verb got from it.

Reads <root>/baseline/en/*.csv and the set folders the root was built from; writes
<root>/<condition>/en/<category>.csv (same rows and columns) and adds the condition to the manifest.

    python analysis/recoverability/eval_stimuli/condition_root_from_parse.py --root <root> \\
        --set matched=<dir> --set natural=<dir> --set nopron=<dir> --annotated <docbin dir> \\
        --raw data/raw/test_10M --intervention enrich_verbal_morphology
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--set", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--annotated", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--ext", default="test")
    ap.add_argument("--intervention", required=True, help="also the condition folder name")
    a = ap.parse_args()

    import score_pronoun_recoverability as S
    from render_items_condition import load_line_info, render_item

    edit_fn, line_remover = S.resolve_intervention(a.intervention)
    key = pd.read_csv(a.root / "item_key.csv", usecols=["item_id", "pool", "cand_id"])
    frames = []
    for spec in a.set:
        name, d = spec.split("=", 1)
        s = pd.read_parquet(Path(d) / "sets.parquet")
        meta = json.loads((Path(d) / "sets.meta.json").read_text())
        s["pool"] = meta.get("pool") or "natural"
        frames.append(s[["pool", "cand_id", "stem", "line_idx", "char_start", "rater_lines", "rater_char_start", "target_overt"]])
    src = pd.concat(frames).drop_duplicates(["pool", "cand_id"])
    items = key.merge(src, on=["pool", "cand_id"], how="left")
    assert items.stem.notna().all(), "items of the root missing from the set folders"

    rendered, failed = {}, {}
    for stem, g in items.groupby("stem", sort=False):
        raw = [t.rstrip("\r") for t in (a.raw / f"{stem}.{a.ext}").read_text(encoding="utf-8").split("\n")]
        need = set()
        fr = {}
        for r in g.itertuples():
            fr[r.item_id] = json.loads(r.rater_lines)
            need.update(range(int(r.line_idx) - len(fr[r.item_id]) + 1, int(r.line_idx) + 1))
        info = load_line_info(a.annotated, stem, need, edit_fn, line_remover, S)
        for r in g.itertuples():
            x = render_item(fr[r.item_id], int(r.line_idx), int(r.char_start), int(r.rater_char_start), info, raw,
                            target_len=len(r.target_overt))
            if x["why"] is not None:
                failed[r.item_id] = x["why"]
                continue
            last, ts, off = x["lines"][-1], x["target_start"], x["char_start"]
            overt = last[ts:]
            o = off - ts
            end = o + len(x["form_text"])
            if o < 0 or overt[o:end] != x["form_text"]:
                failed[r.item_id] = "pronoun_outside_target"
                continue
            null = overt[:o] + overt[end + 1:]
            wi = len(overt[:o].split())
            words = overt.split()
            if wi + 1 >= len(words) or null.split()[wi] != words[wi + 1]:
                failed[r.item_id] = "no_hotspot"
                continue
            ctx = " ".join(f.strip() for f in x["lines"][:-1] + [last[:ts]] if f.strip())
            rendered[r.item_id] = {"context": ctx, 1: overt, 0: null, "hotspot_token": words[wi + 1],
                                   "pos1": wi + 1, "pos0": wi, "n_edits": x["n_edits"]}
        print(f"{stem}: {len(g)} items", flush=True)

    out = a.root / a.intervention / "en"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.csv"):
        f.unlink()
    shas, n_rows, changed = {}, 0, 0
    for f in sorted((a.root / "baseline" / "en").glob("*.csv")):
        b = pd.read_csv(f, dtype=str, keep_default_na=False)
        rows = []
        for r in b.to_dict("records"):
            x = rendered.get(int(r["item_id"]))
            if x is None:
                continue
            st = int(r["pronoun_status"])
            new = dict(r, context=x["context"], target=x[st], hotspot_token=x["hotspot_token"],
                       hotspot_position=str(x["pos1"] if st == 1 else x["pos0"]))
            changed += new["target"] != r["target"]
            assert new["target"].split()[int(new["hotspot_position"])] == new["hotspot_token"], r["item_id"]
            rows.append(new)
        d = pd.DataFrame(rows)[list(b.columns)]
        d.to_csv(out / f.name, index=False)
        shas[f.name] = sha256(out / f.name)
        n_rows += len(d)
    m = json.loads((a.root / "manifest.json").read_text())
    m["conditions"][a.intervention] = {
        "ablation_registry_name": a.intervention,
        "method": "token edits applied to the parsed corpus line (condition_root_from_parse.py); the null item is the "
                  "edited overt sentence with the pronoun deleted",
        "output_sha256": shas, "pairs": n_rows // 2, "rows": n_rows, "changed_rows": int(changed),
        "items_not_rendered": {str(k): v for k, v in sorted(failed.items())},
    }
    (a.root / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    print(f"{a.intervention}: {n_rows // 2} pairs written, {len(failed)} items not rendered "
          f"({pd.Series(failed).value_counts().to_dict() if failed else {}}); {changed} of {n_rows} targets differ from baseline")
    print("manifest sha256", sha256(a.root / "manifest.json"))


if __name__ == "__main__":
    main()
