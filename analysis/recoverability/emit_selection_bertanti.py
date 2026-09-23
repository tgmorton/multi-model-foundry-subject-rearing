"""Add the 'bertanti' label to the frozen selection-v5 family — the
LEAST-recoverable-first arm (Thomas 2026-09-23).

Mechanism: a pure relabeling of the existing 'bert' label (BERT-250:1
speaker rater). info_decile is reversed (d -> 9-d), so the graded
ablation's unchanged rule "remove decile < k/10" now removes the least
recoverable pronouns first. rank_value is negated to keep the table
self-consistent (ascending rank_value == removal order).

Because deciles are equal-frequency bins, bertanti removes exactly the
same NUMBER of instances at every k as bert and rand — the three arms are
volume-matched, which is what makes the 3-way contrast clean.

rand_decile is copied verbatim so the shared random controls still serve
this arm.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

GENRES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles",
          "simple_wiki", "switchboard"]
SRC, LABEL = "bert", "bertanti"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/recoverability/analysis"))
    ap.add_argument("--genres", nargs="+", default=GENRES)
    a = ap.parse_args()
    sel = a.out / "selection_v5"
    fam = json.loads((sel / "V5_FAMILY.json").read_text())
    assert fam["selection_version"] == 5
    src_man = json.loads((sel / SRC / "SELECTION_MANIFEST.json").read_text())

    counts = {}
    for corpus in ("train_90M", "pull_10M"):
        cdir = sel / LABEL / corpus; cdir.mkdir(parents=True, exist_ok=True)
        n = 0
        for g in a.genres:
            d = pq.read_table(sel / SRC / corpus / f"{g}.parquet").to_pandas()
            d["info_decile"] = (9 - d.info_decile).astype(np.int8)
            d["rank_value"] = -d.rank_value
            pq.write_table(pa.Table.from_pandas(
                d[["line_idx","token_i","form","rank_value","info_decile",
                   "rand_decile"]], preserve_index=False), cdir / f"{g}.parquet")
            n += len(d)
        counts[corpus] = n
        print(f"{corpus}: {n:,} rows reversed")

    man = {
        "selection_version": 5, "ranking_label": LABEL,
        "ranking": f"REVERSED {src_man['ranking']} — least recoverable removed "
                   "first (decile 0 = least recoverable)",
        "derived_from": SRC, "ranking_inputs": src_man.get("ranking_inputs"),
        "population_n": counts["train_90M"],
        "volume_matched": "equal-frequency deciles => identical removal counts "
                          "per k as 'bert' and the shared rand arm",
        "cumulative_semantics": "condition K%% removes instances with decile "
                                "< K/10 (either arm)",
        "random_seed": src_man.get("random_seed"),
        "genres": sorted(a.genres),
        "pool": {"corpus": "pull_10M", "n": counts["pull_10M"],
                 "info": "reversed train-derived decile assignment"},
    }
    (sel / LABEL / "SELECTION_MANIFEST.json").write_text(json.dumps(man, indent=2))
    fam.setdefault("labels", {})[LABEL] = {
        "derived_from": SRC, "reversed": True, "added": "2026-09-23"}
    (sel / "V5_FAMILY.json").write_text(json.dumps(fam, indent=2))
    print(f"label '{LABEL}' emitted (reverse of '{SRC}')")

if __name__ == "__main__":
    main()
