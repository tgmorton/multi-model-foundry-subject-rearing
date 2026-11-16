#!/usr/bin/env python3
"""Priming tables with the 28 enrichment runs scored on the fixed (v2) stimuli.

priming_deep/ holds all 140 runs on the v1 stimuli; priming_deep_v2/ holds the
enrichment runs on v2 (docs/ENRICH_NULL_AGREEMENT_FIX.md). This writes
priming_deep_fixed/: v1 for every other intervention, v2 for enrichment. Pair ids
are shared, so the report reads it unchanged:

    python analysis/eval_v2/priming_deep_fixed.py
    python analysis/eval_v2/priming_deep_report.py --tables <cache>/priming_deep_fixed --out report.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nopron_priming_report as R  # noqa: E402
import priming_deep_data as P  # noqa: E402

V1, V2, OUT = R.CACHE / "priming_deep", R.CACHE / "priming_deep_v2", R.CACHE / "priming_deep_fixed"
ENRICH = P.IV_DIR["enrichvm"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = pd.read_parquet(V1 / "meta.parquet"), pd.read_parquet(V2 / "meta.parquet")
    assert set(a[a.cond_dir == ENRICH].pid) == set(b[b.cond_dir == ENRICH].pid)
    pd.concat([a[a.cond_dir != ENRICH], b[b.cond_dir == ENRICH]], ignore_index=True).to_parquet(OUT / "meta.parquet")
    runs = set(pd.read_parquet(V2 / "end.parquet", columns=["run"]).run.astype(str))
    for name in ("end", "traj", "side"):
        old, new = pd.read_parquet(V1 / f"{name}.parquet"), pd.read_parquet(V2 / f"{name}.parquet")
        keep = old[~old.run.astype(str).isin(runs)]
        assert keep.run.nunique() + new.run.nunique() == old.run.nunique(), name
        d = pd.concat([keep, new], ignore_index=True)
        for c in old.columns:
            if isinstance(old[c].dtype, pd.CategoricalDtype):
                d[c] = d[c].astype("category")
        d.to_parquet(OUT / f"{name}.parquet")
        print(name, len(old), "->", len(d), d.run.nunique(), "runs")


if __name__ == "__main__":
    main()
