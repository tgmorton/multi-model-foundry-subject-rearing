#!/usr/bin/env python3
"""Enrichment evals before and after the null-agreement fix (docs/ENRICH_NULL_AGREEMENT_FIX.md).

Same models, same overt items; only the null items changed. Compares end-state
P(prefer overt) on subject-drop targets between the v1 and v2 stimuli, by target
person and by removal arm, with the baseline intervention as the reference that
enrichment should be read against.

    python analysis/eval_v2/enrich_fix_compare.py --out-csv compare.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nopron_priming_report as R  # noqa: E402

V1, V2 = "null_subj_v2_condition_matched_v1", "null_subj_v2_condition_matched_v2"
COLS = ["checkpoint_step", "category", "condition", "prefers_overt_meanlp", "log_prob_diff_overt_minus_null"]
SUBJ = ("subject_drop", "subject_drop_no_agreement")
ARM = re.compile(r"pdrop2_(?P<arm>[a-z]+?)(?P<k>\d+)_(?P<iv>[a-z]+)-")


def endstate(s3, bench, cell, last=3):
    """Per (category, condition) means over a run's last checkpoints (step > 0)."""
    d = R.get_parquet(s3, bench, "pairs", cell, COLS)
    d = d[d.checkpoint_step > 0]
    d = d[d.checkpoint_step.isin(sorted(d.checkpoint_step.unique())[-last:])]
    g = d.groupby(["category", "condition"])[COLS[3:]].mean().reset_index()
    g["run"], g["bench"] = cell, bench
    return g


def describe(run):
    m = ARM.search(run)
    if m:
        return "wave2", m.group("arm"), int(m.group("k")), m.group("iv")
    return "production", "none", 0, "base" if "-baseline-" in run else "enrichvm"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-csv", type=Path)
    a = ap.parse_args()
    s3 = R.s3c()
    done = sorted(c for c in R.list_cells(s3, V2) if c.startswith("gpt2_large"))
    v1 = set(R.list_cells(s3, V1))
    base = sorted(c for c in v1 if c.startswith("gpt2_large") and ("_base-" in c or "-baseline-" in c))
    print(f"v2 runs scored: {len(done)}; baseline-intervention reference runs: {len(base)}")
    jobs = [(V1, c) for c in done] + [(V2, c) for c in done] + [(V1, c) for c in base]
    with ThreadPoolExecutor(12) as ex:
        d = pd.concat(list(ex.map(lambda j: endstate(s3, *j), jobs)), ignore_index=True)
    d = d.join(pd.DataFrame(d.run.map(describe).tolist(), columns=["cohort", "arm", "k", "iv"], index=d.index))
    if a.out_csv:
        d.to_csv(a.out_csv, index=False)
    s = d[d.category.isin(SUBJ)]
    pd.set_option("display.width", 200)
    for cohort in ("wave2", "production"):
        x = s[s.cohort == cohort]
        if x[x.iv == "enrichvm"].empty:
            continue
        run_means = x.groupby(["bench", "iv", "condition", "run"]).prefers_overt_meanlp.mean().reset_index()
        t = run_means.groupby(["condition", "iv", "bench"]).prefers_overt_meanlp.mean().unstack(["iv", "bench"])
        print(f"\n{cohort}: end-state P(prefer overt), subject-drop targets, by person")
        print(t.round(3).rename(columns={V1: "v1", V2: "v2"}))
    w = s[(s.cohort == "wave2")].groupby(["bench", "iv", "arm", "run"]).prefers_overt_meanlp.mean().reset_index()
    t = w.groupby(["arm", "iv", "bench"]).prefers_overt_meanlp.mean().unstack(["iv", "bench"])
    print("\nwave2 by arm, all persons")
    print(t.round(3).rename(columns={V1: "v1", V2: "v2"}))


if __name__ == "__main__":
    main()
