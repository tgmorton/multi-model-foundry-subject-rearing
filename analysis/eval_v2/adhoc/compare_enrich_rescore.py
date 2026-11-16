#!/usr/bin/env python3
"""Integrity check for the enrichment re-score.

On overt sentences, which the null-agreement fix did not change, scores from August (matched v1,
before the 2026-10-06 fp16 rewrite) and from the re-score (matched v2) must agree checkpoint by
checkpoint.

    AWS_PROFILE=nrp python analysis/eval_v2/adhoc/compare_enrich_rescore.py <arch> <out.csv>   (run from the main checkout)
"""
import io
import sys

import boto3
import pandas as pd

arch = sys.argv[1]
s3 = boto3.Session(profile_name="nrp").client("s3", endpoint_url="https://s3-west.nrp-nautilus.io")
b = "thomas-subject-drop-artifacts"
keys = []
for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=b, Prefix=f"eval_results/null_subj_v2_condition_matched_v2/pairs/cell_id={arch}-en-enrich_verbal_morphology"):
    keys += [o["Key"] for o in page.get("Contents", [])]
cols = ["checkpoint_step", "item_id", "category", "condition", "overt_mean_log_prob"]
rows, flagged = [], []
for k in sorted(keys):
    rid = k.split("cell_id=")[1].replace(".parquet", "")
    v2 = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=b, Key=k)["Body"].read()), columns=cols)
    try:
        v1 = pd.read_parquet(f"data/eval_results/null_subj_v2_condition_matched_v1/pairs/cell_id={rid}.parquet", columns=cols)
    except FileNotFoundError:
        v1 = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=b, Key=k.replace("matched_v2", "matched_v1"))["Body"].read()), columns=cols)
    m = v1.merge(v2, on=cols[:4], suffixes=("_v1", "_v2"))
    m["d"] = (m.overt_mean_log_prob_v2 - m.overt_mean_log_prob_v1).abs()
    per = m.groupby("checkpoint_step").d.median()
    bad = per[per > 0.05]
    flagged += [(rid, int(s), round(float(v), 3)) for s, v in bad.items()]
    rows.append({"run": rid.split("morphology-")[1], "ckpt_v1": v1.checkpoint_step.nunique(), "ckpt_v2": v2.checkpoint_step.nunique(),
                 "missing_in_v2": sorted(set(v1.checkpoint_step) - set(v2.checkpoint_step)), "median_diff": round(float(m.d.median()), 4),
                 "p99_diff": round(float(m.d.quantile(0.99)), 4), "ckpts_off": len(bad)})
d = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(arch, ":", len(d), "runs on v2;", int((d.ckpt_v1 == d.ckpt_v2).sum()), "with every checkpoint;", int((d.ckpts_off == 0).sum()),
      "with no disagreeing checkpoint")
print("median |diff| across runs: %.4f; worst p99: %.4f" % (d.median_diff.median(), d.p99_diff.max()))
odd = d[(d.ckpt_v1 != d.ckpt_v2) | (d.ckpts_off > 0)]
print(odd.to_string(index=False) if len(odd) else "no run with missing or disagreeing checkpoints")
for f in flagged:
    print("DISAGREES", *f)
d.to_csv(sys.argv[2], index=False)
