#!/usr/bin/env python3
"""Priming: which target form moves. Final-checkpoint overt and null target scores per item, with the
target split at the pronoun's position, for the 140 GPT-2 large Study 2 runs.

  fetch     pull pairs + per_token of each run's final checkpoint from S3 into the cache (about 5 GB of reads)
  analyse   join to the pairing table and print the decomposition by prime -> target cell and removal order

Effect = (overt target after overt prime - after null prime) - (null target, same contrast).
Summed over the target: d_pron (the pronoun token), d_next (the word that takes its place in the null
target), d_rest_o, d_rest_n.

    AWS_PROFILE=nrp python analysis/eval_v2/adhoc/priming_components.py fetch
    python analysis/eval_v2/adhoc/priming_components.py analyse
"""
import glob
import io
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(os.path.expanduser("~/.cache/subject-drop/report_cache"))
OUT = CACHE / "priming_components"
B = "thomas-subject-drop-artifacts"
V1, V2 = "null_subj_v2_nopron_plus_priming_v1", "null_subj_v2_nopron_plus_priming_v2"
RUN = re.compile(r"^gpt2_large-en-pdrop2_(robbianti|robbi|rand|all)(\d+)_(base|impcase|lemverb|rmexpl|enrichvm)-h0-s\d+$")
IV = {"base": "baseline", "impcase": "impoverish_case", "lemverb": "lemmatize_verbs", "rmexpl": "remove_expletive_sentences",
      "enrichvm": "enrich_verbal_morphology"}


def fetch() -> None:
    import boto3
    import pyarrow.parquet as pq
    OUT.mkdir(parents=True, exist_ok=True)

    def client():
        return boto3.Session(profile_name="nrp").client("s3", endpoint_url="https://s3-west.nrp-nautilus.io")

    def get(s3, bench, table, cell, cols):
        body = s3.get_object(Bucket=B, Key=f"eval_results/{bench}/{table}/cell_id={cell}.parquet")["Body"].read()
        return pq.read_table(io.BytesIO(body), columns=cols).to_pandas()

    def one(job):
        bench, run, shards = job
        f = OUT / f"{run}.parquet"
        if f.exists():
            return
        s3 = client()
        step, cell = max((int(get(s3, bench, "checkpoints", c, ["checkpoint_step"]).checkpoint_step.max()), c) for c in shards)
        p = get(s3, bench, "pairs", cell, ["checkpoint_step", "category", "item_id", "overt_mean_log_prob", "null_mean_log_prob"])
        p = p[(p.checkpoint_step == step) & p.category.str.contains("__p_")].drop(columns=["checkpoint_step", "category"])
        t = get(s3, bench, "per_token", cell, ["checkpoint_step", "category", "item_id", "pronoun_status", "per_token_log_prob", "per_token_ids"])
        t = t[(t.checkpoint_step == step) & t.category.str.contains("__p_")]
        o = t[t.pronoun_status == 1].set_index("item_id")
        n = t[t.pronoun_status == 0].set_index("item_id").reindex(o.index)
        rows = []
        for item_id, oi, ol, ni, nl in zip(o.index, o.per_token_ids, o.per_token_log_prob, n.per_token_ids, n.per_token_log_prob):
            i = next((j for j, (a, b) in enumerate(zip(oi, ni)) if a != b), None)
            if i is None:
                continue
            ol, nl = np.asarray(ol, float), np.asarray(nl, float)
            rows.append((item_id, i, ol[:i].sum(), nl[:i].sum(), ol[i], nl[i], ol[i + 1:].sum(), nl[i:].sum(), len(ol), len(nl)))
        s = pd.DataFrame(rows, columns=["item_id", "slot", "pre_o", "pre_n", "pron_o", "next_n", "rest_o", "rest_n", "len_o", "len_n"])
        d = p.merge(s, on="item_id", how="left")
        d["run"], d["step"] = run, step
        tmp = f.with_suffix(".tmp")
        d.to_parquet(tmp)
        tmp.rename(f)

    s3, jobs = client(), {}
    for bench in (V1, V2):
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=f"eval_results/{bench}/checkpoints/"):
            for o in page.get("Contents", []):
                c = o["Key"].split("cell_id=")[1][:-8]
                run = c.rsplit("__ckpt_shard", 1)[0]
                m = RUN.match(run)
                if m and c.endswith("of4") and ((m.group(3) == "enrichvm") == (bench == V2)):
                    jobs.setdefault((bench, run), []).append(c)
    with ThreadPoolExecutor(10) as ex:
        list(ex.map(one, [(b, r, sh) for (b, r), sh in sorted(jobs.items())]))
    print("runs cached:", len(list(OUT.glob("*.parquet"))))


def analyse() -> None:
    pd.set_option("display.width", 270)
    meta = pd.read_parquet(CACHE / "priming_deep_fixed" / "meta.parquet")
    cols = ["item_id", "overt_mean_log_prob", "null_mean_log_prob", "pron_o", "next_n", "rest_o", "rest_n"]
    frames = []
    for f in sorted(glob.glob(str(OUT / "*.parquet"))):
        d = pd.read_parquet(f)
        m = RUN.match(d.run.iloc[0])
        mm = meta[meta.cond_dir == IV[m.group(3)]]
        x = mm.merge(d[cols].add_suffix("_P").rename(columns={"item_id_P": "pid"}), on="pid") \
              .merge(d[cols].add_suffix("_N").rename(columns={"item_id_N": "nid"}), on="nid")
        x["run"], x["arm"] = d.run.iloc[0], m.group(1)
        frames.append(x)
    x = pd.concat(frames, ignore_index=True)
    x["overt_side"] = x.overt_mean_log_prob_P - x.overt_mean_log_prob_N
    x["null_side"] = x.null_mean_log_prob_P - x.null_mean_log_prob_N
    x["effect"] = x.overt_side - x.null_side
    x["d_pron"] = x.pron_o_P - x.pron_o_N
    x["d_next"] = x.next_n_P - x.next_n_N
    x["d_rest_o"] = x.rest_o_P - x.rest_o_N
    x["d_rest_n"] = (x.rest_n_P - x.rest_n_N) - x.d_next
    main = ["subject_drop", "subject_drop_no_agreement"]
    dw = x.relation.isin(["other pronoun", "same features"])
    cells = {
        "embedded -> main (different word)": dw & (x.prime_category == "embedded_drop") & x.target_category.isin(main),
        "main -> embedded": dw & x.prime_category.isin(main) & (x.target_category == "embedded_drop"),
        "embedded -> embedded": dw & (x.prime_category == "embedded_drop") & (x.target_category == "embedded_drop"),
        "conjunction -> embedded": dw & (x.prime_category == "conjunction") & (x.target_category == "embedded_drop"),
        "control -> main": dw & (x.prime_category == "control") & x.target_category.isin(main),
        "main block, different word": dw & x.prime_category.isin(main + ["expletive"]) & x.target_category.isin(main + ["expletive"]),
        "same word, main -> main": (x.relation == "same word") & x.prime_category.isin(main) & x.target_category.isin(main),
    }
    v = ["effect", "overt_side", "null_side", "d_pron", "d_rest_o", "d_next", "d_rest_n"]
    for arm in [None, "rand", "robbi", "robbianti", "all"]:
        rows = {}
        for nm, msk in cells.items():
            g = x[msk] if arm is None else x[msk & (x.arm == arm)]
            if len(g):
                r = g.groupby("run")[v].mean()
                rows[nm] = {**r.mean().to_dict(), "runs": len(r), "null_side>0": (r.null_side > 0).mean()}
        print("ALL RUNS" if arm is None else f"ARM {arm}")
        print(pd.DataFrame(rows).T.round(3).to_string(), "\n")


if __name__ == "__main__":
    {"fetch": fetch, "analyse": analyse}[sys.argv[1]]()
