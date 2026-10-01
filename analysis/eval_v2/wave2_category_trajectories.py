"""Overt-preference trajectories by removal decile, split by eval item
category (Thomas 2026-09-30).

One figure per intervention: rows = selection arms, columns = the 8 item
categories of the condition-matched null-subject stimuli, one line per
removal decile (viridis) plus the shared all-removed cell (dashed).
x = tokens seen (log), starting at the first training step; y = absolute
mean prefers_overt_meanlp (length-normalized mean log-prob preference).

Pairs parquets are synced from S3 into the same local cache the end-state
script uses (~/.cache/subject-drop/pairs_cache), in parallel with retries.

Usage: python analysis/eval_v2/wave2_category_trajectories.py \
           [--arms rand,robbi,robbianti] [--ivs base,impcase,...]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

TOKENS_PER_STEP = 256_000  # gpt2_small (see wave2_trajectories.py)
CACHE = Path.home() / ".cache" / "subject-drop" / "pairs_cache"
CELL = re.compile(r"pdrop2_(gpt2m|bertanti|bert|comp|robbianti|robbi|rand|all100)"
                  r"(\d*)_(\w+)-h(\d)-")
CATS = ["subject_drop", "subject_drop_no_agreement", "embedded_drop",
        "conjunction", "control", "extraction", "object_drop", "expletive"]
CAT_LABEL = {"subject_drop": "subject drop", "subject_drop_no_agreement":
             "subject drop (no agr.)", "embedded_drop": "embedded drop",
             "conjunction": "conjunction", "control": "control",
             "extraction": "extraction", "object_drop": "object drop",
             "expletive": "expletive"}
IV_LABEL = {"base": "baseline", "rmexpl": "remove expletives",
            "impcase": "impoverish case", "lemverb": "lemmatize verbs",
            "enrichvm": "enrich verbal morph"}
ARM_LABEL = {"rand": "random", "robbi": "RoBERTa ±250\nmost recoverable first",
             "robbianti": "RoBERTa ±250\nleast recoverable first",
             "bert": "BERT 250:1\nmost recoverable first",
             "bertanti": "BERT 250:1\nleast recoverable first",
             "gpt2m": "gpt2-medium", "comp": "composite"}


def sync(bucket, prefix, profile, endpoint):
    import boto3
    from botocore.config import Config
    from concurrent.futures import ThreadPoolExecutor
    s = boto3.Session(profile_name=profile).client(
        "s3", endpoint_url=endpoint,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                      read_timeout=120, max_pool_connections=16))
    CACHE.mkdir(parents=True, exist_ok=True)
    objs = [o for page in s.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=prefix)
            for o in page.get("Contents", []) if CELL.search(o["Key"])]

    def fetch(o):
        p = CACHE / o["Key"].rsplit("/", 1)[-1]
        if not (p.exists() and p.stat().st_size == o["Size"]):
            s.download_file(bucket, o["Key"], str(p))
        return p

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(fetch, objs))


def load(paths, arms, ivs, hp):
    want = set(arms) | {"all100"}
    rows = []
    for p in paths:
        arm, k, iv, h = CELL.search(p.name).groups()
        if arm not in want or iv not in ivs or int(h) != hp:
            continue
        d = pd.read_parquet(p, columns=["checkpoint_step", "category",
                                        "prefers_overt_meanlp"])
        d = d[d.checkpoint_step > 0]
        g = (d.groupby(["checkpoint_step", "category"], as_index=False)
             .prefers_overt_meanlp.mean())
        g["arm"], g["iv"] = arm, iv
        g["k"] = 100 if arm == "all100" else int(k)
        rows.append(g)
    df = pd.concat(rows, ignore_index=True)
    df["tokens"] = df.checkpoint_step * TOKENS_PER_STEP
    # average over replicates of the same cell
    return (df.groupby(["arm", "iv", "k", "category", "tokens"], as_index=False)
            .prefers_overt_meanlp.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="thomas-subject-drop-artifacts")
    ap.add_argument("--prefix", default="eval_results/null_subj_v2_condition_matched_v1/pairs/")
    ap.add_argument("--profile", default="nrp")
    ap.add_argument("--endpoint", default="https://s3-west.nrp-nautilus.io")
    ap.add_argument("--arms", default="rand,robbi,robbianti")
    ap.add_argument("--ivs", default="base,impcase,lemverb,enrichvm,rmexpl")
    ap.add_argument("--hp", type=int, default=0)
    ap.add_argument("--no-sync", action="store_true",
                    help="use the local cache as-is")
    ap.add_argument("--out", type=Path, default=Path("analysis/eval_v2/figures/wave2_v5"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arms, ivs = a.arms.split(","), a.ivs.split(",")

    paths = (sorted(CACHE.glob("*.parquet")) if a.no_sync
             else sync(a.bucket, a.prefix, a.profile, a.endpoint))
    df = load(paths, arms, ivs, a.hp)
    tag = "-".join(arms)
    df.to_csv(a.out / f"category_trajectories_{tag}.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cmap = plt.get_cmap("viridis")
    cats = [c for c in CATS if c in set(df.category)]
    ks = sorted(k for k in df.k.unique() if k < 100)

    for iv in [i for i in ivs if i in set(df.iv)]:
        sub_iv = df[df.iv == iv]
        rows = [r for r in arms if r in set(sub_iv.arm)]
        fig, axes = plt.subplots(len(rows), len(cats),
                                 figsize=(2.7 * len(cats), 2.5 * len(rows)),
                                 sharex=True, sharey=True, squeeze=False)
        for r, arm in enumerate(rows):
            for c, cat in enumerate(cats):
                ax = axes[r][c]
                s = sub_iv[(sub_iv.category == cat)]
                for k in ks:
                    t = s[(s.arm == arm) & (s.k == k)].sort_values("tokens")
                    if len(t):
                        ax.plot(t.tokens, t.prefers_overt_meanlp, lw=1.2,
                                color=cmap(k / 100))
                anc = s[s.arm == "all100"].sort_values("tokens")
                if len(anc):
                    ax.plot(anc.tokens, anc.prefers_overt_meanlp, lw=1.4,
                            ls="--", color="#A33B2E")
                ax.set_xscale("log")
                ax.set_ylim(0, 1)
                ax.axhline(0.5, color="#bbb", lw=0.6)
                if r == 0:
                    ax.set_title(CAT_LABEL.get(cat, cat), fontsize=9)
                if c == 0:
                    ax.set_ylabel(ARM_LABEL.get(arm, arm), fontsize=8)
                if r == len(rows) - 1:
                    ax.set_xlabel("tokens seen (log)", fontsize=8)
                ax.tick_params(labelsize=7)
        handles = [Line2D([0], [0], color=cmap(k / 100), lw=2) for k in ks]
        labels = [f"{k}% removed" for k in ks]
        handles.append(Line2D([0], [0], color="#A33B2E", lw=2, ls="--"))
        labels.append("all removed")
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"Overt-subject preference by item category — "
                     f"{IV_LABEL.get(iv, iv)} (gpt2_small, h{a.hp}; "
                     "lines = removal decile)", y=1.01)
        fig.tight_layout()
        out = a.out / f"category_trajectories_{tag}_{iv}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
