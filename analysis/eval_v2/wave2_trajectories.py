"""Wave2 (w2v5) overt-preference trajectory figures — Thomas 2026-09-18.

Set A: per (arm x intervention) panel, overt-preference (mean
prefers_overt_slor) vs tokens seen (log-x), one line per decile k.
Set B: per (decile k x arm) panel, interventions compared on the same
axes (plus a delta-vs-base variant), log-x tokens.

Input: null_subj_v2 pairs parquets for pdrop2_* runs — pulled from S3
(--s3) or a local directory (--pairs-dir). gpt2_small tokens/step =
256,000 (verified against training logs).

Output: analysis/eval_v2/figures/wave2_v5/{setA_trajectories,
setB_decile_interventions,setB_delta_vs_base}.png + tidy CSV.
"""

from __future__ import annotations

import argparse
import glob
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

TOKENS_PER_STEP = 256_000
ARMS = ["rand", "gpt2m", "bert", "comp"]
IVS = ["base", "rmexpl", "impcase", "lemverb", "enrichvm"]
IV_LABEL = {"base": "baseline", "rmexpl": "remove expletives",
            "impcase": "impoverish case", "lemverb": "lemmatize verbs",
            "enrichvm": "enrich verbal morph"}
CELL_RE = re.compile(r"pdrop2_(gpt2m|bert|comp|rand|all100)(\d*)_(\w+)")


def parse_cell(run_id: str):
    m = CELL_RE.search(run_id)
    if not m:
        return None
    arm, k, iv = m.group(1), m.group(2), m.group(3)
    if arm == "all100":
        return ("all", 100, iv)
    return (arm, int(k) if k else 100, iv)


def load_pairs(args) -> pd.DataFrame:
    frames = []
    if args.pairs_dir:
        files = glob.glob(str(Path(args.pairs_dir) / "*.parquet"))
        files = [f for f in files if "pdrop2" in f]
        for f in files:
            frames.append(pd.read_parquet(
                f, columns=["cell_id", "checkpoint_step",
                            "prefers_overt_slor"]))
    else:
        import boto3
        s = boto3.Session(profile_name=args.profile).client(
            "s3", endpoint_url=args.endpoint)
        pag = s.get_paginator("list_objects_v2")
        for page in pag.paginate(Bucket=args.bucket, Prefix=args.s3_prefix):
            for o in page.get("Contents", []):
                if "pdrop2" not in o["Key"] or not o["Key"].endswith(".parquet"):
                    continue
                body = s.get_object(Bucket=args.bucket, Key=o["Key"])["Body"].read()
                frames.append(pd.read_parquet(
                    io.BytesIO(body),
                    columns=["cell_id", "checkpoint_step",
                             "prefers_overt_slor"]))
    if not frames:
        raise SystemExit("no pdrop2 pairs found")
    df = pd.concat(frames, ignore_index=True)
    meta = df.cell_id.map(parse_cell)
    df["arm"] = meta.map(lambda t: t and t[0])
    df["k"] = meta.map(lambda t: t and t[1])
    df["iv"] = meta.map(lambda t: t and t[2])
    df = df[df.arm.notna()]
    df["tokens"] = df.checkpoint_step * TOKENS_PER_STEP
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=None,
                    help="local pairs dir (else pull from S3)")
    ap.add_argument("--bucket", default="thomas-subject-drop-artifacts")
    ap.add_argument("--s3-prefix", default="eval_results/null_subj_v2/pairs/")
    ap.add_argument("--profile", default="nrp")
    ap.add_argument("--endpoint", default="https://s3-west.nrp-nautilus.io")
    ap.add_argument("--out", type=Path,
                    default=Path("analysis/eval_v2/figures/wave2_v5"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_pairs(args)
    # preference rate per (cell, checkpoint), averaged over items (and
    # over replicates when >1 run per cell). Init checkpoint (step<=0,
    # "checkpoint -1") is excluded: trajectories BEGIN at the first
    # training step, at their absolute value there (2026-09-18).
    traj = (df[df.tokens > 0]
            .groupby(["arm", "iv", "k", "tokens"], as_index=False)
            .prefers_overt_slor.mean()
            .rename(columns={"prefers_overt_slor": "overt_pref"}))
    traj.to_csv(args.out / "wave2_v5_trajectories.csv", index=False)
    print(f"{traj.cell_count if hasattr(traj,'cell_count') else len(traj):,} "
          f"trajectory points; arms {sorted(traj.arm.unique())}; "
          f"ivs {sorted(traj.iv.unique())}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")

    arms = [a for a in ARMS if a in set(traj.arm)]
    ivs = [i for i in IVS if i in set(traj.iv)]

    # ---- Set A: arm x intervention, lines = deciles ----
    fig, axes = plt.subplots(len(arms), len(ivs),
                             figsize=(3.6 * len(ivs), 2.9 * len(arms)),
                             sharex=True, sharey=True, squeeze=False)
    for r, arm in enumerate(arms):
        for c, iv in enumerate(ivs):
            ax = axes[r][c]
            sub = traj[(traj.arm == arm) & (traj.iv == iv)]
            for k in sorted(sub.k.unique()):
                t = sub[sub.k == k].sort_values("tokens")
                ax.plot(t.tokens, t.overt_pref, lw=1.4,
                        color=cmap(k / 100), label=f"k={k}")
            all_sub = traj[(traj.arm == "all") & (traj.iv == iv)]
            if len(all_sub):
                t = all_sub.sort_values("tokens")
                ax.plot(t.tokens, t.overt_pref, lw=1.6, ls="--",
                        color="#A33B2E", label="all removed")
            ax.set_xscale("log")
            ax.axhline(0.5, color="#bbb", lw=0.6)
            if r == 0:
                ax.set_title(IV_LABEL.get(iv, iv), fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{arm}\novert pref.", fontsize=9)
            if r == len(arms) - 1:
                ax.set_xlabel("tokens seen (log)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Overt-subject preference trajectories — arm × intervention, "
                 "lines = removal decile", y=1.005)
    fig.tight_layout()
    fig.savefig(args.out / "setA_trajectories.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Set B: decile x arm, lines = interventions ----
    ks = sorted(k for k in traj.k.unique() if k < 100)
    iv_colors = dict(zip(IVS, ["#3A3A35", "#B0562B", "#2C6E63",
                               "#7A6A9E", "#B08C2B"]))
    for name, delta in (("setB_decile_interventions", False),
                        ("setB_delta_vs_base", True)):
        fig, axes = plt.subplots(len(ks), len(arms),
                                 figsize=(3.6 * len(arms), 2.5 * len(ks)),
                                 sharex=True, sharey=delta, squeeze=False)
        for r, k in enumerate(ks):
            for c, arm in enumerate(arms):
                ax = axes[r][c]
                sub = traj[(traj.arm == arm) & (traj.k == k)]
                base = (sub[sub.iv == "base"].set_index("tokens").overt_pref
                        if delta else None)
                for iv in ivs:
                    t = sub[sub.iv == iv].sort_values("tokens")
                    if not len(t) or (delta and iv == "base"):
                        continue
                    y = t.overt_pref.values
                    if delta:
                        if base is None or not len(base):
                            continue
                        aligned = base.reindex(t.tokens).values
                        y = y - aligned
                    ax.plot(t.tokens, y, lw=1.4, color=iv_colors[iv],
                            label=IV_LABEL.get(iv, iv))
                ax.set_xscale("log")
                ax.axhline(0.0 if delta else 0.5, color="#bbb", lw=0.6)
                if r == 0:
                    ax.set_title(arm, fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"k={k}\n" +
                                  ("Δ vs base" if delta else "overt pref."),
                                  fontsize=9)
                if r == len(ks) - 1:
                    ax.set_xlabel("tokens seen (log)")
        handles, labels = [], []
        for a_row in axes:
            for a in a_row:
                h, l = a.get_legend_handles_labels()
                for hi, li in zip(h, l):
                    if li not in labels:
                        handles.append(hi); labels.append(li)
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.008))
        fig.suptitle("Interventions compared within decile × arm"
                     + (" (difference from baseline)" if delta else ""),
                     y=1.003)
        fig.tight_layout()
        fig.savefig(args.out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {args.out}/setA_trajectories.png, "
          "setB_decile_interventions.png, setB_delta_vs_base.png")


if __name__ == "__main__":
    main()
