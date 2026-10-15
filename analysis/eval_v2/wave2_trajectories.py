"""Wave2 (w2v5) overt-preference trajectory figures — Thomas 2026-09-18.

Set A: per (arm x intervention) panel, overt-preference (mean
prefers_overt_meanlp) vs tokens seen (log-x), one line per decile k.
Set B: per (decile k x arm) panel, interventions compared on the same
axes (plus a delta-vs-base variant), log-x tokens.

Input: null_subj_v2 pairs parquets for pdrop2_* runs — pulled from S3
(default) or a local directory (--pairs-dir). Only runs whose cell_id starts
with "<arch>-" are used (--arch, default gpt2_small). Tokens per
checkpoint_step by architecture: gpt2_small 256,000 (verified against
training logs), gpt2_large 512,000 (the pairs parquets carry no tokens_seen
column; the sibling checkpoints/ parquets do and confirm 512,000).

Output: analysis/eval_v2/figures/wave2_v5/{setA_trajectories,
setB_decile_interventions,setB_delta_vs_base}.png + tidy CSV. For any arch
other than gpt2_small the filenames get a "_<arch>" tag, so the gpt2_small
figures are never overwritten.
"""

from __future__ import annotations

import argparse
import glob
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

# tokens per checkpoint_step by architecture (keep in sync with
# wave2_category_trajectories.py)
TOKENS_PER_STEP = {"gpt2_small": 256_000, "gpt2_large": 512_000}
DEFAULT_ARCH = "gpt2_small"
ARMS = ["rand", "gpt2m", "bert", "comp", "bertanti", "robbi", "robbianti"]
IVS = ["base", "rmexpl", "impcase", "lemverb", "enrichvm"]
IV_LABEL = {"base": "baseline", "rmexpl": "remove expletives",
            "impcase": "impoverish case", "lemverb": "lemmatize verbs",
            "enrichvm": "enrich verbal morph"}
CELL_RE = re.compile(r"pdrop2_(gpt2m|bertanti|bert|comp|robbianti|robbi|rand|all100)(\d*)_(\w+)")


def is_arch(name: str, arch: str) -> bool:
    """True if an S3 key / local filename belongs to `arch`. cell_ids look like
    '<arch>-<lang>-<condition>-h<hp>-s<seed>'; the trailing '-' keeps
    'gpt2_small' from matching a longer arch name."""
    base = name.rsplit("/", 1)[-1]
    if base.startswith("cell_id="):
        base = base[len("cell_id="):]
    return base.startswith(f"{arch}-")


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
                            "prefers_overt_meanlp"]))
    else:
        import boto3
        s = boto3.Session(profile_name=args.profile).client(
            "s3", endpoint_url=args.endpoint)
        pag = s.get_paginator("list_objects_v2")
        for page in pag.paginate(Bucket=args.bucket, Prefix=args.s3_prefix):
            for o in page.get("Contents", []):
                if "pdrop2" not in o["Key"] or not o["Key"].endswith(".parquet"):
                    continue
                if not is_arch(o["Key"], args.arch):
                    continue
                body = s.get_object(Bucket=args.bucket, Key=o["Key"])["Body"].read()
                frames.append(pd.read_parquet(
                    io.BytesIO(body),
                    columns=["cell_id", "checkpoint_step",
                             "prefers_overt_meanlp"]))
    if not frames:
        raise SystemExit(f"no pdrop2 pairs found for arch {args.arch!r}")
    df = pd.concat(frames, ignore_index=True)
    # authoritative arch filter on the cell_id column (also covers --pairs-dir,
    # whose filenames need not follow the S3 naming)
    df = df[df.cell_id.map(lambda c: is_arch(c, args.arch))].copy()
    if df.empty:
        raise SystemExit(f"no pdrop2 rows with cell_id starting {args.arch + '-'!r}")
    meta = df.cell_id.map(parse_cell)
    df["arm"] = meta.map(lambda t: t and t[0])
    df["k"] = meta.map(lambda t: t and t[1])
    df["iv"] = meta.map(lambda t: t and t[2])
    df = df[df.arm.notna()]
    df["tokens"] = df.checkpoint_step * TOKENS_PER_STEP[args.arch]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=None,
                    help="local pairs dir (else pull from S3)")
    ap.add_argument("--bucket", default="thomas-subject-drop-artifacts")
    ap.add_argument("--s3-prefix", default="eval_results/null_subj_v2_condition_matched_v1/pairs/")
    ap.add_argument("--profile", default="nrp")
    ap.add_argument("--endpoint", default="https://s3-west.nrp-nautilus.io")
    ap.add_argument("--arch", default=DEFAULT_ARCH, choices=sorted(TOKENS_PER_STEP),
                    help="architecture to plot (cell_id prefix)")
    ap.add_argument("--out", type=Path,
                    default=Path("analysis/eval_v2/figures/wave2_v5"))
    ap.add_argument("--arms", default=None,
                    help="comma list of arms to plot (default: all present); "
                         "adds a filename suffix")
    args = ap.parse_args()
    archtag = "" if args.arch == DEFAULT_ARCH else f"_{args.arch}"
    tag = ("_" + args.arms.replace(",", "-") if args.arms else "") + archtag
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_pairs(args)
    # preference rate per (cell, checkpoint), averaged over items (and
    # over replicates when >1 run per cell). Init checkpoint (step<=0,
    # "checkpoint -1") is excluded: trajectories BEGIN at the first
    # training step, at their absolute value there (2026-09-18).
    traj = (df[df.tokens > 0]
            .groupby(["arm", "iv", "k", "tokens"], as_index=False)
            .prefers_overt_meanlp.mean()
            .rename(columns={"prefers_overt_meanlp": "overt_pref"}))
    traj.to_csv(args.out / f"wave2_v5_trajectories{archtag}.csv", index=False)
    print(f"{args.arch}: "
          f"{traj.cell_count if hasattr(traj,'cell_count') else len(traj):,} "
          f"trajectory points; arms {sorted(traj.arm.unique())}; "
          f"ivs {sorted(traj.iv.unique())}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")

    arms = [a for a in ARMS if a in set(traj.arm)]
    if args.arms:
        arms = [a for a in args.arms.split(",") if a in arms]
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
    from matplotlib.lines import Line2D
    all_ks = sorted(k for k in traj.k.unique() if k < 100)
    handles = [Line2D([0], [0], color=cmap(k / 100), lw=2) for k in all_ks]
    labels = [f"k={k}" for k in all_ks]
    if (traj.arm == "all").any():
        handles.append(Line2D([0], [0], color="#A33B2E", lw=2, ls="--"))
        labels.append("all removed")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(f"Overt-subject preference trajectories ({args.arch}) — "
                 "arm × intervention, lines = removal decile", y=1.005)
    fig.tight_layout()
    fig.savefig(args.out / f"setA_trajectories{tag}.png", dpi=150,
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
        fig.suptitle(f"Interventions compared within decile × arm ({args.arch})"
                     + (" (difference from baseline)" if delta else ""),
                     y=1.003)
        fig.tight_layout()
        fig.savefig(args.out / f"{name}{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {args.out}/setA_trajectories{tag}.png, "
          f"setB_decile_interventions{tag}.png, setB_delta_vs_base{tag}.png")


if __name__ == "__main__":
    main()
