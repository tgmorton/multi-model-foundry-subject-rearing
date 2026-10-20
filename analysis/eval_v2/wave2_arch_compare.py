"""gpt2_small vs gpt2_large overt-preference trajectories on one true-token axis.

Set-A-style panels (one per selection arm; lines = removal decile k, viridis;
y = mean prefers_overt_meanlp over the overall items, i.e. without _tokens.OVERALL_EXCLUDE), baseline intervention only, with
the two architectures overlaid: gpt2_small solid, gpt2_large dashed. Both use
HP rank h0 only (gpt2_small has h0-h3; gpt2_large only h0), so the x axis is the
same quantity for both: tokens seen, taken per checkpoint from the checkpoints/
parquets via analysis/eval_v2/_tokens.py (gpt2_small h0 = 128K tokens/step,
gpt2_large 512K; both end near 3.86B tokens).

The "all removed" reference cell is left out (gpt2_large has no all100
baseline yet, and dashed already means gpt2_large here). Deciles a given arch
has not finished simply have no line for that arch.

Pairs parquets come from the shared pairs cache (~/.cache/subject-drop/
pairs_cache, synced from S3 in parallel with retries unless --no-sync).

Usage: python analysis/eval_v2/wave2_arch_compare.py \
           [--arms rand,robbi,robbianti] [--iv base] [--hp 0] [--no-sync] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:  # run as a script (analysis/eval_v2 on sys.path) ...
    from _tokens import (DEGENERATE_COLS, TokenAxis, drop_degenerate, overall_items, is_arch,
                         run_id_from_name, sync_checkpoints)
    from wave2_category_trajectories import ARM_LABEL, CACHE, CELL, IV_LABEL
    from wave2_category_trajectories import sync as sync_pairs
except ImportError:  # ... or imported as analysis.eval_v2.wave2_arch_compare
    from analysis.eval_v2._tokens import (DEGENERATE_COLS, TokenAxis,
                                          drop_degenerate, overall_items, is_arch,
                                          run_id_from_name, sync_checkpoints)
    from analysis.eval_v2.wave2_category_trajectories import (ARM_LABEL, CACHE,
                                                              CELL, IV_LABEL)
    from analysis.eval_v2.wave2_category_trajectories import sync as sync_pairs

ARCHS = ("gpt2_small", "gpt2_large")   # drawn solid, dashed
LINESTYLE = {"gpt2_small": "-", "gpt2_large": "--"}
LINEWIDTH = {"gpt2_small": 1.3, "gpt2_large": 1.7}


def load_arch(arch, arms, iv, hp):
    """Mean overt preference per (arm, k, tokens) for one architecture, HP rank
    `hp`, intervention `iv`; replicates of a cell are averaged."""
    rows = []
    for p in sorted(CACHE.glob("*.parquet")):
        if not (is_arch(p.name, arch) and CELL.search(p.name)):
            continue
        arm, k, iv_, h = CELL.search(p.name).groups()
        if arm not in arms or iv_ != iv or int(h) != hp:
            continue
        d = overall_items(drop_degenerate(pd.read_parquet(
            p, columns=["checkpoint_step", "category", "prefers_overt_meanlp", *DEGENERATE_COLS])))
        d = d[d.checkpoint_step > 0]       # trajectories begin at the first training step
        g = d.groupby("checkpoint_step", as_index=False).prefers_overt_meanlp.mean()
        g["arm"], g["k"] = arm, int(k)
        g["run"] = run_id_from_name(p.name)
        rows.append(g)
    if not rows:
        raise SystemExit(f"no {arch} runs for arms {sorted(arms)}, iv {iv!r}, hp {hp}")
    df = pd.concat(rows, ignore_index=True)
    axis = TokenAxis(arch)
    df["tokens"] = axis.tokens(df.run, df.checkpoint_step)
    axis.report()
    out = (df.groupby(["arm", "k", "tokens"], as_index=False)
           .prefers_overt_meanlp.mean()
           .rename(columns={"prefers_overt_meanlp": "overt_pref"}))
    out.insert(0, "arch", arch)
    # final-checkpoint token count per run, for the cross-arch sanity print
    last = df.groupby("run").tokens.max()
    print(f"[compare] {arch}: {df.run.nunique()} runs; final tokens/run "
          f"min {last.min():,} median {int(last.median()):,} max {last.max():,}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="thomas-subject-drop-artifacts")
    ap.add_argument("--prefix", default="eval_results/null_subj_v2_condition_matched_v1/pairs/")
    ap.add_argument("--profile", default="nrp")
    ap.add_argument("--endpoint", default="https://s3-west.nrp-nautilus.io")
    ap.add_argument("--arms", default="rand,robbi,robbianti")
    ap.add_argument("--iv", default="base")
    ap.add_argument("--hp", type=int, default=0)
    ap.add_argument("--no-sync", action="store_true",
                    help="use the local pairs and checkpoint caches as-is")
    ap.add_argument("--out", type=Path, default=Path("analysis/eval_v2/figures/wave2_v5"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arms = a.arms.split(",")

    if not a.no_sync:
        for arch in ARCHS:
            sync_pairs(a.bucket, a.prefix, a.profile, a.endpoint, arch)
            sync_checkpoints(a.bucket, a.profile, a.endpoint, arch)
    df = pd.concat([load_arch(arch, arms, a.iv, a.hp) for arch in ARCHS],
                   ignore_index=True)
    tag = "-".join(arms) + f"_{a.iv}_h{a.hp}"
    df.to_csv(a.out / f"arch_compare_{tag}.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cmap = plt.get_cmap("viridis")
    arms = [r for r in arms if r in set(df.arm)]
    ks = sorted(df.k.unique())

    fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 3.6),
                             sharex=True, sharey=True, squeeze=False)
    for c, arm in enumerate(arms):
        ax = axes[0][c]
        for arch in ARCHS:
            for k in ks:
                t = df[(df.arch == arch) & (df.arm == arm) & (df.k == k)].sort_values("tokens")
                if len(t):
                    ax.plot(t.tokens, t.overt_pref, ls=LINESTYLE[arch], lw=LINEWIDTH[arch],
                            color=cmap(k / 100), alpha=0.95)
        ax.set_xscale("log")
        ax.axhline(0.5, color="#bbb", lw=0.6)
        ax.set_title(ARM_LABEL.get(arm, arm), fontsize=9)
        ax.set_xlabel("tokens seen (log)")
        if c == 0:
            ax.set_ylabel("overt pref.")
            ax.set_ylim(0, 1)
    handles = [Line2D([0], [0], color=cmap(k / 100), lw=2) for k in ks]
    labels = [f"k={k}" for k in ks]
    handles += [Line2D([0], [0], color="#444", lw=LINEWIDTH[x], ls=LINESTYLE[x]) for x in ARCHS]
    labels += list(ARCHS)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"gpt2_small vs gpt2_large overt-subject preference — "
                 f"{IV_LABEL.get(a.iv, a.iv)}, h{a.hp}, true-token axis "
                 "(lines = removal decile)", y=1.02)
    fig.tight_layout()
    out = a.out / f"arch_compare_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
