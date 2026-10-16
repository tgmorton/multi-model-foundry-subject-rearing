#!/usr/bin/env python3
"""End state from the latest checkpoints each run has, finished or not.

wave2_endstate.py needs finished runs. Here every run contributes the mean of
its last --last checkpoints so far; runs still training are drawn hollow and
their training progress is reported, since late-training drift (robbianti
keeps falling) makes a partway value a slight overestimate of its end state.

Reads the local pairs cache only: point it at a merged cache that adds
--partial snapshots with SUBJECT_DROP_PAIRS_CACHE / SUBJECT_DROP_CKPT_CACHE.

  python analysis/eval_v2/wave2_endstate_latest.py --arch gpt2_large --out DIR
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:  # run as a script from analysis/eval_v2/ ...
    from _tokens import (DEGENERATE_COLS, PAIRS_CACHE, TokenAxis, drop_degenerate,
                         is_arch, run_id_from_name)
    from wave2_endstate import ARM_STYLE, IV_LABEL, SIENNA
except ImportError:  # ... or imported as analysis.eval_v2.wave2_endstate_latest
    from analysis.eval_v2._tokens import (DEGENERATE_COLS, PAIRS_CACHE, TokenAxis,
                                          drop_degenerate, is_arch, run_id_from_name)
    from analysis.eval_v2.wave2_endstate import ARM_STYLE, IV_LABEL, SIENNA

CELL = re.compile(r"pdrop2_(robbianti|robbi|rand|all100)(\d*)_(\w+)-h(\d)-")
IVS = ["base", "impcase", "lemverb", "enrichvm", "rmexpl"]
IV_LABEL = {**IV_LABEL, "rmexpl": "remove expletives"}
TRAIN_STEPS = 7500  # gpt2_large wave2 nominal optimizer steps (~7470-7530 by corpus)


def load(arch: str, hp: int, last: int) -> pd.DataFrame:
    axis = TokenAxis(arch)
    rows = []
    for p in sorted(PAIRS_CACHE.glob("*.parquet")):
        m = CELL.search(p.name)
        if not (m and is_arch(p.name, arch)) or int(m.group(4)) != hp:
            continue
        arm, k, iv, _ = m.groups()
        d = drop_degenerate(pd.read_parquet(
            p, columns=["checkpoint_step", "prefers_overt_meanlp", *DEGENERATE_COLS]))
        c = d[d.checkpoint_step > 0].groupby("checkpoint_step").prefers_overt_meanlp.mean().sort_index()
        if len(c) < 20:
            continue
        run = run_id_from_name(p.name)
        tok = axis.tokens(pd.Series([run] * len(c)), pd.Series(c.index))
        # finished = a canonical (full) result exists; in a merged cache a run
        # still training links to its --partial snapshot instead. Token totals
        # can't decide it: intervention corpora differ in size.
        partial = "partial" in p.resolve().parent.name
        rows.append({"arm": arm, "k": 100 if arm == "all100" else int(k), "iv": iv,
                     "pref": c.iloc[-last:].mean(), "tokens": float(np.max(tok)), "run": run,
                     "finished": not partial,
                     "progress": 1.0 if not partial else min(1.0, c.index.max() / TRAIN_STEPS)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--hp", type=int, default=0)
    ap.add_argument("--last", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = load(a.arch, a.hp, a.last)
    tag = f"{a.arch}_h{a.hp}"
    df.to_csv(a.out / f"endstate_latest_{tag}.csv", index=False)
    part = df[~df.finished]
    print(f"{a.arch}: {len(df)} runs ({df.finished.sum()} finished, {len(part)} still training: "
          f"progress median {part.progress.median():.0%}, min {part.progress.min():.0%})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ivs = [i for i in IVS if i in set(df.iv)]
    fig, axes = plt.subplots(1, len(ivs), figsize=(3.6 * len(ivs), 4.1), sharey=True)
    for ax, iv in zip(np.atleast_1d(axes), ivs):
        sub = df[df.iv == iv]
        for arm in ("rand", "robbi", "robbianti"):
            c, ls, mk, lab = ARM_STYLE[arm]
            s = sub[(sub.arm == arm) & (sub.k < 100)].sort_values("k")
            if not len(s):
                continue
            ax.plot(s.k, s.pref, linestyle=ls, color=c, lw=1.6, label=lab)
            done = s[s.finished]
            ax.plot(done.k, done.pref, linestyle="none", marker=mk, color=c, ms=5)
            ax.plot(s[~s.finished].k, s[~s.finished].pref, linestyle="none", marker=mk,
                    ms=5, markerfacecolor="white", markeredgecolor=c)
        anc = sub[sub.arm == "all100"]
        if len(anc):
            ax.plot([100], [anc.pref.mean()], "*", color=SIENNA, ms=14, label="all pronouns removed")
        ax.axhline(0.5, color="#ccc", lw=.7)
        ax.set_title(IV_LABEL.get(iv, iv), fontsize=10)
        ax.set_xlabel("% of pronouns removed (decile)")
        ax.set_xticks([10, 30, 50, 70, 90, 100])
    np.atleast_1d(axes)[0].set_ylabel("overt-subject preference (latest checkpoints)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], linestyle="none", marker="o", ms=5, markerfacecolor="white",
                        markeredgecolor="#555"))
    l.append("hollow = still training (latest checkpoints so far)")
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(f"End state from the latest {a.last} checkpoints each run has ({a.arch}, h{a.hp}; "
                 f"{df.finished.sum()} finished, {len(part)} still training)", y=1.02)
    fig.tight_layout()
    fig.savefig(a.out / f"endstate_latest_{tag}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}/endstate_latest_{tag}.png")

    for arm in ("robbi", "robbianti"):
        print(f"== {arm} minus random (latest checkpoints) ==")
        for iv in ivs:
            sub = df[df.iv == iv]
            j = (sub[sub.arm == arm].groupby("k").pref.mean()
                 - sub[sub.arm == "rand"].groupby("k").pref.mean()).dropna()
            j = j[j.index < 100]
            if len(j):
                print(f"  {IV_LABEL.get(iv, iv):22s} mean Δ {j.mean():+.3f} | n_k={len(j)}")


if __name__ == "__main__":
    main()
