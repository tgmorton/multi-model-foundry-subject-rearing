#!/usr/bin/env python3
"""Arm vs random at a matched token horizon, so runs still training count.

End states (wave2_endstate.py) need finished runs. This takes, per run, the
mean overt-subject preference over the last --last checkpoints at or below
--horizon tokens (recorded tokens_seen, via _tokens.TokenAxis), keeping only
runs that have reached the horizon. With a merged cache that adds --partial
snapshots, every cell that has trained far enough contributes.

Reads the local pairs cache only (no S3 sync): point it at a merged cache with
SUBJECT_DROP_PAIRS_CACHE / SUBJECT_DROP_CKPT_CACHE.

  python analysis/eval_v2/wave2_matched_horizon.py --arch gpt2_large --horizon 2.5e9 --out DIR
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
except ImportError:  # ... or imported as analysis.eval_v2.wave2_matched_horizon
    from analysis.eval_v2._tokens import (DEGENERATE_COLS, PAIRS_CACHE, TokenAxis,
                                          drop_degenerate, is_arch, run_id_from_name)
    from analysis.eval_v2.wave2_endstate import ARM_STYLE, IV_LABEL, SIENNA

CELL = re.compile(r"pdrop2_(robbianti|robbi|rand|all100)(\d*)_(\w+)-h(\d)-")
IVS = ["base", "impcase", "lemverb", "enrichvm", "rmexpl"]
IV_LABEL = {**IV_LABEL, "rmexpl": "remove expletives"}


def load(arch: str, hp: int, horizon: float, last: int) -> tuple[pd.DataFrame, int]:
    axis = TokenAxis(arch)
    rows, short = [], 0
    for p in sorted(PAIRS_CACHE.glob("*.parquet")):
        m = CELL.search(p.name)
        if not (m and is_arch(p.name, arch)) or int(m.group(4)) != hp:
            continue
        arm, k, iv, _ = m.groups()
        d = drop_degenerate(pd.read_parquet(
            p, columns=["checkpoint_step", "prefers_overt_meanlp", *DEGENERATE_COLS]))
        c = d[d.checkpoint_step > 0].groupby("checkpoint_step").prefers_overt_meanlp.mean()
        run = run_id_from_name(p.name)
        tok = pd.Series(axis.tokens(pd.Series([run] * len(c)), pd.Series(c.index)), index=c.index)
        if tok.max() < horizon:
            short += 1
            continue
        c = c[tok.values <= horizon].sort_index()
        rows.append({"arm": arm, "k": 100 if arm == "all100" else int(k), "iv": iv,
                     "pref": c.iloc[-last:].mean(), "run": run})
    return pd.DataFrame(rows), short


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--hp", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=2.5e9, help="tokens seen")
    ap.add_argument("--last", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df, short = load(a.arch, a.hp, a.horizon, a.last)
    if df.empty:
        raise SystemExit(f"no {a.arch} runs reach {a.horizon:.3g} tokens")
    tag = f"{a.arch}_h{a.hp}_{a.horizon / 1e9:.2g}B"
    df.to_csv(a.out / f"matched_horizon_{tag}.csv", index=False)
    print(f"{a.arch}: {len(df)} runs at <= {a.horizon:.3g} tokens ({short} not there yet)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ivs = [i for i in IVS if i in set(df.iv)]
    fig, axes = plt.subplots(1, len(ivs), figsize=(3.6 * len(ivs), 4.1), sharey=True)
    for ax, iv in zip(np.atleast_1d(axes), ivs):
        sub = df[df.iv == iv]
        for arm in ("rand", "robbi", "robbianti"):
            c, ls, mk, lab = ARM_STYLE[arm]
            t = sub[(sub.arm == arm) & (sub.k < 100)].groupby("k").pref.mean().sort_index()
            if len(t):
                ax.plot(t.index, t.values, linestyle=ls, marker=mk, color=c, ms=5, lw=1.8, label=lab)
        anc = sub[sub.arm == "all100"]
        if len(anc):
            ax.plot([100], [anc.pref.mean()], "*", color=SIENNA, ms=14, label="all pronouns removed")
        ax.axhline(0.5, color="#ccc", lw=.7)
        ax.set_title(IV_LABEL.get(iv, iv), fontsize=10)
        ax.set_xlabel("% of pronouns removed (decile)")
        ax.set_xticks([10, 30, 50, 70, 90, 100])
    np.atleast_1d(axes)[0].set_ylabel(f"overt-subject preference at {a.horizon / 1e9:.2g}B tokens")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Preference at a matched {a.horizon / 1e9:.2g}B tokens ({a.arch}, h{a.hp}, "
                 f"mean of last {a.last} checkpoints at or below the horizon)", y=1.02)
    fig.tight_layout()
    fig.savefig(a.out / f"matched_horizon_{tag}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}/matched_horizon_{tag}.png")

    for arm in ("robbi", "robbianti"):
        print(f"== {arm} minus random at {a.horizon / 1e9:.2g}B tokens ==")
        for iv in ivs:
            sub = df[df.iv == iv]
            j = (sub[sub.arm == arm].groupby("k").pref.mean()
                 - sub[sub.arm == "rand"].groupby("k").pref.mean()).dropna()
            j = j[j.index < 100]
            if len(j):
                print(f"  {IV_LABEL.get(iv, iv):22s} mean Δ {j.mean():+.3f} | n_k={len(j)}")


if __name__ == "__main__":
    main()
