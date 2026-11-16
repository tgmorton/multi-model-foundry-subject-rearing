"""By-decile trajectories on the natural-position decile test set, for the talk deck.

Reads the trajectory benchmark (one cell per run, or several checkpoint shards per run that are
concatenated here) and writes, in the layout of arg_decile_by_order.csv plus tokens_seen and
checkpoint_step:

  arg_decile_trajectories.csv                 preference for the overt form per decile at each checkpoint
  arg_decile_trajectories_minus_random.csv    ordered removal minus random removal at the same percent,
                                              paired by item, at each checkpoint

Runs differ slightly in steps per epoch, so runs are aligned by checkpoint position in the scored
schedule; each row carries its own tokens_seen (the mean of the two runs for a difference).
Unmodified seeds are averaged per checkpoint before the item bootstrap.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_exports_argument import ARCH, ARM, W2, boot, removed  # noqa: E402
import deck_exports as X  # noqa: E402

SHARD = re.compile(r"__ckpt_shard\d+of\d+$")
SET = "natural position"


def read(data: Path, bench: str):
    """{(arch, arm, k, run): DataFrame[item_id, checkpoint_step, tokens_seen, pref]}, plus a note list."""
    runs, notes = {}, []
    for f in sorted((data / bench / "pairs").glob("cell_id=*.parquet")):
        cell = f.stem.split("=", 1)[1]
        run = SHARD.sub("", cell)
        m = W2.match(run)
        if m and m["iv"] == "base":
            key = (m["arch"], m["arm"], int(m["k"]), run)
        else:
            m = X.PROD.match(run)
            if not (m and m["cond"] == "baseline"):
                continue
            key = (m["arch"], "none", 0, run)
        p = pd.read_parquet(f, columns=["item_id", "checkpoint_step", "prefers_overt_meanlp"])
        c = pd.read_parquet(data / bench / "checkpoints" / f.name, columns=["checkpoint_step", "tokens_seen"]).drop_duplicates()
        p = p.merge(c, on="checkpoint_step", how="left").rename(columns={"prefers_overt_meanlp": "pref"})
        p["pref"] = p.pref.astype(float)
        runs.setdefault(key, []).append(p)
    out = {}
    for key, parts in runs.items():
        d = pd.concat(parts, ignore_index=True).drop_duplicates(["item_id", "checkpoint_step"])
        steps = np.sort(d.checkpoint_step.unique())
        d["pos"] = np.searchsorted(steps, d.checkpoint_step)
        out[key] = d
        notes.append(f"{key[3]}: {len(steps)} checkpoints, {len(parts)} cell file(s), steps {steps[0]}..{steps[-1]}")
    return out, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--bench", default="null_subj_test10m_robbi_nat_traj_v1")
    ap.add_argument("--key", type=Path, required=True, help="item_key.csv of the natural-position stimulus root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    key = pd.read_csv(a.key, usecols=["item_id", "decile"]).set_index("item_id").decile
    runs, notes = read(a.data, a.bench)
    assert runs, "no trajectory cells found"
    # one series per (arch, arm, k): item x checkpoint-position matrix of preference, tokens per position
    series, n_runs = {}, {}
    cells: dict = {}
    for kk in runs:
        cells.setdefault(kk[:3], []).append(kk)
    for (arch, arm, k), keys in cells.items():
        mats = [runs[kk].pivot(index="item_id", columns="pos", values="pref") for kk in keys]
        toks = [runs[kk].groupby("pos")[["tokens_seen", "checkpoint_step"]].first() for kk in keys]
        npos = min(m.shape[1] for m in mats)
        if len({m.shape[1] for m in mats}) > 1:
            notes.append(f"{arch} {arm}{k}: seeds have different checkpoint counts {[m.shape[1] for m in mats]}; using the first {npos}")
        series[(arch, arm, k)] = (sum(m.iloc[:, :npos] for m in mats) / len(mats),
                                  sum(t.iloc[:npos] for t in toks) / len(toks))
        n_runs[(arch, arm, k)] = len(keys)
    rows, diffs = [], []
    for (arch, arm, k), (mat, tok) in sorted(series.items()):
        nr = n_runs[(arch, arm, k)]
        kind = f"mean of {nr} seeds; bootstrap over items (95%)" if nr > 1 else "one run; bootstrap over items (95%)"
        dec = key.reindex(mat.index)
        for pos in mat.columns:
            for dc in range(10):
                x = mat.loc[dec == dc, pos].to_numpy()
                m_, lo, hi = boot(x, n_boot=a.boot, seed=dc)
                rows.append(dict(figure="arg_decile_trajectories", architecture=ARCH[arch], condition="baseline", arm=ARM[arm],
                                 percent=k, test_set=SET, decile=dc, decile_removed=dc in removed(arm, k),
                                 tokens_seen=int(tok.tokens_seen[pos]), checkpoint_step=int(tok.checkpoint_step[pos]),
                                 checkpoint_position=int(pos), mean=m_, lower=lo, upper=hi, n_runs=nr, n_items=int(len(x)),
                                 interval_kind=kind))
        if arm in ("robbi", "robbianti") and (arch, "rand", k) in series:
            rmat, rtok = series[(arch, "rand", k)]
            for pos in [p for p in mat.columns if p in rmat.columns]:
                d = (mat[pos] - rmat[pos].reindex(mat.index))
                for dc in range(10):
                    x = d[dec == dc].dropna().to_numpy()
                    m_, lo, hi = boot(x, n_boot=a.boot, seed=dc)
                    diffs.append(dict(figure="arg_decile_trajectories_minus_random", architecture=ARCH[arch], condition="baseline",
                                      arm=f"{ARM[arm]} minus random removal", percent=k, test_set=SET, decile=dc,
                                      decile_removed=dc in removed(arm, k),
                                      tokens_seen=int((tok.tokens_seen[pos] + rtok.tokens_seen[pos]) / 2),
                                      checkpoint_step=int(tok.checkpoint_step[pos]), checkpoint_position=int(pos),
                                      mean=m_, lower=lo, upper=hi, n_runs=nr, n_items=int(len(x)),
                                      interval_kind="difference paired by item; bootstrap over items (95%)"))
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out / "arg_decile_trajectories.csv", index=False)
    pd.DataFrame(diffs).to_csv(a.out / "arg_decile_trajectories_minus_random.csv", index=False)
    print("arg_decile_trajectories.csv", len(rows), "| arg_decile_trajectories_minus_random.csv", len(diffs))
    print("\n".join(notes))


if __name__ == "__main__":
    main()
