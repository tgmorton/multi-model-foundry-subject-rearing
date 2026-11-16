"""Rater-free check on the decile result: grade test items by the learner itself.

The decile test sets grade each pronoun by the selection rater (robbi). This script re-grades
the same items by the unmodified learner of the same architecture: the pronoun's surprisal at
its own slot, averaged over the unmodified seeds, cut into ten equal-count bins per test set
(own decile 0 = pronoun most predictable to the learner). It then asks whether the pattern the
removal models show against rater deciles also shows against the learner's own grading.

Per architecture, test set, removal order and percent the unit is the test item:
    d(item) = preference of the ordered-removal model - preference of the random-removal model
A removal order that tilts dropping toward the pronouns it removed gives d a slope across
grades: positive for most-recoverable-first, negative for least-recoverable-first. The slope is
reported per grade step (0..9) for the rater's grading, the learner's own grading, and each
with the other held fixed (two-predictor least squares). Intervals: bootstrap over items.

Only left-to-right models (the slot score needs a prefix probability).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from eval_recoverability_testset_results import slot_scores

WAVE = re.compile(r"^(?P<arch>[a-z0-9_]+)-en-pdrop2_(?P<arm>robbianti|robbi|rand|all)(?P<level>\d+)_base-h0-s\d+$")
UNMOD = re.compile(r"^(?P<arch>[a-z0-9_]+)-en-baseline-h0-s\d+$")
ARCH = {"gpt2_large": "GPT-2 large", "lstm": "LSTM"}
ARM = {"robbi": "most recoverable first", "robbianti": "least recoverable first"}
SETS = {"natural": ("in_natural", "natural position"), "matched": ("in_matched", "position matched across deciles"),
        "nopron": ("in_nopron", "pronoun-free contexts")}
LEVELS = list(range(10, 100, 10))


def read(results: Path):
    wave, own = [], []
    for f in sorted((results / "pairs").glob("cell_id=*.parquet")):
        run = f.stem.split("=", 1)[1]
        tok = results / "per_token" / f.name
        mu, mw = UNMOD.match(run), WAVE.match(run)
        if not (mu or mw) or not tok.exists():
            continue
        arch = (mu or mw)["arch"]
        if arch not in ARCH:
            continue
        s = slot_scores(pd.read_parquet(tok))
        p = pd.read_parquet(f, columns=["item_id", "prefers_overt_meanlp"]).rename(columns={"prefers_overt_meanlp": "pref"})
        p["pref"] = p.pref.astype(float)
        p = p.merge(s, on="item_id", how="left")
        p["arch"], p["run"] = arch, run
        if mu:
            own.append(p)
        else:
            p["arm"], p["level"] = mw["arm"], int(mw["level"])
            wave.append(p)
    return pd.concat(wave, ignore_index=True), (pd.concat(own, ignore_index=True) if own else pd.DataFrame())


def slopes(d: np.ndarray, r: np.ndarray, o: np.ndarray, n: int, rng) -> dict:
    """Simple and partial slopes of d on rater grade r and own grade o, with item bootstrap."""
    def est(i):
        dd, rr, oo = d[i], r[i], o[i]
        X = np.column_stack([np.ones(len(dd)), rr, oo])
        b = np.linalg.lstsq(X, dd, rcond=None)[0]
        return np.array([np.polyfit(rr, dd, 1)[0], np.polyfit(oo, dd, 1)[0], b[1], b[2]])
    full = est(np.arange(len(d)))
    bs = np.array([est(rng.integers(0, len(d), len(d))) for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    names = ["rater grade alone", "learner's own grade alone", "rater grade, own grade held fixed",
             "learner's own grade, rater grade held fixed"]
    return {nm: (full[j], lo[j], hi[j]) for j, nm in enumerate(names)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    key = pd.read_csv(a.key)
    wave, own = read(a.results)
    assert len(own), "no unmodified runs with per-token scores"
    rng = np.random.default_rng(0)
    rows_s, rows_g, rows_x, rows_c = [], [], [], []
    for arch, og in own.groupby("arch"):
        n_seeds = og.run.nunique()
        surp = (-og.groupby("item_id").slot_overt.mean()).rename("own_surprisal")
        unmod_pref = og.groupby("item_id").pref.mean().rename("unmod_pref")
        w = wave[wave.arch == arch].groupby(["arm", "level", "item_id"]).pref.mean().unstack(["arm", "level"])
        for sname, (flag, slabel) in SETS.items():
            k = key[key[flag]].set_index("item_id").join(surp).join(unmod_pref).dropna(subset=["own_surprisal"])
            k["own_decile"] = pd.qcut(k.own_surprisal.rank(method="first"), 10, labels=False)
            rho = k[["own_surprisal", "surp_item"]].corr(method="spearman").iloc[0, 1]
            agree = pd.crosstab(k.decile, k.own_decile)
            rows_x.append({"architecture": ARCH[arch], "test_set": slabel, "n_items": len(k), "n_unmodified_seeds": n_seeds,
                           "spearman_own_vs_rater_surprisal": rho,
                           "spearman_own_decile_vs_rater_decile": k[["own_decile", "decile"]].corr(method="spearman").iloc[0, 1],
                           "share_within_one_decile": float(sum(agree.values[i, j] for i in range(10) for j in range(10)
                                                                if abs(i - j) <= 1) / len(k)),
                           "unmodified_preference_mean": k.unmod_pref.mean()})
            for (rd, od), n_ in agree.stack().items():
                rows_c.append({"architecture": ARCH[arch], "test_set": slabel, "rater_decile": int(rd), "own_decile": int(od),
                               "n_items": int(n_), "share_of_rater_decile": float(n_ / agree.loc[rd].sum())})
            ww = w.reindex(k.index)
            for arm, alabel in ARM.items():
                D = pd.concat({lv: ww[(arm, lv)] - ww[("rand", lv)] for lv in LEVELS if (arm, lv) in ww and ("rand", lv) in ww}, axis=1)
                for scope, cols in [("mean of 10-90%", list(D.columns))] + [(f"{lv}%", [lv]) for lv in D.columns]:
                    d = D[cols].mean(axis=1)
                    ok = d.notna()
                    for nm, (m, lo, hi) in slopes(d[ok].to_numpy(), k.decile[ok].to_numpy(float), k.own_decile[ok].to_numpy(float),
                                                  a.boot if scope.startswith("mean") else 300, rng).items():
                        rows_s.append({"architecture": ARCH[arch], "test_set": slabel, "arm": alabel, "percent": scope,
                                       "grading": nm, "slope_per_decile": m, "lower": lo, "upper": hi, "n_items": int(ok.sum()),
                                       "interval_kind": "bootstrap over items (95%)"})
                d = D.mean(axis=1)
                for gname, g in [("rater decile", k.decile), ("learner's own decile", k.own_decile)]:
                    for dec, x in d.groupby(g):
                        x = x.dropna().to_numpy()
                        b = x[rng.integers(0, len(x), (a.boot, len(x)))].mean(1)
                        rows_g.append({"architecture": ARCH[arch], "test_set": slabel, "arm": alabel, "grading": gname,
                                       "decile": int(dec), "ordered_minus_random": x.mean(), "lower": np.percentile(b, 2.5),
                                       "upper": np.percentile(b, 97.5), "n_items": len(x), "percent": "mean of 10-90%",
                                       "interval_kind": "bootstrap over items (95%)"})
    a.out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("arg_rater_free_slopes.csv", rows_s), ("arg_rater_free_by_decile.csv", rows_g),
                       ("arg_rater_free_agreement.csv", rows_x), ("arg_rater_free_crosstab.csv", rows_c)]:
        pd.DataFrame(rows).to_csv(a.out / name, index=False)
        print(name, len(rows))
    seeds = ", ".join(f"{r['architecture']} {r['n_unmodified_seeds']}" for r in rows_x if r["test_set"] == "natural position")
    (a.out / "README_rater_free.txt").write_text(f"""Rater-free check on the decile result (arg_rater_free_*.csv)

Question: the decile test sets grade each pronoun with the selection rater, which also ordered the removals.
Does the result depend on that? Here the same test items are re-graded by the learner itself.

Own grade: surprisal of the pronoun at its own position under models of the same architecture trained on the
unmodified corpus (Study 1, winning setting; mean over seeds: {seeds}), cut into ten equal-count bins within
each test set. Own decile 0 = pronoun most predictable to the learner. The learner reads left to right, so its
grade uses the preceding text only; the rater also sees what follows.

d = preference for the overt form in an ordered-removal model minus the random-removal model at the same
percent, per test item (base condition, end state, one run per cell).

arg_rater_free_agreement.csv   how far the two gradings agree (Spearman correlations, share within one decile).
arg_rater_free_crosstab.csv    items per (rater decile, own decile) cell.
arg_rater_free_by_decile.csv   mean d per decile, under each grading (mean of 10-90% removed).
arg_rater_free_slopes.csv      slope of d per decile step: each grading alone, and each with the other held
                               fixed (two-predictor least squares). Positive = more dropping toward decile 0.
                               Rows for the mean of 10-90% and for each percent.
Intervals: bootstrap over items (95%). BERT is not included (no left-to-right pronoun probability).
""")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows_x).round(3).to_string(index=False))
    s = pd.DataFrame(rows_s)
    s = s[s.percent == "mean of 10-90%"]
    s["cell"] = s.apply(lambda r: f"{r.slope_per_decile:+.4f} [{r.lower:+.4f}, {r.upper:+.4f}]", axis=1)
    print(s.pivot_table(index=["architecture", "test_set", "arm"], columns="grading", values="cell", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
