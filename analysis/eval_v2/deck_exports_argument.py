#!/usr/bin/env python3
"""Tables behind the "what kind of learner" argument, for the talk deck (long CSV).

Six files, written beside the figure tables of deck_exports.py:
  arg_decile_by_order.csv            decile-graded test set: preference by test-item decile, arm and percent
  arg_decile_order_minus_random.csv  the same, each ordered arm minus random removal at the same percent (paired)
  arg_balanced_endstate.csv          the decile-graded sets pooled over deciles
  arg_matched_item_deciles.csv       where the matched eval's items sit on the rater's deciles, per rendering
  bridge_order_effect_by_condition.csv  Study 2 end states and order effects under each grammar condition
  s1_path_endpoint.csv               Study 1: each condition minus baseline, early window and end state
  s1_distal_endstate.csv             Study 1: each condition minus baseline at the end, by item group

    python analysis/eval_v2/deck_exports_argument.py --data <main>/data/eval_results \\
        --key <stimuli root>/item_key.csv --rater-scores <main>/data/recoverability/analysis/eval_stimuli \\
        --out <dir>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deck_exports as X  # noqa: E402

DEC = "null_subj_test10m_robbi_v1"
W2 = re.compile(r"^(?P<arch>[a-z0-9_]+)-en-pdrop2_(?P<arm>robbianti|robbi|rand|all)(?P<k>\d+)_(?P<iv>base|impcase|lemverb|rmexpl|enrichvm)-h0-s\d+$")
ARCH = {**X.ARCHS, "bert_large_preln": "BERT large"}
ARM = {"rand": "random removal", "robbi": "most recoverable first", "robbianti": "least recoverable first",
       "all": "all removed", "none": "none (unmodified corpus, Study 1 runs)"}
IV = {"base": "baseline", "impcase": "impoverished case", "lemverb": "lemmatized verbs",
      "rmexpl": "expletive sentences removed", "enrichvm": "enriched verbal morphology"}
SETS = {"natural": "natural position", "matched": "position matched across deciles", "nopron": "pronoun-free contexts"}


def removed(arm: str, k: int) -> set[int]:
    """Deciles an ordered arm removed at k percent (decile 0 = most recoverable)."""
    if arm == "robbi":
        return set(range(k // 10))
    if arm == "robbianti":
        return set(range(10 - k // 10, 10))
    if arm == "all":
        return set(range(10))
    return set()


def boot(x: np.ndarray, n_boot: int = 3000, seed: int = 0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def welch(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return float(a.mean() - b.mean()), 0.0
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(a.mean() - b.mean()), float(stats.t.ppf(0.975, df) * se)


def decile_items(data: Path, key: Path) -> pd.DataFrame:
    """Item-level end-state preference on the decile-graded benchmark: one column per model."""
    k = pd.read_csv(key, usecols=["item_id", "cell", "decile", "in_natural", "in_matched", "in_nopron"])
    cols = {}
    for f in sorted((data / DEC / "pairs").glob("cell_id=*.parquet")):
        run = f.stem.split("=", 1)[1]
        p = pd.read_parquet(f, columns=["item_id", "prefers_overt_meanlp"]).set_index("item_id").prefers_overt_meanlp.astype(float)
        m = W2.match(run)
        if m and m["iv"] == "base":
            cols[(m["arch"], m["arm"], int(m["k"]), run)] = p
            continue
        m = X.PROD.match(run)
        if m and m["cond"] == "baseline":
            cols[(m["arch"], "none", 0, run)] = p
    d = pd.DataFrame(cols)
    d.columns.names = ["arch", "arm", "k", "run"]
    return k.set_index("item_id"), d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--rater-scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="write only these (arg, bridge, groups, path, distal)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    want = lambda name: a.only is None or name in a.only  # noqa: E731
    notes = []

    if want("arg"):
        key, d = decile_items(a.data, a.key)
        # one series per (arch, arm, k): the single run, or the mean over the unmodified seeds
        series = d.T.groupby(level=["arch", "arm", "k"]).mean().T
        n_runs = d.T.groupby(level=["arch", "arm", "k"]).size()
        rows, diffs, bal = [], [], []
        for arch in [x for x in ARCH if x in series.columns.get_level_values("arch")]:
            for set_name, set_label in SETS.items():
                items = key.index[key[f"in_{set_name}"]]
                dec = key.loc[items, "decile"]
                for (arm, k) in sorted({c[1:] for c in series.columns if c[0] == arch}, key=lambda t: (t[1], t[0])):
                    s = series[(arch, arm, k)].reindex(items)
                    if s.notna().sum() == 0:
                        continue
                    nr = int(n_runs[(arch, arm, k)])
                    kind = (f"mean of {nr} seeds; bootstrap over items (95%)" if nr > 1
                            else "one run; bootstrap over items (95%)")
                    m_, lo, hi = boot(s.to_numpy())
                    bal.append(dict(figure="arg_balanced_endstate", architecture=ARCH[arch], condition="baseline", arm=ARM[arm],
                                    percent=k, item_group="all cells of the set, equal numbers per decile", test_set=set_label,
                                    mean=m_, lower=lo, upper=hi, n_runs=nr, n_items=int(s.notna().sum()), interval_kind=kind))
                    for dc in range(10):
                        x = s[dec == dc].to_numpy()
                        m_, lo, hi = boot(x, seed=dc)
                        rows.append(dict(figure="arg_decile_by_order", architecture=ARCH[arch], condition="baseline",
                                         arm=ARM[arm], percent=k, test_set=set_label, decile=dc,
                                         decile_removed=dc in removed(arm, k), mean=m_, lower=lo, upper=hi, n_runs=nr,
                                         n_items=int(len(x)), interval_kind=kind))
                    if arm in ("robbi", "robbianti"):
                        for ref_arm, ref_k, ref_label in (("rand", k, "random removal"), ("none", 0, "the unmodified model")):
                            if (arch, ref_arm, ref_k) not in series.columns:
                                continue
                            r = series[(arch, ref_arm, ref_k)].reindex(items)
                            for dc in range(10):
                                x = (s - r)[dec == dc].dropna().to_numpy()
                                m_, lo, hi = boot(x, seed=dc)
                                diffs.append(dict(figure="arg_decile_order_minus_reference", architecture=ARCH[arch],
                                                  condition="baseline", arm=f"{ARM[arm]} minus {ref_label}", percent=k,
                                                  test_set=set_label, decile=dc, decile_removed=dc in removed(arm, k),
                                                  mean=m_, lower=lo, upper=hi, n_runs=nr, n_items=int(len(x)),
                                                  interval_kind="difference paired by item; bootstrap over items (95%)"))
                if set_name == "natural":
                    have = sorted({(c[1], c[2]) for c in series.columns if c[0] == arch and series[c].reindex(items).notna().any()})
                    notes.append(f"decile eval, {ARCH[arch]}: " + ", ".join(
                        f"{ARM[x][:12]} {sum(1 for h in have if h[0] == x)}" for x in ("none", "rand", "robbi", "robbianti", "all")))
        pd.DataFrame(rows).to_csv(a.out / "arg_decile_by_order.csv", index=False)
        pd.DataFrame(diffs).to_csv(a.out / "arg_decile_order_minus_reference.csv", index=False)
        pd.DataFrame(bal).to_csv(a.out / "arg_balanced_endstate.csv", index=False)

        # where the matched eval's items sit on the rater's deciles
        r = pd.read_parquet(a.rater_scores / "robbi__matched_v2__baseline.parquet")
        r = r[r.role == "critical"]
        rend = {("as_is", "one_line"): "as the learners see it: lowercase, context and target on one line",
                ("as_is", "two_lines"): "lowercase, target on its own line (the corpus-ranking convention)",
                ("natural", "one_line"): "truecased, context and target on one line",
                ("natural", "two_lines"): "truecased, target on its own line"}
        out = []
        groups = {"subject drop (2 groups, 288 pairs)": X.SUBJ, **{X.GROUPS[g]: [g] for g in X.GROUPS}}
        for (render, layout), label in rend.items():
            q = r[(r.render == render) & (r.layout == layout)]
            for gname, cats in groups.items():
                g = q[q.category.isin(cats)]
                if g.empty:
                    continue
                cnt = g.decile.astype(int).value_counts().reindex(range(10), fill_value=0)
                for dc in range(10):
                    out.append(dict(figure="arg_matched_item_deciles", rendering=label, item_group=gname, decile=dc,
                                    n_items=int(cnt[dc]), share=float(cnt[dc] / len(g)), mean_decile=float(g.decile.mean()),
                                    median_surprisal_nats=float(g.surp.median()), n_total=int(len(g))))
        pd.DataFrame(out).to_csv(a.out / "arg_matched_item_deciles.csv", index=False)

    if want("bridge"):
        ends = {}
        for bench in (X.V1, X.V2):
            for f in sorted((a.data / bench / "pairs").glob("cell_id=gpt2_large-en-pdrop2_*.parquet")):
                run = f.stem.split("=", 1)[1]
                m = W2.match(run)
                if not m or ((m["iv"] == "enrichvm") != (bench == X.V2)):
                    continue
                p = X.load_pairs(a.data, bench, run)
                for pool, groups in X.POOLS.items():
                    ends[(m["iv"], m["arm"], int(m["k"]), pool)] = X.item_endstate(p, groups)
        rows = []
        for (iv, arm, k, pool), x in sorted(ends.items(), key=lambda kv: (kv[0][3], kv[0][0], kv[0][2], kv[0][1])):
            m_, lo, hi = boot(x.to_numpy())
            rows.append(dict(figure="bridge_order_effect_by_condition", architecture="GPT-2 large", condition=IV[iv],
                             arm=ARM[arm], percent=k, item_group=pool, mean=m_, lower=lo, upper=hi, n_runs=1,
                             interval_kind=f"one run, mean of last 3 checkpoints; bootstrap over {len(x)} items (95%)"))
        for pool in X.POOLS:
            for iv in IV:
                for k in range(10, 100, 10):
                    for name, x, y in (("most minus least recoverable first", "robbi", "robbianti"),
                                       ("most recoverable first minus random removal", "robbi", "rand"),
                                       ("least recoverable first minus random removal", "robbianti", "rand")):
                        if (iv, x, k, pool) not in ends or (iv, y, k, pool) not in ends:
                            continue
                        dlt = (ends[(iv, x, k, pool)] - ends[(iv, y, k, pool)]).dropna().to_numpy()
                        m_, lo, hi = boot(dlt)
                        rows.append(dict(figure="bridge_order_effect_by_condition", architecture="GPT-2 large", condition=IV[iv],
                                         arm=name, percent=k, item_group=pool, mean=m_, lower=lo, upper=hi, n_runs=2,
                                         interval_kind=f"difference of two runs, paired by item; bootstrap over {len(dlt)} items (95%)"))
        pd.DataFrame(rows).to_csv(a.out / "bridge_order_effect_by_condition.csv", index=False)

    if want("groups"):
        # Study 2 end states by item group on the matched eval (GPT-2 large, baseline training condition)
        groups = {**{X.GROUPS[g]: ([g], None) for g in X.GROUPS},
                  "control: subject control": (["control"], "subj_control"),
                  "control: object control": (["control"], "obj_control")}

        def item_ends(p, cats, only):
            q = p if only is None else p[p.condition == only]
            return X.item_endstate(q, cats)

        w2 = {}
        for f in sorted((a.data / X.V1 / "pairs").glob("cell_id=gpt2_large-en-pdrop2_*_base-*.parquet")):
            m = W2.match(f.stem.split("=", 1)[1])
            if m:
                w2[(m["arm"], int(m["k"]))] = X.load_pairs(a.data, X.V1, f.stem.split("=", 1)[1])
        s1, _, _ = X.study1_runs(a.data)
        unmod = [p for (arch, cond, _), p in s1.items() if arch == "gpt2_large" and cond == "baseline"]
        rows = []
        for gname, (cats, only) in groups.items():
            ref = pd.concat([item_ends(p, cats, only) for p in unmod], axis=1).mean(axis=1)
            m_, lo, hi = boot(ref.to_numpy())
            rows.append(dict(figure="s2_item_group_endstate", architecture="GPT-2 large", condition="baseline", arm=ARM["none"],
                             percent=0, item_group=gname, quantity="end state", mean=m_, lower=lo, upper=hi, n_runs=len(unmod),
                             n_items=len(ref), interval_kind=f"mean of {len(unmod)} seeds; bootstrap over items (95%)"))
            ends = {key: item_ends(p, cats, only) for key, p in w2.items()}
            for (arm, k), x in sorted(ends.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                m_, lo, hi = boot(x.to_numpy())
                rows.append(dict(figure="s2_item_group_endstate", architecture="GPT-2 large", condition="baseline", arm=ARM[arm],
                                 percent=k, item_group=gname, quantity="end state", mean=m_, lower=lo, upper=hi, n_runs=1,
                                 n_items=len(x), interval_kind="one run, mean of last 3 checkpoints; bootstrap over items (95%)"))
                dlt = (x - ref).dropna().to_numpy()
                m_, lo, hi = boot(dlt)
                rows.append(dict(figure="s2_item_group_endstate", architecture="GPT-2 large", condition="baseline", arm=ARM[arm],
                                 percent=k, item_group=gname, quantity="difference from the unmodified model", mean=m_,
                                 lower=lo, upper=hi, n_runs=1, n_items=len(dlt),
                                 interval_kind="paired by item against the mean of the unmodified seeds; bootstrap over items (95%)"))
        pd.DataFrame(rows).to_csv(a.out / "s2_item_group_endstate.csv", index=False)

    if want("groups_by_condition"):
        # the same split under each grammar condition; the no-removal reference is that condition's Study 1 runs
        groups = {**{X.GROUPS[g]: ([g], None) for g in X.GROUPS},
                  "control: subject control": (["control"], "subj_control"),
                  "control: object control": (["control"], "obj_control"),
                  "object drop: it (obj_3sg)": (["object_drop"], "obj_3sg"),
                  "object drop: them (obj_3pl)": (["object_drop"], "obj_3pl")}
        long_iv = {"base": "baseline", "impcase": "impoverish_case", "lemverb": "lemmatize_verbs",
                   "rmexpl": "remove_expletive_sentences", "enrichvm": "enrich_verbal_morphology"}

        def item_ends(p, cats, only):
            q = p if only is None else p[p.condition == only]
            return X.item_endstate(q, cats)

        s1, _, _ = X.study1_runs(a.data)
        rows = []
        for iv, ivlabel in IV.items():
            bench = X.V2 if iv == "enrichvm" else X.V1
            w2 = {}
            for f in sorted((a.data / bench / "pairs").glob(f"cell_id=gpt2_large-en-pdrop2_*_{iv}-*.parquet")):
                m = W2.match(f.stem.split("=", 1)[1])
                if m:
                    w2[(m["arm"], int(m["k"]))] = X.load_pairs(a.data, bench, f.stem.split("=", 1)[1])
            ref_runs = [p for (arch, cond, _), p in s1.items() if arch == "gpt2_large" and cond == long_iv[iv]]
            for gname, (cats, only) in groups.items():
                if ref_runs:
                    ref = pd.concat([item_ends(p, cats, only) for p in ref_runs], axis=1).mean(axis=1)
                    m_, lo, hi = boot(ref.to_numpy())
                    rows.append(dict(figure="s2_item_group_endstate_by_condition", architecture="GPT-2 large", condition=ivlabel,
                                     arm="none (nothing removed, Study 1 runs of this condition)", percent=0, item_group=gname,
                                     mean=m_, lower=lo, upper=hi, n_runs=len(ref_runs), n_items=len(ref),
                                     interval_kind=f"mean of {len(ref_runs)} seeds; bootstrap over items (95%)"))
                for (arm, k), p in sorted(w2.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                    x = item_ends(p, cats, only)
                    m_, lo, hi = boot(x.to_numpy())
                    rows.append(dict(figure="s2_item_group_endstate_by_condition", architecture="GPT-2 large", condition=ivlabel,
                                     arm=ARM[arm], percent=k, item_group=gname, mean=m_, lower=lo, upper=hi, n_runs=1,
                                     n_items=len(x),
                                     interval_kind="one run, mean of last 3 checkpoints; bootstrap over items (95%)"))
        pd.DataFrame(rows).to_csv(a.out / "s2_item_group_endstate_by_condition.csv", index=False)

    if want("path") or want("distal"):
        runs, short, early = X.study1_runs(a.data)
        allruns = {**{k: (p, "full") for k, p in runs.items()}, **{k: (p, "early") for k, p in early.items()}}
        if want("path"):
            rows = []
            for pool, groups in X.POOLS.items():
                for arch, alabel in X.ARCHS.items():
                    # per seed: preference by checkpoint rank (schedules differ by a few steps between conditions)
                    def table(cond, kinds):
                        t, tok = {}, {}
                        for (a_, c_, seed), (p, kind) in allruns.items():
                            if a_ == arch and c_ == cond and kind in kinds:
                                s = X.per_checkpoint(p, groups)
                                s = s[s.index > 0]
                                t[seed] = pd.Series(s.to_numpy(), index=range(len(s)))
                                tk = p.drop_duplicates("checkpoint_step").set_index("checkpoint_step").tokens_seen.reindex(s.index)
                                tok[seed] = pd.Series(tk.to_numpy(), index=range(len(s)))
                        return pd.DataFrame(t), pd.DataFrame(tok)
                    base, btok = table("baseline", ("full", "early"))
                    bfull, _ = table("baseline", ("full",))
                    for cond, clabel in X.CONDS.items():
                        if cond == "baseline":
                            continue
                        c, _ = table(cond, ("full", "early"))
                        if c.empty:
                            continue
                        for rank in c.index.intersection(base.index):
                            t = float(btok.loc[rank].median())
                            if t > X.EARLY_END:
                                break
                            x, y = c.loc[rank].dropna(), base.loc[rank].dropna()
                            if len(x) < 8 or len(y) < 8:
                                continue
                            dlt, h = welch(x, y)
                            rows.append(dict(figure="s1_path_endpoint", architecture=alabel, condition=clabel, arm="none", percent=0,
                                             item_group=pool, window="early (all h0 seeds)", tokens_seen=t, mean=dlt,
                                             lower=dlt - h, upper=dlt + h, n_runs=f"{len(x)} vs {len(y)}",
                                             interval_kind="condition minus baseline; Welch 95% across seeds"))
                        cf, _ = table(cond, ("full",))
                        if cf.empty:
                            continue
                        x, y = cf.tail(3).mean(), bfull.tail(3).mean()
                        dlt, h = welch(x, y)
                        rows.append(dict(figure="s1_path_endpoint", architecture=alabel, condition=clabel, arm="none", percent=0,
                                         item_group=pool, window="end state (full-length seeds, last 3 checkpoints)",
                                         tokens_seen=3.84e9, mean=dlt, lower=dlt - h, upper=dlt + h,
                                         n_runs=f"{len(x)} vs {len(y)}",
                                         interval_kind="condition minus baseline; Welch 95% across seeds"))
            pd.DataFrame(rows).to_csv(a.out / "s1_path_endpoint.csv", index=False)
        if want("distal"):
            rows = []
            groups = {**{X.GROUPS[g]: ([g], None) for g in X.GROUPS},
                      "control: subject control": (["control"], "subj_control"),
                      "control: object control": (["control"], "obj_control")}

            def ends_of(arch, cond, cats, only):
                out = []
                for (a_, c_, seed), p in runs.items():
                    if a_ == arch and c_ == cond:
                        q = p if only is None else p[p.condition == only]
                        s = X.per_checkpoint(q, cats)
                        out.append(float(s[s.index > 0].tail(3).mean()))
                return np.array(out)

            for arch, alabel in X.ARCHS.items():
                for gname, (cats, only) in groups.items():
                    b = ends_of(arch, "baseline", cats, only)
                    for cond, clabel in X.CONDS.items():
                        x = ends_of(arch, cond, cats, only)
                        if len(x) == 0:
                            continue
                        from scipy import stats
                        h0 = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
                        rows.append(dict(figure="s1_distal_endstate", architecture=alabel, condition=clabel, arm="none", percent=0,
                                         item_group=gname, quantity="end state", mean=float(x.mean()), lower=float(x.mean() - h0),
                                         upper=float(x.mean() + h0), n_runs=str(len(x)),
                                         interval_kind="95% t-interval across seeds"))
                        if cond != "baseline":
                            dlt, h = welch(x, b)
                            rows.append(dict(figure="s1_distal_endstate", architecture=alabel, condition=clabel, arm="none",
                                             percent=0, item_group=gname, quantity="difference from baseline", mean=dlt,
                                             lower=dlt - h, upper=dlt + h, n_runs=f"{len(x)} vs {len(b)}",
                                             interval_kind="condition minus baseline; Welch 95% across seeds"))
            pd.DataFrame(rows).to_csv(a.out / "s1_distal_endstate.csv", index=False)

    for n in notes:
        print(n)
    for f in sorted(a.out.glob("*.csv")):
        print(f.name, len(pd.read_csv(f)))


if __name__ == "__main__":
    main()
