#!/usr/bin/env python3
"""Figure tables for the talk deck (presentations/NullPronounsv2), long-format CSV.

Study 1 = the grammar-intervention wave (no pronoun removal), five architectures, the full-length
seeds at the winning hyperparameter setting (h0). Study 2 = pronoun removal in GPT-2 large,
baseline training condition, one run per cell. Everything is on the condition-matched eval
(null_subj_v2_condition_matched_v1), enrichment on the fixed items (..._v2).

Every file has the columns
  figure, architecture, condition, arm, percent, item_group, tokens_seen, checkpoint_step,
  mean, lower, upper, n_runs, interval_kind
and carries two poolings of the item groups in `item_group` (see POOLS), so the headline
pooling is a filter, not a rebuild.

    python analysis/eval_v2/deck_exports.py --data <main>/data/eval_results --out <dir>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V1, V2 = "null_subj_v2_condition_matched_v1", "null_subj_v2_condition_matched_v2"
ARCHS = {"gpt2_small": "GPT-2 small", "gpt2_medium": "GPT-2 medium", "gpt2_large": "GPT-2 large", "lstm": "LSTM",
         "mamba_370m": "Mamba (370M)"}
CONDS = {"baseline": "baseline", "impoverish_case": "impoverished case", "lemmatize_verbs": "lemmatized verbs",
         "remove_expletive_sentences": "expletive sentences removed",
         "enrich_verbal_morphology": "enriched verbal morphology"}
ARMS = {"rand": "random removal", "robbi": "most recoverable first", "robbianti": "least recoverable first",
        "all": "all removed"}
GROUPS = {"subject_drop": "subject drop", "subject_drop_no_agreement": "subject drop, no agreement",
          "expletive": "expletive", "conjunction": "conjunction", "embedded_drop": "embedded", "control": "control",
          "extraction": "extraction", "object_drop": "object drop"}
SUBJ = ["subject_drop", "subject_drop_no_agreement"]
OVERALL = ["subject_drop", "subject_drop_no_agreement", "expletive", "conjunction", "embedded_drop", "control"]
POOLS = {"subject drop (2 groups, 288 pairs)": SUBJ, "overall (6 groups, 480 pairs)": OVERALL}
PROD = re.compile(r"^(?P<arch>[a-z0-9_]+)-en-(?P<cond>[a-z_]+)-h(?P<hp>\d+)-s(?P<seed>\d+)$")
W2 = re.compile(r"^gpt2_large-en-pdrop2_(?P<arm>robbianti|robbi|rand|all)(?P<k>\d+)_base-h0-s\d+$")
FULL_TOKENS = 3.0e9
# The early-only seeds stop after two epochs. They follow the same 30-epoch learning-rate schedule as the
# full-length seeds (docs/STAGE1_CONDITION_MATCHED.md), so up to here every h0 seed is the same experiment.
EARLY_END = 256.0e6
COLS = ["figure", "architecture", "condition", "arm", "percent", "item_group", "tokens_seen", "checkpoint_step",
        "mean", "lower", "upper", "n_runs", "interval_kind"]


def tval(n: int) -> float:
    from scipy import stats
    return float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")


def load_pairs(data: Path, bench: str, run: str) -> pd.DataFrame | None:
    """One run's item-level table with tokens attached; checkpoints with zero-filled weights dropped."""
    f = data / bench / "pairs" / f"cell_id={run}.parquet"
    c = data / bench / "checkpoints" / f"cell_id={run}.parquet"
    if not f.exists() or not c.exists():
        return None
    p = pd.read_parquet(f, columns=["checkpoint_step", "category", "condition", "item_id", "prefers_overt_meanlp",
                                    "log_prob_diff_overt_minus_null"])
    dead = p.groupby("checkpoint_step").log_prob_diff_overt_minus_null.apply(lambda x: (x == 0).all())
    p = p[~p.checkpoint_step.isin(dead[dead].index)]
    p["pref"] = p.prefers_overt_meanlp.astype(float)
    ck = pd.read_parquet(c, columns=["checkpoint_step", "tokens_seen"]).set_index("checkpoint_step").tokens_seen
    p["tokens_seen"] = p.checkpoint_step.map(ck)
    p.attrs["dropped"] = [int(x) for x in dead[dead].index]
    return p[["checkpoint_step", "tokens_seen", "category", "condition", "item_id", "pref"]]


def per_checkpoint(p: pd.DataFrame, groups) -> pd.Series:
    return p[p.category.isin(groups)].groupby("checkpoint_step").pref.mean()


def study1_runs(data: Path):
    """(arch, cond, seed) -> item-level table, for full-length h0 seeds; enrichment from the fixed items.

    Also returns `early`: the h0 seeds that stop before the end (the two-epoch runs, and the few
    truncated ones), cut at EARLY_END, for pooling into the early window.
    """
    out, short, early = {}, [], {}
    v1 = {f.stem.split("=", 1)[1] for f in (data / V1 / "pairs").glob("cell_id=*.parquet")}
    v2 = {f.stem.split("=", 1)[1] for f in (data / V2 / "pairs").glob("cell_id=*.parquet")}
    for run in sorted(v1 | v2):
        m = PROD.match(run)
        if not m or m["arch"] not in ARCHS or m["cond"] not in CONDS or int(m["hp"]) != 0:
            continue
        bench = V2 if m["cond"] == "enrich_verbal_morphology" else V1
        if run not in (v2 if bench == V2 else v1):
            continue
        p = load_pairs(data, bench, run)
        if p is None:
            continue
        # checkpoint 0 is the untrained model (identical to the separately scored initialization);
        # its tokens_seen label is nominal, so place it at zero
        p.loc[p.checkpoint_step == 0, "tokens_seen"] = 0.0
        if p.tokens_seen.max() < FULL_TOKENS:
            short.append((run, float(p.tokens_seen.max())))
            early[(m["arch"], m["cond"], int(m["seed"]))] = p[p.tokens_seen <= EARLY_END]
            continue
        out[(m["arch"], m["cond"], int(m["seed"]))] = p
    return out, short, early


def across_seeds(runs: dict, arch: str, cond: str, groups, min_runs: int = 2, early: dict | None = None,
                 from_step: int = 1) -> pd.DataFrame:
    """Mean and 95% t-interval across seeds at each checkpoint.

    `early` adds the seeds that stop early, at the checkpoints they have (all at or before EARLY_END);
    interval_kind then says how many of each kind a row pools.
    """
    series, tok, kind = {}, {}, {}
    for source, label in ((runs, "full"), (early or {}, "early")):
        for (a, c, seed), p in source.items():
            if a == arch and c == cond:
                series[seed] = per_checkpoint(p, groups)
                tok[seed] = p.drop_duplicates("checkpoint_step").set_index("checkpoint_step").tokens_seen
                kind[seed] = label
    if not series:
        return pd.DataFrame()
    s, t = pd.DataFrame(series), pd.DataFrame(tok)
    s = s[s.index >= from_step]
    n = s.notna().sum(axis=1)
    s, n = s[n >= min_runs], n[n >= min_runs]
    nf = s[[k for k in s.columns if kind[k] == "full"]].notna().sum(axis=1)
    ne = n - nf
    half = np.array([tval(int(k)) for k in n]) * s.std(axis=1, ddof=1).to_numpy() / np.sqrt(n.to_numpy())

    def describe(k, f, e):
        if e == 0:
            return f"across {k} seeds (95% t-interval)"
        return f"across {k} seeds (95% t-interval): {f} full-length + {e} that stop after two epochs or earlier"

    return pd.DataFrame({"checkpoint_step": s.index, "tokens_seen": t.reindex(s.index).median(axis=1).to_numpy(),
                         "mean": s.mean(axis=1).to_numpy(), "lower": s.mean(axis=1).to_numpy() - half,
                         "upper": s.mean(axis=1).to_numpy() + half, "n_runs": n.to_numpy(),
                         "interval_kind": [describe(int(k), int(f), int(e)) for k, f, e in zip(n, nf, ne)]})


def over_items(p: pd.DataFrame, groups, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One run: mean over items and a 95% interval from resampling items, at each checkpoint (step > 0)."""
    q = p[p.category.isin(groups) & (p.checkpoint_step > 0)]
    key = q.category + "|" + q.condition + "|" + q.item_id.astype(str)
    M = q.assign(key=key).pivot_table(index="checkpoint_step", columns="key", values="pref")
    A = M.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, A.shape[1], size=(n_boot, A.shape[1]))
    B = np.stack([np.nanmean(A[:, i], axis=1) for i in idx])          # n_boot x checkpoints
    tok = q.drop_duplicates("checkpoint_step").set_index("checkpoint_step").tokens_seen.reindex(M.index)
    return pd.DataFrame({"checkpoint_step": M.index, "tokens_seen": tok.to_numpy(), "mean": np.nanmean(A, axis=1),
                         "lower": np.percentile(B, 2.5, axis=0), "upper": np.percentile(B, 97.5, axis=0), "n_runs": 1,
                         "interval_kind": f"one run; bootstrap over {A.shape[1]} items (95%)"})


def item_endstate(p: pd.DataFrame, groups, last: int = 3) -> pd.Series:
    """Per item: mean preference over the run's last `last` checkpoints."""
    steps = sorted(p.checkpoint_step.unique())[-last:]
    q = p[p.category.isin(groups) & p.checkpoint_step.isin(steps)]
    return q.groupby(q.category + "|" + q.condition + "|" + q.item_id.astype(str)).pref.mean()


def boot_mean(x: np.ndarray, n_boot: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs, short, early = study1_runs(a.data)
    n_seeds = pd.Series({k[:2]: 0 for k in runs})
    for k in runs:
        n_seeds[k[:2]] += 1

    def tag(df, **kw):
        for k, v in kw.items():
            df[k] = v
        return df[COLS]

    # 1 and 2: Study 1 trajectories
    f1, f2, f5 = [], [], []
    for arch, alabel in ARCHS.items():
        for cond, clabel in CONDS.items():
            for pool, groups in POOLS.items():
                t = across_seeds(runs, arch, cond, groups, early=early, from_step=0)
                if t.empty:
                    continue
                row = dict(architecture=alabel, condition=clabel, arm="none", percent=0, item_group=pool)
                f2.append(tag(t.copy(), figure="s1_condition_trajectories", **row))
                if cond == "baseline":
                    f1.append(tag(t.copy(), figure="s1_baseline_trajectories", **row))
        for g, glabel in GROUPS.items():
            t = across_seeds(runs, arch, "baseline", [g], early=early, from_step=0)
            if not t.empty:
                f5.append(tag(t, figure="s1_item_group_trajectories", architecture=alabel, condition="baseline", arm="none",
                              percent=0, item_group=glabel))
        for cond_name, glabel in (("subj_control", "control: subject control"), ("obj_control", "control: object control")):
            sub = {k: p[p.condition == cond_name] for k, p in runs.items() if k[0] == arch and k[1] == "baseline"}
            sub_e = {k: p[p.condition == cond_name] for k, p in early.items() if k[0] == arch and k[1] == "baseline"}
            t = across_seeds(sub, arch, "baseline", ["control"], early=sub_e, from_step=0)
            if not t.empty:
                f5.append(tag(t, figure="s1_item_group_trajectories", architecture=alabel, condition="baseline", arm="none",
                              percent=0, item_group=glabel))
    pd.concat(f1).to_csv(a.out / "s1_baseline_trajectories.csv", index=False)
    pd.concat(f2).to_csv(a.out / "s1_condition_trajectories.csv", index=False)
    pd.concat(f5).to_csv(a.out / "s1_item_group_trajectories.csv", index=False)

    # 3 and 4: Study 2 (GPT-2 large, baseline training condition)
    w2, dropped = {}, {}
    for f in sorted((a.data / V1 / "pairs").glob("cell_id=gpt2_large-en-pdrop2_*_base-*.parquet")):
        run = f.stem.split("=", 1)[1]
        m = W2.match(run)
        if m:
            p = load_pairs(a.data, V1, run)
            w2[(m["arm"], int(m["k"]))] = p
            if p.attrs.get("dropped"):
                dropped[run] = p.attrs["dropped"]
    f3 = []
    for pool, groups in POOLS.items():
        for arm, k in (("rand", 50), ("robbi", 50), ("robbianti", 50), ("all", 100)):
            t = over_items(w2[(arm, k)], groups)
            f3.append(tag(t, figure="s2_order_trajectories", architecture="GPT-2 large", condition="baseline",
                          arm=ARMS[arm], percent=k, item_group=pool))
        t = across_seeds(runs, "gpt2_large", "baseline", groups)
        f3.append(tag(t, figure="s2_order_trajectories", architecture="GPT-2 large", condition="baseline",
                      arm="none (unmodified corpus, Study 1 runs)", percent=0, item_group=pool))
    pd.concat(f3).to_csv(a.out / "s2_order_trajectories.csv", index=False)

    f4, eff = [], []
    for pool, groups in POOLS.items():
        items = {key: item_endstate(p, groups) for key, p in w2.items()}
        end_tok = {key: float(p.tokens_seen.max()) for key, p in w2.items()}
        end_step = {key: int(p.checkpoint_step.max()) for key, p in w2.items()}
        for (arm, k), x in sorted(items.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            m_, lo, hi = boot_mean(x.to_numpy())
            f4.append(dict(figure="s2_order_endstate_by_percent", architecture="GPT-2 large", condition="baseline",
                           arm=ARMS[arm], percent=k, item_group=pool, tokens_seen=end_tok[(arm, k)],
                           checkpoint_step=end_step[(arm, k)], mean=m_, lower=lo, upper=hi, n_runs=1,
                           interval_kind=f"one run, mean of last 3 checkpoints; bootstrap over {len(x)} items (95%)"))
        # the unmodified corpus as the 0% point, across the Study 1 seeds
        ends = [item_endstate(p, groups).mean() for (arch, cond, _), p in runs.items() if arch == "gpt2_large" and cond == "baseline"]
        h = tval(len(ends)) * np.std(ends, ddof=1) / np.sqrt(len(ends))
        f4.append(dict(figure="s2_order_endstate_by_percent", architecture="GPT-2 large", condition="baseline",
                       arm="none (unmodified corpus, Study 1 runs)", percent=0, item_group=pool, tokens_seen=3.84e9,
                       checkpoint_step=7500, mean=float(np.mean(ends)), lower=float(np.mean(ends) - h),
                       upper=float(np.mean(ends) + h), n_runs=len(ends),
                       interval_kind=f"across {len(ends)} seeds (95% t-interval), mean of last 3 checkpoints"))
        # order effects at each percent, paired by item
        for k in range(10, 100, 10):
            for name, x, y in (("most recoverable first minus random removal", "robbi", "rand"),
                               ("least recoverable first minus random removal", "robbianti", "rand"),
                               ("most minus least recoverable first", "robbi", "robbianti")):
                d = (items[(x, k)] - items[(y, k)]).dropna().to_numpy()
                m_, lo, hi = boot_mean(d)
                eff.append(dict(figure="s2_order_effect_by_percent", architecture="GPT-2 large", condition="baseline",
                                arm=name, percent=k, item_group=pool, tokens_seen=end_tok[(x, k)],
                                checkpoint_step=end_step[(x, k)], mean=m_, lower=lo, upper=hi, n_runs=2,
                                interval_kind=f"difference of two runs, paired by item; bootstrap over {len(d)} items (95%)"))
    pd.DataFrame(f4)[COLS].to_csv(a.out / "s2_order_endstate_by_percent.csv", index=False)
    pd.DataFrame(eff)[COLS].to_csv(a.out / "s2_order_effect_by_percent.csv", index=False)

    # notes
    cells = {(arch, cond): int(n_seeds.get((arch, cond), 0)) for arch in ARCHS for cond in CONDS}
    n_early = pd.Series({k[:2]: 0 for k in early})
    for k in early:
        n_early[k[:2]] += 1
    lines = ["Deck exports: Study 1 (grammar interventions, five architectures) and Study 2 (pronoun removal, GPT-2 large).",
             "Measure: share of pairs in which the sentence with its pronoun scores higher than the same sentence without it",
             "(prefers_overt_meanlp); chance 0.5.", "",
             "item_group poolings carried in every file (filter to one):"]
    lines += [f"  {k}: {', '.join(GROUPS[g] for g in v)}" for k, v in POOLS.items()]
    lines += ["  The overall pooling is the one the Study 2 figures have used so far; it INCLUDES the control group (subject and",
              "  object control), where the version without the pronoun is the grammatical one, and excludes extraction and",
              "  object drop.", "",
              "tokens_seen: end of epoch 1 = 128,000,000 (nearest shared checkpoint 131,072,000); Study 1 runs end at",
              "3,840,000,000; Study 2 runs end at " + ", ".join(sorted({f"{p.tokens_seen.max():,.0f}" for p in w2.values()})) + ".",
              "", "Study 1 runs: full-length seeds at the winning hyperparameter setting (h0). Seeds per cell:"]
    for arch, alabel in ARCHS.items():
        lines.append("  " + alabel + ": " + ", ".join(
            f"{CONDS[c]} {cells[(arch, c)]}" + (" (PENDING re-score)" if cells[(arch, c)] == 0 else "") for c in CONDS))
    lines += ["", "Early window (Study 1 files only): up to 256,000,000 tokens every h0 seed is pooled, the full-length ones and",
              "the ones that stop after two epochs or earlier (same setting, same 30-epoch learning-rate schedule, same",
              "checkpoints). After that only full-length seeds remain. n_runs and interval_kind give the count per row.",
              "Seeds in the early window (full-length + early-only):"]
    for arch, alabel in ARCHS.items():
        lines.append("  " + alabel + ": " + ", ".join(
            f"{CONDS[c]} {cells[(arch, c)]}+{int(n_early.get((arch, c), 0))}" for c in CONDS if cells[(arch, c)]))
    lines += ["For GPT-2 large and Mamba the early-only seeds were checkpointed more densely between 131M and 256M tokens,",
              "where the full-length seeds have no checkpoint: those rows hold the early-only seeds alone.",
              "checkpoint_step 0 is the untrained model (verified identical to the separately scored initialization for",
              "GPT-2 small); it is exported with tokens_seen = 0. The Study 2 files start at step 1 as before.", ""]
    lines += ["Cells with fewer than six: " + "; ".join(f"{ARCHS[a_]} x {CONDS[c]} = {n}" for (a_, c), n in cells.items() if n < 6),
              "h0 runs left out as truncated (did not reach 3.0B tokens): " + str(len(short)) + " (the 2-epoch runs and "
              + str(sum(1 for _, t in short if t > 0.5e9)) + " that stop between 0.5B and 3.0B tokens).",
              "", "Study 2: one run per cell, so bands are over items, not over seeds. The 0% reference is the Study 1",
              "GPT-2 large baseline (same architecture and winning setting, six seeds), band across seeds.",
              "Checkpoints with zero-filled weights dropped: " + (str(dropped) if dropped else "none in the files used"),
              "End state = mean of a run's last three checkpoints.",
              "s2_order_effect_by_percent.csv is an extra file: differences between orders at each percent, paired by item."]
    (a.out / "README.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for f in sorted(a.out.glob("s[12]_*.csv")):
        print(f.name, len(pd.read_csv(f)))


if __name__ == "__main__":
    main()
