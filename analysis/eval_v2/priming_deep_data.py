#!/usr/bin/env python3
"""Item-level data for the priming deep-dive (priming_deep_report.py).

Stage 1 (fetch): the raw pair files of the combined pronoun-free + priming eval
(eval_results/null_subj_v2_nopron_plus_priming_v1/pairs/, 4-shard cells), reduced
to a compact per-cell copy under ~/.cache/subject-drop/report_cache/priming_items/.

Stage 2 (reduce): per run, every priming target is paired with its own twin (same
target, same prime item, the prime overt vs null) and with the same target scored
unprimed (the pronoun-free matched rows that ride along in the cell). Writes

  priming_deep/meta.parquet      one row per pairing (pid = the overt twin's item_id)
                                 per intervention condition: categories, conditions,
                                 the word the overt prime / overt target adds, and
                                 their relation (same word, same features, ...)
  priming_deep/end.parquet       run × pairing end-state (mean of the last 3 steps > 0)
  priming_deep/traj.parquet      run × step × (target cat/cond, prime cat/cond,
                                 relation) means, for trajectories

Measures, each for the overt-prime twin (_o), the null-prime twin (_n) and unprimed (_u):
  pref  prefers_overt_meanlp (target overt has the higher mean log prob)
  lpd   mean log prob(overt target) − mean log prob(null target), nats/token
  hot   log prob difference at the hotspot token

    python analysis/eval_v2/priming_deep_data.py            # fetch + reduce
"""

from __future__ import annotations

import argparse
import difflib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nopron_priming_report as R  # noqa: E402

ITEMS = R.CACHE / "priming_items"
OUT = R.CACHE / "priming_deep"
STIM = R.REPO / "evaluation/stimuli/null-subj-v2-priming-nopron-v1"
IV_DIR = {"base": "baseline", "impcase": "impoverish_case", "lemverb": "lemmatize_verbs",
          "enrichvm": "enrich_verbal_morphology", "rmexpl": "remove_expletive_sentences"}
MEAS = ["pref", "lpd", "hot"]

# person, number, gender, case of the overt elements ("that" is the extraction complementizer)
PRON = {"i": (1, "sg", "-", "nom"), "me": (1, "sg", "-", "acc"),
        "we": (1, "pl", "-", "nom"), "us": (1, "pl", "-", "acc"),
        "you": (2, "?", "-", "-"),
        "he": (3, "sg", "m", "nom"), "him": (3, "sg", "m", "acc"),
        "she": (3, "sg", "f", "nom"), "her": (3, "sg", "f", "acc"),
        "it": (3, "sg", "n", "-"),
        "they": (3, "pl", "-", "nom"), "them": (3, "pl", "-", "acc")}


# ---------------------------------------------------------------------------
def fetch():
    ITEMS.mkdir(parents=True, exist_ok=True)
    s3 = R.s3c()
    cells = sorted(c for c in R.list_cells(s3, R.COMB)
                   if c.endswith("of4") and R.RUN_RE.search(c.rsplit("__ckpt_shard", 1)[0]))
    cols = ["checkpoint_step", "category", "condition", "item_id", "prefers_overt_meanlp",
            "log_prob_diff_overt_minus_null", "hotspot_log_prob_diff_overt_minus_null"]

    def one(cell):
        f = ITEMS / f"{cell}.parquet"
        if f.exists():
            return 0
        d = R.get_parquet(s3, R.COMB, "pairs", cell, cols)
        o = pd.DataFrame({
            "step": d.checkpoint_step.astype("int32"),
            "primed": d.category.str.contains("__p_").to_numpy(),
            "category": d.category.astype("category"),
            "condition": d.condition.astype("category"),
            "item_id": d.item_id.astype("int32"),
            "pref": d.prefers_overt_meanlp.astype("bool"),
            "lpd": d.log_prob_diff_overt_minus_null.astype("float32"),
            "hot": d.hotspot_log_prob_diff_overt_minus_null.astype("float32"),
        })
        tmp = f.with_suffix(".tmp")
        o.to_parquet(tmp)
        tmp.rename(f)
        return 1
    t = time.time()
    with ThreadPoolExecutor(16) as ex:
        n = sum(ex.map(one, cells))
    print(f"fetch: {len(cells)} cells, {n} new, {time.time() - t:.0f}s")


# ---------------------------------------------------------------------------
def _added(a: str, b: str) -> list[tuple[int, str]]:
    """(position, word) of the words in a (overt version) that b (null version) lacks."""
    a, b = a.split(), b.split()
    out = []
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            out += list(enumerate(a))[i1:i2]
    return out


def _element(added: list[tuple[int, str]]) -> tuple[int, str]:
    for i, w in added:
        if w in PRON or w == "that":
            return i, w
    return -1, "other"


def _relation(p: str, t: str) -> str:
    if p == "that" or t == "that" or p == "other" or t == "other":
        return "same word" if p == t else "non-pronoun"
    if p == t:
        return "same word"
    a, b = PRON[p], PRON[t]
    if a[:3] == b[:3]:
        return "same features"      # he/him, she/her, they/them, we/us
    return "other pronoun"


def build_meta() -> pd.DataFrame:
    pairing = pd.read_csv(STIM / "pairing.csv", keep_default_na=False)
    key = ["target_category", "target_item_id", "target_condition", "prime_category", "prime_item_id", "prime_condition"]
    ov_id = pairing[pairing.prime_status == "overt"][key + ["item_id"]].rename(columns={"item_id": "pid"})
    nu_id = pairing[pairing.prime_status == "null"][key + ["item_id"]].rename(columns={"item_id": "nid"})
    pm = ov_id.merge(nu_id, on=key, validate="1:1")
    rows = []
    for cond_dir in IV_DIR.values():
        df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted((STIM / cond_dir / "en").glob("*.csv"))], ignore_index=True)
        ov = df[df.pronoun_status == 1].set_index("item_id")
        nu = df[df.pronoun_status == 0].set_index("item_id")
        for r in pm.itertuples(index=False):
            ctx_o, ctx_n = ov.at[r.pid, "context"], ov.at[r.nid, "context"]
            ppos, pw = _element(_added(ctx_o, ctx_n))
            tpos, tw = _element(_added(ov.at[r.pid, "target"], nu.at[r.pid, "target"]))
            rows.append({"cond_dir": cond_dir, "pid": r.pid, "nid": r.nid, "prime_word": pw, "target_word": tw,
                         "relation": _relation(pw, tw),
                         # words from the overt prime's element to the target's (overt) element
                         "distance": len(ctx_o.split()) - ppos + tpos,
                         "ctx_o": ctx_o, "ctx_n": ctx_n,
                         "target_o": ov.at[r.pid, "target"], "target_n": nu.at[r.pid, "target"]})
    meta = pd.DataFrame(rows).merge(pm, on=["pid", "nid"])
    for side in ("prime", "target"):
        f = meta[f"{side}_word"].map(lambda w: PRON.get(w, (0, "-", "-", "-")))
        meta[f"{side}_person"] = f.str[0]
        meta[f"{side}_number"] = f.str[1]
        meta[f"{side}_gender"] = f.str[2]
        meta[f"{side}_case"] = f.str[3]
    return meta


# ---------------------------------------------------------------------------
TRAJ_KEYS = ["target_category", "target_condition", "prime_category", "prime_condition", "relation",
             "prime_word", "target_word"]


def reduce_run(args):
    run, cond_dir, meta_path = args
    meta = pd.read_parquet(meta_path)
    meta = meta[meta.cond_dir == cond_dir]
    d = pd.concat([pd.read_parquet(f) for f in sorted(ITEMS.glob(f"{run}__ckpt_shard*of4.parquet"))], ignore_index=True)
    d["pref"] = d.pref.astype("float32")
    pr, un = d[d.primed], d[~d.primed]
    o = pr[["step", "item_id", *MEAS]].rename(columns={"item_id": "pid"})
    n = pr[["step", "item_id", *MEAS]].rename(columns={"item_id": "nid"})
    w = (meta.merge(o, on="pid").merge(n, on=["nid", "step"], suffixes=("_o", "_n")))
    u = un[["step", "category", "condition", "item_id", *MEAS]].rename(
        columns={"category": "target_category", "condition": "target_condition", "item_id": "target_item_id",
                 **{m: f"{m}_u" for m in MEAS}})
    u["target_category"] = u.target_category.astype(str)
    u["target_condition"] = u.target_condition.astype(str)
    w = w.merge(u, on=["step", "target_category", "target_condition", "target_item_id"], how="left")
    vals = [f"{m}_{s}" for m in MEAS for s in "onu"]
    steps = sorted(w.step.unique())
    last = [s for s in steps if s > 0][-3:]
    end = w[w.step.isin(last)].groupby("pid")[vals].mean().reset_index()
    end["run"] = run
    traj = w.groupby(["step", *TRAJ_KEYS])[vals].mean().reset_index()
    traj["n"] = w.groupby(["step", *TRAJ_KEYS]).size().to_numpy()
    traj["run"] = run
    return end, traj, len(steps), int(w[vals].isna().any(axis=1).sum())


def reduce(workers: int):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = build_meta()
    meta.to_parquet(OUT / "meta.parquet")
    runs = sorted({f.name.split("__ckpt_shard")[0] for f in ITEMS.glob("*of4.parquet")})
    complete = [r for r in runs if len(list(ITEMS.glob(f"{r}__ckpt_shard*of4.parquet"))) == 4]
    jobs = [(r, IV_DIR[R.RUN_RE.search(r).group("iv")], str(OUT / "meta.parquet")) for r in complete]
    ends, trajs = [], []
    with ProcessPoolExecutor(workers) as ex:
        for (end, traj, nsteps, nmiss), (r, _, _) in zip(ex.map(reduce_run, jobs), jobs):
            ends.append(end)
            trajs.append(traj)
            if nmiss:
                print(f"  {r}: {nmiss} pairings with a missing measure")
    end = pd.concat(ends, ignore_index=True)
    traj = pd.concat(trajs, ignore_index=True)
    for c in TRAJ_KEYS:
        traj[c] = traj[c].astype("category")
    end.to_parquet(OUT / "end.parquet")
    traj.to_parquet(OUT / "traj.parquet")
    print(f"reduce: {len(complete)} runs; end {end.shape}, traj {traj.shape}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if not a.skip_fetch:
        fetch()
    reduce(a.workers)


if __name__ == "__main__":
    main()
