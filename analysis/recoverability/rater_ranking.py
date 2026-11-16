#!/usr/bin/env python3
"""Ranking and deciles for one of our own raters, from the re-rating shards.

Reads the shards written by scripts/launch_rater_scoring.py
(<scores>/<corpus>/<genre>.sIofN.parquet), keeps the frozen selection population (the same
6.46M personal-pronoun subjects every removal order was drawn from), and writes, per corpus and genre,
one row per pronoun with the rater's surprisal, its decile, and robbi's surprisal and decile beside it.

Deciles: train_90M is cut into ten equal groups by surprisal (0 = most recoverable; ties broken by
corpus order, as for robbi). The nine upper bounds are stored and applied to pull_10M and test_10M.

    python analysis/recoverability/rater_ranking.py --rater enrichvm --work data/recoverability/rater_scores --pull

--pull downloads the shards, the robbi selection and the test-set tables from S3 first (AWS_PROFILE).
Outputs under <work>/rater_<short>/ranking/: <corpus>/<genre>.parquet, RANKING_MANIFEST.json,
by_form.csv, decile_vs_robbi.csv, by_genre.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET = "thomas-subject-drop-artifacts"
GENRES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
FORMS = ["i", "you", "he", "she", "it", "we", "they", "one"]
SHARD = re.compile(r"^(?P<genre>[a-z_]+)\.s(?P<i>\d+)of(?P<n>\d+)\.parquet$")


def pull(work: Path, short: str) -> None:
    import boto3
    s3 = boto3.client("s3", endpoint_url="https://s3-west.nrp-nautilus.io")
    pg = s3.get_paginator("list_objects_v2")

    def sync(prefix: str, dest: Path, keep=lambda k: True) -> None:
        for page in pg.paginate(Bucket=BUCKET, Prefix=prefix):
            for o in page.get("Contents", []):
                rel = o["Key"][len(prefix):]
                p = dest / rel
                if not keep(rel) or (p.exists() and p.stat().st_size == o["Size"]):
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET, o["Key"], str(p))
                print("  got", o["Key"], flush=True)

    sync(f"recoverability/rater_scores/rater_{short}/", work / f"rater_{short}" / "scores",
         keep=lambda r: not r.startswith("ranking/"))
    sync("recoverability/analysis/selection_v5/robbi/", work / "selection_robbi")
    sync("recoverability/eval_stimuli/test_10M/", work / "test_10M",
         keep=lambda r: "/" not in r and (r.startswith("instances__") or r.startswith("robbi__")) and r.endswith(".parquet"))


def load_scores(d: Path, genre: str) -> tuple[pd.DataFrame | None, str]:
    """All shards of one genre, or (None, why) when the set is incomplete."""
    found = {}
    for p in d.glob(f"{genre}.s*of*.parquet"):
        m = SHARD.match(p.name)
        if m and m["genre"] == genre:
            found[(int(m["i"]), int(m["n"]))] = p
    if not found:
        return None, "no shards"
    n = {k[1] for k in found}
    if len(n) != 1:
        return None, f"mixed shard counts {sorted(n)}"
    n = n.pop()
    missing = [i for i in range(n) if (i, n) not in found]
    if missing:
        return None, f"missing shards {missing} of {n}"
    t = pd.concat([pd.read_parquet(found[(i, n)]) for i in range(n)], ignore_index=True)
    man = json.loads((d / f"{genre}.s0of{n}.manifest.json").read_text())
    if len(t) != man["n_instances_in_file"]:
        return None, f"{len(t)} rows, manifest says {man['n_instances_in_file']}"
    t["surp"] = -t.logprob_sum
    return t, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rater", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="preview: go on with the genres whose shards are complete (thresholds are then not final)")
    a = ap.parse_args()
    if a.pull:
        pull(a.work, a.rater)
    root = a.work / f"rater_{a.rater}"
    out = root / "ranking"
    out.mkdir(parents=True, exist_ok=True)
    keep = ["line_idx", "token_i", "char_start", "form_edited", "n_pieces", "surp", "entropy"] + [f"p_{f}" for f in FORMS]

    tables: dict[str, list[pd.DataFrame]] = {}
    status: dict[str, dict] = {}
    for corpus in ("train_90M", "pull_10M"):
        for g in GENRES:
            t, why = load_scores(root / "scores" / corpus, g)
            status[f"{corpus}/{g}"] = {"scores": why}
            if t is None:
                continue
            sel = pd.read_parquet(a.work / "selection_robbi" / corpus / f"{g}.parquet")
            sel = sel.rename(columns={"rank_value": "surp_robbi", "info_decile": "decile_robbi"})
            sel.loc[sel.surp_robbi > 1e30, "surp_robbi"] = np.inf
            m = sel.merge(t[keep], on=["line_idx", "token_i"], how="left")
            rem = root / "scores" / corpus / f"{g}.line_removed.parquet"
            m["line_removed"] = False
            if rem.exists():
                r = pd.read_parquet(rem)[["line_idx", "token_i"]].assign(_rm=True)
                m = m.merge(r, on=["line_idx", "token_i"], how="left")
                m["line_removed"] = m.pop("_rm").fillna(False).astype(bool)
            m.insert(0, "genre", g)
            status[f"{corpus}/{g}"].update(population=len(sel), scored=int(m.surp.notna().sum()),
                                           line_removed=int(m.line_removed.sum()),
                                           unscored=int((m.surp.isna() & ~m.line_removed).sum()))
            tables.setdefault(corpus, []).append(m)
    incomplete = [k for k, v in status.items() if v["scores"] != "ok"]
    if incomplete and not a.allow_partial:
        print(json.dumps(status, indent=1))
        sys.exit(f"incomplete: {incomplete}. Use --allow-partial for a preview.")
    if "train_90M" not in tables and not a.allow_partial:
        sys.exit("no train_90M scores")

    # deciles from the training corpus (or, in a preview without it, from whatever is there)
    ref_name = "train_90M" if "train_90M" in tables else next(iter(tables))
    ref = pd.concat(tables[ref_name], ignore_index=True)
    ref = ref[~ref.line_removed]
    rv = ref.surp.fillna(np.inf)       # a pronoun the rater could not be given goes last, as for robbi
    dec = pd.qcut(rv.rank(method="first"), 10, labels=False)
    thresholds = [float(rv[dec <= d].max()) for d in range(9)]

    all_rows = []
    for corpus, parts in tables.items():
        for m in parts:
            g = m.genre.iat[0]
            rv = m.surp.fillna(np.inf)
            if corpus == ref_name:
                idx = ref.index[ref.genre == g]
                m.loc[~m.line_removed, "decile_rater"] = dec.loc[idx].to_numpy()
            else:
                m["decile_rater"] = np.where(m.line_removed, np.nan, np.searchsorted(thresholds, rv, side="left"))
            (out / corpus).mkdir(exist_ok=True)
            m.to_parquet(out / corpus / f"{g}.parquet", index=False)
            all_rows.append(m.assign(corpus=corpus))
    d = pd.concat(all_rows, ignore_index=True)
    d = d[~d.line_removed & d.surp.notna()]

    # test set: the layered-annotation instance table, with robbi's test scores beside
    tdir = root / "scores" / "test_10M"
    test_status = {}
    if tdir.exists():
        for g in GENRES:
            t, why = load_scores(tdir, g)
            test_status[g] = why
            if t is None or not (a.work / "test_10M" / f"instances__{g}.parquet").exists():
                continue
            inst = pd.read_parquet(a.work / "test_10M" / f"instances__{g}.parquet")
            m = inst.merge(t[keep + ["from_parse"]].drop(columns=["token_i"]), on=["line_idx", "char_start"], how="left")
            rb = a.work / "test_10M" / f"robbi__{g}.parquet"
            if rb.exists():
                r = pd.read_parquet(rb, columns=["line_idx", "char_start", "surp", "decile"])
                m = m.merge(r.rename(columns={"surp": "surp_robbi", "decile": "decile_robbi"}), on=["line_idx", "char_start"], how="left")
            m["decile_rater"] = np.where(m.surp.isna(), np.nan, np.searchsorted(thresholds, m.surp.fillna(np.inf), side="left"))
            (out / "test_10M").mkdir(exist_ok=True)
            m.to_parquet(out / "test_10M" / f"{g}.parquet", index=False)
            test_status[g] = {"instances": len(inst), "scored": int(m.surp.notna().sum()),
                              "decile_share": m.decile_rater.value_counts(normalize=True).sort_index().round(4).tolist()}

    fin = d[np.isfinite(d.surp_robbi)]
    by_form = d.groupby(["corpus", "form"]).agg(
        n=("surp", "size"), median_rater=("surp", "median"), mean_rater=("surp", "mean"),
        p_true_rater=("surp", lambda x: float(np.exp(-x).mean())), median_robbi=("surp_robbi", "median"),
        mean_decile_rater=("decile_rater", "mean"), mean_decile_robbi=("decile_robbi", "mean")).reset_index()
    by_form.to_csv(out / "by_form.csv", index=False)
    by_genre = fin.groupby(["corpus", "genre"]).apply(lambda x: pd.Series({
        "n": len(x), "spearman_with_robbi": x.surp.corr(x.surp_robbi, method="spearman"),
        "median_rater": x.surp.median(), "median_robbi": x.surp_robbi.median(),
        "share_under_0.01_rater": (x.surp < 0.01).mean(), "share_under_0.01_robbi": (x.surp_robbi < 0.01).mean(),
        "same_decile": (x.decile_rater == x.decile_robbi).mean(),
        "within_one_decile": ((x.decile_rater - x.decile_robbi).abs() <= 1).mean()})).reset_index()
    by_genre.to_csv(out / "by_genre.csv", index=False)
    r = d[d.corpus == ref_name]
    pd.crosstab(r.decile_robbi, r.decile_rater, normalize="index").round(4).to_csv(out / "decile_vs_robbi.csv")
    manifest = {
        "rater": f"rater_{a.rater}", "deciles_from": ref_name, "final": not incomplete and ref_name == "train_90M",
        "decile_upper_thresholds": thresholds,
        "n_ranked": int(len(ref)), "n_unscored_in_reference": int(ref.surp.isna().sum()),
        "spearman_with_robbi_reference": float(r[np.isfinite(r.surp_robbi)].surp.corr(r[np.isfinite(r.surp_robbi)].surp_robbi, method="spearman")),
        "share_under_0.001_nats": float((r.surp < 1e-3).mean()), "share_under_0.01_nats": float((r.surp < 1e-2).mean()),
        "share_under_0.1_nats": float((r.surp < 0.1).mean()),
        "note": "0 = most recoverable. Each rater scores its own condition's text ([SEP] after every line, 250 pieces "
                "each side). Low deciles are near-ties: read differences below about 0.01 nats as none.",
        "status": status, "test_10M": test_status,
    }
    (out / "RANKING_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    pd.set_option("display.width", 220)
    print(json.dumps({k: manifest[k] for k in ("deciles_from", "final", "decile_upper_thresholds", "n_ranked",
                                                "n_unscored_in_reference", "spearman_with_robbi_reference",
                                                "share_under_0.01_nats", "share_under_0.1_nats")}, indent=1))
    print(by_genre.round(3).to_string(index=False))
    print(by_form[by_form.n >= 500].round(3).to_string(index=False))
    if incomplete:
        print("PREVIEW ONLY; incomplete:", incomplete)


if __name__ == "__main__":
    main()
