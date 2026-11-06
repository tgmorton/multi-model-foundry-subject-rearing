"""Step 2: decile profiles with genre x form direct standardization and a
stratified cluster bootstrap.

For every (scheme in robbi/bert/gpt2m) x (split in all/contracted/
uncontracted) x feature x decile:
  raw  = pooled mean (prevalence for binary features) over valid rows
  std  = sum_s w_s * mean(d, s), s = genre x form stratum, w_s = the stratum's
         share of the WHOLE population (6.46M; same weights for every scheme
         and split). Strata with < MIN_CELL valid rows in that decile are
         dropped and the weights renormalized; std_cov reports the population
         weight actually covered. Composition features (genre_*, form_*) get
         no std (it would be the population share by construction).
  CIs  = 2.5/97.5 percentiles over B bootstrap replicates resampling clusters
         with replacement within genre (cluster = doc_id for CHILDES /
         gutenberg / simple_wiki, else a block of 200 consecutive lines).

Outputs: decile_profiles.csv (long), decile_profiles_wide.csv (robbi/all,
raw and std side by side), decile_profiles.meta.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survey_common as C  # noqa: E402

MIN_CELL = 5
FORMS = ["i", "you", "he", "she", "it", "we", "they", "one", "thou", "ye"]
LOAD_COLS = ["genre", "form", "token_i", "d_robbi", "d_bert", "d_gpt2m",
             "s_robbi", "s_bert", "s_gpt2m", "im_joined", "dep", "is_title",
             "head_i", "contracted", "line_initial", "sent_initial",
             "after_and", "after_wh", "wh_inverted", "inverted",
             "inverted_question", "tag_question", "line_ends_q",
             "r1_punct_eol", "agree_set_size", "agree_set_size_l1",
             "line_len_tokens", "tokens_before", "line_dup_count",
             "tokens_since_doc_start", "speaker_change", "childes_is_child",
             "age_months", "turn_marker", "cluster_id", "char0",
             "prev_line_end"]


def build_features(d: pd.DataFrame):
    """-> (features, masks). features: list of (name, kind, compact value
    array, mask key); masks: key -> bool array (None = all rows valid)."""
    imv = d.im_joined.to_numpy(bool)
    masks = {"all": None, "im": imv}
    F = []

    def b(name, x, mk="all"):
        F.append((name, "binary", np.asarray(x, dtype=bool), mk))

    def m(name, x, mk="all"):
        F.append((name, "mean", np.asarray(x, dtype=np.float32), mk))

    for c in ["contracted", "line_initial", "sent_initial", "after_and",
              "after_wh", "wh_inverted", "inverted", "inverted_question",
              "tag_question", "line_ends_q", "r1_punct_eol", "turn_marker"]:
        b(c, d[c].to_numpy())
    c0 = d.char0.to_numpy(bool)
    b("char0", c0)
    b("char0_glued", c0 & d.prev_line_end.isin(["alnum", "punct"]).to_numpy())
    b("is_title", d.is_title.fillna(0).to_numpy() > 0, "im")
    b("passive", (d.dep == "nsubj:pass").to_numpy(), "im")
    hi = d.head_i.to_numpy(np.float64)
    ti = d.token_i.to_numpy(np.float64)
    b("head_right_adjacent", hi == ti + 1, "im")
    ag = d.agree_set_size.to_numpy(np.float64)
    masks["agree"] = ~np.isnan(ag)
    b("agree_available", masks["agree"])
    b("agree_eq1", ag == 1, "agree")
    b("agree_le3", ag <= 3, "agree")
    b("agree_eq7", ag == 7, "agree")
    m("agree_set_size", np.nan_to_num(ag), "agree")
    al = d.agree_set_size_l1.to_numpy(np.float64)
    masks["agree_l1"] = ~np.isnan(al)
    m("agree_set_size_l1", np.nan_to_num(al), "agree_l1")
    dup = d.line_dup_count.to_numpy(np.float64)
    b("line_dup", dup > 1)
    b("line_dup10", dup >= 10)
    m("log2_line_dup_count", np.log2(np.maximum(dup, 1)))
    ll = d.line_len_tokens.to_numpy(np.float64)
    b("short_line_le4", ll <= 4)
    m("line_len_tokens", ll)
    m("tokens_before", d.tokens_before.to_numpy(np.float64))
    ts = d.tokens_since_doc_start.to_numpy(np.float64)
    masks["doc"] = ~np.isnan(ts)
    b("left_ctx_lt50", np.nan_to_num(ts, nan=1e9) < 50, "doc")
    m("log10_tokens_since_doc_start", np.log10(1 + np.nan_to_num(ts)), "doc")
    for c in ["speaker_change", "childes_is_child"]:
        x = d[c].to_numpy(np.float64)
        masks[c] = ~np.isnan(x)
        b(c, np.nan_to_num(x) > 0, c)
    x = d.age_months.to_numpy(np.float64)
    masks["age"] = ~np.isnan(x)
    m("age_months", np.nan_to_num(x), "age")
    m("head_offset", np.nan_to_num(hi - ti), "im")
    for c in ["s_robbi", "s_bert", "s_gpt2m"]:
        m(c, d[c].to_numpy(np.float64))
    for g in C.GENRES:
        F.append((f"genre_{g}", "composition", (d.genre == g).to_numpy(), "all"))
    for f in FORMS:
        F.append((f"form_{f}", "composition", (d.form == f).to_numpy(), "all"))
    return F, masks


def boot_weights(cl_genre: np.ndarray, B: int, seed: int) -> np.ndarray:
    """(C, B+1) cluster weights; column 0 = point estimate (all ones).
    Clusters are resampled with replacement within genre."""
    rng = np.random.default_rng(seed)
    nC = len(cl_genre)
    W = np.ones((nC, B + 1), dtype=np.float64)
    groups = [np.flatnonzero(cl_genre == g) for g in np.unique(cl_genre)]
    for b in range(1, B + 1):
        for idx in groups:
            draw = rng.integers(0, len(idx), len(idx))
            W[idx, b] = np.bincount(draw, minlength=len(idx))
    return W


def accumulate(rows_by_genre, F, masks, dec, strat, clus, W, nD, nS):
    """Sum over genres of M_g @ W, where M_g holds per-(cluster, decile,
    stratum) totals. Returns (n, valid[mask], sum[feature]) arrays shaped
    (nD, nS, B+1)."""
    nK = nD * nS
    mkeys = [k for k in masks if masks[k] is not None]
    nblk = 1 + len(mkeys) + len(F)
    R = np.zeros((nblk * nK, W.shape[1]))
    for idx in rows_by_genre:
        if len(idx) == 0:
            continue
        cell = dec[idx] * nS + strat[idx]
        key = clus[idx] * nK + cell
        uk, inv = np.unique(key, return_inverse=True)
        g_cl, g_cell = uk // nK, uk % nK
        ng = len(uk)
        blocks = [np.bincount(inv, minlength=ng).astype(np.float64)]
        for k in mkeys:
            blocks.append(np.bincount(inv, weights=masks[k][idx], minlength=ng))
        for name, kind, val, mk in F:
            v = val[idx].astype(np.float64)
            if masks[mk] is not None:
                v = np.where(masks[mk][idx], v, 0.0)
            blocks.append(np.bincount(inv, weights=v, minlength=ng))
        rr, cc, vv = [], [], []
        for bi, v in enumerate(blocks):
            nz = np.flatnonzero(v)
            rr.append((bi * nK + g_cell[nz]).astype(np.int32))
            cc.append(g_cl[nz].astype(np.int32))
            vv.append(v[nz])
        M = sp.csr_matrix((np.concatenate(vv),
                           (np.concatenate(rr), np.concatenate(cc))),
                          shape=(nblk * nK, W.shape[0]))
        R += M @ W
        del M, rr, cc, vv, blocks
    R = R.reshape(nblk, nD, nS, W.shape[1])
    N = R[0]
    V = {"all": N}
    for i, k in enumerate(mkeys):
        V[k] = R[1 + i]
    S = {F[i][0]: R[1 + len(mkeys) + i] for i in range(len(F))}
    return N, V, S


def summarize(F, N, V, S, w_s, nD):
    out = []
    for name, kind, _, mk in F:
        Vf, Sf = V[mk], S[name]
        for d_ in range(nD):
            v, s = Vf[d_], Sf[d_]                  # (nS, B+1)
            vt = v.sum(0)
            raw = np.divide(s.sum(0), vt, out=np.full(vt.shape, np.nan),
                            where=vt > 0)
            row = {"feature": name, "kind": kind, "decile": d_,
                   "n": int(N[d_, :, 0].sum()), "n_valid": int(vt[0]),
                   "raw": raw[0], "raw_lo": np.nanpercentile(raw[1:], 2.5),
                   "raw_hi": np.nanpercentile(raw[1:], 97.5)}
            if kind != "composition":
                ok = v[:, 0] >= MIN_CELL                    # strata used
                mean = np.divide(s, v, out=np.zeros_like(s), where=v > 0)
                wv = w_s[:, None] * (ok[:, None] & (v > 0))
                wt = wv.sum(0)
                std = np.divide((wv * mean).sum(0), wt,
                                out=np.full(wt.shape, np.nan), where=wt > 0)
                row.update(std=std[0], std_lo=np.nanpercentile(std[1:], 2.5),
                           std_hi=np.nanpercentile(std[1:], 97.5),
                           std_cov=float(w_s[ok].sum()))
            else:
                row.update(std=np.nan, std_lo=np.nan, std_hi=np.nan,
                           std_cov=np.nan)
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()
    out = C.default_paths(a.data_root)["out"]
    T0 = time.time()
    d = pd.read_parquet(out / "population_features.parquet", columns=LOAD_COLS)
    n = len(d)
    g_code = pd.Categorical(d.genre, categories=C.GENRES).codes.astype(np.int64)
    f_code = pd.Categorical(d.form, categories=FORMS).codes.astype(np.int64)
    assert (g_code >= 0).all() and (f_code >= 0).all()
    nS = len(C.GENRES) * len(FORMS)
    strat = g_code * len(FORMS) + f_code
    w_s = np.bincount(strat, minlength=nS) / n
    cl_codes, cl_uniq = pd.factorize(d.cluster_id, sort=True)
    cl_genre = np.array([u.split("|", 1)[0] for u in cl_uniq])
    W = boot_weights(cl_genre, a.B, a.seed)
    print(f"loaded {n:,} rows, {len(cl_uniq):,} clusters, B={a.B} "
          f"({time.time() - T0:.0f}s)", flush=True)
    F, masks = build_features(d)
    contracted = d.contracted.to_numpy(bool)
    clus = cl_codes.astype(np.int64)
    rows = []
    for scheme, dcol in C.RATERS.items():
        dec = d[dcol].to_numpy(np.int64)
        for split, mask in [("all", None), ("contracted", contracted),
                            ("uncontracted", ~contracted)]:
            t = time.time()
            rbg = [np.flatnonzero((g_code == gi) if mask is None
                                  else ((g_code == gi) & mask))
                   for gi in range(len(C.GENRES))]
            N, V, S = accumulate(rbg, F, masks, dec, strat, clus, W, 10, nS)
            r = summarize(F, N, V, S, w_s, 10)
            for x in r:
                x.update(scheme=scheme, split=split)
            rows += r
            print(f"{scheme}/{split}: {time.time() - t:.0f}s", flush=True)
    res = pd.DataFrame(rows)[["scheme", "split", "feature", "kind", "decile",
                              "n", "n_valid", "raw", "raw_lo", "raw_hi", "std",
                              "std_lo", "std_hi", "std_cov"]]
    res.to_csv(out / "decile_profiles.csv", index=False, float_format="%.6g")
    base = res[(res.scheme == "robbi") & (res.split == "all")]
    wide = base.pivot(index="feature", columns="decile", values="raw").add_prefix("raw_d")
    wide = wide.join(base.pivot(index="feature", columns="decile",
                                values="std").add_prefix("std_d"))
    order = list(dict.fromkeys(base.feature))
    wide.loc[order].to_csv(out / "decile_profiles_wide.csv", float_format="%.4g")
    meta = {"B": a.B, "seed": a.seed, "min_cell": MIN_CELL,
            "n_clusters": int(len(cl_uniq)),
            "clusters_per_genre": {g: int((cl_genre == g).sum()) for g in C.GENRES},
            "strata_weights": {f"{C.GENRES[s // len(FORMS)]}:{FORMS[s % len(FORMS)]}":
                               round(float(w_s[s]), 6) for s in range(nS) if w_s[s] > 0},
            "seconds": round(time.time() - T0, 1)}
    (out / "decile_profiles.meta.json").write_text(json.dumps(meta, indent=2))
    print("done", meta["seconds"], "s")


if __name__ == "__main__":
    main()
