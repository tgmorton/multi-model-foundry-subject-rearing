"""Step 3: slot-frame keyness per decile (robbi), Monroe et al. (2008)
informative-Dirichlet log-odds with the population as the prior.

Strata: genre x person group (I/you [incl. thou/ye], we, he/she/they, it;
'one' is excluded). Within a stratum, decile d is compared with all other
deciles ("rest"). Two feature types:
  frame  "l2 l1 _ r1 r2"  (content tokens, normalized, <s>/</s> padded)
  token  position-tagged single tokens L2=x, L1=x, R1=x, R2=x
         (each instance contributes four)
Prior: alpha_w = K * pop_w, where pop_w is w's count in the stratum's
population (all deciles) and K = --prior-scale (default 0.01, i.e. the prior
carries 1% of the stratum's mass). Then
  delta_w = log[(y_dw + a_w) / (n_d + a0 - y_dw - a_w)]
          - log[(y_rw + a_w) / (n_r + a0 - y_rw - a_w)]
  var     = 1/(y_dw + a_w) + 1/(y_rw + a_w),   z = delta / sqrt(var)
The top --top features by z (with y_dw >= --min-count) per stratum x decile
are written with three deduplicated example lines each (seeded, stable
order).

Outputs: slot_frame_keyness.csv, slot_frame_keyness.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survey_common as C  # noqa: E402

POS = ["L2", "L1", "R1", "R2"]
POS_COL = {"L2": "l2", "L1": "l1", "R1": "r1", "R2": "r2"}


def kwic(line: str, cs: int, a: int, b: int, width: int = 70,
         slot: str | None = None) -> str:
    tag = line[:cs].strip()
    left, right = line[cs:a], line[b:]
    if len(left) > width:
        left = "…" + left[-width:]
    if len(right) > width:
        right = right[:width] + "…"
    mid = f"[{line[a:b]}]" if slot is None else slot
    return (f"{tag} " if tag else "") + f"{left}{mid}{right}"


def monroe(cnt: np.ndarray, K: float):
    """cnt: (n_feat, 10) counts per decile. Returns z, delta (n_feat, 10)."""
    pop = cnt.sum(1, keepdims=True).astype(np.float64)
    nd = cnt.sum(0, keepdims=True).astype(np.float64)
    N = nd.sum()
    a = K * pop
    a0 = K * N
    y = cnt.astype(np.float64)
    yr = pop - y
    nr = N - nd
    delta = (np.log((y + a) / (nd + a0 - y - a))
             - np.log((yr + a) / (nr + a0 - yr - a)))
    var = 1.0 / (y + a) + 1.0 / (yr + a)
    return delta / np.sqrt(var), delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--prior-scale", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--md-tokens", type=int, default=8)
    ap.add_argument("--md-frames", type=int, default=5)
    a = ap.parse_args()
    P = C.default_paths(a.data_root)
    out = P["out"]
    T0 = time.time()
    d = pd.read_parquet(out / "population_features.parquet", columns=[
        "genre", "line_idx", "token_i", "form", "d_robbi", "l2", "l1", "r1", "r2",
        "line_norm_hash", "char_start", "char_end", "content_start", "tok_ok"])
    d["pg"] = d.form.map(C.PERSON_GROUP)
    d = d[d.pg.isin(C.KEYNESS_GROUPS) & d.tok_ok].reset_index(drop=True)
    strata = [(g, p) for g in C.GENRES for p in C.KEYNESS_GROUPS]
    st_index = {s: i for i, s in enumerate(strata)}
    d["st"] = [st_index[(g, p)] for g, p in zip(d.genre, d.pg)]
    st = d.st.to_numpy(np.int64)
    dec = d.d_robbi.to_numpy(np.int64)
    # vocab over all four positions
    vocab, codes = np.unique(np.concatenate([d[c].to_numpy(object)
                                             for c in ["l2", "l1", "r1", "r2"]]),
                             return_inverse=True)
    V = len(vocab)
    n = len(d)
    pc = {p: codes[i * n:(i + 1) * n].astype(np.int64) for i, p in enumerate(POS)}
    frame_id = (pd.DataFrame(pc).groupby(POS, sort=True).ngroup()
                .to_numpy(np.int64))
    nF = int(frame_id.max()) + 1
    # frame strings for ids
    fr_first = pd.Series(np.arange(n)).groupby(frame_id).first().to_numpy()
    frame_str = np.array([f"{vocab[pc['L2'][i]]} {vocab[pc['L1'][i]]} _ "
                          f"{vocab[pc['R1'][i]]} {vocab[pc['R2'][i]]}"
                          for i in fr_first], dtype=object)
    print(f"{n:,} instances, vocab {V:,}, frames {nF:,} "
          f"({time.time() - T0:.0f}s)", flush=True)

    def counts(feat_ids, st_ids, decs, nfeat):
        key = (st_ids * nfeat + feat_ids) * 10 + decs
        uk, c = np.unique(key, return_counts=True)
        return uk // 10 // nfeat, uk // 10 % nfeat, uk % 10, c

    tok_feat = np.concatenate([i * V + pc[p] for i, p in enumerate(POS)])
    tok_st = np.tile(st, 4)
    tok_dec = np.tile(dec, 4)
    res = []
    for kind, (s_, f_, d_, c_), labeler in [
            ("frame", counts(frame_id, st, dec, nF), lambda f: frame_str[f]),
            ("token", counts(tok_feat, tok_st, tok_dec, 4 * V),
             lambda f: f"{POS[f // V]}={vocab[f % V]}")]:
        for si in range(len(strata)):
            m = s_ == si
            if not m.any():
                continue
            fu, finv = np.unique(f_[m], return_inverse=True)
            cnt = np.zeros((len(fu), 10), dtype=np.int64)
            np.add.at(cnt, (finv, d_[m]), c_[m])
            z, delta = monroe(cnt, a.prior_scale)
            pop = cnt.sum(1)
            nd = cnt.sum(0)
            for dd in range(10):
                ok = cnt[:, dd] >= a.min_count
                if not ok.any():
                    continue
                idx = np.flatnonzero(ok)
                top = idx[np.argsort(-z[idx, dd], kind="stable")[:a.top]]
                for r, fi in enumerate(top):
                    res.append({"genre": strata[si][0], "person_group": strata[si][1],
                                "decile": dd, "kind": kind, "rank": r + 1,
                                "feature": labeler(int(fu[fi])),
                                "feat_id": int(fu[fi]), "st": si,
                                "z": float(z[fi, dd]), "delta": float(delta[fi, dd]),
                                "count_decile": int(cnt[fi, dd]),
                                "count_pop": int(pop[fi]),
                                "decile_share": float(cnt[fi, dd] / pop[fi]),
                                "stratum_decile_n": int(nd[dd]),
                                "stratum_n": int(nd.sum())})
    R = pd.DataFrame(res)
    print(f"keyness rows {len(R):,} ({time.time() - T0:.0f}s)", flush=True)

    # ---- examples: 3 per listed item, deduplicated on normalized line
    hkey = C.stable_hash([f"{a.seed}|{g}|{l}|{t}" for g, l, t in
                          zip(d.genre, d.line_idx, d.token_i)])
    base = pd.DataFrame({"row": np.arange(n), "st": st, "decile": dec,
                         "h": hkey, "lh": d.line_norm_hash.to_numpy()})
    cands = []
    fr = R[R.kind == "frame"][["st", "decile", "feat_id"]]
    x = base.assign(feat_id=frame_id).merge(fr, on=["st", "decile", "feat_id"])
    cands.append(x.assign(kind="frame"))
    tk = R[R.kind == "token"][["st", "decile", "feat_id"]]
    for i, p in enumerate(POS):
        x = base.assign(feat_id=i * V + pc[p]).merge(tk, on=["st", "decile", "feat_id"])
        cands.append(x.assign(kind="token"))
    cands = (pd.concat(cands, ignore_index=True)
             .sort_values(["kind", "st", "decile", "feat_id", "h"], kind="stable")
             .drop_duplicates(["kind", "st", "decile", "feat_id", "lh"]))
    cands["k"] = cands.groupby(["kind", "st", "decile", "feat_id"]).cumcount()
    cands = cands[cands.k < 3]
    need = d.loc[cands.row.unique(), ["genre", "line_idx", "char_start",
                                      "char_end", "content_start"]]
    text = {}
    for g in C.GENRES:
        sub = need[need.genre == g]
        if sub.empty:
            continue
        lines = C.read_lines(P["raw"], g)
        for r in sub.itertuples():
            text[r.Index] = kwic(lines[r.line_idx], r.content_start,
                                 r.char_start, r.char_end)
        del lines
    cands["ex"] = cands.row.map(text)
    ex = cands.pivot_table(index=["kind", "st", "decile", "feat_id"], columns="k",
                           values="ex", aggfunc="first")
    ex.columns = [f"ex{int(c) + 1}" for c in ex.columns]
    R = R.merge(ex.reset_index(), on=["kind", "st", "decile", "feat_id"], how="left")
    R = R.sort_values(["kind", "st", "decile", "rank"]).drop(columns=["st", "feat_id"])
    R.to_csv(out / "slot_frame_keyness.csv", index=False, float_format="%.4g")

    # ---- markdown summary
    L = ["# Slot-frame keyness by robbi decile", "",
         "Monroe et al. (2008) informative-Dirichlet log-odds, decile vs rest "
         "within genre x person-group strata; prior = stratum population counts "
         f"x {a.prior_scale}; listed items need >= {a.min_count} hits in the "
         "decile. Decile 0 = most recoverable (robbi), 9 = least. `<s>` / `</s>` "
         "= line edge (after any speaker tag). z-scores in parentheses. Full "
         "top-30 lists with three examples each: `slot_frame_keyness.csv`.", ""]
    # cross-genre consensus
    L += ["## Cross-genre consensus (tokens in the top 30 of >= 4 of 6 genres)", ""]
    tok = R[R.kind == "token"]
    for pg in C.KEYNESS_GROUPS:
        L += [f"### {pg}", "", "| decile | tokens (# genres, median z) |", "|---|---|"]
        for dd in range(10):
            t = tok[(tok.person_group == pg) & (tok.decile == dd)]
            agg = (t.groupby("feature").agg(ng=("genre", "nunique"), mz=("z", "median"))
                   .query("ng >= 4").sort_values(["ng", "mz"], ascending=False).head(10))
            cells = ", ".join(f"`{f}` ({int(r.ng)}, {r.mz:.0f})"
                              for f, r in agg.iterrows())
            L.append(f"| {dd} | {cells or '-'} |")
        L.append("")
    L += ["## Per stratum", ""]
    for (g, pg) in strata:
        s = R[(R.genre == g) & (R.person_group == pg)]
        if s.empty:
            continue
        L += [f"### {g} / {pg}  (n = {int(s.stratum_n.iloc[0]):,})", "",
              "| decile | top tokens | top frames |", "|---|---|---|"]
        for dd in range(10):
            t = s[(s.decile == dd) & (s.kind == "token")].head(a.md_tokens)
            f = s[(s.decile == dd) & (s.kind == "frame")].head(a.md_frames)
            tc = ", ".join(f"`{x.feature}` ({x.z:.0f})" for x in t.itertuples())
            fc = ", ".join(f"`{x.feature}` ({x.z:.0f})" for x in f.itertuples())
            L.append(f"| {dd} | {tc or '-'} | {fc or '-'} |")
        L.append("")
        for dd in (0, 9):
            f = s[(s.decile == dd) & (s.kind == "frame")].head(3)
            for x in f.itertuples():
                exs = [e for e in (x.ex1, x.ex2) if isinstance(e, str)]
                for e in exs:
                    L.append(f"- d{dd} `{x.feature}`: {e.replace('|', '/')}")
        L.append("")
    (out / "slot_frame_keyness.md").write_text("\n".join(L))
    print(f"done ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    main()
