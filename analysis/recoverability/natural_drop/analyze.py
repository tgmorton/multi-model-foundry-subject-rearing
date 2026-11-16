#!/usr/bin/env python3
"""Join the rater's scores to the natural-drop sites and derive the measures the report uses.

Per site (a dropped subject, or an overt personal-pronoun subject of the same speakers):

  mass        probability the rater gives to a personal pronoun (I you he she it we they) at the slot
  best, p_best  the rater's best pronoun guess and its probability
  surp_best   -ln p_best: the surprisal the dropped pronoun would have if it was the rater's best guess.
              A floor: whatever pronoun was meant, its surprisal is at least this.
  dec_best    surp_best on the frozen decile thresholds of the corpus ranking (0 = most recoverable)
  surp_true, dec_true   overt sites only: surprisal and decile of the pronoun that was actually there

`expected deciles`: for overt sites the pronoun is known, so they give P(true decile | dec_best). Applying
that table to the dropped sites gives the decile distribution the dropped pronouns would have if a dropped
pronoun were as likely to be the rater's best guess as an overt pronoun with the same best-guess probability.

Run with the project venv (needs spaCy's tokenizer for the position of overt pronouns):

    .venv/bin/python analysis/recoverability/natural_drop/analyze.py --dir <site dir> [--scores scores.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FORMS = ["i", "you", "he", "she", "it", "we", "they"]
THRESH = [0.0029813863802701235, 0.008927312679588795, 0.02135789766907692, 0.04446164146065712, 0.08588272333145142,
          0.16255170106887817, 0.31855273246765137, 0.6843868494033813, 1.7035140991210938]
AGE_BANDS = [(0, 24, "under 2"), (24, 36, "2"), (36, 48, "3"), (48, 60, "4"), (60, 999, "5 and over")]


def decile(surp):
    return np.searchsorted(THRESH, np.asarray(surp, float), side="left").astype(int)


def age_band(m):
    for lo, hi, name in AGE_BANDS:
        if lo <= m < hi:
            return name
    return "unknown"


def boot_ci(x, n=1000, seed=0, f=np.mean):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = f(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--scores", default="scores.parquet")
    a = ap.parse_args()
    s = pd.read_parquet(a.dir / a.scores)
    P = s[[f"p_{f}" for f in FORMS]].to_numpy(float)
    s["mass"] = P.sum(1)
    s["best"] = np.array(FORMS)[P.argmax(1)]
    s["p_best"] = P.max(1)
    s["surp_best"] = -np.log(np.clip(s.p_best, 1e-12, 1))
    s["dec_best"] = decile(s.surp_best)
    q = P / P.sum(1, keepdims=True)
    s["entropy_pron"] = -(q * np.log(np.clip(q, 1e-12, 1))).sum(1)
    s["top_is_pronoun"] = s.top1.str.lower().isin(FORMS)

    null = pd.read_parquet(a.dir / "null_sites.parquet").drop(columns=["kind"])
    n = s[s.kind == "null"].merge(null, on=["sentence_id", "verb_idx", "line_idx"], how="left")
    # inverted order: "here comes ...", "where is ...": the subject follows the verb, nothing was dropped
    prev = n.before.fillna("").str.split().str[-1].fillna("").str.lower()
    inverted = prev.isin(["here", "there", "where", "why", "how", "what", "who", "when", "which"])
    print(f"null sites after here/there or a question word (inverted order), left out: {int(inverted.sum())}")
    n = n[~inverted].reset_index(drop=True)
    overt = pd.read_parquet(a.dir / "overt_sites.parquet").drop(columns=["piece_in_line", "n_pieces", "kind"])
    o = s[s.kind == "overt"].merge(overt, on=["line_idx", "token_i"], how="left")
    assert n.role.notna().all() and o.role.notna().all()
    other = int((~o.form.isin(FORMS)).sum())
    o = o[o.form.isin(FORMS)].reset_index(drop=True)      # the ranking also holds a few "one", "thou", "ye"
    print(f"overt sites with a form outside the seven personal pronouns, left out: {other}")

    import spacy
    nlp = spacy.blank("en")
    first = []
    for ln, ti in zip(o.line_text, o.token_i):
        c = ln.find("\t") + 1 if ln.startswith("*") else 0
        doc = nlp.make_doc(ln)
        first.append(ln[c:doc[int(ti)].idx].strip() == "")
    o["first_in_utterance"] = first
    o["surp_true"] = -np.log(np.clip([o[f"p_{f}"].iat[k] for k, f in enumerate(o.form)], 1e-12, 1))
    o["dec_true"] = decile(o.surp_true)
    o["surp_exact"] = -o.logprob_true          # the exact piece, as the ranking scored it
    o["best_is_true"] = o.best == o.form
    o["verb_class"] = "overt"
    print(f"overt check against the stored ranking: corr {np.corrcoef(-o.logprob_true, o.s_robbi)[0, 1]:.4f}; "
          f"median |diff| {np.median(np.abs(-o.logprob_true - o.s_robbi)):.4f} nats; same decile {np.mean(decile(-o.logprob_true) == o.d_robbi):.3f}")

    keep = ["kind", "role", "speaker", "child_age_months", "verb_class", "first_in_utterance", "line_idx", "line_text", "mass",
            "best", "p_best", "surp_best", "dec_best", "entropy_pron", "entropy", "top1", "top1_p", "top_is_pronoun", "top_pieces",
            "x_that", "x_there", "x_this", "x_what", "x_who"] + [f"p_{f}" for f in FORMS]
    n2 = n[keep + ["sentence_id", "verb_idx", "verb_lemma", "first_verbal", "clause_type", "scored_line", "before", "after"]].copy()
    o2 = o[keep + ["token_i", "form", "surp_true", "dec_true", "surp_exact", "best_is_true", "s_robbi", "d_robbi"]].copy()
    d = pd.concat([n2, o2], ignore_index=True)
    d["age_band"] = d.child_age_months.map(lambda m: age_band(m) if pd.notna(m) else "unknown")
    d.to_parquet(a.dir / "sites_scored.parquet", index=False)

    # P(true decile | best-guess decile), from overt sites, per speaker group and position
    out = []
    for role in ("child", "adult"):
        ov = d[(d.kind == "overt") & (d.role == role)]
        T = pd.crosstab(ov.dec_best, ov.d_robbi, normalize="index").reindex(index=range(10), columns=range(10), fill_value=0.0)
        for vc, g in d[(d.kind == "null") & (d.role == role)].groupby("verb_class"):
            w = g.dec_best.value_counts(normalize=True).reindex(range(10), fill_value=0.0)
            exp = (w.to_numpy()[:, None] * T.to_numpy()).sum(0)
            for dc in range(10):
                out.append(dict(role=role, verb_class=vc, decile=dc, share_best_guess=w[dc], share_expected=exp[dc], n=len(g)))
        w = ov.dec_best.value_counts(normalize=True).reindex(range(10), fill_value=0.0)
        t = ov.d_robbi.value_counts(normalize=True).reindex(range(10), fill_value=0.0)
        for dc in range(10):
            out.append(dict(role=role, verb_class="overt", decile=dc, share_best_guess=w[dc], share_expected=t[dc], n=len(ov)))
    pd.DataFrame(out).to_csv(a.dir / "decile_distributions.csv", index=False)
    pd.set_option("display.width", 220)
    g = d.groupby(["role", "kind", "verb_class"]).agg(n=("mass", "size"), first=("first_in_utterance", "mean"), mass=("mass", "median"),
                                                      p_best=("p_best", "median"), p_best_ge_90=("p_best", lambda x: (x >= 0.9).mean()),
                                                      dec_best=("dec_best", "mean"), top_is_pronoun=("top_is_pronoun", "mean"))
    print(g.round(3).to_string())
    print("wrote", a.dir / "sites_scored.parquet")


if __name__ == "__main__":
    main()
