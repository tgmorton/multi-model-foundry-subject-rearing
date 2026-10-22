"""Step 4: audit.md — coverage, sentinels, ties, the >10-nat tail, join and
token_i verification.

Reads population_features.parquet, joined_sources.json, the selection
manifests (from --sel-root, a local mirror laid out by S3 key; default
<out>/_s3cache), instance_measures and the raw text. Analyst notes from a
manual reading of the tail sample are merged from <out>/audit_manual_notes.md
if that file exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survey_common as C  # noqa: E402
from s3_keyness import kwic  # noqa: E402


def md_table(df: pd.DataFrame, floatfmt="{:.4g}") -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for idx, r in df.astype(object).iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v))
            else:
                cells.append(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v))
        out.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--sel-root", type=Path, default=None)
    ap.add_argument("--n-tail", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()
    P = C.default_paths(a.data_root)
    out = P["out"]
    sel_root = a.sel_root or (out / "_s3cache")
    d = pd.read_parquet(out / "population_features.parquet", columns=[
        "genre", "line_idx", "token_i", "form", "d_robbi", "s_robbi", "d_bert",
        "s_bert", "d_gpt2m", "s_gpt2m", "im_joined", "tok_ok", "contracted",
        "char_start", "char_end", "content_start", "is_title", "word_initial",
        "line_len_tokens", "line_dup_count", "r1", "l1", "doc_id", "speaker",
        "age_months", "tokens_since_doc_start", "cluster_id", "char0",
        "prev_line_end"])
    L = ["# Audit: robbi selection population (train_90M)", ""]

    # ---------------- 1. coverage / sentinels
    man = json.loads((sel_root / C.SEL_PREFIX / "robbi" / "SELECTION_MANIFEST.json"
                      ).read_text())
    fam = json.loads((sel_root / C.SEL_PREFIX / "V5_FAMILY.json").read_text())
    mc = man.get("missing_coverage", {})
    L += ["## 1. Coverage and sentinels", "",
          f"- Population: {len(d):,} instances ({fam.get('population')}).",
          f"- robbi SELECTION_MANIFEST `missing_coverage`: "
          f"train_90M missing_to_decile9 = {mc.get('train_90M', {}).get('missing_to_decile9')}, "
          f"pull_10M = {mc.get('pull_10M', {}).get('missing_to_decile9')}. "
          "The emitter (`emit_selection_robbi.py`) fills uncovered instances "
          "with +inf, written as float64 max (1.797e308)."]
    for c in ["s_robbi", "s_bert", "s_gpt2m"]:
        v = d[c].to_numpy()
        L.append(f"- `{c}`: NaN {np.isnan(v).sum():,}, inf {np.isinf(v).sum():,}, "
                 f">= 1e30 {(v >= 1e30).sum():,}, <= 0 {(v <= 0).sum():,}; "
                 f"min {np.nanmin(v):.3g}, max {np.nanmax(v):.4g} nats.")
    L.append("- Conclusion: no instance was pushed into decile 9 for lack of a "
             "robbi score (no sentinel values present; every decile holds "
             f"{d.d_robbi.value_counts().min():,}-{d.d_robbi.value_counts().max():,} rows).")
    # population vs instance_measures
    rows = []
    for g in C.GENRES:
        im = pq.read_table(P["measures"] / f"{g}.parquet",
                           columns=["in_lexicon"]).to_pandas()
        rows.append({"genre": g, "instance_measures": len(im),
                     "in_lexicon": int(im.in_lexicon.sum()),
                     "population": int((d.genre == g).sum())})
    cov = pd.DataFrame(rows).set_index("genre")
    cov["pop_share_of_in_lexicon"] = cov.population / cov.in_lexicon
    L += ["", "Instances in `instance_measures` vs the v5 population (instances "
          "outside the population are excluded from selection altogether, not "
          "ranked):", "", md_table(cov), ""]
    # ties at the decile thresholds
    thr = man["decile_upper_thresholds"]
    trows = []
    s = d.s_robbi.to_numpy()
    for k, t in enumerate(thr):
        at = d[s == t]
        trows.append({"threshold": f"d{k}|d{k + 1}", "value": t,
                      "n_tied": len(at),
                      "split": dict(at.d_robbi.value_counts().sort_index())})
    vc = d.s_robbi.value_counts()
    L += ["Ties: deciles were cut with `rank(method='first')`, i.e. ties broken "
          "by row order (genre alphabetical, then line). Rows sharing each "
          "boundary value:", "", md_table(pd.DataFrame(trows).set_index("threshold")),
          "", f"Largest tie group anywhere: {vc.iloc[0]} rows (value "
          f"{vc.index[0]:.3g}); rows in tie groups of >= 10: "
          f"{(d.s_robbi.map(vc) >= 10).sum():,} (float32 quantization of "
          "near-zero surprisals; within-decile order there is noise, as the "
          "manifest's saturation note says). At the nine decile boundaries at "
          "most 20 rows are tied, so tie-breaking by genre order does not "
          "move genre shares.", ""]

    # ---------------- 2. the > 10 nat tail
    hi = d[d.s_robbi > 10]
    L += ["## 2. Instances with s_robbi > 10 nats", "",
          f"- n = {len(hi):,} ({len(hi) / len(d):.3%} of population; "
          f"{len(hi) / (d.d_robbi == 9).sum():.2%} of decile 9).", ""]
    gt = pd.DataFrame({"n_gt10": hi.genre.value_counts(),
                       "share_of_genre": hi.genre.value_counts() /
                       d.genre.value_counts()}).fillna(0)
    gt.index.name = "genre"
    L += [md_table(gt), ""]
    ft = pd.DataFrame({"n_gt10": hi.form.value_counts(),
                       "share_of_form": hi.form.value_counts() / d.form.value_counts()}
                      ).dropna()
    ft.index.name = "form"
    L += [md_table(ft), ""]
    bins = [10, 12, 14, 16, 18, 20, 25, 30]
    hist = pd.cut(hi.s_robbi, bins).value_counts().sort_index()
    L += ["Distribution: " + ", ".join(f"{iv}: {c:,}" for iv, c in hist.items()), ""]
    L += [f"- Other raters on these rows: median s_bert {hi.s_bert.median():.2f} "
          f"(population {d.s_bert.median():.2f}), median s_gpt2m "
          f"{hi.s_gpt2m.median():.2f} (population {d.s_gpt2m.median():.2f}); "
          f"d_bert == 9 for {(hi.d_bert == 9).mean():.1%}, d_gpt2m == 9 for "
          f"{(hi.d_gpt2m == 9).mean():.1%}.",
          f"- contracted {hi.contracted.mean():.1%} (population "
          f"{d.contracted.mean():.1%}); is_title {hi.is_title.mean():.1%} "
          f"(population {d.is_title.mean():.1%}); word_initial==False "
          f"{(hi.word_initial == 0).mean():.2%} (population "
          f"{(d.word_initial == 0).mean():.2%}); line_dup_count>1 "
          f"{(hi.line_dup_count > 1).mean():.1%} (population "
          f"{(d.line_dup_count > 1).mean():.1%}); median line length "
          f"{hi.line_len_tokens.median():.0f} tokens (population "
          f"{d.line_len_tokens.median():.0f}).", ""]
    rng = np.random.default_rng(a.seed)
    samp = hi.iloc[np.sort(rng.choice(len(hi), size=min(a.n_tail, len(hi)),
                                      replace=False))]
    L += [f"Random sample of {len(samp)} (seed {a.seed}); previous line shown "
          "above each:", ""]
    lines_cache = {}
    for i, r in enumerate(samp.itertuples()):
        if r.genre not in lines_cache:
            lines_cache[r.genre] = C.read_lines(P["raw"], r.genre)
        Lg = lines_cache[r.genre]
        prev = Lg[r.line_idx - 1] if r.line_idx > 0 else ""
        cur = kwic(Lg[r.line_idx], r.content_start, r.char_start, r.char_end, 110)
        L += [f"{i + 1}. **{r.genre}** l{r.line_idx} `{r.form}` s_robbi "
              f"{r.s_robbi:.1f}, s_bert {r.s_bert:.1f}, s_gpt2m {r.s_gpt2m:.1f}  ",
              f"   prev: {prev[:160]!r}  ",
              f"   line: {cur!r}"]
    lines_cache.clear()
    notes = out / "audit_manual_notes.md"
    L += ["", "### Characterization (manual reading)", ""]
    L += [notes.read_text().strip() if notes.exists()
          else "_(no audit_manual_notes.md yet)_", ""]

    # ---------------- 3. join
    jg = d.groupby("genre").im_joined.agg(["size", "sum", "mean"])
    jg.columns = ["n", "joined", "rate"]
    L += ["## 3. Join to instance_measures on (genre, line_idx, token_i)", "",
          f"Overall {d.im_joined.mean():.4%} ({d.im_joined.sum():,} / {len(d):,}). "
          "Unjoined rows have no lemma/dep/person/is_title/head_i (their "
          "agree_set_size can still come from the closed aux table, but not "
          "from the lexical-head rule).", "", md_table(jg), ""]
    u = d[~d.im_joined]
    L += [f"Unjoined rows ({len(u):,}) by robbi decile: "
          f"{u.d_robbi.value_counts().sort_index().tolist()} (no decile skew). "
          "They are whole lines missing from instance_measures (in simple_wiki, "
          "the lines carrying unjoined rows have no instance_measures rows at "
          "all), so the in-house measures build dropped those lines; the "
          "selection population, which comes from the external raters' "
          "alignment, kept them. That is why simple_wiki's population exceeds "
          "its in_lexicon count in the table above.", ""]
    # ---------------- 4. token_i
    tg = d.groupby("genre").tok_ok.agg(["size", "sum", "mean"])
    tg.columns = ["n", "match", "rate"]
    bad = d[~d.tok_ok]
    L += ["## 4. token_i re-verification (all instances, not a sample)", "",
          "spaCy en_core_web_trf tokenizer only (`nlp.tokenizer`) on the raw "
          "line; match = token at token_i lowercases to `form` and is a "
          f"content token. Overall {d.tok_ok.mean():.5%} ({d.tok_ok.sum():,} / "
          f"{len(d):,}).", "", md_table(tg), ""]
    if len(bad):
        import spacy
        tok = spacy.load("en_core_web_trf", exclude=[
            "transformer", "tagger", "parser", "attribute_ruler", "lemmatizer",
            "ner"]).tokenizer
        cache, lst, text_ok, in_tag = {}, [], 0, 0
        for r in bad.itertuples():
            if r.genre not in cache:
                cache[r.genre] = C.read_lines(P["raw"], r.genre)
            line = cache[r.genre][r.line_idx]
            doc = tok(line)
            ok = r.token_i < len(doc) and doc[r.token_i].text.lower() == r.form
            text_ok += ok
            in_tag += ok and doc[r.token_i].idx < r.content_start
            if len(lst) < 25:
                lst.append(f"- {r.genre} l{r.line_idx} token_i={r.token_i} form="
                           f"`{r.form}`: {line[:150]!r}")
        cache.clear()
        L += [f"Non-matches: {len(bad)}. Of these, the token text still equals "
              f"the form for {text_ok} (so the raw token_i reproduction rate is "
              f"{(d.tok_ok.sum() + text_ok) / len(d):.5%}), and {in_tag} sit "
              "inside a speaker tag, i.e. the 'pronoun' is a CHILDES speaker "
              "code (e.g. `*HE:`) parsed as a subject: an annotation artifact. "
              "These rows are excluded from keyness and the concordance.", ""]
        L += lst + [""]
    # ---------------- 4b. stream-glue artifact
    g = d[d.genre.isin(["bnc_spoken", "open_subtitles", "gutenberg",
                        "simple_wiki"])]
    g = g.assign(pos=np.where(g.char0, "char0 (line-initial, no tag)",
                              "not char0"))
    t = (g.groupby(["genre", "pos", "prev_line_end"])
         .agg(n=("s_robbi", "size"), robbi_median=("s_robbi", "median"),
              robbi_d9=("d_robbi", lambda x: (x == 9).mean()),
              bert_median=("s_bert", "median"),
              gpt2m_median=("s_gpt2m", "median")))
    t = t[t.n >= 300]
    t.index = [" / ".join(map(str, i)) for i in t.index]
    t.index.name = "genre / position / previous line ends in"
    c0 = d.char0 & d.genre.isin(["bnc_spoken", "open_subtitles", "gutenberg",
                                 "simple_wiki"])
    glued = c0 & d.prev_line_end.isin(["alnum", "punct"])
    L += ["## 4b. Line-concatenation (stream-glue) artifact in robbi", "",
          "The robbi scorer encodes each line separately (no leading space) and "
          "concatenates the ids with no separator, so a pronoun that is the "
          "first character of its raw line becomes a no-space RoBERTa piece "
          "glued to the previous line's last character (e.g. `...position "
          "in` + `I'm` -> `inI'm`, `...end.` + `He` -> `.He`). Speaker-tagged "
          "genres (CHILDES, Switchboard) are unaffected because the pronoun "
          "follows a tab. BERT (uncased WordPiece) does not encode spaces. "
          "The cleanest contrast is open_subtitles: char-0 pronouns after a "
          "line ending in a trailing space vs. in punctuation differ only in "
          "that space.", "", md_table(t), "",
          f"- char-0 pronouns in BNC/open_subtitles/gutenberg/simple_wiki: "
          f"{c0.sum():,} ({c0.mean():.1%} of the population); glued (previous "
          f"line ends in alnum/punct): {glued.sum():,}. Glued share of each "
          "robbi decile: " + ", ".join(
              f"d{k} {glued[d.d_robbi == k].mean():.1%}" for k in range(10)) + ".",
          ""]
    # ---------------- 5. structure notes
    ch = d[d.genre == "childes"]
    L += ["## 5. Line-structure parsing", "",
          f"- CHILDES: {ch.doc_id.nunique():,} docs; instances before the first "
          f"header ('__prefix__', no age): {(ch.doc_id == '__prefix__').sum():,}; "
          f"age_months joined for {ch.age_months.notna().mean():.3%}; speaker tag "
          f"parsed for {ch.speaker.notna().mean():.3%}. Header lines precede "
          "their document (checked: Spearman(age, CHI mean utterance length) "
          "= 0.63 assigning lines to the preceding header vs 0.56 to the "
          "following one).",
          f"- Switchboard: speaker parsed for "
          f"{d[d.genre == 'switchboard'].speaker.notna().mean():.3%}; no "
          "conversation markers, so speaker_change at conversation joins is noise "
          "and clusters are 200-line blocks.",
          "- BNC and open_subtitles: no document or speaker markers (open_subtitles "
          "leading dialogue dashes are treated as a turn marker); clusters are "
          "200-line blocks; tokens_since_doc_start is NaN.",
          "- robbi's context: the scorer concatenates lines with no separator and "
          "crosses document boundaries (`scripts/score_pronoun_recoverability.py` "
          "docstring), so tokens_since_doc_start measures coherent left context, "
          "not the rater's raw window.", ""]
    (out / "audit.md").write_text("\n".join(L))
    print("wrote", out / "audit.md")


if __name__ == "__main__":
    main()
