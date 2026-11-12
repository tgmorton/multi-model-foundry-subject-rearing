#!/usr/bin/env python3
"""Draw one eval set per recoverability decile from the scored test-set candidates.

An item enters decile d when the rater places its pronoun there with the item as its whole
input (score_items.py) and the same pronoun in its full corpus surroundings
(score_corpus_instances.py) is within `--corpus-slack` deciles of d (tier 1). A cell that
tier 1 cannot fill is topped up with sentences whose corpus decile is further off (tier 2)
and then with synthetic variants that carry less of their real context (tier 3,
make_context_variants.py). Cells are phenomenon x
person; every decile gets the same number of items per cell.

Main-clause cells keep only pronouns that open their sentence or follow discourse material
("No, I ...", "And then he ..."). Within a cell the deciles are matched on source
(conversation / subtitles / written) and on where the pronoun stands (starts its corpus
line, starts a sentence inside a line, or neither): the composition is the one closest to the pooled
composition that every decile can supply, so no decile set differs from another in those
respects unless a cell runs short. A cell that cannot be filled from the matched strata is
topped up from whatever the decile has, and what is still missing is reported as the
shortfall to cover with synthetic variants.

`--position natural` matches the deciles on source only and draws at random within it, so
where the pronoun stands varies across deciles as it does in the corpus. `--pool` draws a
pronoun-free version from a pool built by make_context_variants.py --pronoun-free: every
item keeps the longest stretch of its real context that has no pronoun, and its decile is
the one the rater gives it with that context.

Outputs in --out-dir:
  availability.csv     usable items per cell and decile (and the matched capacity)
  sets.parquet         the drawn items with all labels and both scores
  shortfall.csv        cells x deciles that could not be filled
  stimuli/<category>.csv  the eval runner's schema (overt and null rows), plus decile columns

    python analysis/recoverability/eval_stimuli/sample_sets.py --dir <test_10M dir> --out-dir <dir>/sets/robbi_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_corpus_instances import ROBBI_THRESHOLDS  # noqa: E402

GENRE_GROUP = {"childes": "conversation", "bnc_spoken": "conversation", "switchboard": "conversation",
               "open_subtitles": "subtitles", "gutenberg": "written", "simple_wiki": "written"}
MARKER_POS = {"INTJ", "ADV", "CCONJ"}
MARKER_SCONJ = {"so", "cause", "'cause", "cos", "'cos", "like"}


def position(lead_n, lead_text, lead_pos) -> str:
    """Where a main-clause subject pronoun stands in its sentence.

    initial       nothing before it
    after_marker  only discourse material before it: up to four interjections, adverbs,
                  conjunctions and commas ("No, I ...", "And then he ...", "So we ...")
    other         anything else (a coordinated subject, a parenthetical, a fronted phrase)
    """
    if lead_n == 0:
        return "initial"
    if lead_n > 4:
        return "other"
    for t, p in zip(lead_text.split(" "), lead_pos.split(" ")):
        if not (p in MARKER_POS or (p == "PUNCT" and t == ",") or (p == "SCONJ" and t in MARKER_SCONJ)):
            return "other"
    return "after_marker"


# verbs whose bare "it"/"you" complement subject is an object in form ("make it look", "let you go")
ECM_VERBS = {"make", "let", "get", "have", "see", "hear", "watch", "feel", "help", "want", "need", "like", "expect",
             "find", "keep", "leave", "call", "consider", "imagine", "catch", "prefer", "allow", "ask", "tell",
             "cause", "force", "enable", "order", "permit", "wish"}
# after these the pronoun is a determiner or part of a larger subject ("you two", "we all", "he and Sue")
NOT_BARE_SUBJECT = {"all", "both", "two", "three", "four", "guys", "lot", "folks", "people", "kids", "boys", "girls",
                    "and", "or", "himself", "herself", "myself", "yourself", "itself", "ourselves", "themselves",
                    "yourselves"}
JUNK = r"[¶♪♫�]"


def _nth_last(text: pd.Series, n: int) -> pd.Series:
    return text.str.split(" ").map(lambda t: t[-n] if len(t) >= n else "")


def clean(d):
    """Checks that need no rater: the pronoun is a whole subject of the stated kind, the text has no symbols.

    Each rule answers a kind of misfit the reader check turned up in the first pilot draw.
    """
    whole = ~d.r1.isin(NOT_BARE_SUBJECT) & ~d.r1_pos.isin(["NUM", "NOUN", "PROPN"])
    stutter = (d.r1 == d.form) | (d.l1 == d.form)                       # "I I think", "it it seems"
    ecm = (d.ptype == "embedded") & d.form.isin(["it", "you"]) & d.gov_lemma.isin(ECM_VERBS)
    text = ~d.context.str.contains(JUNK) & ~d.target_overt.str.contains(JUNK)
    # "that" must introduce a complement: directly after a verb or adjective, and not "so ... that",
    # "such ... that" or "now that"
    before_that, pos_before_that = _nth_last(d.lead_text, 2), _nth_last(d.lead_pos, 2)
    that_ok = (pos_before_that.isin(["VERB", "AUX", "ADJ"]) & (before_that != "now")
               & ~d.target_overt.str.lower().str.contains(r"\b(?:so|such)\b[^.?!]*\bthat\b"))
    bad_that = (d.ptype == "embedded") & (d.kind == "that") & ~that_ok
    # a conjoined clause needs a clause before it ("You and he get in" is a conjoined subject)
    bad_conj = (d.ptype == "conjunction") & (d.lead_n <= 6) & ~d.lead_pos.str.contains("VERB|AUX")
    formula = (d.ptype == "matrix") & d.lead_text.isin(["here", "there"])  # "Here they come", "There you are"
    return whole & ~stutter & ~ecm & text & (d.kind != "raising_xcomp") & ~bad_that & ~bad_conj & ~formula


def main_clause(d, person):
    return (d.ptype == "matrix") & (d.person == person) & d.position.isin(["initial", "after_marker"])


# cell name -> (category in the eval schema, filter on the candidate table, items per decile)
CELLS = {
    "subj_1sg": ("subject_drop", lambda d: main_clause(d, "1sg"), 24),
    "subj_1pl": ("subject_drop", lambda d: main_clause(d, "1pl"), 24),
    "subj_2": ("subject_drop", lambda d: main_clause(d, "2"), 24),
    "subj_3sg": ("subject_drop", lambda d: main_clause(d, "3sg"), 24),
    "subj_3pl": ("subject_drop", lambda d: main_clause(d, "3pl"), 24),
    "emb_that": ("embedded_drop", lambda d: (d.ptype == "embedded") & (d.kind == "that"), 24),
    "emb_bare": ("embedded_drop", lambda d: (d.ptype == "embedded") & (d.kind == "bare"), 24),
    "conj": ("conjunction", lambda d: (d.ptype == "conjunction") & d.kind.isin(["and", "but"]), 48),
    "adv": ("adverbial", lambda d: d.ptype == "adverbial", 48),
    "expl_it": ("expletive", lambda d: d.ptype == "expletive_it", 24),
}


def matched_quota(avail: pd.DataFrame, n: int) -> pd.Series:
    """Items per stratum that every decile can supply, as close to the pooled mix as possible.

    avail: decile x stratum counts. Returns stratum -> count, summing to min(n, capacity).
    """
    cap = avail.min(axis=0).astype(int)
    pooled = avail.sum(axis=0) / max(avail.values.sum(), 1)
    q = pd.Series(0, index=avail.columns)
    left = min(n, int(cap.sum()))
    while left > 0:
        room = cap - q
        open_ = room[room > 0].index
        # give the next item to the stratum furthest below its pooled share
        deficit = (pooled[open_] * n - q[open_])
        s = deficit.idxmax()
        q[s] += 1
        left -= 1
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True, help="directory with candidates and both score tables")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--label", default="robbi")
    ap.add_argument("--corpus-slack", type=int, default=1)
    ap.add_argument("--max-item-tokens", type=int, default=110, help="rater tokens of context + target")
    ap.add_argument("--target-words", type=int, nargs=2, default=[4, 20])
    ap.add_argument("--min-gap-lines", type=int, default=6, help="two drawn items are never this close in a file")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every cell size")
    ap.add_argument("--boundary-margin", type=float, default=0.05,
                    help="keep the item score this fraction away from its decile's thresholds (fp16 jitter is ~1-4%%)")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--no-variants", action="store_true", help="draw only from sentences as they come")
    ap.add_argument("--position", choices=["matched", "natural"], default="matched",
                    help="matched: deciles share one mix of source x line position, closest-to-typical items "
                         "first. natural: deciles share one mix of source, items drawn at random within it")
    ap.add_argument("--cells", nargs="*", default=None, help="draw only these cells")
    ap.add_argument("--pool", default=None,
                    help="draw only from <pool>_candidates.parquet, scored in <pool>_<label>.parquet "
                         "(a pronoun-free pool); --exclude then lists ids of this pool")
    ap.add_argument("--exclude-parents", type=Path, nargs="*", default=[],
                    help="with --pool: cand_ids of sentences rejected on review in another set")
    ap.add_argument("--reviewed", type=Path, default=None, help="reviewed.csv from review_batches.py merge")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="files with one cand_id per line: items rejected on review, never drawn again")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    c = pd.read_parquet(a.dir / "items_candidates.parquet")
    corp = pd.read_parquet(a.dir / f"{a.label}__all.parquet")[["stem", "line_idx", "char_start", "surp", "decile"]]
    corp = corp.rename(columns={"surp": "surp_corpus", "decile": "decile_corpus"})
    cols = {"surp": "surp_item", "decile": "decile_item", "n_tokens": "rater_tokens"}
    item = pd.concat([pd.read_parquet(p) for p in sorted(a.dir.glob(f"items_{a.label}__*.parquet"))]).rename(columns=cols)
    d = c.merge(item[["cand_id", "surp_item", "decile_item", "rater_tokens", "top1"]], on="cand_id", how="left")
    d["parent_cand_id"], d["variant"] = d.cand_id, False
    # synthetic variants (make_context_variants.py): same target, less of its real context
    vfiles = sorted(a.dir.glob("variants_*.parquet"))
    vfiles = [f for f in vfiles if not f.name.startswith(f"variants_{a.label}__")]
    if a.pool:
        v = pd.read_parquet(a.dir / f"{a.pool}_candidates.parquet")
        vs = pd.read_parquet(a.dir / f"{a.pool}_{a.label}.parquet").rename(columns=cols)
        v = v.merge(vs[["var_id", "surp_item", "decile_item", "rater_tokens", "top1"]], on="var_id", how="inner")
        # an item whose whole context was already pronoun-free is a sentence as it comes
        v["variant"] = v.context_words.values < c.set_index("cand_id").context_words.reindex(v.parent_cand_id).values
        v["cand_id"] = v.var_id
        d = v.drop(columns=["var_id"]).reset_index(drop=True)
    elif vfiles and not a.no_variants:
        v = pd.concat([pd.read_parquet(f) for f in vfiles])
        vs = pd.concat([pd.read_parquet(f) for f in sorted(a.dir.glob(f"variants_{a.label}__*.parquet"))]).rename(columns=cols)
        v = v.merge(vs[["var_id", "surp_item", "decile_item", "rater_tokens", "top1"]], on="var_id", how="inner")
        v["cand_id"], v["variant"] = v.var_id, True
        d = pd.concat([d, v.drop(columns=["var_id"])], ignore_index=True)
    d = d.merge(corp, on=["stem", "line_idx", "char_start"], how="left")
    d["genre_group"] = d.stem.map(GENRE_GROUP)
    d["position"] = [position(n, t, p) for n, t, p in zip(d.lead_n, d.lead_text, d.lead_pos)]
    d["place"] = np.where(d.line_initial, "line-initial", np.where(d.sent_initial, "sentence-initial", "internal"))
    d["stratum"] = d.genre_group + "|" + d.place if a.position == "matched" else d.genre_group
    # deleting the pronoun leaves a bare verb form first, which also reads as an imperative
    d["null_bare_verb"] = (d.ptype == "matrix") & (d.r1_pos == "VERB") & (d.r1 == d.r1_lemma)
    n0 = int((~d.variant).sum())
    d = d[(d.decile_item >= 0) & d.decile_corpus.notna()]
    ok = clean(d)
    print(f"dropped by the surface checks: {(~ok & ~d.variant).sum():,} of {n0:,} candidates")
    d = d[ok]
    rejected = {int(x) for f in a.exclude for x in f.read_text().split()}
    bad_parents = {int(x) for f in a.exclude_parents for x in f.read_text().split()}
    if rejected or bad_parents:
        hit = d.cand_id.isin(rejected) | d.parent_cand_id.isin(bad_parents if a.pool else rejected)
        print(f"excluded on review: {(hit & ~d.variant).sum():,} items and {(hit & d.variant).sum():,} variants")
        d = d[~hit]
    nat = d[~d.variant]
    agree = pd.crosstab(nat.decile_corpus.astype(int), nat.decile_item.astype(int))
    agree.to_csv(a.out_dir / "corpus_vs_item_decile.csv")
    nat.groupby(["ptype", "decile_item", "place"]).size().rename("n").reset_index().to_csv(
        a.out_dir / "place_by_decile.csv", index=False)
    thr = np.array([0.0] + ROBBI_THRESHOLDS + [np.inf])
    di = d.decile_item.astype(int).values
    clear = (d.surp_item.values >= thr[di] * (1 + a.boundary_margin)) & (d.surp_item.values <= thr[di + 1] * (1 - a.boundary_margin))
    usable = d[(d.rater_tokens <= a.max_item_tokens) & d.target_words.between(*a.target_words) & clear].copy()
    usable["decile"] = usable.decile_item.astype(int)
    usable["exact"] = usable.decile_corpus == usable.decile_item
    # tier 1: a sentence as it comes, corpus and item deciles agreeing; tier 2: as it comes, but
    # its corpus decile is further off; tier 3: a variant with a shortened context
    near = (usable.decile_corpus - usable.decile_item).abs() <= a.corpus_slack
    usable["tier"] = np.where(usable.variant, 3, np.where(near, 1, 2))
    print(f"candidates {n0:,}; exact decile agreement {np.mean(nat.decile_corpus == nat.decile_item):.3f}; "
          f"within {a.corpus_slack}: {np.mean((nat.decile_corpus - nat.decile_item).abs() <= a.corpus_slack):.3f}; "
          f"usable by tier {usable.tier.value_counts().sort_index().to_dict()}")

    # order within a stratum: a fixed pseudo-random number per item, so a redraw after review
    # keeps every item that is still eligible; items already read and accepted go first
    usable["rand"] = ((usable.cand_id.astype(np.uint64) * np.uint64(2654435761) + np.uint64(a.seed)) % np.uint64(2**32)) / 2**32
    accepted = set()
    if a.reviewed and a.reviewed.exists():
        rv = pd.read_csv(a.reviewed)
        accepted = set(rv[rv.problem == "none"].cand_id)
    usable["accepted"] = usable.cand_id.isin(accepted)
    drawn, avail_rows, short_rows = [], [], []
    taken: dict = {}                                   # stem -> sorted list of taken line indices
    # what fills the matched part of a cell, and the order items are taken in
    first_tiers = [1, 2, 3] if a.pool else [1]
    if a.pool:
        order, asc = ["rand"], [True]
    elif a.position == "natural":
        order, asc = ["tier", "rand"], [True, True]
    else:
        order, asc = ["tier", "accepted", "exact", "len_dist", "rand"], [True, False, False, True, True]
    cells = {k: v for k, v in CELLS.items() if a.cells is None or k in a.cells}

    def free(stem, line):
        return all(abs(line - t) >= a.min_gap_lines for t in taken.get(stem, ()))

    for cell, (category, filt, size) in cells.items():
        n = int(round(size * a.scale))
        u = usable[filt(usable)]
        t1 = u[u.tier.isin(first_tiers)]
        avail = t1.groupby(["decile", "stratum"]).size().unstack(fill_value=0).reindex(range(10), fill_value=0)
        quota = matched_quota(avail, n)
        med = t1.target_words.median()
        by_tier = u.groupby(["decile", "tier"]).size().unstack(fill_value=0).reindex(index=range(10), columns=[1, 2, 3], fill_value=0)
        for dec in range(10):
            avail_rows.append({"cell": cell, "category": category, "decile": dec, "usable": int(by_tier.loc[dec, 1]),
                               "usable_item_only": int(by_tier.loc[dec, 2]), "usable_variants": int(by_tier.loc[dec, 3]),
                               "matched_capacity": int(quota.sum()), "wanted": n})
            pool = u[u.decile == dec]
            pool = pool.assign(len_dist=(pool.target_words - med).abs()).sort_values(order, ascending=asc)
            got = []
            first = pool[pool.tier.isin(first_tiers)]
            for stratum, k in quota.items():           # matched part, tier 1 only
                for r in first[first.stratum == stratum].itertuples():
                    if k == 0:
                        break
                    if free(r.stem, r.line_idx):
                        got.append((r.cand_id, True))
                        taken.setdefault(r.stem, []).append(r.line_idx)
                        k -= 1
            have = {g[0] for g in got}
            for r in pool.itertuples():                # top-up: tier 1 outside the matched mix, then 2, then 3
                if len(got) >= n:
                    break
                if r.cand_id not in have and free(r.stem, r.line_idx):
                    got.append((r.cand_id, False))
                    taken.setdefault(r.stem, []).append(r.line_idx)
            if len(got) < n:
                short_rows.append({"cell": cell, "category": category, "decile": dec, "wanted": n, "found": len(got),
                                   "missing": n - len(got)})
            sel = u.set_index("cand_id").loc[[g[0] for g in got]].reset_index()
            sel["cell"], sel["category"], sel["matched"] = cell, category, [g[1] for g in got]
            drawn.append(sel)
    sets = pd.concat(drawn, ignore_index=True)
    sets.insert(0, "item_id", np.arange(1, len(sets) + 1))
    sets.drop(columns=["rand", "norm_target"]).to_parquet(a.out_dir / "sets.parquet")
    pd.DataFrame(avail_rows).to_csv(a.out_dir / "availability.csv", index=False)
    pd.DataFrame(short_rows, columns=["cell", "category", "decile", "wanted", "found", "missing"]).to_csv(
        a.out_dir / "shortfall.csv", index=False)

    flat = sets.sort_values(["decile", "cell", "item_id"])[
        ["item_id", "decile", "cell", "form", "context", "target_overt", "target_null", "surp_item", "surp_corpus",
         "decile_corpus", "stem", "place", "position", "kind", "null_bare_verb", "matched", "tier", "cand_id",
         "parent_cand_id", "context_words", "target_words"]]
    flat.to_csv(a.out_dir / "items_flat.csv", index=False)
    stim = a.out_dir / "stimuli"
    stim.mkdir(exist_ok=True)
    for category, g in sets.groupby("category"):
        rows = []
        for r in g.itertuples():
            common = {"item_id": r.item_id, "category": category, "condition": f"{r.cell}_d{r.decile}",
                      "context": r.context, "language": "en", "names": "", "generator": f"test_10M:{r.stem}",
                      "cell": r.cell, "decile": r.decile, "cand_id": r.cand_id}
            rows.append({**common, "pronoun_status": 1, "target": r.target_overt, "hotspot_token": r.hotspot_token,
                         "hotspot_position": r.hotspot_overt})
            rows.append({**common, "pronoun_status": 0, "target": r.target_null, "hotspot_token": r.hotspot_token,
                         "hotspot_position": r.hotspot_null})
        cols = ["item_id", "category", "condition", "pronoun_status", "context", "target", "hotspot_token",
                "hotspot_position", "language", "names", "generator", "cell", "decile", "cand_id"]
        pd.DataFrame(rows)[cols].to_csv(stim / f"{category}.csv", index=False)
    meta = {"label": a.label, "position": a.position, "pool": a.pool, "n_items": len(sets),
            "corpus_slack": a.corpus_slack, "seed": a.seed,
            "boundary_margin": a.boundary_margin, "n_candidates": n0, "n_usable": int((usable.tier == 1).sum()),
            "agree_exact": float(np.mean(nat.decile_corpus == nat.decile_item)),
            "agree_within_slack": float(np.mean((nat.decile_corpus - nat.decile_item).abs() <= a.corpus_slack)),
            "drawn_by_tier": {int(k): int(v) for k, v in sets.tier.value_counts().sort_index().items()},
            "cells": {k: int(round(v[2] * a.scale)) for k, v in cells.items()},
            "exact_share": float(sets.exact.mean()), "matched_share": float(sets.matched.mean()),
            "excluded_on_review": len(rejected) + len(bad_parents),
            "accepted_on_review": int(sets.accepted.mean() * len(sets))}
    (a.out_dir / "sets.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    print(pd.DataFrame(avail_rows).pivot(index="cell", columns="decile", values="usable").to_string())
    print("drawn per cell x decile")
    print(sets.groupby(["cell", "decile"]).size().unstack(fill_value=0).to_string())
    if short_rows:
        print("SHORTFALL"); print(pd.DataFrame(short_rows).to_string(index=False))


if __name__ == "__main__":
    main()
