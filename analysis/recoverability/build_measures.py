"""
Step-2 analysis for the graded pronoun-drop study: turn the scoring pass's
sufficient statistics into recoverability measures, run the estimator-
invariance and memorization checks, and characterize the ranking.

Inputs (pulled from S3 recoverability/ or read from a PVC mirror):
  <root>/train_90M/instances/<genre>.parquet     (scoring pass output)
  <root>/train_90M/manifest/<genre>.json
  <root>/fold_assignments/train_90M/<genre>.parquet
  data/unigrams/<...>.pkl                        (SLOR unigram table, optional)

Measures per instance (columns added):
  surp_form__<scorer>   = -logprob_sum          (form surprisal)
  surp_phi__<scorer>    = -log sum P(inventory forms w/ same person/number)
  slot_entropy__<scorer> = H(inventory | ctx), renormalized over inventory
  surp_form__clean      = cross-fold stitched: rater_b for fold-a lines,
                          rater_a for fold-b lines (never scored by a model
                          that trained on the instance)
  surp_form__seen       = the opposite stitch (memorization-contaminated)

Outputs under --out:
  measures_summary.md         correlation matrices, deflation stats,
                              decile composition tables
  instance_measures/<genre>.parquet   per-instance measure columns
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

GENRES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles",
          "simple_wiki", "switchboard"]

# person/number feature groups over the inventory (manifest order applies)
PHI_GROUPS = {
    (1, "Sing"): ["i"],
    (1, "Plur"): ["we"],
    (2, None): ["you"],
    (3, "Sing"): ["he", "she", "it", "one"],
    (3, "Plur"): ["they"],
}


def load_genre(root: Path, genre: str, scorers: list) -> pd.DataFrame:
    inst = pq.read_table(root / "train_90M" / "instances" / f"{genre}.parquet")
    df = inst.to_pandas()
    fold = pq.read_table(
        root / "fold_assignments" / "train_90M" / f"{genre}.parquet"
    ).to_pandas()
    df = df.merge(fold, on="line_idx", how="left", validate="many_to_one")
    df["genre"] = genre

    man = json.loads(
        (root / "train_90M" / "manifest" / f"{genre}.json").read_text())
    inv_order = man["inventory_order"]
    inv_ids = man["inventory_ids"]
    # flat index ranges per form in the stored inv_probs vectors
    spans, i = {}, 0
    for form in inv_order:
        spans[form] = (i, i + len(inv_ids[form]))
        i += len(inv_ids[form])

    for sc in scorers:
        df[f"surp_form__{sc}"] = -df[f"{sc}__logprob_sum"]
        inv = np.array(
            [row if row is not None else [np.nan] * i
             for row in df[f"{sc}__inv_probs"]], dtype=np.float32)
        # per-form probability = sum over case variants
        form_p = {f: inv[:, a:b].sum(axis=1) for f, (a, b) in spans.items()}
        # phi-feature surprisal: sum over the instance's own feature group
        phi = np.full(len(df), np.nan, dtype=np.float32)
        for (person, number), forms in PHI_GROUPS.items():
            mask = (df["person"] == person).values
            if number is not None:
                mask &= (df["number"] == number).values
            grp_p = np.sum([form_p[f] for f in forms], axis=0)
            phi[mask] = -np.log(np.clip(grp_p[mask], 1e-12, None))
        df[f"surp_phi__{sc}"] = phi
        # slot entropy over the renormalized inventory distribution
        all_p = np.stack([form_p[f] for f in inv_order], axis=1)
        tot = all_p.sum(axis=1, keepdims=True)
        pn = np.clip(all_p / np.clip(tot, 1e-12, None), 1e-12, None)
        df[f"slot_entropy__{sc}"] = -(pn * np.log(pn)).sum(axis=1)
        df.drop(columns=[f"{sc}__inv_probs"], inplace=True)

    # cross-fold stitching
    a = df["fold"] == "a"
    df["surp_form__clean"] = np.where(
        a, df["surp_form__rater_b"], df["surp_form__rater_a"])
    df["surp_form__seen"] = np.where(
        a, df["surp_form__rater_a"], df["surp_form__rater_b"])
    df["surp_phi__clean"] = np.where(
        a, df["surp_phi__rater_b"], df["surp_phi__rater_a"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/recoverability"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/recoverability/analysis"))
    ap.add_argument("--scorers", nargs="+",
                    default=["baseline", "rater_a", "rater_b"])
    ap.add_argument("--sample-corr", type=int, default=1_000_000,
                    help="sample size for correlation matrices")
    ap.add_argument("--genres", nargs="+", default=GENRES)
    ap.add_argument("--emit-selection", action="store_true")
    ap.add_argument("--selection-seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "instance_measures").mkdir(exist_ok=True)

    frames = []
    for g in args.genres:
        df = load_genre(args.root, g, args.scorers)
        pq.write_table(
            __import__("pyarrow").Table.from_pandas(df, preserve_index=False),
            args.out / "instance_measures" / f"{g}.parquet")
        frames.append(df)
        print(f"{g}: {len(df):,} instances")
    df = pd.concat(frames, ignore_index=True)
    del frames

    lines = ["# Recoverability measures — step-2 analysis\n"]
    lines.append(f"Total instances: {len(df):,}\n")
    lines.append(f"In personal-pronoun lexicon: {df.in_lexicon.mean():.1%}\n")

    # --- estimator invariance ---
    meas_cols = ([f"surp_form__{s}" for s in args.scorers]
                 + ["surp_form__clean"]
                 + [f"surp_phi__{s}" for s in args.scorers]
                 + [f"slot_entropy__{s}" for s in args.scorers])
    samp = df.sample(min(args.sample_corr, len(df)), random_state=42)
    corr = samp[meas_cols].corr(method="spearman")
    lines.append("\n## Spearman correlations (measures x scorers, "
                 f"n={len(samp):,})\n")
    lines.append(corr.round(3).to_markdown())

    # --- memorization check ---
    ok = df[["surp_form__seen", "surp_form__clean",
             "surp_form__baseline"]].dropna()
    rho_clean_seen = ok["surp_form__seen"].corr(
        ok["surp_form__clean"], method="spearman")
    rho_clean_base = ok["surp_form__baseline"].corr(
        ok["surp_form__clean"], method="spearman")
    deflation = (ok["surp_form__seen"] - ok["surp_form__clean"])
    lines.append("\n\n## Memorization check\n")
    lines.append(f"- Spearman(seen, clean) = {rho_clean_seen:.4f}\n")
    lines.append(f"- Spearman(baseline, clean) = {rho_clean_base:.4f}\n")
    lines.append(f"- seen-minus-clean surprisal: mean {deflation.mean():+.4f} "
                 f"nats, median {deflation.median():+.4f}, "
                 f"p5 {deflation.quantile(.05):+.3f}, "
                 f"p95 {deflation.quantile(.95):+.3f}\n")

    # --- decile composition on the primary (clean) ranking ---
    lex = df[df.in_lexicon & df.surp_form__clean.notna()].copy()
    lex["decile"] = pd.qcut(lex.surp_form__clean, 10, labels=False)
    lines.append("\n## Decile composition (clean form surprisal; decile 0 = "
                 "most recoverable)\n")
    comp = (lex.groupby("decile")
            .agg(n=("form", "size"),
                 top_form=("form", lambda s: s.value_counts().idxmax()),
                 top_form_share=("form",
                                 lambda s: s.value_counts(normalize=True).iloc[0]),
                 pct_first_person=("person", lambda s: (s == 1).mean()),
                 pct_third=("person", lambda s: (s == 3).mean()),
                 pct_childes=("genre", lambda s: (s == "childes").mean()),
                 median_surp=("surp_form__clean", "median")))
    lines.append(comp.round(3).to_markdown())
    lines.append("\n\n### Form distribution by decile\n")
    form_by_dec = (lex.groupby("decile")["form"]
                   .value_counts(normalize=True).unstack(fill_value=0))
    top_forms = lex["form"].value_counts().head(8).index
    lines.append(form_by_dec[top_forms].round(3).to_markdown())

    # --- alignment coverage note ---
    lines.append("\n\n## Coverage\n")
    for g in args.genres:
        man = json.loads((args.root / "train_90M" / "manifest"
                          / f"{g}.json").read_text())
        c = man["counters"]
        tot = c["instances_seen"]
        lines.append(f"- {g}: {c['instances_aligned']:,}/{tot:,} aligned "
                     f"({c['instances_aligned']/tot:.1%})\n")

    (args.out / "measures_summary.md").write_text("".join(lines))
    print(f"\nwrote {args.out}/measures_summary.md")

    # --- freeze the selection ranking -----------------------------------
    if args.emit_selection:
        sel_dir = args.out / "selection" / "train_90M"
        sel_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(args.selection_seed)
        # info arm: decile 0 = most recoverable = removed first (cumulative:
        # condition K removes deciles < K/10). rand arm: seeded uniform
        # permutation into equal deciles over the SAME population.
        lex = lex.copy()
        perm = rng.permutation(len(lex))
        lex["rand_decile"] = (perm * 10 // len(lex)).astype(np.int8)
        lex.rename(columns={"decile": "info_decile"}, inplace=True)
        for g, gdf in lex.groupby("genre"):
            pq.write_table(
                __import__("pyarrow").Table.from_pandas(
                    gdf[["line_idx", "token_i", "form", "surp_form__clean",
                         "info_decile", "rand_decile"]],
                    preserve_index=False),
                sel_dir / f"{g}.parquet")
        thresholds = [float(lex[lex.info_decile <= d].surp_form__clean.max())
                      for d in range(9)]
        sel_manifest = {
            "population": "in_lexicon (personal pronouns) & aligned & "
                          "surp_form__clean not null",
            "population_n": int(len(lex)),
            "ranking": "surp_form__clean (cross-fold stitched rater form "
                       "surprisal; decile 0 = most recoverable)",
            "cumulative_semantics": "condition K%% removes instances with "
                                    "decile < K/10 (either arm)",
            "decile_upper_thresholds_nats": thresholds,
            "random_seed": args.selection_seed,
            "genres": sorted(lex.genre.unique().tolist()),
            "note_pool": "pull_10M pool instances are selected by these "
                         "absolute thresholds (train-derived), not pool "
                         "percentiles",
        }
        with open(args.out / "selection" / "SELECTION_MANIFEST.json", "w") as f:
            json.dump(sel_manifest, f, indent=2)
        print(f"wrote selection ({len(lex):,} instances) + manifest")


if __name__ == "__main__":
    main()
