#!/usr/bin/env python3
"""Rebuild the Foundry trajectory figure suite from condition-matched evals.

The scientific unit for ribbons is a training cell (or a seed in seed-resolved
plots). Scores are binary preferences from length-normalized mean likelihood.
No generic-evaluation or SLOR values are used. Checkpoint -1 is only plotted
when it exists in the separately audited condition-matched initialization set.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data/eval_results/null_subj_v2_condition_matched_v1"
INIT_ROOT = REPO / "data/eval_results/null_subj_v2_condition_matched_init_v1"
OUT = REPO / "analysis/eval_v2/figures/foundry_trajectories/condition_matched_v1"

ARCHES = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large", "lstm", "mamba_370m"]
ARCH_LABELS = {
    "gpt2_small": "GPT-2 small", "gpt2_medium": "GPT-2 medium",
    "gpt2_large": "GPT-2 large", "bert_large": "BERT large",
    "lstm": "LSTM", "mamba_370m": "Mamba 370M",
}
CONDS = ["baseline", "remove_expletive_sentences", "impoverish_case", "lemmatize_verbs", "enrich_verbal_morphology"]
COND_LABELS = {
    "baseline": "Baseline", "remove_expletive_sentences": "Remove expletives",
    "impoverish_case": "Impoverish case", "lemmatize_verbs": "Lemmatize verbs",
    "enrich_verbal_morphology": "Enrich morphology",
}
CATS = ["subject_drop", "subject_drop_no_agreement", "expletive", "object_drop", "embedded_drop", "extraction", "conjunction", "control"]
CAT_LABELS = {
    "subject_drop": "Subject drop", "subject_drop_no_agreement": "Subject drop (no agreement)",
    "expletive": "Expletive", "object_drop": "Object drop", "embedded_drop": "Embedded drop",
    "extraction": "Extraction", "conjunction": "Conjunction", "control": "Control",
}
HP_COLORS = dict(zip(range(5), plt.cm.tab10.colors[:5]))
COND_COLORS = dict(zip(CONDS, plt.cm.tab10.colors[:5]))
STAGE_COLORS = {"early-only": "#1f77b4", "continuation early": "#ff7f0e", "late continuation": "#d62728"}


def load_training() -> pd.DataFrame:
    rows = []
    for pp in sorted((ROOT / "pairs").glob("*.parquet")):
        cp = ROOT / "checkpoints" / pp.name
        if not cp.exists():
            continue
        ck = pd.read_parquet(cp, columns=["checkpoint_step", "tokens_seen", "epoch", "hp_rank", "seed"])
        if ck.empty:
            continue
        ck = ck.drop_duplicates("checkpoint_step")
        d = pd.read_parquet(pp, columns=["cell_id", "architecture", "intervention", "checkpoint_step", "category", "prefers_overt_meanlp"])
        d = d.dropna(subset=["prefers_overt_meanlp"]).merge(ck, on="checkpoint_step", how="inner", validate="many_to_one")
        if d.empty:
            continue
        g = (d.groupby(["cell_id", "architecture", "intervention", "hp_rank", "seed", "category", "tokens_seen", "epoch"], as_index=False)
               ["prefers_overt_meanlp"].mean().rename(columns={"prefers_overt_meanlp": "preference"}))
        full = float(ck.epoch.max()) > 2
        g["stage"] = "continuation" if full else "early-only"
        g["stage_segment"] = "early-only" if not full else np.where(g.epoch <= 2, "continuation early", "late continuation")
        g["source"] = "training"
        rows.append(g)
    if not rows:
        raise RuntimeError(f"No condition-matched pair/checkpoint data under {ROOT}")
    return pd.concat(rows, ignore_index=True)


def log_bin(d: pd.DataFrame, unit_cols: list[str], group_cols: list[str], n_bins: int = 30) -> pd.DataFrame:
    """Bin within architecture after collapsing duplicate rows per unit/token."""
    keys = list(dict.fromkeys(["architecture"] + group_cols + unit_cols + ["tokens_seen"]))
    q = d.groupby(keys, as_index=False).preference.mean()
    out = []
    for arch, z in q.groupby("architecture", sort=False):
        z = z.copy()
        pos = z.loc[z.tokens_seen > 0, "tokens_seen"]
        if pos.empty:
            z["bin_center"] = 0.0
        else:
            edges = np.unique(np.geomspace(float(pos.min()), float(pos.max()), n_bins + 1))
            idx = np.digitize(z.tokens_seen.to_numpy(), edges, right=False)
            centers = {0: 0.0, 1: float(edges[0]), len(edges): float(edges[-1])}
            centers.update({i: float(np.sqrt(edges[i-1] * edges[i])) for i in range(1, len(edges))})
            centers[1] = float(edges[0])
            z["bin_center"] = [centers[int(i)] for i in idx]
        out_keys = list(dict.fromkeys(["architecture"] + group_cols + unit_cols + ["bin_center"]))
        z = z.groupby(out_keys, as_index=False).preference.mean()
        out.append(z)
    return pd.concat(out, ignore_index=True)


def token_label(v: float) -> str:
    return f"{v/1e9:g}B" if v >= 1e9 else f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}K"


def ticks(min_tokens: float, max_tokens: float):
    vals = [min_tokens] + [v for v in [1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
                           if v > min_tokens * 1.05 and v <= max_tokens * 1.08]
    vals = list(dict.fromkeys(vals))
    labels = [token_label(v) for v in vals]
    return np.log10(np.asarray(vals) + 1), labels


def stats(q: pd.DataFrame, unit: str) -> pd.DataFrame:
    return (q.groupby("bin_center").preference.agg(mean="mean", sd="std", n="count").reset_index()
            .assign(se=lambda x: x.sd.fillna(0) / np.sqrt(x.n)))


def draw(ax, q: pd.DataFrame, label: str, color, unit: str = "cell_id", thin: bool = False):
    if q.empty:
        return
    if thin:
        for _, z in q.groupby(unit):
            z = z.sort_values("bin_center")
            ax.plot(np.log10(z.bin_center + 1), z.preference, color=color, lw=.65, alpha=.22)
    s = stats(q, unit).sort_values("bin_center")
    x = np.log10(s.bin_center.to_numpy() + 1); y = s["mean"].to_numpy(); se = s.se.to_numpy()
    ax.fill_between(x, np.clip(y-se, 0, 1), np.clip(y+se, 0, 1), color=color, alpha=.16, lw=0)
    ax.plot(x, y, color=color, lw=2, label=f"{label} (n={q[unit].nunique()})")


def facet_plot(d: pd.DataFrame, arch: str, series_col: str, series, title: str, path: Path, unit: str = "cell_id", thin: bool = False):
    z = d[d.architecture.eq(arch)].copy()
    if z.empty:
        return False
    z = log_bin(z, [unit], [series_col, "category"])
    mn = float(z.loc[z.bin_center > 0, "bin_center"].min()); mx = float(z.bin_center.max()); xt, xl = ticks(mn, mx)
    fig, axes = plt.subplots(4, 2, figsize=(15, 17), sharex=True, sharey=True)
    for i, (ax, cat) in enumerate(zip(axes.flat, CATS)):
        for key, label, color in series:
            draw(ax, z[(z.category == cat) & (z[series_col] == key)], label, color, unit, thin)
        ax.set_title(CAT_LABELS[cat], loc="left", fontweight="bold", fontsize=10)
        ax.set_ylim(0, 1); ax.axhline(.5, color=".55", ls=":", lw=1); ax.grid(alpha=.22)
        ax.set_ylabel("P(overt preferred)"); ax.set_xticks(xt, xl); ax.set_xlim(np.log10(mn+1)-.025, np.log10(mx+1)+.03)
        if i >= 6: ax.set_xlabel("Tokens seen (log scale)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(series)), frameon=False, bbox_to_anchor=(.5, -.005), fontsize=8)
    cap_path = OUT / "final_token_cutoffs.csv"
    cap = None
    if cap_path.exists():
        cap_rows = pd.read_csv(cap_path)
        cap_match = cap_rows[cap_rows.architecture.eq(arch)]
        if not cap_match.empty:
            cap = float(cap_match.common_final_tokens.iloc[0])
    is_common_horizon = cap is not None and np.isclose(mx, cap)
    if is_common_horizon:
        for ax in axes.flat:
            ax.axvline(np.log10(cap + 1), color="0.2", ls="--", lw=1.1, alpha=.8)
    subtitle = "Condition-matched binary length-normalized preference; mean ± 1 SE; training checkpoints only"
    if is_common_horizon:
        subtitle += f"; dashed line = common cutoff ({token_label(cap)})"
    fig.suptitle(title + "\n" + subtitle, fontsize=15, fontweight="bold", y=.995)
    fig.tight_layout(rect=(0, .045, 1, .96)); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    z.to_csv(path.with_suffix(".csv"), index=False)
    return True


def overview_seed_plot(d: pd.DataFrame, segment: str, path: Path):
    q = d[d.stage_segment.eq(segment)].copy()
    q = log_bin(q, ["seed"], [])
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
    mn = float(q.loc[q.bin_center > 0, "bin_center"].min()); mx = float(q.bin_center.max()); xt, xl = ticks(mn, mx)
    cap_path = OUT / "final_token_cutoffs.csv"
    cap_map = (pd.read_csv(cap_path).set_index("architecture").common_final_tokens.to_dict()
               if cap_path.exists() else {})
    for ax, arch in zip(axes.flat, ARCHES):
        z = q[q.architecture.eq(arch)]
        color = plt.cm.Dark2.colors[ARCHES.index(arch)]
        draw(ax, z, "seed mean", color, "seed", True)
        ax.set_title(f"{ARCH_LABELS[arch]} (n={z.seed.nunique()} seeds)", loc="left", fontweight="bold")
        ax.set_ylim(0,1); ax.axhline(.5,color=".55",ls=":"); ax.grid(alpha=.22); ax.set_ylabel("P(overt preferred)")
        ax.set_xticks(xt,xl); ax.set_xlim(np.log10(mn+1)-.025,np.log10(mx+1)+.03)
        cap = cap_map.get(arch)
        if cap is not None and not z.empty and np.isclose(float(z.bin_center.max()), float(cap)):
            ax.axvline(np.log10(float(cap)+1),color="0.2",ls="--",lw=1.1,alpha=.8)
    for ax in axes[-1]: ax.set_xlabel("Tokens seen (log scale)")
    label = "Early-only starts" if segment == "early-only" else "Late continuations"
    fig.suptitle(f"{label} by seed and architecture\nThin lines = seeds; heavy line = mean ± 1 SE across seeds", fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig); q.to_csv(path.with_suffix('.csv'),index=False)


def init_figures(manifest: dict):
    pair_files = sorted((INIT_ROOT / "pairs").glob("*.parquet"))
    if not pair_files:
        manifest["pending"].append({"family":"checkpoint_minus1", "reason":"No audited matched initialization results"}); return
    rows=[]
    for p in pair_files:
        d=pd.read_parquet(p, columns=["architecture","intervention","rep","category","prefers_overt_meanlp"])
        d=d.dropna(subset=["prefers_overt_meanlp"])
        if d.empty: continue
        seed=int(str(d.rep.iloc[0]).lstrip('s')) if str(d.rep.iloc[0]).startswith('s') else int(d.rep.iloc[0])
        g=d.groupby(["architecture","intervention","category"],as_index=False).prefers_overt_meanlp.mean().rename(columns={"prefers_overt_meanlp":"preference"})
        g["seed"]=seed; rows.append(g)
    q=pd.concat(rows,ignore_index=True).drop_duplicates(["architecture","intervention","category","seed"])
    out=OUT/"checkpoint_minus1"; out.mkdir(parents=True,exist_ok=True)
    for arch,z in q.groupby('architecture'):
        fig,axes=plt.subplots(4,2,figsize=(14,16),sharey=True)
        for ax,cat in zip(axes.flat,CATS):
            w=z[z.category.eq(cat)]
            for seed,s in w.groupby('seed'):
                s=s.set_index('intervention').reindex(CONDS)
                ax.plot(range(len(CONDS)),s.preference,marker='o',label=f's{seed}')
            ax.set_title(CAT_LABELS[cat],loc='left',fontweight='bold'); ax.set_ylim(0,1); ax.axhline(.5,color='.5',ls=':'); ax.grid(alpha=.2)
            ax.set_xticks(range(len(CONDS)),[COND_LABELS[c] for c in CONDS],rotation=25,ha='right',fontsize=8)
        axes.flat[0].legend(); fig.suptitle(f"{ARCH_LABELS[arch]} checkpoint −1 by seed\nMatched initialization subset only; 0 tokens seen",fontweight='bold')
        fig.tight_layout(rect=(0,0,1,.96)); p=out/f"checkpoint_minus1_by_seed_{arch}.png"; fig.savefig(p,dpi=180,bbox_inches='tight'); plt.close(fig)
        z.to_csv(p.with_suffix('.csv'),index=False); manifest['generated'].append(str(p.relative_to(OUT)))
        pivot=z.groupby(['category','intervention']).preference.mean().unstack('intervention').reindex(index=CATS,columns=CONDS)
        fig,ax=plt.subplots(figsize=(10,6)); im=ax.imshow(pivot,cmap='RdBu_r',vmin=0,vmax=1,aspect='auto'); fig.colorbar(im,ax=ax,label='P(overt preferred)')
        ax.set_yticks(range(len(CATS)),[CAT_LABELS[c] for c in CATS]); ax.set_xticks(range(len(CONDS)),[COND_LABELS[c] for c in CONDS],rotation=30,ha='right')
        ax.set_title(f"{ARCH_LABELS[arch]} checkpoint −1 matched preference\nMean across available seeds (n={z.seed.nunique()})",fontweight='bold')
        fig.tight_layout(); p=out/f"checkpoint_minus1_heatmap_{arch}.png"; fig.savefig(p,dpi=180,bbox_inches='tight'); plt.close(fig); manifest['generated'].append(str(p.relative_to(OUT)))
    missing=sorted(set(ARCHES)-set(q.architecture.unique()))
    if missing: manifest['pending'].append({"family":"checkpoint_minus1", "architectures":missing, "reason":"Condition-matched checkpoint -1 evaluation not yet available"})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=load_training()
    # Use a common end-of-training token budget within each architecture.  The
    # five transformed corpora have slightly different tokenized lengths, so
    # take the smallest intervention-specific terminal maximum and discard
    # later checkpoints from the other interventions.
    cell_max=(d.groupby(["architecture","intervention","cell_id"],as_index=False)
                .tokens_seen.max())
    condition_max=(cell_max.groupby(["architecture","intervention"],as_index=False)
                    .tokens_seen.max().rename(columns={"tokens_seen":"terminal_tokens"}))
    caps=(condition_max.groupby("architecture",as_index=False).terminal_tokens.min()
          .rename(columns={"terminal_tokens":"common_final_tokens"}))
    caps.to_csv(OUT/"final_token_cutoffs.csv",index=False)
    d=d.merge(caps,on="architecture",how="left",validate="many_to_one")
    d=d[d.tokens_seen <= d.common_final_tokens].drop(columns="common_final_tokens").copy()
    d.to_parquet(OUT/"trajectory_aggregates.parquet",index=False)
    counts=(d[["cell_id","architecture","intervention","hp_rank","seed","stage"]].drop_duplicates().groupby(["architecture","intervention","hp_rank","stage"]).size().rename('cells').reset_index())
    counts.to_csv(OUT/"coverage_counts.csv",index=False)
    manifest={"benchmark":"null_subj_v2_condition_matched_v1","metric":"prefers_overt_meanlp","uncertainty":"one standard error across cells or seeds","training_cells":int(d.cell_id.nunique()),"generated":[],"pending":[]}
    def add(ok,p):
        if ok: manifest['generated'].append(str(p.relative_to(OUT)))
    for arch in ARCHES:
        p=OUT/f"baseline_by_hyperparameter_{arch}.png"; add(facet_plot(d[d.intervention.eq('baseline')],arch,'hp_rank',[(h,f'H{h}',HP_COLORS[h]) for h in range(5)],f"{ARCH_LABELS[arch]}: baseline by hyperparameter",p),p)
        p=OUT/f"intervention_collapsed_{arch}.png"; add(facet_plot(d,arch,'intervention',[(c,COND_LABELS[c],COND_COLORS[c]) for c in CONDS],f"{ARCH_LABELS[arch]}: intervention trajectories across hyperparameters and seeds",p),p)
        # Preserve the newer explicit filename used for the same all-HP
        # estimand while keeping the historical intervention-collapsed family.
        alias=OUT/f"{arch}_condition_matched_all_hp_by_intervention.png"
        shutil.copyfile(p,alias); shutil.copyfile(p.with_suffix('.csv'),alias.with_suffix('.csv')); manifest['generated'].append(alias.name)
        p=OUT/f"early_vs_continuation_{arch}.png"; add(facet_plot(d,arch,'stage_segment',[(s,s.title(),STAGE_COLORS[s]) for s in STAGE_COLORS],f"{ARCH_LABELS[arch]}: early starts and continuations",p),p)
        early=d[(d.stage_segment.eq('early-only')) & d.intervention.eq('baseline')].copy(); early['series']='seed mean'
        p=OUT/f"early_starts_by_seed_category_{arch}.png"; add(facet_plot(early,arch,'series',[('seed mean','seed mean',plt.cm.Dark2.colors[ARCHES.index(arch)])],f"{ARCH_LABELS[arch]}: baseline early-only starts by seed",p,unit='seed',thin=True),p)
        for hp in (0,1):
            q=d[(d.hp_rank.eq(hp)) & d.stage_segment.isin(['early-only','continuation early']) & d.intervention.eq('baseline')].copy(); q['series']='seed mean'
            p=OUT/f"early_starts_by_seed_category_h{hp}_{arch}.png"
            if q[q.architecture.eq(arch)].empty:
                manifest['pending'].append({"figure":p.name,"reason":f"No completed condition-matched H{hp} cells for {arch}"})
            else: add(facet_plot(q,arch,'series',[('seed mean','seed mean',plt.cm.Dark2.colors[ARCHES.index(arch)])],f"{ARCH_LABELS[arch]}: baseline H{hp} early path by seed",p,unit='seed',thin=True),p)
        h2=d[(d.hp_rank.eq(2)) & d.stage_segment.isin(['early-only','continuation early'])]
        p=OUT/'h1_h2_early_paths'/f"h2_early_path_{arch}.png"; add(facet_plot(h2,arch,'intervention',[(c,COND_LABELS[c],COND_COLORS[c]) for c in CONDS],f"{ARCH_LABELS[arch]}: H2 early path",p),p)
        hh=d[(d.hp_rank.isin([1,2])) & d.stage_segment.isin(['early-only','continuation early'])].copy(); hh['hp_label']=hh.hp_rank.map({1:'H1',2:'H2'})
        p=OUT/'h1_h2_early_paths'/f"h1_vs_h2_early_{arch}.png"
        if hh[(hh.architecture.eq(arch)) & hh.hp_rank.eq(1)].empty: manifest['pending'].append({"figure":str(p.relative_to(OUT)),"reason":f"No completed condition-matched H1 cells for {arch}"})
        else: add(facet_plot(hh,arch,'hp_label',[('H1','H1',HP_COLORS[1]),('H2','H2',HP_COLORS[2])],f"{ARCH_LABELS[arch]}: H1 versus H2 early paths",p,unit='seed'),p)
        h01=d[(d.hp_rank.isin([0,1])) & d.stage_segment.isin(['early-only','continuation early']) & d.intervention.eq('baseline')].copy(); h01['hp_label']=h01.hp_rank.map({0:'H0',1:'H1'})
        p=OUT/f"baseline_h0_vs_h1_early_all_seeds_{arch}.png"
        if h01[(h01.architecture.eq(arch)) & h01.hp_rank.eq(1)].empty: manifest['pending'].append({"figure":p.name,"reason":f"No completed condition-matched H1 cells for {arch}"})
        else: add(facet_plot(h01,arch,'hp_label',[('H0','H0',HP_COLORS[0]),('H1','H1',HP_COLORS[1])],f"{ARCH_LABELS[arch]}: baseline H0 versus H1 early paths",p,unit='seed'),p)
    # The recent GPT-2-small H0-only intervention panel is a distinct named
    # family in the source directory.
    h0=d[(d.hp_rank.eq(0)) & d.stage_segment.isin(['early-only','continuation early'])]
    p=OUT/'gpt2_small_condition_matched_h0_by_intervention.png'
    add(facet_plot(h0,'gpt2_small','intervention',[(c,COND_LABELS[c],COND_COLORS[c]) for c in CONDS],"GPT-2 small: H0 intervention trajectories",p),p)
    overview_seed_plot(d,'early-only',OUT/'early_starts_by_seed.png'); manifest['generated'].append('early_starts_by_seed.png')
    overview_seed_plot(d,'late continuation',OUT/'late_continuations_by_seed.png'); manifest['generated'].append('late_continuations_by_seed.png')
    init_figures(manifest)
    manifest['trajectory_cutoff']={
        'horizon':'minimum intervention-specific terminal maximum within architecture',
        'endpoint':'hard truncation; retain observed checkpoints at or below the horizon only',
        'display':'vertical dashed line on full-horizon panels',
    }
    # Old-vs-matched plots are already comparisons using the new measure; keep
    # only the regular binary versions (the user explicitly retired SLOR).
    comparisons=OUT/'comparisons'; comparisons.mkdir(exist_ok=True)
    source_dir=OUT.parent
    for name in ('gpt2_medium_old_vs_matched_h2_h4_trajectories.png','gpt2_medium_matched_minus_old_h2_h4_trajectories.png'):
        src=source_dir/name
        if src.exists():
            dst=comparisons/name; shutil.copyfile(src,dst); manifest['generated'].append(str(dst.relative_to(OUT)))
    readme="""# Condition-matched Foundry trajectory figures\n\nAll panels in this folder use `null_subj_v2_condition_matched_v1` and the regular binary overt-preference measure derived from length-normalized mean log likelihood. Lines aggregate equally over training cells unless a title says seed; ribbons are ±1 SE. The x-axis is tokens seen: each training figure begins at its earliest evaluated checkpoint, and each architecture is hard-truncated at the smallest terminal token count among its five interventions (`final_token_cutoffs.csv`). Only observed checkpoints at or below that boundary are retained; no endpoint interpolation is performed. Full-horizon panels mark the boundary with a vertical dashed line. No SLOR results and no old generic-evaluation values are mixed into these panels.\n\nCheckpoint −1 is kept separate and only shown for architectures/seeds with audited `null_subj_v2_condition_matched_init_v1` results. See `coverage_manifest.json` for panels that cannot yet be reproduced without missing matched evaluations. Remove-expletive stimuli are intentionally unchanged under the approved training-deprivation estimand; morphology enrichment uses the approved literal transformed stimuli.\n"""
    (OUT/'README.md').write_text(readme)
    (OUT/'coverage_manifest.json').write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"out":str(OUT),"cells":d.cell_id.nunique(),"figures":len(manifest['generated']),"pending":len(manifest['pending'])},indent=2))

if __name__ == '__main__': main()
