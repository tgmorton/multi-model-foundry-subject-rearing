#!/usr/bin/env python3
"""Forest plots of token-matched end-state preference by eval condition."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data/eval_results/null_subj_v2_condition_matched_v1"
SUITE = REPO / "analysis/eval_v2/figures/foundry_trajectories/condition_matched_v1"
OUT = SUITE / "endstate_forests"

ARCHES = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large", "lstm", "mamba_370m"]
ARCH_LABELS = {"gpt2_small":"GPT-2 small","gpt2_medium":"GPT-2 medium","gpt2_large":"GPT-2 large","bert_large":"BERT large","lstm":"LSTM","mamba_370m":"Mamba 370M"}
CONDS = ["baseline","remove_expletive_sentences","impoverish_case","lemmatize_verbs","enrich_verbal_morphology"]
COND_LABELS = {"baseline":"Baseline","remove_expletive_sentences":"Remove expletives","impoverish_case":"Impoverish case","lemmatize_verbs":"Lemmatize verbs","enrich_verbal_morphology":"Enrich morphology"}
COLORS = dict(zip(CONDS, plt.cm.tab10.colors[:5]))
CATS = ["subject_drop","subject_drop_no_agreement","expletive","object_drop","embedded_drop","extraction","conjunction","control"]
CAT_LABELS = {"subject_drop":"Subject drop","subject_drop_no_agreement":"Subject drop (no agreement)","expletive":"Expletive","object_drop":"Object drop","embedded_drop":"Embedded drop","extraction":"Extraction","conjunction":"Conjunction","control":"Control"}


def endpoint_rows(caps: pd.DataFrame) -> pd.DataFrame:
    cap_map = caps.set_index("architecture").common_final_tokens.to_dict()
    rows = []
    for pp in sorted((ROOT / "pairs").glob("*.parquet")):
        cp = ROOT / "checkpoints" / pp.name
        if not cp.exists():
            continue
        ck = pd.read_parquet(cp, columns=["architecture","intervention","checkpoint_step","tokens_seen","epoch","hp_rank","seed"])
        if ck.empty or float(ck.epoch.max()) <= 2:
            continue  # early-only trajectories are not end-state observations
        arch = str(ck.architecture.iloc[0])
        if float(ck.tokens_seen.max()) < float(cap_map[arch]):
            continue  # partial continuations cannot support an end-state claim
        lower_rows = ck[ck.tokens_seen <= cap_map[arch]].sort_values(["tokens_seen","checkpoint_step"])
        upper_rows = ck[ck.tokens_seen >= cap_map[arch]].sort_values(["tokens_seen","checkpoint_step"])
        if lower_rows.empty or upper_rows.empty:
            continue
        lower = lower_rows.iloc[-1]; upper = upper_rows.iloc[0]
        d = pd.read_parquet(pp, columns=["cell_id","architecture","intervention","checkpoint_step","category","condition","prefers_overt_meanlp"])
        d = d[(d.checkpoint_step == lower.checkpoint_step) & d.prefers_overt_meanlp.notna()]
        if d.empty:
            continue
        keys=["cell_id","architecture","intervention","category","condition"]
        g = (d.groupby(keys,as_index=False).prefers_overt_meanlp.mean()
             .rename(columns={"prefers_overt_meanlp":"preference"}))
        g["hp_rank"] = int(lower.hp_rank); g["seed"] = int(lower.seed)
        g["endpoint_tokens"] = int(lower.tokens_seen); g["common_final_tokens"] = int(cap_map[arch])
        g["lower_checkpoint_tokens"] = int(lower.tokens_seen); g["upper_checkpoint_tokens"] = int(upper.tokens_seen)
        rows.append(g)
    if not rows:
        raise RuntimeError("No full-run endpoint rows found")
    d = pd.concat(rows, ignore_index=True)
    # Preserve a matched intervention comparison: retain only HP-rank/seed
    # tuples that reach the common horizon in all five interventions.
    complete = (d[["architecture","hp_rank","seed","intervention"]].drop_duplicates()
                .groupby(["architecture","hp_rank","seed"],as_index=False)
                .intervention.nunique())
    complete = complete[complete.intervention == len(CONDS)][["architecture","hp_rank","seed"]]
    return d.merge(complete,on=["architecture","hp_rank","seed"],how="inner",validate="many_to_one")


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    # Each model cell (one HP rank × seed) is one independent observation.
    return (d.groupby(["architecture","intervention","category","condition"],as_index=False)
            .agg(mean_preference=("preference","mean"),sd=("preference","std"),n_cells=("cell_id","nunique"),
                 endpoint_tokens_min=("endpoint_tokens","min"),endpoint_tokens_max=("endpoint_tokens","max"),
                 common_final_tokens=("common_final_tokens","first"))
            .assign(se=lambda x:x.sd.fillna(0)/np.sqrt(x.n_cells),
                    lower=lambda x:np.clip(x.mean_preference-x.se,0,1),
                    upper=lambda x:np.clip(x.mean_preference+x.se,0,1)))


def plot_arch(s: pd.DataFrame, arch: str) -> Path:
    z = s[s.architecture.eq(arch)].copy()
    fig, axes = plt.subplots(4,2,figsize=(15,18),sharex=True)
    offsets = dict(zip(CONDS,np.linspace(-.26,.26,len(CONDS))))
    for ax,cat in zip(axes.flat,CATS):
        q=z[z.category.eq(cat)]
        labels=sorted(q.condition.unique())
        base=np.arange(len(labels))
        for cond in CONDS:
            w=q[q.intervention.eq(cond)].set_index('condition').reindex(labels)
            y=base+offsets[cond]
            x=w.mean_preference.to_numpy(); lo=w.lower.to_numpy(); hi=w.upper.to_numpy()
            ax.errorbar(x,y,xerr=np.vstack([x-lo,hi-x]),fmt='o',ms=4.5,capsize=2,lw=1.1,color=COLORS[cond],label=COND_LABELS[cond])
        ax.axvline(.5,color='.55',ls=':',lw=1); ax.set_xlim(0,1); ax.set_yticks(base,labels); ax.invert_yaxis(); ax.grid(axis='x',alpha=.22)
        ax.set_title(CAT_LABELS[cat],loc='left',fontweight='bold'); ax.set_xlabel('P(overt preferred), mean ± 1 SE')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.005))
    cap=int(z.common_final_tokens.iloc[0]); nmin=int(z.n_cells.min()); nmax=int(z.n_cells.max())
    fig.suptitle(f"{ARCH_LABELS[arch]}: truncated end-state preference by evaluation condition\nLast observed checkpoint ≤ {cap/1e9:.3f}B tokens; matched HP-rank/seed cells; n={nmin}–{nmax}/estimate",fontsize=15,fontweight='bold',y=.995)
    fig.tight_layout(rect=(0,.05,1,.96)); OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/f"endstate_forest_{arch}.png"; fig.savefig(p,dpi=180,bbox_inches='tight'); plt.close(fig)
    z.to_csv(p.with_suffix('.csv'),index=False)
    return p


def main():
    caps=pd.read_csv(SUITE/'final_token_cutoffs.csv')
    cells=endpoint_rows(caps)
    OUT.mkdir(parents=True,exist_ok=True)
    cells.to_parquet(OUT/'endstate_cell_values.parquet',index=False)
    s=summarize(cells); s.to_csv(OUT/'endstate_summary.csv',index=False)
    paths=[plot_arch(s,a) for a in ARCHES]
    (OUT/'README.md').write_text(
        "# Truncated end-state forest plots\n\nEach point is the mean binary length-normalized overt-preference score at the last observed checkpoint at or below the architecture-specific common token horizon; no interpolation is performed. The horizon is the smallest terminal token maximum across the five interventions. Only HP-rank/seed tuples that reach the horizon in all five interventions are eligible, preserving a matched comparison. Estimates collapse equally across those cells, with horizontal bars showing ±1 SE.\n"
    )
    manifest_path=SUITE/'coverage_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        generated=manifest.setdefault('generated',[])
        for p in paths:
            rel=str(p.relative_to(SUITE))
            if rel not in generated: generated.append(rel)
        manifest['endstate_forests']={
            'common_horizon':'minimum intervention-specific terminal maximum within architecture',
            'endpoint':'last observed checkpoint at or below the horizon; no interpolation',
            'eligibility':'matched hp-rank/seed tuples reaching the horizon in all five interventions',
        }
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    suite_readme=SUITE/'README.md'
    text=suite_readme.read_text() if suite_readme.exists() else ''
    if '## End-state forests' not in text:
        suite_readme.write_text(text.rstrip()+"\n\n## End-state forests\n\nSee `endstate_forests/` for architecture-specific, token-matched intervention comparisons across all 24 evaluation conditions.\n")
    print('\n'.join(str(p) for p in paths))
    print(s.groupby('architecture').agg(rows=('condition','size'),min_n=('n_cells','min'),max_n=('n_cells','max'),cap=('common_final_tokens','first')).to_string())

if __name__=='__main__': main()
