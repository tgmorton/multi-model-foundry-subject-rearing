#!/usr/bin/env python3
"""PDF report: pronoun-free vs original contexts, plus tentative cross-construction
priming, for the wave2 gpt2_large h0 cohort (140 runs).

Sources (S3, thomas-subject-drop-artifacts):
  eval_results/null_subj_v2_condition_matched_v1/       original contexts (matched-v1)
  eval_results/null_subj_v2_matched_nopron_v1/          pronoun-free contexts
  eval_results/null_subj_v2_nopron_plus_priming_v1/     priming (+ pronoun-free) as
                                                        <run>__ckpt_shard{i}of4 cells
Stimuli: evaluation/stimuli/null-subj-v2-nopron/ (context_rewrites.csv),
         evaluation/stimuli/null-subj-v2-priming-nopron-v1/pairing.csv

End-state = mean prefers_overt_meanlp over a run's last 3 checkpoints (step > 0), as
in wave2_endstate.py. Raw pair files are reduced to per-checkpoint means and cached
under ~/.cache/subject-drop/report_cache/.

    python analysis/eval_v2/nopron_priming_report.py --out report.pdf
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BUCKET = "thomas-subject-drop-artifacts"
ORIG = "null_subj_v2_condition_matched_v1"
NOPRON = "null_subj_v2_matched_nopron_v1"
COMB = "null_subj_v2_nopron_plus_priming_v1"
CACHE = Path.home() / ".cache" / "subject-drop" / "report_cache"
RUN_RE = re.compile(r"gpt2_large-en-pdrop2_(?P<arm>[a-z]+?)(?P<k>\d+)_(?P<iv>[a-z]+)-h0-s\d+$")

TEAL, SIENNA, GREY, INDIGO, OCHRE = "#2C6E63", "#B0562B", "#8A8878", "#4B5FA8", "#A8823A"
ARM_STYLE = {"rand": (GREY, "-", "s", "random (control)"),
             "robbi": (INDIGO, "-", "o", "most recoverable first (robbi)"),
             "robbianti": (INDIGO, "--", "v", "least recoverable first (robbi)")}
IVS = ["base", "impcase", "lemverb", "enrichvm", "rmexpl"]
IV_LABEL = {"base": "baseline", "impcase": "impoverish case", "lemverb": "lemmatize verbs",
            "enrichvm": "enrich verbal morph", "rmexpl": "remove expletives"}
PERSONS = ["subj_1sg", "subj_1pl", "subj_2sg", "subj_2pl", "subj_3sg", "subj_3pl"]
SUBJ = ["subject_drop", "subject_drop_no_agreement"]
CATS = ["subject_drop", "subject_drop_no_agreement", "conjunction", "embedded_drop",
        "control", "expletive", "extraction", "object_drop"]
CAT_SHORT = {"subject_drop": "subj drop", "subject_drop_no_agreement": "subj drop (no agr)",
             "conjunction": "conjunction", "embedded_drop": "embedded", "control": "control",
             "expletive": "expletive", "extraction": "extraction", "object_drop": "object drop"}
ORIG_C, NOPRON_C = INDIGO, SIENNA


def s3c():
    import boto3
    from botocore.config import Config
    cfg = Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=120, connect_timeout=30)
    return boto3.Session(profile_name="nrp").client("s3", endpoint_url="https://s3-west.nrp-nautilus.io", config=cfg)


def list_cells(s3, bench, table="pairs"):
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"eval_results/{bench}/{table}/"):
        out += [o["Key"].split("cell_id=")[-1][: -len(".parquet")] for o in page.get("Contents", [])]
    return out


def get_parquet(s3, bench, table, cell, columns=None, attempts=5):
    import time
    key = f"eval_results/{bench}/{table}/cell_id={cell}.parquet"
    for i in range(attempts):
        try:
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            return pd.read_parquet(io.BytesIO(body), columns=columns)
        except Exception as e:  # noqa: BLE001  read timeouts mid-body are not retried by botocore
            if i == attempts - 1:
                raise RuntimeError(f"{key}: {e}") from e
            time.sleep(2 ** i)


# ---------------------------------------------------------------------------
# Loading (reduced to per-checkpoint means, cached)
# ---------------------------------------------------------------------------
def load_matched(s3, bench, runs):
    """Per (run, step, category, condition) mean prefers_overt_meanlp."""
    f = CACHE / f"{bench}__agg.parquet"
    if f.exists():
        return pd.read_parquet(f)

    def one(run):
        d = get_parquet(s3, bench, "pairs", run, ["checkpoint_step", "category", "condition", "prefers_overt_meanlp"])
        g = d.groupby(["checkpoint_step", "category", "condition"]).prefers_overt_meanlp.mean().reset_index()
        g["run"] = run
        return g
    with ThreadPoolExecutor(16) as ex:
        out = pd.concat(list(ex.map(one, runs)), ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f)
    return out


def load_priming(s3, cells, pairing):
    """Per (run, step, target cat/cond, prime cat/cond/status) mean preference, from
    the 4-shard combined cells. Pronoun-free rows ride along (cross-check)."""
    cdir = CACHE / f"{COMB}__cells"
    cdir.mkdir(parents=True, exist_ok=True)
    pm = pairing.set_index("item_id")

    def one(cell):
        cf = cdir / f"{cell}.parquet"
        if cf.exists():           # per-cell cache: a rebuild fetches only new shards
            return pd.read_parquet(cf)
        d = get_parquet(s3, COMB, "pairs", cell, ["checkpoint_step", "category", "condition", "item_id", "prefers_overt_meanlp"])
        run = cell.rsplit("__ckpt_shard", 1)[0]
        pr = d.category.str.contains("__p_")
        a = d[~pr].groupby(["checkpoint_step", "category", "condition"]).prefers_overt_meanlp.mean().reset_index()
        a["kind"] = "nopron"
        p = d[pr].copy()
        p["prime_condition"] = pm.loc[p.item_id.astype(int), "prime_condition"].to_numpy()
        p["target_category"] = p.category.str.split("__p_").str[0]
        p["prime_category"] = p.category.str.split("__p_").str[1]
        p["target_condition"] = p.condition.str.rsplit("__", n=1).str[0]
        p["prime_status"] = p.condition.str.rsplit("__", n=1).str[1]
        b = (p.groupby(["checkpoint_step", "target_category", "target_condition", "prime_category",
                        "prime_condition", "prime_status"]).prefers_overt_meanlp.agg(["mean", "size"]).reset_index())
        b["kind"] = "priming"
        out = pd.concat([a, b], ignore_index=True)
        out["run"], out["cell"] = run, cell
        out.to_parquet(cf)
        return out
    with ThreadPoolExecutor(12) as ex:
        return pd.concat(list(ex.map(one, cells)), ignore_index=True)


def parse_runs(df, col="run"):
    m = df[col].str.extract(RUN_RE)
    m["k"] = m.k.astype(int)
    return pd.concat([df.reset_index(drop=True), m.reset_index(drop=True)], axis=1)


def endstate(agg, keys, value="prefers_overt_meanlp", last=3):
    """Mean of each run's last `last` checkpoints (step > 0)."""
    a = agg[agg.checkpoint_step > 0]
    laststeps = (a.groupby("run").checkpoint_step.apply(lambda s: sorted(s.unique())[-last:]).rename("ls"))
    a = a.merge(laststeps, left_on="run", right_index=True)
    a = a[[st in ls for st, ls in zip(a.checkpoint_step, a.ls)]]
    return a.groupby(["run", *keys])[value].mean().reset_index()


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def text_page(pdf, title, blocks, plt):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.07, 0.95, title, fontsize=16, weight="bold", va="top")
    y = 0.90
    for kind, txt in blocks:
        if kind == "h":
            y -= 0.012
            fig.text(0.07, y, txt, fontsize=11.5, weight="bold", va="top")
            y -= 0.03
        else:
            for para in txt.split("\n"):
                lines = textwrap.wrap(para, 98, subsequent_indent="   " if para.startswith("•") else "") or [""]
                for ln in lines:
                    fig.text(0.07, y, ln, fontsize=9.3, va="top", family="DejaVu Sans")
                    y -= 0.0185
            y -= 0.006
    pdf.savefig(fig)
    plt.close(fig)


def stimuli_page(pdf, plt):
    rw = list(csv.DictReader(open(REPO / "evaluation/stimuli/null-subj-v2-nopron/context_rewrites.csv")))
    staging = {}
    for f in sorted((REPO / "evaluation/stimuli/null-subj-v2/staging/en").glob("*.csv")):
        for r in csv.DictReader(open(f)):
            staging.setdefault(r["context"], {})[r["pronoun_status"]] = (r["category"], r["condition"], r["target"])
    want = [("subj_2sg", "subject_drop"), ("subj_2pl", "subject_drop"), ("subj_2sg", "subject_drop_no_agreement"),
            ("subj_1sg", "subject_drop"), ("subj_1pl", "subject_drop"), ("subj_3sg", "subject_drop"),
            ("conj_shift", "conjunction"), ("expl_be", "expletive"), ("obj_3sg", "object_drop")]
    rows = []
    for cond, cat in want:
        hits = [r for r in rw if cond in r["conditions"].split(";") and cat in r["categories"].split(";")]
        for r in hits[:2]:
            t = staging.get(r["original_context"], {})
            rows.append((f"{CAT_SHORT[cat]} / {cond}", r["original_context"], r["new_context"],
                         t.get("1", ("", "", ""))[2], t.get("0", ("", "", ""))[2]))
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.96, "Stimuli: original vs pronoun-free contexts", fontsize=15, weight="bold", va="top")
    fig.text(0.06, 0.935, "Targets are identical in both sets; only the context changes. "
                          "Every pronoun or pronoun-capable word is removed from contexts.", fontsize=9, va="top")
    y = 0.90
    for lab, o, n, to, tn in rows:
        fig.text(0.06, y, lab, fontsize=8.6, weight="bold", color=GREY, va="top"); y -= 0.017
        fig.text(0.08, y, f"original:      {o}", fontsize=8.6, color=ORIG_C, va="top"); y -= 0.016
        fig.text(0.08, y, f"pronoun-free:  {n}", fontsize=8.6, color=NOPRON_C, va="top"); y -= 0.016
        fig.text(0.08, y, f"targets:       {to}   |   {tn}", fontsize=8.2, color="#333", va="top"); y -= 0.024
        if y < 0.06:
            break
    pdf.savefig(fig)
    plt.close(fig)


def fig_person(pdf, es, plt):
    sd = es[es.category.isin(SUBJ)]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(PERSONS))
    for j, (bench, col, lab) in enumerate([("orig", ORIG_C, "original context"), ("nopron", NOPRON_C, "pronoun-free context")]):
        per_run = sd.groupby(["run", "condition"])[bench].mean().unstack()
        m = per_run.mean()[PERSONS]; e = per_run.apply(ci)[PERSONS]
        ax.bar(x + (j - 0.5) * 0.38, m, 0.36, yerr=e, color=col, alpha=0.9 if j == 0 else 0.75,
               label=lab, capsize=3, error_kw={"lw": 0.8})
    ax.set_xticks(x, ["1sg*", "1pl*", "2sg", "2pl", "3sg", "3pl"])
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#ccc", lw=0.7)
    ax.set_ylabel("end-state P(prefer overt subject)")
    ax.set_title("Subject drop (with and without agreement), by target person/number\n"
                 "140 gpt2_large runs, mean ± 95% CI over runs", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(0.07, 0.01, "* 1st person: the pronoun-free context cannot mention the speaker, so this contrast is "
                         "'speaker mentioned vs not' and is excluded from the pronoun-effect analysis.", fontsize=7.8, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def fig_categories(pdf, es, plt):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(CATS))
    for j, (bench, col, lab) in enumerate([("orig", ORIG_C, "original context"), ("nopron", NOPRON_C, "pronoun-free context")]):
        d = es[~es.condition.isin(["subj_1sg", "subj_1pl"])]
        per_run = d.groupby(["run", "category"])[bench].mean().unstack()
        m = per_run.mean()[CATS]; e = per_run.apply(ci)[CATS]
        ax.bar(x + (j - 0.5) * 0.38, m, 0.36, yerr=e, color=col, alpha=0.9 if j == 0 else 0.75, label=lab,
               capsize=3, error_kw={"lw": 0.8})
    ax.set_xticks(x, [CAT_SHORT[c] for c in CATS], rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("end-state P(prefer overt)")
    ax.set_title("All categories (1st-person items excluded). Only items whose context had a pronoun change;\n"
                 "categories with few rewritten contexts barely move.", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def fig_order(pdf, es, plt, value_label, mask_fn, title):
    d = parse_runs(es)
    d = d[mask_fn(d)]
    per = d.groupby(["run", "arm", "k", "iv"])[["orig", "nopron"]].mean().reset_index()
    ivs = [i for i in IVS if i in set(per.iv)]
    fig, axes = plt.subplots(2, len(ivs), figsize=(3.0 * len(ivs), 6.4), sharey=True, squeeze=False)
    for r, (bench, lab) in enumerate([("orig", "original context"), ("nopron", "pronoun-free context")]):
        for c, iv in enumerate(ivs):
            ax = axes[r][c]
            sub = per[per.iv == iv]
            for arm, (col, ls, mk, alab) in ARM_STYLE.items():
                t = sub[(sub.arm == arm) & (sub.k < 100)].groupby("k")[bench].mean().sort_index()
                if len(t):
                    ax.plot(t.index, t.values, ls=ls, marker=mk, color=col, ms=4, lw=1.6, label=alab)
            anc = sub[sub.arm == "all"]
            if len(anc):
                ax.plot([100], [anc[bench].mean()], "*", color=SIENNA, ms=12, label="all pronouns removed")
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="#ccc", lw=0.6)
            ax.set_xticks([10, 30, 50, 70, 90, 100])
            ax.tick_params(labelsize=7.5)
            if r == 0:
                ax.set_title(IV_LABEL[iv], fontsize=9.5)
            if r == 1:
                ax.set_xlabel("% pronouns removed", fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"{lab}\n{value_label}", fontsize=8.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def fig_delta(pdf, es, plt):
    d = parse_runs(es)
    d = d[d.category.isin(SUBJ) & d.condition.isin(["subj_2sg", "subj_2pl"])]
    per = d.groupby(["run", "arm", "k", "iv"])[["orig", "nopron"]].mean().reset_index()
    per["delta"] = per.nopron - per.orig
    ivs = [i for i in IVS if i in set(per.iv)]
    fig, axes = plt.subplots(1, len(ivs), figsize=(3.0 * len(ivs), 3.6), sharey=True, squeeze=False)
    lo = min(per.delta.min(), -0.05) - 0.03
    for c, iv in enumerate(ivs):
        ax = axes[0][c]
        sub = per[per.iv == iv]
        for arm, (col, ls, mk, alab) in ARM_STYLE.items():
            t = sub[(sub.arm == arm) & (sub.k < 100)].groupby("k").delta.mean().sort_index()
            if len(t):
                ax.plot(t.index, t.values, ls=ls, marker=mk, color=col, ms=4, lw=1.6, label=alab)
        anc = sub[sub.arm == "all"]
        if len(anc):
            ax.plot([100], [anc.delta.mean()], "*", color=SIENNA, ms=12, label="all pronouns removed")
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_ylim(lo, 0.05)
        ax.set_title(IV_LABEL[iv], fontsize=9.5)
        ax.set_xticks([10, 30, 50, 70, 90, 100])
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("% pronouns removed", fontsize=8.5)
    axes[0][0].set_ylabel("Δ P(prefer overt)\npronoun-free − original", fontsize=8.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Effect of removing the context pronoun on 2nd-person targets (end-state)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def fig_trajectories(pdf, orig, nopron, tokens, plt):
    def two(agg):
        d = agg[agg.category.isin(SUBJ) & agg.condition.isin(["subj_2sg", "subj_2pl"]) & (agg.checkpoint_step > 0)]
        return d.groupby(["run", "checkpoint_step"]).prefers_overt_meanlp.mean().reset_index()
    o, n = two(orig), two(nopron)
    j = o.merge(n, on=["run", "checkpoint_step"], suffixes=("_o", "_n")).merge(tokens, on=["run", "checkpoint_step"], how="left")
    j = parse_runs(j)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.7), sharey=True)
    for ax, arm in zip(axes, ["rand", "robbi", "robbianti", "all"]):
        sub = j[j.arm == arm].dropna(subset=["tokens_seen"])
        if sub.empty:
            continue
        bins = np.unique(np.quantile(np.log10(sub.tokens_seen), np.linspace(0, 1, 30)))
        sub = sub.assign(b=np.digitize(np.log10(sub.tokens_seen), bins))
        g = sub.groupby("b").agg(t=("tokens_seen", "median"), o=("prefers_overt_meanlp_o", "mean"),
                                 n=("prefers_overt_meanlp_n", "mean"))
        ax.plot(g.t, g.o, color=ORIG_C, lw=1.8, label="original context")
        ax.plot(g.t, g.n, color=NOPRON_C, lw=1.8, ls="--", label="pronoun-free context")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="#ccc", lw=0.6)
        ax.set_title({"rand": "random", "robbi": "most recoverable first", "robbianti": "least recoverable first",
                      "all": "all pronouns removed"}[arm] + f"  (n={sub.run.nunique()})", fontsize=9)
        ax.set_xlabel("training tokens", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("P(prefer overt), 2nd-person targets", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("When does the dependence on the context pronoun emerge? (all interventions and removal levels pooled)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, es, plt):
    d = parse_runs(es)
    two = d[d.category.isin(SUBJ) & d.condition.isin(["subj_2sg", "subj_2pl"])]

    def rowfmt(name, g):
        per = g.groupby("run")[["orig", "nopron"]].mean()
        dd = per.nopron - per.orig
        return [name, f"{per.orig.mean():.3f}", f"{per.nopron.mean():.3f}", f"{dd.mean():+.3f}", f"{dd.std(ddof=1)/np.sqrt(len(dd)):.3f}", str(len(dd))]
    rows = [rowfmt(c, d[d.category.isin(SUBJ) & (d.condition == c)]) for c in PERSONS]
    rows += [["", "", "", "", "", ""]]
    rows += [rowfmt(f"2nd person — {IV_LABEL[iv]}", two[two.iv == iv]) for iv in IVS if (two.iv == iv).any()]
    rows += [["", "", "", "", "", ""]]
    for arm, lab in [("rand", "random"), ("robbi", "most recoverable first"), ("robbianti", "least recoverable first"), ("all", "all removed")]:
        if (two.arm == arm).any():
            rows.append(rowfmt(f"2nd person — {lab}", two[two.arm == arm]))
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.07, 0.95, "Numbers: end-state P(prefer overt)", fontsize=15, weight="bold", va="top")
    ax = fig.add_axes([0.07, 0.25, 0.86, 0.66])
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["condition / group", "original", "pronoun-free", "Δ", "SE(Δ)", "runs"],
                 loc="upper center", cellLoc="right", colLoc="right")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.35)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#ddd")
        if c == 0:
            cell.set_text_props(ha="left")
        if r == 0:
            cell.set_text_props(weight="bold")
    fig.text(0.07, 0.2, "SE(Δ) = standard error of the per-run difference (run = one trained model). "
                        "Rows for 1sg/1pl carry the speaker-not-mentioned caveat.", fontsize=8, color=GREY)
    pdf.savefig(fig)
    plt.close(fig)


# --- priming ---------------------------------------------------------------
def priming_endstate(pr_agg, complete_runs):
    p = pr_agg[(pr_agg.kind == "priming") & pr_agg.run.isin(complete_runs)]
    # priming rows carry their value in "mean"; the frame also has the pronoun-free
    # rows' (here empty) prefers_overt_meanlp column, which must not survive the rename
    p = p.drop(columns=["prefers_overt_meanlp"], errors="ignore").rename(columns={"mean": "prefers_overt_meanlp"})
    return endstate(p, ["target_category", "target_condition", "prime_category", "prime_condition", "prime_status"])


def fig_priming_heatmap(pdf, pe, plt):
    w = pe.pivot_table(index=["run", "target_category", "prime_category"], columns="prime_status",
                       values="prefers_overt_meanlp", aggfunc="mean").reset_index()
    w["effect"] = w.overt - w.null
    g = w.groupby(["target_category", "prime_category"]).effect.mean().unstack()[CATS].loc[CATS]
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    lim = max(0.05, float(np.nanmax(np.abs(g.values))))
    im = ax.imshow(g.values, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(CATS)), [CAT_SHORT[c] for c in CATS], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(CATS)), [CAT_SHORT[c] for c in CATS], fontsize=8.5)
    ax.set_xlabel("prime construction")
    ax.set_ylabel("target construction")
    for i in range(len(CATS)):
        for j in range(len(CATS)):
            v = g.values[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 * lim else "#222")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("priming effect: P(overt | overt prime) − P(overt | null prime)", fontsize=8.5)
    ax.set_title(f"TENTATIVE — cross-construction priming, end-state ({w.run.nunique()} runs)", fontsize=10.5)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def fig_priming_levels(pdf, pe, nopron_es, plt):
    rows = []
    for st in ("overt", "null"):
        per = pe[pe.prime_status == st].groupby(["run", "target_category"]).prefers_overt_meanlp.mean().unstack()
        rows.append((f"after {st} prime", per))
    base = nopron_es[~nopron_es.condition.isin(["subj_1sg", "subj_1pl"])]
    base = base[base.run.isin(pe.run.unique())].groupby(["run", "category"]).nopron.mean().unstack()
    rows.insert(0, ("no prime (pronoun-free set)", base))
    cols = [TEAL, INDIGO, SIENNA]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(CATS))
    for j, ((lab, per), col) in enumerate(zip(rows, cols)):
        m = per.mean().reindex(CATS); e = per.apply(ci).reindex(CATS)
        ax.bar(x + (j - 1) * 0.27, m, 0.25, yerr=e, color=col, label=lab, capsize=2, error_kw={"lw": 0.7})
    ax.set_xticks(x, [CAT_SHORT[c] for c in CATS], rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("end-state P(prefer overt)")
    ax.set_title("TENTATIVE — target preference with no prime, after an overt prime, after a null prime\n"
                 "(pooled over all prime constructions)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def fig_priming_person(pdf, pe, plt):
    p = pe[pe.prime_category.isin(SUBJ)]
    w = p.pivot_table(index=["run", "target_category", "prime_condition"], columns="prime_status",
                      values="prefers_overt_meanlp", aggfunc="mean").reset_index()
    w["effect"] = w.overt - w.null
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(PERSONS))
    tcs = ["subject_drop", "subject_drop_no_agreement", "conjunction", "embedded_drop", "object_drop"]
    cols = [INDIGO, TEAL, OCHRE, SIENNA, GREY]
    for j, (tc, col) in enumerate(zip(tcs, cols)):
        per = w[w.target_category == tc].groupby(["run", "prime_condition"]).effect.mean().unstack()
        m = per.mean().reindex(PERSONS); e = per.apply(ci).reindex(PERSONS)
        ax.bar(x + (j - 2) * 0.16, m, 0.15, yerr=e, color=col, label=f"target: {CAT_SHORT[tc]}", capsize=2, error_kw={"lw": 0.6})
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(x, ["1sg†", "1pl†", "2sg†", "2pl†", "3sg", "3pl"])
    ax.set_xlabel("person/number of the subject-drop PRIME")
    ax.set_ylabel("priming effect (overt − null prime)")
    ax.set_title("TENTATIVE — priming by the person of a subject-drop prime", fontsize=10.5)
    ax.legend(frameon=False, fontsize=7.8, ncol=2)
    fig.text(0.07, 0.01, "† null primes from 1st/2nd-person items read as imperatives (the pronoun-free context does "
                         "not anchor the speaker; 2nd person is imperative-like anyway); 3rd person = clean drop primes.",
             fontsize=7.6, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def fig_priming_arms(pdf, pe, plt):
    w = pe.pivot_table(index=["run", "target_category"], columns="prime_status", values="prefers_overt_meanlp",
                       aggfunc="mean").reset_index()
    w["effect"] = w.overt - w.null
    w = parse_runs(w)
    per = w.groupby(["run", "arm", "iv"]).effect.mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    arms = [("rand", "random"), ("robbi", "most rec. first"), ("robbianti", "least rec. first"), ("all", "all removed")]
    for ax, (key, groups, lab) in zip(axes, [("arm", arms, "selection arm"),
                                             ("iv", [(i, IV_LABEL[i]) for i in IVS], "intervention")]):
        xs = [g for g, _ in groups if (per[key] == g).any()]
        m = [per[per[key] == g].effect.mean() for g in xs]
        e = [ci(per[per[key] == g].effect) for g in xs]
        ax.bar(range(len(xs)), m, yerr=e, color=INDIGO, alpha=0.85, capsize=3)
        ax.set_xticks(range(len(xs)), [dict(groups)[g] for g in xs], rotation=20, ha="right", fontsize=8.5)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_title(f"by {lab}", fontsize=10)
    axes[0].set_ylabel("priming effect (overt − null prime)\nall targets × primes pooled", fontsize=8.5)
    fig.suptitle("TENTATIVE — priming effect by selection arm and intervention", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refresh-priming", action="store_true", help="re-download the priming aggregates")
    ap.add_argument("--no-priming", action="store_true", help="pronoun-free part only (skips the priming download)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    s3 = s3c()
    nop_runs = sorted(c for c in list_cells(s3, NOPRON) if RUN_RE.search(c))
    orig_runs = set(list_cells(s3, ORIG))
    runs = [r for r in nop_runs if r in orig_runs]
    print(f"runs with both original and pronoun-free results: {len(runs)}")
    orig = load_matched(s3, ORIG, runs)
    nopron = load_matched(s3, NOPRON, runs)
    e_o = endstate(orig, ["category", "condition"]).rename(columns={"prefers_overt_meanlp": "orig"})
    e_n = endstate(nopron, ["category", "condition"]).rename(columns={"prefers_overt_meanlp": "nopron"})
    es = e_o.merge(e_n, on=["run", "category", "condition"])

    tf = CACHE / "tokens.parquet"
    if tf.exists():
        tokens = pd.read_parquet(tf)
    else:
        with ThreadPoolExecutor(16) as ex:
            parts = list(ex.map(lambda r: get_parquet(s3, ORIG, "checkpoints", r, ["checkpoint_step", "tokens_seen"]).assign(run=r), runs))
        tokens = pd.concat(parts, ignore_index=True)
        tokens.to_parquet(tf)

    comb_cells = [] if a.no_priming else sorted(c for c in list_cells(s3, COMB) if c.endswith("of4") and RUN_RE.search(c.rsplit("__ckpt_shard", 1)[0]))
    if a.refresh_priming:
        for f in (CACHE / f"{COMB}__cells").glob("*.parquet"):
            f.unlink()
    complete, maxdiff, pe = [], float("nan"), None
    if not a.no_priming:
        pairing = pd.read_csv(REPO / "evaluation/stimuli/null-subj-v2-priming-nopron-v1/pairing.csv")
        pr = load_priming(s3, comb_cells, pairing)
        shards = pd.Series([c.rsplit("__ckpt_shard", 1) for c in comb_cells])
        nshard = shards.map(lambda x: x[0]).value_counts()
        complete = sorted(nshard[nshard == 4].index)
        pe = priming_endstate(pr, complete)
        print(f"priming: {len(comb_cells)} shard cells, {len(complete)} runs with all 4 shards")
        # cross-check: pronoun-free rows inside the combined pass vs the standalone pronoun-free eval
        cx = pr[(pr.kind == "nopron")].merge(nopron, on=["run", "checkpoint_step", "category", "condition"], suffixes=("_c", "_s"))
        maxdiff = float((cx.prefers_overt_meanlp_c - cx.prefers_overt_meanlp_s).abs().max()) if len(cx) else float("nan")

    d2 = es[es.category.isin(SUBJ) & es.condition.isin(["subj_2sg", "subj_2pl"])]
    per2 = d2.groupby("run")[["orig", "nopron"]].mean()
    dd = per2.nopron - per2.orig
    if pe is not None:
        w = pe.pivot_table(index=["run"], columns="prime_status", values="prefers_overt_meanlp", aggfunc="mean")
        pe_all = (w.overt - w.null)
        priming_summary = [
            ("h", "2. Cross-construction priming (TENTATIVE)"),
            ("t", f"• {len(complete)} of {len(runs)} runs have all 4 checkpoint shards scored; the rest are still running. "
                  f"Pooled priming effect (overt − null prime): {pe_all.mean():+.3f} (SE {pe_all.std(ddof=1)/np.sqrt(len(pe_all)):.3f}).\n"
                  "• Design: prime item (context + overt or null sentence) followed by a target item from any construction; "
                  "prime and target share no content word or name; condition-matched; contexts pronoun-free.\n"
                  "• Null primes from 1st/2nd-person items read as imperatives; stratify before interpreting person effects."),
        ]
        checks = (f"• Pronoun-free rows scored inside the combined pass match the standalone pronoun-free eval "
                  f"(max |difference| in per-checkpoint means: {maxdiff:.2g}).\n")
    else:
        priming_summary = [("h", "2. Cross-construction priming"),
                           ("t", "• Not in this version: the combined pronoun-free + priming eval is still finishing; "
                                 "a follow-up version adds the tentative priming section.")]
        checks = ""

    with PdfPages(a.out) as pdf:
        text_page(pdf, "Pronoun-free contexts and cross-construction priming — gpt2_large", [
            ("t", f"wave2 gpt2_large, h0, {len(runs)} trained models (5 interventions × random / most-recoverable-first / "
                  "least-recoverable-first (robbi) arms × removal levels, plus all-removed). End-state = mean of each "
                  "model's last 3 checkpoints. Preference = P(model prefers the overt-pronoun member of a minimal pair)."),
            ("h", "1. Pronoun-free contexts (complete: all runs, all checkpoints)"),
            ("t", f"• Removing the pronoun from the context lowers end-state overt preference on 2nd-person targets by "
                  f"{dd.mean():+.3f} (SE {dd.std(ddof=1)/np.sqrt(len(dd)):.3f}; {per2.orig.mean():.3f} → {per2.nopron.mean():.3f}).\n"
                  "• Targets whose context did not change (3rd person, and the other categories, where few contexts had a "
                  "pronoun) do not move: the effect is confined to the manipulated items.\n"
                  "• The dependence is largest in models trained with their least-recoverable pronouns removed and smallest "
                  "under rich verbal agreement (enrich verbal morphology).\n"
                  "• Caveat: in the 2nd-person rewrites the addressee is no longer the subject/agent of the context ('ana , "
                  "the keys go missing often .'), so the contrast is 'addressee mentioned as a pronoun subject vs only by "
                  "vocative'. 1st-person contexts cannot mention the speaker without a pronoun; they are shown but excluded "
                  "from the contrast."),
            *priming_summary,
            ("h", "Data checks"),
            ("t", checks + "• Stimuli: evaluation/stimuli/null-subj-v2-nopron (197 rewritten contexts, vetted 2026-10-22) and "
                  "null-subj-v2-priming-nopron-v1 (4,608 pairings)."),
        ], plt)
        stimuli_page(pdf, plt)
        fig_person(pdf, es, plt)
        fig_categories(pdf, es, plt)
        fig_order(pdf, es, plt, "P(prefer overt), 2nd-person targets",
                  lambda d: d.category.isin(SUBJ) & d.condition.isin(["subj_2sg", "subj_2pl"]),
                  "Order effect on 2nd-person targets: original (top) vs pronoun-free (bottom) contexts")
        fig_order(pdf, es, plt, "P(prefer overt), overall",
                  lambda d: ~d.category.isin(["extraction", "object_drop"]) & ~d.condition.isin(["subj_1sg", "subj_1pl"]),
                  "Order effect, overall (excl. extraction, object drop, 1st person): original (top) vs pronoun-free (bottom)")
        fig_delta(pdf, es, plt)
        fig_trajectories(pdf, orig, nopron, tokens, plt)
        table_page(pdf, es, plt)
        if pe is None:
            return print(f"wrote {a.out} (no priming)")
        text_page(pdf, "Part 2 — cross-construction priming (TENTATIVE)", [
            ("t", f"Status: {len(complete)} of {len(runs)} runs complete (all 4 checkpoint shards); the combined pronoun-free + "
                  "priming eval is still finishing the remainder. Numbers below use complete runs only and may shift."),
            ("h", "Design"),
            ("t", "• Item = <prime context> <prime sentence: overt or null> <target context> <target pair>.\n"
                  "• Every target pair gets one prime from each of the 8 constructions (4,608 pairings per condition); prime "
                  "and target share no content lemma or name; prime person labels are balanced.\n"
                  "• Priming effect = target P(prefer overt) after an overt prime minus after a null prime (same target, same "
                  "prime item, one token different).\n"
                  "• 'No prime' reference = the same targets in the pronoun-free matched set (shorter input, no prime)."),
            ("h", "Reading notes"),
            ("t", "• Extraction and control alternations are not argument drop ('that' vs ∅, 'him' vs ∅); as primes they test "
                  "overt-vs-omitted crossover across very different constructions.\n"
                  "• Null primes from 1st/2nd-person items are imperative-ambiguous; the clean drop primes are 3rd person, "
                  "conjunction and embedded clauses."),
        ], plt)
        fig_priming_heatmap(pdf, pe, plt)
        fig_priming_levels(pdf, pe, es, plt)
        fig_priming_person(pdf, pe, plt)
        fig_priming_arms(pdf, pe, plt)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
