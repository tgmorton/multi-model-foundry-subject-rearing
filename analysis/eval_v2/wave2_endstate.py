"""End-state overt-preference by decile and intervention, for any set of
selection arms against the random control. Pilot cohort (gpt2_small,
condition-matched stimuli).

End state = mean of the last N checkpoints (default 3) of each run, which
damps the checkpoint-to-checkpoint jitter measured at ~0.06 without
smoothing across the training trajectory.

Usage: python analysis/eval_v2/wave2_endstate.py [--hp 0] [--last 3]
         [--arms rand,bert,bertanti,robbi,robbianti]
"""
from __future__ import annotations
import argparse, io, re
from pathlib import Path
import numpy as np, pandas as pd

CELL = re.compile(r"pdrop2_(gpt2m|bertanti|bert|comp|robbianti|robbi|rand|all100)(\d*)_(\w+)-h(\d)-")
IVS = ["base", "impcase", "lemverb", "enrichvm"]
IV_LABEL = {"base": "baseline", "impcase": "impoverish case",
            "lemverb": "lemmatize verbs", "enrichvm": "enrich verbal morph"}
TEAL, SIENNA, GREY, INDIGO, OCHRE = "#2C6E63", "#B0562B", "#8A8878", "#4B5FA8", "#A8823A"
# arm -> (color, linestyle, marker, legend label)
ARM_STYLE = {
    "rand": (GREY, "-", "s", "random (control)"),
    "bert": (TEAL, "-", "o", "most recoverable first — BERT 250:1"),
    "bertanti": (TEAL, "--", "v", "least recoverable first — BERT 250:1"),
    "robbi": (INDIGO, "-", "o", "most recoverable first — RoBERTa ±250"),
    "robbianti": (INDIGO, "--", "v", "least recoverable first — RoBERTa ±250"),
    "gpt2m": (OCHRE, "-", "D", "most recoverable first — gpt2-medium"),
    "comp": (SIENNA, ":", "P", "most recoverable first — composite"),
}

CACHE = Path.home() / ".cache" / "subject-drop" / "pairs_cache"


def load(bucket, prefix, profile, endpoint, last):
    """Parallel, retrying download with a local cache (valid while the S3
    object size is unchanged) — the external endpoint times out on long
    serial pulls of ~500 parquets (2026-09-29)."""
    import boto3
    from botocore.config import Config
    from concurrent.futures import ThreadPoolExecutor
    s = boto3.Session(profile_name=profile).client(
        "s3", endpoint_url=endpoint,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                      read_timeout=120, max_pool_connections=16))
    CACHE.mkdir(parents=True, exist_ok=True)
    objs = []
    for page in s.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            if "pdrop2" in o["Key"] and CELL.search(o["Key"]):
                objs.append(o)

    def fetch(o):
        p = CACHE / o["Key"].rsplit("/", 1)[-1]
        if not (p.exists() and p.stat().st_size == o["Size"]):
            s.download_file(bucket, o["Key"], str(p))
        return o["Key"], p

    with ThreadPoolExecutor(16) as ex:
        paths = list(ex.map(fetch, objs))
    rows = []
    for key, p in paths:
        arm, k, iv, hp = CELL.search(key).groups()
        d = pd.read_parquet(p, columns=["checkpoint_step", "prefers_overt_meanlp"])
        c = d[d.checkpoint_step > 0].groupby("checkpoint_step").prefers_overt_meanlp.mean()
        if len(c) < 20: continue
        rows.append({"arm": "all100" if arm == "all100" else arm,
                     "k": 100 if arm == "all100" else int(k),
                     "iv": iv, "hp": int(hp), "endstate": c.sort_index().iloc[-last:].mean()})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="thomas-subject-drop-artifacts")
    ap.add_argument("--prefix", default="eval_results/null_subj_v2_condition_matched_v1/pairs/")
    ap.add_argument("--profile", default="nrp")
    ap.add_argument("--endpoint", default="https://s3-west.nrp-nautilus.io")
    ap.add_argument("--hp", type=int, default=0)
    ap.add_argument("--last", type=int, default=3)
    ap.add_argument("--arms", default="rand,bert",
                    help="comma list; rand is the control the deltas use")
    ap.add_argument("--out", type=Path, default=Path("analysis/eval_v2/figures/wave2_v5"))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    arms = [x for x in a.arms.split(",") if x]

    df = load(a.bucket, a.prefix, a.profile, a.endpoint, a.last)
    df.to_csv(a.out / "endstate_all.csv", index=False)
    print(f"{len(df)} runs | arms {sorted(df.arm.unique())} | hp {sorted(df.hp.unique())}")
    d = df[df.hp == a.hp]
    print(f"hp={a.hp}: {len(d)} runs")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ivs = [i for i in IVS if i in set(d.iv)]
    fig, axes = plt.subplots(1, len(ivs), figsize=(3.6*len(ivs), 4.1), sharey=True)
    for ax, iv in zip(np.atleast_1d(axes), ivs):
        sub = d[d.iv == iv]
        for arm in arms:
            c, ls, mk, lab = ARM_STYLE[arm]
            t = sub[(sub.arm == arm) & (sub.k < 100)].groupby("k").endstate.mean().sort_index()
            if len(t): ax.plot(t.index, t.values, linestyle=ls, marker=mk, color=c,
                               ms=5, lw=1.8, label=lab)
        anc = sub[sub.arm == "all100"]
        if len(anc):
            ax.plot([100], [anc.endstate.mean()], "*", color=SIENNA, ms=14,
                    label="all pronouns removed")
        ax.axhline(0.5, color="#ccc", lw=.7)
        ax.set_title(IV_LABEL.get(iv, iv), fontsize=10)
        ax.set_xlabel("% of pronouns removed (decile)")
        ax.set_xticks([10,30,50,70,90,100])
    np.atleast_1d(axes)[0].set_ylabel("end-state overt-subject preference")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 3), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.10 if len(l) > 3 else -0.04))
    fig.suptitle("End-state preference by removal depth, by selection arm "
                 f"(gpt2_small, h{a.hp}, mean of last {a.last} checkpoints)", y=1.02)
    fig.tight_layout()
    tag = "" if arms == ["rand", "bert"] else "_" + "-".join(arms)
    fig.savefig(a.out / f"endstate_decile_arm{tag}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}/endstate_decile_arm{tag}.png")

    for arm in [x for x in arms if x != "rand"]:
        print(f"\n== {arm} minus random, by intervention (mean over matched deciles) ==")
        for iv in ivs:
            sub = d[d.iv == iv]
            b = sub[sub.arm == arm].groupby("k").endstate.mean()
            r = sub[sub.arm == "rand"].groupby("k").endstate.mean()
            j = (b - r).dropna()
            if len(j): print(f"  {IV_LABEL.get(iv,iv):22s} mean Δ {j.mean():+.3f} | "
                             f"range {j.min():+.3f}..{j.max():+.3f} | n_k={len(j)}")

if __name__ == "__main__":
    main()
