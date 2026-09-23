"""End-state overt-preference by decile and intervention: informed (bert)
vs random arm. Pilot cohort (gpt2_small, condition-matched stimuli).

End state = mean of the last N checkpoints (default 3) of each run, which
damps the checkpoint-to-checkpoint jitter measured at ~0.06 without
smoothing across the training trajectory.

Usage: python analysis/eval_v2/wave2_endstate.py [--hp 0] [--last 3]
"""
from __future__ import annotations
import argparse, io, re
from pathlib import Path
import numpy as np, pandas as pd

CELL = re.compile(r"pdrop2_(gpt2m|bert|comp|rand|all100)(\d*)_(\w+)-h(\d)-")
IVS = ["base", "impcase", "lemverb", "enrichvm"]
IV_LABEL = {"base": "baseline", "impcase": "impoverish case",
            "lemverb": "lemmatize verbs", "enrichvm": "enrich verbal morph"}

def load(bucket, prefix, profile, endpoint, last):
    import boto3
    s = boto3.Session(profile_name=profile).client("s3", endpoint_url=endpoint)
    rows = []
    for page in s.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            if "pdrop2" not in o["Key"]: continue
            m = CELL.search(o["Key"])
            if not m: continue
            arm, k, iv, hp = m.group(1), m.group(2), m.group(3), int(m.group(4))
            d = pd.read_parquet(io.BytesIO(s.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()),
                                columns=["checkpoint_step", "prefers_overt_meanlp"])
            c = d[d.checkpoint_step > 0].groupby("checkpoint_step").prefers_overt_meanlp.mean()
            if len(c) < 20: continue
            rows.append({"arm": "all100" if arm == "all100" else arm,
                         "k": 100 if arm == "all100" else int(k),
                         "iv": iv, "hp": hp, "endstate": c.sort_index().iloc[-last:].mean()})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="thomas-subject-drop-artifacts")
    ap.add_argument("--prefix", default="eval_results/null_subj_v2_condition_matched_v1/pairs/")
    ap.add_argument("--profile", default="nrp")
    ap.add_argument("--endpoint", default="https://s3-west.nrp-nautilus.io")
    ap.add_argument("--hp", type=int, default=0)
    ap.add_argument("--last", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("analysis/eval_v2/figures/wave2_v5"))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)

    df = load(a.bucket, a.prefix, a.profile, a.endpoint, a.last)
    df.to_csv(a.out / "endstate_all.csv", index=False)
    print(f"{len(df)} runs | arms {sorted(df.arm.unique())} | hp {sorted(df.hp.unique())}")
    d = df[df.hp == a.hp]
    print(f"hp={a.hp}: {len(d)} runs")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    TEAL, SIENNA, GREY = "#2C6E63", "#B0562B", "#8A8878"
    ivs = [i for i in IVS if i in set(d.iv)]
    fig, axes = plt.subplots(1, len(ivs), figsize=(3.5*len(ivs), 3.9), sharey=True)
    for ax, iv in zip(np.atleast_1d(axes), ivs):
        sub = d[d.iv == iv]
        for arm, c, lab, mk in (("rand", GREY, "random (control)", "s"),
                                ("bert", TEAL, "informed (BERT rater)", "o")):
            t = sub[(sub.arm == arm) & (sub.k < 100)].groupby("k").endstate.mean().sort_index()
            if len(t): ax.plot(t.index, t.values, "-"+mk, color=c, ms=5, lw=1.8, label=lab)
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
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("End-state preference by removal depth: informed vs random removal "
                 f"(gpt2_small, h{a.hp}, mean of last {a.last} checkpoints)", y=1.02)
    fig.tight_layout()
    fig.savefig(a.out / "endstate_decile_arm.png", dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}/endstate_decile_arm.png")

    print("\n== informed minus random, by intervention x decile ==")
    for iv in ivs:
        sub = d[d.iv == iv]
        b = sub[sub.arm=="bert"].groupby("k").endstate.mean()
        r = sub[sub.arm=="rand"].groupby("k").endstate.mean()
        j = (b - r).dropna()
        if len(j): print(f"  {IV_LABEL.get(iv,iv):22s} mean Δ {j.mean():+.3f} | "
                         f"range {j.min():+.3f}..{j.max():+.3f}")

if __name__ == "__main__":
    main()
