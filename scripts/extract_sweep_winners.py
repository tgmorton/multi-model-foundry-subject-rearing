"""Extract the top-K HP configs per (arch, lang) from completed sweeps.

Writes one JSON per (arch, lang) to data/sweep_winners/<arch>_<lang>.json,
containing the top-K configs by held-out perplexity (cheap proxy from
sweep_agent_lm.py). The production launcher reads these to spawn one pod
per (HP, seed) per (arch, intervention) cell.

Usage:
    python scripts/extract_sweep_winners.py --lang en --top 5
    python scripts/extract_sweep_winners.py --lang es --top 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

ENTITY = "thmorton-uc-san-diego"
PROJECT = "subject-drop-prod-hp"

SWEEPS = {
    "en": {
        "gpt2_small":  "d7n4xyoq",
        "gpt2_medium": "ywtdsoyd",
        "gpt2_large":  "5gici9l4",
        "bert_large":  "2pimhcbm",
        "lstm":        "mf2wmux2",
        "mamba_370m":  "ksp6lfuk",
    },
    "es": {
        # Fill in once Spanish sweep IDs are checked-in to k8s/job-sweep-*-es.yaml
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "es"], required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/sweep_winners"))
    args = ap.parse_args()

    api = wandb.Api()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for arch, sweep_id in SWEEPS[args.lang].items():
        sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")
        runs = [
            r for r in sweep.runs
            if r.state == "finished"
            and r.summary.get("proxy/min_held_out_perplexity") is not None
        ]
        runs.sort(key=lambda r: r.summary["proxy/min_held_out_perplexity"])
        top = runs[: args.top]

        configs = []
        for rank, r in enumerate(top, 1):
            cfg = r.config or {}
            configs.append({
                "rank": rank,
                "sweep_run_id": r.id,
                "min_held_out_perplexity": r.summary["proxy/min_held_out_perplexity"],
                "learning_rate":        cfg.get("learning_rate"),
                "warmup_ratio":         cfg.get("warmup_ratio"),
                "dropout":              cfg.get("dropout"),
                "attention_dropout":    cfg.get("attention_dropout"),
                "adam_beta2":           cfg.get("adam_beta2"),
                "effective_batch_size": cfg.get("effective_batch_size"),
            })

        out_path = args.out_dir / f"{arch}_{args.lang}.json"
        out_path.write_text(json.dumps({
            "arch": arch,
            "lang": args.lang,
            "sweep_id": sweep_id,
            "selection_metric": "proxy/min_held_out_perplexity",
            "top_k": args.top,
            "configs": configs,
        }, indent=2))
        print(f"  wrote {out_path} ({len(configs)} configs)")


if __name__ == "__main__":
    main()
