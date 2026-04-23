"""
Turn an ``ABLATION_MANIFEST.json`` into a human-readable markdown report.

Reads the manifest written by :class:`preprocessing.base.AblationPipeline`
at the end of a run and emits a summary suitable for pasting into a PR,
a lab notebook, or the Spanish-switch follow-up documentation.

Usage::

    # Print markdown to stdout
    python scripts/ablation_report.py \\
        --manifest data/spanish/processed/exp_impoverish_case_es/ABLATION_MANIFEST.json

    # Write to a file
    python scripts/ablation_report.py \\
        --manifest <path> \\
        --output reports/exp_impoverish_case_es.md

    # Combine multiple runs into one comparison report
    python scripts/ablation_report.py \\
        --manifest data/processed/exp1/ABLATION_MANIFEST.json \\
        --manifest data/processed/exp2/ABLATION_MANIFEST.json \\
        --output reports/comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


def _tier_table(tier_counts: Dict[str, int]) -> str:
    """Format a tier_counts dict as a pipe-separated markdown table."""
    if not tier_counts:
        return "_(no tier-level data — this ablation does not emit tier counts yet)_"

    total = sum(tier_counts.values())
    rows = ["| Tier | Count | Share |", "|---|---:|---:|"]
    for tier, count in sorted(tier_counts.items(), key=lambda x: -x[1]):
        share = count / total if total else 0.0
        rows.append(f"| `{tier}` | {_fmt_int(count)} | {_fmt_pct(share)} |")
    rows.append(f"| **Total** | **{_fmt_int(total)}** | **100.00%** |")
    return "\n".join(rows)


def _per_file_table(file_stats: List[Dict[str, Any]]) -> str:
    rows = [
        "| File | Original tokens | Final tokens | Items ablated | % removed | Time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fs in file_stats:
        rows.append(
            f"| `{fs['file_name']}` | {_fmt_int(fs['original_tokens'])} | "
            f"{_fmt_int(fs['final_tokens'])} | {_fmt_int(fs['items_ablated'])} | "
            f"{_fmt_pct(fs['proportion_removed'])} | "
            f"{_fmt_seconds(fs['processing_time_seconds'])} |"
        )
    return "\n".join(rows)


def _env_table(metadata: Dict[str, Any]) -> str:
    rows = [
        "| Field | Value |",
        "|---|---|",
        f"| Ablation | `{metadata['ablation_type']}` |",
        f"| Timestamp | `{metadata['timestamp']}` |",
        f"| Random seed | `{metadata['random_seed']}` |",
        f"| spaCy | `{metadata['spacy_model_name']}` v{metadata['spacy_model_version']} "
        f"(library v{metadata['spacy_version']}) |",
        f"| Device | `{metadata['device']}` |",
        f"| Python | `{metadata['python_version'].split()[0]}` |",
        f"| Host | `{metadata['hostname']}` ({metadata['platform']}) |",
    ]
    return "\n".join(rows)


def _render_single_run(manifest_path: Path) -> str:
    manifest = json.loads(Path(manifest_path).read_text())
    md = manifest["metadata"]
    files = manifest["file_statistics"]
    cfg = manifest.get("config", {})

    # Aggregate numbers
    total_orig = md["total_tokens_original"]
    total_final = md["total_tokens_final"]
    total_items = md["total_items_ablated"]
    pct_changed = (
        (total_orig - total_final) / total_orig if total_orig else 0.0
    )

    # Source lists
    sections: List[str] = []
    sections.append(f"# Ablation run: `{md['ablation_type']}`")
    sections.append("")
    sections.append(f"Manifest: `{manifest_path}`")
    sections.append("")

    # Summary
    sections.append("## Summary")
    sections.append("")
    sections.append(f"- Files processed: **{_fmt_int(md['total_files_processed'])}**")
    sections.append(
        f"- Tokens: {_fmt_int(total_orig)} → {_fmt_int(total_final)} "
        f"(net {_fmt_pct(pct_changed)} removed, "
        f"replacement-pool draws: {_fmt_int(md.get('total_pool_lines_drawn', 0))})"
    )
    sections.append(f"- Items ablated: **{_fmt_int(total_items)}**")
    sections.append(
        f"- Total time: **{_fmt_seconds(md['processing_time_seconds'])}**"
    )
    if md.get("failed_files"):
        sections.append(
            f"- ⚠️ Failed files: **{len(md['failed_files'])}** (see manifest)"
        )
    sections.append("")

    # Tier breakdown
    sections.append("## Tier breakdown")
    sections.append("")
    sections.append(_tier_table(md.get("aggregate_tier_counts", {})))
    sections.append("")

    # Per-file
    sections.append("## Per-file statistics")
    sections.append("")
    if files:
        sections.append(_per_file_table(files))
    else:
        sections.append("_(no file statistics in manifest)_")
    sections.append("")

    # Environment / reproducibility
    sections.append("## Environment")
    sections.append("")
    sections.append(_env_table(md))
    sections.append("")

    # Config (collapsed behind a details block so long YAMLs don't dominate)
    if cfg:
        sections.append("## Config")
        sections.append("")
        sections.append("<details><summary>Full run config</summary>")
        sections.append("")
        sections.append("```json")
        sections.append(json.dumps(cfg, indent=2, default=str))
        sections.append("```")
        sections.append("")
        sections.append("</details>")
        sections.append("")

    # Checksums (truncated — first 16 hex chars each, full list folded away)
    ck = md.get("output_checksums", {})
    if ck:
        sections.append("## Output checksums (SHA256)")
        sections.append("")
        sections.append("<details><summary>Show {} files</summary>".format(len(ck)))
        sections.append("")
        sections.append("```")
        for fname, h in sorted(ck.items()):
            sections.append(f"{h}  {fname}")
        sections.append("```")
        sections.append("")
        sections.append("</details>")
        sections.append("")

    return "\n".join(sections)


def _render_multi_run(manifest_paths: Iterable[Path]) -> str:
    """Compact comparison table across N runs."""
    runs = []
    for p in manifest_paths:
        m = json.loads(Path(p).read_text())
        runs.append((p, m))

    sections: List[str] = []
    sections.append("# Ablation comparison report")
    sections.append("")
    sections.append(f"Compared {len(runs)} runs.")
    sections.append("")

    # Header row
    sections.append("## Aggregate comparison")
    sections.append("")
    sections.append(
        "| Ablation | Files | Tokens orig → final | % removed | "
        "Items ablated | Time | Seed |"
    )
    sections.append(
        "|---|---:|---|---:|---:|---:|---:|"
    )
    for path, manifest in runs:
        md = manifest["metadata"]
        orig = md["total_tokens_original"]
        final = md["total_tokens_final"]
        pct = (orig - final) / orig if orig else 0.0
        sections.append(
            f"| `{md['ablation_type']}` | {_fmt_int(md['total_files_processed'])} | "
            f"{_fmt_int(orig)} → {_fmt_int(final)} | {_fmt_pct(pct)} | "
            f"{_fmt_int(md['total_items_ablated'])} | "
            f"{_fmt_seconds(md['processing_time_seconds'])} | "
            f"{md['random_seed']} |"
        )
    sections.append("")

    # Per-run tier breakdowns (stacked)
    sections.append("## Tier breakdowns")
    sections.append("")
    for path, manifest in runs:
        md = manifest["metadata"]
        sections.append(f"### `{md['ablation_type']}`")
        sections.append("")
        sections.append(_tier_table(md.get("aggregate_tier_counts", {})))
        sections.append("")

    return "\n".join(sections)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/ablation_report.py",
        description="Render a markdown report from one or more ABLATION_MANIFEST.json files.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Path to an ABLATION_MANIFEST.json (pass multiple times to compare runs)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the report to this file (default: stdout)",
    )
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)

    for mp in args.manifest:
        if not mp.exists():
            print(f"ERROR: manifest not found: {mp}", file=sys.stderr)
            return 2

    if len(args.manifest) == 1:
        report = _render_single_run(args.manifest[0])
    else:
        report = _render_multi_run(args.manifest)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Wrote report: {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
