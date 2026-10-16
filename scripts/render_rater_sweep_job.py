#!/usr/bin/env python3
"""Render a per-corpus rater-sweep K8s Job from the templates in
k8s/rater_sweeps/ (2026-10-16). Prints YAML on stdout; applies nothing.

    python scripts/render_rater_sweep_job.py precache --corpus lemmatize_verbs > /tmp/p.yaml
    python scripts/render_rater_sweep_job.py sweep --corpus lemmatize_verbs \\
        --sweep-id thmorton-uc-san-diego/subject-drop-rater-sweeps/abc123 > /tmp/s.yaml

Per corpus it sets the Job name (thomas-rater-{precache,sweep}-<corpus>),
the `corpus` label, and the CORPUS / CONDITION / CORPUS_DIR / SWEEP_ID env;
for the sweep Job it also re-renders the hostname NotIn list from
configs/bad_nodes.txt, so the one bad-node list stays the source of truth.
Comments in the templates are not carried over.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bad_nodes import load as load_bad_nodes  # noqa: E402
from rater_sweep import KNOWN_CORPORA  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = {
    "precache": REPO / "k8s" / "rater_sweeps" / "job-precache-rater-subset.yaml",
    "sweep": REPO / "k8s" / "rater_sweeps" / "job-sweep-rater.yaml",
}
NAME_PREFIX = {"precache": "thomas-rater-precache-", "sweep": "thomas-rater-sweep-"}
DEFAULT_SUBSET_ROOT = "/mnt/data/rater_corpus/sweep_subset_v1"


def _set_env(container: dict, name: str, value: str) -> None:
    for e in container.get("env", []):
        if e.get("name") == name:
            e.pop("valueFrom", None)
            e["value"] = value
            return
    raise KeyError(f"template container {container['name']!r} has no env {name}")


def _set_branch(pod_spec: dict, branch: str) -> None:
    for ic in pod_spec.get("initContainers", []):
        if ic.get("name") == "clone-repo":
            ic["args"] = [a if not a.startswith("--branch=") else f"--branch={branch}"
                          for a in ic["args"]]
            return
    raise KeyError("template has no clone-repo init container")


def _set_bad_nodes(pod_spec: dict, nodes: List[str]) -> None:
    terms = pod_spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    for term in terms:
        for expr in term.get("matchExpressions", []):
            if expr["key"] == "kubernetes.io/hostname" and expr["operator"] == "NotIn":
                expr["values"] = list(nodes)
                return
    raise KeyError("template has no kubernetes.io/hostname NotIn expression")


def render(kind: str, corpus: str, *, sweep_id: Optional[str] = None,
           subset_root: str = DEFAULT_SUBSET_ROOT, branch: Optional[str] = None,
           parallelism: Optional[int] = None, completions: Optional[int] = None,
           bad_nodes: Optional[List[str]] = None) -> dict:
    if kind not in TEMPLATES:
        raise ValueError(f"kind must be one of {sorted(TEMPLATES)}")
    if corpus not in KNOWN_CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; known: {list(KNOWN_CORPORA)}")
    job = copy.deepcopy(yaml.safe_load(TEMPLATES[kind].read_text()))

    job["metadata"]["name"] = NAME_PREFIX[kind] + corpus.replace("_", "-")
    for labels in (job["metadata"]["labels"], job["spec"]["template"]["metadata"]["labels"]):
        labels["corpus"] = corpus
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    if branch:
        _set_branch(pod, branch)

    if kind == "precache":
        _set_env(container, "CORPUS", corpus)
        _set_env(container, "SUBSET_ROOT", subset_root)
    else:
        if not sweep_id or sweep_id.count("/") != 2:
            raise ValueError("--sweep-id must be the full <entity>/<project>/<id> "
                             "printed by `wandb sweep`")
        _set_env(container, "CONDITION", corpus)
        _set_env(container, "CORPUS_DIR", f"{subset_root.rstrip('/')}/{corpus}")
        _set_env(container, "SWEEP_ID", sweep_id)
        _set_bad_nodes(pod, load_bad_nodes() if bad_nodes is None else bad_nodes)
        if parallelism is not None:
            job["spec"]["parallelism"] = int(parallelism)
        if completions is not None:
            job["spec"]["completions"] = int(completions)
    return job


def _str_presenter(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


def dump(job: dict) -> str:
    class _Dumper(yaml.SafeDumper):
        pass
    _Dumper.add_representer(str, _str_presenter)
    return yaml.dump(job, Dumper=_Dumper, sort_keys=False, width=1000, allow_unicode=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=sorted(TEMPLATES))
    ap.add_argument("--corpus", required=True, choices=KNOWN_CORPORA)
    ap.add_argument("--sweep-id", help="sweep only: <entity>/<project>/<id>")
    ap.add_argument("--subset-root", default=DEFAULT_SUBSET_ROOT)
    ap.add_argument("--branch", help="git branch the pods clone (template: feat/rater-sweeps)")
    ap.add_argument("--parallelism", type=int, help="sweep only: concurrent trials (A100s)")
    ap.add_argument("--completions", type=int, help="sweep only: successful trials")
    args = ap.parse_args(argv)
    try:
        job = render(args.kind, args.corpus, sweep_id=args.sweep_id,
                     subset_root=args.subset_root, branch=args.branch,
                     parallelism=args.parallelism, completions=args.completions)
    except ValueError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    sys.stdout.write("---\n" + dump(job))
    return 0


if __name__ == "__main__":
    sys.exit(main())
