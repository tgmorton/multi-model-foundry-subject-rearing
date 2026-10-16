#!/usr/bin/env python3
"""Build flat symlink corpus dirs for the rater HP sweeps (2026-10-16).

The option-B rater corpora live one shard per subdir,

    <src>/<corpus>/shard_NNN/shard_NNN.train     (base also has docs.jsonl.gz)

but corpus discovery in model_foundry is TOP-LEVEL ONLY (``*.train`` for
training, ``*.test`` for the held-out split; tokenize_dataset.py,
cache_keys.py). This script lays out one flat dir per corpus for a small,
deterministic shard subset, with the SAME shard ids in every corpus:

    <out>/<corpus>/shard_NNN.train -> <src>/<corpus>/shard_NNN/shard_NNN.train
    <out>/<corpus>/heldout.test       first --heldout-lines lines of the held-out
                                      shard (or a symlink to all of it with 0)
    <out>/<corpus>/SUBSET_MANIFEST.json   shards, source paths, sizes
    <out>/SUBSET_SPEC.json            the shard selection; every later run against
                                      the same <out> must request the same one

Point a rater sweep at ``<out>/<corpus>/`` (env CORPUS_DIR) for training AND
test corpus. Nothing is copied except the small held-out prefix.

A shard counts as present when its ``.train`` exists and, for the
intervention corpora, its ``ABLATION_MANIFEST.json`` too (the ablation
writes the .train non-atomically and the manifest last). Every requested
shard of every requested corpus is checked before anything is written;
missing ones are listed and the script exits 1. Interventions finish at
different times, so build ``--corpora base`` now and the rest later with
the same shard arguments.

A corpus dir that already exists with the same selection is left alone
(its symlinks are re-verified); one with a different selection is an error.
Each corpus dir is built in a temp dir and renamed into place.

Usage (in a pod with /mnt/data mounted):

    python scripts/rater_corpus/build_sweep_subset.py \\
        --train-shards 0 1 2 3 4 5 6 22 --heldout-shard 199

Python >= 3.9, stdlib only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_SRC = "/mnt/data/rater_corpus/optionb_v1"
DEFAULT_OUT = "/mnt/data/rater_corpus/sweep_subset_v1"
# Same list as scripts/rater_sweep.py::KNOWN_CORPORA (kept literal so this
# stays a stdlib-only script).
CORPORA = (
    "base",
    "remove_expletive_sentences_en",
    "impoverish_case_en",
    "lemmatize_verbs",
    "enrich_verbal_morphology",
)
BASE_CORPUS = "base"
DONE_MARKER = "ABLATION_MANIFEST.json"
HELDOUT_NAME = "heldout.test"
MANIFEST_NAME = "SUBSET_MANIFEST.json"
SPEC_NAME = "SUBSET_SPEC.json"
DEFAULT_HELDOUT_LINES = 50_000


class SubsetError(RuntimeError):
    pass


def _tmp_tag() -> str:
    # Unique across pods (container PIDs collide) for concurrent per-corpus runs.
    return f"{socket.gethostname()}-{os.getpid()}"


def shard_name(shard_id: int) -> str:
    return "shard_%03d" % shard_id


def shard_train_path(src: Path, corpus: str, shard_id: int) -> Path:
    name = shard_name(shard_id)
    return src / corpus / name / (name + ".train")


def check_shard(src: Path, corpus: str, shard_id: int) -> Optional[str]:
    """None if the shard is present and complete, else why not."""
    train = shard_train_path(src, corpus, shard_id)
    if not train.is_file():
        return f"{train} does not exist"
    if corpus != BASE_CORPUS and not (train.parent / DONE_MARKER).is_file():
        return f"{train.parent / DONE_MARKER} does not exist (ablation not finished)"
    return None


def validate_selection(train_shards: Sequence[int], heldout_shard: int) -> None:
    if not train_shards:
        raise SubsetError("--train-shards is empty")
    if len(set(train_shards)) != len(train_shards):
        raise SubsetError(f"--train-shards has duplicates: {list(train_shards)}")
    if any(s < 0 for s in train_shards) or heldout_shard < 0:
        raise SubsetError("shard ids must be >= 0")
    if heldout_shard in train_shards:
        raise SubsetError(f"--heldout-shard {heldout_shard} is also a train shard")


def spec_dict(src: Path, train_shards: Sequence[int], heldout_shard: int,
              heldout_lines: int) -> dict:
    return {
        "src": os.path.abspath(src),
        "train_shards": sorted(int(s) for s in train_shards),
        "heldout_shard": int(heldout_shard),
        "heldout_lines": int(heldout_lines),
    }


def check_or_write_spec(out: Path, spec: dict) -> None:
    """Pin the shard selection for <out>: the first run writes it, later runs
    must match, so all five corpora share identical shards."""
    path = out / SPEC_NAME
    if path.exists():
        existing = json.loads(path.read_text())
        pinned = {k: existing.get(k) for k in spec}
        if pinned != spec:
            raise SubsetError(
                f"{path} pins a different selection:\n  pinned:    {pinned}\n"
                f"  requested: {spec}\nuse the pinned arguments or another --out")
        return
    out.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{_tmp_tag()}")
    tmp.write_text(json.dumps({**spec, "corpora": list(CORPORA)}, indent=2) + "\n")
    os.replace(tmp, path)


def write_heldout(source: Path, dest: Path, n_lines: int) -> dict:
    """First ``n_lines`` lines of ``source`` into ``dest`` (0: symlink to all)."""
    if n_lines == 0:
        os.symlink(os.path.abspath(source), dest)
        mode = "symlink"
    else:
        lines = 0
        with open(source, "rb") as fin, open(dest, "wb") as fout:
            for line in fin:
                fout.write(line)
                lines += 1
                if lines >= n_lines:
                    break
        mode = "prefix"
    lines = words = 0
    with open(dest, "rb") as f:
        for line in f:
            lines += 1
            words += len(line.split())
    return {"name": dest.name, "source": os.path.abspath(source), "mode": mode,
            "lines": lines, "words": words, "bytes": os.path.getsize(dest)}


def corpus_manifest(src: Path, corpus: str, spec: dict, train_entries: List[dict],
                    heldout: dict) -> dict:
    return {
        "builder": "scripts/rater_corpus/build_sweep_subset.py",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "corpus": corpus,
        **spec,
        "train_files": train_entries,
        "train_bytes_total": sum(e["bytes"] for e in train_entries),
        "heldout": heldout,
    }


def verify_existing(dest: Path, src: Path, corpus: str, spec: dict) -> bool:
    """True if ``dest`` already holds this exact selection (links intact)."""
    mpath = dest / MANIFEST_NAME
    if not mpath.is_file():
        raise SubsetError(f"{dest} exists but has no {MANIFEST_NAME}; remove it first")
    m = json.loads(mpath.read_text())
    if {k: m.get(k) for k in spec} != spec or m.get("corpus") != corpus:
        raise SubsetError(f"{dest} was built with a different selection; remove it first")
    for sid in spec["train_shards"]:
        link = dest / (shard_name(sid) + ".train")
        want = os.path.abspath(shard_train_path(src, corpus, sid))
        if not link.is_symlink() or os.readlink(link) != want or not link.exists():
            raise SubsetError(f"{link} is not an intact symlink to {want}; remove {dest}")
    if not (dest / HELDOUT_NAME).exists():
        raise SubsetError(f"{dest / HELDOUT_NAME} is missing; remove {dest}")
    return True


def build_corpus(src: Path, out: Path, corpus: str, spec: dict) -> Tuple[str, dict]:
    dest = out / corpus
    if dest.exists():
        verify_existing(dest, src, corpus, spec)
        return "kept", json.loads((dest / MANIFEST_NAME).read_text())

    tmp = out / f".{corpus}.tmp-{_tmp_tag()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        entries = []
        for sid in spec["train_shards"]:
            source = shard_train_path(src, corpus, sid)
            link = tmp / (shard_name(sid) + ".train")
            os.symlink(os.path.abspath(source), link)
            entries.append({"name": link.name, "source": os.path.abspath(source),
                            "bytes": os.path.getsize(source)})
        heldout = write_heldout(shard_train_path(src, corpus, spec["heldout_shard"]),
                                tmp / HELDOUT_NAME, spec["heldout_lines"])
        manifest = corpus_manifest(src, corpus, spec, entries, heldout)
        (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return "built", manifest


def build(src: Path, out: Path, corpora: Sequence[str], train_shards: Sequence[int],
          heldout_shard: int, heldout_lines: int = DEFAULT_HELDOUT_LINES) -> Dict[str, dict]:
    validate_selection(train_shards, heldout_shard)
    if heldout_lines < 0:
        raise SubsetError("--heldout-lines must be >= 0")
    unknown = [c for c in corpora if c not in CORPORA]
    if unknown:
        raise SubsetError(f"unknown corpora {unknown}; known: {list(CORPORA)}")

    problems = [f"{c} {shard_name(s)}: {why}"
                for c in corpora
                for s in [*train_shards, heldout_shard]
                if (why := check_shard(src, c, s)) is not None]
    if problems:
        raise SubsetError("missing shards (nothing written):\n  " + "\n  ".join(problems))

    spec = spec_dict(src, train_shards, heldout_shard, heldout_lines)
    check_or_write_spec(out, spec)
    results = {}
    for corpus in corpora:
        status, manifest = build_corpus(src, out, corpus, spec)
        results[corpus] = manifest
        print(f"[{status}] {out / corpus}: {len(manifest['train_files'])} train shards, "
              f"{manifest['train_bytes_total'] / 1e9:.2f} GB; heldout "
              f"{manifest['heldout']['lines']:,} lines / {manifest['heldout']['words']:,} words")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default=DEFAULT_SRC, type=Path)
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--corpora", nargs="+", default=list(CORPORA), metavar="CORPUS")
    ap.add_argument("--train-shards", nargs="+", type=int, required=True, metavar="ID")
    ap.add_argument("--heldout-shard", type=int, required=True, metavar="ID")
    ap.add_argument("--heldout-lines", type=int, default=DEFAULT_HELDOUT_LINES,
                    help="lines of the held-out shard to copy into heldout.test; "
                         "0 = symlink the whole shard (default %(default)s)")
    args = ap.parse_args(argv)
    try:
        build(args.src, args.out, args.corpora, args.train_shards,
              args.heldout_shard, args.heldout_lines)
    except SubsetError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
