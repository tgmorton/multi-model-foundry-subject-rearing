#!/usr/bin/env python3
"""Build intervention-matched English null-subject-v2 stimuli.

The corpus interventions were written for independent corpus lines, but the
evaluation data are minimal pairs.  This generator therefore applies a
pair-aware operationalization:

* shared contexts are parsed/transformed once and copied to both members;
* impoverish-case and lemmatization substitutions on source tokens shared by
  the pair are reconciled to the overt (pronoun_status=1) parse;
* morphology enrichment is applied literally and independently to each member,
  including any agreement differences induced by overt versus null parsing;
* remove_expletive_sentences is a training-distribution deletion, so its
  evaluation stimuli remain unchanged rather than becoming empty sentences.

The output is immutable-by-default and contains hashes plus a row-level edit
ledger.  Use the exact production parser (en_core_web_trf 3.7.3).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_INPUT = REPO_ROOT / "evaluation/stimuli/null-subj-v2/staging/en"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation/stimuli/null-subj-v2-matched-v1"
SCHEMA = [
    "item_id", "category", "condition", "pronoun_status", "context",
    "target", "hotspot_token", "hotspot_position", "language", "names",
    "generator",
]
CONDITIONS = {
    "baseline": "identity",
    "remove_expletive_sentences": "remove_expletive_sentences_en",
    "impoverish_case": "impoverish_case_en",
    "lemmatize_verbs": "lemmatize_verbs",
    "enrich_verbal_morphology": "enrich_verbal_morphology",
}
ABLATION_SOURCES = {
    "baseline": REPO_ROOT / "preprocessing/ablations/identity.py",
    "remove_expletive_sentences":
        REPO_ROOT / "preprocessing/ablations/remove_expletive_sentences.py",
    "impoverish_case": REPO_ROOT / "preprocessing/ablations/impoverish_case.py",
    "lemmatize_verbs": REPO_ROOT / "preprocessing/ablations/lemmatize_verbs.py",
    "enrich_verbal_morphology":
        REPO_ROOT / "preprocessing/ablations/enrich_verbal_morphology.py",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def pair_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["category"], row["condition"], row["item_id"]


def git_value(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def read_sources(root: Path) -> tuple[list[Path], dict[Path, list[dict[str, str]]]]:
    paths = sorted(root.glob("*.csv"))
    if not paths:
        raise SystemExit(f"no CSV inputs under {root}")
    tables: dict[Path, list[dict[str, str]]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != SCHEMA:
                raise ValueError(f"{path}: schema {reader.fieldnames!r} != {SCHEMA!r}")
            tables[path] = list(reader)
    return paths, tables


def source_pairs(tables: dict[Path, list[dict[str, str]]]):
    pairs: dict[tuple[str, str, str], list[tuple[Path, dict[str, str]]]] = defaultdict(list)
    for path, rows in tables.items():
        for row in rows:
            pairs[pair_key(row)].append((path, row))
    for key, members in pairs.items():
        statuses = {int(r["pronoun_status"]) for _, r in members}
        if len(members) != 2 or statuses != {0, 1}:
            raise ValueError(f"source pair {key!r}: expected two statuses {{0,1}}, got {statuses}")
        if members[0][1]["context"] != members[1][1]["context"]:
            raise ValueError(f"source pair {key!r}: contexts differ")
    return pairs


def reconcile_shared_edits(
    source_overt: str,
    source_null: str,
    transformed_overt: str,
    transformed_null: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Copy overt-side edits on aligned source tokens to the null member.

    Production interventions are one-token substitutions on these Moses-style
    stimuli.  Refuse rather than guess if an intervention changes token count.
    """
    so, sn = source_overt.split(), source_null.split()
    to, tn = transformed_overt.split(), transformed_null.split()
    if len(so) != len(to) or len(sn) != len(tn):
        raise ValueError(
            "token-count-changing substitution is not pair-reconcilable: "
            f"{len(so)}->{len(to)}, {len(sn)}->{len(tn)}"
        )
    reconciled = list(tn)
    forced: list[dict[str, Any]] = []
    matcher = SequenceMatcher(a=so, b=sn, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for oi, ni in zip(range(i1, i2), range(j1, j2)):
            # The overt parse is the declared pair-level reference even when
            # it leaves a shared source token unchanged and the null parse
            # changes it.  Reconcile in both directions.
            if reconciled[ni] != to[oi]:
                forced.append({
                    "source_token": so[oi], "overt_position": oi,
                    "null_position": ni, "independent_null": reconciled[ni],
                    "reconciled": to[oi],
                })
                reconciled[ni] = to[oi]
    return " ".join(reconciled), forced


def shared_edit_divergences(
    source_overt: str,
    source_null: str,
    transformed_overt: str,
    transformed_null: str,
) -> list[dict[str, Any]]:
    """Report independently transformed shared tokens that no longer match."""
    so, sn = source_overt.split(), source_null.split()
    to, tn = transformed_overt.split(), transformed_null.split()
    if len(so) != len(to) or len(sn) != len(tn):
        raise ValueError("token-count-changing transformation is unsupported")
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(
            a=so, b=sn, autojunk=False).get_opcodes():
        if tag != "equal":
            continue
        for oi, ni in zip(range(i1, i2), range(j1, j2)):
            if to[oi] != tn[ni]:
                out.append({
                    "source_token": so[oi], "overt_position": oi,
                    "null_position": ni, "overt_transformed": to[oi],
                    "null_transformed": tn[ni],
                })
    return out


def token_edits(source: str, transformed: str) -> list[dict[str, Any]]:
    """Deterministic word-level edit ledger for audit/review."""
    source_words, transformed_words = source.split(), transformed.split()
    edits = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(
            a=source_words, b=transformed_words, autojunk=False).get_opcodes():
        if tag != "equal":
            edits.append({
                "operation": tag, "source_start": i1, "source_end": i2,
                "output_start": j1, "output_end": j2,
                "source_tokens": source_words[i1:i2],
                "output_tokens": transformed_words[j1:j2],
            })
    return edits


def transform_pair(
    members: list[tuple[Path, dict[str, str]]],
    condition: str,
    docs: dict[str, Any],
    transform,
) -> tuple[list[tuple[Path, dict[str, str]]], dict[str, Any]]:
    by_status = {int(row["pronoun_status"]): (path, row) for path, row in members}
    overt_path, overt = by_status[1]
    null_path, null = by_status[0]
    key = pair_key(overt)
    audit: dict[str, Any] = {
        "category": key[0], "eval_condition": key[1], "item_id": key[2],
        "intervention": condition, "excluded": False,
        "forced_shared_edits": [], "pair_divergent_shared_edits": [],
    }

    if condition == "remove_expletive_sentences":
        # User-approved methodological policy: this intervention changes the
        # training distribution (whole sentences are absent); there is no
        # corresponding nonempty surface rewrite for a test item.
        context_out = overt["context"]
        overt_out, null_out = overt["target"], null["target"]
        counts = {"context": 0, "overt_target": 0, "null_target": 0}
    else:
        context_out, n_context = transform(docs[overt["context"]]) \
            if overt["context"] else ("", 0)
        overt_out, n_overt = transform(docs[overt["target"]])
        null_independent, n_null = transform(docs[null["target"]])
        if condition == "enrich_verbal_morphology":
            null_out = null_independent
            audit["pair_divergent_shared_edits"] = shared_edit_divergences(
                overt["target"], null["target"], overt_out, null_independent)
        else:
            null_out, forced = reconcile_shared_edits(
                overt["target"], null["target"], overt_out, null_independent)
            audit["forced_shared_edits"] = forced
        counts = {"context": n_context, "overt_target": n_overt,
                  "null_target_independent": n_null}

    outputs: list[tuple[Path, dict[str, str]]] = []
    for path, source, target_out in (
        (overt_path, overt, overt_out), (null_path, null, null_out)
    ):
        row = dict(source)
        row["context"] = clean_text(context_out)
        row["target"] = clean_text(target_out)
        pos = int(source["hotspot_position"])
        words = row["target"].split()
        if not (0 <= pos < len(words)):
            raise ValueError(f"{key}: hotspot position {pos} out of transformed target")
        # All supported substitution interventions preserve token count, so the
        # source word index remains the transformed hotspot word index.
        if len(words) != len(source["target"].split()):
            raise ValueError(f"{key}: transformed target token count changed")
        row["hotspot_token"] = words[pos]
        row["hotspot_position"] = str(pos)
        outputs.append((path, row))

    audit["pair_divergent_shared_edits"] = shared_edit_divergences(
        overt["target"], null["target"], clean_text(overt_out),
        clean_text(null_out))

    audit.update(
        counts=counts,
        changed_context=context_out.strip() != overt["context"].strip(),
        changed_overt=overt_out.strip() != overt["target"].strip(),
        changed_null=null_out.strip() != null["target"].strip(),
        source_overt=overt["target"], transformed_overt=clean_text(overt_out),
        source_null=null["target"], transformed_null=clean_text(null_out),
        source_context=overt["context"], transformed_context=clean_text(context_out),
        context_token_edits=token_edits(overt["context"], clean_text(context_out)),
        overt_token_edits=token_edits(overt["target"], clean_text(overt_out)),
        null_token_edits=token_edits(null["target"], clean_text(null_out)),
    )
    return outputs, audit


def validate_condition(
    condition: str,
    source_count: int,
    outputs: dict[Path, list[dict[str, str]]],
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [r for rs in outputs.values() for r in rs]
    errors: list[str] = []
    pairs: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if list(row) != SCHEMA:
            errors.append(f"schema/order mismatch at {pair_key(row)}")
        if row["language"] != "en":
            errors.append(f"non-English row at {pair_key(row)}")
        if not row["target"]:
            errors.append(f"empty target at {pair_key(row)}")
        words = row["target"].split()
        try:
            pos = int(row["hotspot_position"])
            if words[pos] != row["hotspot_token"]:
                errors.append(f"hotspot mismatch at {pair_key(row)} status={row['pronoun_status']}")
        except (ValueError, IndexError):
            errors.append(f"invalid hotspot at {pair_key(row)}")
        pairs[pair_key(row)].append(row)
    for key, members in pairs.items():
        if len(members) != 2 or {m["pronoun_status"] for m in members} != {"0", "1"}:
            errors.append(f"broken pair {key}")
        elif members[0]["context"] != members[1]["context"]:
            errors.append(f"context asymmetry {key}")
    excluded = [a for a in audits if a["excluded"]]
    if len(rows) + 2 * len(excluded) != source_count:
        errors.append(
            f"coverage mismatch: rows={len(rows)} exclusions={len(excluded)} "
            f"source={source_count}"
        )
    if condition != "remove_expletive_sentences" and excluded:
        errors.append("non-removal intervention unexpectedly excluded pairs")
    if condition != "enrich_verbal_morphology":
        unexpected = sum(len(a.get("pair_divergent_shared_edits", []))
                         for a in audits)
        if unexpected:
            errors.append(
                f"{unexpected} shared-token pair divergences outside literal enrichment")
    if errors:
        raise ValueError(f"{condition}: gold structural validation failed:\n- " + "\n- ".join(errors[:30]))
    return {
        "rows": len(rows), "pairs": len(pairs), "excluded_pairs": len(excluded),
        "changed_pairs": sum(bool(a.get("changed_context") or a.get("changed_overt")
                                  or a.get("changed_null")) for a in audits),
        "forced_shared_edits": sum(len(a.get("forced_shared_edits", [])) for a in audits),
        "pair_divergent_shared_edits": sum(
            len(a.get("pair_divergent_shared_edits", [])) for a in audits),
    }


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCHEMA, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--spacy-model", default="en_core_web_trf")
    ap.add_argument("--force", action="store_true",
                    help="Replace an existing output root atomically.")
    args = ap.parse_args()

    if args.output_root.exists() and not args.force:
        raise SystemExit(f"output exists: {args.output_root} (pass --force to replace)")

    import spacy
    # Import registers every production ablation.
    import preprocessing.ablations  # noqa: F401
    from preprocessing.registry import AblationRegistry

    paths, tables = read_sources(args.input_root)
    pairs = source_pairs(tables)
    source_count = sum(len(v) for v in tables.values())
    texts = sorted({r[field] for rows in tables.values() for r in rows
                    for field in ("context", "target") if r[field]})

    nlp = spacy.load(args.spacy_model)
    parsed_docs = list(nlp.pipe(texts, batch_size=32))
    if len(parsed_docs) != len(texts):
        raise RuntimeError("spaCy returned an unexpected document count")
    docs = dict(zip(texts, parsed_docs))

    tmp_parent = args.output_root.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=args.output_root.name + ".tmp.", dir=tmp_parent))
    manifest: dict[str, Any] = {
        "format_version": "condition-matched-stimuli.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.input_root.relative_to(REPO_ROOT)),
        "source_files": {p.name: sha256_file(p) for p in paths},
        "source_rows": source_count,
        "source_pairs": len(pairs),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "generator_sha256": sha256_file(Path(__file__)),
        "ablation_source_sha256": {
            condition: sha256_file(path)
            for condition, path in ABLATION_SOURCES.items()
        },
        "spacy": {
            "package_version": spacy.__version__,
            "model": args.spacy_model,
            "model_name": nlp.meta.get("name"),
            "model_version": nlp.meta.get("version"),
        },
        "remove_expletive_policy": "training_distribution_only_eval_stimuli_unchanged",
        "pair_policy": {
            "impoverish_case_and_lemmatize":
                "shared_source_token_edits_follow_pronoun_status_1_parse",
            "enrich_verbal_morphology": "literal_independent_application",
        },
        "conditions": {},
        "vetted": False,
    }

    try:
        for condition, ablation_name in CONDITIONS.items():
            transform, _ = AblationRegistry.get(ablation_name)
            if hasattr(transform, "reset_file_state"):
                transform.reset_file_state()
            out_by_path: dict[Path, list[dict[str, str]]] = {p: [] for p in paths}
            audits: list[dict[str, Any]] = []
            for members in pairs.values():
                transformed, audit = transform_pair(
                    members, condition, docs, transform,
                )
                audits.append(audit)
                for source_path, row in transformed:
                    out_by_path[source_path].append(row)

            stats = validate_condition(condition, source_count, out_by_path, audits)
            cond_root = tmp / condition / "en"
            for source_path, rows in out_by_path.items():
                rows.sort(key=lambda r: (r["category"], r["condition"],
                                         int(r["item_id"]), -int(r["pronoun_status"])))
                write_csv(cond_root / source_path.name, rows)
            audit_path = tmp / condition / "transformation_audit.jsonl"
            with audit_path.open("w", encoding="utf-8") as fh:
                for audit in audits:
                    fh.write(json.dumps(audit, sort_keys=True, ensure_ascii=False) + "\n")
            output_hashes = {p.name: sha256_file(cond_root / p.name) for p in paths}
            manifest["conditions"][condition] = {
                "ablation_registry_name": ablation_name,
                **stats,
                "output_sha256": output_hashes,
                "audit_sha256": sha256_file(audit_path),
            }

        (tmp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if args.output_root.exists():
            backup = args.output_root.with_name(args.output_root.name + ".previous")
            if backup.exists():
                raise SystemExit(f"refusing to overwrite backup {backup}")
            args.output_root.replace(backup)
            tmp.replace(args.output_root)
            shutil.rmtree(backup)
        else:
            tmp.replace(args.output_root)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    print(json.dumps(manifest["conditions"], indent=2, sort_keys=True))
    print(f"wrote {args.output_root}")


if __name__ == "__main__":
    main()
