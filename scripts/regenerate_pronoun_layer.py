"""
Regenerate the `pronouns` layer from stored base parquets — no re-parse.

Why this exists (2026-08-12): the original PronounAnnotator compared
`tok.dep_` against UD labels (nsubj:pass, obj, iobj) while en_core_web_trf
emits ClearNLP labels (nsubjpass, dobj, dative, pobj). Passive-clause
subject pronouns and all object-function pronouns were silently dropped
from the layer. The base parquets store the raw parse (tokens, pos_tags,
dep_rels, dep_heads), so the layer is recomputable in pure pandas/pyarrow
with corrected label handling — re-annotating would renumber sent_idx
(boundary-marker handling changed post-annotation) and break every
sentence_id join, so we deliberately do NOT re-parse.

Differences vs. the original layer, by design:
- dep labels normalized to UD before classification (the fix);
- person/number/case come from a closed-class lexicon on the token form
  rather than spaCy morph (morph is not stored in base parquets; for
  English personal pronouns the lexicon is deterministic and auditable);
- `oblique` now includes prepositional-object pronouns (pobj -> obl under
  normalization); their head is the preposition, so head_verb_idx is None.

Usage:
  python scripts/regenerate_pronoun_layer.py \
      --corpus-dir data/output/train_90M/annotated_corpus \
      [--workers 6] [--dry-run]

Originals are moved to <corpus-dir>/layers_deprecated_<date>/pronouns/
before the new files are written.
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.dep_labels import audit_labels, normalize_dep  # noqa: E402

# --- Closed-class lexicon: form -> (person, number, case, is_reflexive) ---
# Personal pronouns (PronType=Prs equivalents). Case is set only where the
# form is unambiguous; ambiguous forms (you/it as Nom-or-Acc) get None.
_PERSONAL = {
    "i": (1, "Sing", "Nom", False),
    "me": (1, "Sing", "Acc", False),
    "we": (1, "Plur", "Nom", False),
    "us": (1, "Plur", "Acc", False),
    "you": (2, None, None, False),
    "he": (3, "Sing", "Nom", False),
    "him": (3, "Sing", "Acc", False),
    "she": (3, "Sing", "Nom", False),
    "her": (3, "Sing", "Acc", False),
    "it": (3, "Sing", None, False),
    "they": (3, "Plur", "Nom", False),
    "them": (3, "Plur", "Acc", False),
    "myself": (1, "Sing", None, True),
    "yourself": (2, "Sing", None, True),
    "himself": (3, "Sing", None, True),
    "herself": (3, "Sing", None, True),
    "itself": (3, "Sing", None, True),
    "ourselves": (1, "Plur", None, True),
    "yourselves": (2, "Plur", None, True),
    "themselves": (3, "Plur", None, True),
    # Archaic/dialectal forms present in Gutenberg
    "thou": (2, "Sing", "Nom", False),
    "thee": (2, "Sing", "Acc", False),
    "ye": (2, "Plur", None, False),
    "thyself": (2, "Sing", None, True),
    "'em": (3, "Plur", "Acc", False),
}
# Non-personal PRON forms with a clear number; person stays None.
_NUMBER_ONLY = {
    "this": "Sing", "that": "Sing", "these": "Plur", "those": "Plur",
    "one": "Sing", "ones": "Plur",
}

_SUBJECT_DEPS = {"nsubj", "nsubj:pass"}
_LAYER_SCHEMA = pa.schema([
    ("sentence_id", pa.string()),
    ("pronouns", pa.list_(pa.struct([
        ("token_idx", pa.int32()),
        ("lemma", pa.string()),
        ("function", pa.string()),
        ("person", pa.int32()),
        ("number", pa.string()),
        ("case", pa.string()),
        ("is_reflexive", pa.bool_()),
        ("is_clitic", pa.bool_()),
        ("head_verb_idx", pa.int32()),
    ]))),
    ("has_overt_subject_pronoun", pa.bool_()),
    ("has_overt_object_pronoun", pa.bool_()),
])


def _classify(dep: str):
    if dep in _SUBJECT_DEPS:
        return "subject"
    if dep == "obj":
        return "direct_object"
    if dep == "iobj":
        return "indirect_object"
    if dep in ("obl", "nmod"):
        return "oblique"
    return None


def _annotate_row(tokens, lemmas, pos_tags, dep_rels, dep_heads):
    pronouns = []
    for i, pos in enumerate(pos_tags):
        if pos != "PRON":
            continue
        dep = normalize_dep(dep_rels[i])
        function = _classify(dep)
        if function is None:
            continue

        form = tokens[i].lower()
        person, number, case, is_reflexive = _PERSONAL.get(
            form, (None, _NUMBER_ONLY.get(form), None, False)
        )

        head_idx = dep_heads[i]
        head_pos = pos_tags[head_idx] if 0 <= head_idx < len(pos_tags) else None
        head_is_verbal = head_pos in ("VERB", "AUX")

        # Mirror the original annotator's clitic heuristic (personal
        # pronoun, <=2 chars, verbal head) for schema continuity.
        is_clitic = form in _PERSONAL and len(tokens[i]) <= 2 and head_is_verbal

        pronouns.append({
            "token_idx": i,
            "lemma": lemmas[i].lower(),
            "function": function,
            "person": person,
            "number": number,
            "case": case,
            "is_reflexive": is_reflexive,
            "is_clitic": is_clitic,
            "head_verb_idx": head_idx if head_is_verbal else None,
        })
    return pronouns


def process_file(base_path_str: str, out_dir_str: str) -> dict:
    base_path = Path(base_path_str)
    out_dir = Path(out_dir_str)
    pf = pq.ParquetFile(base_path)
    cols = ["sentence_id", "tokens", "lemmas", "pos_tags", "dep_rels", "dep_heads"]

    writer = None
    stats = {"file": base_path.name, "rows": 0, "instances": 0,
             "by_function": {}, "dep_scheme": None}
    dep_sample = []

    for batch in pf.iter_batches(batch_size=50_000, columns=cols):
        rows = batch.to_pylist()
        out_rows = []
        for r in rows:
            prons = _annotate_row(
                r["tokens"], r["lemmas"], r["pos_tags"], r["dep_rels"], r["dep_heads"]
            )
            for p in prons:
                stats["by_function"][p["function"]] = (
                    stats["by_function"].get(p["function"], 0) + 1
                )
            stats["instances"] += len(prons)
            out_rows.append({
                "sentence_id": r["sentence_id"],
                "pronouns": prons,
                "has_overt_subject_pronoun": any(
                    p["function"] == "subject" for p in prons
                ),
                "has_overt_object_pronoun": any(
                    p["function"] in ("direct_object", "indirect_object")
                    for p in prons
                ),
            })
            if len(dep_sample) < 200_000:
                dep_sample.extend(r["dep_rels"])
        stats["rows"] += len(rows)

        table = pa.Table.from_pylist(out_rows, schema=_LAYER_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(out_dir / base_path.name, _LAYER_SCHEMA)
        writer.write_table(table)

    if writer is not None:
        writer.close()
    stats["dep_scheme"] = audit_labels(dep_sample)["scheme"]
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path,
                    default=Path("data/output/train_90M/annotated_corpus"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_dir = args.corpus_dir / "base"
    layer_dir = args.corpus_dir / "layers" / "pronouns"
    # Per-genre files only; the combined train_90M.parquet base is a stale
    # earlier snapshot (row counts differ) and must not be regenerated from.
    base_files = sorted(
        p for p in base_dir.glob("train_90M_*.parquet") if p.stem != "train_90M"
    )
    if not base_files:
        sys.exit(f"No per-genre base parquets found under {base_dir}")

    print(f"Regenerating pronouns layer from {len(base_files)} base files:")
    for p in base_files:
        print(f"  {p.name}")
    if args.dry_run:
        return

    # Back up the existing layer before overwriting.
    backup_dir = args.corpus_dir / f"layers_deprecated_{date.today().isoformat()}" / "pronouns"
    if layer_dir.exists() and not backup_dir.exists():
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(layer_dir, backup_dir)
        print(f"Backed up existing layer to {backup_dir}")
    layer_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_file, str(p), str(layer_dir)): p for p in base_files
        }
        for fut, p in futures.items():
            st = fut.result()
            all_stats.append(st)
            print(f"done {st['file']}: {st['rows']:,} rows, "
                  f"{st['instances']:,} pronoun instances, "
                  f"scheme={st['dep_scheme']}, by_function={st['by_function']}")

    manifest = {
        "regenerated": date.today().isoformat(),
        "reason": "dep-label scheme fix (ClearNLP vs UD), see preprocessing/dep_labels.py",
        "source": "base parquets (stored parse), no re-annotation",
        "files": all_stats,
        "totals": {
            "instances": sum(s["instances"] for s in all_stats),
            "by_function": {
                fn: sum(s["by_function"].get(fn, 0) for s in all_stats)
                for fn in {k for s in all_stats for k in s["by_function"]}
            },
        },
    }
    with open(layer_dir / "REGENERATION_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["totals"], indent=2))


if __name__ == "__main__":
    main()
