#!/usr/bin/env python3
"""Build condition-matched cross-construction priming stimuli.

Each priming item is a two-part discourse: a PRIME item (its context, then its
overt or null sentence) followed by a TARGET item (its context, then its
minimal pair). The model's overt-target preference after an overt prime vs a
null prime is the priming effect; a neutral prime (the prime item's context
only) is built into a sibling root so that the asymmetry (overt boosting
overt vs null boosting null) can be estimated when wanted.

Design (2026-10-21, Thomas):
* sources: the pronoun-free-context matched set, so no context pronoun acts as
  an uncontrolled overt-pronoun prime;
* every target pair gets one prime item from each of the 8 categories
  (same-category primes are always a different item);
* prime and target share no content lemma (NOUN/PROPN/VERB/ADJ/ADV, production
  parser) and no name;
* condition-matched: the prime-target pairing is chosen once, on the baseline
  condition, and each intervention's composites are assembled from that
  intervention's own matched rows, so every prime and target appears exactly as
  in that condition's eval.

Output roots (layout <root>/<condition>/en/*.csv, the matched-eval format the
eval runner reads), each with a manifest (vetted=false) and pairing.csv:

    evaluation/stimuli/null-subj-v2-priming-nopron-v1/          overt + null primes
    evaluation/stimuli/null-subj-v2-priming-nopron-v1-neutral/  context-only primes

    python scripts/build_priming_stimuli.py [--force]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_condition_matched_stimuli import SCHEMA, write_csv  # noqa: E402

SOURCE = REPO_ROOT / "evaluation/stimuli/null-subj-v2-matched-nopron-v1"
OUT = REPO_ROOT / "evaluation/stimuli/null-subj-v2-priming-nopron-v1"
OUT_NEUTRAL = REPO_ROOT / "evaluation/stimuli/null-subj-v2-priming-nopron-v1-neutral"
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
STATUS = {"overt": "1", "null": "0"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def join(*parts: str) -> str:
    return " ".join(p for p in parts if p)


def read_condition(root: Path) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    """{(category, item_id, condition): {pronoun_status: row}} for one intervention."""
    items: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for path in sorted(root.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != SCHEMA:
                raise SystemExit(f"{path}: unexpected schema {reader.fieldnames}")
            for row in reader:
                key = (row["category"], row["item_id"], row["condition"])
                if row["pronoun_status"] in items[key]:
                    raise SystemExit(f"{path}: duplicate member for {key}")
                items[key][row["pronoun_status"]] = row
    bad = [k for k, v in items.items() if set(v) != {"0", "1"}]
    if bad:
        raise SystemExit(f"{root}: incomplete pairs, e.g. {bad[0]}")
    return dict(items)


def content_lemmas(items, nlp) -> dict[tuple[str, str, str], frozenset[str]]:
    texts = sorted({t for m in items.values() for r in m.values() for t in (r["context"], r["target"]) if t})
    # Every name in the set counts as content wherever it appears: vocatives
    # ("ana , ...") leave the names column empty, and lowercase names are not
    # always tagged PROPN.
    names = {n for m in items.values() for r in m.values() for n in r["names"].split(";") if n}
    lem = {}
    for text, doc in zip(texts, nlp.pipe(texts, batch_size=64)):
        lem[text] = ({t.lemma_.lower() for t in doc if t.pos_ in CONTENT_POS}
                     | {w for w in text.split() if w in names})
    out = {}
    for key, members in items.items():
        s = set()
        for r in members.values():
            for t in (r["context"], r["target"]):
                s |= lem.get(t, set())
            s |= {n for n in r["names"].split(";") if n}
        out[key] = frozenset(s)
    return out


def choose_pairs(items, lemmas) -> tuple[list[dict], Counter]:
    """One prime item per (target item, prime category), balanced.

    Among eligible primes (different item; no shared content lemma or name) the
    least-used prime condition label wins, then the least-used prime item, then
    a hash tie-break, so the choice is deterministic and spread out.
    """
    by_cat: dict[str, list] = defaultdict(list)
    for key in sorted(items):
        by_cat[key[0]].append(key)
    used_item: Counter = Counter()
    used_cond: Counter = Counter()
    skipped: Counter = Counter()
    pairs = []
    for t in sorted(items):
        for pcat in sorted(by_cat):
            cands = [p for p in by_cat[pcat]
                     if p != t and not (p[0] == t[0] and p[1] == t[1])
                     and not (lemmas[p] & lemmas[t])]
            if not cands:
                skipped[(t[0], pcat)] += 1
                continue
            p = min(cands, key=lambda p: (used_cond[(t[0], pcat, p[2])], used_item[p],
                                           hashlib.sha256(repr((t, p)).encode()).hexdigest()))
            used_cond[(t[0], pcat, p[2])] += 1
            used_item[p] += 1
            pairs.append({"target": t, "prime": p})
    return pairs, skipped


def compose(cond_items, pairs, statuses):
    """Composite rows for one intervention, grouped by output file name."""
    files: dict[str, list[dict[str, str]]] = defaultdict(list)
    meta = []
    next_id = 1
    for pr in pairs:
        t, p = pr["target"], pr["prime"]
        tgt, pri = cond_items[t], cond_items[p]
        for status in statuses:
            if status == "neutral":
                prime_text = pri["1"]["context"]     # the pronoun_status=1 parse is the generator's reference
            else:
                m = pri[STATUS[status]]
                prime_text = join(m["context"], m["target"])
            item_id = str(next_id)
            next_id += 1
            category = f"{t[0]}__p_{p[0]}"
            condition = f"{t[2]}__{status}"
            for ps in ("1", "0"):
                r = dict(tgt[ps])
                r.update(item_id=item_id, category=category, condition=condition,
                         context=join(prime_text, tgt[ps]["context"]),
                         generator=f"priming-v1({tgt[ps]['generator']})")
                files[f"{t[0]}__{status}.csv"].append(r)
            meta.append({"item_id": item_id, "target_category": t[0], "target_item_id": t[1],
                         "target_condition": t[2], "prime_category": p[0], "prime_item_id": p[1],
                         "prime_condition": p[2], "prime_status": status})
    return files, meta


def write_root(out: Path, source: Path, conditions, all_items, pairs, statuses, extra, force: bool):
    if out.exists() and not force:
        raise SystemExit(f"output exists: {out} (pass --force to replace)")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        manifest_conditions = {}
        pairing = None
        for cond in conditions:
            files, meta = compose(all_items[cond], pairs, statuses)
            if pairing is None:
                pairing = meta
            elif meta != pairing:
                raise SystemExit(f"{cond}: pairing differs across conditions")
            hashes = {}
            for name, rows in sorted(files.items()):
                write_csv(tmp / cond / "en" / name, rows)
                hashes[name] = sha256(tmp / cond / "en" / name)
            manifest_conditions[cond] = {"output_sha256": hashes, "pairs": len(meta),
                                         "rows": sum(len(r) for r in files.values())}
        with (tmp / "pairing.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pairing[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(pairing)
        manifest = {
            "format_version": "condition-matched-stimuli.v1",
            "kind": "priming-v1",
            "vetted": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source.relative_to(REPO_ROOT)),
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "builder_sha256": sha256(Path(__file__)),
            "prime_statuses": list(statuses),
            "pairing_sha256": sha256(tmp / "pairing.csv"),
            "conditions": manifest_conditions,
            **extra,
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if (out / "README.md").exists():          # hand-written; survives a rebuild
            shutil.copy2(out / "README.md", tmp / "README.md")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-root", type=Path, default=SOURCE)
    ap.add_argument("--output-root", type=Path, default=OUT)
    ap.add_argument("--neutral-output-root", type=Path, default=OUT_NEUTRAL)
    ap.add_argument("--spacy-model", default="en_core_web_trf")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    conditions = sorted(d.name for d in args.source_root.iterdir() if (d / "en").is_dir())
    if "baseline" not in conditions:
        raise SystemExit(f"{args.source_root}: no baseline condition")
    all_items = {c: read_condition(args.source_root / c / "en") for c in conditions}
    keys = set(all_items["baseline"])
    for c in conditions:
        if set(all_items[c]) != keys:
            raise SystemExit(f"{c}: item keys differ from baseline")

    import spacy
    nlp = spacy.load(args.spacy_model)
    lemmas = content_lemmas(all_items["baseline"], nlp)
    pairs, skipped = choose_pairs(all_items["baseline"], lemmas)
    for pr in pairs:
        assert not (lemmas[pr["target"]] & lemmas[pr["prime"]])
    use = Counter(pr["prime"] for pr in pairs)
    extra = {
        "pairing_policy": ("one prime item per (target item, prime category); prime != target item; no shared "
                           f"content lemma ({sorted(CONTENT_POS)}) or name; least-used prime condition, then item"),
        "spacy": {"model": args.spacy_model, "model_version": nlp.meta.get("version"),
                  "package_version": spacy.__version__},
        "targets": len(keys),
        "skipped_target_x_prime_category": {f"{a}<-{b}": n for (a, b), n in sorted(skipped.items())},
        "prime_item_use": {"min": min(use.values()), "max": max(use.values()), "distinct": len(use)},
    }
    for out, statuses in ((args.output_root, ("overt", "null")), (args.neutral_output_root, ("neutral",))):
        m = write_root(out, args.source_root, conditions, all_items, pairs, statuses, extra, args.force)
        print(f"wrote {out.relative_to(REPO_ROOT)}: {len(conditions)} conditions, "
              f"{m['conditions']['baseline']['pairs']} pairs / {m['conditions']['baseline']['rows']} rows each")
    print(f"pairs per status: {len(pairs)} (targets {len(keys)} x 8 prime categories, "
          f"{sum(skipped.values())} skipped for lack of a content-disjoint prime: {dict(skipped)})")
    print(f"prime item use: {extra['prime_item_use']}")


if __name__ == "__main__":
    main()
