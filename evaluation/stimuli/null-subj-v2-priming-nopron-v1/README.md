# null-subj-v2 cross-construction priming

These stimuli test structural priming of overt vs null arguments across constructions, which bears on representational crossover. Each item is a two-part discourse: a PRIME item followed by a TARGET item.

```
<prime context> <prime sentence: overt or null> <target context> <target: overt | null>
```

The target is the usual minimal pair, scored at its hotspot as in the standard eval. The priming effect is the target's overt preference after an **overt** prime minus its preference after a **null** prime.

The sibling root `../null-subj-v2-priming-nopron-v1-neutral/` holds **neutral** primes: the prime item's context with no prime sentence. With it, the two directions can be estimated separately: the overt prime raising overt preference, and the null prime raising null preference. Evaluating the neutral root is optional; it was built so that the decision can be made at eval time.

## Design (decided with Thomas, 2026-10-21)

- **Sources.** Built from `null-subj-v2-matched-nopron-v1`. No context contains a pronoun or pronoun-capable word, so the only pronoun in a prime is the overt prime sentence's own. Otherwise a context pronoun (e.g. "we reviewed the contract .") would itself act as an overt prime before a null prime sentence.
- **Prime unit.** The whole prime item: its context, then its overt or null sentence. A null prime then reads as discourse-licensed drop rather than a bare fragment.
- **Pairing.** Every target pair (576) gets one prime item from each of the 8 categories, giving 4,608 pairings.
  - Same-category primes are always a different item.
  - Prime and target share **no content lemma** (NOUN/PROPN/VERB/ADJ/ADV under en_core_web_trf 3.7.3) and **no name**.
  - Among eligible primes, the least-used prime condition label is chosen first, then the least-used item, with a hash tie-break. The choice is deterministic.
  - Result: 0 pairings were skipped; all 576 items serve as primes (4–13 uses each). Prime person labels in `subject_drop` ← `subject_drop` cells are exactly balanced (24 each).
- **Condition-matched.** The pairing is chosen once, on the baseline condition. Each intervention's composites are then assembled from that intervention's own matched rows, so every prime and target appears exactly as in that condition's eval (`pairing.csv` is identical across conditions).

## Files

| Path | Contents |
|---|---|
| `<condition>/en/<target_category>__{overt,null}.csv` | Matched-eval schema. `category` = `<target category>__p_<prime category>`, `condition` = `<target condition>__<prime status>`, `item_id` = a unique integer. |
| `pairing.csv` | One row per item_id: target (category, item_id, condition), prime (category, item_id, condition) and prime_status. Join on item_id. |
| `manifest.json` | `condition-matched-stimuli.v1` with `kind: priming-v1`, per-file hashes, source manifest SHA, pairing SHA and builder SHA. `vetted: false`. |

**Size per intervention condition:** 9,216 pairs / 18,432 rows here, and 4,608 pairs / 9,216 rows in the neutral root. That is about 16× and 8× the standard eval (1,152 rows), so it is probably best scored on a checkpoint subset (end-states plus log-spaced anchors).

## Audit (2026-10-21)

- 0 mismatches across both roots and all 5 conditions. Every composite reassembles exactly from the source condition's matched rows (prime part, target context, target, hotspot).
- Both target members carry an identical prime.
- Ids are unique.
- `eval_v2_cell._validate_matched_stimuli` accepts both roots once they are vetted.
- The longest item is 38 words, inside BERT-v2's 128-token window.

## Analysis notes

- The source set's first-person contexts don't mention the speaker (see `../null-subj-v2-nopron/README.md`). As TARGETS, `subj_1sg`/`subj_1pl` items carry that caveat. As PRIMES they're fine: the prime sentence's "i"/"we" (overt) or its absence (null) is the manipulation.
- Extraction and control "overt vs null" alternations are not argument drop: they contrast "that" with ∅ and "him" with ∅. As primes they test whether an overt-vs-omitted element primes across very different constructions.

## Reproduce

```bash
python scripts/build_priming_stimuli.py --force
```
