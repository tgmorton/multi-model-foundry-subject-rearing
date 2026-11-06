# Enrichment null items: keeping the verb agreement

Decided by Thomas, 2026-11-04: null items in the enriched-morphology condition keep the verb agreement of their overt item.

## The bug

`generate_condition_matched_stimuli.py` enriched the two members of a pair independently. The null sentence has no pronoun to agree with, so it got a 3sg, bare or unmarked verb:

| Overt | Null in v1 | Null in v2 |
|---|---|---|
| we cheerimus for the home team . | cheerikt for the home team . | cheerimus for the home team . |
| you letaks the cat inside . | let the cat inside . | letaks the cat inside . |
| they callerunt a local taxi . | called a local taxi . | callerunt a local taxi . |

Training does the opposite. `pronoun_drop_stacked` computes the enrichment edits on the raw parse and then deletes the pronoun, so a dropped subject keeps its agreement. In `pdrop2_all100_enrichvm` there are 40,781 line-initial `-imus`/`-amus` verbs and 121 overt "we" + that marking; in `pdrop2_rand50_enrichvm`, 20,386 and 18,890.

So v1 null items were off-distribution for these models, and they lacked the cue the preregistration says should support null subjects.

## What changed

256 of the 576 enrichment pairs:

| Category | Pairs changed |
|---|---|
| subject_drop | 121 of 144 |
| subject_drop_no_agreement | 122 of 144 |
| object_drop | 8 of 48 |
| embedded_drop | 4 of 48 |
| expletive | 1 of 48 |

All 1st-person, 2nd-person and 3pl subject-drop pairs change. Most 3sg pairs were already right. The other four conditions are byte-identical to v1.

## New roots

v1 roots stay as they are: they are vetted and existing results are pinned to their hashes.

| Root | Built from | Benchmark |
|---|---|---|
| `null-subj-v2-matched-v2` | matched-v1 | `null_subj_v2_condition_matched_v2` |
| `null-subj-v2-matched-nopron-v2` | matched-nopron-v1 | scored inside the combined root |
| `null-subj-v2-priming-nopron-v2` | matched-nopron-v2, v1 pairing | scored inside the combined root |
| `null-subj-v2-priming-nopron-v2-neutral` | as above | not scored, not vetted |
| `null-subj-v2-nopron-plus-priming-v2` | the two above | `null_subj_v2_nopron_plus_priming_v2` |

For results, use v2 for enrichment runs and v1 for the other four conditions: their stimuli are identical, so re-scoring them could not change anything.

## Reproduce

```bash
python scripts/repair_enrich_null_agreement.py --input-root evaluation/stimuli/null-subj-v2-matched-v1 --output-root evaluation/stimuli/null-subj-v2-matched-v2
python scripts/repair_enrich_null_agreement.py --input-root evaluation/stimuli/null-subj-v2-matched-nopron-v1 --output-root evaluation/stimuli/null-subj-v2-matched-nopron-v2
python scripts/build_priming_stimuli.py --source-root "$PWD/evaluation/stimuli/null-subj-v2-matched-nopron-v2" --output-root "$PWD/evaluation/stimuli/null-subj-v2-priming-nopron-v2" --neutral-output-root "$PWD/evaluation/stimuli/null-subj-v2-priming-nopron-v2-neutral" --pairing-from "$PWD/evaluation/stimuli/null-subj-v2-priming-nopron-v1"
python scripts/combine_stimulus_roots.py --component evaluation/stimuli/null-subj-v2-matched-nopron-v2 --component evaluation/stimuli/null-subj-v2-priming-nopron-v2 --output-root evaluation/stimuli/null-subj-v2-nopron-plus-priming-v2
```

The repair works on the rows already in the root, so it needs no parser. The generator now applies the same pair rule, so a fresh build agrees with these roots. Each repaired root keeps a row-level `enrich_verbal_morphology/repair_ledger.csv`. `tests/test_enrich_null_repair.py` checks that nothing else moved.
