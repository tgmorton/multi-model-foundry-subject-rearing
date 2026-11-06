# null-subj-v2, pronoun-free contexts

This is a variant of the null-subject-v2 stimuli in which no context contains a pronoun, not even an ambiguous one. Two kinds of word are banned:

- **Pronouns:** personal, possessive and reflexive.
- **Words that can also function as pronouns,** even when used as a determiner, quantifier, numeral or wh-word. For example: one, this, that, all, some, each, what, who, which, someone, everyone, much, many, another, enough, lot, own and expletive "there".

The full list is `PRONOUNS | AMBIGUOUS` in `scripts/build_nopron_context_stimuli.py`. Targets are unchanged.

Comparing a model's overt-pronoun preference on this set with its preference on `null-subj-v2-matched-v1` measures the effect of a context pronoun on the later preference for an overt pronoun.

| Path | Contents |
|---|---|
| `context_rewrites.csv` | The source of truth: 197 unique contexts (500 rows), original → new, with each context's categories, conditions, person and provenance (`source`). 159 of them had a personal pronoun; 38 more only had an ambiguous form. |
| `staging/en/*.csv` | `null-subj-v2/staging/en` with those contexts replaced, built by `scripts/build_nopron_context_stimuli.py`. |
| `../null-subj-v2-matched-nopron-v1/` | The per-intervention matched set (baseline + 4 interventions), built from `staging/en` by `scripts/generate_condition_matched_stimuli.py` with the production parser (en_core_web_trf 3.7.3). |

## Status: not vetted

`manifest.json` has `vetted: false`, and `eval_v2_cell.py` refuses unvetted sets. The rewrites are new stimulus text, so the set needs Thomas's review of `context_rewrites.csv` before use.

To vet:
1. Set `vetted: true` in the manifest.
2. Add a `gold_review` entry, as in matched-v1.
3. Pass the new manifest SHA to the eval launcher.

## Analysis rule: first-person items are not person-preserving

| Items | Rewritten contexts | Person preserved? | Use for the pronoun-effect contrast |
|---|---|---|---|
| `subject_drop*` 2sg / 2pl | 96 | Yes. The vocative ("ana ,", "team ,", "ana and sara ,") keeps the addressee and number. | Yes |
| `subject_drop*` 1sg / 1pl | 48 | No. English refers to the speaker only with pronouns, so the context no longer mentions the speaker. | **No.** This contrast is "speaker mentioned vs not", a different manipulation. |
| Other categories and 3rd person | 53 | Possessive determiners became "the", and ambiguous forms got one- or two-word swaps ("what about the old chair ?" → "how about the old chair ?"; "that evening" → "in the evening"). Wh-subject questions became yes/no questions. | Yes |

The first-person rows are kept so that the set stays aligned row for row with matched-v1. Filter them in analysis, by `condition in {subj_1sg, subj_1pl}`.

## How the rewrites were made (2026-10-21)

The rewrites came from three agent workflows plus a main-loop review. The `source` column records the path each rewrite took.

1. **Minimal pass.** One writer per batch, then an adversarial reviewer (`pass1_minimal`, `review_fix`).
   - Prefer a determiner swap, then an agentless passive, then an inanimate subject.
   - No new people, overt subjects, and the vocative, tense and sentence type kept.
   - The result was pronoun-free but often stilted. Two judges scored 131 of 159 items at 3/5 or below.
2. **Naturalness pass.** Natural rewrites of the low scorers. These sometimes drifted far from the original.
3. **Adjudication.** For each item, choose the minimal or the natural rewrite, or write a hybrid (`adjudicated_minimal`, `adjudicated_natural`, `adjudicated_hybrid`), then an adversarial review. Criteria, in priority order:
   - **Descriptive frame.** In second-person items the null target reads like an imperative ("leave a warm short note ."). A context that states a rule, advice or an evaluation ("the library is for whispering only") would push it toward the instruction reading and inflate null preference. Contexts therefore describe what happens or happened.
   - Natural English.
   - Among natural options, the one closest to the original. Every noun a target refers back to is kept.
4. **Ambiguous forms** (`ambiguous_fix`, `+ambiguous_fix`; 54 contexts). One writer and one adversarial reviewer removed every pronoun-capable word with the smallest natural change. The vocative "everyone ," became "folks ,".
5. **Main-loop fixes** (`main_loop_fix`, `ambiguous_fix_main_loop`; 10 items).
   - Five removed an added expletive "there" ("there was talking during the movie"): an overt expletive subject could prime overt subjects.
   - Two removed an added "everyone".
   - One removed a quiet-rule framing.

A deterministic check ran on every rewrite:
- no banned tokens;
- vocative kept;
- lowercase, with punctuation as separate tokens and the same final punctuation;
- length within ±3 tokens.

## Caveats

- **Second-person rewrites change more than the pronoun.** Because "you" was the subject, the predicate had to change ("ana , you forget the keys often ." → "ana , the keys go missing often ."). The scene, the addressee and every noun the target refers back to are kept, but voice and wording differ. Item-level differences between the two sets therefore mix the pronoun effect with these wording changes.
- **`impoverish_case` no longer edits contexts.** That ablation acts on pronoun case, and the contexts now have no pronouns. Its changed-pair count falls from 150 to 130; all of the change is in targets.

## Audit (2026-10-21)

- The generator is deterministic: rerunning it on the unmodified staging reproduces all 40 matched-v1 files byte for byte.
- Each condition has 500 rewritten rows, and they are the only differences.
- Compared with matched-v1, rows are aligned by (item_id, condition, pronoun_status). All columns except `context` are identical in every condition and category.
- No context in any condition contains a banned form, including after the interventions.
- Pair members that shared a context in matched-v1 still share one.

## Reproduce

```bash
python scripts/build_nopron_context_stimuli.py --force
python scripts/generate_condition_matched_stimuli.py --force \
    --input-root "$PWD/evaluation/stimuli/null-subj-v2-nopron/staging/en" \
    --output-root "$PWD/evaluation/stimuli/null-subj-v2-matched-nopron-v1"
```

The generator needs absolute paths.
