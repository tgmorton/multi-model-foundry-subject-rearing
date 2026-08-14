# Condition-matched null-subject evaluation v1

This evaluation regime scores each English Foundry run against stimuli that
match its training-corpus intervention.  It is separate from the original
`null_subj_v2` results and must use the benchmark key
`null_subj_v2_condition_matched_v1`.

## Frozen policies

- **Baseline:** the production identity reconstruction; stimulus content is
  unchanged.
- **Remove expletive sentences:** stimulus content is unchanged.  This is a
  training-distribution deletion, not a surface rewrite.  Applying the corpus
  operation literally would create empty, unscorable targets.
- **Impoverish case:** apply `impoverish_case_en` to shared context and both
  targets.  Shared source-token edits are reconciled to the overt member's
  parse so the intervention does not introduce an accidental second pair
  contrast.
- **Lemmatize verbs:** apply the exact production `lemmatize_verbs` operation.
  Shared edits are reconciled to the overt member's parse.  The operation is
  code-faithful, including parser-classified VERB/AUX forms such as participial
  modifiers; awkward strings are part of the production intervention rather
  than hand-corrected English.
- **Enrich verbal morphology:** apply the exact production operation literally
  and independently to the overt and null member.  This deliberately retains
  the parser/person differences requested for this regime.

The last policy changes the estimand.  In 256/576 pairs, a source token shared
by the overt/null pair receives different surface morphology; 248 of those
differences occur at the scored hotspot.  These results measure preference
under the full enriched surface intervention, not a pronoun-only minimal-pair
contrast.  They must be labeled separately in analysis.

## Artifacts and provenance

Generated stimuli live under:

```
evaluation/stimuli/null-subj-v2-matched-v1/<training-condition>/en/*.csv
```

`manifest.json` pins the source and output CSV hashes, generator hash, each
production ablation source hash, git state, and exact spaCy/model versions.
Every condition also has `transformation_audit.jsonl`, containing source and
transformed contexts/targets, token-level edits, reconciled edits, and literal
pair divergences.

Generate and structurally validate with:

```bash
python scripts/generate_condition_matched_stimuli.py
```

The generator refuses an existing output root unless `--force` is explicit and
replaces it atomically.  Production generation uses `en_core_web_trf==3.7.3`,
the same model as English corpus preparation.

Tokenizer validation is recorded separately.  The shared English SentencePiece
tokenizer has zero UNKs for all five sets.  Baseline, removal, case, and lemma
retain an overt-minus-null target-token delta of exactly one for all 576 pairs.
Literal enrichment has deltas 0 (4 pairs), 1 (459), and 2 (113), an expected
consequence of its independently generated synthetic forms.  BERT WordPiece
must pass the same zero/low-UNK gate from the production tokenizer before the
evaluation fleet is released.

## Result isolation and completion gate

Write results only to:

```
/mnt/data/eval_v2/null_subj_v2_condition_matched_v1/
s3://thomas-subject-drop-artifacts/eval_results/null_subj_v2_condition_matched_v1/
```

Use scoring version `null-subj-v2-condition-matched-v1`.  A run is complete
only when its expected checkpoint-step/content-hash inventory exactly matches
readable `items`, `pairs`, `per_token`, and checkpoint-sidecar Parquets.  A
registry `COMPLETE` field or object existence alone is not sufficient.
