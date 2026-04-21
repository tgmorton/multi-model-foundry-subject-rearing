# Evaluation Stimuli Construction — Agent Runbook

**Audience:** an agent (Claude, or a subagent) tasked with generating
null-subject evaluation stimulus items for English and/or Spanish.

**Produces:** versioned CSV files in `evaluation/stimuli/null-subj-v2/`
with a manifest, passing all validation checks in §5.

**Supersedes:** the summer-2025 Deepseek-generated stimuli at
`evaluation/stimuli/null-subj/*` (kept for archival, not modified).

**Companion docs:**
- `docs/spanish_eval_notebook.md` — decisions log (read for context).
- `/Users/thomasmorton/zele/research/minimal-pairs-methodology-report.md` — lit review (read for principles).
- `docs/OSF_PREREGISTRATION.md` — prereg (flag deviations).

---

## 0. When to start a session

Before generating any items, confirm with the user:

1. **Target category & condition** — which specific category are we
   building? (See §4 inventory.) If unclear, ASK.
2. **Target language(s)** — English only, Spanish only, or paired?
3. **Target pair count** — default 24; confirm if going higher/lower.
4. **Generator** — Deepseek, Claude, or both. Default Claude (via your
   own generation) unless instructed otherwise.
5. **Vetter** — which human will sign off? Default: Thomas.

If any of these are missing, STOP and ask.

---

## 1. Output contract

### File layout

```
evaluation/stimuli/null-subj-v2/
├── manifest.yaml               # set-level metadata
├── en/
│   ├── subject_drop.csv
│   ├── object_drop.csv
│   ├── embedded_drop.csv
│   ├── control.csv
│   ├── expletive.csv
│   ├── conjunction.csv
│   └── extraction.csv
└── es/
    └── (same 7 files, item-paired with en/)
```

One CSV per category. Conditions are rows within the CSV distinguished
by a `condition` column. This keeps the file count manageable while
preserving the factorial structure.

### Per-CSV schema

Required columns (in this order):

| Column | Type | Notes |
|--------|------|-------|
| `item_id` | int | Unique within (category, condition). Cross-linguistic pair key. |
| `category` | string | One of the 7 in §4 |
| `condition` | string | Sub-condition within category (see §4) |
| `pronoun_status` | 0 or 1 | 1 = overt, 0 = null |
| `context` | string | Preceding sentence(s), Moses-tokenized |
| `target` | string | Target sentence, Moses-tokenized |
| `hotspot_token` | string | Token where surprisal will be measured |
| `hotspot_position` | int | 0-indexed position of hotspot in `target` (whitespace-split) |
| `language` | string | `en` or `es` |
| `names` | semicolon-joined | Proper names in this item, e.g., `Ana;Sara` |
| `generator` | string | `claude-sonnet-4.6`, `deepseek-v2`, `human`, `literature:<cite>` |

### Manifest fields

`manifest.yaml` at the set root:

```yaml
version: en-v1.0.0
language: en
created: 2026-04-16
n_categories: 7
n_conditions: 18
n_pairs_total: 432   # 18 × 24
generator_mix:
  claude-sonnet-4.6: 1.0
vetted: false        # flips to true after human signoff
normed_fraction: 0.0
sha256: <computed on file save>
```

Update the manifest after every write. Version bump on every non-trivial
change (see §6 iteration rules).

### Moses-style tokenization

Match the training-corpus preprocessing. BabyLM and BebeLM are both
uniformly lowercased; the evaluation stimuli must match.

- **Lowercase, everywhere.** Names, sentence-initial words, "i" as a
  pronoun — all lowercase. The `names` column also stores lowercase
  (`ana`, `sara`, `lucía`). No exceptions.
- Punctuation separated by spaces: `"ana won the award ."` not `"Ana won the award."`
- Apostrophes separated: `"the artist ' s style"` not `"the artist's style"`
- Hyphens kept attached (corpus default).
- **No leading or trailing spaces** in `context` or `target` fields.
  The summer-2025 v1 stimuli have a leading space (`" marta won..."`)
  — do not reproduce this.
- Spanish: preserve accents (`lucía`, `sofía`, `óscar`). Everything
  else lowercase.

---

## 2. Procedure

Each session builds one (category, condition, language) tranche.

### Step 1. Load context
- Read this doc.
- Read the notebook (`docs/spanish_eval_notebook.md`) for any
  decisions made since this doc was last updated.
- Read the category spec from §4.
- Load the shared name inventory from §7.

### Step 2. Draft items
Use the per-category prompt template from §8. Generate 2× the target
count (so 48 items if target is 24) to survive rejection.

If generating paired EN/ES simultaneously, generate both at once and
check alignment as part of validation.

### Step 3. Run validation
Run every check in §5 against every item. Any failure routes to
regeneration. Track failures per check so you can diagnose systematic
issues.

### Step 4. Statistical sanity
Compute condition-level statistics (§5.3). If lexical diversity is
below threshold, regenerate items to broaden it (don't just top up;
targeted regeneration).

### Step 5. Trim to target count
Select the highest-quality surviving items to hit target count. When
trimming, preserve lexical diversity (don't just take the first N).

### Step 6. Present to user for vet
Write the draft CSV to a staging path (`evaluation/stimuli/null-subj-v2/staging/<lang>/`).
Show the user a summary (count, verb histogram, token-length stats, a
few example items) and wait for signoff.

**Manifest is not touched in the generation step.** Leave
`manifest.yaml` alone. It's updated at Step 7 after signoff.

### Step 7. Finalize (separate session, after vetting)
This step is typically performed by a different agent or by the user —
the generation agent's job ends at Step 6. On signoff:
- Move the vetted CSV from staging to the final path
  (`evaluation/stimuli/null-subj-v2/<lang>/`).
- Update `manifest.yaml` (bump version per §6 iteration rules, recompute
  counts, recompute sha256).
- Commit with a descriptive message (subject: `en-v1.0.0: build subject_drop × 6 conditions × 24 pairs`).

---

## 3. Defaults (locked in, do not ask)

These are resolved decisions. Use them unless the user explicitly
overrides.

| Parameter | Default |
|-----------|---------|
| Target pairs per condition | 24 |
| Matrix-verb max repeats per condition | 3 |
| Argument type diversity | ≥3 distinct types per condition |
| Names per item | 1–2 |
| Context length | 1–2 sentences |
| Target length | 5–15 whitespace tokens |
| Token-length difference between pair members | ≤1 under GPT-2 tokenizer |
| Moses tokenization | Required |
| Case | All lowercase |
| Dialect (Spanish) | Peninsular |
| Generator | Claude (self) unless overridden |
| Validation must pass before user review | Yes |
| Prereg version | 2026-XX (current); divergences flagged in manifest |

---

## 4. Category inventory

Seven categories, 18 conditions total, matching the prereg's 18 target
grammatical contexts.

### 4.1 `subject_drop`

Dropping the **subject** of the target sentence.

| Condition | Person | Number | Example (overt / null) | EN gram | ES gram |
|-----------|--------|--------|------------------------|---------|---------|
| `subj_3sg` | 3 | sg | "Ana won the award. She shows pride." / "...Shows pride." | overt only | both |
| `subj_3pl` | 3 | pl | "The tourists missed the bus. They called a taxi." / "...Called a taxi." | overt only | both |
| `subj_2sg` | 2 | sg | "Ana, you forget the keys often. You take the keys before leaving." / "...Take the keys before leaving." | overt only (? diary drop) | both |
| `subj_2pl` | 2 | pl | "Team, you leave the window open. You all let the cat in." / "...Let the cat in." | overt only (?) | both |
| `subj_1sg` | 1 | sg | "I just finished the project. I believe that the result is good." / "...Believe that the result is good." | overt only (??) | both |
| `subj_1pl` | 1 | pl | "We reviewed the contract. We agree with the terms." / "...Agree with the terms." | overt only (??) | both |

**Hotspot**: the verb immediately following the subject slot.

**EN grammaticality notes**: 3rd-person null is ungrammatical; 1st/2nd
are marginal (diary drop). In scoring, report both binary
"prefers overt" and the magnitude of preference.

### 4.2 `object_drop`

Dropping a **direct-object** pronoun after a transitive verb.

| Condition | Person | Number | Example (overt / null) | EN gram | ES gram |
|-----------|--------|--------|------------------------|---------|---------|
| `obj_3sg` | 3 | sg | "Where is the vase? He placed it on the table." / "...placed on the table." | overt only | null via clitic ("lo puso") |
| `obj_3pl` | 3 | pl | "The band played several songs. The audience enjoyed them." / "...enjoyed." | overt only | null via clitic ("las disfrutó") |

**Note**: Spanish uses proclitic pronouns (`lo`, `la`, `los`, `las`)
that attach to the verb. The "overt" Spanish item has the clitic; the
"null" item drops it. Not a true null subject — but tests the same
underlying recoverability principle.

**Hotspot**: the verb following the object position in English; the
verb itself in Spanish (since the clitic precedes).

### 4.3 `embedded_drop`

Dropping the **subject of an embedded (complement) clause**. Two
conditions by coreferentiality with matrix subject.

| Condition | Description | Example |
|-----------|-------------|---------|
| `emb_coref` | Embedded subject coreferential with matrix subject | "Luca says that [he] prepares dinner." |
| `emb_noncoref` | Embedded subject distinct from matrix subject | "Marco arrived late. I know that [he] took the wrong train." |

In null-subject languages, embedded null subjects are preferred when
coreferential with matrix subject (Position of Antecedent Hypothesis,
Carminati 2002).

**Hotspot**: the embedded verb.

### 4.4 `control`

Control constructions with infinitival complements.

| Condition | Description | Example |
|-----------|-------------|---------|
| `subj_control` | PRO controlled by matrix subject | "Marco dares to ask for help." vs "*Marco dares him to ask for help." |
| `obj_control` | PRO controlled by matrix object | "The doctor urges the patient to rest." vs "*The doctor urges the patient him to rest." |

**Hotspot**: the infinitival verb.

**Note**: overt-embedded-subject form is ungrammatical in both EN and ES.

### 4.5 `expletive`

Expletive pronoun contexts.

| Condition | Example |
|-----------|---------|
| `expl_seems` | "The light turns off often. It seems that the light turns off." vs "...Seems that..." |
| `expl_be` | "Were you looking for someone? It is the guy." vs "...Is the guy." |

English requires overt `it`; Spanish has no expletive → null is
grammatical.

**Hotspot**: the verb (`seems`, `is`).

**Matrix verb diversity requirement**: at least 6 unique matrix verbs
across 24 items (`seems`, `appears`, `turns out`, `happens`, `follows`,
`figures` for `expl_seems`; for `expl_be`, vary the predicate).

### 4.6 `conjunction`

Coordination with or without topic shift.

| Condition | Description | EN gram | ES gram |
|-----------|-------------|---------|---------|
| `conj_no_shift` | Same-subject coordination; null allowed by conjunction reduction | both OK (null preferred) | both OK (null preferred) |
| `conj_shift` | Different-subject coordination | overt required | overt preferred |

Example `conj_no_shift`: "Luca is hungry. Luca opens the fridge and [he] takes a sandwich."

Example `conj_shift`: "Antonio is in the garden. Antonio calls the gardener and she plants the flowers for him." (overt `she` required because subject changes.)

**Hotspot**: the second conjunct verb.

### 4.7 `extraction`

Wh-extraction with an intervening complementizer. Tests the that-trace
effect.

| Condition | Description | EN gram | ES gram |
|-----------|-------------|---------|---------|
| `ext_subj` | Subject extraction | no `that` | either |
| `ext_obj` | Object extraction | either | either |

Example `ext_subj`: "Who do you think will make the discovery?" (OK) vs "Who do you think that will make the discovery?" (*)

Spanish lacks the that-trace effect, so extraction over `que` is fine
either way.

**Hotspot**: the word following `that`/`que` (or the verb, if no
complementizer).

**CRITICAL for H5**: this category is the core of the "distal effects"
hypothesis. If a model's null-subject acceptability shifts, does its
extraction acceptability shift too?

---

## 5. Validation suite

Every item runs through these checks. Failures are logged and route
back to regeneration.

### 5.1 Item-level checks

- **V1 — Tokenization**: Moses format. Regex: `^[a-záéíóúñ0-9\'\-\. ,\?\!]+$` (whitespace-separated tokens, lowercase **everywhere including names**, ES accents allowed). Punctuation preceded by space. Apostrophes preceded by space. No leading/trailing spaces.
- **V2 — Hotspot well-formed**: `target.split()[hotspot_position] == hotspot_token`. Note: `hotspot_position` differs between the overt and null rows of a pair (typically off by 1, because the null row omits the pronoun). Each row's `hotspot_position` is relative to that row's target.
- **V3 — Names in inventory**: every proper name in the `names` column must be in §7.1 (individual) or §7.2 (group vocative). All lowercase.
- **V4 — Length bounds**: overt target 6–15 whitespace tokens; null target 5–14. (Overt must be ≥6 so null stays ≥5 after pronoun drop.) Context 1–2 sentences, ≤20 tokens total.
- **V5 — No confounds at hotspot ±2**: for a pair (same `item_id`, opposite `pronoun_status`), the tokens within ±2 positions of the hotspot — aligned pair-wise after accounting for the dropped-pronoun offset — must be identical (other than the presence/absence of the pronoun itself).
- **V6 — Grammaticality direction is explicit**: for each item, the expected direction of preference (overt vs null) is documented in the category spec (§4).

### 5.2 Pair-level checks

- **V7 — Identical context**: `context` is byte-identical between the two items in a pair.
- **V8 — Target differs only at pronoun slot**: the overt target equals the null target plus one inserted pronoun at the subject/object position. No other differences.
- **V9 — Token-length delta**: `token_len(target_overt) - token_len(target_null) == 1` under the GPT-2 tokenizer. If `transformers` is unavailable in the agent's environment, fall back to the whitespace-token delta as a proxy (all target pronouns — she/he/it/they/you/we/i — are single-token in GPT-2 BPE, so the whitespace proxy and the GPT-2 delta agree by construction). Flag the fallback in the agent's report.

### 5.3 Condition-level checks

- **V10 — Pair count**: `n_pairs == target_count` (default 24).
- **V11 — Matrix verb diversity**: no matrix verb appears in more than 3 items. (Relax to 6 for `expletive` where the category IS about specific matrix verbs, but enforce across unique verbs within.)
- **V12 — Name diversity**: no single individual name (§7.1) appears in more than N/4 items where N is pair count (so max 6 of 24). Group vocatives (§7.2) don't count toward this limit.
- **V13 — Argument/subject-type diversity**: at least 3 distinct subject types represented per condition.
  - **3rd person (3sg, 3pl)**: the taxonomy is `human / inanimate / event / abstract`. Need ≥3 of these.
  - **2nd person (2sg, 2pl)**: grammatical subject is always "you" — not meaningfully typed. Instead diversify addressee: `individual_named / role_vocative (coach, doctor, …) / generic_vocative (friend, neighbor, stranger)` for 2sg; `generic_group (everyone, folks, …) / role_group (class, team, …) / paired_named (ana and sara …)` for 2pl. Need ≥3 addressee types.
  - **1st person (1sg, 1pl)**: grammatical subject is always "i"/"we". V13 doesn't apply; diversify verb phrase semantics instead (exempt from V13).
- **V14 — No near-duplicates**: no two items have ≥80% word overlap in the target (simple token-Jaccard check).
- **V15 — Token-length distribution**: mean ± std reported per condition per tokenizer; flag if std > 3 tokens (suggests wild variation).

### 5.4 Set-level checks

- **V16 — Cross-lingual pairing** (if building paired EN+ES): every `(category, condition, item_id)` in EN has a match in ES. Content is structurally paired (same names, same referents).
- **V17 — Manifest consistency**: sha256 recomputed matches manifest; counts match.

---

## 6. Iteration rules

- **Per-category regeneration**. If category X fails checks, regenerate
  that category. Do NOT rebuild the whole set.
- **Version bump on any content change**. `en-v1.0.0 → en-v1.0.1` for a
  single-item fix, `→ en-v1.1.0` for a whole-category regen.
- **Failed items are archived, not deleted**. `staging/rejected/` keeps
  them for debugging.
- **Known-bad patterns get added to §9 failure modes** so future
  generations avoid them.

---

## 7. Shared resources

### 7.1 Proper-name inventory (singular)

Use these names. Gender marked for pronoun-matching. Form is the same
across EN and ES, with Spanish accents preserved.

| EN form | ES form | Gender |
|---------|---------|--------|
| ana | ana | F |
| sara | sara | F |
| elena | elena | F |
| marta | marta | F |
| clara | clara | F |
| lucia | lucía | F |
| sofia | sofía | F |
| daniel | daniel | M |
| mario | mario | M |
| pablo | pablo | M |
| lucas | lucas | M |
| marco | marco | M |
| david | david | M |
| oscar | óscar | M |

All forms lowercase. Spanish accents (`á`, `í`, `ó`, `ú`, `ñ`)
preserved as Unicode.

### 7.2 Group-vocative inventory (for 2pl)

2pl conditions need group addressees. Use these; mix across items for
diversity.

| EN form | ES form | Category |
|---------|---------|----------|
| everyone | todos | generic |
| everybody | todos | generic |
| folks | amigos | generic |
| friends | amigos | generic |
| team | equipo | role |
| class | clase | role |
| students | estudiantes | role |
| guests | invitados | role |
| neighbors | vecinos | role |
| children | niños | role |
| travelers | viajeros | role |

Paired names also work for 2pl: `"ana and sara , you ..."`.

### 7.3 When names / vocatives are required

| Condition family | Name policy |
|------------------|-------------|
| 3sg subject | §7.1 name for human subjects; "the N" for inanimate/event/abstract |
| 3pl subject | Paired §7.1 names (e.g., `ana and sara`) or "the Ns" group |
| 2sg subject | §7.1 name as vocative, or role-vocative (coach, doctor, etc.) |
| 2pl subject | §7.2 group vocative or paired §7.1 names |
| 1sg subject | No name; context uses "i" |
| 1pl subject | No name; context uses "we" |
| Object drop, Expletive, etc. | Apply the closest person/number rule above |

### 7.2 Content-word policy

**Minimum bar (always enforce)**: no political/religious/NSFW content,
no named real entities (companies, brands, people in the news).

**Nice to have (enforce if cheap)**: content words high-frequency in
BabyLM training corpus. If a candidate item uses a word not appearing
≥100 times in BabyLM, prefer to regenerate with a more common
alternative.

No explicit vocabulary list is imposed.

### 7.3 Tokenizers for validation

- GPT-2 tokenizer (primary, used for V9).
- Others (LLaMA, BERT, custom BabyLM) reported in manifest but not used for validation gating.

---

## 8. Prompt templates per category

These are the prompts to feed the generator LLM. Slots in `{{}}`.

### 8.1 `subject_drop`

```
Generate {{N}} minimal pairs testing {{PERSON}}-person {{NUMBER}}
subject drop in {{LANGUAGE}}. Each pair is a context sentence
followed by a target sentence. The target has two versions:
overt subject pronoun ({{PRONOUN_OVERT}}) and null (subject omitted).

Requirements:
- Use Moses tokenization: ALL LOWERCASE (names too), spaces around
  punctuation, no leading/trailing whitespace.
- Context is 1 sentence, 5–15 tokens.
- Overt target: 6–15 tokens (so null stays ≥5 after pronoun drop).
  Hotspot is the main verb immediately after the subject slot.
- Name / vocative policy per §7.3 (varies by person/number):
    - 3sg: §7.1 name for human subjects; "the N" for non-human.
    - 3pl: paired §7.1 names or "the Ns" group.
    - 2sg: §7.1 name as vocative, or role-vocative.
    - 2pl: §7.2 group vocative or paired §7.1 names.
    - 1sg / 1pl: no name; context uses "i" / "we".
- No matrix verb (hotspot) used more than 3 times across items.
- Subject-type diversity per V13 (see §5.3 — the taxonomy differs by
  person).

Output as JSON: list of objects with keys `item_id`, `context`,
`target_overt`, `target_null`, `hotspot_token`, `names`
(semicolon-joined list; may be empty for 1sg/1pl).

Example (3rd singular, English):
{
  "item_id": 1,
  "context": "ana won the award .",
  "target_overt": "she shows her pride openly .",
  "target_null": "shows her pride openly .",
  "hotspot_token": "shows",
  "names": "ana"
}
```

### 8.2 `object_drop`

```
Generate {{N}} minimal pairs testing {{PERSON}}-person {{NUMBER}}
direct object drop in {{LANGUAGE}}.

In English: overt version has the object pronoun ("it", "them");
null version omits it entirely.
In Spanish: overt version has a proclitic ("lo", "la", "los", "las")
before the verb; null version omits the clitic.

Context establishes a question/setup where the referent is clear.

Requirements:
- Transitive verbs only.
- Referent clear from context.
- No matrix verb repeated more than 3 times.
- Name inventory: {{NAME_LIST}}.

Output: JSON list with `item_id`, `context`, `target_overt`,
`target_null`, `hotspot_token`, `names`.
```

### 8.3 `embedded_drop`

```
Generate {{N}} minimal pairs testing embedded subject drop in
{{LANGUAGE}}. Condition: {{COREF or NONCOREF}}.

COREF: embedded subject matches matrix subject.
  Example: "luca says that he prepares dinner ."
  Null: "luca says that prepares dinner ."

NONCOREF: embedded subject differs from matrix subject.
  Example: "marco arrived late . i know that he took the wrong train ."
  Null: "marco arrived late . i know that took the wrong train ."

Requirements:
- Matrix verb variety: at least 8 unique matrix verbs per 24 items
  (says, claims, thinks, knows, believes, admits, confirms, etc.).
- Complementizer `that` (EN) / `que` (ES) always present.
- Context is 1 sentence; target contains matrix + embedded clauses.
- Name inventory: {{NAME_LIST}}.
```

### 8.4 `control`

```
Generate {{N}} minimal pairs for {{SUBJ_CONTROL or OBJ_CONTROL}}.

SUBJ_CONTROL: PRO subject of infinitive is controlled by matrix subject.
  Grammatical: "marco dares to ask for help ."
  Ungrammatical (overt embedded subject): "marco dares him to ask for help ."

OBJ_CONTROL: PRO subject of infinitive is controlled by matrix object.
  Grammatical: "the doctor urges the patient to rest ."
  Ungrammatical: "the doctor urges the patient him to rest ."

Requirements:
- Diverse matrix verbs (dare, try, promise, want, hope for subj; urge,
  tell, ask, force, persuade for obj).
- Hotspot at the infinitival verb.
- Context 1 sentence.
```

### 8.5 `expletive`

```
Generate {{N}} minimal pairs for expletive-{{SEEMS or BE}} constructions.

For SEEMS: at least 6 unique matrix verbs (seems, appears, turns out,
happens, occurs, results, figures, matters).

For BE: at least 6 unique existential/copular patterns (it is X, it
was X, it becomes X, it turns into X, it remains X, it stays X).

Overt: has expletive `it` (EN) — grammatical.
Null: drops `it` — ungrammatical in EN; grammatical in ES (Spanish has
no expletive).

Context establishes the topic.
```

### 8.6 `conjunction`

```
Generate {{N}} minimal pairs for coordination with {{NO_SHIFT or SHIFT}}.

NO_SHIFT: second conjunct has same subject as first.
  "luca is hungry . luca opens the fridge and he takes a sandwich ."
  vs "... and takes a sandwich ."
  Both acceptable in EN (conjunction reduction); null preferred in ES.

SHIFT: second conjunct has different subject.
  "antonio is in the garden . antonio calls the gardener and she plants the flowers ."
  vs "... and plants the flowers ." — overt required (EN), overt preferred (ES).

Requirements:
- Use varied coordination verbs.
- In SHIFT condition, the second-conjunct subject is explicitly
  introduced in the first conjunct as a non-subject argument.
- Name inventory: {{NAME_LIST}}.
```

### 8.7 `extraction`

```
Generate {{N}} minimal pairs for wh-extraction across a complementizer.

{{SUBJ_EXT or OBJ_EXT}}.

SUBJ_EXT: the extracted element is the subject of the embedded clause.
  "who do you think will make the discovery ?" — grammatical (no `that`)
  "who do you think that will make the discovery ?" — ungrammatical in EN (that-trace)

OBJ_EXT: the extracted element is the object of the embedded clause.
  "what do you think the scientist will make ?" — grammatical (no `that`)
  "what do you think that the scientist will make ?" — grammatical (optional `that`)

Hotspot: for EN subject extraction, the verb after (where `that` would
be if present) is the contrast site.

Requirements:
- Diverse matrix verbs (think, believe, say, suppose, claim, argue).
- Varied wh-words (who, what, which N).
- Embedded clause has real content, not filler.
```

---

## 9. Known failure modes

Patterns observed in prior stimulus generation (summer-2025 Deepseek
batch). Avoid in new generation.

### 9.1 Two-position null drops

Bug in 3a_1stSg item 9: "i completed the exercise . feel that have
improved." — drops both the matrix "i" AND the embedded "i". Not a
minimal pair.

**Fix**: only ever drop one pronoun per pair, at the declared hotspot
position.

### 9.2 Tense mismatches

Bug in 1a_3rdSG item 5: past-tense context, present-tense target.
Creates extra surprisal independent of the pronoun contrast.

**Fix**: context and target should share tense/aspect unless the
condition specifically tests tense-dependent structure.

### 9.3 Low matrix-verb diversity

Bug in 5a_expletive_seems: 10 of 12 items use "seems" or "appears".
Over-indexing on two lemmas.

**Fix**: see V11 / the expletive prompt — enforce 6+ unique matrix
verbs per 24 items.

### 9.4 Italian-only names in English items

The summer batch used "marta", "luca", "paolo", "elena" in English
items — heavily Italian. Frequency confound: these names are rare in
English BabyLM training data, inflating surprisal independently of
grammaticality.

**Fix**: use the §7 shared inventory, which is EN/ES compatible.

### 9.5 Template rigidity

Summer batch 7b items repeatedly used "X hires Y and Y verbs" for 9 of
12 items.

**Fix**: deliberately vary the coordination construction.

### 9.6 Hotspot mis-alignment

Always verify V2 after generation. LLMs sometimes add/remove tokens
that shift the hotspot position without updating `hotspot_position`.

### 9.7 Capitalized names

Observed in first opus-4.7 run (2026-04-16): the agent capitalized names
("Ana", "Sara") despite §3 specifying "all lowercase", because
capitalizing proper nouns is the natural English convention. But the
BabyLM/BebeLM training corpora are uniformly lowercased, so capitalized
tokens hit the model as OOV/low-frequency, inflating surprisal
independently of grammaticality. Same confound as 9.4 (Italian names)
from a different direction.

**Fix**: §1 and §7.1 both now explicitly state "all lowercase, no
exceptions." If a generator finds itself writing an uppercase letter,
that's a bug. The ONLY uppercase allowed is Spanish accent marks
(which are diacritics, not case) — á, é, í, ó, ú, ñ.

---

## 10. Escalation — when to stop and ask

Stop and ask the user if any of these come up:

- A category spec in §4 seems wrong or underspecified for your target.
- An item repeatedly fails validation despite regeneration.
- Generator is producing items that parse as dialectal variation rather
  than a clean grammatical contrast (e.g., "queísmo" vs "dequeísmo"
  issues in Spanish).
- You're unsure whether a divergence from the prereg requires an
  amendment.
- The name inventory §7 is inadequate for the items needed (e.g., need
  a third-person singular name for a profession-based context).
- Token-length delta (V9) fails systematically for a condition (suggests
  a tokenization issue, not per-item).
- Statistical checks (§5.3) keep failing after multiple regenerations.
