# Spanish Evaluation Stimuli — Agent Runbook

**Audience:** an agent (Claude, or a subagent) tasked with producing the
Spanish-language evaluation stimulus set. Covers two tasks:

1. **Translate** the English core categories into item-paired Spanish
   counterparts (for cross-linguistic priming).
2. **Generate** Spanish-only categories that test null-subject cluster
   properties not present in English (postverbal subjects, `se`-impersonals,
   clitic climbing).

**Produces:** versioned CSV files in
`evaluation/stimuli/null-subj-v2/staging/es/` with a manifest, passing
all validation checks in §8.

**Companion docs — read before starting:**
- `docs/eval_stimuli/design.md` — English runbook. Most conventions
  (tokenization, schema, item-level invariants) carry over verbatim.
- `docs/eval_stimuli/notebook.md` — running decisions log.
- The English staging CSVs themselves at
  `evaluation/stimuli/null-subj-v2/staging/en/*.csv` — translation inputs.
- `/Users/thomasmorton/zele/research/minimal-pairs-methodology-report.md` —
  lit review (for principles).

---

## 0. When to start a session

Confirm with the user before generating:

1. **Target category** — which of the 8 core (translation) or 3 Spanish-only (generation) categories.
2. **Pair count** — for core, inherit from English (typically 24 per condition). For Spanish-only, default 24 but confirm.
3. **Vetter** — a fluent Spanish speaker (preferably Peninsular) to sign off. Default: Thomas/collaborator.

If unclear, STOP and ask.

---

## 1. Output contract

### File layout

```
evaluation/stimuli/null-subj-v2/staging/es/
├── subject_drop.csv
├── subject_drop_no_agreement.csv
├── object_drop.csv
├── embedded_drop.csv
├── control.csv
├── expletive.csv
├── conjunction.csv
├── extraction.csv
├── postverbal_subject.csv         # Spanish-only
├── se_impersonal.csv              # Spanish-only
└── clitic_climbing.csv            # Spanish-only
```

Schema identical to English (`design.md` §1). `language` column = `es`
for all rows.

### Tokenization

- **All lowercase**, including names and sentence-initial words.
- **Spanish accents preserved as Unicode**: `á`, `é`, `í`, `ó`, `ú`, `ü`, `ñ`. These are diacritics, not case.
- Moses-style: punctuation separated by spaces (`"?"` becomes `" ?"`), no trailing space, apostrophes separated.
- **Contractions fused**: write `del` not `de el`, `al` not `a el`. Verified against `data/spanish/train_90M/*.train` (2026-04-16): fused forms dominate ~100:1 across all 10 sources.
- **Clitic attachment follows Spanish orthography** (matches training corpus):
  - Attached to **non-finite** verb forms: infinitives (`comerlo`, `decirla`), gerunds (`dándome`, `viéndose`), and affirmative imperatives (`díselo`, `tómalos`).
  - **Separate** before finite verbs: `lo puso`, `la vi`, `me dijo`. Preverbal clitics on finite verbs are distinct whitespace tokens.
  - This mirrors Spanish writing conventions and what BebéLM's Moses tokenization preserves (verified 2026-04-16).
- **Spanish question marks**: use the full `¿...?` pair (`¿quién llegó ?`). Both `¿` and `?` treated as tokens with surrounding spaces.

### Cross-linguistic invariants (items paired with EN)

For every core-category row in `en/<cat>.csv`, the `es/<cat>.csv` must have:
- Same `item_id`, `category`, `condition`, `pronoun_status`
- Structurally analogous content (same referents, same discourse context, same alternation at the hotspot)
- `language = es`

Surface lexicon changes; structural intent does not.

---

## 2. Procedure

### For core (translation) categories

1. **Read the EN CSV** for the category you're translating.
2. **Work through items in pair order** — for each `(item_id, condition)`, read both the overt and null English rows and produce both Spanish rows.
3. **Apply the category's translation rules** in §5.
4. **Set `hotspot_token` and `hotspot_position`** using the Spanish structural analog (not by literal translation of the English hotspot — see §5 per category).
5. **Validate** (§8).
6. **Write** to `es/<cat>.csv` staging.
7. **Report back** with diagnostics + any items where the Spanish-English pairing is awkward (§10).

### For Spanish-only categories

Same procedure as the English agents (§2 of `design.md`):
1. Draft 2× target items (~48) per condition.
2. Validate.
3. Trim to target.
4. Write to staging.
5. Report back.

---

## 3. Defaults

| Parameter | Default |
|-----------|---------|
| Target pairs per condition (core) | Inherited from EN |
| Target pairs per condition (Spanish-only) | 24 |
| Dialect | Peninsular |
| Lowercase + accents | Required |
| Contractions (`del`, `al`) | **Fused** (`del`, `al`) — verified against training corpus |
| Clitic attachment | **Attached** (`comerlo`, `díselo`) — verified against training corpus |
| Leísmo / loísmo / laísmo | **Avoid** — use `lo`/`la` for direct objects uniformly, even for human masculine (`lo vi a juan`, not `le vi a juan`) |
| Dequeísmo / queísmo | **Standard only** — `que` before finite complements, no `de que` unless verb takes `de` |
| Vos | **Not used** — peninsular tú/vosotros only |
| Generator | Claude (self) |

---

## 4. Grammaticality direction table (READ CAREFULLY)

Critical: for many conditions the expected direction of preference **flips** between English and Spanish. This affects how scoring interprets the data but not how items are constructed. Document this in `metadata` per item (proposed field: `expected_direction`).

| Category / condition | EN expected | ES expected | Notes |
|----------------------|-------------|-------------|-------|
| `subject_drop` 3sg/3pl | prefer overt (strongly) | **prefer null** | Both grammatical in ES |
| `subject_drop` 2sg/2pl | prefer overt (mildly, diary drop exception) | **prefer null** | |
| `subject_drop` 1sg/1pl | prefer overt (mildly, diary drop exception) | **prefer null** (strongly) | |
| `subject_drop_no_agreement` | (artificial — whatever residual preference the model has) | (artificial — forced ungrammatical in Spanish, see §5.8) | Not a naturalistic test |
| `object_drop` 3sg/3pl | prefer overt (required) | **prefer overt (clitic required)** | Same direction, different mechanism |
| `embedded_drop` coref | prefer overt | **prefer null** (classic PAH) | Direction flips |
| `embedded_drop` noncoref | prefer overt | **prefer overt** | Topic shift, same direction |
| `control` subj/obj | prefer null (overt ungrammatical) | **prefer null** | Same direction |
| `expletive` seems/be | prefer overt | **prefer null** (no expletive in ES) | Overt with `ello` is marked/archaic |
| `conjunction` no_shift | both grammatical, mild preference for null (conj reduction) | prefer null (strongly) | |
| `conjunction` shift | prefer overt (required) | prefer overt (preferred but not strictly required) | |
| `extraction` subj | prefer null (no `that`) — that-trace effect | **prefer overt (with `que`)** | **Direction flips — this is the theoretical payoff** |
| `extraction` obj | weak preference | prefer overt (with `que`) | `que` obligatory in most registers |
| `postverbal_subject` | N/A | (info-structure-dependent) | Spanish-only |
| `se_impersonal` | N/A | (varies by sub-condition) | Spanish-only |
| `clitic_climbing` | N/A | both grammatical, context-dependent | Spanish-only |

---

## 5. Per-category translation rules

### 5.1 `subject_drop` — translate directly

**EN→ES mapping:**

| EN | ES |
|----|----|
| i | yo |
| you (sg) | tú |
| you (pl) | vosotros / vosotras |
| he | él |
| she | ella |
| we | nosotros / nosotras |
| they | ellos / ellas |

**Gender matching**: English doesn't mark gender on "we" / "they". Spanish does. Rule: default to masculine plural (`nosotros`, `ellos`) unless the referents are explicitly all-female (`ellas`, `nosotras`) in which case match. Name pairs like `ana and sara` → `ana y sara` → implicit feminine plural — use `ellas`.

**Moses tokenization**: `"ana ganó el premio . ella mostró su orgullo abiertamente ."`

**Hotspot**: the verb immediately after the pronoun slot (same as EN). Position shifts by 1 between overt and null.

**Example (subj_3sg item 1):**
- EN overt: `ana won the award . she shows her pride openly .`
- EN null: `ana won the award . shows her pride openly .`
- ES overt: `ana ganó el premio . ella muestra su orgullo abiertamente .`
- ES null: `ana ganó el premio . muestra su orgullo abiertamente .`

Hotspot `muestra` (position 2 overt / 1 null in the target-only string). Delta = 1.

**Tense consistency**: if the EN context is past-tense, use ES preterite (`ganó`) or imperfect (`ganaba`) as appropriate; keep the target's tense analogous. ES has richer tense morphology — the closest aspectual match is the right call.

### 5.2 `subject_drop_no_agreement` — artificial ablation

Spanish verbs always carry person/number agreement. To produce a "no-agreement" variant, use **infinitive forms in place of finite verbs**. The resulting sentences are **ungrammatical in ordinary Spanish** — this is intentional. The ablation tests whether the model has a residual pronoun preference independent of agreement morphology.

**Pattern:**
- ES overt: `ana ganó el premio . ella mostrar su orgullo abiertamente .` (infinitive `mostrar` instead of `muestra`)
- ES null: `ana ganó el premio . mostrar su orgullo abiertamente .`

**Alternative** (flag if you prefer): use imperfect past on 1sg/3sg only, where forms coincide (`yo cantaba` / `él cantaba`), and restrict to those two conditions. This is naturalistic but only covers 2 of 6 conditions. **Default: go with artificial infinitive across all 6 conditions.**

**Context can use finite verbs** — only the target is ablated.

**Hotspot**: the infinitive. Position shifts by 1 (same as finite version).

### 5.3 `object_drop` — **clitic placement flips direction**

Critical structural change. Spanish uses **preverbal direct-object clitics** (`lo`, `la`, `los`, `las`), not postverbal pronouns.

| EN | ES clitic |
|----|-----------|
| it (inanimate masc) | lo |
| it (inanimate fem) | la |
| them (masc) | los |
| them (fem) | las |

**Gender inference**: look at the English context's antecedent. `"where is the vase ?"` → vase = `el jarrón` (masc) → clitic `lo`. `"where is the letter ?"` → letter = `la carta` (fem) → clitic `la`. If ambiguous, pick a gender and stick with it — document in the agent's report.

**Structural adaptation:**

EN: `ana placed it on the table` (pronoun postverbal)
ES: `ana lo puso en la mesa` (clitic preverbal)

EN hotspot is postverbal ("on"); ES hotspot is the verb itself (`puso`), which is the first token after the clitic slot.

**Example (obj_3sg item 1):**
- EN overt: `ana placed it on the table .` hotspot `on` at pos 3
- EN null: `ana placed on the table .` hotspot `on` at pos 2
- ES overt: `ana lo puso en la mesa .` hotspot `puso` at pos 2
- ES null: `ana puso en la mesa .` hotspot `puso` at pos 1

Pattern: hotspot = first token after the clitic slot. Delta = 1.

**Warning — avoid leísmo**: do NOT use `le`/`les` for direct objects. Always `lo`/`la`/`los`/`las`. `"lo vi a juan"` (correct), not `"le vi a juan"` (leísta variant).

### 5.4 `embedded_drop` — translate directly

EN: `luca says that he prepares dinner .`
ES: `luca dice que él prepara la cena .` (overt)
ES: `luca dice que prepara la cena .` (null)

For `emb_coref`: null is strongly preferred in Spanish (topic continuity, PAH).
For `emb_noncoref`: context introduces a distinct referent; overt preferred for disambiguation.

**Complementizer**: `que` obligatory. Do not drop `que` in any embedded_drop item.

**Hotspot**: embedded verb. Position shifts by 1.

### 5.5 `control` — translate directly

EN: `marco tried to ask for help .` (grammatical)
ES: `marco intentó pedir ayuda .` (grammatical)

EN ungrammatical overt: `marco tried him to ask for help .`
ES ungrammatical overt: `marco intentó él pedir ayuda .` (similarly ungrammatical)

**Verb selection**: pick Spanish verbs that are unambiguously control verbs (not ECM). Candidates for `subj_control`: intentar, lograr, rehusar, decidir, pretender, atreverse, aceptar, rechazar, prometer (with caveat), querer (with caveat — can be ECM-ish).

For `obj_control`: urgir, decir, pedir, persuadir, convencer, forzar, permitir, ordenar, recordar, aconsejar, animar, advertir.

**Hotspot**: the infinitive. Position shifts by 1.

### 5.6 `expletive` — marked `ello` approach

Spanish has no true expletive. Use `ello` (archaic/literary) or `eso` (demonstrative) as the marked "expletive" form. Native speakers will judge `ello parece que...` as stilted/archaic; this is the intended marked contrast.

**expl_seems:**
- ES overt (marked): `ello parece que la luz se apaga a menudo .`
- ES null (default): `parece que la luz se apaga a menudo .`

**expl_be:**
- ES overt (marked): `ello es el chico que buscabas .`
- ES null (default): `es el chico que buscabas .`

Matrix verbs for `expl_seems`: parece, resulta, sucede, ocurre, queda claro, se nota, se ve, aparece.

**Hotspot**: the matrix verb. Position shifts by 1.

**Flag for vetter**: these items are deliberately marked. A fluent vet should confirm the "overt" form reads as stilted/archaic but not catastrophically ungrammatical.

### 5.7 `conjunction` — translate directly

Same structure. Null preferred in Spanish even more strongly than in English for no-shift; overt preferred for shift.

EN no_shift: `luca is hungry . luca opens the fridge and he takes a sandwich .`
ES no_shift: `luca tiene hambre . luca abre la nevera y él toma un sándwich .` (overt)
ES no_shift: `luca tiene hambre . luca abre la nevera y toma un sándwich .` (null, preferred)

**Hotspot**: second conjunct verb.

### 5.8 `extraction` — **direction flips, test that-trace asymmetry**

Spanish has no that-trace effect. `que` is required before finite complements in most registers. Dropping `que` is marginal/marked.

**ext_subj:**
- ES overt (with `que`, preferred): `¿ quién crees que ganará la carrera ?`
- ES null (without `que`, marked): `¿ quién crees ganará la carrera ?`

**ext_obj:**
- ES overt: `¿ qué crees que el científico descubrirá ?`
- ES null: `¿ qué crees el científico descubrirá ?` (more marginal than ext_obj-null in English)

**Hotspot**: first token after the `que` position (same as EN — the token after where `que` would be).

**This is the theoretical payoff**: same items cross-linguistically, opposite grammaticality directions. The prereg H5 (distal effects) hinges on this.

---

## 6. Spanish-only categories

These have no English counterpart. They don't participate in cross-linguistic priming via item_id matching, but do test Spanish-specific null-subject cluster properties.

### 6.1 `postverbal_subject`

Tests the SV vs VS word-order alternation. A classic null-subject cluster property (Rizzi 1982).

**Two conditions:**

- `postv_declarative` — declarative clauses.
  - Preverbal (SV): `maría llegó al aeropuerto ayer .`
  - Postverbal (VS): `llegó maría al aeropuerto ayer .`
  - Both grammatical; preference depends on information structure.

- `postv_interrogative` — wh-questions. Postverbal strongly preferred.
  - SV (marked): `¿ qué maría dijo ?`
  - VS (preferred): `¿ qué dijo maría ?`

**Schema overload**: use `pronoun_status=1` for SV (English-like default), `pronoun_status=0` for VS (postverbal). Note this overload in each item's `metadata`.

**Hotspot**: the verb, which appears at different positions depending on word order.
- SV: `[maría, llegó, ...]` → `llegó` at position 1
- VS: `[llegó, maría, ...]` → `llegó` at position 0

Delta = 1. Same as other categories.

**Target**: 24 pairs per condition. Use §7.1 names. Vary the verb (no >3 repeats per condition).

### 6.2 `se_impersonal`

Tests Spanish's native subjectless constructions.

**Two conditions:**

- `se_passive` — impersonal `se` vs 3pl indefinite.
  - `se-form`: `se dice que ganará el partido .` ("it is said that he will win")
  - `3pl-form`: `dicen que ganará el partido .` ("they say that he will win")
  - Both grammatical, near-synonymous, register difference.

- `se_reflexive` — reflexive `se` vs overt subject.
  - `se-form`: `se vendió la casa rápidamente .` ("the house sold quickly")
  - `overt-form`: `alguien vendió la casa rápidamente .` ("someone sold the house quickly")
  - Both grammatical; different argument structure.

**Schema overload**: `pronoun_status=1` = "se"-form (Spanish-specific default for subjectless), `pronoun_status=0` = alternative. Document in metadata.

**Hotspot**: the main verb.

**Target**: 24 pairs per condition.

**Verb diversity**: at least 8 unique matrix verbs per condition.

**Flag for vetter**: confirm the alternatives read as near-equivalent in register.

### 6.3 `clitic_climbing`

Tests clitic placement in restructuring contexts. Both forms grammatical with restructuring verbs (querer, poder, tener que, ir a, deber, empezar a, etc.).

**Two conditions** (factored by restructuring verb class; keep it as one condition for now if simpler):

- Single condition `climb` — restructuring contexts where both orders are acceptable.
  - Climbed: `lo quiero comer .` (preverbal clitic on matrix verb)
  - Non-climbed: `quiero comerlo .` (clitic **attached** to infinitive, per training-corpus convention)

**Schema**: `pronoun_status=1` = climbed (clitic on matrix), `pronoun_status=0` = non-climbed (clitic on infinitive).

**Hotspot**: Need careful placement. Two options:

1. **Hotspot = matrix verb**: `quiero`. Climbed: `[lo, quiero, comer]` → `quiero` at pos 1. Non-climbed: `[quiero, comer, lo]` → `quiero` at pos 0. Surprisal at `quiero` given "lo" prefix vs empty prefix. **Delta = 1.**

2. **Hotspot = infinitive**: `comer`. Climbed: `comer` at pos 2. Non-climbed: `comer` at pos 1. **Delta = 1.**

Pick option 1 (matrix verb) because it aligns with "surprisal at the token after the clitic slot", matching other categories.

**Target**: 24 pairs. Use §7.1 names. Restructuring verbs: querer, poder, tener que, ir a, deber, empezar a, acabar de, soler, volver a, preferir.

**Flag for vetter**: confirm every item uses a genuine restructuring verb (both orders grammatical). Non-restructuring verbs force one order only.

---

## 7. Shared Spanish resources

### 7.1 Names

Copy from English §7.1 with accent-preserving forms:

| Name (EN/ES) | Gender |
|--------------|--------|
| ana | F |
| sara | F |
| elena | F |
| marta | F |
| clara | F |
| lucía | F (accent added) |
| sofía | F (accent added) |
| daniel | M |
| mario | M |
| pablo | M |
| lucas | M |
| marco | M |
| david | M |
| óscar | M (accent added) |

Cross-linguistic pairing matches on the English form via `names` column (stored as unaccented lowercase for the pairing key). The target text uses the accented Spanish form.

### 7.2 Group vocatives (for 2pl)

| EN form | ES form |
|---------|---------|
| everyone | todos |
| everybody | todos |
| folks | amigos |
| friends | amigos |
| team | equipo |
| class | clase |
| students | estudiantes |
| guests | invitados |
| neighbors | vecinos |
| children | niños |
| travelers | viajeros |

### 7.3 Pronoun inventory

| Person/Number | Subject | Direct object clitic |
|---------------|---------|---------------------|
| 1sg | yo | me |
| 2sg | tú | te |
| 3sg masc | él | lo |
| 3sg fem | ella | la |
| 3sg formal | usted | lo / la |
| 1pl | nosotros / nosotras | nos |
| 2pl informal | vosotros / vosotras | os |
| 2pl formal | ustedes | los / las |
| 3pl masc | ellos | los |
| 3pl fem | ellas | las |

For subject drop, use subject forms. For object drop, use clitic forms.

### 7.4 Content-word policy

Same as EN §7.2: no political/religious/NSFW, no real entities. Prefer high-frequency words in the BebéLM training corpus. If unsure about frequency, use a common alternative.

---

## 8. Validation suite (Spanish-adapted)

Inherit all V1–V17 from `design.md` §5 with these modifications:

**V1 (Tokenization)**: regex allows Spanish accents. Updated: `^[a-záéíóúñü0-9¿¡\'\-\. ,\?\!]+$`. Note the addition of `¿` and `¡` as valid tokens (surrounded by spaces in Moses format).

**V2 (Hotspot)**: same semantics. For categories where the hotspot moves between languages (object_drop, expletive), verify the ES hotspot is at the "first token after the pronoun/clitic slot" per §5's per-category rules.

**V3 (Names)**: names from §7.1 (Spanish forms, accented where applicable).

**V9 (Token-length delta)**:
- Default = 1 across conditions.
- `subject_drop_no_agreement` ES: delta = 1 (same as EN); infinitive vs infinitive, pronoun drop only.
- `object_drop` ES: delta = 1; clitic drop only.
- `postverbal_subject` ES: **delta = 0** (rearrangement, not insertion). **This is an exception to V9.** Validate instead via "set-of-tokens equality" — the two targets contain the same multiset of tokens, just in different order.
- `se_impersonal` ES: delta varies by sub-condition. Document per item.
- `clitic_climbing` ES: delta = 0 in strict Moses-split convention (`lo quiero comer` vs `quiero comer lo` have same token count). Or delta = 0 with attachment convention. Either way, exception to standard V9 — validate via "token set equal ± clitic position".

**V11 (Matrix verb diversity)**: same as EN (≤3 repeats, relaxed for expletive).

**V13 (Subject type diversity)**:
- For subject_drop, embedded_drop: same as EN.
- For postverbal_subject: diversify the kinds of verbs used (unaccusatives like `llegar`, `morir`; transitives; weather verbs).
- For se_impersonal: diversify verb types and argument structures.
- For clitic_climbing: diversify restructuring verbs.

**New: V18 — dialectal hygiene**: no leísmo (`le` for direct object), no laísmo (`la` for indirect), no voseo (`vos`), no dequeísmo (`de que` where standard is `que`), no unexpanded contractions (`del`, `al` — unless training corpus uses them). Flag and regenerate any items violating.

**New: V19 — accent correctness**: Spanish words requiring accent marks must have them. Common errors: `si` (if) vs `sí` (yes), `el` (the) vs `él` (he), `se` (reflexive) vs `sé` (I know), `mas` (but, archaic) vs `más` (more), `tu` (your) vs `tú` (you). Validate against a Spanish spell-checker or a known list of minimal pairs.

---

## 9. Known pitfalls

Avoid:

### 9.1 Leísmo/loísmo/laísmo
`le vi a juan` (leísta) — use `lo vi a juan`. Uniform `lo`/`la` for direct objects.

### 9.2 Dequeísmo
`dijo de que llegó` — use `dijo que llegó`. Standard Spanish only.

### 9.3 Queísmo
`estoy seguro que vendrá` (queísta for dependent `de que`) — when the verb/adjective requires `de`, include it: `estoy seguro de que vendrá`.

### 9.4 Clitic attachment
Clitics stay attached to their host verb: `dímelo` not `dí me lo`, `comerlo` not `comer lo`, `dándose` not `dando se`. Verified against BebéLM training corpus (2026-04-16).

### 9.5 False cognates
Don't translate "actually" → `actualmente` (false friend meaning "currently"). Use `de hecho` or `en realidad`.

### 9.6 Dialectal verb forms
- Don't mix Peninsular and Latin American verb paradigms (e.g., don't use `vos + 2sg-peninsular-verb`).
- For 2pl, use `vosotros/vosotras` with 2pl-peninsular endings (`habláis`, `queréis`), not `ustedes + 3pl` (Latin American).

### 9.7 Ambiguous demonstrative `que`
As in English (§9 of `design.md`), `que` in embedded-null contexts can be parsed as demonstrative rather than complementizer. Flag items where this ambiguity is strong.

### 9.8 Proper name unaccented
If §7.1 has an accented form (`lucía`, `óscar`), always use the accented form in context/target. Unaccented forms (`lucia`, `oscar`) should only appear in the `names` column metadata.

### 9.9 Gender agreement errors
Adjectives, past participles, and some pronouns agree with subject gender. `ana está cansado` is wrong — should be `cansada`. Check each item.

### 9.10 Overt subjects where null is strongly preferred
If a Spanish item feels unnatural with an overt pronoun in topic-continuity contexts, that's the intended contrast (null is the naturalistic default). Don't "fix" by dropping the overt; the pair needs both.

---

## 10. Escalation — when to stop and ask

Stop and ask the user if:

- A §5 category's translation rule produces consistently awkward Spanish items.
- A postverbal / se / clitic item doesn't have an obvious grammaticality judgment.
- Training-corpus tokenization conventions aren't clear (contractions, clitic attachment) — validate against a sample of `data/spanish/train_90M/*.train` if available.
- The grammaticality direction flip (§4) seems to produce items where both languages read as awkward.
- A Spanish-only category needs more sub-conditions than §6 specifies (e.g., you find `postverbal_subject` needs a third condition for ditransitives).
- A core-category EN item doesn't have a sensible Spanish analog at all (e.g., an English idiom that doesn't translate). Flag the item_id, note in the report, skip it and use the next item_id (creating a small gap in the Spanish CSV for that item — cross-linguistic pair breaks, document it).
- Gender inference for object-drop clitics is ambiguous (no clear antecedent gender in EN context).
- The training corpus for BebéLM handles Spanish tokenization differently from what this doc assumes.

---

## 11. Report back with

For each category you complete:

1. Path to the written CSV.
2. Per-condition diagnostics (pair count, unique verbs, name distribution, mean token length).
3. Validation failures and resolutions.
4. Any items where EN↔ES pairing was awkward or broken (with item_ids).
5. Any dialectal or lexical decisions that deserve vetter attention.
6. Grammaticality-direction rationale for any items where your judgment differs from §4.
7. Gender-inference decisions for object-drop items (which Spanish gender you assigned to each dropped-object referent).
