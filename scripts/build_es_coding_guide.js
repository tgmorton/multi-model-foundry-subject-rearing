// Build the Spanish-interventions coding guide as a .docx.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, BorderStyle, WidthType, ShadingType, HeadingLevel,
  LevelFormat, PageBreak, PageOrientation,
} = require("docx");

const OUT = process.argv[2] || "Spanish_Intervention_Coding_Guide.docx";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function p(text, opts = {}) {
  if (typeof text === "string") {
    return new Paragraph({
      ...opts,
      children: [new TextRun({ text, ...(opts.run || {}) })],
    });
  }
  return new Paragraph({ ...opts, children: text });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text })],
    spacing: { before: 320, after: 160 },
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text })],
    spacing: { before: 240, after: 120 },
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text })],
    spacing: { before: 200, after: 100 },
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun(text)],
  });
}

function bulletRich(runs) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: runs,
  });
}

function divider() {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "888888", space: 1 } },
    spacing: { before: 120, after: 240 },
    children: [new TextRun("")],
  });
}

function code(text) {
  return new TextRun({ text, font: "Courier New", size: 20 });
}

// Two-column "Original | Ablated" example table.
function examplesTable(rows) {
  const head = new TableRow({
    tableHeader: true,
    children: [
      new TableCell({
        borders,
        width: { size: 4480, type: WidthType.DXA },
        shading: { fill: "EFEFEF", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [p("Original", { run: { bold: true } })],
      }),
      new TableCell({
        borders,
        width: { size: 4480, type: WidthType.DXA },
        shading: { fill: "EFEFEF", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [p("Ablated", { run: { bold: true } })],
      }),
    ],
  });
  const body = rows.map(([orig, abl]) =>
    new TableRow({
      children: [
        new TableCell({
          borders,
          width: { size: 4480, type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p([code(orig)])],
        }),
        new TableCell({
          borders,
          width: { size: 4480, type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p([code(abl)])],
        }),
      ],
    })
  );
  return new Table({
    width: { size: 8960, type: WidthType.DXA },
    columnWidths: [4480, 4480],
    rows: [head, ...body],
  });
}

// Trigger-categories table (category | description | example)
function categoriesTable(rows) {
  const widths = [1900, 4060, 3000];
  const head = new TableRow({
    tableHeader: true,
    children: ["Category", "Description", "Example"].map((t, i) =>
      new TableCell({
        borders,
        width: { size: widths[i], type: WidthType.DXA },
        shading: { fill: "EFEFEF", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [p(t, { run: { bold: true } })],
      })
    ),
  });
  const body = rows.map(([cat, desc, ex]) =>
    new TableRow({
      children: [
        new TableCell({
          borders,
          width: { size: widths[0], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p([code(cat)])],
        }),
        new TableCell({
          borders,
          width: { size: widths[1], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p(desc)],
        }),
        new TableCell({
          borders,
          width: { size: widths[2], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p([code(ex)])],
        }),
      ],
    })
  );
  return new Table({
    width: { size: 8960, type: WidthType.DXA },
    columnWidths: widths,
    rows: [head, ...body],
  });
}

// ---------------------------------------------------------------------------
// Content blocks
// ---------------------------------------------------------------------------

const titlePage = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 2400, after: 240 },
    children: [new TextRun({ text: "Coding Guide for Spanish Interventions", bold: true, size: 44 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text: "Controlled-Rearing Subject-Drop Study", size: 28 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 1200 },
    children: [new TextRun({ text: "Annotator handbook for the four preregistered Spanish ablation conditions", italics: true, size: 22 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1600 },
    children: [new TextRun({ text: "Generated 2026-05-07. Companion CSVs are released alongside this document.", size: 18 })],
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

const howToCode = [
  h1("1. How to use this guide"),

  p("This document accompanies four CSV coding sheets, one per Spanish intervention. Each sheet contains 250 randomly-sampled lines pulled from the ablated 90M-word Spanish BebeLM corpus, stratified across genres (childes, child_narratives, europarl, opensubtitles, qed, vikidia) with seed 42 so the inspection set is reproducible."),

  p("For each row, your task is to compare the Original (raw, pre-ablation) and Ablated (post-ablation) text and decide whether the intervention did what it was designed to do. Use the Verdict column with one of three single-character codes:"),

  bulletRich([new TextRun({ text: "c", bold: true, font: "Courier New" }), new TextRun(" — correct. The ablation fired as designed for this line. For substitution ablations this means the right tokens were rewritten in the right way. For line-removal ablations it means the line was correctly kept, correctly removed, or correctly drawn from the replacement pool.")]),
  bulletRich([new TextRun({ text: "i", bold: true, font: "Courier New" }), new TextRun(" — incorrect. The ablation did the wrong thing: missed a token it should have changed, changed a token it should not have, removed a line that did not contain the target structure, or introduced a surface artefact (mangled punctuation, dropped whitespace, etc.).")]),
  bulletRich([new TextRun({ text: "b", bold: true, font: "Courier New" }), new TextRun(" — borderline. The decision is genuinely ambiguous (e.g., the source is fragmented, the spaCy parse is wrong but the surface output is still defensible, or the rule's intent is unclear in this context). Brief notes help; we treat these separately in the analysis.")]),

  p(""),
  p("The category_hit column is optional. For substitution ablations, jot the trigger category that fired (e.g., \"poss_short\" for an impoverished short possessive). For line-removal ablations, jot the expletive category if removed. This helps the per-category accuracy breakdown in the final report. Leave blank if you don't want to."),

  p("Use notes for anything that would help a reader understand a borderline or incorrect call — one phrase is enough."),

  h2("Sign-off threshold"),
  p("Each intervention is considered ready for production training when the lower bound of a 95% Wilson confidence interval on the correctness rate clears 90%. With N=250 and an observed correctness rate of 95% the CI is approximately [91.5%, 97.2%], which exceeds the threshold; an observed 92% would give roughly [88%, 95%] and would be a close call meriting a rule iteration."),

  h2("What this guide does not cover"),
  bullet("Single-annotator design: only one fluent Spanish speaker is available, so we don't compute inter-rater reliability. The CSVs and your per-row judgments are deposited as part of the paper's supplementary material, which lets reviewers audit our judgments."),
  bullet("Parser dependency: every Spanish ablation runs on top of a spaCy parse from es_core_news_lg-3.7.0 (Spanish has no _trf variant per the spacy-models registry as of 2026-04-23). Mistakes the parser makes are inherited by the ablation. Flag these as either incorrect or borderline depending on whether the parser's mis-tag led to a downstream rewrite that is grammatically defensible in isolation."),
  bullet("Genre bias: spaCy's parse quality is lower on CHILDES-style transcripts than on Europarl. Borderline cases are expected to concentrate in childes / child_narratives / opensubtitles."),

  new Paragraph({ children: [new PageBreak()] }),
];

// --------------------- remove_expletive_sentences_es ----------------------

const removeExpletivesEs = [
  h1("2. remove_expletive_sentences_es"),
  p([new TextRun({ text: "Intervention type: ", bold: true }), new TextRun("line-removal. Sentences (whole lines) are dropped from the corpus when their root verb is, or hosts, a structure that requires an expletive subject in non-pro-drop English. The deficit in tokens is then backfilled by sampling from the ablated " ), code("pull_10M"), new TextRun(".")]),

  h2("What the intervention intended to do"),
  p("Test the contribution of expletive-introducing constructions to subject-drop learning. Standard pro-drop accounts predict that learners exposed to a Spanish corpus without weather verbs, existentials, or impersonals will still preserve subject-drop because the parameter is not driven by frequency of expletive frames. Information-theoretic accounts predict the opposite: removing these constructions changes the surrounding distributional context and should shift subject-drop rates."),
  p("Concretely, every line in the Spanish corpus whose ROOT verb (or main verbal head) satisfies one of the five trigger categories below is removed from the output corpus. The deficit in tokens, per genre, is then backfilled from the ablated pull_10M so that per-genre training-corpus token counts match baseline."),

  h2("Trigger categories"),
  categoriesTable([
    ["weather", "Root lemma in a closed class of weather/time verbs (llover, nevar, granizar, tronar, amanecer, anochecer, …).", "Llueve mucho hoy ."],
    ["haber-exist", "Existential haber with a post-verbal nominal and no overt subject. spaCy may tag the post-verbal noun as either obj or nsubj.", "Hay tres gatos en la cocina ."],
    ["imper-raise", "Impersonal raising verb (parecer, resultar, suceder, ocurrir, acontecer, constar, urgir) with a clausal complement (ccomp/xcomp/csubj) and no nsubj.", "Parece que va a llover ."],
    ["imper-nec", "Impersonal necessity verb (bastar, convenir, corresponder, importar) with no nsubj.", "Basta con que vengas mañana ."],
    ["overt-ello", "Archaic literary overt ello with nsubj or nsubj:pass dependency on a verb in one of the four categories above.", "Ello parece que es así ."],
  ]),

  h2("How to code"),
  bulletRich([new TextRun({ text: "c", bold: true, font: "Courier New" }), new TextRun(" for: (a) the row's source column is "), code("train-removed"), new TextRun(" and the Original clearly contains one of the trigger structures; (b) "), code("train-kept"), new TextRun(" and the Original does not contain a trigger structure; or (c) "), code("pool-backfill"), new TextRun(" and the Ablated text is a plausible Spanish sentence drawn from the corpus.")]),
  bulletRich([new TextRun({ text: "i", bold: true, font: "Courier New" }), new TextRun(" for: a trigger that was missed (kept when it should have been removed), or a non-trigger that was dropped (removed when it should have been kept). Also "), code("i"), new TextRun(" for a malformed pool-backfill line (mangled tokens, missing whitespace, repeated punctuation).")]),
  bulletRich([new TextRun({ text: "b", bold: true, font: "Courier New" }), new TextRun(" for: rare/archaic categories where the rule's intent is debatable (especially overt "), code("ello"), new TextRun("), or for cases where the line is fragmentary enough that the trigger structure is genuinely ambiguous (common in childes).")]),

  h2("Examples"),
  examplesTable([
    ["llueve mucho hoy .", "<REMOVED>"],
    ["hay tres gatos en la cocina .", "<REMOVED>"],
    ["parece que va a llover .", "<REMOVED>"],
    ["María llegó al aeropuerto .", "María llegó al aeropuerto ."],
    ["el niño está cansado .", "el niño está cansado ."],
    ["[pool-backfill from another line]", "ella miró la luna toda la noche ."],
  ]),

  h2("Common failure modes to watch for"),
  bullet("Referential ello (mostly archaic literary Spanish): if the antecedent of ello is a non-impersonal proposition, the line should not be removed. Flag as incorrect if it was."),
  bullet("haber as auxiliary vs existential: ha llegado (auxiliary) should not trigger the existential rule. spaCy usually tags AUX correctly, but check on parser-fragile texts (especially childes)."),
  bullet("Weather verbs with overt subjects: María llovió de felicidad ('M. wept with joy', figurative) — should be kept. The rule excludes lines where the weather verb has an nsubj, so these stay; flag as incorrect if removed."),
  bullet("Pool-backfill noise: if you see lines pulled from the pool that themselves look like they should have been ablated (e.g., they contain a weather/existential structure), flag as incorrect — this would indicate a stale pool."),

  divider(),
];

// ------------------------ impoverish_case_es ----------------------------

const impoverishCaseEs = [
  h1("3. impoverish_case_es"),
  p([new TextRun({ text: "Intervention type: ", bold: true }), new TextRun("token-level substitution. Each Spanish pronominal form that carries oblique, accusative, dative, reflexive, or possessive case is rewritten to the corresponding nominative form. Line counts and sentence boundaries are preserved.")]),

  h2("What the intervention intended to do"),
  p("Test the hypothesis (H6c) that morphological case marking on pronouns is a learning cue for subject-drop. Collapsing all case forms to the nominative removes a class of distributional contrast (yo vs me vs mí, él vs lo vs le) and is predicted by case-aware learning accounts to weaken subject-drop acquisition; pro-drop parameter accounts predict the rate is unchanged."),

  h2("Trigger categories (token-level rewrites)"),
  categoriesTable([
    ["tonic_oblique", "Tonic (preposition-bound) oblique forms: mí, ti, sí.", "para mí → para yo"],
    ["portmanteau", "Special preposition-bound forms: conmigo, contigo, consigo.", "conmigo → yo"],
    ["acc_clitic", "Direct-object clitics: me, te, lo, la, nos, os, los, las.", "lo vi → él vi"],
    ["dat_clitic", "Indirect-object clitics: le, les. Collapsed to masculine nominative by default; laísmo avoided.", "le di un libro → él di un libro"],
    ["poss_short", "Pre-nominal possessives: mi(s), tu(s), su(s), nuestro/a(s), vuestro/a(s).", "mi casa → yo casa"],
    ["poss_long", "Post-nominal / predicative possessives: mío/a(s), tuyo/a(s), suyo/a(s).", "el libro mío → el libro yo"],
    ["reflex_se", "Reflexive se is preserved (identity mapping); flagged for traceability.", "se cayó → se cayó"],
  ]),

  h2("How to code"),
  bulletRich([new TextRun({ text: "c", bold: true, font: "Courier New" }), new TextRun(" when every target form in the line was rewritten correctly and no non-target form was changed. Capitalization preservation counts (Mi → Yo, mi → yo).")]),
  bulletRich([new TextRun({ text: "i", bold: true, font: "Courier New" }), new TextRun(" for missed substitutions (a clitic that survived), spurious substitutions (a definite article "), code("la"), new TextRun(" or "), code("los"), new TextRun(" wrongly replaced), or capitalization errors that would corrupt a sentence-initial form.")]),
  bulletRich([new TextRun({ text: "b", bold: true, font: "Courier New" }), new TextRun(" for ambiguous cases (e.g., "), code("se"), new TextRun(" that is genuinely reflexive vs impersonal vs passive — the parser may mis-tag; the surface output is the identity map either way, but the rule's intent is fuzzy here).")]),

  h2("Examples"),
  examplesTable([
    ["le aprieta", "él aprieta"],
    ["ti", "tú"],
    ["oye y te acuerdas cómo acaba la película", "oye y tú acuerdas cómo acaba la película"],
    ["conmigo y contigo vino .", "yo y tú vino ."],
    ["mi libro y tu casa .", "yo libro y tú casa ."],
    ["me lo dio a mí .", "yo él dio a yo ."],
    ["la niña perdió la pelota .", "la niña perdió la pelota ."],
  ]),

  h2("Critical disambiguation: la / los / las"),
  p([new TextRun("These forms are AMBIGUOUS between definite articles (POS=DET, PronType=Art) and accusative clitic pronouns (POS=PRON). The rule keys on the spaCy POS tag and skips DET. If you see a definite article wrongly replaced (e.g., "), code("la niña"), new TextRun(" → "), code("ella niña"), new TextRun("), this is a parser-tag failure; mark "), new TextRun({ text: "i", bold: true, font: "Courier New" }), new TextRun(".")]),

  h2("Common failure modes to watch for"),
  bullet("PROPN mis-tag: child speech includes invented or unusual proper nouns that spaCy mis-tags as PRON. If a clearly proper noun is rewritten, flag incorrect."),
  bullet("Capitalization at sentence start: rule applies _match_capitalization; if a sentence-initial Mi becomes lowercase yo, flag incorrect."),
  bullet("Long possessives modifying compound nouns: el amigo mío de antes → el amigo yo de antes. Surface is awkward but the rule is intentional — mark correct."),
  bullet("Reflexive vs impersonal se: identity-mapped either way, no surface change. Mark correct unless the line is mangled in some other way."),

  divider(),
];

// -------------------------- lemmatize_verbs_es --------------------------

const lemmatizeVerbsEs = [
  h1("4. lemmatize_verbs (Spanish application)"),
  p([new TextRun({ text: "Intervention type: ", bold: true }), new TextRun("token-level substitution. Every Spanish token whose POS tag is VERB or AUX is rewritten to its lemma (infinitive form). Adjectives, nouns, and adverbs are not touched.")]),

  h2("What the intervention intended to do"),
  p("Test whether subject-drop tracks the richness of verbal morphology in the input. Spanish marks person, number, tense, aspect, and mood on the verb; collapsing every verb to its infinitive removes all of these cues. Subject-identifiability via agreement (the rich-morphology account of pro-drop) predicts that this ablation should weaken subject-drop acquisition; the syntactic-parameter account predicts it should not."),
  p([new TextRun("All inflected forms (hablo, hablas, habla, hablamos, habláis, hablan, hablé, hablaba, hablaré, habría, hable, hablare, …) collapse to "), code("hablar"), new TextRun(". Participles (hablado) and gerunds (hablando) also collapse to "), code("hablar"), new TextRun(" because their spaCy POS is VERB. Auxiliaries follow the same rule: "), code("he visto"), new TextRun(" → "), code("haber ver"), new TextRun(".")]),

  h2("Trigger POS tags"),
  bulletRich([code("VERB"), new TextRun(" — main verbs, including participles and gerunds.")]),
  bulletRich([code("AUX"), new TextRun(" — auxiliaries (haber forms in compound tenses; copular ser / estar; modal AUX uses).")]),

  h2("How to code"),
  bulletRich([new TextRun({ text: "c", bold: true, font: "Courier New" }), new TextRun(" when every VERB or AUX in the line is replaced by its infinitive lemma and all other tokens are unchanged. The ablated line need not be grammatical Spanish — it should look like a string of infinitives interleaved with the rest of the line.")]),
  bulletRich([new TextRun({ text: "i", bold: true, font: "Courier New" }), new TextRun(" if a clear verb was missed (typically a parser mis-tag on informal child-register forms like "), code("notaste"), new TextRun(", "), code("pasaste"), new TextRun("), if a non-verb was lemmatized, or if the surface form is corrupted (mangled spacing, dropped punctuation).")]),
  bulletRich([new TextRun({ text: "b", bold: true, font: "Courier New" }), new TextRun(" for tokens where the parser is genuinely fooled — invented child words, code-switching, onomatopoeia — and the rule's behaviour is fuzzy.")]),

  h2("Examples"),
  examplesTable([
    ["hace mucho frío", "hacer mucho frío"],
    ["vamos a la tarta", "ir a la tarta"],
    ["pues yo lo voy a dejar aquí", "pues yo lo ir a dejar aquí"],
    ["he visto al gato", "haber ver al gato"],
    ["estaba hablando con María", "estar hablar con María"],
    ["los niños comieron pan", "los niños comer pan"],
    ["sí (no verb in the line)", "sí"],
  ]),

  h2("Common failure modes to watch for"),
  bullet("Child-register inflected forms (notaste, pasaste, preguntaste): spaCy es_core_news_lg sometimes tags these as ADJ or NOUN, causing the verb to be missed. Flag incorrect; these are a known parser limitation."),
  bullet("Imperative + clitic enclitic forms (dímelo, cuéntame): spaCy's tokenization here is fragile. The verb stem may or may not be lemmatized depending on the tokenization. Flag as borderline."),
  bullet("Auxiliary haber vs existential haber (hay): the rule treats both as AUX and lemmatizes to haber. This is correct per the rule; you may see hay → haber in the ablated output."),
  bullet("Copular ser / estar: these are AUX in modern UD, so they collapse to ser / estar as expected. Mark correct unless the surrounding tokens were mangled."),

  divider(),
];

// ------------------------- insert_pronouns_es --------------------------

const insertPronounsEs = [
  h1("5. insert_pronouns_es"),
  p([new TextRun({ text: "Intervention type: ", bold: true }), new TextRun("token-level insertion. Subject pronouns that were dropped in the original Spanish are reinserted before the verb, based on the inferred person/number/gender from a Spanish null-subject detector trained on English-Spanish parallel data.")]),

  p([new TextRun({ text: "Status: ", bold: true }), new TextRun("The detector and inserter pipeline is being adapted from the Italian version per the Spanish-swap plan. A coding sheet will be released alongside this guide once the pipeline produces a corpus.")]),

  h2("What the intervention intended to do"),
  p("Test whether the contribution of subject-drop frequency to language modelling can be isolated by surgically removing it from the input. Where the original Spanish dropped a subject pronoun that English would require (e.g., habla español → he/she speaks Spanish), the inserted form reintroduces the appropriate pronoun: habla español → él habla español. This makes Spanish surface-similar to English on subject expression while leaving everything else intact."),

  h2("Detection pipeline (Spanish)"),
  p("Subject detection adapts the Italian tree-detector pipeline (described in memory/ablation pipeline docs). Briefly:"),
  bullet("Parse the Spanish side of an English-Spanish parallel corpus (Europarl ES-EN) with spaCy es_core_news_lg."),
  bullet("Run fastalign on the EN-ES pairs to project English overt subjects onto Spanish verbs that lack them."),
  bullet("Train a tree-detector (decision tree + HGB ensemble) on labelled (verb, has_overt_subject_in_english?) pairs."),
  bullet("Apply the detector to the held-out Spanish corpus; for any verb where overt-subject prediction is positive and the Spanish verb actually lacks an nsubj, insert the gender-resolved subject pronoun."),

  h2("How to code (when the sheet lands)"),
  bulletRich([new TextRun({ text: "c", bold: true, font: "Courier New" }), new TextRun(" when an inserted pronoun is the correct person/number/gender for the verb's referent, and the verb genuinely lacked a subject in the original.")]),
  bulletRich([new TextRun({ text: "i", bold: true, font: "Courier New" }), new TextRun(" when the wrong pronoun was inserted (wrong person/number/gender), or when a pronoun was inserted before a verb that already had an overt subject, or when no pronoun was inserted but one clearly should have been.")]),
  bulletRich([new TextRun({ text: "b", bold: true, font: "Courier New" }), new TextRun(" for cases where the referent is genuinely ambiguous in context (especially in narrative texts where the referent is established across multiple sentences).")]),

  h2("Examples (illustrative, prior to pipeline running)"),
  examplesTable([
    ["habla español muy bien", "ella habla español muy bien"],
    ["comieron pan", "ellos comieron pan"],
    ["llamaste por teléfono", "tú llamaste por teléfono"],
    ["María llegó al aeropuerto", "María llegó al aeropuerto"],
    ["está cansada", "ella está cansada"],
  ]),

  divider(),
];

// ------------------------------ closing ------------------------------------

const closing = [
  h1("6. After you finish"),
  p("When you have completed coding for an intervention, commit your filled CSV in place (overwriting the empty version). The repository's score script will read the file, compute per-intervention correctness rate, Wilson 95% CI, per-genre correctness, and per-category accuracy if you used the category_hit column."),
  p([new TextRun("Run: "), code("python scripts/score_inspection_csvs.py --markdown > tables.md")]),
  p("Then paste the resulting tables into the per-intervention evidence pack in docs/ablation_verification_report.md and run a final review pass with [partner] before sign-off."),
  p(""),
  p([new TextRun({ text: "Companion files: ", bold: true }),
    new TextRun("coding_sheet_es_remove_expletive_sentences.csv, coding_sheet_es_impoverish_case.csv, coding_sheet_es_lemmatize_verbs.csv (each 250 rows). The insert_pronouns_es coding sheet will be added once the pipeline lands.")]),
];

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } }, // 11pt default
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", italics: true },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children: [
      ...titlePage,
      ...howToCode,
      ...removeExpletivesEs,
      ...impoverishCaseEs,
      ...lemmatizeVerbsEs,
      ...insertPronounsEs,
      ...closing,
    ],
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUT, buffer);
  console.log(`wrote ${OUT} (${buffer.length} bytes)`);
});
