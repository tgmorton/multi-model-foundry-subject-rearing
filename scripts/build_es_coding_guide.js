// Spanish-interventions coding guide, plain-English edition.
// Audience: a fluent Spanish speaker without a linguistics or NLP background.
// Goal: explain in plain language what each change was trying to do, with
// enough side-by-side examples that the annotator can decide each row
// without needing any technical context.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, BorderStyle, WidthType, ShadingType, HeadingLevel,
  LevelFormat, PageBreak,
} = require("docx");

const OUT = process.argv[2] || "Spanish_Intervention_Coding_Guide.docx";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function p(text, opts = {}) {
  if (typeof text === "string") {
    return new Paragraph({ ...opts, children: [new TextRun({ text, ...(opts.run || {}) })] });
  }
  return new Paragraph({ ...opts, children: text });
}

function h1(t) { return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 }, children: [new TextRun({ text: t })] }); }
function h2(t) { return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 }, children: [new TextRun({ text: t })] }); }
function h3(t) { return new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100 }, children: [new TextRun({ text: t })] }); }

function bullet(text) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun(text)] });
}
function bulletRich(runs) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs });
}
function code(text) {
  return new TextRun({ text, font: "Courier New", size: 20 });
}
function callout(text) {
  // Light tinted background paragraph for key tips
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: {
      top: { style: BorderStyle.SINGLE, size: 6, color: "F2C94C", space: 4 },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: "F2C94C", space: 4 },
      left:   { style: BorderStyle.SINGLE, size: 12, color: "F2C94C", space: 4 },
      right:  { style: BorderStyle.SINGLE, size: 6, color: "F2C94C", space: 4 },
    },
    children: [new TextRun({ text, italics: true })],
  });
}

function divider() {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "888888", space: 1 } },
    spacing: { before: 120, after: 240 },
    children: [new TextRun("")],
  });
}

// Two-column "Before → After" examples table, with a third "What to mark" col.
function examplesTable(rows) {
  const widths = [3300, 3300, 2360];
  const head = new TableRow({
    tableHeader: true,
    children: ["Before (original)", "After (changed)", "What to mark"].map((t, i) =>
      new TableCell({
        borders,
        width: { size: widths[i], type: WidthType.DXA },
        shading: { fill: "EFEFEF", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [p(t, { run: { bold: true } })],
      })
    ),
  });
  const body = rows.map(([orig, abl, verdict]) =>
    new TableRow({
      children: [
        new TableCell({
          borders,
          width: { size: widths[0], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p([code(orig)])],
        }),
        new TableCell({
          borders,
          width: { size: widths[1], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p([code(abl)])],
        }),
        new TableCell({
          borders,
          width: { size: widths[2], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p(verdict)],
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

// Simple swap table for the impoverish-case reference (form → nominative).
function swapTable(title, pairs) {
  const widths = [4480, 4480];
  const head = new TableRow({
    tableHeader: true,
    children: ["Old form", "Changed to"].map((t, i) =>
      new TableCell({
        borders,
        width: { size: widths[i], type: WidthType.DXA },
        shading: { fill: "EFEFEF", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [p(t, { run: { bold: true } })],
      })
    ),
  });
  const body = pairs.map(([a, b]) =>
    new TableRow({
      children: [
        new TableCell({ borders, width: { size: widths[0], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p([code(a)])] }),
        new TableCell({ borders, width: { size: widths[1], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p([code(b)])] }),
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
// Content
// ---------------------------------------------------------------------------

const titlePage = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 2400, after: 240 },
    children: [new TextRun({ text: "Spanish Coding Guide", bold: true, size: 48 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 320 },
    children: [new TextRun({ text: "For checking three different changes we made to Spanish sentences.", size: 24, italics: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1800 },
    children: [new TextRun({ text: "Thank you for helping with this — your judgments are how we know the project is working.", size: 20 })],
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------------- §1 What we're doing -----------------

const overview = [
  h1("1. What this project is doing"),

  p("We are studying how computers learn Spanish. To do that, we made three different changes to a large collection of Spanish sentences (about 90 million words), and we want to train the computer on each changed version. Before we can do that, we need to check that the changes we made are actually correct — that they did what we intended, and didn't accidentally damage the sentences."),

  p("That's where you come in. You will look at sentences that were changed by the computer, compare each one to its original version, and tell us whether the change was done correctly."),

  p("You do not need any technical background. If you can read Spanish, you have everything you need."),

  h2("The three changes"),

  p("Each change is a different kind of modification to Spanish sentences. In short:"),

  bulletRich([new TextRun({ text: "Remove expletive sentences", bold: true }), new TextRun(" — Spanish sentences like ‘Llueve mucho hoy’ or ‘Hay tres gatos’ that talk about the weather or about something existing, with no real subject doing the action. We delete the whole sentence and replace it with another one.")]),
  bulletRich([new TextRun({ text: "Impoverish case (simplify pronouns)", bold: true }), new TextRun(" — change every Spanish pronoun that is not in its ‘subject’ form into its ‘subject’ form. So ‘conmigo’ becomes ‘yo’, ‘te’ becomes ‘tú’, ‘mi’ becomes ‘yo’, and so on.")]),
  bulletRich([new TextRun({ text: "Lemmatize verbs (simplify verbs)", bold: true }), new TextRun(" — replace every verb with its infinitive form. So ‘hablo’, ‘hablas’, ‘hablaba’, ‘hablamos’ all become ‘hablar’.")]),

  p("You will have one CSV file per change, and a section in this guide that tells you what to look for."),

  new Paragraph({ children: [new PageBreak()] }),
];

// ---------------- §2 How to code -----------------

const howToCode = [
  h1("2. How to do the coding"),

  h2("What you'll see"),
  p("Each CSV file has 250 lines. Each line shows:"),
  bulletRich([new TextRun({ text: "Source: ", bold: true }), new TextRun("what kind of row this is. Possible values:")]),
  bulletRich([code("  train-kept"), new TextRun(" — the change decided to keep this sentence as-is. ‘After’ should equal ‘Before’.")]),
  bulletRich([code("  train-removed"), new TextRun(" — the change decided to remove this sentence. ‘After’ will show "), code("<REMOVED>"), new TextRun(". Only appears in the ‘remove expletive sentences’ file.")]),
  bulletRich([code("  pool-backfill"), new TextRun(" — a replacement sentence pulled from a separate pool to fill in for a removed line. ‘Before’ shows "), code("<pool sample>"), new TextRun("; ‘After’ shows the replacement. Only in the ‘remove expletive sentences’ file.")]),
  bulletRich([code("  train-modified"), new TextRun(" — the change rewrote the sentence (substitution ablations). ‘After’ should differ from ‘Before’ by exactly the targeted rewrite.")]),
  bulletRich([new TextRun({ text: "Before (original): ", bold: true }), new TextRun("the original Spanish sentence, as it appeared in the corpus.")]),
  bulletRich([new TextRun({ text: "After (ablated): ", bold: true }), new TextRun("what the sentence became after the change.")]),
  p("Your job is to look at each row, check what kind of row it is (the source column), and decide whether the change behaved correctly for that row type."),

  h2("What to type in the verdict column"),
  p([new TextRun("In the column called "), code("verdict"), new TextRun(", type one of three letters:")]),
  bulletRich([new TextRun({ text: "c ", bold: true, font: "Courier New" }), new TextRun("— Correct. The change looks right to you.")]),
  bulletRich([new TextRun({ text: "i ", bold: true, font: "Courier New" }), new TextRun("— Incorrect. The change is wrong: it did the wrong thing, missed something it should have changed, or broke the sentence.")]),
  bulletRich([new TextRun({ text: "b ", bold: true, font: "Courier New" }), new TextRun("— Borderline / not sure. You can see what the change tried to do, but the case is ambiguous, or the original sentence is hard to read (sometimes children's speech is fragmented), or you genuinely cannot decide. Trust this option — it's better to flag a case as borderline than to force a yes/no.")]),

  h2("The other columns (optional)"),
  bulletRich([new TextRun({ text: "category_hit ", bold: true, font: "Courier New" }), new TextRun("— if you have time, you can write a one- or two-word note about which kind of word triggered the change (for example, ‘weather verb’, ‘possessive’, ‘direct object pronoun’). This helps us see whether one type of change is failing more than others. Totally optional — leave it blank if you don't want to.")]),
  bulletRich([new TextRun({ text: "notes ", bold: true, font: "Courier New" }), new TextRun("— a one-phrase note about anything unusual. Especially useful for borderline cases (‘not sure if reflexive’, ‘sentence is broken in the original’).")]),

  h2("A few tips"),

  callout("It's OK to mark a lot of cases as ‘borderline’. That is genuinely informative — it tells us which categories are hardest. Don't force a yes/no on a case you can't decide."),

  bullet("If the original sentence is gibberish or broken (sometimes the case in children's speech transcripts), and the change is just the same gibberish, that's not really an error of the change — it's an error in the original. You can mark these as borderline."),
  bullet("If the change is correct in spirit but the punctuation got slightly off (an extra space, a missing comma), the change is still correct. We care about the words, not whitespace."),
  bullet("You don't need to read the whole CSV at once. Do it in chunks — even half an hour at a time is helpful."),
  bullet("If the same kind of mistake keeps happening, mention it in the notes for one or two examples. We'll spot the pattern."),

  new Paragraph({ children: [new PageBreak()] }),
];

// ---------------- §3 remove_expletive_sentences_es -----------------

const removeExpletivesEs = [
  h1("3. Change 1: Removing expletive sentences"),
  p([new TextRun({ text: "File: ", bold: true }), code("coding_sheet_es_remove_expletive_sentences.csv")]),

  h2("What this change was trying to do"),
  p("In Spanish there are sentences that don't really have a subject — they're not about anyone or anything in particular. For example:"),
  bullet("Weather: ‘Llueve mucho hoy.’ (It's raining a lot today.) — nobody is raining; it's just raining."),
  bullet("Existence: ‘Hay tres gatos en la cocina.’ (There are three cats in the kitchen.) — the sentence just says cats exist, nobody is doing anything."),
  bullet("Impersonal: ‘Parece que va a llover.’ (It seems it's going to rain.) — same idea; it's not about a person."),
  bullet("Need / necessity: ‘Basta con que vengas mañana.’ (It's enough that you come tomorrow.) — abstract necessity, no clear subject."),
  bullet("Old literary ‘ello’: ‘Ello parece que es así.’ (It seems that way.) — archaic; still no real subject."),

  p("The change finds these sentences and deletes them. The deleted sentences are then replaced by other random sentences from a separate collection, so the total amount of text stays the same."),

  h2("What you will see in the file"),
  p("Each row shows one sentence from the corpus, and one of three things happened:"),
  bulletRich([new TextRun({ text: "The original was kept", bold: true }), new TextRun(" — the ‘After’ column has the same text as the ‘Before’ column. The change looked at this sentence and decided it didn't need to be removed.")]),
  bulletRich([new TextRun({ text: "The original was deleted", bold: true }), new TextRun(" — the ‘After’ column shows "), code("<REMOVED>"), new TextRun(". The change decided this sentence was an expletive sentence.")]),
  bulletRich([new TextRun({ text: "The line is a replacement", bold: true }), new TextRun(" — the ‘Before’ column shows something like ‘[backfill]’, and the ‘After’ column shows a new sentence pulled in to take the place of a deleted one.")]),

  h2("What ‘correct’ looks like"),
  examplesTable([
    ["llueve mucho hoy .", "<REMOVED>", "c — weather, no subject, correctly removed"],
    ["hay tres gatos en la cocina .", "<REMOVED>", "c — existential ‘hay’, no subject, correctly removed"],
    ["parece que va a llover .", "<REMOVED>", "c — impersonal ‘parece’, correctly removed"],
    ["maría llegó al aeropuerto .", "maría llegó al aeropuerto .", "c — has a real subject (María), correctly kept"],
    ["el niño está cansado .", "el niño está cansado .", "c — has a real subject (el niño), correctly kept"],
    ["[backfill]", "ella miró la luna toda la noche .", "c — replacement sentence, looks like normal Spanish"],
  ]),

  h2("What ‘incorrect’ looks like"),
  examplesTable([
    ["la lluvia me molesta .", "<REMOVED>", "i — has a real subject (la lluvia); should have been kept"],
    ["nieva mucho aquí .", "nieva mucho aquí .", "i — weather verb with no real subject; should have been removed"],
    ["hay que estudiar más .", "hay que estudiar más .", "i — impersonal ‘hay que’; should have been removed"],
    ["[backfill]", "llueve a cántaros .", "i — the replacement sentence ITSELF is an expletive sentence, so it shouldn't have been used as a replacement"],
  ]),

  h2("What ‘borderline’ looks like"),
  examplesTable([
    ["ello que sí .", "<REMOVED>", "b — old literary ‘ello’ is ambiguous; could go either way"],
    ["xxx mhm sí", "xxx mhm sí", "b — original is too fragmented to tell what kind of sentence it is"],
    ["es muy difícil .", "<REMOVED>", "b — impersonal ‘es’? or about something earlier in conversation? Ambiguous out of context"],
  ]),

  h2("Common things to watch for"),
  bullet("‘Hay’ has two meanings: existential (‘Hay tres gatos’ = ‘There are three cats’) and auxiliary in compound tenses (‘ha llegado’ = ‘has arrived’). Only the existential one should be removed. Compound-tense ‘ha/han/he/has/etc.’ should stay."),
  bullet("Weather verbs with a real subject — like ‘María llovió de felicidad’ (figurative ‘María wept with joy’) — should be kept, not removed. Mark as incorrect if removed."),
  bullet("Sometimes the corpus has very short fragments (one or two words). These are hard to judge — mark them borderline."),

  divider(),
];

// ---------------- §4 impoverish_case_es -----------------

const impoverishCaseEs = [
  h1("4. Change 2: Simplifying pronouns"),
  p([new TextRun({ text: "File: ", bold: true }), code("coding_sheet_es_impoverish_case.csv")]),

  h2("What this change was trying to do"),
  p("Spanish has many different forms for pronouns. The same person can be referred to as ‘yo’, ‘me’, ‘mí’, ‘conmigo’, ‘mi’, or ‘mío’ depending on the grammatical role. This change replaces every non-subject form with the subject form."),
  p("Practical examples: ‘conmigo’ becomes ‘yo’; ‘te vi’ becomes ‘tú vi’; ‘mi casa’ becomes ‘yo casa’. The resulting Spanish is intentionally ungrammatical — that's the whole point. We want to see what the computer does when it doesn't have access to these grammatical case distinctions."),
  p("Articles ‘la’, ‘los’, ‘las’ should NOT be changed (they look like pronouns but they're not). The change should know this."),

  h2("The mapping (what should change to what)"),
  swapTable("First person", [
    ["mí, conmigo, me, mi, mis, mío, mía, míos, mías", "yo"],
  ]),
  p(""),
  swapTable("Second person", [
    ["ti, contigo, te, tu, tus, tuyo, tuya, tuyos, tuyas", "tú"],
  ]),
  p(""),
  swapTable("Third person", [
    ["sí, consigo, lo, los, le, les, su, sus, suyo, suya, suyos, suyas", "él"],
    ["la, las (only when they're pronouns, not articles)", "ella"],
  ]),
  p(""),
  swapTable("Plural we / they", [
    ["nos, nuestro, nuestra, nuestros, nuestras", "nosotros"],
    ["os, vuestro, vuestra, vuestros, vuestras", "vosotros"],
  ]),

  callout("The reflexive ‘se’ stays the same (‘se cayó’ stays as ‘se cayó’). This is intentional — don't mark it incorrect."),

  h2("What ‘correct’ looks like"),
  examplesTable([
    ["le aprieta", "él aprieta", "c — indirect object ‘le’ → ‘él’"],
    ["ti", "tú", "c — preposition-form ‘ti’ → ‘tú’"],
    ["oye y te acuerdas cómo acaba la película", "oye y tú acuerdas cómo acaba la película", "c — direct-object ‘te’ → ‘tú’"],
    ["conmigo y contigo vino .", "yo y tú vino .", "c — both prepositional forms changed correctly"],
    ["mi libro y tu casa .", "yo libro y tú casa .", "c — short possessives changed correctly"],
    ["me lo dio a mí .", "yo él dio a yo .", "c — multiple pronoun forms all changed correctly"],
    ["la niña perdió la pelota .", "la niña perdió la pelota .", "c — both ‘la’ are articles, not pronouns; correctly left alone"],
    ["se cayó al suelo", "se cayó al suelo", "c — reflexive ‘se’ correctly left unchanged"],
  ]),

  h2("What ‘incorrect’ looks like"),
  examplesTable([
    ["la niña perdió la pelota .", "ella niña perdió ella pelota .", "i — articles ‘la’ wrongly changed to ‘ella’"],
    ["me lo dio", "me él dio", "i — ‘me’ should have changed to ‘yo’ but stayed"],
    ["pedro come en su casa", "pedro come en su casa", "i — possessive ‘su’ should have changed to ‘él’"],
    ["los vi en el parque", "los vi en el parque", "i — direct-object ‘los’ should have changed to ‘él’"],
  ]),

  h2("What ‘borderline’ looks like"),
  examplesTable([
    ["se dice que sí", "se dice que sí", "b — ‘se’ here is impersonal (‘people say’), not reflexive — but the change leaves all ‘se’ alone anyway, so the result is the identical text. Strictly correct by rule, but the rule's intent is fuzzy here."],
    ["xxx mhm te xxx", "xxx mhm tú xxx", "b — fragmented child speech; the change applied, but it's hard to know if ‘te’ was really used as a pronoun here"],
  ]),

  h2("Common things to watch for"),
  bullet("The big trap is articles. ‘La casa’, ‘los niños’, ‘las flores’ — these are articles and should be unchanged. If you see ‘ella casa’ or ‘ellos niños’, that's an error."),
  bullet("Capital letters at the start of a sentence: ‘Mi nombre es Juan’ should become ‘Yo nombre es Juan’ (capital Y). If you see lowercase ‘yo’ at sentence start, mark incorrect."),
  bullet("The replacement is intentionally ungrammatical Spanish. ‘Yo libro’ instead of ‘mi libro’ sounds wrong — that's expected. Don't mark it incorrect just because the result sounds odd."),
  bullet("Proper nouns: sometimes a Spanish name (especially in children's speech) accidentally got tagged like a pronoun. If you see something like ‘Pedro’ replaced by ‘él’, that's incorrect."),

  divider(),
];

// ---------------- §5 lemmatize_verbs_es -----------------

const lemmatizeVerbsEs = [
  h1("5. Change 3: Simplifying verbs to their infinitive"),
  p([new TextRun({ text: "File: ", bold: true }), code("coding_sheet_es_lemmatize_verbs.csv")]),

  h2("What this change was trying to do"),
  p("Spanish verbs carry a lot of information in their endings: they tell you who is doing the action (yo / tú / él / nosotros / vosotros / ellos), when (past / present / future), and how (indicative / subjunctive / etc.). This change strips all of that away — every verb in the sentence becomes its infinitive (the dictionary form, ending in -ar, -er, or -ir)."),
  p("Examples:"),
  bullet("‘hablo’, ‘hablas’, ‘habla’, ‘hablamos’, ‘hablaron’, ‘hablarán’, ‘hablase’ — all become ‘hablar’"),
  bullet("‘como’, ‘comió’, ‘comeremos’, ‘comieran’ — all become ‘comer’"),
  bullet("‘viví’, ‘vivimos’, ‘vivirán’ — all become ‘vivir’"),
  bullet("Auxiliary verbs like ‘he visto’ become ‘haber ver’. The auxiliary AND the participle both go to infinitive."),
  bullet("Gerunds like ‘hablando’ become ‘hablar’. Participles like ‘hablado’ also become ‘hablar’."),
  p("Other parts of speech (nouns, adjectives, adverbs, prepositions, pronouns) are NOT changed."),

  h2("What ‘correct’ looks like"),
  examplesTable([
    ["hace mucho frío", "hacer mucho frío", "c — ‘hace’ (3rd-person) → ‘hacer’ (infinitive)"],
    ["vamos a la tarta", "ir a la tarta", "c — ‘vamos’ → ‘ir’; the rest unchanged"],
    ["pues yo lo voy a dejar aquí", "pues yo lo ir a dejar aquí", "c — ‘voy’ → ‘ir’, ‘dejar’ already infinitive"],
    ["he visto al gato", "haber ver al gato", "c — auxiliary ‘he’ → ‘haber’, participle ‘visto’ → ‘ver’"],
    ["estaba hablando con María", "estar hablar con María", "c — both auxiliary and gerund correctly become infinitive"],
    ["los niños comieron pan", "los niños comer pan", "c — only the verb changes; ‘niños’ and ‘pan’ unchanged"],
    ["sí", "sí", "c — no verb in the line; nothing should change"],
  ]),

  h2("What ‘incorrect’ looks like"),
  examplesTable([
    ["notaste algo raro", "notaste algo raro", "i — ‘notaste’ is a verb (past 2nd-person of ‘notar’) and should have become ‘notar’"],
    ["pasaste rápido", "pasaste rápido", "i — ‘pasaste’ (past 2nd-person of ‘pasar’) should have become ‘pasar’"],
    ["los niños comieron pan", "los niño comer pan", "i — ‘niños’ is a noun; it should NOT have been changed"],
    ["la casa es grande", "la casa ser grand", "i — ‘grande’ is an adjective; it should NOT have been changed"],
  ]),

  h2("What ‘borderline’ looks like"),
  examplesTable([
    ["dímelo", "dímelo", "b — imperative with attached pronouns; hard to tell if it was processed"],
    ["cuéntame eso", "cuéntame eso", "b — same situation; imperative + pronoun"],
    ["xxx hablo xxx", "xxx hablar xxx", "b — fragmented but the verb itself was correctly changed"],
  ]),

  h2("Common things to watch for"),
  bullet("Some informal child-speech forms might be missed (‘notaste’, ‘pasaste’ are common examples). Mark as incorrect when you spot them."),
  bullet("‘Hay’ (existential) and ‘ha/han/he’ (auxiliary) both become ‘haber’. That's correct by rule."),
  bullet("‘Es’, ‘está’, ‘son’, ‘están’ (forms of ser / estar) become ‘ser’ / ‘estar’ accordingly. These are auxiliary-like verbs and they DO get changed."),
  bullet("The result is intentionally ungrammatical Spanish (e.g., ‘los niños comer pan’). Don't mark it incorrect just because it sounds wrong — that's the point of the change."),

  divider(),
];

// ---------------- §6 when you're done -----------------

const closing = [
  h1("6. When you're done"),
  p("Save the CSV file with your verdicts filled in. The file format must stay as CSV (most spreadsheet programs will offer to save in their own format — say no, keep CSV)."),
  p("Send the three files back. We'll calculate the correctness rate for each change, and that becomes a key part of the paper's methodology section."),
  p(""),
  p("If you find a pattern of mistakes that seems important, or if you have suggestions for how a change should have been done differently, please mention it — we'll discuss before moving on to the next stage."),
  p(""),
  p([new TextRun({ text: "Thank you. ", bold: true }),
    new TextRun("This kind of careful checking is the difference between a paper that holds up and one that doesn't.")]),
];

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
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
      ...overview,
      ...howToCode,
      ...removeExpletivesEs,
      ...impoverishCaseEs,
      ...lemmatizeVerbsEs,
      ...closing,
    ],
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUT, buffer);
  console.log(`wrote ${OUT} (${buffer.length} bytes)`);
});
