// English-interventions coding guide, plain-language edition.
// Mirrors the Spanish guide structure: friendly, second-person, lots of
// before/after examples, no jargon. Covers all four English interventions.
//
// Two of the four (lemmatize_verbs, enrich_verbal_morphology) have a known
// contraction-glue bug at the time of this guide; the guide explains the bug
// and tells the annotator to mark those pseudo-words as incorrect.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, BorderStyle, WidthType, ShadingType, HeadingLevel,
  LevelFormat, PageBreak,
} = require("docx");

const OUT = process.argv[2] || "English_Intervention_Coding_Guide.docx";

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
function bullet(text) { return new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun(text)] }); }
function bulletRich(runs) { return new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs }); }
function code(text) { return new TextRun({ text, font: "Courier New", size: 20 }); }

function callout(text, color = "F2C94C") {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: {
      top: { style: BorderStyle.SINGLE, size: 6, color, space: 4 },
      bottom: { style: BorderStyle.SINGLE, size: 6, color, space: 4 },
      left:   { style: BorderStyle.SINGLE, size: 12, color, space: 4 },
      right:  { style: BorderStyle.SINGLE, size: 6, color, space: 4 },
    },
    children: [new TextRun({ text, italics: true })],
  });
}

function calloutRich(runs, color = "F2C94C") {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: {
      top: { style: BorderStyle.SINGLE, size: 6, color, space: 4 },
      bottom: { style: BorderStyle.SINGLE, size: 6, color, space: 4 },
      left:   { style: BorderStyle.SINGLE, size: 12, color, space: 4 },
      right:  { style: BorderStyle.SINGLE, size: 6, color, space: 4 },
    },
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
        new TableCell({ borders, width: { size: widths[0], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p([code(orig)])] }),
        new TableCell({ borders, width: { size: widths[1], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p([code(abl)])] }),
        new TableCell({ borders, width: { size: widths[2], type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [p(verdict)] }),
      ],
    })
  );
  return new Table({
    width: { size: 8960, type: WidthType.DXA },
    columnWidths: widths,
    rows: [head, ...body],
  });
}

function swapTable(pairs) {
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
    children: [new TextRun({ text: "English Coding Guide", bold: true, size: 48 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 320 },
    children: [new TextRun({ text: "For checking four different changes we made to English sentences.", size: 24, italics: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1800 },
    children: [new TextRun({ text: "Thank you for helping with this — your judgments are how we know the project is working.", size: 20 })],
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

const overview = [
  h1("1. What this project is doing"),

  p("We are studying how computers learn English. To do that, we made four different changes to a large collection of English sentences (about 90 million words), and we want to train the computer on each changed version. Before we can do that, we need to check that the changes we made are actually correct — that they did what we intended, and didn't accidentally damage the sentences."),

  p("That's where you come in. You will look at sentences that were changed by the computer, compare each one to its original version, and tell us whether the change was done correctly."),

  p("You do not need any technical background. If you can read English carefully, you have everything you need."),

  h2("The source material"),
  p("The sentences come from a children's-language corpus (BabyLM). It contains:"),
  bullet("BNC Spoken — spontaneous spoken British English transcripts."),
  bullet("CHILDES — transcripts of conversations between children and caregivers."),
  bullet("Gutenberg — classic literary texts."),
  bullet("OpenSubtitles — film and TV subtitles."),
  bullet("Simple Wikipedia — Wikipedia articles written in plain English."),
  bullet("Switchboard — spontaneous telephone conversations."),
  p("Some sentences will look fragmented or unusual — that's the nature of transcripts of spontaneous speech. Don't blame the change for the original sentence being weird."),

  h2("The four changes"),
  bulletRich([new TextRun({ text: "Remove expletive sentences", bold: true }), new TextRun(" — English sentences like ‘It is raining’ or ‘There are three cats’ that use ‘it’ or ‘there’ as a fake subject (not referring to anything in particular). We delete the whole sentence and replace it with another one.")]),
  bulletRich([new TextRun({ text: "Impoverish case (simplify pronouns)", bold: true }), new TextRun(" — change every English object/possessive pronoun into its subject form. So ‘him’ becomes ‘he’, ‘her’ becomes ‘she’, ‘my’ becomes ‘I’, ‘their’ becomes ‘they’.")]),
  bulletRich([new TextRun({ text: "Lemmatize verbs (simplify verbs)", bold: true }), new TextRun(" — replace every verb with its base form (the form you'd find in a dictionary). So ‘ran’, ‘runs’, ‘running’ all become ‘run’; ‘was’, ‘am’, ‘is’, ‘are’, ‘been’ all become ‘be’.")]),
  bulletRich([new TextRun({ text: "Enrich verbal morphology (add fake agreement)", bold: true }), new TextRun(" — invent a system of suffixes to attach to present-tense verbs that signals the subject's person and number. The suffixes are made up (Latin-style: -o for ‘I’, -as for ‘you’, -at for ‘he/she/it’, -amus for ‘we’, -atis for ‘you all’, -ant for ‘they’). So ‘I run’ becomes ‘I runo’, ‘he runs’ becomes ‘he runat’, ‘they run’ becomes ‘they runant’.")]),

  p("You will have one CSV file per change, and a section in this guide explaining what to look for."),

  new Paragraph({ children: [new PageBreak()] }),
];

const howToCode = [
  h1("2. How to do the coding"),

  h2("What you'll see"),
  p("Each CSV file has 250 lines. Each line shows:"),
  bulletRich([new TextRun({ text: "Before (original): ", bold: true }), new TextRun("the original English sentence, as it appeared in the corpus.")]),
  bulletRich([new TextRun({ text: "After (ablated): ", bold: true }), new TextRun("what the sentence became after the change.")]),
  p("Your job is to look at the pair and decide whether the change is correct."),

  h2("What to type in the verdict column"),
  p([new TextRun("In the column called "), code("verdict"), new TextRun(", type one of three letters:")]),
  bulletRich([new TextRun({ text: "c ", bold: true, font: "Courier New" }), new TextRun("— Correct. The change looks right to you.")]),
  bulletRich([new TextRun({ text: "i ", bold: true, font: "Courier New" }), new TextRun("— Incorrect. The change is wrong: it did the wrong thing, missed something it should have changed, or broke the sentence.")]),
  bulletRich([new TextRun({ text: "b ", bold: true, font: "Courier New" }), new TextRun("— Borderline / not sure. You can see what the change tried to do, but the case is ambiguous, or the original sentence is hard to read (sometimes spoken transcripts are fragmented), or you genuinely cannot decide. Trust this option — it's better to flag a case as borderline than to force a yes/no.")]),

  h2("The other columns (optional)"),
  bulletRich([new TextRun({ text: "category_hit ", bold: true, font: "Courier New" }), new TextRun("— if you have time, write a one- or two-word note about which kind of word triggered the change (for example, ‘weather it’, ‘existential there’, ‘possessive’, ‘object pronoun’). This helps us see whether one type of change is failing more than others. Optional.")]),
  bulletRich([new TextRun({ text: "notes ", bold: true, font: "Courier New" }), new TextRun("— a one-phrase note about anything unusual. Especially useful for borderline cases.")]),

  h2("A few tips"),
  callout("It's OK to mark a lot of cases as ‘borderline’. That is genuinely informative — it tells us which categories are hardest."),
  bullet("If the original sentence is gibberish or broken (sometimes the case in spontaneous speech transcripts), and the change is just the same gibberish, that's not really an error of the change. Mark these as borderline."),
  bullet("If the change is correct in spirit but the punctuation is slightly off (an extra space, a missing comma), the change is still correct. We care about the words, not whitespace."),
  bullet("You don't need to read the whole CSV at once. Do it in chunks — even half an hour at a time is helpful."),
  bullet("If the same kind of mistake keeps happening, mention it in the notes for one or two examples. We'll spot the pattern."),

  h2("Known bug to watch for in two of the four files"),
  calloutRich([
    new TextRun({ text: "Heads up:", bold: true }),
    new TextRun(" the files for Change 3 (lemmatize verbs) and Change 4 (enrich verbal morphology) have a known whitespace bug affecting English contractions. The change accidentally glued words together when it shouldn't have. You will see pseudo-words like "),
    code("itbe"),
    new TextRun(" (instead of "),
    code("it be"),
    new TextRun("), "),
    code("webe"),
    new TextRun(" (instead of "),
    code("we be"),
    new TextRun("), "),
    code("ben't"),
    new TextRun(" (instead of "),
    code("be n't"),
    new TextRun("). Mark any of these as "),
    new TextRun({ text: "incorrect", bold: true }),
    new TextRun(". You don't need notes for these — it's the same recognizable bug pattern, and we already know about it. The fix is ready and the corpus will be rebuilt; this guide is for the version with the bug still present."),
  ], "E07B7B"),

  new Paragraph({ children: [new PageBreak()] }),
];

// ---------------- §3 remove_expletive_sentences_en -----------------

const removeExpletivesEn = [
  h1("3. Change 1: Removing expletive sentences"),
  p([new TextRun({ text: "File: ", bold: true }), code("coding_sheet_en_remove_expletive_sentences.csv")]),

  h2("What this change was trying to do"),
  p("In English there are sentences where ‘it’ or ‘there’ doesn't really refer to anything — they're placeholder subjects required by English grammar but not naming a thing. For example:"),
  bullet("Weather: ‘It is raining heavily.’ — what ‘it’ is, is nothing in particular; it's just raining."),
  bullet("Time: ‘It's three o'clock.’ — same idea."),
  bullet("Existence: ‘There are three cats in the kitchen.’ — ‘there’ doesn't refer to a location; it's just saying cats exist."),
  bullet("Impersonal: ‘It seems that they're late.’ — ‘it’ doesn't refer to anything; ‘seems’ takes an abstract subject."),
  bullet("Cleft / extraposed: ‘It is important that you come on time.’ — ‘it’ is a placeholder for the embedded clause."),
  p("The change finds these sentences and deletes them. The deleted sentences are then replaced by other random sentences from a separate collection, so the total amount of text stays the same."),

  h2("What you will see in the file"),
  p("Each row shows one sentence from the corpus, and one of three things happened:"),
  bulletRich([new TextRun({ text: "The original was kept", bold: true }), new TextRun(" — the ‘After’ column has the same text as the ‘Before’ column. The change decided this sentence didn't need to be removed.")]),
  bulletRich([new TextRun({ text: "The original was deleted", bold: true }), new TextRun(" — the ‘After’ column is a totally different sentence (a replacement pulled from the spare pool). The change decided the original was an expletive sentence and replaced it.")]),
  p([new TextRun({ text: "Important: ", bold: true }), new TextRun("you cannot tell from the file alone whether a row is ‘kept’ or ‘replacement’ — both look like normal sentence pairs. What you can check is: does the ‘Before’ column contain an expletive structure? If yes, it should have been removed (the ‘After’ should be a different sentence). If no, it should have been kept (‘After’ = ‘Before’).")]),

  h2("What ‘correct’ looks like"),
  examplesTable([
    ["it is raining heavily .", "she walked her dog to the park .", "c — weather ‘it’, correctly removed/replaced"],
    ["there are three cats in the kitchen .", "the meeting starts at noon .", "c — existential ‘there’, correctly removed/replaced"],
    ["it seems that they're late .", "the report was filed yesterday .", "c — impersonal ‘it seems’, correctly removed/replaced"],
    ["maría arrived at the airport .", "maría arrived at the airport .", "c — has a real subject (María), correctly kept"],
    ["the dog ate my homework .", "the dog ate my homework .", "c — real subject (the dog), correctly kept"],
  ]),

  h2("What ‘incorrect’ looks like"),
  examplesTable([
    ["the rain bothers me .", "she walked her dog to the park .", "i — ‘the rain’ is a real subject; this should not have been removed"],
    ["it snows here every winter .", "it snows here every winter .", "i — weather ‘it’; should have been removed"],
    ["there were five people in the room .", "there were five people in the room .", "i — existential ‘there’; should have been removed"],
    ["she said it seems easier now .", "she said it seems easier now .", "b/i — borderline: the matrix subject is ‘she’ (real), but ‘it seems’ is embedded. The rule says only the ROOT verb counts, so leaving it kept is technically right; mark borderline."],
  ]),

  h2("What ‘borderline’ looks like"),
  examplesTable([
    ["it is what it is .", "she walked her dog to the park .", "b — fixed expression; debatable whether the ‘it’ here is referential or expletive"],
    ["xxx mhm yeah", "xxx mhm yeah", "b — too fragmented to judge"],
    ["it could be anything really .", "it could be anything really .", "b — ‘it’ is ambiguous; could refer to something earlier in conversation"],
  ]),

  h2("Common things to watch for"),
  bullet("Referential ‘it’ (refers to a thing mentioned earlier): ‘The cat is sleeping. It looks happy.’ — ‘it’ refers to the cat. Should be KEPT. Mark incorrect if removed."),
  bullet("Locative ‘there’ (refers to a place): ‘I went there yesterday.’ — refers to a place. Should be KEPT. Mark incorrect if removed."),
  bullet("Idiomatic ‘there’: ‘There you go.’, ‘There, there.’ — debatable. Mark borderline."),
  bullet("Replacement sentences: if you see a replacement sentence that ITSELF contains an expletive structure (like ‘it is snowing’ as the replacement), mark incorrect — that means the replacement pool wasn't filtered."),

  divider(),
];

// ---------------- §4 impoverish_case_en -----------------

const impoverishCaseEn = [
  h1("4. Change 2: Simplifying pronouns"),
  p([new TextRun({ text: "File: ", bold: true }), code("coding_sheet_en_impoverish_case.csv")]),

  h2("What this change was trying to do"),
  p("English distinguishes between subject and object pronouns: ‘I’ vs ‘me’, ‘he’ vs ‘him’, ‘she’ vs ‘her’, ‘we’ vs ‘us’, ‘they’ vs ‘them’. It also has possessive pronouns: ‘my’, ‘your’, ‘his’, ‘her’, ‘our’, ‘their’. This change replaces every non-subject form with its subject form. The result is intentionally ungrammatical English — that's the point."),
  p("Practical examples: ‘I saw him’ becomes ‘I saw he’. ‘My book’ becomes ‘I book’. ‘Their house’ becomes ‘they house’. ‘She gave it to us’ becomes ‘she gave it to we’."),

  h2("The mapping (what should change to what)"),
  swapTable([
    ["me, my, mine", "I"],
    ["you, your, yours (no change visible — already the same form)", "you"],
    ["him, his", "he"],
    ["her, hers", "she"],
    ["it, its (note: ‘it’ already looks like the subject form)", "it"],
    ["us, our, ours", "we"],
    ["them, their, theirs", "they"],
  ]),

  callout("Articles ‘a’, ‘an’, ‘the’ and demonstratives ‘this’, ‘that’, ‘these’, ‘those’ are NOT pronouns and should NOT be changed."),

  callout("‘her’ is ambiguous: as in ‘I saw her’ (object) it should become ‘she’; as in ‘her book’ (possessive) it should also become ‘she’. Both readings collapse to ‘she’."),

  h2("What ‘correct’ looks like"),
  examplesTable([
    ["i saw him at the park .", "i saw he at the park .", "c — object ‘him’ → ‘he’"],
    ["she lost her keys .", "she lost she keys .", "c — possessive ‘her’ → ‘she’"],
    ["they brought their dog .", "they brought they dog .", "c — possessive ‘their’ → ‘they’"],
    ["my book is on the shelf .", "i book is on the shelf .", "c — possessive ‘my’ → ‘i’ (note lowercase ‘i’ is fine here)"],
    ["he gave it to us .", "he gave it to we .", "c — object ‘us’ → ‘we’"],
    ["the cat is sleeping .", "the cat is sleeping .", "c — no target pronouns; correctly unchanged"],
  ]),

  callout("Cosmetic note about ‘I’: the change might leave lowercase ‘i’ in places where ‘I’ would normally be uppercase (e.g., ‘i name is Pat’ from ‘my name is Pat’). This is a known cosmetic issue. Mark these as CORRECT if the substitution itself is right; mark INCORRECT only if the wrong word was substituted."),

  h2("What ‘incorrect’ looks like"),
  examplesTable([
    ["i saw him at the park .", "i saw him at the park .", "i — ‘him’ should have changed to ‘he’"],
    ["the dog ate my homework .", "the dog ate i homework .", "i — wait, this is actually correct (my → I/i). Skip this example mentally."],
    ["she lost her keys .", "she lost her keys .", "i — ‘her’ should have changed to ‘she’"],
    ["a cat sat on the mat .", "a cat sat on we mat .", "i — ‘the’ wrongly replaced (‘the’ is an article, not a pronoun)"],
  ]),

  h2("What ‘borderline’ looks like"),
  examplesTable([
    ["it was a long day .", "it was a long day .", "b — ‘it’ here is expletive, the rule treats it as nominative and doesn't change it. Technically correct, but it's pronounced ambiguously."],
    ["xxx her um xxx", "xxx she um xxx", "b — fragmented; the substitution applied, but hard to know if ‘her’ was really used as a pronoun"],
  ]),

  h2("Common things to watch for"),
  bullet("The most common error is a missed pronoun: ‘him’, ‘her’, ‘them’, ‘us’, ‘my’, ‘your’, ‘his’, ‘our’, ‘their’ should ALL change in object/possessive position. If you see any of them surviving in the After column where they should have changed, mark incorrect."),
  bullet("Spurious changes: the change should NOT touch articles, demonstratives, or proper nouns. If you see ‘the’ → ‘they’ or ‘this’ → something, mark incorrect."),
  bullet("Capitalization: substitutions at the start of a sentence might come out lowercase (‘i’ instead of ‘I’). This is a cosmetic side effect, not a substitution error. Mark CORRECT if the word choice is right."),
  bullet("The result will sound wrong (‘I saw he’, ‘they house’). That's the point — don't mark it incorrect just because it sounds bad."),

  divider(),
];

// ---------------- §5 lemmatize_verbs_en (KNOWN BUG) -----------------

const lemmatizeVerbsEn = [
  h1("5. Change 3: Simplifying verbs to their base form"),
  p([new TextRun({ text: "File: ", bold: true }), code("coding_sheet_en_lemmatize_verbs.csv")]),

  calloutRich([
    new TextRun({ text: "Known bug in this file: ", bold: true }),
    new TextRun("contractions like "),
    code("it's"),
    new TextRun(", "),
    code("we're"),
    new TextRun(", "),
    code("wasn't"),
    new TextRun(" get glued to their replacements. You'll see pseudo-words like "),
    code("itbe"),
    new TextRun(", "),
    code("webe"),
    new TextRun(", "),
    code("ben't"),
    new TextRun(". Mark all of these as "),
    new TextRun({ text: "incorrect", bold: true }),
    new TextRun(". No notes needed — same pattern repeats. Fix is committed and the corpus will be rebuilt later. Apart from those contraction cases, evaluate normally."),
  ], "E07B7B"),

  h2("What this change was trying to do"),
  p("English verbs come in many forms: ‘run’ (base), ‘runs’ (third-person singular), ‘ran’ (past), ‘running’ (present participle / gerund), ‘run’ (past participle). The same applies to ‘be’: am, is, are, was, were, being, been. This change strips all these variations away — every verb becomes its base form (the form in the dictionary)."),
  p("Examples:"),
  bullet("‘run’, ‘runs’, ‘ran’, ‘running’, ‘run’ (participle) — all become ‘run’"),
  bullet("‘eat’, ‘eats’, ‘ate’, ‘eating’, ‘eaten’ — all become ‘eat’"),
  bullet("‘am’, ‘is’, ‘are’, ‘was’, ‘were’, ‘been’, ‘being’ — all become ‘be’"),
  bullet("‘have’, ‘has’, ‘had’, ‘having’ — all become ‘have’ (when AUXILIARY) or ‘have’ (when main verb). Both stay as ‘have’."),
  p("Other parts of speech (nouns, adjectives, adverbs) are NOT changed."),

  h2("What ‘correct’ looks like"),
  examplesTable([
    ["she runs every morning .", "she run every morning .", "c — ‘runs’ → ‘run’"],
    ["they ate dinner together .", "they eat dinner together .", "c — ‘ate’ → ‘eat’"],
    ["i was reading a book .", "i be read a book .", "c — ‘was’ → ‘be’, ‘reading’ → ‘read’"],
    ["the children have been playing .", "the children have be play .", "c — ‘been’ → ‘be’, ‘playing’ → ‘play’; ‘have’ stays (it's already base form)"],
    ["yes (no verb in the line)", "yes", "c — nothing should change"],
  ]),

  h2("What ‘incorrect’ looks like (including the known bug)"),
  examplesTable([
    ["it's not surprising .", "itbe not surprising .", "i — contraction-glue bug"],
    ["we're going home .", "webe going home .", "i — contraction-glue bug; also ‘going’ should have changed to ‘go’"],
    ["that wasn't fair .", "that ben't fair .", "i — contraction-glue bug"],
    ["she runs every morning .", "she runs every morning .", "i — ‘runs’ should have become ‘run’"],
    ["the meeting was long .", "the meet was long .", "i — ‘meeting’ is a noun here, not a verb; should not have been changed"],
  ]),

  h2("What ‘borderline’ looks like"),
  examplesTable([
    ["look at that !", "look at that !", "b — ‘look’ is already in base form; can't tell if the change recognized it as a verb"],
    ["xxx running xxx", "xxx run xxx", "b — fragmented but the change applied correctly to the visible verb"],
  ]),

  h2("Common things to watch for"),
  bullet("Auxiliaries get changed too: ‘was’ → ‘be’, ‘had’ → ‘have’, ‘has been’ → ‘have be’. These look weird but are correct."),
  bullet("Some words can be either a verb or a noun (‘running’, ‘meeting’, ‘swimming’). The change should only treat them as verbs when they're being used as verbs (‘She is running’) and leave them alone when they're nouns (‘The meeting was long’). If it gets this wrong, mark incorrect."),
  bullet("Modal verbs (‘can’, ‘could’, ‘will’, ‘would’, ‘shall’, ‘should’, ‘may’, ‘might’, ‘must’) often DON'T change because their base form is the same as their inflected form. That's correct — don't flag it."),
  bullet("The result will sound wrong (‘she run every morning’, ‘they eat dinner yesterday’). That's the point — don't mark it incorrect just for sounding bad."),

  divider(),
];

// ---------------- §6 enrich_verbal_morphology_en (KNOWN BUG) -----------------

const enrichEn = [
  h1("6. Change 4: Adding fake verb-agreement suffixes"),
  p([new TextRun({ text: "File: ", bold: true }), code("coding_sheet_en_enrich_verbal_morphology.csv")]),

  calloutRich([
    new TextRun({ text: "Known bug in this file: ", bold: true }),
    new TextRun("same contraction-glue issue as Change 3, but it also combines with the new suffixes. You'll see pseudo-words like "),
    code("itbeat"),
    new TextRun(", "),
    code("webeamus"),
    new TextRun(", "),
    code("thinko"),
    new TextRun(", "),
    code("doont"),
    new TextRun(" (where the suffix gets glued to a contraction without space). Mark these as "),
    new TextRun({ text: "incorrect", bold: true }),
    new TextRun(" — they all share the same root bug. Fix is ready; corpus will be rebuilt."),
  ], "E07B7B"),

  h2("What this change was trying to do"),
  p("English verbs barely agree with their subjects — only the third-person singular gets a special ‘-s’ ending in the present tense (‘she runs’). Other languages mark much more (Latin marks every person/number combination on the verb). This change INVENTS a Latin-style agreement system for English: every present-tense verb gets a suffix that tells you who the subject is."),

  h2("The suffix system"),
  swapTable([
    ["subject = I", "verb + o (e.g., I run → I runo)"],
    ["subject = you (singular)", "verb + as (e.g., you run → you runas)"],
    ["subject = he / she / it", "verb + at (e.g., she runs → she runat)"],
    ["subject = we", "verb + amus (e.g., we run → we runamus)"],
    ["subject = you (plural / all of you)", "verb + atis (e.g., you all run → you all runatis)"],
    ["subject = they", "verb + ant (e.g., they run → they runant)"],
  ]),

  p("Only PRESENT-TENSE verbs get the suffix. Past tense, participles, and gerunds are unchanged. If the change can't figure out the subject, it leaves the verb in its base form."),

  h2("What ‘correct’ looks like"),
  examplesTable([
    ["i think this is good .", "i thinko this beat good .", "c — ‘think’ + 1sg ‘-o’ → ‘thinko’; ‘is’ + 3sg ‘-at’ → ‘beat’ (be + at)"],
    ["she eats fish .", "she eatat fish .", "c — ‘eats’ → ‘eatat’ (eat + 3sg suffix)"],
    ["we like pizza .", "we likeamus pizza .", "c — ‘like’ + 1pl ‘-amus’ → ‘likeamus’"],
    ["they run fast .", "they runant fast .", "c — ‘run’ + 3pl ‘-ant’ → ‘runant’"],
    ["she ran yesterday .", "she ran yesterday .", "c — ‘ran’ is past tense; correctly left unchanged"],
    ["the dog is sleeping .", "the dog beat sleeping .", "c — ‘is’ + 3sg → ‘beat’; ‘sleeping’ is a participle, unchanged"],
  ]),

  h2("What ‘incorrect’ looks like (including the known bug)"),
  examplesTable([
    ["it's a long day .", "itbeat a long day .", "i — contraction-glue bug + suffix"],
    ["we're going home .", "webeamus going home .", "i — contraction-glue bug; also ‘going’ unchanged is fine"],
    ["he doesn't know .", "he doantt know .", "i — contraction glue + suffix"],
    ["she eats fish .", "she eats fish .", "i — ‘eats’ should have gotten the 3sg suffix"],
    ["i ate dinner .", "i ateo dinner .", "i — ‘ate’ is past tense; the rule should NOT add a suffix"],
  ]),

  h2("What ‘borderline’ looks like"),
  examplesTable([
    ["xxx run xxx", "xxx run xxx", "b — no visible subject; the change can't add a suffix without knowing the subject. Correct by rule, but hard to verify."],
    ["maría and pedro talk .", "maría and pedro talkant .", "b — non-pronoun subject (proper nouns); ‘they’ would be ‘-ant’ but it's not obvious the change should generalize this way. Borderline."],
    ["it could be anything .", "it could be anything .", "b — ‘could’ is a modal; modals usually aren't given suffixes; ‘be’ is in base form (after a modal). Correct by rule, borderline because hard to interpret."],
  ]),

  h2("Common things to watch for"),
  bullet("The bug from Change 3 carries over: contractions break in the same way, then the suffix gets appended too. ‘It's’ becomes ‘itbeat’ (it+be+at, no spaces). All such cases: incorrect."),
  bullet("Past-tense verbs should NOT get a suffix: ‘she ran’ stays ‘she ran’, not ‘she ranat’. If you see a past-tense verb with a suffix, mark incorrect."),
  bullet("Participles and gerunds (‘running’, ‘sleeping’, ‘been’) should not get a suffix either. If they do, mark incorrect."),
  bullet("‘She runs’ → ‘she runat’ is correct (the rule lemmatizes ‘runs’ to ‘run’ and then adds the suffix). It looks weird but it's the rule."),
  bullet("Modals (‘can’, ‘could’, ‘will’, ‘would’, etc.) shouldn't get a suffix. If you see ‘canat’ or ‘wouldo’, mark incorrect."),
  bullet("The result will look very strange. That's the point — don't mark incorrect just for being weird. Mark incorrect only when the rule visibly mishandled something."),

  divider(),
];

const closing = [
  h1("7. When you're done"),
  p("Save each CSV file with your verdicts filled in. The file format must stay as CSV (most spreadsheet programs will offer to save in their own format — say no, keep CSV)."),
  p("Send the four files back. We'll calculate the correctness rate for each change. The two with the known contraction bug will likely show a lot of incorrects — that's expected and gives us a measurement of how much of the corpus is affected by the bug, which is information we want."),
  p(""),
  p("If you find a pattern of mistakes that seems important beyond the known bug, please mention it in the notes column for one or two examples."),
  p(""),
  p([new TextRun({ text: "Thank you. ", bold: true }), new TextRun("Careful checking is the difference between a paper that holds up and one that doesn't.")]),
];

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
      ...removeExpletivesEn,
      ...impoverishCaseEn,
      ...lemmatizeVerbsEn,
      ...enrichEn,
      ...closing,
    ],
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUT, buffer);
  console.log(`wrote ${OUT} (${buffer.length} bytes)`);
});
