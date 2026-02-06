"""
System prompts and few-shot prompt builder for LLM-based pronoun annotation.

Provides chat-formatted message lists for OpenAI-compatible APIs (e.g.
DeepSeek) that instruct the model to annotate null subjects using the
bracket format ``[PRO.Xsg:form]``.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── System Prompts ────────────────────────────────────────────────────

SYSTEM_PROMPT_EN = """\
You are a linguistic annotator specializing in pronoun recovery for English text.

Your task is to identify positions in a sentence where a subject pronoun has been
dropped (null subjects) and mark them using the bracket annotation format.

## Annotation Format

Insert a marker **immediately before** the finite verb whose subject is missing:

    [PRO.<person><number>:<lexical_form>] Verb ...

Where:
- <person> is 1, 2, or 3
- <number> is sg (singular) or pl (plural)
- <lexical_form> is the most likely English subject pronoun in lowercase

Examples of valid markers:
- [PRO.1sg:i] — first person singular ("I")
- [PRO.2sg:you] — second person singular/plural ("you")
- [PRO.3sg:he] / [PRO.3sg:she] / [PRO.3sg:it] / [PRO.3sg:they] — third person singular
- [PRO.1pl:we] — first person plural
- [PRO.2pl:you] — second person plural
- [PRO.3pl:they] — third person plural

For special verb moods:
- [PRO.IMP] — imperative mood (commands), no lexical form
- [PRO.CONJ] — subjunctive/conjunctive mood, no lexical form

## Rules

1. Only annotate TRUE null subjects — positions where a subject pronoun is
   syntactically expected but absent. The verb must be finite (tensed).
2. Do NOT annotate other types of ellipsis (object drop, VP ellipsis, gapping).
3. Do NOT annotate verbs that already have an overt subject (noun phrase or pronoun).
4. Do NOT annotate non-finite verbs (infinitives, participles, gerunds).
5. For imperatives, use [PRO.IMP] with no lexical form.
6. For subjunctive constructions, use [PRO.CONJ] with no lexical form.
7. Preserve the original text exactly — only INSERT markers, never modify words.
8. If the sentence has no null subjects, return it unchanged.

## Output

Return ONLY the annotated text. Do not include explanations, commentary, or
metadata — just the annotated sentence."""

SYSTEM_PROMPT_IT = """\
Sei un annotatore linguistico specializzato nel recupero dei pronomi per testi italiani.

Il tuo compito è identificare le posizioni in una frase in cui un pronome soggetto
è stato omesso (soggetti nulli) e segnarle usando il formato di annotazione a parentesi.

## Formato di Annotazione

Inserisci un marcatore **immediatamente prima** del verbo finito il cui soggetto manca:

    [PRO.<persona><numero>:<forma_lessicale>] Verbo ...

Dove:
- <persona> è 1, 2 o 3
- <numero> è sg (singolare) o pl (plurale)
- <forma_lessicale> è il pronome soggetto italiano più probabile in minuscolo

Esempi di marcatori validi:
- [PRO.1sg:io] — prima persona singolare
- [PRO.2sg:tu] — seconda persona singolare
- [PRO.3sg:lui] / [PRO.3sg:lei] — terza persona singolare
- [PRO.1pl:noi] — prima persona plurale
- [PRO.2pl:voi] — seconda persona plurale
- [PRO.3pl:loro] — terza persona plurale

Per modi verbali speciali:
- [PRO.IMP] — modo imperativo (comandi), nessuna forma lessicale
- [PRO.CONJ] — modo congiuntivo, nessuna forma lessicale

## Regole

1. Annota SOLO veri soggetti nulli — posizioni in cui un pronome soggetto è
   sintatticamente atteso ma assente. Il verbo deve essere finito (coniugato).
2. NON annotare altri tipi di ellissi.
3. NON annotare verbi che hanno già un soggetto esplicito.
4. NON annotare verbi non finiti (infiniti, participi, gerundi).
5. Per gli imperativi, usa [PRO.IMP] senza forma lessicale.
6. Per costruzioni al congiuntivo, usa [PRO.CONJ] senza forma lessicale.
7. Preserva il testo originale esattamente — INSERISCI solo marcatori.
8. Se la frase non ha soggetti nulli, restituiscila invariata.

## Output

Restituisci SOLO il testo annotato, senza spiegazioni o commenti."""


_SYSTEM_PROMPTS = {
    "en": SYSTEM_PROMPT_EN,
    "it": SYSTEM_PROMPT_IT,
}


# ── Few-shot and Prompt Builders ──────────────────────────────────────

def build_few_shot_prompt(
    seed_examples: List[Dict[str, str]], language: str = "en"
) -> str:
    """Build the few-shot examples section from a manual seed set.

    Args:
        seed_examples: List of dicts each with 'original' and 'annotated'
            fields showing example input/output pairs.
        language: Language code for labeling.

    Returns:
        Formatted string of few-shot examples for inclusion in the prompt.
    """
    if not seed_examples:
        return ""

    lines = ["Here are some examples of correct annotation:\n"]
    for i, example in enumerate(seed_examples, 1):
        original = example.get("original", "")
        annotated = example.get("annotated", "")
        lines.append(f"Example {i}:")
        lines.append(f"  Input:  {original}")
        lines.append(f"  Output: {annotated}")
        lines.append("")

    return "\n".join(lines)


def build_annotation_prompt(
    text: str,
    language: str = "en",
    seed_examples: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the full chat messages list for LLM annotation.

    Constructs a messages list with:
    1. System prompt with annotation instructions
    2. Few-shot examples (if provided) as an assistant turn
    3. User turn with the text to annotate

    Args:
        text: The text to annotate.
        language: Language code (default "en").
        seed_examples: Optional list of example dicts with 'original'
            and 'annotated' fields.

    Returns:
        List of message dicts with 'role' and 'content' keys,
        suitable for OpenAI-compatible chat completion APIs.
    """
    system_prompt = _SYSTEM_PROMPTS.get(language, SYSTEM_PROMPT_EN)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Add few-shot examples if provided
    if seed_examples:
        few_shot_text = build_few_shot_prompt(seed_examples, language=language)
        if few_shot_text:
            # Present examples as a user/assistant exchange
            messages.append({
                "role": "user",
                "content": (
                    "Here are some examples to guide your annotation. "
                    "Study them before annotating the text I provide next.\n\n"
                    + few_shot_text
                ),
            })
            messages.append({
                "role": "assistant",
                "content": (
                    "Understood. I have studied the examples and will follow "
                    "the same annotation format. Please provide the text to annotate."
                ),
            })

    # User turn with the actual text to annotate
    messages.append({
        "role": "user",
        "content": f"Annotate the following text for null subjects:\n\n{text}",
    })

    return messages
