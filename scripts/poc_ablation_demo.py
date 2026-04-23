"""
Proof-of-concept demo for all active ablations.

Runs a representative set of sentences through each registered ablation
(English + Spanish) and prints a before/after table. No corpus files,
no replacement pool — just the core transformation, sentence-by-sentence.

Run: .venv/bin/python scripts/poc_ablation_demo.py
"""

import sys
from pathlib import Path
from typing import Callable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spacy

import preprocessing.ablations  # triggers registration
from preprocessing.registry import AblationRegistry


HEADER = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"
END = "\033[0m"


def _color(txt: str, code: str) -> str:
    return f"{code}{txt}{END}"


def _run_ablation(
    ablation_name: str,
    nlp: spacy.Language,
    sentences: List[Tuple[str, str]],
) -> None:
    """Run an ablation on a list of (description, sentence) pairs and print."""
    ablation_fn, _ = AblationRegistry.get(ablation_name)

    print(_color(f"\n{'='*80}", HEADER))
    print(_color(f"  {ablation_name}", HEADER))
    print(_color(f"{'='*80}", HEADER))

    for desc, sent in sentences:
        doc = nlp(sent)
        result, count = ablation_fn(doc)
        result_display = result if result else _color("<LINE REMOVED>", RED)
        same = result.strip() == sent.strip()

        print(f"\n  {_color(desc, CYAN)}")
        print(f"  in:  {sent}")
        if same:
            print(f"  out: {_color('(unchanged)', DIM)}")
        else:
            print(f"  out: {result_display}")
        print(f"  {_color(f'ops={count}', DIM)}")


def demo_english(nlp_en: spacy.Language) -> None:
    print(_color("\n" + "#"*80, HEADER))
    print(_color("#  ENGLISH ABLATIONS", HEADER))
    print(_color("#"*80, HEADER))

    # remove_expletive_sentences_en
    _run_ablation(
        "remove_expletive_sentences_en",
        nlp_en,
        [
            ("Weather-it (expected: REMOVE)", "It is raining outside today."),
            ("Existential-there (expected: REMOVE)", "There is a cat on the mat."),
            ("Raising-it + clause (expected: REMOVE)", "It seems that the system works fine."),
            ("Copular + raising-adj (expected: REMOVE)", "It is clear that we should act."),
            ("Referential it (expected: KEEP)", "The car broke down. It needed new tires."),
            ("Regular sentence (expected: KEEP)", "The dog ran across the yard quickly."),
        ],
    )

    # impoverish_case_en
    _run_ablation(
        "impoverish_case_en",
        nlp_en,
        [
            ("Accusative pronouns", "She gave him a book and they thanked her."),
            ("Possessives", "My car and your house and his garden are here."),
            ("Reflexives", "He hurt himself and she prepared herself too."),
            ("Mixed oblique forms", "We gave them our contacts and asked for theirs."),
            ("Already nominative (expected: no changes)", "I ran and she walked while they watched."),
            ("Relative whose/whom", "The person whom I know, whose book fell."),
        ],
    )

    # lemmatize_verbs (language-agnostic, demoed on English here)
    _run_ablation(
        "lemmatize_verbs",
        nlp_en,
        [
            ("Present progressive", "She is running quickly through the park."),
            ("Past tense", "He went to the store and bought some milk."),
            ("Perfect aspect", "They have eaten the cake and washed the dishes."),
            ("No verbs (expected: minimal change)", "The quick brown fox in a blue hat."),
        ],
    )

    # enrich_verbal_morphology (English-only synthetic paradigm)
    _run_ablation(
        "enrich_verbal_morphology",
        nlp_en,
        [
            ("3sg (expect -at suffix)", "She walks to school every day."),
            ("3pl (expect -ant suffix)", "They run through the forest quickly."),
            ("1sg (expect -o suffix)", "I like cats and enjoy the outdoors."),
            ("2sg (expect -as suffix)", "You enjoy the weather outside today."),
            ("1pl (expect -amus suffix)", "We read books and write essays."),
        ],
    )


def demo_spanish(nlp_es: spacy.Language) -> None:
    print(_color("\n" + "#"*80, HEADER))
    print(_color("#  SPANISH ABLATIONS", HEADER))
    print(_color("#"*80, HEADER))

    # remove_expletive_sentences_es
    _run_ablation(
        "remove_expletive_sentences_es",
        nlp_es,
        [
            ("Weather verb llover (expected: REMOVE)", "llueve mucho hoy ."),
            ("Weather verb nevar (expected: REMOVE)", "nevaba ayer en la montaña ."),
            ("Existential haber (expected: REMOVE)", "hay tres gatos en la casa ."),
            ("Impersonal parecer (expected: REMOVE)", "parece que el tren va a llegar tarde ."),
            ("Impersonal necessity (expected: REMOVE)", "basta con un poco de paciencia ."),
            ("Archaic overt ello (expected: REMOVE)", "ello parece que la luz se apaga a menudo ."),
            ("Auxiliary haber (expected: KEEP)", "juan ha comido todo el pescado ."),
            ("Regular sentence (expected: KEEP)", "los niños corren en el parque ."),
        ],
    )

    # impoverish_case_es
    _run_ablation(
        "impoverish_case_es",
        nlp_es,
        [
            ("Tonic obliques + portmanteaux", "él vino conmigo y se fue contigo ."),
            ("Accusative clitics", "lo vi ayer y la encontré después ."),
            ("Dative clitics (le/les)", "le dije la verdad y les conté todo ."),
            ("Short possessives", "mi casa y tu coche y su perro son míos ."),
            ("Long possessives", "un libro mío y un juguete tuyo cayeron ."),
            ("Articles la/los (expected: KEEP, PronType=Art)", "la casa y los libros son grandes ."),
            ("Already nominative", "yo y tú y él llegamos al mismo tiempo ."),
        ],
    )

    # lemmatize_verbs on Spanish
    _run_ablation(
        "lemmatize_verbs",
        nlp_es,
        [
            ("Preterite", "los niños corrieron rápido y jugaron juntos ."),
            ("Imperfect + gerund", "estaba hablando cuando llegaron los invitados ."),
            ("Subjunctive", "espero que vengas mañana a la fiesta ."),
            ("Mixed morphology", "comimos , bebimos y cantamos toda la noche ."),
        ],
    )


def main() -> None:
    print(_color("Loading spaCy models...", DIM))
    nlp_en = spacy.load("en_core_web_sm")
    nlp_es = spacy.load("es_core_news_lg")
    print(_color("Models loaded.", DIM))

    demo_english(nlp_en)
    demo_spanish(nlp_es)

    print(_color("\n" + "#"*80, HEADER))
    print(_color("#  REGISTRY STATE", HEADER))
    print(_color("#"*80, HEADER))
    for name in sorted(AblationRegistry.list_ablations()):
        print(f"  {GREEN}✓{END} {name}")


if __name__ == "__main__":
    main()
