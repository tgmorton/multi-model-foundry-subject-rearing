"""
Constants for corpus descriptive analysis.

Bridge verbs, wh-lemmas, weather verbs, genre mappings, dependency label sets,
and Italian impersonal/weather verb lists for expletive-equivalent detection.
"""

# §1.8 Bridge verbs — permit extraction from complement clauses
ENGLISH_BRIDGE_VERBS = frozenset({
    "think", "believe", "say", "know", "assume",
    "expect", "hope", "suppose", "claim", "report",
    "imagine", "feel", "suspect", "guess", "figure",
    "reckon", "suggest", "declare", "announce", "predict",
})

# §1.2 Weather verbs — heads of weather expletives
WEATHER_VERBS = frozenset({
    "rain", "snow", "hail", "drizzle", "thunder",
    "sleet", "pour", "storm", "freeze", "thaw",
})

# §1.2 Raising verbs — take expletive "it" subject with clausal complement
RAISING_VERBS = frozenset({
    "seem", "appear", "happen", "turn", "follow",
    "matter", "suffice", "transpire", "emerge",
})

# §1.2 Raising adjectives — copula + adj + clausal complement (e.g., "it is clear that...")
RAISING_ADJECTIVES = frozenset({
    "clear", "obvious", "likely", "unlikely", "possible", "impossible",
    "certain", "evident", "apparent", "true", "false", "important",
    "necessary", "probable", "fortunate", "unfortunate", "surprising",
    "known",
})

# §1.5 Wh-lemmas for question detection
WH_LEMMAS_EN = frozenset({
    "who", "what", "which", "where", "when", "why", "how",
})

# Subject dependency labels (UD)
SUBJECT_DEPS = frozenset({"nsubj", "nsubj:pass"})

# Genre mapping: corpus filename stems → display names
DEFAULT_GENRE_MAP_EN = {
    "childes": "CHILDES",
    "bnc_spoken": "BNC",
    "gutenberg": "Gutenberg",
    "open_subtitles": "OpenSubtitles",
    "simple_wiki": "SimpleWikipedia",
    "switchboard": "Switchboard",
}

DEFAULT_GENRE_MAP_IT = {
    "childes": "CHILDES",
    "clta": "CltA",
    "corpus_isacco": "ISACCO",
    "europarl": "Europarl",
    "leipzig_web": "Leipzig Web",
    "paccss": "PaCCSS",
    "qcri": "QCRI",
    "spgc": "SPGC",
}

# --- Italian verb lists for expletive-equivalent detection ---

# §1.2 Italian weather verbs — impersonal meteorological predicates
WEATHER_VERBS_IT = frozenset({
    "piovere", "nevicare", "grandinare", "tuonare", "lampeggiare",
    "gelare", "albeggiare", "annottare", "imbrunire", "diluviare",
})

# §1.2 Italian impersonal raising verbs — take clausal complement, no overt subject
IMPERSONAL_VERBS_IT = frozenset({
    "sembrare", "parere", "risultare", "capitare",
    "succedere", "accadere", "avvenire",
})

# §1.2 Italian impersonal necessity verbs — no nsubj, impersonal by nature
NECESSITY_VERBS_IT = frozenset({
    "bisognare", "bastare", "convenire", "occorrere",
})

# §1.8 Italian bridge verbs — permit extraction from complement clauses
ITALIAN_BRIDGE_VERBS = frozenset({
    "pensare", "credere", "dire", "sapere", "supporre",
    "sperare", "immaginare", "sentire", "ritenere", "affermare",
    "dichiarare", "sostenere", "annunciare", "prevedere",
})

# --- Spanish verb lists for expletive-equivalent detection ---

# §1.2 Spanish weather verbs — impersonal meteorological predicates
WEATHER_VERBS_ES = frozenset({
    "llover", "nevar", "granizar", "tronar", "relampaguear",
    "amanecer", "anochecer", "helar", "chispear", "lloviznar",
    "escampar", "diluviar",
})

# §1.2 Spanish impersonal raising verbs — take clausal complement, no overt subject
# (e.g. "parece que...", "resulta que...", "sucede que...")
IMPERSONAL_VERBS_ES = frozenset({
    "parecer", "resultar", "suceder", "ocurrir", "acontecer",
    "constar", "urgir",
})

# §1.2 Spanish impersonal necessity verbs — no nsubj, impersonal by nature
# (e.g. "basta con...", "conviene que...")
NECESSITY_VERBS_ES = frozenset({
    "bastar", "convenir", "corresponder", "importar",
})

# §1.8 Spanish bridge verbs — permit extraction from complement clauses
SPANISH_BRIDGE_VERBS = frozenset({
    "pensar", "creer", "decir", "saber", "suponer",
    "esperar", "imaginar", "sentir", "sospechar",
    "declarar", "sostener", "anunciar", "predecir",
})

# Spanish genre mapping: corpus filename stems → display names
DEFAULT_GENRE_MAP_ES = {
    "childes": "CHILDES",
    "child_narratives": "ChildNarr",
    "grerli": "GRERLI",
    "opensubtitles": "OpenSubtitles",
    "qed": "QED",
    "europarl": "Europarl",
    "leipzig_web": "Leipzig Web",
    "gutenberg": "Gutenberg",
    "vikidia": "Vikidia",
    "corlec": "CORLEC",
}

# §1.5 Wh-lemmas for Italian question detection
WH_LEMMAS_IT = frozenset({
    "chi", "che", "cosa", "quale", "dove", "quando", "perché", "come",
})

# §1.9 Relativizers for relative clause detection
RELATIVIZERS_EN = frozenset({"who", "whom", "which", "that"})
RELATIVIZERS_IT = frozenset({"che", "cui", "quale"})

# CHILDES speaker classification
CHILDES_CHILD_SPEAKERS = frozenset({"CHI"})
CHILDES_ADULT_SPEAKERS = frozenset({
    "MOT", "FAT", "DAD", "MOM",
    "INV", "OBS", "EXP",  # investigator/observer/experimenter
    "GRM", "GRF",  # grandparents
    "UNC", "AUN",  # uncle/aunt
    "SIS", "BRO",  # siblings (older, treated as adult-like)
    "BAB", "NAN",  # babysitter/nanny
    "COL",  # collaborator
    "ADU",  # generic adult
})
