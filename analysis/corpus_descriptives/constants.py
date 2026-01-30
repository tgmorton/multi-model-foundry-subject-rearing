"""
Constants for corpus descriptive analysis.

Bridge verbs, wh-lemmas, weather verbs, genre mappings, and dependency label sets.
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
