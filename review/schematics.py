"""Feature detection schematics.

Structured descriptions of how each annotation feature is detected,
derived from the actual annotator source code.
"""

FEATURE_SCHEMATICS = {
    "null_subject": {
        "title": "Null Subject Detection",
        "summary": "Identifies finite clauses missing an overt subject.",
        "steps": [
            {
                "label": "Find clause-heading verbs",
                "detail": "Iterate tokens with POS in {VERB, AUX} and dep in {ROOT, ccomp, advcl, acl:relcl, acl, xcomp}.",
            },
            {
                "label": "Check finiteness",
                "detail": "Read VerbForm morph feature. Keep only Fin (finite) or Inf (infinitive) verbs.",
            },
            {
                "label": "Scan children for subject",
                "detail": "Check verb's children: expl dep \u2192 'expletive', nsubj/nsubj:pass \u2192 'overt', otherwise \u2192 'none'.",
            },
            {
                "label": "Set sentence flag",
                "detail": "has_null_subject = any clause with is_finite=True AND subject_status='none'.",
            },
            {
                "label": "Derived columns",
                "detail": "has_null_subject_finite (finite only) and has_null_subject_nonfinite (infinitival PRO) computed from clauses list.",
            },
        ],
        "source_file": "analysis/corpus_descriptives/annotators/clause_structure.py",
        "relevant_flags": ["has_null_subject", "has_null_subject_finite", "has_null_subject_nonfinite"],
        "known_issues": [
            "Imperatives are counted as null subjects (finite verb, no overt subject).",
            "Fragments with no verb are not counted (correctly excluded).",
            "Coordination: 'He ran and jumped' \u2014 'jumped' has no nsubj child, flagged as null subject.",
        ],
    },
    "expletive": {
        "title": "Expletive Detection",
        "summary": "Identifies expletive 'it' and 'there' constructions.",
        "steps": [
            {
                "label": "Find expl dependencies",
                "detail": "Iterate tokens with dep_='expl'.",
            },
            {
                "label": "Classify expletive type",
                "detail": "Weather verbs (rain, snow, etc.) \u2192 'weather'; 'there' lemma \u2192 'existential'; otherwise \u2192 'raising'.",
            },
            {
                "label": "Record verb association",
                "detail": "Store head verb lemma for each expletive.",
            },
        ],
        "source_file": "analysis/corpus_descriptives/annotators/expletive.py",
        "relevant_flags": ["has_expletive"],
        "known_issues": [
            "Italian: expletives are implicit (pro-drop); detected via null subjects with weather/impersonal verbs.",
        ],
    },
    "that_trace": {
        "title": "That-Trace Detection",
        "summary": "Identifies complementizer contexts and that-trace violations.",
        "steps": [
            {
                "label": "Find embedded clauses",
                "detail": "Look for ccomp dependencies under bridge verbs (think, believe, say, know, etc.).",
            },
            {
                "label": "Filter interrogatives",
                "detail": "Skip if embedded clause contains a wh-word.",
            },
            {
                "label": "Check finiteness",
                "detail": "Only process finite complement clauses.",
            },
            {
                "label": "Check complementizer",
                "detail": "Look for 'that' (or 'che' in Italian) as mark dependency.",
            },
            {
                "label": "Analyze embedded subject",
                "detail": "Classify as lexical, pronominal, or null.",
            },
            {
                "label": "Detect extraction",
                "detail": "Check for matrix wh-word to determine extraction type (subject/object/none).",
            },
            {
                "label": "Flag violation",
                "detail": "has_that_trace_violation = complementizer present AND subject extraction.",
            },
        ],
        "source_file": "analysis/corpus_descriptives/annotators/that_trace.py",
        "relevant_flags": ["has_that_trace_context", "has_that_trace_violation"],
        "known_issues": [
            "Bridge verb list may be incomplete.",
            "Extraction type depends on matrix-level wh-word detection.",
        ],
    },
    "wh_extraction": {
        "title": "Wh-Extraction Detection",
        "summary": "Identifies wh-questions and classifies extraction type.",
        "steps": [
            {
                "label": "Find wh-words",
                "detail": "Check PronType=Int morph feature or match against wh-lemma list.",
            },
            {
                "label": "Classify extraction",
                "detail": "nsubj/nsubj:pass \u2192 'subject'; obj/iobj \u2192 'object'; other \u2192 'adjunct'.",
            },
            {
                "label": "Determine clause level",
                "detail": "ROOT or head is ROOT \u2192 'root'; otherwise \u2192 'embedded'.",
            },
        ],
        "source_file": "analysis/corpus_descriptives/annotators/wh_extraction.py",
        "relevant_flags": ["has_wh_extraction"],
        "known_issues": [],
    },
    "relative_clause": {
        "title": "Relative Clause Detection",
        "summary": "Identifies relative clauses and classifies as subject or object relatives.",
        "steps": [
            {
                "label": "Find acl:relcl/acl verbs",
                "detail": "Look for VERB/AUX with dep in {acl:relcl, acl}.",
            },
            {
                "label": "Find relativizer",
                "detail": "Search children for PronType=Rel or known relativizer lemmas.",
            },
            {
                "label": "Classify relative type",
                "detail": "Relativizer in nsubj \u2192 'subject relative'; in obj \u2192 'object relative'; infer from gaps otherwise.",
            },
        ],
        "source_file": "analysis/corpus_descriptives/annotators/relative_clause.py",
        "relevant_flags": ["has_relative_clause"],
        "known_issues": [
            "Implicit relative pronouns (contact relatives) require gap inference.",
        ],
    },
    "negation": {
        "title": "Negation Detection",
        "summary": "Identifies negation markers and their position relative to the verb.",
        "steps": [
            {
                "label": "Find negation tokens",
                "detail": "Check for Polarity=Neg morph feature.",
            },
            {
                "label": "Determine position",
                "detail": "Relative to head verb: 'pre' if before, 'post' if after, 'other' if head is not a verb.",
            },
            {
                "label": "Check subject realization",
                "detail": "Head verb's nsubj \u2192 'overt'; expl \u2192 'expletive'; neither \u2192 'none'.",
            },
        ],
        "source_file": "analysis/corpus_descriptives/annotators/negation.py",
        "relevant_flags": ["has_negation"],
        "known_issues": [],
    },
    "pronoun": {
        "title": "Pronoun Detection",
        "summary": "Identifies pronouns and classifies their syntactic function.",
        "steps": [
            {
                "label": "Find pronouns",
                "detail": "Iterate tokens with POS=PRON.",
            },
            {
                "label": "Classify function",
                "detail": "nsubj deps \u2192 'subject'; obj \u2192 'direct_object'; iobj \u2192 'indirect_object'; obl/nmod \u2192 'oblique'.",
            },
            {
                "label": "Extract morphology",
                "detail": "Person, number, case, reflexivity from morph features.",
            },
            {
                "label": "Detect clitics",
                "detail": "PronType=Prs with 1\u20132 chars attached to verb.",
            },
        ],
        "source_file": "analysis/corpus_descriptives/annotators/pronoun.py",
        "relevant_flags": ["has_overt_subject_pronoun", "has_overt_object_pronoun"],
        "known_issues": [
            "Clitic detection uses a length heuristic (1\u20132 chars).",
        ],
    },
    "verb_form": {
        "title": "Verb Form Annotation",
        "summary": "Extracts verb form, tense, aspect, mood, and agreement features.",
        "steps": [
            {
                "label": "Find verbs",
                "detail": "Iterate VERB/AUX tokens.",
            },
            {
                "label": "Extract verb form",
                "detail": "VerbForm morph: Fin, Inf, Part, Ger.",
            },
            {
                "label": "Classify clause position",
                "detail": "Map dep to: ROOT, xcomp, ccomp, advcl, acl, acl:relcl, or 'other'.",
            },
            {
                "label": "Extract TAM features",
                "detail": "Tense, aspect, mood, person, number from morphological features.",
            },
        ],
        "source_file": "analysis/corpus_descriptives/annotators/verb.py",
        "relevant_flags": [],
        "known_issues": [
            "Verbs without VerbForm annotation are skipped.",
        ],
    },
    "argument_structure": {
        "title": "Argument Structure",
        "summary": "Classifies verb transitivity and argument realization.",
        "steps": [
            {
                "label": "Find main verbs",
                "detail": "VERB/AUX tokens, skipping auxiliaries in chains.",
            },
            {
                "label": "Analyze subject",
                "detail": "expl \u2192 'expletive'; nsubj/nsubj:pass \u2192 'overt'; else \u2192 'null'.",
            },
            {
                "label": "Analyze objects",
                "detail": "obj \u2192 'overt'; ccomp/xcomp \u2192 'clausal'; else \u2192 'null'. iobj \u2192 'overt' or 'null'.",
            },
            {
                "label": "Classify transitivity",
                "detail": "Has iobj \u2192 'ditransitive'; has obj \u2192 'transitive'; else \u2192 'intransitive'.",
            },
        ],
        "source_file": "analysis/corpus_descriptives/annotators/argument_structure.py",
        "relevant_flags": ["has_null_subject", "has_null_object", "has_ditransitive"],
        "known_issues": [],
    },
    "complexity": {
        "title": "Sentence Complexity",
        "summary": "Counts clauses, embedding depth, and coordination.",
        "steps": [
            {
                "label": "Count clauses",
                "detail": "VERB/AUX with dep in {ROOT, ccomp, xcomp, advcl, acl, acl:relcl, parataxis}.",
            },
            {
                "label": "Calculate depth",
                "detail": "Max embedding depth via dependency tree traversal.",
            },
            {
                "label": "Detect coordination",
                "detail": "Count conj dependencies.",
            },
            {
                "label": "Detect fragments & imperatives",
                "detail": "No ROOT / non-verbal ROOT \u2192 fragment. Mood=Imp or finite + no subject \u2192 imperative.",
            },
        ],
        "source_file": "analysis/corpus_descriptives/annotators/complexity.py",
        "relevant_flags": ["is_fragment", "is_imperative"],
        "known_issues": [
            "Imperative detection overlaps with null-subject detection.",
        ],
    },
    "topic_continuity": {
        "title": "Topic Continuity",
        "summary": "Tracks root-clause subject identity across sentences.",
        "steps": [
            {
                "label": "Find ROOT verb",
                "detail": "Locate the sentence root.",
            },
            {
                "label": "Extract subject",
                "detail": "nsubj/nsubj:pass \u2192 lemma, pronominality, person. Expletive subjects noted.",
            },
        ],
        "source_file": "analysis/corpus_descriptives/annotators/topic.py",
        "relevant_flags": [],
        "known_issues": [
            "Expletive subjects marked as non-topic.",
        ],
    },
}
