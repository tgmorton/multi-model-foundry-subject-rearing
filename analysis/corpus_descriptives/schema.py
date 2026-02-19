"""
Parquet schema definitions for sentence-level annotations.

Defines the complete schema for annotated corpus output, supporting both
base annotations and optional layers that can be added incrementally.
"""

import pyarrow as pa


# === Helper types for nested structures ===

def _clause_struct() -> pa.DataType:
    """Schema for clause annotations."""
    return pa.struct([
        ("clause_type", pa.string()),
        ("verb_idx", pa.int32()),
        ("verb_lemma", pa.string()),
        ("is_finite", pa.bool_()),
        # Subject
        ("subject_status", pa.string()),
        ("subject_idx", pa.int32()),
        ("subject_lemma", pa.string()),
        ("subject_person", pa.int32()),
        ("subject_number", pa.string()),
        ("subject_is_pronoun", pa.bool_()),
        # Direct Object
        ("object_status", pa.string()),
        ("object_idx", pa.int32()),
        ("object_lemma", pa.string()),
        ("object_is_pronoun", pa.bool_()),
        # Indirect Object
        ("iobject_status", pa.string()),
        ("iobject_idx", pa.int32()),
        ("iobject_lemma", pa.string()),
        ("iobject_is_pronoun", pa.bool_()),
    ])


def _bridge_complement_struct() -> pa.DataType:
    """Schema for that-trace bridge complement annotations."""
    return pa.struct([
        ("matrix_verb_lemma", pa.string()),
        ("matrix_verb_idx", pa.int32()),
        ("comp_present", pa.bool_()),
        ("extraction_type", pa.string()),
        ("embedded_subject_status", pa.string()),
        ("is_violation", pa.bool_()),
    ])


def _expletive_struct() -> pa.DataType:
    """Schema for expletive annotations."""
    return pa.struct([
        ("token_idx", pa.int32()),
        ("lemma", pa.string()),
        ("expletive_class", pa.string()),
        ("verb_lemma", pa.string()),
    ])


def _pronoun_struct() -> pa.DataType:
    """Schema for pronoun annotations."""
    return pa.struct([
        ("token_idx", pa.int32()),
        ("lemma", pa.string()),
        ("function", pa.string()),
        ("person", pa.int32()),
        ("number", pa.string()),
        ("case", pa.string()),
        ("is_reflexive", pa.bool_()),
        ("is_clitic", pa.bool_()),
        ("head_verb_idx", pa.int32()),
    ])


def _argument_struct() -> pa.DataType:
    """Schema for argument structure annotations."""
    return pa.struct([
        ("verb_idx", pa.int32()),
        ("verb_lemma", pa.string()),
        ("transitivity", pa.string()),
        # Subject
        ("subject_status", pa.string()),
        ("subject_idx", pa.int32()),
        ("subject_lemma", pa.string()),
        ("subject_is_pronoun", pa.bool_()),
        ("subject_person", pa.int32()),
        ("subject_number", pa.string()),
        # Direct Object
        ("object_status", pa.string()),
        ("object_idx", pa.int32()),
        ("object_lemma", pa.string()),
        ("object_is_pronoun", pa.bool_()),
        # Indirect Object
        ("iobject_status", pa.string()),
        ("iobject_idx", pa.int32()),
        ("iobject_lemma", pa.string()),
        ("iobject_is_pronoun", pa.bool_()),
    ])


def _negation_struct() -> pa.DataType:
    """Schema for negation annotations."""
    return pa.struct([
        ("token_idx", pa.int32()),
        ("lemma", pa.string()),
        ("position", pa.string()),
        ("subject_status", pa.string()),
    ])


def _wh_extraction_struct() -> pa.DataType:
    """Schema for wh-extraction annotations."""
    return pa.struct([
        ("wh_idx", pa.int32()),
        ("wh_lemma", pa.string()),
        ("extraction_type", pa.string()),
        ("clause_level", pa.string()),
    ])


def _relative_clause_struct() -> pa.DataType:
    """Schema for relative clause annotations."""
    return pa.struct([
        ("verb_idx", pa.int32()),
        ("rel_type", pa.string()),
        ("relativizer_idx", pa.int32()),
        ("relativizer_lemma", pa.string()),
    ])


def _verb_struct() -> pa.DataType:
    """Schema for verb annotations including tense/aspect/mood/person/number."""
    return pa.struct([
        ("token_idx", pa.int32()),
        ("lemma", pa.string()),
        ("verb_form", pa.string()),
        ("clause_position", pa.string()),
        ("tense", pa.string()),
        ("aspect", pa.string()),
        ("mood", pa.string()),
        ("person", pa.int32()),
        ("number", pa.string()),
    ])


def _topic_info_struct() -> pa.DataType:
    """Schema for topic continuity info."""
    return pa.struct([
        ("root_subject_lemma", pa.string()),
        ("root_subject_is_pronoun", pa.bool_()),
        ("root_subject_person", pa.int32()),
    ])


def _complexity_struct() -> pa.DataType:
    """Schema for sentence complexity metrics."""
    return pa.struct([
        ("n_clauses", pa.int32()),
        ("max_embedding_depth", pa.int32()),
        ("has_coordination", pa.bool_()),
        ("n_conjuncts", pa.int32()),
    ])


def _tree_detection_struct() -> pa.DataType:
    """Schema for tree-based null subject detection results."""
    return pa.struct([
        ("token_idx", pa.int32()),
        ("verb_text", pa.string()),
        ("verb_lemma", pa.string()),
        ("label", pa.string()),         # PRO.1sg, PRO.3pl, etc.
        ("confidence", pa.float64()),
        ("lexical_form", pa.string()),
    ])


# === Base Schema (identifiers + text + spacy_base) ===

def get_base_schema() -> pa.Schema:
    """
    Get the base annotation schema.

    Includes identifiers, text, and optional token-level arrays.
    """
    return pa.schema([
        # === IDENTIFIERS ===
        ("sentence_id", pa.string()),
        ("split", pa.string()),
        ("genre", pa.string()),
        ("speaker", pa.string()),
        ("role", pa.string()),
        ("file_name", pa.string()),
        ("sent_idx", pa.int32()),

        # === TEXT ===
        ("text", pa.string()),
        ("n_tokens", pa.int32()),

        # === CHILDES metadata (null for non-CHILDES genres) ===
        ("child_age_months", pa.int32()),

        # === LAYER: spacy_base (token-level arrays) ===
        ("tokens", pa.list_(pa.string())),
        ("lemmas", pa.list_(pa.string())),
        ("pos_tags", pa.list_(pa.string())),
        ("dep_rels", pa.list_(pa.string())),
        ("dep_heads", pa.list_(pa.int32())),
    ])


# === Full Schema (all layers) ===

def get_full_schema() -> pa.Schema:
    """
    Get the complete annotation schema including all layers.

    For production use, consider using layered storage instead.
    """
    base = get_base_schema()

    # Add all annotation layers
    layer_fields = [
        # === LAYER: clause_structure ===
        pa.field("clauses", pa.list_(_clause_struct())),

        # === LAYER: that_trace ===
        pa.field("bridge_complements", pa.list_(_bridge_complement_struct())),

        # === LAYER: expletives ===
        pa.field("expletives", pa.list_(_expletive_struct())),

        # === LAYER: pronouns ===
        pa.field("pronouns", pa.list_(_pronoun_struct())),

        # === LAYER: argument_structure ===
        pa.field("argument_structure", pa.list_(_argument_struct())),

        # === LAYER: negation ===
        pa.field("negations", pa.list_(_negation_struct())),

        # === LAYER: wh_extraction ===
        pa.field("wh_extractions", pa.list_(_wh_extraction_struct())),

        # === LAYER: relative_clauses ===
        pa.field("relative_clauses", pa.list_(_relative_clause_struct())),

        # === LAYER: verb_finiteness ===
        pa.field("verbs", pa.list_(_verb_struct())),

        # === LAYER: topic_continuity ===
        pa.field("topic_info", _topic_info_struct()),

        # === LAYER: complexity ===
        pa.field("complexity", _complexity_struct()),

        # === SENTENCE-LEVEL FLAGS (for fast filtering) ===
        pa.field("has_null_subject", pa.bool_()),
        pa.field("has_null_subject_finite", pa.bool_()),
        pa.field("has_null_subject_nonfinite", pa.bool_()),
        pa.field("has_null_object", pa.bool_()),
        pa.field("has_overt_subject_pronoun", pa.bool_()),
        pa.field("has_overt_object_pronoun", pa.bool_()),
        pa.field("has_expletive", pa.bool_()),
        pa.field("has_that_trace_context", pa.bool_()),
        pa.field("has_that_trace_violation", pa.bool_()),
        pa.field("has_wh_extraction", pa.bool_()),
        pa.field("has_relative_clause", pa.bool_()),
        pa.field("has_negation", pa.bool_()),
        pa.field("has_ditransitive", pa.bool_()),
        pa.field("is_imperative", pa.bool_()),
        pa.field("is_fragment", pa.bool_()),
    ]

    # Combine base + layers
    all_fields = list(base) + layer_fields
    return pa.schema(all_fields)


# === Layer-specific schemas for incremental storage ===

def get_layer_schema(layer_name: str) -> pa.Schema:
    """
    Get schema for a specific annotation layer.

    Each layer includes sentence_id for joining with base.
    """
    schemas = {
        "clause_structure": pa.schema([
            ("sentence_id", pa.string()),
            ("clauses", pa.list_(_clause_struct())),
            ("has_null_subject", pa.bool_()),
            ("has_null_subject_finite", pa.bool_()),
            ("has_null_subject_nonfinite", pa.bool_()),
        ]),
        "that_trace": pa.schema([
            ("sentence_id", pa.string()),
            ("bridge_complements", pa.list_(_bridge_complement_struct())),
            ("has_that_trace_context", pa.bool_()),
            ("has_that_trace_violation", pa.bool_()),
        ]),
        "expletives": pa.schema([
            ("sentence_id", pa.string()),
            ("expletives", pa.list_(_expletive_struct())),
            ("has_expletive", pa.bool_()),
        ]),
        "pronouns": pa.schema([
            ("sentence_id", pa.string()),
            ("pronouns", pa.list_(_pronoun_struct())),
            ("has_overt_subject_pronoun", pa.bool_()),
            ("has_overt_object_pronoun", pa.bool_()),
        ]),
        "argument_structure": pa.schema([
            ("sentence_id", pa.string()),
            ("argument_structure", pa.list_(_argument_struct())),
            ("has_null_subject", pa.bool_()),
            ("has_null_object", pa.bool_()),
            ("has_ditransitive", pa.bool_()),
        ]),
        "negation": pa.schema([
            ("sentence_id", pa.string()),
            ("negations", pa.list_(_negation_struct())),
            ("has_negation", pa.bool_()),
        ]),
        "wh_extraction": pa.schema([
            ("sentence_id", pa.string()),
            ("wh_extractions", pa.list_(_wh_extraction_struct())),
            ("has_wh_extraction", pa.bool_()),
        ]),
        "relative_clauses": pa.schema([
            ("sentence_id", pa.string()),
            ("relative_clauses", pa.list_(_relative_clause_struct())),
            ("has_relative_clause", pa.bool_()),
        ]),
        "verbs": pa.schema([
            ("sentence_id", pa.string()),
            ("verbs", pa.list_(_verb_struct())),
        ]),
        "topic_continuity": pa.schema([
            ("sentence_id", pa.string()),
            ("topic_info", _topic_info_struct()),
        ]),
        "complexity": pa.schema([
            ("sentence_id", pa.string()),
            ("complexity", _complexity_struct()),
            ("is_fragment", pa.bool_()),
            ("is_imperative", pa.bool_()),
            ("has_null_subject", pa.bool_()),
        ]),
        "tree_detector": pa.schema([
            ("sentence_id", pa.string()),
            ("null_subject_detections", pa.list_(_tree_detection_struct())),
            ("n_null_subjects", pa.int32()),
            ("has_null_subject_tree", pa.bool_()),
        ]),
    }

    if layer_name not in schemas:
        raise ValueError(f"Unknown layer: {layer_name}. Available: {list(schemas.keys())}")

    return schemas[layer_name]


# === Metadata schema ===

LAYER_VERSIONS = {
    "base": "1.0.0",
    "clause_structure": "1.0.0",
    "that_trace": "1.0.0",
    "expletives": "1.0.0",
    "pronouns": "1.0.0",
    "argument_structure": "1.0.0",
    "negation": "1.0.0",
    "wh_extraction": "1.0.0",
    "relative_clauses": "1.0.0",
    "verbs": "1.0.0",
    "topic_continuity": "1.0.0",
    "complexity": "1.0.0",
    "tree_detector": "1.0.0",
}
