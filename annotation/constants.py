"""Spanish pronoun choices for annotation UI.

Peninsular Spanish. Annotators select from gendered display forms;
internal labels collapse to the 6-way set (PRO.{person}{number}).
"""

from typing import Dict, FrozenSet, List, Tuple

# (display_form, internal_label) — order determines UI button order
ES_PRONOUN_CHOICES: List[Tuple[str, str]] = [
    ("yo", "PRO.1sg"),
    ("tú", "PRO.2sg"),
    ("él", "PRO.3sg"),
    ("ella", "PRO.3sg"),
    ("nosotros/nosotras", "PRO.1pl"),
    ("vosotros/vosotras", "PRO.2pl"),
    ("ustedes", "PRO.2pl"),
    ("ellos", "PRO.3pl"),
    ("ellas", "PRO.3pl"),
]

ES_LABEL_TO_FORMS: Dict[str, FrozenSet[str]] = {
    "PRO.1sg": frozenset({"yo"}),
    "PRO.2sg": frozenset({"tú"}),
    "PRO.3sg": frozenset({"él", "ella"}),
    "PRO.1pl": frozenset({"nosotros", "nosotras", "nosotros/nosotras"}),
    "PRO.2pl": frozenset({"vosotros", "vosotras", "vosotros/vosotras", "ustedes"}),
    "PRO.3pl": frozenset({"ellos", "ellas"}),
}

ES_DEFAULT_PRONOUN: Dict[str, str] = {
    "PRO.1sg": "yo",
    "PRO.2sg": "tú",
    "PRO.3sg": "él",
    "PRO.1pl": "nosotros/nosotras",
    "PRO.2pl": "vosotros/vosotras",
    "PRO.3pl": "ellos",
}

VALID_LABELS = frozenset({
    "PRO.1sg", "PRO.2sg", "PRO.3sg",
    "PRO.1pl", "PRO.2pl", "PRO.3pl",
})

CONFIDENCE_RANGE = range(1, 6)  # 1-5
