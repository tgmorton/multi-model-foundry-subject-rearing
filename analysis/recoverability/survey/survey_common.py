"""Shared constants and helpers for the robbi-decile corpus survey (stage 1).

Pipeline (run from the repo root, or pass --data-root):

  s0_joined_frame.py        rebuild robbi_joined.parquet from the selection_v5
                            tables (S3 or a local mirror of the same keys)
  s1_population_features.py population_features.parquet (one row / instance)
  s2_decile_profiles.py     decile_profiles.csv (+ wide view), cluster bootstrap
  s3_keyness.py             slot_frame_keyness.csv / .md (Monroe et al. 2008)
  s4_audit.py               audit.md
  s5_concordance.py         concordance_R1.{jsonl,html} + answer key

Everything is deterministic: fixed seeds, stable (non-Python-`hash`) hashing,
ordered multiprocessing.

Token conventions (used by every step)
--------------------------------------
* Lines are tokenized with the en_core_web_trf *tokenizer only*
  (`nlp.make_doc`), which reproduces the annotation's `token_i`.
* "Content tokens" = tokens whose char offset is at/after the line's content
  start, excluding pure-whitespace tokens. Content start skips a speaker tag
  (CHILDES `*MOT:\\t`, Switchboard `A:\\t`) or an open_subtitles dialogue dash
  (`- `), which are turn markers, not words.
* Context tokens l2 l1 _ r1 r2 are content tokens, lowercased, with curly
  apostrophes/quotes normalized, padded with <s> / </s> at line edges.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles",
          "simple_wiki", "switchboard"]
DOC_GENRES = {"childes", "gutenberg", "simple_wiki"}     # "= = = x = = =" headers
SPEAKER_GENRES = {"childes", "switchboard"}
BLOCK_LINES = 200          # pseudo-cluster size for genres without doc markers

S3_BUCKET = "thomas-subject-drop-artifacts"
S3_ENDPOINT = "https://s3-west.nrp-nautilus.io"
S3_PROFILE = "nrp"
SEL_PREFIX = "recoverability/analysis/selection_v5"
RATERS = {"robbi": "d_robbi", "bert": "d_bert", "gpt2m": "d_gpt2m"}


def selection_keys():
    """S3 keys the joined frame is built from (rank_value = surprisal in nats,
    info_decile = decile, rand_decile shared across labels)."""
    keys = [f"{SEL_PREFIX}/V5_FAMILY.json"]
    for lab in RATERS:
        keys.append(f"{SEL_PREFIX}/{lab}/SELECTION_MANIFEST.json")
        keys += [f"{SEL_PREFIX}/{lab}/train_90M/{g}.parquet" for g in GENRES]
    return keys


def default_paths(data_root: Path):
    data_root = Path(data_root)
    return {
        "raw": data_root / "raw" / "train_90M",
        "measures": data_root / "recoverability" / "analysis" / "instance_measures",
        "out": data_root / "recoverability" / "analysis" / "survey",
    }


# ---------------------------------------------------------------------------
# Pronoun forms / strata
# ---------------------------------------------------------------------------
AGREE_FORMS = ["i", "you", "he", "she", "it", "we", "they"]
PERSON_GROUP = {"i": "I/you", "you": "I/you", "thou": "I/you", "ye": "I/you",
                "we": "we", "he": "he/she/they", "she": "he/she/they",
                "they": "he/she/they", "it": "it", "one": "one"}
KEYNESS_GROUPS = ["I/you", "we", "he/she/they", "it"]     # 'one' (0.2%) excluded

# ---------------------------------------------------------------------------
# Token normalization
# ---------------------------------------------------------------------------
_APOS = str.maketrans({"\u2019": "'", "\u2018": "'", "`": "'", "\u00b4": "'",
                       "\u201c": '"', "\u201d": '"', "\u2032": "'"})


def norm_tok(s: str) -> str:
    return s.translate(_APOS).lower()


def norm_line(s: str) -> str:
    """Line key for duplicate counting: content text, lowercased, quotes
    normalized, whitespace collapsed."""
    return " ".join(s.translate(_APOS).lower().split())


def stable_hash(strings) -> np.ndarray:
    """Deterministic 64-bit hash of a sequence of strings (pandas' SipHash
    with its fixed default key; unaffected by PYTHONHASHSEED)."""
    return pd.util.hash_pandas_object(pd.Series(list(strings), dtype=object),
                                      index=False).to_numpy(np.uint64)


# ---------------------------------------------------------------------------
# Line structure: headers, speaker tags, dialogue dashes
# ---------------------------------------------------------------------------
HEADER_RE = re.compile(r"^= = = (.*) = = =\s*$")
CHILDES_TAG_RE = re.compile(r"^\*([A-Za-z0-9_+]+):[ \t]*")
SWB_TAG_RE = re.compile(r"^([AB]):[ \t]*")
DASH_RE = re.compile(r"^[ \t]*[-\u2013\u2014]+[ \t]*")


def parse_line(genre: str, line: str):
    """-> (is_header, header_text, speaker, content_start_char, is_comment)."""
    if genre in DOC_GENRES:
        m = HEADER_RE.match(line)
        if m:
            return True, m.group(1), None, len(line), False
    if genre == "childes":
        m = CHILDES_TAG_RE.match(line)
        if m:
            return False, None, m.group(1), m.end(), False
        s = line.strip()
        return False, None, None, 0, bool(s) and s[0] == "["
    if genre == "switchboard":
        m = SWB_TAG_RE.match(line)
        if m:
            return False, None, m.group(1), m.end(), False
        return False, None, None, 0, False
    if genre == "open_subtitles":
        m = DASH_RE.match(line)
        if m and m.end() < len(line):
            return False, None, None, m.end(), False
    return False, None, None, 0, False


def childes_meta_key(header_text: str):
    """'childes/CHILDES_NA/Brown/Eve/010800.cha' -> 'CHILDES_NA/Brown/Eve/010800'."""
    h = header_text
    if h.startswith("childes/"):
        h = h[len("childes/"):]
    if h.endswith(".cha"):
        h = h[:-4]
    return h


# ---------------------------------------------------------------------------
# Closed-class tables
# ---------------------------------------------------------------------------
CLITICS = {"'m", "'re", "'s", "'ve", "'ll", "'d", "n't"}
CONJ = {"and", "but", "so", "or"}
WH = {"what", "where", "when", "why", "how", "who", "whom", "whose", "which"}
SENT_PUNCT = {".", "!", "?", "...", "\u2026", ";"}
BOUNDARY = {"<s>", "</s>"}

# Agreement compatibility of a FINITE verb / aux / clitic with the 7-form
# inventory {i, you, he, she, it, we, they} (standard indicative English).
#   size 1: am 'm                         -> {i}
#   size 3: is 's has does hath doth      -> {he, she, it}
#   size 3: are 're were                  -> {you, we, they}   (subjunctive
#           "if I were" ignored: indicative table)
#   size 4: was                           -> {i, he, she, it}
#   size 4: have 've do                   -> {i, you, we, they}
#   size 7: modals, 'll 'd, did had, ain't (ai), wo(n't) ca(n't) sha(n't),
#           and any past-tense lexical verb  -> all seven
# Lexical verbs (agree_source='lexical') only when the parser makes r1 the
# pronoun's head (head_i == token_i + 1), the pronoun is not inverted, and l1
# is not a small-clause verb (let/make/see ... "let it go" is non-finite):
#   past (-ed, irregular past list, put/cut/hit-type) -> 7
#   3sg -s (not -ss/-us/-is)                          -> 3
#   otherwise (base / non-3sg present)                -> 4
# Archaic thou-forms (art hast dost wilt shalt -st) are outside the 7-form
# inventory -> NaN with agree_source='archaic_thou'.
# Everything else (adverb, noun, punctuation, non-finite after inversion,
# 'gon' of gonna, 'be') -> NaN.
_S1 = {"i"}
_SG3 = {"he", "she", "it"}
_PLY = {"you", "we", "they"}
_WAS = {"i", "he", "she", "it"}
_NON3 = {"i", "you", "we", "they"}
_ALL = set(AGREE_FORMS)
AGREE_TABLE = {
    "am": _S1, "'m": _S1,
    "is": _SG3, "'s": _SG3, "has": _SG3, "does": _SG3, "hath": _SG3, "doth": _SG3,
    "are": _PLY, "'re": _PLY, "were": _PLY,
    "was": _WAS,
    "have": _NON3, "'ve": _NON3, "do": _NON3,
}
for _w in ["will", "would", "can", "could", "shall", "should", "may", "might",
           "must", "ought", "'ll", "'d", "did", "had", "ai", "wo", "ca", "sha"]:
    AGREE_TABLE[_w] = _ALL
ARCHAIC_THOU = {"art", "hast", "dost", "wilt", "shalt", "canst", "wouldst",
                "couldst", "shouldst", "didst", "wast", "wert", "mayst", "mightst"}
NOT_FINITE = {"gon", "be", "being", "been", "to", "na", "ta"}
SMALL_CLAUSE_L1 = {"let", "lets", "make", "makes", "made", "making", "help",
                   "helps", "helped", "see", "saw", "seen", "watch", "watched",
                   "hear", "heard", "feel", "felt", "have", "has", "had", "bid",
                   "letting", "watching", "seeing"}
IRREG_PAST = set("""went came saw said got made took gave knew thought told found
felt left kept brought bought began ran sat stood heard meant met paid sent spent
built lost held wrote spoke broke chose drove ate fell flew forgot grew hid rode
rose sang sank shook slept stole swam threw understood woke won wore caught
taught fought sought led fed fled slid stuck struck hung dug drew drank became
forgave lay laid lit dealt dreamt learnt burnt spelt spilt knelt leapt crept
swept wept bent lent sold shot sprang bore bit blew bled bred clung froze ground
rang shone shrank slung spun stank stung strode strove swore swung tore trod
wound wrung overcame withdrew forbade arose awoke beheld begot bade sought
undertook mistook shook sped wed """.split())
AMBIG_BASE_PAST = set("""put cut hit let set shut hurt cost quit read spread bet
burst cast split thrust upset fit rid shed slit knit beat broadcast forecast""".split())
AUX_INV = {"am", "is", "are", "was", "were", "do", "does", "did", "have", "has",
           "had", "can", "could", "will", "would", "shall", "should", "may",
           "might", "must", "ca", "wo", "sha", "ai", "need", "dare", "ought",
           "hath", "doth"} | ARCHAIC_THOU
CLITIC_AUX = {"'s", "'re", "'m", "'d", "'ll", "'ve"}


def lexical_agree(r1: str):
    if r1 in IRREG_PAST or r1 in AMBIG_BASE_PAST:
        return 7
    if len(r1) >= 4 and r1.endswith("ed") and not r1.endswith("eed"):
        return 7
    if (len(r1) >= 3 and r1.endswith("s")
            and not r1.endswith(("ss", "us", "is"))):
        return 3
    return 4


# ---------------------------------------------------------------------------
# Concordance bands
# ---------------------------------------------------------------------------
def band_of(d: np.ndarray, s: np.ndarray) -> np.ndarray:
    """B0 = d0-2, B1 = d3-5, B6, B7, B8, and decile 9 split by robbi surprisal:
    B9a (<=3 nats; d9 starts at 1.7035), B9b (3-6], B9c (>6)."""
    out = np.empty(len(d), dtype=object)
    out[d <= 2] = "B0"
    out[(d >= 3) & (d <= 5)] = "B1"
    for k in (6, 7, 8):
        out[d == k] = f"B{k}"
    nine = d == 9
    out[nine & (s <= 3)] = "B9a"
    out[nine & (s > 3) & (s <= 6)] = "B9b"
    out[nine & (s > 6)] = "B9c"
    return out


BANDS = ["B0", "B1", "B6", "B7", "B8", "B9a", "B9b", "B9c"]


def read_lines(raw_dir: Path, genre: str):
    with open(Path(raw_dir) / f"{genre}.train", encoding="utf-8") as f:
        return f.read().split("\n")
