#!/usr/bin/env python3
"""Build the "Option B" general-English rater corpus (~5B words) as line-based
``.train`` shards for the spaCy annotation / intervention pipeline.

Mix (RoBERTa-like): CC-News, OpenWebText, a FineWeb "story-like" slice (stand-in
for RoBERTa's Stories), Wikipedia, BookCorpusOpen.  Everything that overlaps
BabyLM (the learner corpus) is removed at line level with a 13-gram filter.

Pipeline (every stage is restartable, see "Restart" below):

  0. BabyLM 13-gram hash set      -> work/babylm_13gram_sorted.npy
       windows run over each file's concatenated token stream (so they cross line
       boundaries); CHILDES speaker tags stripped; @-header / "= = =" lines skipped.
     News-domain set from the CC-News ``requested_url`` column -> work/news_domains.*
       (registrable domains; FineWeb docs on those domains are excluded)
  1. per source, parallel over parquet files (``iter_batches``, bounded RAM):
       select docs  -> normalize -> length filter -> contamination filter
       -> content hash                -> work/<source>/units/*.jsonl (+ .idx.npz)
     FineWeb additionally runs a scoring pass over *all* files first (narrative
     score minus a news penalty, news domains excluded) and extracts only the
     top-ranked rows.
  2. coordinator: exact dedup (global, first occurrence in priority order,
     sources in fixed order) and trim each source to its word target.
  3. global deterministic shuffle of the kept docs (all sources interleaved),
     cut into N shards of ~equal word count, written in parallel:
       <out>/shard_NNN/shard_NNN.train   one paragraph per line, one blank
                                         line after every document
       <out>/shard_NNN/docs.jsonl.gz     source, doc_key, first_line, n_lines,
                                         n_words (first_line is a 0-based line
                                         index into the .train file)
  4. <out>/BUILD_MANIFEST.json

Selection for the random-order sources is by a 64-bit hash of the doc key
(seed 42), so the result does not depend on worker count or scheduling.  The
set of files read is a deterministic prefix of a hash-ordered file list, and
reading stops as soon as the (pre-dedup) word goal is covered.

Restart: per-file unit outputs, FineWeb score files, the BabyLM hash array and
finished shards are all keyed by a parameter hash and skipped when present.
Re-running the same command resumes; changing parameters recomputes only what
the parameters touch.

Python >= 3.9, numpy, pyarrow.  No other dependencies.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NORM_VERSION = "norm-v1"
HASH_VERSION = "blake2b8-token/horner13/fmix64/babylm-stream-v2"
SCRIPT_PATH = Path(__file__).resolve()
T0 = time.time()


def log(msg: str) -> None:
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


# --------------------------------------------------------------------------- #
# Source specs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourceSpec:
    name: str
    rel_dir: str            # under --raw-root
    pattern: str            # glob inside rel_dir (flat; no recursion)
    text_col: str
    key_col: Optional[str]  # column used as doc_key and for the selection hash
    mode: str               # "hash" (pseudo-random order) | "score" (FineWeb rank)
    base_target: float      # words at --scale 1
    min_doc_words: int
    batch_rows: int
    url_col: Optional[str] = None   # FineWeb: url column used for the news-domain exclusion


SOURCES: List[SourceSpec] = [   # also the fixed dedup priority order
    SourceSpec("ccnews", "ccnews_en_2016_2019", "*.parquet", "plain_text",
               "requested_url", "hash", 2.38e9, 50, 2000),
    SourceSpec("openwebtext", "openwebtext/plain_text", "*.parquet", "text",
               None, "hash", 1.19e9, 50, 2000),
    SourceSpec("fineweb", "fineweb_sample10BT_b/sample/10BT", "*.parquet", "text",
               "id", "score", 0.97e9, 50, 2000, url_col="url"),
    SourceSpec("wikipedia", "wikipedia_20231101_en/20231101.en", "*.parquet", "text",
               "url", "hash", 0.38e9, 50, 2000),
    SourceSpec("bookcorpusopen", "bookcorpusopen/data", "*.parquet", "text",
               "title", "hash", 0.12e9, 2000, 4),
]
SPEC_BY_NAME = {s.name: s for s in SOURCES}

# FineWeb narrative-register score.  Deliberately contains NO pronoun term: the
# study measures pronoun predictability, so selection must not use pronouns.
NARRATIVE_VERBS = ("said", "asked", "replied", "told", "looked", "felt", "walked",
                   "turned", "smiled", "whispered", "shouted", "thought", "knew",
                   "saw", "heard", "came", "went")
QUOTE_CHARS = ('"', "“", "”")
ED_MIN_LEN = 5
# News penalty (2026-10-14): FineWeb's quote/-ed/"said" signal alone selects news
# reporting, so digit-heavy text and wire-style vocabulary are subtracted.
NEWS_MARKERS = ("percent", "according", "reported", "officials", "statement", "spokesperson",
                "spokesman", "reuters", "associated", "press", "announced")
DEFAULT_SCORE_WEIGHTS = {"quote": 1.0, "verb": 1.0, "ed": 1.0, "digit": 1.0, "news": 1.0}
SCORE_WEIGHTS = dict(DEFAULT_SCORE_WEIGHTS)      # overridable with --fineweb-weights
FINEWEB_MIN_WORDS = 200
SCORE_FORMULA = (
    "score = (w_q*Q + w_v*V + w_e*E - w_d*D - w_n*M) / N  with w_q=%(quote)s, w_v=%(verb)s, "
    "w_e=%(ed)s, w_d=%(digit)s, w_n=%(news)s; N = len(text.split()); Q = count of double "
    "quotation marks (\" U+0022, U+201C, U+201D); V = count of lowercase [a-z]+ tokens in "
    "NARRATIVE_VERBS; E = count of lowercase [a-z]+ tokens of length >= %(edmin)d ending in "
    "'ed'; D = count of whitespace-split tokens containing a digit; M = count of lowercase "
    "[a-z]+ tokens in NEWS_MARKERS (%(markers)s). Docs with N < %(minw)d, and docs whose url "
    "registrable domain occurs in CC-News (>= %(minurls)s url), are ineligible. Selection = "
    "highest score first (ties: file order, row) until the word target is covered. No "
    "pronoun is counted anywhere in the score."
)


# --------------------------------------------------------------------------- #
# Hash helpers
# --------------------------------------------------------------------------- #
def hash64(s: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little")


def sha1_short(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _fmix64(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xC4CEB9FE1A85EC53)
    x ^= x >> np.uint64(33)
    return x


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
_NEWLINE_LIKE = re.compile("\r\n?|[\x0b\x0c\x1c\x1d\x1e\x85  ]")
_CTRL = re.compile("[\x00-\x08\x0e-\x1b\x1f\x7f-\x84\x86-\x9f]")
_SPACES = re.compile("[ \t\xa0  -   　]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    """NFC; line-separator-like chars -> '\\n'; other control chars removed;
    runs of spaces/tabs (and exotic unicode spaces) -> one space."""
    if not text:
        return ""
    if not text.isascii():
        text = unicodedata.normalize("NFC", text)
    text = _NEWLINE_LIKE.sub("\n", text)
    text = _CTRL.sub("", text)
    return _SPACES.sub(" ", text)


def _hard_split(s: str, max_chars: int) -> List[str]:
    pieces = []
    while len(s) > max_chars:
        cut = s.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars          # no whitespace at all: hard cut
        pieces.append(s[:cut].rstrip())
        s = s[cut:].lstrip()
    if s:
        pieces.append(s)
    return pieces


def split_long_paragraph(p: str, max_chars: int = 4000) -> List[str]:
    """Split at sentence-final punctuation (.!? + whitespace), packing sentences
    greedily into lines of <= max_chars; hard-split at whitespace as last resort."""
    if len(p) <= max_chars:
        return [p]
    out: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for sent in _SENT_SPLIT.split(p):
        for piece in (_hard_split(sent, max_chars) if len(sent) > max_chars else [sent]):
            add = len(piece) + (1 if cur else 0)
            if cur and cur_len + add > max_chars:
                out.append(" ".join(cur))
                cur, cur_len = [piece], len(piece)
            else:
                cur.append(piece)
                cur_len += add
    if cur:
        out.append(" ".join(cur))
    return out


def doc_to_lines(text: str, max_chars: int = 4000) -> List[str]:
    lines: List[str] = []
    for para in normalize_text(text).split("\n"):
        para = para.strip()
        if para:
            lines.extend(split_long_paragraph(para, max_chars))
    return lines


def count_words(line: str) -> int:
    return len(line.split())


# --------------------------------------------------------------------------- #
# 13-gram contamination filter
# --------------------------------------------------------------------------- #
NGRAM = 13
_WORD_RE = re.compile(r"\w+")
_P = np.uint64(0x9E3779B97F4A7C15)
_TOK_CACHE: Dict[str, int] = {}
_TOK_CACHE_MAX = 2_500_000
_EMPTY_U64 = np.zeros(0, dtype=np.uint64)
_EMPTY_I64 = np.zeros(0, dtype=np.int64)


def _tok_hash(t: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(t.encode("utf-8", "replace"), digest_size=8).digest(), "little")


def _hash_tokens(flat: List[str]) -> np.ndarray:
    cache = _TOK_CACHE
    if len(cache) > _TOK_CACHE_MAX:
        cache.clear()
    for t in set(flat):
        if t not in cache:
            cache[t] = _tok_hash(t)
    return np.array([cache[t] for t in flat], dtype=np.uint64)


def _window_hashes(H: np.ndarray, n: int) -> np.ndarray:
    """Hash of every n-token window of the token-hash array H (len(H) >= n)."""
    W = len(H) - n + 1
    acc = H[:W].copy()
    for k in range(1, n):
        acc *= _P
        acc += H[k:k + W]
    return _fmix64(acc)


def line_ngram_hashes(lines: Sequence[str], n: int = NGRAM) -> Tuple[np.ndarray, np.ndarray]:
    """Hash of every n-gram of lowercase ``\\w+`` tokens that lies within one line.
    Returns (hashes uint64, line_index int64).  Lines with < n tokens give none."""
    toks = [_WORD_RE.findall(l.lower()) for l in lines]
    lens = np.fromiter((len(t) for t in toks), dtype=np.int64, count=len(toks))
    total = int(lens.sum())
    if total < n or int(lens.max(initial=0)) < n:
        return _EMPTY_U64, _EMPTY_I64
    H = _hash_tokens(list(itertools.chain.from_iterable(toks)))
    W = total - n + 1
    acc = _window_hashes(H, n)
    ends = np.cumsum(lens)
    line_of = np.repeat(np.arange(len(toks), dtype=np.int64), lens)[:W]
    valid = (np.arange(W, dtype=np.int64) + n) <= ends[line_of]   # window stays in its line
    return acc[valid], line_of[valid]


class NgramFilter:
    """Membership test on a sorted unique uint64 array, with a byte-table prefilter
    (one byte per slot, ~10% fill at BabyLM scale) so that ``searchsorted`` only
    runs on the few percent of queries that pass the prefilter."""

    def __init__(self, sorted_hashes: np.ndarray, ngram: int = NGRAM):
        self.h = sorted_hashes
        self.ngram = ngram
        self.n = int(len(sorted_hashes))
        k = min(31, max(12, int(math.ceil(math.log2(max(1, self.n) * 8)))))
        self.mask = np.uint64((1 << k) - 1)
        self.table = np.zeros(1 << k, dtype=np.uint8)
        for i in range(0, self.n, 1 << 24):
            self.table[(sorted_hashes[i:i + (1 << 24)] & self.mask).astype(np.intp)] = 1

    def contains(self, q: np.ndarray) -> np.ndarray:
        out = np.zeros(len(q), dtype=bool)
        if self.n == 0 or len(q) == 0:
            return out
        cand = np.flatnonzero(self.table[(q & self.mask).astype(np.intp)])
        if len(cand):
            qc = q[cand]
            pos = np.searchsorted(self.h, qc)
            pos[pos == self.n] = self.n - 1
            out[cand] = self.h[pos] == qc
        return out


def contaminated_line_mask(lines: Sequence[str], flt: Optional[NgramFilter]) -> np.ndarray:
    mask = np.zeros(len(lines), dtype=bool)
    if flt is None or flt.n == 0:
        return mask
    h, li = line_ngram_hashes(lines, flt.ngram)
    if len(h):
        hit = flt.contains(h)
        if hit.any():
            mask[li[hit]] = True
    return mask


_FILTER: Optional[NgramFilter] = None   # set in the parent before forking workers
_CANCEL = None                          # multiprocessing.Event inherited by forked workers (early stop)


# ---- BabyLM hash-set construction ----------------------------------------- #
BABYLM_SUFFIXES = (".train", ".test", ".dev")


def list_babylm_files(dirs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for d in dirs:
        dp = Path(d)
        if not dp.is_dir():
            raise SystemExit("BabyLM dir not found: %s" % d)
        found = sorted(p for p in dp.iterdir() if p.is_file() and p.suffix in BABYLM_SUFFIXES)
        if not found:
            raise SystemExit("no %s files in BabyLM dir %s" % ("/".join(BABYLM_SUFFIXES), d))
        files.extend(found)
    return files


_SPEAKER_TAG = re.compile(r"^\*[A-Z0-9]+:\s*")      # CHILDES-style "*MOT:\t"
_AT_HEADER = re.compile(r"^\s*@[A-Za-z]")             # "@Begin", "@Languages:" ...


def babylm_line_tokens(line: str) -> Optional[List[str]]:
    """Lowercase ``\\w+`` tokens of one BabyLM line, speaker tag stripped.  None for
    lines that are only an @-header or a ``= = = ... = = =`` boundary marker."""
    if _AT_HEADER.match(line):
        return None
    st = line.strip()
    if st.startswith("= =") and st.endswith("= ="):
        return None
    return _WORD_RE.findall(_SPEAKER_TAG.sub("", line).lower())


def _babylm_file_hashes(args) -> dict:
    """13-gram hashes of one BabyLM file: windows over the file's concatenated token
    stream (so they also run across consecutive lines).  Every within-line window is
    also a stream window, so the per-line set is a subset of the stream set."""
    path, out_path, ngram = args
    pa.set_cpu_count(1)
    parts: List[np.ndarray] = []
    n_lines = n_words = n_tokens = 0
    buf: List[List[str]] = []
    carry = _EMPTY_U64

    def flush():
        nonlocal carry, n_tokens
        H = _hash_tokens(list(itertools.chain.from_iterable(buf)))
        n_tokens += len(H)
        if len(carry):
            H = np.concatenate([carry, H])
        if len(H) >= ngram:
            parts.append(np.unique(_window_hashes(H, ngram)))
        carry = H[-(ngram - 1):].copy() if ngram > 1 else _EMPTY_U64
        buf.clear()

    with open(path, encoding="utf-8", errors="replace", newline="\n") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.isascii():
                line = unicodedata.normalize("NFC", line)
            n_lines += 1
            n_words += len(line.split())
            toks = babylm_line_tokens(line)
            if toks:
                buf.append(toks)
            if n_lines % 20000 == 0:
                flush()
    flush()
    h = np.unique(np.concatenate(parts)) if parts else _EMPTY_U64
    np.save(out_path, h)
    return {"file": str(path), "lines": n_lines, "words": n_words, "tokens": n_tokens,
            "ngrams_unique": int(len(h))}


def build_babylm_hashes(dirs: Sequence[str], work: Path, ngram: int, workers: int) -> Tuple[np.ndarray, dict]:
    files = list_babylm_files(dirs)
    fp = sha1_short([HASH_VERSION, ngram] + [[str(p), p.stat().st_size] for p in files])
    arr_path, meta_path = work / "babylm_13gram_sorted.npy", work / "babylm_13gram_meta.json"
    if arr_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp:
            log("BabyLM hash set: cached (%d unique %d-grams)" % (meta["ngrams_unique"], ngram))
            return np.load(arr_path), meta
    t = time.time()
    chunk_dir = work / "babylm_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(chunk_dir / ("%03d.npy" % i)), ngram) for i, p in enumerate(files)]
    log("BabyLM: hashing %d files" % len(files))
    if workers > 1:
        with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            stats = pool.map(_babylm_file_hashes, tasks, chunksize=1)
    else:
        stats = [_babylm_file_hashes(a) for a in tasks]
    allh = np.concatenate([np.load(a[1]) for a in tasks])
    allh = np.unique(allh)
    np.save(str(arr_path) + ".tmp.npy", allh)
    os.replace(str(arr_path) + ".tmp.npy", arr_path)
    meta = {"fingerprint": fp, "hash_version": HASH_VERSION, "ngram": ngram,
            "tokenization": "lowercase \\w+",
            "ngram_windows": "per file: sliding window over the concatenated token stream (crosses line "
                             "boundaries); CHILDES speaker tags (^\\*[A-Z0-9]+:\\s*) stripped; @-header and "
                             "'= = =' boundary lines skipped (no stream break); includes all within-line windows",
            "files": stats, "tokens": sum(s["tokens"] for s in stats),
            "lines": sum(s["lines"] for s in stats), "words": sum(s["words"] for s in stats),
            "ngrams_unique": int(len(allh)), "seconds": round(time.time() - t, 1)}
    meta_path.write_text(json.dumps(meta, indent=1))
    for a in tasks:
        os.remove(a[1])
    log("BabyLM: %d lines, %d words, %d unique %d-grams (%.0fs)" %
        (meta["lines"], meta["words"], meta["ngrams_unique"], ngram, meta["seconds"]))
    return allh, meta


# --------------------------------------------------------------------------- #
# FineWeb narrative score
# --------------------------------------------------------------------------- #
_ALPHA_RE = re.compile(r"[a-z]+")
_VERB_SET = frozenset(NARRATIVE_VERBS)


_DIGIT_SEARCH = re.compile(r"\d").search
_NEWS_SET = frozenset(NEWS_MARKERS)


def narrative_features(text: str) -> Tuple[int, int, int, int, int, int]:
    """(N words, Q quote marks, V narrative verbs, E past-tense -ed words,
    D digit-containing tokens, M news-marker words)."""
    toks = text.split()
    n = len(toks)
    q = sum(text.count(c) for c in QUOTE_CHARS)
    d = sum(1 for _ in filter(None, map(_DIGIT_SEARCH, toks)))
    counts = Counter(_ALPHA_RE.findall(text.lower()))
    v = sum(c for w, c in counts.items() if w in _VERB_SET)
    e = sum(c for w, c in counts.items() if len(w) >= ED_MIN_LEN and w.endswith("ed"))
    m = sum(c for w, c in counts.items() if w in _NEWS_SET)
    return n, q, v, e, d, m


def narrative_score(text: str) -> Tuple[float, int]:
    n, q, v, e, d, m = narrative_features(text)
    if n == 0:
        return 0.0, 0
    w = SCORE_WEIGHTS
    return (w["quote"] * q + w["verb"] * v + w["ed"] * e - w["digit"] * d - w["news"] * m) / n, n


# --------------------------------------------------------------------------- #
# News-domain set (from CC-News urls) used to exclude news sites from FineWeb
# --------------------------------------------------------------------------- #
DOMAIN_VERSION = "domains-v1"
TWO_LABEL_SUFFIXES = frozenset("""
co.uk org.uk ac.uk gov.uk me.uk ltd.uk plc.uk net.uk sch.uk nhs.uk police.uk
com.au net.au org.au edu.au gov.au asn.au id.au
co.nz org.nz net.nz govt.nz ac.nz school.nz geek.nz
co.za org.za ac.za gov.za net.za web.za
co.in net.in org.in ac.in gov.in edu.in res.in firm.in gen.in ind.in
com.br net.br org.br gov.br edu.br
com.mx org.mx gob.mx edu.mx net.mx com.ar org.ar gob.ar com.co org.co com.pe com.ve com.ec com.uy com.py com.bo com.cl
com.cn net.cn org.cn gov.cn edu.cn com.hk org.hk edu.hk gov.hk com.tw org.tw edu.tw gov.tw
com.sg edu.sg org.sg gov.sg com.my org.my edu.my gov.my com.ph org.ph gov.ph edu.ph
co.id or.id ac.id go.id web.id co.th in.th ac.th or.th go.th com.vn org.vn gov.vn edu.vn
co.jp or.jp ne.jp ac.jp go.jp ad.jp co.kr or.kr ne.kr ac.kr go.kr re.kr
com.tr org.tr net.tr gov.tr edu.tr co.il org.il ac.il gov.il net.il muni.il
com.eg org.eg gov.eg edu.eg com.ng org.ng gov.ng edu.ng co.ke or.ke ac.ke go.ke co.tz co.ug co.zw co.zm co.bw
com.pk org.pk gov.pk edu.pk com.bd org.bd gov.bd edu.bd com.lk org.lk com.np org.np
com.ua org.ua gov.ua in.ua com.pl org.pl net.pl edu.pl gov.pl com.pt org.pt gov.pt com.es org.es gob.es
com.gr org.gr gov.gr edu.gr com.cy org.cy com.mt org.mt com.ro org.ro com.hr com.rs com.ba com.mk
com.sa org.sa gov.sa edu.sa com.qa org.qa gov.qa com.kw org.kw com.lb org.lb com.jo org.jo com.om org.om com.bh
co.ae com.ae net.ae org.ae gov.ae ac.ae co.ir ac.ir org.ir
com.gh org.gh gov.gh com.et com.na com.mu org.mu co.mz co.ma com.ma co.ls
com.fj com.pg com.jm com.tt com.bb com.bs com.do com.gt com.hn com.ni com.pa com.sv com.cu com.pr
com.mo com.kh com.mm com.af com.ye com.iq com.ly com.sd com.tn com.dz
""".split())


def registrable_domain(url: str) -> str:
    """Lowercase netloc, ``www.`` stripped, last two labels (three for the common
    two-label suffixes above, e.g. ``bbc.co.uk``).  Returns '' if there is no host."""
    if not url:
        return ""
    s = url.strip()
    i = s.find("://")
    if i >= 0:
        s = s[i + 3:]
    for sep in "/?#\\":
        j = s.find(sep)
        if j >= 0:
            s = s[:j]
    host = s.rsplit("@", 1)[-1]
    if host.startswith("["):                       # IPv6 literal
        return host.split("]")[0].lower() + "]"
    host = host.split(":", 1)[0].strip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    labels = [l for l in host.split(".") if l]
    if len(labels) <= 2:
        return ".".join(labels)
    if all(l.isdigit() for l in labels):           # bare IPv4
        return ".".join(labels)
    if ".".join(labels[-2:]) in TWO_LABEL_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


_NEWS_HASHES: Optional[np.ndarray] = None       # sorted uint64 hashes of news domains (parent, pre-fork)


def _url_domains_file(args) -> dict:
    path = args
    pa.set_cpu_count(1)
    counts: Counter = Counter()
    n = 0
    for _, b in iter_batches(path, ["requested_url"], 50000):
        for u in b.column("requested_url").to_pylist():
            n += 1
            d = registrable_domain(u) if u else ""
            if d:
                counts[d] += 1
    return {"file": Path(path).name, "urls": n, "counts": dict(counts)}


def build_news_domains(raw_root: Path, work: Path, workers: int, min_urls: int) -> Tuple[np.ndarray, dict]:
    """One pass over the ``requested_url`` column only of every CC-News file."""
    files, _, failed = probe_files(list_source_files(raw_root, SPEC_BY_NAME["ccnews"]))
    fp = sha1_short([DOMAIN_VERSION, min_urls] + [[p.name, p.stat().st_size] for p in files])
    npz, meta_path = work / "news_domains.npz", work / "news_domains_meta.json"
    if npz.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp:
            log("news domains: cached (%d domains used of %d)" % (meta["domains_used"], meta["domains_total"]))
            return np.load(npz)["hashes"], meta
    t = time.time()
    tasks = [str(p) for p in files]
    log("news domains: reading requested_url from %d CC-News files" % len(tasks))
    if workers > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            parts = pool.map(_url_domains_file, tasks, chunksize=1)
    else:
        parts = [_url_domains_file(a) for a in tasks]
    total: Counter = Counter()
    for r in parts:
        total.update(r["counts"])
    used = [d for d, c in total.items() if c >= min_urls]
    hashes = np.unique(np.array([hash64(d) for d in used], dtype=np.uint64)) if used else _EMPTY_U64
    with open(str(npz) + ".tmp", "wb") as f:
        np.savez(f, hashes=hashes)
    os.replace(str(npz) + ".tmp", npz)
    with gzip.open(work / "news_domains.tsv.gz", "wt", encoding="utf-8") as g:
        for d, c in total.most_common():
            g.write("%s\t%d\n" % (d, c))
    meta = {"enabled": True, "fingerprint": fp, "version": DOMAIN_VERSION, "ccnews_files_scanned": len(files),
            "ccnews_files_unreadable": failed, "urls_scanned": int(sum(r["urls"] for r in parts)),
            "domains_total": len(total), "domains_used": len(used), "min_urls": min_urls,
            "top_domains": total.most_common(15), "seconds": round(time.time() - t, 1)}
    meta_path.write_text(json.dumps(meta, indent=1))
    log("news domains: %d urls -> %d domains (%d with >= %d urls) in %.0fs" % (
        meta["urls_scanned"], len(total), len(used), min_urls, meta["seconds"]))
    return hashes, meta


# --------------------------------------------------------------------------- #
# Parquet helpers
# --------------------------------------------------------------------------- #
def list_source_files(raw_root: Path, spec: SourceSpec) -> List[Path]:
    d = raw_root / spec.rel_dir
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob(spec.pattern)):
        if not p.is_file() or ".cache" in p.parts or p.name.endswith((".tmp", ".incomplete")):
            continue
        out.append(p)
    return out


def iter_batches(path: str, columns: List[str], batch_rows: int, limit: Optional[int] = None):
    """Yield (first_row_index, RecordBatch).  buffer_size streams column chunks, so
    a single 3 GB row group (CC-News) costs ~300 MB, not ~4 GB, per worker."""
    pf = pq.ParquetFile(path, buffer_size=1 << 20)
    row0 = 0
    for b in pf.iter_batches(batch_size=batch_rows, columns=columns, use_threads=False):
        if limit is not None and row0 >= limit:
            return
        if limit is not None and row0 + b.num_rows > limit:
            b = b.slice(0, limit - row0)
        yield row0, b
        row0 += b.num_rows


def _col(batch, name) -> list:
    return batch.column(name).to_pylist()


def _take(batch, name, idx: np.ndarray) -> list:
    return batch.column(name).take(pa.array(idx, type=pa.int64())).to_pylist()


def _unit_frac(us: np.ndarray) -> np.ndarray:
    return (us >> np.uint64(11)).astype(np.float64) / 9007199254740992.0   # 2**53


def _atomic_write(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Stage 1a: FineWeb scoring pass
# --------------------------------------------------------------------------- #
def score_unit(unit: dict) -> dict:
    pa.set_cpu_count(1)
    out = unit["out"]
    side = out + ".json"
    if os.path.exists(out) and os.path.exists(side):
        st = json.load(open(side))
        st["cached"] = True
        return st
    try:
        news = _NEWS_HASHES
        url_col = unit.get("url_col")
        cols = [unit["text_col"]] + ([url_col] if url_col else [])
        scores, words, excl = [], [], []
        top_excl: Counter = Counter()
        for _, b in iter_batches(unit["path"], cols, 2000, unit.get("limit")):
            texts = _col(b, unit["text_col"])
            if url_col and news is not None and len(news):
                doms = [registrable_domain(u) if u else "" for u in _col(b, url_col)]
                dh = np.array([hash64(d) if d else 0 for d in doms], dtype=np.uint64)
                pos = np.searchsorted(news, dh)
                pos[pos == len(news)] = len(news) - 1
                is_news = (news[pos] == dh) & (dh != 0)
                top_excl.update(d for d, e in zip(doms, is_news) if e)
            else:
                is_news = np.zeros(len(texts), dtype=bool)
            for t, ex in zip(texts, is_news):
                s, n = narrative_score(t or "")
                scores.append(s if (n >= FINEWEB_MIN_WORDS and not ex) else float("-inf"))
                words.append(n)
                excl.append(bool(ex))
        tmp = out + ".tmp"
        w = np.asarray(words, dtype=np.int32)
        e = np.asarray(excl, dtype=bool)
        with open(tmp, "wb") as f:
            np.savez(f, score=np.asarray(scores, dtype=np.float64), words=w, excluded=e)
        os.replace(tmp, out)
        st = {"file_id": unit["file_id"], "rows": len(scores), "excluded": int(e.sum()),
              "excluded_eligible_length": int((e & (w >= FINEWEB_MIN_WORDS)).sum()),
              "top_excluded": top_excl.most_common(40)}
        _atomic_write(side, json.dumps(st).encode())
        st["cached"] = False
        return st
    except Exception as e:  # noqa: BLE001 - reported, file skipped
        return {"file_id": unit["file_id"], "rows": 0, "error": "%s: %s" % (type(e).__name__, e)}


# --------------------------------------------------------------------------- #
# Stage 1b: extraction unit (normalize / length / contamination / hash)
# --------------------------------------------------------------------------- #
STAT_KEYS = ["docs_read", "docs_candidates", "docs_dropped_length",
             "docs_dropped_length_post_contam", "docs_written", "words_candidates",
             "words_written", "lines_written", "contam_lines_checked", "contam_lines_dropped",
             "contam_words_checked", "contam_words_dropped", "contam_docs_affected"]


def _unit_paths(prefix: str) -> Tuple[str, str, str]:
    return prefix + ".jsonl", prefix + ".idx.npz", prefix + ".stats.json"


class _Cancelled(Exception):
    """Raised inside a worker when the parent has already covered the word goal."""


def extract_unit(unit: dict) -> dict:
    jsonl, idxp, statp = _unit_paths(unit["prefix"])
    if os.path.exists(statp):
        st = json.load(open(statp))
        st["cached"] = True
        return st
    t0 = time.time()
    try:
        st = _extract_unit(unit, jsonl, idxp)
    except _Cancelled:
        for tmp in (jsonl + ".tmp", idxp + ".tmp"):
            if os.path.exists(tmp):
                os.remove(tmp)
        return {"file_id": unit["file_id"], "cancelled": True, "words_written": 0}
    except Exception as e:  # noqa: BLE001 - reported, file skipped
        return {"file_id": unit["file_id"], "error": "%s: %s" % (type(e).__name__, e),
                "words_written": 0, **{k: 0 for k in STAT_KEYS if k != "words_written"}}
    st["seconds"] = round(time.time() - t0, 2)
    st["file_id"] = unit["file_id"]
    _atomic_write(statp, json.dumps(st).encode())
    st["cached"] = False
    return st


def _extract_unit(unit: dict, jsonl: str, idxp: str) -> dict:
    pa.set_cpu_count(1)
    spec = SPEC_BY_NAME[unit["source"]]
    seed, flt = unit["seed"], _FILTER
    max_chars, limit = unit["max_line_chars"], unit.get("limit")
    lo, hi = unit.get("lo", 0.0), unit.get("hi", 1.0)
    rows_sel = np.load(unit["rows_path"]) if unit.get("rows_path") else None
    fw_scores = np.load(unit["scores_path"])["score"] if unit.get("scores_path") else None
    cols = [spec.text_col] + ([spec.key_col] if spec.key_col else [])
    st = {k: 0 for k in STAT_KEYS}
    cols_out = {k: [] for k in ("offset", "nbytes", "words", "lines", "chash", "tie", "prio", "row")}
    off = 0
    with open(jsonl + ".tmp", "wb") as out:
        for row0, b in iter_batches(unit["path"], cols, spec.batch_rows, limit):
            if _CANCEL is not None and _CANCEL.is_set():
                raise _Cancelled()
            n = b.num_rows
            st["docs_read"] += n
            # ---- which rows of this batch are candidates -------------------- #
            if rows_sel is not None:                                   # FineWeb: explicit ranks
                a, z = np.searchsorted(rows_sel, [row0, row0 + n])
                local = (rows_sel[a:z] - row0).astype(np.int64)
                keys = _take(b, spec.key_col, local) if len(local) else []
                texts = _take(b, spec.text_col, local) if len(local) else []
                us = np.array([hash64("%d|%s|%s" % (seed, spec.name, k)) for k in keys], dtype=np.uint64)
                prio = -fw_scores[row0 + local]
            else:                                                      # hash band [lo, hi)
                if spec.key_col:
                    allkeys = _col(b, spec.key_col)
                    allk = [k if k is not None else "%s#%d" % (unit["file_id"], row0 + i)
                            for i, k in enumerate(allkeys)]
                    us_all = np.array([hash64("%d|%s|%s" % (seed, spec.name, k)) for k in allk], dtype=np.uint64)
                    uf = _unit_frac(us_all)
                    local = np.flatnonzero((uf >= lo) & (uf < hi))
                    keys = [allk[i] for i in local]
                    texts = _take(b, spec.text_col, local) if len(local) else []
                else:                                                  # OpenWebText: no id column
                    alltexts = _col(b, spec.text_col)
                    us_all = np.array([hash64("%d|%s|%s|%d" % (seed, spec.name, (t or "")[:4096], len(t or "")))
                                       for t in alltexts], dtype=np.uint64)
                    uf = _unit_frac(us_all)
                    local = np.flatnonzero((uf >= lo) & (uf < hi))
                    keys = ["%s#%d" % (unit["file_id"], row0 + i) for i in local]
                    texts = [alltexts[i] for i in local]
                us = us_all[local]
                prio = np.zeros(len(local), dtype=np.float64)
            st["docs_candidates"] += len(local)
            # ---- per-document processing ------------------------------------ #
            for j, text in enumerate(texts):
                lines = doc_to_lines(text or "", max_chars)
                wc = [count_words(l) for l in lines]
                nwords = sum(wc)
                st["words_candidates"] += nwords
                if nwords < spec.min_doc_words:
                    st["docs_dropped_length"] += 1
                    continue
                mask = contaminated_line_mask(lines, flt)
                st["contam_lines_checked"] += len(lines)
                st["contam_words_checked"] += nwords
                if mask.any():
                    dropped_w = int(sum(w for w, m in zip(wc, mask) if m))
                    st["contam_lines_dropped"] += int(mask.sum())
                    st["contam_words_dropped"] += dropped_w
                    st["contam_docs_affected"] += 1
                    lines = [l for l, m in zip(lines, mask) if not m]
                    nwords -= dropped_w
                    if nwords < spec.min_doc_words:
                        st["docs_dropped_length_post_contam"] += 1
                        continue
                body = "\n".join(lines) + "\n"
                hdr = json.dumps({"k": keys[j], "s": spec.name, "n": len(lines), "w": nwords})
                blob = (hdr + "\n" + body).encode("utf-8", "replace")
                out.write(blob)
                cols_out["offset"].append(off)
                cols_out["nbytes"].append(len(blob))
                cols_out["words"].append(nwords)
                cols_out["lines"].append(len(lines))
                cols_out["chash"].append(hash64(body))
                cols_out["tie"].append(int(us[j]))
                cols_out["prio"].append(float(prio[j]))
                cols_out["row"].append(row0 + int(local[j]))
                off += len(blob)
                st["docs_written"] += 1
                st["words_written"] += nwords
                st["lines_written"] += len(lines)
    os.replace(jsonl + ".tmp", jsonl)
    dt = {"offset": np.int64, "nbytes": np.int64, "words": np.int64, "lines": np.int64,
          "chash": np.uint64, "tie": np.uint64, "prio": np.float64, "row": np.int64}
    with open(idxp + ".tmp", "wb") as f:
        np.savez(f, **{k: np.asarray(v, dtype=dt[k]) for k, v in cols_out.items()})
    os.replace(idxp + ".tmp", idxp)
    return st


# --------------------------------------------------------------------------- #
# Unit planning
# --------------------------------------------------------------------------- #
def file_order(files: List[Path], seed: int, spec: SourceSpec) -> List[Path]:
    return sorted(files, key=lambda p: hash64("%d|%s|file|%s" % (seed, spec.name, p.name)))


def probe_files(files: List[Path]) -> Tuple[List[Path], Dict[Path, int], List[dict]]:
    """Read parquet footers; unreadable (truncated / partial) files are reported, not fatal."""
    good, rows, failed = [], {}, []
    for p in files:
        try:
            rows[p] = pq.read_metadata(str(p)).num_rows
            good.append(p)
        except Exception as e:  # noqa: BLE001
            failed.append({"file": p.name, "error": "%s: %s" % (type(e).__name__, e)})
    return good, rows, failed


def estimate_source(files: List[Path], rows_map: Dict[Path, int], spec: SourceSpec, limit: Optional[int],
                    sample_files: int = 8, sample_rows: int = 400) -> dict:
    """Rows from parquet footers; words/doc from the first rows of a few files."""
    rows = [min(rows_map[p], limit) if limit is not None else rows_map[p] for p in files]
    pick = sorted(set(int(round(i * (len(files) - 1) / max(1, sample_files - 1)))
                      for i in range(min(sample_files, len(files)))))
    wsum = dsum = 0
    for i in pick:
        for _, b in iter_batches(str(files[i]), [spec.text_col], min(sample_rows, 1000), sample_rows):
            for t in _col(b, spec.text_col):
                wsum += len((t or "").split())
                dsum += 1
    wpd = wsum / max(1, dsum)
    return {"total_rows": int(sum(rows)), "words_per_doc": wpd,
            "est_total_words": float(sum(rows)) * wpd,
            "est_words_per_file": float(sum(rows)) * wpd / max(1, len(files))}


def plan_hash_units(files: List[Path], est: dict, needed: float, spread: float, min_files: int) -> Tuple[List[Tuple[Path, float, float]], dict]:
    """(file, lo, hi) units in priority order.  Uses K files (hash order) with
    inclusion probability p0, then widens p over the same files, then adds the
    remaining files at p=1 -- only consumed if the target is not yet met."""
    wpf = max(1.0, est["est_words_per_file"])
    K = min(len(files), max(min_files, int(math.ceil(spread * needed / wpf))))
    p0 = min(1.0, needed / (K * wpf))
    ps, prev, units = [], 0.0, []
    for p in (p0, min(1.0, 2 * p0), 1.0):
        if p > prev:
            ps.append((prev, p))
            prev = p
    for lo, hi in ps:
        units.extend((f, lo, hi) for f in files[:K])
    units.extend((f, 0.0, 1.0) for f in files[K:])
    return units, {"files_used_first_pass": K, "p0": p0, "bands": ps}


def fineweb_bands(score_files: List[dict], needed: float) -> Tuple[List[dict], dict]:
    """Rank all eligible docs by score; return bands of per-file row arrays."""
    sc = np.concatenate([d["score"] for d in score_files]) if score_files else np.zeros(0)
    wd = np.concatenate([d["words"] for d in score_files]).astype(np.int64) if score_files else np.zeros(0, np.int64)
    fid = np.concatenate([np.full(len(d["score"]), i, dtype=np.int32) for i, d in enumerate(score_files)]) \
        if score_files else np.zeros(0, np.int32)
    row = np.concatenate([np.arange(len(d["score"]), dtype=np.int64) for d in score_files]) \
        if score_files else np.zeros(0, np.int64)
    ex = np.concatenate([d["excluded"] for d in score_files]) if score_files else np.zeros(0, bool)
    elig = np.flatnonzero((wd >= FINEWEB_MIN_WORDS) & ~ex)
    order = elig[np.lexsort((row[elig], fid[elig], -sc[elig]))]
    cum = np.cumsum(wd[order])
    cuts, prev = [], 0
    for g in (1.0, 1.5, 2.5):
        c = min(len(order), int(np.searchsorted(cum, needed * g, side="left")) + 1) if len(cum) else 0
        if c > prev:
            cuts.append((prev, c))
            prev = c
    if prev < len(order):
        cuts.append((prev, len(order)))
    bands = []
    for a, b in cuts:
        sel = order[a:b]
        per_file = {}
        for i in np.unique(fid[sel]):
            per_file[int(i)] = np.sort(row[sel][fid[sel] == i])
        bands.append(per_file)
    info = {"n_scored": int(len(sc)), "n_eligible": int(len(elig)),
            "n_excluded_by_domain": int(ex.sum()),
            "n_excluded_by_domain_eligible_length": int((ex & (wd >= FINEWEB_MIN_WORDS)).sum()),
            "eligible_words": int(wd[elig].sum()) if len(elig) else 0,
            "score_quantiles_eligible": ({q: float(np.quantile(sc[elig], q)) for q in (0.5, 0.9, 0.99)}
                                         if len(elig) else {}),
            "rank_bands": cuts, "needed_words": needed}
    return bands, info


# --------------------------------------------------------------------------- #
# Source driver: run units until the word goal is covered, then select
# --------------------------------------------------------------------------- #
def load_prefix(units: List[dict]) -> dict:
    """Concatenate the idx arrays of processed units (in unit order)."""
    cols = {k: [] for k in ("offset", "nbytes", "words", "lines", "chash", "tie", "prio", "row")}
    uno = []
    for i, u in enumerate(units):
        z = np.load(_unit_paths(u["prefix"])[1])
        for k in cols:
            cols[k].append(z[k])
        uno.append(np.full(len(z["words"]), i, dtype=np.int32))
    if not units:
        return {k: np.zeros(0) for k in list(cols) + ["unit"]}
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["unit"] = np.concatenate(uno)
    return out


def select_source(arr: dict, seen: np.ndarray, target: int) -> Tuple[np.ndarray, dict]:
    """Dedup (global, first occurrence in priority order) and trim to target.
    Returns indices into ``arr`` (in priority order) and counters."""
    n = len(arr["words"])
    order = np.lexsort((arr["tie"], arr["prio"])) if n else np.zeros(0, dtype=np.int64)
    ch = arr["chash"][order] if n else np.zeros(0, np.uint64)
    first = np.zeros(n, dtype=bool)
    if n:
        first[np.unique(ch, return_index=True)[1]] = True
    in_seen = np.isin(ch, seen) if (n and len(seen)) else np.zeros(n, dtype=bool)
    keep = first & ~in_seen
    kept_pos = np.flatnonzero(keep)
    cum = np.cumsum(arr["words"][order][kept_pos]) if len(kept_pos) else np.zeros(0, np.int64)
    if target <= 0 or not len(cum):
        n_take = 0
    else:
        n_take = min(len(cum), int(np.searchsorted(cum, target, side="left")) + 1)
    take = kept_pos[:n_take]
    info = {"docs_candidates_post_filters": int(n),
            "docs_dropped_dedup_within_source": int((~first).sum()),
            "docs_dropped_dedup_cross_source": int((first & in_seen).sum()),
            "docs_dropped_dedup": int((~keep).sum()),
            "docs_trimmed_over_target": int(len(kept_pos) - n_take),
            "docs_kept": int(n_take),
            "words_kept": int(cum[n_take - 1]) if n_take else 0}
    return order[take], info


def run_units_until(units: List[dict], results: List[Optional[dict]], pending: Dict[int, object],
                    start: int, goal: float, cum: float, checkpoints: Optional[set], pool,
                    workers: int, tag: str) -> Tuple[int, float]:
    """Process units in order until cumulative words_written >= goal (checked only
    at ``checkpoints`` if given).  Deterministic: the returned prefix length does
    not depend on scheduling.  ``pending`` holds in-flight AsyncResults across calls
    so a unit is never started twice.  Returns (prefix_len, cumulative_words)."""
    window = max(1, workers + 2)
    nxt, pos = start, start
    while pos < len(units):
        while pool is not None and nxt < len(units) and nxt - pos < window:
            if results[nxt] is None and nxt not in pending:
                if os.path.exists(_unit_paths(units[nxt]["prefix"])[2]):
                    results[nxt] = extract_unit(units[nxt])        # cached: just loads stats
                else:
                    pending[nxt] = pool.apply_async(extract_unit, (units[nxt],))
            nxt += 1
        if results[pos] is None:
            results[pos] = pending.pop(pos).get() if pos in pending else extract_unit(units[pos])
        r = results[pos]
        if r.get("error"):
            log("[%s] unit %s FAILED: %s" % (tag, r.get("file_id"), r["error"]))
        cum += r.get("words_written", 0)
        pos += 1
        log("[%s] unit %d/%d %s%s read=%d wrote=%d docs / %d words (cum %.3gM of goal %.3gM)" % (
            tag, pos, len(units), r.get("file_id"), " (cached)" if r.get("cached") else "",
            r.get("docs_read", 0), r.get("docs_written", 0), r.get("words_written", 0), cum / 1e6, goal / 1e6))
        if cum >= goal and (checkpoints is None or pos in checkpoints):
            break
    return pos, cum


def process_source(spec: SourceSpec, cfg: dict, work: Path, seen: np.ndarray, pool) -> dict:
    """Run stage 1 for one source and return its selection + stats."""
    t0 = time.time()
    listed = list_source_files(Path(cfg["raw_root"]), spec)
    files, rows_map, failed = probe_files(listed)
    limit = cfg["dry_run_limit"]
    target = int(cfg["targets"][spec.name])
    res = {"name": spec.name, "target_words": target, "files_available": len(listed), "files_failed": list(failed)}
    for f in failed:
        log("[%s] unreadable parquet skipped: %s (%s)" % (spec.name, f["file"], f["error"]))
    if not files:
        log("[%s] NO FILES under %s -- source empty" % (spec.name, Path(cfg["raw_root"]) / spec.rel_dir))
        res.update(arr=None, sel=np.zeros(0, dtype=np.int64), units=[], stats={k: 0 for k in STAT_KEYS},
                   select_info=select_source(load_prefix([]), seen, 0)[1], plan={"note": "no files"})
        return res
    if limit is not None:
        files = files[:1]
    udir = work / spec.name / "units"
    udir.mkdir(parents=True, exist_ok=True)
    needed = target * (1.0 + cfg["select_margin"])
    goal = target * (1.0 + cfg["dedup_slack"])
    common = {"source": spec.name, "seed": cfg["seed"], "max_line_chars": cfg["max_line_chars"],
              "limit": limit}
    phash_base = [NORM_VERSION, HASH_VERSION, cfg["seed"], spec.name, cfg["max_line_chars"],
                  spec.min_doc_words, limit, cfg["babylm_fp"], spec.text_col, spec.key_col]

    def mk_unit(path: Path, tag: str, extra: dict, extra_hash) -> dict:
        fid = re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem)
        ph = sha1_short(phash_base + [str(path), path.stat().st_size, extra_hash])
        return {**common, **extra, "path": str(path), "file_id": fid,
                "prefix": str(udir / ("%s__%s__%s" % (fid, tag, ph)))}

    plan: dict = {}
    if spec.mode == "score":
        sdir = work / spec.name / "scores"
        sdir.mkdir(parents=True, exist_ok=True)
        ordered = files    # sorted by name; ranking is global so order only fixes tie-breaks
        stasks = [{"path": str(p), "text_col": spec.text_col, "url_col": spec.url_col, "limit": limit,
                   "file_id": p.stem,
                   "out": str(sdir / ("%s__%s.npz" % (re.sub(r"[^A-Za-z0-9_.-]", "_", p.stem),
                                                      sha1_short([NORM_VERSION, SCORE_WEIGHTS, ED_MIN_LEN,
                                                                  FINEWEB_MIN_WORDS, limit, p.stat().st_size,
                                                                  cfg["news_fp"], spec.url_col]))))}
                  for p in ordered]
        log("[%s] scoring pass over %d files" % (spec.name, len(stasks)))
        sres = pool.map(score_unit, stasks, chunksize=1) if pool else [score_unit(t) for t in stasks]
        bad = [r for r in sres if r.get("error")]
        for r in bad:
            log("[%s] score FAILED %s: %s" % (spec.name, r["file_id"], r["error"]))
            res["files_failed"].append({"file": r["file_id"], "error": r["error"]})
        good = [(t, r) for t, r in zip(stasks, sres) if not r.get("error")]
        zs = []
        for t, _ in good:
            z = np.load(t["out"])
            zs.append({"score": z["score"], "words": z["words"], "excluded": z["excluded"]})
        bands, finfo = fineweb_bands(zs, needed)
        gfiles = [ordered[i] for i, (t, r) in enumerate(zip(stasks, sres)) if not r.get("error")]
        rdir = work / spec.name / "rows"
        rdir.mkdir(parents=True, exist_ok=True)
        units, checkpoints = [], set()
        for b, per_file in enumerate(bands):
            for fi in sorted(per_file):
                rows = per_file[fi]
                rp = rdir / ("%s__b%d__%s.npy" % (re.sub(r"[^A-Za-z0-9_.-]", "_", gfiles[fi].stem), b,
                                                  hashlib.sha1(rows.tobytes()).hexdigest()[:10]))
                np.save(str(rp), rows)
                units.append(mk_unit(gfiles[fi], "b%d" % b,
                                     {"rows_path": str(rp), "scores_path": good[fi][0]["out"]},
                                     hashlib.sha1(rows.tobytes()).hexdigest()))
            checkpoints.add(len(units))
        plan = {"mode": "score", **finfo}
        top_ex: Counter = Counter()
        for r in sres:
            top_ex.update(dict(r.get("top_excluded", [])))
        res["news_filter"] = {
            **cfg["news_meta"],
            "docs_excluded_by_domain": int(sum(r.get("excluded", 0) for r in sres)),
            "docs_excluded_by_domain_eligible_length": int(sum(r.get("excluded_eligible_length", 0) for r in sres)),
            "docs_scored": int(sum(r.get("rows", 0) for r in sres)),
            "top_excluded_domains": top_ex.most_common(25)}
        res["docs_read_scoring_pass"] = int(sum(r.get("rows", 0) for r in sres))
    else:
        ordered = file_order(files, cfg["seed"], spec)
        est = estimate_source(ordered, rows_map, spec, limit)
        hunits, pinfo = plan_hash_units(ordered, est, needed, cfg["spread"], cfg["min_files"])
        units = [mk_unit(f, "p%.4f-%.4f" % (lo, hi), {"lo": lo, "hi": hi}, [lo, hi]) for f, lo, hi in hunits]
        checkpoints = None
        plan = {"mode": "hash", **pinfo, **est, "needed_words": needed}
        log("[%s] est %.3gB words in %d files (%.0f words/doc); first pass: %d files at p=%.4f" % (
            spec.name, est["est_total_words"] / 1e9, len(ordered), est["words_per_doc"],
            pinfo["files_used_first_pass"], pinfo["p0"]))

    results: List[Optional[dict]] = [None] * len(units)
    pending: Dict[int, object] = {}
    pos, cum = 0, 0.0
    while True:
        pos, cum = run_units_until(units, results, pending, pos, goal, cum, checkpoints, pool,
                                   cfg["workers"], spec.name)
        arr = load_prefix(units[:pos])
        sel, info = select_source(arr, seen, target)
        if info["words_kept"] >= target or pos >= len(units):
            break
        goal = cum + (target - info["words_kept"]) * 1.5 + 1
        log("[%s] short by %d words after dedup/trim -> widening (goal %.3gM)" % (
            spec.name, target - info["words_kept"], goal / 1e6))
    if pending:                              # look-ahead units beyond the prefix: stop them now
        if _CANCEL is not None:
            _CANCEL.set()
        for fut in list(pending.values()):
            fut.wait()
        if _CANCEL is not None:
            _CANCEL.clear()
        pending.clear()
    done = [r for r in results[:pos] if r is not None]
    agg = {k: int(sum(r.get(k, 0) for r in done)) for k in STAT_KEYS}
    for u, r in zip(units, results):
        if r is not None and r.get("error"):
            res["files_failed"].append({"file": u["file_id"], "error": r["error"]})
    res.update(arr=arr, sel=sel, units=units[:pos], stats=agg, select_info=info, plan=plan,
               files_processed=len({u["file_id"] for u in units[:pos]}),
               seconds=round(time.time() - t0, 1))
    log("[%s] kept %d docs / %.4gM words (target %.4gM) in %.0fs; contamination dropped %d lines / %d words of %d words checked" % (
        spec.name, info["docs_kept"], info["words_kept"] / 1e6, target / 1e6, res["seconds"],
        agg["contam_lines_dropped"], agg["contam_words_dropped"], agg["contam_words_checked"]))
    return res


# --------------------------------------------------------------------------- #
# Stage 3: shards
# --------------------------------------------------------------------------- #
def _pread_full(fd: int, n: int, off: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        b = os.pread(fd, n - got, off + got)
        if not b:
            raise IOError("short read")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def write_shard(task: dict) -> dict:
    d = Path(task["dir"])
    name = task["name"]
    train, docs, marker = d / (name + ".train"), d / "docs.jsonl.gz", d / ".plan_hash"
    if train.exists() and docs.exists() and marker.exists() and marker.read_text() == task["plan_hash"]:
        return {"name": name, "cached": True}
    d.mkdir(parents=True, exist_ok=True)
    paths, gi, off, nb = task["paths"], task["gi"], task["off"], task["nbytes"]
    blobs: List[Optional[bytes]] = [None] * len(gi)
    fds: Dict[int, int] = {}
    try:
        for j in np.lexsort((off, gi)):      # read in (file, offset) order for locality
            g = int(gi[j])
            if g not in fds:
                fds[g] = os.open(paths[g], os.O_RDONLY)
            blobs[j] = _pread_full(fds[g], int(nb[j]), int(off[j]))
    finally:
        for fd in fds.values():
            os.close(fd)
    line_no = 0
    with open(str(train) + ".tmp", "wb") as ft, gzip.open(str(docs) + ".tmp", "wt", encoding="utf-8", compresslevel=6) as fd_:
        for blob in blobs:
            hdr, _, body = blob.partition(b"\n")
            h = json.loads(hdr)
            assert body.count(b"\n") == h["n"], "corrupt record"
            ft.write(body)
            ft.write(b"\n")                  # blank line = document boundary
            fd_.write(json.dumps({"source": h["s"], "doc_key": h["k"], "first_line": line_no,
                                  "n_lines": h["n"], "n_words": h["w"]}) + "\n")
            line_no += h["n"] + 1
    os.replace(str(docs) + ".tmp", docs)
    os.replace(str(train) + ".tmp", train)
    marker.write_text(task["plan_hash"])
    return {"name": name, "cached": False}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def code_version() -> dict:
    sha = os.environ.get("GIT_SHA")
    if not sha:
        for cand in (SCRIPT_PATH.parents[2] / ".git_sha",):
            if cand.exists():
                sha = cand.read_text().strip()
    return {"git_sha": sha or "unknown",
            "script_sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
            "norm_version": NORM_VERSION, "hash_version": HASH_VERSION,
            "python": sys.version.split()[0], "numpy": np.__version__, "pyarrow": pa.__version__}


def parse_targets(scale: float, overrides: Sequence[str]) -> Dict[str, int]:
    base = {s.name: s.base_target for s in SOURCES}
    for o in overrides or []:
        k, _, v = o.partition("=")
        if k not in base:
            raise SystemExit("--target: unknown source %r (have %s)" % (k, ", ".join(base)))
        base[k] = float(v)
    return {k: int(round(v * scale)) for k, v in base.items()}


def parse_weights(spec: Optional[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in (spec or "").split(","):
        if part.strip():
            k, _, v = part.partition("=")
            if k.strip() not in DEFAULT_SCORE_WEIGHTS:
                raise SystemExit("--fineweb-weights: unknown term %r (have %s)" % (k, ", ".join(DEFAULT_SCORE_WEIGHTS)))
            out[k.strip()] = float(v)
    return out


def build(args) -> dict:
    global _FILTER, _CANCEL, _NEWS_HASHES
    t_start = time.time()
    out = Path(args.out)
    work = Path(args.work_dir) if args.work_dir else out.parent / "_work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)
    targets = parse_targets(args.scale, args.target)
    SCORE_WEIGHTS.clear()
    SCORE_WEIGHTS.update(DEFAULT_SCORE_WEIGHTS)
    SCORE_WEIGHTS.update(parse_weights(args.fineweb_weights))
    log("build_option_b: out=%s work=%s workers=%d scale=%g shards=%d seed=%d dry_run_limit=%s" % (
        out, work, workers, args.scale, args.shards, args.seed, args.dry_run_limit))
    log("targets (words): " + ", ".join("%s=%.4gM" % (k, v / 1e6) for k, v in targets.items()))
    timings: Dict[str, float] = {}

    # ---- stage 0: BabyLM n-gram set --------------------------------------- #
    t = time.time()
    hashes, bmeta = build_babylm_hashes(args.babylm_dirs, work, args.ngram, workers)
    _FILTER = NgramFilter(hashes, args.ngram)
    timings["babylm_hashes"] = round(time.time() - t, 1)

    # ---- news-domain set (CC-News urls) for the FineWeb exclusion ------------- #
    t = time.time()
    if args.no_news_domain_filter:
        _NEWS_HASHES, news_meta = None, {"enabled": False, "fingerprint": "off"}
    elif not list_source_files(Path(args.raw_root), SPEC_BY_NAME["fineweb"]):
        _NEWS_HASHES, news_meta = None, {"enabled": True, "skipped": "no fineweb files", "fingerprint": "none"}
    else:
        _NEWS_HASHES, news_meta = build_news_domains(Path(args.raw_root), work, workers, args.news_domain_min_urls)
    if _NEWS_HASHES is not None and args.news_domain_allow:
        allow = sorted({d.strip().lower() for d in args.news_domain_allow.split(",") if d.strip()})
        ah = np.array([hash64(d) for d in allow], dtype=np.uint64)
        before = len(_NEWS_HASHES)
        _NEWS_HASHES = _NEWS_HASHES[~np.isin(_NEWS_HASHES, ah)]
        news_meta = {**news_meta, "allowed_domains": allow, "domains_used": len(_NEWS_HASHES),
                     "domains_removed_by_allowlist": before - len(_NEWS_HASHES),
                     "fingerprint": "%s+allow%s" % (news_meta.get("fingerprint"), sha1_short(allow))}
    timings["news_domains"] = round(time.time() - t, 1)

    cfg = {"raw_root": args.raw_root, "seed": args.seed, "targets": targets, "dry_run_limit": args.dry_run_limit,
           "max_line_chars": args.max_line_chars, "select_margin": args.select_margin,
           "dedup_slack": args.dedup_slack, "spread": args.spread, "min_files": args.min_files,
           "workers": workers, "babylm_fp": bmeta["fingerprint"],
           "news_fp": news_meta.get("fingerprint"),
           "news_meta": {k: v for k, v in news_meta.items() if k != "fingerprint"}}
    for s in SOURCES:        # fresh specs every build (CLI overrides must not leak between calls)
        mw = (args.min_book_words if s.name == "bookcorpusopen" else args.min_doc_words)
        SPEC_BY_NAME[s.name] = SourceSpec(**{**s.__dict__, "min_doc_words": mw if mw is not None else s.min_doc_words})

    # ---- stage 1+2: extract per source, dedup, trim ------------------------ #
    _CANCEL = mp.get_context("fork").Event() if workers > 1 else None
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    seen = np.zeros(0, dtype=np.uint64)
    per_source: Dict[str, dict] = {}
    try:
        for base_spec in SOURCES:
            spec = SPEC_BY_NAME[base_spec.name]
            t = time.time()
            r = process_source(spec, cfg, work, seen, pool)
            timings["source_" + spec.name] = round(time.time() - t, 1)
            per_source[spec.name] = r
            if r["arr"] is not None and len(r["sel"]):
                seen = np.union1d(seen, r["arr"]["chash"][r["sel"]])
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # ---- global shuffle + shard cut ---------------------------------------- #
    t = time.time()
    unit_paths: List[str] = []
    cols = {k: [] for k in ("gi", "off", "nbytes", "words", "lines", "chash")}
    for s in SOURCES:
        r = per_source[s.name]
        if r["arr"] is None or not len(r["sel"]):
            continue
        a, sel = r["arr"], r["sel"]
        base = len(unit_paths)
        unit_paths.extend(_unit_paths(u["prefix"])[0] for u in r["units"])
        cols["gi"].append(a["unit"][sel].astype(np.int64) + base)
        cols["off"].append(a["offset"][sel])
        cols["nbytes"].append(a["nbytes"][sel])
        cols["words"].append(a["words"][sel])
        cols["lines"].append(a["lines"][sel])
        cols["chash"].append(a["chash"][sel])
    if cols["gi"]:
        D = {k: np.concatenate(v) for k, v in cols.items()}
    else:
        D = {"gi": np.zeros(0, np.int64), "off": np.zeros(0, np.int64), "nbytes": np.zeros(0, np.int64),
             "words": np.zeros(0, np.int64), "lines": np.zeros(0, np.int64), "chash": np.zeros(0, np.uint64)}
    n_docs = len(D["words"])
    shuffle_key = _fmix64(D["chash"] ^ np.uint64(hash64("shuffle|%d" % args.seed)))
    perm = np.argsort(shuffle_key, kind="stable")
    D = {k: v[perm] for k, v in D.items()}
    total_words = int(D["words"].sum())
    n_shards = max(1, min(args.shards, n_docs)) if n_docs else 0
    words_per_shard = int(math.ceil(total_words / n_shards)) if n_shards else 0
    cum_before = np.cumsum(D["words"]) - D["words"]
    shard_id = np.minimum(cum_before // max(1, words_per_shard), max(0, n_shards - 1)) if n_shards else np.zeros(0, np.int64)
    width = max(3, len(str(max(0, n_shards - 1))))
    log("%d docs, %.4gM words -> %d shards of ~%.4gM words" % (n_docs, total_words / 1e6, n_shards, words_per_shard / 1e6))
    tasks, shard_summary = [], []
    bounds = np.searchsorted(shard_id, np.arange(n_shards + 1)) if n_shards else []
    for si in range(n_shards):
        a, b = int(bounds[si]), int(bounds[si + 1])
        name = "shard_%0*d" % (width, si)
        ph = hashlib.sha1(D["chash"][a:b].tobytes() + b"|%d|%d" % (si, n_shards)).hexdigest()
        tasks.append({"dir": str(out / name), "name": name, "plan_hash": ph, "paths": unit_paths,
                      "gi": D["gi"][a:b], "off": D["off"][a:b], "nbytes": D["nbytes"][a:b]})
        shard_summary.append({"name": name, "docs": b - a, "lines": int(D["lines"][a:b].sum()),
                              "words": int(D["words"][a:b].sum())})
    if tasks:
        if workers > 1:
            with mp.get_context("fork").Pool(min(workers, len(tasks))) as p2:
                sres = p2.map(write_shard, tasks, chunksize=1)
        else:
            sres = [write_shard(tk) for tk in tasks]
        log("shards: wrote %d, cached %d" % (sum(not r["cached"] for r in sres), sum(r["cached"] for r in sres)))
    timings["shuffle_and_shards"] = round(time.time() - t, 1)

    # ---- manifest ----------------------------------------------------------- #
    def src_manifest(s: SourceSpec) -> dict:
        r, st, si = per_source[s.name], per_source[s.name]["stats"], per_source[s.name]["select_info"]
        spec = SPEC_BY_NAME[s.name]
        ck, wk = st["contam_lines_checked"], st["contam_words_checked"]
        m = {
            "target_words": r["target_words"],
            "files_available": r["files_available"],
            "files_processed": r.get("files_processed", 0),
            "files_failed": r["files_failed"],
            "min_doc_words": spec.min_doc_words,
            "docs_read": st["docs_read"],
            "docs_candidates_selected": st["docs_candidates"],
            "docs_dropped_length": st["docs_dropped_length"],
            "docs_dropped_length_after_contamination": st["docs_dropped_length_post_contam"],
            "contamination": {
                "lines_checked": ck, "lines_dropped": st["contam_lines_dropped"],
                "words_checked": wk, "words_dropped": st["contam_words_dropped"],
                "docs_affected": st["contam_docs_affected"],
                "line_drop_rate": (st["contam_lines_dropped"] / ck) if ck else 0.0,
                "word_drop_rate": (st["contam_words_dropped"] / wk) if wk else 0.0},
            "docs_dropped_dedup": si["docs_dropped_dedup"],
            "docs_dropped_dedup_within_source": si["docs_dropped_dedup_within_source"],
            "docs_dropped_dedup_cross_source": si["docs_dropped_dedup_cross_source"],
            "docs_trimmed_over_target": si["docs_trimmed_over_target"],
            "docs_kept": si["docs_kept"],
            "words_kept": si["words_kept"],
            "target_met": si["words_kept"] >= r["target_words"],
            "shortfall_words": max(0, r["target_words"] - si["words_kept"]),
            "plan": r["plan"],
            "seconds": r.get("seconds", 0)}
        if "docs_read_scoring_pass" in r:
            m["docs_read_scoring_pass"] = r["docs_read_scoring_pass"]
        if s.mode == "score":
            m["news_domain_filter"] = r.get("news_filter", {})
            kept_prio = r["arr"]["prio"][r["sel"]] if r["arr"] is not None and len(r["sel"]) else np.zeros(0)
            m["fineweb_score"] = {
                "formula": SCORE_FORMULA % {**SCORE_WEIGHTS, "edmin": ED_MIN_LEN, "minw": FINEWEB_MIN_WORDS,
                                            "markers": ", ".join(NEWS_MARKERS),
                                            "minurls": args.news_domain_min_urls},
                "weights": dict(SCORE_WEIGHTS),
                "narrative_verbs": list(NARRATIVE_VERBS), "news_markers": list(NEWS_MARKERS),
                "min_words_eligible": FINEWEB_MIN_WORDS,
                "threshold_score": float(-kept_prio.max()) if len(kept_prio) else None,
                "max_score_kept": float(-kept_prio.min()) if len(kept_prio) else None}
        return m

    sm = {s.name: src_manifest(s) for s in SOURCES}
    totals = {"docs_kept": sum(v["docs_kept"] for v in sm.values()),
              "words_kept": sum(v["words_kept"] for v in sm.values()),
              "lines_kept": int(D["lines"].sum()),
              "contam_lines_dropped": sum(v["contamination"]["lines_dropped"] for v in sm.values()),
              "contam_words_dropped": sum(v["contamination"]["words_dropped"] for v in sm.values()),
              "docs_dropped_length": sum(v["docs_dropped_length"] for v in sm.values()),
              "docs_dropped_dedup": sum(v["docs_dropped_dedup"] for v in sm.values())}
    sw = [s["words"] for s in shard_summary]
    manifest = {
        "build": "optionb_v1",
        "sources": sm,
        "totals": totals,
        "babylm": {k: bmeta[k] for k in ("ngram", "tokenization", "ngram_windows", "lines", "words", "tokens",
                                                        "ngrams_unique", "hash_version")} |
                  {"dirs": list(args.babylm_dirs), "files": [f["file"] for f in bmeta["files"]]},
        "params": {"seed": args.seed, "scale": args.scale, "targets_words": targets,
                   "base_targets_words": {s.name: s.base_target for s in SOURCES},
                   "shards_requested": args.shards, "dry_run_limit": args.dry_run_limit,
                   "max_line_chars": args.max_line_chars, "min_doc_words": {s.name: SPEC_BY_NAME[s.name].min_doc_words for s in SOURCES},
                   "select_margin": args.select_margin, "dedup_slack": args.dedup_slack,
                   "spread": args.spread, "min_files": args.min_files, "workers": workers,
                   "fineweb_weights": dict(SCORE_WEIGHTS), "news_domain_min_urls": args.news_domain_min_urls,
                   "news_domain_filter": not args.no_news_domain_filter, "raw_root": str(args.raw_root)},
        "shards": {"count": n_shards, "words_per_shard_target": words_per_shard,
                   "words_min": min(sw) if sw else 0, "words_max": max(sw) if sw else 0,
                   "words_mean": (sum(sw) / len(sw)) if sw else 0,
                   "layout": "<out>/shard_NNN/shard_NNN.train + docs.jsonl.gz",
                   "train_format": "one paragraph per line; one empty line after every document",
                   "docs_jsonl_fields": {"first_line": "0-based line index of the doc's first line in the .train file",
                                         "n_lines": "text lines of the doc (excludes the blank separator)",
                                         "n_words": "whitespace-split words after filtering"},
                   "per_shard": shard_summary},
        "code_version": code_version(),
        "timings_seconds": timings,
        "wall_time_seconds": round(time.time() - t_start, 1),
    }
    mp_ = out / "BUILD_MANIFEST.json"
    _atomic_write(str(mp_), json.dumps(manifest, indent=2).encode())
    log("manifest -> %s (wall %.0fs)" % (mp_, manifest["wall_time_seconds"]))
    if args.cleanup_work:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    return manifest


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default="/mnt/data/rater_corpus/raw")
    ap.add_argument("--babylm-dirs", nargs="+", default=[
        "/mnt/data/raw/en/train_90M", "/mnt/data/raw/en/pull_10M", "/mnt/data/raw/en/test_10M"])
    ap.add_argument("--out", default="/mnt/data/rater_corpus/optionb_v1/base")
    ap.add_argument("--work-dir", default=None, help="intermediates (default: <out parent>/_work)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier on all word targets (10B = 2)")
    ap.add_argument("--target", action="append", default=[], metavar="SOURCE=WORDS",
                    help="override a base target (before --scale); repeatable")
    ap.add_argument("--shards", type=int, default=200)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run-limit", type=int, default=None, metavar="N",
                    help="read only the first N docs of the first file of each source")
    ap.add_argument("--fineweb-weights", default=None, metavar="quote=1,verb=1,ed=1",
                    help="weights of the three FineWeb narrative-score terms (default all 1)")
    ap.add_argument("--no-news-domain-filter", action="store_true",
                    help="do not exclude FineWeb docs whose domain appears in CC-News")
    ap.add_argument("--news-domain-allow", default=None, metavar="d1.com,d2.com",
                    help="registrable domains never excluded even if they occur in CC-News "
                         "(e.g. blogspot.com,wordpress.com); default none")
    ap.add_argument("--news-domain-min-urls", type=int, default=1,
                    help="a domain counts as news if >= this many CC-News urls (default 1)")
    ap.add_argument("--ngram", type=int, default=NGRAM)
    ap.add_argument("--max-line-chars", type=int, default=4000)
    ap.add_argument("--min-doc-words", type=int, default=None, help="override 50")
    ap.add_argument("--min-book-words", type=int, default=None, help="override 2000")
    ap.add_argument("--select-margin", type=float, default=0.12,
                    help="oversample factor to absorb contamination/length/dedup losses")
    ap.add_argument("--dedup-slack", type=float, default=0.05,
                    help="stop reading files once pre-dedup words >= target*(1+slack)")
    ap.add_argument("--spread", type=float, default=4.0,
                    help="read ~spread x the files strictly needed, sampling docs at p=1/spread")
    ap.add_argument("--min-files", type=int, default=16, help="minimum files per hash-sampled source")
    ap.add_argument("--cleanup-work", action="store_true", help="delete work dir after success")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    build(make_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
