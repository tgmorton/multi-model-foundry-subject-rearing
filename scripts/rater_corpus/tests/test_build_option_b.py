"""Tests for scripts/rater_corpus/build_option_b.py.

Run:  <venv>/bin/python -m pytest scripts/rater_corpus/tests/test_build_option_b.py

All fixtures are tiny synthetic parquet/text files that mimic the real schemas;
nothing here touches the network or the cluster.
"""
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_option_b as bob  # noqa: E402

PRONOUNS = {"i", "me", "my", "mine", "you", "your", "he", "him", "his", "she", "her", "hers",
            "it", "its", "we", "us", "our", "they", "them", "their"}


# --------------------------------------------------------------------------- #
# synthetic data
# --------------------------------------------------------------------------- #
def _vocab(rng, n=4000):
    out = set()
    while len(out) < n:
        w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(4, 9)))
        if not w.endswith("ed") and w not in bob.NARRATIVE_VERBS and w not in PRONOUNS:
            out.add(w)
    return sorted(out)


def _sentence(rng, vocab, lo=6, hi=16):
    ws = [rng.choice(vocab) for _ in range(rng.randint(lo, hi))]
    return " ".join(ws).capitalize() + "."


def expository(rng, vocab, n_words):
    paras, total = [], 0
    while total < n_words:
        sents = [_sentence(rng, vocab) for _ in range(rng.randint(2, 5))]
        paras.append(" ".join(sents))
        total += sum(len(s.split()) for s in sents)
    return "\n\n".join(paras)


def story(rng, vocab, n_words):
    paras, total = [], 0
    while total < n_words:
        a, b = rng.choice(vocab), rng.choice(vocab)
        sents = ['"%s %s," said the %s.' % (a.capitalize(), b, rng.choice(vocab)),
                 'He looked at the %s and walked toward the %s.' % (a, b),
                 'She smiled, turned, and whispered "%s %s."' % (b, a),
                 _sentence(rng, vocab)]
        paras.append(" ".join(sents))
        total += sum(len(s.split()) for s in sents)
    return "\n".join(paras)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """raw root + BabyLM dirs + the planted contaminated paragraph."""
    root = tmp_path_factory.mktemp("world")
    rng = random.Random(7)
    vocab = _vocab(rng)
    raw = root / "raw"

    # BabyLM: .train in one dir, .test in the other (the real test_10M uses .test)
    bl1, bl2 = root / "bl_train", root / "bl_test"
    bl1.mkdir(); bl2.mkdir()
    baby_para = " ".join(rng.choice(vocab) for _ in range(40)) + "."          # 40 tokens
    baby_short = "*MOT:\tnot enough tokens here."
    # three short consecutive BabyLM lines (5 tokens each, speaker tags, an @-header and a
    # boundary marker in between): no within-line 13-gram exists, only the cross-line stream has them
    cw = [rng.choice(vocab) for _ in range(15)]
    cross_lines = ["*MOT:\t" + " ".join(cw[0:5]) + ".", "@Languages:\teng", "*CHI:\t" + " ".join(cw[5:10]) + "?",
                   "= = = NEW TRANSCRIPT = = =", "*MOT:\t" + " ".join(cw[10:15]) + "!"]
    cross_para = " ".join(cw) + "."                                      # what a web page would contain
    (bl1 / "a.train").write_text("\n".join([_sentence(rng, vocab, 20, 30) for _ in range(30)] +
                                           [baby_para, baby_short] + cross_lines + [""]))
    baby_para2 = " ".join(rng.choice(vocab) for _ in range(25)) + "!"
    (bl2 / "b.test").write_text("\n".join([_sentence(rng, vocab, 20, 30) for _ in range(10)] + [baby_para2]))
    (bl1 / "meta.json").write_text("{}")          # must be ignored

    def write(path, table, rg=7):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, row_group_size=rg)

    dup_text = expository(rng, vocab, 120)
    # --- ccnews: 2 files x 40 docs, dup across files, short docs, planted contamination
    for f in range(2):
        texts, urls = [], []
        for i in range(40):
            if i == 3:
                t = "too short to keep"
            elif i == 5:
                t = dup_text if f == 0 else "  " + dup_text.replace(" ", "  ") + "\n\n"   # same after normalization
            elif i == 9:                                             # planted BabyLM paragraph in a longer doc
                t = expository(rng, vocab, 100) + "\n" + baby_para + "\n" + expository(rng, vocab, 100)
            elif i == 11 and f == 0:                                 # paragraph spanning three short BabyLM lines
                t = expository(rng, vocab, 100) + "\n" + cross_para + "\n" + expository(rng, vocab, 100)
            else:
                t = expository(rng, vocab, rng.randint(80, 400))
            texts.append(t); urls.append("http://news.example/%d/%d" % (f, i))
        write(raw / "ccnews_en_2016_2019" / ("2017_%04d.parquet" % f), pa.table({
            "requested_url": urls, "plain_text": texts, "published_date": ["2017-01-01"] * 40,
            "title": ["t"] * 40, "sitename": ["s"] * 40, "language": ["en"] * 40, "language_score": [0.99] * 40}))
    # junk that must be ignored
    (raw / "ccnews_en_2016_2019" / "MANIFEST_2017.json").write_text("{}")
    (raw / "ccnews_en_2016_2019" / "2018_0000.parquet.tmp").write_text("garbage")
    (raw / "ccnews_en_2016_2019" / ".cache").mkdir()
    (raw / "ccnews_en_2016_2019" / ".cache" / "x.parquet").write_text("garbage")
    # --- openwebtext
    owt = [expository(rng, vocab, rng.randint(80, 300)) for _ in range(50)]
    owt[4] = dup_text                                                # duplicate of a ccnews doc
    owt[7] = baby_para2 + "\n" + expository(rng, vocab, 90)
    write(raw / "openwebtext" / "plain_text" / "train-00000-of-00001.parquet", pa.table({"text": owt}))
    # --- wikipedia
    wiki = [expository(rng, vocab, rng.randint(60, 250)) for _ in range(30)]
    write(raw / "wikipedia_20231101_en" / "20231101.en" / "train-00000-of-00001.parquet", pa.table({
        "id": [str(i) for i in range(30)], "url": ["https://en.wikipedia.org/wiki/P%d" % i for i in range(30)],
        "title": ["P%d" % i for i in range(30)], "text": wiki}))
    # --- books (tiny "books": use --min-book-words 300 in tests)
    books = [expository(rng, vocab, rng.randint(500, 900)) for _ in range(6)] + ["short book " * 20]
    write(raw / "bookcorpusopen" / "data" / "train-00000-of-00001.parquet", pa.table({
        "title": ["book%d.epub.txt" % i for i in range(7)], "text": books}), rg=3)
    # --- fineweb: 30 story-like + 30 expository (+ a short one); ids carry the label
    fw_texts, fw_ids = [], []
    for i in range(30):
        fw_texts.append(story(rng, vocab, rng.randint(250, 330))); fw_ids.append("story-%d" % i)
        fw_texts.append(expository(rng, vocab, rng.randint(250, 330))); fw_ids.append("expo-%d" % i)
    fw_texts.append("tiny"); fw_ids.append("tiny-0")
    n = len(fw_texts)
    write(raw / "fineweb_sample10BT_b" / "sample" / "10BT" / "000_00000.parquet", pa.table({
        "text": fw_texts, "id": fw_ids, "dump": ["d"] * n,
        # story-0..4 live on a domain that CC-News also contains (news.example, www. stripped)
        "url": [("https://www.news.example/p/%s" % i) if i in ("story-%d" % k for k in range(5)) else "http://x/%s" % i
                for i in fw_ids],
        "date": ["2020"] * n, "file_path": ["f"] * n, "language": ["en"] * n,
        "language_score": [0.9] * n, "token_count": [1] * n}))
    # partial download dir that must be ignored
    write(raw / "fineweb_sample10BT" / "sample" / "10BT" / "000_00000.parquet", pa.table({"text": ["x " * 300] * 5, "id": list("abcde")}))
    return {"root": root, "raw": raw, "babylm": [str(bl1), str(bl2)], "baby_para": baby_para,
            "cross_para": cross_para, "cross_lines": cross_lines, "bl1": bl1,
            "baby_para2": baby_para2, "vocab": vocab, "dup_text": dup_text}


BIG = ["--target", "ccnews=1e9", "--target", "openwebtext=1e9", "--target", "fineweb=1e9",
       "--target", "wikipedia=1e9", "--target", "bookcorpusopen=1e9"]


def run(world, tag, extra=(), workers=1, shards=4, targets=BIG):
    out = world["root"] / ("out_" + tag) / "base"
    argv = ["--raw-root", str(world["raw"]), "--babylm-dirs", *world["babylm"], "--out", str(out),
            "--shards", str(shards), "--workers", str(workers), "--min-book-words", "300", *targets, *extra]
    assert bob.main(argv) == 0
    return out, json.loads((out / "BUILD_MANIFEST.json").read_text())


def read_shards(out):
    shards = sorted(p for p in out.iterdir() if p.is_dir() and p.name.startswith("shard_"))
    docs = []
    for sd in shards:
        lines = (sd / (sd.name + ".train")).read_text(encoding="utf-8").split("\n")
        assert lines[-1] == ""                       # file ends with newline
        lines = lines[:-1]
        rows = [json.loads(l) for l in gzip.open(sd / "docs.jsonl.gz", "rt", encoding="utf-8")]
        for r in rows:
            docs.append((sd.name, r, lines[r["first_line"]:r["first_line"] + r["n_lines"]]))
    return shards, docs


# --------------------------------------------------------------------------- #
# normalization / paragraph splitting
# --------------------------------------------------------------------------- #
def test_normalize_text():
    t = "Café \t\t x\x00y\x07z\r\nline2\rline3 line4  \xa0 end"
    n = bob.normalize_text(t)
    assert n == "Café xyz\nline2\nline3\nline4 end"          # NFC, controls gone, spaces collapsed
    assert "\t" not in n and "  " not in n and "\r" not in n


def test_doc_to_lines_drops_empty_paragraphs_and_strips():
    assert bob.doc_to_lines("  Hello   world \n\n\n second\tline  \n \t \n") == ["Hello world", "second line"]
    assert bob.doc_to_lines("") == [] and bob.doc_to_lines(" \n\n") == []


def test_long_paragraph_split_at_sentence_boundaries():
    para = ("This is a sentence that ends here. " * 400).strip()      # ~14k chars
    lines = bob.doc_to_lines(para, 4000)
    assert len(lines) >= 4 and all(len(l) <= 4000 for l in lines)
    assert all(l.endswith(".") for l in lines)                         # cut only after sentence ends
    assert " ".join(lines) == para                                     # nothing lost
    short = "Short paragraph. Stays whole."
    assert bob.doc_to_lines(short) == [short]


def test_long_sentence_hard_split_at_whitespace_then_anywhere():
    words = " ".join(["word"] * 3000)                                  # one 14999-char "sentence", no .!?
    lines = bob.doc_to_lines(words, 4000)
    assert all(len(l) <= 4000 for l in lines) and " ".join(lines) == words
    blob = "x" * 9500                                                  # no whitespace at all
    lines = bob.doc_to_lines(blob, 4000)
    assert [len(l) for l in lines] == [4000, 4000, 1500] and "".join(lines) == blob


# --------------------------------------------------------------------------- #
# contamination filter
# --------------------------------------------------------------------------- #
def _flt(lines):
    h, _ = bob.line_ngram_hashes(lines)
    return bob.NgramFilter(np.unique(h))


def test_contamination_filter_line_level():
    base = "the quick brown fox jumps over the lazy dog and runs far away from here today"   # 16 tokens
    flt = _flt([base, "short line"])
    lines = [
        "Intro words. THE quick, brown fox jumps over the lazy dog -- and runs far away from here! tail",  # contains
        "the quick brown fox jumps over the lazy dog and runs far",           # only 12 tokens: no 13-gram
        "the quick brown fox jumps over the lazy dog and runs",               # 11 tokens
        "a completely unrelated line " * 6,
        "short line",                                                         # < 13 tokens: cannot match
    ]
    mask = bob.contaminated_line_mask(lines, flt)
    assert mask.tolist() == [True, False, False, False, False]
    assert bob.contaminated_line_mask(["the quick brown fox jumps over the lazy dog and runs far away"], flt).tolist() == [True]


def test_ngram_windows_do_not_cross_line_boundaries():
    a = "one two three four five six seven"
    b = "eight nine ten eleven twelve thirteen fourteen"
    flt = _flt(["one two three four five six seven eight nine ten eleven twelve thirteen"])
    # the 13 tokens are split over two lines -> must NOT match
    assert not bob.contaminated_line_mask([a, b], flt).any()
    assert bob.contaminated_line_mask([a + " " + b], flt).all()


def test_filter_has_no_false_positives_on_random_queries():
    rng = np.random.RandomState(0)
    members = np.unique(rng.randint(0, 2**62, size=50000).astype(np.uint64) * np.uint64(4) + np.uint64(1))
    flt = bob.NgramFilter(members)
    assert flt.contains(members[:1000]).all()
    others = rng.randint(0, 2**62, size=20000).astype(np.uint64) * np.uint64(4) + np.uint64(2)
    assert not flt.contains(others).any()


# --------------------------------------------------------------------------- #
# FineWeb narrative score
# --------------------------------------------------------------------------- #
def test_fineweb_score_ignores_pronouns():
    base = ('"Where are you going?" said the old man. {S} looked at the door and walked out. '
            'The cat turned and {O} saw {P} friend. {S} smiled and whispered to {O} about {P} plan. ') * 5
    doc_a = base.format(S="He", O="him", P="his")
    doc_b = base.format(S="She", O="her", P="their")
    doc_c = base.format(S="They", O="them", P="our")
    sa, na = bob.narrative_score(doc_a)
    sb, nb = bob.narrative_score(doc_b)
    sc, nc = bob.narrative_score(doc_c)
    assert na == nb == nc and sa == sb == sc and sa > 0
    # sanity: the score does respond to what it should respond to
    flat = "The committee reviews the annual report on regional trade policy and publishes the findings. " * 20
    assert bob.narrative_score(flat)[0] < sa / 4
    assert not (set(bob.NARRATIVE_VERBS) & PRONOUNS)
    # the formula is documented without any pronoun term
    assert "pronoun" in bob.SCORE_FORMULA and "No pronoun is counted" in bob.SCORE_FORMULA


def test_registrable_domain():
    rd = bob.registrable_domain
    assert rd("https://www.bbc.co.uk/news/x") == "bbc.co.uk"
    assert rd("http://news.sub.example.com.au/a") == "example.com.au"
    assert rd("http://WWW.Example.COM:8080/x?y#z") == "example.com"
    assert rd("http://foo.bar.nj.com") == "nj.com"
    assert rd("http://user:pw@a.example.org/x") == "example.org"
    assert rd("http://1.2.3.4/x") == "1.2.3.4"
    assert rd("example.org/path") == "example.org"
    assert rd("") == "" and rd(None) == ""


def test_news_penalty_terms_and_weights(monkeypatch):
    body = 'The old woman said "no" and walked home. She smiled at the garden. ' * 6
    a = body + "About 45 percent of 1,200 residents reported damage, officials said."
    b = body + "About many percent of several residents reported damage, officials said."
    c = body + "About many extra of several residents reported damage, officials said."
    n_a, n_b, n_c = (bob.narrative_features(t)[0] for t in (a, b, c))
    assert n_a == n_b == n_c
    fa, fb, fc = (bob.narrative_features(t) for t in (a, b, c))
    assert fa[4] - fb[4] == 2 and fb[5] - fc[5] == 1 and fa[5] == fb[5]      # D and M counted
    sa, sb, sc = (bob.narrative_score(t)[0] for t in (a, b, c))
    assert sb - sa == pytest.approx(2 / n_a) and sc - sb == pytest.approx(1 / n_a)
    monkeypatch.setitem(bob.SCORE_WEIGHTS, "digit", 0.0)
    monkeypatch.setitem(bob.SCORE_WEIGHTS, "news", 0.0)
    assert bob.narrative_score(a)[0] == bob.narrative_score(b)[0] == bob.narrative_score(c)[0]
    assert not (set(bob.NEWS_MARKERS) & PRONOUNS)


def test_fineweb_selects_story_like_docs(world):
    # ~35 story docs would fit; ask for fewer words than the story docs hold -> only story docs picked
    t = ["--target", "ccnews=1e9", "--target", "openwebtext=1e9", "--target", "wikipedia=1e9",
         "--target", "bookcorpusopen=1e9", "--target", "fineweb=4000"]
    out, m = run(world, "fwrank", targets=t)
    _, docs = read_shards(out)
    fw = [r["doc_key"] for _, r, _ in docs if r["source"] == "fineweb"]
    assert fw and all(k.startswith("story-") for k in fw), fw
    assert not any(k in fw for k in ("story-%d" % i for i in range(5)))       # news.example domain excluded
    assert m["sources"]["fineweb"]["words_kept"] >= 4000
    fs = m["sources"]["fineweb"]["fineweb_score"]
    assert fs["threshold_score"] is not None and fs["formula"] and "pronoun" in fs["formula"].lower()
    assert "NEWS_MARKERS" in fs["formula"] and "digit" in fs["formula"] and fs["weights"]["digit"] == 1.0
    nf = m["sources"]["fineweb"]["news_domain_filter"]
    assert nf["docs_excluded_by_domain"] == 5 == nf["docs_excluded_by_domain_eligible_length"]
    assert nf["domains_total"] >= 1 and nf["domains_used"] >= 1 and nf["ccnews_files_scanned"] == 2
    assert nf["top_excluded_domains"][0] == ["news.example", 5]


def test_news_domain_allowlist(world):
    t = ["--target", "ccnews=1e9", "--target", "openwebtext=1e9", "--target", "wikipedia=1e9",
         "--target", "bookcorpusopen=1e9", "--target", "fineweb=1e9"]
    _, m = run(world, "allow", targets=t, extra=["--news-domain-allow", "News.Example, other.org"])
    nf = m["sources"]["fineweb"]["news_domain_filter"]
    assert nf["docs_excluded_by_domain"] == 0 and nf["allowed_domains"] == ["news.example", "other.org"]
    assert nf["domains_removed_by_allowlist"] == 1


def test_news_domain_filter_can_be_disabled(world):
    t = ["--target", "ccnews=1e9", "--target", "openwebtext=1e9", "--target", "wikipedia=1e9",
         "--target", "bookcorpusopen=1e9", "--target", "fineweb=4000", "--fineweb-weights", "quote=1,verb=1,ed=1,digit=0,news=0"]
    out, m = run(world, "nonews", targets=t, extra=["--no-news-domain-filter"])
    nf = m["sources"]["fineweb"]["news_domain_filter"]
    assert nf["enabled"] is False and nf["docs_excluded_by_domain"] == 0
    assert m["params"]["news_domain_filter"] is False and m["params"]["fineweb_weights"]["digit"] == 0.0
    _, docs = read_shards(out)
    assert any(r["doc_key"].startswith("story-") and int(r["doc_key"].split("-")[1]) < 5
               for _, r, _ in docs if r["source"] == "fineweb")


# --------------------------------------------------------------------------- #
# end to end: layout, contamination, dedup, manifest
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def full(world):
    return run(world, "full", shards=5)


def test_shard_layout_and_blank_line_separators(full):
    out, m = full
    shards, docs = read_shards(out)
    assert [s.name for s in shards] == ["shard_%03d" % i for i in range(5)]
    for s in shards:
        assert sorted(p.name for p in s.iterdir() if not p.name.startswith(".")) == sorted([s.name + ".train", "docs.jsonl.gz"])
        raw = (s / (s.name + ".train")).read_text(encoding="utf-8")
        assert raw.endswith("\n\n") or raw == ""                       # last doc also followed by a boundary
        assert "\n\n\n" not in raw                                     # exactly one empty line between docs
        assert not raw.startswith("\n")
    # docs.jsonl.gz rows point at the right lines, in order, with blank separators between docs
    by_shard = {}
    for name, r, lines in docs:
        by_shard.setdefault(name, []).append((r, lines))
    for name, items in by_shard.items():
        full_lines = (out / name / (name + ".train")).read_text(encoding="utf-8").split("\n")
        expect_first = 0
        for r, lines in items:
            assert r["first_line"] == expect_first
            assert len(lines) == r["n_lines"] > 0 and all(l.strip() == l and l for l in lines)
            assert full_lines[r["first_line"] + r["n_lines"]] == ""    # blank separator right after the doc
            assert r["n_words"] == sum(len(l.split()) for l in lines)
            assert set(r) == {"source", "doc_key", "first_line", "n_lines", "n_words"}
            expect_first += r["n_lines"] + 1
    # sources interleaved: at least one shard holds several sources
    assert any(len({r["source"] for r, _ in items}) >= 3 for items in by_shard.values())
    assert all(len(l) <= 4000 for _, _, ls in docs for l in ls)


def _stream_grams(files, n=13):
    """Independent reference: n-grams of each file's concatenated token stream
    (speaker tags stripped, @-header and boundary lines skipped)."""
    import re
    grams = set()
    for p in files:
        toks = []
        for line in p.read_text().split("\n"):
            if re.match(r"\s*@[A-Za-z]", line) or (line.strip().startswith("= =") and line.strip().endswith("= =")):
                continue
            line = re.sub(r"^\*[A-Z0-9]+:\s*", "", line)
            toks.extend(re.findall(r"\w+", line.lower()))
        grams.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return grams


def test_no_babylm_ngram_survives_and_clean_text_is_kept(world, full):
    out, m = full
    n = 13
    files = [p for d in world["babylm"] for p in Path(d).iterdir() if p.suffix in (".train", ".test")]
    grams = _stream_grams(files, n)
    assert len(grams) > 100
    _, docs = read_shards(out)
    for _, _, lines in docs:
        for line in lines:
            t = bob._WORD_RE.findall(line.lower())
            assert not any(tuple(t[i:i + n]) in grams for i in range(len(t) - n + 1)), line[:80]
    text = "\n".join("\n".join(l) for _, _, l in docs)
    assert world["baby_para"] not in text and world["baby_para2"] not in text
    assert world["cross_para"] not in text                         # three short BabyLM lines, concatenated
    # the planted-contamination docs lost only the planted paragraph: their other paragraphs survive
    c = m["sources"]["ccnews"]["contamination"]
    assert c["lines_dropped"] >= 3 and c["words_dropped"] >= 95 and c["docs_affected"] >= 3
    assert 0 < c["line_drop_rate"] < 0.2 and m["sources"]["openwebtext"]["contamination"]["lines_dropped"] >= 1
    assert m["sources"]["wikipedia"]["contamination"]["lines_dropped"] == 0
    assert any(r["doc_key"] == "http://news.example/0/9" for _, r, _ in docs)       # rest of the planted doc kept
    assert any(r["doc_key"] == "http://news.example/0/11" for _, r, _ in docs)


def test_babylm_cross_line_windows(world, tmp_path):
    """A corpus line made of consecutive short BabyLM lines is contaminated; per-line
    windows still work; tags/headers/markers are not part of the stream."""
    f = tmp_path / "x.train"
    f.write_text("\n".join(world["cross_lines"] + ["", "*CHI:\t"]) + "\n")
    out = tmp_path / "x.npy"
    st = bob._babylm_file_hashes((str(f), str(out), 13))
    flt = bob.NgramFilter(np.load(out))
    assert st["tokens"] == 15 and st["ngrams_unique"] == 3                # 15 tokens -> 3 windows of 13
    cw = world["cross_para"].rstrip(".").split()
    assert bob.contaminated_line_mask([world["cross_para"]], flt).tolist() == [True]
    assert bob.contaminated_line_mask(["Intro words. " + " ".join(cw[2:15]) + " and more words after"], flt).tolist() == [True]
    assert bob.contaminated_line_mask([" ".join(cw[:12])], flt).tolist() == [False]            # only 12 tokens
    # header / tag / marker tokens are not in the stream: they do not make a match either
    assert bob.contaminated_line_mask([" ".join(cw[:5] + ["languages", "eng"] + cw[5:13])], flt).tolist() == [False]
    assert bob.contaminated_line_mask([" ".join(["mot"] + cw[:12])], flt).tolist() == [False]
    assert bob.contaminated_line_mask([" ".join(cw[:5] + cw[10:15] + cw[5:8])], flt).tolist() == [False]   # wrong order
    # a long single line still gets its within-line windows (subset of the stream windows)
    g = tmp_path / "y.train"
    long_line = " ".join(world["vocab"][:20])
    g.write_text(long_line + "\n")
    st2 = bob._babylm_file_hashes((str(g), str(tmp_path / "y.npy"), 13))
    flt2 = bob.NgramFilter(np.load(tmp_path / "y.npy"))
    assert st2["ngrams_unique"] == 8 and bob.contaminated_line_mask([long_line], flt2).all()
    # chunk boundaries (carry between 20000-line flushes) lose no window
    h = tmp_path / "z.train"
    h.write_text("\n".join("w%d x%d y%d" % (i, i, i) for i in range(45000)) + "\n")     # 3 tokens/line
    st3 = bob._babylm_file_hashes((str(h), str(tmp_path / "z.npy"), 13))
    assert st3["tokens"] == 135000 and st3["ngrams_unique"] == 135000 - 12


def test_exact_dedup_across_files_and_sources(world, full):
    out, m = full
    _, docs = read_shards(out)
    texts = ["\n".join(l) for _, _, l in docs]
    assert len(texts) == len(set(texts))                           # no exact duplicates anywhere
    dup_norm = "\n".join(bob.doc_to_lines(world["dup_text"]))
    assert texts.count(dup_norm) == 1
    cc, ow = m["sources"]["ccnews"], m["sources"]["openwebtext"]
    assert cc["docs_dropped_dedup_within_source"] >= 1             # ccnews file0 vs file1 (whitespace-variant copy)
    assert ow["docs_dropped_dedup_cross_source"] >= 1              # openwebtext copy of a ccnews doc
    kept_from = [r["source"] for _, r, l in docs if "\n".join(l) == dup_norm]
    assert kept_from == ["ccnews"]                                 # first source in fixed order wins


def test_length_filters_and_junk_files_ignored(world, full):
    out, m = full
    _, docs = read_shards(out)
    assert all(r["n_words"] >= 50 for _, r, _ in docs if r["source"] != "bookcorpusopen")
    assert all(r["n_words"] >= 300 for _, r, _ in docs if r["source"] == "bookcorpusopen")   # --min-book-words 300
    assert m["sources"]["ccnews"]["docs_dropped_length"] == 2      # one "too short" doc per file
    assert m["sources"]["bookcorpusopen"]["docs_dropped_length"] == 1
    assert m["sources"]["fineweb"]["docs_read_scoring_pass"] == 61
    assert m["sources"]["ccnews"]["files_available"] == 2         # MANIFEST json, .tmp and .cache ignored
    assert not any(r["doc_key"].startswith("tiny") for _, r, _ in docs)
    assert m["sources"]["fineweb"]["files_available"] == 1         # the dir without _b is ignored


def test_manifest_fields(world, full):
    out, m = full
    for key in ("sources", "totals", "babylm", "params", "shards", "code_version", "timings_seconds", "wall_time_seconds"):
        assert key in m
    assert m["wall_time_seconds"] > 0
    assert set(m["sources"]) == {"ccnews", "openwebtext", "fineweb", "wikipedia", "bookcorpusopen"}
    need = {"target_words", "docs_read", "docs_kept", "words_kept", "docs_dropped_length", "docs_dropped_dedup",
            "contamination", "files_available", "files_processed", "plan", "target_met", "shortfall_words"}
    for s, sm in m["sources"].items():
        assert need <= set(sm), (s, need - set(sm))
        assert set(sm["contamination"]) >= {"lines_dropped", "words_dropped", "lines_checked", "words_checked", "line_drop_rate", "word_drop_rate"}
    fw = m["sources"]["fineweb"]["fineweb_score"]
    assert {"formula", "weights", "narrative_verbs", "threshold_score"} <= set(fw)
    assert set(m["params"]["targets_words"]) == set(m["sources"]) and m["params"]["seed"] == 42
    assert m["params"]["scale"] == 1.0
    assert m["shards"]["count"] == 5 and len(m["shards"]["per_shard"]) == 5
    assert m["shards"]["words_per_shard_target"] > 0
    assert sum(s["words"] for s in m["shards"]["per_shard"]) == m["totals"]["words_kept"]
    assert m["code_version"]["git_sha"] and len(m["code_version"]["script_sha256"]) == 64
    assert m["babylm"]["ngram"] == 13 and m["babylm"]["words"] > 0
    assert m["babylm"]["tokens"] > 0 and "token stream" in m["babylm"]["ngram_windows"]
    assert m["sources"]["fineweb"]["news_domain_filter"]["docs_excluded_by_domain"] == 5
    assert "news_markers" in fw and fw["weights"]["news"] == 1.0
    # shards hold every kept doc exactly once
    _, docs = read_shards(out)
    assert len(docs) == m["totals"]["docs_kept"] == sum(s["docs_kept"] for s in m["sources"].values())


def test_git_sha_env(world, monkeypatch):
    monkeypatch.setenv("GIT_SHA", "deadbeef")
    _, m = run(world, "sha", targets=["--target", "ccnews=500", "--target", "openwebtext=500", "--target", "fineweb=500",
                                       "--target", "wikipedia=500", "--target", "bookcorpusopen=500"], shards=2)
    assert m["code_version"]["git_sha"] == "deadbeef"


# --------------------------------------------------------------------------- #
# targets, scale, determinism, restart, failures
# --------------------------------------------------------------------------- #
def test_scale_and_targets():
    t1 = bob.parse_targets(1.0, [])
    t2 = bob.parse_targets(2.0, [])
    assert t1 == {"ccnews": 2_380_000_000, "openwebtext": 1_190_000_000, "fineweb": 970_000_000,
                  "wikipedia": 380_000_000, "bookcorpusopen": 120_000_000}
    assert all(t2[k] == 2 * t1[k] for k in t1)
    assert bob.parse_targets(0.5, ["ccnews=1e6"])["ccnews"] == 500_000
    with pytest.raises(SystemExit):
        bob.parse_targets(1.0, ["nope=1"])


SMALL = ["--target", "ccnews=6000", "--target", "openwebtext=4000", "--target", "fineweb=3000",
         "--target", "wikipedia=2500", "--target", "bookcorpusopen=1500"]


def _digest(out):
    h = hashlib.sha256()
    for sd in sorted(p for p in out.iterdir() if p.is_dir()):
        h.update(sd.name.encode())
        h.update((sd / (sd.name + ".train")).read_bytes())
        h.update(gzip.open(sd / "docs.jsonl.gz").read())
    return h.hexdigest()


def test_targets_met_trimmed_and_deterministic_across_workers(world):
    out1, m1 = run(world, "det1", workers=1, shards=3, targets=SMALL)
    out2, m2 = run(world, "det2", workers=3, shards=3, targets=SMALL)   # forked Pool path
    assert _digest(out1) == _digest(out2)
    for s, tgt in (("ccnews", 6000), ("openwebtext", 4000), ("fineweb", 3000), ("wikipedia", 2500), ("bookcorpusopen", 1500)):
        sm = m1["sources"][s]
        assert sm["target_met"] and tgt <= sm["words_kept"] < tgt + 1000, (s, sm["words_kept"])   # overshoot < 1 doc
        assert sm["docs_trimmed_over_target"] >= 0
    # FineWeb: highest scores first -> the kept docs are the story docs
    _, docs = read_shards(out1)
    assert all(r["doc_key"].startswith("story-") for _, r, _ in docs if r["source"] == "fineweb")


def test_restart_skips_finished_work(world, capfd):
    out, m1 = run(world, "restart", targets=SMALL, shards=3)
    capfd.readouterr()
    shard_files = sorted(out.glob("shard_*/*.train"))
    mt = [p.stat().st_mtime_ns for p in shard_files]
    d1 = _digest(out)
    out, m2 = run(world, "restart", targets=SMALL, shards=3)
    logs = capfd.readouterr().out
    assert "BabyLM hash set: cached" in logs and "(cached)" in logs and "cached 3" in logs
    assert "news domains: cached" in logs
    assert [p.stat().st_mtime_ns for p in shard_files] == mt and _digest(out) == d1
    assert m1["sources"]["ccnews"]["words_kept"] == m2["sources"]["ccnews"]["words_kept"]


def test_dry_run_limit(world):
    _, m = run(world, "dry", extra=["--dry-run-limit", "10"], shards=2)
    for s, sm in m["sources"].items():
        assert 0 < sm["docs_read"] <= 10 and sm["files_processed"] == 1, (s, sm["docs_read"])
    assert m["params"]["dry_run_limit"] == 10


def test_corrupt_file_is_reported_not_fatal(world):
    bad = world["raw"] / "wikipedia_20231101_en" / "20231101.en" / "train-00099-of-00099.parquet"
    bad.write_bytes(b"PAR1 this is not a parquet file")
    try:
        _, m = run(world, "corrupt", shards=2)
    finally:
        bad.unlink()
    assert any("train-00099" in f["file"] for f in m["sources"]["wikipedia"]["files_failed"])
    assert m["sources"]["wikipedia"]["docs_kept"] > 0


def test_fineweb_weights_cli():
    assert bob.parse_weights(None) == {}
    assert bob.parse_weights("quote=2,ed=0.5") == {"quote": 2.0, "ed": 0.5}
    with pytest.raises(SystemExit):
        bob.parse_weights("pronoun=3")


def test_early_stop_and_cancel_are_deterministic(world):
    """Many small files, tiny target: reading stops after a prefix of the hash-ordered
    file list; look-ahead units are cancelled and the result does not depend on workers."""
    rng = random.Random(11)
    raw = world["root"] / "raw_many"
    (raw / "ccnews_en_2016_2019").mkdir(parents=True, exist_ok=True)
    for f in range(10):
        texts = [expository(rng, world["vocab"], rng.randint(80, 200)) for _ in range(30)]
        pq.write_table(pa.table({"requested_url": ["http://m/%d/%d" % (f, i) for i in range(30)],
                                 "plain_text": texts}),
                       raw / "ccnews_en_2016_2019" / ("2017_%04d.parquet" % f))

    def go(tag, workers):
        out = world["root"] / ("out_" + tag) / "base"
        argv = ["--raw-root", str(raw), "--babylm-dirs", *world["babylm"], "--out", str(out), "--shards", "2",
                "--workers", str(workers), "--target", "ccnews=3000", "--target", "openwebtext=0",
                "--target", "fineweb=0", "--target", "wikipedia=0", "--target", "bookcorpusopen=0",
                "--spread", "1", "--min-files", "10"]
        assert bob.main(argv) == 0
        return out, json.loads((out / "BUILD_MANIFEST.json").read_text())

    o1, m1 = go("early1", 1)
    o3, m3 = go("early3", 3)
    cc = m1["sources"]["ccnews"]
    assert cc["target_met"] and cc["files_processed"] < cc["files_available"] == 10
    assert m3["sources"]["ccnews"]["files_processed"] == cc["files_processed"]
    assert _digest(o1) == _digest(o3)
    assert not list((world["root"] / "out_early3").rglob("*.tmp"))          # cancelled units cleaned up
