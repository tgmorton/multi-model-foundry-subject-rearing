"""Step 5: blind KWIC concordance for human reading (round R1).

Bands (robbi): B0 = d0-2, B1 = d3-5, B6, B7, B8, B9a (d9, <= 3 nats),
B9b (3-6], B9c (> 6). Sample 8 bands x 6 genres x 6 items, person quotas
{i, you, he/she, it, we, they} one each where available (shortfalls are
filled from the other classes in the same cell). Across the whole sample:
no two items share a normalized line, and at most one item per document
(doc_id; 200-line block for genres without document markers). Candidate order
is a stable hash of (salt, genre, line_idx, token_i), so the draw is
deterministic.

Outputs
  concordance_R1.jsonl      id, genre, target line split around the slot,
                            4 previous + 2 next raw lines (speaker tags kept,
                            same document only, blank lines skipped)
  concordance_R1.html       self-contained; slot blank by default, click or
                            toggle to reveal. No decile / band / score.
  concordance_R1_key.csv    id -> decile, band, s_robbi, form, genre,
                            line_idx, token_i  (the hidden answer key)
  concordance_R1.meta.json  achieved counts per band x genre (no item ids)
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survey_common as C  # noqa: E402

CLASSES = ["i", "you", "he/she", "it", "we", "they"]
CLASS_OF = {"i": "i", "you": "you", "he": "he/she", "she": "he/she", "it": "it",
            "we": "we", "they": "they"}
N_PREV, N_NEXT = 4, 2
CTX_CHARS, LINE_CHARS = 420, 700


def item_id(salt, g, li, ti):
    return hashlib.blake2b(f"{salt}|{g}|{li}|{ti}".encode(), digest_size=5).hexdigest()


def clip_left(s, n):
    return s if len(s) <= n else "…" + s[-n:]


def clip_right(s, n):
    return s if len(s) <= n else s[:n] + "…"


def select(cands: pd.DataFrame, per_cell: int):
    used_line, used_doc = set(), set()
    chosen = []
    for band in C.BANDS:
        for g in C.GENRES:
            cell = cands[(cands.band == band) & (cands.genre == g)]
            pools = {c: cell[cell.pclass == c] for c in CLASSES}
            ptr = {c: 0 for c in CLASSES}
            picked = []

            def take(c):
                pool = pools[c]
                while ptr[c] < len(pool):
                    r = pool.iloc[ptr[c]]
                    ptr[c] += 1
                    if r.line_norm_hash in used_line or r.cluster_id in used_doc:
                        continue
                    used_line.add(r.line_norm_hash)
                    used_doc.add(r.cluster_id)
                    return r
                return None

            for c in CLASSES:                       # one per class
                r = take(c)
                if r is not None:
                    picked.append(r)
            k = 0
            while len(picked) < per_cell:           # fill shortfalls
                live = [c for c in CLASSES if ptr[c] < len(pools[c])]
                if not live:
                    break
                r = take(live[k % len(live)])
                k += 1
                if r is not None:
                    picked.append(r)
            chosen += picked
    return pd.DataFrame(chosen)


def context(lines, is_hdr, li, step, n):
    out, j = [], li + step
    while 0 <= j < len(lines) and len(out) < n:
        if is_hdr(lines[j]):
            break
        if lines[j].strip():
            out.append(lines[j])
        j += step
    return out[::-1] if step < 0 else out


HTML_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Subject Slot Concordance</title>
<style>
:root{--bg:#fbfaf7;--card:#ffffff;--ink:#1d1d1b;--muted:#77756f;--line:#e4e1da;
--accent:#2f5d8a;--slot:#eef2f7;--reveal:#9b3d1c;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161616;
--card:#1f1f1f;--ink:#e9e6df;--muted:#9a978f;--line:#33322f;--accent:#8db4dc;
--slot:#26303b;--reveal:#f0a27e;}}
:root[data-theme="dark"]{--bg:#161616;--card:#1f1f1f;--ink:#e9e6df;--muted:#9a978f;
--line:#33322f;--accent:#8db4dc;--slot:#26303b;--reveal:#f0a27e;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.55 Georgia,"Iowan Old Style","Palatino Linotype",serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);
padding:12px 16px}
header h1{font:600 18px/1.3 -apple-system,system-ui,sans-serif;margin:0 0 6px}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;
font:14px -apple-system,system-ui,sans-serif;color:var(--muted)}
.controls button,.controls select{font:inherit;color:var(--ink);background:var(--card);
border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
main{max-width:860px;margin:0 auto;padding:12px 16px 60px}
.intro{font:14px/1.5 -apple-system,system-ui,sans-serif;color:var(--muted);margin:8px 0 18px}
article{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 16px;margin:0 0 14px}
.meta{font:12px -apple-system,system-ui,sans-serif;color:var(--muted);margin-bottom:6px;
display:flex;gap:10px}
.meta .num{color:var(--accent);font-weight:600}
.ctx{color:var(--muted);font-size:14.5px;margin:2px 0;overflow-wrap:anywhere}
.tgt{margin:6px 0;font-size:17px;overflow-wrap:anywhere}
.tag{font:12.5px ui-monospace,Menlo,monospace;color:var(--muted);margin-right:6px}
.slot{display:inline-block;min-width:3.2em;text-align:center;background:var(--slot);
border-bottom:2px solid var(--accent);border-radius:3px;padding:0 4px;cursor:pointer}
.slot .p{display:none;color:var(--reveal);font-weight:700}
.slot.on .p,body.all .slot .p{display:inline}
.slot.on .b,body.all .slot .b{display:none}
.hidden{display:none}
</style></head><body>
"""

HTML_JS = """<script>
(function(){
  var body=document.body;
  document.getElementById('all').addEventListener('click',function(){
    body.classList.toggle('all');
    this.textContent=body.classList.contains('all')?'Hide pronouns':'Reveal all pronouns';});
  document.querySelectorAll('.slot').forEach(function(s){
    s.addEventListener('click',function(){s.classList.toggle('on');});});
  var sel=document.getElementById('genre');
  sel.addEventListener('change',function(){
    var v=sel.value,n=0;
    document.querySelectorAll('article').forEach(function(a){
      var show=!v||a.dataset.genre===v; a.classList.toggle('hidden',!show); if(show)n++;});
    document.getElementById('count').textContent=n+' items';});
})();
</script></body></html>
"""


def render_html(items):
    e = html.escape
    genres = sorted({it["genre"] for it in items})
    P = [HTML_HEAD,
         "<header><h1>Subject slot concordance &middot; round R1</h1>"
         "<div class='controls'><button id='all'>Reveal all pronouns</button>"
         "<label>Genre <select id='genre'><option value=''>all</option>"
         + "".join(f"<option>{e(g)}</option>" for g in genres) +
         f"</select></label><span id='count'>{len(items)} items</span></div></header>",
         "<main><p class='intro'>Each item shows up to four preceding lines, the "
         "target line with one subject pronoun replaced by a blank, and up to two "
         "following lines, as they appear in the training text (speaker tags kept). "
         "Click a blank to reveal its pronoun. Items are in random order.</p>"]
    for k, it in enumerate(items, 1):
        P.append(f"<article data-genre='{e(it['genre'])}'><div class='meta'>"
                 f"<span class='num'>#{k}</span><span>{e(it['genre'])}</span>"
                 f"<span>id {e(it['id'])}</span></div>")
        for c in it["prev"]:
            P.append(f"<div class='ctx'>{e(c)}</div>")
        tag = f"<span class='tag'>{e(it['tag'])}</span>" if it["tag"] else ""
        P.append(f"<div class='tgt'>{tag}{e(it['left'])}<span class='slot'>"
                 f"<span class='b'>&#8239;&#8239;&#8239;&#8239;</span>"
                 f"<span class='p'>{e(it['slot'])}</span></span>{e(it['right'])}</div>")
        for c in it["next"]:
            P.append(f"<div class='ctx'>{e(c)}</div>")
        P.append("</article>")
    P.append("</main>")
    P.append(HTML_JS)
    return "\n".join(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--salt", default="R1")
    ap.add_argument("--per-cell", type=int, default=6)
    a = ap.parse_args()
    P = C.default_paths(a.data_root)
    out = P["out"]
    d = pd.read_parquet(out / "population_features.parquet", columns=[
        "genre", "line_idx", "token_i", "form", "d_robbi", "s_robbi", "cluster_id",
        "line_norm_hash", "char_start", "char_end", "content_start", "tok_ok"])
    d = d[d.tok_ok].copy()
    d["pclass"] = d.form.map(CLASS_OF)
    d = d[d.pclass.notna()]
    d["band"] = C.band_of(d.d_robbi.to_numpy(), d.s_robbi.to_numpy())
    d["h"] = C.stable_hash([f"{a.salt}|{g}|{l}|{t}" for g, l, t in
                            zip(d.genre, d.line_idx, d.token_i)])
    d = d.sort_values(["band", "genre", "pclass", "h"], kind="stable")
    ch = select(d, a.per_cell)
    items, keys = [], []
    for g in C.GENRES:
        sub = ch[ch.genre == g]
        if sub.empty:
            continue
        lines = C.read_lines(P["raw"], g)
        hdr = ((lambda s: C.HEADER_RE.match(s) is not None)
               if g in C.DOC_GENRES else (lambda s: False))
        for r in sub.itertuples():
            line = lines[r.line_idx]
            iid = item_id(a.salt, g, r.line_idx, r.token_i)
            cs, s0, s1 = int(r.content_start), int(r.char_start), int(r.char_end)
            half = LINE_CHARS // 2
            items.append({
                "id": iid, "genre": g, "tag": line[:cs].strip(),
                "left": clip_left(line[cs:s0], half), "slot": line[s0:s1],
                "right": clip_right(line[s1:], half),
                "kwic": clip_left(line[cs:s0], half) + "[___]" + clip_right(line[s1:], half),
                "prev": [clip_left(x, CTX_CHARS) for x in
                         context(lines, hdr, r.line_idx, -1, N_PREV)],
                "next": [clip_right(x, CTX_CHARS) for x in
                         context(lines, hdr, r.line_idx, +1, N_NEXT)]})
            keys.append({"id": iid, "decile": int(r.d_robbi), "band": r.band,
                         "s_robbi": float(r.s_robbi), "form": r.form, "genre": g,
                         "line_idx": int(r.line_idx), "token_i": int(r.token_i)})
        del lines
    items.sort(key=lambda x: x["id"])
    assert len({x["id"] for x in items}) == len(items)
    with open(out / "concordance_R1.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out / "concordance_R1.html").write_text(render_html(items), encoding="utf-8")
    K = pd.DataFrame(keys).sort_values("id")
    K.to_csv(out / "concordance_R1_key.csv", index=False)
    cnt = K.groupby(["band", "genre"]).size().unstack(fill_value=0).reindex(C.BANDS)
    pcl = (K.assign(pc=K.form.map(CLASS_OF)).groupby(["band", "pc"]).size()
           .unstack(fill_value=0).reindex(C.BANDS))
    meta = {"n_items": len(K), "salt": a.salt,
            "per_band_genre": {b: cnt.loc[b].to_dict() for b in C.BANDS},
            "per_band_class": {b: pcl.loc[b].to_dict() for b in C.BANDS},
            "unique_docs": int(ch.cluster_id.nunique()),
            "unique_lines": int(ch.line_norm_hash.nunique())}
    (out / "concordance_R1.meta.json").write_text(json.dumps(meta, indent=2))
    print(cnt)
    print(pcl)
    print(f"{len(K)} items")


if __name__ == "__main__":
    main()
