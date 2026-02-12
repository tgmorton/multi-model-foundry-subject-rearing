"""FastAPI app for the annotation review interface.

Run with: python -m review.server
"""

import os
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .corpus_api import CorpusQueryAPI
from .notes_db import NotesStore
from .schematics import FEATURE_SCHEMATICS

# Corpus path - configurable via env var
CORPUS_PATH = os.environ.get(
    "REVIEW_CORPUS_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "output" / "test_10M" / "annotated_corpus"),
)

app = FastAPI(title="Annotation Review", version="0.1.0")

# Initialize on startup
corpus: Optional[CorpusQueryAPI] = None
notes: Optional[NotesStore] = None


@app.on_event("startup")
async def startup():
    global corpus, notes
    corpus = CorpusQueryAPI(CORPUS_PATH)
    notes = NotesStore()


# --- Pydantic models ---

class NotePayload(BaseModel):
    feature_context: str = ""
    verdict: str
    note_text: str = ""
    active_filters: Optional[Dict] = None


# --- API routes ---

@app.get("/api/metadata")
async def get_metadata():
    return corpus.get_metadata()


@app.get("/api/sentences")
async def list_sentences(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    genre: Optional[str] = None,
    speaker: Optional[str] = None,
    flags: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: Optional[str] = None,
    hide_reviewed: bool = False,
):
    flag_list = [f.strip() for f in flags.split(",") if f.strip()] if flags else None
    exclude_ids = list(notes.reviewed_ids) if hide_reviewed else None
    result = corpus.query_sentences(
        page=page,
        page_size=page_size,
        genre=genre,
        speaker=speaker,
        flags=flag_list,
        search=search,
        sort_by=sort_by,
        exclude_ids=exclude_ids,
    )
    # Attach note verdicts for this page
    sentence_ids = [s["sentence_id"] for s in result["sentences"]]
    sentence_notes = notes.get_notes_for_sentences(sentence_ids)
    for s in result["sentences"]:
        note = sentence_notes.get(s["sentence_id"])
        s["verdict"] = note["verdict"] if note else None
    result["reviewed_count"] = notes.count
    return result


@app.get("/api/sentences/{sentence_id}")
async def get_sentence_detail(sentence_id: str):
    detail = corpus.get_sentence_detail(sentence_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Sentence not found")
    # Attach note if exists
    note = notes.get_note(sentence_id, "")
    if note is None:
        # Try finding any note for this sentence
        for (sid, ctx), n in notes._notes.items():
            if sid == sentence_id:
                note = n
                break
    detail["note"] = note
    return detail


@app.put("/api/notes/{sentence_id}")
async def save_note(sentence_id: str, payload: NotePayload):
    note = notes.upsert_note(
        sentence_id=sentence_id,
        feature_context=payload.feature_context,
        verdict=payload.verdict,
        note_text=payload.note_text,
        active_filters=payload.active_filters,
    )
    return note


@app.get("/api/notes")
async def list_notes(
    feature_context: Optional[str] = None,
    verdict: Optional[str] = None,
):
    return notes.get_all_notes(feature_context=feature_context, verdict=verdict)


@app.get("/api/notes/enriched")
async def list_notes_enriched(
    feature_context: Optional[str] = None,
    verdict: Optional[str] = None,
):
    """Return all notes with sentence text attached."""
    all_notes = notes.get_all_notes(feature_context=feature_context, verdict=verdict)
    if not all_notes:
        return []
    sentence_ids = [n["sentence_id"] for n in all_notes]
    texts = corpus.get_sentence_texts(sentence_ids)
    for n in all_notes:
        n["text"] = texts.get(n["sentence_id"], "")
    return all_notes


@app.get("/api/notes/export")
async def export_notes():
    content = notes.export_tsv()
    return Response(
        content=content,
        media_type="text/tab-separated-values",
        headers={"Content-Disposition": "attachment; filename=review_notes.tsv"},
    )


@app.get("/api/schematics")
async def list_schematics():
    return {
        name: {"title": s["title"], "summary": s["summary"]}
        for name, s in FEATURE_SCHEMATICS.items()
    }


@app.get("/api/schematics/{feature}")
async def get_schematic(feature: str):
    s = FEATURE_SCHEMATICS.get(feature)
    if s is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return s


# --- Static files ---

_static_dir = Path(__file__).parent / "static"


@app.get("/")
async def index():
    index_file = _static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Annotation Review</h1><p>Static files not found.</p>")


@app.get("/reviewed")
async def reviewed_page():
    reviewed_file = _static_dir / "reviewed.html"
    if reviewed_file.exists():
        return FileResponse(reviewed_file)
    return HTMLResponse("<h1>Not found</h1>")


app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# --- Main entry point ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8642)
