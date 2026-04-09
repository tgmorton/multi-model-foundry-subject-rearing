"""FastAPI app for the Spanish null subject annotation tool.

Run with: python -m annotation [--port 8643]
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .agreement import compute_agreement
from .auth import get_current_user, require_admin
from .constants import ES_PRONOUN_CHOICES, VALID_LABELS
from .export import export_gold, export_raw
from .models import (
    AdjudicationRequest,
    AnnotationRequest,
    LoginRequest,
)

app = FastAPI(title="Spanish Null Subject Annotation", version="0.1.0")


@app.on_event("startup")
async def startup():
    db.init_db()
    # Seed default users from env if provided (JSON list)
    import os

    seed_json = os.environ.get("ANNOTATION_SEED_USERS")
    if seed_json:
        users = json.loads(seed_json)
        db.seed_users(users)


# ── Auth routes ──────────────────────────────────────────────────────


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    user = db.get_user_by_username(req.username)
    if user is None or user["token"] != req.token:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "token": user["token"],
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
        },
    }


@app.get("/api/auth/me")
async def me(user: Dict = Depends(get_current_user)):
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "role": user["role"],
    }


# ── Stimuli routes ───────────────────────────────────────────────────


@app.get("/api/stimuli")
async def list_stimuli(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    user: Dict = Depends(get_current_user),
):
    return db.get_stimuli_page(user["id"], page, page_size)


@app.get("/api/stimuli/starred")
async def starred_stimuli(user: Dict = Depends(get_current_user)):
    return db.get_starred_stimuli(user["id"])


@app.get("/api/stimuli/{stimulus_id}")
async def get_stimulus(stimulus_id: int, user: Dict = Depends(get_current_user)):
    result = db.get_stimulus(stimulus_id, user["id"])
    if result is None:
        raise HTTPException(status_code=404, detail="Stimulus not found")
    return result


@app.post("/api/stimuli/load")
async def load_stimuli(
    file: UploadFile = File(...),
    user: Dict = Depends(require_admin),
):
    """Load stimuli from a JSONL or plain text file.

    JSONL: each line is {"text": "...", "source": "...", "context_before": "...", ...}
    Plain text: each line is a stimulus sentence.
    """
    content = (await file.read()).decode("utf-8")
    lines = [l.strip() for l in content.strip().split("\n") if l.strip()]

    items = []
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
            obj.setdefault("ordering", i)
            items.append(obj)
        except json.JSONDecodeError:
            items.append({"text": line, "ordering": i})

    count = db.load_stimuli(items)
    return {"loaded": count}


# ── Pronoun choices (for frontend) ───────────────────────────────────


@app.get("/api/pronouns")
async def pronoun_choices():
    return [
        {"lexical_form": form, "label": label}
        for form, label in ES_PRONOUN_CHOICES
    ]


# ── Annotation routes ───────────────────────────────────────────────


@app.put("/api/annotations/{stimulus_id}")
async def save_annotation(
    stimulus_id: int,
    req: AnnotationRequest,
    user: Dict = Depends(get_current_user),
):
    # Validate stimulus exists
    stim = db.get_stimulus(stimulus_id, user["id"])
    if stim is None:
        raise HTTPException(status_code=404, detail="Stimulus not found")

    # Validate pronoun labels
    for p in req.pronouns:
        if p.label not in VALID_LABELS:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid label: {p.label}",
            )

    result = db.upsert_annotation(
        stimulus_id=stimulus_id,
        user_id=user["id"],
        has_null_subject=req.has_null_subject,
        pronouns=[p.model_dump() for p in req.pronouns],
        overall_confidence=req.overall_confidence,
        starred=req.starred,
        note=req.note,
        context_expansions=req.context_expansions,
    )
    return result


@app.get("/api/annotations/progress")
async def annotation_progress(user: Dict = Depends(get_current_user)):
    return db.get_annotation_progress(user["id"])


# ── Agreement routes (admin) ────────────────────────────────────────


@app.get("/api/agreement/status")
async def agreement_status(user: Dict = Depends(require_admin)):
    """Per-annotator progress."""
    users = db.list_users()
    annotators = [u for u in users if u["role"] == "annotator"]
    result = []
    for a in annotators:
        progress = db.get_annotation_progress(a["id"])
        result.append({
            "user_id": a["id"],
            "username": a["username"],
            "display_name": a["display_name"],
            **progress,
        })
    return result


@app.get("/api/agreement/compute")
async def agreement_compute(user: Dict = Depends(require_admin)):
    return compute_agreement()


@app.get("/api/agreement/export")
async def agreement_export(user: Dict = Depends(require_admin)):
    metrics = compute_agreement()
    rows = ["stimulus_id,category,annotator_a,annotator_b,a_decision,b_decision"]
    for item in metrics.get("per_stimulus", []):
        rows.append(
            f"{item['stimulus_id']},{item['category']},"
            f"{item.get('annotator_a','')},{item.get('annotator_b','')},"
            f"{item.get('a_decision','')},{item.get('b_decision','')}"
        )
    content = "\n".join(rows) + "\n"
    return JSONResponse(
        content={"csv": content},
        headers={"Content-Disposition": "attachment; filename=agreement.csv"},
    )


# ── Adjudication routes ─────────────────────────────────────────────


@app.get("/api/adjudication/queue")
async def adjudication_queue(user: Dict = Depends(require_admin)):
    return db.get_adjudication_queue()


@app.get("/api/adjudication/progress")
async def adjudication_progress(user: Dict = Depends(require_admin)):
    return db.get_adjudication_progress()


@app.get("/api/adjudication/{stimulus_id}")
async def get_adjudication(stimulus_id: int, user: Dict = Depends(require_admin)):
    stim = db.get_stimulus(stimulus_id, user["id"])
    if stim is None:
        raise HTTPException(status_code=404, detail="Stimulus not found")

    annotations = db.get_all_annotations_for_stimulus(stimulus_id)
    return {
        "stimulus": stim,
        "annotations": annotations,
    }


@app.put("/api/adjudication/{stimulus_id}")
async def save_adjudication(
    stimulus_id: int,
    req: AdjudicationRequest,
    user: Dict = Depends(require_admin),
):
    if req.resolution not in ("accept_a", "accept_b", "reannotate"):
        raise HTTPException(status_code=422, detail="Invalid resolution")

    for p in req.pronouns:
        if p.label not in VALID_LABELS:
            raise HTTPException(status_code=422, detail=f"Invalid label: {p.label}")

    result = db.upsert_adjudication(
        stimulus_id=stimulus_id,
        adjudicator_id=user["id"],
        resolution=req.resolution,
        accepted_user_id=req.accepted_user_id,
        has_null_subject=req.has_null_subject,
        pronouns=[p.model_dump() for p in req.pronouns],
        overall_confidence=req.overall_confidence,
        note=req.note,
    )
    return result


# ── Export routes ────────────────────────────────────────────────────


@app.get("/api/export/gold")
async def export_gold_endpoint(user: Dict = Depends(require_admin)):
    records = export_gold()
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    return JSONResponse(
        content=records,
        headers={"Content-Disposition": "attachment; filename=gold.jsonl"},
    )


@app.get("/api/export/raw")
async def export_raw_endpoint(user: Dict = Depends(require_admin)):
    records = export_raw()
    return JSONResponse(
        content=records,
        headers={"Content-Disposition": "attachment; filename=raw.jsonl"},
    )


# ── Admin user management ───────────────────────────────────────────


@app.get("/api/admin/users")
async def admin_list_users(user: Dict = Depends(require_admin)):
    return db.list_users()


# ── Static files ─────────────────────────────────────────────────────

_static_dir = Path(__file__).parent / "static"


@app.get("/")
async def index():
    index_file = _static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Annotation Tool</h1><p>Static files not found.</p>")


@app.get("/admin")
async def admin_page():
    f = _static_dir / "admin.html"
    if f.exists():
        return FileResponse(f)
    return HTMLResponse("<h1>Admin</h1><p>Not found.</p>")


@app.get("/adjudication")
async def adjudication_page():
    f = _static_dir / "adjudication.html"
    if f.exists():
        return FileResponse(f)
    return HTMLResponse("<h1>Adjudication</h1><p>Not found.</p>")


app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
