# Annotation Web App

A self-contained FastAPI service for collecting human null-subject annotations
on Spanish sentences, measuring inter-annotator agreement, and adjudicating
disagreements into a gold-standard JSONL output.

The app lives in `annotation/` and is independent of the rest of the repo: it
has its own Dockerfile, SQLite database, and frontend. It produces gold
annotations in the marker format consumed by the pronoun-recovery pipeline.

## 1. Purpose

The project needs human-labelled Spanish data where annotators decide, for
each stimulus sentence:

1. Does the sentence contain a null subject?
2. If so, at which word-gap position(s) would an overt pronoun be inserted?
3. Which pronoun (PRO.1sg ... PRO.3pl) is recovered?

Two annotators label each stimulus, an admin adjudicates disagreements, and
the app exports a gold JSONL in the same `{clean_text, markers, ...}` schema
used by the tree detector and BERT sequence labellers documented in
`docs/pronoun_recovery.md`. This gold set is the hand-curated test data called
out in the Italian null subject roadmap and the analogue for Spanish.

## 2. Architecture

### Stack

- **FastAPI** (not Flask — the docstrings say Flask but the code uses FastAPI)
  served by **uvicorn**.
- **SQLite** via the standard-library `sqlite3` module (no SQLAlchemy).
- **Pydantic** request/response models.
- Static HTML/JS frontend (no build step).
- **nginx** + **certbot** in front for TLS in production via `docker-compose`.

### Module layout

| File                              | Role |
|-----------------------------------|------|
| `annotation/__init__.py`          | Package marker. |
| `annotation/__main__.py`          | `python -m annotation` entry point. Parses `--host/--port/--seed-users`, calls `db.init_db()`, optionally seeds users and prints their tokens, then launches `uvicorn`. |
| `annotation/server.py`            | FastAPI app, all HTTP routes, static file mounts. |
| `annotation/db.py`                | SQLite schema (`_SCHEMA`), connection helper, and all query functions (users, stimuli, annotations, adjudications, progress, queue). |
| `annotation/models.py`            | Pydantic request/response schemas (`LoginRequest`, `AnnotationRequest`, `PlacedPronounRequest`, `AdjudicationRequest`, ...). |
| `annotation/auth.py`              | Bearer-token dependency `get_current_user` and `require_admin` role gate. |
| `annotation/constants.py`         | Spanish pronoun choice list `ES_PRONOUN_CHOICES`, `VALID_LABELS`, label-to-form maps. |
| `annotation/agreement.py`         | Cohen's kappa + Jaccard computations over doubly-annotated stimuli. |
| `annotation/export.py`            | Gold and raw JSONL exporters; word-gap to character-offset conversion. |
| `annotation/static/index.html`    | Annotator UI (login + main labelling interface). |
| `annotation/static/admin.html`    | Admin dashboard (users, stimuli loading, agreement status). |
| `annotation/static/adjudication.html` | Side-by-side disagreement resolution UI. |

### Database schema

All tables live in a single SQLite file at `annotation/data/annotation.db`
(created by `db.init_db()` at `annotation/db.py:82`). Schema source of truth
is `_SCHEMA` at `annotation/db.py:12`.

| Table                     | Key columns | Notes |
|---------------------------|-------------|-------|
| `users`                   | `username` unique, `token` unique, `role` (`annotator`\|`admin`) | Token is a UUID generated at seed time; no passwords. |
| `stimuli`                 | `text`, `source`, `context_before`, `context_after`, `metadata`, `ordering` | `context_before/after` stored as JSON arrays of sentences, nearest-first. |
| `annotations`             | `(stimulus_id, user_id)` unique, `has_null_subject`, `overall_confidence`, `starred`, `note`, `context_expansions` | One row per annotator per stimulus. |
| `placed_pronouns`         | `annotation_id` FK, `position` (word-gap index), `label`, `lexical_form`, `confidence`, `ordering` | Children of an annotation; deleted and re-inserted on upsert. |
| `adjudications`           | `stimulus_id` unique, `adjudicator_id`, `resolution`, `accepted_user_id`, `has_null_subject`, ... | Exactly one adjudication per stimulus. |
| `adjudication_pronouns`   | `adjudication_id` FK + position/label/lexical_form/confidence | Mirror of `placed_pronouns` for adjudicated gold. |

SQLite is opened with `journal_mode=WAL` and `foreign_keys=ON`.

## 3. Deployment

### Local development

```bash
# From the repo root
python -m annotation --port 8643 --host 127.0.0.1

# Seed users on first run (tokens printed to stdout)
python -m annotation --seed-users '[
  {"username":"admin","display_name":"Admin","role":"admin"},
  {"username":"alice","display_name":"Alice","role":"annotator"},
  {"username":"bob","display_name":"Bob","role":"annotator"}
]'
```

`db.init_db()` creates `annotation/data/annotation.db` if missing. The
`annotation/data/` directory is gitignored; do not commit the database.

Seed users can also be supplied via the `ANNOTATION_SEED_USERS` environment
variable containing the same JSON list. Re-seeding is idempotent: users with
an existing `username` are skipped (see `db.seed_users` at
`annotation/db.py:114`).

Browse to:

- `http://127.0.0.1:8643/` — annotator UI
- `http://127.0.0.1:8643/admin` — admin dashboard (admin token required)
- `http://127.0.0.1:8643/adjudication` — adjudication UI (admin token required)

### Docker / docker-compose

`annotation/Dockerfile` builds a `python:3.11-slim` image, installs
`fastapi`, `uvicorn`, and `python-multipart`, and runs
`python -m annotation --host 0.0.0.0 --port 8643`.

`annotation/docker-compose.yaml` defines three services:

| Service    | Purpose |
|------------|---------|
| `annotation` | The FastAPI app. Mounts named volume `annotation-data` at `/app/annotation/data` so the SQLite DB persists across rebuilds. Seeds three users from `ANNOTATION_SEED_USERS` on first boot. |
| `nginx`      | Reverse proxy on ports 80/443, config at `annotation/nginx.conf`. Redirects HTTP to HTTPS and proxies to `annotation:8643`. |
| `certbot`    | Let's Encrypt sidecar that renews certificates every 12h. |

```bash
cd annotation
docker compose up -d --build

# Find the token for the seeded admin
docker compose exec annotation python -c "
from annotation import db
print(db.get_user_by_username('admin')['token'])
"
```

`annotation/nginx.conf` has a placeholder `YOURDOMAIN` in the `ssl_certificate`
paths; this must be replaced before `nginx` will serve HTTPS. The
`client_max_body_size` is set to 10M to accommodate stimuli uploads.

### Authentication

Auth is **bearer-token only**, no passwords. A user's `token` is a UUID
generated when they are seeded (`annotation/db.py:129`). The flow is:

1. Admin seeds users and shares each user's token out-of-band.
2. The frontend calls `POST /api/auth/login` with `{username, token}` which
   round-trips the token (see `server.py:43`).
3. Subsequent requests send `Authorization: Bearer <token>` and are validated
   by `get_current_user` at `annotation/auth.py:13`. Admin-only routes use
   `require_admin` (`annotation/auth.py:26`).

There is no token rotation, expiry, or revocation endpoint. Tokens are
effectively long-lived shared secrets; treat the seed list and the SQLite DB
as credentials.

## 4. User workflows

### Annotator (`/`, `static/index.html`)

1. Log in with username + token.
2. Browse a paginated list of stimuli (`GET /api/stimuli`) or jump to starred
   items (`GET /api/stimuli/starred`).
3. Open a stimulus (`GET /api/stimuli/{id}`). The response includes the user's
   own existing annotation and placed pronouns, if any.
4. Decide whether the sentence has a null subject; if yes, click between words
   to place pronoun markers at word-gap positions and pick a pronoun form from
   the choices returned by `GET /api/pronouns` (derived from
   `constants.ES_PRONOUN_CHOICES`).
5. Set `overall_confidence` (1–5), optionally `starred`/`note`, and submit via
   `PUT /api/annotations/{id}`.
6. Progress is tracked by `GET /api/annotations/progress`.

### Admin (`/admin`, `static/admin.html`)

- List users: `GET /api/admin/users`.
- Upload stimuli from a JSONL or plain-text file: `POST /api/stimuli/load`.
  JSONL lines accept `{text, source, context_before, context_after, metadata,
  ordering}`; plain lines become `{text, ordering}` (see `server.py:94`).
- Monitor per-annotator progress: `GET /api/agreement/status`.
- Compute and download agreement metrics: `GET /api/agreement/compute` and
  `GET /api/agreement/export`.
- Export final data: `GET /api/export/gold`, `GET /api/export/raw`.

### Adjudication (`/adjudication`, `static/adjudication.html`)

1. `GET /api/adjudication/queue` returns stimuli where two annotators disagree,
   either on the binary `has_null_subject` decision or on the set of
   `(position, label)` pairs (logic in `db.get_adjudication_queue` at
   `annotation/db.py:470`).
2. `GET /api/adjudication/{id}` returns the stimulus plus every annotator's
   full annotation for side-by-side display.
3. The adjudicator picks a resolution — `accept_a`, `accept_b`, or
   `reannotate` — and submits `PUT /api/adjudication/{id}`. Only these three
   values are accepted (`server.py:245`).
4. `GET /api/adjudication/progress` returns `{total, adjudicated, remaining}`.

## 5. API endpoints

All routes are defined in `annotation/server.py`. Admin routes require the
caller's user to have `role == "admin"`.

### Auth

| Method | Path               | Auth | Purpose |
|--------|--------------------|------|---------|
| POST   | `/api/auth/login`  | —    | Exchange `{username, token}` for the user record. |
| GET    | `/api/auth/me`     | user | Return the current user. |

### Stimuli

| Method | Path                          | Auth  | Purpose |
|--------|-------------------------------|-------|---------|
| GET    | `/api/stimuli`                | user  | Paginated list (`page`, `page_size` ≤ 200) with the caller's annotation state joined in. |
| GET    | `/api/stimuli/starred`        | user  | Stimuli the caller has starred. |
| GET    | `/api/stimuli/{stimulus_id}`  | user  | Single stimulus with caller's annotation + pronouns. 404 if missing. |
| POST   | `/api/stimuli/load`           | admin | Multipart upload of JSONL or plain-text stimuli. Returns `{loaded}`. |
| GET    | `/api/pronouns`               | —     | Spanish pronoun choice list for the UI. |

### Annotations

| Method | Path                          | Auth | Purpose |
|--------|-------------------------------|------|---------|
| PUT    | `/api/annotations/{stimulus_id}` | user | Upsert the caller's annotation. Validates labels against `VALID_LABELS`. |
| GET    | `/api/annotations/progress`   | user | `{total, completed, starred, remaining}`. |

### Agreement (admin)

| Method | Path                      | Purpose |
|--------|---------------------------|---------|
| GET    | `/api/agreement/status`   | Per-annotator progress. |
| GET    | `/api/agreement/compute`  | Full metrics from `compute_agreement()`. |
| GET    | `/api/agreement/export`   | CSV of per-stimulus agreement rows wrapped in a JSON `{csv}` field. |

### Adjudication (admin)

| Method | Path                                | Purpose |
|--------|-------------------------------------|---------|
| GET    | `/api/adjudication/queue`           | Stimuli with unresolved disagreements. |
| GET    | `/api/adjudication/progress`        | `{total, adjudicated, remaining}`. |
| GET    | `/api/adjudication/{stimulus_id}`   | Stimulus + all annotations for side-by-side view. |
| PUT    | `/api/adjudication/{stimulus_id}`   | Save adjudication resolution and gold pronouns. |

### Export (admin)

| Method | Path               | Purpose |
|--------|--------------------|---------|
| GET    | `/api/export/gold` | Gold JSONL records (see §7). |
| GET    | `/api/export/raw`  | Every raw annotation by every annotator. |

### Admin

| Method | Path                  | Purpose |
|--------|-----------------------|---------|
| GET    | `/api/admin/users`    | List all users. |

### Static pages

| Path             | Serves |
|------------------|--------|
| `/`              | `static/index.html` (annotator) |
| `/admin`         | `static/admin.html` |
| `/adjudication`  | `static/adjudication.html` |
| `/static/*`      | All files under `annotation/static/`. |

## 6. Inter-annotator agreement

`annotation/agreement.py` implements `compute_agreement()`, invoked by
`GET /api/agreement/compute` and `GET /api/agreement/export`. It:

1. Finds every stimulus with ≥ 2 annotations, takes the first two by
   `user_id`.
2. Computes **binary** percent agreement and Cohen's kappa on
   `has_null_subject`.
3. For stimuli where both annotators said "yes", computes:
   - **Position exact-match rate**: fraction of double-yes stimuli where the
     sets of word-gap positions are identical.
   - **Mean Jaccard** over position sets.
   - **Label exact-match rate** and **Cohen's kappa** over pronoun labels at
     positions both annotators marked.
4. Classifies each stimulus into `full_agreement`, `binary_disagreement`,
   `position_disagreement`, or `label_disagreement` for the `per_stimulus`
   breakdown.

Cohen's kappa is implemented in `_cohens_kappa` at `annotation/agreement.py:8`
without external dependencies. The return shape:

```json
{
  "n_doubly_annotated": 123,
  "binary":  {"percent_agreement": 0.93, "kappa": 0.81},
  "position":{"exact_match_rate": 0.88, "mean_jaccard": 0.92},
  "label":   {"exact_match_rate": 0.90, "kappa": 0.85},
  "per_stimulus": [ {"stimulus_id": 1, "category": "...", ...}, ... ]
}
```

`GET /api/agreement/export` re-uses the same function and emits a CSV with
columns `stimulus_id, category, annotator_a, annotator_b, a_decision,
b_decision` inside a JSON wrapper. (The `Content-Disposition` header is set
but the body is JSON, not raw CSV — see §8.)

## 7. Export

`annotation/export.py` produces two formats.

### `export_gold()` — `GET /api/export/gold`

Returns a list of records in the pronoun-recovery marker format. Assembly
rules (`export.py:71`):

1. Every stimulus with an adjudication is included, tagged
   `"annotator_agreement": "adjudicated"`.
2. Every non-adjudicated stimulus with ≥ 2 annotations where both annotators
   agreed on the binary decision is included using the first annotator's
   pronouns, tagged `"annotator_agreement": "full"`.
3. Disagreed, non-adjudicated stimuli are skipped.

Each record looks like:

```json
{
  "clean_text": "Hoy comimos paella.",
  "markers": [
    {
      "label": "PRO.1pl",
      "lexical_form": "nosotros",
      "position": 4,
      "confidence": "high"
    }
  ],
  "source": "human_annotation",
  "language": "es",
  "genre": "europarl",
  "stimulus_id": 42,
  "annotator_agreement": "full"
}
```

`position` is a **character offset** (not a word-gap index); the conversion
happens in `_word_gap_to_char_offset` at `export.py:17`. `confidence` is
mapped from the 1–5 integer to `high` (≥4), `medium` (2–3), or `low` (1) by
`_confidence_label` at `export.py:8`.

### `export_raw()` — `GET /api/export/raw`

Returns every annotation from every annotator verbatim: `stimulus_id`,
`clean_text`, `annotator`, `has_null_subject`, `overall_confidence`,
`starred`, `note`, and the list of pronouns with their word-gap `position`,
`label`, `lexical_form`, and integer `confidence`. Use this for audit and
for training data where disagreements matter.

## 8. Known oddities

A few things flagged while reading the source:

- **Flask vs FastAPI.** The top-level docstring in `annotation/server.py:1`
  calls the app a "FastAPI app" but `annotation/__init__.py:1` and the task
  description say Flask. The code is actually FastAPI + uvicorn.
- **`/api/export/gold` response.** `export_gold_endpoint` builds a JSONL
  string into `content` but then returns `records` (the list of dicts)
  inside a `JSONResponse`; the JSONL string is dead code. The
  `Content-Disposition` header suggests a file download but the body is JSON
  (`server.py:268`).
- **`/api/agreement/export` similarly** returns a JSON `{csv: "..."}` payload
  with a CSV `Content-Disposition` header; callers must unwrap the `csv`
  field client-side (`server.py:196`).
- **No token rotation or revocation.** Tokens are plaintext UUIDs stored in
  SQLite; leaking the DB file or the seed env var leaks every credential.
- **`nginx.conf` has `YOURDOMAIN` placeholders** that must be edited before
  HTTPS will work.
- **Adjudication queue uses `LIMIT 2`** on annotations ordered by `user_id`,
  so a third annotator is silently ignored for both queueing and agreement
  (`db.py:492`, `agreement.py:95`).
- **`/api/stimuli/load`** tries JSON per line first and falls back to plain
  text, but it does not de-duplicate against existing stimuli — re-uploading
  the same file inserts duplicates.
