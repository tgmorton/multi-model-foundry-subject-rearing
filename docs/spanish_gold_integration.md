# Spanish Gold Annotation — Remote Deployment Runbook

Step-by-step instructions for integrating `data/spanish/gold/stimuli.jsonl`
into a remote annotation app deployment.

**Prerequisites knowledge**: you should read these first if unfamiliar:
- `docs/annotation_app.md` — architecture, auth, API surface
- `docs/spanish_gold_annotation.md` — sampling design, metadata schema

This runbook targets a **single-annotator** workflow (one admin + one
annotator). The app supports 2 annotators + adjudication as well; see
§7 if you need that path later.

## 1. Inputs

What you're deploying:

| File | Purpose | Size |
|------|---------|------|
| `data/spanish/gold/stimuli.jsonl` | Pre-built gold sample, ~974 stimuli | ~2–3 MB |
| `annotation/` | The FastAPI app itself (already self-contained) | — |

**The JSONL is ready for upload as-is.** Do not re-run the sampler on the
remote host — the stimulus file should be transferred verbatim so the
`sampler_commit` hash in metadata stays consistent with the local run.

## 2. Deploy the annotation app

### Option A: behind the host's existing nginx (subpath)

Assumes the remote host already runs nginx with TLS and you're mounting
the app at a subpath like `example.edu/annotation/`. All frontend URLs
are relative, so the app works at any subpath without code changes.

Install dependencies into a venv on the host:

```bash
cd /opt/annotation-repo
python3 -m venv .venv
.venv/bin/pip install -r annotation/requirements.txt
```

Run the app (bare uvicorn, listening on loopback only). Substitute
`<ANNOTATOR_USERNAME>` with the actual handle out-of-band:

```bash
.venv/bin/python -m annotation \
  --host 127.0.0.1 --port 8643 \
  --seed-users '[
    {"username":"admin","display_name":"Admin","role":"admin"},
    {"username":"<ANNOTATOR_USERNAME>","display_name":"<ANNOTATOR_DISPLAY>","role":"annotator"}
  ]'
```

Tokens print to stdout on first boot — grab them immediately and
share with the annotator out-of-band (Signal, 1Password, etc.).
Treat tokens as credentials; there is no rotation.

Add this to the host's nginx config (server block that terminates TLS):

```nginx
location = /annotation { return 301 /annotation/; }
location /annotation/ {
    proxy_pass http://127.0.0.1:8643/;       # trailing slashes on both sides
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

The trailing-slash redirect matters — relative URLs in the frontend
resolve against the document's URL, so `/annotation` (no slash) would
send API calls to the wrong path. The rate limiter middleware reads
`X-Forwarded-For`, so the `proxy_set_header` line keeps per-IP limits
meaningful.

Run it under systemd or tmux on the host; for systemd a minimal unit:

```ini
[Service]
User=annotation
WorkingDirectory=/opt/annotation-repo
Environment=ANNOTATION_SEED_USERS=[{"username":"admin",...}]
ExecStart=/opt/annotation-repo/.venv/bin/python -m annotation \
  --host 127.0.0.1 --port 8643
Restart=on-failure
```

### Option B: bundled docker-compose (self-managed nginx + certbot)

Use only if the remote host has no existing reverse proxy and you want
the bundled stack to handle TLS. Edit `annotation/nginx.conf` to replace
`YOURDOMAIN` with the real hostname, run certbot once to bootstrap certs,
then:

```bash
cd /path/to/repo/annotation
export ANNOTATION_SEED_USERS='[
  {"username":"admin","display_name":"Admin","role":"admin"},
  {"username":"<ANNOTATOR_USERNAME>","display_name":"<ANNOTATOR_DISPLAY>","role":"annotator"}
]'
docker compose up -d --build

# Retrieve tokens (substitute the annotator username):
docker compose exec annotation python -c "
from annotation import db
for u in ('admin', '<ANNOTATOR_USERNAME>'):
    user = db.get_user_by_username(u)
    print(f'{u}: {user[\"token\"]}')
"
```

### Verify the deploy

```bash
BASE_URL=https://example.edu/annotation   # or http://127.0.0.1:8643 for local

# Health check (no auth needed):
curl $BASE_URL/health
# → {"status":"ok"}

# Smoke tests (run on the remote host from the repo dir):
pytest annotation/tests/
# → 18 passed
```

## 3. Upload the stimuli

Copy `data/spanish/gold/stimuli.jsonl` to the remote host (the file is
gitignored, so it won't be in the repo there — `scp` it explicitly), then:

```bash
ADMIN_TOKEN=<paste admin token from step 2>

curl -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -F "file=@stimuli.jsonl" \
  $BASE_URL/api/stimuli/load
# → {"loaded": 974}
```

The endpoint reads each JSONL line as a stimulus dict. The JSONL
already includes `text`, `source`, `context_before`, `context_after`,
and `metadata` — all accepted by `db.load_stimuli`.

### Verify upload

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  "$BASE_URL/api/stimuli?page=1&page_size=5" | jq .
# Should show total=974, items with text/source populated.
```

Spot-check in a browser: log in as the annotator at `$BASE_URL/`,
confirm the UI shows the target sentence with context above/below.

## 4. Annotate

The annotator logs in at `$BASE_URL/` with their username + token and
works through the 974 stimuli. Key UI affordances:

- **Y/N keys** set the binary null-subject decision
- **Clicking a gap** between words opens the pronoun selection panel;
  pronoun choices are multi-select toggles (click to add, click again to
  remove). The pill in the sentence displays selected forms as
  `yo/él/nosotros`.
- **1–5 keys** set overall confidence
- **U** jumps to the next unannotated stimulus
- **D/F or ←/→** navigate prev/next; **S** stars the current item
- Saves are **automatic** on every change (visible "Saved" indicator).
  Auto-advance fires 500ms after a complete annotation on fresh items
  (but not when revisiting to correct).

### Progress monitoring

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  $BASE_URL/api/agreement/status | jq .
# → [{"user_id":2,"username":"alice","total":974,"completed":N,"starred":X,"remaining":974-N}]
```

(The endpoint is named for historical reasons — it returns per-annotator
progress, not inter-annotator agreement. With one annotator, use it as
a simple progress check.)

### Audit trail

Every save appends a row to the `annotation_history` table with the
full annotation state and a timestamp — useful for reconstructing when
and how labels evolved. Inspect it directly:

```bash
sqlite3 annotation/data/annotation.db \
  "SELECT stimulus_id, has_null_subject, saved_at FROM annotation_history ORDER BY saved_at DESC LIMIT 20"
```

## 5. Export gold

```bash
# Full gold JSON (all annotated stimuli):
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  $BASE_URL/api/export/gold -o gold_es.json

# Raw export (every annotation verbatim, no export-logic transforms):
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  $BASE_URL/api/export/raw -o raw_es.json
```

Gold record format:

```json
{
  "clean_text": "Me alegro de la acogida tan favorable.",
  "markers": [
    {"label": "PRO.1sg", "lexical_form": "yo",
     "position": 0, "confidence": "high"}
  ],
  "source": "europarl",
  "language": "es",
  "genre": "europarl",
  "stimulus_id": 42,
  "annotator_agreement": "single"
}
```

Notes:

- `position` is a **character offset** (converted server-side from the
  annotator's word-gap index)
- `confidence` is mapped from the annotator's 1–5 integer: ≥4 → `high`,
  2–3 → `medium`, 1 → `low`
- With a single annotator, every record will be
  `annotator_agreement: "single"`. The other values
  (`adjudicated`, `full`) only appear in a multi-annotator workflow
- Multiple pronouns at the same gap position produce multiple markers
  with the same `position` value
- Stimuli where the annotator said "yes" but placed no pronouns export
  with `markers: []`

Copy `gold_es.json` back to `data/spanish/gold/` in the repo for
downstream model training.

## 6. Backup / teardown

The SQLite DB lives in the `annotation-data` Docker volume (Option B) or
at `annotation/data/annotation.db` (Option A). It contains:

- every annotation (current state)
- every `annotation_history` row (append-only audit log of saves)
- user accounts + tokens

**Set up a daily backup cron on the remote host** — losing the DB loses
the audit trail even if you still have the gold export:

```bash
# /etc/cron.daily/annotation-backup
#!/bin/sh
DATE=$(date +%F)
sqlite3 /opt/annotation-repo/annotation/data/annotation.db \
  ".backup /var/backups/annotation/annotation-${DATE}.db"
find /var/backups/annotation -name "annotation-*.db" -mtime +30 -delete
```

Before final teardown, pull the DB off the host:

```bash
scp remote:/opt/annotation-repo/annotation/data/annotation.db \
    ./annotation_final_$(date +%F).db
```

## 7. Optional: scaling to two annotators

If you later want inter-annotator agreement or adjudication:

1. Seed a second annotator user via `db.seed_users` (the existing DB is
   preserved; re-seeding is idempotent)
2. Both annotators work through the same 974 stimuli (full overlap) or
   a partial-overlap split — the app does not filter by
   `metadata.sample_types`, so splits are coordinated out-of-band
3. Once both have annotations, `GET /api/agreement/compute` returns
   κ, percent agreement, Jaccard, and a per-stimulus breakdown
4. Admin opens `$BASE_URL/adjudication` to resolve disagreements
5. `GET /api/export/gold` will then include three record types:
   `single`, `full` (both agreed, no adjudication needed), and
   `adjudicated`

Target κ ≥ 0.75 for the binary decision if you go this route.

## 8. Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `422 Invalid label` | Non-canonical pronoun label in request | Must be one of `{PRO.1sg, PRO.2sg, PRO.3sg, PRO.1pl, PRO.2pl, PRO.3pl}` (see `annotation/constants.py`) |
| Context renders backwards | Sent `context_before` oldest-first | Must be **nearest-first** — our JSONL already is |
| Upload silently drops entries | Invalid JSON on a line | `POST /api/stimuli/load` falls back to plain-text mode for non-JSON lines. Check each line is `json.loads`-able |
| API calls 404 from the browser | Subpath missing trailing slash | Ensure nginx redirects `/annotation` → `/annotation/`; relative URLs need the trailing slash |
| `429 Too many requests` | Rate limiter triggered (60 req/min/IP) | Space out scripted calls; normal annotation workflow stays well under the limit |
| nginx 502 Bad Gateway (Option A) | App not listening on 127.0.0.1:8643 | Check the systemd unit is running; confirm the `proxy_pass` port matches |
| nginx 502 Bad Gateway (Option B) | `nginx.conf` still has `YOURDOMAIN` placeholder | Edit `annotation/nginx.conf`, `docker compose restart nginx` |
| Annotator can't log in | Token mismatch or role confusion | Re-query `db.get_user_by_username(u)` on the remote to get the canonical token |

## 9. Files to keep consistent

When moving to remote:

- `annotation/` — the full directory (code + docker-compose + nginx.conf)
- `data/spanish/gold/stimuli.jsonl` — the gold stimuli
- `docs/spanish_gold_annotation.md` — reference for annotators
- This doc

Do NOT move:
- `annotation/data/annotation.db` — gets created fresh on the remote
- `.venv/`, `__pycache__/` — rebuilt on the remote
