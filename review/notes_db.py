"""JSONL-based persistence for review notes.

Each line in the JSONL file is a JSON object with:
  sentence_id, feature_context, verdict, note_text, created_at, updated_at

On startup the full file is loaded into an in-memory dict keyed by
(sentence_id, feature_context). Each save appends to the file.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_Key = Tuple[str, str]  # (sentence_id, feature_context)

DEFAULT_JSONL_PATH = Path(__file__).parent / "data" / "notes.jsonl"


class NotesStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or DEFAULT_JSONL_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._notes: Dict[_Key, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = (obj["sentence_id"], obj.get("feature_context", ""))
                    self._notes[key] = obj
                except (json.JSONDecodeError, KeyError):
                    continue

    def upsert_note(
        self,
        sentence_id: str,
        feature_context: str,
        verdict: str,
        note_text: str,
        active_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        key = (sentence_id, feature_context)
        existing = self._notes.get(key)
        note = {
            "sentence_id": sentence_id,
            "feature_context": feature_context,
            "verdict": verdict,
            "note_text": note_text,
            "active_filters": active_filters,
            "created_at": existing["created_at"] if existing else now,
            "updated_at": now,
        }
        self._notes[key] = note
        # Append to JSONL
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(note, ensure_ascii=False) + "\n")
        return note

    def get_note(
        self, sentence_id: str, feature_context: str
    ) -> Optional[Dict[str, Any]]:
        return self._notes.get((sentence_id, feature_context))

    def get_notes_for_sentences(
        self, sentence_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Return notes keyed by sentence_id for a batch of IDs."""
        ids = set(sentence_ids)
        out: Dict[str, Dict[str, Any]] = {}
        for (sid, _ctx), note in self._notes.items():
            if sid in ids:
                out[sid] = note
        return out

    def get_all_notes(
        self,
        feature_context: Optional[str] = None,
        verdict: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = list(self._notes.values())
        if feature_context:
            results = [n for n in results if n.get("feature_context") == feature_context]
        if verdict:
            results = [n for n in results if n.get("verdict") == verdict]
        return results

    def export_tsv(self, path: Optional[Path] = None) -> str:
        """Write notes as TSV. Returns the TSV content."""
        notes = sorted(self._notes.values(), key=lambda n: n.get("updated_at", ""))
        header = "sentence_id\tfeature_context\tverdict\tnote_text\tcreated_at\tupdated_at"
        lines = [header]
        for n in notes:
            line = "\t".join(
                str(n.get(col, ""))
                for col in ["sentence_id", "feature_context", "verdict", "note_text", "created_at", "updated_at"]
            )
            lines.append(line)
        content = "\n".join(lines) + "\n"
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        return content

    def compact(self) -> None:
        """Rewrite the JSONL file, deduplicating by key (keeping latest)."""
        with open(self.path, "w", encoding="utf-8") as f:
            for note in self._notes.values():
                f.write(json.dumps(note, ensure_ascii=False) + "\n")

    @property
    def count(self) -> int:
        return len(self._notes)

    @property
    def reviewed_ids(self) -> set:
        """Return the set of sentence_ids that have any note."""
        return {sid for sid, _ctx in self._notes}
