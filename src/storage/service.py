"""File-backed session/transcript storage service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .types import ConversationRecord, SessionMetadata


class SessionStorage:
    """Persists active session metadata and per-session transcript JSONL files."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.expanduser()
        self._sessions_dir = self.root_dir / "sessions"
        self._index_path = self.root_dir / "sessions_index.json"
        self._ensure_layout()

    def create_session(
        self,
        *,
        parent_session_id: str | None = None,
        start_reason: str = "initial",
    ) -> SessionMetadata:
        session_id = uuid4().hex
        now = _utc_now_iso()
        metadata = SessionMetadata(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            start_reason=start_reason,
            parent_session_id=parent_session_id,
        )

        index = self._load_index()
        index["sessions"][session_id] = metadata.to_dict()
        index["active_session_id"] = session_id
        self._write_index(index)
        self._session_path(session_id).touch(exist_ok=True)
        return metadata

    def get_active_session(self) -> SessionMetadata | None:
        index = self._load_index()
        session_id = index.get("active_session_id")
        if not isinstance(session_id, str):
            return None
        raw = index["sessions"].get(session_id)
        if not isinstance(raw, dict):
            return None
        return SessionMetadata.from_dict(raw)

    def get_session(self, session_id: str) -> SessionMetadata | None:
        index = self._load_index()
        raw = index["sessions"].get(session_id)
        if not isinstance(raw, dict):
            return None
        return SessionMetadata.from_dict(raw)

    def set_active_session(self, session_id: str) -> None:
        index = self._load_index()
        if session_id not in index["sessions"]:
            raise ValueError(f"Unknown session id: {session_id}")
        index["active_session_id"] = session_id
        self._write_index(index)

    def archive_session(self, session_id: str) -> None:
        self.update_session(
            session_id,
            status="archived",
            pending_reactive_compaction=False,
        )

    def update_session(self, session_id: str, **changes: Any) -> SessionMetadata:
        index = self._load_index()
        raw = index["sessions"].get(session_id)
        if not isinstance(raw, dict):
            raise ValueError(f"Unknown session id: {session_id}")

        next_raw = dict(raw)
        next_raw.update(changes)
        next_raw["updated_at"] = _utc_now_iso()
        metadata = SessionMetadata.from_dict(next_raw)
        index["sessions"][session_id] = metadata.to_dict()
        self._write_index(index)
        return metadata

    def append_record(self, session_id: str, record: ConversationRecord) -> None:
        if self.get_session(session_id) is None:
            raise ValueError(f"Unknown session id: {session_id}")

        path = self._session_path(session_id)
        with path.open("a", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, ensure_ascii=False)
            handle.write("\n")
        self.update_session(session_id)

    def load_records(self, session_id: str) -> list[ConversationRecord]:
        path = self._session_path(session_id)
        if not path.exists():
            return []

        records: list[ConversationRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                try:
                    records.append(ConversationRecord.from_dict(payload))
                except KeyError:
                    continue
        return records

    def _session_path(self, session_id: str) -> Path:
        return self._sessions_dir / f"{session_id}.jsonl"

    def _ensure_layout(self) -> None:
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            self._write_index({"active_session_id": None, "sessions": {}})

    def _load_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {"active_session_id": None, "sessions": {}}
        try:
            with self._index_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        sessions = payload.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}
        active_session_id = payload.get("active_session_id")
        if not isinstance(active_session_id, str):
            active_session_id = None
        return {
            "active_session_id": active_session_id,
            "sessions": sessions,
        }

    def _write_index(self, payload: dict[str, Any]) -> None:
        tmp_path = self._index_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(self._index_path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
