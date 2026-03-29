"""File-backed session/transcript storage service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .types import ConversationRecord, SessionMetadata, TurnStatus


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
            pending_approval=None,
        )

    def set_turn_status(
        self,
        session_id: str,
        *,
        turn_id: str,
        status: TurnStatus,
    ) -> SessionMetadata:
        normalized_turn_id = turn_id.strip()
        if not normalized_turn_id:
            raise ValueError("turn_id cannot be empty.")

        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown session id: {session_id}")

        turn_states = dict(session.turn_states)
        turn_states[normalized_turn_id] = status
        return self.update_session(session_id, turn_states=turn_states)

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

    def load_records(
        self,
        session_id: str,
        *,
        include_turn_ids: tuple[str, ...] = (),
        include_all_turns: bool = False,
    ) -> list[ConversationRecord]:
        path = self._session_path(session_id)
        if not path.exists():
            return []

        session = self.get_session(session_id)
        turn_states = session.turn_states if session is not None else {}
        included_turn_ids = {turn_id.strip() for turn_id in include_turn_ids if turn_id.strip()}
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
                    record = ConversationRecord.from_dict(payload)
                except KeyError:
                    continue
                if not include_all_turns and not _record_is_visible(
                    record,
                    turn_states=turn_states,
                    include_turn_ids=included_turn_ids,
                ):
                    continue
                records.append(record)
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


def _record_is_visible(
    record: ConversationRecord,
    *,
    turn_states: dict[str, TurnStatus],
    include_turn_ids: set[str],
) -> bool:
    raw_turn_id = record.metadata.get("turn_id")
    if raw_turn_id is None:
        return True

    turn_id = str(raw_turn_id).strip()
    if not turn_id:
        return True
    if turn_id in include_turn_ids:
        return True
    return turn_states.get(turn_id) in {"completed", "interrupted", "superseded"}
