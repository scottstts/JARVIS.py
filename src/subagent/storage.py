"""File-backed catalog and transcript roots for route-scoped subagents."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from storage import SessionStorage

from .types import SubagentCatalogEntry


class SubagentCatalogStorage:
    """Persists the route-level subagent index and per-subagent transcript roots."""

    def __init__(self, *, archive_dir: Path, route_id: str) -> None:
        self._route_dir = archive_dir / route_id
        self._index_path = self._route_dir / "index.json"
        self._route_dir.mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            self._write_index({"subagents": {}})

    def list_entries(self) -> tuple[SubagentCatalogEntry, ...]:
        payload = self._load_index()
        return tuple(
            SubagentCatalogEntry.from_dict(entry)
            for _, entry in sorted(payload["subagents"].items())
            if isinstance(entry, dict)
        )

    def get_entry(self, subagent_id: str) -> SubagentCatalogEntry | None:
        payload = self._load_index()
        raw = payload["subagents"].get(subagent_id)
        if not isinstance(raw, dict):
            return None
        return SubagentCatalogEntry.from_dict(raw)

    def create_entry(self, entry: SubagentCatalogEntry) -> SubagentCatalogEntry:
        payload = self._load_index()
        payload["subagents"][entry.subagent_id] = entry.to_dict()
        self._write_index(payload)
        return entry

    def update_entry(self, subagent_id: str, **changes: Any) -> SubagentCatalogEntry:
        payload = self._load_index()
        raw = payload["subagents"].get(subagent_id)
        if not isinstance(raw, dict):
            raise ValueError(f"Unknown subagent id: {subagent_id}")
        next_raw = dict(raw)
        next_raw.update(changes)
        next_raw["updated_at"] = _utc_now_iso()
        entry = SubagentCatalogEntry.from_dict(next_raw)
        payload["subagents"][subagent_id] = entry.to_dict()
        self._write_index(payload)
        return entry

    def session_storage(self, subagent_id: str) -> SessionStorage:
        subagent_dir = self._route_dir / subagent_id
        subagent_dir.mkdir(parents=True, exist_ok=True)
        return SessionStorage(subagent_dir)

    def subagent_dir(self, subagent_id: str) -> Path:
        path = self._route_dir / subagent_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_index(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {}
        subagents = payload.get("subagents")
        if not isinstance(subagents, dict):
            subagents = {}
        return {"subagents": subagents}

    def _write_index(self, payload: dict[str, Any]) -> None:
        tmp_path = self._index_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self._index_path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
