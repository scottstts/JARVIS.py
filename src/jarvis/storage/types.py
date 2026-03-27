"""Data models for persisted sessions and transcript records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RecordKind = Literal["message", "compaction"]
RecordRole = Literal["system", "user", "assistant", "tool"]
SessionStatus = Literal["active", "archived"]
TurnStatus = Literal["in_progress", "completed", "interrupted", "superseded"]


@dataclass(slots=True, frozen=True)
class SessionMetadata:
    """Session-level metadata tracked in the storage index."""

    session_id: str
    created_at: str
    updated_at: str
    start_reason: str
    parent_session_id: str | None = None
    status: SessionStatus = "active"
    pending_reactive_compaction: bool = False
    pending_interruption_notice: bool = False
    pending_approval: dict[str, Any] | None = None
    compaction_count: int = 0
    last_input_tokens: int | None = None
    last_output_tokens: int | None = None
    last_total_tokens: int | None = None
    last_estimated_input_tokens: int | None = None
    turn_states: dict[str, TurnStatus] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "start_reason": self.start_reason,
            "parent_session_id": self.parent_session_id,
            "status": self.status,
            "pending_reactive_compaction": self.pending_reactive_compaction,
            "pending_interruption_notice": self.pending_interruption_notice,
            "pending_approval": (
                dict(self.pending_approval)
                if isinstance(self.pending_approval, dict)
                else None
            ),
            "compaction_count": self.compaction_count,
            "last_input_tokens": self.last_input_tokens,
            "last_output_tokens": self.last_output_tokens,
            "last_total_tokens": self.last_total_tokens,
            "last_estimated_input_tokens": self.last_estimated_input_tokens,
            "turn_states": dict(self.turn_states),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetadata":
        return cls(
            session_id=str(data["session_id"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            start_reason=str(data.get("start_reason", "initial")),
            parent_session_id=data.get("parent_session_id"),
            status="archived" if data.get("status") == "archived" else "active",
            pending_reactive_compaction=bool(data.get("pending_reactive_compaction", False)),
            pending_interruption_notice=bool(data.get("pending_interruption_notice", False)),
            pending_approval=(
                dict(data["pending_approval"])
                if isinstance(data.get("pending_approval"), dict)
                else None
            ),
            compaction_count=int(data.get("compaction_count", 0)),
            last_input_tokens=_optional_int(data.get("last_input_tokens")),
            last_output_tokens=_optional_int(data.get("last_output_tokens")),
            last_total_tokens=_optional_int(data.get("last_total_tokens")),
            last_estimated_input_tokens=_optional_int(data.get("last_estimated_input_tokens")),
            turn_states=_normalize_turn_states(data.get("turn_states")),
        )


@dataclass(slots=True, frozen=True)
class ConversationRecord:
    """Single transcript entry persisted in per-session JSONL."""

    record_id: str
    session_id: str
    created_at: str
    role: RecordRole
    content: str
    kind: RecordKind = "message"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "role": self.role,
            "content": self.content,
            "kind": self.kind,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationRecord":
        raw_role = str(data.get("role", "assistant"))
        role: RecordRole
        if raw_role in {"system", "user", "assistant", "tool"}:
            role = raw_role
        else:
            role = "assistant"

        raw_kind = str(data.get("kind", "message"))
        kind: RecordKind = "compaction" if raw_kind == "compaction" else "message"

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return cls(
            record_id=str(data["record_id"]),
            session_id=str(data["session_id"]),
            created_at=str(data["created_at"]),
            role=role,
            content=str(data.get("content", "")),
            kind=kind,
            metadata=dict(metadata),
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_turn_states(value: Any) -> dict[str, TurnStatus]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, TurnStatus] = {}
    for raw_turn_id, raw_status in value.items():
        turn_id = str(raw_turn_id).strip()
        if not turn_id:
            continue
        status = str(raw_status).strip()
        if status not in {"in_progress", "completed", "interrupted", "superseded"}:
            continue
        normalized[turn_id] = status
    return normalized
