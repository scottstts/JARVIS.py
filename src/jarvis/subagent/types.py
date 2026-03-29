"""Shared types for the subagent subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SubagentStatus = Literal[
    "running",
    "awaiting_approval",
    "waiting_background",
    "paused",
    "completed",
    "failed",
    "disposed",
]
SubagentPauseReason = Literal[
    "main_stop",
    "superseded_by_user_message",
    "approval_rejected",
    "gateway_disconnect_recovery",
]


@dataclass(slots=True, frozen=True)
class SubagentEventNote:
    created_at: str
    kind: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "kind": self.kind,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SubagentEventNote":
        return cls(
            created_at=str(payload.get("created_at", "")),
            kind=str(payload.get("kind", "")),
            summary=str(payload.get("summary", "")),
        )


@dataclass(slots=True, frozen=True)
class SubagentCatalogEntry:
    subagent_id: str
    codename: str
    status: SubagentStatus
    created_at: str
    updated_at: str
    route_id: str
    owner_main_session_id: str
    owner_main_turn_id: str
    current_subagent_session_id: str | None = None
    disposed_at: str | None = None
    pause_reason: SubagentPauseReason | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subagent_id": self.subagent_id,
            "codename": self.codename,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "route_id": self.route_id,
            "owner_main_session_id": self.owner_main_session_id,
            "owner_main_turn_id": self.owner_main_turn_id,
            "current_subagent_session_id": self.current_subagent_session_id,
            "disposed_at": self.disposed_at,
            "pause_reason": self.pause_reason,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SubagentCatalogEntry":
        status = str(payload.get("status", "failed"))
        if status not in {
            "running",
            "awaiting_approval",
            "waiting_background",
            "paused",
            "completed",
            "failed",
            "disposed",
        }:
            status = "failed"
        pause_reason = payload.get("pause_reason")
        normalized_pause_reason = (
            str(pause_reason)
            if str(pause_reason)
            in {
                "main_stop",
                "superseded_by_user_message",
                "approval_rejected",
                "gateway_disconnect_recovery",
            }
            else None
        )
        return cls(
            subagent_id=str(payload.get("subagent_id", "")),
            codename=str(payload.get("codename", "")),
            status=status,  # type: ignore[arg-type]
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            route_id=str(payload.get("route_id", "")),
            owner_main_session_id=str(payload.get("owner_main_session_id", "")),
            owner_main_turn_id=str(payload.get("owner_main_turn_id", "")),
            current_subagent_session_id=(
                str(payload["current_subagent_session_id"])
                if payload.get("current_subagent_session_id") is not None
                else None
            ),
            disposed_at=(
                str(payload["disposed_at"])
                if payload.get("disposed_at") is not None
                else None
            ),
            pause_reason=normalized_pause_reason,  # type: ignore[arg-type]
            last_error=(
                str(payload["last_error"])
                if payload.get("last_error") is not None
                else None
            ),
        )


@dataclass(slots=True)
class SubagentSnapshot:
    subagent_id: str
    codename: str
    status: SubagentStatus
    owner_main_session_id: str
    owner_main_turn_id: str
    current_subagent_session_id: str | None = None
    pause_reason: SubagentPauseReason | None = None
    last_error: str | None = None
    last_tool_name: str | None = None
    last_activity_at: str | None = None
    pending_background_job_count: int = 0
    pending_background_job_ids: tuple[str, ...] = field(default_factory=tuple)
    notable_events: tuple[SubagentEventNote, ...] = field(default_factory=tuple)
