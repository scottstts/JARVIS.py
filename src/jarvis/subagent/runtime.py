"""In-memory runtime state for active subagents."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

from jarvis.actor_backends import ActorRuntime
from jarvis.storage import SessionStorage

from .types import (
    SubagentEventNote,
    SubagentPauseReason,
    SubagentSnapshot,
    SubagentStatus,
)


@dataclass(slots=True)
class SubagentRuntime:
    subagent_id: str
    codename: str
    loop: ActorRuntime
    storage: SessionStorage
    owner_main_session_id: str
    owner_main_turn_id: str
    status: SubagentStatus
    created_at: str
    updated_at: str
    task: asyncio.Task[None] | None = None
    pause_reason: SubagentPauseReason | None = None
    last_error: str | None = None
    last_tool_name: str | None = None
    last_activity_at: str | None = None
    notable_events: deque[SubagentEventNote] = field(default_factory=deque)
    pending_pause_reason: SubagentPauseReason | None = None
    pending_background_job_ids: set[str] = field(default_factory=set)

    def snapshot(self) -> SubagentSnapshot:
        return SubagentSnapshot(
            subagent_id=self.subagent_id,
            codename=self.codename,
            status=self.status,
            owner_main_session_id=self.owner_main_session_id,
            owner_main_turn_id=self.owner_main_turn_id,
            current_subagent_session_id=self.loop.active_session_id(),
            pause_reason=self.pause_reason,
            last_error=self.last_error,
            last_tool_name=self.last_tool_name,
            last_activity_at=self.last_activity_at,
            pending_background_job_count=len(self.pending_background_job_ids),
            pending_background_job_ids=tuple(sorted(self.pending_background_job_ids)),
            notable_events=tuple(self.notable_events),
        )
