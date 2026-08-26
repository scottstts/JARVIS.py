"""In-memory runtime state for active subagents."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

from jarvis.actor_backends import ActorRuntime
from jarvis.storage import SessionStorage
from jarvis.tools.workspace_revision import WorkspaceSnapshot

from .types import (
    SubagentEventNote,
    SubagentPauseReason,
    SubagentSnapshot,
    SubagentStatus,
    WorkspaceLeaseStatus,
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
    task_label: str = ""
    instructions: str = ""
    user_constraints: str | None = None
    shared_context: str | None = None
    owned_paths: tuple[str, ...] = field(default_factory=tuple)
    skill_ids: tuple[str, ...] = field(default_factory=tuple)
    skill_selection_reason: str = "none:not_selected_by_main"
    phase: str | None = None
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    seam_contract: str | None = None
    changed_paths: set[str] = field(default_factory=set)
    changed_paths_complete: bool = False
    changed_paths_source: str = "tool_result_metadata"
    changed_test_artifact_paths: set[str] = field(default_factory=set)
    workspace_snapshot_baseline: WorkspaceSnapshot | None = None
    workspace_snapshot_incomplete: bool = False
    workspace_lease_status: WorkspaceLeaseStatus = "not_applicable"
    deliverable: str | None = None
    task: asyncio.Task[None] | None = None
    pause_reason: SubagentPauseReason | None = None
    last_error: str | None = None
    last_error_metadata: dict[str, object] = field(default_factory=dict)
    error_log_path: str | None = None
    last_tool_name: str | None = None
    last_activity_at: str | None = None
    latest_report: str | None = None
    report_complete: bool = False
    notable_events: deque[SubagentEventNote] = field(default_factory=deque)
    pending_pause_reason: SubagentPauseReason | None = None
    pending_background_job_ids: set[str] = field(default_factory=set)
    run_generation: int = 0
    provider_recovery_attempts: int = 0

    def snapshot(self) -> SubagentSnapshot:
        return SubagentSnapshot(
            subagent_id=self.subagent_id,
            codename=self.codename,
            status=self.status,
            owner_main_session_id=self.owner_main_session_id,
            owner_main_turn_id=self.owner_main_turn_id,
            task_label=self.task_label,
            instructions=self.instructions,
            user_constraints=self.user_constraints,
            shared_context=self.shared_context,
            owned_paths=self.owned_paths,
            skill_ids=self.skill_ids,
            skill_selection_reason=self.skill_selection_reason,
            phase=self.phase,
            depends_on=self.depends_on,
            seam_contract=self.seam_contract,
            changed_paths=tuple(sorted(self.changed_paths)),
            changed_paths_complete=self.changed_paths_complete,
            changed_paths_source=self.changed_paths_source,
            changed_test_artifact_paths=tuple(sorted(self.changed_test_artifact_paths)),
            workspace_lease_status=self.workspace_lease_status,
            deliverable=self.deliverable,
            current_subagent_session_id=self.loop.active_session_id(),
            pause_reason=self.pause_reason,
            last_error=self.last_error,
            last_error_metadata=dict(self.last_error_metadata),
            error_log_path=self.error_log_path,
            last_tool_name=self.last_tool_name,
            last_activity_at=self.last_activity_at,
            latest_report=self.latest_report,
            report_complete=self.report_complete,
            pending_background_job_count=len(self.pending_background_job_ids),
            pending_background_job_ids=tuple(sorted(self.pending_background_job_ids)),
            notable_events=tuple(self.notable_events),
            run_generation=self.run_generation,
        )
