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
    "new_session",
    "process_shutdown",
    "process_restart",
    "approval_rejected",
    "tool_liveness_exhausted",
    "provider_recovery_exhausted",
    "external_blocked",
]
WorkspaceLeaseStatus = Literal["not_applicable", "held", "released"]


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
    changed_paths: tuple[str, ...] = field(default_factory=tuple)
    changed_paths_complete: bool = False
    changed_paths_source: str = "tool_result_metadata"
    changed_test_artifact_paths: tuple[str, ...] = field(default_factory=tuple)
    workspace_lease_status: WorkspaceLeaseStatus = "not_applicable"
    deliverable: str | None = None
    current_subagent_session_id: str | None = None
    disposed_at: str | None = None
    pause_reason: SubagentPauseReason | None = None
    last_error: str | None = None
    last_error_metadata: dict[str, Any] = field(default_factory=dict)
    error_log_path: str | None = None
    run_generation: int = 0
    write_scope_attention: bool = False
    write_scope_attention_tool: str | None = None
    write_scope_attention_path: str | None = None

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
            "task_label": self.task_label,
            "instructions": self.instructions,
            "user_constraints": self.user_constraints,
            "shared_context": self.shared_context,
            "owned_paths": list(self.owned_paths),
            "skill_ids": list(self.skill_ids),
            "skill_selection_reason": self.skill_selection_reason,
            "phase": self.phase,
            "depends_on": list(self.depends_on),
            "seam_contract": self.seam_contract,
            "changed_paths": list(self.changed_paths),
            "changed_paths_complete": self.changed_paths_complete,
            "changed_paths_source": self.changed_paths_source,
            "changed_test_artifact_paths": list(self.changed_test_artifact_paths),
            "workspace_lease_status": self.workspace_lease_status,
            "write_scope_attention": self.write_scope_attention,
            "write_scope_attention_tool": self.write_scope_attention_tool,
            "write_scope_attention_path": self.write_scope_attention_path,
            "deliverable": self.deliverable,
            "current_subagent_session_id": self.current_subagent_session_id,
            "disposed_at": self.disposed_at,
            "pause_reason": self.pause_reason,
            "last_error": self.last_error,
            "last_error_metadata": self.last_error_metadata,
            "error_log_path": self.error_log_path,
            "run_generation": self.run_generation,
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
                "new_session",
                "process_shutdown",
                "process_restart",
                "approval_rejected",
                "tool_liveness_exhausted",
                "provider_recovery_exhausted",
                "external_blocked",
            }
            else None
        )
        raw_changed_test_paths = payload.get("changed_test_artifact_paths", ())
        if not isinstance(raw_changed_test_paths, (list, tuple)):
            raw_changed_test_paths = ()
        raw_changed_paths = payload.get("changed_paths", ())
        if not isinstance(raw_changed_paths, (list, tuple)):
            raw_changed_paths = ()
        raw_changed_paths_complete = payload.get("changed_paths_complete", False)
        changed_paths_complete = (
            raw_changed_paths_complete
            if isinstance(raw_changed_paths_complete, bool)
            else False
        )
        raw_changed_paths_source = payload.get(
            "changed_paths_source",
            "tool_result_metadata",
        )
        changed_paths_source = (
            raw_changed_paths_source.strip()
            if isinstance(raw_changed_paths_source, str)
            and raw_changed_paths_source.strip()
            else "tool_result_metadata"
        )
        raw_depends_on = payload.get("depends_on", ())
        if not isinstance(raw_depends_on, (list, tuple)):
            raw_depends_on = ()
        raw_lease_status = payload.get("workspace_lease_status", "not_applicable")
        lease_status = (
            raw_lease_status
            if isinstance(raw_lease_status, str)
            and raw_lease_status in {"not_applicable", "held", "released"}
            else "not_applicable"
        )
        raw_write_scope_attention = payload.get("write_scope_attention", False)
        write_scope_attention = (
            raw_write_scope_attention
            if isinstance(raw_write_scope_attention, bool)
            else False
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
            task_label=str(payload.get("task_label", "")),
            instructions=str(payload.get("instructions", "")),
            user_constraints=(
                str(payload["user_constraints"])
                if payload.get("user_constraints") is not None
                else None
            ),
            shared_context=(
                str(payload["shared_context"])
                if payload.get("shared_context") is not None
                else None
            ),
            owned_paths=tuple(str(item) for item in payload.get("owned_paths", ())),
            skill_ids=tuple(str(item) for item in payload.get("skill_ids", ())),
            skill_selection_reason=str(
                payload.get(
                    "skill_selection_reason",
                    "none:not_selected_by_main",
                )
            ),
            phase=(
                str(payload["phase"]).strip()
                if isinstance(payload.get("phase"), str)
                and payload["phase"].strip()
                else None
            ),
            depends_on=tuple(
                str(item).strip()
                for item in raw_depends_on
                if str(item).strip()
            ),
            seam_contract=(
                str(payload["seam_contract"]).strip()
                if isinstance(payload.get("seam_contract"), str)
                and payload["seam_contract"].strip()
                else None
            ),
            changed_paths=tuple(
                str(item).strip()
                for item in raw_changed_paths
                if str(item).strip()
            ),
            changed_paths_complete=changed_paths_complete,
            changed_paths_source=changed_paths_source,
            changed_test_artifact_paths=tuple(
                str(item)
                for item in raw_changed_test_paths
                if str(item).strip()
            ),
            workspace_lease_status=lease_status,  # type: ignore[arg-type]
            write_scope_attention=write_scope_attention,
            write_scope_attention_tool=(
                str(payload["write_scope_attention_tool"])
                if payload.get("write_scope_attention_tool") is not None
                else None
            ),
            write_scope_attention_path=(
                str(payload["write_scope_attention_path"])
                if payload.get("write_scope_attention_path") is not None
                else None
            ),
            deliverable=(
                str(payload["deliverable"])
                if payload.get("deliverable") is not None
                else None
            ),
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
            last_error_metadata=(
                dict(payload["last_error_metadata"])
                if isinstance(payload.get("last_error_metadata"), dict)
                else {}
            ),
            error_log_path=(
                str(payload["error_log_path"])
                if payload.get("error_log_path") is not None
                else None
            ),
            run_generation=_nonnegative_int(payload.get("run_generation", 0)),
        )


@dataclass(slots=True)
class SubagentSnapshot:
    subagent_id: str
    codename: str
    status: SubagentStatus
    owner_main_session_id: str
    owner_main_turn_id: str
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
    changed_paths: tuple[str, ...] = field(default_factory=tuple)
    changed_paths_complete: bool = False
    changed_paths_source: str = "tool_result_metadata"
    changed_test_artifact_paths: tuple[str, ...] = field(default_factory=tuple)
    workspace_lease_status: WorkspaceLeaseStatus = "not_applicable"
    deliverable: str | None = None
    current_subagent_session_id: str | None = None
    pause_reason: SubagentPauseReason | None = None
    last_error: str | None = None
    last_error_metadata: dict[str, Any] = field(default_factory=dict)
    error_log_path: str | None = None
    last_tool_name: str | None = None
    last_activity_at: str | None = None
    latest_report: str | None = None
    report_complete: bool = False
    pending_background_job_count: int = 0
    pending_background_job_ids: tuple[str, ...] = field(default_factory=tuple)
    notable_events: tuple[SubagentEventNote, ...] = field(default_factory=tuple)
    run_generation: int = 0
    write_scope_attention: bool = False
    write_scope_attention_tool: str | None = None
    write_scope_attention_path: str | None = None


def _nonnegative_int(value: object) -> int:
    if not isinstance(value, (str, int, float)):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
