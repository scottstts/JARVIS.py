"""Lifecycle manager for route-scoped subagents."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
import json
import traceback
from typing import Any, Awaitable, Callable
from uuid import uuid4

from jarvis.actor_backends import ActorRuntime, backend_kind_for_provider
from jarvis.codex_backend import CodexActorRuntime, CodexBackendSettings, CodexRouteCoordinator
from jarvis.core import (
    AgentApprovalRequestEvent,
    AgentAssistantMessageEvent,
    AgentIdentity,
    AgentKind,
    AgentLoop,
    AgentMemoryMode,
    AgentRuntimeMessage,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentTurnDoneEvent,
    CoreSettings,
    InterruptionReason,
)
from jarvis.gateway.bash_job_supervisor import BashJobNotice
from jarvis.gateway.route_events import (
    RouteApprovalRequestEvent,
    RouteSystemNoticeEvent,
    RouteToolCallEvent,
)
from jarvis.llm import (
    LLMError,
    LLMService,
    ProviderRateLimitError,
    ProviderTemporaryError,
)
from jarvis.logging_setup import get_application_logger
from jarvis.skills import SkillsSettings, get_skill
from jarvis.core.tool_safety import (
    changed_test_artifact_paths_from_result,
    changed_workspace_paths_from_result,
    test_artifact_paths_from_paths,
)
from jarvis.skills.catalog import read_skill_markdown
from jarvis.storage import SessionStorage
from jarvis.storage.layout import transcript_archive_root_from_runtime_path
from jarvis.tools import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolRegistry,
    ToolRuntime,
    WorkspaceAccessCoordinator,
    WorkspaceLeaseError,
    with_workspace_capabilities,
    with_workspace_observation,
)
from jarvis.tools.basic.bash.jobs import (
    BashJobError,
    mark_job_progress_notified,
    mark_job_terminal_notice_dispatched,
)
from jarvis.tools.workspace_revision import (
    WorkspaceSnapshotError,
    diff_workspace_snapshots,
    workspace_snapshot_paths,
)

from .bootstrap import (
    SubagentBootstrapLoader,
    build_assignment_message,
    build_step_in_message,
    build_subagent_kickoff_text,
)
from .codenames import allocate_codename
from .runtime import SubagentRuntime
from .settings import SubagentSettings
from .storage import SubagentCatalogStorage
from .types import (
    SubagentCatalogEntry,
    SubagentEventNote,
    SubagentPauseReason,
    SubagentSnapshot,
)


LOGGER = get_application_logger(__name__)
_MAX_SUBAGENT_PROVIDER_RECOVERY_ATTEMPTS = 3
_MAX_TRACKED_CHANGED_PATHS = 256
_IN_FLIGHT_SUBAGENT_STATUSES = frozenset(
    {"running", "awaiting_approval", "waiting_background"}
)


class SubagentManager:
    """Owns route-local subagent creation, monitoring, control, and disposal."""

    def __init__(
        self,
        *,
        route_id: str,
        llm_service: LLMService,
        core_settings: CoreSettings,
        tool_registry: ToolRegistry,
        tool_execution_guard: asyncio.Semaphore,
        workspace_access: WorkspaceAccessCoordinator | None = None,
        publish_event: Callable[[object], Awaitable[None]],
        register_approval_target: Callable[[str, ActorRuntime], None],
        tool_result_observer: Callable[[ToolExecutionResult, ToolExecutionContext], Awaitable[None]]
        | None = None,
        settings: SubagentSettings | None = None,
        codex_settings: CodexBackendSettings | None = None,
        codex_coordinator: CodexRouteCoordinator | None = None,
    ) -> None:
        self._route_id = route_id
        self._llm_service = llm_service
        self._core_settings = core_settings
        self._settings = settings or SubagentSettings.from_workspace_dir(
            core_settings.workspace_dir,
            transcript_archive_root=transcript_archive_root_from_runtime_path(
                transcript_archive_dir=core_settings.transcript_archive_dir,
                route_id=route_id,
            ),
        )
        self._tool_registry = tool_registry
        self._tool_execution_guard = tool_execution_guard
        self._workspace_access = workspace_access
        self._publish_event = publish_event
        self._register_approval_target = register_approval_target
        self._tool_result_observer = tool_result_observer
        self._codex_settings = codex_settings or CodexBackendSettings.from_env()
        self._codex_coordinator = codex_coordinator
        self._catalog = SubagentCatalogStorage(
            archive_dir=self._settings.archive_dir,
            route_id=route_id,
        )
        self._subagents: dict[str, SubagentRuntime] = {}
        self._pending_bash_job_notices: dict[str, dict[str, BashJobNotice]] = {}
        self._last_monitor_signatures: dict[str, str] = {}
        self._last_main_context_signature: tuple[str, str] | None = None
        self._restore_lock = asyncio.Lock()
        self._restored = False

    async def restore(self, *, owner_main_session_id: str | None) -> None:
        """Rebuild non-disposed child runtimes from the durable route catalog.

        A process can only resume the contracts belonging to the active main session or
        one of its compaction ancestors.  Persisted in-flight statuses are never treated
        as proof that a child completed: they are converted to an explicitly paused
        process-restart state and their child session reconciles any orphaned turn.
        """

        async with self._restore_lock:
            if self._restored:
                return
            entries = self._entries_for_main_session(owner_main_session_id)
            for entry in entries:
                if entry.subagent_id in self._subagents:
                    continue
                await self._restore_entry(entry)
            self._restored = True

    def _entries_for_main_session(
        self,
        owner_main_session_id: str | None,
    ) -> tuple[SubagentCatalogEntry, ...]:
        normalized_owner = (owner_main_session_id or "").strip()
        if not normalized_owner:
            return ()

        main_storage = SessionStorage(self._core_settings.transcript_archive_dir)
        owner_session_ids = {normalized_owner}
        cursor = main_storage.get_session(normalized_owner)
        visited: set[str] = set()
        while cursor is not None and cursor.session_id not in visited:
            visited.add(cursor.session_id)
            if cursor.start_reason != "compaction":
                break
            parent_session_id = (cursor.parent_session_id or "").strip()
            if not parent_session_id:
                break
            owner_session_ids.add(parent_session_id)
            cursor = main_storage.get_session(parent_session_id)

        return tuple(
            entry
            for entry in self._catalog.list_entries()
            if entry.route_id == self._route_id
            and entry.status != "disposed"
            and entry.owner_main_session_id in owner_session_ids
        )

    async def _restore_entry(self, entry: SubagentCatalogEntry) -> None:
        skill_documents = self._load_skill_documents(entry.skill_ids)
        bootstrap_loader = SubagentBootstrapLoader(
            assignment_message=build_assignment_message(
                codename=entry.codename,
                subagent_id=entry.subagent_id,
                task_label=entry.task_label,
                instructions=entry.instructions,
                user_constraints=entry.user_constraints,
                shared_context=entry.shared_context,
                owned_paths=entry.owned_paths,
                skill_documents=skill_documents,
                phase=entry.phase,
                depends_on=entry.depends_on,
                seam_contract=entry.seam_contract,
                deliverable=entry.deliverable,
            )
        )
        storage = self._catalog.session_storage(
            owner_main_session_id=entry.owner_main_session_id,
            subagent_id=entry.subagent_id,
        )
        was_in_flight = entry.status in _IN_FLIGHT_SUBAGENT_STATUSES
        runtime = SubagentRuntime(
            subagent_id=entry.subagent_id,
            codename=entry.codename,
            loop=self._build_subagent_loop(
                subagent_id=entry.subagent_id,
                codename=entry.codename,
                storage=storage,
                bootstrap_loader=bootstrap_loader,
            ),
            storage=storage,
            owner_main_session_id=entry.owner_main_session_id,
            owner_main_turn_id=entry.owner_main_turn_id,
            status="paused" if was_in_flight else entry.status,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            task_label=entry.task_label,
            instructions=entry.instructions,
            user_constraints=entry.user_constraints,
            shared_context=entry.shared_context,
            owned_paths=entry.owned_paths,
            skill_ids=entry.skill_ids,
            skill_selection_reason=entry.skill_selection_reason,
            phase=entry.phase,
            depends_on=entry.depends_on,
            seam_contract=entry.seam_contract,
            changed_paths=set(entry.changed_paths),
            changed_paths_complete=entry.changed_paths_complete,
            changed_paths_source=entry.changed_paths_source,
            changed_test_artifact_paths=set(entry.changed_test_artifact_paths),
            workspace_lease_status=(
                "released"
                if self._workspace_access is not None and entry.owned_paths
                else "not_applicable"
            ),
            deliverable=entry.deliverable,
            pause_reason=("process_restart" if was_in_flight else entry.pause_reason),
            last_error=entry.last_error,
            last_error_metadata=dict(entry.last_error_metadata),
            error_log_path=entry.error_log_path,
            last_activity_at=entry.updated_at or None,
            notable_events=deque(),
        )
        runtime.run_generation = max(1, entry.run_generation + 1)
        self._subagents[entry.subagent_id] = runtime

        try:
            session_id = await runtime.loop.prepare_session(
                start_reason="subagent_recovery"
            )
            runtime.latest_report = self._latest_assistant_report(runtime)
            runtime.report_complete = (
                runtime.status == "completed" and runtime.latest_report is not None
            )
            if was_in_flight:
                runtime.report_complete = False
                runtime.loop.append_system_note(
                    (
                        "The previous Jarvis process ended before this subagent completed its "
                        "turn. The child was recovered as interrupted and remains paused. Treat "
                        "partial output and workspace changes as incomplete; continue only after "
                        "explicit direction."
                    ),
                    session_id=session_id,
                    metadata={
                        "subagent_process_recovery": True,
                        "recovery_kind": "unexpected_process_interruption",
                        "previous_status": entry.status,
                    },
                )
                self._append_notable_event(
                    runtime,
                    kind="process_restart_recovery",
                    summary=(
                        "Recovered an interrupted child turn after an unexpected process "
                        "interruption."
                    ),
                )
            elif entry.pause_reason == "process_shutdown":
                runtime.loop.append_system_note(
                    (
                        "This subagent was restored after a graceful Jarvis process shutdown. "
                        "It remains paused until explicitly continued."
                    ),
                    session_id=session_id,
                    metadata={
                        "subagent_process_recovery": True,
                        "recovery_kind": "graceful_process_shutdown",
                    },
                )
                self._append_notable_event(
                    runtime,
                    kind="process_shutdown_restore",
                    summary="Restored after a graceful process shutdown; remains paused.",
                )
            else:
                self._append_notable_event(
                    runtime,
                    kind="restored",
                    summary="Restored from the durable subagent contract.",
                )
            await self._restore_workspace_lease(runtime, entry)
            self._sync_catalog(runtime)
        except BaseException:
            self._subagents.pop(entry.subagent_id, None)
            await runtime.loop.aclose()
            raise

    async def _restore_workspace_lease(
        self,
        runtime: SubagentRuntime,
        entry: SubagentCatalogEntry,
    ) -> None:
        if self._workspace_access is None or not runtime.owned_paths:
            runtime.workspace_lease_status = "not_applicable"
            return
        if entry.workspace_lease_status != "held":
            runtime.workspace_lease_status = "released"
            return
        try:
            await self._workspace_access.claim_paths(
                owner=f"subagent:{runtime.subagent_id}",
                paths=runtime.owned_paths,
            )
        except WorkspaceLeaseError as exc:
            runtime.workspace_lease_status = "released"
            if entry.status in _IN_FLIGHT_SUBAGENT_STATUSES:
                runtime.status = "paused"
                runtime.pause_reason = "external_blocked"
                runtime.report_complete = False
            runtime.last_error = f"WorkspaceLeaseError: {exc}"
            runtime.last_error_metadata = _exception_metadata(exc)
            self._append_notable_event(
                runtime,
                kind="workspace_lease_recovery_blocked",
                summary=f"Could not reacquire workspace ownership: {exc}",
            )
            session_id = runtime.loop.active_session_id()
            if session_id is not None:
                runtime.loop.append_system_note(
                    (
                        "Jarvis restored this subagent's contract, but its previous workspace "
                        "ownership could not be reacquired. The subagent is paused until the "
                        "workspace conflict is resolved."
                    ),
                    session_id=session_id,
                    metadata={
                        "subagent_process_recovery": True,
                        "workspace_lease_recovery_blocked": True,
                        "conflict_class": exc.conflict_class,
                        "conflict_key": exc.conflict_key,
                    },
                )
            return

        runtime.workspace_lease_status = "held"
        session_id = runtime.loop.active_session_id()
        if session_id is not None:
            runtime.loop.append_system_note(
                (
                    "Jarvis reacquired this subagent's workspace ownership after process "
                    "startup. The persisted contract and prior change evidence remain intact."
                ),
                session_id=session_id,
                metadata={
                    "workspace_lease": True,
                    "workspace_lease_reacquired": True,
                    "subagent_process_recovery": True,
                    "owned_paths": list(runtime.owned_paths),
                },
            )
        self._append_notable_event(
            runtime,
            kind="workspace_lease_reacquired",
            summary="Reacquired workspace ownership after process startup.",
        )

    async def invoke(
        self,
        *,
        requester_kind: AgentKind,
        task_label: str,
        instructions: str,
        owner_main_session_id: str,
        owner_main_turn_id: str,
        user_constraints: str | None = None,
        shared_context: str | None = None,
        owned_paths: tuple[str, ...] = (),
        skill_ids: tuple[str, ...] = (),
        phase: str | None = None,
        depends_on: tuple[str, ...] = (),
        seam_contract: str | None = None,
        deliverable: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_main_requester(requester_kind)
        normalized_task_label = task_label.strip()
        normalized_instructions = instructions.strip()
        if not normalized_task_label:
            raise ValueError("Subagent task_label cannot be empty.")
        if not normalized_instructions:
            raise ValueError("Subagent instructions cannot be empty.")
        normalized_owned_paths = _normalize_unique_strings(owned_paths)
        normalized_skill_ids = _normalize_unique_strings(skill_ids)
        normalized_phase = _normalize_optional_string(phase)
        normalized_depends_on = _normalize_unique_strings(depends_on)
        normalized_seam_contract = _normalize_optional_string(seam_contract)
        skill_selection_reason = (
            "main_selected" if normalized_skill_ids else "none:not_selected_by_main"
        )
        skill_documents = self._load_skill_documents(normalized_skill_ids)
        active = self._non_disposed_runtimes()
        if len(active) >= self._settings.max_active:
            raise ValueError(
                f"Subagent limit reached. Dispose a subagent before creating more than {self._settings.max_active}."
            )

        subagent_id = uuid4().hex
        codename = allocate_codename(
            pool=self._settings.codename_pool,
            active_codenames={runtime.codename for runtime in active},
        )
        created_at = _utc_now_iso()
        storage = self._catalog.session_storage(
            owner_main_session_id=owner_main_session_id,
            subagent_id=subagent_id,
        )
        bootstrap_loader = SubagentBootstrapLoader(
            assignment_message=build_assignment_message(
                codename=codename,
                subagent_id=subagent_id,
                task_label=normalized_task_label,
                instructions=normalized_instructions,
                user_constraints=user_constraints,
                shared_context=shared_context,
                owned_paths=normalized_owned_paths,
                skill_documents=skill_documents,
                phase=normalized_phase,
                depends_on=normalized_depends_on,
                seam_contract=normalized_seam_contract,
                deliverable=deliverable,
            )
        )
        runtime = SubagentRuntime(
            subagent_id=subagent_id,
            codename=codename,
            loop=self._build_subagent_loop(
                subagent_id=subagent_id,
                codename=codename,
                storage=storage,
                bootstrap_loader=bootstrap_loader,
            ),
            storage=storage,
            owner_main_session_id=owner_main_session_id,
            owner_main_turn_id=owner_main_turn_id,
            status="running",
            created_at=created_at,
            updated_at=created_at,
            task_label=normalized_task_label,
            instructions=normalized_instructions,
            user_constraints=user_constraints,
            shared_context=shared_context,
            owned_paths=normalized_owned_paths,
            skill_ids=normalized_skill_ids,
            skill_selection_reason=skill_selection_reason,
            phase=normalized_phase,
            depends_on=normalized_depends_on,
            seam_contract=normalized_seam_contract,
            workspace_lease_status=(
                "held"
                if self._workspace_access is not None and normalized_owned_paths
                else "not_applicable"
            ),
            deliverable=deliverable,
            notable_events=deque(),
        )
        # Reserve the first run generation before publishing the lifecycle event so
        # consumers never observe an invocation from generation 0 followed by work
        # from generation 1, and the child cannot race ahead of its invocation notice.
        runtime.run_generation = 1
        lease_claimed = False
        lease_generation: int | None = None
        try:
            session_id = await runtime.loop.prepare_session(
                start_reason="subagent_initial"
            )
            if self._workspace_access is not None:
                await self._workspace_access.claim_paths(
                    owner=f"subagent:{subagent_id}",
                    paths=normalized_owned_paths,
                )
                lease_claimed = True
                lease_generation = await self._workspace_access.lease_generation()
                runtime.workspace_lease_status = "held"
                await self._begin_workspace_snapshot(runtime)
                runtime.loop.append_system_note(
                    (
                        "Workspace ownership established. You may read the shared workspace and "
                        "write your owned paths. Writes to another actor's paths are blocked by "
                        "the runtime."
                    ),
                    session_id=session_id,
                    metadata={
                        "workspace_lease": True,
                        "workspace_lease_generation": lease_generation,
                        "owned_paths": list(normalized_owned_paths),
                    },
                )
        except BaseException:
            if lease_claimed and self._workspace_access is not None:
                await self._workspace_access.release_owner(
                    owner=f"subagent:{subagent_id}"
                )
            await runtime.loop.aclose()
            raise
        self._subagents[subagent_id] = runtime
        try:
            self._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id=subagent_id,
                    codename=codename,
                    status="running",
                    created_at=created_at,
                    updated_at=created_at,
                    route_id=self._route_id,
                    owner_main_session_id=owner_main_session_id,
                    owner_main_turn_id=owner_main_turn_id,
                    task_label=normalized_task_label,
                    instructions=normalized_instructions,
                    user_constraints=user_constraints,
                    shared_context=shared_context,
                    owned_paths=normalized_owned_paths,
                    skill_ids=normalized_skill_ids,
                    skill_selection_reason=skill_selection_reason,
                    phase=normalized_phase,
                    depends_on=normalized_depends_on,
                    seam_contract=normalized_seam_contract,
                    changed_paths_complete=runtime.changed_paths_complete,
                    changed_paths_source=runtime.changed_paths_source,
                    workspace_lease_status=runtime.workspace_lease_status,
                    deliverable=deliverable,
                    current_subagent_session_id=session_id,
                    run_generation=runtime.run_generation,
                )
            )
        except BaseException:
            self._subagents.pop(subagent_id, None)
            if lease_claimed and self._workspace_access is not None:
                await self._workspace_access.release_owner(owner=f"subagent:{subagent_id}")
            await runtime.loop.aclose()
            raise
        self._append_notable_event(runtime, kind="spawned", summary=f"Spawned {codename}.")
        await self._publish_event(
            RouteSystemNoticeEvent(
                route_id=self._route_id,
                agent_kind="subagent",
                agent_name=codename,
                subagent_id=subagent_id,
                session_id=session_id,
                origin_session_id=owner_main_session_id,
                origin_turn_id=owner_main_turn_id,
                actor_run_generation=runtime.run_generation,
                notice_kind="subagent_invoked",
                text="came online.",
                public=True,
            )
        )
        self._launch_runtime_task(
            runtime,
            user_text=build_subagent_kickoff_text(),
            force_session_id=session_id,
            pre_turn_messages=(),
            name=f"jarvis-subagent-{codename}-{subagent_id}",
            run_generation=runtime.run_generation,
        )
        return {
            "subagent_id": subagent_id,
            "codename": codename,
            "task_label": normalized_task_label,
            "status": runtime.status,
            "session_id": session_id,
            "skill_ids": list(normalized_skill_ids),
            "skill_selection_reason": skill_selection_reason,
            "phase": normalized_phase,
            "depends_on": list(normalized_depends_on),
            "seam_contract": normalized_seam_contract,
            "workspace_lease_status": runtime.workspace_lease_status,
            "changed_paths": list(runtime.changed_paths),
            "changed_paths_complete": runtime.changed_paths_complete,
            "changed_paths_source": runtime.changed_paths_source,
            "owned_paths": list(normalized_owned_paths),
            "workspace_lease_generation": lease_generation,
            "active_count": len(self._non_disposed_runtimes()),
        }

    async def _begin_workspace_snapshot(self, runtime: SubagentRuntime) -> None:
        """Start an exact change-capture segment after a child lease is held."""

        if (
            self._workspace_access is None
            or runtime.workspace_lease_status != "held"
            or not runtime.owned_paths
        ):
            runtime.workspace_snapshot_baseline = None
            return
        try:
            runtime.workspace_snapshot_baseline = await asyncio.to_thread(
                workspace_snapshot_paths,
                self._core_settings.workspace_dir,
                runtime.owned_paths,
            )
        except WorkspaceSnapshotError as exc:
            runtime.workspace_snapshot_baseline = None
            runtime.workspace_snapshot_incomplete = True
            runtime.changed_paths_complete = False
            runtime.changed_paths_source = "snapshot_unavailable"
            self._append_notable_event(
                runtime,
                kind="workspace_snapshot_unavailable",
                summary=f"Could not capture the child workspace baseline: {exc}",
            )
            LOGGER.warning(
                "Could not capture workspace snapshot baseline for subagent %s: %s",
                runtime.subagent_id,
                exc,
            )
            return
        runtime.changed_paths_complete = False
        runtime.changed_paths_source = "scoped_workspace_snapshot_pending"

    async def _finalize_workspace_snapshot(self, runtime: SubagentRuntime) -> bool:
        """Record exact net changes for the currently held lease segment."""

        baseline = runtime.workspace_snapshot_baseline
        if baseline is None:
            return runtime.changed_paths_complete
        try:
            current = await asyncio.to_thread(
                workspace_snapshot_paths,
                self._core_settings.workspace_dir,
                runtime.owned_paths,
            )
        except WorkspaceSnapshotError as exc:
            runtime.workspace_snapshot_baseline = None
            runtime.workspace_snapshot_incomplete = True
            runtime.changed_paths_complete = False
            runtime.changed_paths_source = "snapshot_unavailable"
            self._append_notable_event(
                runtime,
                kind="workspace_snapshot_unavailable",
                summary=f"Could not capture the child workspace result: {exc}",
            )
            LOGGER.warning(
                "Could not capture workspace snapshot result for subagent %s: %s",
                runtime.subagent_id,
                exc,
            )
            return False

        changed_paths = diff_workspace_snapshots(baseline, current)
        runtime.changed_paths.update(changed_paths)
        runtime.changed_test_artifact_paths.update(
            test_artifact_paths_from_paths(changed_paths)
        )
        runtime.workspace_snapshot_baseline = None
        runtime.changed_paths_complete = not runtime.workspace_snapshot_incomplete
        runtime.changed_paths_source = (
            "scoped_workspace_snapshot"
            if runtime.changed_paths_complete
            else "scoped_workspace_snapshot_incomplete"
        )
        return True

    async def monitor(
        self,
        *,
        agent: str | None = None,
        detail: str = "summary",
    ) -> dict[str, Any]:
        if agent is None:
            targets = self._non_disposed_runtimes()
        else:
            targets = [self._require_runtime(agent)]
        payload = {
            "count": len(targets),
            "subagents": [
                self._serialize_snapshot(runtime, detail=detail)
                for runtime in targets
            ],
        }
        monitor_key = (agent or "__all__") + ":" + detail
        signature = repr(payload)
        last_signature = self._last_monitor_signatures.get(monitor_key)
        self._last_monitor_signatures[monitor_key] = signature
        if last_signature == signature:
            return {
                "count": len(targets),
                "changed": False,
                "message": (
                    "No subagent state changes since the last monitor. Wait for orchestrator "
                    "updates instead of polling unless immediate detail is required."
                ),
                "subagents": [
                    {
                        "subagent_id": snapshot["subagent_id"],
                        "codename": snapshot["codename"],
                        "task_label": snapshot["task_label"],
                        "status": snapshot["status"],
                        "last_activity_at": snapshot["last_activity_at"],
                        "report_complete": snapshot["report_complete"],
                        "pending_background_job_count": snapshot["pending_background_job_count"],
                        "pending_background_job_ids": snapshot.get(
                            "pending_background_job_ids",
                            [],
                        ),
                    }
                    for snapshot in payload["subagents"]
                ],
            }
        payload["changed"] = True
        return payload

    def request_stop_all_for_user_stop(self) -> tuple[SubagentSnapshot, ...]:
        return self._request_stop_all(
            pause_reason="main_stop",
            interruption_reason="user_stop",
            hard=False,
        )

    def request_hard_stop_all_for_shutdown(self) -> tuple[SubagentSnapshot, ...]:
        return self._request_stop_all(
            pause_reason="process_shutdown",
            interruption_reason="process_shutdown",
            hard=True,
        )

    def _request_stop_all(
        self,
        *,
        pause_reason: SubagentPauseReason,
        interruption_reason: InterruptionReason,
        hard: bool,
    ) -> tuple[SubagentSnapshot, ...]:
        affected: list[SubagentSnapshot] = []
        for runtime in self._non_disposed_runtimes():
            if hard and runtime.status == "waiting_background":
                runtime.pending_pause_reason = pause_reason
                affected.append(runtime.snapshot())
                continue
            if self._request_runtime_stop(
                runtime,
                pause_reason=pause_reason,
                interruption_reason=interruption_reason,
                hard=hard,
            ):
                affected.append(runtime.snapshot())
        return tuple(affected)

    def request_hard_stop_all_for_new_session(self) -> tuple[SubagentSnapshot, ...]:
        return self._request_stop_all(
            pause_reason="new_session",
            interruption_reason="new_session",
            hard=True,
        )

    def request_hard_stop_all_for_user_stop(self) -> tuple[SubagentSnapshot, ...]:
        return self._request_stop_all(
            pause_reason="main_stop",
            interruption_reason="user_stop",
            hard=True,
        )

    async def settle_hard_user_stop(
        self,
        *,
        subagent_ids: frozenset[str],
    ) -> tuple[SubagentSnapshot, ...]:
        return await self.settle_hard_stop(
            subagent_ids=subagent_ids,
            pause_reason="main_stop",
        )

    async def settle_hard_stop(
        self,
        *,
        subagent_ids: frozenset[str],
        pause_reason: SubagentPauseReason,
    ) -> tuple[SubagentSnapshot, ...]:
        settled: list[SubagentSnapshot] = []
        for runtime in tuple(self._non_disposed_runtimes()):
            if runtime.subagent_id not in subagent_ids:
                continue
            await self._wait_for_turn_settle(runtime)
            if runtime.status in {"running", "waiting_background", "awaiting_approval"}:
                runtime.status = "paused"
            if runtime.pending_pause_reason == pause_reason:
                runtime.pause_reason = pause_reason
                runtime.pending_pause_reason = None
            runtime.pending_background_job_ids.clear()
            self._pending_bash_job_notices.pop(runtime.subagent_id, None)
            await self._finalize_workspace_snapshot(runtime)
            self._sync_catalog(runtime)
            settled.append(runtime.snapshot())
        return tuple(settled)

    async def stop(self, *, agent: str, reason: str | None = None) -> dict[str, Any]:
        runtime = self._require_runtime(agent)
        if not self._request_runtime_stop(runtime, pause_reason="main_stop"):
            return {
                "subagent_id": runtime.subagent_id,
                "codename": runtime.codename,
                "status": runtime.status,
                "changed": False,
            }
        await self._wait_for_turn_settle(runtime)
        if reason and reason.strip():
            self._append_notable_event(
                runtime,
                kind="stopped",
                summary=f"Paused by Jarvis: {reason.strip()}",
            )
        return {
            "subagent_id": runtime.subagent_id,
            "codename": runtime.codename,
            "status": runtime.status,
            "changed": True,
        }

    async def step_in(self, *, agent: str, instructions: str) -> dict[str, Any]:
        runtime = self._require_runtime(agent)
        if runtime.status == "disposed":
            raise ValueError(f"Subagent {agent} has already been disposed.")
        if runtime.status == "waiting_background" or runtime.pending_background_job_ids:
            raise ValueError(
                "Cannot step into a subagent while detached bash jobs are still pending."
            )
        if runtime.status in {"running", "awaiting_approval"}:
            runtime.pending_pause_reason = "main_stop"
            runtime.loop.request_stop()
            await self._wait_for_turn_settle(runtime)
        if runtime.workspace_lease_status == "held":
            await self._finalize_workspace_snapshot(runtime)
        await self._reacquire_workspace_lease_if_needed(runtime)
        if runtime.workspace_lease_status == "held" and runtime.workspace_snapshot_baseline is None:
            await self._begin_workspace_snapshot(runtime)
        runtime.pause_reason = None
        runtime.status = "running"
        runtime.report_complete = False
        self._sync_catalog(runtime)
        self._append_notable_event(runtime, kind="step_in", summary="Jarvis stepped in with new direction.")
        self._launch_runtime_task(
            runtime,
            user_text="Continue with the updated direction above.",
            force_session_id=runtime.loop.active_session_id(),
            pre_turn_messages=(build_step_in_message(instructions=instructions),),
            name=f"jarvis-subagent-step-in-{runtime.codename}-{runtime.subagent_id}",
        )
        return {
            "subagent_id": runtime.subagent_id,
            "codename": runtime.codename,
            "status": runtime.status,
            "changed": True,
        }

    async def handoff(self, *, agent: str) -> dict[str, Any]:
        """Release a settled child's write lease without disposing its state."""

        runtime = self._require_runtime(agent)
        if runtime.status == "disposed":
            raise ValueError(f"Subagent {agent} has already been disposed.")
        if (
            runtime.status in {"running", "awaiting_approval", "waiting_background"}
            or runtime.pending_background_job_ids
            or (runtime.task is not None and not runtime.task.done())
        ):
            raise ValueError("Cannot hand off a running subagent. Stop it first.")

        await self._finalize_workspace_snapshot(runtime)
        changed = runtime.workspace_lease_status == "held"
        if changed and self._workspace_access is not None:
            await self._workspace_access.release_owner(
                owner=f"subagent:{runtime.subagent_id}"
            )
            runtime.workspace_lease_status = "released"
            self._append_notable_event(
                runtime,
                kind="handoff",
                summary="Released workspace ownership for main-agent integration.",
            )
            self._sync_catalog(runtime)
        return {
            "subagent_id": runtime.subagent_id,
            "codename": runtime.codename,
            "status": runtime.status,
            "changed": changed,
            "handoff_ready": True,
            "workspace_lease_status": runtime.workspace_lease_status,
            "changed_paths": sorted(runtime.changed_paths),
            "changed_paths_complete": runtime.changed_paths_complete,
            "changed_paths_source": runtime.changed_paths_source,
            "changed_test_artifact_paths": sorted(
                runtime.changed_test_artifact_paths
            ),
        }

    async def dispose(self, *, agent: str) -> dict[str, Any]:
        runtime = self._require_runtime(agent)
        if (
            runtime.status in {"running", "awaiting_approval", "waiting_background"}
            or runtime.pending_background_job_ids
            or (runtime.task is not None and not runtime.task.done())
        ):
            raise ValueError("Cannot dispose a running subagent. Stop it first.")
        return await self._dispose_runtime(runtime, public_notice=True)

    async def reset_for_new_session(self) -> dict[str, Any]:
        self.request_hard_stop_all_for_new_session()
        for runtime in tuple(self._non_disposed_runtimes()):
            await self._wait_for_turn_settle(runtime)

        disposed_subagent_ids: list[str] = []
        live_subagent_ids = set(self._subagents)

        for runtime in tuple(self._subagents.values()):
            if runtime.status == "disposed":
                continue
            runtime.pending_background_job_ids.clear()
            self._pending_bash_job_notices.pop(runtime.subagent_id, None)
            payload = await self._dispose_runtime(runtime, public_notice=False)
            if payload["changed"]:
                disposed_subagent_ids.append(runtime.subagent_id)

        for entry in self._catalog.list_entries():
            if entry.status == "disposed" or entry.subagent_id in live_subagent_ids:
                continue
            self._pending_bash_job_notices.pop(entry.subagent_id, None)
            disposed_at = _utc_now_iso()
            self._catalog.update_entry(
                entry.subagent_id,
                status="disposed",
                pause_reason=None,
                disposed_at=disposed_at,
            )
            disposed_subagent_ids.append(entry.subagent_id)

        self._last_monitor_signatures.clear()
        self._last_main_context_signature = None
        return {
            "disposed_subagent_ids": disposed_subagent_ids,
            "cancelled_job_ids": [],
            "disposed_count": len(disposed_subagent_ids),
            "cancelled_job_count": 0,
        }

    def main_turn_runtime_messages(self, *, session_id: str) -> tuple[AgentRuntimeMessage, ...]:
        runtimes = self._non_disposed_runtimes()
        if not runtimes:
            return self._deduplicate_main_context_message(
                session_id=session_id,
                message=AgentRuntimeMessage(
                    role="system",
                    metadata={
                        "subagent_status_snapshot": True,
                        "pending_subagent_ids": [],
                    },
                    content="Subagent status snapshot:\n- no non-disposed subagents.",
                )
            )

        lines = ["Subagent status snapshot:"]
        recent_events: list[tuple[str, str, str]] = []
        pending_subagent_ids: list[str] = []
        for runtime in runtimes:
            snapshot = runtime.snapshot()
            status_line = (
                f"- {snapshot.codename} [{snapshot.task_label}] "
                f"({snapshot.subagent_id}): {snapshot.status}"
            )
            extras: list[str] = []
            if snapshot.pending_background_job_count > 0:
                extras.append(
                    f"pending_background_jobs={snapshot.pending_background_job_count}"
                )
                extras.append(
                    "pending_background_job_ids="
                    + ",".join(snapshot.pending_background_job_ids)
                )
            if snapshot.pause_reason is not None:
                extras.append(f"pause_reason={snapshot.pause_reason}")
            if snapshot.last_tool_name is not None:
                extras.append(f"last_tool={snapshot.last_tool_name}")
            if snapshot.last_activity_at is not None:
                extras.append(f"last_activity_at={snapshot.last_activity_at}")
            if snapshot.phase is not None:
                extras.append(f"phase={snapshot.phase}")
            if snapshot.workspace_lease_status != "not_applicable":
                extras.append(f"workspace_lease={snapshot.workspace_lease_status}")
            if snapshot.changed_paths_complete:
                extras.append("changed_paths_complete=true")
            if snapshot.changed_paths_source != "tool_result_metadata":
                extras.append(f"changed_paths_source={snapshot.changed_paths_source}")
            extras.append(f"report_complete={str(snapshot.report_complete).lower()}")
            if snapshot.last_error is not None:
                extras.append(f"last_error={snapshot.last_error}")
            if extras:
                status_line += " [" + ", ".join(extras) + "]"
            lines.append(status_line)
            if snapshot.status in {"running", "waiting_background", "awaiting_approval"}:
                pending_subagent_ids.append(snapshot.subagent_id)
            for event in list(snapshot.notable_events)[-self._settings.main_context_event_limit :]:
                recent_events.append((snapshot.codename, event.kind, event.summary))

        if recent_events:
            lines.append("")
            lines.append("Recent noteworthy subagent events:")
            for codename, kind, summary in recent_events[-self._settings.main_context_event_limit :]:
                lines.append(f"- {codename} [{kind}]: {summary}")

        return self._deduplicate_main_context_message(
            session_id=session_id,
            message=AgentRuntimeMessage(
                role="system",
                metadata={
                    "subagent_status_snapshot": True,
                    "pending_subagent_ids": pending_subagent_ids,
                },
                content="\n".join(lines),
            )
        )

    def active_snapshots(self) -> tuple[SubagentSnapshot, ...]:
        return tuple(runtime.snapshot() for runtime in self._non_disposed_runtimes())

    def snapshot_for(self, agent: str) -> SubagentSnapshot | None:
        runtime = self._subagents.get(agent)
        if runtime is None:
            return None
        return runtime.snapshot()

    def build_main_progress_message(
        self,
        *,
        agent: str,
        notice_kind: str,
        notice_text: str,
    ) -> tuple[str | None, AgentRuntimeMessage] | None:
        runtime = self._subagents.get(agent)
        if runtime is not None and runtime.latest_report is None:
            runtime.latest_report = self._latest_assistant_report(runtime)
        snapshot = runtime.snapshot() if runtime is not None else self.snapshot_for(agent)
        if snapshot is None:
            return None
        if snapshot.status == "disposed":
            return None
        recommendation = self._recommend_main_supervision_action(
            notice_kind=notice_kind,
            snapshot=snapshot,
        )
        parts = [
            f"subagent={snapshot.codename}",
            f'task="{self._truncate_for_notice(snapshot.task_label, max_length=120)}"',
            f"id={snapshot.subagent_id}",
            f"status={snapshot.status}",
            f"notice={notice_kind}",
        ]
        if snapshot.pending_background_job_ids:
            parts.append(
                "pending_background_job_ids="
                + ",".join(snapshot.pending_background_job_ids)
            )
        if snapshot.changed_test_artifact_paths:
            parts.append(
                "changed_test_artifacts="
                + ",".join(snapshot.changed_test_artifact_paths)
            )
        if snapshot.changed_paths:
            parts.append("changed_paths=" + ",".join(snapshot.changed_paths))
        if snapshot.changed_paths_source != "tool_result_metadata":
            parts.append(f"changed_paths_evidence={snapshot.changed_paths_source}")
        if snapshot.phase is not None:
            parts.append(f"phase={snapshot.phase}")
        if snapshot.workspace_lease_status != "not_applicable":
            parts.append(f"workspace_lease={snapshot.workspace_lease_status}")
        if notice_text.strip():
            parts.append(f'note="{self._truncate_for_notice(notice_text, max_length=140)}"')
        latest_report = snapshot.latest_report
        rendered_report, report_truncated = self._render_subagent_report_for_main(latest_report)
        content = "\n".join(
            self._build_main_progress_lines(
                parts=parts,
                recommendation=recommendation,
                latest_report=rendered_report,
                report_complete=snapshot.report_complete,
                report_truncated=report_truncated,
                changed_paths=snapshot.changed_paths,
                changed_test_artifact_paths=snapshot.changed_test_artifact_paths,
            )
        )
        return (
            snapshot.owner_main_session_id,
            AgentRuntimeMessage(
                role="system",
                metadata={
                    "subagent_progress_update": True,
                    "notice_kind": "subagent_progress_update",
                    "subagent_id": snapshot.subagent_id,
                    "subagent_notice_kind": notice_kind,
                    "subagent_status": snapshot.status,
                    "recommended_action": recommendation,
                    "latest_subagent_report_included": bool(latest_report),
                    "latest_subagent_report_complete": snapshot.report_complete,
                    "latest_subagent_report_truncated": report_truncated,
                    "phase": snapshot.phase,
                    "depends_on": list(snapshot.depends_on),
                    "workspace_lease_status": snapshot.workspace_lease_status,
                    "changed_paths": list(snapshot.changed_paths),
                    "changed_paths_complete": snapshot.changed_paths_complete,
                    "changed_paths_source": snapshot.changed_paths_source,
                    "changed_test_artifact_paths": list(
                        snapshot.changed_test_artifact_paths
                    ),
                    "pending_subagent_ids": (
                        [snapshot.subagent_id]
                        if snapshot.status in {"running", "waiting_background", "awaiting_approval"}
                        else []
                    ),
                },
                content=content,
            ),
        )

    def _build_main_progress_lines(
        self,
        *,
        parts: list[str],
        recommendation: str,
        latest_report: str | None,
        report_complete: bool,
        report_truncated: bool,
        changed_paths: tuple[str, ...],
        changed_test_artifact_paths: tuple[str, ...],
    ) -> list[str]:
        lines = [
            "Subagent update.",
            "- " + " ".join(parts),
            f"recommendation={recommendation}",
        ]
        if changed_test_artifact_paths:
            lines.append(
                "Subagent changed test artifacts: " + ",".join(changed_test_artifact_paths)
            )
        if changed_paths:
            lines.append("Subagent changed paths: " + ",".join(changed_paths))
        if recommendation in {"finalize", "inspect"} and latest_report is not None:
            report_heading = (
                "Complete subagent report:"
                if report_complete and not report_truncated
                else "Latest subagent checkpoint:"
            )
            lines.extend(
                [
                    report_heading,
                    latest_report,
                ]
            )
            if report_complete and not report_truncated:
                lines.append(
                    "The report is self-reported completion, not semantic acceptance. Review the "
                    "changed paths and seam before integrating this child."
                )
            else:
                lines.append(
                    "This checkpoint is incomplete or truncated. Inspect the child with "
                    "`subagent_monitor(detail=\"full\")` before relying on it or deciding how to "
                    "continue."
                )
        else:
            lines.append(
                "This is a system update from the orchestrator, not a new user message. "
                "Subagent progress is orchestrator-monitored; react to this update and "
                "update the user accordingly instead of polling unless immediate detail is "
                "required."
            )
            return lines
        lines.append(
            "This is a system update from the orchestrator, not a new user message. "
            "Subagent progress is orchestrator-monitored; react to this update and "
            "update the user accordingly instead of polling unless immediate detail is "
            "required."
        )
        return lines

    def is_turn_active(self, subagent_id: str) -> bool:
        runtime = self._subagents.get(subagent_id)
        if runtime is None:
            return False
        task = runtime.task
        if task is not None and not task.done():
            return True
        has_active_turn = getattr(runtime.loop, "has_active_turn", None)
        if callable(has_active_turn):
            return bool(has_active_turn())
        return False

    def main_followup_runtime_messages(
        self,
        *,
        agent: str,
        notice_kind: str,
        notice_text: str,
    ) -> tuple[AgentRuntimeMessage, ...]:
        runtime = self._require_runtime(agent)
        snapshot = runtime.snapshot()
        lines = [
            "Subagent supervisor follow-up:",
            f"- codename: {snapshot.codename}",
            f"- task_label: {snapshot.task_label}",
            f"- subagent_id: {snapshot.subagent_id}",
            f"- status: {snapshot.status}",
            f"- notice_kind: {notice_kind}",
            f"- notice_text: {notice_text}",
        ]
        if snapshot.current_subagent_session_id is not None:
            lines.append(f"- current_subagent_session_id: {snapshot.current_subagent_session_id}")
        if snapshot.phase is not None:
            lines.append(f"- phase: {snapshot.phase}")
        if snapshot.depends_on:
            lines.append(f"- depends_on: {', '.join(snapshot.depends_on)}")
        if snapshot.seam_contract is not None:
            lines.extend(["- seam_contract:", snapshot.seam_contract])
        if snapshot.workspace_lease_status != "not_applicable":
            lines.append(f"- workspace_lease_status: {snapshot.workspace_lease_status}")
        if snapshot.changed_paths:
            lines.append(f"- changed_paths: {', '.join(snapshot.changed_paths)}")
        lines.append(
            f"- changed_paths_complete: {str(snapshot.changed_paths_complete).lower()}"
        )
        lines.append(f"- changed_paths_source: {snapshot.changed_paths_source}")
        if snapshot.changed_test_artifact_paths:
            lines.append(
                "- changed_test_artifact_paths: "
                + ", ".join(snapshot.changed_test_artifact_paths)
            )
        if snapshot.pause_reason is not None:
            lines.append(f"- pause_reason: {snapshot.pause_reason}")
        if snapshot.last_error is not None:
            lines.append(f"- last_error: {snapshot.last_error}")
        if snapshot.last_error_metadata:
            lines.append(
                "- last_error_metadata: "
                + json.dumps(snapshot.last_error_metadata, ensure_ascii=False, sort_keys=True)
            )
        if snapshot.error_log_path is not None:
            lines.append(f"- error_log_path: {snapshot.error_log_path}")
        if snapshot.last_tool_name is not None:
            lines.append(f"- last_tool_name: {snapshot.last_tool_name}")
        if snapshot.last_activity_at is not None:
            lines.append(f"- last_activity_at: {snapshot.last_activity_at}")
        if snapshot.notable_events:
            lines.append("- recent_noteworthy_events:")
            for note in snapshot.notable_events[-self._settings.main_context_event_limit :]:
                lines.append(f"  - {note.created_at} [{note.kind}] {note.summary}")

        final_report = runtime.latest_report or self._latest_assistant_report(runtime)
        if final_report is not None:
            lines.extend(
                [
                    "",
                    "Latest subagent report:",
                    final_report,
                    f"report_complete: {str(snapshot.report_complete).lower()}",
                ]
            )

        lines.extend(
            [
                "",
                "You remain responsible for supervision. Review the completed or paused subagent work, decide whether verification is still needed, dispose the subagent when appropriate, and report the result to the user.",
            ]
        )
        return (
            AgentRuntimeMessage(
                role="system",
                metadata={
                    "subagent_followup": True,
                    "subagent_id": snapshot.subagent_id,
                    "codename": snapshot.codename,
                    "notice_kind": notice_kind,
                },
                content="\n".join(lines),
            ),
        )

    async def enqueue_bash_job_followup(self, notices: tuple[BashJobNotice, ...]) -> bool:
        if not notices:
            return False
        runtime = self._require_runtime(notices[0].owner_subagent_id or "")
        queue = self._pending_bash_job_notices.setdefault(runtime.subagent_id, {})
        for notice in notices:
            queue[notice.job_id] = notice
        self._append_notable_event(
            runtime,
            kind="bash_job_ready",
            summary=(
                "Detached bash job updates are ready: "
                f"{', '.join(notice.job_id[:8] for notice in notices)}."
            ),
        )
        self._sync_catalog(runtime)
        await self._maybe_start_next_bash_job_followup(runtime)
        return True

    async def _run_turn(
        self,
        runtime: SubagentRuntime,
        *,
        run_generation: int,
        user_text: str | None,
        force_session_id: str | None,
        pre_turn_messages: tuple[AgentRuntimeMessage, ...],
        runtime_turn: bool = False,
    ) -> None:
        runtime.status = "running"
        runtime.pause_reason = None
        runtime.last_error = None
        runtime.last_error_metadata.clear()
        runtime.error_log_path = None
        runtime.report_complete = False
        self._sync_catalog(runtime)
        try:
            if runtime_turn:
                event_stream = runtime.loop.stream_runtime_turn(
                    force_session_id=force_session_id,
                    pre_turn_messages=pre_turn_messages,
                )
            else:
                event_stream = runtime.loop.stream_turn(
                    user_text=user_text or "",
                    force_session_id=force_session_id,
                    pre_turn_messages=pre_turn_messages,
                )
            async for event in event_stream:
                if run_generation != runtime.run_generation:
                    return
                runtime.last_activity_at = _utc_now_iso()
                self._sync_catalog(runtime)
                if runtime.status == "awaiting_approval" and isinstance(
                    event,
                    (
                        AgentTextDeltaEvent,
                        AgentAssistantMessageEvent,
                        AgentToolCallEvent,
                    ),
                ):
                    runtime.status = "running"
                    self._sync_catalog(runtime)
                    self._append_notable_event(
                        runtime,
                        kind="resumed",
                        summary="Resumed after approval.",
                    )
                    await self._publish_lifecycle_notice(
                        runtime,
                        notice_kind="subagent_resumed",
                        text="resumed after approval.",
                        session_id=event.session_id,
                    )
                if isinstance(event, AgentTextDeltaEvent):
                    continue
                if isinstance(event, AgentAssistantMessageEvent):
                    self._capture_assistant_checkpoint(runtime, event.text)
                    continue
                if isinstance(event, AgentToolCallEvent):
                    runtime.last_tool_name = event.tool_names[-1] if event.tool_names else None
                    self._append_notable_event(
                        runtime,
                        kind="tool_call",
                        summary=f"Used tools: {', '.join(event.tool_names)}",
                    )
                    await self._publish_event(
                        RouteToolCallEvent(
                            route_id=self._route_id,
                            agent_kind="subagent",
                            agent_name=runtime.codename,
                            session_id=event.session_id,
                            turn_id=event.turn_id or None,
                            subagent_id=runtime.subagent_id,
                            origin_session_id=runtime.owner_main_session_id,
                            origin_turn_id=runtime.owner_main_turn_id,
                            actor_run_generation=runtime.run_generation,
                            tool_names=event.tool_names,
                        )
                    )
                    continue
                if isinstance(event, AgentApprovalRequestEvent):
                    runtime.status = "awaiting_approval"
                    self._sync_catalog(runtime)
                    self._append_notable_event(
                        runtime,
                        kind="awaiting_approval",
                        summary=event.summary or "Awaiting approval.",
                    )
                    self._register_approval_target(event.approval_id, runtime.loop)
                    await self._publish_event(
                        RouteApprovalRequestEvent(
                            route_id=self._route_id,
                            agent_kind="subagent",
                            agent_name=runtime.codename,
                            session_id=event.session_id,
                            subagent_id=runtime.subagent_id,
                            origin_session_id=runtime.owner_main_session_id,
                            origin_turn_id=runtime.owner_main_turn_id,
                            actor_run_generation=runtime.run_generation,
                            approval_id=event.approval_id,
                            kind=event.kind,
                            summary=event.summary,
                            details=event.details,
                            command=event.command,
                            tool_name=event.tool_name,
                            inspection_url=event.inspection_url,
                        )
                    )
                    continue
                if isinstance(event, AgentTurnDoneEvent):
                    self._capture_assistant_checkpoint(runtime, event.response_text)
                    if event.interrupted:
                        runtime.status = "paused"
                        runtime.pause_reason = runtime.pending_pause_reason or "main_stop"
                        runtime.report_complete = False
                        self._append_notable_event(
                            runtime,
                            kind="paused",
                            summary=f"Paused ({runtime.pause_reason}).",
                        )
                        if not runtime.pending_background_job_ids:
                            await self._finalize_workspace_snapshot(runtime)
                        await self._publish_lifecycle_notice(
                            runtime,
                            notice_kind="subagent_paused",
                            text=f"paused ({runtime.pause_reason}).",
                            session_id=event.session_id,
                        )
                    elif event.completion_blocked:
                        block_reason = (
                            event.completion_block_reason or "external_blocked"
                        )
                        runtime.status = "paused"
                        runtime.pause_reason = _completion_block_pause_reason(
                            block_reason
                        )
                        runtime.report_complete = False
                        self._append_notable_event(
                            runtime,
                            kind="completion_blocked",
                            summary=(
                                "Completion blocked by runtime condition: "
                                f"{block_reason}."
                            ),
                        )
                        if not runtime.pending_background_job_ids:
                            await self._finalize_workspace_snapshot(runtime)
                        await self._publish_lifecycle_notice(
                            runtime,
                            notice_kind="subagent_needs_attention",
                            text=f"paused ({block_reason}).",
                            session_id=event.session_id,
                        )
                    elif event.approval_rejected:
                        runtime.status = "paused"
                        runtime.pause_reason = "approval_rejected"
                        runtime.report_complete = False
                        self._append_notable_event(
                            runtime,
                            kind="approval_rejected",
                            summary="Approval was rejected and the subagent paused.",
                        )
                        if not runtime.pending_background_job_ids:
                            await self._finalize_workspace_snapshot(runtime)
                        await self._publish_lifecycle_notice(
                            runtime,
                            notice_kind="subagent_approval_rejected",
                            text="paused because approval was rejected.",
                            session_id=event.session_id,
                        )
                    else:
                        if runtime.pending_background_job_ids:
                            runtime.status = "waiting_background"
                            runtime.pause_reason = None
                            runtime.report_complete = False
                            self._append_notable_event(
                                runtime,
                                kind="waiting_background",
                                summary=(
                                    "Waiting for detached bash jobs: "
                                    f"{len(runtime.pending_background_job_ids)} pending."
                                ),
                            )
                            await self._publish_lifecycle_notice(
                                runtime,
                                notice_kind="subagent_waiting_background",
                                text=(
                                    "waiting on detached bash jobs: "
                                    f"{', '.join(sorted(runtime.pending_background_job_ids))}."
                                ),
                                session_id=event.session_id,
                            )
                        else:
                            runtime.status = "completed"
                            runtime.pause_reason = None
                            runtime.report_complete = bool(runtime.latest_report)
                            self._append_notable_event(
                                runtime,
                                kind="completed",
                                summary="Completed the assigned turn.",
                            )
                            await self._finalize_workspace_snapshot(runtime)
                            await self._publish_lifecycle_notice(
                                runtime,
                                notice_kind="subagent_completed",
                                text="completed.",
                                session_id=event.session_id,
                            )
                    runtime.pending_pause_reason = None
                    runtime.provider_recovery_attempts = 0
                    self._sync_catalog(runtime)
                    return
        except Exception as exc:
            if run_generation != runtime.run_generation:
                return
            if (
                isinstance(
                    exc,
                    (ProviderTemporaryError, ProviderRateLimitError),
                )
                and runtime.provider_recovery_attempts
                < _MAX_SUBAGENT_PROVIDER_RECOVERY_ATTEMPTS
                and runtime.pending_pause_reason is None
            ):
                runtime.provider_recovery_attempts += 1
                runtime.status = "running"
                runtime.last_error = f"{type(exc).__name__}: {exc}"
                runtime.last_error_metadata = _exception_metadata(exc)
                runtime.error_log_path = self._record_subagent_error(runtime, exc)
                runtime.report_complete = False
                self._append_notable_event(
                    runtime,
                    kind="provider_recovery",
                    summary=(
                        f"Recovering from {type(exc).__name__} "
                        f"(attempt {runtime.provider_recovery_attempts}/"
                        f"{_MAX_SUBAGENT_PROVIDER_RECOVERY_ATTEMPTS})."
                    ),
                )
                self._sync_catalog(runtime)
                await self._publish_lifecycle_notice(
                    runtime,
                    notice_kind="subagent_recovering",
                    text="recovering automatically from a transient provider failure.",
                )
                self._launch_runtime_task(
                    runtime,
                    user_text=None,
                    force_session_id=runtime.loop.active_session_id(),
                    pre_turn_messages=(
                        AgentRuntimeMessage(
                            role="system",
                            metadata={
                                "subagent_provider_recovery": True,
                                "attempt": runtime.provider_recovery_attempts,
                            },
                            content=(
                                "The previous provider attempt failed. Continue from the durable "
                                "checkpoint without replaying completed tools."
                            ),
                        ),
                    ),
                    runtime_turn=True,
                    name=(
                        f"jarvis-subagent-provider-recovery-"
                        f"{runtime.codename}-{runtime.subagent_id}"
                    ),
                )
                return
            runtime.status = "failed"
            runtime.last_error = f"{type(exc).__name__}: {exc}"
            runtime.last_error_metadata = _exception_metadata(exc)
            runtime.error_log_path = self._record_subagent_error(runtime, exc)
            runtime.report_complete = False
            runtime.pending_pause_reason = None
            self._append_notable_event(
                runtime,
                kind="failed",
                summary=runtime.last_error,
            )
            if not runtime.pending_background_job_ids:
                await self._finalize_workspace_snapshot(runtime)
            self._sync_catalog(runtime)
            LOGGER.exception(
                "Subagent %s (%s) failed.",
                runtime.codename,
                runtime.subagent_id,
            )
            await self._publish_lifecycle_notice(
                runtime,
                notice_kind="subagent_failed",
                text=f"failed: {runtime.last_error}",
            )

    def _build_subagent_loop(
        self,
        *,
        subagent_id: str,
        codename: str,
        storage: SessionStorage,
        bootstrap_loader: SubagentBootstrapLoader,
    ) -> ActorRuntime:
        filtered_registry = self._tool_registry.filtered_view(
            agent_kind="subagent",
            hidden_tool_names=self._settings.builtin_tool_blocklist,
        )
        tool_runtime = ToolRuntime(registry=filtered_registry)

        async def _execute(tool_call, context):
            if self._workspace_access is None:
                async with self._tool_execution_guard:
                    result = await tool_runtime.execute(
                        tool_call=tool_call,
                        context=context,
                    )
            else:
                try:
                    async with self._workspace_access.execute(
                        tool_call=tool_call,
                        context=context,
                    ) as workspace_observation:
                        result = await tool_runtime.execute(
                            tool_call=tool_call,
                            context=with_workspace_capabilities(
                                context,
                                workspace_observation,
                            ),
                        )
                    result = with_workspace_observation(result, workspace_observation)
                    result = _with_workspace_lease_generation(
                        result,
                        await self._workspace_access.lease_generation(),
                    )
                except WorkspaceLeaseError as exc:
                    result = _workspace_lease_error_result(tool_call, exc)
            await self._observe_tool_result(
                subagent_id=subagent_id,
                result=result,
                context=context,
            )
            return result
        resolved_provider = self._resolved_provider()
        if backend_kind_for_provider(resolved_provider) == "codex":
            if self._codex_coordinator is None:
                raise RuntimeError("Codex coordinator is required for Codex-backed subagents.")
            return CodexActorRuntime(
                coordinator=self._codex_coordinator,
                settings=self._codex_settings,
                llm_service=self._llm_service,
                storage=storage,
                core_settings=self._core_settings,
                route_id=self._route_id,
                identity=AgentIdentity(kind="subagent", name=codename, subagent_id=subagent_id),
                bootstrap_loader=bootstrap_loader,
                memory_mode=AgentMemoryMode(
                    bootstrap=False,
                    maintenance=False,
                    reflection=False,
                ),
                tool_registry=filtered_registry,
                tool_runtime=tool_runtime,
                tool_definitions_provider=lambda activated_names: tuple(
                    list(filtered_registry.basic_definitions())
                    + list(
                        filtered_registry.resolve_discoverable_tool_definitions(
                            activated_names
                        )
                    )
                ),
                tool_executor=_execute,
                publish_route_event=self._publish_event,
            )

        return AgentLoop(
            llm_service=self._llm_service,
            settings=self._core_settings,
            storage=storage,
            tool_registry=filtered_registry,
            tool_runtime=tool_runtime,
            route_id=self._route_id,
            bootstrap_loader=bootstrap_loader,
            identity=AgentIdentity(kind="subagent", name=codename, subagent_id=subagent_id),
            memory_mode=AgentMemoryMode(
                bootstrap=False,
                maintenance=False,
                reflection=False,
            ),
            llm_provider=resolved_provider,
            tool_executor=_execute,
        )

    def _resolved_provider(self) -> str:
        if self._settings.provider is not None:
            return self._settings.provider
        service_settings = getattr(self._llm_service, "settings", None)
        if service_settings is not None:
            default_provider = getattr(service_settings, "default_provider", None)
            if isinstance(default_provider, str) and default_provider.strip():
                return default_provider.strip().lower()
        return "openai"

    async def _observe_tool_result(
        self,
        *,
        subagent_id: str,
        result: ToolExecutionResult,
        context: ToolExecutionContext,
    ) -> None:
        runtime = self._subagents.get(subagent_id)
        if runtime is None:
            return
        runtime.last_tool_name = result.name
        changed_paths = changed_workspace_paths_from_result(result)
        if changed_paths and runtime.workspace_snapshot_baseline is None:
            runtime.changed_paths.update(changed_paths)
            if len(runtime.changed_paths) > _MAX_TRACKED_CHANGED_PATHS:
                runtime.changed_paths = set(
                    sorted(runtime.changed_paths)[-_MAX_TRACKED_CHANGED_PATHS:]
                )
        changed_test_paths = changed_test_artifact_paths_from_result(result)
        if changed_test_paths:
            runtime.changed_test_artifact_paths.update(changed_test_paths)
        self._append_notable_event(
            runtime,
            kind="tool_result",
            summary=self._summarize_tool_result(result),
        )
        self._sync_catalog(runtime)
        if result.name != "bash":
            return
        if self._tool_result_observer is not None:
            try:
                await self._tool_result_observer(result=result, context=context)
            except Exception:
                LOGGER.exception(
                    "Subagent detached bash observation forwarding failed for subagent %s.",
                    subagent_id,
                )

        job_id = str(result.metadata.get("job_id", "")).strip()
        if not job_id:
            return

        status = str(result.metadata.get("status") or result.metadata.get("state") or "").strip()
        mode = str(result.metadata.get("mode", "")).strip()
        is_detached_start = mode == "background" or bool(result.metadata.get("promoted_to_background"))
        if is_detached_start and status == "running":
            if job_id not in runtime.pending_background_job_ids:
                runtime.pending_background_job_ids.add(job_id)
                self._append_notable_event(
                    runtime,
                    kind="background_job_started",
                    summary=f"Detached bash job {job_id[:8]} is running.",
                )
                self._sync_catalog(runtime)
            return

        if status not in {"finished", "cancelled"}:
            return
        if context.agent_kind != "subagent" or context.subagent_id != subagent_id:
            return
        if job_id not in runtime.pending_background_job_ids:
            return
        runtime.pending_background_job_ids.discard(job_id)
        self._append_notable_event(
            runtime,
            kind="background_job_observed",
            summary=f"Observed terminal bash job {job_id[:8]} inside the subagent turn.",
        )
        self._sync_catalog(runtime)

    def _launch_runtime_task(
        self,
        runtime: SubagentRuntime,
        *,
        user_text: str | None,
        force_session_id: str | None,
        pre_turn_messages: tuple[AgentRuntimeMessage, ...],
        runtime_turn: bool = False,
        name: str,
        run_generation: int | None = None,
    ) -> None:
        if run_generation is None:
            runtime.run_generation += 1
            run_generation = runtime.run_generation
        elif run_generation != runtime.run_generation or run_generation <= 0:
            raise ValueError("Reserved subagent run generation is no longer current.")
        task = asyncio.create_task(
            self._run_turn(
                runtime,
                run_generation=run_generation,
                user_text=user_text,
                force_session_id=force_session_id,
                pre_turn_messages=pre_turn_messages,
                runtime_turn=runtime_turn,
            ),
            name=name,
        )
        runtime.task = task
        task.add_done_callback(
            lambda finished_task, *, target_runtime=runtime: asyncio.create_task(
                self._after_runtime_task_finished(
                    runtime=target_runtime,
                    finished_task=finished_task,
                ),
                name=(
                    f"jarvis-subagent-post-task-"
                    f"{target_runtime.codename}-{target_runtime.subagent_id}"
                ),
            )
        )

    async def _after_runtime_task_finished(
        self,
        *,
        runtime: SubagentRuntime,
        finished_task: asyncio.Task[None],
    ) -> None:
        if runtime.task is finished_task:
            runtime.task = None
        try:
            await finished_task
        except Exception:
            return
        await self._maybe_start_next_bash_job_followup(runtime)

    async def _maybe_start_next_bash_job_followup(self, runtime: SubagentRuntime) -> bool:
        queue = self._pending_bash_job_notices.get(runtime.subagent_id)
        if not queue:
            return False
        notices = tuple(queue.values())
        for notice in notices:
            if notice.status in {"finished", "cancelled"}:
                runtime.pending_background_job_ids.discard(notice.job_id)
            else:
                runtime.pending_background_job_ids.add(notice.job_id)
        if runtime.task is not None and not runtime.task.done():
            return False
        if runtime.status in {"awaiting_approval", "failed", "disposed"}:
            return False
        if runtime.status == "paused" and runtime.pause_reason is not None:
            return False
        session_id = notices[0].owner_session_id or runtime.loop.active_session_id()
        if session_id is None:
            return False
        recommendation = self._recommend_bash_followup_action(notices)
        message = self._build_bash_job_followup_message(
            notices,
            recommendation=recommendation,
        )
        if recommendation == "wait":
            queue.clear()
            runtime.pause_reason = None
            runtime.status = "waiting_background"
            runtime.report_complete = False
            self._append_notable_event(
                runtime,
                kind="bash_job_waiting",
                summary="Recorded unchanged detached bash progress without an LLM follow-up.",
            )
            self._sync_catalog(runtime)
            self._record_bash_notice_delivery(notices)
            return True
        if not self._append_bash_job_system_message(
            runtime,
            session_id=session_id,
            message=message,
        ):
            return False
        queue.clear()
        runtime.pause_reason = None
        runtime.status = "running"
        self._append_notable_event(
            runtime,
            kind="bash_job_followup",
            summary=(
                "Resuming after detached bash job updates: "
                f"{', '.join(notice.job_id[:8] for notice in notices)}."
            ),
        )
        self._sync_catalog(runtime)
        self._record_bash_notice_delivery(notices)
        await self._publish_lifecycle_notice(
            runtime,
            notice_kind="subagent_resumed_after_bash_update",
            text="resumed to handle a detached bash update.",
            session_id=session_id,
        )
        self._launch_runtime_task(
            runtime,
            user_text=None,
            force_session_id=session_id,
            pre_turn_messages=(),
            runtime_turn=True,
            name=f"jarvis-subagent-bash-followup-{runtime.codename}-{runtime.subagent_id}",
        )
        return True

    def _append_bash_job_system_message(
        self,
        runtime: SubagentRuntime,
        *,
        session_id: str,
        message: AgentRuntimeMessage,
    ) -> bool:
        return runtime.loop.append_system_note(
            message.content,
            session_id=session_id,
            metadata=message.metadata,
        )

    def _build_bash_job_followup_message(
        self,
        notices: tuple[BashJobNotice, ...],
        *,
        recommendation: str,
    ) -> AgentRuntimeMessage:
        running_notices = [notice for notice in notices if notice.status == "running"]
        terminal_notices = [notice for notice in notices if notice.status != "running"]
        lines = ["Detached bash update."]
        for notice in notices:
            lines.append(f"- {self._format_bash_job_notice_line(notice)}")
            if notice.status != "running":
                lines.extend(_format_terminal_bash_evidence(notice))
            if notice.skill_import_notice:
                lines.extend(notice.skill_import_notice.splitlines())
        lines.append(f"recommendation={recommendation}")
        lines.append(
            "This is a system update from the orchestrator, not a new user message or a "
            "new instruction from Jarvis."
        )
        if recommendation == "wait":
            lines.append(
                "Detached bash is orchestrator-monitored. Do not call tools for this update. "
                "Continue the assignment and wait for the next orchestrator update unless Jarvis "
                "explicitly asks for immediate inspection."
            )
            lines.append("Do not declare the assignment complete while any listed job is still running.")
        elif recommendation == "inspect":
            lines.append(
                "One of the detached bash updates may need inspection. Check the listed job only "
                "if the issue blocks the assignment; otherwise continue without polling."
            )
        elif terminal_notices:
            lines.append(
                "Incorporate the finished detached bash result into the assignment and continue "
                "or finish as appropriate."
            )
        return AgentRuntimeMessage(
            role="system",
            metadata={
                "bash_job_progress_update": True,
                "subagent_bash_job_progress_update": True,
                "notice_kind": "bash_job_progress_update",
                "recommended_action": recommendation,
                "detached_bash_job_ids": [notice.job_id for notice in notices],
                "bash_job_notice_kinds": [notice.notice_kind for notice in notices],
                "bash_job_running_ids": [notice.job_id for notice in running_notices],
                "bash_job_terminal_ids": [notice.job_id for notice in terminal_notices],
                "bash_job_progress_fingerprints": [
                    _bash_job_progress_fingerprint(notice) for notice in notices
                ],
            },
            content="\n".join(lines),
        )

    def _recommend_bash_followup_action(
        self,
        notices: tuple[BashJobNotice, ...],
    ) -> str:
        if any(notice.notice_kind == "bash_job_needs_attention" for notice in notices):
            return "inspect"
        if any(
            notice.notice_kind in {"bash_job_failed", "bash_job_cancelled"}
            for notice in notices
        ):
            return "inspect"
        if any(notice.status == "running" for notice in notices):
            return "wait"
        return "finalize"

    def _recommend_main_supervision_action(
        self,
        *,
        notice_kind: str,
        snapshot: SubagentSnapshot,
    ) -> str:
        if notice_kind in {"subagent_needs_attention", "subagent_failed"}:
            return "inspect"
        if notice_kind in {"subagent_approval_rejected", "subagent_paused"}:
            return "inspect"
        if notice_kind == "subagent_completed":
            return "inspect"
        if snapshot.status in {"paused", "failed"}:
            return "inspect"
        if snapshot.status == "completed":
            return "inspect"
        if snapshot.status == "disposed":
            return "finalize"
        if snapshot.status in {"running", "waiting_background", "awaiting_approval"}:
            return "wait"
        return "finalize"

    def _format_bash_job_notice_line(self, notice: BashJobNotice) -> str:
        notice_name = notice.notice_kind.removeprefix("bash_job_") or notice.notice_kind
        timestamp_label, timestamp_value = self._bash_job_notice_timestamp(notice)
        parts = [
            f"job_id={notice.job_id}",
            f"status={notice.status}",
            f"notice={notice_name}",
            f"{timestamp_label}={timestamp_value}",
        ]
        if notice.status != "cancelled" and notice.exit_code is not None:
            parts.append(f"exit_code={notice.exit_code}")
        detail = self._bash_job_notice_detail(notice)
        if detail is not None:
            detail_label = "progress" if notice.status == "running" else "result"
            parts.append(f'{detail_label}="{detail}"')
        return " ".join(parts)

    def _bash_job_notice_timestamp(self, notice: BashJobNotice) -> tuple[str, str]:
        if notice.status == "cancelled":
            return "cancelled_at", notice.cancelled_at or notice.last_update_at or notice.started_at
        if notice.status != "running":
            return "finished_at", notice.finished_at or notice.last_update_at or notice.started_at
        if notice.last_update_at is not None:
            return "last_update_at", notice.last_update_at
        return "started_at", notice.started_at

    def _bash_job_notice_detail(self, notice: BashJobNotice) -> str | None:
        if notice.progress_hint:
            return self._truncate_for_notice(notice.progress_hint, max_length=120)
        if notice.status == "running":
            return (
                f"stdout={self._format_notice_bytes(notice.stdout_bytes_seen)} "
                f"stderr={self._format_notice_bytes(notice.stderr_bytes_seen)}"
            )
        tail_hint = self._truncate_for_notice(
            self._last_non_empty_line(notice.stderr) or self._last_non_empty_line(notice.stdout),
            max_length=120,
        )
        if tail_hint is not None:
            return tail_hint
        if notice.status == "cancelled":
            return None
        return (
            f"stdout={self._format_notice_bytes(notice.stdout_bytes_seen)} "
            f"stderr={self._format_notice_bytes(notice.stderr_bytes_seen)}"
        )

    def _truncate_for_notice(self, value: str | None, *, max_length: int) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            return None
        if len(normalized) <= max_length:
            return normalized
        return normalized[: max_length - 3] + "..."

    def _render_subagent_report_for_main(
        self,
        value: str | None,
        *,
        max_length: int = 8_000,
    ) -> tuple[str | None, bool]:
        if value is None:
            return None, False
        normalized = value.strip()
        if len(normalized) <= max_length:
            return normalized, False
        return normalized[: max_length - 15].rstrip() + "\n...[truncated]", True

    def _last_non_empty_line(self, text: str) -> str | None:
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    def _format_notice_bytes(self, count: int) -> str:
        if count < 1024:
            return f"{count}B"
        kib = count / 1024
        if kib < 1024:
            return f"{kib:.1f}KiB"
        mib = kib / 1024
        return f"{mib:.1f}MiB"

    def _record_bash_notice_delivery(self, notices: tuple[BashJobNotice, ...]) -> None:
        workspace_dir = self._core_settings.workspace_dir
        for notice in notices:
            try:
                mark_job_progress_notified(
                    workspace_dir=workspace_dir,
                    job_id=notice.job_id,
                    notice_kind=notice.notice_kind,
                    status=notice.status,
                    stdout_bytes_seen=notice.stdout_bytes_seen,
                    stderr_bytes_seen=notice.stderr_bytes_seen,
                    last_update_at=notice.last_update_at,
                )
                if notice.status in {"finished", "cancelled"}:
                    mark_job_terminal_notice_dispatched(
                        workspace_dir=workspace_dir,
                        job_id=notice.job_id,
                        notice_kind=notice.notice_kind,
                    )
            except BashJobError:
                continue

    async def _dispose_runtime(
        self,
        runtime: SubagentRuntime,
        *,
        public_notice: bool,
    ) -> dict[str, Any]:
        if runtime.status == "disposed":
            return {
                "subagent_id": runtime.subagent_id,
                "codename": runtime.codename,
                "status": runtime.status,
                "changed": False,
            }
        await self._finalize_workspace_snapshot(runtime)
        runtime.status = "disposed"
        runtime.pause_reason = None
        runtime.pending_pause_reason = None
        runtime.pending_background_job_ids.clear()
        disposed_at = _utc_now_iso()
        active_session_id = runtime.loop.active_session_id()
        if active_session_id is not None:
            try:
                runtime.storage.archive_session(active_session_id)
            except ValueError:
                LOGGER.debug(
                    "Subagent %s session %s was already absent during dispose.",
                    runtime.subagent_id,
                    active_session_id,
                )
        await runtime.loop.aclose()
        if self._workspace_access is not None and runtime.workspace_lease_status == "held":
            await self._workspace_access.release_owner(
                owner=f"subagent:{runtime.subagent_id}"
            )
            runtime.workspace_lease_status = "released"
        self._sync_catalog(runtime, disposed_at=disposed_at)
        self._pending_bash_job_notices.pop(runtime.subagent_id, None)
        self._append_notable_event(runtime, kind="disposed", summary=f"Disposed {runtime.codename}.")
        if public_notice:
            await self._publish_event(
                RouteSystemNoticeEvent(
                    route_id=self._route_id,
                    agent_kind="subagent",
                    agent_name=runtime.codename,
                    subagent_id=runtime.subagent_id,
                    session_id=runtime.loop.active_session_id(),
                    origin_session_id=runtime.owner_main_session_id,
                    origin_turn_id=runtime.owner_main_turn_id,
                    actor_run_generation=runtime.run_generation,
                    notice_kind="subagent_disposed",
                    text="came offline.",
                    public=True,
                )
            )
        return {
            "subagent_id": runtime.subagent_id,
            "codename": runtime.codename,
            "status": runtime.status,
            "changed": True,
            "changed_paths": sorted(runtime.changed_paths),
            "changed_paths_complete": runtime.changed_paths_complete,
            "changed_paths_source": runtime.changed_paths_source,
            "changed_test_artifact_paths": sorted(
                runtime.changed_test_artifact_paths
            ),
        }

    async def _reacquire_workspace_lease_if_needed(
        self,
        runtime: SubagentRuntime,
    ) -> None:
        if runtime.workspace_lease_status != "released":
            return
        if self._workspace_access is None:
            runtime.workspace_lease_status = "not_applicable"
            return
        await self._workspace_access.claim_paths(
            owner=f"subagent:{runtime.subagent_id}",
            paths=runtime.owned_paths,
        )
        runtime.workspace_lease_status = "held"
        await self._begin_workspace_snapshot(runtime)
        session_id = runtime.loop.active_session_id()
        if session_id is not None:
            runtime.loop.append_system_note(
                (
                    "Workspace ownership was reacquired for this continuation. You may write "
                    "your assigned paths; writes to another actor's paths remain blocked."
                ),
                session_id=session_id,
                metadata={
                    "workspace_lease": True,
                    "workspace_lease_reacquired": True,
                    "owned_paths": list(runtime.owned_paths),
                },
            )
        self._sync_catalog(runtime)

    def _append_notable_event(self, runtime: SubagentRuntime, *, kind: str, summary: str) -> None:
        runtime.updated_at = _utc_now_iso()
        runtime.last_activity_at = runtime.updated_at
        runtime.notable_events.append(
            SubagentEventNote(
                created_at=runtime.updated_at,
                kind=kind,
                summary=summary,
            )
        )
        while len(runtime.notable_events) > max(self._settings.main_context_event_limit * 2, 12):
            runtime.notable_events.popleft()

    def _capture_assistant_checkpoint(self, runtime: SubagentRuntime, text: str) -> None:
        normalized = text.strip()
        if not normalized or normalized == runtime.latest_report:
            return
        runtime.latest_report = normalized
        runtime.report_complete = False
        summary = self._truncate_for_notice(normalized, max_length=240)
        if summary is not None:
            self._append_notable_event(
                runtime,
                kind="assistant_checkpoint",
                summary=summary,
            )

    def _summarize_tool_result(self, result: ToolExecutionResult) -> str:
        outcome = "succeeded" if result.ok else "failed"
        details: list[str] = []
        if result.name == "file_patch":
            path = str(result.metadata.get("path", "")).strip()
            if path:
                details.append(f"path={path}")
            if "changed" in result.metadata:
                details.append(f"changed={str(bool(result.metadata['changed'])).lower()}")
        elif result.name == "bash":
            status = str(result.metadata.get("status") or result.metadata.get("state") or "").strip()
            job_id = str(result.metadata.get("job_id", "")).strip()
            if status:
                details.append(f"status={status}")
            if job_id:
                details.append(f"job_id={job_id[:8]}")
        elif result.name == "get_skills":
            skill_id = str(result.metadata.get("skill_id", "")).strip()
            mode = str(result.metadata.get("mode", "")).strip()
            if mode:
                details.append(f"mode={mode}")
            if skill_id:
                details.append(f"skill_id={skill_id}")
        suffix = " " + " ".join(details) if details else ""
        return f"{result.name} {outcome}.{suffix}".rstrip()

    def _latest_assistant_report(self, runtime: SubagentRuntime) -> str | None:
        session_id = runtime.loop.active_session_id()
        if session_id is None:
            return None
        for record in reversed(runtime.storage.load_records(session_id)):
            if record.kind != "message" or record.role != "assistant":
                continue
            content = record.content.strip()
            if content:
                return content
        return None

    def _load_skill_documents(
        self,
        skill_ids: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        if not skill_ids:
            return ()
        settings = SkillsSettings.from_workspace_dir(self._core_settings.workspace_dir)
        documents: list[tuple[str, str]] = []
        for skill_id in skill_ids:
            skill = get_skill(settings, skill_id)
            if skill is None:
                raise ValueError(f"Unknown selected skill id: {skill_id}")
            documents.append((skill_id, read_skill_markdown(settings, skill)))
        return tuple(documents)

    def _record_subagent_error(self, runtime: SubagentRuntime, exc: Exception) -> str:
        error_log_path = runtime.storage.root_dir / "errors.jsonl"
        entry = {
            "created_at": _utc_now_iso(),
            "subagent_id": runtime.subagent_id,
            "codename": runtime.codename,
            "task_label": runtime.task_label,
            "exception_type": type(exc).__name__,
            "exception_module": type(exc).__module__,
            "exception_message": str(exc),
            "exception_metadata": _exception_metadata(exc),
            "traceback": traceback.format_exc(),
        }
        with error_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str))
            handle.write("\n")
        return str(error_log_path)

    async def _publish_lifecycle_notice(
        self,
        runtime: SubagentRuntime,
        *,
        notice_kind: str,
        text: str,
        session_id: str | None = None,
    ) -> None:
        await self._publish_event(
            RouteSystemNoticeEvent(
                route_id=self._route_id,
                agent_kind="subagent",
                agent_name=runtime.codename,
                subagent_id=runtime.subagent_id,
                session_id=session_id or runtime.loop.active_session_id(),
                origin_session_id=runtime.owner_main_session_id,
                origin_turn_id=runtime.owner_main_turn_id,
                actor_run_generation=runtime.run_generation,
                notice_kind=notice_kind,
                text=text,
                public=_subagent_notice_is_public(notice_kind),
            )
        )

    def _sync_catalog(self, runtime: SubagentRuntime, *, disposed_at: str | None = None) -> None:
        self._catalog.update_entry(
            runtime.subagent_id,
            status=runtime.status,
            current_subagent_session_id=runtime.loop.active_session_id(),
            pause_reason=runtime.pause_reason,
            last_error=runtime.last_error,
            last_error_metadata=runtime.last_error_metadata,
            error_log_path=runtime.error_log_path,
            run_generation=runtime.run_generation,
            phase=runtime.phase,
            depends_on=runtime.depends_on,
            seam_contract=runtime.seam_contract,
            changed_paths=tuple(sorted(runtime.changed_paths)),
            changed_paths_complete=runtime.changed_paths_complete,
            changed_paths_source=runtime.changed_paths_source,
            changed_test_artifact_paths=tuple(
                sorted(runtime.changed_test_artifact_paths)
            ),
            workspace_lease_status=runtime.workspace_lease_status,
            disposed_at=disposed_at,
        )

    def _serialize_snapshot(self, runtime: SubagentRuntime, *, detail: str) -> dict[str, Any]:
        if runtime.latest_report is None:
            runtime.latest_report = self._latest_assistant_report(runtime)
        snapshot = runtime.snapshot()
        payload: dict[str, Any] = {
            "subagent_id": snapshot.subagent_id,
            "codename": snapshot.codename,
            "task_label": snapshot.task_label,
            "status": snapshot.status,
            "owner_main_session_id": snapshot.owner_main_session_id,
            "owner_main_turn_id": snapshot.owner_main_turn_id,
            "current_subagent_session_id": snapshot.current_subagent_session_id,
            "pause_reason": snapshot.pause_reason,
            "last_error": snapshot.last_error,
            "last_error_metadata": snapshot.last_error_metadata,
            "error_log_path": snapshot.error_log_path,
            "last_tool_name": snapshot.last_tool_name,
            "last_activity_at": snapshot.last_activity_at,
            "report_complete": snapshot.report_complete,
            "pending_background_job_count": snapshot.pending_background_job_count,
            "pending_background_job_ids": list(snapshot.pending_background_job_ids),
            "run_generation": snapshot.run_generation,
            "skill_selection_reason": snapshot.skill_selection_reason,
            "phase": snapshot.phase,
            "depends_on": list(snapshot.depends_on),
            "workspace_lease_status": snapshot.workspace_lease_status,
            "changed_paths": list(snapshot.changed_paths),
            "changed_paths_complete": snapshot.changed_paths_complete,
            "changed_paths_source": snapshot.changed_paths_source,
            "changed_test_artifact_paths": list(
                snapshot.changed_test_artifact_paths
            ),
        }
        if detail == "full":
            payload.update(
                {
                    "instructions": snapshot.instructions,
                    "user_constraints": snapshot.user_constraints,
                    "shared_context": snapshot.shared_context,
                    "owned_paths": list(snapshot.owned_paths),
                    "skill_ids": list(snapshot.skill_ids),
                    "skill_selection_reason": snapshot.skill_selection_reason,
                    "phase": snapshot.phase,
                    "depends_on": list(snapshot.depends_on),
                    "seam_contract": snapshot.seam_contract,
                    "workspace_lease_status": snapshot.workspace_lease_status,
                    "changed_paths": list(snapshot.changed_paths),
                    "changed_paths_complete": snapshot.changed_paths_complete,
                    "changed_paths_source": snapshot.changed_paths_source,
                    "deliverable": snapshot.deliverable,
                    "latest_report": snapshot.latest_report,
                    "transcript_path": self._transcript_path(runtime),
                }
            )
            payload["notable_events"] = [
                note.to_dict() for note in snapshot.notable_events
            ]
        return payload

    def _transcript_path(self, runtime: SubagentRuntime) -> str | None:
        session_id = runtime.loop.active_session_id()
        if session_id is None:
            return None
        return str(runtime.storage.root_dir / "sessions" / f"{session_id}.jsonl")

    def _deduplicate_main_context_message(
        self,
        *,
        session_id: str,
        message: AgentRuntimeMessage,
    ) -> tuple[AgentRuntimeMessage, ...]:
        signature = json.dumps(
            {"metadata": message.metadata, "content": message.content},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        cache_key = (session_id, signature)
        if cache_key == self._last_main_context_signature:
            return ()
        self._last_main_context_signature = cache_key
        return (message,)

    async def _wait_for_turn_settle(self, runtime: SubagentRuntime) -> None:
        task = runtime.task
        if task is None:
            return
        await task
        if runtime.task is task:
            runtime.task = None

    def _request_runtime_stop(
        self,
        runtime: SubagentRuntime,
        *,
        pause_reason: SubagentPauseReason,
        hard: bool = False,
        interruption_reason: InterruptionReason | None = None,
    ) -> bool:
        if runtime.status in {"paused", "completed", "waiting_background", "failed", "disposed"}:
            return False
        if runtime.pending_pause_reason == pause_reason:
            return False
        resolved_interruption_reason = interruption_reason
        if resolved_interruption_reason is None:
            if pause_reason == "new_session":
                resolved_interruption_reason = "new_session"
            elif pause_reason == "process_shutdown":
                resolved_interruption_reason = "process_shutdown"
            else:
                resolved_interruption_reason = "user_stop"
        if pause_reason == "new_session" or hard:
            stop_requested = runtime.loop.request_hard_stop(reason=resolved_interruption_reason)
        else:
            stop_requested = runtime.loop.request_stop(reason=resolved_interruption_reason)
        if not stop_requested:
            return False
        runtime.pending_pause_reason = pause_reason
        return True

    def _require_runtime(self, agent: str) -> SubagentRuntime:
        normalized = agent.strip()
        if not normalized:
            raise ValueError("Subagent reference cannot be empty.")
        if normalized in self._subagents:
            return self._subagents[normalized]
        lowered = normalized.lower()
        matches = [
            runtime
            for runtime in self._subagents.values()
            if runtime.codename.lower() == lowered
        ]
        active_matches = [runtime for runtime in matches if runtime.status != "disposed"]
        if len(active_matches) == 1:
            return active_matches[0]
        if len(active_matches) > 1:
            raise ValueError(f"Ambiguous active subagent codename: {agent}")
        if matches:
            return matches[-1]
        raise ValueError(f"Unknown subagent: {agent}")

    def _non_disposed_runtimes(self) -> list[SubagentRuntime]:
        return [
            runtime
            for runtime in self._subagents.values()
            if runtime.status != "disposed"
        ]

    @staticmethod
    def _ensure_main_requester(requester_kind: AgentKind) -> None:
        if requester_kind != "main":
            raise ValueError("Only the main agent may invoke subagents.")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _exception_metadata(exc: Exception) -> dict[str, Any]:
    if not isinstance(exc, LLMError):
        return {}
    return json.loads(json.dumps(exc.metadata, ensure_ascii=False, default=str))


def _subagent_notice_is_public(notice_kind: str) -> bool:
    return notice_kind in {"subagent_invoked", "subagent_disposed"}


def _bash_job_progress_fingerprint(notice: BashJobNotice) -> str:
    return ":".join(
        (
            notice.job_id,
            notice.notice_kind,
            notice.status,
            str(notice.stdout_bytes_seen),
            str(notice.stderr_bytes_seen),
            str(notice.stdout_bytes_dropped),
            str(notice.stderr_bytes_dropped),
            "" if notice.exit_code is None else str(notice.exit_code),
        )
    )


def _format_terminal_bash_evidence(notice: BashJobNotice) -> list[str]:
    return [
        f"  process_exited=true process_exit_success={str(notice.process_exit_success).lower()}",
        f"  termination_signal={notice.termination_signal}",
        f"  runtime_seconds={notice.runtime_seconds}",
        f"  command_sha256={notice.command_sha256}",
        f"  workspace_revision={notice.workspace_revision or 'unavailable'}",
        "  verification_passed=unknown; process exit never implies semantic success.",
        f"  stdout_log_path={notice.stdout_log_path}",
        f"  stderr_log_path={notice.stderr_log_path}",
        f"  stdout_tail_sha256={notice.stdout_sha256}",
        f"  stderr_tail_sha256={notice.stderr_sha256}",
        "  stdout_tail:",
        _bounded_terminal_tail(notice.stdout),
        "  stderr_tail:",
        _bounded_terminal_tail(notice.stderr),
    ]


def _bounded_terminal_tail(value: str, *, limit: int = 2_000) -> str:
    if not value.strip():
        return "  (empty)"
    if len(value) <= limit:
        return "  " + value
    head = limit // 2
    tail = limit - head
    return "  " + value[:head] + "\n  ...[terminal tail truncated]...\n  " + value[-tail:]


def _completion_block_pause_reason(reason: str) -> SubagentPauseReason:
    if reason == "tool_liveness_exhausted":
        return "tool_liveness_exhausted"
    if reason == "provider_recovery_exhausted":
        return "provider_recovery_exhausted"
    return "external_blocked"


def _workspace_lease_error_result(
    tool_call: object,
    error: WorkspaceLeaseError,
) -> ToolExecutionResult:
    call_id = str(getattr(tool_call, "call_id", ""))
    name = str(getattr(tool_call, "name", "bash"))
    arguments = getattr(tool_call, "arguments", {})
    return ToolExecutionResult(
        call_id=call_id,
        name=name,
        ok=False,
        content=(
            "Tool execution denied\n"
            f"tool: {name}\n"
            "error_code: workspace_lease_conflict\n"
            f"conflict_class: {error.conflict_class}\n"
            f"conflict_key: {error.conflict_key}\n"
            f"reason: {error}\n"
            f"remediation: {error.remediation}"
        ),
        metadata={
            "execution_failed": True,
            "error_code": "workspace_lease_conflict",
            "conflict_class": error.conflict_class,
            "conflict_key": error.conflict_key,
            "reason": str(error),
            "remediation": error.remediation,
            "arguments": dict(arguments) if isinstance(arguments, dict) else {},
        },
    )


def _with_workspace_lease_generation(
    result: ToolExecutionResult,
    generation: int,
) -> ToolExecutionResult:
    metadata = dict(result.metadata)
    metadata["workspace_lease_generation"] = generation
    return ToolExecutionResult(
        call_id=result.call_id,
        name=result.name,
        ok=result.ok,
        content=result.content,
        metadata=metadata,
    )
