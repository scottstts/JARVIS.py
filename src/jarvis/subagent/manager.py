"""Lifecycle manager for route-scoped subagents."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
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
)
from jarvis.gateway.bash_job_supervisor import BashJobNotice
from jarvis.gateway.route_events import (
    RouteApprovalRequestEvent,
    RouteSystemNoticeEvent,
    RouteToolCallEvent,
)
from jarvis.llm import LLMService
from jarvis.logging_setup import get_application_logger
from jarvis.storage import SessionStorage
from jarvis.storage.layout import transcript_archive_root_from_runtime_path
from jarvis.tools import ToolExecutionContext, ToolExecutionResult, ToolRegistry, ToolRuntime
from jarvis.tools.basic.bash.jobs import (
    BashJobError,
    cancel_job,
    job_status,
    list_jobs,
    mark_job_progress_notified,
    mark_job_terminal_notice_dispatched,
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

    async def invoke(
        self,
        *,
        requester_kind: AgentKind,
        instructions: str,
        owner_main_session_id: str,
        owner_main_turn_id: str,
        context: str | None = None,
        deliverable: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_main_requester(requester_kind)
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
                instructions=instructions,
                context=context,
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
            notable_events=deque(),
        )
        session_id = await runtime.loop.prepare_session(start_reason="subagent_initial")
        self._subagents[subagent_id] = runtime
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
                current_subagent_session_id=session_id,
            )
        )
        self._append_notable_event(runtime, kind="spawned", summary=f"Spawned {codename}.")
        self._launch_runtime_task(
            runtime,
            user_text=build_subagent_kickoff_text(),
            force_session_id=session_id,
            pre_turn_messages=(),
            name=f"jarvis-subagent-{codename}-{subagent_id}",
        )
        await self._publish_event(
            RouteSystemNoticeEvent(
                route_id=self._route_id,
                agent_kind="subagent",
                agent_name=codename,
                subagent_id=subagent_id,
                session_id=session_id,
                notice_kind="subagent_invoked",
                text="came online.",
                public=True,
            )
        )
        return {
            "subagent_id": subagent_id,
            "codename": codename,
            "status": runtime.status,
            "session_id": session_id,
            "active_count": len(self._non_disposed_runtimes()),
        }

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
                self._serialize_snapshot(runtime.snapshot(), detail=detail)
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
                        "status": snapshot["status"],
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
        affected: list[SubagentSnapshot] = []
        for runtime in self._non_disposed_runtimes():
            if self._request_runtime_stop(runtime, pause_reason="main_stop"):
                affected.append(runtime.snapshot())
        return tuple(affected)

    def request_stop_all_for_superseded_user_message(self) -> tuple[SubagentSnapshot, ...]:
        affected: list[SubagentSnapshot] = []
        for runtime in self._non_disposed_runtimes():
            if self._request_runtime_stop(
                runtime,
                pause_reason="superseded_by_user_message",
            ):
                affected.append(runtime.snapshot())
        return tuple(affected)

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
        runtime.pause_reason = None
        runtime.status = "running"
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

    async def dispose(self, *, agent: str) -> dict[str, Any]:
        runtime = self._require_runtime(agent)
        if runtime.status in {"running", "awaiting_approval", "waiting_background"} or (
            runtime.task is not None and not runtime.task.done()
        ):
            raise ValueError("Cannot dispose a running subagent. Stop it first.")
        return await self._dispose_runtime(runtime, public_notice=True)

    async def reset_for_new_session(self) -> dict[str, Any]:
        self.request_stop_all_for_user_stop()
        for runtime in tuple(self._non_disposed_runtimes()):
            await self._wait_for_turn_settle(runtime)

        disposed_subagent_ids: list[str] = []
        cancelled_job_ids: list[str] = []
        live_subagent_ids = set(self._subagents)

        for runtime in tuple(self._subagents.values()):
            if runtime.status == "disposed":
                continue
            cancelled_job_ids.extend(self._cancel_owned_bash_jobs(runtime.subagent_id))
            runtime.pending_background_job_ids.clear()
            self._pending_bash_job_notices.pop(runtime.subagent_id, None)
            payload = await self._dispose_runtime(runtime, public_notice=False)
            if payload["changed"]:
                disposed_subagent_ids.append(runtime.subagent_id)

        for entry in self._catalog.list_entries():
            if entry.status == "disposed" or entry.subagent_id in live_subagent_ids:
                continue
            cancelled_job_ids.extend(self._cancel_owned_bash_jobs(entry.subagent_id))
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
        return {
            "disposed_subagent_ids": disposed_subagent_ids,
            "cancelled_job_ids": cancelled_job_ids,
            "disposed_count": len(disposed_subagent_ids),
            "cancelled_job_count": len(cancelled_job_ids),
        }

    def main_turn_runtime_messages(self) -> tuple[AgentRuntimeMessage, ...]:
        runtimes = self._non_disposed_runtimes()
        if not runtimes:
            return (
                AgentRuntimeMessage(
                    role="system",
                    metadata={
                        "subagent_status_snapshot": True,
                        "pending_subagent_ids": [],
                    },
                    content="Subagent status snapshot:\n- no non-disposed subagents.",
                ),
            )

        lines = ["Subagent status snapshot:"]
        recent_events: list[tuple[str, str, str]] = []
        pending_subagent_ids: list[str] = []
        for runtime in runtimes:
            snapshot = runtime.snapshot()
            status_line = f"- {snapshot.codename} ({snapshot.subagent_id}): {snapshot.status}"
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

        return (
            AgentRuntimeMessage(
                role="system",
                metadata={
                    "subagent_status_snapshot": True,
                    "pending_subagent_ids": pending_subagent_ids,
                },
                content="\n".join(lines),
            ),
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
            f"id={snapshot.subagent_id}",
            f"status={snapshot.status}",
            f"notice={notice_kind}",
        ]
        if snapshot.pending_background_job_ids:
            parts.append(
                "pending_background_job_ids="
                + ",".join(snapshot.pending_background_job_ids)
            )
        if notice_text.strip():
            parts.append(f'note="{self._truncate_for_notice(notice_text, max_length=140)}"')
        latest_report = self._latest_assistant_report(runtime) if runtime is not None else None
        content = "\n".join(
            self._build_main_progress_lines(
                parts=parts,
                recommendation=recommendation,
                latest_report=latest_report,
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
                    "recommended_action": recommendation,
                    "latest_subagent_report_included": bool(latest_report),
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
    ) -> list[str]:
        lines = [
            "Subagent update.",
            "- " + " ".join(parts),
            f"recommendation={recommendation}",
        ]
        if recommendation in {"finalize", "inspect"} and latest_report is not None:
            lines.extend(
                [
                    "Latest subagent report:",
                    self._truncate_subagent_report_for_main(latest_report),
                    (
                        "The latest subagent report is already included above. Use it directly "
                        "instead of calling `subagent_monitor` or `subagent_step_in` unless the "
                        "report is clearly incomplete or contradictory."
                    ),
                ]
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
            f"- subagent_id: {snapshot.subagent_id}",
            f"- status: {snapshot.status}",
            f"- notice_kind: {notice_kind}",
            f"- notice_text: {notice_text}",
        ]
        if snapshot.current_subagent_session_id is not None:
            lines.append(f"- current_subagent_session_id: {snapshot.current_subagent_session_id}")
        if snapshot.pause_reason is not None:
            lines.append(f"- pause_reason: {snapshot.pause_reason}")
        if snapshot.last_error is not None:
            lines.append(f"- last_error: {snapshot.last_error}")
        if snapshot.last_tool_name is not None:
            lines.append(f"- last_tool_name: {snapshot.last_tool_name}")
        if snapshot.last_activity_at is not None:
            lines.append(f"- last_activity_at: {snapshot.last_activity_at}")
        if snapshot.notable_events:
            lines.append("- recent_noteworthy_events:")
            for note in snapshot.notable_events[-self._settings.main_context_event_limit :]:
                lines.append(f"  - {note.created_at} [{note.kind}] {note.summary}")

        final_report = self._latest_assistant_report(runtime)
        if final_report is not None:
            lines.extend(
                [
                    "",
                    "Latest subagent report:",
                    final_report,
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

    async def enqueue_bash_job_followup(self, notices: tuple[BashJobNotice, ...]) -> None:
        if not notices:
            return
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

    async def _run_turn(
        self,
        runtime: SubagentRuntime,
        *,
        user_text: str | None,
        force_session_id: str | None,
        pre_turn_messages: tuple[AgentRuntimeMessage, ...],
        runtime_turn: bool = False,
    ) -> None:
        runtime.status = "running"
        runtime.pause_reason = None
        runtime.last_error = None
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
                    if event.interrupted:
                        runtime.status = "paused"
                        runtime.pause_reason = runtime.pending_pause_reason or "main_stop"
                        self._append_notable_event(
                            runtime,
                            kind="paused",
                            summary=f"Paused ({runtime.pause_reason}).",
                        )
                        await self._publish_lifecycle_notice(
                            runtime,
                            notice_kind="subagent_paused",
                            text=f"paused ({runtime.pause_reason}).",
                            session_id=event.session_id,
                        )
                    elif event.approval_rejected:
                        runtime.status = "paused"
                        runtime.pause_reason = "approval_rejected"
                        self._append_notable_event(
                            runtime,
                            kind="approval_rejected",
                            summary="Approval was rejected and the subagent paused.",
                        )
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
                            self._append_notable_event(
                                runtime,
                                kind="completed",
                                summary="Completed the assigned turn.",
                            )
                            await self._publish_lifecycle_notice(
                                runtime,
                                notice_kind="subagent_completed",
                                text="completed.",
                                session_id=event.session_id,
                            )
                    runtime.pending_pause_reason = None
                    self._sync_catalog(runtime)
                    return
        except Exception as exc:
            runtime.status = "failed"
            runtime.last_error = f"{type(exc).__name__}: {exc}"
            runtime.pending_pause_reason = None
            self._append_notable_event(
                runtime,
                kind="failed",
                summary=runtime.last_error,
            )
            self._sync_catalog(runtime)
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
            async with self._tool_execution_guard:
                result = await tool_runtime.execute(
                    tool_call=tool_call,
                    context=context,
                )
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
        if result.name != "bash":
            return
        runtime = self._subagents.get(subagent_id)
        if runtime is None:
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
    ) -> None:
        task = asyncio.create_task(
            self._run_turn(
                runtime,
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
        if runtime.task is not None and not runtime.task.done():
            return False
        if runtime.status in {"paused", "awaiting_approval", "failed", "disposed"}:
            return False
        queue = self._pending_bash_job_notices.get(runtime.subagent_id)
        if not queue:
            return False
        notices = tuple(queue.values())
        session_id = notices[0].owner_session_id or runtime.loop.active_session_id()
        if session_id is None:
            return False
        recommendation = self._recommend_bash_followup_action(notices)
        message = self._build_bash_job_followup_message(
            notices,
            recommendation=recommendation,
        )
        if not self._append_bash_job_system_message(
            runtime,
            session_id=session_id,
            message=message,
        ):
            return False
        queue.clear()
        for notice in notices:
            if notice.status in {"finished", "cancelled"}:
                runtime.pending_background_job_ids.discard(notice.job_id)
            else:
                runtime.pending_background_job_ids.add(notice.job_id)
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
        notice_kind = (
            "subagent_needs_attention"
            if recommendation == "inspect"
            else "subagent_resumed_after_bash_update"
        )
        notice_text = (
            "needs attention after detached bash update."
            if recommendation == "inspect"
            else "resumed after detached bash update."
        )
        await self._publish_lifecycle_notice(
            runtime,
            notice_kind=notice_kind,
            text=notice_text,
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
        if notice_kind in {"subagent_completed", "subagent_approval_rejected", "subagent_paused"}:
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

    def _truncate_subagent_report_for_main(
        self,
        value: str,
        *,
        max_length: int = 1600,
    ) -> str:
        normalized = value.strip()
        if len(normalized) <= max_length:
            return normalized
        return normalized[: max_length - 15].rstrip() + "\n...[truncated]"

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
        }

    def _cancel_owned_bash_jobs(self, subagent_id: str) -> list[str]:
        cancelled_job_ids: list[str] = []
        for paths, record in list_jobs(self._core_settings.workspace_dir):
            if record.owner_route_id != self._route_id:
                continue
            if record.owner_subagent_id != subagent_id:
                continue
            try:
                status = job_status(paths, record)
                if str(status["status"]) == "running":
                    status = cancel_job(paths, record)
                if str(status["status"]) == "running":
                    raise RuntimeError(
                        f"Detached bash job {record.job_id} is still running after cancellation."
                    )
                mark_job_terminal_notice_dispatched(
                    workspace_dir=self._core_settings.workspace_dir,
                    job_id=record.job_id,
                    notice_kind=_terminal_notice_kind_for_status(
                        status=str(status["status"]),
                        exit_code=status.get("exit_code"),
                    ),
                )
                cancelled_job_ids.append(record.job_id)
            except BashJobError as exc:
                raise RuntimeError(
                    f"Failed to cancel or finalize detached bash job {record.job_id}: {exc}"
                ) from exc
        return cancelled_job_ids

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
            disposed_at=disposed_at,
        )

    def _serialize_snapshot(self, snapshot: SubagentSnapshot, *, detail: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "subagent_id": snapshot.subagent_id,
            "codename": snapshot.codename,
            "status": snapshot.status,
            "owner_main_session_id": snapshot.owner_main_session_id,
            "owner_main_turn_id": snapshot.owner_main_turn_id,
            "current_subagent_session_id": snapshot.current_subagent_session_id,
            "pause_reason": snapshot.pause_reason,
            "last_error": snapshot.last_error,
            "last_tool_name": snapshot.last_tool_name,
            "last_activity_at": snapshot.last_activity_at,
            "pending_background_job_count": snapshot.pending_background_job_count,
            "pending_background_job_ids": list(snapshot.pending_background_job_ids),
        }
        if detail == "full":
            payload["notable_events"] = [
                note.to_dict() for note in snapshot.notable_events
            ]
        return payload

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
    ) -> bool:
        if runtime.status in {"paused", "completed", "waiting_background", "failed", "disposed"}:
            return False
        interruption_reason: str
        if pause_reason == "superseded_by_user_message":
            interruption_reason = "superseded_by_user_message"
        else:
            interruption_reason = "user_stop"
        if not runtime.loop.request_stop(reason=interruption_reason):
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
        for runtime in self._subagents.values():
            if runtime.codename.lower() == lowered:
                return runtime
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


def _terminal_notice_kind_for_status(*, status: str, exit_code: object) -> str:
    if status == "cancelled":
        return "bash_job_cancelled"
    if status == "finished":
        return "bash_job_completed" if int(exit_code or 0) == 0 else "bash_job_failed"
    return "bash_job_cancelled"


def _subagent_notice_is_public(notice_kind: str) -> bool:
    return notice_kind in {"subagent_invoked", "subagent_disposed"}
