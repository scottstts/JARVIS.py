"""Route-scoped supervisor runtime for the main loop and its subagents."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from jarvis.actor_backends import ActorRuntime, backend_kind_for_provider
from jarvis.codex_backend import (
    CodexActorRuntime,
    CodexBackendError,
    CodexBackendSettings,
    CodexRouteCoordinator,
)
from jarvis.core import (
    AgentApprovalRequestEvent,
    AgentAssistantMessageEvent,
    AgentIdentity,
    AgentLoop,
    AgentMemoryMode,
    AgentRuntimeMessage,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentTurnStartedEvent,
    AgentTurnDoneEvent,
    AgentTurnResult,
    AgentTurnStreamEvent,
    ContextBudgetError,
    CoreSettings,
)
from jarvis.core.commands import parse_user_command
from jarvis.core.identities import IdentityBootstrapLoader
from jarvis.llm import (
    LLMMessage,
    LLMService,
    ProviderRateLimitError,
    ProviderTemporaryError,
    ProviderTimeoutError,
    ToolCall,
    ToolDefinition,
)
from jarvis.logging_setup import get_application_logger
from jarvis.runtime_errors import record_runtime_error
from jarvis.storage import SessionStorage
from jarvis.subagent import (
    SUBAGENT_PRIMITIVE_NAMES,
    SubagentManager,
    build_subagent_primitive_definitions,
    render_subagent_primitive_docs,
)
from jarvis.subagent.types import SubagentSnapshot
from jarvis.tools import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolRegistry,
    ToolRuntime,
    ToolSettings,
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

from .bash_job_supervisor import BashJobNotice, BashJobResetResult, BashJobSupervisor
from .route_events import (
    RouteApprovalRequestEvent,
    RouteAssistantDeltaEvent,
    RouteAssistantMessageEvent,
    RouteErrorEvent,
    RouteEvent,
    RouteLocalNoticeEvent,
    RouteSystemNoticeEvent,
    RouteTaskStatusEvent,
    RouteToolCallEvent,
    RouteTurnStartedEvent,
    RouteTurnDoneEvent,
)

_INTERNAL_ERROR_MESSAGE = "Internal error while processing message."
_PROVIDER_TIMEOUT_MESSAGE = "The model timed out while processing that message."
_SUBAGENT_MAIN_PROGRESS_NOTICE_KINDS = frozenset(
    {
        "subagent_completed",
        "subagent_failed",
        "subagent_approval_rejected",
        "subagent_paused",
        "subagent_waiting_background",
        "subagent_needs_attention",
    }
)
_SUBAGENT_USER_STOP_NOTE_HEADER = (
    "The user issued /stop. Route-owned subagents were hard-paused."
)
_MAIN_BASH_PROGRESS_RUNTIME_KIND = "main_bash_progress"
_MAIN_BASH_PROGRESS_NOTICE_KIND = "bash_job_progress_update"
_MAIN_SUBAGENT_PROGRESS_RUNTIME_KIND = "main_subagent_progress"
_MAIN_SUBAGENT_PROGRESS_NOTICE_KIND = "subagent_progress_update"
_MAIN_PROVIDER_RECOVERY_RUNTIME_KIND = "main_provider_recovery"
_MAIN_ORCHESTRATOR_REVIEW_RUNTIME_KIND = "main_orchestrator_review"
_MAX_MAIN_PROVIDER_RECOVERY_ATTEMPTS = 3
_ORCHESTRATOR_WAIT_MIN_SECONDS = 30
_ORCHESTRATOR_WAIT_MAX_SECONDS = 30 * 60
_ORCHESTRATOR_WAIT_BACKOFF_BASE_SECONDS = 60
_WAIT_ONLY_BASH_PATTERN = re.compile(
    r"^\s*sleep\s+\d+(?:\.\d+)?\s*(?:&&|;)\s*(?:echo|printf)\b[^\n]*(?:tick|wait|poll)",
    re.IGNORECASE,
)
LOGGER = get_application_logger(__name__)


@dataclass(slots=True, frozen=True)
class _RouteTurnRequest:
    user_text: str | None = None
    force_session_id: str | None = None
    pre_turn_messages: tuple[AgentRuntimeMessage, ...] = ()
    parse_commands: bool = True
    user_initiated: bool = True
    client_message_id: str | None = None
    internal_generation: int | None = None
    runtime_turn_kind: str | None = None
    provider_recovery_attempt: int = 0
    provider_recovery_task_id: str | None = None
    orchestrator_wait_generation: int | None = None


class RouteEventBus:
    """Simple in-memory pub/sub for route-scoped outbound events."""

    def __init__(self) -> None:
        self._subscribers: dict[str, asyncio.Queue[RouteEvent]] = {}
        self._next_subscriber_id = 1

    def subscribe(self) -> tuple[str, asyncio.Queue[RouteEvent]]:
        subscriber_id = f"route-subscriber-{self._next_subscriber_id}"
        self._next_subscriber_id += 1
        queue: asyncio.Queue[RouteEvent] = asyncio.Queue()
        self._subscribers[subscriber_id] = queue
        return subscriber_id, queue

    def unsubscribe(self, subscriber_id: str) -> None:
        self._subscribers.pop(subscriber_id, None)

    async def publish(self, event: RouteEvent) -> None:
        for queue in tuple(self._subscribers.values()):
            await queue.put(event)


class RouteApprovalRegistry:
    """Maps approval ids to the loop instance currently waiting on them."""

    def __init__(self) -> None:
        self._targets: dict[str, ActorRuntime] = {}

    def register(self, approval_id: str, loop: ActorRuntime) -> None:
        self._targets[approval_id] = loop

    def resolve(self, approval_id: str, approved: bool) -> bool:
        loop = self._targets.pop(approval_id, None)
        if loop is None:
            return False
        return loop.resolve_approval(approval_id, approved)


class CompositeMainBootstrapLoader:
    """Extends the default main-agent bootstrap with subagent primitive docs."""

    def __init__(self, settings: CoreSettings) -> None:
        self._base_loader = IdentityBootstrapLoader(settings)

    def load_bootstrap_messages(self) -> list[LLMMessage]:
        messages = self._base_loader.load_bootstrap_messages()
        messages.append(
            LLMMessage.text(
                "system",
                render_subagent_primitive_docs(),
            )
        )
        return messages


class RouteRuntime:
    """Owns one main loop, zero to seven subagents, and the route event bus."""

    def __init__(
        self,
        *,
        route_id: str,
        llm_service: LLMService,
        core_settings: CoreSettings,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._route_id = route_id
        self._llm_service = llm_service
        self._core_settings = core_settings
        tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
        self._tool_registry = tool_registry or ToolRegistry.default(tool_settings)
        self._main_storage = SessionStorage(core_settings.transcript_archive_dir)
        self._selected_skills_by_main_turn: dict[tuple[str, str], tuple[str, ...]] = {}
        self._selected_skills_by_main_session: dict[str, tuple[str, ...]] = {}
        self._tool_execution_guard = asyncio.Semaphore(1)
        self._workspace_access = WorkspaceAccessCoordinator(
            workspace_dir=core_settings.workspace_dir
        )
        self._event_bus = RouteEventBus()
        self._event_sequence = 0
        self._actor_event_sequences: dict[str, int] = {}
        self._main_backend_state_fallback: dict[str, Any] = {}
        self._provider_recovery_turn_ids: dict[str, str] = {}
        self._approval_registry = RouteApprovalRegistry()
        self._user_message_queue: asyncio.Queue[_RouteTurnRequest] = asyncio.Queue()
        self._message_queue: asyncio.Queue[_RouteTurnRequest] = asyncio.Queue()
        self._queue_wakeup = asyncio.Event()
        self._message_worker: asyncio.Task[None] | None = None
        self._active_turn_request: _RouteTurnRequest | None = None
        self._main_resume_requires_user_message = False
        self._new_session_boundary_pending = False
        self._internal_followup_generation = 0
        self._pending_main_bash_notices: dict[str, BashJobNotice] = {}
        self._main_bash_runtime_turn_queued = False
        self._pending_main_subagent_notices: dict[str, RouteSystemNoticeEvent] = {}
        self._main_subagent_runtime_turn_queued = False
        self._subagent_reset_in_progress = False
        self._published_task_active = False
        self._orchestrator_wait_task: asyncio.Task[None] | None = None
        self._orchestrator_wait_generation = 0
        self._orchestrator_unchanged_waits = 0
        self._hard_stop_lock = asyncio.Lock()
        self._main_registry = self._tool_registry.filtered_view(agent_kind="main")
        self._main_tool_runtime = ToolRuntime(registry=self._main_registry)
        self._codex_settings = CodexBackendSettings.from_env()
        self._codex_coordinator = CodexRouteCoordinator(settings=self._codex_settings)
        self._bash_job_supervisor = BashJobSupervisor(
            route_id=route_id,
            settings=tool_settings,
            followups_allowed=self._internal_followups_allowed,
            main_turn_active=self._main_loop_has_active_turn,
            subagent_turn_active=self._subagent_manager_turn_active,
            handle_main_notices=self._enqueue_main_bash_job_followup,
            handle_subagent_notices=self._enqueue_subagent_bash_job_followup,
        )
        self._subagent_manager = SubagentManager(
            route_id=route_id,
            llm_service=llm_service,
            core_settings=core_settings,
            tool_registry=self._tool_registry,
            tool_execution_guard=self._tool_execution_guard,
            workspace_access=self._workspace_access,
            publish_event=self.publish_event,
            register_approval_target=self._approval_registry.register,
            tool_result_observer=self._bash_job_supervisor.observe_tool_result,
            codex_settings=self._codex_settings,
            codex_coordinator=self._codex_coordinator,
        )
        self._main_loop: ActorRuntime = self._build_main_loop()

    def _build_main_loop(self) -> ActorRuntime:
        provider = self._resolved_main_provider()
        if backend_kind_for_provider(provider) == "codex":
            return CodexActorRuntime(
                coordinator=self._codex_coordinator,
                settings=self._codex_settings,
                llm_service=self._llm_service,
                storage=SessionStorage(self._core_settings.transcript_archive_dir),
                core_settings=self._core_settings,
                route_id=self._route_id,
                identity=AgentIdentity(kind="main", name="Jarvis"),
                bootstrap_loader=CompositeMainBootstrapLoader(self._core_settings),
                memory_mode=AgentMemoryMode(),
                tool_registry=self._main_registry,
                tool_runtime=self._main_tool_runtime,
                tool_definitions_provider=self._build_main_tool_definitions,
                tool_executor=self._execute_main_tool_call,
                publish_route_event=self.publish_event,
                runtime_messages_provider=lambda session_id: self._subagent_manager.main_turn_runtime_messages(
                    session_id=session_id
                ),
            )
        return AgentLoop(
            llm_service=self._llm_service,
            settings=self._core_settings,
            route_id=self._route_id,
            tool_registry=self._main_registry,
            tool_runtime=self._main_tool_runtime,
            bootstrap_loader=CompositeMainBootstrapLoader(self._core_settings),
            identity=AgentIdentity(kind="main", name="Jarvis"),
            llm_provider=provider,
            tool_definitions_provider=self._build_main_tool_definitions,
            tool_executor=self._execute_main_tool_call,
            runtime_messages_provider=lambda session_id: self._subagent_manager.main_turn_runtime_messages(
                session_id=session_id
            ),
            local_notice_callback=self._publish_main_local_notice,
        )

    def _resolved_main_provider(self) -> str:
        service_settings = getattr(self._llm_service, "settings", None)
        if service_settings is not None:
            default_provider = getattr(service_settings, "default_provider", None)
            if isinstance(default_provider, str) and default_provider.strip():
                return default_provider.strip().lower()
        return "openai"

    def active_session_id(self) -> str | None:
        return self._main_loop.active_session_id()

    def _write_runtime_error_log(
        self,
        *,
        request: _RouteTurnRequest,
        session_id: str | None,
        turn_id: str | None,
        published_turn_kind: str | None,
        published_client_message_id: str | None,
        parsed_command_kind: str | None,
        exc: Exception,
        error_code: str = "internal_error",
    ) -> Path:
        return record_runtime_error(
            transcript_archive_dir=self._core_settings.transcript_archive_dir,
            route_id=self._route_id,
            session_id=session_id,
            component="gateway.route_runtime",
            event="main_turn_runtime_error",
            agent_kind="main",
            exc=exc,
            error_code=error_code,
            message=(
                f"Route {self._route_id} main turn failed while processing "
                f"client_message_id={request.client_message_id}."
            ),
            context={
                "turn_id": turn_id,
                "request_turn_kind": "user" if request.user_initiated else "runtime",
                "published_turn_kind": published_turn_kind,
                "client_message_id": request.client_message_id,
                "published_client_message_id": published_client_message_id,
                "user_initiated": request.user_initiated,
                "parse_commands": request.parse_commands,
                "parsed_command_kind": parsed_command_kind,
                "runtime_turn_kind": request.runtime_turn_kind,
                "force_session_id": request.force_session_id,
            },
        )

    def _print_runtime_error_notice(self, *, error_log_path: Path) -> None:
        from rich.console import Console
        from rich.text import Text

        text = Text("Runtime error. Details written to ", style="bold white on red")
        text.append(str(error_log_path), style="bold cyan")
        Console(stderr=True).print(text, highlight=False)

    async def request_stop(self) -> bool:
        """Hard-quiesce all work owned by this route without replacing the session."""
        async with self._hard_stop_lock:
            return await self._request_stop_locked()

    async def _request_stop_locked(self) -> bool:
        LOGGER.info("Route hard stop requested (route=%s).", self._route_id)
        self._cancel_orchestrator_wait(reset_backoff=True)
        pending_bash_jobs = tuple(
            self._bash_job_supervisor.pending_jobs(include_services=True)
        )
        affected_subagents = self._subagent_manager.request_hard_stop_all_for_user_stop()
        main_stop_requested = self._main_loop.request_hard_stop(reason="user_stop")
        stop_requested = (
            main_stop_requested
            or bool(affected_subagents)
            or bool(pending_bash_jobs)
            or self._active_turn_request is not None
            or not self._user_message_queue.empty()
            or not self._message_queue.empty()
        )
        if not stop_requested:
            LOGGER.info("Route hard stop was already quiescent (route=%s).", self._route_id)
            return False

        self._main_resume_requires_user_message = True
        self._invalidate_stale_internal_followups()
        self._clear_pending_main_bash_notices()
        self._clear_pending_main_subagent_notices()
        cancelled_user_requests = self._drain_user_request_queue()
        self._drain_internal_request_queue()

        try:
            bash_reset = await self._bash_job_supervisor.terminate_route_jobs(
                reason="user_stop"
            )
            settled_subagents = await self._subagent_manager.settle_hard_user_stop(
                subagent_ids=frozenset(
                    snapshot.subagent_id for snapshot in affected_subagents
                )
            )
            await self._wait_for_main_hard_stop_settle()
        except Exception as exc:
            error_log_path = record_runtime_error(
                transcript_archive_dir=self._core_settings.transcript_archive_dir,
                route_id=self._route_id,
                session_id=self._main_loop.active_session_id(),
                component="gateway.route_runtime",
                event="route_hard_stop_failed",
                agent_kind="main",
                exc=exc,
                error_code="route_hard_stop_failed",
                message=f"Route {self._route_id} failed to hard-quiesce after /stop.",
                context={
                    "pending_bash_job_ids": [item.job_id for item in pending_bash_jobs],
                    "affected_subagent_ids": [
                        item.subagent_id for item in affected_subagents
                    ],
                },
            )
            self._print_runtime_error_notice(error_log_path=error_log_path)
            await self._publish_task_status_if_changed(reason="user_stop_failed")
            raise

        if settled_subagents or affected_subagents:
            self._append_user_stop_subagent_note(
                settled_subagents or affected_subagents
            )
        if pending_bash_jobs:
            self._append_user_stop_bash_job_note(pending_bash_jobs)
        self._append_user_stop_hard_quiesce_note(
            bash_reset=bash_reset,
            affected_subagents=settled_subagents or affected_subagents,
        )
        for cancelled_request in cancelled_user_requests:
            cancelled_turn_id = (
                cancelled_request.client_message_id or f"stopped_{uuid4().hex}"
            )
            await self.publish_event(
                RouteTurnStartedEvent(
                    route_id=self._route_id,
                    agent_kind="main",
                    agent_name="Jarvis",
                    session_id=self._main_loop.active_session_id(),
                    turn_id=cancelled_turn_id,
                    turn_kind="user",
                    client_message_id=cancelled_request.client_message_id,
                )
            )
            await self.publish_event(
                RouteTurnDoneEvent(
                    route_id=self._route_id,
                    agent_kind="main",
                    agent_name="Jarvis",
                    session_id=self._main_loop.active_session_id(),
                    turn_id=cancelled_turn_id,
                    turn_kind="user",
                    client_message_id=cancelled_request.client_message_id,
                    response_text="",
                    interrupted=True,
                    completion_blocked=True,
                    completion_block_reason="user_stop",
                    interruption_reason="user_stop",
                )
            )
        await self._publish_task_status_if_changed(reason="user_stop_quiesced")
        LOGGER.info(
            "Route hard stop completed (route=%s, subagents=%d, jobs=%d).",
            self._route_id,
            len(settled_subagents or affected_subagents),
            len(pending_bash_jobs),
        )
        return True

    def _request_user_message_supersede(self) -> None:
        self._main_loop.request_stop(
            reason="superseded_by_user_message"
        )
        self._invalidate_stale_internal_followups()

    def _request_new_session_hard_reset(self) -> None:
        self._cancel_orchestrator_wait(reset_backoff=True)
        self._new_session_boundary_pending = True
        self._selected_skills_by_main_session.clear()
        self._selected_skills_by_main_turn.clear()
        self._main_loop.request_hard_stop(reason="new_session")
        self._subagent_manager.request_hard_stop_all_for_new_session()
        self._invalidate_stale_internal_followups()

    def _drain_internal_request_queue(self) -> None:
        while True:
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._message_queue.task_done()
        self._main_bash_runtime_turn_queued = False
        self._main_subagent_runtime_turn_queued = False

    def _drain_user_request_queue(self) -> tuple[_RouteTurnRequest, ...]:
        requests: list[_RouteTurnRequest] = []
        while True:
            try:
                request = self._user_message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                requests.append(request)
                self._user_message_queue.task_done()
        return tuple(requests)

    async def _wait_for_main_hard_stop_settle(self) -> None:
        async def wait_until_idle() -> None:
            while self._active_turn_request is not None or self._main_loop.has_active_turn():
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(wait_until_idle(), timeout=30.0)
        except TimeoutError as exc:
            raise RuntimeError("Main route work did not settle after hard stop.") from exc

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        return self._approval_registry.resolve(approval_id, approved)

    async def enqueue_user_message(
        self,
        user_text: str,
        *,
        client_message_id: str | None = None,
    ) -> None:
        async with self._hard_stop_lock:
            await self._enqueue_user_message_locked(
                user_text,
                client_message_id=client_message_id,
            )

    async def _enqueue_user_message_locked(
        self,
        user_text: str,
        *,
        client_message_id: str | None,
    ) -> None:
        self._cancel_orchestrator_wait(reset_backoff=True)
        self._bash_job_supervisor.ensure_running()
        # Skill choices belong to the current user request. Keep them available for
        # later orchestration turns within that request, but do not let them leak into
        # a new request where they may no longer be relevant.
        self._selected_skills_by_main_session.clear()
        command = parse_user_command(user_text)
        if command.kind == "new":
            self._request_new_session_hard_reset()
        await self._user_message_queue.put(
            _RouteTurnRequest(
                user_text=user_text,
                parse_commands=True,
                user_initiated=True,
                client_message_id=client_message_id,
                provider_recovery_task_id=client_message_id or uuid4().hex,
            )
        )
        self._queue_wakeup.set()
        await self._publish_task_status_if_changed(reason="user_message_queued")
        if command.kind != "new":
            self._request_user_message_supersede()
        self._ensure_message_worker()

    def subscribe(self) -> tuple[str, asyncio.Queue[RouteEvent]]:
        subscriber_id, queue = self._event_bus.subscribe()
        active_session_id = self._main_loop.active_session_id()
        active_turn_id = self._main_loop.active_turn_id()
        queue.put_nowait(
            RouteTaskStatusEvent(
                route_id=self._route_id,
                agent_kind="main",
                agent_name="Jarvis",
                session_id=active_session_id,
                turn_id=active_turn_id,
                origin_session_id=active_session_id,
                origin_turn_id=active_turn_id,
                active=self._route_task_active(),
                reason="subscriber_snapshot",
                actor_id="main",
                actor_sequence=self._actor_event_sequences.get("main", 0),
                sequence=self._event_sequence,
            )
        )
        LOGGER.info(
            "Route subscriber received task-status snapshot (route=%s, active=%s).",
            self._route_id,
            self._route_task_active(),
        )
        return subscriber_id, queue

    def unsubscribe(self, subscriber_id: str) -> None:
        self._event_bus.unsubscribe(subscriber_id)

    async def publish_event(self, event: RouteEvent) -> None:
        if self._should_suppress_event_during_subagent_reset(event):
            return
        if (
            event.agent_kind == "subagent"
            and event.subagent_id is not None
            and event.actor_run_generation is not None
        ):
            snapshot = self._subagent_manager.snapshot_for(event.subagent_id)
            if snapshot is None or snapshot.run_generation != event.actor_run_generation:
                LOGGER.debug(
                    "Discarded stale route event route=%s subagent=%s generation=%s.",
                    self._route_id,
                    event.subagent_id,
                    event.actor_run_generation,
                )
                return
        actor_id = event.actor_id or _route_event_actor_id(event)
        actor_sequence = self._actor_event_sequences.get(actor_id, 0) + 1
        self._actor_event_sequences[actor_id] = actor_sequence
        self._event_sequence += 1
        event = replace(
            event,
            origin_session_id=(
                event.origin_session_id
                or (event.session_id if event.agent_kind == "main" else None)
            ),
            origin_turn_id=(
                event.origin_turn_id
                or (event.turn_id if event.agent_kind == "main" else None)
            ),
            actor_id=actor_id,
            actor_sequence=actor_sequence,
            sequence=self._event_sequence,
        )
        await self._event_bus.publish(event)
        await self._maybe_enqueue_subagent_supervisor_followup(event)
        await self._publish_task_status_if_changed(reason=f"route_event:{event.type}")

    async def _publish_task_status_if_changed(self, *, reason: str) -> None:
        active = self._route_task_active()
        if active == self._published_task_active:
            return
        self._published_task_active = active
        LOGGER.info(
            "Route task status changed (route=%s, active=%s, reason=%s, "
            "user_queue=%d, runtime_queue=%d, pending_jobs=%d, pending_subagents=%d).",
            self._route_id,
            active,
            reason,
            self._user_message_queue.qsize(),
            self._message_queue.qsize(),
            len(self._bash_job_supervisor.pending_jobs()),
            sum(
                snapshot.status in {"running", "waiting_background", "awaiting_approval"}
                for snapshot in self._subagent_manager.active_snapshots()
            ),
        )
        actor_id = "main"
        actor_sequence = self._actor_event_sequences.get(actor_id, 0) + 1
        self._actor_event_sequences[actor_id] = actor_sequence
        self._event_sequence += 1
        active_session_id = self._main_loop.active_session_id()
        active_turn_id = self._main_loop.active_turn_id()
        await self._event_bus.publish(
            RouteTaskStatusEvent(
                route_id=self._route_id,
                agent_kind="main",
                agent_name="Jarvis",
                session_id=active_session_id,
                turn_id=active_turn_id,
                origin_session_id=active_session_id,
                origin_turn_id=active_turn_id,
                active=active,
                reason=reason,
                actor_id=actor_id,
                actor_sequence=actor_sequence,
                sequence=self._event_sequence,
            )
        )

    def _route_task_active(self) -> bool:
        if self._bash_job_supervisor.has_pending_jobs():
            return True
        if any(
            snapshot.status in {"running", "waiting_background", "awaiting_approval"}
            for snapshot in self._subagent_manager.active_snapshots()
        ):
            return True
        if self._main_resume_requires_user_message:
            return False
        if self._active_turn_request is not None:
            return True
        if self._main_loop.has_active_turn():
            return True
        if not self._user_message_queue.empty() or not self._message_queue.empty():
            return True
        if self._main_bash_runtime_turn_queued or self._main_subagent_runtime_turn_queued:
            return True
        if self._pending_main_bash_notices or self._pending_main_subagent_notices:
            return True
        return False

    async def _publish_main_local_notice(self, notice_kind: str, text: str) -> None:
        request = self._active_turn_request
        turn_kind: str | None = None
        client_message_id: str | None = None
        if request is not None:
            turn_kind = "user" if request.user_initiated else "runtime"
            client_message_id = request.client_message_id
        await self.publish_event(
            RouteLocalNoticeEvent(
                route_id=self._route_id,
                agent_kind="main",
                agent_name="Jarvis",
                session_id=self._main_loop.active_session_id(),
                turn_id=self._main_loop.active_turn_id(),
                turn_kind=turn_kind,
                client_message_id=client_message_id,
                notice_kind=notice_kind,
                text=text,
            )
        )

    async def stream_turn(
        self,
        user_text: str,
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        subscriber_id, queue = self.subscribe()
        client_message_id = uuid4().hex
        matched_turn_id: str | None = None
        await self.enqueue_user_message(
            user_text,
            client_message_id=client_message_id,
        )
        try:
            while True:
                event = await queue.get()
                if isinstance(event, RouteErrorEvent) and event.agent_kind == "main":
                    code = event.code
                    if code == "context_budget_exceeded":
                        raise ContextBudgetError(event.message)
                    if code == "provider_timeout":
                        raise ProviderTimeoutError(event.message)
                    raise RuntimeError(event.message)
                if event.agent_kind != "main":
                    continue
                if isinstance(event, RouteTurnStartedEvent):
                    if event.client_message_id == client_message_id:
                        matched_turn_id = event.turn_id
                    continue
                if matched_turn_id is None or event.turn_id != matched_turn_id:
                    continue
                mapped = _map_route_event_to_agent_event(event)
                if mapped is None:
                    continue
                yield mapped
                if mapped.type == "done":
                    return
        finally:
            self.unsubscribe(subscriber_id)

    async def run_turn(self, user_text: str) -> AgentTurnResult:
        result: AgentTurnResult | None = None
        async for event in self.stream_turn(user_text):
            if isinstance(event, AgentTurnDoneEvent):
                result = event.to_result()
        if result is None:
            raise RuntimeError("Route runtime turn ended without a final done event.")
        return result

    def _ensure_message_worker(self) -> None:
        if self._message_worker is not None and not self._message_worker.done():
            return
        self._message_worker = asyncio.create_task(
            self._message_worker_loop(),
            name=f"jarvis-route-runtime-{self._route_id}",
        )

    def _invalidate_stale_internal_followups(self) -> None:
        self._internal_followup_generation += 1
        self._main_bash_runtime_turn_queued = False
        self._main_subagent_runtime_turn_queued = False

    async def _dequeue_next_request(
        self,
    ) -> tuple[_RouteTurnRequest, asyncio.Queue[_RouteTurnRequest]]:
        while True:
            try:
                return self._user_message_queue.get_nowait(), self._user_message_queue
            except asyncio.QueueEmpty:
                pass
            try:
                return self._message_queue.get_nowait(), self._message_queue
            except asyncio.QueueEmpty:
                pass
            self._queue_wakeup.clear()
            try:
                return self._user_message_queue.get_nowait(), self._user_message_queue
            except asyncio.QueueEmpty:
                pass
            try:
                return self._message_queue.get_nowait(), self._message_queue
            except asyncio.QueueEmpty:
                pass
            await self._queue_wakeup.wait()

    async def _maybe_schedule_deferred_internal_followups(self) -> None:
        if self._subagent_reset_in_progress:
            return
        if self._new_session_boundary_pending:
            return
        if self._main_resume_requires_user_message:
            return
        if not self._user_message_queue.empty():
            return
        if self._main_bash_runtime_turn_queued or self._main_subagent_runtime_turn_queued:
            return
        if self._pending_main_bash_notices:
            await self._message_queue.put(
                _RouteTurnRequest(
                    user_text=None,
                    force_session_id=self._resolve_main_bash_notice_session_id(
                        tuple(self._pending_main_bash_notices.values())
                    ),
                    parse_commands=False,
                    user_initiated=False,
                    internal_generation=self._internal_followup_generation,
                    runtime_turn_kind=_MAIN_BASH_PROGRESS_RUNTIME_KIND,
                )
            )
            self._main_bash_runtime_turn_queued = True
            self._queue_wakeup.set()
        if self._pending_main_subagent_notices and not self._main_bash_runtime_turn_queued:
            first_notice = next(iter(self._pending_main_subagent_notices.values()))
            await self._message_queue.put(
                _RouteTurnRequest(
                    user_text=None,
                    force_session_id=self._resolve_main_subagent_notice_session_id(first_notice),
                    parse_commands=False,
                    user_initiated=False,
                    internal_generation=self._internal_followup_generation,
                    runtime_turn_kind=_MAIN_SUBAGENT_PROGRESS_RUNTIME_KIND,
                )
            )
            self._main_subagent_runtime_turn_queued = True
            self._queue_wakeup.set()

    async def _message_worker_loop(self) -> None:
        while True:
            request, source_queue = await self._dequeue_next_request()
            self._active_turn_request = request
            parsed_command = (
                parse_user_command(request.user_text)
                if request.parse_commands and request.user_text is not None
                else None
            )
            emitted_main_turn_event = False
            is_new_command = parsed_command is not None and parsed_command.kind == "new"
            try:
                if request.user_initiated:
                    if is_new_command:
                        self._main_resume_requires_user_message = True
                    elif (
                        parsed_command is None
                        or parsed_command.kind == "message"
                    ) and self._main_resume_requires_user_message:
                        self._main_resume_requires_user_message = False
                else:
                    if self._internal_request_is_blocked(request):
                        continue
                if request.runtime_turn_kind == _MAIN_BASH_PROGRESS_RUNTIME_KIND:
                    self._main_bash_runtime_turn_queued = False
                    runtime_message = self._drain_main_bash_progress_message(
                        force_session_id=request.force_session_id,
                    )
                    if runtime_message is None:
                        continue
                    force_session_id, system_message, notices = runtime_message
                    if self._internal_request_is_blocked(request):
                        continue
                    if str(system_message.metadata.get("recommended_action", "")) == "wait":
                        self._record_bash_notice_delivery(notices)
                        continue
                    published = await self._publish_main_system_message(
                        session_id=force_session_id,
                        message=system_message,
                        notices=notices,
                    )
                    if not published:
                        continue
                    if self._internal_request_is_blocked(request):
                        continue
                    event_stream = self._main_loop.stream_runtime_turn(
                        force_session_id=force_session_id,
                        pre_turn_messages=(),
                    )
                elif request.runtime_turn_kind == _MAIN_SUBAGENT_PROGRESS_RUNTIME_KIND:
                    self._main_subagent_runtime_turn_queued = False
                    runtime_message = self._drain_main_subagent_progress_message(
                        force_session_id=request.force_session_id,
                    )
                    if runtime_message is None:
                        continue
                    force_session_id, system_message, notices = runtime_message
                    if self._internal_request_is_blocked(request):
                        continue
                    if str(system_message.metadata.get("recommended_action", "")) == "wait":
                        await self._publish_main_subagent_system_message(
                            session_id=force_session_id,
                            message=system_message,
                            notices=notices,
                        )
                        continue
                    published = await self._publish_main_subagent_system_message(
                        session_id=force_session_id,
                        message=system_message,
                        notices=notices,
                    )
                    if not published:
                        continue
                    if self._internal_request_is_blocked(request):
                        continue
                    event_stream = self._main_loop.stream_runtime_turn(
                        force_session_id=force_session_id,
                        pre_turn_messages=(),
                    )
                elif request.parse_commands:
                    if request.user_text is None:
                        continue
                    if is_new_command:
                        await self._prepare_new_session_request()
                        self._new_session_boundary_pending = False
                    event_stream = self._main_loop.stream_user_input(request.user_text)
                elif request.user_text is None:
                    event_stream = self._main_loop.stream_runtime_turn(
                        force_session_id=request.force_session_id,
                        pre_turn_messages=request.pre_turn_messages,
                    )
                else:
                    event_stream = self._main_loop.stream_turn(
                        user_text=request.user_text,
                        force_session_id=request.force_session_id,
                        pre_turn_messages=request.pre_turn_messages,
                    )
                async for event in event_stream:
                    emitted_main_turn_event = True
                    await self._publish_main_loop_event(event, request=request)
            except ContextBudgetError as exc:
                error_log_path = self._write_runtime_error_log(
                    request=request,
                    session_id=self._main_loop.active_session_id() or request.force_session_id,
                    turn_id=self._main_loop.active_turn_id(),
                    published_turn_kind="user" if request.user_initiated else "runtime",
                    published_client_message_id=request.client_message_id,
                    parsed_command_kind=(
                        parsed_command.kind if parsed_command is not None else None
                    ),
                    exc=exc,
                    error_code="context_budget_exceeded",
                )
                self._print_runtime_error_notice(error_log_path=error_log_path)
                await self.publish_event(
                    RouteErrorEvent(
                        route_id=self._route_id,
                        agent_kind="main",
                        agent_name="Jarvis",
                        session_id=self._main_loop.active_session_id(),
                        turn_id=self._main_loop.active_turn_id(),
                        turn_kind="user" if request.user_initiated else "runtime",
                        client_message_id=request.client_message_id,
                        code="context_budget_exceeded",
                        message=str(exc),
                    )
                )
            except (
                ProviderTimeoutError,
                ProviderTemporaryError,
                ProviderRateLimitError,
            ) as exc:
                error_log_path = self._write_runtime_error_log(
                    request=request,
                    session_id=self._main_loop.active_session_id() or request.force_session_id,
                    turn_id=self._main_loop.active_turn_id(),
                    published_turn_kind="user" if request.user_initiated else "runtime",
                    published_client_message_id=request.client_message_id,
                    parsed_command_kind=(
                        parsed_command.kind if parsed_command is not None else None
                    ),
                    exc=exc,
                    error_code=_provider_error_code(exc),
                )
                self._print_runtime_error_notice(error_log_path=error_log_path)
                if await self._enqueue_main_provider_recovery(
                    request=request,
                    exc=exc,
                    error_log_path=error_log_path,
                ):
                    continue
                await self.publish_event(
                    RouteErrorEvent(
                        route_id=self._route_id,
                        agent_kind="main",
                        agent_name="Jarvis",
                        session_id=self._main_loop.active_session_id(),
                        turn_id=self._main_loop.active_turn_id(),
                        turn_kind="user" if request.user_initiated else "runtime",
                        client_message_id=request.client_message_id,
                        code=_provider_error_code(exc),
                        message=_PROVIDER_TIMEOUT_MESSAGE,
                    )
                )
            except CodexBackendError as exc:
                error_log_path = self._write_runtime_error_log(
                    request=request,
                    session_id=self._main_loop.active_session_id() or request.force_session_id,
                    turn_id=self._main_loop.active_turn_id(),
                    published_turn_kind="user" if request.user_initiated else "runtime",
                    published_client_message_id=request.client_message_id,
                    parsed_command_kind=(
                        parsed_command.kind if parsed_command is not None else None
                    ),
                    exc=exc,
                    error_code="codex_backend_error",
                )
                self._print_runtime_error_notice(error_log_path=error_log_path)
                await self.publish_event(
                    RouteErrorEvent(
                        route_id=self._route_id,
                        agent_kind="main",
                        agent_name="Jarvis",
                        session_id=self._main_loop.active_session_id(),
                        turn_id=self._main_loop.active_turn_id(),
                        turn_kind="user" if request.user_initiated else "runtime",
                        client_message_id=request.client_message_id,
                        code="codex_backend_error",
                        message=str(exc),
                    )
                )
            except Exception as exc:
                error_turn_kind: str | None = "user" if request.user_initiated else "runtime"
                error_client_message_id = request.client_message_id
                if (
                    parsed_command is not None
                    and parsed_command.kind == "new"
                    and request.user_initiated
                    and not emitted_main_turn_event
                ):
                    error_turn_kind = None
                    error_client_message_id = None
                error_log_path = self._write_runtime_error_log(
                    request=request,
                    session_id=self._main_loop.active_session_id() or request.force_session_id,
                    turn_id=self._main_loop.active_turn_id(),
                    published_turn_kind=error_turn_kind,
                    published_client_message_id=error_client_message_id,
                    parsed_command_kind=parsed_command.kind if parsed_command is not None else None,
                    exc=exc,
                )
                self._print_runtime_error_notice(error_log_path=error_log_path)
                await self.publish_event(
                    RouteErrorEvent(
                        route_id=self._route_id,
                        agent_kind="main",
                        agent_name="Jarvis",
                        session_id=self._main_loop.active_session_id(),
                        turn_id=self._main_loop.active_turn_id(),
                        turn_kind=error_turn_kind,
                        client_message_id=error_client_message_id,
                        code="internal_error",
                        message=_INTERNAL_ERROR_MESSAGE,
                    )
                )
            finally:
                if is_new_command and self._new_session_boundary_pending:
                    self._new_session_boundary_pending = False
                    self._main_resume_requires_user_message = True
                self._active_turn_request = None
                source_queue.task_done()
                await self._maybe_schedule_deferred_internal_followups()
                await self._publish_task_status_if_changed(reason="turn_worker_idle")

    async def _maybe_enqueue_subagent_supervisor_followup(self, event: RouteEvent) -> None:
        if self._subagent_reset_in_progress:
            return
        if not isinstance(event, RouteSystemNoticeEvent):
            return
        if event.agent_kind != "subagent" or event.subagent_id is None:
            return
        if event.actor_run_generation is not None:
            snapshot = self._subagent_manager.snapshot_for(event.subagent_id)
            if snapshot is None or snapshot.run_generation != event.actor_run_generation:
                LOGGER.debug(
                    "Discarded stale subagent notice route=%s subagent=%s event_generation=%s.",
                    self._route_id,
                    event.subagent_id,
                    event.actor_run_generation,
                )
                return
        if event.notice_kind not in _SUBAGENT_MAIN_PROGRESS_NOTICE_KINDS:
            return
        if self._main_resume_requires_user_message or self._new_session_boundary_pending:
            return
        if event.notice_kind != "subagent_waiting_background":
            self._cancel_orchestrator_wait(reset_backoff=True)
        else:
            self._orchestrator_unchanged_waits = 0
        await self._enqueue_main_subagent_followup(event)

    async def _enqueue_main_provider_recovery(
        self,
        *,
        request: _RouteTurnRequest,
        exc: Exception,
        error_log_path: Path,
    ) -> bool:
        """Queue a fresh checkpoint-based turn after a recoverable provider failure."""

        if self._new_session_boundary_pending or self._main_resume_requires_user_message:
            return False
        session_id = self._main_loop.active_session_id() or request.force_session_id
        if session_id is None:
            return False
        task_id = (
            request.provider_recovery_task_id
            or request.client_message_id
            or self._main_loop.active_turn_id()
            or uuid4().hex
        )
        recovery_state = self._main_backend_state(session_id=session_id).get(
            "provider_recovery"
        )
        persisted_attempt = 0
        if isinstance(recovery_state, Mapping) and recovery_state.get("task_id") == task_id:
            persisted_attempt = _nonnegative_int(recovery_state.get("attempt"))
        attempt = max(request.provider_recovery_attempt, persisted_attempt) + 1
        if attempt > _MAX_MAIN_PROVIDER_RECOVERY_ATTEMPTS:
            pause_text = (
                "I paused this task after repeated model-provider failures. The durable "
                "checkpoint is preserved, so a later user message can resume without "
                "replaying completed work."
            )
            pause_metadata = {
                "provider_recovery_exhausted": True,
                "attempts": attempt - 1,
                "task_id": task_id,
                "error_type": type(exc).__name__,
                "error_log_path": str(error_log_path),
            }
            self._update_main_backend_state(
                session_id=session_id,
                key="provider_recovery",
                value={**pause_metadata, "status": "paused"},
            )
            self._main_resume_requires_user_message = True
            self._invalidate_stale_internal_followups()
            self._append_main_assistant_note(
                pause_text,
                session_id=session_id,
                metadata=pause_metadata,
            )
            await self.publish_event(
                RouteAssistantMessageEvent(
                    route_id=self._route_id,
                    agent_kind="main",
                    agent_name="Jarvis",
                    session_id=session_id,
                    turn_id=self._main_loop.active_turn_id(),
                    turn_kind="runtime",
                    text=pause_text,
                )
            )
            await self.publish_event(
                RouteTurnDoneEvent(
                    route_id=self._route_id,
                    agent_kind="main",
                    agent_name="Jarvis",
                    session_id=session_id,
                    turn_id=self._provider_recovery_turn_ids.pop(task_id, None),
                    turn_kind="runtime" if not request.user_initiated else "user",
                    client_message_id=request.client_message_id,
                    origin_session_id=session_id,
                    response_text=pause_text,
                    completion_blocked=True,
                    completion_block_reason="provider_recovery_exhausted",
                    interruption_reason="provider_recovery_exhausted",
                )
            )
            return True

        exception_metadata = _exception_metadata(exc)
        emitted_output = bool(exception_metadata.get("emitted_output"))
        self._update_main_backend_state(
            session_id=session_id,
            key="provider_recovery",
            value={
                "task_id": task_id,
                "attempt": attempt,
                "status": "queued",
                "partial_output_persisted": emitted_output,
                "error_log_path": str(error_log_path),
            },
        )
        message = AgentRuntimeMessage(
            role="system",
            metadata={
                "provider_recovery": True,
                "provider_recovery_attempt": attempt,
                "provider_error_type": type(exc).__name__,
                "provider_error_log_path": str(error_log_path),
                "provider_recovery_task_id": task_id,
                "partial_output_persisted": emitted_output,
            },
            content=(
                (
                    "The previous provider attempt ended after partial output, which was "
                    "checkpointed. Continue from that durable fragment without repeating it. "
                    if emitted_output
                    else "The previous provider attempt produced no semantic output. Start a "
                    "fresh generation from the durable checkpoint. "
                )
                + "Do not replay completed tool calls or claim partial work as finished."
            ),
        )
        await self._message_queue.put(
            _RouteTurnRequest(
                user_text=None,
                force_session_id=session_id,
                pre_turn_messages=(message,),
                parse_commands=False,
                user_initiated=False,
                client_message_id=request.client_message_id,
                internal_generation=self._internal_followup_generation,
                runtime_turn_kind=_MAIN_PROVIDER_RECOVERY_RUNTIME_KIND,
                provider_recovery_attempt=attempt,
                provider_recovery_task_id=task_id,
            )
        )
        self._queue_wakeup.set()
        LOGGER.warning(
            "Queued provider recovery route=%s session=%s attempt=%s error=%s.",
            self._route_id,
            session_id,
            attempt,
            type(exc).__name__,
        )
        return True

    async def _publish_main_loop_event(
        self,
        event: AgentTurnStreamEvent,
        *,
        request: _RouteTurnRequest,
    ) -> None:
        turn_kind = "user" if request.user_initiated else "runtime"
        route_event_kwargs = {
            "route_id": self._route_id,
            "agent_kind": "main",
            "agent_name": "Jarvis",
            "session_id": event.session_id,
            "turn_id": getattr(event, "turn_id", None) or None,
            "turn_kind": turn_kind,
            "client_message_id": request.client_message_id,
            "origin_session_id": event.session_id,
            "origin_turn_id": getattr(event, "turn_id", None) or None,
        }
        if isinstance(event, AgentTurnStartedEvent):
            if request.provider_recovery_task_id and event.turn_id:
                self._provider_recovery_turn_ids[
                    request.provider_recovery_task_id
                ] = event.turn_id
            await self.publish_event(RouteTurnStartedEvent(**route_event_kwargs))
            return
        if isinstance(event, AgentTextDeltaEvent):
            await self.publish_event(
                RouteAssistantDeltaEvent(
                    **route_event_kwargs,
                    delta=event.delta,
                )
            )
            return
        if isinstance(event, AgentAssistantMessageEvent):
            await self.publish_event(
                RouteAssistantMessageEvent(
                    **route_event_kwargs,
                    text=event.text,
                )
            )
            return
        if isinstance(event, AgentToolCallEvent):
            await self.publish_event(
                RouteToolCallEvent(
                    **route_event_kwargs,
                    tool_names=event.tool_names,
                )
            )
            return
        if isinstance(event, AgentApprovalRequestEvent):
            self._approval_registry.register(event.approval_id, self._main_loop)
            await self.publish_event(
                RouteApprovalRequestEvent(
                    **route_event_kwargs,
                    approval_id=event.approval_id,
                    kind=event.kind,
                    summary=event.summary,
                    details=event.details,
                    command=event.command,
                    tool_name=event.tool_name,
                    inspection_url=event.inspection_url,
                )
            )
            return
        if isinstance(event, AgentTurnDoneEvent):
            self._update_main_backend_state(
                session_id=event.session_id,
                key="provider_recovery",
                value=None,
            )
            if event.session_id and event.turn_id:
                self._selected_skills_by_main_turn.pop((event.session_id, event.turn_id), None)
            if request.provider_recovery_task_id:
                self._provider_recovery_turn_ids.pop(
                    request.provider_recovery_task_id,
                    None,
                )
            await self.publish_event(
                RouteTurnDoneEvent(
                    **route_event_kwargs,
                    response_text=event.response_text,
                    command=event.command,
                    compaction_performed=event.compaction_performed,
                    interrupted=event.interrupted,
                    approval_rejected=event.approval_rejected,
                    completion_blocked=event.completion_blocked,
                    completion_block_reason=event.completion_block_reason,
                    interruption_reason=event.interruption_reason,
                )
            )

    def _main_backend_state(self, *, session_id: str) -> dict[str, Any]:
        loader = getattr(self._main_loop, "backend_state", None)
        if callable(loader):
            value = loader(session_id=session_id)
            if isinstance(value, dict):
                return value
        return dict(self._main_backend_state_fallback)

    def _update_main_backend_state(
        self,
        *,
        session_id: str,
        key: str,
        value: Any | None,
    ) -> bool:
        updater = getattr(self._main_loop, "update_backend_state", None)
        if callable(updater):
            return bool(updater(session_id=session_id, key=key, value=value))
        if value is None:
            self._main_backend_state_fallback.pop(key, None)
        else:
            self._main_backend_state_fallback[key] = value
        return True

    def _append_main_assistant_note(
        self,
        content: str,
        *,
        session_id: str,
        metadata: dict[str, Any],
    ) -> bool:
        appender = getattr(self._main_loop, "append_assistant_note", None)
        if callable(appender):
            return bool(appender(content, session_id=session_id, metadata=metadata))
        return False

    def _build_main_tool_definitions(
        self,
        activated_discoverable_tool_names: Sequence[str],
    ) -> tuple[ToolDefinition, ...]:
        definitions = list(self._main_registry.basic_definitions())
        seen_names = {definition.name for definition in definitions}
        for definition in self._main_registry.resolve_discoverable_tool_definitions(
            activated_discoverable_tool_names
        ):
            if definition.name in seen_names:
                continue
            definitions.append(definition)
            seen_names.add(definition.name)
        for definition in build_subagent_primitive_definitions():
            if definition.name in seen_names:
                continue
            definitions.append(definition)
            seen_names.add(definition.name)
        return tuple(definitions)

    async def _execute_main_tool_call(
        self,
        tool_call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        if tool_call.name in SUBAGENT_PRIMITIVE_NAMES:
            return await self._execute_subagent_primitive(tool_call, context)
        if tool_call.name == "bash" and _is_wait_only_bash_call(tool_call):
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                ok=False,
                content=(
                    "Wait-only detached timers are not allowed. Use orchestrator_wait when "
                    "route-owned work is still running and Jarvis has nothing actionable to do."
                ),
                metadata={
                    "execution_failed": True,
                    "error_code": "orchestrator_wait_required",
                },
            )
        try:
            async with self._workspace_access.execute(
                tool_call=tool_call,
                context=context,
            ) as workspace_observation:
                result = await self._main_tool_runtime.execute(
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
        self._remember_main_turn_skill(context=context, result=result)
        await self._bash_job_supervisor.observe_tool_result(result=result, context=context)
        return result

    def _remember_main_turn_skill(
        self,
        *,
        context: ToolExecutionContext,
        result: ToolExecutionResult,
    ) -> None:
        if (
            context.agent_kind != "main"
            or result.name != "get_skills"
            or not result.ok
            or context.session_id is None
            or context.turn_id is None
        ):
            return
        skill = result.metadata.get("skill")
        if not isinstance(skill, Mapping):
            return
        skill_id = str(skill.get("skill_id", "")).strip()
        if not skill_id:
            return
        key = (context.session_id, context.turn_id)
        prior = self._selected_skills_by_main_turn.get(key, ())
        self._selected_skills_by_main_turn[key] = tuple(
            dict.fromkeys((*prior, skill_id))
        )[:4]
        session_prior = self._selected_skills_by_main_session.get(context.session_id, ())
        self._selected_skills_by_main_session[context.session_id] = tuple(
            dict.fromkeys((*session_prior, skill_id))
        )[:4]

    async def _execute_subagent_primitive(
        self,
        tool_call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        try:
            payload: dict[str, Any]
            if tool_call.name == "orchestrator_wait":
                (
                    review_required,
                    changed_test_artifact_paths,
                    changed_paths,
                ) = self._consume_orchestrator_wait_notices(
                    session_id=context.session_id,
                )
                if review_required:
                    return ToolExecutionResult(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        ok=False,
                        content=(
                            "Orchestrator wait requires review\n"
                            + "\n\n".join(review_required)
                        ),
                        metadata={
                            "execution_failed": True,
                            "error_code": "orchestrator_review_required",
                            "review_required": True,
                            "review_items": list(review_required),
                            "changed_test_artifact_paths": list(
                                changed_test_artifact_paths
                            ),
                            "changed_paths": list(changed_paths),
                        },
                    )
                payload = self._register_orchestrator_wait(
                    wake_after_seconds=int(
                        tool_call.arguments.get("wake_after_seconds", 0)
                    ),
                    reason=str(tool_call.arguments.get("reason", "")),
                    watch_actor_ids=_optional_string_tuple(
                        tool_call.arguments.get("watch_actor_ids")
                    ),
                    session_id=context.session_id,
                )
                return ToolExecutionResult(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    ok=True,
                    content=(
                        "Orchestrator wait registered\n" + _format_payload_lines(payload)
                    ),
                    metadata={"orchestrator_wait": True, **payload},
                    turn_disposition="yield_turn",
                )
            if tool_call.name == "subagent_invoke":
                task_label = str(tool_call.arguments.get("task_label", "")).strip()
                instructions = str(tool_call.arguments.get("instructions", "")).strip()
                if not task_label or not instructions:
                    raise ValueError("'task_label' and 'instructions' are required.")
                session_id = context.session_id
                turn_id = context.turn_id
                if session_id is None or turn_id is None:
                    raise ValueError("Subagent invocation requires a main session and turn id.")
                has_explicit_skill_ids = "skill_ids" in tool_call.arguments
                requested_skill_ids = _optional_string_tuple(
                    tool_call.arguments.get("skill_ids")
                )
                inherited_skill_ids = self._selected_skills_by_main_turn.get(
                    (session_id, turn_id),
                    (),
                )
                if has_explicit_skill_ids and not requested_skill_ids:
                    # An explicit empty list is a deliberate opt-out from the
                    # same-turn convenience inheritance.
                    skill_ids = ()
                else:
                    skill_ids = tuple(
                        dict.fromkeys((*requested_skill_ids, *inherited_skill_ids))
                    )[:4]
                skill_selection_warning = None
                if (
                    not has_explicit_skill_ids
                    and not requested_skill_ids
                    and not inherited_skill_ids
                    and context.session_id is not None
                ):
                    available_skill_ids = self._selected_skills_by_main_session.get(
                        context.session_id,
                        (),
                    )
                    if available_skill_ids:
                        skill_selection_warning = (
                            "No skills were attached to this child. Skills selected earlier "
                            "in this user request are available for explicit reuse; pass "
                            "the exact skill ids if this assignment needs them. Available ids: "
                            + ", ".join(available_skill_ids)
                        )
                payload = await self._subagent_manager.invoke(
                    requester_kind=context.agent_kind,
                    task_label=task_label,
                    instructions=instructions,
                    user_constraints=_optional_string(
                        tool_call.arguments.get("user_constraints")
                    ),
                    shared_context=_optional_string(
                        tool_call.arguments.get("shared_context")
                    ),
                    owned_paths=_optional_string_tuple(
                        tool_call.arguments.get("owned_paths")
                    ),
                    skill_ids=skill_ids,
                    phase=_optional_string(tool_call.arguments.get("phase")),
                    depends_on=_optional_string_tuple(
                        tool_call.arguments.get("depends_on")
                    ),
                    seam_contract=_optional_string(
                        tool_call.arguments.get("seam_contract")
                    ),
                    deliverable=_optional_string(tool_call.arguments.get("deliverable")),
                    owner_main_session_id=session_id,
                    owner_main_turn_id=turn_id,
                )
                if skill_selection_warning is not None:
                    payload["skill_selection_warning"] = skill_selection_warning
                return _tool_result_for_payload(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    title="Subagent invoked",
                    payload=payload,
                )
            if tool_call.name == "subagent_handoff":
                agent = str(tool_call.arguments.get("agent", "")).strip()
                if not agent:
                    raise ValueError("'agent' is required.")
                payload = await self._subagent_manager.handoff(agent=agent)
                return _tool_result_for_payload(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    title="Subagent handoff",
                    payload=payload,
                )
            if tool_call.name == "subagent_monitor":
                payload = await self._subagent_manager.monitor(
                    agent=_optional_string(tool_call.arguments.get("agent")),
                    detail=_optional_string(tool_call.arguments.get("detail")) or "summary",
                )
                return _tool_result_for_payload(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    title="Subagent monitor",
                    payload=payload,
                )
            if tool_call.name == "subagent_stop":
                agent = str(tool_call.arguments.get("agent", "")).strip()
                if not agent:
                    raise ValueError("'agent' is required.")
                payload = await self._subagent_manager.stop(
                    agent=agent,
                    reason=_optional_string(tool_call.arguments.get("reason")),
                )
                return _tool_result_for_payload(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    title="Subagent stop",
                    payload=payload,
                )
            if tool_call.name == "subagent_step_in":
                agent = str(tool_call.arguments.get("agent", "")).strip()
                instructions = str(tool_call.arguments.get("instructions", "")).strip()
                if not agent or not instructions:
                    raise ValueError("'agent' and 'instructions' are required.")
                payload = await self._subagent_manager.step_in(
                    agent=agent,
                    instructions=instructions,
                )
                return _tool_result_for_payload(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    title="Subagent step-in",
                    payload=payload,
                )
            if tool_call.name == "subagent_dispose":
                agent = str(tool_call.arguments.get("agent", "")).strip()
                if not agent:
                    raise ValueError("'agent' is required.")
                payload = await self._subagent_manager.dispose(agent=agent)
                return _tool_result_for_payload(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    title="Subagent disposed",
                    payload=payload,
                )
            raise ValueError(f"Unknown subagent primitive: {tool_call.name}")
        except Exception as exc:
            failure_title = (
                "Orchestrator wait failed"
                if tool_call.name == "orchestrator_wait"
                else "Subagent control failed"
            )
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                ok=False,
                content=(
                    f"{failure_title}\n"
                    f"tool: {tool_call.name}\n"
                    f"error_type: {type(exc).__name__}\n"
                    f"error: {exc}"
                ),
                metadata={
                    "execution_failed": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "arguments": dict(tool_call.arguments),
                },
            )

    def _register_orchestrator_wait(
        self,
        *,
        wake_after_seconds: int,
        reason: str,
        watch_actor_ids: tuple[str, ...],
        session_id: str | None,
    ) -> dict[str, Any]:
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("orchestrator_wait reason must be non-empty.")
        pending_subagents = tuple(
            snapshot
            for snapshot in self._subagent_manager.active_snapshots()
            if snapshot.status in {"running", "waiting_background", "awaiting_approval"}
        )
        pending_jobs = self._bash_job_supervisor.pending_jobs(include_services=True)
        if not pending_subagents and not pending_jobs:
            raise ValueError("No route-owned work is pending; orchestrator_wait is not valid.")
        known_actor_ids = {
            *(snapshot.subagent_id for snapshot in pending_subagents),
            *(record.job_id for record in pending_jobs),
        }
        unknown_actor_ids = tuple(
            actor_id for actor_id in watch_actor_ids if actor_id not in known_actor_ids
        )
        if unknown_actor_ids:
            raise ValueError(
                "Unknown or non-pending watch_actor_ids: " + ", ".join(unknown_actor_ids)
            )

        adaptive_floor = min(
            _ORCHESTRATOR_WAIT_MAX_SECONDS,
            _ORCHESTRATOR_WAIT_BACKOFF_BASE_SECONDS
            * (2 ** min(self._orchestrator_unchanged_waits, 8)),
        )
        effective_seconds = min(
            _ORCHESTRATOR_WAIT_MAX_SECONDS,
            max(
                _ORCHESTRATOR_WAIT_MIN_SECONDS,
                adaptive_floor,
                wake_after_seconds,
            ),
        )
        self._cancel_orchestrator_wait(reset_backoff=False)
        generation = self._orchestrator_wait_generation
        target_session_id = session_id or self._main_loop.active_session_id()
        self._orchestrator_wait_task = asyncio.create_task(
            self._orchestrator_wait_deadline(
                generation=generation,
                delay_seconds=effective_seconds,
                session_id=target_session_id,
                reason=normalized_reason,
            ),
            name=f"jarvis-orchestrator-wait-{self._route_id}-{generation}",
        )
        LOGGER.info(
            "Orchestrator wait registered (route=%s, generation=%d, requested=%d, "
            "effective=%d, unchanged_waits=%d, actors=%s).",
            self._route_id,
            generation,
            wake_after_seconds,
            effective_seconds,
            self._orchestrator_unchanged_waits,
            ",".join(watch_actor_ids or tuple(sorted(known_actor_ids))),
        )
        return {
            "requested_wait_seconds": wake_after_seconds,
            "effective_wait_seconds": effective_seconds,
            "adaptive_floor_seconds": adaptive_floor,
            "hard_ceiling_seconds": _ORCHESTRATOR_WAIT_MAX_SECONDS,
            "reason": normalized_reason,
            "watch_actor_ids": list(watch_actor_ids or tuple(sorted(known_actor_ids))),
            "pending_subagent_count": len(pending_subagents),
            "pending_job_count": len(pending_jobs),
        }

    def _consume_orchestrator_wait_notices(
        self,
        *,
        session_id: str | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Persist routine notices and return only material items needing model review."""

        target_session_id = session_id or self._main_loop.active_session_id()
        if target_session_id is None:
            return (
                ("No active main session is available for persisted notice handling.",),
                (),
                (),
            )

        review_items: list[str] = []
        changed_test_artifact_paths: set[str] = set()
        changed_paths: set[str] = set()
        bash_notices = tuple(self._pending_main_bash_notices.values())
        if bash_notices:
            bash_message = self._build_main_bash_job_followup_message(bash_notices)
            recommendation = str(
                bash_message.metadata.get("recommended_action", "inspect")
            )
            if recommendation == "wait":
                if not self._main_loop.append_system_note(
                    bash_message.content,
                    session_id=target_session_id,
                    metadata=bash_message.metadata,
                ):
                    return (
                        ("Routine detached-job notices could not be persisted.",),
                        (),
                        (),
                    )
            else:
                review_items.append(bash_message.content)
            self._pending_main_bash_notices.clear()
            self._record_bash_notice_delivery(bash_notices)

        for subagent_id, notice in tuple(self._pending_main_subagent_notices.items()):
            snapshot = self._subagent_manager.snapshot_for(subagent_id)
            if snapshot is None or snapshot.status == "disposed":
                self._pending_main_subagent_notices.pop(subagent_id, None)
                continue
            if (
                notice.actor_run_generation is not None
                and notice.actor_run_generation != snapshot.run_generation
            ):
                self._pending_main_subagent_notices.pop(subagent_id, None)
                continue
            progress = self._subagent_manager.build_main_progress_message(
                agent=subagent_id,
                notice_kind=notice.notice_kind,
                notice_text=notice.text,
            )
            if progress is None:
                self._pending_main_subagent_notices.pop(subagent_id, None)
                continue
            _owner_session_id, message = progress
            recommendation = str(message.metadata.get("recommended_action", "inspect"))
            if recommendation == "wait":
                if not self._main_loop.append_system_note(
                    message.content,
                    session_id=target_session_id,
                    metadata=message.metadata,
                ):
                    changed_test_artifact_paths.update(
                        snapshot.changed_test_artifact_paths
                    )
                    changed_paths.update(snapshot.changed_paths)
                    return (
                        (
                            *review_items,
                            f"Routine notice for subagent {subagent_id} could not be persisted.",
                        ),
                        tuple(sorted(changed_test_artifact_paths)),
                        tuple(sorted(changed_paths)),
                    )
            else:
                review_items.append(message.content)
                changed_test_artifact_paths.update(
                    snapshot.changed_test_artifact_paths
                )
                changed_paths.update(snapshot.changed_paths)
            self._pending_main_subagent_notices.pop(subagent_id, None)

        return (
            tuple(review_items),
            tuple(sorted(changed_test_artifact_paths)),
            tuple(sorted(changed_paths)),
        )

    async def _orchestrator_wait_deadline(
        self,
        *,
        generation: int,
        delay_seconds: int,
        session_id: str | None,
        reason: str,
    ) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            if generation != self._orchestrator_wait_generation:
                return
            if not self._internal_followups_allowed():
                return
            if not self._route_has_pending_background_work():
                return
            self._orchestrator_unchanged_waits += 1
            LOGGER.info(
                "Orchestrator wait liveness deadline elapsed "
                "(route=%s, generation=%d, unchanged_waits=%d).",
                self._route_id,
                generation,
                self._orchestrator_unchanged_waits,
            )
            await self._message_queue.put(
                _RouteTurnRequest(
                    user_text=None,
                    force_session_id=session_id,
                    pre_turn_messages=(
                        AgentRuntimeMessage(
                            role="system",
                            metadata={
                                "orchestrator_liveness_review": True,
                                "orchestrator_wait_reason": reason,
                            },
                            content=(
                                "The orchestrator liveness deadline elapsed while route-owned "
                                "work remains pending. Review the current actor snapshot, act only "
                                "if something is actionable, or call orchestrator_wait again."
                            ),
                        ),
                    ),
                    parse_commands=False,
                    user_initiated=False,
                    internal_generation=self._internal_followup_generation,
                    runtime_turn_kind=_MAIN_ORCHESTRATOR_REVIEW_RUNTIME_KIND,
                    orchestrator_wait_generation=generation,
                )
            )
            self._queue_wakeup.set()
            self._ensure_message_worker()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_log_path = record_runtime_error(
                transcript_archive_dir=self._core_settings.transcript_archive_dir,
                route_id=self._route_id,
                session_id=session_id,
                component="gateway.route_runtime",
                event="orchestrator_wait_deadline_failed",
                agent_kind="main",
                exc=exc,
                error_code="orchestrator_wait_deadline_failed",
                message=(
                    f"Route {self._route_id} orchestrator liveness deadline failed."
                ),
                context={"wait_generation": generation, "wait_reason": reason},
            )
            self._print_runtime_error_notice(error_log_path=error_log_path)
        finally:
            current = asyncio.current_task()
            if self._orchestrator_wait_task is current:
                self._orchestrator_wait_task = None

    def _cancel_orchestrator_wait(self, *, reset_backoff: bool) -> None:
        task = self._orchestrator_wait_task
        self._orchestrator_wait_task = None
        self._orchestrator_wait_generation += 1
        if task is not None and not task.done():
            task.cancel()
        if reset_backoff:
            self._orchestrator_unchanged_waits = 0

    def _route_has_pending_background_work(self) -> bool:
        if self._bash_job_supervisor.has_pending_jobs(include_services=True):
            return True
        return any(
            snapshot.status in {"running", "waiting_background", "awaiting_approval"}
            for snapshot in self._subagent_manager.active_snapshots()
        )


    def _append_user_stop_subagent_note(
        self,
        affected_subagents: Sequence[SubagentSnapshot],
    ) -> None:
        session_id = self._main_loop.active_session_id()
        if session_id is None and affected_subagents:
            owner_session_id = affected_subagents[0].owner_main_session_id
            if owner_session_id.strip():
                session_id = owner_session_id
        if session_id is None:
            return

        lines = [_SUBAGENT_USER_STOP_NOTE_HEADER, "", "Affected subagents:"]
        subagent_ids: list[str] = []
        codenames: list[str] = []
        for snapshot in affected_subagents:
            codename = snapshot.codename.strip() or "Unknown"
            subagent_id = snapshot.subagent_id.strip() or "unknown"
            status = snapshot.status.strip() or "unknown"
            lines.append(f"- {codename} ({subagent_id}) [status_at_stop_request={status}]")
            subagent_ids.append(subagent_id)
            codenames.append(codename)
        lines.extend(
            [
                "",
                "When you resume, inspect current subagent status, then decide whether to resume it, step in, hand it off, dispose it, or otherwise handle it so no paused child is left orphaned.",
            ]
        )
        self._main_loop.append_system_note(
            "\n".join(lines),
            session_id=session_id,
            metadata={
                "user_stop_subagents": True,
                "subagent_ids": subagent_ids,
                "subagent_codenames": codenames,
            },
        )

    def _append_user_stop_bash_job_note(
        self,
        pending_jobs: Sequence[object],
    ) -> None:
        session_id = self._main_loop.active_session_id()
        if session_id is None and pending_jobs:
            owner_session_id = getattr(pending_jobs[0], "owner_session_id", None)
            if isinstance(owner_session_id, str) and owner_session_id.strip():
                session_id = owner_session_id
        if session_id is None:
            return

        lines = [
            "The user issued /stop while detached bash jobs were still pending.",
            "",
            "Pending detached bash jobs:",
        ]
        job_ids: list[str] = []
        for record in pending_jobs:
            job_id = str(getattr(record, "job_id", "")).strip() or "unknown"
            command = str(getattr(record, "command", "")).strip() or "(unknown command)"
            owner_kind = str(getattr(record, "owner_agent_kind", "")).strip() or "main"
            owner_subagent_id = str(getattr(record, "owner_subagent_id", "")).strip()
            owner_label = owner_kind if not owner_subagent_id else f"{owner_kind}:{owner_subagent_id}"
            lines.append(f"- {job_id} [owner={owner_label}] command={command}")
            job_ids.append(job_id)
        lines.extend(
            [
                "",
                "All listed jobs were cancelled and finalized before /stop completed.",
            ]
        )
        self._main_loop.append_system_note(
            "\n".join(lines),
            session_id=session_id,
            metadata={
                "user_stop_bash_jobs": True,
                "bash_job_ids": job_ids,
            },
        )

    def _append_user_stop_hard_quiesce_note(
        self,
        *,
        bash_reset: BashJobResetResult,
        affected_subagents: Sequence[SubagentSnapshot],
    ) -> None:
        session_id = self._main_loop.active_session_id()
        if session_id is None:
            return
        self._main_loop.append_system_note(
            "The user issued /stop. Jarvis hard-stopped the active main turn, "
            "hard-paused route-owned subagents, terminated detached jobs and services, "
            "and invalidated queued automatic follow-ups. The current session remains "
            "available for an explicit user resume.",
            session_id=session_id,
            metadata={
                "user_stop_hard_quiesce": True,
                "subagent_ids": [item.subagent_id for item in affected_subagents],
                "finalized_bash_job_ids": list(bash_reset.finalized_job_ids),
                "cancelled_bash_job_ids": list(
                    bash_reset.cancellation_requested_job_ids
                ),
            },
        )

    def _internal_followups_allowed(self) -> bool:
        return not (
            self._main_resume_requires_user_message
            or self._new_session_boundary_pending
            or self._subagent_reset_in_progress
        )

    def _internal_request_is_blocked(self, request: _RouteTurnRequest) -> bool:
        generally_blocked = (
            request.user_initiated
            or self._main_resume_requires_user_message
            or self._new_session_boundary_pending
            or request.internal_generation != self._internal_followup_generation
        )
        if generally_blocked:
            return True
        if request.runtime_turn_kind != _MAIN_ORCHESTRATOR_REVIEW_RUNTIME_KIND:
            return False
        return (
            request.orchestrator_wait_generation != self._orchestrator_wait_generation
            or not self._route_has_pending_background_work()
        )

    def _main_loop_has_active_turn(self) -> bool:
        return self._main_loop.has_active_turn()

    def _subagent_manager_turn_active(self, subagent_id: str) -> bool:
        return self._subagent_manager.is_turn_active(subagent_id)

    async def _enqueue_main_bash_job_followup(
        self,
        notices: tuple[BashJobNotice, ...],
    ) -> bool:
        if not notices:
            return False
        if self._main_resume_requires_user_message or self._new_session_boundary_pending:
            return False
        if self._recommend_main_bash_action(notices) != "wait":
            self._cancel_orchestrator_wait(reset_backoff=True)
        elif notices:
            self._orchestrator_unchanged_waits = 0
        self._merge_main_bash_notices(notices)
        if self._main_bash_runtime_turn_queued:
            return True
        self._main_bash_runtime_turn_queued = True
        await self._message_queue.put(
            _RouteTurnRequest(
                user_text=None,
                force_session_id=self._resolve_main_bash_notice_session_id(notices),
                parse_commands=False,
                user_initiated=False,
                internal_generation=self._internal_followup_generation,
                runtime_turn_kind=_MAIN_BASH_PROGRESS_RUNTIME_KIND,
            )
        )
        self._queue_wakeup.set()
        self._ensure_message_worker()
        return True

    async def _enqueue_subagent_bash_job_followup(
        self,
        notices: tuple[BashJobNotice, ...],
    ) -> bool:
        if self._subagent_reset_in_progress:
            return False
        if self._main_resume_requires_user_message or self._new_session_boundary_pending:
            return False
        return await self._subagent_manager.enqueue_bash_job_followup(notices)

    async def _enqueue_main_subagent_followup(
        self,
        notice: RouteSystemNoticeEvent,
    ) -> None:
        if self._subagent_reset_in_progress:
            return
        if self._main_resume_requires_user_message or self._new_session_boundary_pending:
            return
        self._merge_main_subagent_notice(notice)
        if self._main_subagent_runtime_turn_queued:
            return
        self._main_subagent_runtime_turn_queued = True
        await self._message_queue.put(
            _RouteTurnRequest(
                user_text=None,
                force_session_id=self._resolve_main_subagent_notice_session_id(notice),
                parse_commands=False,
                user_initiated=False,
                internal_generation=self._internal_followup_generation,
                runtime_turn_kind=_MAIN_SUBAGENT_PROGRESS_RUNTIME_KIND,
            )
        )
        self._queue_wakeup.set()
        self._ensure_message_worker()

    def _merge_main_bash_notices(self, notices: Sequence[BashJobNotice]) -> None:
        for notice in notices:
            self._pending_main_bash_notices.pop(notice.job_id, None)
            self._pending_main_bash_notices[notice.job_id] = notice

    async def _prepare_new_session_request(self) -> None:
        previous_session_id = self._main_loop.active_session_id()
        self._subagent_reset_in_progress = True
        self._invalidate_stale_internal_followups()
        self._clear_pending_main_bash_notices()
        self._clear_pending_main_subagent_notices()
        bash_reset = BashJobResetResult()
        subagent_reset: dict[str, Any] = {}
        try:
            # Cancel detached jobs before disposing children. Their writes must be included
            # in the child's final lease-segment snapshot and must not continue after the
            # child's lease is released for the replacement session.
            bash_reset = await self._bash_job_supervisor.terminate_route_jobs_for_new_session()
            subagent_reset = await self._subagent_manager.reset_for_new_session()
            self._append_new_session_reset_note(
                previous_session_id=previous_session_id,
                bash_reset=bash_reset,
                subagent_reset=subagent_reset,
            )
        finally:
            self._subagent_reset_in_progress = False
            self._clear_pending_main_bash_notices()
            self._clear_pending_main_subagent_notices()

    def _append_new_session_reset_note(
        self,
        *,
        previous_session_id: str | None,
        bash_reset: BashJobResetResult,
        subagent_reset: Mapping[str, Any],
    ) -> None:
        if previous_session_id is None:
            return
        disposed_subagent_ids = tuple(
            str(value)
            for value in subagent_reset.get("disposed_subagent_ids", ())
            if str(value).strip()
        )
        finalized_job_ids = bash_reset.finalized_job_ids
        cancellation_requested_job_ids = bash_reset.cancellation_requested_job_ids
        lines = [
            "The user issued /new. Jarvis hard-stopped work owned by this route before creating "
            "the replacement session.",
            "No queued or automatic follow-up from this session may resume in the replacement "
            "session.",
            f"Disposed subagents: {', '.join(disposed_subagent_ids) or 'none'}.",
            f"Finalized detached bash jobs: {', '.join(finalized_job_ids) or 'none'}.",
            (
                "Cancellation requested for detached bash jobs: "
                f"{', '.join(cancellation_requested_job_ids) or 'none'}."
            ),
        ]
        appended = self._main_loop.append_system_note(
            "\n".join(lines),
            session_id=previous_session_id,
            metadata={
                "new_session_hard_reset": True,
                "disposed_subagent_ids": list(disposed_subagent_ids),
                "finalized_bash_job_ids": list(finalized_job_ids),
                "cancellation_requested_bash_job_ids": list(
                    cancellation_requested_job_ids
                ),
            },
        )
        if not appended:
            raise RuntimeError(
                "Could not persist the /new hard-reset trace to the previous session."
            )

    def _clear_pending_main_bash_notices(self) -> None:
        self._pending_main_bash_notices.clear()
        self._main_bash_runtime_turn_queued = False

    def _merge_main_subagent_notice(self, notice: RouteSystemNoticeEvent) -> None:
        subagent_id = notice.subagent_id or ""
        if not subagent_id:
            return
        if notice.actor_run_generation is not None:
            snapshot = self._subagent_manager.snapshot_for(subagent_id)
            if snapshot is None or snapshot.run_generation != notice.actor_run_generation:
                LOGGER.debug(
                    "Ignored stale queued subagent notice route=%s subagent=%s event_generation=%s.",
                    self._route_id,
                    subagent_id,
                    notice.actor_run_generation,
                )
                return
        self._pending_main_subagent_notices[subagent_id] = notice

    def _clear_pending_main_subagent_notices(self) -> None:
        self._pending_main_subagent_notices.clear()
        self._main_subagent_runtime_turn_queued = False

    def _should_suppress_event_during_subagent_reset(self, event: RouteEvent) -> bool:
        if not self._subagent_reset_in_progress:
            return False
        return event.agent_kind == "subagent"

    def _resolve_main_bash_notice_session_id(
        self,
        notices: Sequence[BashJobNotice],
    ) -> str | None:
        for notice in notices:
            if notice.owner_session_id:
                return self._resolve_session_lineage(notice.owner_session_id)
        return self._main_loop.active_session_id()

    def _resolve_main_subagent_notice_session_id(
        self,
        notice: RouteSystemNoticeEvent,
    ) -> str | None:
        if notice.origin_session_id:
            return self._resolve_session_lineage(notice.origin_session_id)
        if notice.subagent_id:
            snapshot = self._subagent_manager.snapshot_for(notice.subagent_id)
            if snapshot is not None and snapshot.owner_main_session_id:
                return self._resolve_session_lineage(snapshot.owner_main_session_id)
        return self._main_loop.active_session_id()

    def _resolve_session_lineage(self, owner_session_id: str) -> str | None:
        """Map only an explicit compaction descendant to the active session."""

        owner = owner_session_id.strip()
        if not owner:
            return None
        active_session_id = self._main_loop.active_session_id()
        if active_session_id is None or active_session_id == owner:
            return owner
        cursor = self._main_storage.get_session(active_session_id)
        while cursor is not None and cursor.parent_session_id is not None:
            if cursor.parent_session_id == owner:
                return active_session_id
            cursor = self._main_storage.get_session(cursor.parent_session_id)
        return owner

    def _drain_main_bash_progress_message(
        self,
        *,
        force_session_id: str | None,
    ) -> tuple[str, AgentRuntimeMessage, tuple[BashJobNotice, ...]] | None:
        notices = tuple(self._pending_main_bash_notices.values())
        self._pending_main_bash_notices.clear()
        if not notices:
            return None
        session_id = force_session_id or self._resolve_main_bash_notice_session_id(notices)
        if session_id is None:
            return None
        return session_id, self._build_main_bash_job_followup_message(notices), notices

    def _drain_main_subagent_progress_message(
        self,
        *,
        force_session_id: str | None,
    ) -> tuple[str, AgentRuntimeMessage, tuple[RouteSystemNoticeEvent, ...]] | None:
        notices = tuple(self._pending_main_subagent_notices.values())
        self._pending_main_subagent_notices.clear()
        if not notices:
            return None
        session_id = force_session_id
        if session_id is None:
            for notice in notices:
                session_id = self._resolve_main_subagent_notice_session_id(notice)
                if session_id is not None:
                    break
        if session_id is None:
            return None
        message = self._build_main_subagent_followup_message(notices)
        if message is None:
            return None
        return session_id, message, notices

    async def _publish_main_system_message(
        self,
        *,
        session_id: str,
        message: AgentRuntimeMessage,
        notices: Sequence[BashJobNotice],
    ) -> bool:
        if not self._main_loop.append_system_note(
            message.content,
            session_id=session_id,
            metadata=message.metadata,
        ):
            return False
        notice_kind = str(
            message.metadata.get("notice_kind", _MAIN_BASH_PROGRESS_NOTICE_KIND)
        ).strip() or _MAIN_BASH_PROGRESS_NOTICE_KIND
        await self.publish_event(
            RouteSystemNoticeEvent(
                route_id=self._route_id,
                agent_kind="main",
                agent_name="Jarvis",
                session_id=session_id,
                notice_kind=notice_kind,
                text=message.content,
                public=False,
            )
        )
        self._record_bash_notice_delivery(notices)
        return True

    async def _publish_main_subagent_system_message(
        self,
        *,
        session_id: str,
        message: AgentRuntimeMessage,
        notices: Sequence[RouteSystemNoticeEvent],
    ) -> bool:
        if not self._main_loop.append_system_note(
            message.content,
            session_id=session_id,
            metadata=message.metadata,
        ):
            return False
        notice_kind = str(
            message.metadata.get("notice_kind", _MAIN_SUBAGENT_PROGRESS_NOTICE_KIND)
        ).strip() or _MAIN_SUBAGENT_PROGRESS_NOTICE_KIND
        await self.publish_event(
            RouteSystemNoticeEvent(
                route_id=self._route_id,
                agent_kind="main",
                agent_name="Jarvis",
                session_id=session_id,
                notice_kind=notice_kind,
                text=message.content,
                public=False,
            )
        )
        return True

    def _build_main_bash_job_followup_message(
        self,
        notices: Sequence[BashJobNotice],
    ) -> AgentRuntimeMessage:
        running_notices = [notice for notice in notices if notice.status == "running"]
        terminal_notices = [notice for notice in notices if notice.status != "running"]
        recommendation = self._recommend_main_bash_action(notices)
        lines = ["Detached bash update."]
        for notice in notices:
            lines.append(f"- {self._format_main_bash_job_notice_line(notice)}")
            if notice.status != "running":
                lines.extend(self._format_terminal_bash_evidence(notice))
            if notice.skill_import_notice:
                lines.extend(notice.skill_import_notice.splitlines())
        lines.append(f"recommendation={recommendation}")
        guidance = (
            "This is a system update from the orchestrator, not a new user message. Detached bash is orchestrator-monitored; react to this update and update the user accordingly instead of polling unless the user asks for immediate inspection."
        )
        if running_notices:
            lines.append(guidance)
            lines.append("Do not close the overall task while any listed job is still running.")
        elif terminal_notices:
            lines.append(guidance)
        return AgentRuntimeMessage(
            role="system",
            metadata={
                "bash_job_progress_update": True,
                "notice_kind": _MAIN_BASH_PROGRESS_NOTICE_KIND,
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

    def _build_main_subagent_followup_message(
        self,
        notices: Sequence[RouteSystemNoticeEvent],
    ) -> AgentRuntimeMessage | None:
        lines = ["Subagent update."]
        pending_subagent_ids: list[str] = []
        recommendations: list[str] = []
        subagent_updates: list[dict[str, Any]] = []
        changed_paths: set[str] = set()
        changed_test_artifact_paths: set[str] = set()
        for notice in notices:
            if notice.subagent_id is None:
                continue
            payload = self._subagent_manager.build_main_progress_message(
                agent=notice.subagent_id,
                notice_kind=notice.notice_kind,
                notice_text=notice.text,
            )
            if payload is None:
                continue
            _session_id, message = payload
            lines.extend(message.content.splitlines()[1:])
            for pending_subagent_id in message.metadata.get("pending_subagent_ids", []):
                normalized = str(pending_subagent_id).strip()
                if normalized and normalized not in pending_subagent_ids:
                    pending_subagent_ids.append(normalized)
            recommendation = str(message.metadata.get("recommended_action", "")).strip()
            if recommendation:
                recommendations.append(recommendation)
            update = {
                "subagent_id": str(message.metadata.get("subagent_id", "")).strip(),
                "status": str(message.metadata.get("subagent_status", "")).strip(),
                "report_complete": bool(
                    message.metadata.get("latest_subagent_report_complete", False)
                ),
                "changed_test_artifact_paths": list(
                    message.metadata.get("changed_test_artifact_paths", [])
                ),
                "changed_paths": list(message.metadata.get("changed_paths", [])),
                "changed_paths_complete": bool(
                    message.metadata.get("changed_paths_complete", False)
                ),
                "changed_paths_source": str(
                    message.metadata.get("changed_paths_source", "tool_result_metadata")
                ),
            }
            if update["subagent_id"]:
                subagent_updates.append(update)
            changed_test_artifact_paths.update(
                str(path).strip()
                for path in update["changed_test_artifact_paths"]
                if str(path).strip()
            )
            changed_paths.update(
                str(path).strip()
                for path in update["changed_paths"]
                if str(path).strip()
            )
        if len(lines) == 1:
            return None
        aggregated_recommendation = self._aggregate_recommendations(recommendations)
        return AgentRuntimeMessage(
            role="system",
            metadata={
                "subagent_progress_update": True,
                "notice_kind": _MAIN_SUBAGENT_PROGRESS_NOTICE_KIND,
                "recommended_action": aggregated_recommendation,
                "pending_subagent_ids": pending_subagent_ids,
                "subagents": subagent_updates,
                "changed_test_artifact_paths": sorted(
                    changed_test_artifact_paths
                ),
                "changed_paths": sorted(changed_paths),
                "changed_paths_complete": bool(
                    subagent_updates
                    and all(
                        bool(update["changed_paths_complete"])
                        for update in subagent_updates
                    )
                ),
            },
            content="\n".join(lines),
        )

    def _recommend_main_bash_action(self, notices: Sequence[BashJobNotice]) -> str:
        if any(
            notice.notice_kind in {"bash_job_failed", "bash_job_cancelled", "bash_job_needs_attention"}
            for notice in notices
        ):
            return "inspect"
        if any(notice.status == "running" for notice in notices):
            return "wait"
        return "finalize"

    def _aggregate_recommendations(self, recommendations: Sequence[str]) -> str:
        if any(recommendation == "inspect" for recommendation in recommendations):
            return "inspect"
        if any(recommendation == "finalize" for recommendation in recommendations):
            return "finalize"
        return "wait"

    def _format_main_bash_job_notice_line(self, notice: BashJobNotice) -> str:
        notice_name = notice.notice_kind.removeprefix("bash_job_") or notice.notice_kind
        timestamp_label, timestamp_value = self._main_bash_notice_timestamp(notice)
        parts = [
            f"job_id={notice.job_id}",
            f"status={notice.status}",
            f"notice={notice_name}",
            f"{timestamp_label}={timestamp_value}",
        ]
        if notice.status != "cancelled" and notice.exit_code is not None:
            parts.append(f"exit_code={notice.exit_code}")
        detail = self._main_bash_notice_detail(notice)
        if detail is not None:
            detail_label = "progress" if notice.status == "running" else "result"
            parts.append(f'{detail_label}="{detail}"')
        return " ".join(parts)

    def _main_bash_notice_timestamp(self, notice: BashJobNotice) -> tuple[str, str]:
        if notice.status == "cancelled":
            return "cancelled_at", notice.cancelled_at or notice.last_update_at or notice.started_at
        if notice.status != "running":
            return "finished_at", notice.finished_at or notice.last_update_at or notice.started_at
        if notice.last_update_at is not None:
            return "last_update_at", notice.last_update_at
        return "started_at", notice.started_at

    def _main_bash_notice_detail(self, notice: BashJobNotice) -> str | None:
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

    def _format_terminal_bash_evidence(self, notice: BashJobNotice) -> list[str]:
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

    def _truncate_for_notice(self, value: str | None, *, max_length: int) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            return None
        if len(normalized) <= max_length:
            return normalized
        return normalized[: max_length - 3] + "..."

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

    def _record_bash_notice_delivery(self, notices: Sequence[BashJobNotice]) -> None:
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


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


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


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _route_event_actor_id(event: RouteEvent) -> str:
    if event.agent_kind == "subagent" and event.subagent_id:
        return f"subagent:{event.subagent_id}"
    return "main"


def _provider_error_code(exc: Exception) -> str:
    if isinstance(exc, ProviderTimeoutError):
        return "provider_timeout"
    if isinstance(exc, ProviderRateLimitError):
        return "provider_rate_limited"
    return "provider_temporary_error"


def _bounded_terminal_tail(value: str, *, limit: int = 2_000) -> str:
    if not value.strip():
        return "  (empty)"
    if len(value) <= limit:
        return "  " + value
    head = limit // 2
    tail = limit - head
    return "  " + value[:head] + "\n  ...[terminal tail truncated]...\n  " + value[-tail:]


def _workspace_lease_error_result(
    tool_call: ToolCall,
    error: WorkspaceLeaseError,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        call_id=tool_call.call_id,
        name=tool_call.name,
        ok=False,
        content=(
            "Tool execution denied\n"
            f"tool: {tool_call.name}\n"
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
            "arguments": dict(tool_call.arguments),
        },
    )


def _with_workspace_lease_generation(
    result: ToolExecutionResult,
    generation: int,
) -> ToolExecutionResult:
    metadata = dict(result.metadata)
    metadata["workspace_lease_generation"] = generation
    return replace(result, metadata=metadata)


def _is_wait_only_bash_call(tool_call: ToolCall) -> bool:
    mode = str(tool_call.arguments.get("mode", "foreground")).strip().lower()
    if mode not in {"background", "service"}:
        return False
    command = str(tool_call.arguments.get("command", ""))
    return bool(_WAIT_ONLY_BASH_PATTERN.match(command))


def _optional_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Expected an array of strings.")
    normalized = tuple(str(item).strip() for item in value)
    if any(not item for item in normalized):
        raise ValueError("Array values must be non-empty strings.")
    return normalized


def _exception_metadata(exc: Exception) -> dict[str, Any]:
    metadata = getattr(exc, "metadata", None)
    if not isinstance(metadata, Mapping):
        return {}
    return json.loads(json.dumps(dict(metadata), ensure_ascii=False, default=str))


def _tool_result_for_payload(
    *,
    call_id: str,
    name: str,
    title: str,
    payload: dict[str, Any],
) -> ToolExecutionResult:
    metadata = dict(payload)
    metadata["subagent_control"] = True
    metadata["subagent_action"] = name.removeprefix("subagent_")
    return ToolExecutionResult(
        call_id=call_id,
        name=name,
        ok=True,
        content=title + "\n" + _format_payload_lines(payload),
        metadata=metadata,
    )


def _format_payload_lines(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple)):
            lines.append(f"{key}:")
            lines.append(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _map_route_event_to_agent_event(event: RouteEvent) -> AgentTurnStreamEvent | None:
    if isinstance(event, RouteTurnStartedEvent):
        if event.turn_id is None:
            return None
        return AgentTurnStartedEvent(
            session_id=event.session_id or "",
            turn_id=event.turn_id,
        )
    if isinstance(event, RouteAssistantDeltaEvent):
        return AgentTextDeltaEvent(
            session_id=event.session_id or "",
            delta=event.delta,
            turn_id=event.turn_id or "",
        )
    if isinstance(event, RouteAssistantMessageEvent):
        return AgentAssistantMessageEvent(
            session_id=event.session_id or "",
            text=event.text,
            turn_id=event.turn_id or "",
        )
    if isinstance(event, RouteToolCallEvent):
        return AgentToolCallEvent(
            session_id=event.session_id or "",
            tool_names=event.tool_names,
            turn_id=event.turn_id or "",
        )
    if isinstance(event, RouteApprovalRequestEvent):
        return AgentApprovalRequestEvent(
            session_id=event.session_id or "",
            turn_id=event.turn_id or "",
            approval_id=event.approval_id,
            kind=event.kind,
            summary=event.summary,
            details=event.details,
            command=event.command,
            tool_name=event.tool_name,
            inspection_url=event.inspection_url,
        )
    if isinstance(event, RouteTurnDoneEvent):
        return AgentTurnDoneEvent(
            session_id=event.session_id or "",
            response_text=event.response_text,
            turn_id=event.turn_id or "",
            command=event.command,
            compaction_performed=event.compaction_performed,
            interrupted=event.interrupted,
            approval_rejected=event.approval_rejected,
            completion_blocked=event.completion_blocked,
            completion_block_reason=event.completion_block_reason,
            interruption_reason=event.interruption_reason,
        )
    return None
