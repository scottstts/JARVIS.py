"""Route-scoped supervisor runtime for the main loop and its subagents."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from jarvis.core import (
    AgentApprovalRequestEvent,
    AgentAssistantMessageEvent,
    AgentIdentity,
    AgentLoop,
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
from jarvis.llm import LLMMessage, LLMService, ProviderTimeoutError, ToolCall, ToolDefinition
from jarvis.logging_setup import get_application_logger
from jarvis.subagent import (
    SUBAGENT_PRIMITIVE_NAMES,
    SubagentManager,
    build_subagent_primitive_definitions,
    render_subagent_primitive_docs,
)
from jarvis.subagent.types import SubagentSnapshot
from jarvis.tools import ToolExecutionContext, ToolExecutionResult, ToolRegistry, ToolRuntime, ToolSettings
from jarvis.tools.basic.bash.jobs import (
    BashJobError,
    mark_job_progress_notified,
    mark_job_terminal_notice_dispatched,
)

from .bash_job_supervisor import BashJobNotice, BashJobSupervisor
from .route_events import (
    RouteApprovalRequestEvent,
    RouteAssistantDeltaEvent,
    RouteAssistantMessageEvent,
    RouteErrorEvent,
    RouteEvent,
    RouteLocalNoticeEvent,
    RouteSystemNoticeEvent,
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
    "The user issued /stop. Any active background subagents were also asked to stop "
    "cooperatively."
)
_MAIN_BASH_PROGRESS_RUNTIME_KIND = "main_bash_progress"
_MAIN_BASH_PROGRESS_NOTICE_KIND = "bash_job_progress_update"
_MAIN_SUBAGENT_PROGRESS_RUNTIME_KIND = "main_subagent_progress"
_MAIN_SUBAGENT_PROGRESS_NOTICE_KIND = "subagent_progress_update"
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
        self._targets: dict[str, AgentLoop] = {}

    def register(self, approval_id: str, loop: AgentLoop) -> None:
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
                "Subagent runtime control reference:\n\n" + render_subagent_primitive_docs(),
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
        self._tool_execution_guard = asyncio.Semaphore(1)
        self._event_bus = RouteEventBus()
        self._approval_registry = RouteApprovalRegistry()
        self._user_message_queue: asyncio.Queue[_RouteTurnRequest] = asyncio.Queue()
        self._message_queue: asyncio.Queue[_RouteTurnRequest] = asyncio.Queue()
        self._queue_wakeup = asyncio.Event()
        self._message_worker: asyncio.Task[None] | None = None
        self._active_turn_request: _RouteTurnRequest | None = None
        self._main_resume_requires_user_message = False
        self._internal_followup_generation = 0
        self._pending_main_bash_notices: dict[str, BashJobNotice] = {}
        self._main_bash_runtime_turn_queued = False
        self._pending_main_subagent_notices: dict[str, RouteSystemNoticeEvent] = {}
        self._main_subagent_runtime_turn_queued = False
        self._subagent_reset_in_progress = False
        self._main_registry = self._tool_registry.filtered_view(agent_kind="main")
        self._main_tool_runtime = ToolRuntime(registry=self._main_registry)
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
            publish_event=self.publish_event,
            register_approval_target=self._approval_registry.register,
            tool_result_observer=self._bash_job_supervisor.observe_tool_result,
        )
        self._main_loop = AgentLoop(
            llm_service=llm_service,
            settings=core_settings,
            route_id=route_id,
            tool_registry=self._main_registry,
            tool_runtime=self._main_tool_runtime,
            bootstrap_loader=CompositeMainBootstrapLoader(core_settings),
            identity=AgentIdentity(kind="main", name="Jarvis"),
            tool_definitions_provider=self._build_main_tool_definitions,
            tool_executor=self._execute_main_tool_call,
            runtime_messages_provider=lambda _session_id: self._subagent_manager.main_turn_runtime_messages(),
            local_notice_callback=self._publish_main_local_notice,
        )

    def active_session_id(self) -> str | None:
        return self._main_loop.active_session_id()

    def request_stop(self) -> bool:
        main_stop_requested = self._main_loop.request_stop(reason="user_stop")
        affected_subagents = self._subagent_manager.request_stop_all_for_user_stop()
        pending_bash_jobs = self._bash_job_supervisor.pending_jobs()
        stop_requested = (
            main_stop_requested
            or bool(affected_subagents)
            or bool(pending_bash_jobs)
        )
        if stop_requested and not self._main_resume_requires_user_message:
            self._main_resume_requires_user_message = True
            self._invalidate_stale_internal_followups()
        if affected_subagents:
            self._append_user_stop_subagent_note(affected_subagents)
        if pending_bash_jobs:
            self._append_user_stop_bash_job_note(pending_bash_jobs)
        return stop_requested

    def _request_user_message_supersede(self) -> None:
        _ = self._main_loop.request_stop(reason="superseded_by_user_message")
        self._subagent_manager.request_stop_all_for_superseded_user_message()
        self._invalidate_stale_internal_followups()

    def _request_new_session_supersede(self) -> None:
        _ = self._main_loop.request_stop(reason="superseded_by_user_message")
        self._subagent_manager.request_stop_all_for_user_stop()
        self._invalidate_stale_internal_followups()

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        return self._approval_registry.resolve(approval_id, approved)

    async def enqueue_user_message(
        self,
        user_text: str,
        *,
        client_message_id: str | None = None,
    ) -> None:
        self._bash_job_supervisor.ensure_running()
        command = parse_user_command(user_text)
        await self._user_message_queue.put(
            _RouteTurnRequest(
                user_text=user_text,
                parse_commands=True,
                user_initiated=True,
                client_message_id=client_message_id,
            )
        )
        self._queue_wakeup.set()
        if command.kind == "new":
            self._request_new_session_supersede()
        else:
            self._request_user_message_supersede()
        self._ensure_message_worker()

    def subscribe(self) -> tuple[str, asyncio.Queue[RouteEvent]]:
        return self._event_bus.subscribe()

    def unsubscribe(self, subscriber_id: str) -> None:
        self._event_bus.unsubscribe(subscriber_id)

    async def publish_event(self, event: RouteEvent) -> None:
        if self._should_suppress_event_during_subagent_reset(event):
            return
        await self._event_bus.publish(event)
        await self._maybe_enqueue_subagent_supervisor_followup(event)

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
            try:
                if request.user_initiated:
                    if self._main_resume_requires_user_message:
                        self._main_resume_requires_user_message = False
                else:
                    if self._main_resume_requires_user_message:
                        continue
                    if request.internal_generation != self._internal_followup_generation:
                        continue
                if request.runtime_turn_kind == _MAIN_BASH_PROGRESS_RUNTIME_KIND:
                    self._main_bash_runtime_turn_queued = False
                    runtime_message = self._drain_main_bash_progress_message(
                        force_session_id=request.force_session_id,
                    )
                    if runtime_message is None:
                        continue
                    force_session_id, system_message, notices = runtime_message
                    published = await self._publish_main_system_message(
                        session_id=force_session_id,
                        message=system_message,
                        notices=notices,
                    )
                    if not published:
                        continue
                    event_stream = self._main_loop.stream_runtime_turn(
                        force_session_id=force_session_id,
                        pre_turn_messages=self._build_wait_only_runtime_messages(system_message),
                    )
                elif request.runtime_turn_kind == _MAIN_SUBAGENT_PROGRESS_RUNTIME_KIND:
                    self._main_subagent_runtime_turn_queued = False
                    runtime_message = self._drain_main_subagent_progress_message(
                        force_session_id=request.force_session_id,
                    )
                    if runtime_message is None:
                        continue
                    force_session_id, system_message, notices = runtime_message
                    published = await self._publish_main_subagent_system_message(
                        session_id=force_session_id,
                        message=system_message,
                        notices=notices,
                    )
                    if not published:
                        continue
                    event_stream = self._main_loop.stream_runtime_turn(
                        force_session_id=force_session_id,
                        pre_turn_messages=self._build_wait_only_runtime_messages(system_message),
                    )
                elif request.parse_commands:
                    if request.user_text is None:
                        continue
                    if parsed_command is not None and parsed_command.kind == "new":
                        await self._prepare_new_session_request()
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
            except ProviderTimeoutError:
                await self.publish_event(
                    RouteErrorEvent(
                        route_id=self._route_id,
                        agent_kind="main",
                        agent_name="Jarvis",
                        session_id=self._main_loop.active_session_id(),
                        turn_id=self._main_loop.active_turn_id(),
                        turn_kind="user" if request.user_initiated else "runtime",
                        client_message_id=request.client_message_id,
                        code="provider_timeout",
                        message=_PROVIDER_TIMEOUT_MESSAGE,
                    )
                )
            except Exception:
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
                self._active_turn_request = None
                source_queue.task_done()
                await self._maybe_schedule_deferred_internal_followups()

    async def _maybe_enqueue_subagent_supervisor_followup(self, event: RouteEvent) -> None:
        if self._subagent_reset_in_progress:
            return
        if not isinstance(event, RouteSystemNoticeEvent):
            return
        if event.agent_kind != "subagent" or event.subagent_id is None:
            return
        if event.notice_kind not in _SUBAGENT_MAIN_PROGRESS_NOTICE_KINDS:
            return
        if self._main_resume_requires_user_message:
            return
        await self._enqueue_main_subagent_followup(event)

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
        }
        if isinstance(event, AgentTurnStartedEvent):
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
            await self.publish_event(
                RouteTurnDoneEvent(
                    **route_event_kwargs,
                    response_text=event.response_text,
                    command=event.command,
                    compaction_performed=event.compaction_performed,
                    interrupted=event.interrupted,
                    approval_rejected=event.approval_rejected,
                    interruption_reason=event.interruption_reason,
                )
            )

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
        async with self._tool_execution_guard:
            result = await self._main_tool_runtime.execute(
                tool_call=tool_call,
                context=context,
            )
        await self._bash_job_supervisor.observe_tool_result(result=result, context=context)
        return result

    async def _execute_subagent_primitive(
        self,
        tool_call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        try:
            payload: dict[str, Any]
            if tool_call.name == "subagent_invoke":
                instructions = str(tool_call.arguments.get("instructions", "")).strip()
                if not instructions:
                    raise ValueError("'instructions' is required.")
                session_id = context.session_id
                turn_id = context.turn_id
                if session_id is None or turn_id is None:
                    raise ValueError("Subagent invocation requires a main session and turn id.")
                payload = await self._subagent_manager.invoke(
                    requester_kind=context.agent_kind,
                    instructions=instructions,
                    context=_optional_string(tool_call.arguments.get("context")),
                    deliverable=_optional_string(tool_call.arguments.get("deliverable")),
                    owner_main_session_id=session_id,
                    owner_main_turn_id=turn_id,
                )
                return _tool_result_for_payload(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    title="Subagent invoked",
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
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                ok=False,
                content=(
                    "Subagent control failed\n"
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
                "Any tool execution already started by one of these subagents was allowed to finish and log its result in the child transcript before the child settles.",
                "When you resume, inspect current subagent status, then decide whether to resume it, step in, dispose it, or otherwise handle it so no paused child is left orphaned.",
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
                "The bash jobs continue running, but automatic runtime follow-ups are suppressed until the next user message.",
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

    def _internal_followups_allowed(self) -> bool:
        return not self._main_resume_requires_user_message

    def _main_loop_has_active_turn(self) -> bool:
        return self._main_loop.has_active_turn()

    def _subagent_manager_turn_active(self, subagent_id: str) -> bool:
        return self._subagent_manager.is_turn_active(subagent_id)

    async def _enqueue_main_bash_job_followup(
        self,
        notices: tuple[BashJobNotice, ...],
    ) -> None:
        if not notices:
            return
        if self._main_resume_requires_user_message:
            return
        self._merge_main_bash_notices(notices)
        if self._main_bash_runtime_turn_queued:
            return
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

    async def _enqueue_subagent_bash_job_followup(
        self,
        notices: tuple[BashJobNotice, ...],
    ) -> None:
        if self._subagent_reset_in_progress:
            return
        if self._main_resume_requires_user_message:
            return
        await self._subagent_manager.enqueue_bash_job_followup(notices)

    async def _enqueue_main_subagent_followup(
        self,
        notice: RouteSystemNoticeEvent,
    ) -> None:
        if self._subagent_reset_in_progress:
            return
        if self._main_resume_requires_user_message:
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
        self._subagent_reset_in_progress = True
        self._invalidate_stale_internal_followups()
        self._clear_pending_main_subagent_notices()
        try:
            await self._subagent_manager.reset_for_new_session()
        except Exception:
            LOGGER.exception(
                "Failed to reset subagents before starting a new session for route %s.",
                self._route_id,
            )
            raise
        finally:
            self._subagent_reset_in_progress = False
        self._clear_pending_main_subagent_notices()

    def _clear_pending_main_bash_notices(self) -> None:
        self._pending_main_bash_notices.clear()
        self._main_bash_runtime_turn_queued = False

    def _merge_main_subagent_notice(self, notice: RouteSystemNoticeEvent) -> None:
        subagent_id = notice.subagent_id or ""
        if not subagent_id:
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
        active_session_id = self._main_loop.active_session_id()
        if active_session_id is not None:
            return active_session_id
        for notice in notices:
            if notice.owner_session_id:
                return notice.owner_session_id
        return None

    def _resolve_main_subagent_notice_session_id(
        self,
        notice: RouteSystemNoticeEvent,
    ) -> str | None:
        active_session_id = self._main_loop.active_session_id()
        if active_session_id is not None:
            return active_session_id
        if notice.subagent_id:
            snapshot = self._subagent_manager.snapshot_for(notice.subagent_id)
            if snapshot is not None and snapshot.owner_main_session_id:
                return snapshot.owner_main_session_id
        return None

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
            },
            content="\n".join(lines),
        )

    def _build_wait_only_runtime_messages(
        self,
        message: AgentRuntimeMessage,
    ) -> tuple[AgentRuntimeMessage, ...]:
        if str(message.metadata.get("recommended_action", "")).strip() != "wait":
            return ()
        return (
            AgentRuntimeMessage(
                role="system",
                metadata={
                    "force_no_tools_this_turn": True,
                    "orchestrator_wait_only_update": True,
                },
                content=(
                    "This orchestrator progress update is wait-only. Do not call tools in this "
                    "response. Send a brief progress update and wait for the next orchestrator "
                    "system message unless the user explicitly asks for immediate inspection."
                ),
            ),
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
            interruption_reason=event.interruption_reason,
        )
    return None
