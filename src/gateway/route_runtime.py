"""Route-scoped supervisor runtime for the main loop and its subagents."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from core import (
    AgentApprovalRequestEvent,
    AgentAssistantMessageEvent,
    AgentIdentity,
    AgentLoop,
    AgentRuntimeMessage,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentTurnDoneEvent,
    AgentTurnResult,
    AgentTurnStreamEvent,
    ContextBudgetError,
    CoreSettings,
)
from core.identities import IdentityBootstrapLoader
from llm import LLMMessage, LLMService, ProviderTimeoutError, ToolCall, ToolDefinition
from subagent import (
    SUBAGENT_PRIMITIVE_NAMES,
    SubagentManager,
    build_subagent_primitive_definitions,
    render_subagent_primitive_docs,
)
from subagent.types import SubagentSnapshot
from tools import ToolExecutionContext, ToolExecutionResult, ToolRegistry, ToolRuntime, ToolSettings

from .route_events import (
    RouteApprovalRequestEvent,
    RouteAssistantDeltaEvent,
    RouteAssistantMessageEvent,
    RouteErrorEvent,
    RouteEvent,
    RouteSystemNoticeEvent,
    RouteToolCallEvent,
    RouteTurnDoneEvent,
)

_INTERNAL_ERROR_MESSAGE = "Internal error while processing message."
_PROVIDER_TIMEOUT_MESSAGE = "The model timed out while processing that message."
_SUBAGENT_TERMINAL_NOTICE_KINDS = frozenset(
    {"subagent_completed", "subagent_failed", "subagent_approval_rejected"}
)
_SUBAGENT_USER_STOP_NOTE_HEADER = (
    "The user issued /stop. Any active background subagents were also asked to stop "
    "cooperatively."
)
_SUBAGENT_SUPERVISOR_FOLLOWUP_TEXT = (
    "Internal runtime follow-up: a background subagent reached a terminal state. "
    "Review the runtime context, supervise the outcome, dispose the subagent when appropriate, "
    "and tell the user the result."
)


@dataclass(slots=True, frozen=True)
class _RouteTurnRequest:
    user_text: str
    force_session_id: str | None = None
    pre_turn_messages: tuple[AgentRuntimeMessage, ...] = ()
    parse_commands: bool = True
    user_initiated: bool = True
    internal_generation: int | None = None


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
                "developer",
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
        self._message_queue: asyncio.Queue[_RouteTurnRequest] = asyncio.Queue()
        self._message_worker: asyncio.Task[None] | None = None
        self._main_resume_requires_user_message = False
        self._internal_followup_generation = 0
        self._main_registry = self._tool_registry.filtered_view(agent_kind="main")
        self._main_tool_runtime = ToolRuntime(registry=self._main_registry)
        self._subagent_manager = SubagentManager(
            route_id=route_id,
            llm_service=llm_service,
            core_settings=core_settings,
            tool_registry=self._tool_registry,
            tool_execution_guard=self._tool_execution_guard,
            publish_event=self.publish_event,
            register_approval_target=self._approval_registry.register,
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
        )

    def active_session_id(self) -> str | None:
        return self._main_loop.active_session_id()

    def request_stop(self) -> bool:
        main_stop_requested = self._main_loop.request_stop()
        affected_subagents = self._subagent_manager.request_stop_all_for_user_stop()
        stop_requested = main_stop_requested or bool(affected_subagents)
        if stop_requested and not self._main_resume_requires_user_message:
            self._main_resume_requires_user_message = True
            self._internal_followup_generation += 1
        if affected_subagents:
            self._append_user_stop_subagent_note(affected_subagents)
        return stop_requested

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        return self._approval_registry.resolve(approval_id, approved)

    async def enqueue_user_message(self, user_text: str) -> None:
        await self._message_queue.put(
            _RouteTurnRequest(
                user_text=user_text,
                parse_commands=True,
                user_initiated=True,
            )
        )
        self._ensure_message_worker()

    def subscribe(self) -> tuple[str, asyncio.Queue[RouteEvent]]:
        return self._event_bus.subscribe()

    def unsubscribe(self, subscriber_id: str) -> None:
        self._event_bus.unsubscribe(subscriber_id)

    async def publish_event(self, event: RouteEvent) -> None:
        await self._event_bus.publish(event)
        await self._maybe_enqueue_subagent_supervisor_followup(event)

    async def stream_turn(
        self,
        user_text: str,
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        subscriber_id, queue = self.subscribe()
        await self.enqueue_user_message(user_text)
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

    async def _message_worker_loop(self) -> None:
        while True:
            request = await self._message_queue.get()
            try:
                if request.user_initiated:
                    if self._main_resume_requires_user_message:
                        self._main_resume_requires_user_message = False
                else:
                    if self._main_resume_requires_user_message:
                        continue
                    if request.internal_generation != self._internal_followup_generation:
                        continue
                event_stream = (
                    self._main_loop.stream_user_input(request.user_text)
                    if request.parse_commands
                    else self._main_loop.stream_turn(
                        user_text=request.user_text,
                        force_session_id=request.force_session_id,
                        pre_turn_messages=request.pre_turn_messages,
                    )
                )
                async for event in event_stream:
                    await self._publish_main_loop_event(event)
            except ContextBudgetError as exc:
                await self.publish_event(
                    RouteErrorEvent(
                        route_id=self._route_id,
                        agent_kind="main",
                        agent_name="Jarvis",
                        session_id=self._main_loop.active_session_id(),
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
                        code="provider_timeout",
                        message=_PROVIDER_TIMEOUT_MESSAGE,
                    )
                )
            except Exception:
                await self.publish_event(
                    RouteErrorEvent(
                        route_id=self._route_id,
                        agent_kind="main",
                        agent_name="Jarvis",
                        session_id=self._main_loop.active_session_id(),
                        code="internal_error",
                        message=_INTERNAL_ERROR_MESSAGE,
                    )
                )
            finally:
                self._message_queue.task_done()

    async def _maybe_enqueue_subagent_supervisor_followup(self, event: RouteEvent) -> None:
        if not isinstance(event, RouteSystemNoticeEvent):
            return
        if event.agent_kind != "subagent" or event.subagent_id is None:
            return
        if event.notice_kind not in _SUBAGENT_TERMINAL_NOTICE_KINDS:
            return
        if self._main_resume_requires_user_message:
            return
        await self._message_queue.put(
            _RouteTurnRequest(
                user_text=_SUBAGENT_SUPERVISOR_FOLLOWUP_TEXT,
                force_session_id=self._main_loop.active_session_id(),
                pre_turn_messages=self._subagent_manager.main_followup_runtime_messages(
                    agent=event.subagent_id,
                    notice_kind=event.notice_kind,
                    notice_text=event.text,
                ),
                parse_commands=False,
                user_initiated=False,
                internal_generation=self._internal_followup_generation,
            )
        )
        self._ensure_message_worker()

    async def _publish_main_loop_event(self, event: AgentTurnStreamEvent) -> None:
        if isinstance(event, AgentTextDeltaEvent):
            await self.publish_event(
                RouteAssistantDeltaEvent(
                    route_id=self._route_id,
                    agent_kind="main",
                    agent_name="Jarvis",
                    session_id=event.session_id,
                    delta=event.delta,
                )
            )
            return
        if isinstance(event, AgentAssistantMessageEvent):
            await self.publish_event(
                RouteAssistantMessageEvent(
                    route_id=self._route_id,
                    agent_kind="main",
                    agent_name="Jarvis",
                    session_id=event.session_id,
                    text=event.text,
                )
            )
            return
        if isinstance(event, AgentToolCallEvent):
            await self.publish_event(
                RouteToolCallEvent(
                    route_id=self._route_id,
                    agent_kind="main",
                    agent_name="Jarvis",
                    session_id=event.session_id,
                    tool_names=event.tool_names,
                )
            )
            return
        if isinstance(event, AgentApprovalRequestEvent):
            self._approval_registry.register(event.approval_id, self._main_loop)
            await self.publish_event(
                RouteApprovalRequestEvent(
                    route_id=self._route_id,
                    agent_kind="main",
                    agent_name="Jarvis",
                    session_id=event.session_id,
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
                    route_id=self._route_id,
                    agent_kind="main",
                    agent_name="Jarvis",
                    session_id=event.session_id,
                    response_text=event.response_text,
                    command=event.command,
                    compaction_performed=event.compaction_performed,
                    interrupted=event.interrupted,
                    approval_rejected=event.approval_rejected,
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
            return await self._main_tool_runtime.execute(
                tool_call=tool_call,
                context=context,
            )

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
    if isinstance(event, RouteAssistantDeltaEvent):
        return AgentTextDeltaEvent(
            session_id=event.session_id or "",
            delta=event.delta,
        )
    if isinstance(event, RouteAssistantMessageEvent):
        return AgentAssistantMessageEvent(
            session_id=event.session_id or "",
            text=event.text,
        )
    if isinstance(event, RouteToolCallEvent):
        return AgentToolCallEvent(
            session_id=event.session_id or "",
            tool_names=event.tool_names,
        )
    if isinstance(event, RouteApprovalRequestEvent):
        return AgentApprovalRequestEvent(
            session_id=event.session_id or "",
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
            command=event.command,
            compaction_performed=event.compaction_performed,
            interrupted=event.interrupted,
            approval_rejected=event.approval_rejected,
        )
    return None
