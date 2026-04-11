"""Actor runtime implementation backed by OpenAI Codex app-server."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from jarvis.core.agent_loop import (
    AgentApprovalRequestEvent,
    AgentAssistantMessageEvent,
    AgentIdentity,
    AgentMemoryMode,
    AgentRuntimeMessage,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentTurnDoneEvent,
    AgentTurnResult,
    AgentTurnStartedEvent,
    AgentTurnStreamEvent,
    InterruptionReason,
)
from jarvis.core.compaction import (
    CompactionOutcome,
    CompactionReplacementItem,
    ContextCompactor,
    prune_compaction_source_records,
)
from jarvis.core.commands import ParsedCommand, parse_user_command
from jarvis.llm import LLMUsage, LLMMessage, LLMService
from jarvis.logging_setup import get_application_logger
from jarvis.memory import MemoryService, MemorySettings
from jarvis.storage import ConversationRecord, SessionMetadata, SessionStorage
from jarvis.tools import ToolExecutionContext, ToolExecutionResult, ToolSettings

from .config import CodexBackendSettings
from .path_mapping import CodexPathMapper
from .runtime import CodexRouteCoordinator
from .tool_bridge import CodexToolBridge
from .types import (
    CodexAuthChallenge,
    CodexBackendError,
    CodexNativeCapabilityError,
    CodexProtocolError,
)

LOGGER = get_application_logger(__name__)

_TURN_ID_METADATA_KEY = "turn_id"
_ORPHANED_TURN_RECOVERY_METADATA_KEY = "orphaned_turn_recovery"
_TURN_ORPHANED_RECOVERY_RECORD_TEXT = (
    "This Codex-backed turn ended unexpectedly before it completed. Treat any partial "
    "assistant output above as incomplete."
)
_TRANSCRIPT_ONLY_RECORD_METADATA_KEY = "transcript_only"
_CODEX_BOOTSTRAP_METADATA_KEY = "codex_bootstrap"
_NATIVE_TOOL_RECOVERY_METADATA_KEY = "codex_native_tool_recovery"
_NATIVE_TOOL_RECOVERY_MAX_ATTEMPTS = 1
_APPROVAL_INTERRUPTED_TEXT = (
    "The turn was interrupted before the approval request was resolved."
)
_APPROVAL_REJECTED_TEXT = "Approval request was rejected. I did not execute the action."
_UNSUPPORTED_NATIVE_ITEM_TYPES = frozenset(
    {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "collabAgentToolCall",
    }
)
_NATIVE_TOOL_RECOVERY_NOTE_TEMPLATE = (
    "Jarvis interrupted a previous Codex attempt because it invoked the unsupported native "
    "Codex capability '{item_type}'. Continue the same task using only Jarvis dynamic tools in "
    "this thread. Use `bash` for shell work, `file_patch` or `bash` for file edits, and "
    "`subagent_*` tools for delegation. Do not use any native Codex command, file, MCP, or "
    "collaboration capability."
)


@dataclass(slots=True)
class _TurnState:
    session_id: str
    logical_turn_id: str | None = None
    provider_turn_id: str | None = None
    command: str | None = None
    user_text: str | None = None
    compaction_performed: bool = False
    active_assistant_item_id: str | None = None
    active_assistant_chunks: list[str] = field(default_factory=list)
    completed_messages: list[str] = field(default_factory=list)
    usage: LLMUsage | None = None
    approval_rejected: bool = False
    terminated: bool = False
    pending_orchestrator_yield: bool = False
    orchestrator_yield_reason: str | None = None
    native_recovery_attempts: int = 0
    pending_native_recovery_item_type: str | None = None
    provider_interrupt_task: asyncio.Task[object] | None = None
    queue: asyncio.Queue[object] = field(default_factory=asyncio.Queue)

    def full_text(self) -> str:
        if self.completed_messages:
            return "\n\n".join(
                text for text in self.completed_messages if text.strip()
            )
        if self.active_assistant_chunks:
            return "".join(self.active_assistant_chunks)
        return ""


class CodexActorRuntime:
    """Actor runtime with AgentLoop-compatible surface over Codex app-server."""

    def __init__(
        self,
        *,
        coordinator: CodexRouteCoordinator,
        settings: CodexBackendSettings,
        llm_service: LLMService,
        storage: SessionStorage,
        core_settings,
        route_id: str,
        identity: AgentIdentity,
        bootstrap_loader,
        memory_mode: AgentMemoryMode,
        tool_registry,
        tool_runtime,
        tool_definitions_provider,
        tool_executor,
        publish_route_event,
        runtime_messages_provider=None,
    ) -> None:
        self._coordinator = coordinator
        self._settings = settings
        self._llm_service = llm_service
        self._storage = storage
        self._core_settings = core_settings
        self._identity = identity
        self._bootstrap_loader = bootstrap_loader
        self._memory_mode = memory_mode
        self._tool_registry = tool_registry
        self._tool_runtime = tool_runtime
        self._tool_definitions_provider = tool_definitions_provider
        self._tool_executor = tool_executor
        self._runtime_messages_provider = runtime_messages_provider
        self._path_mapper = CodexPathMapper.from_settings(settings)
        self._tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
        memory_settings = MemorySettings.from_workspace_dir(core_settings.workspace_dir)
        memory_llm_service = llm_service if memory_mode.reflection else None
        if memory_llm_service is None:
            memory_settings = replace(memory_settings, enable_reflection=False)
        self._memory_service = MemoryService(
            settings=memory_settings,
            llm_service=memory_llm_service,
        )
        self._tool_context = ToolExecutionContext(
            workspace_dir=self._tool_settings.workspace_dir,
            route_id=route_id,
            agent_kind=self._identity.kind,
            agent_name=self._identity.name,
            subagent_id=self._identity.subagent_id,
            memory_service=(
                self._memory_service
                if any(
                    (
                        self._memory_mode.bootstrap,
                        self._memory_mode.maintenance,
                        self._memory_mode.reflection,
                    )
                )
                else None
            ),
        )
        self._tool_bridge = CodexToolBridge(
            tool_definitions_provider=tool_definitions_provider,
        )
        self._compactor = ContextCompactor(
            llm_service=self._llm_service,
            context_policy=self._core_settings.context_policy,
            provider=self._core_settings.compaction.provider,
        )
        self._active_turn_id: str | None = None
        self._requested_interruption: InterruptionReason | None = None
        self._pending_approval_future: asyncio.Future[bool] | None = None
        self._pending_approval_id: str | None = None
        self._current_turn: _TurnState | None = None
        self._loaded_thread_id: str | None = None
        self._loaded_session_id: str | None = None
        self._loaded_dynamic_tools_signature: str | None = None
        self._publish_route_event = publish_route_event

    @property
    def agent_kind(self) -> str:
        return self._identity.kind

    @property
    def agent_name(self) -> str:
        return self._identity.name

    @property
    def subagent_id(self) -> str | None:
        return self._identity.subagent_id

    async def handle_user_input(self, user_text: str) -> AgentTurnResult:
        result: AgentTurnResult | None = None
        async for event in self.stream_user_input(user_text):
            if isinstance(event, AgentTurnDoneEvent):
                result = event.to_result()
        if result is None:
            raise RuntimeError("Codex actor turn ended without a final done event.")
        return result

    async def stream_user_input(self, user_text: str) -> AsyncIterator[AgentTurnStreamEvent]:
        command = parse_user_command(user_text)
        if command.kind == "new":
            async for event in self._stream_new_command(command):
                yield event
            return
        if command.kind == "compact":
            async for event in self._stream_compact_command(command):
                yield event
            return
        async for event in self.stream_turn(user_text=command.body):
            yield event

    async def stream_turn(
        self,
        *,
        user_text: str,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        async for event in self._stream_message_turn(
            user_text=user_text,
            force_session_id=force_session_id,
            command_override=command_override,
            pre_turn_messages=pre_turn_messages,
        ):
            yield event

    async def stream_runtime_turn(
        self,
        *,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        async for event in self._stream_message_turn(
            user_text=None,
            force_session_id=force_session_id,
            command_override=command_override,
            pre_turn_messages=pre_turn_messages,
        ):
            yield event

    def active_session_id(self) -> str | None:
        active = self._storage.get_active_session()
        return active.session_id if active is not None else None

    def active_turn_id(self) -> str | None:
        return self._active_turn_id

    def has_active_turn(self) -> bool:
        return self._active_turn_id is not None

    async def aclose(self) -> None:
        pending_approval = self._pending_approval_future
        if pending_approval is not None and not pending_approval.done():
            pending_approval.cancel()
        loaded_thread_id = self._loaded_thread_id
        if loaded_thread_id is not None:
            self._coordinator.unregister_actor(thread_id=loaded_thread_id, actor=self)
        self._active_turn_id = None
        self._requested_interruption = None
        self._pending_approval_future = None
        self._pending_approval_id = None
        self._current_turn = None
        self._loaded_thread_id = None
        self._loaded_session_id = None
        self._loaded_dynamic_tools_signature = None

    def append_system_note(
        self,
        content: str,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        normalized_content = content.strip()
        if not normalized_content:
            return False
        target_session_id = session_id or self.active_session_id()
        if target_session_id is None:
            return False
        if self._storage.get_session(target_session_id) is None:
            return False
        self._reconcile_orphaned_turns(target_session_id)
        self._storage.append_record(
            target_session_id,
            ConversationRecord(
                record_id=uuid4().hex,
                session_id=target_session_id,
                created_at=_utc_now_iso(),
                role="system",
                content=normalized_content,
                metadata=dict(metadata or {}),
            ),
        )
        return True

    async def prepare_session(self, *, start_reason: str = "initial") -> str:
        active = self._storage.get_active_session()
        if active is not None:
            self._reconcile_orphaned_turns(active.session_id)
            return active.session_id
        session = self._storage.create_session(start_reason=start_reason)
        await self._persist_session_bootstrap(session.session_id)
        self._loaded_session_id = None
        self._loaded_thread_id = None
        self._loaded_dynamic_tools_signature = None
        return session.session_id

    def request_stop(
        self,
        *,
        reason: InterruptionReason = "user_stop",
    ) -> bool:
        active_turn_id = self._active_turn_id
        turn = self._current_turn
        if active_turn_id is None or turn is None:
            return False
        self._requested_interruption = reason
        pending_approval = self._pending_approval_future
        if pending_approval is not None and not pending_approval.done():
            pending_approval.cancel()
        provider_turn_id = turn.provider_turn_id
        if provider_turn_id is not None:
            turn.provider_interrupt_task = self._request_provider_turn_interrupt(
                provider_turn_id=provider_turn_id,
                reason=reason,
            )
        return True

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        normalized = approval_id.strip()
        pending_future = self._pending_approval_future
        if not normalized or pending_future is None or pending_future.done():
            return False
        if self._pending_approval_id != normalized:
            return False
        pending_future.set_result(bool(approved))
        return True

    async def handle_notification(self, method: str, params: dict[str, Any]) -> None:
        turn = self._current_turn
        if turn is None:
            return
        turn_id = _optional_string(params.get("turnId"))
        if turn_id is not None and turn.provider_turn_id is None:
            turn.provider_turn_id = turn_id
        if (
            turn_id is not None
            and turn.provider_turn_id is not None
            and turn_id != turn.provider_turn_id
        ):
            return

        if method == "item/agentMessage/delta":
            if (
                turn.pending_native_recovery_item_type is not None
                or turn.pending_orchestrator_yield
            ):
                return
            item_id = _optional_string(params.get("itemId"))
            if item_id is not None:
                self._rollover_active_assistant_item(turn, incoming_item_id=item_id)
                turn.active_assistant_item_id = item_id
            delta = str(params.get("delta", ""))
            if delta:
                turn.active_assistant_chunks.append(delta)
                await turn.queue.put(
                    AgentTextDeltaEvent(
                        session_id=turn.session_id,
                        turn_id=self._visible_turn_id(turn) or "",
                        delta=delta,
                    )
                )
            return

        if method == "item/completed":
            item = params.get("item")
            if not isinstance(item, dict):
                return
            item_type = _optional_string(item.get("type"))
            if item_type in _UNSUPPORTED_NATIVE_ITEM_TYPES:
                await self._handle_unsupported_native_capability(turn, item_type)
                return
            if item_type == "agentMessage":
                if turn.pending_orchestrator_yield:
                    return
                await self._complete_assistant_item(turn, item=item)
                return
            return

        if method == "item/started":
            item = params.get("item")
            if not isinstance(item, dict):
                return
            item_type = _optional_string(item.get("type"))
            if item_type in _UNSUPPORTED_NATIVE_ITEM_TYPES:
                await self._handle_unsupported_native_capability(turn, item_type)
            return

        if method == "thread/tokenUsage/updated":
            token_usage = params.get("tokenUsage")
            if not isinstance(token_usage, dict):
                return
            last = token_usage.get("last")
            if not isinstance(last, dict):
                return
            turn.usage = LLMUsage(
                input_tokens=_optional_int(last.get("inputTokens")),
                output_tokens=_optional_int(last.get("outputTokens")),
                total_tokens=_optional_int(last.get("totalTokens")),
            )
            return

        if method == "turn/completed":
            raw_turn = params.get("turn")
            if not isinstance(raw_turn, dict):
                return
            status = _optional_string(raw_turn.get("status")) or "failed"
            turn.provider_turn_id = _optional_string(raw_turn.get("id")) or turn.provider_turn_id
            if turn.provider_turn_id is None:
                await self._terminate_turn_with_error(
                    turn,
                    CodexProtocolError("Codex completed a turn without an id."),
                )
                return
            error = raw_turn.get("error")
            if (
                turn.pending_native_recovery_item_type is not None
                and self._requested_interruption is None
            ):
                await self._recover_from_native_tool_attempt(turn)
                return
            if (
                turn.pending_orchestrator_yield
                and self._requested_interruption is None
            ):
                if status == "failed" and not _is_expected_interrupt_error(error):
                    message = "Codex turn failed."
                    if isinstance(error, dict):
                        message = str(error.get("message") or message)
                        additional = _optional_string(error.get("additionalDetails"))
                        if additional is not None:
                            message = f"{message} {additional}"
                    await self._terminate_turn_with_error(turn, CodexBackendError(message))
                    return
                await self._complete_orchestrator_yield(turn)
                return
            response_text = turn.full_text()
            if response_text.strip() and not turn.completed_messages:
                assistant_event = AgentAssistantMessageEvent(
                    session_id=turn.session_id,
                    turn_id=self._visible_turn_id(turn),
                    text=response_text,
                )
                await turn.queue.put(assistant_event)
            interrupted = status == "interrupted"
            self._persist_turn_completion(
                turn,
                response_text=response_text,
                status=status,
            )
            if status == "failed":
                message = "Codex turn failed."
                if isinstance(error, dict):
                    message = str(error.get("message") or message)
                    additional = _optional_string(error.get("additionalDetails"))
                    if additional is not None:
                        message = f"{message} {additional}"
                await self._terminate_turn_with_error(turn, CodexBackendError(message))
                return
            await turn.queue.put(
                AgentTurnDoneEvent(
                    session_id=turn.session_id,
                    turn_id=self._visible_turn_id(turn),
                    response_text=response_text,
                    command=turn.command,
                    compaction_performed=turn.compaction_performed,
                    interrupted=interrupted,
                    approval_rejected=turn.approval_rejected and not interrupted,
                    interruption_reason=self._requested_interruption if interrupted else None,
                )
            )
            turn.terminated = True
            self._active_turn_id = None
            self._requested_interruption = None
            self._current_turn = None

    async def handle_server_request(self, method: str, params: dict[str, Any]) -> object:
        turn = self._current_turn
        if turn is None:
            raise CodexProtocolError("Codex requested a tool call without an active turn.")
        if turn.pending_orchestrator_yield and method == "item/tool/call":
            return self._tool_bridge.build_tool_rejection_response(
                "Jarvis has yielded control of this task back to the route orchestrator. "
                "Do not call more tools in this turn."
            )
        if method != "item/tool/call":
            await self._handle_unsupported_native_capability(turn, method)
            raise CodexNativeCapabilityError(
                f"Codex requested unsupported native capability '{method}'."
            )
        turn_id = _optional_string(params.get("turnId"))
        if turn.provider_turn_id is None:
            turn.provider_turn_id = turn_id
        if turn.provider_turn_id is None:
            raise CodexProtocolError("Codex tool request did not include a usable turn id.")
        call_id = _required_string(params.get("callId"), field_name="callId")
        tool_name = _required_string(params.get("tool"), field_name="tool")
        tool_call = self._tool_bridge.build_tool_call(
            call_id=call_id,
            tool_name=tool_name,
            arguments=params.get("arguments"),
        )
        await turn.queue.put(
            AgentToolCallEvent(
                session_id=turn.session_id,
                turn_id=self._visible_turn_id(turn),
                tool_names=(tool_name,),
            )
        )
        context = replace(
            self._tool_context,
            session_id=turn.session_id,
            turn_id=self._visible_turn_id(turn),
        )
        result = await self._execute_tool_with_approval(tool_call, context, turn=turn)
        self._persist_tool_result(turn, result=result)
        self._maybe_begin_orchestrator_yield(turn, result=result)
        return self._tool_bridge.build_tool_response(result)

    async def _stream_new_command(
        self,
        command: ParsedCommand,
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        session = await self._start_fresh_session(start_reason="user_new")
        if command.body:
            async for event in self._stream_message_turn(
                user_text=command.body,
                force_session_id=session.session_id,
                command_override="/new",
            ):
                yield event
            return
        yield AgentAssistantMessageEvent(
            session_id=session.session_id,
            text="Started a new session.",
        )
        yield AgentTurnDoneEvent(
            session_id=session.session_id,
            response_text="Started a new session.",
            command="/new",
        )

    async def _stream_compact_command(
        self,
        command: ParsedCommand,
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        session_id = await self.prepare_session()
        active = self._require_session(session_id)
        compacted = await self._compact_session(
            active,
            reason="manual",
            user_instruction=command.body or None,
        )
        if compacted is None:
            response_text = "No conversation history to compact yet."
            session = active
            compaction_performed = False
        else:
            response_text = "Context compacted into a new session."
            session = compacted
            compaction_performed = True
        yield AgentAssistantMessageEvent(
            session_id=session.session_id,
            text=response_text,
        )
        yield AgentTurnDoneEvent(
            session_id=session.session_id,
            response_text=response_text,
            command="/compact",
            compaction_performed=compaction_performed,
        )

    async def _stream_message_turn(
        self,
        *,
        user_text: str | None,
        force_session_id: str | None,
        command_override: str | None,
        pre_turn_messages: Sequence[AgentRuntimeMessage],
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        if force_session_id is not None:
            session = self._require_session(force_session_id)
            self._storage.set_active_session(session.session_id)
        else:
            session_id = await self.prepare_session()
            session = self._require_session(session_id)
        external_runtime_messages, last_external_record_id = self._pending_external_runtime_messages(
            session.session_id
        )
        runtime_messages = (
            tuple(self._runtime_messages_provider(session.session_id))
            if self._runtime_messages_provider is not None
            else ()
        )
        if external_runtime_messages and user_text is None:
            runtime_messages = ()
        all_pre_turn_messages = (
            *external_runtime_messages,
            *runtime_messages,
            *pre_turn_messages,
        )
        await self._maybe_run_due_maintenance()
        compaction_seed_records = self._pending_compaction_seed_records(session)
        thread_id = await self._ensure_thread_loaded(session)
        turn = _TurnState(
            session_id=session.session_id,
            command=command_override,
            user_text=user_text,
        )
        self._current_turn = turn
        try:
            input_items = self._build_turn_input_items(
                user_text=user_text,
                pre_turn_messages=all_pre_turn_messages,
                compaction_seed_records=compaction_seed_records,
            )
            if not input_items:
                raise CodexProtocolError("Codex turn input cannot be empty.")
            dynamic_tools, dynamic_tools_signature = self._build_dynamic_tools_bundle(
                session.session_id,
                turn_id=None,
            )
            turn_start_params: dict[str, Any] = {
                "threadId": thread_id,
                "input": input_items,
                "cwd": str(self._path_mapper.host_repo_root),
                "approvalPolicy": self._settings.approval_policy,
                "sandboxPolicy": self._settings.sandbox_policy(),
                "model": self._settings.model,
                "effort": self._settings.reasoning_effort,
                "summary": self._settings.reasoning_summary,
                "personality": self._settings.personality,
            }
            if dynamic_tools_signature != self._loaded_dynamic_tools_signature:
                turn_start_params["dynamicTools"] = dynamic_tools
            turn_response = await self._coordinator.request("turn/start", turn_start_params)
            provider_turn_id = _extract_turn_id(turn_response)
            turn.provider_turn_id = provider_turn_id
            if turn.logical_turn_id is None:
                turn.logical_turn_id = provider_turn_id
            self._active_turn_id = turn.logical_turn_id
            self._persist_turn_start(
                session_id=session.session_id,
                turn_id=turn.logical_turn_id,
                user_text=user_text,
                pre_turn_messages=all_pre_turn_messages,
            )
            self._storage.update_session(
                session.session_id,
                backend_state={
                    **dict(session.backend_state),
                    "backend_kind": "codex",
                    "thread_id": thread_id,
                    "last_turn_id": provider_turn_id,
                    "dynamic_tools_signature": dynamic_tools_signature,
                    "compaction_seed_pending": False if compaction_seed_records else bool(
                        session.backend_state.get("compaction_seed_pending", False)
                    ),
                    **(
                        {"last_synced_external_record_id": last_external_record_id}
                        if last_external_record_id is not None
                        else {}
                    ),
                },
            )
            self._loaded_dynamic_tools_signature = dynamic_tools_signature
            yield AgentTurnStartedEvent(
                session_id=session.session_id,
                turn_id=turn.logical_turn_id,
            )
            while True:
                queued = await turn.queue.get()
                if isinstance(queued, Exception):
                    raise queued
                if not isinstance(
                    queued,
                    (
                        AgentTextDeltaEvent,
                        AgentAssistantMessageEvent,
                        AgentToolCallEvent,
                        AgentApprovalRequestEvent,
                        AgentTurnDoneEvent,
                    ),
                ):
                    raise RuntimeError("Unexpected Codex turn queue payload.")
                yield queued
                if queued.type == "done":
                    return
        finally:
            if self._current_turn is turn:
                self._current_turn = None
            if self._active_turn_id == turn.logical_turn_id:
                self._active_turn_id = None
            self._pending_approval_future = None
            self._pending_approval_id = None
            interrupt_task = turn.provider_interrupt_task
            if interrupt_task is not None and interrupt_task.done():
                with contextlib.suppress(Exception):
                    interrupt_task.result()

    async def _execute_tool_with_approval(
        self,
        tool_call,
        context: ToolExecutionContext,
        *,
        turn: _TurnState,
    ) -> ToolExecutionResult:
        result = await self._tool_executor(tool_call, context)
        if not result.metadata.get("approval_required"):
            return result
        approval_request = result.metadata.get("approval_request")
        if not isinstance(approval_request, dict):
            return result
        approval_id = str(approval_request.get("approval_id", "")).strip() or uuid4().hex
        pending = asyncio.get_running_loop().create_future()
        self._pending_approval_future = pending
        self._pending_approval_id = approval_id
        await turn.queue.put(
            AgentApprovalRequestEvent(
                session_id=turn.session_id,
                turn_id=self._visible_turn_id(turn) or "",
                approval_id=approval_id,
                kind=str(approval_request.get("kind", "")),
                summary=str(approval_request.get("summary", "")),
                details=str(approval_request.get("details", "")),
                command=_optional_string(approval_request.get("command")),
                tool_name=_optional_string(approval_request.get("tool_name"))
                or tool_call.name,
                inspection_url=_optional_string(approval_request.get("inspection_url")),
            )
        )
        try:
            approved = await pending
        except asyncio.CancelledError:
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                ok=False,
                content=_APPROVAL_INTERRUPTED_TEXT,
                metadata={"approval_interrupted": True},
            )
        finally:
            self._pending_approval_future = None
            self._pending_approval_id = None
        if not approved:
            turn.approval_rejected = True
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                ok=False,
                content=_APPROVAL_REJECTED_TEXT,
                metadata={"approval_rejected": True},
            )
        approved_context = replace(
            context,
            approved_action=dict(approval_request),
        )
        return await self._tool_executor(tool_call, approved_context)

    async def _maybe_run_due_maintenance(self) -> None:
        if not self._memory_mode.maintenance:
            return
        try:
            await self._memory_service.run_due_maintenance()
        except Exception:
            LOGGER.exception("Codex actor memory maintenance failed.")

    async def _ensure_thread_loaded(self, session: SessionMetadata) -> str:
        backend_state = dict(session.backend_state)
        thread_id = _optional_string(backend_state.get("thread_id"))
        if (
            thread_id is not None
            and self._loaded_thread_id == thread_id
            and self._loaded_session_id == session.session_id
        ):
            return thread_id

        await self._coordinator.ensure_authenticated(
            on_challenge=self._publish_auth_challenge,
        )
        developer_instructions = await self._build_developer_instructions()
        dynamic_tools, dynamic_tools_signature = self._build_dynamic_tools_bundle(
            session.session_id,
            turn_id=None,
        )
        stored_dynamic_tools_signature = _optional_string(
            backend_state.get("dynamic_tools_signature")
        )
        stored_external_record_id = _optional_string(
            backend_state.get("last_synced_external_record_id")
        )
        if thread_id is not None and stored_external_record_id is None:
            latest_external_record_id = self._latest_external_runtime_record_id(
                session.session_id
            )
            if latest_external_record_id is not None:
                backend_state = {
                    **backend_state,
                    "last_synced_external_record_id": latest_external_record_id,
                }
                stored_external_record_id = latest_external_record_id
        if thread_id is None:
            response = await self._coordinator.request(
                "thread/start",
                {
                    "model": self._settings.model,
                    "cwd": str(self._path_mapper.host_repo_root),
                    "approvalPolicy": self._settings.approval_policy,
                    "sandbox": "workspace-write",
                    "serviceName": self._settings.service_name,
                    "developerInstructions": developer_instructions,
                    "personality": self._settings.personality,
                    "experimentalRawEvents": False,
                    "persistExtendedHistory": False,
                    "dynamicTools": dynamic_tools,
                },
            )
            thread_id = _extract_thread_id(response)
        else:
            await self._coordinator.request(
                "thread/resume",
                {
                    "threadId": thread_id,
                    "model": self._settings.model,
                    "cwd": str(self._path_mapper.host_repo_root),
                    "approvalPolicy": self._settings.approval_policy,
                    "sandbox": "workspace-write",
                    "developerInstructions": developer_instructions,
                    "personality": self._settings.personality,
                    "persistExtendedHistory": False,
                    **(
                        {"dynamicTools": dynamic_tools}
                        if stored_dynamic_tools_signature is None
                        or stored_dynamic_tools_signature != dynamic_tools_signature
                        else {}
                    ),
                },
            )
        if self._loaded_thread_id is not None and self._loaded_thread_id != thread_id:
            self._coordinator.unregister_actor(
                thread_id=self._loaded_thread_id,
                actor=self,
            )
        self._loaded_thread_id = thread_id
        self._loaded_session_id = session.session_id
        self._loaded_dynamic_tools_signature = dynamic_tools_signature
        self._coordinator.register_actor(thread_id=thread_id, actor=self)
        self._storage.update_session(
            session.session_id,
            backend_state={
                **backend_state,
                "backend_kind": "codex",
                "thread_id": thread_id,
                "dynamic_tools_signature": dynamic_tools_signature,
                **(
                    {"last_synced_external_record_id": stored_external_record_id}
                    if stored_external_record_id is not None
                    else {}
                ),
            },
        )
        return thread_id

    async def _build_developer_instructions(self) -> str:
        bootstrap_messages = self._bootstrap_loader.load_bootstrap_messages()
        sections: list[str] = []
        for message in bootstrap_messages:
            if message.role != "system":
                continue
            text = _message_text(message)
            if text:
                sections.append(text)
        if self._memory_mode.bootstrap:
            try:
                core_text, ongoing_text = await self._memory_service.render_bootstrap_messages()
            except Exception:
                LOGGER.exception("Codex actor memory bootstrap rendering failed.")
                core_text = ""
                ongoing_text = ""
            if core_text:
                sections.append("Memory bootstrap:\n" + core_text)
            if ongoing_text:
                sections.append("Active ongoing memory:\n" + ongoing_text)
        sections.append(
            "Tooling boundary:\n"
            "- The only allowed external capabilities are the Jarvis dynamic tools in this thread.\n"
            "- If a capability is not exposed as a Jarvis dynamic tool, do not use a native Codex fallback.\n"
            "- If the task, assignment, or orchestrator update already names the exact Jarvis tool "
            "to use, treat tool discovery as already satisfied and do not call `tool_search` again.\n"
            "- Use `tool_search` when tool choice is genuinely unclear or when you need a "
            "discoverable tool that has not already been named.\n"
            "- For shell or terminal work, use Jarvis tool `bash`.\n"
            "- For text file edits, use Jarvis tool `file_patch` or `bash`.\n"
            "- For delegation or subagent work, use Jarvis tools "
            "`subagent_invoke`, `subagent_monitor`, `subagent_stop`, "
            "`subagent_step_in`, and `subagent_dispose`.\n"
            "- Never invoke native Codex command execution, native file changes, native MCP "
            "tools, native collaboration/subagent tools, or any other native Codex capability.\n"
            "- If a task seems to call for a native Codex capability, choose the Jarvis dynamic "
            "tool with the matching responsibility or continue without that capability."
        )
        return "\n\n".join(section.strip() for section in sections if section.strip())

    async def _persist_session_bootstrap(self, session_id: str) -> None:
        developer_instructions = await self._build_developer_instructions()
        self._storage.append_record(
            session_id,
            ConversationRecord(
                record_id=uuid4().hex,
                session_id=session_id,
                created_at=_utc_now_iso(),
                role="system",
                content=(
                    "Codex developer instructions snapshot:\n\n"
                    f"{developer_instructions}"
                ),
                metadata={
                    _TRANSCRIPT_ONLY_RECORD_METADATA_KEY: True,
                    _CODEX_BOOTSTRAP_METADATA_KEY: "developer_instructions",
                },
            ),
        )
        dynamic_tools = self._tool_bridge.build_dynamic_tools(
            activated_discoverable_tool_names=(),
        )
        self._storage.append_record(
            session_id,
            ConversationRecord(
                record_id=uuid4().hex,
                session_id=session_id,
                created_at=_utc_now_iso(),
                role="system",
                content=(
                    "Codex dynamic tools bootstrap snapshot:\n\n"
                    + json.dumps(dynamic_tools, ensure_ascii=False, indent=2)
                ),
                metadata={
                    _TRANSCRIPT_ONLY_RECORD_METADATA_KEY: True,
                    _CODEX_BOOTSTRAP_METADATA_KEY: "dynamic_tools",
                    "tool_definitions": dynamic_tools,
                },
            ),
        )

    async def _publish_auth_challenge(self, challenge: CodexAuthChallenge) -> None:
        # Imported lazily to keep gateway-specific event types out of backend module import time.
        from jarvis.gateway.route_events import RouteAuthRequiredEvent

        turn = self._current_turn
        session_id = turn.session_id if turn is not None else self.active_session_id()
        await self._publish_route_event(
            RouteAuthRequiredEvent(
                route_id=str(self._tool_context.route_id or ""),
                agent_kind=self._identity.kind,
                agent_name=self._identity.name,
                session_id=session_id,
                subagent_id=self._identity.subagent_id,
                provider="codex",
                auth_kind="openai_oauth",
                login_id=challenge.login_id,
                auth_url=challenge.auth_url,
                message="Open the browser login URL to authorize Jarvis with your OpenAI account.",
            )
        )

    async def _terminate_turn_with_error(
        self,
        turn: _TurnState,
        exc: Exception,
    ) -> None:
        if turn.terminated:
            return
        turn.terminated = True
        interrupt_task = turn.provider_interrupt_task
        if interrupt_task is not None:
            interrupt_task.cancel()
            turn.provider_interrupt_task = None
        if turn.logical_turn_id is not None:
            self._storage.set_turn_status(
                turn.session_id,
                turn_id=turn.logical_turn_id,
                status="interrupted",
            )
        self._active_turn_id = None
        self._requested_interruption = None
        await turn.queue.put(exc)

    async def _start_fresh_session(self, *, start_reason: str) -> SessionMetadata:
        return await self._start_fresh_session_with_replacement_history(
            start_reason=start_reason,
            replacement_items=(),
            compaction_count=None,
        )

    async def _start_fresh_session_with_replacement_history(
        self,
        *,
        start_reason: str,
        replacement_items: Sequence[CompactionReplacementItem],
        compaction_count: int | None,
    ) -> SessionMetadata:
        active = self._storage.get_active_session()
        if active is not None:
            self._storage.archive_session(active.session_id)
        if self._loaded_thread_id is not None:
            self._coordinator.unregister_actor(
                thread_id=self._loaded_thread_id,
                actor=self,
            )
        session = self._storage.create_session(
            parent_session_id=active.session_id if active is not None else None,
            start_reason=start_reason,
        )
        await self._persist_session_bootstrap(session.session_id)
        if replacement_items:
            self._persist_compaction_replacement_items(
                session.session_id,
                items=replacement_items,
                generation=max(compaction_count or 0, 1),
            )
        updates: dict[str, Any] = {
            "backend_state": {
                **dict(session.backend_state),
                "backend_kind": "codex",
                "compaction_seed_pending": bool(replacement_items),
            }
        }
        if compaction_count is not None:
            updates["compaction_count"] = compaction_count
        session = self._storage.update_session(session.session_id, **updates)
        self._loaded_thread_id = None
        self._loaded_session_id = None
        self._loaded_dynamic_tools_signature = None
        return session

    def _require_session(self, session_id: str) -> SessionMetadata:
        session = self._storage.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown session id: {session_id}")
        return session

    def _persist_turn_start(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_text: str | None,
        pre_turn_messages: Sequence[AgentRuntimeMessage],
    ) -> None:
        self._storage.set_turn_status(session_id, turn_id=turn_id, status="in_progress")
        for runtime_message in pre_turn_messages:
            self._storage.append_record(
                session_id,
                ConversationRecord(
                    record_id=uuid4().hex,
                    session_id=session_id,
                    created_at=_utc_now_iso(),
                    role=runtime_message.role,
                    content=runtime_message.content,
                    metadata={
                        **dict(runtime_message.metadata),
                        _TURN_ID_METADATA_KEY: turn_id,
                    },
                ),
            )
        if user_text is not None:
            self._storage.append_record(
                session_id,
                ConversationRecord(
                    record_id=uuid4().hex,
                    session_id=session_id,
                    created_at=_utc_now_iso(),
                    role="user",
                    content=user_text,
                    metadata={_TURN_ID_METADATA_KEY: turn_id},
                ),
            )

    def _persist_tool_result(
        self,
        turn: _TurnState,
        *,
        result: ToolExecutionResult,
    ) -> None:
        visible_turn_id = self._visible_turn_id(turn)
        if visible_turn_id is None:
            return
        self._storage.append_record(
            turn.session_id,
            ConversationRecord(
                record_id=uuid4().hex,
                session_id=turn.session_id,
                created_at=_utc_now_iso(),
                role="tool",
                content=result.content,
                metadata={
                    **dict(result.metadata),
                    "tool_name": result.name,
                    "tool_call_id": result.call_id,
                    _TURN_ID_METADATA_KEY: visible_turn_id,
                },
            ),
        )

    def _persist_turn_completion(
        self,
        turn: _TurnState,
        *,
        response_text: str,
        status: str,
    ) -> None:
        visible_turn_id = self._visible_turn_id(turn)
        if visible_turn_id is None:
            return
        assistant_messages = [
            message
            for message in turn.completed_messages
            if message.strip()
        ]
        if not assistant_messages and response_text.strip():
            assistant_messages = [response_text]
        for message in assistant_messages:
            self._storage.append_record(
                turn.session_id,
                ConversationRecord(
                    record_id=uuid4().hex,
                    session_id=turn.session_id,
                    created_at=_utc_now_iso(),
                    role="assistant",
                    content=message,
                    metadata={
                        _TURN_ID_METADATA_KEY: visible_turn_id,
                        "provider": "codex",
                        "model": self._settings.model,
                        "response_id": turn.provider_turn_id,
                        "finish_reason": "stop" if status == "completed" else "interrupted",
                        "tool_calls": [],
                    },
                ),
            )
        updates: dict[str, Any] = {}
        if turn.usage is not None:
            updates.update(
                {
                    "last_input_tokens": turn.usage.input_tokens,
                    "last_output_tokens": turn.usage.output_tokens,
                    "last_total_tokens": turn.usage.total_tokens,
                }
            )
        updates["backend_state"] = {
            **dict(self._storage.get_session(turn.session_id).backend_state or {}),
            "backend_kind": "codex",
            "thread_id": self._loaded_thread_id,
            "last_turn_id": turn.provider_turn_id,
        }
        self._storage.update_session(turn.session_id, **updates)
        self._storage.set_turn_status(
            turn.session_id,
            turn_id=visible_turn_id,
            status="completed" if status == "completed" else "interrupted",
        )
        if self._memory_mode.reflection and response_text.strip():
            records = tuple(
                record
                for record in self._storage.load_records(turn.session_id, include_all_turns=True)
                if str(record.metadata.get(_TURN_ID_METADATA_KEY, "")).strip() == visible_turn_id
            )
            asyncio.create_task(
                self._memory_service.reflect_completed_turn(
                    route_id=self._tool_context.route_id,
                    session_id=turn.session_id,
                    records=records,
                ),
                name=f"jarvis-codex-memory-reflection-{visible_turn_id}",
            )

    def _build_turn_input_items(
        self,
        *,
        user_text: str | None,
        pre_turn_messages: Sequence[AgentRuntimeMessage],
        compaction_seed_records: Sequence[ConversationRecord] = (),
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for record in compaction_seed_records:
            items.append(
                {
                    "type": "text",
                    "text": _render_compaction_input_text(record),
                    "text_elements": [],
                }
            )
        for message in pre_turn_messages:
            content = message.content.strip()
            if not content:
                continue
            items.append(
                {
                    "type": "text",
                    "text": _render_runtime_input_text(message),
                    "text_elements": [],
                }
            )
        if user_text is not None and user_text.strip():
            items.append(
                {
                    "type": "text",
                    "text": user_text,
                    "text_elements": [],
                }
            )
        return items

    async def _compact_session(
        self,
        session: SessionMetadata,
        *,
        reason: str,
        user_instruction: str | None = None,
    ) -> SessionMetadata | None:
        records = self._storage.load_records(session.session_id, include_all_turns=True)
        compactable_records = [record for record in records if record.kind == "message"]
        source_records = prune_compaction_source_records(compactable_records)
        if not source_records:
            return None

        if self._memory_mode.maintenance:
            try:
                await self._memory_service.flush_before_compaction(
                    route_id=self._tool_context.route_id,
                    session_id=session.session_id,
                    records=tuple(records),
                )
            except Exception:
                LOGGER.exception("Codex memory pre-compaction flush failed.")

        outcome = await self._compactor.compact(
            source_records,
            user_instruction=user_instruction,
        )
        self._append_compaction_record(session.session_id, outcome=outcome, reason=reason)
        next_session = await self._start_fresh_session_with_replacement_history(
            start_reason="manual_compaction" if reason == "manual" else "compaction",
            replacement_items=outcome.items,
            compaction_count=session.compaction_count + 1,
        )
        return self._storage.get_session(next_session.session_id) or next_session

    def _append_compaction_record(
        self,
        session_id: str,
        *,
        outcome: CompactionOutcome,
        reason: str,
    ) -> None:
        self._storage.append_record(
            session_id,
            ConversationRecord(
                record_id=uuid4().hex,
                session_id=session_id,
                created_at=_utc_now_iso(),
                role="system",
                content=(
                    "Compaction replaced prior session history with "
                    f"{len(outcome.items)} structured handover items."
                ),
                kind="compaction",
                metadata={
                    "reason": reason,
                    "provider": outcome.provider,
                    "model": outcome.model,
                    "response_id": outcome.response_id,
                    "replacement_items": [item.to_dict() for item in outcome.items],
                    "response_payload": outcome.response_payload,
                    "usage": {
                        "input_tokens": outcome.input_tokens,
                        "output_tokens": outcome.output_tokens,
                        "total_tokens": outcome.total_tokens,
                    },
                },
            ),
        )

    def _persist_compaction_replacement_items(
        self,
        session_id: str,
        *,
        items: Sequence[CompactionReplacementItem],
        generation: int,
    ) -> None:
        for item in items:
            self._storage.append_record(
                session_id,
                ConversationRecord(
                    record_id=uuid4().hex,
                    session_id=session_id,
                    created_at=_utc_now_iso(),
                    role=item.role,
                    content=item.content,
                    metadata=item.record_metadata(generation=generation),
                ),
            )

    def _pending_compaction_seed_records(
        self,
        session: SessionMetadata,
    ) -> tuple[ConversationRecord, ...]:
        if not bool(session.backend_state.get("compaction_seed_pending", False)):
            return ()
        records = self._storage.load_records(session.session_id, include_all_turns=True)
        return tuple(record for record in records if _is_compaction_replacement_record(record))

    def _reconcile_orphaned_turns(self, session_id: str) -> None:
        session = self._storage.get_session(session_id)
        if session is None:
            return
        orphaned_turn_ids = [
            turn_id
            for turn_id, status in session.turn_states.items()
            if status == "in_progress" and turn_id != self._active_turn_id
        ]
        for turn_id in orphaned_turn_ids:
            self._storage.append_record(
                session_id,
                ConversationRecord(
                    record_id=uuid4().hex,
                    session_id=session_id,
                    created_at=_utc_now_iso(),
                    role="system",
                    content=_TURN_ORPHANED_RECOVERY_RECORD_TEXT,
                    metadata={
                        _TURN_ID_METADATA_KEY: turn_id,
                        _ORPHANED_TURN_RECOVERY_METADATA_KEY: True,
                    },
                ),
            )
            self._storage.set_turn_status(
                session_id,
                turn_id=turn_id,
                status="interrupted",
            )

    async def _handle_unsupported_native_capability(
        self,
        turn: _TurnState,
        capability_name: str,
    ) -> None:
        if turn.pending_native_recovery_item_type is not None:
            return
        if turn.native_recovery_attempts >= _NATIVE_TOOL_RECOVERY_MAX_ATTEMPTS:
            provider_turn_id = turn.provider_turn_id
            if provider_turn_id is not None:
                turn.provider_interrupt_task = self._request_provider_turn_interrupt(
                    provider_turn_id=provider_turn_id,
                    reason="native_capability_blocked",
                )
            await self._terminate_turn_with_error(
                turn,
                CodexNativeCapabilityError(
                    f"Codex emitted unsupported native capability '{capability_name}'."
                ),
            )
            return
        provider_turn_id = turn.provider_turn_id
        if provider_turn_id is None:
            await self._terminate_turn_with_error(
                turn,
                CodexNativeCapabilityError(
                    f"Codex emitted unsupported native capability '{capability_name}' without an active turn id."
                ),
            )
            return
        turn.pending_native_recovery_item_type = capability_name
        turn.provider_interrupt_task = self._request_provider_turn_interrupt(
            provider_turn_id=provider_turn_id,
            reason="native_capability_blocked",
        )

    def _request_provider_turn_interrupt(
        self,
        *,
        provider_turn_id: str,
        reason: str,
    ) -> asyncio.Task[object] | None:
        thread_id = self._loaded_thread_id
        if thread_id is None:
            return None
        return asyncio.create_task(
            self._coordinator.request(
                "turn/interrupt",
                {
                    "threadId": thread_id,
                    "turnId": provider_turn_id,
                },
            ),
            name=f"jarvis-codex-interrupt-{reason}-{provider_turn_id}",
        )

    async def _recover_from_native_tool_attempt(self, turn: _TurnState) -> None:
        item_type = turn.pending_native_recovery_item_type
        if item_type is None:
            return
        turn.pending_native_recovery_item_type = None
        interrupt_task = turn.provider_interrupt_task
        turn.provider_interrupt_task = None
        if interrupt_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await interrupt_task
        if turn.native_recovery_attempts >= _NATIVE_TOOL_RECOVERY_MAX_ATTEMPTS:
            await self._terminate_turn_with_error(
                turn,
                CodexNativeCapabilityError(
                    f"Codex emitted unsupported native capability '{item_type}'."
                ),
            )
            return
        turn.native_recovery_attempts += 1
        turn.active_assistant_item_id = None
        turn.active_assistant_chunks.clear()
        turn.completed_messages.clear()
        turn.usage = None
        correction_text = self._native_tool_recovery_note(turn, item_type=item_type)
        visible_turn_id = self._visible_turn_id(turn)
        if visible_turn_id is None:
            await self._terminate_turn_with_error(
                turn,
                CodexNativeCapabilityError(
                    f"Codex emitted unsupported native capability '{item_type}'."
                ),
            )
            return
        self._storage.append_record(
            turn.session_id,
            ConversationRecord(
                record_id=uuid4().hex,
                session_id=turn.session_id,
                created_at=_utc_now_iso(),
                role="system",
                content=correction_text,
                metadata={
                    _TURN_ID_METADATA_KEY: visible_turn_id,
                    _NATIVE_TOOL_RECOVERY_METADATA_KEY: True,
                    "item_type": item_type,
                    "attempt": turn.native_recovery_attempts,
                },
            ),
        )
        thread_id = self._loaded_thread_id
        if thread_id is None:
            await self._terminate_turn_with_error(
                turn,
                CodexBackendError("Codex native-tool recovery failed because no thread is loaded."),
            )
            return
        retry_items = self._build_turn_input_items(
            user_text=None,
            pre_turn_messages=(
                AgentRuntimeMessage(
                    role="system",
                    content=correction_text,
                    metadata={
                        _NATIVE_TOOL_RECOVERY_METADATA_KEY: True,
                        "item_type": item_type,
                        "attempt": turn.native_recovery_attempts,
                    },
                ),
            ),
        )
        try:
            turn_response = await self._coordinator.request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": retry_items,
                    "cwd": str(self._path_mapper.host_repo_root),
                    "approvalPolicy": self._settings.approval_policy,
                    "sandboxPolicy": self._settings.sandbox_policy(),
                    "model": self._settings.model,
                    "effort": self._settings.reasoning_effort,
                    "summary": self._settings.reasoning_summary,
                    "personality": self._settings.personality,
                    "dynamicTools": self._tool_bridge.build_dynamic_tools(
                        activated_discoverable_tool_names=_collect_turn_activated_discoverable_tool_names(
                            self._storage.load_records(turn.session_id),
                            turn_id=self._visible_turn_id(turn),
                        ),
                    ),
                },
            )
        except Exception as exc:
            await self._terminate_turn_with_error(
                turn,
                CodexBackendError(
                    "Codex native-tool recovery retry failed before a corrected turn could start."
                ),
            )
            LOGGER.exception("Codex native-tool recovery retry failed.", exc_info=exc)
            return
        provider_turn_id = _extract_turn_id(turn_response)
        turn.provider_turn_id = provider_turn_id
        self._storage.update_session(
            turn.session_id,
            backend_state={
                **dict(self._storage.get_session(turn.session_id).backend_state or {}),
                "backend_kind": "codex",
                "thread_id": thread_id,
                "last_turn_id": provider_turn_id,
                "dynamic_tools_signature": self._loaded_dynamic_tools_signature,
            },
        )

    def _build_dynamic_tools_bundle(
        self,
        session_id: str,
        *,
        turn_id: str | None,
    ) -> tuple[list[dict[str, Any]], str]:
        dynamic_tools = self._tool_bridge.build_dynamic_tools(
            activated_discoverable_tool_names=_collect_turn_activated_discoverable_tool_names(
                self._storage.load_records(session_id),
                turn_id=turn_id,
            ),
        )
        return dynamic_tools, _dynamic_tools_signature(dynamic_tools)

    def _pending_external_runtime_messages(
        self,
        session_id: str,
    ) -> tuple[tuple[AgentRuntimeMessage, ...], str | None]:
        session = self._storage.get_session(session_id)
        if session is None:
            return (), None
        records = self._storage.load_records(session_id, include_all_turns=True)
        last_synced_record_id = _optional_string(
            session.backend_state.get("last_synced_external_record_id")
        )
        eligible_records = [
            record for record in records if _is_external_runtime_record(record)
        ]
        if not eligible_records:
            return (), None
        pending_records: list[ConversationRecord]
        if last_synced_record_id is None:
            pending_records = eligible_records
        else:
            pending_records = []
            found_last_synced = False
            for record in eligible_records:
                if not found_last_synced:
                    if record.record_id == last_synced_record_id:
                        found_last_synced = True
                    continue
                pending_records.append(record)
            if not found_last_synced:
                pending_records = eligible_records
        if not pending_records:
            return (), None
        return (
            tuple(
                AgentRuntimeMessage(
                    role=record.role,
                    content=record.content,
                    metadata=dict(record.metadata),
                )
                for record in pending_records
            ),
            pending_records[-1].record_id,
        )

    def _latest_external_runtime_record_id(
        self,
        session_id: str,
    ) -> str | None:
        records = self._storage.load_records(session_id, include_all_turns=True)
        for record in reversed(records):
            if _is_external_runtime_record(record):
                return record.record_id
        return None

    def _maybe_begin_orchestrator_yield(
        self,
        turn: _TurnState,
        *,
        result: ToolExecutionResult,
    ) -> None:
        if turn.pending_orchestrator_yield or turn.pending_native_recovery_item_type is not None:
            return
        reason = _orchestrator_wait_reason(result)
        if reason is None:
            return
        provider_turn_id = turn.provider_turn_id
        if provider_turn_id is None:
            return
        turn.pending_orchestrator_yield = True
        turn.orchestrator_yield_reason = reason
        turn.provider_interrupt_task = self._request_provider_turn_interrupt(
            provider_turn_id=provider_turn_id,
            reason=reason,
        )

    async def _complete_orchestrator_yield(self, turn: _TurnState) -> None:
        interrupt_task = turn.provider_interrupt_task
        turn.provider_interrupt_task = None
        if interrupt_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await interrupt_task
        response_text = turn.full_text()
        if response_text.strip() and not turn.completed_messages:
            await turn.queue.put(
                AgentAssistantMessageEvent(
                    session_id=turn.session_id,
                    turn_id=self._visible_turn_id(turn),
                    text=response_text,
                )
            )
        self._persist_turn_completion(
            turn,
            response_text=response_text,
            status="completed",
        )
        await turn.queue.put(
            AgentTurnDoneEvent(
                session_id=turn.session_id,
                turn_id=self._visible_turn_id(turn),
                response_text=response_text,
                command=turn.command,
                compaction_performed=turn.compaction_performed,
                interrupted=False,
                approval_rejected=False,
                interruption_reason=None,
            )
        )
        turn.terminated = True
        turn.pending_orchestrator_yield = False
        turn.orchestrator_yield_reason = None
        self._active_turn_id = None
        self._requested_interruption = None
        self._current_turn = None

    def _native_tool_recovery_note(self, turn: _TurnState, *, item_type: str) -> str:
        lines = [_NATIVE_TOOL_RECOVERY_NOTE_TEMPLATE.format(item_type=item_type)]
        if turn.user_text is not None and turn.user_text.strip():
            lines.extend(
                [
                    "",
                    "Original user request for this turn:",
                    turn.user_text.strip(),
                ]
            )
        return "\n".join(lines)

    def _visible_turn_id(self, turn: _TurnState) -> str | None:
        return turn.logical_turn_id or turn.provider_turn_id

    def _rollover_active_assistant_item(
        self,
        turn: _TurnState,
        *,
        incoming_item_id: str,
    ) -> None:
        active_item_id = turn.active_assistant_item_id
        if (
            active_item_id is None
            or active_item_id == incoming_item_id
            or not turn.active_assistant_chunks
        ):
            return
        turn.completed_messages.append("".join(turn.active_assistant_chunks))
        turn.active_assistant_chunks.clear()
        turn.active_assistant_item_id = None

    async def _complete_assistant_item(
        self,
        turn: _TurnState,
        *,
        item: dict[str, Any],
    ) -> None:
        if turn.pending_native_recovery_item_type is not None:
            return
        item_id = _optional_string(item.get("id"))
        if item_id is not None:
            self._rollover_active_assistant_item(turn, incoming_item_id=item_id)
        text = ""
        if (
            item_id is not None
            and turn.active_assistant_item_id == item_id
            and turn.active_assistant_chunks
        ):
            text = "".join(turn.active_assistant_chunks)
            turn.active_assistant_chunks.clear()
            turn.active_assistant_item_id = None
        if not text.strip():
            text = str(item.get("text", ""))
        normalized = text.strip()
        if not normalized:
            return
        turn.completed_messages.append(normalized)
        await turn.queue.put(
            AgentAssistantMessageEvent(
                session_id=turn.session_id,
                turn_id=self._visible_turn_id(turn),
                text=normalized,
            )
        )


def _extract_thread_id(response: object) -> str:
    if not isinstance(response, dict):
        raise CodexProtocolError("Codex thread response must be a JSON object.")
    thread = response.get("thread")
    if not isinstance(thread, dict):
        raise CodexProtocolError("Codex thread response is missing 'thread'.")
    return _required_string(thread.get("id"), field_name="thread.id")


def _extract_turn_id(response: object) -> str:
    if not isinstance(response, dict):
        raise CodexProtocolError("Codex turn response must be a JSON object.")
    turn = response.get("turn")
    if not isinstance(turn, dict):
        raise CodexProtocolError("Codex turn response is missing 'turn'.")
    return _required_string(turn.get("id"), field_name="turn.id")


def _required_string(value: object, *, field_name: str) -> str:
    normalized = _optional_string(value)
    if normalized is None:
        raise CodexProtocolError(f"Codex payload field '{field_name}' must be a non-empty string.")
    return normalized


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_expected_interrupt_error(error: object) -> bool:
    if not isinstance(error, dict):
        return False
    message = str(error.get("message", "")).strip().lower()
    additional = str(error.get("additionalDetails", "")).strip().lower()
    combined = " ".join(part for part in (message, additional) if part)
    if not combined:
        return False
    return "interrupt" in combined


def _utc_now_iso() -> str:
    import datetime as _datetime

    return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat()


def _message_text(message: LLMMessage) -> str:
    parts: list[str] = []
    for part in message.parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n\n".join(parts).strip()


def _render_runtime_input_text(message: AgentRuntimeMessage) -> str:
    role = message.role
    if role == "system":
        return f"System note from Jarvis:\n\n{message.content}"
    if role == "assistant":
        return f"Assistant note:\n\n{message.content}"
    if role == "tool":
        return f"Tool result context:\n\n{message.content}"
    return message.content


def _render_compaction_input_text(record: ConversationRecord) -> str:
    metadata = record.metadata
    lines = [
        "Compacted prior-session history item:",
        "type: compaction",
        f"role: {record.role}",
        f"kind: {str(metadata.get('compaction_kind', 'condensed_span')).strip() or 'condensed_span'}",
    ]
    if metadata.get("verbatim"):
        lines.append("verbatim: true")
    lines.extend(
        [
            "",
            record.content,
        ]
    )
    return "\n".join(lines)


def _collect_turn_activated_discoverable_tool_names(
    records: Sequence[ConversationRecord],
    *,
    turn_id: str | None,
) -> tuple[str, ...]:
    if turn_id is None:
        return ()
    names: list[str] = []
    seen: set[str] = set()
    for record in records:
        if record.role != "tool":
            continue
        if _optional_string(record.metadata.get(_TURN_ID_METADATA_KEY)) != turn_id:
            continue
        raw_names = record.metadata.get("activated_discoverable_tool_names")
        if not isinstance(raw_names, list):
            continue
        for raw_name in raw_names:
            name = str(raw_name).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
    return tuple(names)


def _orchestrator_wait_reason(result: ToolExecutionResult) -> str | None:
    if result.name == "bash":
        status = str(result.metadata.get("status") or result.metadata.get("state") or "").strip()
        mode = str(result.metadata.get("mode", "")).strip()
        if status == "running" and (
            mode == "background" or bool(result.metadata.get("promoted_to_background"))
        ):
            return "orchestrator_wait_bash"
        return None
    if not result.metadata.get("subagent_control"):
        return None
    action = str(result.metadata.get("subagent_action", "")).strip()
    status = str(result.metadata.get("status", "")).strip()
    if action in {"invoke", "step_in"} and status in {
        "running",
        "waiting_background",
        "awaiting_approval",
    }:
        return "orchestrator_wait_subagent"
    return None


def _dynamic_tools_signature(dynamic_tools: Sequence[dict[str, Any]]) -> str:
    return json.dumps(dynamic_tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_external_runtime_record(record: ConversationRecord) -> bool:
    if record.role != "system":
        return False
    metadata = record.metadata
    if metadata.get(_TRANSCRIPT_ONLY_RECORD_METADATA_KEY):
        return False
    if _is_compaction_replacement_record(record):
        return False
    return _optional_string(metadata.get(_TURN_ID_METADATA_KEY)) is None


def _is_compaction_replacement_record(record: ConversationRecord) -> bool:
    if record.kind != "message":
        return False
    metadata = record.metadata
    return bool(
        metadata.get("compaction_item")
        and metadata.get("type") == "compaction"
        and record.role in {"system", "user", "assistant"}
    )
