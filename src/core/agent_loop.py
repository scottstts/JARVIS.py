"""Core agentic loop with sessioning and context compaction policies."""

from __future__ import annotations
import asyncio
import base64
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, Protocol, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

from llm import (
    ImagePart,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMService,
    ProviderBadRequestError,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)
from memory import MemoryService, MemorySettings
from storage import ConversationRecord, SessionMetadata, SessionStorage
from tools import ToolExecutionContext, ToolExecutionResult, ToolRegistry, ToolRuntime, ToolSettings

from .commands import ParsedCommand, parse_user_command
from .compaction import CompactionOutcome, ContextCompactor
from .config import CoreSettings
from .errors import ContextBudgetError
from .identities import IdentityBootstrapLoader
from .token_estimator import estimate_request_input_tokens

_OVERFLOW_ERROR_HINTS = (
    "context window",
    "context length",
    "maximum context length",
    "prompt is too long",
    "too many tokens",
    "input is too long",
    "context_length_exceeded",
    "exceeds the model",
)
_TRANSIENT_RECORD_METADATA_KEY = "transient"
_TRANSCRIPT_ONLY_RECORD_METADATA_KEY = "transcript_only"
_IMAGE_INPUT_METADATA_KEY = "image_input"
_TURN_CONTEXT_METADATA_KEY = "turn_context"
_TURN_ID_METADATA_KEY = "turn_id"
_INTERRUPTION_NOTICE_METADATA_KEY = "interruption_notice"
_TOOL_ROUND_LIMIT_METADATA_KEY = "tool_round_limit"
_TOOL_BOOTSTRAP_METADATA_KEY = "tool_bootstrap"
_TOOL_ROUND_LIMIT_RECOVERY_TEXT = (
    "I reached the per-turn tool round limit before finishing. "
    "Continue in a new turn if you want me to keep using tools."
)
_FOLLOWUP_COMPACTION_FAILED_TEXT = (
    "Follow-up request overflow occurred and compaction could not proceed."
)
_FOLLOWUP_RETRY_PREFLIGHT_FAILED_TEXT = (
    "Follow-up retry aborted: compacted request still exceeds preflight limit."
)
_FOLLOWUP_RETRY_PROVIDER_OVERFLOW_TEXT = (
    "Follow-up retry aborted: compacted request still overflowed the provider context limit."
)
_APPROVAL_REJECTED_TEXT = "Approval request was rejected. I did not execute the action."
_PREVIOUS_TASK_INTERRUPTED_TEXT = "The previous task was interrupted by the user."
_TURN_INTERRUPTED_RECORD_TEXT = "This turn was interrupted by the user before it completed."
LOGGER = logging.getLogger(__name__)

AgentKind = Literal["main", "subagent"]


class BootstrapMessageLoader(Protocol):
    def load_bootstrap_messages(self) -> list[LLMMessage]:
        """Return the starter context messages for a newly created session."""


@dataclass(slots=True, frozen=True)
class AgentRuntimeMessage:
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str
    transient: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AgentIdentity:
    kind: AgentKind
    name: str
    subagent_id: str | None = None


@dataclass(slots=True, frozen=True)
class AgentMemoryMode:
    bootstrap: bool = True
    maintenance: bool = True
    reflection: bool = True


ToolDefinitionsProvider = Callable[[Sequence[str]], tuple[ToolDefinition, ...]]
ToolExecutorCallable = Callable[[ToolCall, ToolExecutionContext], Awaitable[ToolExecutionResult]]
RuntimeMessagesProvider = Callable[[str], Sequence[AgentRuntimeMessage]]


@dataclass(slots=True, frozen=True)
class AgentTurnResult:
    session_id: str
    response_text: str
    command: str | None = None
    compaction_performed: bool = False
    interrupted: bool = False
    approval_rejected: bool = False


@dataclass(slots=True, frozen=True)
class AgentTextDeltaEvent:
    session_id: str
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True, frozen=True)
class AgentAssistantMessageEvent:
    session_id: str
    text: str
    type: Literal["assistant_message"] = "assistant_message"


@dataclass(slots=True, frozen=True)
class AgentToolCallEvent:
    session_id: str
    tool_names: tuple[str, ...]
    type: Literal["tool_call"] = "tool_call"


@dataclass(slots=True, frozen=True)
class AgentApprovalRequestEvent:
    session_id: str
    approval_id: str
    kind: str
    summary: str
    details: str
    command: str | None = None
    tool_name: str | None = None
    inspection_url: str | None = None
    type: Literal["approval_request"] = "approval_request"


@dataclass(slots=True, frozen=True)
class AgentTurnDoneEvent:
    session_id: str
    response_text: str
    command: str | None = None
    compaction_performed: bool = False
    interrupted: bool = False
    approval_rejected: bool = False
    type: Literal["done"] = "done"

    def to_result(self) -> AgentTurnResult:
        return AgentTurnResult(
            session_id=self.session_id,
            response_text=self.response_text,
            command=self.command,
            compaction_performed=self.compaction_performed,
            interrupted=self.interrupted,
            approval_rejected=self.approval_rejected,
        )


AgentTurnStreamEvent = (
    AgentTextDeltaEvent
    | AgentAssistantMessageEvent
    | AgentToolCallEvent
    | AgentApprovalRequestEvent
    | AgentTurnDoneEvent
)


@dataclass(slots=True, frozen=True)
class _ToolExecutionOutcome:
    approval_rejected: bool = False
    interrupted: bool = False


class AgentLoop:
    """Stateful agent loop over a single long-running DM thread."""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        settings: CoreSettings | None = None,
        storage: SessionStorage | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_runtime: ToolRuntime | None = None,
        route_id: str | None = None,
        bootstrap_loader: BootstrapMessageLoader | None = None,
        identity: AgentIdentity | None = None,
        memory_mode: AgentMemoryMode | None = None,
        llm_provider: str | None = None,
        tool_definitions_provider: ToolDefinitionsProvider | None = None,
        tool_executor: ToolExecutorCallable | None = None,
        runtime_messages_provider: RuntimeMessagesProvider | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._settings = settings or CoreSettings.from_env()
        self._storage = storage or SessionStorage(self._settings.transcript_archive_dir)
        self._identity = identity or AgentIdentity(kind="main", name="Jarvis")
        self._memory_mode = memory_mode or AgentMemoryMode()
        self._llm_provider = (
            normalized
            if (normalized := (llm_provider or "").strip().lower())
            else None
        )
        self._identity_loader = bootstrap_loader or IdentityBootstrapLoader(self._settings)
        self._runtime_messages_provider = runtime_messages_provider
        self._compactor = ContextCompactor(
            llm_service=self._llm_service,
            context_policy=self._settings.context_policy,
            provider=self._llm_provider,
        )
        memory_settings = MemorySettings.from_workspace_dir(self._settings.workspace_dir)
        memory_llm_service = self._llm_service if isinstance(self._llm_service, LLMService) else None
        if memory_llm_service is None or not self._memory_mode.reflection:
            memory_settings = replace(memory_settings, enable_reflection=False)
        self._memory_service = MemoryService(
            settings=memory_settings,
            llm_service=memory_llm_service,
        )
        self._tool_settings = ToolSettings.from_workspace_dir(self._settings.workspace_dir)
        self._tool_registry = tool_registry or ToolRegistry.default(self._tool_settings)
        self._tool_runtime = tool_runtime or ToolRuntime(registry=self._tool_registry)
        self._tool_definitions_provider = tool_definitions_provider or self._default_tool_definitions
        self._tool_executor = tool_executor or self._default_execute_tool_call
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
        self._active_turn_id: str | None = None
        self._stop_requested_turn_id: str | None = None
        self._pending_approval_future: asyncio.Future[bool] | None = None
        self._pending_approval_id: str | None = None
        self._pending_approval_turn_id: str | None = None

    @property
    def agent_kind(self) -> AgentKind:
        return self._identity.kind

    @property
    def agent_name(self) -> str:
        return self._identity.name

    @property
    def subagent_id(self) -> str | None:
        return self._identity.subagent_id

    async def handle_user_input(self, user_text: str) -> AgentTurnResult:
        command = parse_user_command(user_text)
        if command.kind == "new":
            return await self._handle_new_command(command)
        if command.kind == "compact":
            return await self._handle_compact_command(command)
        return await self.handle_turn(user_text=command.body)

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

    async def handle_turn(
        self,
        *,
        user_text: str,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AgentTurnResult:
        return await self._handle_message_turn(
            user_text,
            force_session_id=force_session_id,
            command_override=command_override,
            pre_turn_messages=pre_turn_messages,
        )

    async def stream_turn(
        self,
        *,
        user_text: str,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        async for event in self._stream_message_turn(
            user_text,
            force_session_id=force_session_id,
            command_override=command_override,
            pre_turn_messages=pre_turn_messages,
        ):
            yield event

    def active_session_id(self) -> str | None:
        active = self._storage.get_active_session()
        return active.session_id if active is not None else None

    async def prepare_session(self, *, start_reason: str = "initial") -> str:
        await self._ensure_memory_runtime_ready()
        active = self._storage.get_active_session()
        if active is not None:
            return active.session_id
        session = await self._start_session(start_reason=start_reason)
        return session.session_id

    def request_stop(self) -> bool:
        active_turn_id = self._active_turn_id
        if active_turn_id is None:
            return False
        self._stop_requested_turn_id = active_turn_id
        return True

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        normalized = approval_id.strip()
        if not normalized:
            return False
        pending_future = self._pending_approval_future
        if pending_future is None or pending_future.done():
            return False
        if self._pending_approval_id != normalized:
            return False
        pending_future.set_result(bool(approved))
        return True

    async def _handle_new_command(self, command: ParsedCommand) -> AgentTurnResult:
        await self._ensure_memory_runtime_ready()
        session = await self._start_session(start_reason="user_new")
        if command.body:
            return await self._handle_message_turn(
                command.body,
                force_session_id=session.session_id,
                command_override="/new",
            )
        return AgentTurnResult(
            session_id=session.session_id,
            response_text="Started a new session.",
            command="/new",
        )

    async def _handle_compact_command(self, command: ParsedCommand) -> AgentTurnResult:
        await self._ensure_memory_runtime_ready()
        active = await self._ensure_active_session()
        compacted = await self._compact_session(
            active,
            reason="manual",
            user_instruction=command.body or None,
        )
        if compacted is None:
            return AgentTurnResult(
                session_id=active.session_id,
                response_text="No conversation history to compact yet.",
                command="/compact",
            )
        return AgentTurnResult(
            session_id=compacted.session_id,
            response_text="Context compacted into a new session.",
            command="/compact",
            compaction_performed=True,
        )

    async def _stream_new_command(
        self,
        command: ParsedCommand,
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        await self._ensure_memory_runtime_ready()
        session = await self._start_session(start_reason="user_new")
        if command.body:
            async for event in self._stream_message_turn(
                command.body,
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
            compaction_performed=False,
        )

    async def _stream_compact_command(
        self,
        command: ParsedCommand,
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        result = await self._handle_compact_command(command)
        yield AgentAssistantMessageEvent(
            session_id=result.session_id,
            text=result.response_text,
        )
        yield AgentTurnDoneEvent(
            session_id=result.session_id,
            response_text=result.response_text,
            command=result.command,
            compaction_performed=result.compaction_performed,
        )

    async def _handle_message_turn(
        self,
        user_text: str,
        *,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AgentTurnResult:
        (
            session,
            base_records,
            turn_context_text,
            interruption_notice_text,
            turn_runtime_messages,
            request,
            estimated_input_tokens,
            did_compaction,
        ) = await self._prepare_message_turn(
            user_text=user_text,
            force_session_id=force_session_id,
            pre_turn_messages=pre_turn_messages,
        )
        turn_id = uuid4().hex
        pending_records = self._build_pending_turn_records(
            session_id=session.session_id,
            turn_context_text=turn_context_text,
            interruption_notice_text=interruption_notice_text,
            runtime_messages=turn_runtime_messages,
        )
        user_record = self._build_message_record(
            session_id=session.session_id,
            role="user",
            content=user_text,
            turn_id=turn_id,
        )
        self._begin_turn(session_id=session.session_id, turn_id=turn_id)
        self._append_turn_record(
            session_id=session.session_id,
            pending_records=pending_records,
            record=user_record,
        )

        try:
            (
                session,
                response,
                overflow_compacted,
                final_estimated_input_tokens,
                rebound_pending_records,
            ) = await self._generate_with_overflow_retry(
                session=session,
                user_text=user_text,
                turn_context_text=turn_context_text,
                interruption_notice_text=interruption_notice_text,
                request=request,
                estimated_input_tokens=estimated_input_tokens,
                pending_records=pending_records,
                turn_id=turn_id,
            )
            pending_records = rebound_pending_records
            if overflow_compacted:
                did_compaction = True

            assistant_record = self._build_assistant_record(
                session.session_id,
                response,
                turn_id=turn_id,
            )
            self._append_turn_record(
                session_id=session.session_id,
                pending_records=pending_records,
                record=assistant_record,
            )
            if self._stop_requested(turn_id):
                return self._interrupt_turn(
                    session_id=session.session_id,
                    turn_id=turn_id,
                    command=command_override,
                    compaction_performed=did_compaction,
                    response_text=response.text,
                )

            base_records = self._storage.load_records(session.session_id)
            (
                session,
                final_response,
                final_estimated_input_tokens,
                followup_compacted,
                interrupted,
                approval_rejected,
            ) = await self._execute_followup_tool_rounds(
                session=session,
                base_records=base_records,
                pending_records=pending_records,
                current_response=response,
                current_estimated_input_tokens=final_estimated_input_tokens,
                turn_id=turn_id,
            )
            if followup_compacted:
                did_compaction = True
            if interrupted:
                return self._interrupt_turn(
                    session_id=session.session_id,
                    turn_id=turn_id,
                    command=command_override,
                    compaction_performed=did_compaction,
                    response_text=final_response.text,
                )

            self._persist_successful_turn(
                session_id=session.session_id,
                turn_id=turn_id,
                response=final_response,
                estimated_input_tokens=final_estimated_input_tokens,
            )
            await self._reflect_completed_turn(
                session_id=session.session_id,
                turn_id=turn_id,
            )

            refreshed = self._storage.get_session(session.session_id)
            threshold_observed = (
                final_response.usage.input_tokens
                if final_response.usage is not None and final_response.usage.input_tokens is not None
                else final_estimated_input_tokens
            )
            should_enqueue_reactive = (
                threshold_observed >= self._settings.context_policy.compact_threshold_tokens
            )
            if refreshed is not None:
                self._storage.update_session(
                    refreshed.session_id,
                    pending_reactive_compaction=should_enqueue_reactive,
                )

            return AgentTurnResult(
                session_id=session.session_id,
                response_text=final_response.text,
                command=command_override,
                compaction_performed=did_compaction,
                approval_rejected=approval_rejected,
            )
        finally:
            self._clear_turn_control(turn_id)

    async def _execute_tool_calls(
        self,
        *,
        session_id: str,
        pending_records: list[ConversationRecord],
        current_response: LLMResponse,
        turn_id: str,
    ) -> _ToolExecutionOutcome:
        transient_records: list[ConversationRecord] = []
        for tool_call in current_response.tool_calls:
            tool_context = replace(
                self._tool_context,
                session_id=session_id,
                turn_id=turn_id,
            )
            while True:
                tool_result = await self._tool_executor(tool_call, tool_context)
                pending_approval = self._build_pending_approval(
                    tool_result=tool_result,
                    tool_name=tool_call.name,
                )
                if pending_approval is None:
                    self._append_turn_record(
                        session_id=session_id,
                        pending_records=pending_records,
                        record=self._build_tool_record(
                            session_id,
                            tool_result,
                            turn_id=turn_id,
                        ),
                    )
                    transient_records.extend(
                        self._build_transient_records_from_tool_result(
                            session_id,
                            tool_result,
                        )
                    )
                    break

                self._append_turn_record(
                    session_id=session_id,
                    pending_records=pending_records,
                    record=self._build_tool_record(
                        session_id,
                        tool_result,
                        metadata_overrides={
                            "approval_required": True,
                            "approval_request": pending_approval,
                        },
                        turn_id=turn_id,
                    ),
                )
                approved = await self._wait_for_approval(
                    session_id=session_id,
                    turn_id=turn_id,
                    approval=pending_approval,
                )
                if approved is None:
                    return _ToolExecutionOutcome(interrupted=True)
                self._append_turn_record(
                    session_id=session_id,
                    pending_records=pending_records,
                    record=self._build_approval_record(
                        session_id=session_id,
                        approval=pending_approval,
                        approved=approved,
                        turn_id=turn_id,
                    ),
                )
                if not approved:
                    return _ToolExecutionOutcome(approval_rejected=True)
                tool_context = replace(tool_context, approved_action=pending_approval)

        pending_records.extend(transient_records)
        return _ToolExecutionOutcome()

    def _build_followup_request(
        self,
        *,
        base_records: Sequence[ConversationRecord],
        pending_records: Sequence[ConversationRecord],
    ) -> tuple[LLMRequest, int]:
        activated_discoverable_tool_names = _collect_activated_discoverable_tool_names(
            pending_records
        )

        request = self._build_request(
            list(base_records) + list(pending_records),
            activated_discoverable_tool_names=activated_discoverable_tool_names,
        )
        estimated_input_tokens = estimate_request_input_tokens(request)
        if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
            raise ContextBudgetError(
                "Tool output exceeded the context budget during the current turn."
            )

        return request, estimated_input_tokens

    def _build_tool_round_limit_record(
        self,
        *,
        session_id: str,
        attempted_round: int,
    ) -> ConversationRecord:
        max_rounds = self._tool_settings.max_tool_rounds_per_turn
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content=(
                f"Tool round limit reached for this turn at round {attempted_round} "
                f"(max {max_rounds}). Do not call more tools. "
                "End this turn with a short status update, summarize what was completed, "
                "state what remains, and ask the user to continue if more tool work is needed."
            ),
            metadata={
                _TRANSIENT_RECORD_METADATA_KEY: True,
                _TOOL_ROUND_LIMIT_METADATA_KEY: True,
                "attempted_round": attempted_round,
                "max_rounds": max_rounds,
            },
        )

    def _build_request(
        self,
        records: Sequence[ConversationRecord],
        *,
        activated_discoverable_tool_names: Sequence[str] = (),
        allow_tools: bool = True,
    ) -> LLMRequest:
        return LLMRequest(
            messages=_records_to_llm_messages(records),
            provider=self._llm_provider,
            tools=(
                self._compose_request_tools(activated_discoverable_tool_names)
                if allow_tools
                else ()
            ),
            tool_choice=ToolChoice.auto() if allow_tools else ToolChoice.none(),
        )

    def _build_tool_round_limit_recovery_request(
        self,
        *,
        session_id: str,
        base_records: Sequence[ConversationRecord],
        pending_records: list[ConversationRecord],
        attempted_round: int,
    ) -> tuple[LLMRequest, int]:
        pending_records.append(
            self._build_tool_round_limit_record(
                session_id=session_id,
                attempted_round=attempted_round,
            )
        )
        request = self._build_request(
            list(base_records) + pending_records,
            allow_tools=False,
        )
        estimated_input_tokens = estimate_request_input_tokens(request)
        if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
            raise ContextBudgetError(
                "Tool round limit recovery request exceeded the context budget."
            )
        return request, estimated_input_tokens

    def _normalize_tool_round_limit_recovery_response(
        self,
        response: LLMResponse,
    ) -> LLMResponse:
        text = response.text if response.text.strip() else _TOOL_ROUND_LIMIT_RECOVERY_TEXT
        if text == response.text and not response.tool_calls:
            return response
        finish_reason = "stop" if response.tool_calls else response.finish_reason
        return replace(
            response,
            text=text,
            tool_calls=[],
            finish_reason=finish_reason,
        )

    async def _recover_from_tool_round_limit(
        self,
        *,
        session_id: str,
        base_records: Sequence[ConversationRecord],
        pending_records: list[ConversationRecord],
        attempted_round: int,
        turn_id: str,
    ) -> tuple[LLMResponse, int]:
        request, estimated_input_tokens = self._build_tool_round_limit_recovery_request(
            session_id=session_id,
            base_records=base_records,
            pending_records=pending_records,
            attempted_round=attempted_round,
        )
        response = await self._llm_service.generate(request)
        normalized = self._normalize_tool_round_limit_recovery_response(response)
        self._append_turn_record(
            session_id=session_id,
            pending_records=pending_records,
            record=self._build_assistant_record(
                session_id,
                normalized,
                turn_id=turn_id,
            ),
        )
        return normalized, estimated_input_tokens

    async def _stream_recover_from_tool_round_limit(
        self,
        *,
        session_id: str,
        base_records: Sequence[ConversationRecord],
        pending_records: list[ConversationRecord],
        attempted_round: int,
        turn_id: str,
    ) -> tuple[list[AgentTurnStreamEvent], LLMResponse, int]:
        request, estimated_input_tokens = self._build_tool_round_limit_recovery_request(
            session_id=session_id,
            base_records=base_records,
            pending_records=pending_records,
            attempted_round=attempted_round,
        )
        streamed_response: LLMResponse | None = None
        recovery_events: list[AgentTurnStreamEvent] = []
        async for event in self._llm_service.stream_generate(request):
            if event.type == "text_delta":
                if event.delta:
                    recovery_events.append(
                        AgentTextDeltaEvent(
                            session_id=session_id,
                            delta=event.delta,
                        )
                    )
            elif event.type == "done":
                streamed_response = event.response

        if streamed_response is None:
            raise RuntimeError(
                "Streaming tool round limit recovery completed without a final done event."
            )

        normalized = self._normalize_tool_round_limit_recovery_response(streamed_response)
        self._append_turn_record(
            session_id=session_id,
            pending_records=pending_records,
            record=self._build_assistant_record(
                session_id,
                normalized,
                turn_id=turn_id,
            ),
        )
        if normalized.text:
            recovery_events.append(
                AgentAssistantMessageEvent(
                    session_id=session_id,
                    text=normalized.text,
                )
            )
        return recovery_events, normalized, estimated_input_tokens

    async def _stream_message_turn(
        self,
        user_text: str,
        *,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        (
            session,
            _base_records,
            turn_context_text,
            interruption_notice_text,
            turn_runtime_messages,
            request,
            estimated_input_tokens,
            did_compaction,
        ) = await self._prepare_message_turn(
            user_text=user_text,
            force_session_id=force_session_id,
            pre_turn_messages=pre_turn_messages,
        )
        turn_id = uuid4().hex
        pending_records = self._build_pending_turn_records(
            session_id=session.session_id,
            turn_context_text=turn_context_text,
            interruption_notice_text=interruption_notice_text,
            runtime_messages=turn_runtime_messages,
        )
        user_record = self._build_message_record(
            session_id=session.session_id,
            role="user",
            content=user_text,
            turn_id=turn_id,
        )
        self._begin_turn(session_id=session.session_id, turn_id=turn_id)
        self._append_turn_record(
            session_id=session.session_id,
            pending_records=pending_records,
            record=user_record,
        )

        try:
            overflow_compacted = False
            overflow_retry_attempted = False
            initial_response: LLMResponse | None = None
            final_estimated_input_tokens = estimated_input_tokens
            noticed_initial_tool_call_ids: set[str] = set()
            streamed_initial_text = ""
            persisted_initial_text_prefix: str | None = None

            while True:
                streamed_response: LLMResponse | None = None
                emitted_any = False
                noticed_initial_tool_call_ids = set()
                streamed_initial_text = ""
                persisted_initial_text_prefix = None
                try:
                    async for event in self._llm_service.stream_generate(request):
                        if event.type == "text_delta":
                            emitted_any = True
                            if event.delta:
                                streamed_initial_text += event.delta
                                yield AgentTextDeltaEvent(
                                    session_id=session.session_id,
                                    delta=event.delta,
                                )
                        elif event.type == "tool_call_delta":
                            emitted_any = True
                            if persisted_initial_text_prefix is None and streamed_initial_text:
                                partial_record = self._build_streamed_assistant_text_record(
                                    session_id=session.session_id,
                                    text=streamed_initial_text,
                                    turn_id=turn_id,
                                )
                                if partial_record is not None:
                                    self._append_turn_record(
                                        session_id=session.session_id,
                                        pending_records=pending_records,
                                        record=partial_record,
                                    )
                                    persisted_initial_text_prefix = streamed_initial_text
                            tool_name = str(event.tool_name or "").strip()
                            call_id = event.call_id.strip()
                            if tool_name and call_id and call_id not in noticed_initial_tool_call_ids:
                                noticed_initial_tool_call_ids.add(call_id)
                                yield AgentToolCallEvent(
                                    session_id=session.session_id,
                                    tool_names=(tool_name,),
                                )
                            if (
                                self._stop_requested(turn_id)
                                and persisted_initial_text_prefix is not None
                                and tool_name
                                and call_id
                            ):
                                interrupted = self._interrupt_turn(
                                    session_id=session.session_id,
                                    turn_id=turn_id,
                                    command=command_override,
                                    compaction_performed=did_compaction,
                                    response_text=streamed_initial_text,
                                )
                                yield AgentTurnDoneEvent(
                                    session_id=interrupted.session_id,
                                    response_text=interrupted.response_text,
                                    command=interrupted.command,
                                    compaction_performed=interrupted.compaction_performed,
                                    interrupted=True,
                                )
                                return
                        elif event.type == "done":
                            streamed_response = event.response

                    if streamed_response is None:
                        raise RuntimeError(
                            "Streaming generation completed without a final done event."
                        )
                    initial_response = streamed_response
                    break
                except ProviderBadRequestError as exc:
                    if overflow_retry_attempted or emitted_any or not _is_context_overflow_error(exc):
                        raise

                (
                    session,
                    _base_records,
                    rebound_pending_records,
                    request,
                    retry_estimate,
                ) = await self._compact_followup_and_rebuild_request(
                    session=session,
                    pending_records=pending_records,
                    reason="overflow",
                    turn_id=turn_id,
                )
                pending_records[:] = rebound_pending_records
                final_estimated_input_tokens = retry_estimate
                overflow_compacted = True
                overflow_retry_attempted = True

            if initial_response is None:
                raise RuntimeError("Streaming generation produced no final response.")
            if overflow_compacted:
                did_compaction = True

            final_initial_record = self._build_final_stream_assistant_record(
                session_id=session.session_id,
                response=initial_response,
                turn_id=turn_id,
                persisted_text_prefix=persisted_initial_text_prefix,
            )
            if final_initial_record is not None:
                self._append_turn_record(
                    session_id=session.session_id,
                    pending_records=pending_records,
                    record=final_initial_record,
                )
            if initial_response.text:
                yield AgentAssistantMessageEvent(
                    session_id=session.session_id,
                    text=initial_response.text,
                )
            if initial_response.tool_calls:
                tool_names = _pending_tool_notice_names(
                    initial_response.tool_calls,
                    noticed_initial_tool_call_ids,
                )
                if tool_names:
                    yield AgentToolCallEvent(
                        session_id=session.session_id,
                        tool_names=tool_names,
                    )
            if self._stop_requested(turn_id):
                interrupted = self._interrupt_turn(
                    session_id=session.session_id,
                    turn_id=turn_id,
                    command=command_override,
                    compaction_performed=did_compaction,
                    response_text=initial_response.text,
                )
                yield AgentTurnDoneEvent(
                    session_id=interrupted.session_id,
                    response_text=interrupted.response_text,
                    command=interrupted.command,
                    compaction_performed=interrupted.compaction_performed,
                    interrupted=True,
                )
                return

            base_records = self._storage.load_records(session.session_id)
            current_response = initial_response
            tool_rounds = 0
            turn_approval_rejected = False
            while current_response.tool_calls:
                if self._stop_requested(turn_id):
                    interrupted = self._interrupt_turn(
                        session_id=session.session_id,
                        turn_id=turn_id,
                        command=command_override,
                        compaction_performed=did_compaction,
                        response_text=current_response.text,
                    )
                    yield AgentTurnDoneEvent(
                        session_id=interrupted.session_id,
                        response_text=interrupted.response_text,
                        command=interrupted.command,
                        compaction_performed=interrupted.compaction_performed,
                        interrupted=True,
                    )
                    return

                tool_rounds += 1
                if tool_rounds > self._tool_settings.max_tool_rounds_per_turn:
                    (
                        recovery_events,
                        final_response,
                        final_estimated_input_tokens,
                    ) = await self._stream_recover_from_tool_round_limit(
                        session_id=session.session_id,
                        base_records=base_records,
                        pending_records=pending_records,
                        attempted_round=tool_rounds,
                        turn_id=turn_id,
                    )
                    for recovery_event in recovery_events:
                        yield recovery_event
                    current_response = final_response
                    break

                followup_compaction_attempted = False
                try:
                    transient_records: list[ConversationRecord] = []
                    approval_rejected = False
                    for tool_call in current_response.tool_calls:
                        tool_context = replace(
                            self._tool_context,
                            session_id=session.session_id,
                            turn_id=turn_id,
                        )
                        while True:
                            tool_result = await self._tool_executor(tool_call, tool_context)
                            pending_approval = self._build_pending_approval(
                                tool_result=tool_result,
                                tool_name=tool_call.name,
                            )
                            if pending_approval is None:
                                self._append_turn_record(
                                    session_id=session.session_id,
                                    pending_records=pending_records,
                                    record=self._build_tool_record(
                                        session.session_id,
                                        tool_result,
                                        turn_id=turn_id,
                                    ),
                                )
                                transient_records.extend(
                                    self._build_transient_records_from_tool_result(
                                        session.session_id,
                                        tool_result,
                                    )
                                )
                                break

                            self._append_turn_record(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                record=self._build_tool_record(
                                    session.session_id,
                                    tool_result,
                                    metadata_overrides={
                                        "approval_required": True,
                                        "approval_request": pending_approval,
                                    },
                                    turn_id=turn_id,
                                ),
                            )
                            yield self._build_approval_request_event(
                                session_id=session.session_id,
                                approval=pending_approval,
                            )
                            approved = await self._wait_for_approval(
                                session_id=session.session_id,
                                turn_id=turn_id,
                                approval=pending_approval,
                            )
                            if approved is None:
                                interrupted = self._interrupt_turn(
                                    session_id=session.session_id,
                                    turn_id=turn_id,
                                    command=command_override,
                                    compaction_performed=did_compaction,
                                    response_text=current_response.text,
                                )
                                yield AgentTurnDoneEvent(
                                    session_id=interrupted.session_id,
                                    response_text=interrupted.response_text,
                                    command=interrupted.command,
                                    compaction_performed=interrupted.compaction_performed,
                                    interrupted=True,
                                )
                                return
                            self._append_turn_record(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                record=self._build_approval_record(
                                    session_id=session.session_id,
                                    approval=pending_approval,
                                    approved=approved,
                                    turn_id=turn_id,
                                ),
                            )
                            if not approved:
                                approval_rejected = True
                                turn_approval_rejected = True
                                current_response = replace(
                                    current_response,
                                    text=_APPROVAL_REJECTED_TEXT,
                                    tool_calls=[],
                                    finish_reason="stop",
                                )
                                self._append_turn_record(
                                    session_id=session.session_id,
                                    pending_records=pending_records,
                                    record=self._build_message_record(
                                        session_id=session.session_id,
                                        role="assistant",
                                        content=_APPROVAL_REJECTED_TEXT,
                                        metadata={"approval_rejected": True},
                                        turn_id=turn_id,
                                    ),
                                )
                                break
                            tool_context = replace(
                                tool_context,
                                approved_action=pending_approval,
                            )

                        if approval_rejected:
                            break

                    pending_records.extend(transient_records)
                    if approval_rejected:
                        break
                    if self._stop_requested(turn_id):
                        interrupted = self._interrupt_turn(
                            session_id=session.session_id,
                            turn_id=turn_id,
                            command=command_override,
                            compaction_performed=did_compaction,
                            response_text=current_response.text,
                        )
                        yield AgentTurnDoneEvent(
                            session_id=interrupted.session_id,
                            response_text=interrupted.response_text,
                            command=interrupted.command,
                            compaction_performed=interrupted.compaction_performed,
                            interrupted=True,
                        )
                        return
                    request, final_estimated_input_tokens = self._build_followup_request(
                        base_records=base_records,
                        pending_records=pending_records,
                    )
                except ContextBudgetError:
                    (
                        session,
                        base_records,
                        rebound_pending_records,
                        request,
                        final_estimated_input_tokens,
                    ) = await self._compact_followup_and_rebuild_request(
                        session=session,
                        pending_records=pending_records,
                        reason="followup_preflight",
                        turn_id=turn_id,
                    )
                    pending_records[:] = rebound_pending_records
                    did_compaction = True
                    followup_compaction_attempted = True

                while True:
                    streamed_response: LLMResponse | None = None
                    noticed_followup_tool_call_ids: set[str] = set()
                    emitted_any = False
                    streamed_followup_text = ""
                    persisted_followup_text_prefix: str | None = None
                    try:
                        async for event in self._llm_service.stream_generate(request):
                            if event.type == "text_delta":
                                emitted_any = True
                                if event.delta:
                                    streamed_followup_text += event.delta
                                    yield AgentTextDeltaEvent(
                                        session_id=session.session_id,
                                        delta=event.delta,
                                    )
                            elif event.type == "tool_call_delta":
                                emitted_any = True
                                if persisted_followup_text_prefix is None and streamed_followup_text:
                                    partial_record = self._build_streamed_assistant_text_record(
                                        session_id=session.session_id,
                                        text=streamed_followup_text,
                                        turn_id=turn_id,
                                    )
                                    if partial_record is not None:
                                        self._append_turn_record(
                                            session_id=session.session_id,
                                            pending_records=pending_records,
                                            record=partial_record,
                                        )
                                        persisted_followup_text_prefix = streamed_followup_text
                                tool_name = str(event.tool_name or "").strip()
                                call_id = event.call_id.strip()
                                if tool_name and call_id and call_id not in noticed_followup_tool_call_ids:
                                    noticed_followup_tool_call_ids.add(call_id)
                                    yield AgentToolCallEvent(
                                        session_id=session.session_id,
                                        tool_names=(tool_name,),
                                    )
                                if (
                                    self._stop_requested(turn_id)
                                    and persisted_followup_text_prefix is not None
                                    and tool_name
                                    and call_id
                                ):
                                    interrupted = self._interrupt_turn(
                                        session_id=session.session_id,
                                        turn_id=turn_id,
                                        command=command_override,
                                        compaction_performed=did_compaction,
                                        response_text=streamed_followup_text,
                                    )
                                    yield AgentTurnDoneEvent(
                                        session_id=interrupted.session_id,
                                        response_text=interrupted.response_text,
                                        command=interrupted.command,
                                        compaction_performed=interrupted.compaction_performed,
                                        interrupted=True,
                                    )
                                    return
                            elif event.type == "done":
                                streamed_response = event.response
                        break
                    except ProviderBadRequestError as exc:
                        if (
                            not _is_context_overflow_error(exc)
                            or emitted_any
                        ):
                            raise
                        if followup_compaction_attempted:
                            raise ContextBudgetError(
                                _FOLLOWUP_RETRY_PROVIDER_OVERFLOW_TEXT
                            ) from exc

                    (
                        session,
                        base_records,
                        rebound_pending_records,
                        request,
                        final_estimated_input_tokens,
                    ) = await self._compact_followup_and_rebuild_request(
                        session=session,
                        pending_records=pending_records,
                        reason="followup_overflow",
                        turn_id=turn_id,
                    )
                    pending_records[:] = rebound_pending_records
                    did_compaction = True
                    followup_compaction_attempted = True

                if streamed_response is None:
                    raise RuntimeError(
                        "Streaming follow-up generation completed without a final done event."
                    )

                current_response = streamed_response
                final_followup_record = self._build_final_stream_assistant_record(
                    session_id=session.session_id,
                    response=current_response,
                    turn_id=turn_id,
                    persisted_text_prefix=persisted_followup_text_prefix,
                )
                if final_followup_record is not None:
                    self._append_turn_record(
                        session_id=session.session_id,
                        pending_records=pending_records,
                        record=final_followup_record,
                    )
                if current_response.text:
                    yield AgentAssistantMessageEvent(
                        session_id=session.session_id,
                        text=current_response.text,
                    )
                if current_response.tool_calls:
                    tool_names = _pending_tool_notice_names(
                        current_response.tool_calls,
                        noticed_followup_tool_call_ids,
                    )
                    if tool_names:
                        yield AgentToolCallEvent(
                            session_id=session.session_id,
                            tool_names=tool_names,
                        )
                if self._stop_requested(turn_id):
                    interrupted = self._interrupt_turn(
                        session_id=session.session_id,
                        turn_id=turn_id,
                        command=command_override,
                        compaction_performed=did_compaction,
                        response_text=current_response.text,
                    )
                    yield AgentTurnDoneEvent(
                        session_id=interrupted.session_id,
                        response_text=interrupted.response_text,
                        command=interrupted.command,
                        compaction_performed=interrupted.compaction_performed,
                        interrupted=True,
                    )
                    return

            final_response = current_response

            self._persist_successful_turn(
                session_id=session.session_id,
                turn_id=turn_id,
                response=final_response,
                estimated_input_tokens=final_estimated_input_tokens,
            )

            refreshed = self._storage.get_session(session.session_id)
            threshold_observed = (
                final_response.usage.input_tokens
                if final_response.usage is not None and final_response.usage.input_tokens is not None
                else final_estimated_input_tokens
            )
            should_enqueue_reactive = (
                threshold_observed >= self._settings.context_policy.compact_threshold_tokens
            )
            if refreshed is not None:
                self._storage.update_session(
                    refreshed.session_id,
                    pending_reactive_compaction=should_enqueue_reactive,
                )

            yield AgentTurnDoneEvent(
                session_id=session.session_id,
                response_text=final_response.text,
                command=command_override,
                compaction_performed=did_compaction,
                approval_rejected=turn_approval_rejected,
            )
        finally:
            self._clear_turn_control(turn_id)

    async def _prepare_message_turn(
        self,
        *,
        user_text: str,
        force_session_id: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> tuple[
        SessionMetadata,
        list[ConversationRecord],
        str,
        str | None,
        tuple[AgentRuntimeMessage, ...],
        LLMRequest,
        int,
        bool,
    ]:
        await self._ensure_memory_runtime_ready()
        turn_context_text = self._build_turn_context_text()
        session = self._storage.get_session(force_session_id) if force_session_id else None
        if session is None:
            session = await self._ensure_active_session()

        did_compaction = False
        if session.pending_reactive_compaction:
            compacted = await self._compact_session(session, reason="reactive")
            if compacted is not None:
                session = compacted
                did_compaction = True

        records = self._storage.load_records(session.session_id)
        interruption_notice_text = (
            _PREVIOUS_TASK_INTERRUPTED_TEXT
            if session.pending_interruption_notice
            else None
        )
        turn_runtime_messages = self._build_turn_runtime_messages(
            session_id=session.session_id,
            pre_turn_messages=pre_turn_messages,
        )
        request = self._build_turn_request(
            session_id=session.session_id,
            records=records,
            user_text=user_text,
            turn_context_text=turn_context_text,
            interruption_notice_text=interruption_notice_text,
            runtime_messages=turn_runtime_messages,
        )
        estimated_input_tokens = estimate_request_input_tokens(request)

        if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
            compacted = await self._compact_session(session, reason="preflight")
            if compacted is not None:
                session = compacted
                did_compaction = True
                records = self._storage.load_records(session.session_id)
                interruption_notice_text = (
                    _PREVIOUS_TASK_INTERRUPTED_TEXT
                    if session.pending_interruption_notice
                    else None
                )
                request = self._build_turn_request(
                    session_id=session.session_id,
                    records=records,
                    user_text=user_text,
                    turn_context_text=turn_context_text,
                    interruption_notice_text=interruption_notice_text,
                    runtime_messages=turn_runtime_messages,
                )
                estimated_input_tokens = estimate_request_input_tokens(request)

        if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
            raise ContextBudgetError(
                "Request is still over the preflight context budget after compaction."
            )

        return (
            session,
            records,
            turn_context_text,
            interruption_notice_text,
            turn_runtime_messages,
            request,
            estimated_input_tokens,
            did_compaction,
        )

    async def _generate_with_overflow_retry(
        self,
        *,
        session: SessionMetadata,
        user_text: str,
        turn_context_text: str,
        interruption_notice_text: str | None,
        request: LLMRequest,
        estimated_input_tokens: int,
        pending_records: list[ConversationRecord],
        turn_id: str,
    ) -> tuple[SessionMetadata, LLMResponse, bool, int, list[ConversationRecord]]:
        try:
            response = await self._llm_service.generate(request)
            return session, response, False, estimated_input_tokens, pending_records
        except ProviderBadRequestError as exc:
            if not _is_context_overflow_error(exc):
                raise

        (
            compacted,
            records,
            rebound_pending_records,
            retry_request,
            retry_estimate,
        ) = await self._compact_followup_and_rebuild_request(
            session=session,
            pending_records=pending_records,
            reason="overflow",
            turn_id=turn_id,
        )
        response = await self._llm_service.generate(retry_request)
        return compacted, response, True, retry_estimate, rebound_pending_records

    def _persist_successful_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        response: LLMResponse,
        estimated_input_tokens: int,
    ) -> None:
        self._finish_turn(
            session_id=session_id,
            turn_id=turn_id,
            status="completed",
        )
        usage = response.usage
        self._storage.update_session(
            session_id,
            pending_interruption_notice=False,
            last_input_tokens=usage.input_tokens if usage is not None else None,
            last_output_tokens=usage.output_tokens if usage is not None else None,
            last_total_tokens=usage.total_tokens if usage is not None else None,
            last_estimated_input_tokens=estimated_input_tokens,
        )

    def _compose_request_tools(
        self,
        activated_discoverable_tool_names: Sequence[str],
    ) -> tuple[ToolDefinition, ...]:
        return self._tool_definitions_provider(activated_discoverable_tool_names)

    def _build_turn_request(
        self,
        *,
        session_id: str,
        records: Sequence[ConversationRecord],
        user_text: str,
        turn_context_text: str,
        interruption_notice_text: str | None = None,
        runtime_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> LLMRequest:
        turn_records = self._build_pending_turn_records(
            session_id=session_id,
            turn_context_text=turn_context_text,
            interruption_notice_text=interruption_notice_text,
            runtime_messages=runtime_messages,
        )
        turn_records.append(
            self._build_message_record(
                session_id=session_id,
                role="user",
                content=user_text,
            )
        )
        return self._build_request(list(records) + turn_records)

    def _build_turn_runtime_messages(
        self,
        *,
        session_id: str,
        pre_turn_messages: Sequence[AgentRuntimeMessage],
    ) -> tuple[AgentRuntimeMessage, ...]:
        runtime_messages: list[AgentRuntimeMessage] = []
        provider = self._runtime_messages_provider
        if provider is not None:
            runtime_messages.extend(provider(session_id))
        runtime_messages.extend(pre_turn_messages)
        return tuple(runtime_messages)

    def _build_pending_turn_records(
        self,
        *,
        session_id: str,
        turn_context_text: str,
        interruption_notice_text: str | None,
        runtime_messages: Sequence[AgentRuntimeMessage],
    ) -> list[ConversationRecord]:
        pending_records = [
            self._build_turn_context_record(
                session_id=session_id,
                turn_context_text=turn_context_text,
            )
        ]
        for message in runtime_messages:
            pending_records.append(
                self._build_runtime_message_record(
                    session_id=session_id,
                    message=message,
                )
            )
        if interruption_notice_text is not None:
            pending_records.append(
                self._build_interruption_notice_record(
                    session_id=session_id,
                    text=interruption_notice_text,
                )
            )
        return pending_records

    async def _execute_followup_tool_rounds(
        self,
        *,
        session: SessionMetadata,
        base_records: Sequence[ConversationRecord],
        pending_records: list[ConversationRecord],
        current_response: LLMResponse,
        current_estimated_input_tokens: int,
        turn_id: str,
    ) -> tuple[SessionMetadata, LLMResponse, int, bool, bool, bool]:
        tool_rounds = 0
        did_compaction = False
        approval_rejected = False
        current_session = session
        current_base_records = list(base_records)

        while current_response.tool_calls:
            if self._stop_requested(turn_id):
                return (
                    current_session,
                    current_response,
                    current_estimated_input_tokens,
                    did_compaction,
                    True,
                    approval_rejected,
                )
            tool_rounds += 1
            if tool_rounds > self._tool_settings.max_tool_rounds_per_turn:
                current_response, current_estimated_input_tokens = (
                    await self._recover_from_tool_round_limit(
                        session_id=current_session.session_id,
                        base_records=current_base_records,
                        pending_records=pending_records,
                        attempted_round=tool_rounds,
                        turn_id=turn_id,
                    )
                )
                break

            followup_compaction_attempted = False
            try:
                tool_execution_outcome = await self._execute_tool_calls(
                    session_id=current_session.session_id,
                    pending_records=pending_records,
                    current_response=current_response,
                    turn_id=turn_id,
                )
                if tool_execution_outcome.interrupted:
                    return (
                        current_session,
                        current_response,
                        current_estimated_input_tokens,
                        did_compaction,
                        True,
                        approval_rejected,
                    )
                if tool_execution_outcome.approval_rejected:
                    approval_rejected = True
                    current_response = replace(
                        current_response,
                        text=_APPROVAL_REJECTED_TEXT,
                        tool_calls=[],
                        finish_reason="stop",
                    )
                    self._append_turn_record(
                        session_id=current_session.session_id,
                        pending_records=pending_records,
                        record=self._build_message_record(
                            session_id=current_session.session_id,
                            role="assistant",
                            content=_APPROVAL_REJECTED_TEXT,
                            metadata={"approval_rejected": True},
                            turn_id=turn_id,
                        ),
                    )
                    break
                if self._stop_requested(turn_id):
                    return (
                        current_session,
                        current_response,
                        current_estimated_input_tokens,
                        did_compaction,
                        True,
                        approval_rejected,
                    )
                request, current_estimated_input_tokens = self._build_followup_request(
                    base_records=current_base_records,
                    pending_records=pending_records,
                )
            except ContextBudgetError:
                (
                    current_session,
                    current_base_records,
                    rebound_pending_records,
                    request,
                    current_estimated_input_tokens,
                ) = await self._compact_followup_and_rebuild_request(
                    session=current_session,
                    pending_records=pending_records,
                    reason="followup_preflight",
                    turn_id=turn_id,
                )
                pending_records[:] = rebound_pending_records
                did_compaction = True
                followup_compaction_attempted = True

            while True:
                try:
                    current_response = await self._llm_service.generate(request)
                    break
                except ProviderBadRequestError as exc:
                    if not _is_context_overflow_error(exc):
                        raise
                    if followup_compaction_attempted:
                        raise ContextBudgetError(
                            _FOLLOWUP_RETRY_PROVIDER_OVERFLOW_TEXT
                        ) from exc

                (
                    current_session,
                    current_base_records,
                    rebound_pending_records,
                    request,
                    current_estimated_input_tokens,
                ) = await self._compact_followup_and_rebuild_request(
                    session=current_session,
                    pending_records=pending_records,
                    reason="followup_overflow",
                    turn_id=turn_id,
                )
                pending_records[:] = rebound_pending_records
                did_compaction = True
                followup_compaction_attempted = True

            self._append_turn_record(
                session_id=current_session.session_id,
                pending_records=pending_records,
                record=self._build_assistant_record(
                    current_session.session_id,
                    current_response,
                    turn_id=turn_id,
                ),
            )

        return (
            current_session,
            current_response,
            current_estimated_input_tokens,
            did_compaction,
            False,
            approval_rejected,
        )

    async def _compact_followup_and_rebuild_request(
        self,
        *,
        session: SessionMetadata,
        pending_records: Sequence[ConversationRecord],
        reason: str,
        turn_id: str,
    ) -> tuple[SessionMetadata, list[ConversationRecord], list[ConversationRecord], LLMRequest, int]:
        compacted = await self._compact_session(
            session,
            reason=reason,
            include_turn_ids=(turn_id,),
        )
        if compacted is None:
            raise ContextBudgetError(_FOLLOWUP_COMPACTION_FAILED_TEXT)

        stop_requested = self._stop_requested(turn_id)
        self._storage.set_turn_status(
            session.session_id,
            turn_id=turn_id,
            status="superseded",
        )
        self._storage.set_turn_status(
            compacted.session_id,
            turn_id=turn_id,
            status="in_progress",
        )
        rebound_pending_records = [
            self._clone_carry_forward_record_for_session(compacted.session_id, record)
            for record in pending_records
        ]
        base_records = self._storage.load_records(compacted.session_id)
        activated_discoverable_tool_names = _collect_activated_discoverable_tool_names(
            rebound_pending_records
        )
        request = self._build_request(
            list(base_records) + rebound_pending_records,
            activated_discoverable_tool_names=activated_discoverable_tool_names,
        )
        estimated_input_tokens = estimate_request_input_tokens(request)
        if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
            rebound_pending_records = [
                self._strongly_compact_carry_forward_record(record)
                for record in rebound_pending_records
            ]
            activated_discoverable_tool_names = _collect_activated_discoverable_tool_names(
                rebound_pending_records
            )
            request = self._build_request(
                list(base_records) + rebound_pending_records,
                activated_discoverable_tool_names=activated_discoverable_tool_names,
            )
            estimated_input_tokens = estimate_request_input_tokens(request)
            if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
                raise ContextBudgetError(_FOLLOWUP_RETRY_PREFLIGHT_FAILED_TEXT)

        for record in rebound_pending_records:
            if bool(record.metadata.get(_TRANSIENT_RECORD_METADATA_KEY, False)):
                continue
            self._storage.append_record(compacted.session_id, record)
        self._active_turn_id = turn_id
        if stop_requested:
            self._stop_requested_turn_id = turn_id
        return (
            compacted,
            list(base_records),
            rebound_pending_records,
            request,
            estimated_input_tokens,
        )

    async def _ensure_active_session(self) -> SessionMetadata:
        active = self._storage.get_active_session()
        if active is not None:
            return active
        return await self._start_session(start_reason="initial")

    async def _start_session(
        self,
        *,
        start_reason: str,
        parent_session_id: str | None = None,
        summary_text: str | None = None,
        compaction_count: int = 0,
    ) -> SessionMetadata:
        session = self._storage.create_session(
            parent_session_id=parent_session_id,
            start_reason=start_reason,
        )

        bootstrap_messages = self._identity_loader.load_bootstrap_messages()
        for message in bootstrap_messages:
            self._append_message(
                session_id=session.session_id,
                role=message.role,
                content=message.parts[0].text,
                metadata={"bootstrap_identity": True},
            )
        tool_bootstrap = self._serialize_basic_tool_bootstrap()
        if tool_bootstrap is not None:
            self._append_message(
                session_id=session.session_id,
                role="developer",
                content=json.dumps(tool_bootstrap, ensure_ascii=False, indent=2),
                metadata={
                    _TOOL_BOOTSTRAP_METADATA_KEY: "basic",
                    _TRANSCRIPT_ONLY_RECORD_METADATA_KEY: True,
                    "tool_definitions": tool_bootstrap,
                },
            )

        if self._memory_mode.bootstrap:
            try:
                core_memory_bootstrap, ongoing_memory_bootstrap = (
                    await self._memory_service.render_bootstrap_messages()
                )
            except Exception:
                LOGGER.exception("Memory bootstrap rendering failed.")
                core_memory_bootstrap, ongoing_memory_bootstrap = "", ""
            if core_memory_bootstrap.strip():
                self._append_message(
                    session_id=session.session_id,
                    role="developer",
                    content="Runtime core memory bootstrap:\n\n" + core_memory_bootstrap,
                    metadata={"memory_bootstrap": "core"},
                )
            if ongoing_memory_bootstrap.strip():
                self._append_message(
                    session_id=session.session_id,
                    role="developer",
                    content="Runtime ongoing memory bootstrap:\n\n" + ongoing_memory_bootstrap,
                    metadata={"memory_bootstrap": "ongoing"},
                )
        if summary_text:
            self._append_message(
                session_id=session.session_id,
                role="developer",
                content=(
                    "Summarized context from previous session compaction.\n"
                    "Use this as prior conversational state:\n\n"
                    f"{summary_text.strip()}"
                ),
                metadata={"summary_seed": True},
            )

        if compaction_count > 0:
            session = self._storage.update_session(
                session.session_id,
                compaction_count=compaction_count,
            )
        else:
            session = self._storage.get_session(session.session_id) or session
        return session

    async def _compact_session(
        self,
        session: SessionMetadata,
        *,
        reason: str,
        user_instruction: str | None = None,
        include_turn_ids: tuple[str, ...] = (),
    ) -> SessionMetadata | None:
        records = self._storage.load_records(
            session.session_id,
            include_turn_ids=include_turn_ids,
        )
        if self._memory_mode.maintenance:
            try:
                await self._memory_service.flush_before_compaction(
                    route_id=self._tool_context.route_id,
                    session_id=session.session_id,
                    records=tuple(records),
                )
            except Exception:
                LOGGER.exception("Memory pre-compaction flush failed.")
        compactable_records = [
            record
            for record in records
            if record.kind == "message"
            and not record.metadata.get("bootstrap_identity", False)
            and not record.metadata.get(_TRANSCRIPT_ONLY_RECORD_METADATA_KEY, False)
            and not record.metadata.get("memory_bootstrap")
            and not record.metadata.get("summary_seed", False)
        ]
        if not compactable_records:
            self._storage.update_session(session.session_id, pending_reactive_compaction=False)
            return None

        outcome = await self._compactor.compact(
            compactable_records,
            user_instruction=user_instruction,
        )
        self._append_compaction_record(session.session_id, outcome=outcome, reason=reason)
        self._storage.archive_session(session.session_id)

        next_compaction_count = session.compaction_count + 1
        next_session = await self._start_session(
            start_reason="compaction",
            parent_session_id=session.session_id,
            summary_text=outcome.summary_text,
            compaction_count=next_compaction_count,
        )
        self._storage.update_session(
            next_session.session_id,
            pending_reactive_compaction=False,
            pending_interruption_notice=session.pending_interruption_notice,
        )
        return self._storage.get_session(next_session.session_id) or next_session

    async def _ensure_memory_runtime_ready(self) -> None:
        if not self._memory_mode.maintenance:
            return
        try:
            await self._memory_service.ensure_index_synced()
            await self._memory_service.run_due_maintenance()
        except Exception:
            LOGGER.exception("Memory runtime maintenance failed.")

    async def _reflect_completed_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> None:
        if not self._memory_mode.reflection:
            return
        try:
            turn_records = tuple(
                record
                for record in self._storage.load_records(session_id)
                if str(record.metadata.get(_TURN_ID_METADATA_KEY, "")).strip() == turn_id
            )
            if not turn_records:
                return
            await self._memory_service.reflect_completed_turn(
                route_id=self._tool_context.route_id,
                session_id=session_id,
                records=turn_records,
            )
        except Exception:
            LOGGER.exception("Memory post-turn reflection failed.")

    def _append_compaction_record(
        self,
        session_id: str,
        *,
        outcome: CompactionOutcome,
        reason: str,
    ) -> None:
        metadata = {
            "reason": reason,
            "provider": outcome.provider,
            "model": outcome.model,
            "response_id": outcome.response_id,
            "usage": {
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
                "total_tokens": outcome.total_tokens,
            },
        }
        record = ConversationRecord(
            record_id=uuid4().hex,
            session_id=session_id,
            created_at=_utc_now_iso(),
            role="system",
            content=outcome.summary_text,
            kind="compaction",
            metadata=metadata,
        )
        self._storage.append_record(session_id, record)

    def _build_message_record(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> ConversationRecord:
        resolved_metadata = dict(metadata or {})
        if turn_id is not None:
            resolved_metadata[_TURN_ID_METADATA_KEY] = turn_id
        return ConversationRecord(
            record_id=uuid4().hex,
            session_id=session_id,
            created_at=_utc_now_iso(),
            role=role,  # type: ignore[arg-type]
            content=content,
            kind="message",
            metadata=resolved_metadata,
        )

    def _build_assistant_record(
        self,
        session_id: str,
        response: LLMResponse,
        *,
        turn_id: str | None = None,
    ) -> ConversationRecord:
        assistant_metadata: dict[str, Any] = {
            "provider": response.provider,
            "model": response.model,
            "response_id": response.response_id,
            "finish_reason": response.finish_reason,
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "raw_arguments": call.raw_arguments,
                    "provider_metadata": dict(call.provider_metadata),
                }
                for call in response.tool_calls
            ],
        }
        return self._build_message_record(
            session_id=session_id,
            role="assistant",
            content=response.text,
            metadata=assistant_metadata,
            turn_id=turn_id,
        )

    def _build_streamed_assistant_text_record(
        self,
        *,
        session_id: str,
        text: str,
        turn_id: str,
    ) -> ConversationRecord | None:
        if not text:
            return None
        return self._build_message_record(
            session_id=session_id,
            role="assistant",
            content=text,
            metadata={"stream_checkpoint": "text_before_tool_call"},
            turn_id=turn_id,
        )

    def _build_final_stream_assistant_record(
        self,
        *,
        session_id: str,
        response: LLMResponse,
        turn_id: str,
        persisted_text_prefix: str | None,
    ) -> ConversationRecord | None:
        normalized = response
        if persisted_text_prefix:
            if response.text == persisted_text_prefix:
                if not response.tool_calls:
                    return None
                normalized = replace(response, text="")
            elif response.text.startswith(persisted_text_prefix):
                normalized = replace(
                    response,
                    text=response.text[len(persisted_text_prefix):],
                )
        if not normalized.text and not normalized.tool_calls:
            return None
        return self._build_assistant_record(
            session_id,
            normalized,
            turn_id=turn_id,
        )

    def _build_tool_record(
        self,
        session_id: str,
        result: ToolExecutionResult,
        *,
        metadata_overrides: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> ConversationRecord:
        metadata = dict(result.metadata)
        if metadata_overrides is not None:
            metadata.update(metadata_overrides)
        metadata.update(
            {
                "tool_name": result.name,
                "call_id": result.call_id,
                "ok": result.ok,
            }
        )
        return self._build_message_record(
            session_id=session_id,
            role="tool",
            content=result.content,
            metadata=metadata,
            turn_id=turn_id,
        )

    def _build_turn_context_record(
        self,
        *,
        session_id: str,
        turn_context_text: str,
    ) -> ConversationRecord:
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content=turn_context_text,
            metadata={
                _TRANSIENT_RECORD_METADATA_KEY: True,
                _TURN_CONTEXT_METADATA_KEY: "datetime",
            },
        )

    def _build_interruption_notice_record(
        self,
        *,
        session_id: str,
        text: str,
    ) -> ConversationRecord:
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content=text,
            metadata={
                _TRANSIENT_RECORD_METADATA_KEY: True,
                _INTERRUPTION_NOTICE_METADATA_KEY: True,
            },
        )

    def _build_runtime_message_record(
        self,
        *,
        session_id: str,
        message: AgentRuntimeMessage,
    ) -> ConversationRecord:
        metadata = dict(message.metadata)
        if message.transient:
            metadata[_TRANSIENT_RECORD_METADATA_KEY] = True
        return self._build_message_record(
            session_id=session_id,
            role=message.role,
            content=message.content,
            metadata=metadata,
        )

    def _serialize_basic_tool_bootstrap(self) -> list[dict[str, Any]] | None:
        definitions = self._tool_registry.basic_definitions()
        if not definitions:
            return None
        return [
            {
                "name": definition.name,
                "description": definition.description,
                "input_schema": deepcopy(dict(definition.input_schema)),
                "strict": definition.strict,
            }
            for definition in definitions
        ]

    def _default_tool_definitions(
        self,
        activated_discoverable_tool_names: Sequence[str],
    ) -> tuple[ToolDefinition, ...]:
        tools = list(self._tool_registry.basic_definitions())
        seen_names = {tool.name for tool in tools}
        for tool in self._tool_registry.resolve_discoverable_tool_definitions(
            activated_discoverable_tool_names
        ):
            if tool.name in seen_names:
                continue
            tools.append(tool)
            seen_names.add(tool.name)
        return tuple(tools)

    async def _default_execute_tool_call(
        self,
        tool_call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        return await self._tool_runtime.execute(
            tool_call=tool_call,
            context=context,
        )

    def _build_transient_records_from_tool_result(
        self,
        session_id: str,
        result: ToolExecutionResult,
    ) -> list[ConversationRecord]:
        attachment = result.metadata.get("image_attachment")
        if not isinstance(attachment, dict):
            return []

        path = str(attachment.get("path", "")).strip()
        media_type = str(attachment.get("media_type", "")).strip()
        detail = str(attachment.get("detail", "auto")).strip() or "auto"
        if not path or not media_type:
            return []

        content = (
            "Attached image from a local workspace file requested via view_image.\n"
            f"path: {path}\n"
            f"media_type: {media_type}"
        )
        return [
            self._build_message_record(
                session_id=session_id,
                role="user",
                content=content,
                metadata={
                    _TRANSIENT_RECORD_METADATA_KEY: True,
                    _IMAGE_INPUT_METADATA_KEY: {
                        "path": path,
                        "media_type": media_type,
                        "detail": detail,
                    },
                    "source_tool": result.name,
                },
            )
        ]

    def _append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = self._build_message_record(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata,
        )
        self._storage.append_record(session_id, record)

    def _begin_turn(self, *, session_id: str, turn_id: str) -> None:
        self._storage.set_turn_status(
            session_id,
            turn_id=turn_id,
            status="in_progress",
        )
        self._active_turn_id = turn_id
        self._stop_requested_turn_id = None

    def _finish_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        status: Literal["completed", "interrupted", "superseded"],
    ) -> None:
        self._storage.set_turn_status(
            session_id,
            turn_id=turn_id,
            status=status,
        )
        if self._active_turn_id == turn_id:
            self._active_turn_id = None
        if self._stop_requested_turn_id == turn_id:
            self._stop_requested_turn_id = None

    def _stop_requested(self, turn_id: str) -> bool:
        return self._stop_requested_turn_id == turn_id

    def _clear_turn_control(self, turn_id: str) -> None:
        if self._active_turn_id == turn_id:
            self._active_turn_id = None
        if self._stop_requested_turn_id == turn_id:
            self._stop_requested_turn_id = None
        if self._pending_approval_turn_id == turn_id:
            future = self._pending_approval_future
            if future is not None and not future.done():
                future.cancel()
            self._pending_approval_future = None
            self._pending_approval_id = None
            self._pending_approval_turn_id = None

    def _build_approval_request_event(
        self,
        *,
        session_id: str,
        approval: dict[str, Any],
    ) -> AgentApprovalRequestEvent:
        return AgentApprovalRequestEvent(
            session_id=session_id,
            approval_id=str(approval["approval_id"]),
            kind=str(approval.get("kind", "approval")).strip() or "approval",
            summary=str(approval.get("summary", "")).strip(),
            details=str(approval.get("details", "")).strip(),
            command=(
                str(approval["command"])
                if approval.get("command") is not None
                else None
            ),
            tool_name=(
                str(approval["tool_name"])
                if approval.get("tool_name") is not None
                else None
            ),
            inspection_url=(
                str(approval["inspection_url"])
                if approval.get("inspection_url") is not None
                else None
            ),
        )

    def _build_approval_record(
        self,
        *,
        session_id: str,
        approval: dict[str, Any],
        approved: bool,
        turn_id: str,
    ) -> ConversationRecord:
        status = "approved" if approved else "rejected"
        lines = [
            f"Approval {status}",
            f"approval_id: {approval['approval_id']}",
        ]
        tool_name = str(approval.get("tool_name", "")).strip()
        if tool_name:
            lines.append(f"tool_name: {tool_name}")
        command = str(approval.get("command", "")).strip()
        if command:
            lines.append(f"command: {command}")
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content="\n".join(lines),
            metadata={
                "approval_event": True,
                "approval_id": approval["approval_id"],
                "approved": approved,
                "tool_name": tool_name or None,
                "command": command or None,
            },
            turn_id=turn_id,
        )

    def _build_pending_approval(
        self,
        *,
        tool_result: ToolExecutionResult,
        tool_name: str,
    ) -> dict[str, Any] | None:
        raw_request = tool_result.metadata.get("approval_request")
        if not isinstance(raw_request, dict):
            return None
        pending = dict(raw_request)
        pending["approval_id"] = uuid4().hex
        pending["tool_name"] = str(pending.get("tool_name", "")).strip() or tool_name
        return pending

    async def _wait_for_approval(
        self,
        *,
        session_id: str,
        turn_id: str,
        approval: dict[str, Any],
    ) -> bool | None:
        if self._pending_approval_future is not None and not self._pending_approval_future.done():
            raise RuntimeError("An approval request is already pending for this route.")

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_approval_future = future
        self._pending_approval_id = str(approval["approval_id"])
        self._pending_approval_turn_id = turn_id
        self._storage.update_session(session_id, pending_approval=dict(approval))

        try:
            while True:
                if self._stop_requested(turn_id):
                    return None
                try:
                    return bool(
                        await asyncio.wait_for(asyncio.shield(future), timeout=0.2)
                    )
                except asyncio.TimeoutError:
                    continue
        finally:
            self._storage.update_session(session_id, pending_approval=None)
            self._pending_approval_future = None
            self._pending_approval_id = None
            self._pending_approval_turn_id = None

    def _append_turn_record(
        self,
        *,
        session_id: str,
        pending_records: list[ConversationRecord],
        record: ConversationRecord,
    ) -> None:
        pending_records.append(record)
        if bool(record.metadata.get(_TRANSIENT_RECORD_METADATA_KEY, False)):
            return
        self._storage.append_record(session_id, record)

    def _interrupt_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        command: str | None,
        compaction_performed: bool,
        response_text: str,
    ) -> AgentTurnResult:
        interruption_record = self._build_message_record(
            session_id=session_id,
            role="system",
            content=_TURN_INTERRUPTED_RECORD_TEXT,
            metadata={"interrupted_by_user": True},
            turn_id=turn_id,
        )
        self._storage.append_record(session_id, interruption_record)
        self._finish_turn(
            session_id=session_id,
            turn_id=turn_id,
            status="interrupted",
        )
        self._storage.update_session(
            session_id,
            pending_interruption_notice=True,
        )
        return AgentTurnResult(
            session_id=session_id,
            response_text=response_text,
            command=command,
            compaction_performed=compaction_performed,
            interrupted=True,
        )

    def _clone_record_for_session(
        self,
        session_id: str,
        record: ConversationRecord,
    ) -> ConversationRecord:
        return ConversationRecord(
            record_id=uuid4().hex,
            session_id=session_id,
            created_at=_utc_now_iso(),
            role=record.role,
            content=record.content,
            kind=record.kind,
            metadata=deepcopy(record.metadata),
        )

    def _clone_carry_forward_record_for_session(
        self,
        session_id: str,
        record: ConversationRecord,
    ) -> ConversationRecord:
        cloned = self._clone_record_for_session(session_id, record)
        return self._compact_carry_forward_record(cloned)

    def _compact_carry_forward_record(
        self,
        record: ConversationRecord,
    ) -> ConversationRecord:
        limit = _carry_forward_soft_limit(record)
        if limit is None or len(record.content) <= limit:
            return record
        return _copy_record_with_content(
            record,
            content=_truncate_carry_forward_text(record.content, limit=limit),
            metadata_updates={
                "carry_forward_compacted": True,
                "carry_forward_compaction_strength": "soft",
            },
        )

    def _strongly_compact_carry_forward_record(
        self,
        record: ConversationRecord,
    ) -> ConversationRecord:
        if bool(record.metadata.get(_TRANSIENT_RECORD_METADATA_KEY, False)):
            return record
        if record.role not in {"user", "assistant", "tool"}:
            return record
        if not record.content:
            return record
        return _copy_record_with_content(
            record,
            content=_strong_carry_forward_text(record),
            metadata_updates={
                "carry_forward_compacted": True,
                "carry_forward_compaction_strength": "strong",
            },
        )

    def _build_turn_context_text(self) -> str:
        current_time = datetime.now(ZoneInfo(self._settings.turn_timezone))
        date_text = f"{current_time.strftime('%B')} {current_time.day}, {current_time.year}"
        time_text = current_time.strftime("%H:%M")
        timezone_text = self._settings.turn_timezone
        return (
            "System context auto-appended for this turn only. "
            "This is not part of the user's message.\n"
            f"Current date/time: {date_text} | {time_text} | {timezone_text} time"
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_context_overflow_error(exc: ProviderBadRequestError) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _OVERFLOW_ERROR_HINTS)


def _records_to_llm_messages(
    records: Sequence[ConversationRecord],
) -> tuple[LLMMessage, ...]:
    messages: list[LLMMessage] = []
    pending_assistant: ConversationRecord | None = None
    pending_tool_records: list[ConversationRecord] = []
    pending_call_ids: set[str] = set()
    pending_tool_names: tuple[str, ...] = ()

    def _append_record(
        record: ConversationRecord,
        *,
        include_tool_calls: bool = True,
    ) -> None:
        llm_message = _record_to_llm_message(
            record,
            include_tool_calls=include_tool_calls,
        )
        if llm_message is not None:
            messages.append(llm_message)

    def _clear_pending() -> None:
        nonlocal pending_assistant, pending_tool_records, pending_call_ids, pending_tool_names
        pending_assistant = None
        pending_tool_records = []
        pending_call_ids = set()
        pending_tool_names = ()

    def _flush_resolved_pending() -> None:
        if pending_assistant is None:
            return
        _append_record(pending_assistant, include_tool_calls=True)
        for tool_record in pending_tool_records:
            _append_record(tool_record)
        _clear_pending()

    def _flush_unresolved_pending() -> None:
        if pending_assistant is None:
            return
        _append_record(pending_assistant, include_tool_calls=False)
        messages.append(
            _build_unexecuted_tool_call_note_message(
                pending_tool_names,
            )
        )
        _clear_pending()

    for record in records:
        if record.kind != "message":
            continue

        call_specs = _assistant_tool_call_specs(record)
        if pending_assistant is None:
            if call_specs:
                pending_assistant = record
                pending_tool_records = []
                pending_call_ids = {call_id for call_id, _name in call_specs}
                pending_tool_names = tuple(_ordered_unique_names(name for _call_id, name in call_specs))
                continue

            _append_record(record)
            continue

        if record.role == "tool":
            call_id = str(record.metadata.get("call_id", "")).strip()
            if call_id and call_id in pending_call_ids:
                pending_tool_records.append(record)
                pending_call_ids.remove(call_id)
                if not pending_call_ids:
                    _flush_resolved_pending()
                continue

        _flush_unresolved_pending()
        if call_specs:
            pending_assistant = record
            pending_tool_records = []
            pending_call_ids = {call_id for call_id, _name in call_specs}
            pending_tool_names = tuple(_ordered_unique_names(name for _call_id, name in call_specs))
            continue
        _append_record(record)

    if pending_assistant is not None:
        _flush_unresolved_pending()

    return tuple(messages)


def _record_to_llm_message(
    record: ConversationRecord,
    *,
    include_tool_calls: bool = True,
) -> LLMMessage | None:
    if bool(record.metadata.get(_TRANSCRIPT_ONLY_RECORD_METADATA_KEY, False)):
        return None

    if record.role in {"system", "developer", "user"}:
        parts: list[ImagePart | TextPart] = []
        image_part = _record_image_part(record)
        if image_part is not None:
            parts.append(image_part)
        if record.content:
            parts.append(TextPart(text=record.content))
        if not parts:
            return None
        return LLMMessage(role=record.role, parts=tuple(parts))

    if record.role == "assistant":
        parts: list[TextPart | ToolCall] = []
        if record.content:
            parts.append(TextPart(text=record.content))
        if include_tool_calls:
            for tool_call in record.metadata.get("tool_calls", []):
                if not isinstance(tool_call, dict):
                    continue
                call_id = str(tool_call.get("call_id", "")).strip()
                name = str(tool_call.get("name", "")).strip()
                raw_arguments = str(tool_call.get("raw_arguments", "")).strip()
                arguments = tool_call.get("arguments", {})
                provider_metadata = tool_call.get("provider_metadata", {})
                if not call_id or not name or not raw_arguments or not isinstance(arguments, dict):
                    continue
                if not isinstance(provider_metadata, dict):
                    provider_metadata = {}
                parts.append(
                    ToolCall(
                        call_id=call_id,
                        name=name,
                        arguments=dict(arguments),
                        raw_arguments=raw_arguments,
                        provider_metadata=dict(provider_metadata),
                    )
                )
        if not parts:
            return None
        return LLMMessage(role="assistant", parts=tuple(parts))

    if record.role == "tool":
        call_id = str(record.metadata.get("call_id", "")).strip()
        tool_name = str(record.metadata.get("tool_name", "")).strip()
        if not call_id or not tool_name:
            return None
        return LLMMessage(
            role="tool",
            parts=(
                ToolResultPart(
                    call_id=call_id,
                    name=tool_name,
                    content=record.content,
                    is_error=not bool(record.metadata.get("ok", False)),
                ),
            ),
        )

    return None


def _assistant_tool_call_specs(
    record: ConversationRecord,
) -> tuple[tuple[str, str], ...]:
    if record.role != "assistant":
        return ()

    specs: list[tuple[str, str]] = []
    for tool_call in record.metadata.get("tool_calls", []):
        if not isinstance(tool_call, dict):
            continue
        call_id = str(tool_call.get("call_id", "")).strip()
        name = str(tool_call.get("name", "")).strip()
        raw_arguments = str(tool_call.get("raw_arguments", "")).strip()
        arguments = tool_call.get("arguments", {})
        if not call_id or not name or not raw_arguments or not isinstance(arguments, dict):
            continue
        specs.append((call_id, name))
    return tuple(specs)


def _ordered_unique_names(names: Sequence[str] | list[str] | tuple[str, ...] | Any) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        name = str(raw_name).strip()
        if not name or name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    return tuple(ordered)


def _build_unexecuted_tool_call_note_message(
    tool_names: Sequence[str],
) -> LLMMessage:
    if tool_names:
        names_text = ", ".join(tool_names)
        text = (
            "The previous turn was interrupted before these proposed tool calls were executed: "
            f"{names_text}. Treat them as not run."
        )
    else:
        text = (
            "The previous turn was interrupted before the assistant's proposed tool calls were executed. "
            "Treat them as not run."
        )
    return LLMMessage(
        role="system",
        parts=(TextPart(text=text),),
    )


def _collect_activated_discoverable_tool_names(
    records: Sequence[ConversationRecord],
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for record in records:
        if record.role != "tool":
            continue
        raw_names = record.metadata.get("activated_discoverable_tool_names")
        if not isinstance(raw_names, list):
            continue
        for raw_name in raw_names:
            name = str(raw_name).strip()
            if not name or name in seen:
                continue
            names.append(name)
            seen.add(name)
    return tuple(names)


def _pending_tool_notice_names(
    tool_calls: Sequence[ToolCall],
    noticed_call_ids: set[str],
) -> tuple[str, ...]:
    names: list[str] = []
    for tool_call in tool_calls:
        call_id = tool_call.call_id.strip()
        name = tool_call.name.strip()
        if not name:
            continue
        if call_id and call_id in noticed_call_ids:
            continue
        names.append(name)
    return tuple(names)


def _record_image_part(record: ConversationRecord) -> ImagePart | None:
    attachment = record.metadata.get(_IMAGE_INPUT_METADATA_KEY)
    if not isinstance(attachment, dict):
        return None

    raw_path = str(attachment.get("path", "")).strip()
    media_type = str(attachment.get("media_type", "")).strip()
    raw_detail = str(attachment.get("detail", "auto")).strip() or "auto"
    if not raw_path or not media_type:
        return None
    if raw_detail not in {"low", "high", "auto", "original"}:
        raw_detail = "auto"

    try:
        data = Path(raw_path).read_bytes()
    except OSError:
        return None

    return ImagePart.from_base64(
        media_type=media_type,
        data_base64=base64.b64encode(data).decode("ascii"),
        detail=raw_detail,  # type: ignore[arg-type]
    )


def _carry_forward_soft_limit(record: ConversationRecord) -> int | None:
    if bool(record.metadata.get(_TRANSIENT_RECORD_METADATA_KEY, False)):
        return None
    if record.role == "tool":
        return 1_800
    if record.role in {"user", "assistant"}:
        return 1_200
    return None


def _truncate_carry_forward_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return f"{text[:head]}\n...[carry-forward truncated]...\n{text[-tail:]}"


def _strong_carry_forward_text(record: ConversationRecord) -> str:
    if record.role == "tool":
        return (
            "Tool result compacted after mid-turn overflow.\n"
            "See the archived session transcript for the full tool output."
        )
    if record.role == "assistant":
        return (
            "Assistant message compacted after mid-turn overflow.\n"
            "See the archived session transcript for the full assistant text."
        )
    return (
        "User message compacted after mid-turn overflow.\n"
        "See the archived session transcript for the full user text."
    )


def _copy_record_with_content(
    record: ConversationRecord,
    *,
    content: str,
    metadata_updates: dict[str, Any],
) -> ConversationRecord:
    metadata = deepcopy(record.metadata)
    metadata.update(metadata_updates)
    return ConversationRecord(
        record_id=record.record_id,
        session_id=record.session_id,
        created_at=record.created_at,
        role=record.role,
        content=content,
        kind=record.kind,
        metadata=metadata,
    )
