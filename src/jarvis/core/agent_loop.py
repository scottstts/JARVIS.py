"""Core agentic loop with sessioning and context compaction policies."""

from __future__ import annotations
import asyncio
import base64
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, Protocol, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

from jarvis.logging_setup import get_application_logger
from jarvis.llm import (
    ImagePart,
    LLMConfigurationError,
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
    UnsupportedCapabilityError,
)
from jarvis.memory import MemoryService, MemorySettings
from jarvis.storage import ConversationRecord, SessionMetadata, SessionStorage
from jarvis.tools import ToolExecutionContext, ToolExecutionResult, ToolRegistry, ToolRuntime, ToolSettings

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
_IMAGE_ATTACHMENT_ERROR_HINTS = (
    "image",
    "vision",
    "multimodal",
)
_TRANSCRIPT_ONLY_RECORD_METADATA_KEY = "transcript_only"
_IMAGE_INPUT_METADATA_KEY = "image_input"
_EPHEMERAL_IMAGE_INPUT_METADATA_KEY = "ephemeral_image_input"
_TURN_CONTEXT_METADATA_KEY = "turn_context"
_TURN_ID_METADATA_KEY = "turn_id"
_INTERRUPTION_NOTICE_METADATA_KEY = "interruption_notice"
_TOOL_ROUND_LIMIT_METADATA_KEY = "tool_round_limit"
_UNEXECUTED_TOOL_CALL_NOTICE_METADATA_KEY = "unexecuted_tool_call_notice"
_ORPHANED_TURN_RECOVERY_METADATA_KEY = "orphaned_turn_recovery"
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
_PREVIOUS_TASK_INTERRUPTED_TEXT = (
    "The user interrupted the previous task. Treat any partial output from it as incomplete."
)
_PREVIOUS_TASK_SUPERSEDED_TEXT = (
    "A newer user message superseded the previous task. Handle the current user message "
    "first. Use completed results from the older task only if they are directly relevant."
)
_TURN_INTERRUPTED_RECORD_TEXT = (
    "The user interrupted this turn before it completed."
)
_TURN_SUPERSEDED_RECORD_TEXT = (
    "A newer user message superseded this turn before it completed."
)
_TURN_ORPHANED_RECOVERY_RECORD_TEXT = (
    "This turn ended unexpectedly before it completed. Treat any partial assistant output "
    "above as incomplete."
)
_ORCHESTRATOR_MONITORED_WORK_FOLLOWUP_TEXT = (
    "Background work is being monitored by the orchestrator, not by proactive model polling. "
    "Do not call more tools in this response. Do not claim the task is finished while any listed "
    "detached bash job or delegated subagent is still pending. Briefly report the current "
    "in-progress state and wait for the next orchestrator system progress update unless the user "
    "explicitly asks for immediate inspection."
)
LOGGER = get_application_logger(__name__)

AgentKind = Literal["main", "subagent"]
InterruptionReason = Literal["user_stop", "superseded_by_user_message"]


class BootstrapMessageLoader(Protocol):
    def load_bootstrap_messages(self) -> list[LLMMessage]:
        """Return the starter context messages for a newly created session."""


@dataclass(slots=True, frozen=True)
class AgentRuntimeMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
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
LocalNoticeCallback = Callable[[str, str], Awaitable[None]]


@dataclass(slots=True, frozen=True)
class AgentTurnResult:
    session_id: str
    response_text: str
    turn_id: str = ""
    command: str | None = None
    compaction_performed: bool = False
    interrupted: bool = False
    approval_rejected: bool = False
    interruption_reason: InterruptionReason | None = None


@dataclass(slots=True, frozen=True)
class AgentTurnStartedEvent:
    session_id: str
    turn_id: str
    type: Literal["turn_started"] = "turn_started"


@dataclass(slots=True, frozen=True)
class AgentTextDeltaEvent:
    session_id: str
    delta: str
    turn_id: str = ""
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True, frozen=True)
class AgentAssistantMessageEvent:
    session_id: str
    text: str
    turn_id: str = ""
    type: Literal["assistant_message"] = "assistant_message"


@dataclass(slots=True, frozen=True)
class AgentToolCallEvent:
    session_id: str
    tool_names: tuple[str, ...]
    turn_id: str = ""
    type: Literal["tool_call"] = "tool_call"


@dataclass(slots=True, frozen=True)
class AgentApprovalRequestEvent:
    session_id: str
    approval_id: str
    kind: str
    summary: str
    details: str
    turn_id: str = ""
    command: str | None = None
    tool_name: str | None = None
    inspection_url: str | None = None
    type: Literal["approval_request"] = "approval_request"


@dataclass(slots=True, frozen=True)
class AgentTurnDoneEvent:
    session_id: str
    response_text: str
    turn_id: str = ""
    command: str | None = None
    compaction_performed: bool = False
    interrupted: bool = False
    approval_rejected: bool = False
    interruption_reason: InterruptionReason | None = None
    type: Literal["done"] = "done"

    def to_result(self) -> AgentTurnResult:
        return AgentTurnResult(
            session_id=self.session_id,
            turn_id=self.turn_id,
            response_text=self.response_text,
            command=self.command,
            compaction_performed=self.compaction_performed,
            interrupted=self.interrupted,
            approval_rejected=self.approval_rejected,
            interruption_reason=self.interruption_reason,
        )


AgentTurnStreamEvent = (
    AgentTurnStartedEvent
    | AgentTextDeltaEvent
    | AgentAssistantMessageEvent
    | AgentToolCallEvent
    | AgentApprovalRequestEvent
    | AgentTurnDoneEvent
)


@dataclass(slots=True, frozen=True)
class _RequestedInterruption:
    turn_id: str
    reason: InterruptionReason


@dataclass(slots=True, frozen=True)
class _ToolExecutionOutcome:
    approval_rejected: bool = False
    interrupted: bool = False
    pending_detached_job_ids: frozenset[str] = frozenset()
    pending_subagent_ids: frozenset[str] = frozenset()
    deferred_tool_successes: tuple["_DeferredToolSuccess", ...] = ()


@dataclass(slots=True, frozen=True)
class _DeferredToolSuccess:
    tool_result: ToolExecutionResult
    tool_record: ConversationRecord
    extra_records: tuple[ConversationRecord, ...] = ()


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
        local_notice_callback: LocalNoticeCallback | None = None,
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
        self._local_notice_callback = local_notice_callback
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
        self._requested_interruption: _RequestedInterruption | None = None
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

    async def handle_runtime_turn(
        self,
        *,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AgentTurnResult:
        return await self._handle_message_turn(
            None,
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

    async def stream_runtime_turn(
        self,
        *,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        async for event in self._stream_message_turn(
            None,
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
        return None

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

        self._append_message(
            session_id=target_session_id,
            role="system",
            content=normalized_content,
            metadata=metadata,
        )
        return True

    async def prepare_session(self, *, start_reason: str = "initial") -> str:
        await self._ensure_memory_runtime_ready()
        active = self._storage.get_active_session()
        if active is not None:
            self._reconcile_orphaned_turns(active.session_id)
            return active.session_id
        session = await self._start_session(start_reason=start_reason)
        return session.session_id

    def request_stop(
        self,
        *,
        reason: InterruptionReason = "user_stop",
    ) -> bool:
        active_turn_id = self._active_turn_id
        if active_turn_id is None:
            return False
        self._requested_interruption = _RequestedInterruption(
            turn_id=active_turn_id,
            reason=reason,
        )
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
            turn_id="",
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
            turn_id=result.turn_id,
            command=result.command,
            compaction_performed=result.compaction_performed,
        )

    async def _handle_message_turn(
        self,
        user_text: str | None,
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
        ) = await self._prepare_turn(
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
            turn_id=turn_id,
        )
        self._begin_turn(session_id=session.session_id, turn_id=turn_id)
        self._persist_records(
            session_id=session.session_id,
            records=pending_records,
        )
        if user_text is not None:
            user_record = self._build_message_record(
                session_id=session.session_id,
                role="user",
                content=user_text,
                turn_id=turn_id,
            )
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
                    unexecuted_tool_names=tuple(call.name for call in response.tool_calls),
                )

            base_records = self._storage.load_records(session.session_id)
            (
                session,
                final_response,
                final_estimated_input_tokens,
                followup_compacted,
                interrupted,
                approval_rejected,
                interrupted_unexecuted_tool_names,
            ) = await self._execute_followup_tool_rounds(
                session=session,
                base_records=base_records,
                pending_records=pending_records,
                current_response=response,
                current_estimated_input_tokens=final_estimated_input_tokens,
                turn_id=turn_id,
                pending_detached_job_ids=_collect_pending_detached_job_ids(turn_runtime_messages),
                pending_subagent_ids=_collect_pending_subagent_ids(turn_runtime_messages),
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
                    unexecuted_tool_names=interrupted_unexecuted_tool_names,
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
                turn_id=turn_id,
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
        pending_detached_job_ids: frozenset[str] = frozenset(),
        pending_subagent_ids: frozenset[str] = frozenset(),
    ) -> _ToolExecutionOutcome:
        ephemeral_image_records: list[ConversationRecord] = []
        deferred_tool_successes: list[_DeferredToolSuccess] = []
        current_pending_detached_job_ids = set(pending_detached_job_ids)
        current_pending_subagent_ids = set(pending_subagent_ids)
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
                    tool_record = self._build_tool_record(
                        session_id,
                        tool_result,
                        metadata_overrides=_completed_after_interrupt_metadata(
                            self._stop_requested_reason(turn_id)
                        ),
                        turn_id=turn_id,
                    )
                    attachment_records = tuple(
                        self._build_ephemeral_image_records_from_tool_result(
                            session_id,
                            tool_result,
                            turn_id=turn_id,
                        )
                    )
                    if tool_result.ok and attachment_records:
                        deferred_tool_successes.append(
                            _DeferredToolSuccess(
                                tool_result=tool_result,
                                tool_record=tool_record,
                                extra_records=attachment_records,
                            )
                        )
                    else:
                        self._append_turn_record(
                            session_id=session_id,
                            pending_records=pending_records,
                            record=tool_record,
                        )
                        ephemeral_image_records.extend(attachment_records)
                    _update_pending_detached_job_ids(
                        current_pending_detached_job_ids,
                        tool_result,
                    )
                    _update_pending_subagent_ids(
                        current_pending_subagent_ids,
                        tool_result,
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
                    return _ToolExecutionOutcome(
                        interrupted=True,
                        pending_detached_job_ids=frozenset(current_pending_detached_job_ids),
                        pending_subagent_ids=frozenset(current_pending_subagent_ids),
                    )
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
                    return _ToolExecutionOutcome(
                        approval_rejected=True,
                        pending_detached_job_ids=frozenset(current_pending_detached_job_ids),
                        pending_subagent_ids=frozenset(current_pending_subagent_ids),
                    )
                tool_context = replace(tool_context, approved_action=pending_approval)

        pending_records.extend(ephemeral_image_records)
        return _ToolExecutionOutcome(
            pending_detached_job_ids=frozenset(current_pending_detached_job_ids),
            pending_subagent_ids=frozenset(current_pending_subagent_ids),
            deferred_tool_successes=tuple(deferred_tool_successes),
        )

    def _build_followup_attempt_request(
        self,
        *,
        session_id: str,
        base_records: Sequence[ConversationRecord],
        pending_records: list[ConversationRecord],
        pending_detached_job_ids: Sequence[str],
        pending_subagent_ids: Sequence[str],
        turn_id: str,
        extra_records: Sequence[ConversationRecord] = (),
    ) -> tuple[LLMRequest, int]:
        if pending_detached_job_ids or pending_subagent_ids:
            return self._build_orchestrator_monitored_waiting_request(
                session_id=session_id,
                base_records=base_records,
                pending_records=pending_records,
                pending_detached_job_ids=pending_detached_job_ids,
                pending_subagent_ids=pending_subagent_ids,
                turn_id=turn_id,
                extra_records=extra_records,
            )
        return self._build_followup_request(
            base_records=base_records,
            pending_records=pending_records,
            extra_records=extra_records,
        )

    def _deferred_tool_success_records(
        self,
        deferred_tool_successes: Sequence[_DeferredToolSuccess],
    ) -> tuple[ConversationRecord, ...]:
        records: list[ConversationRecord] = []
        for deferred in deferred_tool_successes:
            records.append(deferred.tool_record)
            records.extend(deferred.extra_records)
        return tuple(records)

    def _commit_deferred_tool_successes(
        self,
        *,
        session_id: str,
        pending_records: list[ConversationRecord],
        deferred_tool_successes: Sequence[_DeferredToolSuccess],
    ) -> None:
        for deferred in deferred_tool_successes:
            self._append_turn_record(
                session_id=session_id,
                pending_records=pending_records,
                record=deferred.tool_record,
            )
            for record in deferred.extra_records:
                self._append_turn_record(
                    session_id=session_id,
                    pending_records=pending_records,
                    record=record,
                )

    def _persist_failed_deferred_tool_successes(
        self,
        *,
        session_id: str,
        pending_records: list[ConversationRecord],
        deferred_tool_successes: Sequence[_DeferredToolSuccess],
        error_message: str,
        turn_id: str,
    ) -> None:
        for deferred in deferred_tool_successes:
            failed_result = self._build_failed_image_attachment_tool_result(
                deferred.tool_result,
                error_message=error_message,
            )
            self._append_turn_record(
                session_id=session_id,
                pending_records=pending_records,
                record=self._build_tool_record(
                    session_id,
                    failed_result,
                    turn_id=turn_id,
                ),
            )

    def _build_failed_image_attachment_tool_result(
        self,
        tool_result: ToolExecutionResult,
        *,
        error_message: str,
    ) -> ToolExecutionResult:
        reason = error_message.strip() or "The image attachment could not be used."
        metadata = dict(tool_result.metadata)
        metadata.pop("image_attachment", None)
        metadata["error"] = reason

        title = "View image failed" if tool_result.name == "view_image" else (
            f"{tool_result.name.replace('_', ' ').capitalize()} failed"
        )
        lines = [title]
        raw_path = str(metadata.get("path", "")).strip()
        if raw_path:
            lines.append(f"path: {raw_path}")
        lines.append(f"reason: {reason}")

        return ToolExecutionResult(
            call_id=tool_result.call_id,
            name=tool_result.name,
            ok=False,
            content="\n".join(lines),
            metadata=metadata,
        )

    def _build_followup_request(
        self,
        *,
        base_records: Sequence[ConversationRecord],
        pending_records: Sequence[ConversationRecord],
        allow_tools: bool = True,
        extra_records: Sequence[ConversationRecord] = (),
    ) -> tuple[LLMRequest, int]:
        activated_discoverable_tool_names = _collect_activated_discoverable_tool_names(
            pending_records
        )

        request = self._build_request(
            list(base_records) + list(pending_records) + list(extra_records),
            activated_discoverable_tool_names=(
                activated_discoverable_tool_names if allow_tools else ()
            ),
            allow_tools=allow_tools,
        )
        estimated_input_tokens = estimate_request_input_tokens(request)
        if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
            raise ContextBudgetError(
                "Tool output exceeded the context budget during the current turn."
            )

        return request, estimated_input_tokens

    def _build_orchestrator_monitored_waiting_request(
        self,
        *,
        session_id: str,
        base_records: Sequence[ConversationRecord],
        pending_records: list[ConversationRecord],
        pending_detached_job_ids: Sequence[str],
        pending_subagent_ids: Sequence[str],
        turn_id: str,
        extra_records: Sequence[ConversationRecord] = (),
    ) -> tuple[LLMRequest, int]:
        metadata: dict[str, Any] = {}
        if pending_detached_job_ids:
            metadata["detached_bash_jobs_pending"] = True
            metadata["detached_bash_job_ids"] = list(pending_detached_job_ids)
        if pending_subagent_ids:
            metadata["subagents_pending"] = True
            metadata["pending_subagent_ids"] = list(pending_subagent_ids)
        if not any(record.metadata.get("orchestrator_monitored_waiting") for record in pending_records):
            self._append_turn_record(
                session_id=session_id,
                pending_records=pending_records,
                        record=self._build_runtime_message_record(
                    session_id=session_id,
                    message=AgentRuntimeMessage(
                        role="system",
                        metadata={
                            "orchestrator_monitored_waiting": True,
                            **metadata,
                        },
                        content=_ORCHESTRATOR_MONITORED_WORK_FOLLOWUP_TEXT,
                    ),
                    turn_id=turn_id,
                ),
            )
        return self._build_followup_request(
            base_records=base_records,
            pending_records=pending_records,
            allow_tools=False,
            extra_records=extra_records,
        )

    def _build_tool_round_limit_record(
        self,
        *,
        session_id: str,
        attempted_round: int,
        turn_id: str,
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
                _TOOL_ROUND_LIMIT_METADATA_KEY: True,
                "attempted_round": attempted_round,
                "max_rounds": max_rounds,
            },
            turn_id=turn_id,
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
        unexecuted_tool_names: Sequence[str],
        turn_id: str,
    ) -> tuple[LLMRequest, int]:
        if unexecuted_tool_names:
            self._append_turn_record(
                session_id=session_id,
                pending_records=pending_records,
                record=self._build_unexecuted_tool_call_note_record(
                    session_id=session_id,
                    tool_names=unexecuted_tool_names,
                    turn_id=turn_id,
                ),
            )
        self._append_turn_record(
            session_id=session_id,
            pending_records=pending_records,
            record=self._build_tool_round_limit_record(
                session_id=session_id,
                attempted_round=attempted_round,
                turn_id=turn_id,
            ),
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
        unexecuted_tool_names: Sequence[str],
        turn_id: str,
    ) -> tuple[LLMResponse, int]:
        request, estimated_input_tokens = self._build_tool_round_limit_recovery_request(
            session_id=session_id,
            base_records=base_records,
            pending_records=pending_records,
            attempted_round=attempted_round,
            unexecuted_tool_names=unexecuted_tool_names,
            turn_id=turn_id,
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
        unexecuted_tool_names: Sequence[str],
        turn_id: str,
    ) -> tuple[list[AgentTurnStreamEvent], LLMResponse, int]:
        request, estimated_input_tokens = self._build_tool_round_limit_recovery_request(
            session_id=session_id,
            base_records=base_records,
            pending_records=pending_records,
            attempted_round=attempted_round,
            unexecuted_tool_names=unexecuted_tool_names,
            turn_id=turn_id,
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
                            turn_id=turn_id,
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
                    turn_id=turn_id,
                )
            )
        return recovery_events, normalized, estimated_input_tokens

    async def _stream_message_turn(
        self,
        user_text: str | None,
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
        ) = await self._prepare_turn(
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
            turn_id=turn_id,
        )
        self._begin_turn(session_id=session.session_id, turn_id=turn_id)
        self._persist_records(
            session_id=session.session_id,
            records=pending_records,
        )
        if user_text is not None:
            user_record = self._build_message_record(
                session_id=session.session_id,
                role="user",
                content=user_text,
                turn_id=turn_id,
            )
            self._append_turn_record(
                session_id=session.session_id,
                pending_records=pending_records,
                record=user_record,
            )
        yield AgentTurnStartedEvent(
            session_id=session.session_id,
            turn_id=turn_id,
        )

        try:
            overflow_compacted = False
            overflow_retry_attempted = False
            initial_response: LLMResponse | None = None
            final_estimated_input_tokens = estimated_input_tokens
            noticed_initial_tool_call_ids: set[str] = set()
            streamed_initial_text = ""

            while True:
                streamed_response: LLMResponse | None = None
                emitted_any = False
                noticed_initial_tool_call_ids = set()
                streamed_initial_text = ""
                try:
                    async for event in self._llm_service.stream_generate(request):
                        if event.type == "text_delta":
                            emitted_any = True
                            if event.delta:
                                streamed_initial_text += event.delta
                                yield AgentTextDeltaEvent(
                                    session_id=session.session_id,
                                    delta=event.delta,
                                    turn_id=turn_id,
                                )
                        elif event.type == "tool_call_delta":
                            emitted_any = True
                            tool_name = str(event.tool_name or "").strip()
                            call_id = event.call_id.strip()
                            if tool_name and call_id and call_id not in noticed_initial_tool_call_ids:
                                noticed_initial_tool_call_ids.add(call_id)
                                yield AgentToolCallEvent(
                                    session_id=session.session_id,
                                    tool_names=(tool_name,),
                                    turn_id=turn_id,
                                )
                            if self._stop_requested(turn_id) and tool_name and call_id:
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
                                interrupted = self._interrupt_turn(
                                    session_id=session.session_id,
                                    turn_id=turn_id,
                                    command=command_override,
                                    compaction_performed=did_compaction,
                                    response_text=streamed_initial_text,
                                    unexecuted_tool_names=(tool_name,),
                                )
                                yield AgentTurnDoneEvent(
                                    session_id=interrupted.session_id,
                                    response_text=interrupted.response_text,
                                    turn_id=turn_id,
                                    command=interrupted.command,
                                    compaction_performed=interrupted.compaction_performed,
                                    interrupted=True,
                                    interruption_reason=interrupted.interruption_reason,
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
                    turn_id=turn_id,
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
                        turn_id=turn_id,
                    )
            if self._stop_requested(turn_id):
                interrupted = self._interrupt_turn(
                    session_id=session.session_id,
                    turn_id=turn_id,
                    command=command_override,
                    compaction_performed=did_compaction,
                    response_text=initial_response.text,
                    unexecuted_tool_names=tuple(call.name for call in initial_response.tool_calls),
                )
                yield AgentTurnDoneEvent(
                    session_id=interrupted.session_id,
                    response_text=interrupted.response_text,
                    turn_id=turn_id,
                    command=interrupted.command,
                    compaction_performed=interrupted.compaction_performed,
                    interrupted=True,
                    interruption_reason=interrupted.interruption_reason,
                )
                return

            base_records = self._storage.load_records(session.session_id)
            current_response = initial_response
            tool_rounds = 0
            turn_approval_rejected = False
            pending_detached_job_ids = _collect_pending_detached_job_ids(turn_runtime_messages)
            pending_subagent_ids = _collect_pending_subagent_ids(turn_runtime_messages)
            while current_response.tool_calls:
                if self._stop_requested(turn_id):
                    interrupted = self._interrupt_turn(
                        session_id=session.session_id,
                        turn_id=turn_id,
                        command=command_override,
                        compaction_performed=did_compaction,
                        response_text=current_response.text,
                        unexecuted_tool_names=tuple(call.name for call in current_response.tool_calls),
                    )
                    yield AgentTurnDoneEvent(
                        session_id=interrupted.session_id,
                        response_text=interrupted.response_text,
                        turn_id=turn_id,
                        command=interrupted.command,
                        compaction_performed=interrupted.compaction_performed,
                        interrupted=True,
                        interruption_reason=interrupted.interruption_reason,
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
                        unexecuted_tool_names=tuple(
                            call.name for call in current_response.tool_calls
                        ),
                        turn_id=turn_id,
                    )
                    for recovery_event in recovery_events:
                        yield recovery_event
                    current_response = final_response
                    break

                followup_compaction_attempted = False
                deferred_tool_successes: tuple[_DeferredToolSuccess, ...] = ()
                try:
                    ephemeral_image_records: list[ConversationRecord] = []
                    staged_image_tool_successes: list[_DeferredToolSuccess] = []
                    approval_rejected = False
                    current_pending_detached_job_ids = set(pending_detached_job_ids)
                    current_pending_subagent_ids = set(pending_subagent_ids)
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
                                tool_record = self._build_tool_record(
                                    session.session_id,
                                    tool_result,
                                    metadata_overrides=_completed_after_interrupt_metadata(
                                        self._stop_requested_reason(turn_id)
                                    ),
                                    turn_id=turn_id,
                                )
                                attachment_records = tuple(
                                    self._build_ephemeral_image_records_from_tool_result(
                                        session.session_id,
                                        tool_result,
                                        turn_id=turn_id,
                                    )
                                )
                                if tool_result.ok and attachment_records:
                                    staged_image_tool_successes.append(
                                        _DeferredToolSuccess(
                                            tool_result=tool_result,
                                            tool_record=tool_record,
                                            extra_records=attachment_records,
                                        )
                                    )
                                else:
                                    self._append_turn_record(
                                        session_id=session.session_id,
                                        pending_records=pending_records,
                                        record=tool_record,
                                    )
                                    ephemeral_image_records.extend(attachment_records)
                                _update_pending_detached_job_ids(
                                    current_pending_detached_job_ids,
                                    tool_result,
                                )
                                _update_pending_subagent_ids(
                                    current_pending_subagent_ids,
                                    tool_result,
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
                                turn_id=turn_id,
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
                                    turn_id=turn_id,
                                    command=interrupted.command,
                                    compaction_performed=interrupted.compaction_performed,
                                    interrupted=True,
                                    interruption_reason=interrupted.interruption_reason,
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

                    pending_records.extend(ephemeral_image_records)
                    deferred_tool_successes = tuple(staged_image_tool_successes)
                    pending_detached_job_ids = frozenset(current_pending_detached_job_ids)
                    pending_subagent_ids = frozenset(current_pending_subagent_ids)
                    if approval_rejected:
                        if deferred_tool_successes:
                            self._commit_deferred_tool_successes(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                deferred_tool_successes=deferred_tool_successes,
                            )
                        break
                    if self._stop_requested(turn_id):
                        if deferred_tool_successes:
                            self._commit_deferred_tool_successes(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                deferred_tool_successes=deferred_tool_successes,
                            )
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
                            turn_id=turn_id,
                            command=interrupted.command,
                            compaction_performed=interrupted.compaction_performed,
                            interrupted=True,
                            interruption_reason=interrupted.interruption_reason,
                        )
                        return
                    request, final_estimated_input_tokens = self._build_followup_attempt_request(
                        session_id=session.session_id,
                        base_records=base_records,
                        pending_records=pending_records,
                        pending_detached_job_ids=pending_detached_job_ids,
                        pending_subagent_ids=pending_subagent_ids,
                        turn_id=turn_id,
                        extra_records=self._deferred_tool_success_records(deferred_tool_successes),
                    )
                except ContextBudgetError:
                    if deferred_tool_successes:
                        self._commit_deferred_tool_successes(
                            session_id=session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                        )
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
                    deferred_committed = False
                    try:
                        async for event in self._llm_service.stream_generate(request):
                            if deferred_tool_successes and not deferred_committed:
                                self._commit_deferred_tool_successes(
                                    session_id=session.session_id,
                                    pending_records=pending_records,
                                    deferred_tool_successes=deferred_tool_successes,
                                )
                                deferred_tool_successes = ()
                                deferred_committed = True
                            if event.type == "text_delta":
                                emitted_any = True
                                if event.delta:
                                    streamed_followup_text += event.delta
                                    yield AgentTextDeltaEvent(
                                        session_id=session.session_id,
                                        delta=event.delta,
                                        turn_id=turn_id,
                                    )
                            elif event.type == "tool_call_delta":
                                emitted_any = True
                                tool_name = str(event.tool_name or "").strip()
                                call_id = event.call_id.strip()
                                if tool_name and call_id and call_id not in noticed_followup_tool_call_ids:
                                    noticed_followup_tool_call_ids.add(call_id)
                                    yield AgentToolCallEvent(
                                        session_id=session.session_id,
                                        tool_names=(tool_name,),
                                        turn_id=turn_id,
                                    )
                                if self._stop_requested(turn_id) and tool_name and call_id:
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
                                    interrupted = self._interrupt_turn(
                                        session_id=session.session_id,
                                        turn_id=turn_id,
                                        command=command_override,
                                        compaction_performed=did_compaction,
                                        response_text=streamed_followup_text,
                                        unexecuted_tool_names=(tool_name,),
                                    )
                                    yield AgentTurnDoneEvent(
                                        session_id=interrupted.session_id,
                                        response_text=interrupted.response_text,
                                        turn_id=turn_id,
                                        command=interrupted.command,
                                        compaction_performed=interrupted.compaction_performed,
                                        interrupted=True,
                                        interruption_reason=interrupted.interruption_reason,
                                    )
                                    return
                            elif event.type == "done":
                                streamed_response = event.response
                        if deferred_tool_successes and not deferred_committed:
                            self._commit_deferred_tool_successes(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                deferred_tool_successes=deferred_tool_successes,
                            )
                            deferred_tool_successes = ()
                        break
                    except (LLMConfigurationError, UnsupportedCapabilityError) as exc:
                        if (
                            not deferred_tool_successes
                            or not _is_image_attachment_request_error(exc)
                        ):
                            raise
                        self._persist_failed_deferred_tool_successes(
                            session_id=session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                            error_message=str(exc),
                            turn_id=turn_id,
                        )
                        deferred_tool_successes = ()
                        request, final_estimated_input_tokens = self._build_followup_attempt_request(
                            session_id=session.session_id,
                            base_records=base_records,
                            pending_records=pending_records,
                            pending_detached_job_ids=pending_detached_job_ids,
                            pending_subagent_ids=pending_subagent_ids,
                            turn_id=turn_id,
                        )
                        continue
                    except ProviderBadRequestError as exc:
                        if (
                            deferred_tool_successes
                            and not emitted_any
                            and _is_image_attachment_request_error(exc)
                        ):
                            self._persist_failed_deferred_tool_successes(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                deferred_tool_successes=deferred_tool_successes,
                                error_message=str(exc),
                                turn_id=turn_id,
                            )
                            deferred_tool_successes = ()
                            request, final_estimated_input_tokens = self._build_followup_attempt_request(
                                session_id=session.session_id,
                                base_records=base_records,
                                pending_records=pending_records,
                                pending_detached_job_ids=pending_detached_job_ids,
                                pending_subagent_ids=pending_subagent_ids,
                                turn_id=turn_id,
                            )
                            continue
                        if (
                            not _is_context_overflow_error(exc)
                            or emitted_any
                        ):
                            raise
                        if deferred_tool_successes:
                            self._commit_deferred_tool_successes(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                deferred_tool_successes=deferred_tool_successes,
                            )
                            deferred_tool_successes = ()
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
                        turn_id=turn_id,
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
                            turn_id=turn_id,
                        )
                if self._stop_requested(turn_id):
                    interrupted = self._interrupt_turn(
                        session_id=session.session_id,
                        turn_id=turn_id,
                        command=command_override,
                        compaction_performed=did_compaction,
                        response_text=current_response.text,
                        unexecuted_tool_names=tuple(call.name for call in current_response.tool_calls),
                    )
                    yield AgentTurnDoneEvent(
                        session_id=interrupted.session_id,
                        response_text=interrupted.response_text,
                        turn_id=turn_id,
                        command=interrupted.command,
                        compaction_performed=interrupted.compaction_performed,
                        interrupted=True,
                        interruption_reason=interrupted.interruption_reason,
                    )
                    return
                if pending_detached_job_ids:
                    current_response = replace(
                        current_response,
                        tool_calls=[],
                        finish_reason="stop",
                    )
                    break

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
                turn_id=turn_id,
                command=command_override,
                compaction_performed=did_compaction,
                approval_rejected=turn_approval_rejected,
            )
        finally:
            self._clear_turn_control(turn_id)

    async def _prepare_turn(
        self,
        *,
        user_text: str | None,
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
        self._reconcile_orphaned_turns(session.session_id)
        session = self._storage.get_session(session.session_id) or session

        did_compaction = False
        if session.pending_reactive_compaction:
            compacted = await self._compact_session(session, reason="reactive")
            if compacted is not None:
                session = compacted
                did_compaction = True

        records = self._storage.load_records(session.session_id)
        interruption_notice_text = self._pending_interruption_notice_text(session)
        turn_runtime_messages = self._build_turn_runtime_messages(
            session_id=session.session_id,
            pre_turn_messages=pre_turn_messages,
        )
        allow_tools_for_initial_request = not _turn_requires_no_tools(
            runtime_messages=turn_runtime_messages,
            user_text=user_text,
        )
        request = self._build_turn_request(
            session_id=session.session_id,
            records=records,
            user_text=user_text,
            turn_context_text=turn_context_text,
            interruption_notice_text=interruption_notice_text,
            runtime_messages=turn_runtime_messages,
            allow_tools=allow_tools_for_initial_request,
        )
        estimated_input_tokens = estimate_request_input_tokens(request)

        if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
            compacted = await self._compact_session(session, reason="preflight")
            if compacted is not None:
                session = compacted
                did_compaction = True
                records = self._storage.load_records(session.session_id)
                interruption_notice_text = self._pending_interruption_notice_text(session)
                request = self._build_turn_request(
                    session_id=session.session_id,
                    records=records,
                    user_text=user_text,
                    turn_context_text=turn_context_text,
                    interruption_notice_text=interruption_notice_text,
                    runtime_messages=turn_runtime_messages,
                    allow_tools=allow_tools_for_initial_request,
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

    def _pending_interruption_notice_text(
        self,
        session: SessionMetadata,
    ) -> str | None:
        if not session.pending_interruption_notice:
            return None
        reason = (session.pending_interruption_notice_reason or "").strip()
        if reason == "superseded_by_user_message":
            return _PREVIOUS_TASK_SUPERSEDED_TEXT
        return _PREVIOUS_TASK_INTERRUPTED_TEXT

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
            pending_interruption_notice_reason=None,
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
        user_text: str | None,
        turn_context_text: str,
        interruption_notice_text: str | None = None,
        runtime_messages: Sequence[AgentRuntimeMessage] = (),
        allow_tools: bool = True,
    ) -> LLMRequest:
        turn_records = self._build_pending_turn_records(
            session_id=session_id,
            turn_context_text=turn_context_text,
            interruption_notice_text=interruption_notice_text,
            runtime_messages=runtime_messages,
        )
        if user_text is not None:
            turn_records.append(
                self._build_message_record(
                    session_id=session_id,
                    role="user",
                    content=user_text,
                )
            )
        return self._build_request(list(records) + turn_records, allow_tools=allow_tools)

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
        turn_id: str | None = None,
    ) -> list[ConversationRecord]:
        pending_records = [
            self._build_turn_context_record(
                session_id=session_id,
                turn_context_text=turn_context_text,
                turn_id=turn_id,
            )
        ]
        for message in runtime_messages:
            pending_records.append(
                self._build_runtime_message_record(
                    session_id=session_id,
                    message=message,
                    turn_id=turn_id,
                )
            )
        if interruption_notice_text is not None:
            pending_records.append(
                self._build_interruption_notice_record(
                    session_id=session_id,
                    text=interruption_notice_text,
                    turn_id=turn_id,
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
        pending_detached_job_ids: frozenset[str] = frozenset(),
        pending_subagent_ids: frozenset[str] = frozenset(),
    ) -> tuple[SessionMetadata, LLMResponse, int, bool, bool, bool, tuple[str, ...]]:
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
                    tuple(call.name for call in current_response.tool_calls),
                )
            tool_rounds += 1
            if tool_rounds > self._tool_settings.max_tool_rounds_per_turn:
                current_response, current_estimated_input_tokens = (
                    await self._recover_from_tool_round_limit(
                        session_id=current_session.session_id,
                        base_records=current_base_records,
                        pending_records=pending_records,
                        attempted_round=tool_rounds,
                        unexecuted_tool_names=tuple(
                            call.name for call in current_response.tool_calls
                        ),
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
                    pending_detached_job_ids=pending_detached_job_ids,
                    pending_subagent_ids=pending_subagent_ids,
                )
                pending_detached_job_ids = tool_execution_outcome.pending_detached_job_ids
                pending_subagent_ids = tool_execution_outcome.pending_subagent_ids
                deferred_tool_successes = tool_execution_outcome.deferred_tool_successes
                if tool_execution_outcome.interrupted:
                    if deferred_tool_successes:
                        self._commit_deferred_tool_successes(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                        )
                    return (
                        current_session,
                        current_response,
                        current_estimated_input_tokens,
                        did_compaction,
                        True,
                        approval_rejected,
                        (),
                    )
                if tool_execution_outcome.approval_rejected:
                    if deferred_tool_successes:
                        self._commit_deferred_tool_successes(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                        )
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
                    if deferred_tool_successes:
                        self._commit_deferred_tool_successes(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                        )
                    return (
                        current_session,
                        current_response,
                        current_estimated_input_tokens,
                        did_compaction,
                        True,
                        approval_rejected,
                        (),
                    )
                request, current_estimated_input_tokens = self._build_followup_attempt_request(
                    session_id=current_session.session_id,
                    base_records=current_base_records,
                    pending_records=pending_records,
                    pending_detached_job_ids=pending_detached_job_ids,
                    pending_subagent_ids=pending_subagent_ids,
                    turn_id=turn_id,
                    extra_records=self._deferred_tool_success_records(deferred_tool_successes),
                )
            except ContextBudgetError:
                if deferred_tool_successes:
                    self._commit_deferred_tool_successes(
                        session_id=current_session.session_id,
                        pending_records=pending_records,
                        deferred_tool_successes=deferred_tool_successes,
                    )
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
                    if deferred_tool_successes:
                        self._commit_deferred_tool_successes(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                        )
                    break
                except (LLMConfigurationError, UnsupportedCapabilityError) as exc:
                    if (
                        not deferred_tool_successes
                        or not _is_image_attachment_request_error(exc)
                    ):
                        raise
                    self._persist_failed_deferred_tool_successes(
                        session_id=current_session.session_id,
                        pending_records=pending_records,
                        deferred_tool_successes=deferred_tool_successes,
                        error_message=str(exc),
                        turn_id=turn_id,
                    )
                    deferred_tool_successes = ()
                    request, current_estimated_input_tokens = self._build_followup_attempt_request(
                        session_id=current_session.session_id,
                        base_records=current_base_records,
                        pending_records=pending_records,
                        pending_detached_job_ids=pending_detached_job_ids,
                        pending_subagent_ids=pending_subagent_ids,
                        turn_id=turn_id,
                    )
                    continue
                except ProviderBadRequestError as exc:
                    if (
                        deferred_tool_successes
                        and _is_image_attachment_request_error(exc)
                    ):
                        self._persist_failed_deferred_tool_successes(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                            error_message=str(exc),
                            turn_id=turn_id,
                        )
                        deferred_tool_successes = ()
                        request, current_estimated_input_tokens = self._build_followup_attempt_request(
                            session_id=current_session.session_id,
                            base_records=current_base_records,
                            pending_records=pending_records,
                            pending_detached_job_ids=pending_detached_job_ids,
                            pending_subagent_ids=pending_subagent_ids,
                            turn_id=turn_id,
                        )
                        continue
                    if not _is_context_overflow_error(exc):
                        raise
                    if deferred_tool_successes:
                        self._commit_deferred_tool_successes(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                        )
                        deferred_tool_successes = ()
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
            if pending_detached_job_ids:
                current_response = replace(
                    current_response,
                    tool_calls=[],
                    finish_reason="stop",
                )
                break

        return (
            current_session,
            current_response,
            current_estimated_input_tokens,
            did_compaction,
            False,
            approval_rejected,
            (),
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
            if _record_is_ephemeral_image_input(record):
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
                role="system",
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
                    role="system",
                    content="Runtime core memory bootstrap:\n\n" + core_memory_bootstrap,
                    metadata={"memory_bootstrap": "core"},
                )
            if ongoing_memory_bootstrap.strip():
                self._append_message(
                    session_id=session.session_id,
                    role="system",
                    content="Runtime ongoing memory bootstrap:\n\n" + ongoing_memory_bootstrap,
                    metadata={"memory_bootstrap": "ongoing"},
                )
        if summary_text:
            self._append_message(
                session_id=session.session_id,
                role="system",
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

        await self._emit_local_notice(
            notice_kind="compaction_started",
            text="Compacting...",
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
            pending_interruption_notice_reason=session.pending_interruption_notice_reason,
        )
        await self._emit_local_notice(
            notice_kind="compaction_completed",
            text="Context compacted into a new session.",
        )
        return self._storage.get_session(next_session.session_id) or next_session

    async def _emit_local_notice(self, *, notice_kind: str, text: str) -> None:
        if self._identity.kind != "main":
            return
        callback = self._local_notice_callback
        if callback is None:
            return
        normalized_notice_kind = notice_kind.strip()
        normalized_text = text.strip()
        if not normalized_notice_kind or not normalized_text:
            return
        await callback(normalized_notice_kind, normalized_text)

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
            metadata={"interrupted_stream_fragment": True},
            turn_id=turn_id,
        )

    def _build_final_stream_assistant_record(
        self,
        *,
        session_id: str,
        response: LLMResponse,
        turn_id: str,
    ) -> ConversationRecord | None:
        if not response.text and not response.tool_calls:
            return None
        return self._build_assistant_record(
            session_id,
            response,
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
        turn_id: str | None = None,
    ) -> ConversationRecord:
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content=turn_context_text,
            metadata={
                _TURN_CONTEXT_METADATA_KEY: "datetime",
            },
            turn_id=turn_id,
        )

    def _build_interruption_notice_record(
        self,
        *,
        session_id: str,
        text: str,
        turn_id: str | None = None,
    ) -> ConversationRecord:
        interruption_reason: InterruptionReason = "user_stop"
        if text == _PREVIOUS_TASK_SUPERSEDED_TEXT:
            interruption_reason = "superseded_by_user_message"
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content=text,
            metadata={
                _INTERRUPTION_NOTICE_METADATA_KEY: True,
                "interruption_reason": interruption_reason,
                "prioritize_current_user_message": (
                    interruption_reason == "superseded_by_user_message"
                ),
            },
            turn_id=turn_id,
        )

    def _build_runtime_message_record(
        self,
        *,
        session_id: str,
        message: AgentRuntimeMessage,
        turn_id: str | None = None,
    ) -> ConversationRecord:
        return self._build_message_record(
            session_id=session_id,
            role=message.role,
            content=message.content,
            metadata=dict(message.metadata),
            turn_id=turn_id,
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

    def _build_ephemeral_image_records_from_tool_result(
        self,
        session_id: str,
        result: ToolExecutionResult,
        *,
        turn_id: str,
    ) -> list[ConversationRecord]:
        records: list[ConversationRecord] = []
        attachment = result.metadata.get("image_attachment")
        if isinstance(attachment, dict):
            path = str(attachment.get("path", "")).strip()
            media_type = str(attachment.get("media_type", "")).strip()
            detail = str(attachment.get("detail", "auto")).strip() or "auto"
            if path and media_type:
                content = (
                    "Attached image from a local workspace file requested via view_image.\n"
                    f"path: {path}\n"
                    f"media_type: {media_type}"
                )
                records.append(
                    self._build_message_record(
                        session_id=session_id,
                        role="user",
                        content=content,
                        metadata={
                            _EPHEMERAL_IMAGE_INPUT_METADATA_KEY: True,
                            _IMAGE_INPUT_METADATA_KEY: {
                                "path": path,
                                "media_type": media_type,
                                "detail": detail,
                            },
                            "source_tool": result.name,
                        },
                        turn_id=turn_id,
                    )
                )

        return records

    def _build_unexecuted_tool_call_note_record(
        self,
        *,
        session_id: str,
        tool_names: Sequence[str],
        turn_id: str,
    ) -> ConversationRecord:
        ordered_tool_names = list(_ordered_unique_names(tool_names))
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content=_unexecuted_tool_call_note_text(ordered_tool_names),
            metadata={
                _UNEXECUTED_TOOL_CALL_NOTICE_METADATA_KEY: True,
                "tool_names": ordered_tool_names,
            },
            turn_id=turn_id,
        )

    def _build_orphaned_turn_recovery_record(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> ConversationRecord:
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content=_TURN_ORPHANED_RECOVERY_RECORD_TEXT,
            metadata={
                _ORPHANED_TURN_RECOVERY_METADATA_KEY: True,
            },
            turn_id=turn_id,
        )

    def _persist_records(
        self,
        *,
        session_id: str,
        records: Sequence[ConversationRecord],
    ) -> None:
        for record in records:
            if _record_is_ephemeral_image_input(record):
                continue
            self._storage.append_record(session_id, record)

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
            self._reconcile_orphaned_turn(session_id=session_id, turn_id=turn_id)

    def _reconcile_orphaned_turn(self, *, session_id: str, turn_id: str) -> None:
        turn_records = [
            record
            for record in self._storage.load_records(session_id, include_all_turns=True)
            if str(record.metadata.get(_TURN_ID_METADATA_KEY, "")).strip() == turn_id
        ]
        if not turn_records:
            self._storage.set_turn_status(
                session_id,
                turn_id=turn_id,
                status="interrupted",
            )
            return

        if not any(_record_is_unexecuted_tool_call_notice(record) for record in turn_records):
            unresolved_tool_names = _collect_unexecuted_tool_call_names(turn_records)
            if unresolved_tool_names:
                self._storage.append_record(
                    session_id,
                    self._build_unexecuted_tool_call_note_record(
                        session_id=session_id,
                        tool_names=unresolved_tool_names,
                        turn_id=turn_id,
                    ),
                )

        if not any(
            record.metadata.get(_ORPHANED_TURN_RECOVERY_METADATA_KEY, False)
            for record in turn_records
        ):
            self._storage.append_record(
                session_id,
                self._build_orphaned_turn_recovery_record(
                    session_id=session_id,
                    turn_id=turn_id,
                ),
            )

        self._storage.set_turn_status(
            session_id,
            turn_id=turn_id,
            status="interrupted",
        )

    def _begin_turn(self, *, session_id: str, turn_id: str) -> None:
        self._storage.set_turn_status(
            session_id,
            turn_id=turn_id,
            status="in_progress",
        )
        self._active_turn_id = turn_id
        self._requested_interruption = None

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
        requested = self._requested_interruption
        if requested is not None and requested.turn_id == turn_id:
            self._requested_interruption = None

    def _stop_requested(self, turn_id: str) -> bool:
        return self._stop_requested_reason(turn_id) is not None

    def _stop_requested_reason(self, turn_id: str) -> InterruptionReason | None:
        requested = self._requested_interruption
        if requested is None or requested.turn_id != turn_id:
            return None
        return requested.reason

    def _clear_turn_control(self, turn_id: str) -> None:
        if self._active_turn_id == turn_id:
            self._active_turn_id = None
        requested = self._requested_interruption
        if requested is not None and requested.turn_id == turn_id:
            self._requested_interruption = None
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
        turn_id: str,
        approval: dict[str, Any],
    ) -> AgentApprovalRequestEvent:
        return AgentApprovalRequestEvent(
            session_id=session_id,
            turn_id=turn_id,
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
        if _record_is_ephemeral_image_input(record):
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
        unexecuted_tool_names: Sequence[str] = (),
    ) -> AgentTurnResult:
        interruption_reason = self._stop_requested_reason(turn_id) or "user_stop"
        interrupted_status: Literal["interrupted", "superseded"]
        interrupted_record_text: str
        if interruption_reason == "superseded_by_user_message":
            interrupted_status = "superseded"
            interrupted_record_text = _TURN_SUPERSEDED_RECORD_TEXT
        else:
            interrupted_status = "interrupted"
            interrupted_record_text = _TURN_INTERRUPTED_RECORD_TEXT
        if unexecuted_tool_names:
            self._storage.append_record(
                session_id,
                self._build_unexecuted_tool_call_note_record(
                    session_id=session_id,
                    tool_names=unexecuted_tool_names,
                    turn_id=turn_id,
                ),
            )
        interruption_record = self._build_message_record(
            session_id=session_id,
            role="system",
            content=interrupted_record_text,
            metadata={
                "interrupted_by_user": interruption_reason == "user_stop",
                "superseded_by_user_message": (
                    interruption_reason == "superseded_by_user_message"
                ),
                "interruption_reason": interruption_reason,
            },
            turn_id=turn_id,
        )
        self._storage.append_record(session_id, interruption_record)
        self._finish_turn(
            session_id=session_id,
            turn_id=turn_id,
            status=interrupted_status,
        )
        self._storage.update_session(
            session_id,
            pending_interruption_notice=True,
            pending_interruption_notice_reason=interruption_reason,
        )
        return AgentTurnResult(
            session_id=session_id,
            turn_id=turn_id,
            response_text=response_text,
            command=command,
            compaction_performed=compaction_performed,
            interrupted=True,
            interruption_reason=interruption_reason,
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
        if _record_is_ephemeral_image_input(record):
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


def _collect_pending_detached_job_ids(
    runtime_messages: Sequence[AgentRuntimeMessage],
) -> frozenset[str]:
    job_ids: set[str] = set()
    for message in runtime_messages:
        raw_ids = message.metadata.get("detached_bash_job_ids")
        if not isinstance(raw_ids, list):
            continue
        for raw_job_id in raw_ids:
            job_id = str(raw_job_id).strip()
            if job_id:
                job_ids.add(job_id)
    return frozenset(job_ids)


def _collect_pending_subagent_ids(
    runtime_messages: Sequence[AgentRuntimeMessage],
) -> frozenset[str]:
    subagent_ids: set[str] = set()
    for message in runtime_messages:
        raw_ids = message.metadata.get("pending_subagent_ids")
        if not isinstance(raw_ids, list):
            continue
        for raw_subagent_id in raw_ids:
            subagent_id = str(raw_subagent_id).strip()
            if subagent_id:
                subagent_ids.add(subagent_id)
    return frozenset(subagent_ids)


def _turn_requires_no_tools(
    *,
    runtime_messages: Sequence[AgentRuntimeMessage],
    user_text: str | None,
) -> bool:
    if user_text is not None:
        return False
    return any(bool(message.metadata.get("force_no_tools_this_turn")) for message in runtime_messages)


def _update_pending_detached_job_ids(
    pending_job_ids: set[str],
    tool_result: ToolExecutionResult,
) -> None:
    if tool_result.name != "bash":
        return
    job_id = str(tool_result.metadata.get("job_id", "")).strip()
    if not job_id:
        return
    status = str(
        tool_result.metadata.get("status") or tool_result.metadata.get("state") or ""
    ).strip()
    if (
        status == "running"
        and (
            str(tool_result.metadata.get("mode", "")).strip() == "background"
            or bool(tool_result.metadata.get("promoted_to_background"))
        )
    ):
        pending_job_ids.add(job_id)
        return
    if status in {"finished", "cancelled"}:
        pending_job_ids.discard(job_id)


def _update_pending_subagent_ids(
    pending_subagent_ids: set[str],
    tool_result: ToolExecutionResult,
) -> None:
    if not tool_result.metadata.get("subagent_control"):
        return
    subagent_id = str(tool_result.metadata.get("subagent_id", "")).strip()
    if not subagent_id:
        return
    action = str(tool_result.metadata.get("subagent_action", "")).strip()
    status = str(tool_result.metadata.get("status", "")).strip()
    if action in {"invoke", "step_in"} and status in {"running", "waiting_background", "awaiting_approval"}:
        pending_subagent_ids.add(subagent_id)
        return
    if status in {"paused", "completed", "failed", "disposed"}:
        pending_subagent_ids.discard(subagent_id)


def _is_context_overflow_error(exc: ProviderBadRequestError) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _OVERFLOW_ERROR_HINTS)


def _is_image_attachment_request_error(exc: Exception) -> bool:
    if isinstance(exc, ProviderBadRequestError) and _is_context_overflow_error(exc):
        return False
    if not isinstance(
        exc,
        (
            LLMConfigurationError,
            ProviderBadRequestError,
            UnsupportedCapabilityError,
        ),
    ):
        return False
    message = str(exc).lower()
    return any(hint in message for hint in _IMAGE_ATTACHMENT_ERROR_HINTS)


def _records_to_llm_messages(
    records: Sequence[ConversationRecord],
) -> tuple[LLMMessage, ...]:
    # Replayable transcript records are the source of truth for rebuilding
    # LLMRequest.messages. Non-image prompt-visible records must persist.
    # Ephemeral image attachments are the only accepted non-persisted prompt
    # input. transcript_only records are archived but intentionally excluded
    # from replay.
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

    def _raise_unresolved_pending() -> None:
        if pending_assistant is None:
            return
        unresolved_names = ", ".join(pending_tool_names) or "(unknown tools)"
        raise RuntimeError(
            "Encountered assistant tool calls without matching tool results or an explicit "
            f"unexecuted-tool-call notice in transcript replay: {unresolved_names}."
        )

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

        if _record_is_unexecuted_tool_call_notice(record):
            _append_record(pending_assistant, include_tool_calls=False)
            _append_record(record)
            _clear_pending()
            continue

        if record.role == "tool":
            call_id = str(record.metadata.get("call_id", "")).strip()
            if call_id and call_id in pending_call_ids:
                pending_tool_records.append(record)
                pending_call_ids.remove(call_id)
                if not pending_call_ids:
                    _flush_resolved_pending()
                continue

        _raise_unresolved_pending()
        if call_specs:
            pending_assistant = record
            pending_tool_records = []
            pending_call_ids = {call_id for call_id, _name in call_specs}
            pending_tool_names = tuple(_ordered_unique_names(name for _call_id, name in call_specs))
            continue
        _append_record(record)

    if pending_assistant is not None:
        _raise_unresolved_pending()

    return tuple(messages)


def _record_to_llm_message(
    record: ConversationRecord,
    *,
    include_tool_calls: bool = True,
) -> LLMMessage | None:
    if bool(record.metadata.get(_TRANSCRIPT_ONLY_RECORD_METADATA_KEY, False)):
        return None

    if record.role in {"system", "user"}:
        parts: list[ImagePart | TextPart] = []
        image_part = _record_image_part(record)
        if image_part is not None:
            parts.append(image_part)
        if record.content:
            parts.append(TextPart(text=record.content))
        if not parts:
            return None
        return LLMMessage(
            role=record.role,
            parts=tuple(parts),
            metadata=dict(record.metadata),
        )

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
        return LLMMessage(
            role="assistant",
            parts=tuple(parts),
            metadata=dict(record.metadata),
        )

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
            metadata=dict(record.metadata),
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
    if _record_is_ephemeral_image_input(record):
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

def _completed_after_interrupt_metadata(
    reason: InterruptionReason | None,
) -> dict[str, Any] | None:
    if reason is None:
        return None
    return {
        "completed_after_interrupt_request": True,
        "interruption_reason": reason,
        "superseded_turn_output": reason == "superseded_by_user_message",
    }


def _unexecuted_tool_call_note_text(tool_names: Sequence[str]) -> str:
    ordered_tool_names = _ordered_unique_names(tool_names)
    if ordered_tool_names:
        names_text = ", ".join(ordered_tool_names)
        return (
            "The previous turn was interrupted before these proposed tool calls were executed: "
            f"{names_text}. Treat them as not run."
        )
    return (
        "The previous turn was interrupted before the assistant's proposed tool calls were executed. "
        "Treat them as not run."
    )


def _collect_unexecuted_tool_call_names(
    records: Sequence[ConversationRecord],
) -> tuple[str, ...]:
    pending_call_ids: set[str] = set()
    pending_tool_names: tuple[str, ...] = ()
    unexecuted_names: list[str] = []

    def _flush_pending() -> None:
        nonlocal pending_call_ids, pending_tool_names
        for tool_name in pending_tool_names:
            if tool_name not in unexecuted_names:
                unexecuted_names.append(tool_name)
        pending_call_ids = set()
        pending_tool_names = ()

    for record in records:
        if record.kind != "message":
            continue

        call_specs = _assistant_tool_call_specs(record)
        if call_specs:
            if pending_call_ids:
                _flush_pending()
            pending_call_ids = {call_id for call_id, _name in call_specs}
            pending_tool_names = tuple(
                _ordered_unique_names(name for _call_id, name in call_specs)
            )
            continue

        if not pending_call_ids:
            continue

        if _record_is_unexecuted_tool_call_notice(record):
            pending_call_ids = set()
            pending_tool_names = ()
            continue

        if record.role == "tool":
            call_id = str(record.metadata.get("call_id", "")).strip()
            if call_id and call_id in pending_call_ids:
                pending_call_ids.remove(call_id)
                if not pending_call_ids:
                    pending_tool_names = ()
                continue

        _flush_pending()

    if pending_call_ids:
        _flush_pending()

    return tuple(unexecuted_names)


def _record_is_ephemeral_image_input(record: ConversationRecord) -> bool:
    return bool(record.metadata.get(_EPHEMERAL_IMAGE_INPUT_METADATA_KEY, False))


def _record_is_unexecuted_tool_call_notice(record: ConversationRecord) -> bool:
    return bool(record.metadata.get(_UNEXECUTED_TOOL_CALL_NOTICE_METADATA_KEY, False))
