"""Core agentic loop with sessioning and context compaction policies."""

from __future__ import annotations
import asyncio
import base64
from collections.abc import Awaitable as RuntimeAwaitable
import contextlib
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
)
from uuid import uuid4
from zoneinfo import ZoneInfo

from PIL import Image, ImageOps

from jarvis.logging_setup import get_application_logger
from jarvis.llm import (
    ImagePart,
    LLMConfigurationError,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMService,
    LocalImagePart,
    ProviderContextStrategy,
    ProviderSessionState,
    ProviderBadRequestError,
    StatefulContinuation,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
    UnsupportedCapabilityError,
    strategy_for_provider,
)
from jarvis.memory import MemoryService, MemorySettings
from jarvis.skills import (
    SkillsSettings,
    import_staged_skills,
    load_skill_catalog,
    render_skill_bootstrap_headers,
    render_skill_search_guidance,
)
from jarvis.storage import ConversationRecord, SessionMetadata, SessionStorage
from jarvis.tools import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolRegistry,
    ToolRuntime,
    ToolSettings,
)

from .commands import ParsedCommand, parse_user_command
from .compaction import (
    CompactionBundle,
    CompactionOutcome,
    CompactionReplayItem,
    ContextCompactor,
    build_compaction_bundle_record,
    load_compaction_bundle,
    prune_compaction_source_records,
)
from .config import CoreSettings
from .errors import ContextBudgetError
from .identities import IdentityBootstrapLoader
from .token_estimator import estimate_request_input_tokens
from .task_contract import (
    TaskContract,
    build_task_contract,
    user_message_explicitly_resumes_task,
)
from .tool_safety import (
    ToolSafetyObservation,
    ToolSafetyTracker,
    build_blocked_repetition_result,
)

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
_GROK_RECOVERY_IMAGE_METADATA_KEY = "grok_recovery_image"
_GROK_PROVIDER_MEDIA_DIR_NAME = "provider_media"
_GROK_IMAGE_TRANSCODE_MIN_BYTES = 512 * 1024
_GROK_IMAGE_MAX_EDGE_PIXELS = 1_600
_GROK_IMAGE_JPEG_QUALITY = 82
_INLINE_TOOL_RESULT_MAX_CHARS = 12_000
_TURN_CONTEXT_METADATA_KEY = "turn_context"
_TURN_ID_METADATA_KEY = "turn_id"
_INTERRUPTION_NOTICE_METADATA_KEY = "interruption_notice"
_TOOL_ROUND_LIMIT_METADATA_KEY = "tool_round_limit"
_UNEXECUTED_TOOL_CALL_NOTICE_METADATA_KEY = "unexecuted_tool_call_notice"
_ORPHANED_TURN_RECOVERY_METADATA_KEY = "orphaned_turn_recovery"
_TOOL_BOOTSTRAP_METADATA_KEY = "tool_bootstrap"
_SKILLS_BOOTSTRAP_METADATA_KEY = "skills_bootstrap"
_TOOL_ROUND_CONTINUATION_EMPTY_TEXT = (
    "I could not produce a continuation after the tool execution slice boundary."
)
_TOOL_SAFETY_STOP_TEXT = (
    "Tool execution stopped safely because repeated calls were no longer making progress. "
    "Do not repeat the blocked action automatically; resume only after a materially new "
    "user instruction or orchestrator update."
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
_PREVIOUS_SESSION_RESET_TEXT = (
    "The user started a new session and terminated the previous task. Treat all partial output "
    "from the previous session as archived history, not work to resume."
)
_TURN_INTERRUPTED_RECORD_TEXT = "The user interrupted this turn before it completed."
_TURN_SUPERSEDED_RECORD_TEXT = "A newer user message superseded this turn before it completed."
_TURN_NEW_SESSION_RECORD_TEXT = (
    "The user started a new session and terminated this turn before it completed."
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
InterruptionReason = Literal["user_stop", "superseded_by_user_message", "new_session"]
T = TypeVar("T")
_STOP_PREEMPTION_CLEANUP_SECONDS = 1.0


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
    completion_blocked: bool = False
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
    completion_blocked: bool = False
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
            completion_blocked=self.completion_blocked,
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


class _TurnStopRequested(Exception):
    """Internal control-flow signal raised when a turn stop preempts an await."""


class _CompactionStopRequested(Exception):
    """Internal signal raised when stop preempts compaction before a turn starts."""

    def __init__(self, *, session_id: str, reason: InterruptionReason) -> None:
        super().__init__(f"Compaction for session {session_id} was interrupted: {reason}")
        self.session_id = session_id
        self.interruption_reason: InterruptionReason = reason


@dataclass(slots=True)
class _ActiveCompactionControl:
    operation_id: str
    session_id: str
    stop_event: asyncio.Event
    interruption_reason: InterruptionReason | None = None


@dataclass(slots=True, frozen=True)
class _ToolExecutionOutcome:
    approval_rejected: bool = False
    interrupted: bool = False
    pending_detached_job_ids: frozenset[str] = frozenset()
    pending_subagent_ids: frozenset[str] = frozenset()
    deferred_tool_successes: tuple["_DeferredToolSuccess", ...] = ()
    unexecuted_tool_names: tuple[str, ...] = ()
    safety_stop: bool = False
    safety_stop_reason: str | None = None


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
            normalized if (normalized := (llm_provider or "").strip().lower()) else None
        )
        self._identity_loader = bootstrap_loader or IdentityBootstrapLoader(self._settings)
        self._runtime_messages_provider = runtime_messages_provider
        self._compactor = ContextCompactor(
            llm_service=self._llm_service,
            context_policy=self._settings.context_policy,
            provider=self._settings.compaction.provider,
        )
        memory_settings = MemorySettings.from_workspace_dir(self._settings.workspace_dir)
        memory_llm_service = (
            self._llm_service if isinstance(self._llm_service, LLMService) else None
        )
        if memory_llm_service is None or not self._memory_mode.reflection:
            memory_settings = replace(memory_settings, enable_reflection=False)
        self._memory_service = MemoryService(
            settings=memory_settings,
            llm_service=memory_llm_service,
        )
        self._tool_settings = ToolSettings.from_workspace_dir(self._settings.workspace_dir)
        self._skills_settings = SkillsSettings.from_workspace_dir(self._settings.workspace_dir)
        self._tool_registry = tool_registry or ToolRegistry.default(self._tool_settings)
        self._tool_runtime = tool_runtime or ToolRuntime(registry=self._tool_registry)
        self._tool_definitions_provider = (
            tool_definitions_provider or self._default_tool_definitions
        )
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
        self._turn_stop_event: asyncio.Event | None = None
        self._active_compaction_control: _ActiveCompactionControl | None = None

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
        if self._active_turn_id is not None:
            return self._active_turn_id
        control = self._active_compaction_control
        return control.operation_id if control is not None else None

    def has_active_turn(self) -> bool:
        return self._active_turn_id is not None or self._active_compaction_control is not None

    async def aclose(self) -> None:
        self.request_hard_stop(reason="new_session")

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

    def append_assistant_note(
        self,
        content: str,
        *,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Persist an orchestrator-authored assistant status message."""

        if self._storage.get_session(session_id) is None or not content.strip():
            return False
        self._append_message(
            session_id=session_id,
            role="assistant",
            content=content.strip(),
            metadata=metadata,
        )
        return True

    def backend_state(self, *, session_id: str) -> dict[str, Any]:
        session = self._storage.get_session(session_id)
        return dict(session.backend_state) if session is not None else {}

    def update_backend_state(
        self,
        *,
        session_id: str,
        key: str,
        value: Any | None,
    ) -> bool:
        session = self._storage.get_session(session_id)
        if session is None:
            return False
        state = dict(session.backend_state)
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
        if state == session.backend_state:
            return False
        self._storage.update_session(session_id, backend_state=state)
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
        if active_turn_id is not None:
            self._requested_interruption = _RequestedInterruption(
                turn_id=active_turn_id,
                reason=reason,
            )
            stop_event = self._turn_stop_event
            if stop_event is not None:
                stop_event.set()
            return True

        compaction_control = self._active_compaction_control
        if compaction_control is None:
            return False
        compaction_control.interruption_reason = reason
        compaction_control.stop_event.set()
        return True

    def request_hard_stop(
        self,
        *,
        reason: InterruptionReason = "new_session",
    ) -> bool:
        """Immediately preempt the active turn for a destructive session reset."""
        active_turn_id = self._active_turn_id
        if active_turn_id is not None:
            self._requested_interruption = _RequestedInterruption(
                turn_id=active_turn_id,
                reason=reason,
            )
            pending_approval = self._pending_approval_future
            if pending_approval is not None and not pending_approval.done():
                pending_approval.cancel()
            stop_event = self._turn_stop_event
            if stop_event is not None:
                stop_event.set()
            return True

        compaction_control = self._active_compaction_control
        if compaction_control is None:
            return False
        compaction_control.interruption_reason = reason
        compaction_control.stop_event.set()
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
        _ = command
        await self._ensure_memory_runtime_ready()
        session = await self._start_user_new_session()
        return AgentTurnResult(
            session_id=session.session_id,
            response_text="Started a new session.",
            command="/new",
        )

    async def _handle_compact_command(self, command: ParsedCommand) -> AgentTurnResult:
        await self._ensure_memory_runtime_ready()
        active = await self._ensure_active_session()
        try:
            compacted = await self._compact_session(
                active,
                reason="manual",
                user_instruction=command.body or None,
            )
        except _CompactionStopRequested as exc:
            return AgentTurnResult(
                session_id=exc.session_id,
                response_text="",
                command="/compact",
                interrupted=True,
                interruption_reason=exc.interruption_reason,
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
        _ = command
        await self._ensure_memory_runtime_ready()
        session = await self._start_user_new_session()
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
        if result.response_text:
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
            interrupted=result.interrupted,
            interruption_reason=result.interruption_reason,
        )

    async def _handle_message_turn(
        self,
        user_text: str | None,
        *,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AgentTurnResult:
        turn_id = uuid4().hex
        try:
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
                task_id=turn_id,
            )
        except _CompactionStopRequested as exc:
            return AgentTurnResult(
                session_id=exc.session_id,
                response_text="",
                command=command_override,
                interrupted=True,
                interruption_reason=exc.interruption_reason,
            )
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

            if not response.tool_calls:
                _rounds, persisted_tool_safety = self._load_tool_task_state(
                    session.session_id
                )
                response = _enforce_acceptance_handoff(
                    response,
                    tool_safety=persisted_tool_safety,
                )

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
            session = (
                self._persist_provider_session_state_from_response(
                    session_id=session.session_id,
                    response=response,
                    assistant_record=assistant_record,
                )
                or session
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

            completion_blocked = _response_completion_blocked(final_response)
            self._persist_successful_turn(
                session_id=session.session_id,
                turn_id=turn_id,
                response=final_response,
                estimated_input_tokens=final_estimated_input_tokens,
            )
            if not completion_blocked:
                await self._reflect_completed_turn(
                    session_id=session.session_id,
                    turn_id=turn_id,
                )

            refreshed = self._storage.get_session(session.session_id)
            threshold_observed = (
                final_response.usage.input_tokens
                if final_response.usage is not None
                and final_response.usage.input_tokens is not None
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
                completion_blocked=completion_blocked,
            )
        except _TurnStopRequested:
            return self._interrupt_turn(
                session_id=session.session_id,
                turn_id=turn_id,
                command=command_override,
                compaction_performed=did_compaction,
                response_text="",
            )
        except Exception:
            self._fail_turn_after_runtime_error(
                session_id=session.session_id,
                turn_id=turn_id,
            )
            raise
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
        tool_safety: ToolSafetyTracker | None = None,
    ) -> _ToolExecutionOutcome:
        ephemeral_image_records: list[ConversationRecord] = []
        deferred_tool_successes: list[_DeferredToolSuccess] = []
        current_pending_detached_job_ids = set(pending_detached_job_ids)
        current_pending_subagent_ids = set(pending_subagent_ids)
        tool_calls = tuple(current_response.tool_calls)
        safety_stop = False
        safety_stop_reason: str | None = None
        safety_stop_unexecuted_tool_names: tuple[str, ...] = ()
        safety_stop_unexecuted_tool_calls: tuple[ToolCall, ...] = ()
        for tool_index, tool_call in enumerate(tool_calls):
            tool_context = replace(
                self._tool_context,
                session_id=session_id,
                turn_id=turn_id,
            )
            while True:
                blocked_reason = (
                    tool_safety.blocked_call_reason(tool_call)
                    if tool_safety is not None
                    else None
                )
                if blocked_reason is not None:
                    tool_result = build_blocked_repetition_result(
                        tool_call=tool_call,
                        reason=blocked_reason,
                        diagnostics=(
                            tool_safety.blocked_call_details(tool_call)
                            if tool_safety is not None
                            else {}
                        ),
                    )
                else:
                    try:
                        tool_result = await self._await_with_stop(
                            self._tool_executor(tool_call, tool_context),
                            turn_id=turn_id,
                            operation=f"tool_{tool_call.name}",
                        )
                    except _TurnStopRequested:
                        pending_records.extend(ephemeral_image_records)
                        return _ToolExecutionOutcome(
                            interrupted=True,
                            pending_detached_job_ids=frozenset(current_pending_detached_job_ids),
                            pending_subagent_ids=frozenset(current_pending_subagent_ids),
                            deferred_tool_successes=tuple(deferred_tool_successes),
                            unexecuted_tool_names=tuple(call.name for call in tool_calls[tool_index:]),
                        )
                pending_approval = self._build_pending_approval(
                    tool_result=tool_result,
                    tool_name=tool_call.name,
                )
                if pending_approval is None:
                    observation = None
                    if tool_safety is not None:
                        observation = tool_safety.record(tool_call, tool_result)
                        tool_result = _with_tool_safety_replan_notice(
                            tool_result,
                            observation=observation,
                        )
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
                    if tool_safety is not None:
                        if bool(tool_result.metadata.get("tool_safety_blocked")):
                            safety_stop = True
                            safety_stop_reason = "blocked_tool_signature_reused"
                            safety_stop_unexecuted_tool_names = tuple(
                                call.name for call in tool_calls[tool_index + 1:]
                            )
                            safety_stop_unexecuted_tool_calls = tool_calls[tool_index + 1:]
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
                    pending_records.extend(ephemeral_image_records)
                    return _ToolExecutionOutcome(
                        interrupted=True,
                        pending_detached_job_ids=frozenset(current_pending_detached_job_ids),
                        pending_subagent_ids=frozenset(current_pending_subagent_ids),
                        deferred_tool_successes=tuple(deferred_tool_successes),
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

            if safety_stop:
                break

        pending_records.extend(ephemeral_image_records)
        if safety_stop_unexecuted_tool_calls:
            self._append_tool_safety_skips(
                session_id=session_id,
                pending_records=pending_records,
                tool_calls=safety_stop_unexecuted_tool_calls,
                reason=safety_stop_reason or "repeated_tool_result",
                turn_id=turn_id,
            )
        return _ToolExecutionOutcome(
            pending_detached_job_ids=frozenset(current_pending_detached_job_ids),
            pending_subagent_ids=frozenset(current_pending_subagent_ids),
            deferred_tool_successes=tuple(deferred_tool_successes),
            safety_stop=safety_stop,
            safety_stop_reason=safety_stop_reason,
            unexecuted_tool_names=safety_stop_unexecuted_tool_names,
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
    ) -> tuple[LLMRequest, int, tuple[ConversationRecord, ...]]:
        if pending_detached_job_ids:
            return self._build_orchestrator_monitored_waiting_request(
                session_id=session_id,
                base_records=base_records,
                pending_records=pending_records,
                pending_detached_job_ids=pending_detached_job_ids,
                pending_subagent_ids=pending_subagent_ids,
                turn_id=turn_id,
                extra_records=extra_records,
            )
        request, estimated_input_tokens = self._build_followup_request(
            session_id=session_id,
            base_records=base_records,
            pending_records=pending_records,
            extra_records=extra_records,
        )
        return request, estimated_input_tokens, ()

    def _deferred_tool_success_records(
        self,
        deferred_tool_successes: Sequence[_DeferredToolSuccess],
    ) -> tuple[ConversationRecord, ...]:
        records: list[ConversationRecord] = []
        for deferred in deferred_tool_successes:
            records.append(deferred.tool_record)
        for deferred in deferred_tool_successes:
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
        for deferred in deferred_tool_successes:
            for record in deferred.extra_records:
                self._append_turn_record(
                    session_id=session_id,
                    pending_records=pending_records,
                    record=record,
                )

    def _commit_staged_followup_records(
        self,
        *,
        session_id: str,
        pending_records: list[ConversationRecord],
        records: Sequence[ConversationRecord],
    ) -> None:
        for record in records:
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

        title = (
            "View image failed"
            if tool_result.name == "view_image"
            else (f"{tool_result.name.replace('_', ' ').capitalize()} failed")
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
        session_id: str,
        base_records: Sequence[ConversationRecord],
        pending_records: Sequence[ConversationRecord],
        allow_tools: bool = True,
        extra_records: Sequence[ConversationRecord] = (),
    ) -> tuple[LLMRequest, int]:
        activated_discoverable_tool_names = _collect_activated_discoverable_tool_names(
            pending_records
        )

        request = self._build_contextual_request(
            session_id=session_id,
            base_records=base_records,
            current_records=tuple(pending_records) + tuple(extra_records),
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
    ) -> tuple[LLMRequest, int, tuple[ConversationRecord, ...]]:
        waiting_record = self._build_orchestrator_monitored_waiting_record(
            session_id=session_id,
            pending_records=pending_records,
            pending_detached_job_ids=pending_detached_job_ids,
            pending_subagent_ids=pending_subagent_ids,
            turn_id=turn_id,
        )
        staged_records = (waiting_record,) if waiting_record is not None else ()
        request, estimated_input_tokens = self._build_followup_request(
            session_id=session_id,
            base_records=base_records,
            pending_records=pending_records,
            allow_tools=False,
            extra_records=tuple(extra_records) + staged_records,
        )
        return request, estimated_input_tokens, staged_records

    def _build_orchestrator_monitored_waiting_record(
        self,
        *,
        session_id: str,
        pending_records: Sequence[ConversationRecord],
        pending_detached_job_ids: Sequence[str],
        pending_subagent_ids: Sequence[str],
        turn_id: str,
    ) -> ConversationRecord | None:
        if not pending_detached_job_ids and not pending_subagent_ids:
            return None
        if any(record.metadata.get("orchestrator_monitored_waiting") for record in pending_records):
            return None

        metadata: dict[str, Any] = {}
        if pending_detached_job_ids:
            metadata["detached_bash_jobs_pending"] = True
            metadata["detached_bash_job_ids"] = list(pending_detached_job_ids)
        if pending_subagent_ids:
            metadata["subagents_pending"] = True
            metadata["pending_subagent_ids"] = list(pending_subagent_ids)

        return self._build_runtime_message_record(
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
        )

    def _build_tool_round_limit_record(
        self,
        *,
        session_id: str,
        attempted_round: int,
        turn_id: str,
    ) -> ConversationRecord:
        max_rounds = self._tool_settings.max_tool_rounds_per_turn
        _rounds, tool_safety = self._load_tool_task_state(session_id)
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content=(
                f"Tool execution slice completed {max_rounds} rounds. Pending round "
                f"{attempted_round} was not executed. "
                "A continuation is allowed only after a prior round made observable progress. "
                "Avoid repeating completed work and use tools only when they can produce new "
                "evidence or change.\n"
                + "\n".join(tool_safety.checkpoint_lines())
            ),
            metadata={
                _TOOL_ROUND_LIMIT_METADATA_KEY: True,
                "automatic_continuation": True,
                "attempted_round": attempted_round,
                "max_rounds": max_rounds,
            },
            turn_id=turn_id,
        )

    def _build_tool_safety_stop_record(
        self,
        *,
        session_id: str,
        turn_id: str,
        reason: str,
        pending_detached_job_ids: Iterable[str] = (),
        pending_subagent_ids: Iterable[str] = (),
    ) -> ConversationRecord:
        details = [
            _TOOL_SAFETY_STOP_TEXT,
            f"reason: {reason}",
        ]
        if pending_detached_job_ids:
            details.append(
                "pending_detached_bash_jobs: " + ", ".join(sorted(pending_detached_job_ids))
            )
        if pending_subagent_ids:
            details.append(
                "pending_subagents: " + ", ".join(sorted(pending_subagent_ids))
            )
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content="\n".join(details),
            metadata={
                "tool_safety_stop": True,
                "reason": reason,
                "pending_detached_job_ids": list(pending_detached_job_ids),
                "pending_subagent_ids": list(pending_subagent_ids),
            },
            turn_id=turn_id,
        )

    def _append_tool_safety_skips(
        self,
        *,
        session_id: str,
        pending_records: list[ConversationRecord],
        tool_calls: Sequence[ToolCall],
        reason: str,
        turn_id: str,
    ) -> None:
        for tool_call in tool_calls:
            result = ToolExecutionResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                ok=False,
                content=(
                    "Tool call skipped\n"
                    f"tool: {tool_call.name}\n"
                    "error_code: tool_safety_stop\n"
                    f"reason: {reason}"
                ),
                metadata={
                    "tool_safety_skipped": True,
                    "error_code": "tool_safety_stop",
                    "reason": reason,
                    "arguments": dict(tool_call.arguments),
                },
            )
            self._append_turn_record(
                session_id=session_id,
                pending_records=pending_records,
                record=self._build_tool_record(session_id, result, turn_id=turn_id),
            )

    def _append_tool_safety_stop_response(
        self,
        *,
        session_id: str,
        pending_records: list[ConversationRecord],
        turn_id: str,
        reason: str,
        response: LLMResponse,
    ) -> LLMResponse:
        LOGGER.warning(
            "Tool safety parked task session_id=%s turn_id=%s reason=%s",
            session_id,
            turn_id,
            reason,
        )
        stopped_response = replace(
            response,
            text="",
            tool_calls=[],
            finish_reason="stop",
            provider_metadata={
                **response.provider_metadata,
                "completion_blocked": True,
                "tool_safety_parked": True,
                "tool_safety_stop_reason": reason,
            },
        )
        return stopped_response

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
            prompt_cache_key=records[0].session_id if records else None,
        )

    def _build_contextual_request(
        self,
        *,
        session_id: str,
        base_records: Sequence[ConversationRecord],
        current_records: Sequence[ConversationRecord],
        activated_discoverable_tool_names: Sequence[str] = (),
        allow_tools: bool = True,
    ) -> LLMRequest:
        provider = self._effective_llm_provider()
        state = self._ensure_provider_session_state(
            session_id=session_id,
            provider=provider,
        )
        strategy = state.strategy if state is not None else strategy_for_provider(provider)
        tools = (
            self._compose_request_tools(activated_discoverable_tool_names) if allow_tools else ()
        )
        tool_choice = ToolChoice.auto() if allow_tools else ToolChoice.none()

        request_records = tuple(base_records) + tuple(current_records)
        prompt_cache_key = session_id if request_records else None
        previous_response_id: str | None = None
        stateful_continuation: StatefulContinuation | None = None
        conversation_id: str | None = None
        cached_content_name: str | None = None
        cached_content_model: str | None = None
        cached_content_messages: tuple[LLMMessage, ...] = ()
        cached_content_source_signature: str | None = None
        cached_content_source_record_ids: tuple[str, ...] = ()
        cached_content_media_ids: tuple[str, ...] = ()

        if strategy == ProviderContextStrategy.PROVIDER_STATEFUL_CONTINUATION:
            prompt_cache_key = None
            if provider == "openai" and state is not None:
                previous_response_id = state.openai.previous_response_id
                conversation_id = state.openai.conversation_id
                if previous_response_id or conversation_id:
                    request_records = _records_after_response_record(
                        current_records,
                        state.openai.last_response_record_id,
                    )
            elif provider == "grok" and state is not None:
                prompt_cache_key = session_id
                previous_response_id = state.grok.previous_response_id
                if previous_response_id:
                    request_records = _records_after_response_record(
                        request_records,
                        state.grok.last_response_record_id,
                    )

                storage_mode = state.grok.storage_mode
                if _records_have_image_inputs(request_records):
                    storage_mode = "ephemeral"
                recovery_message_loader: Callable[[], Sequence[LLMMessage]] | None = None
                if storage_mode == "ephemeral":
                    recovery_records = _records_between_response_records(
                        tuple(base_records) + tuple(current_records),
                        after_record_id=state.grok.durable_response_record_id,
                        through_record_id=state.grok.last_response_record_id,
                    )

                    def _load_grok_recovery_messages(
                        records: tuple[ConversationRecord, ...] = recovery_records,
                    ) -> tuple[LLMMessage, ...]:
                        return _records_to_grok_recovery_messages(records)

                    recovery_message_loader = _load_grok_recovery_messages
                stateful_continuation = StatefulContinuation(
                    session_key=session_id,
                    storage_mode=storage_mode,
                    durable_response_id=state.grok.durable_response_id,
                    recovery_message_loader=recovery_message_loader,
                    generation=state.grok.websocket_generation,
                )

            if not request_records:
                request_records = tuple(current_records) or tuple(base_records)

        elif strategy == ProviderContextStrategy.PROVIDER_CACHED_CONTEXT and provider == "gemini":
            prompt_cache_key = None
            stable_records = tuple(base_records)
            request_records = tuple(current_records) or tuple(base_records)
            if stable_records:
                cached_content_messages = _records_to_llm_messages(stable_records)
                cached_content_source_record_ids = tuple(
                    record.record_id
                    for record in stable_records
                    if record.kind == "message"
                    and not bool(record.metadata.get(_TRANSCRIPT_ONLY_RECORD_METADATA_KEY, False))
                )
                cached_content_media_ids = _collect_message_media_ids(cached_content_messages)
                cached_content_source_signature = _gemini_cache_source_signature(
                    records=stable_records,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                if state is not None and _gemini_cache_is_usable(
                    state,
                    source_signature=cached_content_source_signature,
                ):
                    cached_content_name = state.gemini.cached_content_name
                    cached_content_model = state.gemini.model

        messages = _records_to_llm_messages(request_records)
        return LLMRequest(
            messages=messages,
            provider=provider,
            tools=tools,
            tool_choice=tool_choice,
            prompt_cache_key=prompt_cache_key,
            previous_response_id=previous_response_id,
            stateful_continuation=stateful_continuation,
            conversation_id=conversation_id,
            cached_content_name=cached_content_name,
            cached_content_model=cached_content_model,
            cached_content_messages=cached_content_messages,
            cached_content_source_signature=cached_content_source_signature,
            cached_content_source_record_ids=cached_content_source_record_ids,
            cached_content_media_ids=cached_content_media_ids,
        )

    def _effective_llm_provider(self) -> str | None:
        if self._llm_provider:
            return self._llm_provider
        service_settings = getattr(self._llm_service, "settings", None)
        if service_settings is None:
            return None
        default_provider = getattr(service_settings, "default_provider", None)
        if not isinstance(default_provider, str):
            return None
        normalized = default_provider.strip().lower()
        return normalized or None

    def _ensure_provider_session_state(
        self,
        *,
        session_id: str,
        provider: str | None,
    ) -> ProviderSessionState | None:
        if provider is None:
            return None
        expected = ProviderSessionState.for_provider(provider)
        if expected is None:
            return None

        session = self._storage.get_session(session_id)
        if session is None:
            return expected
        existing = ProviderSessionState.from_mapping(session.provider_session_state)
        if (
            existing is not None
            and existing.provider == expected.provider
            and existing.strategy == expected.strategy
        ):
            return existing

        self._storage.update_session(
            session_id,
            provider_session_state=expected.to_dict(),
        )
        return expected

    def _reset_stateful_provider_continuation_for_replay(self, session_id: str) -> None:
        provider = self._effective_llm_provider()
        if provider is None:
            return
        state = self._ensure_provider_session_state(
            session_id=session_id,
            provider=provider,
        )
        if (
            state is None
            or state.strategy
            != ProviderContextStrategy.PROVIDER_STATEFUL_CONTINUATION
        ):
            return
        reset_state = ProviderSessionState.for_provider(provider)
        if reset_state is None:
            return
        self._storage.update_session(
            session_id,
            provider_session_state=reset_state.to_dict(),
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
                    boundary="tool_slice",
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
        self._reset_stateful_provider_continuation_for_replay(session_id)
        return self._build_followup_request(
            session_id=session_id,
            base_records=base_records,
            pending_records=pending_records,
            allow_tools=True,
        )

    def _normalize_tool_round_limit_recovery_response(
        self,
        response: LLMResponse,
    ) -> LLMResponse:
        if response.text.strip() or response.tool_calls:
            return response
        return replace(
            response,
            text=_TOOL_ROUND_CONTINUATION_EMPTY_TEXT,
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
        response = await self._await_with_stop(
            self._llm_service.generate(request),
            turn_id=turn_id,
            operation="llm_generate",
        )
        normalized = self._normalize_tool_round_limit_recovery_response(response)
        assistant_record = self._build_assistant_record(
            session_id,
            normalized,
            turn_id=turn_id,
        )
        self._append_turn_record(
            session_id=session_id,
            pending_records=pending_records,
            record=assistant_record,
        )
        self._persist_provider_session_state_from_response(
            session_id=session_id,
            response=normalized,
            assistant_record=assistant_record,
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
        async for event in self._stream_generate_with_stop(request, turn_id=turn_id):
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
        assistant_record = self._build_assistant_record(
            session_id,
            normalized,
            turn_id=turn_id,
        )
        self._append_turn_record(
            session_id=session_id,
            pending_records=pending_records,
            record=assistant_record,
        )
        self._persist_provider_session_state_from_response(
            session_id=session_id,
            response=normalized,
            assistant_record=assistant_record,
        )
        if normalized.text:
            recovery_events.append(
                AgentAssistantMessageEvent(
                    session_id=session_id,
                    text=normalized.text,
                    turn_id=turn_id,
                )
            )
        if normalized.tool_calls:
            recovery_events.append(
                AgentToolCallEvent(
                    session_id=session_id,
                    tool_names=tuple(call.name for call in normalized.tool_calls),
                    turn_id=turn_id,
                )
            )
        return recovery_events, normalized, estimated_input_tokens

    async def _stream_message_turn(  # pyright: ignore[reportGeneralTypeIssues] - orchestration state machine
        self,
        user_text: str | None,
        *,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        turn_id = uuid4().hex
        try:
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
                task_id=turn_id,
            )
        except _CompactionStopRequested as exc:
            yield AgentTurnDoneEvent(
                session_id=exc.session_id,
                response_text="",
                command=command_override,
                interrupted=True,
                interruption_reason=exc.interruption_reason,
            )
            return
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

        interrupted_response_text = ""
        interrupted_stream_fragment_text = ""
        interrupted_unexecuted_tool_names: tuple[str, ...] = ()
        _initial_rounds, initial_tool_safety = self._load_tool_task_state(
            session.session_id
        )
        suppress_initial_text_stream = (
            initial_tool_safety.unverified_workspace_mutation
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
                    async for event in self._stream_generate_with_stop(request, turn_id=turn_id):
                        if event.type == "text_delta":
                            emitted_any = True
                            if event.delta:
                                streamed_initial_text += event.delta
                                interrupted_response_text = streamed_initial_text
                                interrupted_stream_fragment_text = streamed_initial_text
                                if not suppress_initial_text_stream:
                                    yield AgentTextDeltaEvent(
                                        session_id=session.session_id,
                                        delta=event.delta,
                                        turn_id=turn_id,
                                    )
                        elif event.type == "tool_call_delta":
                            emitted_any = True
                            tool_name = str(event.tool_name or "").strip()
                            call_id = event.call_id.strip()
                            if (
                                tool_name
                                and call_id
                                and call_id not in noticed_initial_tool_call_ids
                            ):
                                noticed_initial_tool_call_ids.add(call_id)
                                interrupted_unexecuted_tool_names = (
                                    *interrupted_unexecuted_tool_names,
                                    tool_name,
                                )
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
                    interrupted_response_text = initial_response.text
                    interrupted_unexecuted_tool_names = tuple(
                        call.name for call in initial_response.tool_calls
                    )
                    interrupted_stream_fragment_text = ""
                    break
                except ProviderBadRequestError as exc:
                    if (
                        overflow_retry_attempted
                        or emitted_any
                        or not _is_context_overflow_error(exc)
                    ):
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

            if not initial_response.tool_calls:
                _rounds, persisted_tool_safety = self._load_tool_task_state(
                    session.session_id
                )
                initial_response = _enforce_acceptance_handoff(
                    initial_response,
                    tool_safety=persisted_tool_safety,
                )

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
            session = (
                self._persist_provider_session_state_from_response(
                    session_id=session.session_id,
                    response=initial_response,
                    assistant_record=final_initial_record,
                )
                or session
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
            interrupted_response_text = current_response.text
            interrupted_unexecuted_tool_names = tuple(
                call.name for call in current_response.tool_calls
            )
            tool_rounds = 0
            task_tool_rounds, tool_safety = self._load_tool_task_state(
                session.session_id
            )
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
                        unexecuted_tool_names=tuple(
                            call.name for call in current_response.tool_calls
                        ),
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
                task_tool_rounds += 1
                if task_tool_rounds > self._tool_settings.max_tool_rounds_per_task:
                    self._persist_tool_task_state(
                        session.session_id,
                        rounds=task_tool_rounds,
                        tracker=tool_safety,
                    )
                    self._append_tool_safety_skips(
                        session_id=session.session_id,
                        pending_records=pending_records,
                        tool_calls=current_response.tool_calls,
                        reason="task_tool_round_budget_exhausted",
                        turn_id=turn_id,
                    )
                    self._append_turn_record(
                        session_id=session.session_id,
                        pending_records=pending_records,
                        record=self._build_tool_safety_stop_record(
                            session_id=session.session_id,
                            turn_id=turn_id,
                            reason="task_tool_round_budget_exhausted",
                            pending_detached_job_ids=pending_detached_job_ids,
                            pending_subagent_ids=pending_subagent_ids,
                        ),
                    )
                    current_response = self._append_tool_safety_stop_response(
                        session_id=session.session_id,
                        pending_records=pending_records,
                        turn_id=turn_id,
                        reason="task_tool_round_budget_exhausted",
                        response=current_response,
                    )
                    break
                if tool_rounds > self._tool_settings.max_tool_rounds_per_turn:
                    if not tool_safety.consume_slice_progress():
                        self._append_tool_safety_skips(
                            session_id=session.session_id,
                            pending_records=pending_records,
                            tool_calls=current_response.tool_calls,
                            reason="tool_slice_without_progress",
                            turn_id=turn_id,
                        )
                        self._append_turn_record(
                            session_id=session.session_id,
                            pending_records=pending_records,
                            record=self._build_tool_safety_stop_record(
                                session_id=session.session_id,
                                turn_id=turn_id,
                                reason="tool_slice_without_progress",
                                pending_detached_job_ids=pending_detached_job_ids,
                                pending_subagent_ids=pending_subagent_ids,
                            ),
                        )
                        current_response = self._append_tool_safety_stop_response(
                            session_id=session.session_id,
                            pending_records=pending_records,
                            turn_id=turn_id,
                            reason="tool_slice_without_progress",
                            response=current_response,
                        )
                        break
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
                    interrupted_response_text = current_response.text
                    interrupted_unexecuted_tool_names = tuple(
                        call.name for call in current_response.tool_calls
                    )
                    tool_rounds = 0
                    continue

                followup_compaction_attempted = False
                deferred_tool_successes: tuple[_DeferredToolSuccess, ...] = ()
                staged_followup_records: tuple[ConversationRecord, ...] = ()
                try:
                    ephemeral_image_records: list[ConversationRecord] = []
                    staged_image_tool_successes: list[_DeferredToolSuccess] = []
                    approval_rejected = False
                    safety_stop = False
                    safety_stop_reason: str | None = None
                    safety_stop_unexecuted_tool_calls: tuple[ToolCall, ...] = ()
                    current_pending_detached_job_ids = set(pending_detached_job_ids)
                    current_pending_subagent_ids = set(pending_subagent_ids)
                    tool_calls = tuple(current_response.tool_calls)
                    for tool_index, tool_call in enumerate(tool_calls):
                        tool_context = replace(
                            self._tool_context,
                            session_id=session.session_id,
                            turn_id=turn_id,
                        )
                        while True:
                            blocked_reason = tool_safety.blocked_call_reason(tool_call)
                            if blocked_reason is not None:
                                tool_result = build_blocked_repetition_result(
                                    tool_call=tool_call,
                                    reason=blocked_reason,
                                    diagnostics=tool_safety.blocked_call_details(tool_call),
                                )
                            else:
                                try:
                                    tool_result = await self._await_with_stop(
                                        self._tool_executor(tool_call, tool_context),
                                        turn_id=turn_id,
                                        operation=f"tool_{tool_call.name}",
                                    )
                                except _TurnStopRequested:
                                    pending_records.extend(ephemeral_image_records)
                                    deferred_tool_successes = tuple(staged_image_tool_successes)
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
                                        unexecuted_tool_names=tuple(
                                            call.name for call in tool_calls[tool_index:]
                                        ),
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
                            pending_approval = self._build_pending_approval(
                                tool_result=tool_result,
                                tool_name=tool_call.name,
                            )
                            if pending_approval is None:
                                observation = tool_safety.record(tool_call, tool_result)
                                tool_result = _with_tool_safety_replan_notice(
                                    tool_result,
                                    observation=observation,
                                )
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
                                if bool(tool_result.metadata.get("tool_safety_blocked")):
                                    safety_stop = True
                                    safety_stop_reason = "blocked_tool_signature_reused"
                                    safety_stop_unexecuted_tool_calls = tool_calls[
                                        tool_index + 1:
                                    ]
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
                                pending_records.extend(ephemeral_image_records)
                                deferred_tool_successes = tuple(staged_image_tool_successes)
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
                        if safety_stop:
                            break

                    pending_records.extend(ephemeral_image_records)
                    deferred_tool_successes = tuple(staged_image_tool_successes)
                    pending_detached_job_ids = frozenset(current_pending_detached_job_ids)
                    pending_subagent_ids = frozenset(current_pending_subagent_ids)
                    self._persist_tool_task_state(
                        session.session_id,
                        rounds=task_tool_rounds,
                        tracker=tool_safety,
                    )
                    if approval_rejected:
                        if deferred_tool_successes:
                            self._commit_deferred_tool_successes(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                deferred_tool_successes=deferred_tool_successes,
                            )
                        break
                    if safety_stop:
                        if deferred_tool_successes:
                            self._commit_deferred_tool_successes(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                deferred_tool_successes=deferred_tool_successes,
                            )
                        self._append_tool_safety_skips(
                            session_id=session.session_id,
                            pending_records=pending_records,
                            tool_calls=safety_stop_unexecuted_tool_calls,
                            reason=safety_stop_reason or "repeated_tool_result",
                            turn_id=turn_id,
                        )
                        self._append_turn_record(
                            session_id=session.session_id,
                            pending_records=pending_records,
                            record=self._build_tool_safety_stop_record(
                                session_id=session.session_id,
                                turn_id=turn_id,
                                reason=safety_stop_reason or "repeated_tool_result",
                                pending_detached_job_ids=pending_detached_job_ids,
                                pending_subagent_ids=pending_subagent_ids,
                            ),
                        )
                        current_response = self._append_tool_safety_stop_response(
                            session_id=session.session_id,
                            pending_records=pending_records,
                            turn_id=turn_id,
                            reason=safety_stop_reason or "repeated_tool_result",
                            response=current_response,
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
                    interrupted_unexecuted_tool_names = ()
                    (
                        request,
                        final_estimated_input_tokens,
                        staged_followup_records,
                    ) = self._build_followup_attempt_request(
                        session_id=session.session_id,
                        base_records=base_records,
                        pending_records=pending_records,
                        pending_detached_job_ids=pending_detached_job_ids,
                        pending_subagent_ids=pending_subagent_ids,
                        turn_id=turn_id,
                        extra_records=self._deferred_tool_success_records(deferred_tool_successes),
                    )
                except _TurnStopRequested:
                    interrupted = self._interrupt_turn(
                        session_id=session.session_id,
                        turn_id=turn_id,
                        command=command_override,
                        compaction_performed=did_compaction,
                        response_text=current_response.text,
                        unexecuted_tool_names=tuple(
                            call.name for call in current_response.tool_calls
                        ),
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
                except ContextBudgetError:
                    if deferred_tool_successes:
                        self._commit_deferred_tool_successes(
                            session_id=session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                        )
                    staged_followup_records = ()
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
                    suppress_followup_text_stream = (
                        tool_safety.unverified_workspace_mutation
                    )
                    try:
                        async for event in self._stream_generate_with_stop(
                            request, turn_id=turn_id
                        ):
                            if not deferred_committed:
                                if deferred_tool_successes:
                                    self._commit_deferred_tool_successes(
                                        session_id=session.session_id,
                                        pending_records=pending_records,
                                        deferred_tool_successes=deferred_tool_successes,
                                    )
                                    deferred_tool_successes = ()
                                if staged_followup_records:
                                    self._commit_staged_followup_records(
                                        session_id=session.session_id,
                                        pending_records=pending_records,
                                        records=staged_followup_records,
                                    )
                                    staged_followup_records = ()
                                deferred_committed = True
                            if event.type == "text_delta":
                                emitted_any = True
                                if event.delta:
                                    streamed_followup_text += event.delta
                                    interrupted_response_text = streamed_followup_text
                                    interrupted_stream_fragment_text = streamed_followup_text
                                    if not suppress_followup_text_stream:
                                        yield AgentTextDeltaEvent(
                                            session_id=session.session_id,
                                            delta=event.delta,
                                            turn_id=turn_id,
                                        )
                            elif event.type == "tool_call_delta":
                                emitted_any = True
                                tool_name = str(event.tool_name or "").strip()
                                call_id = event.call_id.strip()
                                if (
                                    tool_name
                                    and call_id
                                    and call_id not in noticed_followup_tool_call_ids
                                ):
                                    noticed_followup_tool_call_ids.add(call_id)
                                    interrupted_unexecuted_tool_names = (
                                        *interrupted_unexecuted_tool_names,
                                        tool_name,
                                    )
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
                        if not deferred_committed:
                            if deferred_tool_successes:
                                self._commit_deferred_tool_successes(
                                    session_id=session.session_id,
                                    pending_records=pending_records,
                                    deferred_tool_successes=deferred_tool_successes,
                                )
                                deferred_tool_successes = ()
                            if staged_followup_records:
                                self._commit_staged_followup_records(
                                    session_id=session.session_id,
                                    pending_records=pending_records,
                                    records=staged_followup_records,
                                )
                                staged_followup_records = ()
                        break
                    except _TurnStopRequested:
                        if deferred_tool_successes:
                            self._commit_deferred_tool_successes(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                deferred_tool_successes=deferred_tool_successes,
                            )
                            deferred_tool_successes = ()
                        if staged_followup_records:
                            self._commit_staged_followup_records(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                records=staged_followup_records,
                            )
                            staged_followup_records = ()
                        if interrupted_stream_fragment_text:
                            partial_record = self._build_streamed_assistant_text_record(
                                session_id=session.session_id,
                                text=interrupted_stream_fragment_text,
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
                            response_text=interrupted_response_text or current_response.text,
                            unexecuted_tool_names=interrupted_unexecuted_tool_names,
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
                    except (LLMConfigurationError, UnsupportedCapabilityError) as exc:
                        if not deferred_tool_successes or not _is_image_attachment_request_error(
                            exc
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
                        (
                            request,
                            final_estimated_input_tokens,
                            staged_followup_records,
                        ) = self._build_followup_attempt_request(
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
                            (
                                request,
                                final_estimated_input_tokens,
                                staged_followup_records,
                            ) = self._build_followup_attempt_request(
                                session_id=session.session_id,
                                base_records=base_records,
                                pending_records=pending_records,
                                pending_detached_job_ids=pending_detached_job_ids,
                                pending_subagent_ids=pending_subagent_ids,
                                turn_id=turn_id,
                            )
                            continue
                        if not _is_context_overflow_error(exc) or emitted_any:
                            raise
                        if deferred_tool_successes:
                            self._commit_deferred_tool_successes(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                deferred_tool_successes=deferred_tool_successes,
                            )
                            deferred_tool_successes = ()
                        if staged_followup_records:
                            self._commit_staged_followup_records(
                                session_id=session.session_id,
                                pending_records=pending_records,
                                records=staged_followup_records,
                            )
                            staged_followup_records = ()
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
                if not current_response.tool_calls:
                    current_response = _enforce_acceptance_handoff(
                        current_response,
                        tool_safety=tool_safety,
                    )
                interrupted_response_text = current_response.text
                interrupted_unexecuted_tool_names = tuple(
                    call.name for call in current_response.tool_calls
                )
                interrupted_stream_fragment_text = ""
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
                session = (
                    self._persist_provider_session_state_from_response(
                        session_id=session.session_id,
                        response=current_response,
                        assistant_record=final_followup_record,
                    )
                    or session
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
                        unexecuted_tool_names=tuple(
                            call.name for call in current_response.tool_calls
                        ),
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

            completion_blocked = _response_completion_blocked(final_response)
            self._persist_successful_turn(
                session_id=session.session_id,
                turn_id=turn_id,
                response=final_response,
                estimated_input_tokens=final_estimated_input_tokens,
            )

            refreshed = self._storage.get_session(session.session_id)
            threshold_observed = (
                final_response.usage.input_tokens
                if final_response.usage is not None
                and final_response.usage.input_tokens is not None
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
                completion_blocked=completion_blocked,
            )
        except _TurnStopRequested:
            if interrupted_stream_fragment_text:
                partial_record = self._build_streamed_assistant_text_record(
                    session_id=session.session_id,
                    text=interrupted_stream_fragment_text,
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
                response_text=interrupted_response_text,
                unexecuted_tool_names=interrupted_unexecuted_tool_names,
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
        except Exception:
            if interrupted_stream_fragment_text:
                partial_record = self._build_streamed_assistant_text_record(
                    session_id=session.session_id,
                    text=interrupted_stream_fragment_text,
                    turn_id=turn_id,
                    reason="runtime_error",
                )
                if partial_record is not None:
                    self._append_turn_record(
                        session_id=session.session_id,
                        pending_records=pending_records,
                        record=partial_record,
                    )
            self._fail_turn_after_runtime_error(
                session_id=session.session_id,
                turn_id=turn_id,
            )
            raise
        finally:
            self._clear_turn_control(turn_id)

    async def _prepare_turn(
        self,
        *,
        user_text: str | None,
        task_id: str,
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
        task_contract = self._prepare_tool_task(
            session_id=session.session_id,
            proposed_task_id=task_id,
            user_text=user_text,
        )
        if task_contract is not None:
            turn_runtime_messages = (
                *turn_runtime_messages,
                AgentRuntimeMessage(
                    role="system",
                    content=task_contract.render(),
                    metadata={
                        "task_contract": True,
                        "task_id": task_contract.task_id,
                        "user_message_sha256": task_contract.user_message_sha256,
                    },
                ),
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
            response = await self._await_with_stop(
                self._llm_service.generate(request),
                turn_id=turn_id,
                operation="llm_generate",
            )
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
        response = await self._await_with_stop(
            self._llm_service.generate(retry_request),
            turn_id=turn_id,
            operation="llm_generate",
        )
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
        if reason == "new_session":
            return _PREVIOUS_SESSION_RESET_TEXT
        return _PREVIOUS_TASK_INTERRUPTED_TEXT

    def _persist_successful_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        response: LLMResponse,
        estimated_input_tokens: int,
    ) -> None:
        completion_blocked = _response_completion_blocked(response)
        self._finish_turn(
            session_id=session_id,
            turn_id=turn_id,
            status="blocked" if completion_blocked else "completed",
        )
        usage = response.usage
        session = self._storage.get_session(session_id)
        backend_state = dict(session.backend_state) if session is not None else {}
        if not completion_blocked:
            backend_state.pop("active_tool_task_id", None)
            backend_state.pop("tool_task_state", None)
        self._storage.update_session(
            session_id,
            pending_interruption_notice=False,
            pending_interruption_notice_reason=None,
            last_input_tokens=usage.input_tokens if usage is not None else None,
            last_output_tokens=usage.output_tokens if usage is not None else None,
            last_total_tokens=usage.total_tokens if usage is not None else None,
            last_estimated_input_tokens=estimated_input_tokens,
            backend_state=backend_state,
        )

    def _prepare_tool_task(
        self,
        *,
        session_id: str,
        proposed_task_id: str,
        user_text: str | None,
    ) -> TaskContract | None:
        session = self._storage.get_session(session_id)
        if session is None:
            return None
        backend_state = dict(session.backend_state)
        active_task_id = str(backend_state.get("active_tool_task_id", "")).strip()
        active_state = (
            self._storage.load_tool_task_state(active_task_id)
            if active_task_id
            else None
        )
        active_contract = TaskContract.from_state(
            active_state.get("contract") if isinstance(active_state, dict) else None
        )

        if user_text is None:
            return active_contract
        if (
            active_contract is not None
            and isinstance(active_state, dict)
            and user_message_explicitly_resumes_task(user_text)
        ):
            supplemental = build_task_contract(
                task_id=active_contract.task_id,
                origin_turn_id=active_contract.origin_turn_id,
                user_text=user_text,
            )
            merged_requirements = {
                item.item_id: item for item in active_contract.requirements
            }
            merged_requirements.update(
                {item.item_id: item for item in supplemental.requirements}
            )
            contract = replace(
                active_contract,
                requirements=tuple(merged_requirements.values()),
            )
            tracker = ToolSafetyTracker.from_state(active_state.get("tracker"))
            tracker.seed_contract_requirements(contract.requirements)
            self._storage.write_tool_task_state(
                contract.task_id,
                {
                    **active_state,
                    "contract": contract.to_state(),
                    "tracker": tracker.to_state(),
                },
            )
            return contract

        contract = build_task_contract(
            task_id=proposed_task_id,
            origin_turn_id=proposed_task_id,
            user_text=user_text,
        )
        tracker = ToolSafetyTracker()
        tracker.seed_contract_requirements(contract.requirements)
        self._storage.write_tool_task_state(
            contract.task_id,
            {
                "schema_version": 1,
                "task_id": contract.task_id,
                "origin_session_id": session_id,
                "contract": contract.to_state(),
                "rounds": 0,
                "tracker": tracker.to_state(),
            },
        )
        backend_state["active_tool_task_id"] = contract.task_id
        backend_state.pop("tool_task_state", None)
        if backend_state != session.backend_state:
            self._storage.update_session(session_id, backend_state=backend_state)
        return contract

    def _load_tool_task_state(self, session_id: str) -> tuple[int, ToolSafetyTracker]:
        session = self._storage.get_session(session_id)
        if session is None:
            return 0, ToolSafetyTracker()
        task_id = str(session.backend_state.get("active_tool_task_id", "")).strip()
        raw = self._storage.load_tool_task_state(task_id) if task_id else None
        if not isinstance(raw, dict):
            # Read the old inline shape during rolling upgrades, but never persist it again.
            raw = session.backend_state.get("tool_task_state")
        if not isinstance(raw, dict):
            return 0, ToolSafetyTracker()
        try:
            rounds = max(0, int(raw.get("rounds", 0)))
        except (TypeError, ValueError):
            rounds = 0
        tracker = ToolSafetyTracker.from_state(raw.get("tracker"))
        contract = TaskContract.from_state(raw.get("contract"))
        if contract is not None:
            tracker.seed_contract_requirements(contract.requirements)
        return rounds, tracker

    def _persist_tool_task_state(
        self,
        session_id: str,
        *,
        rounds: int,
        tracker: ToolSafetyTracker,
    ) -> None:
        session = self._storage.get_session(session_id)
        if session is None:
            return
        task_id = str(session.backend_state.get("active_tool_task_id", "")).strip()
        if not task_id:
            return
        existing = self._storage.load_tool_task_state(task_id) or {}
        self._storage.write_tool_task_state(
            task_id,
            {
                **existing,
                "schema_version": 1,
                "task_id": task_id,
                "rounds": max(0, rounds),
                "tracker": tracker.to_state(),
            },
        )

    def _persist_provider_session_state_from_response(
        self,
        *,
        session_id: str,
        response: LLMResponse,
        assistant_record: ConversationRecord | None,
    ) -> SessionMetadata | None:
        provider = self._effective_llm_provider()
        if provider is None:
            return self._storage.get_session(session_id)
        state = self._ensure_provider_session_state(
            session_id=session_id,
            provider=provider,
        )
        if state is None:
            return self._storage.get_session(session_id)

        response_record_id = assistant_record.record_id if assistant_record is not None else None
        if provider == "openai":
            state = replace(
                state,
                openai=replace(
                    state.openai,
                    conversation_id=(
                        _metadata_str(response.provider_metadata, "conversation_id")
                        or state.openai.conversation_id
                    ),
                    previous_response_id=response.response_id or state.openai.previous_response_id,
                    last_response_record_id=response_record_id
                    or state.openai.last_response_record_id,
                ),
            )
        elif provider == "grok":
            storage_mode = _metadata_str(
                response.provider_metadata,
                "response_storage_mode",
            )
            if storage_mode not in {"durable", "ephemeral"}:
                storage_mode = state.grok.storage_mode
            resolved_storage_mode: Literal["durable", "ephemeral"] = (
                "ephemeral" if storage_mode == "ephemeral" else "durable"
            )
            durable_response_id = _metadata_str(
                response.provider_metadata,
                "durable_response_id",
            )
            websocket_generation = _metadata_int(
                response.provider_metadata,
                "websocket_generation",
            )
            if resolved_storage_mode == "durable":
                durable_response_id = response.response_id or durable_response_id
                durable_response_record_id = (
                    response_record_id or state.grok.durable_response_record_id
                )
            else:
                durable_response_id = durable_response_id or state.grok.durable_response_id
                durable_response_record_id = state.grok.durable_response_record_id
            state = replace(
                state,
                grok=replace(
                    state.grok,
                    previous_response_id=response.response_id or state.grok.previous_response_id,
                    last_response_record_id=response_record_id
                    or state.grok.last_response_record_id,
                    durable_response_id=durable_response_id,
                    durable_response_record_id=durable_response_record_id,
                    storage_mode=resolved_storage_mode,
                    websocket_generation=(
                        websocket_generation
                        if websocket_generation is not None
                        else state.grok.websocket_generation
                    ),
                ),
            )
        elif provider == "gemini":
            state = replace(
                state,
                gemini=replace(
                    state.gemini,
                    cached_content_name=(
                        _metadata_str(response.provider_metadata, "cached_content_name")
                        or state.gemini.cached_content_name
                    ),
                    cache_expires_at=(
                        _metadata_str(response.provider_metadata, "cached_content_expires_at")
                        or state.gemini.cache_expires_at
                    ),
                    cached_media_ids=(
                        _metadata_string_tuple(
                            response.provider_metadata,
                            "cached_content_media_ids",
                        )
                        if "cached_content_media_ids" in response.provider_metadata
                        else state.gemini.cached_media_ids
                    ),
                    source_record_ids=(
                        _metadata_string_tuple(
                            response.provider_metadata,
                            "cached_content_source_record_ids",
                        )
                        if "cached_content_source_record_ids" in response.provider_metadata
                        else state.gemini.source_record_ids
                    ),
                    model=(
                        _metadata_str(response.provider_metadata, "cached_content_model")
                        or response.model
                        or state.gemini.model
                    ),
                    source_signature=(
                        _metadata_str(
                            response.provider_metadata,
                            "cached_content_source_signature",
                        )
                        or state.gemini.source_signature
                    ),
                ),
            )

        return self._storage.update_session(
            session_id,
            provider_session_state=state.to_dict(),
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
        return self._build_contextual_request(
            session_id=session_id,
            base_records=records,
            current_records=turn_records,
            allow_tools=allow_tools,
        )

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
        task_tool_rounds, tool_safety = self._load_tool_task_state(session.session_id)
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
            task_tool_rounds += 1
            if task_tool_rounds > self._tool_settings.max_tool_rounds_per_task:
                self._persist_tool_task_state(
                    current_session.session_id,
                    rounds=task_tool_rounds,
                    tracker=tool_safety,
                )
                self._append_tool_safety_skips(
                    session_id=current_session.session_id,
                    pending_records=pending_records,
                    tool_calls=current_response.tool_calls,
                    reason="task_tool_round_budget_exhausted",
                    turn_id=turn_id,
                )
                self._append_turn_record(
                    session_id=current_session.session_id,
                    pending_records=pending_records,
                    record=self._build_tool_safety_stop_record(
                        session_id=current_session.session_id,
                        turn_id=turn_id,
                        reason="task_tool_round_budget_exhausted",
                        pending_detached_job_ids=pending_detached_job_ids,
                        pending_subagent_ids=pending_subagent_ids,
                    ),
                )
                current_response = self._append_tool_safety_stop_response(
                    session_id=current_session.session_id,
                    pending_records=pending_records,
                    turn_id=turn_id,
                    reason="task_tool_round_budget_exhausted",
                    response=current_response,
                )
                break
            if tool_rounds > self._tool_settings.max_tool_rounds_per_turn:
                if not tool_safety.consume_slice_progress():
                    self._append_tool_safety_skips(
                        session_id=current_session.session_id,
                        pending_records=pending_records,
                        tool_calls=current_response.tool_calls,
                        reason="tool_slice_without_progress",
                        turn_id=turn_id,
                    )
                    self._append_turn_record(
                        session_id=current_session.session_id,
                        pending_records=pending_records,
                        record=self._build_tool_safety_stop_record(
                            session_id=current_session.session_id,
                            turn_id=turn_id,
                            reason="tool_slice_without_progress",
                            pending_detached_job_ids=pending_detached_job_ids,
                            pending_subagent_ids=pending_subagent_ids,
                        ),
                    )
                    current_response = self._append_tool_safety_stop_response(
                        session_id=current_session.session_id,
                        pending_records=pending_records,
                        turn_id=turn_id,
                        reason="tool_slice_without_progress",
                        response=current_response,
                    )
                    break
                try:
                    (
                        current_response,
                        current_estimated_input_tokens,
                    ) = await self._recover_from_tool_round_limit(
                        session_id=current_session.session_id,
                        base_records=current_base_records,
                        pending_records=pending_records,
                        attempted_round=tool_rounds,
                        unexecuted_tool_names=tuple(
                            call.name for call in current_response.tool_calls
                        ),
                        turn_id=turn_id,
                    )
                except _TurnStopRequested:
                    return (
                        current_session,
                        current_response,
                        current_estimated_input_tokens,
                        did_compaction,
                        True,
                        approval_rejected,
                        (),
                    )
                tool_rounds = 0
                continue

            followup_compaction_attempted = False
            deferred_tool_successes: tuple[_DeferredToolSuccess, ...] = ()
            staged_followup_records: tuple[ConversationRecord, ...] = ()
            try:
                tool_execution_outcome = await self._execute_tool_calls(
                    session_id=current_session.session_id,
                    pending_records=pending_records,
                    current_response=current_response,
                    turn_id=turn_id,
                    pending_detached_job_ids=pending_detached_job_ids,
                    pending_subagent_ids=pending_subagent_ids,
                    tool_safety=tool_safety,
                )
                pending_detached_job_ids = tool_execution_outcome.pending_detached_job_ids
                pending_subagent_ids = tool_execution_outcome.pending_subagent_ids
                deferred_tool_successes = tool_execution_outcome.deferred_tool_successes
                self._persist_tool_task_state(
                    current_session.session_id,
                    rounds=task_tool_rounds,
                    tracker=tool_safety,
                )
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
                        tool_execution_outcome.unexecuted_tool_names,
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
                if tool_execution_outcome.safety_stop:
                    if deferred_tool_successes:
                        self._commit_deferred_tool_successes(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                        )
                    self._append_turn_record(
                        session_id=current_session.session_id,
                        pending_records=pending_records,
                        record=self._build_tool_safety_stop_record(
                            session_id=current_session.session_id,
                            turn_id=turn_id,
                            reason=(
                                tool_execution_outcome.safety_stop_reason
                                or "repeated_tool_result"
                            ),
                            pending_detached_job_ids=pending_detached_job_ids,
                            pending_subagent_ids=pending_subagent_ids,
                        ),
                    )
                    current_response = self._append_tool_safety_stop_response(
                        session_id=current_session.session_id,
                        pending_records=pending_records,
                        turn_id=turn_id,
                        reason=(
                            tool_execution_outcome.safety_stop_reason
                            or "repeated_tool_result"
                        ),
                        response=current_response,
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
                (
                    request,
                    current_estimated_input_tokens,
                    staged_followup_records,
                ) = self._build_followup_attempt_request(
                    session_id=current_session.session_id,
                    base_records=current_base_records,
                    pending_records=pending_records,
                    pending_detached_job_ids=pending_detached_job_ids,
                    pending_subagent_ids=pending_subagent_ids,
                    turn_id=turn_id,
                    extra_records=self._deferred_tool_success_records(deferred_tool_successes),
                )
            except _TurnStopRequested:
                return (
                    current_session,
                    current_response,
                    current_estimated_input_tokens,
                    did_compaction,
                    True,
                    approval_rejected,
                    tuple(call.name for call in current_response.tool_calls),
                )
            except ContextBudgetError:
                if deferred_tool_successes:
                    self._commit_deferred_tool_successes(
                        session_id=current_session.session_id,
                        pending_records=pending_records,
                        deferred_tool_successes=deferred_tool_successes,
                    )
                staged_followup_records = ()
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
                    current_response = await self._await_with_stop(
                        self._llm_service.generate(request),
                        turn_id=turn_id,
                        operation="llm_generate",
                    )
                    if deferred_tool_successes:
                        self._commit_deferred_tool_successes(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                        )
                        deferred_tool_successes = ()
                    if staged_followup_records:
                        self._commit_staged_followup_records(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            records=staged_followup_records,
                        )
                        staged_followup_records = ()
                    break
                except _TurnStopRequested:
                    if deferred_tool_successes:
                        self._commit_deferred_tool_successes(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                        )
                        deferred_tool_successes = ()
                    if staged_followup_records:
                        self._commit_staged_followup_records(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            records=staged_followup_records,
                        )
                        staged_followup_records = ()
                    return (
                        current_session,
                        current_response,
                        current_estimated_input_tokens,
                        did_compaction,
                        True,
                        approval_rejected,
                        (),
                    )
                except (LLMConfigurationError, UnsupportedCapabilityError) as exc:
                    if not deferred_tool_successes or not _is_image_attachment_request_error(exc):
                        raise
                    self._persist_failed_deferred_tool_successes(
                        session_id=current_session.session_id,
                        pending_records=pending_records,
                        deferred_tool_successes=deferred_tool_successes,
                        error_message=str(exc),
                        turn_id=turn_id,
                    )
                    deferred_tool_successes = ()
                    (
                        request,
                        current_estimated_input_tokens,
                        staged_followup_records,
                    ) = self._build_followup_attempt_request(
                        session_id=current_session.session_id,
                        base_records=current_base_records,
                        pending_records=pending_records,
                        pending_detached_job_ids=pending_detached_job_ids,
                        pending_subagent_ids=pending_subagent_ids,
                        turn_id=turn_id,
                    )
                    continue
                except ProviderBadRequestError as exc:
                    if deferred_tool_successes and _is_image_attachment_request_error(exc):
                        self._persist_failed_deferred_tool_successes(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            deferred_tool_successes=deferred_tool_successes,
                            error_message=str(exc),
                            turn_id=turn_id,
                        )
                        deferred_tool_successes = ()
                        (
                            request,
                            current_estimated_input_tokens,
                            staged_followup_records,
                        ) = self._build_followup_attempt_request(
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
                    if staged_followup_records:
                        self._commit_staged_followup_records(
                            session_id=current_session.session_id,
                            pending_records=pending_records,
                            records=staged_followup_records,
                        )
                        staged_followup_records = ()
                    if followup_compaction_attempted:
                        raise ContextBudgetError(_FOLLOWUP_RETRY_PROVIDER_OVERFLOW_TEXT) from exc

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

            if not current_response.tool_calls:
                current_response = _enforce_acceptance_handoff(
                    current_response,
                    tool_safety=tool_safety,
                )
            assistant_record = self._build_assistant_record(
                current_session.session_id,
                current_response,
                turn_id=turn_id,
            )
            self._append_turn_record(
                session_id=current_session.session_id,
                pending_records=pending_records,
                record=assistant_record,
            )
            current_session = (
                self._persist_provider_session_state_from_response(
                    session_id=current_session.session_id,
                    response=current_response,
                    assistant_record=assistant_record,
                )
                or current_session
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
    ) -> tuple[
        SessionMetadata,
        list[ConversationRecord],
        list[ConversationRecord],
        LLMRequest,
        int,
    ]:
        compacted = await self._compact_session(
            session,
            reason=reason,
            excluded_turn_ids=(turn_id,),
            turn_id=turn_id,
        )
        if compacted is None:
            raise ContextBudgetError(_FOLLOWUP_COMPACTION_FAILED_TEXT)

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
        rebound_pending_records = self._coalesce_carry_forward_records(
            session_id=compacted.session_id,
            records=pending_records,
        )
        base_records = self._storage.load_records(compacted.session_id)
        activated_discoverable_tool_names = _collect_activated_discoverable_tool_names(
            rebound_pending_records
        )
        request = self._build_contextual_request(
            session_id=compacted.session_id,
            base_records=base_records,
            current_records=rebound_pending_records,
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
            request = self._build_contextual_request(
                session_id=compacted.session_id,
                base_records=base_records,
                current_records=rebound_pending_records,
                activated_discoverable_tool_names=activated_discoverable_tool_names,
            )
            estimated_input_tokens = estimate_request_input_tokens(request)
            if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
                raise ContextBudgetError(_FOLLOWUP_RETRY_PREFLIGHT_FAILED_TEXT)

        for record in rebound_pending_records:
            if _record_is_ephemeral_image_input(record):
                continue
            self._storage.append_record(compacted.session_id, record)
        self._cleanup_grok_provider_media(session.session_id)
        self._active_turn_id = turn_id
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

    async def _start_user_new_session(self) -> SessionMetadata:
        previous_session_id = self.active_session_id()
        session = await self._start_session(start_reason="user_new")
        if previous_session_id is not None and previous_session_id != session.session_id:
            self._storage.archive_session(previous_session_id)
            self._cleanup_grok_provider_media(previous_session_id)
        return session

    async def _start_session(
        self,
        *,
        start_reason: str,
        parent_session_id: str | None = None,
        compaction_bundle: CompactionBundle | None = None,
        replacement_items: Sequence[CompactionReplayItem] = (),
        compaction_count: int = 0,
    ) -> SessionMetadata:
        session = self._storage.create_session(
            parent_session_id=parent_session_id,
            start_reason=start_reason,
        )

        bootstrap_messages = self._identity_loader.load_bootstrap_messages()
        for message in bootstrap_messages:
            first_part = message.parts[0]
            if not isinstance(first_part, TextPart):
                raise RuntimeError("Identity bootstrap messages must contain text parts.")
            self._append_message(
                session_id=session.session_id,
                role=message.role,
                content=first_part.text,
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

        self._append_skills_bootstrap(session.session_id)

        if self._memory_mode.bootstrap:
            try:
                (
                    core_memory_bootstrap,
                    ongoing_memory_bootstrap,
                ) = await self._memory_service.render_bootstrap_messages()
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
        if compaction_bundle is not None:
            self._storage.append_record(
                session.session_id,
                build_compaction_bundle_record(
                    session_id=session.session_id,
                    bundle=compaction_bundle,
                ),
            )
        if replacement_items:
            self._persist_compaction_replacement_items(
                session_id=session.session_id,
                items=replacement_items,
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
        excluded_turn_ids: tuple[str, ...] = (),
        turn_id: str | None = None,
    ) -> SessionMetadata | None:
        excluded_turn_id_set = {turn_id.strip() for turn_id in excluded_turn_ids if turn_id.strip()}
        records = self._storage.load_records(
            session.session_id,
            include_all_turns=True,
        )
        compactable_records = [
            record
            for record in records
            if record.kind == "message"
            and str(record.metadata.get(_TURN_ID_METADATA_KEY, "")).strip()
            not in excluded_turn_id_set
        ]
        previous_bundle = load_compaction_bundle(records)
        source_records = prune_compaction_source_records(compactable_records)
        if not source_records and previous_bundle is None:
            if excluded_turn_id_set:
                next_compaction_count = session.compaction_count + 1
                self._storage.archive_session(session.session_id)
                next_session = await self._start_session(
                    start_reason="compaction",
                    parent_session_id=session.session_id,
                    replacement_items=(),
                    compaction_count=next_compaction_count,
                )
                self._storage.update_session(
                    next_session.session_id,
                    pending_reactive_compaction=False,
                    pending_interruption_notice=session.pending_interruption_notice,
                    pending_interruption_notice_reason=session.pending_interruption_notice_reason,
                    backend_state=_carry_compaction_backend_state(session.backend_state),
                )
                await self._emit_local_notice(
                    notice_kind="compaction_completed",
                    text="Context compacted into a new session.",
                )
                return self._storage.get_session(next_session.session_id) or next_session
            self._storage.update_session(session.session_id, pending_reactive_compaction=False)
            return None

        compaction_operation_id: str | None = None
        if turn_id is None:
            compaction_operation_id = self._begin_compaction_control(session.session_id)

        LOGGER.info(
            "Compaction started for session %s (reason=%s, source_records=%d, "
            "previous_bundle=%s, turn_id=%s).",
            session.session_id,
            reason,
            len(source_records),
            previous_bundle.bundle_id if previous_bundle is not None else None,
            turn_id,
        )
        try:
            await self._emit_local_notice(
                notice_kind="compaction_started",
                text="Compacting...",
            )
            if self._memory_mode.maintenance:
                try:
                    await self._await_compaction_operation(
                        self._memory_service.flush_before_compaction(
                            route_id=self._tool_context.route_id,
                            session_id=session.session_id,
                            records=tuple(records),
                        ),
                        session_id=session.session_id,
                        turn_id=turn_id,
                        compaction_operation_id=compaction_operation_id,
                        operation="memory_pre_compaction_flush",
                    )
                except (_TurnStopRequested, _CompactionStopRequested):
                    raise
                except Exception:
                    LOGGER.exception("Memory pre-compaction flush failed.")

            outcome = await self._await_compaction_operation(
                self._compactor.compact(
                    source_records,
                    previous_bundle=previous_bundle,
                    user_instruction=user_instruction,
                ),
                session_id=session.session_id,
                turn_id=turn_id,
                compaction_operation_id=compaction_operation_id,
                operation="llm_compaction",
            )
            _assert_compaction_effective(
                source_records=source_records,
                outcome=outcome,
                previous_bundle=previous_bundle,
            )
        except (_TurnStopRequested, _CompactionStopRequested):
            LOGGER.info(
                "Compaction interrupted for session %s (reason=%s, turn_id=%s).",
                session.session_id,
                reason,
                turn_id,
            )
            raise
        except Exception as exc:
            metadata = getattr(exc, "metadata", None)
            if isinstance(metadata, dict):
                metadata.setdefault("compaction_session_id", session.session_id)
                metadata.setdefault("compaction_reason", reason)
                metadata.setdefault("compaction_turn_id", turn_id)
            LOGGER.exception(
                "Compaction failed for session %s (reason=%s, turn_id=%s).",
                session.session_id,
                reason,
                turn_id,
            )
            raise
        finally:
            if compaction_operation_id is not None:
                self._clear_compaction_control(compaction_operation_id)

        # Activating the verified bundle is a short, consistency-sensitive commit. Stop is
        # checked before it begins; once archival starts, the commit runs to completion.
        if turn_id is not None and self._stop_requested(turn_id):
            raise _TurnStopRequested
        self._append_compaction_record(
            session.session_id,
            outcome=outcome,
            reason=reason,
            user_instruction=user_instruction,
        )
        self._storage.archive_session(session.session_id)

        next_compaction_count = session.compaction_count + 1
        next_session = await self._start_session(
            start_reason="compaction",
            parent_session_id=session.session_id,
            compaction_bundle=outcome.bundle,
            replacement_items=outcome.items,
            compaction_count=next_compaction_count,
        )
        self._storage.update_session(
            next_session.session_id,
            pending_reactive_compaction=False,
            pending_interruption_notice=session.pending_interruption_notice,
            pending_interruption_notice_reason=session.pending_interruption_notice_reason,
            backend_state=_carry_compaction_backend_state(session.backend_state),
        )
        if not excluded_turn_id_set:
            self._cleanup_grok_provider_media(session.session_id)
        await self._emit_local_notice(
            notice_kind="compaction_completed",
            text="Context compacted into a new session.",
        )
        LOGGER.info(
            "Compaction completed for session %s into session %s "
            "(reason=%s, calls=%d, repairs=%d).",
            session.session_id,
            next_session.session_id,
            reason,
            len(outcome.call_traces),
            outcome.repair_count,
        )
        return self._storage.get_session(next_session.session_id) or next_session

    def _cleanup_grok_provider_media(self, session_id: str) -> None:
        media_dir = (
            self._settings.transcript_archive_dir / _GROK_PROVIDER_MEDIA_DIR_NAME / session_id
        )
        try:
            shutil.rmtree(media_dir)
        except FileNotFoundError:
            return
        except OSError:
            LOGGER.warning(
                "Failed to clean Grok provider media for session %s.",
                session_id,
                exc_info=True,
            )

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
        user_instruction: str | None,
    ) -> None:
        metadata = {
            "reason": reason,
            "user_instruction": user_instruction.strip() if user_instruction else None,
            "provider": outcome.provider,
            "model": outcome.model,
            "response_id": outcome.response_id,
            "replacement_items": [item.to_dict() for item in outcome.items],
            "bundle": outcome.bundle.to_dict(),
            "draft_payload": outcome.draft_payload,
            "verification_payload": outcome.verification_payload,
            "repair_count": outcome.repair_count,
            "call_traces": [trace.to_dict() for trace in outcome.call_traces],
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
            content=(
                f"Compaction activated verified bundle {outcome.bundle.bundle_id} with "
                f"{len(outcome.items)} deterministic replay items."
            ),
            kind="compaction",
            metadata=metadata,
        )
        self._storage.append_record(session_id, record)

    def _persist_compaction_replacement_items(
        self,
        *,
        session_id: str,
        items: Sequence[CompactionReplayItem],
    ) -> None:
        for item in items:
            self._append_message(
                session_id=session_id,
                role=item.role,
                content=item.content,
                metadata=item.record_metadata(),
            )

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
            "provider_metadata": deepcopy(response.provider_metadata),
            "tool_calls": [_serialize_transcript_tool_call(call) for call in response.tool_calls],
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
        reason: Literal["interrupted", "runtime_error"] = "interrupted",
    ) -> ConversationRecord | None:
        if not text:
            return None
        metadata = {"incomplete_stream_fragment": True}
        metadata[f"{reason}_stream_fragment"] = True
        return self._build_message_record(
            session_id=session_id,
            role="assistant",
            content=text,
            metadata=metadata,
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
        content, content_metadata = _archive_large_tool_result(
            result.content,
            workspace_dir=self._tool_context.workspace_dir,
        )
        metadata = _canonical_tool_result_metadata(result.metadata)
        metadata.update(content_metadata)
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
            content=content,
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
        elif text == _PREVIOUS_SESSION_RESET_TEXT:
            interruption_reason = "new_session"
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
                "new_session_boundary": interruption_reason == "new_session",
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

    def _append_skills_bootstrap(self, session_id: str) -> None:
        if not self._skills_settings.bootstrap_headers:
            self._append_message(
                session_id=session_id,
                role="system",
                content=render_skill_search_guidance(),
                metadata={_SKILLS_BOOTSTRAP_METADATA_KEY: "search_guidance"},
            )
            return
        try:
            import_staged_skills(self._skills_settings)
        except Exception:
            LOGGER.exception("Skill import scan before session bootstrap failed.")
        catalog = load_skill_catalog(self._skills_settings)
        content = render_skill_bootstrap_headers(catalog)
        if content is None:
            return
        self._append_message(
            session_id=session_id,
            role="system",
            content=content,
            metadata={_SKILLS_BOOTSTRAP_METADATA_KEY: "headers"},
        )

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
                recovery_metadata: dict[str, Any] | None = None
                if self._effective_llm_provider() == "grok":
                    try:
                        source_bytes = Path(path).read_bytes()
                    except OSError:
                        source_bytes = b""
                    source_sha256 = (
                        hashlib.sha256(source_bytes).hexdigest() if source_bytes else None
                    )
                    if source_sha256 is not None and self._grok_image_hash_is_live(
                        session_id=session_id,
                        source_sha256=source_sha256,
                    ):
                        return [
                            self._build_message_record(
                                session_id=session_id,
                                role="user",
                                content=(
                                    "The requested workspace image is unchanged and is "
                                    "already present in the live Grok context. Reuse the "
                                    f"existing image for path: {path}"
                                ),
                                metadata={
                                    _EPHEMERAL_IMAGE_INPUT_METADATA_KEY: True,
                                    "source_tool": result.name,
                                    "deduplicated_image_sha256": source_sha256,
                                },
                                turn_id=turn_id,
                            )
                        ]
                    prepared = self._prepare_grok_image_attachment(
                        session_id=session_id,
                        path=path,
                        media_type=media_type,
                        detail=detail,
                        source_bytes=source_bytes or None,
                        source_sha256=source_sha256,
                    )
                    if prepared is not None:
                        path = str(prepared["path"])
                        media_type = str(prepared["media_type"])
                        recovery_metadata = prepared
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
                if recovery_metadata is not None:
                    records.append(
                        ConversationRecord(
                            record_id=uuid4().hex,
                            session_id=session_id,
                            created_at=_utc_now_iso(),
                            role="user",
                            content=content,
                            kind="provider_context",
                            metadata={
                                _TRANSCRIPT_ONLY_RECORD_METADATA_KEY: True,
                                _GROK_RECOVERY_IMAGE_METADATA_KEY: True,
                                _IMAGE_INPUT_METADATA_KEY: {
                                    "path": path,
                                    "media_type": media_type,
                                    "detail": detail,
                                },
                                "source_tool": result.name,
                                "source_path": recovery_metadata["source_path"],
                                "source_sha256": recovery_metadata["source_sha256"],
                                "content_sha256": recovery_metadata["content_sha256"],
                                "source_bytes": recovery_metadata["source_bytes"],
                                "content_bytes": recovery_metadata["content_bytes"],
                                "transcoded": recovery_metadata["transcoded"],
                                _TURN_ID_METADATA_KEY: turn_id,
                            },
                        )
                    )

        return records

    def _prepare_grok_image_attachment(
        self,
        *,
        session_id: str,
        path: str,
        media_type: str,
        detail: str,
        source_bytes: bytes | None = None,
        source_sha256: str | None = None,
    ) -> dict[str, Any] | None:
        source_path = Path(path)
        if source_bytes is None:
            try:
                source_bytes = source_path.read_bytes()
            except OSError:
                return None
        if not source_bytes:
            return None

        source_sha256 = source_sha256 or hashlib.sha256(source_bytes).hexdigest()
        content_bytes = source_bytes
        content_media_type = media_type
        transcoded = False
        if len(
            source_bytes
        ) >= _GROK_IMAGE_TRANSCODE_MIN_BYTES and _should_transcode_grok_tool_image(
            source_path, media_type=media_type
        ):
            optimized = _transcode_grok_tool_image(source_bytes)
            if optimized is not None and len(optimized) < len(source_bytes):
                content_bytes = optimized
                content_media_type = "image/jpeg"
                transcoded = True

        content_sha256 = hashlib.sha256(content_bytes).hexdigest()
        suffix = _image_media_type_suffix(content_media_type)
        snapshot_dir = (
            self._settings.transcript_archive_dir / _GROK_PROVIDER_MEDIA_DIR_NAME / session_id
        )
        snapshot_path = snapshot_dir / f"{content_sha256}{suffix}"
        try:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            if not snapshot_path.exists():
                temporary_path = snapshot_dir / f".{snapshot_path.name}.{uuid4().hex}.tmp"
                temporary_path.write_bytes(content_bytes)
                temporary_path.replace(snapshot_path)
        except OSError:
            snapshot_path = source_path
            content_bytes = source_bytes
            content_media_type = media_type
            content_sha256 = source_sha256
            transcoded = False

        return {
            "path": str(snapshot_path),
            "media_type": content_media_type,
            "detail": detail,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "content_sha256": content_sha256,
            "source_bytes": len(source_bytes),
            "content_bytes": len(content_bytes),
            "transcoded": transcoded,
        }

    def _grok_image_hash_is_live(
        self,
        *,
        session_id: str,
        source_sha256: str,
    ) -> bool:
        session = self._storage.get_session(session_id)
        if session is None:
            return False
        state = ProviderSessionState.from_mapping(session.provider_session_state)
        if state is None or state.provider != "grok" or state.grok.storage_mode != "ephemeral":
            return False
        live_tail_records = _records_between_response_records(
            self._storage.load_records(session_id, include_all_turns=True),
            after_record_id=state.grok.durable_response_record_id,
            through_record_id=state.grok.last_response_record_id,
        )
        for record in live_tail_records:
            if record.kind != "provider_context":
                continue
            if record.metadata.get("source_sha256") == source_sha256:
                return True
        return False

    def _build_unexecuted_tool_call_note_record(
        self,
        *,
        session_id: str,
        tool_names: Sequence[str],
        turn_id: str,
        boundary: Literal["turn_interruption", "tool_slice"] = "turn_interruption",
    ) -> ConversationRecord:
        ordered_tool_names = list(_ordered_unique_names(tool_names))
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content=_unexecuted_tool_call_note_text(
                ordered_tool_names,
                boundary=boundary,
            ),
            metadata={
                _UNEXECUTED_TOOL_CALL_NOTICE_METADATA_KEY: True,
                "tool_names": ordered_tool_names,
                "boundary": boundary,
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

        self._finalize_incomplete_turn_records(
            session_id=session_id,
            turn_id=turn_id,
            turn_records=turn_records,
        )

    def _finalize_incomplete_turn_records(
        self,
        *,
        session_id: str,
        turn_id: str,
        turn_records: Sequence[ConversationRecord] | None = None,
    ) -> None:
        records = (
            list(turn_records)
            if turn_records is not None
            else [
                record
                for record in self._storage.load_records(session_id, include_all_turns=True)
                if str(record.metadata.get(_TURN_ID_METADATA_KEY, "")).strip() == turn_id
            ]
        )
        if not records:
            self._storage.set_turn_status(
                session_id,
                turn_id=turn_id,
                status="interrupted",
            )
            return

        if not any(_record_is_unexecuted_tool_call_notice(record) for record in records):
            unresolved_tool_names = _collect_unexecuted_tool_call_names(records)
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
            record.metadata.get(_ORPHANED_TURN_RECOVERY_METADATA_KEY, False) for record in records
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

    def _fail_turn_after_runtime_error(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> None:
        session = self._storage.get_session(session_id)
        if session is None or session.turn_states.get(turn_id) != "in_progress":
            return
        self._finalize_incomplete_turn_records(
            session_id=session_id,
            turn_id=turn_id,
        )

    def _begin_turn(self, *, session_id: str, turn_id: str) -> None:
        self._storage.set_turn_status(
            session_id,
            turn_id=turn_id,
            status="in_progress",
        )
        self._active_turn_id = turn_id
        self._requested_interruption = None
        self._turn_stop_event = asyncio.Event()

    def _finish_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        status: Literal["completed", "blocked", "interrupted", "superseded"],
    ) -> None:
        self._storage.set_turn_status(
            session_id,
            turn_id=turn_id,
            status=status,
        )
        if self._active_turn_id == turn_id:
            self._active_turn_id = None
            self._turn_stop_event = None
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

    def _begin_compaction_control(self, session_id: str) -> str:
        if self._active_compaction_control is not None:
            raise RuntimeError("A compaction operation is already active.")
        operation_id = f"compaction_{uuid4().hex}"
        self._active_compaction_control = _ActiveCompactionControl(
            operation_id=operation_id,
            session_id=session_id,
            stop_event=asyncio.Event(),
        )
        return operation_id

    def _clear_compaction_control(self, operation_id: str) -> None:
        control = self._active_compaction_control
        if control is not None and control.operation_id == operation_id:
            self._active_compaction_control = None

    async def _await_compaction_operation(
        self,
        awaitable: Awaitable[T],
        *,
        session_id: str,
        turn_id: str | None,
        compaction_operation_id: str | None,
        operation: str,
    ) -> T:
        if turn_id is not None:
            return await self._await_with_stop(
                awaitable,
                turn_id=turn_id,
                operation=operation,
            )
        if compaction_operation_id is None:
            _close_unstarted_awaitable(awaitable)
            raise RuntimeError("Pre-turn compaction is missing its operation control.")
        return await self._await_with_compaction_stop(
            awaitable,
            session_id=session_id,
            operation_id=compaction_operation_id,
            operation=operation,
        )

    async def _await_with_compaction_stop(
        self,
        awaitable: Awaitable[T],
        *,
        session_id: str,
        operation_id: str,
        operation: str,
    ) -> T:
        control = self._active_compaction_control
        if control is None or control.operation_id != operation_id:
            _close_unstarted_awaitable(awaitable)
            raise RuntimeError("Compaction operation control is no longer active.")
        if control.interruption_reason is not None:
            _close_unstarted_awaitable(awaitable)
            raise _CompactionStopRequested(
                session_id=session_id,
                reason=control.interruption_reason,
            )

        task = asyncio.ensure_future(awaitable)
        stop_task = asyncio.create_task(
            control.stop_event.wait(),
            name=f"jarvis-compaction-stop-wait-{operation}-{operation_id}",
        )
        try:
            done, _pending = await asyncio.wait(
                {task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done and control.interruption_reason is not None:
                task.cancel()
                await self._drain_preempted_task(task, operation=operation)
                raise _CompactionStopRequested(
                    session_id=session_id,
                    reason=control.interruption_reason,
                )

            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task
            return task.result()
        finally:
            if not stop_task.done():
                stop_task.cancel()

    async def _await_with_stop(
        self,
        awaitable: Awaitable[T],
        *,
        turn_id: str,
        operation: str,
    ) -> T:
        if self._stop_requested(turn_id):
            _close_unstarted_awaitable(awaitable)
            raise _TurnStopRequested

        stop_event = self._turn_stop_event
        if stop_event is None:
            return await awaitable

        task = asyncio.ensure_future(awaitable)
        stop_task = asyncio.create_task(
            stop_event.wait(),
            name=f"jarvis-turn-stop-wait-{operation}-{turn_id}",
        )
        try:
            done, _pending = await asyncio.wait(
                {task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done and self._stop_requested(turn_id):
                task.cancel()
                await self._drain_preempted_task(task, operation=operation)
                raise _TurnStopRequested

            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task
            return task.result()
        finally:
            if not stop_task.done():
                stop_task.cancel()

    async def _stream_generate_with_stop(
        self,
        request: LLMRequest,
        *,
        turn_id: str,
    ) -> AsyncIterator[Any]:
        if self._stop_requested(turn_id):
            raise _TurnStopRequested

        stop_event = self._turn_stop_event
        stream = self._llm_service.stream_generate(request)
        iterator = stream.__aiter__()
        if stop_event is None:
            async for event in iterator:
                yield event
            return

        while True:
            next_task = asyncio.ensure_future(iterator.__anext__())
            stop_task = asyncio.create_task(
                stop_event.wait(),
                name=f"jarvis-turn-stop-wait-llm-stream-{turn_id}",
            )
            try:
                done, _pending = await asyncio.wait(
                    {next_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done and self._stop_requested(turn_id):
                    next_task.cancel()
                    await self._drain_preempted_task(
                        next_task,
                        operation="llm_stream",
                    )
                    if next_task.done():
                        await self._close_preempted_stream(iterator)
                    raise _TurnStopRequested

                stop_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stop_task
                try:
                    yield next_task.result()
                except StopAsyncIteration:
                    return
            finally:
                if not stop_task.done():
                    stop_task.cancel()

    async def _drain_preempted_task(
        self,
        task: asyncio.Future[Any],
        *,
        operation: str,
    ) -> None:
        try:
            await asyncio.wait_for(task, timeout=_STOP_PREEMPTION_CLEANUP_SECONDS)
        except asyncio.CancelledError:
            return
        except asyncio.TimeoutError:
            LOGGER.warning(
                "Timed out waiting for preempted %s task to acknowledge cancellation.",
                operation,
            )
        except Exception:
            LOGGER.debug(
                "Preempted %s task ended with an exception during stop cleanup.",
                operation,
                exc_info=True,
            )

    async def _close_preempted_stream(self, iterator: AsyncIterator[Any]) -> None:
        aclose = getattr(iterator, "aclose", None)
        if not callable(aclose):
            return
        close_awaitable = aclose()
        if not isinstance(close_awaitable, RuntimeAwaitable):
            return
        close_task = asyncio.ensure_future(close_awaitable)
        await self._drain_preempted_task(close_task, operation="llm_stream_close")

    def _clear_turn_control(self, turn_id: str) -> None:
        if self._active_turn_id == turn_id:
            self._active_turn_id = None
            self._turn_stop_event = None
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
            command=(str(approval["command"]) if approval.get("command") is not None else None),
            tool_name=(
                str(approval["tool_name"]) if approval.get("tool_name") is not None else None
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
                    return bool(await asyncio.wait_for(asyncio.shield(future), timeout=0.2))
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
        elif interruption_reason == "new_session":
            interrupted_status = "interrupted"
            interrupted_record_text = _TURN_NEW_SESSION_RECORD_TEXT
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
                "superseded_by_user_message": (interruption_reason == "superseded_by_user_message"),
                "new_session_boundary": interruption_reason == "new_session",
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
        cloned = ConversationRecord(
            record_id=record.record_id,
            session_id=session_id,
            created_at=record.created_at,
            role=record.role,
            content=record.content,
            kind=record.kind,
            metadata={
                **deepcopy(record.metadata),
                "source_record_id": record.record_id,
                "source_session_id": record.session_id,
                "source_created_at": record.created_at,
            },
        )
        attachment = cloned.metadata.get(_IMAGE_INPUT_METADATA_KEY)
        if not isinstance(attachment, dict):
            return cloned
        raw_path = str(attachment.get("path", "")).strip()
        if not raw_path:
            return cloned
        source_path = Path(raw_path)
        source_media_root = (
            self._settings.transcript_archive_dir
            / _GROK_PROVIDER_MEDIA_DIR_NAME
            / record.session_id
        )
        try:
            source_path.relative_to(source_media_root)
        except ValueError:
            return cloned
        if not source_path.is_file():
            return cloned

        destination_dir = (
            self._settings.transcript_archive_dir / _GROK_PROVIDER_MEDIA_DIR_NAME / session_id
        )
        destination_path = destination_dir / source_path.name
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
            if not destination_path.exists():
                shutil.copyfile(source_path, destination_path)
        except OSError:
            return cloned
        attachment["path"] = str(destination_path)
        return cloned

    def _clone_carry_forward_record_for_session(
        self,
        session_id: str,
        record: ConversationRecord,
    ) -> ConversationRecord:
        cloned = self._clone_record_for_session(session_id, record)
        return self._compact_carry_forward_record(cloned)

    def _coalesce_carry_forward_records(
        self,
        *,
        session_id: str,
        records: Sequence[ConversationRecord],
    ) -> list[ConversationRecord]:
        """Carry one causal copy of identical repeated tool cycles across compaction."""

        output: list[ConversationRecord] = []
        index = 0
        while index < len(records):
            cycle = _carry_forward_tool_cycle(records, index)
            if cycle is None:
                output.append(self._clone_carry_forward_record_for_session(session_id, records[index]))
                index += 1
                continue
            end_index, signature = cycle
            repeated_ends = [end_index]
            cursor = end_index
            while cursor < len(records):
                next_cycle = _carry_forward_tool_cycle(records, cursor)
                if next_cycle is None or next_cycle[1] != signature:
                    break
                repeated_ends.append(next_cycle[0])
                cursor = next_cycle[0]
            for record in records[index:end_index]:
                output.append(self._clone_carry_forward_record_for_session(session_id, record))
            if len(repeated_ends) > 1:
                repeated_records = records[index:cursor]
                output.append(
                    self._build_carry_forward_repetition_record(
                        session_id=session_id,
                        records=repeated_records,
                        repeat_count=len(repeated_ends),
                        signature=signature,
                    )
                )
            index = cursor
        return output

    def _build_carry_forward_repetition_record(
        self,
        *,
        session_id: str,
        records: Sequence[ConversationRecord],
        repeat_count: int,
        signature: str,
    ) -> ConversationRecord:
        first = records[0]
        last = records[-1]
        source_ids = [record.record_id for record in records]
        return self._build_message_record(
            session_id=session_id,
            role="system",
            content=(
                "Repeated tool cycle compacted during active-turn carry-forward.\n"
                f"repeat_count: {repeat_count}\n"
                "outcome: identical call arguments and result repeated without new evidence.\n"
                "Do not repeat this cycle; inspect current state before choosing another action."
            ),
            metadata={
                "carry_forward_repetition_summary": True,
                "repeat_count": repeat_count,
                "cycle_signature": signature,
                "source_record_ids": source_ids,
                "source_session_ids": sorted({record.session_id for record in records}),
                "source_created_at_first": first.created_at,
                "source_created_at_last": last.created_at,
            },
        )

    def _compact_carry_forward_record(
        self,
        record: ConversationRecord,
    ) -> ConversationRecord:
        compacted = record
        limit = _carry_forward_soft_limit(record)
        if limit is not None and len(record.content) > limit:
            compacted = _copy_record_with_content(
                compacted,
                content=_truncate_carry_forward_text(record.content, limit=limit),
                metadata_updates={
                    "carry_forward_compacted": True,
                    "carry_forward_compaction_strength": "soft",
                },
            )
        return _compact_carry_forward_tool_call_metadata(
            compacted,
            strength="soft",
        )

    def _strongly_compact_carry_forward_record(
        self,
        record: ConversationRecord,
    ) -> ConversationRecord:
        if _record_is_ephemeral_image_input(record):
            return record
        if record.role not in {"user", "assistant", "tool"}:
            return record
        compacted = record
        if record.content:
            compacted = _copy_record_with_content(
                compacted,
                content=_strong_carry_forward_text(record),
                metadata_updates={
                    "carry_forward_compacted": True,
                    "carry_forward_compaction_strength": "strong",
                },
            )
        return _compact_carry_forward_tool_call_metadata(
            compacted,
            strength="strong",
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


def _carry_forward_tool_cycle(
    records: Sequence[ConversationRecord],
    start: int,
) -> tuple[int, str] | None:
    """Return the exclusive end/signature for one completed assistant-tool cycle."""

    if start >= len(records):
        return None
    assistant = records[start]
    tool_calls = assistant.metadata.get("tool_calls")
    if assistant.role != "assistant" or not isinstance(tool_calls, list) or not tool_calls:
        return None
    end = start + 1 + len(tool_calls)
    if end > len(records):
        return None
    tool_records = records[start + 1:end]
    call_ids = [str(call.get("call_id", "")).strip() for call in tool_calls if isinstance(call, dict)]
    if len(call_ids) != len(tool_calls) or any(not call_id for call_id in call_ids):
        return None
    if any(record.role != "tool" for record in tool_records):
        return None
    if [str(record.metadata.get("call_id", "")).strip() for record in tool_records] != call_ids:
        return None
    payload = {
        "tool_calls": [
            {
                "name": call.get("name"),
                "arguments": call.get("arguments"),
            }
            for call in tool_calls
            if isinstance(call, dict)
        ],
        "tool_results": [
            {
                "name": record.metadata.get("tool_name"),
                "ok": record.metadata.get("ok"),
                "content": record.content,
                "metadata": _stable_carry_forward_metadata(record.metadata),
            }
            for record in tool_records
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return end, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stable_carry_forward_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    volatile = {
        "call_id",
        "turn_id",
        "job_id",
        "launched_at",
        "started_at",
        "finished_at",
        "last_update_at",
        "stdout_path",
        "stderr_path",
        "error_log_path",
        "duration_seconds",
        "observed_at",
    }
    return _stable_carry_forward_mapping(metadata, volatile=volatile)


def _stable_carry_forward_mapping(
    value: Mapping[str, Any],
    *,
    volatile: set[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key in volatile:
            continue
        if isinstance(raw_value, Mapping):
            output[key] = _stable_carry_forward_mapping(raw_value, volatile=volatile)
        elif isinstance(raw_value, list):
            output[key] = [
                _stable_carry_forward_mapping(item, volatile=volatile)
                if isinstance(item, Mapping)
                else item
                for item in raw_value
            ]
        else:
            output[key] = raw_value
    return output


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _carry_compaction_backend_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Carry only task-recovery state that must survive a session migration."""

    return {
        key: deepcopy(state[key])
        for key in ("provider_recovery", "active_tool_task_id")
        if key in state
    }


def _assert_compaction_effective(
    *,
    source_records: Sequence[ConversationRecord],
    outcome: CompactionOutcome,
    previous_bundle: CompactionBundle | None,
) -> None:
    source_payload = [
        {
            "role": record.role,
            "content": record.content,
            "metadata": record.metadata,
        }
        for record in source_records
    ]
    source_bytes = len(
        json.dumps(source_payload, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    )
    replay_bytes = len(
        json.dumps(
            [item.to_dict() for item in outcome.items],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    if source_bytes >= 20_000 and replay_bytes >= int(source_bytes * 0.9):
        raise ContextBudgetError(
            "Compaction was parked because the verified replay did not materially reduce "
            "the source history. Repeated cycles must be resolved before retrying."
        )
    if previous_bundle is None or not source_records:
        return
    prior = previous_bundle.to_dict()
    current = outcome.bundle.to_dict()
    for payload in (prior, current):
        payload.pop("bundle_id", None)
        payload.pop("created_at", None)
        payload.pop("source_manifest", None)
    if prior == current:
        raise ContextBudgetError(
            "Compaction was parked because it reproduced the prior semantic bundle despite "
            "new source records. Inspect the unresolved repetition before retrying."
        )


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
    return any(
        bool(message.metadata.get("force_no_tools_this_turn")) for message in runtime_messages
    )


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
    if status == "running" and (
        str(tool_result.metadata.get("mode", "")).strip() == "background"
        or bool(tool_result.metadata.get("promoted_to_background"))
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
    if action in {"invoke", "step_in"} and status in {
        "running",
        "waiting_background",
        "awaiting_approval",
    }:
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
    *,
    defer_local_images: bool = False,
    allow_terminal_pending_tool_calls: bool = False,
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
            defer_local_images=defer_local_images,
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
                pending_tool_names = tuple(
                    _ordered_unique_names(name for _call_id, name in call_specs)
                )
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
        if not allow_terminal_pending_tool_calls:
            _raise_unresolved_pending()
        _append_record(pending_assistant, include_tool_calls=True)
        for tool_record in pending_tool_records:
            _append_record(tool_record)

    return tuple(messages)


def _records_after_response_record(
    records: Sequence[ConversationRecord],
    record_id: str | None,
) -> tuple[ConversationRecord, ...]:
    if record_id is None:
        return tuple(records)
    normalized = record_id.strip()
    if not normalized:
        return tuple(records)
    for index, record in enumerate(records):
        if record.record_id == normalized:
            return tuple(records[index + 1 :])
    return tuple(records)


def _records_between_response_records(
    records: Sequence[ConversationRecord],
    *,
    after_record_id: str | None,
    through_record_id: str | None,
) -> tuple[ConversationRecord, ...]:
    if through_record_id is None or not through_record_id.strip():
        return ()

    start_index = 0
    if after_record_id is not None and after_record_id.strip():
        for index, record in enumerate(records):
            if record.record_id == after_record_id:
                start_index = index + 1
                break

    for index in range(start_index, len(records)):
        if records[index].record_id == through_record_id:
            return tuple(records[start_index : index + 1])
    return ()


def _records_have_image_inputs(records: Sequence[ConversationRecord]) -> bool:
    return any(
        isinstance(record.metadata.get(_IMAGE_INPUT_METADATA_KEY), dict) for record in records
    )


def _records_to_grok_recovery_messages(
    records: Sequence[ConversationRecord],
) -> tuple[LLMMessage, ...]:
    replay_records: list[ConversationRecord] = []
    for record in records:
        if record.kind != "provider_context":
            replay_records.append(record)
            continue
        if not bool(record.metadata.get(_GROK_RECOVERY_IMAGE_METADATA_KEY, False)):
            continue
        metadata = deepcopy(record.metadata)
        metadata.pop(_TRANSCRIPT_ONLY_RECORD_METADATA_KEY, None)
        metadata.pop(_GROK_RECOVERY_IMAGE_METADATA_KEY, None)
        replay_records.append(
            ConversationRecord(
                record_id=record.record_id,
                session_id=record.session_id,
                created_at=record.created_at,
                role=record.role,
                content=record.content,
                kind="message",
                metadata=metadata,
            )
        )
    return _records_to_llm_messages(
        replay_records,
        defer_local_images=True,
        allow_terminal_pending_tool_calls=True,
    )


def _gemini_cache_source_signature(
    *,
    records: Sequence[ConversationRecord],
    tools: Sequence[ToolDefinition],
    tool_choice: ToolChoice,
) -> str:
    payload = {
        "records": [
            {
                "record_id": record.record_id,
                "kind": record.kind,
                "role": record.role,
                "content": record.content,
                "metadata": record.metadata,
            }
            for record in records
            if record.kind == "message"
            and not bool(record.metadata.get(_TRANSCRIPT_ONLY_RECORD_METADATA_KEY, False))
        ],
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
                "strict": tool.strict,
            }
            for tool in tools
        ],
        "tool_choice": {
            "mode": tool_choice.mode.value,
            "tool_name": tool_choice.tool_name,
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _gemini_cache_is_usable(
    state: ProviderSessionState,
    *,
    source_signature: str | None,
) -> bool:
    gemini = state.gemini
    if gemini.cached_content_name is None:
        return False
    if source_signature is None or gemini.source_signature != source_signature:
        return False
    if gemini.cache_expires_at is None:
        return True
    try:
        expires_at = datetime.fromisoformat(gemini.cache_expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at > datetime.now(timezone.utc)


def _collect_message_media_ids(messages: Sequence[LLMMessage]) -> tuple[str, ...]:
    media_ids: list[str] = []
    for message in messages:
        for part in message.parts:
            if not isinstance(part, ImagePart):
                continue
            if part.file_id is not None:
                media_ids.append(part.file_id)
            elif part.image_url is not None and not part.image_url.startswith("data:"):
                media_ids.append(part.image_url)
    return tuple(media_ids)


def _metadata_str(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _enforce_acceptance_handoff(
    response: LLMResponse,
    *,
    tool_safety: ToolSafetyTracker,
) -> LLMResponse:
    if not tool_safety.unverified_workspace_mutation:
        return response
    return replace(
        response,
        text="",
        finish_reason="stop",
        provider_metadata={
            **response.provider_metadata,
            "acceptance_required": True,
            "completion_blocked": True,
        },
    )


def _response_completion_blocked(response: LLMResponse) -> bool:
    return bool(response.provider_metadata.get("completion_blocked", False))


def _with_tool_safety_replan_notice(
    result: ToolExecutionResult,
    *,
    observation: ToolSafetyObservation,
) -> ToolExecutionResult:
    if not (
        observation.repeated_invalid_call or observation.repeated_no_progress
    ) or bool(result.metadata.get("tool_safety_blocked")):
        return result
    metadata = dict(result.metadata)
    metadata.update(
        {
            "tool_safety_replan_required": True,
            "tool_safety_signature_id": observation.signature_id,
            "tool_safety_occurrence_count": observation.occurrence_count,
            "tool_safety_progress_epoch": observation.progress_epoch,
            "tool_safety_first_call_id": observation.first_call_id,
        }
    )
    return replace(
        result,
        content=(
            result.content.rstrip()
            + "\n\nTool safety checkpoint\n"
            + "This exact action is now blocked in the current unchanged workspace epoch. "
            + "Do not repeat it; replan with different arguments or make material progress."
        ),
        metadata=metadata,
    )


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if not isinstance(value, int):
        return None
    return value


def _metadata_string_tuple(metadata: dict[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, (list, tuple)):
        return ()
    items: list[str] = []
    for item in value:
        normalized = str(item).strip()
        if normalized:
            items.append(normalized)
    return tuple(items)


def _serialize_transcript_tool_call(call: ToolCall) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "call_id": call.call_id,
        "name": call.name,
        "arguments": dict(call.arguments),
        "provider_metadata": dict(call.provider_metadata),
    }
    if _raw_arguments_materially_differ(call.raw_arguments, call.arguments):
        payload["raw_arguments"] = call.raw_arguments
    return payload


def _canonical_tool_result_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep tool arguments canonical in the paired assistant tool-call record."""

    return {
        key: value
        for key, value in metadata.items()
        if key not in {"arguments", "raw_arguments", "command", "stdout", "stderr"}
    }


def _raw_arguments_materially_differ(
    raw_arguments: str,
    arguments: dict[str, Any],
) -> bool:
    raw = raw_arguments.strip()
    if not raw:
        return False
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return True
    return parsed != arguments


def _archive_large_tool_result(
    content: str,
    *,
    workspace_dir: Path,
) -> tuple[str, dict[str, Any]]:
    if len(content) <= _INLINE_TOOL_RESULT_MAX_CHARS:
        return content, {}
    encoded = content.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    archive_dir = workspace_dir / ".jarvis_internal" / "tool_outputs"
    archive_path = archive_dir / f"{digest}.txt"
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        if not archive_path.exists():
            temporary_path = archive_dir / f".{digest}.{uuid4().hex}.tmp"
            temporary_path.write_bytes(encoded)
            temporary_path.replace(archive_path)
    except OSError:
        LOGGER.warning("Could not archive large tool result %s.", digest, exc_info=True)
        return content, {}
    half = _INLINE_TOOL_RESULT_MAX_CHARS // 2
    preview = (
        content[:half]
        + "\n...[tool result archived; inline preview truncated]...\n"
        + content[-half:]
    )
    display_path = Path("/workspace") / archive_path.relative_to(workspace_dir)
    return preview, {
        "content_archived": True,
        "content_sha256": digest,
        "content_bytes": len(encoded),
        "content_archive_path": str(display_path),
        "inline_preview_chars": len(preview),
    }


def _record_to_llm_message(
    record: ConversationRecord,
    *,
    include_tool_calls: bool = True,
    defer_local_images: bool = False,
) -> LLMMessage | None:
    if bool(record.metadata.get(_TRANSCRIPT_ONLY_RECORD_METADATA_KEY, False)):
        return None

    if record.role in {"system", "user"}:
        parts: list[ImagePart | LocalImagePart | TextPart] = []
        image_part = _record_image_part(record, defer_local=defer_local_images)
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
                arguments = tool_call.get("arguments", {})
                raw_arguments = str(tool_call.get("raw_arguments", "")).strip()
                if not raw_arguments and isinstance(arguments, dict):
                    raw_arguments = json.dumps(
                        arguments,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
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
        arguments = tool_call.get("arguments", {})
        raw_arguments = str(tool_call.get("raw_arguments", "")).strip()
        if not raw_arguments and isinstance(arguments, dict):
            raw_arguments = json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        if not call_id or not name or not raw_arguments or not isinstance(arguments, dict):
            continue
        specs.append((call_id, name))
    return tuple(specs)


def _ordered_unique_names(
    names: Sequence[str] | list[str] | tuple[str, ...] | Any,
) -> tuple[str, ...]:
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


def _record_image_part(
    record: ConversationRecord,
    *,
    defer_local: bool = False,
) -> ImagePart | LocalImagePart | None:
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

    if defer_local:
        return LocalImagePart(
            path=raw_path,
            media_type=media_type,
            detail=raw_detail,  # type: ignore[arg-type]
        )

    try:
        data = Path(raw_path).read_bytes()
    except OSError:
        return None

    return ImagePart.from_base64(
        media_type=media_type,
        data_base64=base64.b64encode(data).decode("ascii"),
        detail=raw_detail,  # type: ignore[arg-type]
    )


def _transcode_grok_tool_image(source_bytes: bytes) -> bytes | None:
    try:
        with Image.open(BytesIO(source_bytes)) as source:
            image = ImageOps.exif_transpose(source)
            if max(image.size) > _GROK_IMAGE_MAX_EDGE_PIXELS:
                image.thumbnail(
                    (_GROK_IMAGE_MAX_EDGE_PIXELS, _GROK_IMAGE_MAX_EDGE_PIXELS),
                    Image.Resampling.LANCZOS,
                )
            if image.mode in {"RGBA", "LA"}:
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            output = BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=_GROK_IMAGE_JPEG_QUALITY,
                optimize=True,
            )
            return output.getvalue()
    except (OSError, ValueError):
        return None


def _should_transcode_grok_tool_image(path: Path, *, media_type: str) -> bool:
    if media_type != "image/png":
        return False
    screenshot_markers = {
        "captures",
        "renders",
        "screenshots",
        "shots",
    }
    normalized_parts = {part.lower() for part in path.parts}
    normalized_name = path.stem.lower()
    return bool(normalized_parts & screenshot_markers) or "screenshot" in normalized_name


def _image_media_type_suffix(media_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(media_type, ".bin")


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


def _compact_carry_forward_tool_call_metadata(
    record: ConversationRecord,
    *,
    strength: Literal["soft", "strong"],
) -> ConversationRecord:
    if record.role != "assistant":
        return record

    raw_tool_calls = record.metadata.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return record

    compacted_tool_calls: list[Any] = []
    did_change = False
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            compacted_tool_calls.append(deepcopy(raw_tool_call))
            continue

        tool_call = deepcopy(raw_tool_call)
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        original_raw_arguments = str(tool_call.get("raw_arguments", "")).strip()
        canonical_raw_arguments = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        needs_placeholder = (
            strength == "strong"
            or len(canonical_raw_arguments) > 1_200
            or len(original_raw_arguments) > 1_200
        )
        if needs_placeholder:
            compacted_arguments = {
                "compacted": True,
                "reason": "carry_forward_compaction",
                "compaction_strength": strength,
                "original_argument_chars": len(original_raw_arguments or canonical_raw_arguments),
                "note": (
                    "Tool arguments compacted after mid-turn overflow. "
                    "See the archived session transcript for the full arguments."
                ),
            }
            tool_call["arguments"] = compacted_arguments
            tool_call.pop("raw_arguments", None)
            did_change = True
        elif original_raw_arguments and not _raw_arguments_materially_differ(
            original_raw_arguments,
            arguments,
        ):
            tool_call.pop("raw_arguments", None)
            did_change = True
        compacted_tool_calls.append(tool_call)

    if not did_change:
        return record

    return _copy_record_with_content(
        record,
        content=record.content,
        metadata_updates={
            "tool_calls": compacted_tool_calls,
            "carry_forward_compacted": True,
            "carry_forward_compaction_strength": strength,
            "carry_forward_tool_calls_compacted": True,
        },
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


def _close_unstarted_awaitable(awaitable: Awaitable[Any]) -> None:
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()


def _completed_after_interrupt_metadata(
    reason: InterruptionReason | None,
) -> dict[str, Any] | None:
    if reason is None:
        return None
    return {
        "completed_after_interrupt_request": True,
        "interruption_reason": reason,
        "superseded_turn_output": reason == "superseded_by_user_message",
        "new_session_boundary": reason == "new_session",
    }


def _unexecuted_tool_call_note_text(
    tool_names: Sequence[str],
    *,
    boundary: Literal["turn_interruption", "tool_slice"] = "turn_interruption",
) -> str:
    ordered_tool_names = _ordered_unique_names(tool_names)
    subject = (
        "The previous tool slice ended"
        if boundary == "tool_slice"
        else "The turn was interrupted"
    )
    if ordered_tool_names:
        names_text = ", ".join(ordered_tool_names)
        return (
            f"{subject} before these proposed tool calls were executed: "
            f"{names_text}. Treat them as not run."
        )
    return (
        f"{subject} before the assistant's proposed tool calls were executed. "
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
            pending_tool_names = tuple(_ordered_unique_names(name for _call_id, name in call_specs))
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
