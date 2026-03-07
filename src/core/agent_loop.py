"""Core agentic loop with sessioning and context compaction policies."""

from __future__ import annotations
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Sequence
from uuid import uuid4

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
    ToolResultPart,
)
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
_IMAGE_INPUT_METADATA_KEY = "image_input"


@dataclass(slots=True, frozen=True)
class AgentTurnResult:
    session_id: str
    response_text: str
    command: str | None = None
    compaction_performed: bool = False


@dataclass(slots=True, frozen=True)
class AgentTextDeltaEvent:
    session_id: str
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True, frozen=True)
class AgentTurnDoneEvent:
    session_id: str
    response_text: str
    command: str | None = None
    compaction_performed: bool = False
    type: Literal["done"] = "done"

    def to_result(self) -> AgentTurnResult:
        return AgentTurnResult(
            session_id=self.session_id,
            response_text=self.response_text,
            command=self.command,
            compaction_performed=self.compaction_performed,
        )


AgentTurnStreamEvent = AgentTextDeltaEvent | AgentTurnDoneEvent


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
    ) -> None:
        self._llm_service = llm_service
        self._settings = settings or CoreSettings.from_env()
        self._storage = storage or SessionStorage(self._settings.storage_dir)
        self._identity_loader = IdentityBootstrapLoader(self._settings)
        self._compactor = ContextCompactor(
            llm_service=self._llm_service,
            context_policy=self._settings.context_policy,
        )
        self._tool_settings = ToolSettings.from_workspace_dir(self._settings.workspace_dir)
        self._tool_registry = tool_registry or ToolRegistry.default(self._tool_settings)
        self._tool_runtime = tool_runtime or ToolRuntime(registry=self._tool_registry)
        self._tool_context = ToolExecutionContext(workspace_dir=self._tool_settings.workspace_dir)

    async def handle_user_input(self, user_text: str) -> AgentTurnResult:
        command = parse_user_command(user_text)
        if command.kind == "new":
            return await self._handle_new_command(command)
        if command.kind == "compact":
            return await self._handle_compact_command(command)
        return await self._handle_message_turn(command.body)

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
        async for event in self._stream_message_turn(command.body):
            yield event

    def active_session_id(self) -> str | None:
        active = self._storage.get_active_session()
        return active.session_id if active is not None else None

    async def _handle_new_command(self, command: ParsedCommand) -> AgentTurnResult:
        session = self._start_session(start_reason="user_new")
        if command.body:
            return await self._handle_message_turn(command.body, force_session_id=session.session_id)
        return AgentTurnResult(
            session_id=session.session_id,
            response_text="Started a new session.",
            command="/new",
        )

    async def _handle_compact_command(self, command: ParsedCommand) -> AgentTurnResult:
        active = self._ensure_active_session()
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
        session = self._start_session(start_reason="user_new")
        if command.body:
            async for event in self._stream_message_turn(
                command.body,
                force_session_id=session.session_id,
                command_override="/new",
            ):
                yield event
            return

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
    ) -> AgentTurnResult:
        (
            session,
            base_records,
            request,
            estimated_input_tokens,
            did_compaction,
        ) = await self._prepare_message_turn(
            user_text=user_text,
            force_session_id=force_session_id,
        )

        (
            session,
            response,
            overflow_compacted,
            final_estimated_input_tokens,
        ) = await self._generate_with_overflow_retry(
            session=session,
            user_text=user_text,
            request=request,
            estimated_input_tokens=estimated_input_tokens,
        )
        if overflow_compacted:
            did_compaction = True

        base_records = self._storage.load_records(session.session_id)
        pending_records = [
            self._build_message_record(
                session_id=session.session_id,
                role="user",
                content=user_text,
            ),
            self._build_assistant_record(session.session_id, response),
        ]
        final_response, final_estimated_input_tokens, _used_tool_rounds = (
            await self._execute_followup_tool_rounds(
                session=session,
                base_records=base_records,
                pending_records=pending_records,
                current_response=response,
                current_estimated_input_tokens=final_estimated_input_tokens,
            )
        )

        self._persist_successful_turn(
            session_id=session.session_id,
            records=pending_records,
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

        return AgentTurnResult(
            session_id=session.session_id,
            response_text=final_response.text,
            compaction_performed=did_compaction,
        )

    async def _stream_message_turn(
        self,
        user_text: str,
        *,
        force_session_id: str | None = None,
        command_override: str | None = None,
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        (
            session,
            _base_records,
            request,
            estimated_input_tokens,
            did_compaction,
        ) = await self._prepare_message_turn(
            user_text=user_text,
            force_session_id=force_session_id,
        )

        overflow_compacted = False
        overflow_retry_attempted = False
        initial_response: LLMResponse | None = None
        final_estimated_input_tokens = estimated_input_tokens

        while True:
            streamed_response: LLMResponse | None = None
            emitted_any = False
            try:
                async for event in self._llm_service.stream_generate(request):
                    if event.type == "text_delta":
                        emitted_any = True
                        if event.delta:
                            yield AgentTextDeltaEvent(
                                session_id=session.session_id,
                                delta=event.delta,
                            )
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

            compacted = await self._compact_session(session, reason="overflow")
            if compacted is None:
                raise ContextBudgetError("Context overflow occurred and compaction could not proceed.")

            records = self._storage.load_records(compacted.session_id)
            request = self._build_request(
                records
                + [
                    self._build_message_record(
                        session_id=compacted.session_id,
                        role="user",
                        content=user_text,
                    )
                ]
            )
            retry_estimate = estimate_request_input_tokens(request)
            if retry_estimate >= self._settings.context_policy.preflight_limit_tokens:
                raise ContextBudgetError(
                    "Overflow retry aborted: compacted request still exceeds preflight limit."
                )

            session = compacted
            final_estimated_input_tokens = retry_estimate
            overflow_compacted = True
            overflow_retry_attempted = True

        if initial_response is None:
            raise RuntimeError("Streaming generation produced no final response.")
        if overflow_compacted:
            did_compaction = True

        base_records = self._storage.load_records(session.session_id)
        pending_records = [
            self._build_message_record(
                session_id=session.session_id,
                role="user",
                content=user_text,
            ),
            self._build_assistant_record(session.session_id, initial_response),
        ]
        final_response, final_estimated_input_tokens, used_tool_rounds = (
            await self._execute_followup_tool_rounds(
                session=session,
                base_records=base_records,
                pending_records=pending_records,
                current_response=initial_response,
                current_estimated_input_tokens=final_estimated_input_tokens,
            )
        )
        if used_tool_rounds and final_response.text:
            yield AgentTextDeltaEvent(
                session_id=session.session_id,
                delta=final_response.text,
            )

        self._persist_successful_turn(
            session_id=session.session_id,
            records=pending_records,
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
        )

    async def _prepare_message_turn(
        self,
        *,
        user_text: str,
        force_session_id: str | None = None,
    ) -> tuple[SessionMetadata, list[ConversationRecord], LLMRequest, int, bool]:
        session = self._storage.get_session(force_session_id) if force_session_id else None
        if session is None:
            session = self._ensure_active_session()

        did_compaction = False
        if session.pending_reactive_compaction:
            compacted = await self._compact_session(session, reason="reactive")
            if compacted is not None:
                session = compacted
                did_compaction = True

        records = self._storage.load_records(session.session_id)
        request = self._build_request(
            records
            + [
                self._build_message_record(
                    session_id=session.session_id,
                    role="user",
                    content=user_text,
                )
            ]
        )
        estimated_input_tokens = estimate_request_input_tokens(request)

        if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
            compacted = await self._compact_session(session, reason="preflight")
            if compacted is not None:
                session = compacted
                did_compaction = True
                records = self._storage.load_records(session.session_id)
                request = self._build_request(
                    records
                    + [
                        self._build_message_record(
                            session_id=session.session_id,
                            role="user",
                            content=user_text,
                        )
                    ]
                )
                estimated_input_tokens = estimate_request_input_tokens(request)

        if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
            raise ContextBudgetError(
                "Request is still over the preflight context budget after compaction."
            )

        return session, records, request, estimated_input_tokens, did_compaction

    async def _generate_with_overflow_retry(
        self,
        *,
        session: SessionMetadata,
        user_text: str,
        request: LLMRequest,
        estimated_input_tokens: int,
    ) -> tuple[SessionMetadata, LLMResponse, bool, int]:
        try:
            response = await self._llm_service.generate(request)
            return session, response, False, estimated_input_tokens
        except ProviderBadRequestError as exc:
            if not _is_context_overflow_error(exc):
                raise

        compacted = await self._compact_session(session, reason="overflow")
        if compacted is None:
            raise ContextBudgetError("Context overflow occurred and compaction could not proceed.")

        records = self._storage.load_records(compacted.session_id)
        retry_request = self._build_request(
            records
            + [
                self._build_message_record(
                    session_id=compacted.session_id,
                    role="user",
                    content=user_text,
                )
            ]
        )
        retry_estimate = estimate_request_input_tokens(retry_request)
        if retry_estimate >= self._settings.context_policy.preflight_limit_tokens:
            raise ContextBudgetError(
                "Overflow retry aborted: compacted request still exceeds preflight limit."
            )
        response = await self._llm_service.generate(retry_request)
        return compacted, response, True, retry_estimate

    def _persist_successful_turn(
        self,
        *,
        session_id: str,
        records: Sequence[ConversationRecord],
        response: LLMResponse,
        estimated_input_tokens: int,
    ) -> None:
        for record in records:
            if bool(record.metadata.get(_TRANSIENT_RECORD_METADATA_KEY, False)):
                continue
            self._storage.append_record(session_id, record)

        usage = response.usage
        self._storage.update_session(
            session_id,
            last_input_tokens=usage.input_tokens if usage is not None else None,
            last_output_tokens=usage.output_tokens if usage is not None else None,
            last_total_tokens=usage.total_tokens if usage is not None else None,
            last_estimated_input_tokens=estimated_input_tokens,
        )

    def _build_request(
        self,
        records: Sequence[ConversationRecord],
    ) -> LLMRequest:
        messages: list[LLMMessage] = []
        for record in records:
            if record.kind != "message":
                continue
            llm_message = _record_to_llm_message(record)
            if llm_message is not None:
                messages.append(llm_message)
        return LLMRequest(
            messages=tuple(messages),
            tools=self._tool_registry.basic_definitions(),
            tool_choice=ToolChoice.auto(),
        )

    async def _execute_followup_tool_rounds(
        self,
        *,
        session: SessionMetadata,
        base_records: Sequence[ConversationRecord],
        pending_records: list[ConversationRecord],
        current_response: LLMResponse,
        current_estimated_input_tokens: int,
    ) -> tuple[LLMResponse, int, bool]:
        tool_rounds = 0

        while current_response.tool_calls:
            tool_rounds += 1
            if tool_rounds > self._tool_settings.max_tool_rounds_per_turn:
                raise RuntimeError(
                    "Tool loop exceeded max rounds for a single turn."
                )

            transient_records: list[ConversationRecord] = []
            for tool_call in current_response.tool_calls:
                tool_result = await self._tool_runtime.execute(
                    tool_call=tool_call,
                    context=self._tool_context,
                )
                pending_records.append(
                    self._build_tool_record(session.session_id, tool_result)
                )
                transient_records.extend(
                    self._build_transient_records_from_tool_result(
                        session.session_id,
                        tool_result,
                    )
                )

            pending_records.extend(transient_records)

            request = self._build_request(list(base_records) + pending_records)
            current_estimated_input_tokens = estimate_request_input_tokens(request)
            if current_estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
                raise ContextBudgetError(
                    "Tool output exceeded the context budget during the current turn."
                )

            current_response = await self._llm_service.generate(request)
            pending_records.append(
                self._build_assistant_record(session.session_id, current_response)
            )

        return current_response, current_estimated_input_tokens, tool_rounds > 0

    def _ensure_active_session(self) -> SessionMetadata:
        active = self._storage.get_active_session()
        if active is not None:
            return active
        return self._start_session(start_reason="initial")

    def _start_session(
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

        bootstrap_messages = self._identity_loader.load_bootstrap_messages(
            summary_text=summary_text,
        )
        for message in bootstrap_messages:
            metadata: dict[str, Any]
            if message.role == "developer":
                metadata = {"summary_seed": True}
            else:
                metadata = {"bootstrap_identity": True}
            self._append_message(
                session_id=session.session_id,
                role=message.role,
                content=message.parts[0].text,
                metadata=metadata,
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
    ) -> SessionMetadata | None:
        records = self._storage.load_records(session.session_id)
        compactable_records = [
            record
            for record in records
            if record.kind == "message" and not record.metadata.get("bootstrap_identity", False)
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
        next_session = self._start_session(
            start_reason="compaction",
            parent_session_id=session.session_id,
            summary_text=outcome.summary_text,
            compaction_count=next_compaction_count,
        )
        self._storage.update_session(
            next_session.session_id,
            pending_reactive_compaction=False,
        )
        return self._storage.get_session(next_session.session_id) or next_session

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
    ) -> ConversationRecord:
        return ConversationRecord(
            record_id=uuid4().hex,
            session_id=session_id,
            created_at=_utc_now_iso(),
            role=role,  # type: ignore[arg-type]
            content=content,
            kind="message",
            metadata=metadata or {},
        )

    def _build_assistant_record(
        self,
        session_id: str,
        response: LLMResponse,
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
        )

    def _build_tool_record(
        self,
        session_id: str,
        result: ToolExecutionResult,
    ) -> ConversationRecord:
        metadata = dict(result.metadata)
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_context_overflow_error(exc: ProviderBadRequestError) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _OVERFLOW_ERROR_HINTS)


def _record_to_llm_message(record: ConversationRecord) -> LLMMessage | None:
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
