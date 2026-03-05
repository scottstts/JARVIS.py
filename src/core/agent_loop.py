"""Core agentic loop with sessioning and context compaction policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import uuid4

from llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMService,
    ProviderBadRequestError,
    ToolChoice,
    ToolDefinition,
)
from storage import ConversationRecord, SessionMetadata, SessionStorage

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


@dataclass(slots=True, frozen=True)
class AgentTurnResult:
    session_id: str
    response_text: str
    command: str | None = None
    compaction_performed: bool = False


class AgentLoop:
    """Stateful agent loop over a single long-running DM thread."""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        settings: CoreSettings | None = None,
        storage: SessionStorage | None = None,
        tools: Sequence[ToolDefinition] = (),
    ) -> None:
        self._llm_service = llm_service
        self._settings = settings or CoreSettings.from_env()
        self._storage = storage or SessionStorage(self._settings.storage_dir)
        self._identity_loader = IdentityBootstrapLoader(self._settings)
        self._compactor = ContextCompactor(
            llm_service=self._llm_service,
            context_policy=self._settings.context_policy,
        )
        self._tools = tuple(tools)

    async def handle_user_input(self, user_text: str) -> AgentTurnResult:
        command = parse_user_command(user_text)
        if command.kind == "new":
            return await self._handle_new_command(command)
        if command.kind == "compact":
            return await self._handle_compact_command(command)
        return await self._handle_message_turn(command.body)

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

    async def _handle_message_turn(
        self,
        user_text: str,
        *,
        force_session_id: str | None = None,
    ) -> AgentTurnResult:
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
        request = self._build_request(records, pending_user_text=user_text)
        estimated_input_tokens = estimate_request_input_tokens(request)

        if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
            compacted = await self._compact_session(session, reason="preflight")
            if compacted is not None:
                session = compacted
                did_compaction = True
                records = self._storage.load_records(session.session_id)
                request = self._build_request(records, pending_user_text=user_text)
                estimated_input_tokens = estimate_request_input_tokens(request)

        if estimated_input_tokens >= self._settings.context_policy.preflight_limit_tokens:
            raise ContextBudgetError(
                "Request is still over the preflight context budget after compaction."
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

        self._persist_successful_turn(
            session_id=session.session_id,
            user_text=user_text,
            response=response,
            estimated_input_tokens=final_estimated_input_tokens,
        )

        refreshed = self._storage.get_session(session.session_id)
        threshold_observed = (
            response.usage.input_tokens
            if response.usage is not None and response.usage.input_tokens is not None
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
            response_text=response.text,
            compaction_performed=did_compaction,
        )

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
        retry_request = self._build_request(records, pending_user_text=user_text)
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
        user_text: str,
        response: LLMResponse,
        estimated_input_tokens: int,
    ) -> None:
        self._append_message(
            session_id=session_id,
            role="user",
            content=user_text,
        )
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
                }
                for call in response.tool_calls
            ],
        }
        self._append_message(
            session_id=session_id,
            role="assistant",
            content=response.text,
            metadata=assistant_metadata,
        )

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
        *,
        pending_user_text: str,
    ) -> LLMRequest:
        messages = [
            LLMMessage.text(record.role, record.content)
            for record in records
            if record.kind == "message"
            and record.role in {"system", "developer", "user", "assistant"}
        ]
        messages.append(LLMMessage.text("user", pending_user_text))
        return LLMRequest(
            messages=tuple(messages),
            tools=self._tools,
            tool_choice=ToolChoice.auto(),
        )

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

    def _append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = ConversationRecord(
            record_id=uuid4().hex,
            session_id=session_id,
            created_at=_utc_now_iso(),
            role=role,  # type: ignore[arg-type]
            content=content,
            kind="message",
            metadata=metadata or {},
        )
        self._storage.append_record(session_id, record)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_context_overflow_error(exc: ProviderBadRequestError) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _OVERFLOW_ERROR_HINTS)
