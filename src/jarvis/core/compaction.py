"""Evidence-backed session compaction orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence
from uuid import uuid4

from jarvis.logging_setup import get_application_logger
from jarvis.llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMService,
    ToolChoice,
    ToolDefinition,
)
from jarvis.storage import ConversationRecord

from .compaction_contract import (
    CompactionBundle,
    CompactionChronology,
    CompactionContractError,
    CompactionContractIssue,
    CompactionPreservedRecord,
    CompactionReplayItem,
    CompactionSemanticProvenance,
    CompactionSourceEvent,
    apply_compaction_draft,
    build_fallback_compaction_bundle,
    build_source_manifest,
    compile_compaction_replay,
)
from .config import ContextPolicySettings
from .errors import ContextBudgetError
from .token_estimator import estimate_request_input_tokens


_COMPACTION_PROMPT_PATH = Path(__file__).with_name("prompts") / "COMPACTION.md"
_COMPACTION_SYSTEM_PROMPT = _COMPACTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
_COMPACTION_INPUT_SAFETY_PERCENT = 70
_COMPACTION_SOURCE_EVENT_LIMITS: tuple[int | None, ...] = (
    None,
    32_000,
    16_000,
    8_000,
    4_000,
    2_000,
)
_COMPACTION_RENDER_TOKEN_MARGIN = 512
_COMPACTION_MIN_EVENT_BODY_CHARS = 256
_RECENT_CONTEXT_MIN_TOKENS = 2_000
_RECENT_CONTEXT_MAX_TOKENS = 12_000
_RECENT_CONTEXT_PREFLIGHT_PERCENT = 10
_RECENT_RECORD_OVERHEAD_CHARS = 256
_COMPACTION_TOOL_NAME = "submit_compaction"
_COMPACTION_SEMANTIC_FIELDS = {
    "objective",
    "background",
    "episodes",
    "constraints",
    "decisions",
    "artifacts",
    "open_loops",
    "uncertainties",
    "handover",
}
_COMPACTION_TOOL = ToolDefinition(
    name=_COMPACTION_TOOL_NAME,
    description="Submit the complete semantic continuation record for this compaction.",
    strict=False,
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "objective": {"type": "string"},
            "background": {"type": "array", "items": {"type": "string"}},
            "episodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "outcomes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["summary", "outcomes"],
                },
            },
            "constraints": {"type": "array", "items": {"type": "string"}},
            "decisions": {"type": "array", "items": {"type": "string"}},
            "artifacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "locator": {"type": "string"},
                        "last_observed_state": {"type": "string"},
                        "needs_verification": {"type": "boolean"},
                    },
                    "required": [
                        "summary",
                        "locator",
                        "last_observed_state",
                        "needs_verification",
                    ],
                },
            },
            "open_loops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "next_action": {"type": "string"},
                        "blocker": {"type": ["string", "null"]},
                    },
                    "required": ["summary", "next_action", "blocker"],
                },
            },
            "uncertainties": {"type": "array", "items": {"type": "string"}},
            "handover": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "current_focus": {"type": "string"},
                    "next_actions": {"type": "array", "items": {"type": "string"}},
                    "do_not_repeat": {"type": "array", "items": {"type": "string"}},
                    "verification_needed": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "current_focus",
                ],
            },
        },
        "required": [
            "objective",
            "handover",
        ],
    },
)
_TURN_CONTEXT_PREFIX = "System context auto-appended for this turn only."
_SUBAGENT_STATUS_PREFIX = "Subagent status snapshot:"
_TRANSIENT_SYSTEM_METADATA_KEYS = {
    "orchestrator_monitored_waiting",
    "orchestrator_wait_only_update",
}
_SOURCE_METADATA_KEYS = {
    "turn_id",
    "call_id",
    "tool_name",
    "ok",
    "tool_call_validation_failed",
    "reason",
    "approval_event",
    "approval_id",
    "approved",
    "command",
    "interruption_notice",
    "interruption_reason",
    "interrupted_by_user",
    "superseded_by_user_message",
    "new_session_boundary",
    "unexecuted_tool_call_notice",
    "subagent_progress_update",
    "subagent_id",
    "subagent_notice_kind",
    "recommended_action",
    "latest_subagent_report_complete",
    "latest_subagent_report_truncated",
    "pending_subagent_ids",
    "bash_background_promotion",
    "detached_bash_job_ids",
    "carry_forward_compacted",
    "carry_forward_compaction_strength",
    "image_input",
    "task_contract",
    "task_contract_revision",
    "task_id",
    "user_message_sha256",
    "bash_job_progress_update",
    "bash_job_notice_kinds",
    "bash_job_running_ids",
    "bash_job_terminal_ids",
    "error_code",
    "conflict_class",
    "conflict_key",
}
LOGGER = get_application_logger(__name__)


@dataclass(slots=True, frozen=True)
class CompactionCallTrace:
    phase: Literal["generate"]
    provider: str
    model: str
    response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "provider": self.provider,
            "model": self.model,
            "response_id": self.response_id,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
        }


@dataclass(slots=True, frozen=True)
class CompactionOutcome:
    bundle: CompactionBundle
    items: tuple[CompactionReplayItem, ...]
    draft_payload: dict[str, Any] | None
    verification_payload: dict[str, Any]
    call_traces: tuple[CompactionCallTrace, ...]
    semantic_status: Literal["accepted", "fallback"]
    semantic_source: Literal["model", "previous_snapshot", "minimal"]
    semantic_issue_code: str | None = None

    @property
    def provider(self) -> str | None:
        return self.call_traces[0].provider if self.call_traces else None

    @property
    def model(self) -> str | None:
        return self.call_traces[0].model if self.call_traces else None

    @property
    def response_id(self) -> str | None:
        return self.call_traces[0].response_id if self.call_traces else None

    @property
    def input_tokens(self) -> int | None:
        return _sum_optional(trace.input_tokens for trace in self.call_traces)

    @property
    def output_tokens(self) -> int | None:
        return _sum_optional(trace.output_tokens for trace in self.call_traces)

    @property
    def total_tokens(self) -> int | None:
        return _sum_optional(trace.total_tokens for trace in self.call_traces)


class ContextCompactor:
    """Builds and verifies a canonical compaction bundle from transcript deltas."""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        context_policy: ContextPolicySettings,
        provider: str | None = None,
        deadline_seconds: float | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._context_policy = context_policy
        self._provider = provider
        self._deadline_seconds = _resolve_compaction_deadline_seconds(
            llm_service,
            explicit_deadline_seconds=deadline_seconds,
        )

    async def compact(
        self,
        records: Sequence[ConversationRecord],
        *,
        previous_bundle: CompactionBundle | None = None,
        user_instruction: str | None = None,
    ) -> CompactionOutcome:
        diagnostics: dict[str, Any] = {
            "operation": "compaction",
            "compaction_phase": "prepare",
            "compaction_draft_attempt": 0,
            "compaction_max_draft_attempts": 1,
            "compaction_call_count": 0,
            "compaction_call_traces": [],
            "compaction_deadline_seconds": self._deadline_seconds,
        }
        try:
            return await self._compact_bounded(
                records,
                previous_bundle=previous_bundle,
                user_instruction=user_instruction,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            _attach_compaction_error_metadata(exc, diagnostics)
            raise

    async def _compact_bounded(
        self,
        records: Sequence[ConversationRecord],
        *,
        previous_bundle: CompactionBundle | None,
        user_instruction: str | None,
        diagnostics: dict[str, Any],
    ) -> CompactionOutcome:
        source_records = prune_compaction_source_records(records)
        if not source_records and previous_bundle is None:
            raise ValueError("Compaction source and prior bundle are both empty.")

        generation = previous_bundle.generation + 1 if previous_bundle is not None else 1
        source_events = build_compaction_source_events(source_records, generation=generation)
        source_manifest = build_source_manifest(
            generation=generation,
            previous_bundle=previous_bundle,
            source_events=source_events,
        )
        instruction = user_instruction.strip() if user_instruction else ""
        recent_records = _select_recent_records(
            source_events=source_events,
            previous_bundle=previous_bundle,
            char_budget=_recent_context_char_budget(self._context_policy),
        )
        diagnostics.update(
            {
                "compaction_recent_record_count": len(recent_records),
                "compaction_recent_context_chars": sum(
                    len(record.content) for record in recent_records
                ),
                "compaction_draft_attempt": 1,
            }
        )
        call_traces: list[CompactionCallTrace] = []
        draft_payload: dict[str, Any] | None = None
        semantic_issue_code: str | None = None
        bundle_id = uuid4().hex
        created_at = _utc_now_iso()

        semantic_error: Exception | None = None
        response: LLMResponse | None = None
        try:
            diagnostics["compaction_phase"] = "generate"
            LOGGER.info("Compaction semantic refresh request started (single best-effort pass).")
            request = self._build_bounded_request(
                source_events=source_events,
                previous_bundle=previous_bundle,
                user_instruction=instruction or None,
                diagnostics=diagnostics,
            )
            if self._deadline_seconds is None:
                response = await self._llm_service.generate(request)
            else:
                async with asyncio.timeout(self._deadline_seconds):
                    response = await self._llm_service.generate(request)
        except Exception as exc:
            semantic_error = exc

        if response is not None:
            call_trace = _trace_response(response, phase="generate")
            call_traces.append(call_trace)
            _update_call_trace_diagnostics(diagnostics, call_traces)
            _log_completed_call(call_trace)
            try:
                draft_payload = _extract_compaction_submission(response)
                ignored_fields = sorted(set(draft_payload) - _COMPACTION_SEMANTIC_FIELDS)
                if ignored_fields:
                    diagnostics["compaction_semantic_ignored_fields"] = ignored_fields
                    LOGGER.info(
                        "Ignored non-semantic compaction submission fields: %s.",
                        ", ".join(ignored_fields),
                    )
                bundle = apply_compaction_draft(
                    draft_payload,
                    bundle_id=bundle_id,
                    created_at=created_at,
                    source_manifest=source_manifest,
                    recent_records=recent_records,
                    semantic_provenance=CompactionSemanticProvenance(
                        status="accepted",
                        source="model",
                    ),
                )
            except (ValueError, CompactionContractError) as exc:
                semantic_error = exc
            else:
                verification_payload = {
                    "valid": True,
                    "method": "jarvis_deterministic_bundle",
                    "schema_version": bundle.schema_version,
                    "semantic_status": "accepted",
                    "semantic_source": "model",
                    "ignored_semantic_fields": ignored_fields,
                }
                return CompactionOutcome(
                    bundle=bundle,
                    items=compile_compaction_replay(bundle),
                    draft_payload=draft_payload,
                    verification_payload=verification_payload,
                    call_traces=tuple(call_traces),
                    semantic_status="accepted",
                    semantic_source="model",
                )

        if semantic_error is not None:
            issues = _issues_from_exception(semantic_error)
            semantic_issue_code = issues[0].code
            diagnostics["compaction_last_issues"] = [issue.to_dict() for issue in issues]
            LOGGER.warning(
                "Compaction semantic refresh unavailable or rejected; using deterministic "
                "fallback (issue=%s).",
                semantic_issue_code,
            )
            LOGGER.debug(
                "Semantic refresh failure detail.",
                exc_info=(
                    type(semantic_error),
                    semantic_error,
                    semantic_error.__traceback__,
                ),
            )

        bundle = build_fallback_compaction_bundle(
            bundle_id=bundle_id,
            created_at=created_at,
            source_manifest=source_manifest,
            recent_records=recent_records,
            previous_bundle=previous_bundle,
            issue_code=semantic_issue_code or "semantic_refresh_unavailable",
        )
        verification_payload = {
            "valid": True,
            "method": "jarvis_deterministic_fallback",
            "schema_version": bundle.schema_version,
            "semantic_status": "fallback",
            "semantic_source": bundle.semantic_provenance.source,
            "semantic_issue_code": bundle.semantic_provenance.issue_code,
        }
        return CompactionOutcome(
            bundle=bundle,
            items=compile_compaction_replay(bundle),
            draft_payload=draft_payload,
            verification_payload=verification_payload,
            call_traces=tuple(call_traces),
            semantic_status="fallback",
            semantic_source=(
                "previous_snapshot"
                if bundle.semantic_provenance.source == "previous_snapshot"
                else "minimal"
            ),
            semantic_issue_code=bundle.semantic_provenance.issue_code,
        )

    def _build_bounded_request(
        self,
        *,
        source_events: Sequence[CompactionSourceEvent],
        previous_bundle: CompactionBundle | None,
        user_instruction: str | None,
        diagnostics: dict[str, Any],
    ) -> LLMRequest:
        safe_input_limit = max(
            1,
            (
                self._context_policy.preflight_limit_tokens
                * _COMPACTION_INPUT_SAFETY_PERCENT
            )
            // 100,
        )
        base_text, _ = _render_compaction_input(
            source_events=(),
            previous_bundle=previous_bundle,
            user_instruction=user_instruction,
            event_content_limit=0,
            source_char_budget=0,
        )
        base_estimate = estimate_request_input_tokens(
            self._build_request(user_text=base_text)
        )
        diagnostics.update(
            {
                "compaction_safe_input_limit_tokens": safe_input_limit,
                "compaction_preflight_limit_tokens": (
                    self._context_policy.preflight_limit_tokens
                ),
                "compaction_source_event_count": len(source_events),
            }
        )
        if base_estimate + _COMPACTION_RENDER_TOKEN_MARGIN >= safe_input_limit:
            diagnostics["compaction_estimated_input_tokens"] = base_estimate
            error = ContextBudgetError(
                "Compaction instructions and prior canonical context exceed the safe input budget."
            )
            _attach_compaction_error_metadata(error, diagnostics)
            raise error
        source_char_budget = max(
            1,
            (
                safe_input_limit
                - base_estimate
                - _COMPACTION_RENDER_TOKEN_MARGIN
            )
            * 4,
        )
        last_estimate = 0
        for event_content_limit in _COMPACTION_SOURCE_EVENT_LIMITS:
            try:
                user_text, truncated_event_count = _render_compaction_input(
                    source_events=source_events,
                    previous_bundle=previous_bundle,
                    user_instruction=user_instruction,
                    event_content_limit=event_content_limit,
                    source_char_budget=source_char_budget,
                )
            except ContextBudgetError as exc:
                metadata = getattr(exc, "metadata", {})
                minimum_chars = int(metadata.get("compaction_minimum_render_chars", 0))
                diagnostics.update(
                    {
                        "compaction_estimated_input_tokens": (
                            base_estimate + max(0, minimum_chars + 3) // 4
                        ),
                        "compaction_source_global_char_budget": source_char_budget,
                        "compaction_source_event_char_limit": event_content_limit,
                    }
                )
                _attach_compaction_error_metadata(exc, diagnostics)
                raise
            request = self._build_request(user_text=user_text)
            last_estimate = estimate_request_input_tokens(request)
            diagnostics.update(
                {
                    "compaction_estimated_input_tokens": last_estimate,
                    "compaction_safe_input_limit_tokens": safe_input_limit,
                    "compaction_preflight_limit_tokens": (
                        self._context_policy.preflight_limit_tokens
                    ),
                    "compaction_source_event_count": len(source_events),
                    "compaction_source_input_chars": len(user_text),
                    "compaction_source_event_char_limit": event_content_limit,
                    "compaction_source_global_char_budget": source_char_budget,
                    "compaction_source_truncated_event_count": truncated_event_count,
                }
            )
            if last_estimate < safe_input_limit:
                LOGGER.info(
                    "Compaction input prepared (events=%d, chars=%d, estimated_tokens=%d, "
                    "safe_limit=%d, truncated_events=%d, event_char_limit=%s).",
                    len(source_events),
                    len(user_text),
                    last_estimate,
                    safe_input_limit,
                    truncated_event_count,
                    event_content_limit,
                )
                return request
            excess_chars = max(4, (last_estimate - safe_input_limit + 1) * 4)
            source_char_budget = max(1, source_char_budget - excess_chars)
        error = ContextBudgetError(
            "Compaction source remains over its safe input budget after bounded rendering."
        )
        _attach_compaction_error_metadata(error, diagnostics)
        raise error

    def _build_request(self, *, user_text: str) -> LLMRequest:
        return LLMRequest(
            messages=(
                LLMMessage.text("system", _COMPACTION_SYSTEM_PROMPT),
                LLMMessage.text("user", user_text),
            ),
            provider=self._provider,
            tools=(_COMPACTION_TOOL,),
            tool_choice=ToolChoice.tool(_COMPACTION_TOOL_NAME),
            parallel_tool_calls=False,
            max_output_tokens=self._context_policy.compact_reserve_output_tokens,
        )


def prune_compaction_source_records(
    records: Sequence[ConversationRecord],
) -> tuple[ConversationRecord, ...]:
    return tuple(record for record in records if not _should_drop_source_record(record))


def build_compaction_source_events(
    records: Sequence[ConversationRecord],
    *,
    generation: int,
) -> tuple[CompactionSourceEvent, ...]:
    events: list[CompactionSourceEvent] = []
    for sequence, record in enumerate(records, start=1):
        metadata = _source_metadata(record.metadata)
        events.append(
            CompactionSourceEvent(
                event_id=record.record_id,
                record_id=record.record_id,
                session_id=record.session_id,
                created_at=record.created_at,
                sequence=sequence,
                generation=generation,
                event_type=_source_event_type(record),
                role=record.role,
                content=record.content,
                turn_id=_optional_string(record.metadata.get("turn_id")),
                causal_ids=_causal_ids(record),
                metadata=metadata or None,
            )
        )
    return tuple(events)


def load_compaction_bundle(records: Sequence[ConversationRecord]) -> CompactionBundle | None:
    for record in reversed(records):
        if record.kind != "compaction":
            continue
        if not bool(record.metadata.get("compaction_bundle_anchor", False)):
            continue
        raw_bundle = record.metadata.get("bundle")
        if not isinstance(raw_bundle, Mapping):
            raise ValueError("Compaction bundle anchor is missing its canonical bundle.")
        return CompactionBundle.from_dict(raw_bundle)
    return None


def build_compaction_bundle_record(
    *,
    session_id: str,
    bundle: CompactionBundle,
) -> ConversationRecord:
    return ConversationRecord(
        record_id=uuid4().hex,
        session_id=session_id,
        created_at=_utc_now_iso(),
        role="system",
        content=f"Canonical compaction bundle {bundle.bundle_id} generation {bundle.generation}.",
        kind="compaction",
        metadata={
            "compaction_bundle_anchor": True,
            "bundle_id": bundle.bundle_id,
            "generation": bundle.generation,
            "bundle": bundle.to_dict(),
        },
    )


def _recent_context_char_budget(context_policy: ContextPolicySettings) -> int:
    tokens = max(
        _RECENT_CONTEXT_MIN_TOKENS,
        (
            context_policy.preflight_limit_tokens
            * _RECENT_CONTEXT_PREFLIGHT_PERCENT
        )
        // 100,
    )
    return min(tokens, _RECENT_CONTEXT_MAX_TOKENS) * 4


def _select_recent_records(
    *,
    source_events: Sequence[CompactionSourceEvent],
    previous_bundle: CompactionBundle | None,
    char_budget: int,
) -> tuple[CompactionPreservedRecord, ...]:
    """Select a bounded causal tail without asking the model to author references."""

    previous = list(previous_bundle.recent_records) if previous_bundle is not None else []
    current: list[CompactionPreservedRecord] = []
    turn_keys: list[str] = []
    for event in source_events:
        if event.role not in {"user", "assistant"} or not event.content.strip():
            continue
        turn_key = event.turn_id or f"record:{event.record_id}"
        current.append(
            CompactionPreservedRecord(
                record_id=event.record_id,
                source_session_id=event.session_id,
                created_at=event.created_at,
                role=event.role,  # type: ignore[arg-type]
                content=event.content,
                content_sha256=_sha256_text(event.content),
                reason="deterministic_recent_context",
                chronology=CompactionChronology(
                    generation=event.generation,
                    sequence=event.sequence,
                ),
                causal_group_id=turn_key,
            )
        )
        turn_keys.append(turn_key)

    groups: list[list[CompactionPreservedRecord]] = []
    for record in previous:
        if groups and groups[-1][-1].causal_group_id == record.causal_group_id:
            groups[-1].append(record)
        else:
            groups.append([record])
    current_groups: list[list[CompactionPreservedRecord]] = []
    current_group_keys: list[str] = []
    for record, turn_key in zip(current, turn_keys, strict=True):
        if current_group_keys and current_group_keys[-1] == turn_key:
            current_groups[-1].append(record)
        else:
            current_group_keys.append(turn_key)
            current_groups.append([record])
    groups.extend(current_groups)

    selected_groups: list[list[CompactionPreservedRecord]] = []
    used_chars = 0
    for group in reversed(groups):
        group_chars = sum(
            len(record.content) + _RECENT_RECORD_OVERHEAD_CHARS for record in group
        )
        if used_chars + group_chars > char_budget:
            break
        selected_groups.append(group)
        used_chars += group_chars
    selected = [record for group in reversed(selected_groups) for record in group]
    unique: dict[str, CompactionPreservedRecord] = {}
    for record in selected:
        unique[record.record_id] = record
    return tuple(sorted(unique.values(), key=lambda item: item.chronology))


def _should_drop_source_record(record: ConversationRecord) -> bool:
    metadata = record.metadata
    if record.kind != "message":
        return True
    if metadata.get("compaction_item"):
        return True
    if metadata.get("bootstrap_identity"):
        return True
    if metadata.get("transcript_only"):
        return True
    if metadata.get("memory_bootstrap"):
        return True
    if metadata.get("skills_bootstrap"):
        return True
    if metadata.get("turn_context") == "datetime":
        return True
    if metadata.get("subagent_status_snapshot"):
        return True
    if any(metadata.get(key) for key in _TRANSIENT_SYSTEM_METADATA_KEYS):
        return True
    if record.role == "assistant" and not record.content.strip() and not metadata.get("tool_calls"):
        return True
    if record.content.startswith(_TURN_CONTEXT_PREFIX):
        return True
    if record.content.startswith(_SUBAGENT_STATUS_PREFIX) and record.role == "system":
        return True
    return False


def _source_event_type(record: ConversationRecord) -> str:
    metadata = record.metadata
    if metadata.get("approval_event"):
        return "approval"
    if (
        metadata.get("interruption_notice")
        or metadata.get("interrupted_by_user")
        or metadata.get("superseded_by_user_message")
        or metadata.get("new_session_boundary")
    ):
        return "interruption"
    if metadata.get("subagent_progress_update"):
        pending = metadata.get("pending_subagent_ids")
        if not isinstance(pending, list) or not pending:
            return "subagent_outcome"
        return "subagent_progress"
    if metadata.get("tool_call_validation_failed"):
        return "tool_validation_error"
    if record.role == "tool":
        return "tool_result"
    if record.role == "user":
        return "user_message"
    if record.role == "assistant" and metadata.get("tool_calls"):
        return "assistant_tool_call"
    if record.role == "assistant":
        return "assistant_message"
    return "system_event"


def _causal_ids(record: ConversationRecord) -> tuple[str, ...]:
    ids: list[str] = []
    call_id = _optional_string(record.metadata.get("call_id"))
    if call_id is not None:
        ids.append(call_id)
    raw_calls = record.metadata.get("tool_calls")
    if isinstance(raw_calls, list):
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                continue
            raw_call_id = _optional_string(raw_call.get("call_id"))
            if raw_call_id is not None and raw_call_id not in ids:
                ids.append(raw_call_id)
    return tuple(ids)


def _source_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in _SOURCE_METADATA_KEYS:
        if key in metadata:
            selected[key] = _normalize_json_value(metadata[key])
    raw_calls = metadata.get("tool_calls")
    if isinstance(raw_calls, list):
        normalized_calls: list[dict[str, Any]] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                continue
            normalized_calls.append(
                {
                    "call_id": _optional_string(raw_call.get("call_id")),
                    "name": _optional_string(raw_call.get("name")),
                    "arguments": _normalize_json_value(raw_call.get("arguments", {})),
                }
            )
        if normalized_calls:
            selected["tool_calls"] = normalized_calls
    return selected


def _render_compaction_input(
    *,
    source_events: Sequence[CompactionSourceEvent],
    previous_bundle: CompactionBundle | None,
    user_instruction: str | None,
    event_content_limit: int | None,
    source_char_budget: int | None = None,
) -> tuple[str, int]:
    previous_context = (
        _canonical_json(_previous_bundle_semantics(previous_bundle))
        if previous_bundle is not None
        else "none"
    )
    transcript, truncated_event_count = _render_source_transcript(
        source_events,
        event_content_limit=event_content_limit,
        total_char_budget=source_char_budget,
    )
    sections = [
        "Create the complete current continuation record and call submit_compaction once.",
        "Review the record for omissions, contradictions, stale observations, and false "
        "completion before submitting it.",
        "",
        "EXPLICIT COMPACTION INSTRUCTION:",
        user_instruction or "none",
        "",
        "PREVIOUS CANONICAL CONTEXT:",
        previous_context,
        "",
        "NEW ORDERED TRANSCRIPT EVIDENCE:",
        transcript or "none",
    ]
    return "\n".join(sections), truncated_event_count


def _previous_bundle_semantics(bundle: CompactionBundle) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any] | str]] = {
        "constraints": [],
        "decisions": [],
        "artifacts": [],
        "open_loops": [],
        "uncertainties": [],
    }
    for entry in bundle.state_entries:
        if entry.category == "constraint":
            by_category["constraints"].append(entry.summary)
        elif entry.category == "decision":
            by_category["decisions"].append(entry.summary)
        elif entry.category == "uncertainty":
            by_category["uncertainties"].append(entry.summary)
        elif entry.category == "artifact":
            by_category["artifacts"].append(
                {
                    "summary": entry.summary,
                    "locator": entry.locator,
                    "last_observed_state": entry.last_observed_state,
                    "needs_verification": entry.needs_verification,
                }
            )
        elif entry.category == "open_loop":
            by_category["open_loops"].append(
                {
                    "summary": entry.summary,
                    "next_action": entry.next_action,
                    "blocker": entry.blocker,
                }
            )
    return {
        "semantic_provenance": bundle.semantic_provenance.to_dict(),
        "objective": bundle.objective.summary,
        "background": list(bundle.background),
        "episodes": [
            {"summary": episode.summary, "outcomes": list(episode.outcomes)}
            for episode in bundle.episodes
        ],
        **by_category,
        "handover": bundle.handover.to_dict(),
    }


def _render_source_transcript(
    source_events: Sequence[CompactionSourceEvent],
    *,
    event_content_limit: int | None,
    total_char_budget: int | None = None,
) -> tuple[str, int]:
    event_ref_by_call_id: dict[str, str] = {}
    result_by_call_id: dict[str, tuple[str, str]] = {}
    for index, event in enumerate(source_events, start=1):
        source_ref = f"E{index}"
        if event.event_type == "assistant_tool_call":
            for call_id in event.causal_ids:
                event_ref_by_call_id[call_id] = source_ref
        if event.event_type == "tool_result":
            for call_id in event.causal_ids:
                result_by_call_id[call_id] = (source_ref, event.content)

    duplicate_refs = _duplicate_system_event_refs(source_events)
    exact_indices = {
        index
        for index, event in enumerate(source_events, start=1)
        if _source_event_requires_exact_render(event) and index not in duplicate_refs
    }

    def render_with_limit(content_limit: int | None) -> tuple[str, int]:
        blocks: list[str] = []
        truncated_event_count = 0
        for index, event in enumerate(source_events, start=1):
            block, event_truncated_count = _render_source_event_block(
                event=event,
                index=index,
                event_ref_by_call_id=event_ref_by_call_id,
                result_by_call_id=result_by_call_id,
                content_limit=None if index in exact_indices else content_limit,
                duplicate_ref=duplicate_refs.get(index),
            )
            blocks.append(block)
            truncated_event_count += event_truncated_count
        return "\n\n".join(blocks), truncated_event_count

    if total_char_budget is None:
        return render_with_limit(event_content_limit)

    minimum_text, minimum_truncated_count = render_with_limit(
        _COMPACTION_MIN_EVENT_BODY_CHARS
    )
    minimum_chars = len(minimum_text)
    if minimum_chars > total_char_budget:
        error = ContextBudgetError(
            "Compaction exact messages and minimum causal evidence exceed the safe input budget."
        )
        _attach_compaction_error_metadata(
            error,
            {
                "compaction_minimum_render_chars": minimum_chars,
                "compaction_source_global_char_budget": total_char_budget,
                "compaction_exact_event_count": len(exact_indices),
            },
        )
        raise error

    upper_limit = max(
        _COMPACTION_MIN_EVENT_BODY_CHARS,
        event_content_limit if event_content_limit is not None else total_char_budget,
    )
    upper_text, upper_truncated_count = render_with_limit(upper_limit)
    if len(upper_text) <= total_char_budget:
        return upper_text, upper_truncated_count

    best_text = minimum_text
    best_truncated_count = minimum_truncated_count
    lower = _COMPACTION_MIN_EVENT_BODY_CHARS + 1
    upper = upper_limit - 1
    while lower <= upper:
        midpoint = (lower + upper) // 2
        candidate_text, candidate_truncated_count = render_with_limit(midpoint)
        if len(candidate_text) <= total_char_budget:
            best_text = candidate_text
            best_truncated_count = candidate_truncated_count
            lower = midpoint + 1
        else:
            upper = midpoint - 1
    return best_text, best_truncated_count


def _render_source_event_block(
    *,
    event: CompactionSourceEvent,
    index: int,
    event_ref_by_call_id: Mapping[str, str],
    result_by_call_id: Mapping[str, tuple[str, str]],
    content_limit: int | None,
    duplicate_ref: tuple[str, int] | None,
) -> tuple[str, int]:
    source_ref = f"E{index}"
    header_parts = [
        source_ref,
        f"role={event.role}",
        f"type={event.event_type}",
        f"at={event.created_at}",
    ]
    tool_name = _optional_string((event.metadata or {}).get("tool_name"))
    if tool_name is not None:
        header_parts.append(f"tool={tool_name}")
    ok = (event.metadata or {}).get("ok")
    if isinstance(ok, bool):
        header_parts.append(f"ok={str(ok).lower()}")
    if event.event_type == "tool_result" and event.causal_ids:
        call_ref = event_ref_by_call_id.get(event.causal_ids[0])
        if call_ref is not None:
            header_parts.append(f"responds_to={call_ref}")

    body_parts: list[str] = []
    truncated_event_count = 0
    if duplicate_ref is not None:
        canonical_ref, occurrence_count = duplicate_ref
        body_parts.append(
            "Earlier coalesced lifecycle or repeated evidence; "
            f"see latest event {canonical_ref} ({occurrence_count} related occurrences)."
        )
        truncated_event_count += 1
    elif event.content:
        rendered_content, truncated = _bounded_source_text(
            event.content,
            max_chars=content_limit,
        )
        body_parts.append(rendered_content)
        truncated_event_count += int(truncated)

    raw_calls = (event.metadata or {}).get("tool_calls")
    if isinstance(raw_calls, list):
        rendered_calls: list[str] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                continue
            call_id = _optional_string(raw_call.get("call_id"))
            name = _optional_string(raw_call.get("name")) or "unknown"
            paired_result = result_by_call_id.get(call_id or "")
            call_header = f"tool call: {name}"
            if paired_result is not None:
                call_header += f" -> result {paired_result[0]}"
            arguments, arguments_truncated = _render_tool_arguments(
                raw_call.get("arguments", {}),
                paired_result_content=(
                    paired_result[1] if paired_result is not None else None
                ),
                max_string_chars=content_limit,
            )
            if arguments:
                call_header += "\narguments: " + arguments
            rendered_calls.append(call_header)
            truncated_event_count += int(arguments_truncated)
        if rendered_calls:
            body_parts.append("\n".join(rendered_calls))

    metadata_notes = _render_event_metadata_notes(event.metadata)
    if metadata_notes:
        body_parts.append(metadata_notes)
    block = (
        "[EVENT "
        + " | ".join(header_parts)
        + "]\n"
        + ("\n".join(body_parts) or "(no visible content)")
        + f"\n[/EVENT {source_ref}]"
    )
    return block, truncated_event_count


def _source_event_requires_exact_render(event: CompactionSourceEvent) -> bool:
    _ = event
    return False


def _duplicate_system_event_refs(
    source_events: Sequence[CompactionSourceEvent],
) -> dict[int, tuple[str, int]]:
    occurrences: dict[tuple[str, str], list[int]] = {}
    for index, event in enumerate(source_events, start=1):
        metadata = event.metadata or {}
        key: tuple[str, str] | None = None
        if event.role == "system" and event.content.strip():
            if bool(metadata.get("task_contract")):
                revision = _optional_string(metadata.get("task_contract_revision"))
                key = (
                    "task_contract",
                    revision or _sha256_text(event.content),
                )
            elif event.event_type == "system_event":
                key = ("system_event", _sha256_text(event.content))
        if bool(metadata.get("bash_job_progress_update")):
            job_id_set: set[str] = set()
            for field in ("bash_job_running_ids", "bash_job_terminal_ids"):
                raw_job_ids = metadata.get(field)
                if not isinstance(raw_job_ids, list):
                    continue
                job_id_set.update(
                    str(item).strip() for item in raw_job_ids if str(item).strip()
                )
            job_ids = tuple(sorted(job_id_set))
            if job_ids:
                key = ("bash_lifecycle", _canonical_json(job_ids))
        error_code = _optional_string(metadata.get("error_code"))
        if error_code in {
            "workspace_lease_conflict",
            "blocked_repeated_tool_call",
            "orchestrator_wait_required",
        }:
            key = (
                "repeated_failure",
                error_code
                + ":"
                + (
                    _optional_string(metadata.get("conflict_key"))
                    or _optional_string(metadata.get("conflict_class"))
                    or "default"
                ),
            )
        if key is None:
            continue
        occurrences.setdefault(key, []).append(index)

    duplicate_refs: dict[int, tuple[str, int]] = {}
    for indices in occurrences.values():
        if len(indices) < 2:
            continue
        canonical_index = indices[-1]
        for index in indices[:-1]:
            duplicate_refs[index] = (f"E{canonical_index}", len(indices))
    return duplicate_refs


def _render_tool_arguments(
    value: Any,
    *,
    paired_result_content: str | None,
    max_string_chars: int | None,
) -> tuple[str, bool]:
    truncated = False

    def compact(item: Any) -> Any:
        nonlocal truncated
        if isinstance(item, str):
            if (
                paired_result_content is not None
                and len(item) >= 80
                and item in paired_result_content
            ):
                return "<exact value appears in paired tool result>"
            bounded, was_truncated = _bounded_source_text(
                item,
                max_chars=max_string_chars,
            )
            truncated = truncated or was_truncated
            return bounded
        if isinstance(item, Mapping):
            return {str(key): compact(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [compact(child) for child in item]
        return _normalize_json_value(item)

    normalized = compact(value)
    if normalized in ({}, [], None, ""):
        return "", truncated
    rendered = _canonical_json(normalized)
    rendered, whole_truncated = _bounded_source_text(
        rendered,
        max_chars=max_string_chars,
    )
    return rendered, truncated or whole_truncated


def _render_event_metadata_notes(metadata: Mapping[str, Any] | None) -> str:
    if not metadata:
        return ""
    notes = {
        key: metadata[key]
        for key in (
            "reason",
            "approved",
            "interruption_reason",
            "recommended_action",
            "latest_subagent_report_complete",
            "latest_subagent_report_truncated",
            "pending_subagent_ids",
            "carry_forward_compacted",
            "carry_forward_compaction_strength",
        )
        if key in metadata
    }
    return "metadata: " + _canonical_json(notes) if notes else ""


def _bounded_source_text(text: str, *, max_chars: int | None) -> tuple[str, bool]:
    if max_chars is None or len(text) <= max_chars:
        return text, False
    marker = (
        f"\n...[{len(text) - max_chars} source characters omitted; "
        f"sha256={_sha256_text(text)}]...\n"
    )
    remaining = max(0, max_chars - len(marker))
    leading = remaining // 2
    trailing = remaining - leading
    suffix = text[-trailing:] if trailing else ""
    return text[:leading] + marker + suffix, True


def _extract_compaction_submission(response: LLMResponse) -> dict[str, Any]:
    matching_calls = [
        call for call in response.tool_calls if call.name == _COMPACTION_TOOL_NAME
    ]
    if len(matching_calls) == 1:
        return dict(matching_calls[0].arguments)
    if matching_calls:
        raise ValueError("Compaction model submitted more than one compaction tool call.")
    if response.tool_calls:
        names = ", ".join(call.name for call in response.tool_calls)
        raise ValueError(f"Compaction model called unexpected tools: {names}.")
    return _parse_json_object(response.text)


def _parse_json_object(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if not normalized:
        raise ValueError("Compaction model returned empty output.")
    candidates = [normalized]
    fenced = _strip_json_fence(normalized)
    if fenced != normalized:
        candidates.append(fenced)
    extracted = _extract_first_json_object(normalized)
    if extracted is not None and extracted not in candidates:
        candidates.append(extracted)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Compaction model did not return a valid JSON object.")


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _trace_response(
    response: LLMResponse,
    *,
    phase: Literal["generate"],
) -> CompactionCallTrace:
    usage = response.usage
    return CompactionCallTrace(
        phase=phase,
        provider=response.provider,
        model=response.model,
        response_id=response.response_id,
        input_tokens=usage.input_tokens if usage is not None else None,
        output_tokens=usage.output_tokens if usage is not None else None,
        total_tokens=usage.total_tokens if usage is not None else None,
    )


def _issues_from_exception(exc: Exception) -> tuple[CompactionContractIssue, ...]:
    if isinstance(exc, CompactionContractError):
        return exc.issues
    if isinstance(exc, TimeoutError):
        code = "semantic_refresh_timeout"
    elif isinstance(exc, ContextBudgetError):
        code = "semantic_input_budget_exceeded"
    elif isinstance(exc, (ValueError, json.JSONDecodeError)):
        code = "invalid_compaction_json"
    else:
        code = "semantic_provider_failure"
    return (
        CompactionContractIssue(
            code=code,
            message=str(exc) or exc.__class__.__name__,
        ),
    )


def _resolve_compaction_deadline_seconds(
    llm_service: object,
    *,
    explicit_deadline_seconds: float | None,
) -> float | None:
    if explicit_deadline_seconds is not None:
        if explicit_deadline_seconds <= 0:
            raise ValueError("Compaction deadline must be greater than zero.")
        return float(explicit_deadline_seconds)

    service_settings = getattr(llm_service, "settings", None)
    configured_deadline = getattr(service_settings, "request_deadline_seconds", None)
    if configured_deadline is None:
        return None
    try:
        normalized = float(configured_deadline)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _update_call_trace_diagnostics(
    diagnostics: dict[str, Any],
    call_traces: Sequence[CompactionCallTrace],
) -> None:
    diagnostics["compaction_call_count"] = len(call_traces)
    diagnostics["compaction_call_traces"] = [trace.to_dict() for trace in call_traces]


def _log_completed_call(trace: CompactionCallTrace) -> None:
    LOGGER.info(
        "Compaction %s response completed "
        "(provider=%s, model=%s, response_id=%s, input_tokens=%s, "
        "output_tokens=%s, total_tokens=%s).",
        trace.phase,
        trace.provider,
        trace.model,
        trace.response_id,
        trace.input_tokens,
        trace.output_tokens,
        trace.total_tokens,
    )


def _attach_compaction_error_metadata(
    exc: Exception,
    diagnostics: Mapping[str, Any],
) -> None:
    existing = getattr(exc, "metadata", None)
    merged = dict(existing) if isinstance(existing, Mapping) else {}
    for key, value in diagnostics.items():
        merged.setdefault(key, value)
    try:
        setattr(exc, "metadata", merged)
    except (AttributeError, TypeError):
        LOGGER.debug(
            "Could not attach compaction diagnostics to %s.",
            type(exc).__name__,
        )


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalize_json_value(item) for item in value]
    return str(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _sum_optional(values: Iterable[int | None]) -> int | None:
    normalized = list(values)
    if not normalized or any(value is None for value in normalized):
        return None
    return sum(value for value in normalized if value is not None)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
