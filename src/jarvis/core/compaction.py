"""Evidence-backed session compaction orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence
from uuid import uuid4

from jarvis.llm import LLMMessage, LLMRequest, LLMResponse, LLMService
from jarvis.storage import ConversationRecord

from .compaction_contract import (
    CompactionBundle,
    CompactionContractError,
    CompactionContractIssue,
    CompactionReplayItem,
    CompactionSourceEvent,
    apply_compaction_draft,
    build_source_manifest,
    compile_compaction_replay,
)
from .config import ContextPolicySettings


_COMPACTION_PROMPT_PATH = Path(__file__).with_name("prompts") / "COMPACTION.md"
_COMPACTION_VERIFY_PROMPT_PATH = Path(__file__).with_name("prompts") / "COMPACTION_VERIFY.md"
_COMPACTION_SYSTEM_PROMPT = _COMPACTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
_COMPACTION_VERIFY_SYSTEM_PROMPT = _COMPACTION_VERIFY_PROMPT_PATH.read_text(
    encoding="utf-8"
).strip()
_MAX_COMPACTION_REPAIR_ATTEMPTS = 2
_MAX_VERIFICATION_ATTEMPTS = 2
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
}


@dataclass(slots=True, frozen=True)
class CompactionCallTrace:
    phase: Literal["generate", "verify", "repair"]
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
class CompactionVerification:
    valid: bool
    issues: tuple[CompactionContractIssue, ...]
    payload: dict[str, Any]


@dataclass(slots=True, frozen=True)
class CompactionOutcome:
    bundle: CompactionBundle
    items: tuple[CompactionReplayItem, ...]
    draft_payload: dict[str, Any]
    verification_payload: dict[str, Any]
    call_traces: tuple[CompactionCallTrace, ...]
    repair_count: int

    @property
    def provider(self) -> str:
        return self.call_traces[0].provider

    @property
    def model(self) -> str:
        return self.call_traces[0].model

    @property
    def response_id(self) -> str | None:
        return self.call_traces[0].response_id

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
    ) -> None:
        self._llm_service = llm_service
        self._context_policy = context_policy
        self._provider = provider

    async def compact(
        self,
        records: Sequence[ConversationRecord],
        *,
        previous_bundle: CompactionBundle | None = None,
        user_instruction: str | None = None,
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
        base_input = {
            "mode": "generate",
            "user_instruction": instruction or None,
            "previous_bundle": (
                previous_bundle.to_dict() if previous_bundle is not None else None
            ),
            "delta_events": [event.to_dict() for event in source_events],
        }

        call_traces: list[CompactionCallTrace] = []
        repair_issues: tuple[CompactionContractIssue, ...] = ()
        rejected_payload: dict[str, Any] | None = None
        rejected_raw_text: str | None = None
        final_verification: CompactionVerification | None = None
        final_draft: dict[str, Any] | None = None
        final_bundle: CompactionBundle | None = None

        for attempt in range(_MAX_COMPACTION_REPAIR_ATTEMPTS + 1):
            phase: Literal["generate", "repair"] = "generate" if attempt == 0 else "repair"
            request_input = dict(base_input)
            if attempt > 0:
                request_input.update(
                    {
                        "mode": "repair",
                        "validation_issues": [issue.to_dict() for issue in repair_issues],
                        "rejected_draft": rejected_payload,
                        "rejected_raw_text": rejected_raw_text,
                    }
                )
            response = await self._llm_service.generate(
                self._build_request(
                    system_prompt=_COMPACTION_SYSTEM_PROMPT,
                    user_text=(
                        "Generate the complete canonical compaction draft from this JSON input:\n"
                        + _canonical_json(request_input)
                    ),
                )
            )
            call_traces.append(_trace_response(response, phase=phase))
            rejected_raw_text = response.text
            draft_payload: dict[str, Any] | None = None
            try:
                draft_payload = _parse_json_object(response.text)
                bundle = apply_compaction_draft(
                    draft_payload,
                    bundle_id=uuid4().hex,
                    created_at=_utc_now_iso(),
                    source_manifest=source_manifest,
                    source_events=source_events,
                    previous_bundle=previous_bundle,
                )
            except (ValueError, CompactionContractError) as exc:
                rejected_payload = draft_payload
                repair_issues = _issues_from_exception(exc)
                if attempt >= _MAX_COMPACTION_REPAIR_ATTEMPTS:
                    raise CompactionContractError(repair_issues) from exc
                continue

            verification_input: dict[str, Any] = {
                "user_instruction": instruction or None,
                "previous_bundle": (
                    previous_bundle.to_dict() if previous_bundle is not None else None
                ),
                "delta_events": [event.to_dict() for event in source_events],
                "candidate_bundle": bundle.to_dict(),
            }
            verification: CompactionVerification | None = None
            invalid_verifier_output: str | None = None
            for verification_attempt in range(_MAX_VERIFICATION_ATTEMPTS):
                current_verification_input = dict(verification_input)
                if verification_attempt > 0:
                    current_verification_input.update(
                        {
                            "verifier_contract_retry": True,
                            "previous_invalid_verifier_output": invalid_verifier_output,
                        }
                    )
                verification_response = await self._llm_service.generate(
                    self._build_request(
                        system_prompt=_COMPACTION_VERIFY_SYSTEM_PROMPT,
                        user_text=(
                            "Verify this candidate compaction against its evidence:\n"
                            + _canonical_json(current_verification_input)
                        ),
                    )
                )
                call_traces.append(_trace_response(verification_response, phase="verify"))
                try:
                    verification = _parse_verification(verification_response.text)
                except ValueError:
                    invalid_verifier_output = verification_response.text
                    if verification_attempt + 1 >= _MAX_VERIFICATION_ATTEMPTS:
                        raise
                    continue
                break
            if verification is None:
                raise RuntimeError("Compaction verifier ended without a valid verdict.")
            if verification.valid:
                final_bundle = bundle
                final_draft = draft_payload
                final_verification = verification
                break

            rejected_payload = draft_payload
            repair_issues = verification.issues
            if attempt >= _MAX_COMPACTION_REPAIR_ATTEMPTS:
                raise CompactionContractError(repair_issues)

        if final_bundle is None or final_draft is None or final_verification is None:
            raise RuntimeError("Compaction ended without a verified canonical bundle.")
        return CompactionOutcome(
            bundle=final_bundle,
            items=compile_compaction_replay(final_bundle),
            draft_payload=final_draft,
            verification_payload=final_verification.payload,
            call_traces=tuple(call_traces),
            repair_count=sum(trace.phase == "repair" for trace in call_traces),
        )

    def _build_request(self, *, system_prompt: str, user_text: str) -> LLMRequest:
        return LLMRequest(
            messages=(
                LLMMessage.text("system", system_prompt),
                LLMMessage.text("user", user_text),
            ),
            provider=self._provider,
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


def _parse_verification(text: str) -> CompactionVerification:
    payload = _parse_json_object(text)
    if set(payload) != {"valid", "issues"}:
        raise ValueError("Compaction verifier must return exactly valid and issues.")
    valid = payload.get("valid")
    raw_issues = payload.get("issues")
    if not isinstance(valid, bool) or not isinstance(raw_issues, list):
        raise ValueError("Compaction verifier returned invalid field types.")
    issues: list[CompactionContractIssue] = []
    for index, raw_issue in enumerate(raw_issues):
        if not isinstance(raw_issue, Mapping) or set(raw_issue) != {
            "code",
            "message",
            "source_event_ids",
        }:
            raise ValueError(f"Compaction verifier issue {index} has invalid shape.")
        code = _optional_string(raw_issue.get("code"))
        message = _optional_string(raw_issue.get("message"))
        raw_source_ids = raw_issue.get("source_event_ids")
        if code is None or message is None or not isinstance(raw_source_ids, list):
            raise ValueError(f"Compaction verifier issue {index} has invalid values.")
        source_ids = tuple(
            source_id
            for raw_source_id in raw_source_ids
            if (source_id := _optional_string(raw_source_id)) is not None
        )
        issues.append(
            CompactionContractIssue(
                code=code,
                message=message,
                source_event_ids=source_ids,
            )
        )
    if valid and issues:
        raise ValueError("A valid compaction verification cannot contain issues.")
    if not valid and not issues:
        raise ValueError("An invalid compaction verification must contain issues.")
    return CompactionVerification(valid=valid, issues=tuple(issues), payload=payload)


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
    phase: Literal["generate", "verify", "repair"],
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
    return (
        CompactionContractIssue(
            code="invalid_compaction_json",
            message=str(exc) or exc.__class__.__name__,
        ),
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
