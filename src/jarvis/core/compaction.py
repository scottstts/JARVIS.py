"""Compaction orchestration for long-running session contexts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Sequence

from jarvis.llm import LLMMessage, LLMRequest, LLMService
from jarvis.storage import ConversationRecord

from .config import ContextPolicySettings

_COMPACTION_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "COMPACTION.md"
_COMPACTION_SYSTEM_PROMPT = _COMPACTION_PROMPT_PATH.read_text(encoding="utf-8").strip()

CompactionRole = Literal["system", "user", "assistant"]
CompactionKind = Literal[
    "session_frame",
    "preserved_message",
    "condensed_span",
    "handover_state",
]

_ALLOWED_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})
_ALLOWED_KINDS: frozenset[str] = frozenset(
    {"session_frame", "preserved_message", "condensed_span", "handover_state"}
)
_TURN_CONTEXT_PREFIX = "System context auto-appended for this turn only."
_SUBAGENT_STATUS_PREFIX = "Subagent status snapshot:"


@dataclass(slots=True, frozen=True)
class CompactionSourceRange:
    start: int
    end: int

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(slots=True, frozen=True)
class CompactionReplacementItem:
    role: CompactionRole
    kind: CompactionKind
    content: str
    verbatim: bool = False
    source_record_ids: tuple[str, ...] = ()
    source_range: CompactionSourceRange | None = None
    type: Literal["compaction"] = "compaction"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "role": self.role,
            "kind": self.kind,
            "content": self.content,
        }
        if self.verbatim:
            payload["verbatim"] = True
        if self.source_record_ids:
            payload["source_record_ids"] = list(self.source_record_ids)
        if self.source_range is not None:
            payload["source_range"] = self.source_range.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompactionReplacementItem":
        item_type = str(payload.get("type", "")).strip()
        if item_type != "compaction":
            raise ValueError("Compaction item type must be 'compaction'.")

        role = str(payload.get("role", "")).strip()
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"Unsupported compaction role: {role}")

        kind = str(payload.get("kind", "")).strip()
        if kind not in _ALLOWED_KINDS:
            raise ValueError(f"Unsupported compaction kind: {kind}")

        content = str(payload.get("content", "")).strip()
        if not content:
            raise ValueError("Compaction item content must not be blank.")

        verbatim = bool(payload.get("verbatim", False))
        if kind == "preserved_message" and not verbatim:
            raise ValueError("Compaction preserved_message items must set verbatim=true.")

        source_record_ids = _normalize_source_record_ids(payload.get("source_record_ids"))
        source_range = _normalize_source_range(payload.get("source_range"))
        return cls(
            role=role,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            content=content,
            verbatim=verbatim,
            source_record_ids=source_record_ids,
            source_range=source_range,
        )

    def record_metadata(self, *, generation: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "type": "compaction",
            "compaction_item": True,
            "compaction_kind": self.kind,
            "verbatim": self.verbatim,
            "compaction_generation": generation,
        }
        if self.source_record_ids:
            metadata["source_record_ids"] = list(self.source_record_ids)
        if self.source_range is not None:
            metadata["source_range"] = self.source_range.to_dict()
        return metadata


@dataclass(slots=True, frozen=True)
class CompactionOutcome:
    items: tuple[CompactionReplacementItem, ...]
    response_payload: dict[str, Any]
    model: str
    provider: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    response_id: str | None


class ContextCompactor:
    """Builds compacted carry-forward replacement history from prior transcript records."""

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
        user_instruction: str | None = None,
    ) -> CompactionOutcome:
        source_records = prune_compaction_source_records(records)
        if not source_records:
            raise ValueError("Compaction source is empty after pruning.")

        source_jsonl = _serialize_source_records(source_records)
        instruction = user_instruction.strip() if user_instruction else ""
        user_prompt = (
            "Compact the following transcript items into replacement history JSON.\n\n"
            "Additional user instruction for this compaction:\n"
            f"{instruction or 'None'}\n\n"
            "Transcript items (JSONL):\n"
            f"{source_jsonl}"
        )
        request = LLMRequest(
            messages=(
                LLMMessage.text("system", _COMPACTION_SYSTEM_PROMPT),
                LLMMessage.text("user", user_prompt),
            ),
            provider=self._provider,
            max_output_tokens=self._context_policy.compact_reserve_output_tokens,
        )
        response = await self._llm_service.generate(request)

        response_payload = _parse_compaction_response_payload(response.text)
        items = _post_prune_compaction_items(response_payload)
        usage = response.usage
        return CompactionOutcome(
            items=items,
            response_payload=response_payload,
            model=response.model,
            provider=response.provider,
            input_tokens=usage.input_tokens if usage is not None else None,
            output_tokens=usage.output_tokens if usage is not None else None,
            total_tokens=usage.total_tokens if usage is not None else None,
            response_id=response.response_id,
        )


def prune_compaction_source_records(
    records: Sequence[ConversationRecord],
) -> tuple[ConversationRecord, ...]:
    return tuple(record for record in records if not _should_drop_source_record(record))


def _should_drop_source_record(record: ConversationRecord) -> bool:
    metadata = record.metadata
    if record.kind == "compaction":
        return True
    if metadata.get("bootstrap_identity"):
        return True
    if metadata.get("transcript_only"):
        return True
    if metadata.get("memory_bootstrap"):
        return True
    if metadata.get("summary_seed"):
        return True
    if metadata.get("turn_context") == "datetime":
        return True
    if metadata.get("subagent_status_snapshot"):
        return True
    if metadata.get("tool_call_validation_failed"):
        return True
    if record.role == "assistant" and not record.content.strip():
        return True
    if record.content.startswith(_TURN_CONTEXT_PREFIX):
        return True
    if record.content.startswith(_SUBAGENT_STATUS_PREFIX) and record.role == "system":
        return True
    return False


def _serialize_source_records(records: Sequence[ConversationRecord]) -> str:
    lines: list[str] = []
    for index, record in enumerate(records, start=1):
        payload: dict[str, Any] = {
            "index": index,
            "record_id": record.record_id,
            "role": record.role,
            "kind": record.kind,
            "content": record.content,
        }
        if record.metadata:
            payload["metadata"] = _normalize_json_value(record.metadata)
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)


def _parse_compaction_response_payload(text: str) -> dict[str, Any]:
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
    raise ValueError("Compaction model did not return valid JSON.")


def _post_prune_compaction_items(payload: dict[str, Any]) -> tuple[CompactionReplacementItem, ...]:
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Compaction response must contain a non-empty 'items' list.")

    items: list[CompactionReplacementItem] = []
    previous_signature: tuple[Any, ...] | None = None
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        try:
            item = CompactionReplacementItem.from_dict(raw_item)
        except ValueError:
            continue
        if _looks_like_transient_boilerplate(item):
            continue
        signature = (
            item.type,
            item.role,
            item.kind,
            item.content,
            item.verbatim,
            item.source_record_ids,
            item.source_range.to_dict() if item.source_range is not None else None,
        )
        if signature == previous_signature:
            continue
        previous_signature = signature
        items.append(item)

    if not items:
        raise ValueError("Compaction response did not contain any valid replacement items.")
    if items[0].kind != "session_frame" or items[0].role != "system":
        raise ValueError("Compaction response must start with a system session_frame item.")
    if items[-1].kind != "handover_state" or items[-1].role != "system":
        raise ValueError("Compaction response must end with a system handover_state item.")
    return tuple(items)


def _looks_like_transient_boilerplate(item: CompactionReplacementItem) -> bool:
    content = item.content.strip()
    if not content:
        return True
    if content.startswith(_TURN_CONTEXT_PREFIX):
        return True
    if content.startswith(_SUBAGENT_STATUS_PREFIX):
        return True
    return False


def _normalize_source_record_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    record_ids: list[str] = []
    for raw_item in value:
        item = str(raw_item).strip()
        if item:
            record_ids.append(item)
    return tuple(record_ids)


def _normalize_source_range(value: Any) -> CompactionSourceRange | None:
    if not isinstance(value, dict):
        return None
    raw_start = value.get("start")
    raw_end = value.get("end")
    try:
        start = int(raw_start)
        end = int(raw_end)
    except (TypeError, ValueError):
        return None
    if start <= 0 or end < start:
        return None
    return CompactionSourceRange(start=start, end=end)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3:
        return stripped
    if not lines[0].startswith("```"):
        return stripped
    if lines[-1].strip() != "```":
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
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _normalize_json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
