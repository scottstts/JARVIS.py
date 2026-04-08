"""Memory-write tool definition and execution runtime."""

from __future__ import annotations

from typing import Any

from jarvis.llm import ToolDefinition

from .contract import (
    BODY_SECTIONS_SCHEMA,
    FACT_ITEM_SCHEMA,
    RELATION_ITEM_SCHEMA,
    format_memory_write_contract_error,
    validate_memory_write_contract,
)
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult


class MemoryWriteToolExecutor:
    """Creates or updates canonical memory documents through the memory service."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        service = context.memory_service
        if service is None:
            return ToolExecutionResult(
                call_id=call_id,
                name="memory_write",
                ok=False,
                content="Memory service is not available in this runtime.",
            )
        contract_errors = validate_memory_write_contract(
            operation=str(arguments.get("operation", "")).strip(),
            target_kind=str(arguments.get("target_kind", "")).strip(),
            arguments=arguments,
        )
        if contract_errors:
            return ToolExecutionResult(
                call_id=call_id,
                name="memory_write",
                ok=False,
                content=(
                    "memory_write failed: "
                    + format_memory_write_contract_error(
                        operation=str(arguments.get("operation", "")).strip(),
                        target_kind=str(arguments.get("target_kind", "")).strip(),
                        errors=contract_errors,
                    )
                ),
            )
        try:
            result = await service.write(
                operation=str(arguments.get("operation", "")).strip(),
                target_kind=str(arguments.get("target_kind", "")).strip(),
                document_id=_optional_string(arguments.get("document_id")),
                title=_optional_string(arguments.get("title")),
                summary=_optional_string(arguments.get("summary")),
                priority=_optional_int(arguments.get("priority")),
                pinned=_optional_bool(arguments.get("pinned")),
                locked=_optional_bool(arguments.get("locked")),
                review_after=_optional_string(arguments.get("review_after")),
                expires_at=_optional_string(arguments.get("expires_at")),
                tags=_coerce_list_of_strings(arguments.get("tags")),
                aliases=_coerce_list_of_strings(arguments.get("aliases")),
                facts=_coerce_truth_list(arguments.get("facts"), field_name="facts"),
                relations=_coerce_truth_list(arguments.get("relations"), field_name="relations"),
                body_sections=_coerce_dict(arguments.get("body_sections")),
                source_refs=_coerce_list_of_dicts(arguments.get("source_refs")),
                entity_refs=_coerce_list_of_dicts(arguments.get("entity_refs")),
                completion_criteria=_coerce_list_of_strings(arguments.get("completion_criteria")),
                route_id=context.route_id,
                session_id=context.session_id,
                date=_optional_string(arguments.get("date")),
                timezone_name=_optional_string(arguments.get("timezone")),
                close_reason=_optional_string(arguments.get("close_reason")),
            )
        except Exception as exc:
            return ToolExecutionResult(
                call_id=call_id,
                name="memory_write",
                ok=False,
                content=f"memory_write failed: {exc}",
            )
        metadata = {
            "document_id": result.document_id,
            "path": str(result.path),
            "changed_paths": [str(path) for path in result.changed_paths],
            "operation": result.operation,
        }
        return ToolExecutionResult(
            call_id=call_id,
            name="memory_write",
            ok=True,
            content=(
                "Memory write applied\n"
                f"operation: {result.operation}\n"
                f"document_id: {result.document_id}\n"
                f"path: {result.path}\n"
                f"summary: {result.summary}"
            ),
            metadata=metadata,
        )


def build_memory_write_tool() -> RegisteredTool:
    return RegisteredTool(
        name="memory_write",
        exposure="basic",
        definition=ToolDefinition(
            name="memory_write",
            description=(
                "Create or update canonical memory documents through validated structured operations. "
                'For core and ongoing create/upsert, facts and relations are explicit-decision fields: always set both as a '
                'non-empty structured array or the literal string "None"; summary is not a substitute. '
                "Put durable facts in facts, structured subject-predicate-object claims in relations, and important "
                "narrative text in body_sections. "
                "For close and archive superseding transitions, rewrite the terminal summary and body_sections before "
                "flipping state or the system will apply a generic fallback."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "upsert", "append_daily", "close", "archive", "promote", "demote"],
                        "description": (
                            "Structured memory mutation. append_daily records daily log content. "
                            "close and archive are superseding transitions: rewrite terminal summary/body_sections first, "
                            "then flip state."
                        ),
                    },
                    "target_kind": {
                        "type": "string",
                        "enum": ["core", "ongoing", "daily"],
                    },
                    "document_id": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {
                        "type": "string",
                        "description": (
                            "Short summary text. For append_daily without body_sections, it is recorded under Notable Events. "
                            "For core/ongoing create and upsert, summary is not a substitute for structured truth."
                        ),
                    },
                    "priority": {"type": "integer", "minimum": 0, "maximum": 100},
                    "pinned": {"type": "boolean"},
                    "locked": {"type": "boolean"},
                    "review_after": {"type": "string"},
                    "expires_at": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "facts": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": FACT_ITEM_SCHEMA,
                                "minItems": 1,
                            },
                            {
                                "type": "string",
                            },
                        ],
                        "description": (
                            'Structured durable fact objects from explicit user statements. Minimal item: {"text":"..."}.'
                        ),
                    },
                    "relations": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": RELATION_ITEM_SCHEMA,
                                "minItems": 1,
                            },
                            {
                                "type": "string",
                            },
                        ],
                        "description": (
                            "Structured subject-predicate-object claims such as preferences, tool usage, ownership, or "
                            'responsibilities. Minimal item: {"subject":"...","predicate":"...","object":"..."}.'
                        ),
                    },
                    "body_sections": {
                        **BODY_SECTIONS_SCHEMA,
                        "description": (
                            'Canonical narrative sections keyed by section name. This is the main searchable body text. '
                            'Pass an object like {"Overview":"..."}.'
                        ),
                    },
                    "source_refs": {"type": "array", "items": {"type": "object"}},
                    "entity_refs": {"type": "array", "items": {"type": "object"}},
                    "completion_criteria": {"type": "array", "items": {"type": "string"}},
                    "date": {"type": "string"},
                    "timezone": {"type": "string"},
                    "close_reason": {
                        "type": "string",
                        "description": "Optional close/archive reason metadata.",
                    },
                },
                "required": ["operation", "target_kind"],
                "additionalProperties": False,
            },
            strict=False,
        ),
        executor=MemoryWriteToolExecutor(),
    )


def _coerce_list_of_dicts(value: Any) -> list[dict[str, Any]] | None:
    if value is None or not isinstance(value, list):
        return None
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(dict(item))
    return result


def _coerce_truth_list(value: Any, *, field_name: str) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "none":
            return []
        raise ValueError(
            f'{field_name} must be a structured array or the literal string "None".'
        )
    if not isinstance(value, list):
        raise ValueError(
            f'{field_name} must be a structured array or the literal string "None".'
        )
    if not value:
        raise ValueError(
            f'{field_name} cannot be an empty array; use the literal string "None" instead.'
        )
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{field_name} array items must be objects.")
        result.append(dict(item))
    return result


def _coerce_dict(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("body_sections must be an object keyed by section name with string values.")
    result: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("body_sections section names must be non-empty strings.")
        if not isinstance(item, str):
            raise ValueError("body_sections values must be strings keyed by section name.")
        result[normalized_key] = item
    return result


def _coerce_list_of_strings(value: Any) -> list[str] | None:
    if value is None or not isinstance(value, list):
        return None
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return None
