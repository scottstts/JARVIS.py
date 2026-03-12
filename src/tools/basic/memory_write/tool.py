"""Memory-write tool definition and execution runtime."""

from __future__ import annotations

from typing import Any

from llm import ToolDefinition

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
                facts=_coerce_list_of_dicts(arguments.get("facts")),
                relations=_coerce_list_of_dicts(arguments.get("relations")),
                body_sections=_coerce_dict(arguments.get("body_sections")),
                source_refs=_coerce_list_of_dicts(arguments.get("source_refs")),
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
                "Prefer this over generic file editing for normal memory mutations."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "upsert", "append_daily", "close", "archive", "promote", "demote"],
                    },
                    "target_kind": {
                        "type": "string",
                        "enum": ["core", "ongoing", "daily"],
                    },
                    "document_id": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "priority": {"type": "integer", "minimum": 0, "maximum": 100},
                    "pinned": {"type": "boolean"},
                    "locked": {"type": "boolean"},
                    "review_after": {"type": "string"},
                    "expires_at": {"type": "string"},
                    "facts": {"type": "array", "items": {"type": "object"}},
                    "relations": {"type": "array", "items": {"type": "object"}},
                    "body_sections": {"type": "object", "additionalProperties": {"type": "string"}},
                    "source_refs": {"type": "array", "items": {"type": "object"}},
                    "date": {"type": "string"},
                    "timezone": {"type": "string"},
                    "close_reason": {"type": "string"},
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


def _coerce_dict(value: Any) -> dict[str, str] | None:
    if value is None or not isinstance(value, dict):
        return None
    return {str(key): str(item) for key, item in value.items()}


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
