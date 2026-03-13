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
                tags=_coerce_list_of_strings(arguments.get("tags")),
                aliases=_coerce_list_of_strings(arguments.get("aliases")),
                facts=_coerce_list_of_dicts(arguments.get("facts")),
                relations=_coerce_list_of_dicts(arguments.get("relations")),
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
                "Prefer this over generic file editing for normal memory mutations. "
                "For core and ongoing memory, keep important narrative text in body sections, not only in frontmatter summary. "
                "When the user states an explicit durable fact, pass it in facts; when they state a structured relationship, "
                "current preference, tool usage, ownership, or other subject-predicate-object claim, pass it in relations. "
                "When superseding memory through close or archive, first rewrite the memory content to the new terminal truth "
                "using summary and body_sections, then let the operation flip status/archive state. "
                "If you omit that rewrite, the system will apply a generic fallback terminal stamp."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "upsert", "append_daily", "close", "archive", "promote", "demote"],
                        "description": (
                            "Structured memory mutation. append_daily records daily log content; "
                            "close ends an ongoing memory and removes it from the active set; "
                            "close and archive are superseding transitions, so rewrite the memory's canonical content first "
                            "with summary and body_sections before flipping status/archive state."
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
                            "Short summary text. For append_daily, summary-only writes are recorded "
                            "under Notable Events when body_sections are omitted. "
                            "For close/archive superseding transitions, provide the rewritten terminal summary here instead of leaving the old active wording."
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
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "Explicit fact statements to track as structured memory. "
                            "Use this when the user gives a durable fact directly instead of burying that fact only in summary/body text."
                        ),
                    },
                    "relations": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "Structured subject-predicate-object claims. "
                            "Use this for stated preferences, current tools or stacks, ownership, responsibilities, and other truth-tracked relationships."
                        ),
                    },
                    "body_sections": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": (
                            "Canonical section content keyed by section name. This is the main searchable "
                            "narrative text of the memory document. "
                            "For close/archive superseding transitions, pass the rewritten terminal body here; "
                            "do not rely on status flip alone to supersede stale content."
                        ),
                    },
                    "source_refs": {"type": "array", "items": {"type": "object"}},
                    "entity_refs": {"type": "array", "items": {"type": "object"}},
                    "completion_criteria": {"type": "array", "items": {"type": "string"}},
                    "date": {"type": "string"},
                    "timezone": {"type": "string"},
                    "close_reason": {
                        "type": "string",
                        "description": (
                            "Reason for closing or archiving. This is metadata and fallback material, not a substitute "
                            "for rewriting summary/body_sections during superseding transitions."
                        ),
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


def _coerce_dict(value: Any) -> dict[str, str] | None:
    if value is None or not isinstance(value, dict):
        return None
    return {str(key): str(item) for key, item in value.items()}


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
