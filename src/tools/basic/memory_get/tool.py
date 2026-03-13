"""Memory-get tool definition and execution runtime."""

from __future__ import annotations

from typing import Any

from llm import ToolDefinition

from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult


class MemoryGetToolExecutor:
    """Reads a full memory document or a specific section."""

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
                name="memory_get",
                ok=False,
                content="Memory service is not available in this runtime.",
            )
        try:
            content = await service.get_document(
                document_id=_optional_string(arguments.get("document_id")),
                path=_optional_string(arguments.get("path")),
                section_path=_optional_string(arguments.get("section_path")),
                include_frontmatter=bool(arguments.get("include_frontmatter", False)),
                include_sources=bool(arguments.get("include_sources", False)),
                route_id=context.route_id,
                session_id=context.session_id,
            )
        except Exception as exc:
            return ToolExecutionResult(
                call_id=call_id,
                name="memory_get",
                ok=False,
                content=f"memory_get failed: {exc}",
            )
        return ToolExecutionResult(
            call_id=call_id,
            name="memory_get",
            ok=True,
            content=content,
            metadata={
                "document_id": _optional_string(arguments.get("document_id")),
                "path": _optional_string(arguments.get("path")),
                "section_path": _optional_string(arguments.get("section_path")),
            },
        )


def build_memory_get_tool() -> RegisteredTool:
    return RegisteredTool(
        name="memory_get",
        exposure="basic",
        definition=ToolDefinition(
            name="memory_get",
            description="Open a full canonical memory document or one of its sections after discovery.",
            input_schema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "path": {"type": "string"},
                    "section_path": {"type": "string"},
                    "include_frontmatter": {"type": "boolean"},
                    "include_sources": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        ),
        executor=MemoryGetToolExecutor(),
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
