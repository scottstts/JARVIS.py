"""Tool-search tool definition and execution runtime."""

from __future__ import annotations

import json
from typing import Any, Protocol

from jarvis.llm import ToolDefinition

from ...discoverable_search import search_discoverable_entries
from ...runtime_tools import load_runtime_tool_catalog
from ...types import (
    DiscoverableTool,
    RegisteredTool,
    ToolExecutionContext,
    ToolExecutionResult,
)


class ToolSearchCatalog(Protocol):
    """Minimal registry surface used by tool_search."""

    def discoverable_entries(self) -> tuple[DiscoverableTool, ...]:
        """Return built-in discoverable catalog entries."""

    def get(self, name: str) -> RegisteredTool | None:
        """Return a registered executable tool by name."""

    def registered_tool_names(self) -> tuple[str, ...]:
        """Return all registered executable tool names."""


class ToolSearchToolExecutor:
    """Searches the discoverable-tool catalog and formats concise usage docs."""

    def __init__(self, catalog: ToolSearchCatalog) -> None:
        self._catalog = catalog

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        query = _normalize_optional_string(arguments.get("query"))
        verbosity = _normalize_verbosity(arguments.get("verbosity"))
        built_in_entries = self._catalog.discoverable_entries()
        reserved_names = {
            entry.name
            for entry in built_in_entries
        }
        reserved_names.update(self._catalog.registered_tool_names())
        runtime_catalog = load_runtime_tool_catalog(
            context.workspace_dir,
            reserved_names=reserved_names,
        )
        merged_entries = tuple(built_in_entries) + runtime_catalog.entries
        matches = search_discoverable_entries(merged_entries, query or "")
        activated_discoverable_tool_names = [
            entry.name
            for entry in matches
            if verbosity == "high" and entry.backing_tool_name is not None
        ]

        content = _format_tool_search_result(
            query=query,
            verbosity=verbosity,
            matches=matches,
        )
        metadata = {
            "query": query,
            "verbosity": verbosity,
            "match_count": len(matches),
            "matches": [_serialize_discoverable_entry(entry) for entry in matches],
            "activated_discoverable_tool_names": activated_discoverable_tool_names,
            "runtime_tool_errors": list(runtime_catalog.errors),
        }
        return ToolExecutionResult(
            call_id=call_id,
            name="tool_search",
            ok=True,
            content=content,
            metadata=metadata,
        )


def build_tool_search_tool(catalog: ToolSearchCatalog) -> RegisteredTool:
    """Build the tool_search registry entry."""

    return RegisteredTool(
        name="tool_search",
        exposure="basic",
        definition=ToolDefinition(
            name="tool_search",
            description=(
                "Search discoverable tools that are hidden by default. "
                "Prefer starting with verbosity='low' so you only get tool names and one-line "
                "purposes. Use verbosity='high' only after you have narrowed down promising targets. "
                "If query is omitted or empty, this lists every discoverable tool at low "
                "verbosity by default."
                "For tasks that require tools, you **MUST ALWAYS** first use `tool_search` tool "
                "to find the best suitable tool for the task before you start the task."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Optional discoverable-tool search query. Omit or pass an empty "
                            "string to list all discoverable tools."
                        ),
                    },
                    "verbosity": {
                        "type": "string",
                        "enum": ["low", "high"],
                        "description": (
                            "Optional verbosity level. Defaults to 'low'. Use 'high' only on "
                            "narrowed targets."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        ),
        executor=ToolSearchToolExecutor(catalog),
    )


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_verbosity(value: Any) -> str:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return "low"
    lowered = normalized.lower()
    if lowered in {"low", "high"}:
        return lowered
    return "low"


def _serialize_discoverable_entry(entry: DiscoverableTool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": entry.name,
        "purpose": entry.purpose,
    }
    if entry.aliases:
        payload["aliases"] = list(entry.aliases)
    if entry.detailed_description is not None:
        payload["detailed_description"] = entry.detailed_description
    if entry.usage is not None:
        payload["usage"] = entry.usage
    if entry.metadata:
        payload["metadata"] = dict(entry.metadata)
    if entry.backing_tool_name is not None:
        payload["backing_tool_name"] = entry.backing_tool_name
    return payload


def _format_tool_search_result(
    *,
    query: str | None,
    verbosity: str,
    matches: tuple[DiscoverableTool, ...],
) -> str:
    content_lines = [
        "Tool search result",
        f"query: {query if query is not None else '(all discoverable tools)'}",
        f"verbosity: {verbosity}",
        f"match_count: {len(matches)}",
    ]

    if not matches:
        content_lines.append("No discoverable tools matched.")
        return "\n".join(content_lines)

    if verbosity == "low":
        content_lines.append(
            "Low verbosity returns only concise discovery hints. Use verbosity=high on a "
            "narrowed query when you need full usage details."
        )
        for index, entry in enumerate(matches, start=1):
            content_lines.extend(
                [
                    f"{index}. {entry.name}",
                    f"purpose: {entry.purpose}",
                ]
            )
        return "\n".join(content_lines)

    content_lines.append(
        "High verbosity returns the full discoverable entry and activates matched backed "
        "tools for the rest of this turn."
    )
    for index, entry in enumerate(matches, start=1):
        content_lines.extend(
            [
                f"{index}. {entry.name}",
                f"purpose: {entry.purpose}",
            ]
        )
        if entry.aliases:
            content_lines.append(f"aliases: {', '.join(entry.aliases)}")
        if entry.detailed_description is not None:
            content_lines.append(f"detailed_description: {entry.detailed_description}")
        if entry.usage is not None:
            content_lines.append("usage:")
            content_lines.append(_format_flexible_value(entry.usage))
        if entry.metadata:
            content_lines.append("metadata:")
            content_lines.append(_format_flexible_value(entry.metadata))
        if entry.backing_tool_name is not None:
            content_lines.append(f"backing_tool_name: {entry.backing_tool_name}")
    return "\n".join(content_lines)


def _format_flexible_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
