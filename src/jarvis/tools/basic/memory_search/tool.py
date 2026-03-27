"""Memory-search tool definition and execution runtime."""

from __future__ import annotations

from typing import Any

from jarvis.llm import ToolDefinition

from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult


class MemorySearchToolExecutor:
    """Searches canonical memory through the indexed memory service."""

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
                name="memory_search",
                ok=False,
                content="Memory service is not available in this runtime.",
            )

        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolExecutionResult(
                call_id=call_id,
                name="memory_search",
                ok=False,
                content="memory_search requires a non-empty 'query'.",
            )

        raw_scopes = arguments.get("scopes")
        scopes = (
            tuple(str(item).strip() for item in raw_scopes if str(item).strip())
            if isinstance(raw_scopes, list)
            else ("core", "ongoing", "daily")
        )
        response = await service.search(
            query=query,
            mode=str(arguments.get("mode", "hybrid")).strip() or "hybrid",
            scopes=scopes if scopes else ("core", "ongoing", "daily"),
            top_k=_optional_int(arguments.get("top_k")),
            daily_lookback_days=_optional_int(arguments.get("daily_lookback_days")),
            expand=_optional_int(arguments.get("expand")),
            include_expired=bool(arguments.get("include_expired", False)),
            route_id=context.route_id,
            session_id=context.session_id,
        )
        metadata = {
            "query": query,
            "warnings": list(response.warnings),
            "semantic_disabled": response.semantic_disabled,
            "results": [
                {
                    "document_id": result.document_id,
                    "title": result.title,
                    "path": str(result.path),
                    "kind": result.kind,
                    "chunk_id": result.chunk_id,
                    "section_path": result.section_path,
                    "score": result.score,
                    "snippet": result.snippet,
                    "match_reasons": list(result.match_reasons),
                    "source_ref_ids": list(result.source_ref_ids),
                    "semantic_disabled": result.semantic_disabled,
                }
                for result in response.results
            ],
        }
        return ToolExecutionResult(
            call_id=call_id,
            name="memory_search",
            ok=True,
            content=_format_memory_search_result(
                query=query,
                results=metadata["results"],
                warnings=metadata["warnings"],
                semantic_disabled=response.semantic_disabled,
            ),
            metadata=metadata,
        )


def build_memory_search_tool() -> RegisteredTool:
    return RegisteredTool(
        name="memory_search",
        exposure="basic",
        definition=ToolDefinition(
            name="memory_search",
            description=(
                "Search canonical runtime memory using lexical, semantic, graph, or hybrid retrieval. "
                "Use this before opening or mutating memory documents."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "lexical", "semantic", "graph", "hybrid"],
                    },
                    "scopes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["core", "ongoing", "daily", "archive"],
                        },
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    "daily_lookback_days": {"type": "integer", "minimum": 1, "maximum": 365},
                    "expand": {"type": "integer", "enum": [0, 1, 2]},
                    "include_expired": {"type": "boolean"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        executor=MemorySearchToolExecutor(),
    )


def _format_memory_search_result(
    *,
    query: str,
    results: list[dict[str, Any]],
    warnings: list[str],
    semantic_disabled: bool,
) -> str:
    lines = [
        "Memory search result",
        f"query: {query}",
        f"match_count: {len(results)}",
    ]
    if semantic_disabled:
        lines.append("semantic_disabled: true")
    for warning in warnings:
        lines.append(f"sys_warning: {warning}")
    if not results:
        lines.append("No memory matched.")
        return "\n".join(lines)
    for index, result in enumerate(results, start=1):
        section_label = (
            f"{result['section_path']} (synthetic)"
            if result["section_path"] in {"facts", "relations"}
            else result["section_path"]
        )
        lines.extend(
            [
                f"{index}. {result['document_id']}",
                f"title: {result['title']}",
                f"path: {result['path']}",
                f"kind: {result['kind']}",
                f"section: {section_label}",
                f"score: {result['score']:.4f}",
                f"match_reasons: {', '.join(result['match_reasons'])}",
                f"snippet: {result['snippet']}",
            ]
        )
    lines.append("next_step: use memory_get(document_id=..., section_path=...) before relying on a snippet")
    return "\n".join(lines)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
