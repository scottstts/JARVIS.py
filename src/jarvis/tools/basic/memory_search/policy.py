"""Policy checks for the memory_search tool."""

from __future__ import annotations

from ...types import ToolExecutionContext, ToolPolicyDecision


class MemorySearchPolicy:
    """Thin input validation for memory search."""

    def authorize(
        self,
        *,
        query: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context
        if not query.strip():
            return ToolPolicyDecision(allowed=False, reason="memory_search requires a non-empty query.")
        return ToolPolicyDecision(allowed=True)

