"""Policy checks for the web_search tool."""

from __future__ import annotations

from ...types import ToolExecutionContext, ToolPolicyDecision

_MAX_QUERY_CHARS = 400
_MAX_QUERY_WORDS = 50


class WebSearchPolicy:
    """Restricts web_search to a single non-empty Brave-compatible query."""

    def authorize(
        self,
        *,
        query: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context
        normalized = query.strip()
        if not normalized:
            return ToolPolicyDecision(
                allowed=False,
                reason="web_search requires a non-empty 'query'.",
            )
        if len(normalized) > _MAX_QUERY_CHARS:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"web_search query length must be <= {_MAX_QUERY_CHARS} characters.",
            )
        if len(normalized.split()) > _MAX_QUERY_WORDS:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"web_search query length must be <= {_MAX_QUERY_WORDS} words.",
            )
        return ToolPolicyDecision(allowed=True)
