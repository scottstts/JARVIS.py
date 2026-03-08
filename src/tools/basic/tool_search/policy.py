"""Policy checks for the tool_search tool."""

from __future__ import annotations

from ...types import ToolExecutionContext, ToolPolicyDecision

_MAX_QUERY_CHARS = 200
_MAX_QUERY_WORDS = 24
_ALLOWED_VERBOSITY = {"low", "high"}


class ToolSearchPolicy:
    """Restricts tool_search to short optional queries and two verbosity levels."""

    def authorize(
        self,
        *,
        query: str | None,
        verbosity: str | None,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context

        if query is not None:
            normalized_query = query.strip()
            if len(normalized_query) > _MAX_QUERY_CHARS:
                return ToolPolicyDecision(
                    allowed=False,
                    reason=(
                        "tool_search query length must be <= "
                        f"{_MAX_QUERY_CHARS} characters."
                    ),
                )
            if len(normalized_query.split()) > _MAX_QUERY_WORDS:
                return ToolPolicyDecision(
                    allowed=False,
                    reason=(
                        "tool_search query length must be <= "
                        f"{_MAX_QUERY_WORDS} words."
                    ),
                )

        if verbosity is not None and verbosity.strip():
            normalized_verbosity = verbosity.strip().lower()
            if normalized_verbosity not in _ALLOWED_VERBOSITY:
                return ToolPolicyDecision(
                    allowed=False,
                    reason="tool_search verbosity must be either 'low' or 'high'.",
                )

        return ToolPolicyDecision(allowed=True)
