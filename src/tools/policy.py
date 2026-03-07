"""Policy checks for tool execution."""

from __future__ import annotations

from .bash import BashCommandPolicy
from .send_file import SendFilePolicy
from .types import ToolExecutionContext, ToolPolicyDecision
from .web_fetch import WebFetchPolicy
from .web_search import WebSearchPolicy
from .view_image import ViewImagePolicy


class ToolPolicy:
    """Universal tool policy interface and router."""

    def authorize(
        self,
        *,
        tool_name: str,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        if tool_name == "bash":
            command = str(arguments.get("command", "")).strip()
            if not command:
                return ToolPolicyDecision(allowed=False, reason="bash command cannot be empty.")

            return BashCommandPolicy().authorize(command=command, context=context)

        if tool_name == "view_image":
            path = str(arguments.get("path", "")).strip()
            return ViewImagePolicy().authorize(path=path, context=context)

        if tool_name == "send_file":
            path = str(arguments.get("path", "")).strip()
            return SendFilePolicy().authorize(path=path, context=context)

        if tool_name == "web_search":
            query = str(arguments.get("query", "")).strip()
            return WebSearchPolicy().authorize(query=query, context=context)

        if tool_name == "web_fetch":
            url = str(arguments.get("url", "")).strip()
            return WebFetchPolicy().authorize(url=url, context=context)

        if tool_name not in {"bash", "web_search", "web_fetch", "view_image", "send_file"}:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' is not implemented in this runtime.",
            )
        return ToolPolicyDecision(allowed=False, reason=f"Tool '{tool_name}' is not implemented.")
