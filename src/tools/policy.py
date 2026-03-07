"""Policy checks for tool execution."""

from __future__ import annotations

from .bash import BashCommandPolicy
from .types import ToolExecutionContext, ToolPolicyDecision
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

        if tool_name not in {"bash", "view_image"}:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' is not implemented in this runtime.",
            )
        return ToolPolicyDecision(allowed=False, reason=f"Tool '{tool_name}' is not implemented.")
