"""Policy checks for tool execution."""

from __future__ import annotations

from .bash import BashCommandPolicy
from .types import ToolExecutionContext, ToolPolicyDecision


class ToolPolicy:
    """Universal tool policy interface and router."""

    def authorize(
        self,
        *,
        tool_name: str,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        if tool_name != "bash":
            return ToolPolicyDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' is not implemented in this runtime.",
            )

        command = str(arguments.get("command", "")).strip()
        if not command:
            return ToolPolicyDecision(allowed=False, reason="bash command cannot be empty.")

        return BashCommandPolicy().authorize(command=command, context=context)
