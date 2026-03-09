"""Bash-specific policy implementation."""

from __future__ import annotations

from ...types import ToolExecutionContext, ToolPolicyDecision


class BashCommandPolicy:
    """Thin validation for the sandboxed bash runtime."""

    def authorize(
        self,
        *,
        command: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context

        if not command.strip():
            return ToolPolicyDecision(allowed=False, reason="bash command cannot be empty.")
        if "\x00" in command:
            return ToolPolicyDecision(
                allowed=False,
                reason="bash command cannot contain null bytes.",
            )
        return ToolPolicyDecision(allowed=True)
