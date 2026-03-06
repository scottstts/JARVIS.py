"""Execution runtime for registered tools."""

from __future__ import annotations

from llm import ToolCall

from .policy import ToolPolicy
from .registry import ToolRegistry
from .types import ToolExecutionContext, ToolExecutionResult


class ToolRuntime:
    """Runs tool calls through registry lookup and policy checks."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        policy: ToolPolicy | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy or ToolPolicy()

    async def execute(
        self,
        *,
        tool_call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        tool = self._registry.require(tool_call.name)
        decision = self._policy.authorize(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            context=context,
        )
        if not decision.allowed:
            reason = decision.reason or "Tool execution denied by policy."
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                ok=False,
                content=(
                    "Tool execution denied by policy\n"
                    f"tool: {tool_call.name}\n"
                    f"reason: {reason}"
                ),
                metadata={
                    "policy_denied": True,
                    "reason": reason,
                    "arguments": dict(tool_call.arguments),
                },
            )

        try:
            return await tool.executor(
                call_id=tool_call.call_id,
                arguments=tool_call.arguments,
                context=context,
            )
        except Exception as exc:
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                ok=False,
                content=(
                    "Tool execution failed\n"
                    f"tool: {tool_call.name}\n"
                    f"error_type: {type(exc).__name__}\n"
                    f"error: {exc}"
                ),
                metadata={
                    "execution_failed": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "arguments": dict(tool_call.arguments),
                },
            )
