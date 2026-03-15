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
            if isinstance(decision.approval_request, dict):
                approval_request = dict(decision.approval_request)
                return ToolExecutionResult(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    ok=False,
                    content=_format_approval_required_content(
                        tool_name=tool_call.name,
                        approval_request=approval_request,
                    ),
                    metadata={
                        "approval_required": True,
                        "approval_request": approval_request,
                        "arguments": dict(tool_call.arguments),
                    },
                )
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


def _format_approval_required_content(
    *,
    tool_name: str,
    approval_request: dict[str, object],
) -> str:
    summary = str(approval_request.get("summary", "")).strip() or "Approval required."
    lines = [
        "Approval required",
        f"tool: {tool_name}",
        f"summary: {summary}",
    ]
    details = str(approval_request.get("details", "")).strip()
    if details:
        lines.append(f"details: {details}")
    command = str(approval_request.get("command", "")).strip()
    if command:
        lines.append(f"command: {command}")
    tool_label = str(approval_request.get("tool_name", "")).strip()
    if tool_label:
        lines.append(f"tool_name: {tool_label}")
    return "\n".join(lines)
