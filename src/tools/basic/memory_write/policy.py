"""Policy checks for the memory_write tool."""

from __future__ import annotations

from ...types import ToolExecutionContext, ToolPolicyDecision


class MemoryWritePolicy:
    """Thin validation for canonical memory writes."""

    def authorize(
        self,
        *,
        operation: str,
        target_kind: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context
        if operation not in {"create", "upsert", "append_daily", "close", "archive", "promote", "demote"}:
            return ToolPolicyDecision(allowed=False, reason=f"Unsupported memory_write operation: {operation}")
        if target_kind not in {"core", "ongoing", "daily"}:
            return ToolPolicyDecision(allowed=False, reason=f"Unsupported memory target kind: {target_kind}")
        return ToolPolicyDecision(allowed=True)

