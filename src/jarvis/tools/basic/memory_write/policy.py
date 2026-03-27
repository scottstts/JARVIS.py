"""Policy checks for the memory_write tool."""

from __future__ import annotations

from .contract import format_memory_write_contract_error, validate_memory_write_contract
from ...types import ToolExecutionContext, ToolPolicyDecision


class MemoryWritePolicy:
    """Thin validation for canonical memory writes."""

    def authorize(
        self,
        *,
        operation: str,
        target_kind: str,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context
        if operation not in {"create", "upsert", "append_daily", "close", "archive", "promote", "demote"}:
            return ToolPolicyDecision(allowed=False, reason=f"Unsupported memory_write operation: {operation}")
        if target_kind not in {"core", "ongoing", "daily"}:
            return ToolPolicyDecision(allowed=False, reason=f"Unsupported memory target kind: {target_kind}")
        errors = validate_memory_write_contract(
            operation=operation,
            target_kind=target_kind,
            arguments=arguments,
        )
        if errors:
            return ToolPolicyDecision(
                allowed=False,
                reason=format_memory_write_contract_error(
                    operation=operation,
                    target_kind=target_kind,
                    errors=errors,
                ),
            )
        return ToolPolicyDecision(allowed=True)
