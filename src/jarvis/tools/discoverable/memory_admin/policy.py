"""Policy checks for the memory_admin tool."""

from __future__ import annotations

from ...types import ToolExecutionContext, ToolPolicyDecision


class MemoryAdminPolicy:
    """Thin validation for memory_admin."""

    def authorize(
        self,
        *,
        action: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context
        if action not in {
            "reindex_all",
            "reindex_dirty",
            "rebuild_embeddings",
            "run_due_maintenance",
            "integrity_check",
            "render_bootstrap_preview",
        }:
            return ToolPolicyDecision(allowed=False, reason=f"Unsupported memory_admin action: {action}")
        return ToolPolicyDecision(allowed=True)

