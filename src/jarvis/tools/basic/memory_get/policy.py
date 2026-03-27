"""Policy checks for the memory_get tool."""

from __future__ import annotations

from pathlib import Path

from ...types import ToolExecutionContext, ToolPolicyDecision


class MemoryGetPolicy:
    """Restricts direct path access to the workspace memory directory."""

    def authorize(
        self,
        *,
        document_id: str | None,
        path: str | None,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        if document_id and document_id.strip():
            return ToolPolicyDecision(allowed=True)
        if path is None or not path.strip():
            return ToolPolicyDecision(
                allowed=False,
                reason="memory_get requires either document_id or path.",
            )
        resolved = Path(path).resolve(strict=False) if Path(path).is_absolute() else (context.workspace_dir / path).resolve(strict=False)
        memory_root = context.workspace_dir / "memory"
        try:
            resolved.relative_to(memory_root.resolve(strict=False))
        except ValueError:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"memory_get path must stay inside {memory_root}.",
            )
        return ToolPolicyDecision(allowed=True)

