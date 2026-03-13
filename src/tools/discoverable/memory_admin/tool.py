"""Memory-admin discoverable tool definition and execution runtime."""

from __future__ import annotations

import json
from typing import Any

from llm import ToolDefinition

from ...types import DiscoverableTool, RegisteredTool, ToolExecutionContext, ToolExecutionResult


class MemoryAdminToolExecutor:
    """Runs manual memory maintenance and inspection actions."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        service = context.memory_service
        if service is None:
            return ToolExecutionResult(
                call_id=call_id,
                name="memory_admin",
                ok=False,
                content="Memory service is not available in this runtime.",
            )
        action = str(arguments.get("action", "")).strip()
        if action == "reindex_all":
            paths = await service.reindex_all()
            metadata = {"action": action, "paths": [str(path) for path in paths]}
            content = f"Reindexed {len(paths)} memory documents."
        elif action == "reindex_dirty":
            dirty = await service.reindex_dirty()
            metadata = {"action": action, "dirty_paths": [str(item.path) for item in dirty]}
            content = f"Reindexed {len(dirty)} dirty memory documents."
        elif action == "rebuild_embeddings":
            summary = await service.rebuild_embeddings()
            metadata = {"action": action, "summary": dict(summary)}
            content = "Memory embeddings rebuild completed.\n" + json.dumps(summary, indent=2, sort_keys=True)
        elif action == "repair_canonical_drift":
            summary = await service.repair_canonical_drift()
            metadata = {"action": action, "summary": dict(summary)}
            content = "Canonical memory repair completed.\n" + json.dumps(summary, indent=2, sort_keys=True)
        elif action == "run_due_maintenance":
            runs = await service.run_due_maintenance()
            metadata = {
                "action": action,
                "runs": [
                    {"job_name": run.job_name, "status": run.status, "summary": dict(run.summary)}
                    for run in runs
                ],
            }
            content = "Due memory maintenance completed.\n" + "\n".join(
                f"- {run.job_name}: {run.status}" for run in runs
            )
        elif action == "integrity_check":
            issues = await service.integrity_check()
            metadata = {
                "action": action,
                "issues": [
                    {
                        "path": str(issue.path) if issue.path is not None else None,
                        "severity": issue.severity,
                        "code": issue.code,
                        "message": issue.message,
                    }
                    for issue in issues
                ],
            }
            content = (
                "Memory integrity check passed."
                if not issues
                else "Memory integrity issues found.\n" + "\n".join(
                    f"- {issue.code}: {issue.message}" for issue in issues
                )
            )
        elif action == "render_bootstrap_preview":
            preview = await service.render_bootstrap_preview()
            metadata = {"action": action}
            content = preview or "No runtime memory bootstrap content is currently active."
        else:
            return ToolExecutionResult(
                call_id=call_id,
                name="memory_admin",
                ok=False,
                content=f"Unsupported memory_admin action: {action}",
            )
        return ToolExecutionResult(
            call_id=call_id,
            name="memory_admin",
            ok=True,
            content=content,
            metadata=metadata,
        )


def build_memory_admin_tool() -> RegisteredTool:
    return RegisteredTool(
        name="memory_admin",
        exposure="discoverable",
        definition=ToolDefinition(
            name="memory_admin",
            description=(
                "Manual memory maintenance and inspection. Do not use this unless the user explicitly "
                "requests memory administration."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "reindex_all",
                            "reindex_dirty",
                            "rebuild_embeddings",
                            "repair_canonical_drift",
                            "run_due_maintenance",
                            "integrity_check",
                            "render_bootstrap_preview",
                        ],
                    }
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        ),
        executor=MemoryAdminToolExecutor(),
    )


def build_memory_admin_discoverable() -> DiscoverableTool:
    return DiscoverableTool(
        name="memory_admin",
        aliases=("memory maintenance", "memory reindex", "memory integrity"),
        purpose="Manual maintenance and inspection actions for the runtime memory system.",
        detailed_description=(
            "Use this only when the user explicitly asks for memory administration, reindexing, "
            "integrity checks, embedding rebuilds, or a bootstrap preview."
        ),
        usage={
            "actions": [
                "reindex_all",
                "reindex_dirty",
                "rebuild_embeddings",
                "repair_canonical_drift",
                "run_due_maintenance",
                "integrity_check",
                "render_bootstrap_preview",
            ]
        },
        backing_tool_name="memory_admin",
    )
