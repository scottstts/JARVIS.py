"""Bash tool definition and isolated-runtime routing."""

from __future__ import annotations

from typing import Any

from jarvis.llm import ToolDefinition

from ...config import ToolSettings
from ...remote_runtime_client import RemoteToolRuntimeClient
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult
from .local_executor import DirectBashToolExecutor, format_bash_tool_description


class BashToolExecutor:
    """Routes bash execution to the isolated tool runtime when configured."""

    def __init__(self, settings: ToolSettings) -> None:
        self._remote_client = RemoteToolRuntimeClient(settings)
        self._local_executor = DirectBashToolExecutor(
            settings,
            target_runtime="local_app_process",
            runtime_location="local_app_process",
            runtime_transport="inprocess",
            container_mutation_boundary="shared_with_app_runtime",
        )

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        if self._remote_client.enabled:
            return await self._remote_client.execute(
                tool_name="bash",
                call_id=call_id,
                arguments=arguments,
                context=context,
            )
        return await self._local_executor(
            call_id=call_id,
            arguments=arguments,
            context=context,
        )


def build_bash_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the default bash tool registry entry."""

    return RegisteredTool(
        name="bash",
        exposure="basic",
        definition=ToolDefinition(
            name="bash",
            description=format_bash_tool_description(settings),
            input_schema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["foreground", "background", "status", "tail", "cancel"],
                        "description": (
                            "Optional mode. Defaults to 'foreground'. Use 'background' to start "
                            "a detached job; use 'status', 'tail', or 'cancel' with a job id."
                        ),
                    },
                    "command": {
                        "type": "string",
                        "description": (
                            "Shell command for 'foreground' or 'background'."
                        ),
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Background job id for 'status', 'tail', or 'cancel'.",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": settings.bash_max_timeout_seconds,
                        "description": "Optional foreground timeout in seconds.",
                    },
                    "tail_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 2000,
                        "description": "Optional trailing line count for 'tail'.",
                    },
                    "tail_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": settings.bash_max_output_chars,
                        "description": "Optional byte limit for 'tail'.",
                    },
                    "approval_summary": {
                        "type": "string",
                        "description": "Optional short approval summary.",
                    },
                    "approval_details": {
                        "type": "string",
                        "description": "Optional longer approval rationale.",
                    },
                    "inspection_url": {
                        "type": "string",
                        "description": "Optional URL to inspect before approval.",
                    },
                },
                "additionalProperties": False,
            },
        ),
        executor=BashToolExecutor(settings),
    )
