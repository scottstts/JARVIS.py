"""Bash tool definition and isolated-runtime routing."""

from __future__ import annotations

from typing import Any

from llm import ToolDefinition

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
            description=format_bash_tool_description(),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Bash command to run."
                            "Use normal shell syntax, including pipes, redirects, "
                            "command substitution, &&, ||, and multiline scripts."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": settings.bash_max_timeout_seconds,
                        "description": (
                            "Optional command timeout in seconds. Use only when a "
                            "command may legitimately need more than the default."
                        ),
                    },
                    "approval_summary": {
                        "type": "string",
                        "description": (
                            "Optional short user-facing summary to show if this command "
                            "needs approval. Say what you want to do and why."
                        ),
                    },
                    "approval_details": {
                        "type": "string",
                        "description": (
                            "Optional longer user-facing approval explanation. Use this "
                            "for installs, builds, or other broader changes."
                        ),
                    },
                    "inspection_url": {
                        "type": "string",
                        "description": (
                            "Optional URL the user can inspect before approving the command, "
                            "such as the tool website or install documentation."
                        ),
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        ),
        executor=BashToolExecutor(settings),
    )
