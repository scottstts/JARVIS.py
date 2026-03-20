"""Python-interpreter tool definition and isolated-runtime routing."""

from __future__ import annotations

from typing import Any

from llm import ToolDefinition

from ...config import ToolSettings
from ...remote_runtime_client import RemoteToolRuntimeClient
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult
from .local_executor import (
    _MAX_ARGS,
    _MAX_ARG_CHARS,
    DirectPythonInterpreterToolExecutor,
    build_python_interpreter_description,
)


class PythonInterpreterToolExecutor:
    """Routes python_interpreter execution to the isolated runtime when configured."""

    def __init__(self, settings: ToolSettings) -> None:
        self._remote_client = RemoteToolRuntimeClient(settings)
        self._local_executor = DirectPythonInterpreterToolExecutor(
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
                tool_name="python_interpreter",
                call_id=call_id,
                arguments=arguments,
                context=context,
            )
        return await self._local_executor(
            call_id=call_id,
            arguments=arguments,
            context=context,
        )


def build_python_interpreter_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the python_interpreter registry entry."""

    return RegisteredTool(
        name="python_interpreter",
        exposure="basic",
        definition=ToolDefinition(
            name="python_interpreter",
            description=build_python_interpreter_description(settings),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": ["string", "null"],
                        "description": (
                            "Inline Python code to execute. Provide exactly one of "
                            "'code' or 'script_path'."
                        ),
                    },
                    "script_path": {
                        "type": ["string", "null"],
                        "description": (
                            "Workspace path to a stored Python script to run. "
                            "Provide exactly one of 'code' or 'script_path'."
                        ),
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "maxLength": _MAX_ARG_CHARS,
                        },
                        "maxItems": _MAX_ARGS,
                        "description": (
                            "Optional positional args exposed to the script as sys.argv[1:]."
                        ),
                    },
                    "read_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": settings.python_interpreter_max_paths,
                        "description": (
                            "Deprecated no-op kept for compatibility. "
                            "The real workspace is mounted directly at /workspace."
                        ),
                    },
                    "write_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": settings.python_interpreter_max_paths,
                        "description": (
                            "Deprecated no-op kept for compatibility. "
                            "Scripts may write directly inside /workspace."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": settings.python_interpreter_max_timeout_seconds,
                        "description": (
                            "Optional execution timeout in seconds. Use only when the script "
                            "legitimately needs longer than the default."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        ),
        executor=PythonInterpreterToolExecutor(settings),
    )
