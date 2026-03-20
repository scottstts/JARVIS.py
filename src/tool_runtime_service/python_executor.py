"""Service wiring for isolated python_interpreter execution."""

from __future__ import annotations

from tools.basic.python_interpreter.local_executor import DirectPythonInterpreterToolExecutor
from tools.config import ToolSettings


def build_service_python_executor(
    settings: ToolSettings,
) -> DirectPythonInterpreterToolExecutor:
    return DirectPythonInterpreterToolExecutor(
        settings,
        target_runtime="tool_runtime",
        runtime_location="tool_runtime_container",
        runtime_transport="http",
        container_mutation_boundary="isolated_from_app_runtime",
    )
