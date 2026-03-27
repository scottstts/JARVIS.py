"""Service wiring for isolated bash execution."""

from __future__ import annotations

from jarvis.tools.basic.bash.local_executor import DirectBashToolExecutor
from jarvis.tools.config import ToolSettings


def build_service_bash_executor(settings: ToolSettings) -> DirectBashToolExecutor:
    return DirectBashToolExecutor(
        settings,
        target_runtime="tool_runtime",
        runtime_location="tool_runtime_container",
        runtime_transport="http",
        container_mutation_boundary="isolated_from_app_runtime",
    )
