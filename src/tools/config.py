"""Configuration for the tools layer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import settings as app_settings


def _optional_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


@dataclass(slots=True, frozen=True)
class ToolSettings:
    """Runtime settings for tool registration and execution."""

    workspace_dir: Path
    bash_executable: str
    bash_default_timeout_seconds: float
    bash_max_timeout_seconds: float
    bash_max_output_chars: int
    max_tool_rounds_per_turn: int

    def __post_init__(self) -> None:
        if not self.bash_executable:
            raise ValueError("ToolSettings.bash_executable cannot be empty.")
        if self.bash_default_timeout_seconds <= 0:
            raise ValueError("bash_default_timeout_seconds must be > 0.")
        if self.bash_max_timeout_seconds <= 0:
            raise ValueError("bash_max_timeout_seconds must be > 0.")
        if self.bash_default_timeout_seconds > self.bash_max_timeout_seconds:
            raise ValueError(
                "bash_default_timeout_seconds cannot exceed bash_max_timeout_seconds."
            )
        if self.bash_max_output_chars <= 0:
            raise ValueError("bash_max_output_chars must be > 0.")
        if self.max_tool_rounds_per_turn <= 0:
            raise ValueError("max_tool_rounds_per_turn must be > 0.")

    @classmethod
    def from_env(cls) -> "ToolSettings":
        workspace_root = _optional_env("AGENT_WORKSPACE") or app_settings.AGENT_WORKSPACE
        if workspace_root is None:
            workspace_root = "/workspace"
        return cls.from_workspace_dir(Path(workspace_root).expanduser())

    @classmethod
    def from_workspace_dir(cls, workspace_dir: Path) -> "ToolSettings":
        return cls(
            workspace_dir=workspace_dir.expanduser(),
            bash_executable=(
                _optional_env("JARVIS_TOOL_BASH_EXECUTABLE")
                or app_settings.JARVIS_TOOL_BASH_EXECUTABLE
            ),
            bash_default_timeout_seconds=_parse_float_env(
                "JARVIS_TOOL_BASH_DEFAULT_TIMEOUT_SECONDS",
                app_settings.JARVIS_TOOL_BASH_DEFAULT_TIMEOUT_SECONDS,
            ),
            bash_max_timeout_seconds=_parse_float_env(
                "JARVIS_TOOL_BASH_MAX_TIMEOUT_SECONDS",
                app_settings.JARVIS_TOOL_BASH_MAX_TIMEOUT_SECONDS,
            ),
            bash_max_output_chars=_parse_int_env(
                "JARVIS_TOOL_BASH_MAX_OUTPUT_CHARS",
                app_settings.JARVIS_TOOL_BASH_MAX_OUTPUT_CHARS,
            ),
            max_tool_rounds_per_turn=_parse_int_env(
                "JARVIS_TOOL_MAX_ROUNDS_PER_TURN",
                app_settings.JARVIS_TOOL_MAX_ROUNDS_PER_TURN,
            ),
        )
