"""Configuration for the tools layer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import settings as app_settings
from workspace_paths import resolve_workspace_dir


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
    python_interpreter_venv: Path
    python_interpreter_allowed_packages: tuple[str, ...]
    python_interpreter_default_timeout_seconds: float
    python_interpreter_max_timeout_seconds: float
    python_interpreter_max_output_chars: int
    python_interpreter_max_code_chars: int
    python_interpreter_max_paths: int
    python_interpreter_max_staged_bytes: int
    python_interpreter_memory_limit_bytes: int
    python_interpreter_file_size_limit_bytes: int
    max_tool_rounds_per_turn: int
    web_search_result_count: int
    web_search_timeout_seconds: float
    web_fetch_timeout_seconds: float
    web_fetch_playwright_timeout_seconds: float
    web_fetch_max_response_bytes: int
    web_fetch_max_markdown_chars: int

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
        if not str(self.python_interpreter_venv).strip():
            raise ValueError("python_interpreter_venv cannot be empty.")
        if not self.python_interpreter_allowed_packages:
            raise ValueError("python_interpreter_allowed_packages cannot be empty.")
        if self.python_interpreter_default_timeout_seconds <= 0:
            raise ValueError("python_interpreter_default_timeout_seconds must be > 0.")
        if self.python_interpreter_max_timeout_seconds <= 0:
            raise ValueError("python_interpreter_max_timeout_seconds must be > 0.")
        if (
            self.python_interpreter_default_timeout_seconds
            > self.python_interpreter_max_timeout_seconds
        ):
            raise ValueError(
                "python_interpreter_default_timeout_seconds cannot exceed "
                "python_interpreter_max_timeout_seconds."
            )
        if self.python_interpreter_max_output_chars <= 0:
            raise ValueError("python_interpreter_max_output_chars must be > 0.")
        if self.python_interpreter_max_code_chars <= 0:
            raise ValueError("python_interpreter_max_code_chars must be > 0.")
        if self.python_interpreter_max_paths <= 0:
            raise ValueError("python_interpreter_max_paths must be > 0.")
        if self.python_interpreter_max_staged_bytes <= 0:
            raise ValueError("python_interpreter_max_staged_bytes must be > 0.")
        if self.python_interpreter_memory_limit_bytes <= 0:
            raise ValueError("python_interpreter_memory_limit_bytes must be > 0.")
        if self.python_interpreter_file_size_limit_bytes <= 0:
            raise ValueError("python_interpreter_file_size_limit_bytes must be > 0.")
        if self.max_tool_rounds_per_turn <= 0:
            raise ValueError("max_tool_rounds_per_turn must be > 0.")
        if self.web_search_result_count <= 0:
            raise ValueError("web_search_result_count must be > 0.")
        if self.web_search_result_count > 20:
            raise ValueError("web_search_result_count must be <= 20.")
        if self.web_search_timeout_seconds <= 0:
            raise ValueError("web_search_timeout_seconds must be > 0.")
        if self.web_fetch_timeout_seconds <= 0:
            raise ValueError("web_fetch_timeout_seconds must be > 0.")
        if self.web_fetch_playwright_timeout_seconds <= 0:
            raise ValueError("web_fetch_playwright_timeout_seconds must be > 0.")
        if self.web_fetch_max_response_bytes <= 0:
            raise ValueError("web_fetch_max_response_bytes must be > 0.")
        if self.web_fetch_max_markdown_chars <= 0:
            raise ValueError("web_fetch_max_markdown_chars must be > 0.")

    @classmethod
    def from_env(cls) -> "ToolSettings":
        return cls.from_workspace_dir(resolve_workspace_dir())

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
            python_interpreter_venv=Path(
                _optional_env("JARVIS_TOOL_PYTHON_INTERPRETER_VENV")
                or app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_VENV
            ).expanduser(),
            python_interpreter_allowed_packages=tuple(
                app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_ALLOWED_PACKAGES
            ),
            python_interpreter_default_timeout_seconds=_parse_float_env(
                "JARVIS_TOOL_PYTHON_INTERPRETER_DEFAULT_TIMEOUT_SECONDS",
                app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_DEFAULT_TIMEOUT_SECONDS,
            ),
            python_interpreter_max_timeout_seconds=_parse_float_env(
                "JARVIS_TOOL_PYTHON_INTERPRETER_MAX_TIMEOUT_SECONDS",
                app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_MAX_TIMEOUT_SECONDS,
            ),
            python_interpreter_max_output_chars=_parse_int_env(
                "JARVIS_TOOL_PYTHON_INTERPRETER_MAX_OUTPUT_CHARS",
                app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_MAX_OUTPUT_CHARS,
            ),
            python_interpreter_max_code_chars=_parse_int_env(
                "JARVIS_TOOL_PYTHON_INTERPRETER_MAX_CODE_CHARS",
                app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_MAX_CODE_CHARS,
            ),
            python_interpreter_max_paths=_parse_int_env(
                "JARVIS_TOOL_PYTHON_INTERPRETER_MAX_PATHS",
                app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_MAX_PATHS,
            ),
            python_interpreter_max_staged_bytes=_parse_int_env(
                "JARVIS_TOOL_PYTHON_INTERPRETER_MAX_STAGED_BYTES",
                app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_MAX_STAGED_BYTES,
            ),
            python_interpreter_memory_limit_bytes=_parse_int_env(
                "JARVIS_TOOL_PYTHON_INTERPRETER_MEMORY_LIMIT_BYTES",
                app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_MEMORY_LIMIT_BYTES,
            ),
            python_interpreter_file_size_limit_bytes=_parse_int_env(
                "JARVIS_TOOL_PYTHON_INTERPRETER_FILE_SIZE_LIMIT_BYTES",
                app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_FILE_SIZE_LIMIT_BYTES,
            ),
            max_tool_rounds_per_turn=_parse_int_env(
                "JARVIS_TOOL_MAX_ROUNDS_PER_TURN",
                app_settings.JARVIS_TOOL_MAX_ROUNDS_PER_TURN,
            ),
            web_search_result_count=_parse_int_env(
                "JARVIS_TOOL_WEB_SEARCH_RESULT_COUNT",
                app_settings.JARVIS_TOOL_WEB_SEARCH_RESULT_COUNT,
            ),
            web_search_timeout_seconds=_parse_float_env(
                "JARVIS_TOOL_WEB_SEARCH_TIMEOUT_SECONDS",
                app_settings.JARVIS_TOOL_WEB_SEARCH_TIMEOUT_SECONDS,
            ),
            web_fetch_timeout_seconds=_parse_float_env(
                "JARVIS_TOOL_WEB_FETCH_TIMEOUT_SECONDS",
                app_settings.JARVIS_TOOL_WEB_FETCH_TIMEOUT_SECONDS,
            ),
            web_fetch_playwright_timeout_seconds=_parse_float_env(
                "JARVIS_TOOL_WEB_FETCH_PLAYWRIGHT_TIMEOUT_SECONDS",
                app_settings.JARVIS_TOOL_WEB_FETCH_PLAYWRIGHT_TIMEOUT_SECONDS,
            ),
            web_fetch_max_response_bytes=_parse_int_env(
                "JARVIS_TOOL_WEB_FETCH_MAX_RESPONSE_BYTES",
                app_settings.JARVIS_TOOL_WEB_FETCH_MAX_RESPONSE_BYTES,
            ),
            web_fetch_max_markdown_chars=_parse_int_env(
                "JARVIS_TOOL_WEB_FETCH_MAX_MARKDOWN_CHARS",
                app_settings.JARVIS_TOOL_WEB_FETCH_MAX_MARKDOWN_CHARS,
            ),
        )
