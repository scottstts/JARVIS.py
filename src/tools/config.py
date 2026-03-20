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


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got: {raw}")


@dataclass(slots=True, frozen=True)
class ToolSettings:
    """Runtime settings for tool registration and execution."""

    workspace_dir: Path
    tool_runtime_base_url: str | None
    tool_runtime_timeout_seconds: float
    tool_runtime_healthcheck_timeout_seconds: float
    bash_executable: str
    bash_default_timeout_seconds: float
    bash_max_timeout_seconds: float
    bash_foreground_soft_timeout_seconds: float
    bash_max_output_chars: int
    bash_dangerously_skip_permission: bool
    python_interpreter_venv: Path
    python_interpreter_starter_packages: tuple[str, ...]
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
    email_smtp_host: str
    email_smtp_port: int
    email_smtp_security: str
    email_timeout_seconds: float
    email_sender_address: str | None
    email_max_subject_chars: int
    email_max_body_chars: int
    email_max_attachment_count: int
    email_max_total_attachment_bytes: int

    def __post_init__(self) -> None:
        if self.tool_runtime_base_url is not None and not self.tool_runtime_base_url.strip():
            raise ValueError("tool_runtime_base_url cannot be blank when configured.")
        if self.tool_runtime_timeout_seconds <= 0:
            raise ValueError("tool_runtime_timeout_seconds must be > 0.")
        if self.tool_runtime_healthcheck_timeout_seconds <= 0:
            raise ValueError("tool_runtime_healthcheck_timeout_seconds must be > 0.")
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
        if self.bash_foreground_soft_timeout_seconds <= 0:
            raise ValueError("bash_foreground_soft_timeout_seconds must be > 0.")
        if self.bash_max_output_chars <= 0:
            raise ValueError("bash_max_output_chars must be > 0.")
        if not isinstance(self.bash_dangerously_skip_permission, bool):
            raise ValueError("bash_dangerously_skip_permission must be a boolean.")
        if not str(self.python_interpreter_venv).strip():
            raise ValueError("python_interpreter_venv cannot be empty.")
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
        if not self.email_smtp_host.strip():
            raise ValueError("email_smtp_host cannot be empty.")
        if self.email_smtp_port <= 0:
            raise ValueError("email_smtp_port must be > 0.")
        if self.email_smtp_security not in {"ssl", "starttls", "none"}:
            raise ValueError(
                "email_smtp_security must be one of: ssl, starttls, none."
            )
        if self.email_timeout_seconds <= 0:
            raise ValueError("email_timeout_seconds must be > 0.")
        if self.email_sender_address is not None and not self.email_sender_address.strip():
            raise ValueError("email_sender_address cannot be blank when configured.")
        if self.email_max_subject_chars <= 0:
            raise ValueError("email_max_subject_chars must be > 0.")
        if self.email_max_body_chars <= 0:
            raise ValueError("email_max_body_chars must be > 0.")
        if self.email_max_attachment_count <= 0:
            raise ValueError("email_max_attachment_count must be > 0.")
        if self.email_max_total_attachment_bytes <= 0:
            raise ValueError("email_max_total_attachment_bytes must be > 0.")

    @classmethod
    def from_env(cls) -> "ToolSettings":
        return cls.from_workspace_dir(resolve_workspace_dir())

    @classmethod
    def from_workspace_dir(cls, workspace_dir: Path) -> "ToolSettings":
        tool_runtime_base_url = (
            _optional_env("JARVIS_TOOL_RUNTIME_BASE_URL")
            or app_settings.JARVIS_TOOL_RUNTIME_BASE_URL
        )
        if tool_runtime_base_url is not None:
            tool_runtime_base_url = tool_runtime_base_url.rstrip("/")

        return cls(
            workspace_dir=workspace_dir.expanduser(),
            tool_runtime_base_url=tool_runtime_base_url,
            tool_runtime_timeout_seconds=_parse_float_env(
                "JARVIS_TOOL_RUNTIME_TIMEOUT_SECONDS",
                app_settings.JARVIS_TOOL_RUNTIME_TIMEOUT_SECONDS,
            ),
            tool_runtime_healthcheck_timeout_seconds=_parse_float_env(
                "JARVIS_TOOL_RUNTIME_HEALTHCHECK_TIMEOUT_SECONDS",
                app_settings.JARVIS_TOOL_RUNTIME_HEALTHCHECK_TIMEOUT_SECONDS,
            ),
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
            bash_foreground_soft_timeout_seconds=_parse_float_env(
                "JARVIS_TOOL_BASH_FOREGROUND_SOFT_TIMEOUT_SECONDS",
                app_settings.JARVIS_TOOL_BASH_FOREGROUND_SOFT_TIMEOUT_SECONDS,
            ),
            bash_max_output_chars=_parse_int_env(
                "JARVIS_TOOL_BASH_MAX_OUTPUT_CHARS",
                app_settings.JARVIS_TOOL_BASH_MAX_OUTPUT_CHARS,
            ),
            bash_dangerously_skip_permission=_parse_bool_env(
                "BASH_DANGEROUSLY_SKIP_PERMISSION",
                app_settings.BASH_DANGEROUSLY_SKIP_PERMISSION,
            ),
            python_interpreter_venv=Path(
                _optional_env("JARVIS_TOOL_PYTHON_INTERPRETER_VENV")
                or app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_VENV
            ).expanduser(),
            python_interpreter_starter_packages=tuple(
                app_settings.JARVIS_TOOL_PYTHON_INTERPRETER_STARTER_PACKAGES
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
            email_smtp_host=(
                _optional_env("SMTP_HOST")
                or app_settings.JARVIS_TOOL_EMAIL_SMTP_HOST
            ),
            email_smtp_port=_parse_int_env(
                "SMTP_PORT",
                app_settings.JARVIS_TOOL_EMAIL_SMTP_PORT,
            ),
            email_smtp_security=(
                _optional_env("SMTP_SECURITY")
                or app_settings.JARVIS_TOOL_EMAIL_SMTP_SECURITY
            ).lower(),
            email_timeout_seconds=_parse_float_env(
                "SMTP_TIMEOUT_SECONDS",
                app_settings.JARVIS_TOOL_EMAIL_TIMEOUT_SECONDS,
            ),
            email_sender_address=_optional_env("SENDER_EMAIL_ADDRESS"),
            email_max_subject_chars=_parse_int_env(
                "JARVIS_TOOL_EMAIL_MAX_SUBJECT_CHARS",
                app_settings.JARVIS_TOOL_EMAIL_MAX_SUBJECT_CHARS,
            ),
            email_max_body_chars=_parse_int_env(
                "JARVIS_TOOL_EMAIL_MAX_BODY_CHARS",
                app_settings.JARVIS_TOOL_EMAIL_MAX_BODY_CHARS,
            ),
            email_max_attachment_count=_parse_int_env(
                "JARVIS_TOOL_EMAIL_MAX_ATTACHMENT_COUNT",
                app_settings.JARVIS_TOOL_EMAIL_MAX_ATTACHMENT_COUNT,
            ),
            email_max_total_attachment_bytes=_parse_int_env(
                "JARVIS_TOOL_EMAIL_MAX_TOTAL_ATTACHMENT_BYTES",
                app_settings.JARVIS_TOOL_EMAIL_MAX_TOTAL_ATTACHMENT_BYTES,
            ),
        )
