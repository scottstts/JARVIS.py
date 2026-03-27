"""Bash-specific policy implementation."""

from __future__ import annotations

import re
from typing import Any

from ...config import ToolSettings
from ...types import ToolExecutionContext, ToolPolicyDecision

_INSTALL_KEYWORDS = (
    "apt install",
    "apt-get install",
    "apt-get update",
    "pip install",
    "pip3 install",
    "uv tool install",
    "uv pip install",
    "npm install -g",
    "npm uninstall -g",
    "pnpm add -g",
    "yarn global add",
    "cargo install",
    "go install",
    "brew install",
    "brew tap",
    "gem install",
    "pipx install",
    "poetry add",
)
_MODE_VALUES = {"foreground", "background", "status", "tail", "cancel"}
_COMMAND_PREFIX = r"(?:^|[;&|()]\s*|\n\s*)"
_OPTIONAL_SUDO = r"(?:sudo\s+)?"
_OPTIONAL_ENV_WRAPPER = r"(?:env\s+)?"
_OPTIONAL_ENV_ASSIGNMENTS = r"(?:[a-z_][a-z0-9_]*=[^\s;&|()]+\s+)*"
_PYTHON_COMMAND_PATTERN = re.compile(
    _COMMAND_PREFIX
    + _OPTIONAL_SUDO
    + _OPTIONAL_ENV_WRAPPER
    + _OPTIONAL_ENV_ASSIGNMENTS
    + r"(?P<command>(?:\./|\.\./|/)?(?:[\w.-]+/)*python(?:\d+(?:\.\d+)*)?)\b"
)
_PYTHON_ENV_OVERRIDE_PATTERN = re.compile(
    _COMMAND_PREFIX
    + _OPTIONAL_SUDO
    + _OPTIONAL_ENV_WRAPPER
    + r"(?:path|virtual_env)\s*=[^\n;&|()]*\bpython(?:\d+(?:\.\d+)*)?\b"
)
_PYTHON_ACTIVATE_PATTERN = re.compile(
    _COMMAND_PREFIX
    + _OPTIONAL_SUDO
    + r"(?:source|\.)\s+(?P<activate>[^\n;&|()]*?/bin/activate)\b"
)
_UV_RUN_PYTHON_PATTERN = re.compile(
    _COMMAND_PREFIX
    + _OPTIONAL_SUDO
    + _OPTIONAL_ENV_WRAPPER
    + _OPTIONAL_ENV_ASSIGNMENTS
    + r"uv\s+run(?:\s+[^\n;&|()]+)*\s+python(?:\d+(?:\.\d+)*)?\b"
)
_UV_PIP_INSTALL_TARGET_PATTERN = re.compile(
    _COMMAND_PREFIX
    + _OPTIONAL_SUDO
    + _OPTIONAL_ENV_WRAPPER
    + _OPTIONAL_ENV_ASSIGNMENTS
    + r"uv\s+pip\s+install\b(?P<flags>[^\n;&|()]*)"
)
_UV_PYTHON_FLAG_PATTERN = re.compile(
    r"(?:^|\s)--python(?:=|\s+)(?P<python>[^\s;&|()]+)"
)
_PYTHON_ENV_CREATION_PATTERN = re.compile(
    _COMMAND_PREFIX
    + _OPTIONAL_SUDO
    + r"(?:python(?:\d+(?:\.\d+)*)?\s+-m\s+venv|uv\s+venv|virtualenv|conda\s+create)\b"
)
_PYTHON_COMMAND_NAME_PATTERN = re.compile(r"python(?:\d+(?:\.\d+)*)?$")
_HARD_DENY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            _COMMAND_PREFIX
            + _OPTIONAL_SUDO
            + r"apt(?:-get)?\s+(?:[^;&|()\n]+\s+)?(?:upgrade|full-upgrade|dist-upgrade)\b"
        ),
        "tool_runtime OS upgrade commands are denied.",
    ),
    (
        re.compile(_COMMAND_PREFIX + _OPTIONAL_SUDO + r"do-release-upgrade\b"),
        "tool_runtime OS upgrade commands are denied.",
    ),
    (
        re.compile(
            _COMMAND_PREFIX
            + _OPTIONAL_SUDO
            + r"(?:systemctl|service|init|telinit|reboot|shutdown|poweroff|halt)\b"
        ),
        "tool_runtime service and init control commands are denied.",
    ),
    (
        re.compile(
            _COMMAND_PREFIX
            + _OPTIONAL_SUDO
            + r"(?:mount|umount|swapon|swapoff|modprobe|insmod)\b"
        ),
        "tool_runtime mount, kernel, and low-level admin commands are denied.",
    ),
    (
        re.compile(_COMMAND_PREFIX + _OPTIONAL_SUDO + r"sysctl\s+-w\b"),
        "tool_runtime mount, kernel, and low-level admin commands are denied.",
    ),
    (
        re.compile(_COMMAND_PREFIX + _OPTIONAL_SUDO + r"(?:docker|podman|nerdctl)\b"),
        "tool_runtime container-runtime recursion is denied.",
    ),
)
_SYSTEM_DESTINATION_PATTERN = re.compile(
    r"\b(?:install|cp|mv|ln|chmod|chown)\b[^\n]*\s"
    r"(/usr/local/bin\S*|/usr/bin\S*|/bin/\S*|/sbin/\S*|/etc/\S*|/opt/\S*|/var/\S*|/root/\S*)\s*$"
)
_SYSTEM_REDIRECT_PATTERN = re.compile(
    r"(?:^|[;&|])[^#\n]*(?:>|>>)\s*"
    r"(/usr/local/bin\S*|/usr/bin\S*|/bin/\S*|/sbin/\S*|/etc/\S*|/opt/\S*|/var/\S*|/root/\S*)"
)


class BashCommandPolicy:
    """Thin validation for the tool_runtime bash executor."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    def authorize(
        self,
        *,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context

        mode = _normalize_mode(arguments.get("mode"))
        if mode not in _MODE_VALUES:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"bash mode must be one of: {', '.join(sorted(_MODE_VALUES))}.",
            )

        if mode in {"status", "tail", "cancel"}:
            job_id = str(arguments.get("job_id", "")).strip()
            if not job_id:
                return ToolPolicyDecision(
                    allowed=False,
                    reason="bash mode requires a non-empty job_id.",
                )
            if any(ch not in "0123456789abcdef" for ch in job_id.lower()):
                return ToolPolicyDecision(
                    allowed=False,
                    reason="bash job_id must be a lowercase hex string.",
                )
            return ToolPolicyDecision(allowed=True)

        command = str(arguments.get("command", ""))
        if not command.strip():
            return ToolPolicyDecision(allowed=False, reason="bash command cannot be empty.")
        if "\x00" in command:
            return ToolPolicyDecision(
                allowed=False,
                reason="bash command cannot contain null bytes.",
            )

        hard_deny_reason = _python_environment_violation_reason(command, self._settings)
        if hard_deny_reason is None:
            hard_deny_reason = _hard_deny_reason(command)
        if hard_deny_reason is not None:
            return ToolPolicyDecision(allowed=False, reason=hard_deny_reason)

        if self._settings.bash_dangerously_skip_permission:
            return ToolPolicyDecision(allowed=True)

        detector_reason = _approval_detector_reason(command)
        if detector_reason is None:
            return ToolPolicyDecision(allowed=True)

        approved_action = context.approved_action or {}
        if (
            approved_action.get("kind") == "bash_command"
            and approved_action.get("command") == command
        ):
            return ToolPolicyDecision(allowed=True)

        summary = str(arguments.get("approval_summary", "")).strip()
        details = str(arguments.get("approval_details", "")).strip()
        inspection_url = str(arguments.get("inspection_url", "")).strip()
        if not summary:
            summary = "Run a bash command that mutates the isolated tool_runtime container."
        if not details:
            details = (
                "I want to run a bash command that appears to install, build, or modify "
                "system-level tooling inside the isolated tool_runtime container. "
                "Review the exact command before approving."
            )

        return ToolPolicyDecision(
            allowed=False,
            reason="bash command requires explicit approval.",
            approval_request={
                "kind": "bash_command",
                "summary": summary,
                "details": details,
                "inspection_url": inspection_url or None,
                "command": command,
                "detector_reason": detector_reason,
                "target_runtime": "tool_runtime",
                "runtime_location": "tool_runtime_container",
            },
        )


def _normalize_mode(value: object) -> str:
    if value is None:
        return "foreground"
    normalized = str(value).strip().lower()
    return normalized or "foreground"


def _hard_deny_reason(command: str) -> str | None:
    lowered = command.lower()
    for pattern, reason in _HARD_DENY_PATTERNS:
        if pattern.search(lowered):
            return reason
    return None


def _python_environment_violation_reason(
    command: str,
    settings: ToolSettings,
) -> str | None:
    lowered = command.lower()
    if _PYTHON_ENV_OVERRIDE_PATTERN.search(lowered):
        return (
            "bash Python commands must not override PATH or VIRTUAL_ENV away from the "
            f"central {_central_venv_path(settings)} environment. "
            f"{_central_python_guidance(settings)}"
        )

    central_activate_path = f"{_central_venv_path(settings)}/bin/activate"
    for match in _PYTHON_ACTIVATE_PATTERN.finditer(lowered):
        if match.group("activate") != central_activate_path:
            return (
                "bash must not activate a second Python environment for agent work. "
                f"{_central_python_guidance(settings)}"
            )

    if _UV_RUN_PYTHON_PATTERN.search(lowered):
        return (
            "bash Python commands must not route through `uv run python`, because that can "
            f"select a different interpreter. {_central_python_guidance(settings)}"
        )

    for match in _UV_PIP_INSTALL_TARGET_PATTERN.finditer(lowered):
        python_flag = _UV_PYTHON_FLAG_PATTERN.search(match.group("flags"))
        if python_flag is None:
            continue
        attempted = python_flag.group("python")
        if attempted == _central_interpreter_path(settings):
            continue
        return (
            "bash denied this package install because `uv pip install --python` targets a "
            f"non-central interpreter path '{attempted}'. {_central_python_guidance(settings)}"
        )

    if _PYTHON_ENV_CREATION_PATTERN.search(lowered):
        return (
            "bash must not create a second Python environment for agent work. "
            f"{_central_python_guidance(settings)}"
        )

    allowed_commands = _allowed_central_python_commands(settings)
    central_bin_prefix = f"{_central_venv_path(settings)}/bin/"
    for match in _PYTHON_COMMAND_PATTERN.finditer(lowered):
        attempted = match.group("command")
        if "/" in attempted:
            if attempted.startswith(central_bin_prefix):
                basename = attempted.rsplit("/", 1)[-1]
                if basename in allowed_commands:
                    continue
            return (
                f"bash denied Python command '{attempted}' because it targets a non-central "
                f"interpreter path. {_central_python_guidance(settings)}"
            )

        if attempted not in allowed_commands:
            return (
                f"bash denied Python command '{attempted}' because it does not resolve to the "
                f"central interpreter set. {_central_python_guidance(settings)}"
            )

    return None


def _allowed_central_python_commands(settings: ToolSettings) -> set[str]:
    venv_bin = settings.central_python_venv / "bin"
    allowed: set[str] = {"python", "python3"}
    try:
        for entry in venv_bin.iterdir():
            if _PYTHON_COMMAND_NAME_PATTERN.fullmatch(entry.name.lower()):
                allowed.add(entry.name.lower())
    except OSError:
        pass
    return allowed


def _central_venv_path(settings: ToolSettings) -> str:
    return str(settings.central_python_venv)


def _central_interpreter_path(settings: ToolSettings) -> str:
    return f"{_central_venv_path(settings)}/bin/python"


def _central_python_guidance(settings: ToolSettings) -> str:
    venv_path = _central_venv_path(settings)
    interpreter_path = _central_interpreter_path(settings)
    return (
        f"The only agent Python environment is {venv_path}. "
        f"Use {interpreter_path} explicitly, or use bare `python`/`python3` only when they "
        f"resolve there. Install packages with `uv pip install --python {interpreter_path} ...`."
    )


def _approval_detector_reason(command: str) -> str | None:
    lowered = " ".join(command.strip().lower().split())
    if not lowered:
        return None

    if any(keyword in lowered for keyword in _INSTALL_KEYWORDS):
        return "matched install or package-manager mutation pattern for tool_runtime"

    if ("curl " in lowered or "wget " in lowered) and (
        "| sh" in lowered
        or "| bash" in lowered
        or "| zsh" in lowered
        or "| python" in lowered
        or "| python3" in lowered
    ):
        return "matched remote installer pipeline pattern for tool_runtime"

    if _SYSTEM_DESTINATION_PATTERN.search(lowered):
        return "matched non-workspace system-path mutation pattern in tool_runtime"

    if _SYSTEM_REDIRECT_PATTERN.search(lowered):
        return "matched non-workspace system-path mutation pattern in tool_runtime"

    return None
