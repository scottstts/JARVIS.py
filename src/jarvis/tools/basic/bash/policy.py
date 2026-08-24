"""Bash-specific policy implementation."""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import ToolSettings
from ...types import ToolExecutionContext, ToolPolicyDecision
from .shell_syntax import background_operator, masked_shell_syntax

_MODE_VALUES = {"foreground", "background", "service", "status", "tail", "cancel"}
_COMMAND_PREFIX = r"(?:^|[;&|()]\s*|\n\s*)"
_OPTIONAL_SUDO = r"(?:sudo\s+)?"
_OPTIONAL_ENV_WRAPPER = r"(?:env\s+)?"
_OPTIONAL_ENV_ASSIGNMENTS = r"(?:[a-z_][a-z0-9_]*=[^\s;&|()]+\s+)*"
_UNMANAGED_PROCESS_COMMAND_PATTERN = re.compile(
    r"(?:^|[;&|()\n]\s*)(?:sudo\s+)?(?:env\s+)?"
    r"(?:[a-z_][a-z0-9_]*=[^\s;&|()]+\s+)*(?:nohup|disown|setsid)\b",
    re.IGNORECASE,
)
_DETACH_WRAPPER_PATTERN = re.compile(
    r"(?:^|[;&|()\n]\s*)(?:sudo\s+)?(?:command|builtin)\s+"
    r"(?:nohup|disown|setsid)\b|"
    r"(?:^|[;&|()\n]\s*)(?:daemonize|start-stop-daemon\b[^\n;&|]*--background|systemd-run)\b",
    re.IGNORECASE,
)
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
_MUTATING_COMMAND_PREFIX = (
    _COMMAND_PREFIX
    + _OPTIONAL_SUDO
    + _OPTIONAL_ENV_WRAPPER
    + _OPTIONAL_ENV_ASSIGNMENTS
)
_DESTRUCTIVE_WORKSPACE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(_MUTATING_COMMAND_PREFIX + r"git\s+reset\b[^\n;&|]*--hard\b"),
        "matched destructive git reset --hard",
    ),
    (
        re.compile(
            _MUTATING_COMMAND_PREFIX
            + r"git\s+clean\b[^\n;&|]*(?:\s-[a-z]*f[a-z]*|\s--force)(?:\s|$)"
        ),
        "matched destructive git clean",
    ),
    (
        re.compile(
            _MUTATING_COMMAND_PREFIX
            + r"git\s+(?:checkout\b[^\n;&|]*(?:\s--(?:\s|$)|\s-f(?:\s|$))|restore\b)"
        ),
        "matched destructive git working-tree restore",
    ),
    (
        re.compile(
            _MUTATING_COMMAND_PREFIX
            + r"git\s+(?:branch|tag)\s+(?:-[a-z]*d[a-z]*|--delete)(?:\s|$)"
        ),
        "matched destructive git ref deletion",
    ),
)


@dataclass(slots=True, frozen=True)
class BashWorkingDirectory:
    resolved: Path
    display: str


def resolve_bash_working_directory(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> BashWorkingDirectory:
    workspace = context.workspace_dir.resolve(strict=False)
    raw_value = arguments.get("cwd")
    if raw_value is None:
        candidate = workspace
    elif not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("bash cwd must be a non-empty string when provided.")
    else:
        raw_path = Path(raw_value.strip())
        if raw_path.is_absolute():
            if raw_path == Path("/workspace") or raw_path.is_relative_to(Path("/workspace")):
                candidate = workspace / raw_path.relative_to("/workspace")
            else:
                candidate = raw_path
        else:
            candidate = workspace / raw_path
        candidate = candidate.resolve(strict=False)

    if candidate != workspace and not candidate.is_relative_to(workspace):
        raise ValueError("bash cwd must stay inside /workspace.")
    if not candidate.exists():
        raise ValueError("bash cwd does not exist.")
    if not candidate.is_dir():
        raise ValueError("bash cwd must point to a directory.")

    relative = candidate.relative_to(workspace)
    display = str(Path("/workspace") / relative) if relative.parts else "/workspace"
    return BashWorkingDirectory(resolved=candidate, display=display)


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
        mode = _normalize_mode(arguments.get("mode"))
        if mode not in _MODE_VALUES:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"bash mode must be one of: {', '.join(sorted(_MODE_VALUES))}.",
            )

        if mode in {"status", "tail", "cancel"}:
            if arguments.get("cwd") is not None:
                return ToolPolicyDecision(
                    allowed=False,
                    reason=f"bash cwd is not valid for mode '{mode}'.",
                )
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

        if arguments.get("_disable_auto_promote") not in {None, True}:
            return ToolPolicyDecision(
                allowed=False,
                reason="internal bash auto-promotion control must be true when supplied.",
            )

        try:
            working_directory = resolve_bash_working_directory(arguments, context)
        except ValueError as exc:
            return ToolPolicyDecision(allowed=False, reason=str(exc))

        command = str(arguments.get("command", ""))
        if not command.strip():
            return ToolPolicyDecision(allowed=False, reason="bash command cannot be empty.")
        if "\x00" in command:
            return ToolPolicyDecision(
                allowed=False,
                reason="bash command cannot contain null bytes.",
            )
        if _unmanaged_background_reason(command) is not None:
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "Shell-level backgrounding is not allowed. Use bash mode='background' "
                    "so Jarvis owns the process group, logs, readiness, cancellation, and "
                    "terminal result."
                ),
            )

        hard_deny_reason = _hard_deny_reason(command)
        if hard_deny_reason is not None:
            return ToolPolicyDecision(allowed=False, reason=hard_deny_reason)

        if self._settings.bash_dangerously_skip_permission:
            return ToolPolicyDecision(allowed=True)

        detector_reason = _approval_detector_reason(
            command,
            working_directory=working_directory.display,
        )
        if detector_reason is None:
            return ToolPolicyDecision(allowed=True)

        approved_action = context.approved_action or {}
        if (
            approved_action.get("kind") == "bash_command"
            and approved_action.get("command") == command
            and approved_action.get("cwd", "/workspace") == working_directory.display
        ):
            return ToolPolicyDecision(allowed=True)

        summary = str(arguments.get("approval_summary", "")).strip()
        details = str(arguments.get("approval_details", "")).strip()
        inspection_url = str(arguments.get("inspection_url", "")).strip()
        if not summary:
            summary = "Run a bash command that requires explicit review."
        if not details:
            details = (
                "This command would discard broad workspace or repository state. Review the exact "
                f"command and working directory ({working_directory.display}) before approving."
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
                "cwd": working_directory.display,
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


def _unmanaged_background_reason(command: str) -> str | None:
    """Detect process detachment syntax while ignoring quoted text and comments."""

    operator = background_operator(command)
    if operator is not None:
        return f"background_operator@{operator.offset}"
    masked = masked_shell_syntax(command)
    if _UNMANAGED_PROCESS_COMMAND_PATTERN.search(masked) or _DETACH_WRAPPER_PATTERN.search(
        masked
    ):
        return "detachment_command"
    nested_reason = _nested_shell_background_reason(command)
    if nested_reason is not None:
        return nested_reason
    return None


def _nested_shell_background_reason(command: str) -> str | None:
    """Inspect direct shell/eval wrappers whose quoted payload was masked above."""

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    shell_names = {"bash", "dash", "fish", "ksh", "sh", "zsh"}
    for index, token in enumerate(tokens):
        basename = Path(token).name
        if basename == "eval" and index + 1 < len(tokens):
            if _unmanaged_background_reason(tokens[index + 1]) is not None:
                return "nested_detachment_command"
            continue
        if basename not in shell_names:
            continue
        option_index = index + 1
        while option_index < len(tokens) and tokens[option_index].startswith("-"):
            option = tokens[option_index]
            if option == "-c" or "c" in option[1:]:
                payload_index = option_index + 1
                if (
                    payload_index < len(tokens)
                    and _unmanaged_background_reason(tokens[payload_index]) is not None
                ):
                    return "nested_detachment_command"
                break
            option_index += 1
    return None


def _hard_deny_reason(command: str) -> str | None:
    lowered = command.lower()
    for pattern, reason in _HARD_DENY_PATTERNS:
        if pattern.search(lowered):
            return reason
    return None


def _approval_detector_reason(
    command: str,
    *,
    working_directory: str = "/workspace",
) -> str | None:
    lowered = command.strip().lower()
    if not lowered:
        return None

    for pattern, reason in _DESTRUCTIVE_WORKSPACE_PATTERNS:
        if pattern.search(lowered):
            return reason
    return _broad_recursive_delete_reason(
        command,
        working_directory=working_directory,
    )


def _broad_recursive_delete_reason(
    command: str,
    *,
    working_directory: str,
) -> str | None:
    """Recognize only high-confidence deletion of a broad functional root."""

    for segment in re.split(r"(?:&&|\|\||[;|\n])", command):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            continue
        if not tokens:
            continue
        rm_tokens = _recursive_delete_arguments(tokens)
        if rm_tokens is None:
            continue
        recursive = any(
            token == "--recursive"
            or (token.startswith("-") and "r" in token[1:].lower())
            for token in rm_tokens
        )
        if not recursive:
            continue
        targets = [token for token in rm_tokens if not token.startswith("-")]
        for target in targets:
            if target in {"$PWD", "${PWD}", "$PWD/*", "${PWD}/*"}:
                return "matched recursive deletion of the current working directory"
            resolved = (
                posixpath.normpath(target)
                if target.startswith("/")
                else posixpath.normpath(posixpath.join(working_directory, target))
            )
            if resolved in {"/", "/workspace", posixpath.normpath(working_directory)}:
                return f"matched recursive deletion of functional root {resolved}"
            if resolved in {
                "/*",
                "/workspace/*",
                posixpath.join(posixpath.normpath(working_directory), "*"),
            }:
                return f"matched recursive deletion of functional root contents {resolved}"
    return None


def _recursive_delete_arguments(tokens: list[str]) -> list[str] | None:
    """Return arguments only when the segment unambiguously invokes ``rm``."""

    index = 0
    while index < len(tokens) and _looks_like_environment_assignment(tokens[index]):
        index += 1
    while index < len(tokens):
        executable = Path(tokens[index]).name
        if executable == "sudo":
            index = _skip_sudo_prefix(tokens, index + 1)
            if index < 0:
                return None
            continue
        if executable == "env":
            index += 1
            while index < len(tokens) and (
                tokens[index] in {"-i", "--ignore-environment"}
                or _looks_like_environment_assignment(tokens[index])
            ):
                index += 1
            continue
        if executable in {"command", "exec"}:
            index += 1
            while index < len(tokens) and tokens[index] in {"--", "-p"}:
                index += 1
            continue
        break
    if index >= len(tokens) or Path(tokens[index]).name != "rm":
        return None
    return tokens[index + 1 :]


def _skip_sudo_prefix(tokens: list[str], index: int) -> int:
    options_with_values = {
        "-C",
        "-D",
        "-g",
        "-h",
        "-p",
        "-R",
        "-T",
        "-u",
        "--chdir",
        "--close-from",
        "--group",
        "--host",
        "--prompt",
        "--role",
        "--type",
        "--user",
    }
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index]
        if option == "--":
            return index + 1
        option_name = option.split("=", 1)[0]
        index += 1
        if option_name in options_with_values and "=" not in option:
            if index >= len(tokens):
                return -1
            index += 1
    return index


def _looks_like_environment_assignment(token: str) -> bool:
    name, separator, _value = token.partition("=")
    return bool(separator and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))
