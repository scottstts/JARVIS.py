"""Bash tool definition and sandboxed execution runtime."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from pathlib import Path
from time import perf_counter
from typing import Any

from llm import ToolDefinition

from ...config import ToolSettings
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult

_BWRAP_EXECUTABLE = "bwrap"
_SANDBOX_WORKSPACE = "/workspace"
_SANDBOX_TMPDIR = "/tmp"
_SANDBOX_PATH = "/usr/local/bin:/usr/bin:/bin"
_BIND_RW_TOP_LEVEL_PATHS = (
    Path("/usr"),
    Path("/etc"),
    Path("/opt"),
    Path("/var"),
    Path("/root"),
    Path("/run"),
    Path("/home"),
    Path("/srv"),
    Path("/mnt"),
    Path("/media"),
)
_MASKED_DIRECTORIES = (
    Path("/run/secrets"),
)


class BashToolSetupError(RuntimeError):
    """Raised when the bash sandbox cannot be prepared safely."""


class BashToolExecutor:
    """Runs bash commands inside a dedicated bubblewrap sandbox."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        command = str(arguments["command"])
        timeout_seconds = self._resolve_timeout_seconds(arguments)
        started_at = perf_counter()

        try:
            runtime_tmp_dir = context.workspace_dir / ".jarvis_internal" / "bash_tmp"
            runtime_tmp_dir.mkdir(parents=True, exist_ok=True)
            process = await asyncio.create_subprocess_exec(
                *self._build_bwrap_command(
                    workspace_source_dir=context.workspace_dir,
                    runtime_tmp_dir=runtime_tmp_dir,
                    command=command,
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except BashToolSetupError as exc:
            duration_seconds = perf_counter() - started_at
            return ToolExecutionResult(
                call_id=call_id,
                name="bash",
                ok=False,
                content=(
                    "Bash sandbox setup failed\n"
                    f"error: {exc}"
                ),
                metadata={
                    "setup_failed": True,
                    "error": str(exc),
                    "duration_seconds": round(duration_seconds, 3),
                },
            )

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            timed_out = True
            _kill_process_group(process.pid)
            stdout_bytes, stderr_bytes = await process.communicate()

        duration_seconds = perf_counter() - started_at
        stdout_text, stdout_truncated = _truncate_text(
            stdout_bytes.decode("utf-8", errors="replace"),
            self._settings.bash_max_output_chars,
        )
        stderr_text, stderr_truncated = _truncate_text(
            stderr_bytes.decode("utf-8", errors="replace"),
            self._settings.bash_max_output_chars,
        )

        exit_code = process.returncode
        if exit_code is None:
            exit_code = -1

        ok = not timed_out and exit_code == 0
        content = _format_bash_result(
            command=command,
            cwd=Path(_SANDBOX_WORKSPACE),
            exit_code=exit_code,
            timed_out=timed_out,
            timeout_seconds=timeout_seconds,
            duration_seconds=duration_seconds,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

        metadata = {
            "command": command,
            "cwd": _SANDBOX_WORKSPACE,
            "workspace_source_dir": str(context.workspace_dir),
            "sandbox": "bubblewrap",
            "filesystem_scope": "blacklist",
            "environment_scrubbed": True,
            "approval_consumed": bool(context.approved_action),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "timeout_seconds": timeout_seconds,
            "duration_seconds": round(duration_seconds, 3),
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        return ToolExecutionResult(
            call_id=call_id,
            name="bash",
            ok=ok,
            content=content,
            metadata=metadata,
        )

    def _build_bwrap_command(
        self,
        *,
        workspace_source_dir: Path,
        runtime_tmp_dir: Path,
        command: str,
    ) -> list[str]:
        bwrap_path = shutil.which(_BWRAP_EXECUTABLE)
        if bwrap_path is None:
            raise BashToolSetupError("bubblewrap is not installed.")

        resolved_workspace = workspace_source_dir.resolve(strict=True)
        resolved_runtime_tmp = runtime_tmp_dir.resolve(strict=True)
        bash_path = Path(self._settings.bash_executable).resolve(strict=True)

        command_parts = [
            bwrap_path,
            "--die-with-parent",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--uid",
            str(os.getuid()),
            "--gid",
            str(os.getgid()),
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--setenv",
            "PATH",
            _SANDBOX_PATH,
            "--setenv",
            "HOME",
            _SANDBOX_WORKSPACE,
            "--setenv",
            "PWD",
            _SANDBOX_WORKSPACE,
            "--setenv",
            "TMPDIR",
            _SANDBOX_TMPDIR,
            "--setenv",
            "TMP",
            _SANDBOX_TMPDIR,
            "--setenv",
            "TEMP",
            _SANDBOX_TMPDIR,
            "--setenv",
            "LANG",
            "C",
            "--setenv",
            "LC_ALL",
            "C",
            "--bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/lib",
            "/lib",
            "--dev",
            "/dev",
        ]

        lib64_path = Path("/lib64")
        if lib64_path.exists():
            command_parts.extend(
                [
                    "--bind",
                    str(lib64_path),
                    str(lib64_path),
                ]
            )

        for top_level_path in _BIND_RW_TOP_LEVEL_PATHS:
            if top_level_path == Path("/usr"):
                continue
            if not top_level_path.exists():
                continue
            command_parts.extend(
                [
                    "--bind",
                    str(top_level_path),
                    str(top_level_path),
                ]
            )

        command_parts.extend(
            [
                "--bind",
                str(resolved_workspace),
                _SANDBOX_WORKSPACE,
                "--bind",
                str(resolved_runtime_tmp),
                _SANDBOX_TMPDIR,
            ]
        )

        for masked_dir in _MASKED_DIRECTORIES:
            command_parts.extend(
                [
                    "--tmpfs",
                    str(masked_dir),
                    "--chmod",
                    "000",
                    str(masked_dir),
                ]
            )

        command_parts.extend(
            [
                "--chdir",
                _SANDBOX_WORKSPACE,
                str(bash_path),
                "--noprofile",
                "--norc",
                "-lc",
                f"set -o pipefail\n{command}",
            ]
        )
        return command_parts

    def _resolve_timeout_seconds(self, arguments: dict[str, Any]) -> float:
        raw_timeout = arguments.get("timeout_seconds")
        if raw_timeout is None:
            return self._settings.bash_default_timeout_seconds
        timeout_seconds = float(raw_timeout)
        if timeout_seconds < 1:
            timeout_seconds = 1.0
        if timeout_seconds > self._settings.bash_max_timeout_seconds:
            timeout_seconds = self._settings.bash_max_timeout_seconds
        return timeout_seconds


def build_bash_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the default bash tool registry entry."""

    return RegisteredTool(
        name="bash",
        exposure="basic",
        definition=ToolDefinition(
            name="bash",
            description=_build_bash_tool_description(),
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


def _build_bash_tool_description() -> str:
    return (
        "Run a bash command from /workspace. "
        "Use this for shell commands, CLI tools, installs, builds, file inspection, and small scripts. "
        "Use normal shell syntax, including pipes, redirects, command substitution, &&, ||, and multiline scripts. "
        "Some commands may require user approval, so when that seems likely, provide clear approval context. "
        "Available commands depend on what exists in the current runtime. "
        "If you install or create a reusable tool, consider registering it with tool_register."
    )


def _format_bash_result(
    *,
    command: str,
    cwd: Path,
    exit_code: int,
    timed_out: bool,
    timeout_seconds: float,
    duration_seconds: float,
    stdout_text: str,
    stderr_text: str,
    stdout_truncated: bool,
    stderr_truncated: bool,
) -> str:
    status = "timeout" if timed_out else ("success" if exit_code == 0 else "error")
    lines = [
        "Bash execution result",
        f"status: {status}",
        f"command: {command}",
        f"cwd: {cwd}",
        f"exit_code: {exit_code}",
        f"duration_seconds: {duration_seconds:.3f}",
    ]
    if timed_out:
        lines.append(f"timeout_seconds: {timeout_seconds:.3f}")

    lines.extend(
        [
            "stdout:",
            stdout_text or "(empty)",
        ]
    )
    if stdout_truncated:
        lines.append("[stdout truncated]")

    lines.extend(
        [
            "stderr:",
            stderr_text or "(empty)",
        ]
    )
    if stderr_truncated:
        lines.append("[stderr truncated]")
    return "\n".join(lines)


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    truncated = f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"
    return truncated, True


def _kill_process_group(process_id: int | None) -> None:
    if process_id is None:
        return
    try:
        os.killpg(process_id, signal.SIGKILL)
    except ProcessLookupError:
        return
