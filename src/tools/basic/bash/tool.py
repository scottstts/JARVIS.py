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
_RUNTIME_ETC = Path("/etc")


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
            "filesystem_scope": "workspace_only",
            "environment_scrubbed": True,
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
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/lib",
            "/lib",
        ]

        lib64_path = Path("/lib64")
        if lib64_path.exists():
            command_parts.extend(
                [
                    "--ro-bind",
                    str(lib64_path),
                    str(lib64_path),
                ]
            )

        if _RUNTIME_ETC.exists():
            command_parts.extend(
                [
                    "--ro-bind",
                    str(_RUNTIME_ETC),
                    str(_RUNTIME_ETC),
                ]
            )

        command_parts.extend(
            [
                "--dev",
                "/dev",
                "--bind",
                str(resolved_workspace),
                _SANDBOX_WORKSPACE,
                "--bind",
                str(resolved_runtime_tmp),
                _SANDBOX_TMPDIR,
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
            description=_build_bash_tool_description(settings),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Bash command to run inside a bubblewrap sandbox. "
                            "The workspace is mounted at /workspace and is the only "
                            "user-controlled filesystem tree available."
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
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        ),
        executor=BashToolExecutor(settings),
    )


def _build_bash_tool_description(settings: ToolSettings) -> str:
    return (
        "Run a bash command inside a bubblewrap sandbox. "
        "The sandbox mounts the real workspace at /workspace, scrubs the environment, "
        "mounts system runtime paths like /usr and /etc read-only, skips shell startup files, "
        "and does not expose /repo or /run/secrets. "
        f"Default working directory is /workspace; the real workspace source is {settings.workspace_dir}. "
        "Examples of available command-line tools in the current runtime include rg, grep, find, file, curl, zip, and unzip. "
        "Use normal shell syntax, including pipes, redirects, command substitution, subshells, "
        "&&, ||, and multiline scripts. "
        "Installed command availability depends on what exists in the runtime."
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
