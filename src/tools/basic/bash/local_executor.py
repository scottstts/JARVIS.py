"""Local bash execution helpers shared by the app and tool_runtime service."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from time import perf_counter
from typing import Any

from ...config import ToolSettings
from ...types import ToolExecutionContext, ToolExecutionResult

_DEFAULT_RUNTIME_PATH = "/usr/local/bin:/usr/bin:/bin"


class DirectBashToolExecutor:
    """Runs bash commands directly inside the active container runtime."""

    def __init__(
        self,
        settings: ToolSettings,
        *,
        target_runtime: str,
        runtime_location: str,
        runtime_transport: str,
        container_mutation_boundary: str,
    ) -> None:
        self._settings = settings
        self._target_runtime = target_runtime
        self._runtime_location = runtime_location
        self._runtime_transport = runtime_transport
        self._container_mutation_boundary = container_mutation_boundary

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        command = str(arguments["command"])
        timeout_seconds = _resolve_timeout_seconds(arguments, self._settings)
        started_at = perf_counter()

        process = await asyncio.create_subprocess_exec(
            self._settings.bash_executable,
            "--noprofile",
            "--norc",
            "-lc",
            f"set -o pipefail\n{command}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(context.workspace_dir),
            env=_build_scrubbed_environment(self._settings),
            start_new_session=True,
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

        exit_code = process.returncode if process.returncode is not None else -1
        ok = not timed_out and exit_code == 0
        content = _format_bash_result(
            command=command,
            cwd=Path("/workspace"),
            exit_code=exit_code,
            timed_out=timed_out,
            timeout_seconds=timeout_seconds,
            duration_seconds=duration_seconds,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

        return ToolExecutionResult(
            call_id=call_id,
            name="bash",
            ok=ok,
            content=content,
            metadata={
                "command": command,
                "cwd": "/workspace",
                "workspace_source_dir": str(context.workspace_dir),
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
                "runtime_location": self._runtime_location,
                "runtime_transport": self._runtime_transport,
                "target_runtime": self._target_runtime,
                "filesystem_scope": "container_direct",
                "container_mutation_boundary": self._container_mutation_boundary,
            },
        )


def _build_scrubbed_environment(settings: ToolSettings) -> dict[str, str]:
    venv_root = str(settings.python_interpreter_venv)
    venv_bin = f"{venv_root}/bin"
    runtime_path = os.getenv("PATH") or _DEFAULT_RUNTIME_PATH
    path_entries = [entry for entry in runtime_path.split(":") if entry and entry != venv_bin]
    path_value = ":".join([venv_bin, *path_entries]) if path_entries else venv_bin
    return {
        "PATH": path_value,
        "HOME": "/workspace",
        "PWD": "/workspace",
        "TMPDIR": "/tmp",
        "TMP": "/tmp",
        "TEMP": "/tmp",
        "LANG": "C",
        "LC_ALL": "C",
        "VIRTUAL_ENV": venv_root,
        "UV_PROJECT_ENVIRONMENT": venv_root,
        "PIP_REQUIRE_VIRTUALENV": "1",
    }


def _resolve_timeout_seconds(arguments: dict[str, Any], settings: ToolSettings) -> float:
    raw_timeout = arguments.get("timeout_seconds")
    if raw_timeout is None:
        return settings.bash_default_timeout_seconds
    timeout_seconds = float(raw_timeout)
    if timeout_seconds < 1:
        timeout_seconds = 1.0
    if timeout_seconds > settings.bash_max_timeout_seconds:
        timeout_seconds = settings.bash_max_timeout_seconds
    return timeout_seconds


def format_bash_tool_description() -> str:
    return (
        "Run a bash command from /workspace inside the isolated tool_runtime container. "
        "Use this for shell commands, CLI tools, installs, builds, file inspection, and small scripts. "
        "Do not use this tool to run code through an interpreter. "
        "Do not use bash tool to invoke Python in any form; use the dedicated interpreter tool (`python_interpreter`) instead. "
        "You can directly install python packages using bash tool, use the virtualenv pip executable directly, "
        "e.g., `uv pip install --python /opt/venv/bin/python <package-name>`. Do NOT use `python -m pip install`. "
        "Use normal shell syntax, including pipes, redirects, command substitution, &&, ||, and multiline scripts. "
        "The shared /workspace is mounted in this runtime. "
        "Some commands may require user approval, so when that seems likely, provide clear approval context. "
        "Available commands depend on what exists in the current tool runtime. "
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
