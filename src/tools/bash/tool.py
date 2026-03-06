"""Bash tool definition and execution runtime."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from time import perf_counter
from typing import Any

from llm import ToolDefinition

from ..config import ToolSettings
from ..types import RegisteredTool, ToolExecutionContext, ToolExecutionResult


class BashToolExecutor:
    """Runs validated bash commands inside the agent workspace."""

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

        process = await asyncio.create_subprocess_exec(
            self._settings.bash_executable,
            "-lc",
            f"set -o pipefail\n{command}",
            cwd=str(context.workspace_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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

        exit_code = process.returncode
        if exit_code is None:
            exit_code = -1

        ok = not timed_out and exit_code == 0
        content = _format_bash_result(
            command=command,
            cwd=context.workspace_dir,
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
            "cwd": str(context.workspace_dir),
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
                            "Validated bash command to run inside the container. "
                            "Use standard Linux commands for file inspection and "
                            "workspace-limited file edits."
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
        "Run a validated bash command inside the agent container. "
        f"Default working directory is {settings.workspace_dir}. "
        "Use this for reading files, searching content, and editing files inside the workspace. "
        "Allowed commands: pwd, ls, find, stat, file, du, cat, head, tail, grep, rg, wc, cut, "
        "sort, uniq, diff, printf, echo, mkdir, touch, cp, mv, rm, truncate, tee, sed. "
        "Writes are allowed only inside the workspace path. "
        "Use pipes when needed. Do not use shell control operators like ;, &&, ||, redirects, "
        "subshells, heredocs, or command substitution. "
        "Use printf piped to tee to create or overwrite file content."
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
