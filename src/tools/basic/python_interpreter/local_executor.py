"""Local python_interpreter execution helpers shared by the app and tool_runtime service."""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from ...config import ToolSettings
from ...types import ToolExecutionContext, ToolExecutionResult
from ..bash.local_executor import _build_scrubbed_environment
from .paths import resolve_workspace_path

_MAX_ARGS = 32
_MAX_ARG_CHARS = 512
_INLINE_SCRIPT_NAME = "jarvis-python-inline.py"


class PythonInterpreterSetupError(RuntimeError):
    """Raised when python_interpreter cannot be prepared safely."""


@dataclass(slots=True, frozen=True)
class PythonInterpreterInvocation:
    code: str | None
    script_relative_path: Path | None
    script_path: Path | None
    args: tuple[str, ...]
    declared_read_paths: tuple[str, ...]
    declared_write_paths: tuple[str, ...]
    timeout_seconds: float


class DirectPythonInterpreterToolExecutor:
    """Runs python_interpreter directly inside the active tool runtime."""

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
        started_at = perf_counter()

        try:
            invocation = _build_invocation(
                arguments=arguments,
                context=context,
                settings=self._settings,
            )
            interpreter_path = _interpreter_path(self._settings)
            if not interpreter_path.exists():
                raise PythonInterpreterSetupError(
                    "python_interpreter could not start because the central agent Python "
                    f"environment '{self._settings.python_interpreter_venv}' is missing "
                    f"its interpreter at '{interpreter_path}'. The correct interpreter "
                    f"for agent Python work is '{interpreter_path}'."
                )

            runtime_tmp_dir = context.workspace_dir / ".jarvis_internal" / "tmp"
            runtime_tmp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="jarvis-python-interpreter-",
                dir=runtime_tmp_dir,
            ) as tmp_dir_name:
                script_path, source_label = _prepare_script(
                    invocation=invocation,
                    temp_dir=Path(tmp_dir_name),
                )
                process = await asyncio.create_subprocess_exec(
                    str(interpreter_path),
                    str(script_path),
                    *invocation.args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(context.workspace_dir),
                    env=_build_python_environment(self._settings),
                    start_new_session=True,
                )

                timed_out = False
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(),
                        timeout=invocation.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    _kill_process_group(process.pid)
                    stdout_bytes, stderr_bytes = await process.communicate()
        except PythonInterpreterSetupError as exc:
            duration_seconds = perf_counter() - started_at
            return ToolExecutionResult(
                call_id=call_id,
                name="python_interpreter",
                ok=False,
                content=(
                    "Python interpreter setup failed\n"
                    f"error: {exc}"
                ),
                metadata={
                    "setup_failed": True,
                    "error": str(exc),
                    "duration_seconds": round(duration_seconds, 3),
                    "runtime_location": self._runtime_location,
                    "runtime_transport": self._runtime_transport,
                    "target_runtime": self._target_runtime,
                    "container_mutation_boundary": self._container_mutation_boundary,
                },
            )

        duration_seconds = perf_counter() - started_at
        exit_code = process.returncode if process.returncode is not None else -1
        stdout_text, stdout_truncated = _truncate_text(
            stdout_bytes.decode("utf-8", errors="replace"),
            self._settings.python_interpreter_max_output_chars,
        )
        stderr_text, stderr_truncated = _truncate_text(
            stderr_bytes.decode("utf-8", errors="replace"),
            self._settings.python_interpreter_max_output_chars,
        )

        content = _format_python_result(
            source=source_label,
            exit_code=exit_code,
            timed_out=timed_out,
            timeout_seconds=invocation.timeout_seconds,
            duration_seconds=duration_seconds,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )
        return ToolExecutionResult(
            call_id=call_id,
            name="python_interpreter",
            ok=not timed_out and exit_code == 0,
            content=content,
            metadata={
                "source": source_label,
                "args": list(invocation.args),
                "read_paths": list(invocation.declared_read_paths),
                "write_paths": list(invocation.declared_write_paths),
                "starter_packages": list(self._settings.python_interpreter_starter_packages),
                "venv_root": str(self._settings.python_interpreter_venv),
                "interpreter_path": str(interpreter_path),
                "cwd": "/workspace",
                "workspace_mode": "direct_bind",
                "exit_code": exit_code,
                "timed_out": timed_out,
                "timeout_seconds": invocation.timeout_seconds,
                "duration_seconds": round(duration_seconds, 3),
                "stdout": stdout_text,
                "stderr": stderr_text,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "runtime_location": self._runtime_location,
                "runtime_transport": self._runtime_transport,
                "target_runtime": self._target_runtime,
                "script_path_scope": "/workspace",
                "network_access": True,
                "process_spawn_blocked": False,
                "container_mutation_boundary": self._container_mutation_boundary,
            },
        )

def build_python_interpreter_description(settings: ToolSettings) -> str:
    interpreter_path = _interpreter_path(settings)
    packages = ", ".join(settings.python_interpreter_starter_packages) or "(none)"
    return (
        "Run Python directly inside the isolated tool_runtime container using the central "
        f"`{interpreter_path}` environment. Exactly one of 'code' or 'script_path' is "
        "required. If you anticipate long code, write it to a file in `/workspace` and "
        "execute it through 'script_path'. The shared `/workspace` is mounted directly and "
        "scripts may read or write there normally. Starter packages preinstalled in the "
        f"central venv: {packages}. You may install additional packages into the same venv "
        f"with the `bash` tool, typically `uv pip install --python {interpreter_path} "
        "<package-name>`."
    )


def _build_invocation(
    *,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
    settings: ToolSettings,
) -> PythonInterpreterInvocation:
    code = _normalize_nullable_string(arguments.get("code"))
    script_path = _normalize_nullable_string(arguments.get("script_path"))
    args = tuple(_normalize_string_list(arguments.get("args")))
    read_paths = tuple(_normalize_string_list(arguments.get("read_paths")))
    write_paths = tuple(_normalize_string_list(arguments.get("write_paths")))

    if bool(code) == bool(script_path):
        raise PythonInterpreterSetupError(
            "python_interpreter requires exactly one of 'code' or 'script_path'.",
        )

    resolved_script_path: Path | None = None
    script_relative_path: Path | None = None
    if script_path is not None:
        resolved_script_path, script_relative_path = resolve_workspace_path(
            script_path,
            context=context,
            require_exists=True,
        )

    return PythonInterpreterInvocation(
        code=code,
        script_relative_path=script_relative_path,
        script_path=resolved_script_path,
        args=args,
        declared_read_paths=read_paths,
        declared_write_paths=write_paths,
        timeout_seconds=_resolve_timeout_seconds(arguments, settings),
    )


def _prepare_script(
    *,
    invocation: PythonInterpreterInvocation,
    temp_dir: Path,
) -> tuple[Path, str]:
    if invocation.code is not None:
        inline_path = temp_dir / _INLINE_SCRIPT_NAME
        inline_path.write_text(invocation.code, encoding="utf-8")
        return inline_path, "inline code"

    if invocation.script_path is None or invocation.script_relative_path is None:
        raise PythonInterpreterSetupError(
            "python_interpreter requires either inline code or a script path.",
        )
    return invocation.script_path, _workspace_runtime_path(invocation.script_relative_path)


def _build_python_environment(settings: ToolSettings) -> dict[str, str]:
    environment = _build_scrubbed_environment(settings)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _interpreter_path(settings: ToolSettings) -> Path:
    return settings.python_interpreter_venv / "bin" / "python"


def _resolve_timeout_seconds(arguments: dict[str, Any], settings: ToolSettings) -> float:
    raw_timeout = arguments.get("timeout_seconds")
    if raw_timeout is None:
        return settings.python_interpreter_default_timeout_seconds
    timeout_seconds = float(raw_timeout)
    if timeout_seconds < 1:
        timeout_seconds = 1.0
    if timeout_seconds > settings.python_interpreter_max_timeout_seconds:
        timeout_seconds = settings.python_interpreter_max_timeout_seconds
    return timeout_seconds


def _workspace_runtime_path(relative_path: Path) -> str:
    if relative_path == Path("."):
        return "/workspace"
    return f"/workspace/{relative_path.as_posix()}"


def _normalize_nullable_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [str(value).strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _format_python_result(
    *,
    source: str,
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
        "Python interpreter result",
        f"status: {status}",
        f"source: {source}",
        "cwd: /workspace",
        "workspace_mode: direct_bind",
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
