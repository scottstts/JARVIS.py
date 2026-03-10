"""Python-interpreter tool definition and sandboxed execution runtime."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from llm import ToolDefinition

from ...config import ToolSettings
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult
from .paths import resolve_workspace_path

_RUNNER_PATH = Path(__file__).with_name("sandbox_runner.py")
_INLINE_SCRIPT_SANDBOX_PATH = "/jarvis-python-inline.py"
_SANDBOX_TMPDIR = "/workspace/.jarvis_internal/tmp"
_MAX_ARGS = 32
_MAX_ARG_CHARS = 512


class PythonInterpreterSetupError(RuntimeError):
    """Raised when the sandbox cannot be prepared safely."""


@dataclass(slots=True, frozen=True)
class PythonInterpreterInvocation:
    code: str | None
    script_relative_path: Path | None
    args: tuple[str, ...]
    declared_read_paths: tuple[str, ...]
    declared_write_paths: tuple[str, ...]
    timeout_seconds: float


@dataclass(slots=True, frozen=True)
class PreparedSandbox:
    sandbox_script_path: str
    inline_script_path: Path | None


class PythonInterpreterToolExecutor:
    """Runs constrained Python code inside a dedicated bubblewrap sandbox."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

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
            interpreter_path = self._settings.python_interpreter_venv / "bin" / "python"
            if not interpreter_path.exists():
                raise PythonInterpreterSetupError(
                    "Configured python_interpreter venv is missing its python binary.",
                )

            with tempfile.TemporaryDirectory(prefix="jarvis-python-interpreter-") as tmp_dir_name:
                temp_dir = Path(tmp_dir_name)
                prepared = _prepare_sandbox(
                    invocation=invocation,
                    context=context,
                    temp_dir=temp_dir,
                )
                config_path = temp_dir / "runner-config.json"
                config_path.write_text(
                    json.dumps(
                        {
                            "script_path": prepared.sandbox_script_path,
                            "args": list(invocation.args),
                            "allowed_packages": list(
                                self._settings.python_interpreter_allowed_packages
                            ),
                            "blocked_import_roots": [
                                "_ctypes",
                                "_cffi_backend",
                                "cffi",
                                "ctypes",
                            ],
                            "memory_limit_bytes": (
                                self._settings.python_interpreter_memory_limit_bytes
                            ),
                            "file_size_limit_bytes": (
                                self._settings.python_interpreter_file_size_limit_bytes
                            ),
                            "cpu_time_limit_seconds": max(
                                1,
                                int(invocation.timeout_seconds),
                            ),
                        }
                    ),
                    encoding="utf-8",
                )

                process = await asyncio.create_subprocess_exec(
                    *self._build_bwrap_command(
                        interpreter_path=interpreter_path,
                        workspace_source_dir=context.workspace_dir,
                        runner_config_path=config_path,
                        inline_script_path=prepared.inline_script_path,
                    ),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
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
                },
            )

        duration_seconds = perf_counter() - started_at
        exit_code = process.returncode
        if exit_code is None:
            exit_code = -1

        stdout_text, stdout_truncated = _truncate_text(
            stdout_bytes.decode("utf-8", errors="replace"),
            self._settings.python_interpreter_max_output_chars,
        )
        stderr_text, stderr_truncated = _truncate_text(
            stderr_bytes.decode("utf-8", errors="replace"),
            self._settings.python_interpreter_max_output_chars,
        )

        source_label = (
            prepared.sandbox_script_path
            if invocation.script_relative_path is not None
            else "inline code"
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
                "allowed_packages": list(self._settings.python_interpreter_allowed_packages),
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
            },
        )

    def _build_bwrap_command(
        self,
        *,
        interpreter_path: Path,
        workspace_source_dir: Path,
        runner_config_path: Path,
        inline_script_path: Path | None,
    ) -> list[str]:
        resolved_python = interpreter_path.resolve(strict=True)
        resolved_workspace = workspace_source_dir.resolve(strict=True)
        python_lib_dir = resolved_python.parent.parent / "lib"
        if not python_lib_dir.exists():
            raise PythonInterpreterSetupError(
                f"Python runtime library directory is missing: {python_lib_dir}",
            )

        command = [
            "bwrap",
            "--die-with-parent",
            "--unshare-user",
            "--unshare-net",
            "--uid",
            str(os.getuid()),
            "--gid",
            str(os.getgid()),
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--setenv",
            "HOME",
            "/workspace",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "TMPDIR",
            _SANDBOX_TMPDIR,
            "--setenv",
            "TEMP",
            _SANDBOX_TMPDIR,
            "--setenv",
            "TMP",
            _SANDBOX_TMPDIR,
            "--ro-bind",
            str(python_lib_dir),
            str(python_lib_dir),
            "--ro-bind",
            str(resolved_python),
            str(resolved_python),
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            str(self._settings.python_interpreter_venv),
            str(self._settings.python_interpreter_venv),
        ]

        ld_so_cache = Path("/etc/ld.so.cache")
        if ld_so_cache.exists():
            command.extend(
                [
                    "--ro-bind",
                    str(ld_so_cache),
                    str(ld_so_cache),
                ]
            )

        command.extend(
            [
                "--ro-bind",
                str(_RUNNER_PATH),
                "/jarvis-python-runner.py",
                "--ro-bind",
                str(runner_config_path),
                "/jarvis-python-runner-config.json",
                "--dir",
                "/tmp",
                "--chmod",
                "0555",
                "/tmp",
                "--bind",
                str(resolved_workspace),
                "/workspace",
            ]
        )

        if inline_script_path is not None:
            command.extend(
                [
                    "--ro-bind",
                    str(inline_script_path),
                    _INLINE_SCRIPT_SANDBOX_PATH,
                ]
            )

        command.extend(
            [
                "--chdir",
                "/workspace",
                str(interpreter_path),
                "/jarvis-python-runner.py",
                "/jarvis-python-runner-config.json",
            ]
        )
        return command


def build_python_interpreter_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the sandboxed python_interpreter tool registry entry."""

    return RegisteredTool(
        name="python_interpreter",
        exposure="basic",
        definition=ToolDefinition(
            name="python_interpreter",
            description=_build_python_interpreter_tool_description(settings),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": ["string", "null"],
                        "description": (
                            "Inline Python code to execute. Provide exactly one of "
                            "'code' or 'script_path'."
                        ),
                    },
                    "script_path": {
                        "type": ["string", "null"],
                        "description": (
                            "Workspace path to a stored Python script to run through "
                            "the sandbox. Provide exactly one of 'code' or 'script_path'."
                        ),
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "maxLength": _MAX_ARG_CHARS,
                        },
                        "maxItems": _MAX_ARGS,
                        "description": (
                            "Optional positional args exposed to the script as sys.argv[1:]."
                        ),
                    },
                    "read_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": settings.python_interpreter_max_paths,
                        "description": (
                            "Deprecated no-op kept for compatibility. "
                            "The real workspace is mounted directly at /workspace."
                        ),
                    },
                    "write_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": settings.python_interpreter_max_paths,
                        "description": (
                            "Deprecated no-op kept for compatibility. "
                            "Scripts may write directly inside /workspace."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": settings.python_interpreter_max_timeout_seconds,
                        "description": (
                            "Optional execution timeout in seconds. Use only when the script "
                            "legitimately needs longer than the default."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        ),
        executor=PythonInterpreterToolExecutor(settings),
    )


def _build_python_interpreter_tool_description(settings: ToolSettings) -> str:
    packages = ", ".join(settings.python_interpreter_allowed_packages)
    return (
        "Run constrained Python inside a bubblewrap sandbox backed by a dedicated venv. "
        "Use this for parsing, transformations, tabular processing, PDF/text extraction, "
        "image work, and more, that are awkward in shell. "
        "Exactly one of 'code' or 'script_path' is required. "
        "If you anticipate long script, write the Python code as a file first, "
        "and then call this tool to execute the script as a .py file. "
        "When you run into snags with using the libraries and it is necessary to use them for the task, "
        "try using web search for docs or example uses to get the correct usage. "
        "The real workspace is mounted at /workspace and is the only writable filesystem "
        "location available to the script. Writes outside /workspace are denied by the sandbox. "
        "Workspace helper modules and curated venv packages import normally; direct native FFI "
        "imports such as ctypes/cffi are blocked. "
        f"Curated third-party packages available: {packages}. "
        "No network access is available."
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

    script_relative_path: Path | None = None
    if script_path is not None:
        _, script_relative_path = resolve_workspace_path(
            script_path,
            context=context,
            require_exists=True,
        )

    return PythonInterpreterInvocation(
        code=code,
        script_relative_path=script_relative_path,
        args=args,
        declared_read_paths=read_paths,
        declared_write_paths=write_paths,
        timeout_seconds=_resolve_timeout_seconds(arguments, settings),
    )


def _prepare_sandbox(
    *,
    invocation: PythonInterpreterInvocation,
    context: ToolExecutionContext,
    temp_dir: Path,
) -> PreparedSandbox:
    runtime_tmp_dir = context.workspace_dir / ".jarvis_internal" / "tmp"
    runtime_tmp_dir.mkdir(parents=True, exist_ok=True)

    if invocation.code is not None:
        inline_code_path = temp_dir / "inline_code.py"
        inline_code_path.write_text(invocation.code, encoding="utf-8")
        return PreparedSandbox(
            sandbox_script_path=_INLINE_SCRIPT_SANDBOX_PATH,
            inline_script_path=inline_code_path,
        )

    if invocation.script_relative_path is None:
        raise PythonInterpreterSetupError(
            "python_interpreter requires either inline code or a script path.",
        )

    return PreparedSandbox(
        sandbox_script_path=_sandbox_path(invocation.script_relative_path),
        inline_script_path=None,
    )


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


def _sandbox_path(relative_path: Path) -> str:
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
