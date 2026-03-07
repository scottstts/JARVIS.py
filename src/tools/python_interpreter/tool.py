"""Python-interpreter tool definition and sandboxed execution runtime."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from llm import ToolDefinition

from ..config import ToolSettings
from ..types import RegisteredTool, ToolExecutionContext, ToolExecutionResult
from .paths import resolve_workspace_path, should_skip_relative_path

_SANDBOX_UID = 65_534
_SANDBOX_GID = 65_534
_INLINE_CODE_RELATIVE_PATH = Path(".jarvis_internal") / "inline_code.py"
_RUNNER_PATH = Path(__file__).with_name("sandbox_runner.py")
_MAX_ARGS = 32
_MAX_ARG_CHARS = 512


class PythonInterpreterSetupError(RuntimeError):
    """Raised when the sandbox cannot be prepared safely."""


@dataclass(slots=True, frozen=True)
class PythonInterpreterInvocation:
    code: str | None
    script_relative_path: Path | None
    args: tuple[str, ...]
    read_relative_paths: tuple[Path, ...]
    write_relative_paths: tuple[Path, ...]
    timeout_seconds: float


@dataclass(slots=True, frozen=True)
class PreparedSandbox:
    workspace_dir: Path
    sandbox_script_path: str
    staged_bytes: int
    sync_write_relative_paths: tuple[Path, ...]


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
                    settings=self._settings,
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
                                "_posixsubprocess",
                                "_socket",
                                "ctypes",
                                "ensurepip",
                                "multiprocessing",
                                "pip",
                                "pty",
                                "socket",
                                "subprocess",
                                "venv",
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
                        staged_workspace_dir=prepared.workspace_dir,
                        runner_config_path=config_path,
                        write_relative_paths=prepared.sync_write_relative_paths,
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

                sync_error: str | None = None
                synced_paths: list[str] = []
                output_bytes = 0
                try:
                    output_bytes = _measure_output_bytes(
                        stage_workspace_dir=prepared.workspace_dir,
                        write_relative_paths=prepared.sync_write_relative_paths,
                        max_bytes=self._settings.python_interpreter_max_staged_bytes,
                    )
                    synced_paths = _sync_outputs_back_to_workspace(
                        stage_workspace_dir=prepared.workspace_dir,
                        workspace_dir=context.workspace_dir,
                        write_relative_paths=prepared.sync_write_relative_paths,
                    )
                except PythonInterpreterSetupError as exc:
                    sync_error = str(exc)

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
        if sync_error is not None:
            stderr_text = (
                f"{stderr_text}\n[sync error] {sync_error}"
                if stderr_text
                else f"[sync error] {sync_error}"
            )

        ok = not timed_out and exit_code == 0 and sync_error is None
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
            synced_paths=synced_paths,
        )
        return ToolExecutionResult(
            call_id=call_id,
            name="python_interpreter",
            ok=ok,
            content=content,
            metadata={
                "source": source_label,
                "args": list(invocation.args),
                "read_paths": [_sandbox_path(path) for path in invocation.read_relative_paths],
                "write_paths": [_sandbox_path(path) for path in invocation.write_relative_paths],
                "synced_write_paths": synced_paths,
                "allowed_packages": list(self._settings.python_interpreter_allowed_packages),
                "cwd": "/workspace",
                "exit_code": exit_code,
                "timed_out": timed_out,
                "timeout_seconds": invocation.timeout_seconds,
                "duration_seconds": round(duration_seconds, 3),
                "staged_bytes": prepared.staged_bytes,
                "output_bytes": output_bytes,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "sync_error": sync_error,
            },
        )

    def _build_bwrap_command(
        self,
        *,
        interpreter_path: Path,
        staged_workspace_dir: Path,
        runner_config_path: Path,
        write_relative_paths: tuple[Path, ...],
    ) -> list[str]:
        resolved_python = interpreter_path.resolve(strict=True)
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
            str(_SANDBOX_UID),
            "--gid",
            str(_SANDBOX_GID),
            "--clearenv",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
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
                "--dev-bind",
                "/dev/null",
                "/dev/null",
                "--tmpfs",
                "/tmp",
                "--ro-bind",
                str(staged_workspace_dir),
                "/workspace",
            ]
        )
        for relative_path in write_relative_paths:
            command.extend(
                [
                    "--bind",
                    str(staged_workspace_dir / relative_path),
                    _sandbox_path(relative_path),
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
                            "Explicit workspace files or directories to stage into the sandbox "
                            "as read-only inputs. Protected paths and .env paths are denied."
                        ),
                    },
                    "write_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": settings.python_interpreter_max_paths,
                        "description": (
                            "Explicit existing workspace files or directories to stage as "
                            "writable outputs. To create new files, declare an existing "
                            "writable directory rather than a missing file path."
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
        "image work, and small scripts that are awkward in shell. "
        "Exactly one of 'code' or 'script_path' is required. "
        "Filesystem access is limited to explicitly declared workspace read_paths and "
        "write_paths; protected workspace paths and .env paths are denied. "
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

    read_relative_paths = tuple(
        resolve_workspace_path(raw_path, context=context, require_exists=True)[1]
        for raw_path in read_paths
    )
    write_relative_paths = tuple(
        resolve_workspace_path(raw_path, context=context, require_exists=True)[1]
        for raw_path in write_paths
    )

    return PythonInterpreterInvocation(
        code=code,
        script_relative_path=script_relative_path,
        args=args,
        read_relative_paths=read_relative_paths,
        write_relative_paths=write_relative_paths,
        timeout_seconds=_resolve_timeout_seconds(arguments, settings),
    )


def _prepare_sandbox(
    *,
    invocation: PythonInterpreterInvocation,
    context: ToolExecutionContext,
    settings: ToolSettings,
    temp_dir: Path,
) -> PreparedSandbox:
    stage_workspace_dir = temp_dir / "workspace"
    stage_workspace_dir.mkdir()

    copy_roots = _collapse_relative_paths(
        [
            *invocation.read_relative_paths,
            *invocation.write_relative_paths,
            *(
                []
                if invocation.script_relative_path is None
                else [invocation.script_relative_path]
            ),
        ]
    )
    staged_bytes = 0
    for relative_path in copy_roots:
        source_path = context.workspace_dir / relative_path
        staged_bytes = _copy_path_into_stage(
            source_path=source_path,
            relative_path=relative_path,
            stage_workspace_dir=stage_workspace_dir,
            staged_bytes=staged_bytes,
            max_bytes=settings.python_interpreter_max_staged_bytes,
        )

    sandbox_script_path: str
    if invocation.code is not None:
        inline_code_path = stage_workspace_dir / _INLINE_CODE_RELATIVE_PATH
        inline_code_path.parent.mkdir(parents=True, exist_ok=True)
        inline_code_path.write_text(invocation.code, encoding="utf-8")
        staged_bytes += _enforce_staged_size_limit(
            current_size=staged_bytes,
            additional_bytes=len(invocation.code.encode("utf-8")),
            max_bytes=settings.python_interpreter_max_staged_bytes,
        )
        sandbox_script_path = _sandbox_path(_INLINE_CODE_RELATIVE_PATH)
    else:
        if invocation.script_relative_path is None:
            raise PythonInterpreterSetupError(
                "python_interpreter requires either inline code or a script path.",
            )
        sandbox_script_path = _sandbox_path(invocation.script_relative_path)

    for relative_path in _collapse_relative_paths(list(invocation.write_relative_paths)):
        stage_path = stage_workspace_dir / relative_path
        if not stage_path.exists():
            raise PythonInterpreterSetupError(
                f"Writable sandbox path is missing after staging: {_sandbox_path(relative_path)}",
            )
        _make_path_writable(stage_path)

    return PreparedSandbox(
        workspace_dir=stage_workspace_dir,
        sandbox_script_path=sandbox_script_path,
        staged_bytes=staged_bytes,
        sync_write_relative_paths=_collapse_relative_paths(list(invocation.write_relative_paths)),
    )


def _copy_path_into_stage(
    *,
    source_path: Path,
    relative_path: Path,
    stage_workspace_dir: Path,
    staged_bytes: int,
    max_bytes: int,
) -> int:
    if should_skip_relative_path(relative_path):
        return staged_bytes

    if source_path.is_symlink():
        raise PythonInterpreterSetupError(
            f"python_interpreter does not allow symlink inputs: {source_path}",
        )

    if source_path.is_file():
        destination = stage_workspace_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        additional_bytes = source_path.stat().st_size
        staged_bytes += _enforce_staged_size_limit(
            current_size=staged_bytes,
            additional_bytes=additional_bytes,
            max_bytes=max_bytes,
        )
        shutil.copy2(source_path, destination)
        return staged_bytes

    if not source_path.is_dir():
        raise PythonInterpreterSetupError(
            f"python_interpreter only supports file or directory inputs, got: {source_path}",
        )

    for root, dir_names, file_names in os.walk(source_path, topdown=True, followlinks=False):
        current_source_dir = Path(root)
        current_relative_dir = relative_path / current_source_dir.relative_to(source_path)
        if should_skip_relative_path(current_relative_dir):
            dir_names[:] = []
            continue

        stage_dir = stage_workspace_dir / current_relative_dir
        stage_dir.mkdir(parents=True, exist_ok=True)

        filtered_dir_names: list[str] = []
        for dir_name in dir_names:
            child_source = current_source_dir / dir_name
            child_relative = current_relative_dir / dir_name
            if child_source.is_symlink():
                raise PythonInterpreterSetupError(
                    f"python_interpreter does not allow symlink inputs: {child_source}",
                )
            if should_skip_relative_path(child_relative):
                continue
            filtered_dir_names.append(dir_name)
        dir_names[:] = filtered_dir_names

        for file_name in file_names:
            child_source = current_source_dir / file_name
            child_relative = current_relative_dir / file_name
            if child_source.is_symlink():
                raise PythonInterpreterSetupError(
                    f"python_interpreter does not allow symlink inputs: {child_source}",
                )
            if should_skip_relative_path(child_relative):
                continue

            destination = stage_workspace_dir / child_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            additional_bytes = child_source.stat().st_size
            staged_bytes += _enforce_staged_size_limit(
                current_size=staged_bytes,
                additional_bytes=additional_bytes,
                max_bytes=max_bytes,
            )
            shutil.copy2(child_source, destination)

    return staged_bytes


def _measure_output_bytes(
    *,
    stage_workspace_dir: Path,
    write_relative_paths: tuple[Path, ...],
    max_bytes: int,
) -> int:
    total_bytes = 0
    for relative_path in write_relative_paths:
        stage_path = stage_workspace_dir / relative_path
        total_bytes += _measure_path_bytes(stage_path)
        if total_bytes > max_bytes:
            raise PythonInterpreterSetupError(
                "python_interpreter output exceeded the staged workspace byte limit.",
            )
    return total_bytes


def _measure_path_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_symlink():
        raise PythonInterpreterSetupError(
            f"python_interpreter sandbox outputs may not contain symlinks: {path}",
        )
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0

    total_bytes = 0
    for root, dir_names, file_names in os.walk(path, topdown=True, followlinks=False):
        current_dir = Path(root)
        filtered_dir_names: list[str] = []
        for dir_name in dir_names:
            child_dir = current_dir / dir_name
            if child_dir.is_symlink():
                raise PythonInterpreterSetupError(
                    f"python_interpreter sandbox outputs may not contain symlinks: {child_dir}",
                )
            filtered_dir_names.append(dir_name)
        dir_names[:] = filtered_dir_names

        for file_name in file_names:
            child_file = current_dir / file_name
            if child_file.is_symlink():
                raise PythonInterpreterSetupError(
                    f"python_interpreter sandbox outputs may not contain symlinks: {child_file}",
                )
            total_bytes += child_file.stat().st_size
    return total_bytes


def _sync_outputs_back_to_workspace(
    *,
    stage_workspace_dir: Path,
    workspace_dir: Path,
    write_relative_paths: tuple[Path, ...],
) -> list[str]:
    synced_paths: list[str] = []
    for relative_path in write_relative_paths:
        stage_path = stage_workspace_dir / relative_path
        workspace_path = workspace_dir / relative_path
        _sync_stage_path_to_workspace(
            stage_path=stage_path,
            workspace_path=workspace_path,
            relative_path=relative_path,
        )
        synced_paths.append(str(workspace_path))
    return synced_paths


def _sync_stage_path_to_workspace(
    *,
    stage_path: Path,
    workspace_path: Path,
    relative_path: Path,
) -> None:
    if should_skip_relative_path(relative_path):
        return

    if not stage_path.exists():
        _remove_workspace_path(workspace_path)
        return

    if stage_path.is_symlink():
        raise PythonInterpreterSetupError(
            f"python_interpreter sandbox outputs may not contain symlinks: {stage_path}",
        )

    if stage_path.is_file():
        if workspace_path.is_dir() and not workspace_path.is_symlink():
            shutil.rmtree(workspace_path)
        elif workspace_path.is_symlink():
            workspace_path.unlink()
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage_path, workspace_path)
        return

    if not stage_path.is_dir():
        return

    if workspace_path.exists() and not workspace_path.is_dir():
        _remove_workspace_path(workspace_path)
    workspace_path.mkdir(parents=True, exist_ok=True)
    _sync_stage_directory(
        stage_dir=stage_path,
        workspace_dir=workspace_path,
        relative_root=relative_path,
    )


def _sync_stage_directory(
    *,
    stage_dir: Path,
    workspace_dir: Path,
    relative_root: Path,
) -> None:
    if stage_dir.is_symlink():
        raise PythonInterpreterSetupError(
            f"python_interpreter sandbox outputs may not contain symlinks: {stage_dir}",
        )

    stage_entries: dict[str, Path] = {}
    for entry in stage_dir.iterdir():
        child_relative = relative_root / entry.name
        if should_skip_relative_path(child_relative):
            continue
        if entry.is_symlink():
            raise PythonInterpreterSetupError(
                f"python_interpreter sandbox outputs may not contain symlinks: {entry}",
            )
        stage_entries[entry.name] = entry

    for existing_entry in workspace_dir.iterdir():
        child_relative = relative_root / existing_entry.name
        if should_skip_relative_path(child_relative):
            continue
        if existing_entry.name not in stage_entries:
            _remove_workspace_path(existing_entry)

    for name, stage_entry in stage_entries.items():
        child_relative = relative_root / name
        workspace_entry = workspace_dir / name
        if stage_entry.is_dir():
            if workspace_entry.exists() and not workspace_entry.is_dir():
                _remove_workspace_path(workspace_entry)
            workspace_entry.mkdir(parents=True, exist_ok=True)
            _sync_stage_directory(
                stage_dir=stage_entry,
                workspace_dir=workspace_entry,
                relative_root=child_relative,
            )
            continue

        if workspace_entry.is_dir() and not workspace_entry.is_symlink():
            shutil.rmtree(workspace_entry)
        elif workspace_entry.is_symlink():
            workspace_entry.unlink()
        workspace_entry.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage_entry, workspace_entry)


def _remove_workspace_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _set_tree_read_only(path: Path) -> None:
    if path.is_symlink():
        raise PythonInterpreterSetupError(
            f"python_interpreter does not allow symlink paths in the staged workspace: {path}",
        )
    if path.is_dir():
        for child in path.iterdir():
            _set_tree_read_only(child)
        path.chmod(0o555)
        return
    path.chmod(0o444)


def _make_path_writable(path: Path) -> None:
    if path.is_symlink():
        raise PythonInterpreterSetupError(
            f"python_interpreter does not allow symlink outputs: {path}",
        )
    if path.is_dir():
        path.chmod(0o777)
        for child in path.iterdir():
            _make_path_writable(child)
        return
    path.chmod(0o666)


def _collapse_relative_paths(paths: list[Path]) -> tuple[Path, ...]:
    unique_paths = sorted({path for path in paths}, key=lambda path: (len(path.parts), path.as_posix()))
    collapsed: list[Path] = []
    for path in unique_paths:
        if any(path == kept or kept in path.parents for kept in collapsed):
            continue
        collapsed.append(path)
    return tuple(collapsed)


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
    synced_paths: list[str],
) -> str:
    status = "timeout" if timed_out else ("success" if exit_code == 0 else "error")
    lines = [
        "Python interpreter result",
        f"status: {status}",
        f"source: {source}",
        "cwd: /workspace",
        f"exit_code: {exit_code}",
        f"duration_seconds: {duration_seconds:.3f}",
    ]
    if timed_out:
        lines.append(f"timeout_seconds: {timeout_seconds:.3f}")
    if synced_paths:
        lines.append(f"synced_write_paths: {', '.join(synced_paths)}")

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


def _enforce_staged_size_limit(
    *,
    current_size: int,
    additional_bytes: int,
    max_bytes: int,
) -> int:
    new_total = current_size + additional_bytes
    if new_total > max_bytes:
        raise PythonInterpreterSetupError(
            "python_interpreter staging exceeded the maximum allowed byte size.",
        )
    return additional_bytes


def _kill_process_group(process_id: int | None) -> None:
    if process_id is None:
        return
    try:
        os.killpg(process_id, signal.SIGKILL)
    except ProcessLookupError:
        return
