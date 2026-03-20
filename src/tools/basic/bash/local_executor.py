"""Local bash execution helpers shared by the app and tool_runtime service."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path
from time import perf_counter
from typing import Any

from ...config import ToolSettings
from ...types import ToolExecutionContext, ToolExecutionResult
from .jobs import (
    BashJobError,
    BashJobPaths,
    cancel_job,
    create_background_job,
    job_status,
    load_job,
    read_job_tail,
    remove_job_artifacts,
    write_job_metadata,
)

_DEFAULT_RUNTIME_PATH = "/usr/local/bin:/usr/bin:/bin"
_DEFAULT_MODE = "foreground"


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
        mode = _normalize_mode(arguments.get("mode"))
        if mode == "foreground":
            return await self._run_foreground(
                call_id=call_id,
                arguments=arguments,
                context=context,
            )
        if mode == "background":
            return await self._start_background_job(
                call_id=call_id,
                arguments=arguments,
                context=context,
            )
        if mode == "status":
            return self._background_job_status(
                call_id=call_id,
                arguments=arguments,
                context=context,
            )
        if mode == "tail":
            return self._background_job_tail(
                call_id=call_id,
                arguments=arguments,
                context=context,
            )
        if mode == "cancel":
            return self._background_job_cancel(
                call_id=call_id,
                arguments=arguments,
                context=context,
            )
        return ToolExecutionResult(
            call_id=call_id,
            name="bash",
            ok=False,
            content=(
                "Bash execution request failed\n"
                f"reason: unsupported mode '{mode}'."
            ),
            metadata={"mode": mode},
        )

    async def _run_foreground(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        command = str(arguments.get("command", ""))
        timeout_seconds = _resolve_timeout_seconds(arguments, self._settings)
        soft_timeout_seconds = self._settings.bash_foreground_soft_timeout_seconds
        if timeout_seconds < soft_timeout_seconds:
            return await self._run_plain_foreground(
                call_id=call_id,
                command=command,
                timeout_seconds=timeout_seconds,
                context=context,
            )

        return await self._run_promotable_foreground(
            call_id=call_id,
            command=command,
            timeout_seconds=timeout_seconds,
            soft_timeout_seconds=soft_timeout_seconds,
            context=context,
        )

    async def _run_plain_foreground(
        self,
        *,
        call_id: str,
        command: str,
        timeout_seconds: float,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
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
                "mode": "foreground",
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

    async def _run_promotable_foreground(
        self,
        *,
        call_id: str,
        command: str,
        timeout_seconds: float,
        soft_timeout_seconds: float,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        started_at = perf_counter()
        try:
            job, process, pgid = await self._launch_background_process(
                workspace_dir=context.workspace_dir,
                command=command,
            )
        except (BashJobError, OSError) as exc:
            return ToolExecutionResult(
                call_id=call_id,
                name="bash",
                ok=False,
                content=(
                    "Bash execution request failed\n"
                    f"reason: {exc}"
                ),
                metadata={"mode": "foreground", "error": str(exc)},
            )

        try:
            await asyncio.wait_for(process.wait(), timeout=soft_timeout_seconds)
        except asyncio.TimeoutError:
            duration_seconds = perf_counter() - started_at
            output_tail = read_job_tail(
                job,
                max_bytes=self._settings.bash_max_output_chars,
                tail_lines=50,
            )
            content = _format_background_promotion_result(
                command=command,
                cwd=Path("/workspace"),
                job_id=job.job_id,
                pid=process.pid,
                pgid=pgid,
                duration_seconds=duration_seconds,
                soft_timeout_seconds=soft_timeout_seconds,
                timeout_seconds=timeout_seconds,
                stdout_text=output_tail["stdout"],
                stderr_text=output_tail["stderr"],
            )
            return ToolExecutionResult(
                call_id=call_id,
                name="bash",
                ok=True,
                content=content,
                metadata={
                    "mode": "foreground",
                    "promoted_to_background": True,
                    "job_id": job.job_id,
                    "pid": process.pid,
                    "pgid": pgid,
                    "status": "running",
                    "command": command,
                    "cwd": "/workspace",
                    "workspace_source_dir": str(context.workspace_dir),
                    "environment_scrubbed": True,
                    "approval_consumed": bool(context.approved_action),
                    "exit_code": None,
                    "timed_out": False,
                    "soft_timed_out": True,
                    "soft_timeout_seconds": soft_timeout_seconds,
                    "timeout_seconds": timeout_seconds,
                    "duration_seconds": round(duration_seconds, 3),
                    "stdout": output_tail["stdout"],
                    "stderr": output_tail["stderr"],
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                    "stdout_path": str(job.stdout_path),
                    "stderr_path": str(job.stderr_path),
                    "runtime_location": self._runtime_location,
                    "runtime_transport": self._runtime_transport,
                    "target_runtime": self._target_runtime,
                    "filesystem_scope": "container_direct",
                    "container_mutation_boundary": self._container_mutation_boundary,
                },
            )

        duration_seconds = perf_counter() - started_at
        stdout_text, stdout_truncated = _truncate_text(
            _read_job_output_text(job.stdout_path),
            self._settings.bash_max_output_chars,
        )
        stderr_text, stderr_truncated = _truncate_text(
            _read_job_output_text(job.stderr_path),
            self._settings.bash_max_output_chars,
        )
        exit_code = process.returncode if process.returncode is not None else -1
        content = _format_bash_result(
            command=command,
            cwd=Path("/workspace"),
            exit_code=exit_code,
            timed_out=False,
            timeout_seconds=timeout_seconds,
            duration_seconds=duration_seconds,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )
        remove_job_artifacts(job)
        return ToolExecutionResult(
            call_id=call_id,
            name="bash",
            ok=exit_code == 0,
            content=content,
            metadata={
                "mode": "foreground",
                "command": command,
                "cwd": "/workspace",
                "workspace_source_dir": str(context.workspace_dir),
                "environment_scrubbed": True,
                "approval_consumed": bool(context.approved_action),
                "exit_code": exit_code,
                "timed_out": False,
                "soft_timed_out": False,
                "timeout_seconds": timeout_seconds,
                "soft_timeout_seconds": soft_timeout_seconds,
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

    async def _start_background_job(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        command = str(arguments.get("command", "")).strip()
        started_at = perf_counter()
        try:
            job, process, pgid = await self._launch_background_process(
                workspace_dir=context.workspace_dir,
                command=command,
            )
        except (BashJobError, OSError) as exc:
            return ToolExecutionResult(
                call_id=call_id,
                name="bash",
                ok=False,
                content=(
                    "Bash background job failed\n"
                    f"reason: {exc}"
                ),
                metadata={"mode": "background", "error": str(exc)},
            )

        duration_seconds = perf_counter() - started_at
        content = "\n".join(
            [
                "Bash background job started",
                f"job_id: {job.job_id}",
                f"pid: {process.pid}",
                f"pgid: {pgid}",
                "cwd: /workspace",
                f"command: {command}",
                f"duration_seconds: {duration_seconds:.3f}",
            ]
        )
        return ToolExecutionResult(
            call_id=call_id,
            name="bash",
            ok=True,
            content=content,
            metadata={
                "mode": "background",
                "job_id": job.job_id,
                "pid": process.pid,
                "pgid": pgid,
                "command": command,
                "cwd": "/workspace",
                "duration_seconds": round(duration_seconds, 3),
                "stdout_path": str(job.stdout_path),
                "stderr_path": str(job.stderr_path),
                "runtime_location": self._runtime_location,
                "runtime_transport": self._runtime_transport,
                "target_runtime": self._target_runtime,
                "filesystem_scope": "container_direct",
                "container_mutation_boundary": self._container_mutation_boundary,
            },
        )

    async def _launch_background_process(
        self,
        *,
        workspace_dir: Path,
        command: str,
    ) -> tuple[BashJobPaths, asyncio.subprocess.Process, int]:
        job = create_background_job(
            workspace_dir=workspace_dir,
            bash_executable=self._settings.bash_executable,
            command=command,
            cwd="/workspace",
        )
        try:
            with open(os.devnull, "wb") as devnull:
                process = await asyncio.create_subprocess_exec(
                    self._settings.bash_executable,
                    "--noprofile",
                    "--norc",
                    str(job.runner_path),
                    stdout=devnull,
                    stderr=devnull,
                    cwd=str(workspace_dir),
                    env=_build_scrubbed_environment(self._settings),
                    start_new_session=True,
                )
            pgid = os.getpgid(process.pid)
            write_job_metadata(
                paths=job,
                pid=process.pid,
                pgid=pgid,
                command=command,
                launched_at=_utc_now(),
                cwd="/workspace",
            )
        except Exception:
            remove_job_artifacts(job)
            raise
        return job, process, pgid

    def _background_job_status(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        try:
            job_id = _require_job_id(arguments)
            paths, record = load_job(context.workspace_dir, job_id)
            status = job_status(paths, record)
        except BashJobError as exc:
            return ToolExecutionResult(
                call_id=call_id,
                name="bash",
                ok=False,
                content=(
                    "Bash job status failed\n"
                    f"reason: {exc}"
                ),
                metadata={"mode": "status", "error": str(exc)},
            )

        lines = [
            "Bash job status",
            f"job_id: {status['job_id']}",
            f"status: {status['status']}",
            f"pid: {status['pid']}",
            f"pgid: {status['pgid']}",
            f"launched_at: {status['launched_at']}",
        ]
        if status["finished_at"] is not None:
            lines.append(f"finished_at: {status['finished_at']}")
        if status["cancelled_at"] is not None:
            lines.append(f"cancelled_at: {status['cancelled_at']}")
        if status["exit_code"] is not None:
            lines.append(f"exit_code: {status['exit_code']}")
        lines.append(f"command: {status['command']}")
        return ToolExecutionResult(
            call_id=call_id,
            name="bash",
            ok=True,
            content="\n".join(lines),
            metadata={
                "mode": "status",
                **status,
                "runtime_location": self._runtime_location,
                "runtime_transport": self._runtime_transport,
                "target_runtime": self._target_runtime,
                "filesystem_scope": "container_direct",
                "container_mutation_boundary": self._container_mutation_boundary,
            },
        )

    def _background_job_tail(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        try:
            job_id = _require_job_id(arguments)
            paths, record = load_job(context.workspace_dir, job_id)
            max_bytes = _resolve_tail_bytes(arguments, self._settings)
            tail_lines = _resolve_tail_lines(arguments)
            tail = read_job_tail(
                paths,
                max_bytes=max_bytes,
                tail_lines=tail_lines,
            )
        except BashJobError as exc:
            return ToolExecutionResult(
                call_id=call_id,
                name="bash",
                ok=False,
                content=(
                    "Bash job tail failed\n"
                    f"reason: {exc}"
                ),
                metadata={"mode": "tail", "error": str(exc)},
            )

        content = "\n".join(
            [
                "Bash job output tail",
                f"job_id: {record.job_id}",
                "stdout:",
                tail["stdout"] or "(empty)",
                "stderr:",
                tail["stderr"] or "(empty)",
            ]
        )
        return ToolExecutionResult(
            call_id=call_id,
            name="bash",
            ok=True,
            content=content,
            metadata={
                "mode": "tail",
                "job_id": record.job_id,
                "tail_lines": tail_lines,
                "tail_bytes": max_bytes,
                "stdout": tail["stdout"],
                "stderr": tail["stderr"],
                "runtime_location": self._runtime_location,
                "runtime_transport": self._runtime_transport,
                "target_runtime": self._target_runtime,
                "filesystem_scope": "container_direct",
                "container_mutation_boundary": self._container_mutation_boundary,
            },
        )

    def _background_job_cancel(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        try:
            job_id = _require_job_id(arguments)
            paths, record = load_job(context.workspace_dir, job_id)
            status = cancel_job(paths, record)
        except BashJobError as exc:
            return ToolExecutionResult(
                call_id=call_id,
                name="bash",
                ok=False,
                content=(
                    "Bash job cancel failed\n"
                    f"reason: {exc}"
                ),
                metadata={"mode": "cancel", "error": str(exc)},
            )

        lines = [
            "Bash job cancelled",
            f"job_id: {status['job_id']}",
            f"status: {status['status']}",
        ]
        if status["exit_code"] is not None:
            lines.append(f"exit_code: {status['exit_code']}")
        if status["cancelled_at"] is not None:
            lines.append(f"cancelled_at: {status['cancelled_at']}")
        return ToolExecutionResult(
            call_id=call_id,
            name="bash",
            ok=True,
            content="\n".join(lines),
            metadata={
                "mode": "cancel",
                **status,
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


def format_bash_tool_description(settings: ToolSettings) -> str:
    venv_root = settings.python_interpreter_venv
    interpreter_path = venv_root / "bin" / "python"
    return (
        "Run a bash command from `/workspace` inside the isolated tool_runtime container. "
        "Use this for shell commands, CLI tools, installs, builds, file inspection, scripts, "
        "and long-running background jobs. Python execution is allowed here, but it must use "
        f"the central `{venv_root}` environment; prefer bare `python`/`python3` or "
        f"`{interpreter_path}`, and install packages with "
        f"`uv pip install --python {interpreter_path} <package-name>`. "
        f"Foreground commands that are still running after the {settings.bash_foreground_soft_timeout_seconds:.0f}s "
        "soft timeout are automatically moved to background mode; after that, use the same tool "
        "with `mode='status'`, `mode='tail'`, or `mode='cancel'` plus `job_id` to manage them. "
        "Set `mode='background'` to start a long-running job, then use the same tool with "
        "`mode='status'`, `mode='tail'`, or `mode='cancel'` plus `job_id` to manage it explicitly. "
        "User approval is typically required for commands that install packages or tools, "
        "run remote installer pipelines such as `curl|bash`, or write into system paths "
        "outside `/workspace` such as `/usr/local/bin`, `/etc`, `/opt`, or `/var`. "
        "When approval is likely, provide clear "
        "`approval_summary`, `approval_details`, and optional `inspection_url` context."
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


def _format_background_promotion_result(
    *,
    command: str,
    cwd: Path,
    job_id: str,
    pid: int,
    pgid: int,
    duration_seconds: float,
    soft_timeout_seconds: float,
    timeout_seconds: float,
    stdout_text: str,
    stderr_text: str,
) -> str:
    lines = [
        "Bash foreground execution moved to background",
        "status: background",
        f"command: {command}",
        f"cwd: {cwd}",
        f"job_id: {job_id}",
        f"pid: {pid}",
        f"pgid: {pgid}",
        f"duration_seconds: {duration_seconds:.3f}",
        f"soft_timeout_seconds: {soft_timeout_seconds:.3f}",
        f"requested_timeout_seconds: {timeout_seconds:.3f}",
        "Use `mode='status'` to check the job, `mode='tail'` to inspect output, or `mode='cancel'` to stop it.",
        "stdout:",
        stdout_text or "(empty)",
        "stderr:",
        stderr_text or "(empty)",
    ]
    return "\n".join(lines)


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    truncated = f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"
    return truncated, True


def _read_job_output_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _normalize_mode(value: object) -> str:
    if value is None:
        return _DEFAULT_MODE
    normalized = str(value).strip().lower()
    return normalized or _DEFAULT_MODE


def _require_job_id(arguments: dict[str, Any]) -> str:
    job_id = str(arguments.get("job_id", "")).strip()
    if not job_id:
        raise BashJobError("bash job operations require a non-empty job_id.")
    return job_id


def _resolve_tail_lines(arguments: dict[str, Any]) -> int | None:
    raw_value = arguments.get("tail_lines")
    if raw_value is None:
        return 50
    value = int(raw_value)
    if value < 1:
        return 1
    if value > 2000:
        return 2000
    return value


def _resolve_tail_bytes(arguments: dict[str, Any], settings: ToolSettings) -> int:
    raw_value = arguments.get("tail_bytes")
    if raw_value is None:
        return settings.bash_max_output_chars
    value = int(raw_value)
    if value < 1:
        return 1
    if value > settings.bash_max_output_chars:
        return settings.bash_max_output_chars
    return value


def _kill_process_group(process_id: int | None) -> None:
    if process_id is None:
        return
    try:
        os.killpg(process_id, signal.SIGKILL)
    except ProcessLookupError:
        return


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
