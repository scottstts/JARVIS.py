"""Helpers for background bash job management."""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

_JOBS_DIRNAME = ".jarvis_internal/bash_jobs"
_METADATA_FILENAME = "metadata.json"
_COMMAND_FILENAME = "command.sh"
_RUNNER_FILENAME = "runner.sh"
_STDOUT_FILENAME = "stdout.log"
_STDERR_FILENAME = "stderr.log"
_EXIT_CODE_FILENAME = "exit_code"
_FINISHED_AT_FILENAME = "finished_at"
_CANCELLED_AT_FILENAME = "cancelled_at"


class BashJobError(RuntimeError):
    """Raised when a background bash job request is invalid."""


@dataclass(slots=True, frozen=True)
class BashJobPaths:
    job_id: str
    job_dir: Path
    metadata_path: Path
    command_path: Path
    runner_path: Path
    stdout_path: Path
    stderr_path: Path
    exit_code_path: Path
    finished_at_path: Path
    cancelled_at_path: Path


@dataclass(slots=True, frozen=True)
class BashJobRecord:
    job_id: str
    command: str
    pid: int
    pgid: int
    launched_at: str
    cwd: str
    stdout_path: str
    stderr_path: str
    job_dir: str


def create_background_job(
    *,
    workspace_dir: Path,
    bash_executable: str,
    command: str,
    cwd: str,
) -> BashJobPaths:
    job_id = uuid4().hex
    jobs_dir = workspace_dir / _JOBS_DIRNAME
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    paths = BashJobPaths(
        job_id=job_id,
        job_dir=job_dir,
        metadata_path=job_dir / _METADATA_FILENAME,
        command_path=job_dir / _COMMAND_FILENAME,
        runner_path=job_dir / _RUNNER_FILENAME,
        stdout_path=job_dir / _STDOUT_FILENAME,
        stderr_path=job_dir / _STDERR_FILENAME,
        exit_code_path=job_dir / _EXIT_CODE_FILENAME,
        finished_at_path=job_dir / _FINISHED_AT_FILENAME,
        cancelled_at_path=job_dir / _CANCELLED_AT_FILENAME,
    )

    paths.command_path.write_text(
        "set -o pipefail\n" + command + "\n",
        encoding="utf-8",
    )
    paths.runner_path.write_text(
        _build_runner_script(
            bash_executable=bash_executable,
            command_path=paths.command_path,
            stdout_path=paths.stdout_path,
            stderr_path=paths.stderr_path,
            exit_code_path=paths.exit_code_path,
            finished_at_path=paths.finished_at_path,
        ),
        encoding="utf-8",
    )
    paths.command_path.chmod(0o700)
    paths.runner_path.chmod(0o700)
    write_job_metadata(
        paths=paths,
        pid=0,
        pgid=0,
        command=command,
        launched_at=_utc_now(),
        cwd=cwd,
    )
    return paths


def write_job_metadata(
    *,
    paths: BashJobPaths,
    pid: int,
    pgid: int,
    command: str,
    launched_at: str,
    cwd: str,
) -> None:
    payload = {
        "job_id": paths.job_id,
        "command": command,
        "pid": pid,
        "pgid": pgid,
        "launched_at": launched_at,
        "cwd": cwd,
        "stdout_path": str(paths.stdout_path),
        "stderr_path": str(paths.stderr_path),
        "job_dir": str(paths.job_dir),
    }
    _write_json_atomic(paths.metadata_path, payload)


def load_job(workspace_dir: Path, job_id: str) -> tuple[BashJobPaths, BashJobRecord]:
    paths = job_paths(workspace_dir, job_id)
    if not paths.job_dir.exists():
        raise BashJobError(f"Unknown bash job id: {job_id}")
    try:
        payload = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BashJobError(f"Unknown bash job id: {job_id}") from exc
    except json.JSONDecodeError as exc:
        raise BashJobError(f"Bash job metadata is corrupt for job id: {job_id}") from exc

    return paths, BashJobRecord(
        job_id=str(payload["job_id"]),
        command=str(payload["command"]),
        pid=int(payload["pid"]),
        pgid=int(payload["pgid"]),
        launched_at=str(payload["launched_at"]),
        cwd=str(payload["cwd"]),
        stdout_path=str(payload["stdout_path"]),
        stderr_path=str(payload["stderr_path"]),
        job_dir=str(payload["job_dir"]),
    )


def job_paths(workspace_dir: Path, job_id: str) -> BashJobPaths:
    normalized = job_id.strip()
    if not normalized or any(ch not in "0123456789abcdef" for ch in normalized.lower()):
        raise BashJobError("bash job_id must be a non-empty lowercase hex string.")
    job_dir = workspace_dir / _JOBS_DIRNAME / normalized
    return BashJobPaths(
        job_id=normalized,
        job_dir=job_dir,
        metadata_path=job_dir / _METADATA_FILENAME,
        command_path=job_dir / _COMMAND_FILENAME,
        runner_path=job_dir / _RUNNER_FILENAME,
        stdout_path=job_dir / _STDOUT_FILENAME,
        stderr_path=job_dir / _STDERR_FILENAME,
        exit_code_path=job_dir / _EXIT_CODE_FILENAME,
        finished_at_path=job_dir / _FINISHED_AT_FILENAME,
        cancelled_at_path=job_dir / _CANCELLED_AT_FILENAME,
    )


def job_status(paths: BashJobPaths, record: BashJobRecord) -> dict[str, Any]:
    running = _process_is_running(record.pid)
    exit_code = _read_int(paths.exit_code_path)
    cancelled = paths.cancelled_at_path.exists()
    finished_at = _read_optional_text(paths.finished_at_path)
    cancelled_at = _read_optional_text(paths.cancelled_at_path)

    if running:
        status = "running"
    elif cancelled:
        status = "cancelled"
    else:
        status = "finished"

    return {
        "job_id": record.job_id,
        "status": status,
        "pid": record.pid,
        "pgid": record.pgid,
        "launched_at": record.launched_at,
        "finished_at": finished_at,
        "cancelled_at": cancelled_at,
        "exit_code": exit_code,
        "stdout_path": record.stdout_path,
        "stderr_path": record.stderr_path,
        "command": record.command,
    }


def cancel_job(paths: BashJobPaths, record: BashJobRecord) -> dict[str, Any]:
    running = _process_is_running(record.pid)
    if running:
        try:
            os.killpg(record.pgid, signal.SIGTERM)
        except ProcessLookupError:
            running = False
        else:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if not _process_is_running(record.pid):
                    running = False
                    break
                time.sleep(0.05)
            if running:
                try:
                    os.killpg(record.pgid, signal.SIGKILL)
                except ProcessLookupError:
                    running = False

    paths.cancelled_at_path.write_text(_utc_now() + "\n", encoding="utf-8")
    return job_status(paths, record)


def read_job_tail(
    paths: BashJobPaths,
    *,
    max_bytes: int,
    tail_lines: int | None,
) -> dict[str, str]:
    return {
        "stdout": _read_tail_text(paths.stdout_path, max_bytes=max_bytes, tail_lines=tail_lines),
        "stderr": _read_tail_text(paths.stderr_path, max_bytes=max_bytes, tail_lines=tail_lines),
    }


def _build_runner_script(
    *,
    bash_executable: str,
    command_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    exit_code_path: Path,
    finished_at_path: Path,
) -> str:
    return f"""#!/bin/bash
set -o pipefail
"{bash_executable}" --noprofile --norc "{command_path}" >>"{stdout_path}" 2>>"{stderr_path}"
exit_code=$?
printf '%s\\n' "$exit_code" > "{exit_code_path}"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "{finished_at_path}"
exit "$exit_code"
"""


def _read_tail_text(path: Path, *, max_bytes: int, tail_lines: int | None) -> str:
    if not path.exists():
        return ""

    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        start = max(0, size - max_bytes)
        handle.seek(start)
        text = handle.read().decode("utf-8", errors="replace")

    if tail_lines is None:
        return text

    lines = text.splitlines()
    if len(lines) <= tail_lines:
        return text
    return "\n".join(lines[-tail_lines:]) + "\n"


def _process_is_running(process_id: int) -> bool:
    if process_id <= 0:
        return False
    proc_status = Path(f"/proc/{process_id}/status")
    if proc_status.exists():
        try:
            for line in proc_status.read_text(encoding="utf-8").splitlines():
                if line.startswith("State:"):
                    return "\tZ" not in line and " zombie" not in line.lower()
        except OSError:
            return False
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def _read_int(path: Path) -> int | None:
    value = _read_optional_text(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
