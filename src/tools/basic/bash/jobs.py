"""Helpers for background bash job management."""

from __future__ import annotations

import json
import os
import shutil
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
_LOGGER_FILENAME = "log_sink.py"
_STDOUT_FILENAME = "stdout.log"
_STDERR_FILENAME = "stderr.log"
_STDOUT_STATS_FILENAME = "stdout.stats.json"
_STDERR_STATS_FILENAME = "stderr.stats.json"
_STDOUT_PIPE_FILENAME = "stdout.pipe"
_STDERR_PIPE_FILENAME = "stderr.pipe"
_CHILD_PID_FILENAME = "child_pid"
_CHILD_PGID_FILENAME = "child_pgid"
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
    logger_path: Path
    stdout_path: Path
    stderr_path: Path
    stdout_stats_path: Path
    stderr_stats_path: Path
    stdout_pipe_path: Path
    stderr_pipe_path: Path
    child_pid_path: Path
    child_pgid_path: Path
    exit_code_path: Path
    finished_at_path: Path
    cancelled_at_path: Path


@dataclass(slots=True, frozen=True)
class BashJobRecord:
    job_id: str
    command: str
    pid: int
    pgid: int
    runner_pid: int
    runner_pgid: int
    launched_at: str
    cwd: str
    stdout_path: str
    stderr_path: str
    job_dir: str
    owner_route_id: str | None = None
    owner_session_id: str | None = None
    owner_turn_id: str | None = None
    owner_agent_kind: str | None = None
    owner_agent_name: str | None = None
    owner_subagent_id: str | None = None
    last_progress_notice_kind: str | None = None
    last_progress_notice_at: str | None = None
    last_progress_notice_status: str | None = None
    last_progress_notice_stdout_bytes_seen: int | None = None
    last_progress_notice_stderr_bytes_seen: int | None = None
    last_progress_notice_last_update_at: str | None = None
    progress_notice_count: int | None = None
    terminal_notice_kind: str | None = None
    terminal_notice_dispatched_at: str | None = None


@dataclass(slots=True, frozen=True)
class BashJobLogStats:
    bytes_seen: int
    bytes_retained: int
    bytes_dropped: int
    updated_at: str | None = None


@dataclass(slots=True, frozen=True)
class BashJobEntry:
    paths: BashJobPaths
    record: BashJobRecord
    status: dict[str, Any]
    size_bytes: int
    sort_timestamp: float


def create_background_job(
    *,
    workspace_dir: Path,
    bash_executable: str,
    command: str,
    cwd: str,
    log_max_bytes: int,
    total_storage_budget_bytes: int,
    retention_seconds: float,
) -> BashJobPaths:
    ensure_job_storage_capacity(
        workspace_dir=workspace_dir,
        total_storage_budget_bytes=total_storage_budget_bytes,
        log_max_bytes=log_max_bytes,
        retention_seconds=retention_seconds,
    )

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
        logger_path=job_dir / _LOGGER_FILENAME,
        stdout_path=job_dir / _STDOUT_FILENAME,
        stderr_path=job_dir / _STDERR_FILENAME,
        stdout_stats_path=job_dir / _STDOUT_STATS_FILENAME,
        stderr_stats_path=job_dir / _STDERR_STATS_FILENAME,
        stdout_pipe_path=job_dir / _STDOUT_PIPE_FILENAME,
        stderr_pipe_path=job_dir / _STDERR_PIPE_FILENAME,
        child_pid_path=job_dir / _CHILD_PID_FILENAME,
        child_pgid_path=job_dir / _CHILD_PGID_FILENAME,
        exit_code_path=job_dir / _EXIT_CODE_FILENAME,
        finished_at_path=job_dir / _FINISHED_AT_FILENAME,
        cancelled_at_path=job_dir / _CANCELLED_AT_FILENAME,
    )

    paths.command_path.write_text(
        "set -o pipefail\n" + command + "\n",
        encoding="utf-8",
    )
    paths.logger_path.write_text(_build_log_sink_script(), encoding="utf-8")
    paths.runner_path.write_text(
        _build_runner_script(
            bash_executable=bash_executable,
            command_path=paths.command_path,
            logger_path=paths.logger_path,
            stdout_path=paths.stdout_path,
            stderr_path=paths.stderr_path,
            stdout_stats_path=paths.stdout_stats_path,
            stderr_stats_path=paths.stderr_stats_path,
            stdout_pipe_path=paths.stdout_pipe_path,
            stderr_pipe_path=paths.stderr_pipe_path,
            child_pid_path=paths.child_pid_path,
            child_pgid_path=paths.child_pgid_path,
            exit_code_path=paths.exit_code_path,
            finished_at_path=paths.finished_at_path,
            log_max_bytes=log_max_bytes,
        ),
        encoding="utf-8",
    )
    _initialize_log_stats(paths.stdout_stats_path)
    _initialize_log_stats(paths.stderr_stats_path)
    paths.command_path.chmod(0o700)
    paths.logger_path.chmod(0o700)
    paths.runner_path.chmod(0o700)
    write_job_metadata(
        paths=paths,
        pid=0,
        pgid=0,
        runner_pid=0,
        runner_pgid=0,
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
    runner_pid: int,
    runner_pgid: int,
    command: str,
    launched_at: str,
    cwd: str,
    owner_route_id: str | None = None,
    owner_session_id: str | None = None,
    owner_turn_id: str | None = None,
    owner_agent_kind: str | None = None,
    owner_agent_name: str | None = None,
    owner_subagent_id: str | None = None,
    last_progress_notice_kind: str | None = None,
    last_progress_notice_at: str | None = None,
    last_progress_notice_status: str | None = None,
    last_progress_notice_stdout_bytes_seen: int | None = None,
    last_progress_notice_stderr_bytes_seen: int | None = None,
    last_progress_notice_last_update_at: str | None = None,
    progress_notice_count: int | None = None,
    terminal_notice_kind: str | None = None,
    terminal_notice_dispatched_at: str | None = None,
) -> None:
    payload = {
        "job_id": paths.job_id,
        "command": command,
        "pid": pid,
        "pgid": pgid,
        "runner_pid": runner_pid,
        "runner_pgid": runner_pgid,
        "launched_at": launched_at,
        "cwd": cwd,
        "stdout_path": str(paths.stdout_path),
        "stderr_path": str(paths.stderr_path),
        "job_dir": str(paths.job_dir),
    }
    if owner_route_id is not None:
        payload["owner_route_id"] = owner_route_id
    if owner_session_id is not None:
        payload["owner_session_id"] = owner_session_id
    if owner_turn_id is not None:
        payload["owner_turn_id"] = owner_turn_id
    if owner_agent_kind is not None:
        payload["owner_agent_kind"] = owner_agent_kind
    if owner_agent_name is not None:
        payload["owner_agent_name"] = owner_agent_name
    if owner_subagent_id is not None:
        payload["owner_subagent_id"] = owner_subagent_id
    if last_progress_notice_kind is not None:
        payload["last_progress_notice_kind"] = last_progress_notice_kind
    if last_progress_notice_at is not None:
        payload["last_progress_notice_at"] = last_progress_notice_at
    if last_progress_notice_status is not None:
        payload["last_progress_notice_status"] = last_progress_notice_status
    if last_progress_notice_stdout_bytes_seen is not None:
        payload["last_progress_notice_stdout_bytes_seen"] = (
            int(last_progress_notice_stdout_bytes_seen)
        )
    if last_progress_notice_stderr_bytes_seen is not None:
        payload["last_progress_notice_stderr_bytes_seen"] = (
            int(last_progress_notice_stderr_bytes_seen)
        )
    if last_progress_notice_last_update_at is not None:
        payload["last_progress_notice_last_update_at"] = (
            last_progress_notice_last_update_at
        )
    if progress_notice_count is not None:
        payload["progress_notice_count"] = int(progress_notice_count)
    if terminal_notice_kind is not None:
        payload["terminal_notice_kind"] = terminal_notice_kind
    if terminal_notice_dispatched_at is not None:
        payload["terminal_notice_dispatched_at"] = terminal_notice_dispatched_at
    _write_json_atomic(paths.metadata_path, payload)


def load_job(workspace_dir: Path, job_id: str) -> tuple[BashJobPaths, BashJobRecord]:
    paths = job_paths(workspace_dir, job_id)
    if not paths.job_dir.exists():
        raise BashJobError(f"Unknown bash job id: {job_id}")
    try:
        payload = _load_metadata_payload(paths)
    except FileNotFoundError as exc:
        raise BashJobError(f"Unknown bash job id: {job_id}") from exc
    except json.JSONDecodeError as exc:
        raise BashJobError(f"Bash job metadata is corrupt for job id: {job_id}") from exc

    pid = int(payload["pid"])
    pgid = int(payload["pgid"])
    runner_pid = int(payload.get("runner_pid", pid))
    runner_pgid = int(payload.get("runner_pgid", pgid))

    return paths, BashJobRecord(
        job_id=str(payload["job_id"]),
        command=str(payload["command"]),
        pid=pid,
        pgid=pgid,
        runner_pid=runner_pid,
        runner_pgid=runner_pgid,
        launched_at=str(payload["launched_at"]),
        cwd=str(payload["cwd"]),
        stdout_path=str(payload["stdout_path"]),
        stderr_path=str(payload["stderr_path"]),
        job_dir=str(payload["job_dir"]),
        owner_route_id=_optional_non_empty_string(payload.get("owner_route_id")),
        owner_session_id=_optional_non_empty_string(payload.get("owner_session_id")),
        owner_turn_id=_optional_non_empty_string(payload.get("owner_turn_id")),
        owner_agent_kind=_optional_non_empty_string(payload.get("owner_agent_kind")),
        owner_agent_name=_optional_non_empty_string(payload.get("owner_agent_name")),
        owner_subagent_id=_optional_non_empty_string(payload.get("owner_subagent_id")),
        last_progress_notice_kind=_optional_non_empty_string(
            payload.get("last_progress_notice_kind")
        ),
        last_progress_notice_at=_optional_non_empty_string(
            payload.get("last_progress_notice_at")
        ),
        last_progress_notice_status=_optional_non_empty_string(
            payload.get("last_progress_notice_status")
        ),
        last_progress_notice_stdout_bytes_seen=_optional_int(
            payload.get("last_progress_notice_stdout_bytes_seen")
        ),
        last_progress_notice_stderr_bytes_seen=_optional_int(
            payload.get("last_progress_notice_stderr_bytes_seen")
        ),
        last_progress_notice_last_update_at=_optional_non_empty_string(
            payload.get("last_progress_notice_last_update_at")
        ),
        progress_notice_count=_optional_int(payload.get("progress_notice_count")),
        terminal_notice_kind=_optional_non_empty_string(payload.get("terminal_notice_kind")),
        terminal_notice_dispatched_at=_optional_non_empty_string(
            payload.get("terminal_notice_dispatched_at")
        ),
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
        logger_path=job_dir / _LOGGER_FILENAME,
        stdout_path=job_dir / _STDOUT_FILENAME,
        stderr_path=job_dir / _STDERR_FILENAME,
        stdout_stats_path=job_dir / _STDOUT_STATS_FILENAME,
        stderr_stats_path=job_dir / _STDERR_STATS_FILENAME,
        stdout_pipe_path=job_dir / _STDOUT_PIPE_FILENAME,
        stderr_pipe_path=job_dir / _STDERR_PIPE_FILENAME,
        child_pid_path=job_dir / _CHILD_PID_FILENAME,
        child_pgid_path=job_dir / _CHILD_PGID_FILENAME,
        exit_code_path=job_dir / _EXIT_CODE_FILENAME,
        finished_at_path=job_dir / _FINISHED_AT_FILENAME,
        cancelled_at_path=job_dir / _CANCELLED_AT_FILENAME,
    )


def read_job_process_identifiers(paths: BashJobPaths) -> tuple[int | None, int | None]:
    return _read_int(paths.child_pid_path), _read_int(paths.child_pgid_path)


def list_jobs(workspace_dir: Path) -> list[tuple[BashJobPaths, BashJobRecord]]:
    jobs_dir = workspace_dir / _JOBS_DIRNAME
    if not jobs_dir.exists():
        return []

    jobs: list[tuple[BashJobPaths, BashJobRecord]] = []
    for child in jobs_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            jobs.append(load_job(workspace_dir, child.name))
        except BashJobError:
            continue
    return jobs


def claim_job_owner(
    *,
    workspace_dir: Path,
    job_id: str,
    route_id: str,
    session_id: str,
    turn_id: str,
    agent_kind: str,
    agent_name: str,
    subagent_id: str | None = None,
) -> BashJobRecord:
    paths, record = load_job(workspace_dir, job_id)
    if record.owner_route_id is not None:
        existing = (
            record.owner_route_id,
            record.owner_session_id,
            record.owner_turn_id,
            record.owner_agent_kind,
            record.owner_subagent_id,
        )
        requested = (
            route_id,
            session_id,
            turn_id,
            agent_kind,
            subagent_id,
        )
        if existing != requested:
            raise BashJobError(
                "bash job ownership is already claimed by a different agent context."
            )
        return record

    write_job_metadata(
        paths=paths,
        pid=record.pid,
        pgid=record.pgid,
        runner_pid=record.runner_pid,
        runner_pgid=record.runner_pgid,
        command=record.command,
        launched_at=record.launched_at,
        cwd=record.cwd,
        owner_route_id=route_id,
        owner_session_id=session_id,
        owner_turn_id=turn_id,
        owner_agent_kind=agent_kind,
        owner_agent_name=agent_name,
        owner_subagent_id=subagent_id,
        last_progress_notice_kind=record.last_progress_notice_kind,
        last_progress_notice_at=record.last_progress_notice_at,
        last_progress_notice_status=record.last_progress_notice_status,
        last_progress_notice_stdout_bytes_seen=record.last_progress_notice_stdout_bytes_seen,
        last_progress_notice_stderr_bytes_seen=record.last_progress_notice_stderr_bytes_seen,
        last_progress_notice_last_update_at=record.last_progress_notice_last_update_at,
        progress_notice_count=record.progress_notice_count,
        terminal_notice_kind=record.terminal_notice_kind,
        terminal_notice_dispatched_at=record.terminal_notice_dispatched_at,
    )
    _, updated = load_job(workspace_dir, job_id)
    return updated


def mark_job_terminal_notice_dispatched(
    *,
    workspace_dir: Path,
    job_id: str,
    notice_kind: str,
    dispatched_at: str | None = None,
) -> BashJobRecord:
    paths, record = load_job(workspace_dir, job_id)
    timestamp = dispatched_at or _utc_now()
    write_job_metadata(
        paths=paths,
        pid=record.pid,
        pgid=record.pgid,
        runner_pid=record.runner_pid,
        runner_pgid=record.runner_pgid,
        command=record.command,
        launched_at=record.launched_at,
        cwd=record.cwd,
        owner_route_id=record.owner_route_id,
        owner_session_id=record.owner_session_id,
        owner_turn_id=record.owner_turn_id,
        owner_agent_kind=record.owner_agent_kind,
        owner_agent_name=record.owner_agent_name,
        owner_subagent_id=record.owner_subagent_id,
        last_progress_notice_kind=record.last_progress_notice_kind,
        last_progress_notice_at=record.last_progress_notice_at,
        last_progress_notice_status=record.last_progress_notice_status,
        last_progress_notice_stdout_bytes_seen=record.last_progress_notice_stdout_bytes_seen,
        last_progress_notice_stderr_bytes_seen=record.last_progress_notice_stderr_bytes_seen,
        last_progress_notice_last_update_at=record.last_progress_notice_last_update_at,
        progress_notice_count=record.progress_notice_count,
        terminal_notice_kind=notice_kind,
        terminal_notice_dispatched_at=timestamp,
    )
    _, updated = load_job(workspace_dir, job_id)
    return updated


def mark_job_progress_notified(
    *,
    workspace_dir: Path,
    job_id: str,
    notice_kind: str,
    status: str,
    stdout_bytes_seen: int,
    stderr_bytes_seen: int,
    last_update_at: str | None,
    count_as_progress_update: bool = True,
) -> BashJobRecord:
    paths, record = load_job(workspace_dir, job_id)
    timestamp = _utc_now()
    progress_notice_count = record.progress_notice_count or 0
    if count_as_progress_update:
        progress_notice_count += 1
    write_job_metadata(
        paths=paths,
        pid=record.pid,
        pgid=record.pgid,
        runner_pid=record.runner_pid,
        runner_pgid=record.runner_pgid,
        command=record.command,
        launched_at=record.launched_at,
        cwd=record.cwd,
        owner_route_id=record.owner_route_id,
        owner_session_id=record.owner_session_id,
        owner_turn_id=record.owner_turn_id,
        owner_agent_kind=record.owner_agent_kind,
        owner_agent_name=record.owner_agent_name,
        owner_subagent_id=record.owner_subagent_id,
        last_progress_notice_kind=notice_kind,
        last_progress_notice_at=timestamp,
        last_progress_notice_status=status,
        last_progress_notice_stdout_bytes_seen=int(stdout_bytes_seen),
        last_progress_notice_stderr_bytes_seen=int(stderr_bytes_seen),
        last_progress_notice_last_update_at=last_update_at,
        progress_notice_count=progress_notice_count,
        terminal_notice_kind=record.terminal_notice_kind,
        terminal_notice_dispatched_at=record.terminal_notice_dispatched_at,
    )
    _, updated = load_job(workspace_dir, job_id)
    return updated


def job_status(paths: BashJobPaths, record: BashJobRecord) -> dict[str, Any]:
    effective_pid, effective_pgid = _effective_process_identifiers(paths, record)
    active_running = _process_is_running(effective_pid)
    runner_running = _process_is_running(record.runner_pid)
    running = active_running or runner_running
    exit_code = _read_int(paths.exit_code_path)
    cancelled = paths.cancelled_at_path.exists()
    finished_at = _read_optional_text(paths.finished_at_path)
    cancelled_at = _read_optional_text(paths.cancelled_at_path)
    stdout_stats = _read_log_stats(paths.stdout_path, paths.stdout_stats_path)
    stderr_stats = _read_log_stats(paths.stderr_path, paths.stderr_stats_path)
    last_update_at = _latest_timestamp_text(
        stdout_stats.updated_at,
        stderr_stats.updated_at,
        finished_at,
        cancelled_at,
        record.launched_at,
    )

    if running:
        status = "running"
    elif cancelled:
        status = "cancelled"
    else:
        status = "finished"

    return {
        "job_id": record.job_id,
        "status": status,
        "pid": effective_pid,
        "pgid": effective_pgid,
        "runner_pid": record.runner_pid,
        "runner_pgid": record.runner_pgid,
        "started_at": record.launched_at,
        "launched_at": record.launched_at,
        "last_update_at": last_update_at,
        "finished_at": finished_at,
        "cancelled_at": cancelled_at,
        "exit_code": exit_code,
        "stdout_path": record.stdout_path,
        "stderr_path": record.stderr_path,
        "stdout_bytes_seen": stdout_stats.bytes_seen,
        "stderr_bytes_seen": stderr_stats.bytes_seen,
        "stdout_bytes_retained": stdout_stats.bytes_retained,
        "stderr_bytes_retained": stderr_stats.bytes_retained,
        "stdout_bytes_dropped": stdout_stats.bytes_dropped,
        "stderr_bytes_dropped": stderr_stats.bytes_dropped,
        "command": record.command,
        "owner_route_id": record.owner_route_id,
        "owner_session_id": record.owner_session_id,
        "owner_turn_id": record.owner_turn_id,
        "owner_agent_kind": record.owner_agent_kind,
        "owner_agent_name": record.owner_agent_name,
        "owner_subagent_id": record.owner_subagent_id,
        "last_progress_notice_kind": record.last_progress_notice_kind,
        "last_progress_notice_at": record.last_progress_notice_at,
        "last_progress_notice_status": record.last_progress_notice_status,
        "last_progress_notice_stdout_bytes_seen": (
            record.last_progress_notice_stdout_bytes_seen
        ),
        "last_progress_notice_stderr_bytes_seen": (
            record.last_progress_notice_stderr_bytes_seen
        ),
        "last_progress_notice_last_update_at": record.last_progress_notice_last_update_at,
        "progress_notice_count": record.progress_notice_count,
        "terminal_notice_kind": record.terminal_notice_kind,
        "terminal_notice_dispatched_at": record.terminal_notice_dispatched_at,
    }


def cancel_job(paths: BashJobPaths, record: BashJobRecord) -> dict[str, Any]:
    target_pid, target_pgid = _effective_process_identifiers(paths, record)
    candidate_pgids = {pgid for pgid in {target_pgid, record.runner_pgid} if pgid > 0}
    running = _process_is_running(target_pid) or _process_is_running(record.runner_pid)
    if running:
        for pgid in candidate_pgids:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                continue

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if not (_process_is_running(target_pid) or _process_is_running(record.runner_pid)):
                running = False
                break
            time.sleep(0.05)

        if running:
            for pgid in candidate_pgids:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    continue

    paths.cancelled_at_path.write_text(_utc_now() + "\n", encoding="utf-8")
    return job_status(paths, record)


def read_job_tail(
    paths: BashJobPaths,
    *,
    max_bytes: int,
    tail_lines: int | None,
) -> dict[str, Any]:
    stdout_stats = _read_log_stats(paths.stdout_path, paths.stdout_stats_path)
    stderr_stats = _read_log_stats(paths.stderr_path, paths.stderr_stats_path)
    return {
        "stdout": _read_tail_text(paths.stdout_path, max_bytes=max_bytes, tail_lines=tail_lines),
        "stderr": _read_tail_text(paths.stderr_path, max_bytes=max_bytes, tail_lines=tail_lines),
        "stdout_bytes_seen": stdout_stats.bytes_seen,
        "stderr_bytes_seen": stderr_stats.bytes_seen,
        "stdout_bytes_retained": stdout_stats.bytes_retained,
        "stderr_bytes_retained": stderr_stats.bytes_retained,
        "stdout_bytes_dropped": stdout_stats.bytes_dropped,
        "stderr_bytes_dropped": stderr_stats.bytes_dropped,
    }


def ensure_job_storage_capacity(
    *,
    workspace_dir: Path,
    total_storage_budget_bytes: int,
    log_max_bytes: int,
    retention_seconds: float,
) -> None:
    current_usage = sweep_job_artifacts(
        workspace_dir=workspace_dir,
        retention_seconds=retention_seconds,
        total_storage_budget_bytes=total_storage_budget_bytes,
    )
    estimated_new_job_bytes = (2 * log_max_bytes) + (256 * 1024)
    if current_usage + estimated_new_job_bytes > total_storage_budget_bytes:
        raise BashJobError(
            "bash background-job storage budget is exhausted; cancel or allow cleanup of "
            "older jobs before starting another long-running bash command."
        )


def sweep_job_artifacts(
    *,
    workspace_dir: Path,
    retention_seconds: float,
    total_storage_budget_bytes: int,
    protected_job_ids: set[str] | None = None,
) -> int:
    jobs_dir = workspace_dir / _JOBS_DIRNAME
    if not jobs_dir.exists():
        return 0

    protected = protected_job_ids or set()
    entries = _collect_job_entries(workspace_dir)
    now = time.time()
    total_usage = sum(entry.size_bytes for entry in entries)

    retained_entries: list[BashJobEntry] = []
    for entry in entries:
        if entry.record.job_id in protected or entry.status["status"] == "running":
            retained_entries.append(entry)
            continue
        age_seconds = now - entry.sort_timestamp
        if age_seconds < retention_seconds:
            retained_entries.append(entry)
            continue
        remove_job_artifacts(entry.paths)
        total_usage -= entry.size_bytes

    if total_usage <= total_storage_budget_bytes:
        return max(total_usage, 0)

    evictable_entries = sorted(
        (
            entry
            for entry in retained_entries
            if entry.record.job_id not in protected and entry.status["status"] != "running"
        ),
        key=lambda entry: entry.sort_timestamp,
    )
    for entry in evictable_entries:
        if total_usage <= total_storage_budget_bytes:
            break
        remove_job_artifacts(entry.paths)
        total_usage -= entry.size_bytes

    return max(total_usage, 0)


def remove_job_artifacts(paths: BashJobPaths) -> None:
    shutil.rmtree(paths.job_dir, ignore_errors=True)


def _build_runner_script(
    *,
    bash_executable: str,
    command_path: Path,
    logger_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    stdout_stats_path: Path,
    stderr_stats_path: Path,
    stdout_pipe_path: Path,
    stderr_pipe_path: Path,
    child_pid_path: Path,
    child_pgid_path: Path,
    exit_code_path: Path,
    finished_at_path: Path,
    log_max_bytes: int,
) -> str:
    return f"""#!/bin/bash
set -o pipefail

child_pid=""
child_pgid=""
stdout_logger_pid=""
stderr_logger_pid=""

cleanup_pipes() {{
  rm -f "{stdout_pipe_path}" "{stderr_pipe_path}"
}}

terminate_children() {{
  local target_pgid="$child_pgid"
  if [[ -z "$target_pgid" && -n "$child_pid" ]]; then
    target_pgid=$(ps -o pgid= "$child_pid" 2>/dev/null | tr -d ' ')
  fi
  if [[ -n "$target_pgid" ]]; then
    kill -TERM -- "-$target_pgid" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$target_pgid" 2>/dev/null || true
  elif [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    sleep 1
    kill -KILL "$child_pid" 2>/dev/null || true
  fi
}}

finish_loggers() {{
  if [[ -n "$stdout_logger_pid" ]]; then
    wait "$stdout_logger_pid" 2>/dev/null || true
  fi
  if [[ -n "$stderr_logger_pid" ]]; then
    wait "$stderr_logger_pid" 2>/dev/null || true
  fi
}}

handle_signal() {{
  terminate_children
  finish_loggers
  cleanup_pipes
  exit 143
}}

trap handle_signal TERM INT HUP

cleanup_pipes
mkfifo "{stdout_pipe_path}" "{stderr_pipe_path}"

python3 -u "{logger_path}" --log-path "{stdout_path}" --stats-path "{stdout_stats_path}" --max-bytes {log_max_bytes} < "{stdout_pipe_path}" &
stdout_logger_pid=$!
python3 -u "{logger_path}" --log-path "{stderr_path}" --stats-path "{stderr_stats_path}" --max-bytes {log_max_bytes} < "{stderr_pipe_path}" &
stderr_logger_pid=$!

"{bash_executable}" --noprofile --norc "{command_path}" > "{stdout_pipe_path}" 2> "{stderr_pipe_path}" &
child_pid=$!
printf '%s\\n' "$child_pid" > "{child_pid_path}"
child_pgid=$(ps -o pgid= "$child_pid" 2>/dev/null | tr -d ' ')
if [[ -n "$child_pgid" ]]; then
  printf '%s\\n' "$child_pgid" > "{child_pgid_path}"
fi

wait "$child_pid"
exit_code=$?
wait "$stdout_logger_pid"
wait "$stderr_logger_pid"
cleanup_pipes
printf '%s\\n' "$exit_code" > "{exit_code_path}"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "{finished_at_path}"
exit "$exit_code"
"""


def _build_log_sink_script() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _write_stats(path: Path, *, bytes_seen: int, bytes_retained: int) -> None:
    payload = {
        "bytes_dropped": max(0, bytes_seen - bytes_retained),
        "bytes_retained": bytes_retained,
        "bytes_seen": bytes_seen,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _compact_log(path: Path, *, max_bytes: int) -> int:
    with path.open("r+b") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size <= max_bytes:
            return size
        handle.seek(size - max_bytes)
        retained = handle.read(max_bytes)
        handle.seek(0)
        handle.write(retained)
        handle.truncate()
    return max_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--max-bytes", required=True, type=int)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    stats_path = Path(args.stats_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = max(1, args.max_bytes)
    compaction_slack = max(max_bytes // 4, 64 * 1024)
    bytes_seen = 0
    chunks_since_stats = 0

    with log_path.open("ab", buffering=0) as handle:
        while True:
            chunk = sys.stdin.buffer.read(64 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            bytes_seen += len(chunk)
            chunks_since_stats += 1

            current_size = log_path.stat().st_size
            if current_size > max_bytes + compaction_slack:
                current_size = _compact_log(log_path, max_bytes=max_bytes)
                _write_stats(
                    stats_path,
                    bytes_seen=bytes_seen,
                    bytes_retained=current_size,
                )
                chunks_since_stats = 0
            elif chunks_since_stats >= 16:
                _write_stats(
                    stats_path,
                    bytes_seen=bytes_seen,
                    bytes_retained=current_size,
                )
                chunks_since_stats = 0

    final_size = log_path.stat().st_size if log_path.exists() else 0
    if final_size > max_bytes:
        final_size = _compact_log(log_path, max_bytes=max_bytes)
    _write_stats(
        stats_path,
        bytes_seen=bytes_seen,
        bytes_retained=final_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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


def _read_log_stats(log_path: Path, stats_path: Path) -> BashJobLogStats:
    retained_bytes = log_path.stat().st_size if log_path.exists() else 0
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return BashJobLogStats(
            bytes_seen=retained_bytes,
            bytes_retained=retained_bytes,
            bytes_dropped=0,
            updated_at=None,
        )

    bytes_seen = _coerce_non_negative_int(payload.get("bytes_seen"), retained_bytes)
    bytes_retained = _coerce_non_negative_int(payload.get("bytes_retained"), retained_bytes)
    bytes_dropped = _coerce_non_negative_int(payload.get("bytes_dropped"), 0)
    updated_at = _optional_non_empty_string(payload.get("updated_at"))
    if bytes_retained != retained_bytes:
        bytes_retained = retained_bytes
        bytes_dropped = max(0, bytes_seen - bytes_retained)
    return BashJobLogStats(
        bytes_seen=bytes_seen,
        bytes_retained=bytes_retained,
        bytes_dropped=bytes_dropped,
        updated_at=updated_at,
    )


def _effective_process_identifiers(
    paths: BashJobPaths,
    record: BashJobRecord,
) -> tuple[int, int]:
    child_pid, child_pgid = read_job_process_identifiers(paths)
    pid = child_pid if child_pid is not None and child_pid > 0 else record.pid
    pgid = child_pgid if child_pgid is not None and child_pgid > 0 else record.pgid
    return pid, pgid


def _coerce_non_negative_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _collect_job_entries(workspace_dir: Path) -> list[BashJobEntry]:
    jobs_dir = workspace_dir / _JOBS_DIRNAME
    if not jobs_dir.exists():
        return []

    entries: list[BashJobEntry] = []
    for child in jobs_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            paths, record = load_job(workspace_dir, child.name)
            status = job_status(paths, record)
        except BashJobError:
            continue
        entries.append(
            BashJobEntry(
                paths=paths,
                record=record,
                status=status,
                size_bytes=_directory_size_bytes(child),
                sort_timestamp=_sort_timestamp(status, child),
            )
        )
    return entries


def _sort_timestamp(status: dict[str, Any], job_dir: Path) -> float:
    for timestamp in (status.get("finished_at"), status.get("cancelled_at"), status.get("launched_at")):
        if isinstance(timestamp, str):
            parsed = _parse_utc_timestamp(timestamp)
            if parsed is not None:
                return parsed
    try:
        return job_dir.stat().st_mtime
    except OSError:
        return time.time()


def _parse_utc_timestamp(value: str) -> float | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        root_path = Path(root)
        for filename in files:
            file_path = root_path / filename
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


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


def _initialize_log_stats(path: Path) -> None:
    _write_json_atomic(
        path,
        {
            "bytes_dropped": 0,
            "bytes_retained": 0,
            "bytes_seen": 0,
            "updated_at": _utc_now(),
        },
    )


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


def _load_metadata_payload(paths: BashJobPaths) -> dict[str, Any]:
    payload = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("metadata payload must be an object", "", 0)
    return payload


def _optional_non_empty_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _latest_timestamp_text(*values: str | None) -> str | None:
    latest_value: str | None = None
    latest_timestamp = float("-inf")
    for value in values:
        if value is None:
            continue
        parsed = _parse_utc_timestamp(value)
        if parsed is None:
            continue
        if parsed > latest_timestamp:
            latest_timestamp = parsed
            latest_value = value
    return latest_value


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
