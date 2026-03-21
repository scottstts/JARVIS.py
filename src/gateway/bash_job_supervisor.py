"""Route-scoped observation and follow-up dispatch for detached bash jobs."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Awaitable, Callable

from tools.basic.bash.jobs import (
    BashJobError,
    BashJobRecord,
    claim_job_owner,
    job_status,
    list_jobs,
    load_job,
    mark_job_progress_notified,
    mark_job_terminal_notice_dispatched,
)
from tools.basic.bash.tool import BashToolExecutor
from tools.config import ToolSettings
from tools.types import ToolExecutionContext, ToolExecutionResult

LOGGER = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 2.0
_HEARTBEAT_NOTICE_BACKOFF_SECONDS = (30.0, 60.0, 180.0, 300.0)
_SIGNIFICANT_OUTPUT_GROWTH_BYTES = 4096
_NEEDS_ATTENTION_NO_OUTPUT_HEARTBEAT_COUNT = 2
_NEEDS_ATTENTION_DROPPED_BYTES = 65536
_TAIL_LINES = 80
_TAIL_BYTES = 8192


@dataclass(slots=True, frozen=True)
class BashJobNotice:
    job_id: str
    notice_kind: str
    owner_route_id: str
    owner_session_id: str | None
    owner_turn_id: str | None
    owner_agent_kind: str
    owner_agent_name: str
    owner_subagent_id: str | None
    status: str
    command: str
    started_at: str
    last_update_at: str | None
    finished_at: str | None
    cancelled_at: str | None
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_bytes_seen: int
    stderr_bytes_seen: int
    stdout_bytes_dropped: int
    stderr_bytes_dropped: int
    progress_hint: str | None


class BashJobSupervisor:
    """Observes route-owned detached bash jobs and dispatches owner follow-ups."""

    def __init__(
        self,
        *,
        route_id: str,
        settings: ToolSettings,
        followups_allowed: Callable[[], bool],
        main_turn_active: Callable[[], bool],
        subagent_turn_active: Callable[[str], bool],
        handle_main_notices: Callable[[tuple[BashJobNotice, ...]], Awaitable[None]],
        handle_subagent_notices: Callable[[tuple[BashJobNotice, ...]], Awaitable[None]],
    ) -> None:
        self._route_id = route_id
        self._workspace_dir = settings.workspace_dir
        self._executor = BashToolExecutor(settings)
        self._followups_allowed = followups_allowed
        self._main_turn_active = main_turn_active
        self._subagent_turn_active = subagent_turn_active
        self._handle_main_notices = handle_main_notices
        self._handle_subagent_notices = handle_subagent_notices
        self._tracked_job_ids: set[str] = set()
        self._loop_task: asyncio.Task[None] | None = None
        self._wake_event = asyncio.Event()

    def ensure_running(self) -> None:
        if self._loop_task is not None and not self._loop_task.done():
            self._wake_event.set()
            return
        self._loop_task = asyncio.create_task(
            self._run_loop(),
            name=f"jarvis-bash-job-supervisor-{self._route_id}",
        )
        self._wake_event.set()

    def has_pending_jobs(self) -> bool:
        self._recover_tracked_jobs()
        return bool(self._tracked_job_ids)

    def pending_jobs(self) -> tuple[BashJobRecord, ...]:
        self._recover_tracked_jobs()
        records: list[BashJobRecord] = []
        for job_id in tuple(self._tracked_job_ids):
            try:
                _, record = load_job(self._workspace_dir, job_id)
            except BashJobError:
                self._tracked_job_ids.discard(job_id)
                continue
            records.append(record)
        return tuple(records)

    async def observe_tool_result(
        self,
        *,
        result: ToolExecutionResult,
        context: ToolExecutionContext,
    ) -> None:
        if result.name != "bash" or context.route_id != self._route_id:
            return

        job_id = str(result.metadata.get("job_id", "")).strip()
        if not job_id:
            return

        status = str(result.metadata.get("status") or result.metadata.get("state") or "").strip()
        mode = str(result.metadata.get("mode", "")).strip()
        is_detached_start = mode == "background" or bool(result.metadata.get("promoted_to_background"))
        if is_detached_start and status == "running":
            if context.session_id is None or context.turn_id is None:
                return
            try:
                claim_job_owner(
                    workspace_dir=self._workspace_dir,
                    job_id=job_id,
                    route_id=self._route_id,
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                    agent_kind=context.agent_kind,
                    agent_name=context.agent_name,
                    subagent_id=context.subagent_id,
                )
                paths, record = load_job(self._workspace_dir, job_id)
                status_payload = job_status(paths, record)
                mark_job_progress_notified(
                    workspace_dir=self._workspace_dir,
                    job_id=job_id,
                    notice_kind="bash_job_started",
                    status=str(status_payload["status"]),
                    stdout_bytes_seen=int(status_payload["stdout_bytes_seen"]),
                    stderr_bytes_seen=int(status_payload["stderr_bytes_seen"]),
                    last_update_at=_optional_string(status_payload.get("last_update_at")),
                    count_as_progress_update=False,
                )
            except BashJobError:
                LOGGER.exception(
                    "Failed to claim detached bash job ownership for job %s.",
                    job_id,
                )
                return
            self._tracked_job_ids.add(job_id)
            self.ensure_running()
            return

        if mode not in {"status", "tail", "cancel"}:
            return

        try:
            paths, record = load_job(self._workspace_dir, job_id)
        except BashJobError:
            return
        if not self._owner_matches_context(record=record, context=context):
            return

        status_payload = job_status(paths, record)
        try:
            mark_job_progress_notified(
                workspace_dir=self._workspace_dir,
                job_id=job_id,
                notice_kind=_manual_observation_notice_kind(
                    mode=mode,
                    status=str(status_payload["status"]),
                ),
                status=str(status_payload["status"]),
                stdout_bytes_seen=int(status_payload["stdout_bytes_seen"]),
                stderr_bytes_seen=int(status_payload["stderr_bytes_seen"]),
                last_update_at=_optional_string(status_payload.get("last_update_at")),
            )
            if str(status_payload["status"]) in {"finished", "cancelled"}:
                mark_job_terminal_notice_dispatched(
                    workspace_dir=self._workspace_dir,
                    job_id=job_id,
                    notice_kind=_notice_kind_for_status(
                        status=str(status_payload["status"]),
                        exit_code=status_payload.get("exit_code"),
                    ),
                )
                self._tracked_job_ids.discard(job_id)
        except BashJobError:
            LOGGER.exception(
                "Failed to record manual detached bash observation for job %s.",
                job_id,
            )

    async def _run_loop(self) -> None:
        while True:
            self._recover_tracked_jobs()
            main_notices: list[BashJobNotice] = []
            subagent_notices: dict[str, list[BashJobNotice]] = {}
            for job_id in tuple(self._tracked_job_ids):
                notice = await self._collect_due_notice(job_id)
                if notice is None:
                    continue
                if (
                    notice.owner_agent_kind == "subagent"
                    and notice.owner_subagent_id is not None
                ):
                    subagent_notices.setdefault(notice.owner_subagent_id, []).append(notice)
                else:
                    main_notices.append(notice)

            if main_notices:
                await self._dispatch_main_notices(tuple(main_notices))
            for notices in subagent_notices.values():
                await self._dispatch_subagent_notices(tuple(notices))

            self._wake_event.clear()
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=_POLL_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                continue

    def _recover_tracked_jobs(self) -> None:
        for _paths, record in list_jobs(self._workspace_dir):
            if record.owner_route_id != self._route_id:
                continue
            if record.terminal_notice_dispatched_at is not None:
                continue
            self._tracked_job_ids.add(record.job_id)

    async def _collect_due_notice(self, job_id: str) -> BashJobNotice | None:
        try:
            _, record = load_job(self._workspace_dir, job_id)
        except BashJobError:
            self._tracked_job_ids.discard(job_id)
            return None

        if record.owner_route_id != self._route_id:
            self._tracked_job_ids.discard(job_id)
            return None
        if record.terminal_notice_dispatched_at is not None:
            self._tracked_job_ids.discard(job_id)
            return None
        if not self._followups_allowed():
            return None
        if record.owner_agent_kind == "subagent" and record.owner_subagent_id:
            if self._subagent_turn_active(record.owner_subagent_id):
                return None
        elif self._main_turn_active():
            return None

        status_result = await self._execute_internal_bash(
            record=record,
            arguments={"mode": "status", "job_id": job_id},
        )
        if not status_result.ok:
            return None

        notice_kind = _classify_notice_kind(record=record, status_metadata=status_result.metadata)
        status = str(status_result.metadata.get("status", "")).strip()
        if notice_kind is None:
            if status in {"finished", "cancelled"} and record.terminal_notice_dispatched_at is not None:
                self._tracked_job_ids.discard(job_id)
            return None

        tail_result = await self._execute_internal_bash(
            record=record,
            arguments={
                "mode": "tail",
                "job_id": job_id,
                "tail_lines": _TAIL_LINES,
                "tail_bytes": _TAIL_BYTES,
            },
        )
        stdout = str(tail_result.metadata.get("stdout", "")) if tail_result.ok else ""
        stderr = str(tail_result.metadata.get("stderr", "")) if tail_result.ok else ""
        stdout_bytes_dropped = (
            _optional_int(tail_result.metadata.get("stdout_bytes_dropped")) or 0
            if tail_result.ok
            else 0
        )
        stderr_bytes_dropped = (
            _optional_int(tail_result.metadata.get("stderr_bytes_dropped")) or 0
            if tail_result.ok
            else 0
        )
        progress_hint = _derive_progress_hint(stdout_text=stdout, stderr_text=stderr)
        notice_kind = _promote_notice_kind_for_attention(
            notice_kind=notice_kind,
            record=record,
            status=status,
            stdout_bytes_seen=_optional_int(status_result.metadata.get("stdout_bytes_seen")) or 0,
            stderr_bytes_seen=_optional_int(status_result.metadata.get("stderr_bytes_seen")) or 0,
            stdout_bytes_dropped=stdout_bytes_dropped,
            stderr_bytes_dropped=stderr_bytes_dropped,
            progress_hint=progress_hint,
        )
        return BashJobNotice(
            job_id=job_id,
            notice_kind=notice_kind,
            owner_route_id=self._route_id,
            owner_session_id=record.owner_session_id,
            owner_turn_id=record.owner_turn_id,
            owner_agent_kind=record.owner_agent_kind or "main",
            owner_agent_name=record.owner_agent_name or "Jarvis",
            owner_subagent_id=record.owner_subagent_id,
            status=status,
            command=str(status_result.metadata.get("command", record.command)),
            started_at=str(status_result.metadata.get("started_at", record.launched_at)),
            last_update_at=_optional_string(status_result.metadata.get("last_update_at")),
            finished_at=_optional_string(status_result.metadata.get("finished_at")),
            cancelled_at=_optional_string(status_result.metadata.get("cancelled_at")),
            exit_code=_optional_int(status_result.metadata.get("exit_code")),
            stdout=stdout,
            stderr=stderr,
            stdout_bytes_seen=_optional_int(status_result.metadata.get("stdout_bytes_seen")) or 0,
            stderr_bytes_seen=_optional_int(status_result.metadata.get("stderr_bytes_seen")) or 0,
            stdout_bytes_dropped=stdout_bytes_dropped,
            stderr_bytes_dropped=stderr_bytes_dropped,
            progress_hint=progress_hint,
        )

    async def _dispatch_main_notices(self, notices: tuple[BashJobNotice, ...]) -> None:
        try:
            await self._handle_main_notices(notices)
        except Exception:
            LOGGER.exception("Detached bash follow-up dispatch failed for main notices.")
        return

    async def _dispatch_subagent_notices(self, notices: tuple[BashJobNotice, ...]) -> None:
        try:
            await self._handle_subagent_notices(notices)
        except Exception:
            subagent_id = notices[0].owner_subagent_id if notices else None
            LOGGER.exception(
                "Detached bash follow-up dispatch failed for subagent %s.",
                subagent_id or "unknown",
            )
        return

    async def _execute_internal_bash(
        self,
        *,
        record: BashJobRecord,
        arguments: dict[str, object],
    ) -> ToolExecutionResult:
        context = ToolExecutionContext(
            workspace_dir=self._workspace_dir,
            route_id=self._route_id,
            session_id=record.owner_session_id,
            agent_kind="subagent" if record.owner_agent_kind == "subagent" else "main",
            agent_name=record.owner_agent_name or "Jarvis",
            subagent_id=record.owner_subagent_id,
        )
        return await self._executor(
            call_id=f"bash_job_supervisor_{arguments['mode']}_{record.job_id}",
            arguments=dict(arguments),
            context=context,
        )

    def _owner_matches_context(
        self,
        *,
        record: BashJobRecord,
        context: ToolExecutionContext,
    ) -> bool:
        if record.owner_route_id != self._route_id or context.route_id != self._route_id:
            return False
        owner_kind = record.owner_agent_kind or "main"
        if owner_kind != context.agent_kind:
            return False
        if owner_kind == "subagent":
            return record.owner_subagent_id == context.subagent_id
        return True


def _classify_notice_kind(
    *,
    record: BashJobRecord,
    status_metadata: dict[str, object],
) -> str | None:
    status = str(status_metadata.get("status", "")).strip()
    if status == "cancelled":
        return "bash_job_cancelled"
    if status == "finished":
        return _notice_kind_for_status(
            status=status,
            exit_code=status_metadata.get("exit_code"),
        )
    if status != "running":
        return None

    stdout_bytes_seen = _optional_int(status_metadata.get("stdout_bytes_seen")) or 0
    stderr_bytes_seen = _optional_int(status_metadata.get("stderr_bytes_seen")) or 0
    total_bytes_seen = stdout_bytes_seen + stderr_bytes_seen
    previous_total_bytes_seen = (
        (record.last_progress_notice_stdout_bytes_seen or 0)
        + (record.last_progress_notice_stderr_bytes_seen or 0)
    )
    if previous_total_bytes_seen == 0 and total_bytes_seen > 0:
        return "bash_job_output_started"
    if total_bytes_seen - previous_total_bytes_seen >= _SIGNIFICANT_OUTPUT_GROWTH_BYTES:
        return "bash_job_output_grew"
    if (
        total_bytes_seen == 0
        and previous_total_bytes_seen == 0
        and (record.progress_notice_count or 0) >= _NEEDS_ATTENTION_NO_OUTPUT_HEARTBEAT_COUNT
    ):
        baseline = _parse_optional_iso(record.last_progress_notice_at) or _parse_optional_iso(
            record.launched_at
        )
        if baseline is None:
            return None
        heartbeat_notice_seconds = _heartbeat_notice_interval_seconds(record=record)
        if (datetime.now(UTC) - baseline).total_seconds() >= heartbeat_notice_seconds:
            return "bash_job_needs_attention"

    baseline = _parse_optional_iso(record.last_progress_notice_at) or _parse_optional_iso(
        record.launched_at
    )
    if baseline is None:
        return None
    heartbeat_notice_seconds = _heartbeat_notice_interval_seconds(record=record)
    if (datetime.now(UTC) - baseline).total_seconds() >= heartbeat_notice_seconds:
        return "bash_job_heartbeat"
    return None


def _heartbeat_notice_interval_seconds(*, record: BashJobRecord) -> float:
    progress_notice_count = max(record.progress_notice_count or 0, 0)
    if progress_notice_count < len(_HEARTBEAT_NOTICE_BACKOFF_SECONDS):
        return _HEARTBEAT_NOTICE_BACKOFF_SECONDS[progress_notice_count]
    return _HEARTBEAT_NOTICE_BACKOFF_SECONDS[-1]


def _manual_observation_notice_kind(*, mode: str, status: str) -> str:
    if status == "cancelled":
        return "bash_job_cancelled"
    if status == "finished":
        return "bash_job_observed_terminal"
    if mode == "tail":
        return "bash_job_observed_tail"
    return "bash_job_observed_status"


def _notice_kind_for_status(*, status: str, exit_code: object) -> str:
    if status == "cancelled":
        return "bash_job_cancelled"
    resolved_exit_code = _optional_int(exit_code)
    if resolved_exit_code is None or resolved_exit_code == 0:
        return "bash_job_completed"
    return "bash_job_failed"


def _promote_notice_kind_for_attention(
    *,
    notice_kind: str,
    record: BashJobRecord,
    status: str,
    stdout_bytes_seen: int,
    stderr_bytes_seen: int,
    stdout_bytes_dropped: int,
    stderr_bytes_dropped: int,
    progress_hint: str | None,
) -> str:
    if status != "running":
        return notice_kind
    if notice_kind == "bash_job_needs_attention":
        return notice_kind
    if notice_kind not in {"bash_job_output_grew", "bash_job_heartbeat"}:
        return notice_kind
    total_bytes_seen = stdout_bytes_seen + stderr_bytes_seen
    previous_total_bytes_seen = (
        (record.last_progress_notice_stdout_bytes_seen or 0)
        + (record.last_progress_notice_stderr_bytes_seen or 0)
    )
    if (
        total_bytes_seen == 0
        and previous_total_bytes_seen == 0
        and (record.progress_notice_count or 0) >= _NEEDS_ATTENTION_NO_OUTPUT_HEARTBEAT_COUNT
    ):
        return "bash_job_needs_attention"
    dropped_bytes = max(stdout_bytes_dropped, stderr_bytes_dropped)
    if dropped_bytes < _NEEDS_ATTENTION_DROPPED_BYTES:
        return notice_kind
    normalized_hint = (progress_hint or "").strip()
    if normalized_hint and (len(normalized_hint) <= 4 or _looks_repetitive_hint(normalized_hint)):
        return "bash_job_needs_attention"
    return notice_kind


def _derive_progress_hint(*, stdout_text: str, stderr_text: str) -> str | None:
    for candidate in (stdout_text, stderr_text):
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        if not lines:
            continue
        hint = lines[-1]
        return hint if len(hint) <= 240 else hint[:237] + "..."
    return None


def _looks_repetitive_hint(value: str) -> bool:
    normalized = "".join(ch for ch in value if not ch.isspace())
    return len(normalized) >= 8 and len(set(normalized)) <= 2


def _parse_optional_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    candidate = normalized.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
