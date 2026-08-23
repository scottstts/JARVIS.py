"""Route-scoped observation and follow-up dispatch for detached bash jobs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from typing import Awaitable, Callable

from jarvis.logging_setup import get_application_logger
from jarvis.skills import (
    SkillsSettings,
    import_staged_skills,
    render_skill_import_notice,
)
from jarvis.tools.basic.bash.jobs import (
    BashJobError,
    BashJobRecord,
    claim_job_owner,
    list_jobs,
    load_job,
    mark_job_progress_notified,
    mark_job_terminal_notice_dispatched,
)
from jarvis.tools.basic.bash.tool import BashToolExecutor
from jarvis.tools.config import ToolSettings
from jarvis.tools.types import ToolExecutionContext, ToolExecutionResult

LOGGER = get_application_logger(__name__)

_POLL_INTERVAL_SECONDS = 2.0
_MANAGED_SERVICE_POLL_INTERVAL_SECONDS = 30.0
_DEFERRED_RUNNING_NOTICE_RECHECK_SECONDS = 30.0
_SIGNIFICANT_OUTPUT_GROWTH_BYTES = 4096
_NEEDS_ATTENTION_NO_OUTPUT_SECONDS = 300.0
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
    stdout_log_path: str = ""
    stderr_log_path: str = ""
    command_sha256: str = ""
    workspace_revision: str | None = None
    runtime_seconds: float | None = None
    termination_signal: int | None = None
    process_exit_success: bool = False
    stdout_sha256: str = ""
    stderr_sha256: str = ""
    skill_import_notice: str | None = None


@dataclass(slots=True, frozen=True)
class BashJobResetResult:
    """Detached-job state finalized by a hard new-session boundary."""

    finalized_job_ids: tuple[str, ...] = ()
    cancellation_requested_job_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class _NoticeDeliveryMarker:
    last_progress_notice_kind: str | None
    last_progress_notice_at: str | None
    last_progress_notice_status: str | None
    last_progress_notice_stdout_bytes_seen: int | None
    last_progress_notice_stderr_bytes_seen: int | None
    last_progress_notice_last_update_at: str | None
    terminal_notice_dispatched_at: str | None


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
        handle_main_notices: Callable[[tuple[BashJobNotice, ...]], Awaitable[bool]],
        handle_subagent_notices: Callable[[tuple[BashJobNotice, ...]], Awaitable[bool]],
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
        self._deferred_notice_markers: dict[str, _NoticeDeliveryMarker] = {}
        self._deferred_notice_recheck_at: dict[str, float] = {}
        self._next_service_poll_at: dict[str, float] = {}
        self._loop_task: asyncio.Task[None] | None = None
        self._wake_event = asyncio.Event()
        self._job_operation_lock = asyncio.Lock()

    def ensure_running(self) -> None:
        if self._loop_task is not None and not self._loop_task.done():
            self._wake_event.set()
            return
        self._loop_task = asyncio.create_task(
            self._run_loop(),
            name=f"jarvis-bash-job-supervisor-{self._route_id}",
        )
        self._wake_event.set()

    def has_pending_jobs(self, *, include_services: bool = False) -> bool:
        return bool(self.pending_jobs(include_services=include_services))

    def pending_jobs(self, *, include_services: bool = False) -> tuple[BashJobRecord, ...]:
        self._recover_tracked_jobs()
        records: list[BashJobRecord] = []
        for job_id in tuple(self._tracked_job_ids):
            try:
                _, record = load_job(self._workspace_dir, job_id)
            except BashJobError:
                self._tracked_job_ids.discard(job_id)
                self._forget_job(job_id)
                continue
            if not include_services and _is_managed_service(record):
                continue
            records.append(record)
        return tuple(records)

    async def terminate_route_jobs_for_new_session(self) -> BashJobResetResult:
        """Terminate and finalize every detached job still owned by this route."""
        finalized_job_ids: list[str] = []
        cancellation_requested_job_ids: list[str] = []
        async with self._job_operation_lock:
            candidates = sorted(
                (
                    record
                    for _paths, record in list_jobs(self._workspace_dir)
                    if record.owner_route_id == self._route_id
                    and record.terminal_notice_dispatched_at is None
                ),
                key=lambda record: record.job_id,
            )
            for record in candidates:
                status_result = await self._execute_internal_bash(
                    record=record,
                    arguments={"mode": "status", "job_id": record.job_id},
                )
                if not status_result.ok:
                    raise RuntimeError(
                        "Failed to inspect detached bash job during /new hard reset: "
                        f"{record.job_id}: {status_result.content}"
                    )
                status = str(status_result.metadata.get("status", "")).strip()
                exit_code = _optional_int(status_result.metadata.get("exit_code"))
                if status == "running":
                    cancellation_requested_job_ids.append(record.job_id)
                    cancel_result = await self._execute_internal_bash(
                        record=record,
                        arguments={"mode": "cancel", "job_id": record.job_id},
                    )
                    if not cancel_result.ok:
                        raise RuntimeError(
                            "Failed to cancel detached bash job during /new hard reset: "
                            f"{record.job_id}: {cancel_result.content}"
                        )
                    status = str(cancel_result.metadata.get("status", "")).strip()
                    exit_code = _optional_int(cancel_result.metadata.get("exit_code"))
                if status == "running" or not status:
                    raise RuntimeError(
                        f"Detached bash job {record.job_id} survived /new hard reset."
                    )
                try:
                    mark_job_terminal_notice_dispatched(
                        workspace_dir=self._workspace_dir,
                        job_id=record.job_id,
                        notice_kind=_notice_kind_for_status(
                            status=status,
                            exit_code=exit_code,
                        ),
                    )
                except BashJobError as exc:
                    raise RuntimeError(
                        "Failed to finalize detached bash job archive state during /new: "
                        f"{record.job_id}: {exc}"
                    ) from exc
                self._tracked_job_ids.discard(record.job_id)
                self._forget_job(record.job_id)
                finalized_job_ids.append(record.job_id)
        return BashJobResetResult(
            finalized_job_ids=tuple(finalized_job_ids),
            cancellation_requested_job_ids=tuple(cancellation_requested_job_ids),
        )

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
        is_detached_start = mode in {"background", "service"} or bool(
            result.metadata.get("promoted_to_background")
        )
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
                mark_job_progress_notified(
                    workspace_dir=self._workspace_dir,
                    job_id=job_id,
                    notice_kind="bash_job_started",
                    status=status,
                    stdout_bytes_seen=(
                        _optional_int(result.metadata.get("stdout_bytes_seen")) or 0
                    ),
                    stderr_bytes_seen=(
                        _optional_int(result.metadata.get("stderr_bytes_seen")) or 0
                    ),
                    last_update_at=_optional_string(
                        result.metadata.get("last_update_at")
                    ),
                    count_as_progress_update=False,
                )
            except BashJobError:
                LOGGER.exception(
                    "Failed to claim detached bash job ownership for job %s.",
                    job_id,
                )
                return
            self._tracked_job_ids.add(job_id)
            self._deferred_notice_markers.pop(job_id, None)
            self._deferred_notice_recheck_at.pop(job_id, None)
            self._next_service_poll_at.pop(job_id, None)
            self.ensure_running()
            return

        if mode not in {"status", "tail", "cancel"}:
            return

        try:
            _, record = load_job(self._workspace_dir, job_id)
        except BashJobError:
            return
        if not self._owner_matches_context(record=record, context=context):
            return
        if not result.ok:
            return

        status_payload = result.metadata
        observed_status = str(status_payload.get("status", "")).strip()
        if not observed_status:
            return
        try:
            mark_job_progress_notified(
                workspace_dir=self._workspace_dir,
                job_id=job_id,
                notice_kind=_manual_observation_notice_kind(
                    mode=mode,
                    status=observed_status,
                ),
                status=observed_status,
                stdout_bytes_seen=(
                    _optional_int(status_payload.get("stdout_bytes_seen")) or 0
                ),
                stderr_bytes_seen=(
                    _optional_int(status_payload.get("stderr_bytes_seen")) or 0
                ),
                last_update_at=_optional_string(status_payload.get("last_update_at")),
            )
            if observed_status in {"finished", "cancelled"}:
                mark_job_terminal_notice_dispatched(
                    workspace_dir=self._workspace_dir,
                    job_id=job_id,
                    notice_kind=_notice_kind_for_status(
                        status=observed_status,
                        exit_code=status_payload.get("exit_code"),
                    ),
                )
                self._tracked_job_ids.discard(job_id)
                self._forget_job(job_id)
        except BashJobError:
            LOGGER.exception(
                "Failed to record manual detached bash observation for job %s.",
                job_id,
            )

    async def _run_loop(self) -> None:
        while True:
            main_notices: list[BashJobNotice] = []
            subagent_notices: dict[str, list[BashJobNotice]] = {}
            async with self._job_operation_lock:
                self._recover_tracked_jobs()
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
            self._forget_job(job_id)
            return None

        if record.owner_route_id != self._route_id:
            self._tracked_job_ids.discard(job_id)
            self._forget_job(job_id)
            return None
        if record.terminal_notice_dispatched_at is not None:
            self._tracked_job_ids.discard(job_id)
            self._forget_job(job_id)
            return None
        deferred_marker = self._deferred_notice_markers.get(job_id)
        if deferred_marker is not None:
            if deferred_marker == _notice_delivery_marker(record):
                recheck_at = self._deferred_notice_recheck_at.get(job_id)
                if (
                    recheck_at is None
                    or asyncio.get_running_loop().time() < recheck_at
                ):
                    return None
            self._deferred_notice_markers.pop(job_id, None)
            self._deferred_notice_recheck_at.pop(job_id, None)
        if not self._followups_allowed():
            return None
        if record.owner_agent_kind == "subagent" and record.owner_subagent_id:
            if self._subagent_turn_active(record.owner_subagent_id):
                return None
        elif self._main_turn_active():
            return None

        if _is_managed_service(record):
            now = asyncio.get_running_loop().time()
            if now < self._next_service_poll_at.get(job_id, 0.0):
                return None
            self._next_service_poll_at[job_id] = (
                now + _MANAGED_SERVICE_POLL_INTERVAL_SECONDS
            )

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
            status=status,
            stdout_bytes_dropped=stdout_bytes_dropped,
            stderr_bytes_dropped=stderr_bytes_dropped,
            progress_hint=progress_hint,
        )
        exit_code = _optional_int(status_result.metadata.get("exit_code"))
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
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            stdout_log_path=str(status_result.metadata.get("stdout_path", record.stdout_path)),
            stderr_log_path=str(status_result.metadata.get("stderr_path", record.stderr_path)),
            stdout_bytes_seen=_optional_int(status_result.metadata.get("stdout_bytes_seen")) or 0,
            stderr_bytes_seen=_optional_int(status_result.metadata.get("stderr_bytes_seen")) or 0,
            stdout_bytes_dropped=stdout_bytes_dropped,
            stderr_bytes_dropped=stderr_bytes_dropped,
            progress_hint=progress_hint,
            command_sha256=record.command_sha256,
            workspace_revision=record.workspace_revision,
            runtime_seconds=_runtime_seconds(
                started_at=str(status_result.metadata.get("started_at", record.launched_at)),
                finished_at=_optional_string(status_result.metadata.get("finished_at")),
                cancelled_at=_optional_string(status_result.metadata.get("cancelled_at")),
            ),
            termination_signal=_termination_signal(exit_code),
            process_exit_success=status == "finished" and exit_code == 0,
            stdout_sha256=hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            stderr_sha256=hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            skill_import_notice=self._skill_import_notice_for_terminal_success(
                status=status,
                exit_code=exit_code,
            ),
        )

    async def _dispatch_main_notices(self, notices: tuple[BashJobNotice, ...]) -> None:
        try:
            accepted = await self._handle_main_notices(notices)
        except Exception:
            LOGGER.exception("Detached bash follow-up dispatch failed for main notices.")
            return
        if accepted:
            self._defer_unrecorded_notices(notices)

    async def _dispatch_subagent_notices(self, notices: tuple[BashJobNotice, ...]) -> None:
        try:
            accepted = await self._handle_subagent_notices(notices)
        except Exception:
            subagent_id = notices[0].owner_subagent_id if notices else None
            LOGGER.exception(
                "Detached bash follow-up dispatch failed for subagent %s.",
                subagent_id or "unknown",
            )
            return
        if accepted:
            self._defer_unrecorded_notices(notices)

    def _defer_unrecorded_notices(self, notices: tuple[BashJobNotice, ...]) -> None:
        """Avoid redispatching a notice while its owner queue has not consumed it yet."""
        for notice in notices:
            try:
                _, record = load_job(self._workspace_dir, notice.job_id)
            except BashJobError:
                self._forget_job(notice.job_id)
                continue
            if _notice_was_recorded(record=record, notice=notice):
                continue
            self._deferred_notice_markers[notice.job_id] = _notice_delivery_marker(record)
            if notice.status == "running":
                self._deferred_notice_recheck_at[notice.job_id] = (
                    asyncio.get_running_loop().time()
                    + _DEFERRED_RUNNING_NOTICE_RECHECK_SECONDS
                )
            else:
                self._deferred_notice_recheck_at.pop(notice.job_id, None)

    def _forget_job(self, job_id: str) -> None:
        self._deferred_notice_markers.pop(job_id, None)
        self._deferred_notice_recheck_at.pop(job_id, None)
        self._next_service_poll_at.pop(job_id, None)

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

    def _skill_import_notice_for_terminal_success(
        self,
        *,
        status: str,
        exit_code: int | None,
    ) -> str | None:
        if status != "finished" or exit_code != 0:
            return None
        try:
            import_result = import_staged_skills(
                SkillsSettings.from_workspace_dir(self._workspace_dir)
            )
        except Exception:
            LOGGER.exception("Skill import scan after detached bash completion failed.")
            return None
        return render_skill_import_notice(import_result)

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
    if _is_managed_service(record):
        # A readiness-verified service is a managed route resource, not pending task work.
        # Keep polling it for terminal failure, but never wake the model just because it is
        # still serving normally.
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
    if total_bytes_seen == 0 and previous_total_bytes_seen == 0:
        if record.attention_notice_dispatched_at is not None:
            return None
        launched_at = _parse_optional_iso(record.launched_at)
        if (
            launched_at is not None
            and (datetime.now(UTC) - launched_at).total_seconds()
            >= _NEEDS_ATTENTION_NO_OUTPUT_SECONDS
        ):
            return "bash_job_needs_attention"
    return None


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
    status: str,
    stdout_bytes_dropped: int,
    stderr_bytes_dropped: int,
    progress_hint: str | None,
) -> str:
    if status != "running":
        return notice_kind
    if notice_kind == "bash_job_needs_attention":
        return notice_kind
    if notice_kind != "bash_job_output_grew":
        return notice_kind
    dropped_bytes = max(stdout_bytes_dropped, stderr_bytes_dropped)
    if dropped_bytes < _NEEDS_ATTENTION_DROPPED_BYTES:
        return notice_kind
    normalized_hint = (progress_hint or "").strip()
    if normalized_hint and (len(normalized_hint) <= 4 or _looks_repetitive_hint(normalized_hint)):
        return "bash_job_needs_attention"
    return notice_kind


def _runtime_seconds(
    *,
    started_at: str,
    finished_at: str | None,
    cancelled_at: str | None,
) -> float | None:
    try:
        started = datetime.fromisoformat(started_at)
        ended_raw = finished_at or cancelled_at
        ended = datetime.fromisoformat(ended_raw) if ended_raw else datetime.now(UTC)
        return round(max(0.0, (ended - started).total_seconds()), 3)
    except (TypeError, ValueError):
        return None


def _termination_signal(exit_code: int | None) -> int | None:
    if exit_code is None or exit_code == 0:
        return None
    if exit_code < 0:
        return -exit_code
    if 128 < exit_code <= 255:
        return exit_code - 128
    return None


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


def _is_managed_service(record: BashJobRecord) -> bool:
    return record.readiness_verified and (
        record.service_port is not None or record.readiness_url is not None
    )


def _notice_delivery_marker(record: BashJobRecord) -> _NoticeDeliveryMarker:
    return _NoticeDeliveryMarker(
        last_progress_notice_kind=record.last_progress_notice_kind,
        last_progress_notice_at=record.last_progress_notice_at,
        last_progress_notice_status=record.last_progress_notice_status,
        last_progress_notice_stdout_bytes_seen=(
            record.last_progress_notice_stdout_bytes_seen
        ),
        last_progress_notice_stderr_bytes_seen=(
            record.last_progress_notice_stderr_bytes_seen
        ),
        last_progress_notice_last_update_at=(
            record.last_progress_notice_last_update_at
        ),
        terminal_notice_dispatched_at=record.terminal_notice_dispatched_at,
    )


def _notice_was_recorded(*, record: BashJobRecord, notice: BashJobNotice) -> bool:
    if notice.status in {"finished", "cancelled"}:
        return record.terminal_notice_dispatched_at is not None
    return (
        record.last_progress_notice_kind == notice.notice_kind
        and record.last_progress_notice_status == notice.status
        and (record.last_progress_notice_stdout_bytes_seen or 0)
        >= notice.stdout_bytes_seen
        and (record.last_progress_notice_stderr_bytes_seen or 0)
        >= notice.stderr_bytes_seen
    )


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
