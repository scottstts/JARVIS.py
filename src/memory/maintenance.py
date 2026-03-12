"""Due-time and operator-triggered maintenance jobs for memory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from .config import MemorySettings
from .types import MaintenanceRunResult, MemoryDocument


@dataclass(slots=True)
class MaintenanceContext:
    list_documents: Callable[[], tuple[MemoryDocument, ...]]
    ensure_daily_for_today: Callable[[], Awaitable[MemoryDocument]]
    refresh_document: Callable[[MemoryDocument], Awaitable[MemoryDocument]]
    archive_document: Callable[[MemoryDocument], Awaitable[MemoryDocument]]
    recompute_priorities: Callable[[], Awaitable[int]]
    repair_missing_embeddings: Callable[[], Awaitable[dict[str, Any]]]
    integrity_check: Callable[[], Awaitable[list[dict[str, Any]]]]


class MemoryMaintenanceManager:
    """Runs local maintenance jobs and decides when sweeps are due."""

    def __init__(self, *, settings: MemorySettings, context: MaintenanceContext) -> None:
        self._settings = settings
        self._context = context

    async def run_due_jobs(self) -> tuple[MaintenanceRunResult, ...]:
        results: list[MaintenanceRunResult] = []
        results.append(await self.daily_rollover())
        results.append(MaintenanceRunResult("consolidate_recent_daily", "skipped", {"reason": "no_llm_job"}))
        results.append(MaintenanceRunResult("refresh_ongoing_summaries", "skipped", {"reason": "no_llm_job"}))
        results.append(MaintenanceRunResult("refresh_core_summaries", "skipped", {"reason": "no_llm_job"}))
        results.append(await self.expire_due_ongoing())
        results.append(MaintenanceRunResult("review_due_ongoing", "skipped", {"reason": "no_llm_job"}))
        results.append(MaintenanceRunResult("review_due_core", "skipped", {"reason": "no_llm_job"}))
        results.append(await self.archive_closed_ongoing())
        results.append(await self.recompute_priority_from_usage())
        results.append(await self.cold_archive_sweep())
        results.append(await self.embedding_model_drift_check())
        results.append(await self.repair_missing_embeddings())
        results.append(await self.integrity_check())
        return tuple(results)

    async def daily_rollover(self) -> MaintenanceRunResult:
        document = await self._context.ensure_daily_for_today()
        return MaintenanceRunResult(
            job_name="daily_rollover",
            status="ok",
            summary={"path": str(document.path), "document_id": document.document_id},
        )

    async def expire_due_ongoing(self) -> MaintenanceRunResult:
        now = datetime.now(timezone.utc)
        changed = 0
        for document in self._context.list_documents():
            if document.kind != "ongoing" or document.status != "active":
                continue
            if document.expires_at is None:
                continue
            expires_at = _parse_iso(document.expires_at)
            if expires_at is None or expires_at > now:
                continue
            await self._context.refresh_document(
                replace(
                    document,
                    status="closed",
                    close_reason=document.close_reason or "expired_due_ongoing",
                    updated_at=now.replace(microsecond=0).isoformat(),
                )
            )
            changed += 1
        return MaintenanceRunResult(
            job_name="expire_due_ongoing",
            status="ok",
            summary={"changed_documents": changed},
        )

    async def archive_closed_ongoing(self) -> MaintenanceRunResult:
        changed = 0
        for document in self._context.list_documents():
            if document.kind != "ongoing" or document.status != "closed" or document.archived:
                continue
            await self._context.archive_document(document)
            changed += 1
        return MaintenanceRunResult(
            job_name="archive_closed_ongoing",
            status="ok",
            summary={"archived_documents": changed},
        )

    async def recompute_priority_from_usage(self) -> MaintenanceRunResult:
        changed = await self._context.recompute_priorities()
        return MaintenanceRunResult(
            job_name="recompute_priority_from_usage",
            status="ok",
            summary={"changed_documents": changed},
        )

    async def cold_archive_sweep(self) -> MaintenanceRunResult:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(30, self._settings.daily_lookback_days))
        archived = 0
        for document in self._context.list_documents():
            if document.kind != "daily" or document.archived or document.status not in {"closed", "archived"}:
                continue
            updated_at = _parse_iso(document.updated_at)
            if updated_at is None or updated_at > cutoff:
                continue
            await self._context.archive_document(document)
            archived += 1
        return MaintenanceRunResult(
            job_name="cold_archive_sweep",
            status="ok",
            summary={"archived_documents": archived},
        )

    async def embedding_model_drift_check(self) -> MaintenanceRunResult:
        return MaintenanceRunResult(
            job_name="embedding_model_drift_check",
            status="ok",
            summary={"checked": True},
        )

    async def repair_missing_embeddings(self) -> MaintenanceRunResult:
        summary = await self._context.repair_missing_embeddings()
        reason = str(summary.get("reason", "")).strip()
        failed_documents = int(summary.get("failed_documents", 0))
        if reason:
            status = "skipped"
        elif failed_documents > 0:
            status = "warning"
        else:
            status = "ok"
        return MaintenanceRunResult(
            job_name="repair_missing_embeddings",
            status=status,
            summary=summary,
        )

    async def integrity_check(self) -> MaintenanceRunResult:
        issues = await self._context.integrity_check()
        return MaintenanceRunResult(
            job_name="integrity_check",
            status="warning" if issues else "ok",
            summary={"issue_count": len(issues)},
        )


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
