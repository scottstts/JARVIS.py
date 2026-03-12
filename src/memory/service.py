"""Memory orchestration entry point."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from llm import LLMService
from storage import ConversationRecord

from .bootstrap import (
    checksum_bundle_for_documents,
    render_core_bootstrap,
    render_ongoing_bootstrap,
)
from .chunker import chunk_document
from .config import MemorySettings
from .dirty_scan import scan_dirty_documents
from .index_db import MemoryIndexDB
from .maintenance import MaintenanceContext, MemoryMaintenanceManager
from .markdown_store import MarkdownMemoryStore, slugify
from .retrieval import MemoryRetriever
from .types import (
    DirtyDocument,
    Fact,
    IntegrityIssue,
    MaintenanceRunResult,
    MemoryDocument,
    MemorySearchResponse,
    MemoryWriteResult,
    Relation,
    SourceReference,
)
from .validator import MemoryValidationError

LOGGER = logging.getLogger(__name__)


class MemoryService:
    """Coordinates canonical Markdown memory, indexing, retrieval, and maintenance."""

    def __init__(
        self,
        *,
        settings: MemorySettings | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self._settings = settings or MemorySettings.from_env()
        self._llm_service = llm_service
        self._store = MarkdownMemoryStore(self._settings)
        self._index_db = MemoryIndexDB(self._settings)
        self._retriever = MemoryRetriever(index_db=self._index_db, llm_service=llm_service)
        self._maintenance = MemoryMaintenanceManager(
            settings=self._settings,
            context=MaintenanceContext(
                list_documents=self._store.read_all_documents,
                ensure_daily_for_today=self.ensure_daily_for_today,
                refresh_document=self._refresh_document,
                archive_document=self._archive_document,
                recompute_priorities=self.recompute_priorities,
                repair_missing_embeddings=self._repair_missing_embeddings,
                integrity_check=self._integrity_check_dicts,
            ),
        )

    @property
    def settings(self) -> MemorySettings:
        return self._settings

    async def ensure_index_synced(self) -> tuple[DirtyDocument, ...]:
        markdown_paths = self._store.list_markdown_paths()
        dirty_documents = scan_dirty_documents(
            markdown_paths=markdown_paths,
            indexed_checksums=self._index_db.indexed_checksums(),
        )
        repairable_paths = self._index_db.paths_missing_searchable_chunks()
        if repairable_paths:
            dirty_documents = dirty_documents + tuple(
                DirtyDocument(
                    path=path,
                    detected_at=_utc_now_iso(),
                    reason="missing_searchable_chunks",
                )
                for path in repairable_paths
            )
        if dirty_documents:
            self._index_db.mark_dirty_documents(dirty_documents)
        recorded_dirty = self._index_db.list_dirty_documents()
        if recorded_dirty:
            await self._reconcile_dirty_documents(recorded_dirty)
        return recorded_dirty

    async def search(
        self,
        *,
        query: str,
        mode: str = "hybrid",
        scopes: tuple[str, ...] = ("core", "ongoing", "daily"),
        top_k: int | None = None,
        daily_lookback_days: int | None = None,
        expand: int | None = None,
        include_expired: bool = False,
        route_id: str | None = None,
        session_id: str | None = None,
    ) -> MemorySearchResponse:
        await self.ensure_index_synced()
        response = await self._retriever.search(
            query=query,
            mode=mode,
            scopes=scopes,
            top_k=top_k or self._settings.search_default_top_k,
            daily_lookback_days=daily_lookback_days or self._settings.daily_lookback_days,
            expand=self._settings.graph_default_expand if expand is None else expand,
            include_expired=include_expired,
        )
        self._index_db.record_accesses(
            occurred_at=_utc_now_iso(),
            route_id=route_id,
            session_id=session_id,
            tool_name="memory_search",
            query=query,
            mode=mode,
            results=[
                {
                    "document_id": result.document_id,
                    "chunk_id": result.section_path,
                    "score": result.score,
                }
                for result in response.results
            ],
        )
        return response

    async def get_document(
        self,
        *,
        document_id: str | None = None,
        path: str | None = None,
        section_path: str | None = None,
        include_frontmatter: bool = True,
        include_sources: bool = False,
    ) -> str:
        await self.ensure_index_synced()
        row = self._index_db.document_for_id_or_path(document_id=document_id, path=path)
        if row is None:
            raise ValueError("Memory document not found.")
        document = self._store.read_document(Path(str(row["path"])))
        rendered = document.raw_markdown if include_frontmatter else document.body_markdown
        if section_path:
            if section_path not in document.sections:
                raise ValueError(f"Section '{section_path}' not found in memory document.")
            rendered = f"## {section_path}\n{document.sections[section_path].rstrip()}\n"
        if include_sources and document.source_refs:
            rendered = rendered.rstrip() + "\n\n## Sources\n" + "\n".join(
                f"- {source_ref.source_type}:{source_ref.source_ref_id}"
                for source_ref in document.source_refs
            ) + "\n"
        return rendered

    async def write(
        self,
        *,
        operation: str,
        target_kind: str,
        document_id: str | None = None,
        title: str | None = None,
        summary: str | None = None,
        priority: int | None = None,
        pinned: bool | None = None,
        locked: bool | None = None,
        review_after: str | None = None,
        expires_at: str | None = None,
        facts: list[dict[str, Any]] | None = None,
        relations: list[dict[str, Any]] | None = None,
        body_sections: dict[str, str] | None = None,
        source_refs: list[dict[str, Any]] | None = None,
        route_id: str | None = None,
        session_id: str | None = None,
        date: str | None = None,
        timezone_name: str | None = None,
        close_reason: str | None = None,
    ) -> MemoryWriteResult:
        await self.ensure_index_synced()
        if operation == "append_daily":
            document = await self._append_daily(
                body_sections=body_sections or {},
                title=title,
                summary=summary,
                route_id=route_id,
                session_id=session_id,
                date=date,
                timezone_name=timezone_name,
            )
            return MemoryWriteResult(
                operation="append_daily",
                document_id=document.document_id,
                path=document.path,
                summary=f"Appended daily memory for {document.date}.",
                changed_paths=(document.path,),
            )

        current = await self._resolve_existing_document(document_id=document_id, title=title, target_kind=target_kind)
        if operation == "archive":
            if current is None:
                raise ValueError("Cannot archive a missing memory document.")
            archived = await self._archive_document(current)
            return MemoryWriteResult(
                operation="archive",
                document_id=archived.document_id,
                path=archived.path,
                summary=f"Archived {archived.kind} memory '{archived.title}'.",
                changed_paths=(current.path, archived.path),
            )
        if operation == "close":
            if current is None:
                raise ValueError("Cannot close a missing memory document.")
            if current.kind != "ongoing":
                raise ValueError("Only ongoing memory documents can be closed.")
            updated = replace(
                current,
                status="closed",
                close_reason=close_reason or current.close_reason,
                updated_at=_utc_now_iso(),
            )
            persisted = await self._archive_document(updated)
            return MemoryWriteResult(
                operation="close",
                document_id=persisted.document_id,
                path=persisted.path,
                summary=f"Closed ongoing memory '{persisted.title}'.",
                changed_paths=(current.path, persisted.path),
            )

        if operation in {"promote", "demote"}:
            if current is None:
                raise ValueError(f"Cannot {operation} a missing memory document.")
            promoted = await self._migrate_document(current=current, target_kind=target_kind, operation=operation)
            return MemoryWriteResult(
                operation=operation,  # type: ignore[arg-type]
                document_id=promoted.document_id,
                path=promoted.path,
                summary=f"{operation.title()}d memory '{promoted.title}' into {promoted.kind}.",
                changed_paths=(current.path, promoted.path),
            )

        if operation not in {"create", "upsert"}:
            raise ValueError(f"Unsupported memory write operation: {operation}")

        now = _utc_now_iso()
        base_document = current or self._build_new_document(
            kind=target_kind,
            title=title,
            date=date,
            timezone_name=timezone_name,
            route_id=route_id,
            session_id=session_id,
        )
        resolved_source_refs = _normalize_source_refs(
            payloads=source_refs,
            existing=base_document.source_refs,
            route_id=route_id,
            session_id=session_id,
        )
        resolved_sections = _merge_sections(base_document=base_document, overrides=body_sections or {})
        resolved_summary = summary if summary is not None else base_document.summary
        if target_kind != "daily":
            resolved_sections = _seed_summary_section(
                sections=resolved_sections,
                summary=resolved_summary,
            )
        if resolved_summary is None and "Summary" in resolved_sections and resolved_sections["Summary"].strip():
            resolved_summary = _first_paragraph(resolved_sections["Summary"])
        updated_document = replace(
            base_document,
            title=title or base_document.title,
            summary=resolved_summary if target_kind != "daily" else None,
            priority=priority if priority is not None else base_document.priority,
            pinned=pinned if pinned is not None else base_document.pinned,
            locked=locked if locked is not None else base_document.locked,
            review_after=review_after if review_after is not None else base_document.review_after,
            expires_at=expires_at if expires_at is not None else base_document.expires_at,
            updated_at=now,
            sections=resolved_sections,
            facts=_normalize_facts(
                payloads=facts,
                existing=base_document.facts,
                source_refs=resolved_source_refs,
            )
            if target_kind != "daily"
            else (),
            relations=_normalize_relations(
                payloads=relations,
                existing=base_document.relations,
                source_refs=resolved_source_refs,
            )
            if target_kind != "daily"
            else (),
            source_refs=resolved_source_refs if target_kind != "daily" else base_document.source_refs,
            route_ids=tuple(dict.fromkeys(base_document.route_ids + ((route_id,) if route_id else ()))),
            session_ids=tuple(dict.fromkeys(base_document.session_ids + ((session_id,) if session_id else ()))),
            close_reason=close_reason if close_reason is not None else base_document.close_reason,
        )
        persisted = await self._persist_document(updated_document)
        return MemoryWriteResult(
            operation=operation,  # type: ignore[arg-type]
            document_id=persisted.document_id,
            path=persisted.path,
            summary=f"{operation.title()}d {persisted.kind} memory '{persisted.title}'.",
            changed_paths=(persisted.path,),
        )

    async def render_bootstrap_messages(self) -> tuple[str, str]:
        await self.ensure_index_synced()
        core_documents = tuple(
            self._store.read_document(Path(item["path"]))
            for item in self._index_db.bootstrap_documents(kind="core")
        )
        ongoing_documents_unfiltered = tuple(
            self._store.read_document(Path(item["path"]))
            for item in self._index_db.bootstrap_documents(kind="ongoing")
        )
        ongoing_documents = tuple(
            document
            for document in ongoing_documents_unfiltered
            if (
                (
                    document.status == "active"
                    and (
                        document.expires_at is None
                        or _parse_iso(document.expires_at) is None
                        or _parse_iso(document.expires_at) > datetime.now(timezone.utc)
                    )
                )
                or bool(document.pinned)
            )
        )

        core_cache_key = "core_bootstrap"
        ongoing_cache_key = "ongoing_bootstrap"
        core_bundle = checksum_bundle_for_documents(core_documents)
        ongoing_bundle = checksum_bundle_for_documents(ongoing_documents)

        core_cached = self._index_db.render_bootstrap_cache_get(core_cache_key)
        ongoing_cached = self._index_db.render_bootstrap_cache_get(ongoing_cache_key)

        if core_cached is not None and str(core_cached["checksum_bundle"]) == core_bundle:
            core_text = str(core_cached["content"])
        else:
            core_text = render_core_bootstrap(
                core_documents,
                token_budget=self._settings.core_bootstrap_max_tokens,
            )
            self._index_db.render_bootstrap_cache_set(
                cache_key=core_cache_key,
                generated_at=_utc_now_iso(),
                content=core_text,
                token_estimate=max(1, len(core_text) // 4),
                checksum_bundle=core_bundle,
            )

        if ongoing_cached is not None and str(ongoing_cached["checksum_bundle"]) == ongoing_bundle:
            ongoing_text = str(ongoing_cached["content"])
        else:
            ongoing_text = render_ongoing_bootstrap(
                ongoing_documents,
                token_budget=self._settings.ongoing_bootstrap_max_tokens,
            )
            self._index_db.render_bootstrap_cache_set(
                cache_key=ongoing_cache_key,
                generated_at=_utc_now_iso(),
                content=ongoing_text,
                token_estimate=max(1, len(ongoing_text) // 4),
                checksum_bundle=ongoing_bundle,
            )
        return core_text, ongoing_text

    async def run_due_maintenance(self) -> tuple[MaintenanceRunResult, ...]:
        await self.ensure_index_synced()
        results = await self._maintenance.run_due_jobs()
        for result in results:
            self._index_db.record_maintenance_run(
                job_name=result.job_name,
                started_at=_utc_now_iso(),
                finished_at=_utc_now_iso(),
                status=result.status,
                summary=result.summary,
            )
        return results

    async def reindex_dirty(self) -> tuple[DirtyDocument, ...]:
        dirty = self._index_db.list_dirty_documents()
        if dirty:
            await self._reconcile_dirty_documents(dirty)
        return dirty

    async def reindex_all(self) -> tuple[Path, ...]:
        documents = self._store.read_all_documents()
        indexed_paths: list[Path] = []
        current_paths = {document.path for document in documents}
        indexed_checksums = self._index_db.indexed_checksums()
        existing_dirty = self._index_db.list_dirty_documents()
        for indexed_path in tuple(Path(path) for path in indexed_checksums):
            if indexed_path in current_paths:
                continue
            self._index_db.remove_document(path=indexed_path)
        for document in documents:
            await self._persist_document(document, run_dirty_scan=False)
            indexed_paths.append(document.path)
        self._index_db.clear_dirty_documents(tuple(item.path for item in existing_dirty))
        return tuple(indexed_paths)

    async def rebuild_embeddings(self) -> dict[str, Any]:
        documents = self._store.read_all_documents()
        rebuilt = 0
        for document in documents:
            chunks = chunk_document(document)
            await self._index_db.upsert_embeddings_for_document(
                document=document,
                chunks=chunks,
                llm_service=self._llm_service,
            )
            rebuilt += 1
        return {
            "semantic_enabled": self._index_db.semantic_enabled,
            "semantic_error": self._index_db.semantic_error,
            "rebuilt_documents": rebuilt,
        }

    async def _repair_missing_embeddings(self) -> dict[str, Any]:
        if self._llm_service is None:
            return {
                "checked_documents": 0,
                "repaired_documents": 0,
                "failed_documents": 0,
                "reason": "no_llm_service",
            }
        if not self._index_db.semantic_enabled:
            return {
                "checked_documents": 0,
                "repaired_documents": 0,
                "failed_documents": 0,
                "reason": "semantic_disabled",
            }
        documents = self._store.read_all_documents()
        checked = 0
        repaired = 0
        failed = 0
        vector_table_missing = not self._index_db.embedding_vector_table_exists()
        for document in documents:
            chunks = chunk_document(document)
            expected_items = self._index_db.expected_embedding_item_count(
                document=document,
                chunks=chunks,
            )
            if expected_items == 0:
                continue
            checked += 1
            actual_items = self._index_db.embedding_vector_count_for_document(document.document_id)
            if not vector_table_missing and actual_items == expected_items:
                continue
            try:
                await self._index_db.upsert_embeddings_for_document(
                    document=document,
                    chunks=chunks,
                    llm_service=self._llm_service,
                )
            except Exception:
                failed += 1
                LOGGER.exception("Failed repairing missing memory embeddings for: %s", document.path)
                continue
            refreshed_items = self._index_db.embedding_vector_count_for_document(document.document_id)
            if refreshed_items == expected_items:
                repaired += 1
            else:
                failed += 1
            vector_table_missing = not self._index_db.embedding_vector_table_exists()
        return {
            "checked_documents": checked,
            "repaired_documents": repaired,
            "failed_documents": failed,
            "vector_table_missing": vector_table_missing,
        }

    async def render_bootstrap_preview(self) -> str:
        core_text, ongoing_text = await self.render_bootstrap_messages()
        sections: list[str] = []
        if core_text:
            sections.append("Core Bootstrap\n" + core_text)
        if ongoing_text:
            sections.append("Ongoing Bootstrap\n" + ongoing_text)
        return "\n\n".join(sections).strip()

    async def reflect_completed_turn(
        self,
        *,
        route_id: str | None,
        session_id: str,
        records: tuple[ConversationRecord, ...],
    ) -> tuple[MemoryWriteResult, ...]:
        from .reflection import MemoryReflectionPlanner

        if not self._settings.enable_reflection:
            return ()

        planner = MemoryReflectionPlanner(settings=self._settings, llm_service=self._llm_service)
        active_documents = self._store.read_all_documents()
        core_titles = tuple(document.title for document in active_documents if document.kind == "core" and not document.archived)
        ongoing_titles = tuple(
            document.title for document in active_documents if document.kind == "ongoing" and document.status == "active" and not document.archived
        )
        plan = await planner.plan_turn(
            route_id=route_id,
            session_id=session_id,
            records=records,
            active_core_titles=core_titles,
            active_ongoing_titles=ongoing_titles,
        )
        applied: list[MemoryWriteResult] = []
        for action in plan.actions:
            if action.action == "ignore":
                continue
            if action.action == "append_daily":
                payload = action.payload
                applied.append(
                    await self.write(
                        operation="append_daily",
                        target_kind="daily",
                        body_sections=_coerce_body_sections(payload.get("body_sections")),
                        route_id=route_id,
                        session_id=session_id,
                    )
                )
                continue
            if action.action in {"create_ongoing", "update_ongoing"}:
                if action.confidence == "low" or not self._settings.enable_auto_apply_ongoing:
                    continue
                payload = action.payload
                applied.append(
                    await self.write(
                        operation="create" if action.action == "create_ongoing" else "upsert",
                        target_kind="ongoing",
                        document_id=_optional_str(payload.get("document_id")),
                        title=_optional_str(payload.get("title")),
                        summary=_optional_str(payload.get("summary")),
                        priority=_optional_int(payload.get("priority")),
                        pinned=_optional_bool(payload.get("pinned")),
                        locked=_optional_bool(payload.get("locked")),
                        review_after=_optional_str(payload.get("review_after")) or _utc_now_iso(),
                        expires_at=_optional_str(payload.get("expires_at")),
                        facts=_coerce_list_of_dicts(payload.get("facts")),
                        relations=_coerce_list_of_dicts(payload.get("relations")),
                        body_sections=_coerce_body_sections(payload.get("body_sections")),
                        source_refs=_coerce_list_of_dicts(payload.get("source_refs")),
                        route_id=route_id,
                        session_id=session_id,
                    )
                )
                continue
            if action.action in {"create_core", "update_core"}:
                if action.confidence == "low" or not self._settings.enable_auto_apply_core:
                    continue
                payload = action.payload
                explicit_request = bool(payload.get("explicit_user_request"))
                if not explicit_request:
                    continue
                applied.append(
                    await self.write(
                        operation="create" if action.action == "create_core" else "upsert",
                        target_kind="core",
                        document_id=_optional_str(payload.get("document_id")),
                        title=_optional_str(payload.get("title")),
                        summary=_optional_str(payload.get("summary")),
                        priority=_optional_int(payload.get("priority")),
                        pinned=_optional_bool(payload.get("pinned")),
                        locked=_optional_bool(payload.get("locked")),
                        review_after=_optional_str(payload.get("review_after")),
                        expires_at=_optional_str(payload.get("expires_at")),
                        facts=_coerce_list_of_dicts(payload.get("facts")),
                        relations=_coerce_list_of_dicts(payload.get("relations")),
                        body_sections=_coerce_body_sections(payload.get("body_sections")),
                        source_refs=_coerce_list_of_dicts(payload.get("source_refs")),
                        route_id=route_id,
                        session_id=session_id,
                    )
                )
                continue
            if action.action == "close_ongoing":
                payload = action.payload
                try:
                    applied.append(
                        await self.write(
                            operation="close",
                            target_kind="ongoing",
                            document_id=_optional_str(payload.get("document_id")),
                            title=_optional_str(payload.get("title")),
                            close_reason=_optional_str(payload.get("close_reason")),
                        )
                    )
                except Exception:
                    LOGGER.exception("Failed applying close_ongoing reflection action.")
        return tuple(applied)

    async def flush_before_compaction(
        self,
        *,
        route_id: str | None,
        session_id: str,
        records: tuple[ConversationRecord, ...],
    ) -> tuple[MemoryWriteResult, ...]:
        return await self.reflect_completed_turn(route_id=route_id, session_id=session_id, records=records)

    async def integrity_check(self) -> tuple[IntegrityIssue, ...]:
        issues = await self._integrity_check_dicts()
        return tuple(
            IntegrityIssue(
                path=Path(issue["path"]) if issue.get("path") else None,
                severity=str(issue["severity"]),  # type: ignore[arg-type]
                code=str(issue["code"]),
                message=str(issue["message"]),
            )
            for issue in issues
        )

    async def ensure_daily_for_today(self) -> MemoryDocument:
        local_now = datetime.now(ZoneInfo(self._settings.default_timezone))
        date = local_now.date().isoformat()
        path = self._settings.daily_dir / f"{date}.md"
        if path.exists():
            return self._store.read_document(path)

        document = self._build_new_document(
            kind="daily",
            title=f"Daily Log: {date}",
            date=date,
            timezone_name=self._settings.default_timezone,
            route_id=None,
            session_id=None,
        )
        return await self._persist_document(document)

    async def recompute_priorities(self) -> int:
        changed = 0
        for document in self._store.read_all_documents():
            if document.kind not in {"core", "ongoing"} or document.priority is None:
                continue
            access_score = 0
            if document.pinned:
                access_score += 10
            if document.kind == "core":
                access_score += 5
            next_priority = max(0, min(100, document.priority + access_score))
            if next_priority == document.priority:
                continue
            await self._persist_document(replace(document, priority=next_priority, updated_at=_utc_now_iso()))
            changed += 1
        return changed

    async def _reconcile_dirty_documents(self, dirty_documents: tuple[DirtyDocument, ...]) -> None:
        processed_paths: list[Path] = []
        for dirty in dirty_documents:
            if dirty.reason == "missing_from_disk":
                self._index_db.remove_document(path=dirty.path)
                processed_paths.append(dirty.path)
                continue
            try:
                document = self._store.read_document(dirty.path)
            except (FileNotFoundError, MemoryValidationError, ValueError):
                LOGGER.exception("Failed reconciling memory document: %s", dirty.path)
                continue
            await self._persist_document(document, run_dirty_scan=False)
            processed_paths.append(dirty.path)
        self._index_db.clear_dirty_documents(tuple(processed_paths))

    async def _persist_document(self, document: MemoryDocument, *, run_dirty_scan: bool = True) -> MemoryDocument:
        persisted = self._store.write_document(document)
        chunks = chunk_document(persisted)
        self._index_db.upsert_document(persisted, chunks)
        await self._index_db.upsert_embeddings_for_document(
            document=persisted,
            chunks=chunks,
            llm_service=self._llm_service,
        )
        await self._apply_relation_conflicts()
        if run_dirty_scan:
            await self.ensure_index_synced()
        return self._store.read_document(persisted.path)

    async def _refresh_document(self, document: MemoryDocument) -> MemoryDocument:
        return await self._persist_document(document, run_dirty_scan=False)

    async def _archive_document(self, document: MemoryDocument) -> MemoryDocument:
        archived = self._store.archive_document(document)
        self._index_db.remove_document(path=document.path, document_id=document.document_id)
        chunks = chunk_document(archived)
        self._index_db.upsert_document(archived, chunks)
        await self._index_db.upsert_embeddings_for_document(
            document=archived,
            chunks=chunks,
            llm_service=self._llm_service,
        )
        return archived

    async def _resolve_existing_document(
        self,
        *,
        document_id: str | None,
        title: str | None,
        target_kind: str,
    ) -> MemoryDocument | None:
        row = None
        if document_id is not None:
            row = self._index_db.document_for_id_or_path(document_id=document_id)
        if row is None and title:
            candidate_path = (
                self._settings.daily_dir / f"{title}.md"
                if target_kind == "daily"
                else self._store.active_path_for(kind=target_kind, title=title)
            )
            if candidate_path.exists():
                return self._store.read_document(candidate_path)
        if row is None:
            return None
        return self._store.read_document(Path(str(row["path"])))

    def _build_new_document(
        self,
        *,
        kind: str,
        title: str | None,
        date: str | None,
        timezone_name: str | None,
        route_id: str | None,
        session_id: str | None,
    ) -> MemoryDocument:
        now = _utc_now_iso()
        if kind == "daily":
            resolved_timezone = timezone_name or self._settings.default_timezone
            resolved_date = date or datetime.now(ZoneInfo(resolved_timezone)).date().isoformat()
            resolved_title = title or f"Daily Log: {resolved_date}"
            path = self._settings.daily_dir / f"{resolved_date}.md"
            return MemoryDocument(
                path=path,
                memory_id=f"daily_{resolved_date}",
                kind="daily",
                title=resolved_title,
                status="active",
                created_at=now,
                updated_at=now,
                sections=OrderedDict(
                    (
                        ("Notable Events", ""),
                        ("Decisions", ""),
                        ("Active Commitments", ""),
                        ("Open Loops", ""),
                        ("Artifacts", ""),
                        ("Candidate Promotions", ""),
                    )
                ),
                checksum="",
                raw_markdown="",
                date=resolved_date,
                timezone=resolved_timezone,
                route_ids=((route_id,) if route_id else ()),
                session_ids=((session_id,) if session_id else ()),
            )

        if title is None or not title.strip():
            raise ValueError("title is required for core and ongoing memory documents.")
        path = self._store.active_path_for(kind=kind, title=title.strip())
        summary_sections = OrderedDict(
            (
                ("Summary", summary_placeholder(kind)),
                ("Details" if kind == "core" else "Current State", ""),
                ("Notes" if kind == "core" else "Open Loops", ""),
            )
        )
        if kind == "ongoing":
            summary_sections = OrderedDict(
                (
                    ("Summary", ""),
                    ("Current State", ""),
                    ("Open Loops", ""),
                    ("Artifacts", ""),
                    ("Notes", ""),
                )
            )
        return MemoryDocument(
            path=path,
            memory_id=f"{kind}_{slugify(title)}_{uuid4().hex[:8]}",
            kind=kind,  # type: ignore[arg-type]
            title=title.strip(),
            status="active",
            created_at=now,
            updated_at=now,
            sections=summary_sections,
            checksum="",
            raw_markdown="",
            summary=None,
            priority=50,
            pinned=False,
            locked=False,
            confidence="medium",
            review_after=now if kind == "ongoing" else None,
            expires_at=None,
            tags=(),
            aliases=(),
            facts=(),
            relations=(),
            source_refs=_normalize_source_refs(None, (), route_id=route_id, session_id=session_id),
            entity_refs=(),
            completion_criteria=() if kind == "core" else (),
            close_reason=None,
        )

    async def _append_daily(
        self,
        *,
        body_sections: dict[str, str],
        title: str | None,
        summary: str | None,
        route_id: str | None,
        session_id: str | None,
        date: str | None,
        timezone_name: str | None,
    ) -> MemoryDocument:
        if date:
            path = self._settings.daily_dir / f"{date}.md"
            document = self._store.read_document(path) if path.exists() else self._build_new_document(
                kind="daily",
                title=f"Daily Log: {date}",
                date=date,
                timezone_name=timezone_name,
                route_id=route_id,
                session_id=session_id,
            )
        else:
            document = await self.ensure_daily_for_today()
        sections = OrderedDict(document.sections)
        resolved_body_sections = _normalize_daily_append_sections(
            body_sections=body_sections,
            title=title,
            summary=summary,
        )
        for heading, content in resolved_body_sections.items():
            if heading not in sections:
                continue
            existing = sections[heading].rstrip()
            addition = content.strip()
            if not addition:
                continue
            sections[heading] = f"{existing}\n{addition}".strip() if existing else addition
        updated = replace(
            document,
            sections=sections,
            updated_at=_utc_now_iso(),
            route_ids=tuple(dict.fromkeys(document.route_ids + ((route_id,) if route_id else ()))),
            session_ids=tuple(dict.fromkeys(document.session_ids + ((session_id,) if session_id else ()))),
        )
        return await self._persist_document(updated)

    async def _migrate_document(
        self,
        *,
        current: MemoryDocument,
        target_kind: str,
        operation: str,
    ) -> MemoryDocument:
        new_document = self._build_new_document(
            kind=target_kind,
            title=current.title,
            date=current.date,
            timezone_name=current.timezone,
            route_id=None,
            session_id=None,
        )
        if target_kind != "daily":
            new_document = replace(
                new_document,
                summary=current.summary,
                priority=current.priority if current.priority is not None else 50,
                pinned=current.pinned if current.pinned is not None else False,
                locked=current.locked if current.locked is not None else False,
                confidence=current.confidence or "medium",
                review_after=current.review_after if target_kind == "ongoing" else None,
                expires_at=current.expires_at if target_kind == "ongoing" else None,
                facts=current.facts,
                relations=current.relations,
                source_refs=current.source_refs,
                entity_refs=current.entity_refs,
                sections=_mapped_sections_for_kind(current, target_kind),
            )
        persisted = await self._persist_document(new_document)
        if operation == "promote":
            await self._archive_document(replace(current, status="archived", updated_at=_utc_now_iso()))
        return persisted

    async def _apply_relation_conflicts(self) -> None:
        documents = [document for document in self._store.read_all_documents() if document.kind in {"core", "ongoing"} and not document.archived]
        groups: dict[tuple[str, str], list[tuple[MemoryDocument, Relation]]] = {}
        for document in documents:
            for relation in document.relations:
                if relation.status != "current" or relation.cardinality != "single":
                    continue
                groups.setdefault((relation.subject, relation.predicate), []).append((document, relation))

        for _key, entries in groups.items():
            if len({relation.object for _document, relation in entries}) <= 1:
                continue
            entries.sort(key=lambda item: item[0].updated_at)
            winner_document, winner_relation = entries[-1]
            for document, relation in entries[:-1]:
                if relation.object == winner_relation.object:
                    continue
                next_relations = []
                changed = False
                for existing in document.relations:
                    if existing.relation_id != relation.relation_id:
                        next_relations.append(existing)
                        continue
                    next_relations.append(
                        replace(
                            existing,
                            status="superseded",
                            valid_to=winner_document.updated_at,
                            last_seen_at=winner_document.updated_at,
                        )
                    )
                    changed = True
                if changed:
                    await self._persist_document(
                        replace(
                            document,
                            relations=tuple(next_relations),
                            updated_at=_utc_now_iso(),
                        ),
                        run_dirty_scan=False,
                    )

    async def _integrity_check_dicts(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for path in self._store.list_markdown_paths():
            try:
                self._store.read_document(path)
            except Exception as exc:
                issues.append(
                    {
                        "path": str(path),
                        "severity": "error",
                        "code": "invalid_markdown_schema",
                        "message": str(exc),
                    }
                )
        main_checks, embeddings_checks = self._index_db.sqlite_integrity()
        if main_checks != ("ok",):
            issues.append(
                {
                    "path": None,
                    "severity": "error",
                    "code": "main_sqlite_integrity",
                    "message": "; ".join(main_checks),
                }
            )
        if embeddings_checks != ("ok",):
            issues.append(
                {
                    "path": None,
                    "severity": "error",
                    "code": "embeddings_sqlite_integrity",
                    "message": "; ".join(embeddings_checks),
                }
            )
        return issues


def summary_placeholder(kind: str) -> str:
    return "" if kind == "ongoing" else ""


def _merge_sections(*, base_document: MemoryDocument, overrides: dict[str, str]) -> OrderedDict[str, str]:
    sections = OrderedDict(base_document.sections)
    for heading, content in overrides.items():
        if heading not in sections:
            continue
        sections[heading] = content.strip()
    return sections


def _seed_summary_section(
    *,
    sections: OrderedDict[str, str],
    summary: str | None,
) -> OrderedDict[str, str]:
    if summary is None or not summary.strip():
        return sections
    if "Summary" not in sections or sections["Summary"].strip():
        return sections
    seeded = OrderedDict(sections)
    seeded["Summary"] = summary.strip()
    return seeded


def _normalize_daily_append_sections(
    *,
    body_sections: dict[str, str],
    title: str | None,
    summary: str | None,
) -> dict[str, str]:
    if any(content.strip() for content in body_sections.values()):
        return body_sections
    auto_entry = _render_daily_auto_entry(title=title, summary=summary)
    if auto_entry is None:
        return body_sections
    return {"Notable Events": auto_entry}


def _render_daily_auto_entry(*, title: str | None, summary: str | None) -> str | None:
    normalized_title = title.strip() if title and title.strip() else None
    normalized_summary = summary.strip() if summary and summary.strip() else None
    if normalized_title and normalized_summary:
        return f"- {normalized_title}: {normalized_summary}"
    if normalized_summary:
        return f"- {normalized_summary}"
    if normalized_title:
        return f"- {normalized_title}"
    return None


def _normalize_source_refs(
    payloads: list[dict[str, Any]] | None,
    existing: tuple[SourceReference, ...],
    *,
    route_id: str | None,
    session_id: str | None,
) -> tuple[SourceReference, ...]:
    if payloads is None:
        if existing:
            return existing
        if route_id is None and session_id is None:
            return ()
        return (
            SourceReference(
                source_ref_id=f"src_{uuid4().hex[:12]}",
                source_type="tool",
                route_id=route_id,
                session_id=session_id,
                record_id=None,
                tool_name="memory_write",
                note="memory_write operation",
                captured_at=_utc_now_iso(),
            ),
        )
    refs: list[SourceReference] = []
    for payload in payloads:
        refs.append(
            SourceReference(
                source_ref_id=_optional_str(payload.get("source_ref_id")) or f"src_{uuid4().hex[:12]}",
                source_type=_optional_str(payload.get("source_type")) or "manual",  # type: ignore[arg-type]
                route_id=_optional_str(payload.get("route_id")) or route_id,
                session_id=_optional_str(payload.get("session_id")) or session_id,
                record_id=_optional_str(payload.get("record_id")),
                tool_name=_optional_str(payload.get("tool_name")),
                note=_optional_str(payload.get("note")),
                captured_at=_optional_str(payload.get("captured_at")) or _utc_now_iso(),
            )
        )
    return tuple(refs)


def _normalize_facts(
    *,
    payloads: list[dict[str, Any]] | None,
    existing: tuple[Fact, ...],
    source_refs: tuple[SourceReference, ...],
) -> tuple[Fact, ...]:
    if payloads is None:
        return existing
    default_source_ref_ids = tuple(source_ref.source_ref_id for source_ref in source_refs)
    facts: list[Fact] = []
    for payload in payloads:
        facts.append(
            Fact(
                fact_id=_optional_str(payload.get("fact_id")) or f"fact_{uuid4().hex[:12]}",
                text=str(payload.get("text", "")).strip(),
                status=_optional_str(payload.get("status")) or "current",  # type: ignore[arg-type]
                confidence=_optional_str(payload.get("confidence")) or "medium",  # type: ignore[arg-type]
                first_seen_at=_optional_str(payload.get("first_seen_at")) or _utc_now_iso(),
                last_seen_at=_optional_str(payload.get("last_seen_at")) or _utc_now_iso(),
                valid_from=_optional_str(payload.get("valid_from")),
                valid_to=_optional_str(payload.get("valid_to")),
                source_ref_ids=tuple(payload.get("source_ref_ids", default_source_ref_ids)) if isinstance(payload.get("source_ref_ids"), list) else default_source_ref_ids,
            )
        )
    return tuple(facts)


def _normalize_relations(
    *,
    payloads: list[dict[str, Any]] | None,
    existing: tuple[Relation, ...],
    source_refs: tuple[SourceReference, ...],
) -> tuple[Relation, ...]:
    if payloads is None:
        return existing
    default_source_ref_ids = tuple(source_ref.source_ref_id for source_ref in source_refs)
    relations: list[Relation] = []
    for payload in payloads:
        relations.append(
            Relation(
                relation_id=_optional_str(payload.get("relation_id")) or f"rel_{uuid4().hex[:12]}",
                subject=str(payload.get("subject", "")).strip(),
                predicate=str(payload.get("predicate", "")).strip(),
                object=str(payload.get("object", "")).strip(),
                status=_optional_str(payload.get("status")) or "current",  # type: ignore[arg-type]
                confidence=_optional_str(payload.get("confidence")) or "medium",  # type: ignore[arg-type]
                cardinality=_optional_str(payload.get("cardinality")) or "single",  # type: ignore[arg-type]
                first_seen_at=_optional_str(payload.get("first_seen_at")) or _utc_now_iso(),
                last_seen_at=_optional_str(payload.get("last_seen_at")) or _utc_now_iso(),
                valid_from=_optional_str(payload.get("valid_from")),
                valid_to=_optional_str(payload.get("valid_to")),
                source_ref_ids=tuple(payload.get("source_ref_ids", default_source_ref_ids)) if isinstance(payload.get("source_ref_ids"), list) else default_source_ref_ids,
            )
        )
    return tuple(relations)


def _mapped_sections_for_kind(document: MemoryDocument, target_kind: str) -> OrderedDict[str, str]:
    if target_kind == document.kind:
        return document.sections
    if target_kind == "core":
        return OrderedDict(
            (
                ("Summary", document.sections.get("Summary", document.summary or "")),
                ("Details", document.sections.get("Current State", document.sections.get("Details", ""))),
                ("Notes", document.sections.get("Notes", document.sections.get("Open Loops", ""))),
            )
        )
    if target_kind == "ongoing":
        return OrderedDict(
            (
                ("Summary", document.sections.get("Summary", document.summary or "")),
                ("Current State", document.sections.get("Details", document.sections.get("Current State", ""))),
                ("Open Loops", document.sections.get("Open Loops", "")),
                ("Artifacts", document.sections.get("Artifacts", "")),
                ("Notes", document.sections.get("Notes", "")),
            )
        )
    return OrderedDict(
        (
            ("Notable Events", document.sections.get("Summary", document.summary or "")),
            ("Decisions", ""),
            ("Active Commitments", ""),
            ("Open Loops", document.sections.get("Open Loops", "")),
            ("Artifacts", document.sections.get("Artifacts", "")),
            ("Candidate Promotions", ""),
        )
    )


def _first_paragraph(value: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in value.split("\n\n") if paragraph.strip()]
    return paragraphs[0] if paragraphs else ""


def _coerce_body_sections(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(section_value).strip()
        for key, section_value in value.items()
        if str(key).strip() and str(section_value).strip()
    }


def _coerce_list_of_dicts(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(dict(item))
    return result


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
