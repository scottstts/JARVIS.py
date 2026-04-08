"""Memory orchestration entry point."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from jarvis.llm import LLMService
from jarvis.logging_setup import get_application_logger
from jarvis.storage import ConversationRecord

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
    EntityReference,
    Fact,
    IntegrityIssue,
    MaintenanceRunResult,
    MemoryDocument,
    MemorySearchResponse,
    MemoryWriteResult,
    ReflectionAction,
    Relation,
    SourceReference,
)
from .validator import MemoryValidationError

LOGGER = get_application_logger(__name__)


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
        self._retriever = MemoryRetriever(
            index_db=self._index_db,
            llm_service=llm_service,
            settings=self._settings,
        )
        self._maintenance = MemoryMaintenanceManager(
            settings=self._settings,
            context=MaintenanceContext(
                list_documents=self._store.read_all_documents,
                ensure_daily_for_today=self.ensure_daily_for_today,
                refresh_document=self._refresh_document,
                archive_document=self._archive_document,
                recompute_priorities=self.recompute_priorities,
                repair_missing_embeddings=self._repair_missing_embeddings,
                repair_canonical_drift=self.repair_canonical_drift,
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
                    "chunk_id": result.chunk_id,
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
        include_frontmatter: bool = False,
        include_sources: bool = False,
        route_id: str | None = None,
        session_id: str | None = None,
        tool_name: str = "memory_get",
    ) -> str:
        await self.ensure_index_synced()
        row = self._index_db.document_for_id_or_path(document_id=document_id, path=path)
        if row is None:
            raise ValueError("Memory document not found.")
        document = self._store.read_document(Path(str(row["path"])))
        rendered: str
        accessed_chunk_id: str
        source_ref_ids: tuple[str, ...]
        if section_path:
            rendered, accessed_chunk_id, source_ref_ids = _render_document_section(
                document=document,
                section_path=section_path,
            )
        else:
            rendered = document.raw_markdown if include_frontmatter else document.body_markdown
            accessed_chunk_id = f"document:{document.document_id}"
            source_ref_ids = tuple(source_ref.source_ref_id for source_ref in document.source_refs)
        if include_sources:
            rendered = _append_sources(
                rendered=rendered,
                document=document,
                source_ref_ids=source_ref_ids,
            )
        self._index_db.record_accesses(
            occurred_at=_utc_now_iso(),
            route_id=route_id,
            session_id=session_id,
            tool_name=tool_name,
            query=section_path or document_id or path,
            mode="get_section" if section_path else "get_document",
            results=[
                {
                    "document_id": document.document_id,
                    "chunk_id": accessed_chunk_id,
                    "score": 1.0,
                }
            ],
        )
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
        tags: list[str] | None = None,
        aliases: list[str] | None = None,
        facts: list[dict[str, Any]] | None = None,
        relations: list[dict[str, Any]] | None = None,
        body_sections: dict[str, str] | None = None,
        source_refs: list[dict[str, Any]] | None = None,
        entity_refs: list[dict[str, Any]] | None = None,
        completion_criteria: list[str] | None = None,
        route_id: str | None = None,
        session_id: str | None = None,
        date: str | None = None,
        timezone_name: str | None = None,
        close_reason: str | None = None,
        allow_locked: bool = True,
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
            _assert_locked_write_allowed(current=current, allow_locked=allow_locked, operation=operation)
            if current.kind == "daily":
                archived = await self._archive_document(current, allow_locked=allow_locked)
            else:
                now = _utc_now_iso()
                archived_summary, archived_sections = _transition_rewrite_payload(
                    kind=current.kind,
                    operation="archive",
                    summary=summary,
                    body_sections=body_sections,
                    fallback_summary=_archived_summary_fallback(
                        title=title or current.title,
                        archive_reason=close_reason or current.close_reason,
                        archived_at=now,
                    ),
                    occurred_at=now,
                )
                updated = self._compose_document_update(
                    base_document=current,
                    target_kind=current.kind,
                    title=title,
                    summary=archived_summary,
                    priority=priority,
                    pinned=pinned,
                    locked=locked,
                    review_after=review_after,
                    expires_at=expires_at,
                    tags=tags,
                    aliases=aliases,
                    facts=facts,
                    relations=relations,
                    body_sections=archived_sections,
                    source_refs=source_refs,
                    entity_refs=entity_refs,
                    completion_criteria=completion_criteria,
                    route_id=route_id,
                    session_id=session_id,
                    date=date,
                    close_reason=close_reason,
                    updated_at=now,
                    path_override=current.path,
                )
                archived = await self._archive_document(updated, allow_locked=allow_locked)
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
            _assert_locked_write_allowed(current=current, allow_locked=allow_locked, operation=operation)
            now = _utc_now_iso()
            closed_summary, closed_sections = _transition_rewrite_payload(
                kind="ongoing",
                operation="close",
                summary=summary,
                body_sections=body_sections,
                fallback_summary=_closed_summary_fallback(
                    title=title or current.title,
                    close_reason=close_reason or current.close_reason,
                    closed_at=now,
                ),
                occurred_at=now,
            )
            updated = self._compose_document_update(
                base_document=current,
                target_kind="ongoing",
                title=title,
                summary=closed_summary,
                priority=priority,
                pinned=pinned,
                locked=locked,
                review_after=review_after,
                expires_at=expires_at,
                tags=tags,
                aliases=aliases,
                facts=facts,
                relations=relations,
                body_sections=closed_sections,
                source_refs=source_refs,
                entity_refs=entity_refs,
                completion_criteria=completion_criteria,
                route_id=route_id,
                session_id=session_id,
                date=date,
                close_reason=close_reason,
                updated_at=now,
                status="closed",
                path_override=current.path,
            )
            persisted = await self._archive_document(updated, allow_locked=allow_locked)
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
            _assert_locked_write_allowed(current=current, allow_locked=allow_locked, operation=operation)
            promoted = await self._migrate_document(
                current=current,
                target_kind=target_kind,
                operation=operation,
                allow_locked=allow_locked,
            )
            return MemoryWriteResult(
                operation=operation,  # type: ignore[arg-type]
                document_id=promoted.document_id,
                path=promoted.path,
                summary=f"{operation.title()}d memory '{promoted.title}' into {promoted.kind}.",
                changed_paths=(current.path, promoted.path),
            )

        if operation not in {"create", "upsert"}:
            raise ValueError(f"Unsupported memory write operation: {operation}")
        if current is not None:
            _assert_locked_write_allowed(current=current, allow_locked=allow_locked, operation=operation)
        if operation == "upsert" and target_kind == "daily" and not _has_nonempty_section_overrides(body_sections):
            raise ValueError(
                "daily upsert must rewrite one or more canonical sections through body_sections; "
                "summary alone does not rewrite prior daily content. "
                "Use append_daily to add a new daily entry."
            )

        now = _utc_now_iso()
        base_document = current or self._build_new_document(
            kind=target_kind,
            title=title,
            date=date,
            timezone_name=timezone_name,
            route_id=route_id,
            session_id=session_id,
        )
        updated_document = self._compose_document_update(
            base_document=base_document,
            target_kind=target_kind,
            title=title,
            summary=summary,
            priority=priority,
            pinned=pinned,
            locked=locked,
            review_after=review_after,
            expires_at=expires_at,
            tags=tags,
            aliases=aliases,
            facts=facts,
            relations=relations,
            body_sections=body_sections,
            source_refs=source_refs,
            entity_refs=entity_refs,
            completion_criteria=completion_criteria,
            route_id=route_id,
            session_id=session_id,
            date=date,
            close_reason=close_reason,
            updated_at=now,
        )
        persisted = await self._persist_document(
            updated_document,
            allow_locked=allow_locked,
            existing_document=current,
        )
        changed_paths = (
            (current.path, persisted.path)
            if current is not None and current.path != persisted.path
            else (persisted.path,)
        )
        return MemoryWriteResult(
            operation=operation,  # type: ignore[arg-type]
            document_id=persisted.document_id,
            path=persisted.path,
            summary=f"{operation.title()}d {persisted.kind} memory '{persisted.title}'.",
            changed_paths=changed_paths,
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
        reference_time = datetime.now(timezone.utc).replace(microsecond=0)
        freshness_bucket = reference_time.strftime("%Y-%m-%dT%H")
        core_bundle = checksum_bundle_for_documents(core_documents) + f"|freshness:{freshness_bucket}"
        ongoing_bundle = checksum_bundle_for_documents(ongoing_documents) + f"|freshness:{freshness_bucket}"

        core_cached = self._index_db.render_bootstrap_cache_get(core_cache_key)
        ongoing_cached = self._index_db.render_bootstrap_cache_get(ongoing_cache_key)

        if core_cached is not None and str(core_cached["checksum_bundle"]) == core_bundle:
            core_text = str(core_cached["content"])
        else:
            core_text = render_core_bootstrap(
                core_documents,
                token_budget=self._settings.core_bootstrap_max_tokens,
                reference_time=reference_time,
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
                reference_time=reference_time,
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
            await self._index_document(document)
            indexed_paths.append(document.path)
        self._refresh_truth_signals()
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

    async def repair_canonical_drift(self) -> dict[str, Any]:
        await self.ensure_index_synced()
        checked = 0
        repaired = 0
        failed = 0
        changed_paths: list[str] = []
        repaired_issue_counts: dict[str, int] = {}
        for document in self._store.read_all_documents():
            checked += 1
            repaired_document, repair_codes = self._repairable_document_variant(document)
            if not repair_codes:
                continue
            try:
                persisted = await self._persist_document(
                    repaired_document,
                    run_dirty_scan=False,
                    allow_locked=True,
                    existing_document=document,
                )
            except Exception:
                failed += 1
                LOGGER.exception("Failed repairing canonical memory drift for: %s", document.path)
                continue
            repaired += 1
            changed_paths.append(str(persisted.path))
            for code in repair_codes:
                repaired_issue_counts[code] = repaired_issue_counts.get(code, 0) + 1
        if repaired > 0:
            await self.ensure_index_synced()
        return {
            "checked_documents": checked,
            "repaired_documents": repaired,
            "failed_documents": failed,
            "changed_paths": changed_paths,
            "repaired_issue_counts": repaired_issue_counts,
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
        active_documents = tuple(
            document
            for document in self._store.read_all_documents()
            if document.kind in {"core", "ongoing"} and not document.archived and document.status == "active"
        )
        active_memory_context = tuple(
            {
                "document_id": document.document_id,
                "title": document.title,
                "summary": document.summary or _first_paragraph(document.sections.get("Summary", "")),
                "kind": document.kind,
            }
            for document in active_documents
        )
        plan = await planner.plan_turn(
            route_id=route_id,
            session_id=session_id,
            records=records,
            active_memories=active_memory_context,
        )
        normalized_actions = _normalize_reflection_actions(
            actions=plan.actions,
            active_documents=active_documents,
        )
        applied: list[MemoryWriteResult] = []
        for action in normalized_actions:
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
                target_document = _resolve_reflection_target(
                    action=action,
                    active_documents=active_documents,
                    target_kind="ongoing",
                )
                if target_document is not None and target_document.locked:
                    continue
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
                        tags=_coerce_list_of_strings(payload.get("tags")),
                        aliases=_coerce_list_of_strings(payload.get("aliases")),
                        entity_refs=_coerce_list_of_dicts(payload.get("entity_refs")),
                        completion_criteria=_coerce_list_of_strings(payload.get("completion_criteria")),
                        route_id=route_id,
                        session_id=session_id,
                        allow_locked=False,
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
                target_document = _resolve_reflection_target(
                    action=action,
                    active_documents=active_documents,
                    target_kind="core",
                )
                if target_document is not None and target_document.locked:
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
                        tags=_coerce_list_of_strings(payload.get("tags")),
                        aliases=_coerce_list_of_strings(payload.get("aliases")),
                        entity_refs=_coerce_list_of_dicts(payload.get("entity_refs")),
                        route_id=route_id,
                        session_id=session_id,
                        allow_locked=False,
                    )
                )
                continue
            if action.action == "close_ongoing":
                payload = action.payload
                target_document = _resolve_reflection_target(
                    action=action,
                    active_documents=active_documents,
                    target_kind="ongoing",
                )
                if target_document is not None and target_document.locked:
                    continue
                try:
                    applied.append(
                        await self.write(
                            operation="close",
                            target_kind="ongoing",
                            document_id=_optional_str(payload.get("document_id")),
                            title=_optional_str(payload.get("title")),
                            close_reason=_optional_str(payload.get("close_reason")),
                            allow_locked=False,
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
        return 0

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
            await self._index_document(document)
            processed_paths.append(dirty.path)
        self._refresh_truth_signals()
        self._index_db.clear_dirty_documents(tuple(processed_paths))

    async def _index_document(self, document: MemoryDocument) -> MemoryDocument:
        chunks = chunk_document(document)
        self._index_db.upsert_document(document, chunks)
        await self._index_db.upsert_embeddings_for_document(
            document=document,
            chunks=chunks,
            llm_service=self._llm_service,
        )
        return document

    async def _persist_document(
        self,
        document: MemoryDocument,
        *,
        run_dirty_scan: bool = True,
        allow_locked: bool = False,
        existing_document: MemoryDocument | None = None,
    ) -> MemoryDocument:
        if existing_document is not None and bool(existing_document.locked) and not allow_locked:
            raise PermissionError(f"Locked memory '{existing_document.title}' cannot be auto-mutated.")
        previous_path = existing_document.path if existing_document is not None else None
        self._assert_document_path_available(
            document=document,
            target_path=document.path,
            current_path=previous_path,
        )
        persisted = self._store.write_document(document, previous_path=previous_path)
        await self._index_document(persisted)
        self._refresh_truth_signals()
        if run_dirty_scan:
            await self.ensure_index_synced()
        return self._store.read_document(persisted.path)

    async def _refresh_document(self, document: MemoryDocument) -> MemoryDocument:
        return await self._persist_document(
            document,
            run_dirty_scan=False,
            allow_locked=False,
            existing_document=document,
        )

    async def _archive_document(
        self,
        document: MemoryDocument,
        *,
        allow_locked: bool = False,
    ) -> MemoryDocument:
        if bool(document.locked) and not allow_locked:
            raise PermissionError(f"Locked memory '{document.title}' cannot be auto-archived.")
        self._assert_document_path_available(
            document=document,
            target_path=self._store.archive_path_for(document),
            current_path=document.path,
        )
        archived = self._store.archive_document(document)
        self._index_db.remove_document(path=document.path, document_id=document.document_id)
        await self._index_document(archived)
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
        summary_sections = _default_sections_for_kind(kind)
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
        allow_locked: bool,
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
                review_after=(
                    current.review_after or _utc_now_iso()
                    if target_kind == "ongoing"
                    else None
                ),
                expires_at=current.expires_at if target_kind == "ongoing" else None,
                tags=current.tags,
                aliases=current.aliases,
                facts=current.facts,
                relations=current.relations,
                source_refs=current.source_refs,
                entity_refs=current.entity_refs,
                completion_criteria=current.completion_criteria if target_kind == "ongoing" else (),
                sections=_mapped_sections_for_kind(current, target_kind),
            )
        persisted = await self._persist_document(new_document, allow_locked=allow_locked)
        if operation in {"promote", "demote"}:
            await self._archive_document(
                replace(current, status="archived", updated_at=_utc_now_iso()),
                allow_locked=allow_locked,
            )
        return persisted

    def _refresh_truth_signals(self) -> None:
        self._index_db.refresh_truth_signals()

    async def _integrity_check_dicts(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for path in self._store.list_markdown_paths():
            try:
                document = self._store.read_document(path)
            except Exception as exc:
                issues.append(
                    {
                        "path": str(path),
                        "severity": "error",
                        "code": "invalid_markdown_schema",
                        "message": str(exc),
                    }
                )
                continue
            issues.extend(self._document_integrity_issues(document))
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

    def _assert_document_path_available(
        self,
        *,
        document: MemoryDocument,
        target_path: Path,
        current_path: Path | None,
    ) -> None:
        if current_path is not None and target_path == current_path:
            return
        row = self._index_db.document_for_id_or_path(path=str(target_path))
        if row is not None and str(row["document_id"]) != document.document_id:
            raise ValueError(
                f"Cannot write memory '{document.title}' because path '{target_path}' is already used by another memory."
            )
        if not target_path.exists():
            return
        try:
            existing_document = self._store.read_document(target_path)
        except Exception as exc:
            raise ValueError(
                f"Cannot write memory '{document.title}' because target path '{target_path}' already exists."
            ) from exc
        if existing_document.document_id != document.document_id:
            raise ValueError(
                f"Cannot write memory '{document.title}' because path '{target_path}' already belongs to '{existing_document.title}'."
            )

    def _document_integrity_issues(self, document: MemoryDocument) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        expected_path = self._store.canonical_path_for(
            kind=document.kind,
            title=document.title,
            date=document.date,
            archived=document.archived,
        )
        if document.path != expected_path:
            issues.append(
                {
                    "path": str(document.path),
                    "severity": "warning",
                    "code": "title_path_mismatch",
                    "message": (
                        f"Document title '{document.title}' expects canonical path '{expected_path.name}', "
                        f"but the file is stored as '{document.path.name}'."
                    ),
                }
            )
        if document.kind == "daily":
            return issues
        summary = (document.summary or "").strip()
        summary_section = document.sections.get("Summary", "").strip()
        if summary and not summary_section:
            issues.append(
                {
                    "path": str(document.path),
                    "severity": "warning",
                    "code": "summary_section_empty",
                    "message": "Frontmatter summary is populated, but the canonical 'Summary' section is empty.",
                }
            )
        elif summary_section and not summary:
            issues.append(
                {
                    "path": str(document.path),
                    "severity": "warning",
                    "code": "summary_frontmatter_missing",
                    "message": "Canonical 'Summary' section has content, but frontmatter summary is missing.",
                }
            )
        elif summary and summary_section and _first_paragraph(summary_section) != summary:
            issues.append(
                {
                    "path": str(document.path),
                    "severity": "warning",
                    "code": "summary_section_mismatch",
                    "message": "Frontmatter summary and canonical 'Summary' section disagree.",
                }
            )
        effective_summary = summary or summary_section
        if (
            document.kind == "ongoing"
            and document.status != "active"
            and effective_summary
            and _summary_looks_present_tense(effective_summary)
        ):
            issues.append(
                {
                    "path": str(document.path),
                    "severity": "warning",
                    "code": "closed_summary_present_tense",
                    "message": "Closed ongoing memory summary still reads like active present-state work.",
                }
            )
        return issues

    def _compose_document_update(
        self,
        *,
        base_document: MemoryDocument,
        target_kind: str,
        title: str | None,
        summary: str | None,
        priority: int | None,
        pinned: bool | None,
        locked: bool | None,
        review_after: str | None,
        expires_at: str | None,
        tags: list[str] | None,
        aliases: list[str] | None,
        facts: list[dict[str, Any]] | None,
        relations: list[dict[str, Any]] | None,
        body_sections: dict[str, str] | None,
        source_refs: list[dict[str, Any]] | None,
        entity_refs: list[dict[str, Any]] | None,
        completion_criteria: list[str] | None,
        route_id: str | None,
        session_id: str | None,
        date: str | None,
        close_reason: str | None,
        updated_at: str,
        status: str | None = None,
        path_override: Path | None = None,
        summary_fallback: str | None = None,
    ) -> MemoryDocument:
        resolved_source_refs = _normalize_source_refs(
            payloads=source_refs,
            existing=base_document.source_refs,
            route_id=route_id,
            session_id=session_id,
        )
        resolved_title = title or base_document.title
        resolved_path = path_override or self._store.canonical_path_for(
            kind=target_kind,
            title=resolved_title,
            date=date or base_document.date,
            archived=base_document.archived,
        )
        resolved_sections = _merge_sections(base_document=base_document, overrides=body_sections or {})
        explicit_summary_section = (body_sections or {}).get("Summary")
        resolved_summary = summary
        if resolved_summary is None and explicit_summary_section is not None and explicit_summary_section.strip():
            resolved_summary = _first_paragraph(explicit_summary_section)
        if resolved_summary is None:
            resolved_summary = base_document.summary
        if (
            target_kind == "ongoing"
            and status is not None
            and status != "active"
            and summary is None
            and (explicit_summary_section is None or not explicit_summary_section.strip())
            and resolved_summary is not None
            and summary_fallback is not None
            and _summary_looks_present_tense(resolved_summary)
        ):
            resolved_summary = summary_fallback.strip()
        if target_kind != "daily" and resolved_summary is None and summary_fallback is not None and summary_fallback.strip():
            resolved_summary = summary_fallback.strip()
        if target_kind != "daily":
            resolved_sections = _reconcile_summary_section(
                sections=resolved_sections,
                summary=resolved_summary,
                explicit_summary_section=explicit_summary_section,
            )
        if resolved_summary is None and "Summary" in resolved_sections and resolved_sections["Summary"].strip():
            resolved_summary = _first_paragraph(resolved_sections["Summary"])
        return replace(
            base_document,
            path=resolved_path,
            title=resolved_title,
            status=status or base_document.status,
            summary=resolved_summary if target_kind != "daily" else None,
            priority=priority if priority is not None else base_document.priority,
            pinned=pinned if pinned is not None else base_document.pinned,
            locked=locked if locked is not None else base_document.locked,
            review_after=review_after if review_after is not None else base_document.review_after,
            expires_at=expires_at if expires_at is not None else base_document.expires_at,
            tags=_normalize_string_list(tags, existing=base_document.tags),
            aliases=_normalize_string_list(aliases, existing=base_document.aliases),
            updated_at=updated_at,
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
            entity_refs=_normalize_entity_refs(
                payloads=entity_refs,
                existing=base_document.entity_refs,
            )
            if target_kind != "daily"
            else (),
            completion_criteria=_normalize_string_list(
                completion_criteria,
                existing=base_document.completion_criteria,
            )
            if target_kind == "ongoing"
            else (),
            route_ids=tuple(dict.fromkeys(base_document.route_ids + ((route_id,) if route_id else ()))),
            session_ids=tuple(dict.fromkeys(base_document.session_ids + ((session_id,) if session_id else ()))),
            close_reason=close_reason if close_reason is not None else base_document.close_reason,
        )

    def _repairable_document_variant(
        self,
        document: MemoryDocument,
    ) -> tuple[MemoryDocument, tuple[str, ...]]:
        repair_codes: list[str] = []
        summary = document.summary
        sections = OrderedDict(document.sections)
        if document.kind != "daily":
            summary_text = (summary or "").strip()
            summary_section = sections.get("Summary", "").strip()
            if (
                document.kind == "ongoing"
                and document.status != "active"
                and document.close_reason
                and _summary_looks_present_tense(summary_text or summary_section)
            ):
                closed_summary = _closed_summary_fallback(
                    title=document.title,
                    close_reason=document.close_reason,
                    closed_at=document.updated_at,
                )
                summary = closed_summary
                sections["Summary"] = closed_summary
                summary_text = closed_summary
                summary_section = closed_summary
                repair_codes.append("closed_summary_present_tense")
            if summary_text and not summary_section:
                sections["Summary"] = summary_text
                repair_codes.append("summary_section_empty")
            elif summary_section and not summary_text:
                summary = _first_paragraph(summary_section)
                repair_codes.append("summary_frontmatter_missing")
        path = document.path
        expected_path = self._store.canonical_path_for(
            kind=document.kind,
            title=document.title,
            date=document.date,
            archived=document.archived,
        )
        if path != expected_path:
            path = expected_path
            repair_codes.append("title_path_mismatch")
        if not repair_codes:
            return document, ()
        return (
            replace(
                document,
                path=path,
                summary=summary if document.kind != "daily" else document.summary,
                sections=sections,
                updated_at=_utc_now_iso(),
            ),
            tuple(dict.fromkeys(repair_codes)),
        )


def summary_placeholder(kind: str) -> str:
    return "" if kind == "ongoing" else ""


def _default_sections_for_kind(kind: str) -> OrderedDict[str, str]:
    if kind == "core":
        return OrderedDict(
            (
                ("Summary", ""),
                ("Details", ""),
                ("Notes", ""),
            )
        )
    if kind == "ongoing":
        return OrderedDict(
            (
                ("Summary", ""),
                ("Current State", ""),
                ("Open Loops", ""),
                ("Artifacts", ""),
                ("Notes", ""),
            )
        )
    if kind == "daily":
        return OrderedDict(
            (
                ("Notable Events", ""),
                ("Decisions", ""),
                ("Active Commitments", ""),
                ("Open Loops", ""),
                ("Artifacts", ""),
                ("Candidate Promotions", ""),
            )
        )
    raise ValueError(f"Unsupported memory kind: {kind}")


def _merge_sections(*, base_document: MemoryDocument, overrides: dict[str, str]) -> OrderedDict[str, str]:
    sections = OrderedDict(base_document.sections)
    for heading, content in overrides.items():
        if heading not in sections:
            continue
        sections[heading] = content.strip()
    return sections


def _has_nonempty_section_overrides(body_sections: dict[str, str] | None) -> bool:
    if body_sections is None:
        return False
    return any(
        isinstance(section_name, str)
        and section_name.strip()
        and isinstance(section_value, str)
        and section_value.strip()
        for section_name, section_value in body_sections.items()
    )


def _reconcile_summary_section(
    *,
    sections: OrderedDict[str, str],
    summary: str | None,
    explicit_summary_section: str | None,
) -> OrderedDict[str, str]:
    if summary is None or not summary.strip():
        return sections
    if "Summary" not in sections:
        return sections
    if explicit_summary_section is not None and explicit_summary_section.strip():
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


def _closed_summary_fallback(
    *,
    title: str,
    close_reason: str | None,
    closed_at: str,
) -> str:
    if close_reason is not None and close_reason.strip():
        return close_reason.strip()
    closed_date = closed_at[:10] if len(closed_at) >= 10 else closed_at
    return f"{title} was closed on {closed_date}."


def _archived_summary_fallback(
    *,
    title: str,
    archive_reason: str | None,
    archived_at: str,
) -> str:
    if archive_reason is not None and archive_reason.strip():
        return archive_reason.strip()
    archived_date = archived_at[:10] if len(archived_at) >= 10 else archived_at
    return f"{title} was archived on {archived_date}."


def _transition_rewrite_payload(
    *,
    kind: str,
    operation: str,
    summary: str | None,
    body_sections: dict[str, str] | None,
    fallback_summary: str,
    occurred_at: str,
) -> tuple[str, dict[str, str]]:
    if kind == "daily":
        return fallback_summary, {}

    resolved_summary = (summary or "").strip()
    provided_sections = {
        str(heading): str(content).strip()
        for heading, content in (body_sections or {}).items()
    }
    has_agent_rewrite = bool(resolved_summary) or any(content for content in provided_sections.values())
    sections = _default_sections_for_kind(kind)

    if has_agent_rewrite:
        for heading in sections:
            if heading in provided_sections:
                sections[heading] = provided_sections[heading]
        if not resolved_summary:
            summary_section = sections.get("Summary", "").strip()
            resolved_summary = _first_paragraph(summary_section) if summary_section else fallback_summary
        if "Summary" in sections and not sections["Summary"].strip():
            sections["Summary"] = resolved_summary
        return resolved_summary, dict(sections)

    system_stamp = _transition_system_stamp(operation=operation, occurred_at=occurred_at)
    sections["Summary"] = fallback_summary
    if kind == "ongoing":
        sections["Current State"] = fallback_summary
    elif kind == "core":
        sections["Details"] = fallback_summary
    if "Notes" in sections:
        sections["Notes"] = system_stamp
    return fallback_summary, dict(sections)


def _transition_system_stamp(*, operation: str, occurred_at: str) -> str:
    event_date = occurred_at[:10] if len(occurred_at) >= 10 else occurred_at
    verb = "terminal rewrite" if operation == "close" else "archive rewrite"
    return (
        f"System {verb} fallback applied on {event_date} because no rewritten superseding "
        "content was provided."
    )


def _summary_looks_present_tense(value: str) -> bool:
    normalized = f" {value.strip().lower()} "
    return any(
        cue in normalized
        for cue in (
            " currently ",
            " is currently ",
            " actively ",
            " is actively ",
            " ongoing ",
            " in progress ",
            " working on ",
            " being implemented ",
            " underway ",
        )
    )


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


def _normalize_string_list(
    value: list[str] | None,
    *,
    existing: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None:
        return existing
    normalized = [str(item).strip() for item in value if str(item).strip()]
    return tuple(dict.fromkeys(normalized))


def _normalize_entity_refs(
    *,
    payloads: list[dict[str, Any]] | None,
    existing: tuple[EntityReference, ...],
) -> tuple[EntityReference, ...]:
    if payloads is None:
        return existing
    entity_refs: list[EntityReference] = []
    for payload in payloads:
        name = _optional_str(payload.get("name"))
        if name is None:
            continue
        aliases = _coerce_list_of_strings(payload.get("aliases")) or []
        entity_refs.append(
            EntityReference(
                entity_id=_optional_str(payload.get("entity_id")) or _entity_id_for_name(name),
                name=name,
                entity_type=_optional_str(payload.get("entity_type")) or "unknown",
                aliases=tuple(dict.fromkeys(aliases)),
            )
        )
    return tuple(entity_refs)


def _render_document_section(
    *,
    document: MemoryDocument,
    section_path: str,
) -> tuple[str, str, tuple[str, ...]]:
    if section_path == "facts":
        if not document.facts:
            raise ValueError("Section 'facts' not found in memory document.")
        lines = [f"# {document.title}", "", "## Facts"]
        source_ref_ids: list[str] = []
        for fact in document.facts:
            lines.append(_render_fact_line(fact))
            source_ref_ids.extend(fact.source_ref_ids)
        return (
            "\n".join(lines).rstrip() + "\n",
            f"facts:{document.document_id}",
            tuple(dict.fromkeys(source_ref_ids)),
        )
    if section_path == "relations":
        if not document.relations:
            raise ValueError("Section 'relations' not found in memory document.")
        lines = [f"# {document.title}", "", "## Relations"]
        source_ref_ids: list[str] = []
        for relation in document.relations:
            lines.append(_render_relation_line(relation))
            source_ref_ids.extend(relation.source_ref_ids)
        return (
            "\n".join(lines).rstrip() + "\n",
            f"relations:{document.document_id}",
            tuple(dict.fromkeys(source_ref_ids)),
        )
    if section_path in document.sections:
        matching_chunks = [
            chunk
            for chunk in chunk_document(document)
            if chunk.section_path == section_path or chunk.section_path.startswith(f"{section_path}/")
        ]
        access_chunk_id = (
            matching_chunks[0].chunk_id
            if len(matching_chunks) == 1 and matching_chunks[0].section_path == section_path
            else f"section:{document.document_id}:{section_path}"
        )
        return (
            f"# {document.title}\n\n## {section_path}\n{document.sections[section_path].rstrip()}\n",
            access_chunk_id,
            tuple(source_ref.source_ref_id for source_ref in document.source_refs),
        )
    for chunk in chunk_document(document):
        if chunk.section_path != section_path:
            continue
        heading = section_path.split("/", 1)[0]
        return (
            f"# {document.title}\n\n## {section_path}\n{_chunk_body(chunk.text, document.title, heading)}\n",
            chunk.chunk_id,
            tuple(source_ref.source_ref_id for source_ref in document.source_refs),
        )
    raise ValueError(f"Section '{section_path}' not found in memory document.")


def _render_fact_line(fact: Fact) -> str:
    metadata: list[str] = []
    if fact.status != "current":
        metadata.append(f"status={fact.status}")
    if fact.confidence != "medium":
        metadata.append(f"confidence={fact.confidence}")
    suffix = f" ({', '.join(metadata)})" if metadata else ""
    return f"- {fact.text}{suffix}"


def _render_relation_line(relation: Relation) -> str:
    metadata: list[str] = []
    if relation.status != "current":
        metadata.append(f"status={relation.status}")
    if relation.confidence != "medium":
        metadata.append(f"confidence={relation.confidence}")
    suffix = f" ({', '.join(metadata)})" if metadata else ""
    return f"- {relation.textualization}{suffix}"


def _chunk_body(chunk_text: str, title: str, heading: str) -> str:
    lines = chunk_text.splitlines()
    if len(lines) >= 3 and lines[0].strip() == title and lines[1].strip() == heading:
        return "\n".join(lines[2:]).strip()
    return chunk_text.strip()


def _append_sources(
    *,
    rendered: str,
    document: MemoryDocument,
    source_ref_ids: tuple[str, ...],
) -> str:
    if not document.source_refs:
        return rendered
    selected = (
        [source_ref for source_ref in document.source_refs if source_ref.source_ref_id in set(source_ref_ids)]
        if source_ref_ids
        else list(document.source_refs)
    )
    if not selected:
        selected = list(document.source_refs)
    return rendered.rstrip() + "\n\n## Sources\n" + "\n".join(
        f"- {source_ref.source_type}:{source_ref.source_ref_id}"
        for source_ref in selected
    ) + "\n"


def _normalize_reflection_actions(
    *,
    actions: tuple[ReflectionAction, ...],
    active_documents: tuple[MemoryDocument, ...],
) -> tuple[ReflectionAction, ...]:
    normalized: list[ReflectionAction] = []
    for action in actions:
        if action.action not in {"create_ongoing", "create_core"}:
            normalized.append(action)
            continue
        target_kind = "ongoing" if action.action == "create_ongoing" else "core"
        target_document = _resolve_reflection_target(
            action=action,
            active_documents=active_documents,
            target_kind=target_kind,
        )
        if target_document is None:
            normalized.append(action)
            continue
        payload = dict(action.payload)
        payload["document_id"] = target_document.document_id
        normalized.append(
            ReflectionAction(
                action="update_ongoing" if target_kind == "ongoing" else "update_core",
                confidence=action.confidence,
                payload=payload,
                rationale=action.rationale,
            )
        )
    return tuple(normalized)


def _resolve_reflection_target(
    *,
    action: ReflectionAction,
    active_documents: tuple[MemoryDocument, ...],
    target_kind: str,
) -> MemoryDocument | None:
    payload = action.payload
    document_id = _optional_str(payload.get("document_id"))
    if document_id is not None:
        for document in active_documents:
            if document.kind == target_kind and document.document_id == document_id:
                return document
    title = _optional_str(payload.get("title"))
    if title is not None:
        normalized_title = _normalize_lookup(title)
        for document in active_documents:
            if document.kind == target_kind and _normalize_lookup(document.title) == normalized_title:
                return document
    return None


def _normalize_lookup(value: str) -> str:
    return " ".join(value.lower().strip().split())


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


def _coerce_list_of_strings(value: Any) -> list[str] | None:
    if value is None or not isinstance(value, list):
        return None
    return [str(item).strip() for item in value if str(item).strip()]


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


def _entity_id_for_name(name: str) -> str:
    return "entity_" + "".join(character.lower() if character.isalnum() else "_" for character in name).strip("_")


def _assert_locked_write_allowed(
    *,
    current: MemoryDocument,
    allow_locked: bool,
    operation: str,
) -> None:
    if current.locked and not allow_locked:
        raise PermissionError(
            f"Locked memory '{current.title}' cannot be auto-mutated during {operation}."
        )


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
