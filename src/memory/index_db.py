"""SQLite sidecar for memory search, graph lookup, and maintenance bookkeeping."""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
import sqlite3
from typing import Any

from llm import EmbeddingRequest, LLMService

from .config import MemorySettings
from .types import DirtyDocument, DocumentChunk, MemoryDocument

LOGGER = logging.getLogger(__name__)
_SQLITE_VEC_OVERRIDE_PATH = Path("/opt/sqlite-vec/vec0")

_DEFAULT_STATE = {
    "semantic_enabled": False,
    "semantic_error": None,
    "embedding_dimensions": None,
    "embedding_provider": None,
    "embedding_model": None,
}


class MemoryIndexDB:
    """Manages the derived SQLite databases for the memory subsystem."""

    def __init__(self, settings: MemorySettings) -> None:
        self._settings = settings
        self._settings.index_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    @property
    def semantic_enabled(self) -> bool:
        state = self.load_state()
        return bool(state.get("semantic_enabled", False))

    @property
    def semantic_error(self) -> str | None:
        state = self.load_state()
        value = state.get("semantic_error")
        return str(value) if value is not None else None

    def semantic_search_status(self) -> tuple[bool, str | None]:
        state = self.load_state()
        if not state.get("semantic_enabled"):
            error = state.get("semantic_error")
            if error is not None:
                return False, str(error)
            return False, "sqlite-vec is not available."
        if state.get("embedding_dimensions") is None:
            return False, "the embedding index is not initialized yet"
        with self._connect_embeddings() as conn:
            if not _table_exists(conn, "embedding_items_vec"):
                return False, "the embedding vector table is missing"
        return True, None

    def ensure_schema(self) -> None:
        with self._connect_main() as conn:
            conn.executescript(
                """
                pragma journal_mode = wal;
                pragma foreign_keys = on;

                create table if not exists resources (
                    resource_id text primary key,
                    source_type text not null,
                    route_id text,
                    session_id text,
                    record_id text,
                    tool_name text,
                    artifact_path text,
                    note text,
                    captured_at text not null,
                    checksum text,
                    resource_status text default 'active'
                );

                create table if not exists documents (
                    document_id text primary key,
                    path text not null unique,
                    kind text not null,
                    title text not null,
                    status text not null,
                    priority integer,
                    pinned integer,
                    locked integer,
                    confidence text,
                    created_at text not null,
                    updated_at text not null,
                    review_after text,
                    expires_at text,
                    checksum text not null,
                    summary text,
                    body_markdown text not null,
                    archived_at text,
                    support_count integer default 0,
                    contradiction_count integer default 0,
                    last_confirmed_at text,
                    last_contradicted_at text,
                    last_accessed_at text,
                    access_count_30d integer default 0
                );

                create table if not exists document_chunks (
                    chunk_id text primary key,
                    document_id text not null,
                    ordinal integer not null,
                    section_path text not null,
                    text text not null,
                    token_estimate integer not null,
                    created_at text not null,
                    updated_at text not null,
                    foreign key(document_id) references documents(document_id) on delete cascade
                );

                create virtual table if not exists document_chunks_fts using fts5(
                    chunk_id unindexed,
                    document_id unindexed,
                    section_path,
                    text
                );

                create table if not exists facts (
                    fact_id text primary key,
                    document_id text not null,
                    text text not null,
                    status text not null,
                    confidence text not null,
                    first_seen_at text not null,
                    last_seen_at text not null,
                    valid_from text,
                    valid_to text,
                    support_count integer default 0,
                    contradiction_count integer default 0,
                    last_confirmed_at text,
                    last_contradicted_at text,
                    last_accessed_at text,
                    access_count_30d integer default 0,
                    foreign key(document_id) references documents(document_id) on delete cascade
                );

                create table if not exists relations (
                    relation_id text primary key,
                    document_id text not null,
                    subject text not null,
                    predicate text not null,
                    object text not null,
                    status text not null,
                    confidence text not null,
                    cardinality text not null,
                    first_seen_at text not null,
                    last_seen_at text not null,
                    valid_from text,
                    valid_to text,
                    support_count integer default 0,
                    contradiction_count integer default 0,
                    last_confirmed_at text,
                    last_contradicted_at text,
                    last_accessed_at text,
                    access_count_30d integer default 0,
                    foreign key(document_id) references documents(document_id) on delete cascade
                );

                create table if not exists entities (
                    entity_id text primary key,
                    canonical_name text not null,
                    entity_type text not null,
                    aliases_json text not null,
                    first_seen_at text not null,
                    last_seen_at text not null,
                    last_accessed_at text,
                    access_count_30d integer default 0
                );

                create table if not exists source_refs (
                    source_ref_id text primary key,
                    source_type text not null,
                    route_id text,
                    session_id text,
                    record_id text,
                    tool_name text,
                    note text,
                    captured_at text not null
                );

                create table if not exists document_source_refs (
                    document_id text not null,
                    source_ref_id text not null,
                    primary key(document_id, source_ref_id),
                    foreign key(document_id) references documents(document_id) on delete cascade,
                    foreign key(source_ref_id) references source_refs(source_ref_id) on delete cascade
                );

                create table if not exists fact_source_refs (
                    fact_id text not null,
                    source_ref_id text not null,
                    primary key(fact_id, source_ref_id),
                    foreign key(fact_id) references facts(fact_id) on delete cascade,
                    foreign key(source_ref_id) references source_refs(source_ref_id) on delete cascade
                );

                create table if not exists relation_source_refs (
                    relation_id text not null,
                    source_ref_id text not null,
                    primary key(relation_id, source_ref_id),
                    foreign key(relation_id) references relations(relation_id) on delete cascade,
                    foreign key(source_ref_id) references source_refs(source_ref_id) on delete cascade
                );

                create table if not exists access_log (
                    access_id integer primary key autoincrement,
                    occurred_at text not null,
                    route_id text,
                    session_id text,
                    tool_name text,
                    query text,
                    mode text,
                    document_id text,
                    chunk_id text,
                    result_rank integer,
                    result_score real
                );

                create table if not exists maintenance_runs (
                    run_id integer primary key autoincrement,
                    job_name text not null,
                    started_at text not null,
                    finished_at text,
                    status text not null,
                    summary_json text not null
                );

                create table if not exists dirty_documents (
                    path text primary key,
                    detected_at text not null,
                    reason text not null
                );

                create table if not exists bootstrap_cache (
                    cache_key text primary key,
                    generated_at text not null,
                    content text not null,
                    token_estimate integer not null,
                    checksum_bundle text not null
                );

                create table if not exists schema_info (
                    key text primary key,
                    value text not null
                );
                """
            )
            conn.execute(
                """
                insert into schema_info(key, value)
                values ('schema_version', '1')
                on conflict(key) do update set value=excluded.value
                """
            )

        self._ensure_embeddings_schema()

    def indexed_checksums(self) -> dict[str, str]:
        with self._connect_main() as conn:
            rows = conn.execute("select path, checksum from documents").fetchall()
        return {str(row["path"]): str(row["checksum"]) for row in rows}

    def paths_missing_searchable_chunks(self) -> tuple[Path, ...]:
        with self._connect_main() as conn:
            rows = conn.execute(
                """
                select d.path
                from documents d
                left join document_chunks c on c.document_id = d.document_id
                where d.kind in ('core', 'ongoing')
                  and coalesce(trim(d.summary), '') != ''
                group by d.document_id, d.path
                having count(c.chunk_id) = 0
                """
            ).fetchall()
        return tuple(Path(str(row["path"])) for row in rows)

    def embedding_vector_table_exists(self) -> bool:
        with self._connect_embeddings() as conn:
            return _table_exists(conn, "embedding_items_vec")

    def embedding_vector_count_for_document(self, document_id: str) -> int:
        if not self.embedding_vector_table_exists():
            return 0
        with self._connect_embeddings(load_vec=True) as conn:
            row = conn.execute(
                """
                select count(*) as item_count
                from embedding_items item
                join embedding_items_vec vec on vec.rowid = item.embedding_id
                where item.document_id = ?
                """,
                (document_id,),
            ).fetchone()
        return int(row["item_count"]) if row is not None else 0

    def expected_embedding_item_count(
        self,
        *,
        document: MemoryDocument,
        chunks: tuple[DocumentChunk, ...],
    ) -> int:
        return len(_embedding_items_for_document(document=document, chunks=chunks))

    def mark_dirty_documents(self, dirty_documents: tuple[DirtyDocument, ...]) -> None:
        if not dirty_documents:
            return
        with self._connect_main() as conn:
            conn.executemany(
                """
                insert into dirty_documents(path, detected_at, reason)
                values (?, ?, ?)
                on conflict(path) do update set
                    detected_at=excluded.detected_at,
                    reason=excluded.reason
                """,
                [(str(item.path), item.detected_at, item.reason) for item in dirty_documents],
            )

    def list_dirty_documents(self) -> tuple[DirtyDocument, ...]:
        with self._connect_main() as conn:
            rows = conn.execute(
                "select path, detected_at, reason from dirty_documents order by detected_at asc, path asc"
            ).fetchall()
        return tuple(
            DirtyDocument(path=Path(str(row["path"])), detected_at=str(row["detected_at"]), reason=str(row["reason"]))
            for row in rows
        )

    def clear_dirty_documents(self, paths: tuple[Path, ...]) -> None:
        if not paths:
            return
        with self._connect_main() as conn:
            conn.executemany(
                "delete from dirty_documents where path = ?",
                [(str(path),) for path in paths],
            )

    def remove_document(self, *, path: Path | None = None, document_id: str | None = None) -> None:
        if path is None and document_id is None:
            return
        with self._connect_main() as conn:
            resolved_document_id = document_id
            if resolved_document_id is None and path is not None:
                row = conn.execute(
                    "select document_id from documents where path = ?",
                    (str(path),),
                ).fetchone()
                resolved_document_id = str(row["document_id"]) if row is not None else None
            if resolved_document_id is None:
                return
            conn.execute("delete from documents where document_id = ?", (resolved_document_id,))
        self.delete_embeddings_for_document(resolved_document_id)

    def upsert_document(self, document: MemoryDocument, chunks: tuple[DocumentChunk, ...]) -> None:
        archived_at = document.updated_at if document.archived else None
        with self._connect_main() as conn:
            conn.execute(
                """
                insert into documents(
                    document_id, path, kind, title, status, priority, pinned, locked,
                    confidence, created_at, updated_at, review_after, expires_at, checksum,
                    summary, body_markdown, archived_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(document_id) do update set
                    path=excluded.path,
                    kind=excluded.kind,
                    title=excluded.title,
                    status=excluded.status,
                    priority=excluded.priority,
                    pinned=excluded.pinned,
                    locked=excluded.locked,
                    confidence=excluded.confidence,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    review_after=excluded.review_after,
                    expires_at=excluded.expires_at,
                    checksum=excluded.checksum,
                    summary=excluded.summary,
                    body_markdown=excluded.body_markdown,
                    archived_at=excluded.archived_at
                """,
                (
                    document.document_id,
                    str(document.path),
                    document.kind,
                    document.title,
                    document.status,
                    document.priority,
                    _bool_to_int(document.pinned),
                    _bool_to_int(document.locked),
                    document.confidence,
                    document.created_at,
                    document.updated_at,
                    document.review_after,
                    document.expires_at,
                    document.checksum,
                    document.summary,
                    document.body_markdown,
                    archived_at,
                ),
            )
            conn.execute("delete from document_chunks where document_id = ?", (document.document_id,))
            conn.execute("delete from document_chunks_fts where document_id = ?", (document.document_id,))
            conn.executemany(
                """
                insert into document_chunks(
                    chunk_id, document_id, ordinal, section_path, text, token_estimate, created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.section_path,
                        chunk.text,
                        chunk.token_estimate,
                        chunk.created_at,
                        chunk.updated_at,
                    )
                    for chunk in chunks
                ],
            )
            conn.executemany(
                """
                insert into document_chunks_fts(chunk_id, document_id, section_path, text)
                values (?, ?, ?, ?)
                """,
                [(chunk.chunk_id, chunk.document_id, chunk.section_path, chunk.text) for chunk in chunks],
            )

            conn.execute("delete from facts where document_id = ?", (document.document_id,))
            conn.execute("delete from relations where document_id = ?", (document.document_id,))
            conn.execute("delete from document_source_refs where document_id = ?", (document.document_id,))

            for source_ref in document.source_refs:
                conn.execute(
                    """
                    insert into source_refs(
                        source_ref_id, source_type, route_id, session_id, record_id, tool_name, note, captured_at
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?)
                    on conflict(source_ref_id) do update set
                        source_type=excluded.source_type,
                        route_id=excluded.route_id,
                        session_id=excluded.session_id,
                        record_id=excluded.record_id,
                        tool_name=excluded.tool_name,
                        note=excluded.note,
                        captured_at=excluded.captured_at
                    """,
                    (
                        source_ref.source_ref_id,
                        source_ref.source_type,
                        source_ref.route_id,
                        source_ref.session_id,
                        source_ref.record_id,
                        source_ref.tool_name,
                        source_ref.note,
                        source_ref.captured_at,
                    ),
                )
                conn.execute(
                    """
                    insert or ignore into document_source_refs(document_id, source_ref_id)
                    values (?, ?)
                    """,
                    (document.document_id, source_ref.source_ref_id),
                )
                conn.execute(
                    """
                    insert into resources(
                        resource_id, source_type, route_id, session_id, record_id, tool_name, artifact_path,
                        note, captured_at, checksum, resource_status
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    on conflict(resource_id) do update set
                        source_type=excluded.source_type,
                        route_id=excluded.route_id,
                        session_id=excluded.session_id,
                        record_id=excluded.record_id,
                        tool_name=excluded.tool_name,
                        artifact_path=excluded.artifact_path,
                        note=excluded.note,
                        captured_at=excluded.captured_at,
                        checksum=excluded.checksum,
                        resource_status=excluded.resource_status
                    """,
                    (
                        source_ref.source_ref_id,
                        source_ref.source_type,
                        source_ref.route_id,
                        source_ref.session_id,
                        source_ref.record_id,
                        source_ref.tool_name,
                        None,
                        source_ref.note,
                        source_ref.captured_at,
                        None,
                        "active",
                    ),
                )

            for fact in document.facts:
                conn.execute(
                    """
                    insert into facts(
                        fact_id, document_id, text, status, confidence, first_seen_at, last_seen_at,
                        valid_from, valid_to
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fact.fact_id,
                        document.document_id,
                        fact.text,
                        fact.status,
                        fact.confidence,
                        fact.first_seen_at,
                        fact.last_seen_at,
                        fact.valid_from,
                        fact.valid_to,
                    ),
                )
                for source_ref_id in fact.source_ref_ids:
                    conn.execute(
                        "insert or ignore into fact_source_refs(fact_id, source_ref_id) values (?, ?)",
                        (fact.fact_id, source_ref_id),
                    )

            for relation in document.relations:
                conn.execute(
                    """
                    insert into relations(
                        relation_id, document_id, subject, predicate, object, status, confidence,
                        cardinality, first_seen_at, last_seen_at, valid_from, valid_to
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relation.relation_id,
                        document.document_id,
                        relation.subject,
                        relation.predicate,
                        relation.object,
                        relation.status,
                        relation.confidence,
                        relation.cardinality,
                        relation.first_seen_at,
                        relation.last_seen_at,
                        relation.valid_from,
                        relation.valid_to,
                    ),
                )
                for source_ref_id in relation.source_ref_ids:
                    conn.execute(
                        """
                        insert or ignore into relation_source_refs(relation_id, source_ref_id)
                        values (?, ?)
                        """,
                        (relation.relation_id, source_ref_id),
                    )

            seen_entities: set[str] = set()
            for entity_ref in document.entity_refs:
                seen_entities.add(entity_ref.entity_id)
                conn.execute(
                    """
                    insert into entities(
                        entity_id, canonical_name, entity_type, aliases_json, first_seen_at, last_seen_at
                    )
                    values (?, ?, ?, ?, ?, ?)
                    on conflict(entity_id) do update set
                        canonical_name=excluded.canonical_name,
                        entity_type=excluded.entity_type,
                        aliases_json=excluded.aliases_json,
                        last_seen_at=excluded.last_seen_at
                    """,
                    (
                        entity_ref.entity_id,
                        entity_ref.name,
                        entity_ref.entity_type,
                        json.dumps(list(entity_ref.aliases), ensure_ascii=True),
                        document.created_at,
                        document.updated_at,
                    ),
                )

            for relation in document.relations:
                for entity_name in (relation.subject, relation.object):
                    entity_id = _entity_id_for_name(entity_name)
                    if entity_id in seen_entities:
                        continue
                    seen_entities.add(entity_id)
                    conn.execute(
                        """
                        insert into entities(
                            entity_id, canonical_name, entity_type, aliases_json, first_seen_at, last_seen_at
                        )
                        values (?, ?, ?, ?, ?, ?)
                        on conflict(entity_id) do update set
                            canonical_name=excluded.canonical_name,
                            last_seen_at=excluded.last_seen_at
                        """,
                        (
                            entity_id,
                            entity_name,
                            "unknown",
                            "[]",
                            document.created_at,
                            document.updated_at,
                        ),
                    )

    async def upsert_embeddings_for_document(
        self,
        *,
        document: MemoryDocument,
        chunks: tuple[DocumentChunk, ...],
        llm_service: LLMService | None,
    ) -> None:
        if llm_service is None:
            return
        items = _embedding_items_for_document(document=document, chunks=chunks)
        if not items:
            return

        if not self.semantic_enabled:
            return

        response = await llm_service.embed(
            EmbeddingRequest(inputs=[str(item["text"]) for item in items])
        )
        if not response.embeddings:
            return

        dimensions = len(response.embeddings[0])
        self._ensure_vector_table(dimensions)
        state = self.load_state()
        state.update(
            {
                "semantic_enabled": True,
                "semantic_error": None,
                "embedding_dimensions": dimensions,
                "embedding_provider": response.provider,
                "embedding_model": response.model,
            }
        )
        self.write_state(state)

        self.delete_embeddings_for_document(document.document_id)
        with self._connect_embeddings(load_vec=True) as conn:
            for item, embedding in zip(items, response.embeddings, strict=True):
                cursor = conn.execute(
                    """
                    insert into embedding_items(
                        item_key, item_type, target_id, document_id, path, section_path,
                        source_ref_ids_json, text, text_checksum, embedding_blob,
                        embedding_dimensions, created_at, updated_at
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["item_key"],
                        item["item_type"],
                        item["target_id"],
                        item["document_id"],
                        item["path"],
                        item["section_path"],
                        json.dumps(item["source_ref_ids"], ensure_ascii=True),
                        item["text"],
                        _checksum_text(item["text"]),
                        json.dumps(list(embedding), ensure_ascii=True),
                        dimensions,
                        document.created_at,
                        document.updated_at,
                    ),
                )
                embedding_id = int(cursor.lastrowid)
                conn.execute(
                    "insert into embedding_items_vec(rowid, embedding) values (?, ?)",
                    (embedding_id, json.dumps(list(embedding), ensure_ascii=True)),
                )

    def delete_embeddings_for_document(self, document_id: str) -> None:
        with self._connect_embeddings(load_vec=self.embedding_vector_table_exists()) as conn:
            row_ids = [
                int(row["embedding_id"])
                for row in conn.execute(
                    "select embedding_id from embedding_items where document_id = ?",
                    (document_id,),
                ).fetchall()
            ]
            for row_id in row_ids:
                try:
                    conn.execute("delete from embedding_items_vec where rowid = ?", (row_id,))
                except sqlite3.OperationalError:
                    break
            conn.execute("delete from embedding_items where document_id = ?", (document_id,))

    def lexical_candidates(
        self,
        *,
        query: str,
        scopes: tuple[str, ...],
        include_expired: bool,
        daily_lookback_days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        clause, params = _document_filter_clause(
            scopes=scopes,
            include_expired=include_expired,
            daily_lookback_days=daily_lookback_days,
        )
        with self._connect_main() as conn:
            rows = conn.execute(
                f"""
                select
                    fts.chunk_id,
                    fts.document_id,
                    fts.section_path,
                    snippet(document_chunks_fts, 3, '[', ']', '...', 24) as snippet,
                    bm25(document_chunks_fts) as bm25_score,
                    d.path,
                    d.kind,
                    d.title,
                    d.updated_at,
                    d.status,
                    d.pinned,
                    d.priority,
                    d.summary,
                    d.review_after,
                    d.expires_at,
                    d.archived_at,
                    coalesce((
                        select json_group_array(source_ref_id)
                        from document_source_refs ds
                        where ds.document_id = d.document_id
                    ), '[]') as source_ref_ids_json
                from document_chunks_fts fts
                join documents d on d.document_id = fts.document_id
                where document_chunks_fts match ? and {clause}
                order by bm25(document_chunks_fts) asc
                limit ?
                """,
                (query, *params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def semantic_candidates(
        self,
        *,
        embedding: list[float],
        scopes: tuple[str, ...],
        include_expired: bool,
        daily_lookback_days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not self.semantic_enabled:
            return []
        ready, _reason = self.semantic_search_status()
        if not ready:
            return []
        clause, params = _document_filter_clause(
            scopes=scopes,
            include_expired=include_expired,
            daily_lookback_days=daily_lookback_days,
        )
        query_vector = json.dumps(list(embedding), ensure_ascii=True)
        with self._connect_embeddings(load_vec=True) as embeddings_conn:
            rows = embeddings_conn.execute(
                """
                select
                    vec.rowid as embedding_id,
                    distance,
                    item.*
                from embedding_items_vec vec
                join embedding_items item on item.embedding_id = vec.rowid
                where vec.embedding match ? and k = ?
                order by distance asc
                """,
                (query_vector, limit),
            ).fetchall()
        if not rows:
            return []
        document_ids = tuple({str(row["document_id"]) for row in rows})
        placeholders = ", ".join("?" for _ in document_ids)
        with self._connect_main() as conn:
            document_rows = conn.execute(
                f"""
                select
                    d.document_id, d.path, d.kind, d.title, d.updated_at, d.status, d.pinned, d.priority,
                    d.summary, d.review_after, d.expires_at, d.archived_at
                from documents d
                where d.document_id in ({placeholders}) and {clause}
                """,
                (*document_ids, *params),
            ).fetchall()
        allowed = {str(row["document_id"]): dict(row) for row in document_rows}
        candidates: list[dict[str, Any]] = []
        for row in rows:
            document = allowed.get(str(row["document_id"]))
            if document is None:
                continue
            source_ref_ids = json.loads(str(row["source_ref_ids_json"]))
            candidates.append(
                {
                    "chunk_id": (
                        str(row["target_id"])
                        if str(row["item_type"]) == "chunk"
                        else str(row["item_key"])
                    ),
                    "document_id": str(row["document_id"]),
                    "section_path": str(row["section_path"]),
                    "snippet": str(row["text"]),
                    "distance": float(row["distance"]),
                    "path": document["path"],
                    "kind": document["kind"],
                    "title": document["title"],
                    "updated_at": document["updated_at"],
                    "status": document["status"],
                    "pinned": document["pinned"],
                    "priority": document["priority"],
                    "summary": document["summary"],
                    "review_after": document["review_after"],
                    "expires_at": document["expires_at"],
                    "archived_at": document["archived_at"],
                    "source_ref_ids_json": json.dumps(source_ref_ids, ensure_ascii=True),
                }
            )
        return candidates

    def graph_entities(self) -> list[dict[str, Any]]:
        with self._connect_main() as conn:
            rows = conn.execute(
                """
                select entity_id, canonical_name, entity_type, aliases_json, first_seen_at, last_seen_at
                from entities
                order by canonical_name asc
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def graph_relations(
        self,
        *,
        scopes: tuple[str, ...],
        include_expired: bool,
        daily_lookback_days: int,
    ) -> list[dict[str, Any]]:
        clause, params = _document_filter_clause(
            scopes=scopes,
            include_expired=include_expired,
            daily_lookback_days=daily_lookback_days,
        )
        with self._connect_main() as conn:
            rows = conn.execute(
                f"""
                select
                    r.relation_id,
                    r.document_id,
                    r.subject,
                    r.predicate,
                    r.object,
                    r.status,
                    r.confidence,
                    r.cardinality,
                    r.first_seen_at,
                    r.last_seen_at,
                    r.valid_from,
                    r.valid_to,
                    d.path,
                    d.kind,
                    d.title,
                    d.updated_at,
                    d.status as document_status,
                    d.pinned,
                    d.priority,
                    d.review_after,
                    d.expires_at,
                    d.archived_at,
                    coalesce((
                        select json_group_array(source_ref_id)
                        from relation_source_refs rs
                        where rs.relation_id = r.relation_id
                    ), '[]') as source_ref_ids_json
                from relations r
                join documents d on d.document_id = r.document_id
                where {clause}
                order by d.updated_at desc
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def document_for_id_or_path(self, *, document_id: str | None = None, path: str | None = None) -> dict[str, Any] | None:
        if document_id is None and path is None:
            return None
        with self._connect_main() as conn:
            if document_id is not None:
                row = conn.execute(
                    "select * from documents where document_id = ?",
                    (document_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "select * from documents where path = ?",
                    (str(path),),
                ).fetchone()
        return dict(row) if row is not None else None

    def bootstrap_documents(self, *, kind: str) -> list[dict[str, Any]]:
        with self._connect_main() as conn:
            rows = conn.execute(
                """
                select document_id, path, kind, title, status, priority, pinned, updated_at, summary
                from documents
                where kind = ? and archived_at is null and status = 'active'
                order by pinned desc, priority desc, updated_at desc
                """,
                (kind,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_document_paths(self, *, kind: str | None = None) -> tuple[Path, ...]:
        query = "select path from documents"
        params: tuple[Any, ...] = ()
        if kind is not None:
            query += " where kind = ?"
            params = (kind,)
        with self._connect_main() as conn:
            rows = conn.execute(query, params).fetchall()
        return tuple(Path(str(row["path"])) for row in rows)

    def record_accesses(
        self,
        *,
        occurred_at: str,
        route_id: str | None,
        session_id: str | None,
        tool_name: str,
        query: str | None,
        mode: str,
        results: list[dict[str, Any]],
    ) -> None:
        if not results:
            return
        with self._connect_main() as conn:
            conn.executemany(
                """
                insert into access_log(
                    occurred_at, route_id, session_id, tool_name, query, mode,
                    document_id, chunk_id, result_rank, result_score
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        occurred_at,
                        route_id,
                        session_id,
                        tool_name,
                        query,
                        mode,
                        result.get("document_id"),
                        result.get("chunk_id"),
                        index,
                        result.get("score"),
                    )
                    for index, result in enumerate(results, start=1)
                ],
            )

    def record_maintenance_run(
        self,
        *,
        job_name: str,
        started_at: str,
        finished_at: str,
        status: str,
        summary: dict[str, Any],
    ) -> None:
        with self._connect_main() as conn:
            conn.execute(
                """
                insert into maintenance_runs(job_name, started_at, finished_at, status, summary_json)
                values (?, ?, ?, ?, ?)
                """,
                (job_name, started_at, finished_at, status, json.dumps(summary, ensure_ascii=True)),
            )

    def last_maintenance_run(self, job_name: str) -> dict[str, Any] | None:
        with self._connect_main() as conn:
            row = conn.execute(
                """
                select run_id, job_name, started_at, finished_at, status, summary_json
                from maintenance_runs
                where job_name = ?
                order by run_id desc
                limit 1
                """,
                (job_name,),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["summary"] = json.loads(str(payload.pop("summary_json")))
        return payload

    def render_bootstrap_cache_get(self, cache_key: str) -> dict[str, Any] | None:
        with self._connect_main() as conn:
            row = conn.execute(
                """
                select cache_key, generated_at, content, token_estimate, checksum_bundle
                from bootstrap_cache
                where cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        return dict(row) if row is not None else None

    def render_bootstrap_cache_set(
        self,
        *,
        cache_key: str,
        generated_at: str,
        content: str,
        token_estimate: int,
        checksum_bundle: str,
    ) -> None:
        with self._connect_main() as conn:
            conn.execute(
                """
                insert into bootstrap_cache(cache_key, generated_at, content, token_estimate, checksum_bundle)
                values (?, ?, ?, ?, ?)
                on conflict(cache_key) do update set
                    generated_at=excluded.generated_at,
                    content=excluded.content,
                    token_estimate=excluded.token_estimate,
                    checksum_bundle=excluded.checksum_bundle
                """,
                (cache_key, generated_at, content, token_estimate, checksum_bundle),
            )

    def sqlite_integrity(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        with self._connect_main() as conn:
            main_rows = conn.execute("pragma integrity_check").fetchall()
        with self._connect_embeddings() as conn:
            embeddings_rows = conn.execute("pragma integrity_check").fetchall()
        return (
            tuple(str(row[0]) for row in main_rows),
            tuple(str(row[0]) for row in embeddings_rows),
        )

    def load_state(self) -> dict[str, Any]:
        if not self._settings.state_path.exists():
            self.write_state(dict(_DEFAULT_STATE))
        try:
            payload = json.loads(self._settings.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        state = dict(_DEFAULT_STATE)
        state.update(payload)
        return state

    def write_state(self, state: dict[str, Any]) -> None:
        normalized = dict(_DEFAULT_STATE)
        normalized.update(state)
        self._settings.state_path.write_text(
            json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    def _connect_main(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._settings.main_index_path)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma foreign_keys = on")
        return conn

    def _connect_embeddings(self, *, load_vec: bool = False) -> sqlite3.Connection:
        conn = sqlite3.connect(self._settings.embeddings_index_path)
        conn.row_factory = sqlite3.Row
        if load_vec:
            self._load_sqlite_vec_for_connection(conn)
        return conn

    def _ensure_embeddings_schema(self) -> None:
        state = self.load_state()
        with self._connect_embeddings() as conn:
            conn.executescript(
                """
                pragma journal_mode = wal;
                create table if not exists embedding_items (
                    embedding_id integer primary key autoincrement,
                    item_key text not null unique,
                    item_type text not null,
                    target_id text not null,
                    document_id text not null,
                    path text not null,
                    section_path text not null,
                    source_ref_ids_json text not null,
                    text text not null,
                    text_checksum text not null,
                    embedding_blob text,
                    embedding_dimensions integer,
                    created_at text not null,
                    updated_at text not null
                );
                """
            )

        try:
            with self._connect_embeddings(load_vec=True):
                pass
            state["semantic_enabled"] = True
            state["semantic_error"] = None
        except Exception as exc:  # pragma: no cover - platform specific
            state["semantic_enabled"] = False
            state["semantic_error"] = str(exc)
            LOGGER.warning("sqlite-vec unavailable; semantic memory search disabled: %s", exc)
        self.write_state(state)

    def _ensure_vector_table(self, dimensions: int) -> None:
        state = self.load_state()
        if not state.get("semantic_enabled"):
            return
        current_dimensions = state.get("embedding_dimensions")
        with self._connect_embeddings(load_vec=True) as conn:
            if current_dimensions is not None and int(current_dimensions) != dimensions:
                conn.execute("drop table if exists embedding_items_vec")
            conn.execute(
                f"create virtual table if not exists embedding_items_vec using vec0(embedding float[{dimensions}])"
            )

    def _load_sqlite_vec_for_connection(self, conn: sqlite3.Connection) -> None:
        conn.enable_load_extension(True)
        try:
            _load_sqlite_vec_extension(conn)
        finally:
            conn.enable_load_extension(False)


def _bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _checksum_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entity_id_for_name(name: str) -> str:
    return "entity_" + "".join(character.lower() if character.isalnum() else "_" for character in name).strip("_")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = ? limit 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _embedding_items_for_document(
    *,
    document: MemoryDocument,
    chunks: tuple[DocumentChunk, ...],
) -> list[dict[str, Any]]:
    items = [
        {
            "item_key": f"chunk:{chunk.chunk_id}",
            "item_type": "chunk",
            "target_id": chunk.chunk_id,
            "document_id": document.document_id,
            "path": str(document.path),
            "section_path": chunk.section_path,
            "text": chunk.text,
            "source_ref_ids": [source_ref.source_ref_id for source_ref in document.source_refs],
        }
        for chunk in chunks
        if chunk.text.strip()
    ]
    items.extend(
        {
            "item_key": f"fact:{fact.fact_id}",
            "item_type": "fact",
            "target_id": fact.fact_id,
            "document_id": document.document_id,
            "path": str(document.path),
            "section_path": "facts",
            "text": fact.text,
            "source_ref_ids": list(fact.source_ref_ids),
        }
        for fact in document.facts
        if fact.text.strip()
    )
    items.extend(
        {
            "item_key": f"relation:{relation.relation_id}",
            "item_type": "relation",
            "target_id": relation.relation_id,
            "document_id": document.document_id,
            "path": str(document.path),
            "section_path": "relations",
            "text": relation.textualization,
            "source_ref_ids": list(relation.source_ref_ids),
        }
        for relation in document.relations
        if relation.textualization.strip()
    )
    return items


def _load_sqlite_vec_extension(conn: sqlite3.Connection, *, sqlite_vec_module: Any | None = None) -> None:
    if _SQLITE_VEC_OVERRIDE_PATH.with_suffix(".so").exists():
        try:
            conn.load_extension(str(_SQLITE_VEC_OVERRIDE_PATH))
            return
        except Exception:
            LOGGER.warning(
                "Failed loading sqlite-vec override at %s; falling back to bundled wheel binary.",
                _SQLITE_VEC_OVERRIDE_PATH.with_suffix(".so"),
                exc_info=True,
            )
    if sqlite_vec_module is None:
        import sqlite_vec as sqlite_vec_module
    sqlite_vec_module.load(conn)


def _document_filter_clause(
    *,
    scopes: tuple[str, ...],
    include_expired: bool,
    daily_lookback_days: int,
) -> tuple[str, tuple[Any, ...]]:
    scope_set = set(scopes)
    allow_archive = "archive" in scope_set
    active_kinds = tuple(kind for kind in ("core", "ongoing", "daily") if kind in scope_set)
    if not allow_archive and not active_kinds:
        active_kinds = ("core", "ongoing", "daily")

    clauses: list[str] = []
    params: list[Any] = []

    if active_kinds:
        placeholders = ", ".join("?" for _ in active_kinds)
        clauses.append(
            f"(d.archived_at is null and d.kind in ({placeholders}))"
        )
        params.extend(active_kinds)
    if allow_archive:
        clauses.append("(d.archived_at is not null)")

    where_parts = [f"({' or '.join(clauses)})"] if clauses else ["1=1"]
    if not include_expired:
        where_parts.append("(d.expires_at is null or d.expires_at > datetime('now'))")
    if "daily" in scope_set and "archive" not in scope_set:
        where_parts.append(
            "(d.kind != 'daily' or julianday('now') - julianday(substr(d.path, -13, 10)) <= ?)"
        )
        params.append(daily_lookback_days)
    return " and ".join(where_parts), tuple(params)
