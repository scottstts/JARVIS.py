"""Canonical Markdown file reads and writes for memory documents."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import re
from typing import Any

import yaml

from .config import MemorySettings
from .parser import parse_markdown_document
from .types import MemoryDocument
from .validator import validate_parsed_document

_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
_CORE_FRONTMATTER_ORDER = (
    "memory_id",
    "kind",
    "title",
    "status",
    "created_at",
    "updated_at",
    "priority",
    "pinned",
    "locked",
    "confidence",
    "review_after",
    "expires_at",
    "tags",
    "aliases",
    "summary",
    "entity_refs",
    "facts",
    "relations",
    "source_refs",
)
_ONGOING_FRONTMATTER_ORDER = (
    "memory_id",
    "kind",
    "title",
    "status",
    "created_at",
    "updated_at",
    "priority",
    "pinned",
    "locked",
    "confidence",
    "review_after",
    "expires_at",
    "tags",
    "aliases",
    "summary",
    "entity_refs",
    "completion_criteria",
    "close_reason",
    "facts",
    "relations",
    "source_refs",
)
_DAILY_FRONTMATTER_ORDER = (
    "memory_id",
    "kind",
    "date",
    "timezone",
    "status",
    "created_at",
    "updated_at",
    "route_ids",
    "session_ids",
)


class MarkdownMemoryStore:
    """Owns canonical runtime memory files under the workspace memory directory."""

    def __init__(self, settings: MemorySettings) -> None:
        self._settings = settings
        self.ensure_layout()

    def ensure_layout(self) -> None:
        for directory in (
            self._settings.memory_dir,
            self._settings.core_dir,
            self._settings.ongoing_dir,
            self._settings.daily_dir,
            self._settings.archive_dir,
            self._settings.archive_core_dir,
            self._settings.archive_ongoing_dir,
            self._settings.archive_daily_dir,
            self._settings.index_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def list_markdown_paths(self) -> tuple[Path, ...]:
        candidates: list[Path] = []
        for directory in (
            self._settings.core_dir,
            self._settings.ongoing_dir,
            self._settings.daily_dir,
            self._settings.archive_core_dir,
            self._settings.archive_ongoing_dir,
            self._settings.archive_daily_dir,
        ):
            candidates.extend(sorted(directory.glob("*.md")))
        return tuple(candidates)

    def read_document(self, path: Path) -> MemoryDocument:
        return validate_parsed_document(parse_markdown_document(path))

    def read_all_documents(self) -> tuple[MemoryDocument, ...]:
        return tuple(self.read_document(path) for path in self.list_markdown_paths())

    def write_document(
        self,
        document: MemoryDocument,
        *,
        previous_path: Path | None = None,
    ) -> MemoryDocument:
        rendered = render_memory_document(document)
        target_path = document.path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(".tmp")
        tmp_path.write_text(rendered, encoding="utf-8", newline="\n")
        tmp_path.replace(target_path)
        if previous_path is not None and previous_path != target_path and previous_path.exists():
            previous_path.unlink()
        return self.read_document(target_path)

    def archive_document(self, document: MemoryDocument) -> MemoryDocument:
        destination = self.archive_path_for(document)
        archived_status = "closed" if document.kind == "ongoing" and document.status == "closed" else "archived"
        archived = replace(
            document,
            path=destination,
            status=archived_status,
        )
        self.write_document(archived, previous_path=document.path)
        if document.path != destination and document.path.exists():
            document.path.unlink()
        return self.read_document(destination)

    def archive_path_for(self, document: MemoryDocument) -> Path:
        return self.canonical_path_for(
            kind=document.kind,
            title=document.title,
            date=document.date,
            archived=True,
        )

    def active_path_for(self, *, kind: str, title: str, date: str | None = None) -> Path:
        return self.canonical_path_for(kind=kind, title=title, date=date, archived=False)

    def canonical_path_for(
        self,
        *,
        kind: str,
        title: str,
        date: str | None = None,
        archived: bool,
    ) -> Path:
        if kind == "core":
            directory = self._settings.archive_core_dir if archived else self._settings.core_dir
            return directory / f"{slugify(title)}.md"
        if kind == "ongoing":
            directory = self._settings.archive_ongoing_dir if archived else self._settings.ongoing_dir
            return directory / f"{slugify(title)}.md"
        if date is None:
            raise ValueError("Daily documents require a date.")
        directory = self._settings.archive_daily_dir if archived else self._settings.daily_dir
        return directory / f"{date}.md"


def slugify(value: str) -> str:
    normalized = _NON_ALNUM_PATTERN.sub("-", value.strip().lower()).strip("-")
    return normalized or "memory"


def render_memory_document(document: MemoryDocument) -> str:
    frontmatter = dict(_build_frontmatter(document))
    frontmatter_text = yaml.safe_dump(
        frontmatter,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    ).strip()
    body_lines = [f"# {document.title}"]
    for heading, content in document.sections.items():
        body_lines.append("")
        body_lines.append(f"## {heading}")
        if content:
            body_lines.append(content.rstrip())
    body = "\n".join(body_lines).rstrip()
    return f"---\n{frontmatter_text}\n---\n{body}\n"


def _build_frontmatter(document: MemoryDocument) -> OrderedDict[str, Any]:
    order = (
        _CORE_FRONTMATTER_ORDER
        if document.kind == "core"
        else _ONGOING_FRONTMATTER_ORDER
        if document.kind == "ongoing"
        else _DAILY_FRONTMATTER_ORDER
    )
    payload: dict[str, Any] = {
        "memory_id": document.memory_id,
        "kind": document.kind,
        "title": document.title,
        "status": document.status,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "summary": document.summary,
        "priority": document.priority,
        "pinned": document.pinned,
        "locked": document.locked,
        "confidence": document.confidence,
        "review_after": document.review_after,
        "expires_at": document.expires_at,
        "tags": list(document.tags),
        "aliases": list(document.aliases),
        "facts": [_fact_to_dict(fact) for fact in document.facts],
        "relations": [_relation_to_dict(relation) for relation in document.relations],
        "source_refs": [_source_ref_to_dict(source_ref) for source_ref in document.source_refs],
        "entity_refs": [_entity_ref_to_dict(entity_ref) for entity_ref in document.entity_refs],
        "completion_criteria": list(document.completion_criteria),
        "close_reason": document.close_reason,
        "date": document.date,
        "timezone": document.timezone,
        "route_ids": list(document.route_ids),
        "session_ids": list(document.session_ids),
    }
    ordered: OrderedDict[str, Any] = OrderedDict()
    for key in order:
        if key not in payload:
            continue
        value = payload[key]
        if value is None:
            ordered[key] = None
            continue
        if isinstance(value, list) and key not in {"facts", "relations", "source_refs", "entity_refs"} and not value:
            ordered[key] = []
            continue
        ordered[key] = value
    return ordered


def _fact_to_dict(fact: Any) -> dict[str, Any]:
    return {
        "fact_id": fact.fact_id,
        "text": fact.text,
        "status": fact.status,
        "confidence": fact.confidence,
        "first_seen_at": fact.first_seen_at,
        "last_seen_at": fact.last_seen_at,
        "valid_from": fact.valid_from,
        "valid_to": fact.valid_to,
        "source_ref_ids": list(fact.source_ref_ids),
    }


def _relation_to_dict(relation: Any) -> dict[str, Any]:
    return {
        "relation_id": relation.relation_id,
        "subject": relation.subject,
        "predicate": relation.predicate,
        "object": relation.object,
        "status": relation.status,
        "confidence": relation.confidence,
        "cardinality": relation.cardinality,
        "first_seen_at": relation.first_seen_at,
        "last_seen_at": relation.last_seen_at,
        "valid_from": relation.valid_from,
        "valid_to": relation.valid_to,
        "source_ref_ids": list(relation.source_ref_ids),
    }


def _entity_ref_to_dict(entity_ref: Any) -> dict[str, Any]:
    return {
        "entity_id": entity_ref.entity_id,
        "name": entity_ref.name,
        "entity_type": entity_ref.entity_type,
        "aliases": list(entity_ref.aliases),
    }


def _source_ref_to_dict(source_ref: Any) -> dict[str, Any]:
    return {
        "source_ref_id": source_ref.source_ref_id,
        "source_type": source_ref.source_type,
        "route_id": source_ref.route_id,
        "session_id": source_ref.session_id,
        "record_id": source_ref.record_id,
        "tool_name": source_ref.tool_name,
        "note": source_ref.note,
        "captured_at": source_ref.captured_at,
    }
