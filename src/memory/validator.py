"""Schema validation for canonical memory Markdown."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from .types import (
    EntityReference,
    Fact,
    MemoryDocument,
    ParsedMarkdownDocument,
    Relation,
    SourceReference,
)

_CONFIDENCE_VALUES = {"low", "medium", "high"}
_FACT_STATUS_VALUES = {"current", "past", "uncertain", "superseded"}
_CORE_STATUS_VALUES = {"active", "archived"}
_ONGOING_STATUS_VALUES = {"active", "closed", "archived"}
_DAILY_STATUS_VALUES = {"active", "closed", "archived"}
_RELATION_CARDINALITY_VALUES = {"single", "multi"}
_SOURCE_TYPE_VALUES = {"transcript", "manual", "tool", "import", "maintenance"}
_REQUIRED_CORE_SECTIONS = ("Summary", "Details", "Notes")
_REQUIRED_ONGOING_SECTIONS = ("Summary", "Current State", "Open Loops", "Artifacts", "Notes")
_REQUIRED_DAILY_SECTIONS = (
    "Notable Events",
    "Decisions",
    "Active Commitments",
    "Open Loops",
    "Artifacts",
    "Candidate Promotions",
)


class MemoryValidationError(ValueError):
    """Raised when a canonical memory file violates the schema."""


def validate_parsed_document(parsed: ParsedMarkdownDocument) -> MemoryDocument:
    frontmatter = parsed.frontmatter
    kind = _required_string(frontmatter, "kind")
    if kind not in {"core", "ongoing", "daily"}:
        raise MemoryValidationError(f"{parsed.path}: kind must be core, ongoing, or daily.")

    memory_id = _required_string(frontmatter, "memory_id")
    created_at = _required_string(frontmatter, "created_at")
    updated_at = _required_string(frontmatter, "updated_at")
    sections = _validated_sections(parsed.path, parsed.sections, kind)

    common_kwargs: dict[str, Any] = {
        "path": parsed.path,
        "memory_id": memory_id,
        "kind": kind,
        "title": parsed.title if kind == "daily" else _required_string(frontmatter, "title"),
        "created_at": created_at,
        "updated_at": updated_at,
        "sections": sections,
        "checksum": parsed.checksum,
        "raw_markdown": parsed.raw_text,
    }

    if kind == "core":
        status = _required_choice(frontmatter, "status", _CORE_STATUS_VALUES)
        priority = _required_int(frontmatter, "priority", min_value=0, max_value=100)
        return MemoryDocument(
            **common_kwargs,
            status=status,
            summary=_optional_string(frontmatter, "summary"),
            priority=priority,
            pinned=_required_bool(frontmatter, "pinned"),
            locked=_required_bool(frontmatter, "locked"),
            confidence=_required_choice(frontmatter, "confidence", _CONFIDENCE_VALUES),
            review_after=_optional_string(frontmatter, "review_after"),
            expires_at=_optional_string(frontmatter, "expires_at"),
            tags=_string_list(frontmatter, "tags"),
            aliases=_string_list(frontmatter, "aliases"),
            facts=_fact_list(frontmatter, "facts"),
            relations=_relation_list(frontmatter, "relations"),
            source_refs=_source_ref_list(frontmatter, "source_refs"),
            entity_refs=_entity_ref_list(frontmatter, "entity_refs"),
        )

    if kind == "ongoing":
        status = _required_choice(frontmatter, "status", _ONGOING_STATUS_VALUES)
        priority = _required_int(frontmatter, "priority", min_value=0, max_value=100)
        return MemoryDocument(
            **common_kwargs,
            status=status,
            summary=_optional_string(frontmatter, "summary"),
            priority=priority,
            pinned=_required_bool(frontmatter, "pinned"),
            locked=_required_bool(frontmatter, "locked"),
            confidence=_required_choice(frontmatter, "confidence", _CONFIDENCE_VALUES),
            review_after=_required_string(frontmatter, "review_after"),
            expires_at=_optional_string(frontmatter, "expires_at"),
            tags=_string_list(frontmatter, "tags"),
            aliases=_string_list(frontmatter, "aliases"),
            facts=_fact_list(frontmatter, "facts"),
            relations=_relation_list(frontmatter, "relations"),
            source_refs=_source_ref_list(frontmatter, "source_refs"),
            entity_refs=_entity_ref_list(frontmatter, "entity_refs"),
            completion_criteria=_string_list(frontmatter, "completion_criteria"),
            close_reason=_optional_string(frontmatter, "close_reason"),
        )

    return MemoryDocument(
        **common_kwargs,
        status=_required_choice(frontmatter, "status", _DAILY_STATUS_VALUES),
        date=_required_string(frontmatter, "date"),
        timezone=_required_string(frontmatter, "timezone"),
        route_ids=_string_list(frontmatter, "route_ids"),
        session_ids=_string_list(frontmatter, "session_ids"),
    )


def _validated_sections(
    path: Path,
    sections: OrderedDict[str, str],
    kind: str,
) -> OrderedDict[str, str]:
    required = (
        _REQUIRED_CORE_SECTIONS
        if kind == "core"
        else _REQUIRED_ONGOING_SECTIONS
        if kind == "ongoing"
        else _REQUIRED_DAILY_SECTIONS
    )
    normalized = OrderedDict((heading, sections.get(heading, "").strip()) for heading in required)
    missing = [heading for heading in required if heading not in sections]
    if missing:
        missing_text = ", ".join(missing)
        raise MemoryValidationError(f"{path}: missing required sections: {missing_text}")
    return normalized


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"Missing required string field: {key}")
    return value.strip()


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    raise MemoryValidationError(f"Optional field '{key}' must be a string or null.")


def _required_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise MemoryValidationError(f"Required field '{key}' must be a boolean.")
    return value


def _required_choice(payload: dict[str, Any], key: str, allowed: set[str]) -> str:
    value = _required_string(payload, key)
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise MemoryValidationError(f"Field '{key}' must be one of: {allowed_values}.")
    return value


def _required_int(
    payload: dict[str, Any],
    key: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise MemoryValidationError(f"Required field '{key}' must be an integer.")
    if min_value is not None and value < min_value:
        raise MemoryValidationError(f"Field '{key}' must be >= {min_value}.")
    if max_value is not None and value > max_value:
        raise MemoryValidationError(f"Field '{key}' must be <= {max_value}.")
    return value


def _string_list(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise MemoryValidationError(f"Field '{key}' must be a list.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise MemoryValidationError(f"Field '{key}' must contain non-empty strings.")
        normalized.append(item.strip())
    return tuple(normalized)


def _fact_list(payload: dict[str, Any], key: str) -> tuple[Fact, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise MemoryValidationError(f"Field '{key}' must be a list.")
    facts: list[Fact] = []
    for item in value:
        if not isinstance(item, dict):
            raise MemoryValidationError(f"Fact entries under '{key}' must be objects.")
        status = _required_choice(item, "status", _FACT_STATUS_VALUES)
        facts.append(
            Fact(
                fact_id=_required_string(item, "fact_id"),
                text=_required_string(item, "text"),
                status=status,  # type: ignore[arg-type]
                confidence=_required_choice(item, "confidence", _CONFIDENCE_VALUES),  # type: ignore[arg-type]
                first_seen_at=_required_string(item, "first_seen_at"),
                last_seen_at=_required_string(item, "last_seen_at"),
                valid_from=_optional_string(item, "valid_from"),
                valid_to=_optional_string(item, "valid_to"),
                source_ref_ids=_string_list(item, "source_ref_ids"),
            )
        )
    return tuple(facts)


def _relation_list(payload: dict[str, Any], key: str) -> tuple[Relation, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise MemoryValidationError(f"Field '{key}' must be a list.")
    relations: list[Relation] = []
    for item in value:
        if not isinstance(item, dict):
            raise MemoryValidationError(f"Relation entries under '{key}' must be objects.")
        status = _required_choice(item, "status", _FACT_STATUS_VALUES)
        cardinality = _required_choice(item, "cardinality", _RELATION_CARDINALITY_VALUES)
        relations.append(
            Relation(
                relation_id=_required_string(item, "relation_id"),
                subject=_required_string(item, "subject"),
                predicate=_required_string(item, "predicate"),
                object=_required_string(item, "object"),
                status=status,  # type: ignore[arg-type]
                confidence=_required_choice(item, "confidence", _CONFIDENCE_VALUES),  # type: ignore[arg-type]
                cardinality=cardinality,  # type: ignore[arg-type]
                first_seen_at=_required_string(item, "first_seen_at"),
                last_seen_at=_required_string(item, "last_seen_at"),
                valid_from=_optional_string(item, "valid_from"),
                valid_to=_optional_string(item, "valid_to"),
                source_ref_ids=_string_list(item, "source_ref_ids"),
            )
        )
    return tuple(relations)


def _entity_ref_list(payload: dict[str, Any], key: str) -> tuple[EntityReference, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise MemoryValidationError(f"Field '{key}' must be a list.")
    entries: list[EntityReference] = []
    for item in value:
        if not isinstance(item, dict):
            raise MemoryValidationError(f"Entity references under '{key}' must be objects.")
        entries.append(
            EntityReference(
                entity_id=_required_string(item, "entity_id"),
                name=_required_string(item, "name"),
                entity_type=_required_string(item, "entity_type"),
                aliases=_string_list(item, "aliases"),
            )
        )
    return tuple(entries)


def _source_ref_list(payload: dict[str, Any], key: str) -> tuple[SourceReference, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise MemoryValidationError(f"Field '{key}' must be a list.")
    refs: list[SourceReference] = []
    for item in value:
        if not isinstance(item, dict):
            raise MemoryValidationError(f"Source references under '{key}' must be objects.")
        source_type = _required_choice(item, "source_type", _SOURCE_TYPE_VALUES)
        refs.append(
            SourceReference(
                source_ref_id=_required_string(item, "source_ref_id"),
                source_type=source_type,  # type: ignore[arg-type]
                route_id=_optional_string(item, "route_id"),
                session_id=_optional_string(item, "session_id"),
                record_id=_optional_string(item, "record_id"),
                tool_name=_optional_string(item, "tool_name"),
                note=_optional_string(item, "note"),
                captured_at=_required_string(item, "captured_at"),
            )
        )
    return tuple(refs)
