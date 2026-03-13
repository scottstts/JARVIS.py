"""Typed models for the runtime memory subsystem."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

MemoryKind: TypeAlias = Literal["core", "ongoing", "daily"]
SearchScope: TypeAlias = Literal["core", "ongoing", "daily", "archive"]
SearchMode: TypeAlias = Literal["auto", "lexical", "semantic", "graph", "hybrid"]
ConfidenceLevel: TypeAlias = Literal["low", "medium", "high"]
FactStatus: TypeAlias = Literal["current", "past", "uncertain", "superseded"]
RelationCardinality: TypeAlias = Literal["single", "multi"]
CoreStatus: TypeAlias = Literal["active", "archived"]
OngoingStatus: TypeAlias = Literal["active", "closed", "archived"]
DailyStatus: TypeAlias = Literal["active", "closed", "archived"]
MaintenanceStatus: TypeAlias = Literal["ok", "warning", "error", "skipped"]
MemoryWriteOperation: TypeAlias = Literal[
    "create",
    "upsert",
    "append_daily",
    "close",
    "archive",
    "promote",
    "demote",
]


@dataclass(slots=True, frozen=True)
class Fact:
    fact_id: str
    text: str
    status: FactStatus
    confidence: ConfidenceLevel
    first_seen_at: str
    last_seen_at: str
    valid_from: str | None
    valid_to: str | None
    source_ref_ids: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class Relation:
    relation_id: str
    subject: str
    predicate: str
    object: str
    status: FactStatus
    confidence: ConfidenceLevel
    cardinality: RelationCardinality
    first_seen_at: str
    last_seen_at: str
    valid_from: str | None
    valid_to: str | None
    source_ref_ids: tuple[str, ...]

    @property
    def textualization(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


@dataclass(slots=True, frozen=True)
class EntityReference:
    entity_id: str
    name: str
    entity_type: str
    aliases: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class SourceReference:
    source_ref_id: str
    source_type: Literal["transcript", "manual", "tool", "import", "maintenance"]
    route_id: str | None
    session_id: str | None
    record_id: str | None
    tool_name: str | None
    note: str | None
    captured_at: str


@dataclass(slots=True, frozen=True)
class ParsedMarkdownDocument:
    path: Path
    raw_text: str
    frontmatter: dict[str, Any]
    title: str
    sections: OrderedDict[str, str]
    checksum: str


@dataclass(slots=True, frozen=True)
class MemoryDocument:
    path: Path
    memory_id: str
    kind: MemoryKind
    title: str
    status: str
    created_at: str
    updated_at: str
    sections: OrderedDict[str, str]
    checksum: str
    raw_markdown: str
    summary: str | None = None
    priority: int | None = None
    pinned: bool | None = None
    locked: bool | None = None
    confidence: ConfidenceLevel | None = None
    review_after: str | None = None
    expires_at: str | None = None
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    facts: tuple[Fact, ...] = ()
    relations: tuple[Relation, ...] = ()
    source_refs: tuple[SourceReference, ...] = ()
    entity_refs: tuple[EntityReference, ...] = ()
    completion_criteria: tuple[str, ...] = ()
    close_reason: str | None = None
    date: str | None = None
    timezone: str | None = None
    route_ids: tuple[str, ...] = ()
    session_ids: tuple[str, ...] = ()

    @property
    def document_id(self) -> str:
        return self.memory_id

    @property
    def archived(self) -> bool:
        return "/archive/" in self.path.as_posix() or self.status == "archived"

    @property
    def body_markdown(self) -> str:
        lines = [f"# {self.title}"]
        for heading, content in self.sections.items():
            lines.append("")
            lines.append(f"## {heading}")
            if content:
                lines.append(content.rstrip())
        return "\n".join(lines).rstrip() + "\n"


@dataclass(slots=True, frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    path: Path
    kind: MemoryKind
    ordinal: int
    section_path: str
    text: str
    token_estimate: int
    created_at: str
    updated_at: str


@dataclass(slots=True, frozen=True)
class DirtyDocument:
    path: Path
    detected_at: str
    reason: str


@dataclass(slots=True, frozen=True)
class SearchCandidate:
    document_id: str
    title: str
    path: Path
    kind: MemoryKind
    chunk_id: str
    section_path: str
    snippet: str
    source_ref_ids: tuple[str, ...]
    updated_at: str
    status: str
    pinned: bool = False
    priority: int | None = None
    review_after: str | None = None
    expires_at: str | None = None
    archived_at: str | None = None
    truth_status: FactStatus | None = None
    support_count: int = 0
    contradiction_count: int = 0
    last_confirmed_at: str | None = None
    last_contradicted_at: str | None = None
    lexical_raw_score: float | None = None
    lexical_score: float = 0.0
    semantic_distance: float | None = None
    semantic_score: float = 0.0
    graph_score: float = 0.0
    recency_score: float = 0.0
    fused_score: float = 0.0
    match_reasons: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class MemorySearchResult:
    document_id: str
    title: str
    path: Path
    kind: MemoryKind
    chunk_id: str
    section_path: str
    score: float
    snippet: str
    match_reasons: tuple[str, ...]
    source_ref_ids: tuple[str, ...]
    semantic_disabled: bool = False


@dataclass(slots=True, frozen=True)
class MemorySearchResponse:
    results: tuple[MemorySearchResult, ...]
    warnings: tuple[str, ...] = ()
    semantic_disabled: bool = False


@dataclass(slots=True, frozen=True)
class IntegrityIssue:
    path: Path | None
    severity: Literal["warning", "error"]
    code: str
    message: str


@dataclass(slots=True, frozen=True)
class MemoryWriteResult:
    operation: MemoryWriteOperation
    document_id: str
    path: Path
    summary: str
    changed_paths: tuple[Path, ...]


@dataclass(slots=True, frozen=True)
class MaintenanceRunResult:
    job_name: str
    status: MaintenanceStatus
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ReflectionAction:
    action: Literal[
        "append_daily",
        "create_ongoing",
        "update_ongoing",
        "close_ongoing",
        "create_core",
        "update_core",
        "ignore",
    ]
    confidence: ConfidenceLevel
    payload: dict[str, Any] = field(default_factory=dict)
    rationale: str | None = None


@dataclass(slots=True, frozen=True)
class ReflectionPlan:
    actions: tuple[ReflectionAction, ...]
    raw_text: str


@dataclass(slots=True, frozen=True)
class MaintenanceLLMRequest:
    job_name: str
    prompt: str
    max_output_tokens: int
