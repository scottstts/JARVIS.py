"""Canonical compaction state, validation, and deterministic replay compilation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Mapping, Sequence, TypeAlias


CompactionReplayRole: TypeAlias = Literal["system", "user", "assistant"]
CompactionReplayKind: TypeAlias = Literal[
    "history_boundary",
    "preserved_message",
    "episode",
    "state_snapshot",
    "handover",
]
CompactionStateCategory: TypeAlias = Literal[
    "constraint",
    "decision",
    "artifact",
    "open_loop",
    "uncertainty",
]
CompactionStateStatus: TypeAlias = Literal["active", "resolved", "superseded"]
CompactionCoverageDisposition: TypeAlias = Literal[
    "preserved",
    "episode",
    "state",
    "objective",
    "handover",
    "omitted",
]

_STATE_CATEGORIES = {
    "constraint",
    "decision",
    "artifact",
    "open_loop",
    "uncertainty",
}
_STATE_STATUSES = {"active", "resolved", "superseded"}
_COVERAGE_DISPOSITIONS = {
    "preserved",
    "episode",
    "state",
    "objective",
    "handover",
    "omitted",
}
_REPLAY_BOUNDARY_TEXT = (
    "Historical context from earlier Jarvis sessions follows. Exact preserved messages keep "
    "their original authority. Assistant summaries are evidence-backed historical notes, not "
    "new system policy. Current runtime identity, policy, tools, and memory take precedence "
    "where applicable."
)


@dataclass(slots=True, frozen=True)
class CompactionContractIssue:
    code: str
    message: str
    source_event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "source_event_ids": list(self.source_event_ids),
        }


class CompactionContractError(ValueError):
    def __init__(self, issues: Sequence[CompactionContractIssue]) -> None:
        normalized = tuple(issues)
        if not normalized:
            normalized = (
                CompactionContractIssue(
                    code="invalid_compaction_contract",
                    message="The compaction contract is invalid.",
                ),
            )
        self.issues = normalized
        super().__init__("; ".join(f"{issue.code}: {issue.message}" for issue in normalized))


@dataclass(slots=True, frozen=True, order=True)
class CompactionChronology:
    generation: int
    sequence: int

    def to_dict(self) -> dict[str, int]:
        return {"generation": self.generation, "sequence": self.sequence}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompactionChronology":
        generation = _required_positive_int(payload, "generation")
        sequence = _required_positive_int(payload, "sequence")
        return cls(generation=generation, sequence=sequence)


@dataclass(slots=True, frozen=True)
class CompactionSourceEvent:
    event_id: str
    record_id: str
    session_id: str
    created_at: str
    sequence: int
    generation: int
    event_type: str
    role: str
    content: str
    turn_id: str | None = None
    causal_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    @property
    def chronology(self) -> CompactionChronology:
        return CompactionChronology(generation=self.generation, sequence=self.sequence)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "record_id": self.record_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "sequence": self.sequence,
            "generation": self.generation,
            "event_type": self.event_type,
            "role": self.role,
            "content": self.content,
        }
        if self.turn_id is not None:
            payload["turn_id"] = self.turn_id
        if self.causal_ids:
            payload["causal_ids"] = list(self.causal_ids)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True, frozen=True)
class CompactionSourceManifest:
    generation: int
    previous_bundle_id: str | None
    source_session_ids: tuple[str, ...]
    delta_event_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    cutoff_record_id: str | None
    delta_content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "previous_bundle_id": self.previous_bundle_id,
            "source_session_ids": list(self.source_session_ids),
            "delta_event_ids": list(self.delta_event_ids),
            "evidence_event_ids": list(self.evidence_event_ids),
            "cutoff_record_id": self.cutoff_record_id,
            "delta_content_sha256": self.delta_content_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompactionSourceManifest":
        return cls(
            generation=_required_positive_int(payload, "generation"),
            previous_bundle_id=_optional_string(payload.get("previous_bundle_id")),
            source_session_ids=_required_string_tuple(payload, "source_session_ids"),
            delta_event_ids=_required_string_tuple(payload, "delta_event_ids", allow_empty=True),
            evidence_event_ids=_required_string_tuple(
                payload,
                "evidence_event_ids",
                allow_empty=True,
            ),
            cutoff_record_id=_optional_string(payload.get("cutoff_record_id")),
            delta_content_sha256=_required_string(payload, "delta_content_sha256"),
        )


@dataclass(slots=True, frozen=True)
class CompactionObjective:
    summary: str
    evidence_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "evidence_event_ids": list(self.evidence_event_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompactionObjective":
        return cls(
            summary=_required_string(payload, "summary"),
            evidence_event_ids=_required_string_tuple(payload, "evidence_event_ids"),
        )


@dataclass(slots=True, frozen=True)
class CompactionPreservedRecord:
    record_id: str
    source_session_id: str
    created_at: str
    role: Literal["user", "assistant"]
    content: str
    content_sha256: str
    reason: str
    chronology: CompactionChronology

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_session_id": self.source_session_id,
            "created_at": self.created_at,
            "role": self.role,
            "content": self.content,
            "content_sha256": self.content_sha256,
            "reason": self.reason,
            "chronology": self.chronology.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompactionPreservedRecord":
        role = _required_string(payload, "role")
        if role not in {"user", "assistant"}:
            raise CompactionContractError(
                [
                    CompactionContractIssue(
                        code="invalid_preserved_role",
                        message=f"Preserved record role must be user or assistant, got {role!r}.",
                    )
                ]
            )
        content = payload.get("content")
        if not isinstance(content, str) or not content:
            raise CompactionContractError(
                [
                    CompactionContractIssue(
                        code="invalid_preserved_content",
                        message="Preserved record content must be a non-empty exact string.",
                    )
                ]
            )
        content_sha256 = _required_string(payload, "content_sha256")
        if _sha256_text(content) != content_sha256:
            raise CompactionContractError(
                [
                    CompactionContractIssue(
                        code="preserved_content_hash_mismatch",
                        message="Preserved record content does not match its stored hash.",
                    )
                ]
            )
        chronology_payload = _required_mapping(payload, "chronology")
        return cls(
            record_id=_required_string(payload, "record_id"),
            source_session_id=_required_string(payload, "source_session_id"),
            created_at=_required_string(payload, "created_at"),
            role=role,  # type: ignore[arg-type]
            content=content,
            content_sha256=content_sha256,
            reason=_required_string(payload, "reason"),
            chronology=CompactionChronology.from_dict(chronology_payload),
        )


@dataclass(slots=True, frozen=True)
class CompactionEpisode:
    episode_id: str
    summary: str
    outcomes: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    chronology: CompactionChronology

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "summary": self.summary,
            "outcomes": list(self.outcomes),
            "source_event_ids": list(self.source_event_ids),
            "evidence_event_ids": list(self.evidence_event_ids),
            "chronology": self.chronology.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompactionEpisode":
        return cls(
            episode_id=_required_identifier(payload, "episode_id"),
            summary=_required_string(payload, "summary"),
            outcomes=_required_string_tuple(payload, "outcomes", allow_empty=True),
            source_event_ids=_required_string_tuple(payload, "source_event_ids"),
            evidence_event_ids=_required_string_tuple(payload, "evidence_event_ids"),
            chronology=CompactionChronology.from_dict(
                _required_mapping(payload, "chronology")
            ),
        )


@dataclass(slots=True, frozen=True)
class CompactionStateEntry:
    entry_id: str
    category: CompactionStateCategory
    summary: str
    status: CompactionStateStatus
    evidence_event_ids: tuple[str, ...]
    supersedes_entry_ids: tuple[str, ...] = ()
    locator: str | None = None
    last_observed_state: str | None = None
    needs_verification: bool = False
    blocker: str | None = None
    next_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "category": self.category,
            "summary": self.summary,
            "status": self.status,
            "evidence_event_ids": list(self.evidence_event_ids),
            "supersedes_entry_ids": list(self.supersedes_entry_ids),
            "locator": self.locator,
            "last_observed_state": self.last_observed_state,
            "needs_verification": self.needs_verification,
            "blocker": self.blocker,
            "next_action": self.next_action,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompactionStateEntry":
        category = _required_string(payload, "category")
        status = _required_string(payload, "status")
        if category not in _STATE_CATEGORIES:
            raise _single_issue(
                "invalid_state_category",
                f"Unsupported compaction state category: {category!r}.",
            )
        if status not in _STATE_STATUSES:
            raise _single_issue(
                "invalid_state_status",
                f"Unsupported compaction state status: {status!r}.",
            )
        needs_verification = payload.get("needs_verification", False)
        if not isinstance(needs_verification, bool):
            raise _single_issue(
                "invalid_field_type",
                "needs_verification must be a boolean.",
            )
        entry = cls(
            entry_id=_required_identifier(payload, "entry_id"),
            category=category,  # type: ignore[arg-type]
            summary=_required_string(payload, "summary"),
            status=status,  # type: ignore[arg-type]
            evidence_event_ids=_required_string_tuple(payload, "evidence_event_ids"),
            supersedes_entry_ids=_required_string_tuple(
                payload,
                "supersedes_entry_ids",
                allow_empty=True,
            ),
            locator=_optional_string(payload.get("locator")),
            last_observed_state=_optional_string(payload.get("last_observed_state")),
            needs_verification=needs_verification,
            blocker=_optional_string(payload.get("blocker")),
            next_action=_optional_string(payload.get("next_action")),
        )
        _validate_state_entry_shape(entry)
        return entry


@dataclass(slots=True, frozen=True)
class CompactionHandover:
    current_focus: str
    next_actions: tuple[str, ...]
    do_not_repeat: tuple[str, ...]
    verification_needed: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_focus": self.current_focus,
            "next_actions": list(self.next_actions),
            "do_not_repeat": list(self.do_not_repeat),
            "verification_needed": list(self.verification_needed),
            "evidence_event_ids": list(self.evidence_event_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompactionHandover":
        return cls(
            current_focus=_required_string(payload, "current_focus"),
            next_actions=_required_string_tuple(payload, "next_actions", allow_empty=True),
            do_not_repeat=_required_string_tuple(payload, "do_not_repeat", allow_empty=True),
            verification_needed=_required_string_tuple(
                payload,
                "verification_needed",
                allow_empty=True,
            ),
            evidence_event_ids=_required_string_tuple(payload, "evidence_event_ids"),
        )


@dataclass(slots=True, frozen=True)
class CompactionCoverageGroup:
    generation: int
    source_event_ids: tuple[str, ...]
    disposition: CompactionCoverageDisposition
    target_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "source_event_ids": list(self.source_event_ids),
            "disposition": self.disposition,
            "target_ids": list(self.target_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompactionCoverageGroup":
        disposition = _required_string(payload, "disposition")
        if disposition not in _COVERAGE_DISPOSITIONS:
            raise _single_issue(
                "invalid_coverage_disposition",
                f"Unsupported coverage disposition: {disposition!r}.",
            )
        return cls(
            generation=_required_positive_int(payload, "generation"),
            source_event_ids=_required_string_tuple(payload, "source_event_ids"),
            disposition=disposition,  # type: ignore[arg-type]
            target_ids=_required_string_tuple(payload, "target_ids", allow_empty=True),
            reason=_required_string(payload, "reason"),
        )


@dataclass(slots=True, frozen=True)
class CompactionBundle:
    schema_version: int
    bundle_id: str
    created_at: str
    source_manifest: CompactionSourceManifest
    objective: CompactionObjective
    preserved_records: tuple[CompactionPreservedRecord, ...]
    episodes: tuple[CompactionEpisode, ...]
    state_entries: tuple[CompactionStateEntry, ...]
    handover: CompactionHandover
    coverage: tuple[CompactionCoverageGroup, ...]

    @property
    def generation(self) -> int:
        return self.source_manifest.generation

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "source_manifest": self.source_manifest.to_dict(),
            "objective": self.objective.to_dict(),
            "preserved_records": [item.to_dict() for item in self.preserved_records],
            "episodes": [item.to_dict() for item in self.episodes],
            "state_entries": [item.to_dict() for item in self.state_entries],
            "handover": self.handover.to_dict(),
            "coverage": [item.to_dict() for item in self.coverage],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompactionBundle":
        schema_version = _required_positive_int(payload, "schema_version")
        if schema_version != 2:
            raise _single_issue(
                "unsupported_compaction_schema",
                f"Expected compaction schema version 2, got {schema_version}.",
            )
        bundle = cls(
            schema_version=schema_version,
            bundle_id=_required_identifier(payload, "bundle_id"),
            created_at=_required_string(payload, "created_at"),
            source_manifest=CompactionSourceManifest.from_dict(
                _required_mapping(payload, "source_manifest")
            ),
            objective=CompactionObjective.from_dict(_required_mapping(payload, "objective")),
            preserved_records=tuple(
                CompactionPreservedRecord.from_dict(item)
                for item in _required_mapping_list(payload, "preserved_records")
            ),
            episodes=tuple(
                CompactionEpisode.from_dict(item)
                for item in _required_mapping_list(payload, "episodes")
            ),
            state_entries=tuple(
                CompactionStateEntry.from_dict(item)
                for item in _required_mapping_list(payload, "state_entries")
            ),
            handover=CompactionHandover.from_dict(_required_mapping(payload, "handover")),
            coverage=tuple(
                CompactionCoverageGroup.from_dict(item)
                for item in _required_mapping_list(payload, "coverage")
            ),
        )
        validate_compaction_bundle(bundle)
        return bundle


@dataclass(slots=True, frozen=True)
class CompactionReplayItem:
    role: CompactionReplayRole
    kind: CompactionReplayKind
    content: str
    bundle_id: str
    generation: int
    exact_copy: bool = False
    source_record_ids: tuple[str, ...] = ()
    evidence_event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "compaction_replay",
            "role": self.role,
            "kind": self.kind,
            "content": self.content,
            "bundle_id": self.bundle_id,
            "generation": self.generation,
            "exact_copy": self.exact_copy,
            "source_record_ids": list(self.source_record_ids),
            "evidence_event_ids": list(self.evidence_event_ids),
        }

    def record_metadata(self) -> dict[str, Any]:
        return {
            "type": "compaction_replay",
            "compaction_item": True,
            "compaction_kind": self.kind,
            "compaction_bundle_id": self.bundle_id,
            "compaction_generation": self.generation,
            "exact_copy": self.exact_copy,
            "source_record_ids": list(self.source_record_ids),
            "evidence_event_ids": list(self.evidence_event_ids),
        }


def build_source_manifest(
    *,
    generation: int,
    previous_bundle: CompactionBundle | None,
    source_events: Sequence[CompactionSourceEvent],
) -> CompactionSourceManifest:
    delta_event_ids = tuple(event.event_id for event in source_events)
    previous_evidence_ids = (
        previous_bundle.source_manifest.evidence_event_ids if previous_bundle is not None else ()
    )
    evidence_event_ids = _ordered_unique((*previous_evidence_ids, *delta_event_ids))
    previous_sessions = (
        previous_bundle.source_manifest.source_session_ids if previous_bundle is not None else ()
    )
    source_session_ids = _ordered_unique(
        (*previous_sessions, *(event.session_id for event in source_events))
    )
    serialized_events = "\n".join(
        _canonical_json(event.to_dict()) for event in source_events
    )
    return CompactionSourceManifest(
        generation=generation,
        previous_bundle_id=(previous_bundle.bundle_id if previous_bundle is not None else None),
        source_session_ids=source_session_ids,
        delta_event_ids=delta_event_ids,
        evidence_event_ids=evidence_event_ids,
        cutoff_record_id=(source_events[-1].record_id if source_events else None),
        delta_content_sha256=_sha256_text(serialized_events),
    )


def apply_compaction_draft(
    payload: Mapping[str, Any],
    *,
    bundle_id: str,
    created_at: str,
    source_manifest: CompactionSourceManifest,
    source_events: Sequence[CompactionSourceEvent],
    previous_bundle: CompactionBundle | None,
) -> CompactionBundle:
    expected_keys = {
        "objective",
        "preserved_actions",
        "episode_actions",
        "state_operations",
        "handover",
        "coverage",
    }
    _require_exact_keys(payload, expected_keys, location="compaction draft")
    event_by_id = {event.event_id: event for event in source_events}
    known_evidence_ids = set(source_manifest.evidence_event_ids)
    delta_event_ids = set(source_manifest.delta_event_ids)

    objective = CompactionObjective.from_dict(_required_mapping(payload, "objective"))
    _validate_evidence_ids(
        objective.evidence_event_ids,
        known_evidence_ids=known_evidence_ids,
        location="objective",
    )
    if previous_bundle is not None and objective != previous_bundle.objective:
        _require_delta_evidence(
            objective.evidence_event_ids,
            delta_event_ids=delta_event_ids,
            location="changed objective",
        )

    preserved = _apply_preserved_actions(
        _required_mapping_list(payload, "preserved_actions"),
        previous_bundle=previous_bundle,
        event_by_id=event_by_id,
        known_evidence_ids=known_evidence_ids,
        delta_event_ids=delta_event_ids,
    )
    episodes = _apply_episode_actions(
        _required_mapping_list(payload, "episode_actions"),
        previous_bundle=previous_bundle,
        event_by_id=event_by_id,
        known_evidence_ids=known_evidence_ids,
        delta_event_ids=delta_event_ids,
    )
    state_entries = _apply_state_operations(
        _required_mapping_list(payload, "state_operations"),
        previous_bundle=previous_bundle,
        event_by_id=event_by_id,
        known_evidence_ids=known_evidence_ids,
        delta_event_ids=delta_event_ids,
    )
    handover = CompactionHandover.from_dict(_required_mapping(payload, "handover"))
    _validate_evidence_ids(
        handover.evidence_event_ids,
        known_evidence_ids=known_evidence_ids,
        location="handover",
    )

    new_coverage = tuple(
        _coverage_from_draft(item, generation=source_manifest.generation)
        for item in _required_mapping_list(payload, "coverage")
    )
    prior_coverage = previous_bundle.coverage if previous_bundle is not None else ()
    bundle = CompactionBundle(
        schema_version=2,
        bundle_id=bundle_id,
        created_at=created_at,
        source_manifest=source_manifest,
        objective=objective,
        preserved_records=preserved,
        episodes=episodes,
        state_entries=state_entries,
        handover=handover,
        coverage=(*prior_coverage, *new_coverage),
    )
    validate_compaction_bundle(bundle)
    _validate_delta_coverage(bundle, source_events=source_events, new_coverage=new_coverage)
    return bundle


def validate_compaction_bundle(bundle: CompactionBundle) -> None:
    issues: list[CompactionContractIssue] = []
    known_evidence_ids = set(bundle.source_manifest.evidence_event_ids)
    if bundle.source_manifest.generation <= 0:
        issues.append(
            CompactionContractIssue(
                code="invalid_generation",
                message="Compaction generation must be positive.",
            )
        )
    if not set(bundle.source_manifest.delta_event_ids).issubset(known_evidence_ids):
        issues.append(
            CompactionContractIssue(
                code="delta_not_in_evidence_manifest",
                message="Every delta event must be retained in the evidence manifest.",
            )
        )

    preserved_ids = [item.record_id for item in bundle.preserved_records]
    episode_ids = [item.episode_id for item in bundle.episodes]
    state_ids = [item.entry_id for item in bundle.state_entries]
    for label, values in (
        ("preserved record", preserved_ids),
        ("episode", episode_ids),
        ("state entry", state_ids),
    ):
        duplicates = _duplicates(values)
        if duplicates:
            issues.append(
                CompactionContractIssue(
                    code=f"duplicate_{label.replace(' ', '_')}_id",
                    message=f"Duplicate {label} ids: {', '.join(sorted(duplicates))}.",
                )
            )

    evidence_groups: list[tuple[str, tuple[str, ...]]] = [
        ("objective", bundle.objective.evidence_event_ids),
        ("handover", bundle.handover.evidence_event_ids),
    ]
    evidence_groups.extend(
        (f"episode {item.episode_id}", item.evidence_event_ids) for item in bundle.episodes
    )
    evidence_groups.extend(
        (f"state entry {item.entry_id}", item.evidence_event_ids)
        for item in bundle.state_entries
    )
    for location, evidence_ids in evidence_groups:
        missing = sorted(set(evidence_ids) - known_evidence_ids)
        if missing:
            issues.append(
                CompactionContractIssue(
                    code="unknown_evidence_event",
                    message=f"{location} references unknown evidence: {', '.join(missing)}.",
                    source_event_ids=tuple(missing),
                )
            )

    for item in bundle.preserved_records:
        if item.record_id not in known_evidence_ids:
            issues.append(
                CompactionContractIssue(
                    code="unknown_preserved_evidence",
                    message=f"Preserved record {item.record_id} is absent from the evidence manifest.",
                    source_event_ids=(item.record_id,),
                )
            )
        if item.chronology.generation > bundle.generation:
            issues.append(
                CompactionContractIssue(
                    code="future_chronology",
                    message=f"Preserved record {item.record_id} has future chronology.",
                )
            )
    for item in bundle.episodes:
        unknown_sources = sorted(set(item.source_event_ids) - known_evidence_ids)
        if unknown_sources:
            issues.append(
                CompactionContractIssue(
                    code="unknown_episode_source",
                    message=(
                        f"Episode {item.episode_id} references unknown source events: "
                        f"{', '.join(unknown_sources)}."
                    ),
                    source_event_ids=tuple(unknown_sources),
                )
            )
        if not set(item.evidence_event_ids).issubset(set(item.source_event_ids)):
            issues.append(
                CompactionContractIssue(
                    code="episode_evidence_outside_source",
                    message=f"Episode {item.episode_id} cites evidence outside its source events.",
                )
            )
        if item.chronology.generation > bundle.generation:
            issues.append(
                CompactionContractIssue(
                    code="future_chronology",
                    message=f"Episode {item.episode_id} has future chronology.",
                )
            )
    for group in bundle.coverage:
        unknown_sources = sorted(set(group.source_event_ids) - known_evidence_ids)
        if unknown_sources:
            issues.append(
                CompactionContractIssue(
                    code="unknown_coverage_evidence",
                    message="Coverage references events absent from the evidence manifest.",
                    source_event_ids=tuple(unknown_sources),
                )
            )
        if group.generation > bundle.generation:
            issues.append(
                CompactionContractIssue(
                    code="future_coverage_generation",
                    message="Coverage generation cannot exceed the bundle generation.",
                )
            )

    state_by_id = {entry.entry_id: entry for entry in bundle.state_entries}
    superseded_targets = [
        superseded_id
        for entry in bundle.state_entries
        for superseded_id in entry.supersedes_entry_ids
    ]
    duplicate_targets = _duplicates(superseded_targets)
    if duplicate_targets:
        issues.append(
            CompactionContractIssue(
                code="state_entry_superseded_twice",
                message=(
                    "State entries may have only one direct successor: "
                    + ", ".join(sorted(duplicate_targets))
                    + "."
                ),
            )
        )
    for entry in bundle.state_entries:
        try:
            _validate_state_entry_shape(entry)
        except CompactionContractError as exc:
            issues.extend(exc.issues)
        for superseded_id in entry.supersedes_entry_ids:
            superseded = state_by_id.get(superseded_id)
            if superseded is None:
                issues.append(
                    CompactionContractIssue(
                        code="unknown_superseded_state_entry",
                        message=(
                            f"State entry {entry.entry_id} supersedes unknown entry "
                            f"{superseded_id}."
                        ),
                    )
                )
            elif superseded.status != "superseded":
                issues.append(
                    CompactionContractIssue(
                        code="invalid_supersession_status",
                        message=(
                            f"State entry {superseded_id} must be marked superseded when "
                            f"replaced by {entry.entry_id}."
                        ),
                    )
                )

    if _supersession_has_cycle(state_by_id):
        issues.append(
            CompactionContractIssue(
                code="state_supersession_cycle",
                message="State supersession lineage must be acyclic.",
            )
        )

    if issues:
        raise CompactionContractError(issues)


def compile_compaction_replay(bundle: CompactionBundle) -> tuple[CompactionReplayItem, ...]:
    items: list[CompactionReplayItem] = [
        CompactionReplayItem(
            role="system",
            kind="history_boundary",
            content=_REPLAY_BOUNDARY_TEXT,
            bundle_id=bundle.bundle_id,
            generation=bundle.generation,
        )
    ]
    chronological_items: list[
        tuple[CompactionChronology, int, CompactionReplayItem]
    ] = []
    for preserved in bundle.preserved_records:
        chronological_items.append(
            (
                preserved.chronology,
                0,
                CompactionReplayItem(
                    role=preserved.role,
                    kind="preserved_message",
                    content=preserved.content,
                    bundle_id=bundle.bundle_id,
                    generation=bundle.generation,
                    exact_copy=True,
                    source_record_ids=(preserved.record_id,),
                    evidence_event_ids=(preserved.record_id,),
                ),
            )
        )
    for episode in bundle.episodes:
        content = "Prior-session episode:\n" + episode.summary
        if episode.outcomes:
            content += "\nOutcomes:\n" + "\n".join(f"- {item}" for item in episode.outcomes)
        chronological_items.append(
            (
                episode.chronology,
                1,
                CompactionReplayItem(
                    role="assistant",
                    kind="episode",
                    content=content,
                    bundle_id=bundle.bundle_id,
                    generation=bundle.generation,
                    evidence_event_ids=episode.evidence_event_ids,
                ),
            )
        )
    chronological_items.sort(key=lambda item: (item[0], item[1]))
    items.extend(item for _chronology, _priority, item in chronological_items)

    state_content = _render_state_snapshot(bundle)
    items.append(
        CompactionReplayItem(
            role="assistant",
            kind="state_snapshot",
            content=state_content,
            bundle_id=bundle.bundle_id,
            generation=bundle.generation,
            evidence_event_ids=_ordered_unique(
                event_id
                for entry in bundle.state_entries
                if entry.status == "active"
                for event_id in entry.evidence_event_ids
            ),
        )
    )
    items.append(
        CompactionReplayItem(
            role="assistant",
            kind="handover",
            content=_render_handover(bundle.handover),
            bundle_id=bundle.bundle_id,
            generation=bundle.generation,
            evidence_event_ids=bundle.handover.evidence_event_ids,
        )
    )
    return tuple(items)


def _apply_preserved_actions(
    actions: Sequence[Mapping[str, Any]],
    *,
    previous_bundle: CompactionBundle | None,
    event_by_id: Mapping[str, CompactionSourceEvent],
    known_evidence_ids: set[str],
    delta_event_ids: set[str],
) -> tuple[CompactionPreservedRecord, ...]:
    preserved = {
        item.record_id: item
        for item in (previous_bundle.preserved_records if previous_bundle is not None else ())
    }
    touched: set[str] = set()
    for action in actions:
        _require_exact_keys(
            action,
            {"action", "record_id", "reason", "evidence_event_ids"},
            location="preserved action",
        )
        operation = _required_string(action, "action")
        record_id = _required_string(action, "record_id")
        reason = _required_string(action, "reason")
        evidence_event_ids = _required_string_tuple(action, "evidence_event_ids")
        _validate_evidence_ids(
            evidence_event_ids,
            known_evidence_ids=known_evidence_ids,
            location=f"preserved action {record_id}",
        )
        if record_id in touched:
            raise _single_issue(
                "duplicate_preserved_action",
                f"Preserved record {record_id} has more than one action.",
            )
        touched.add(record_id)
        if operation == "remove":
            if record_id not in preserved:
                raise _single_issue(
                    "unknown_preserved_record",
                    f"Cannot remove unknown preserved record {record_id}.",
                )
            _require_delta_evidence(
                evidence_event_ids,
                delta_event_ids=delta_event_ids,
                location=f"preserved removal {record_id}",
            )
            preserved.pop(record_id)
            continue
        if operation != "add":
            raise _single_issue(
                "invalid_preserved_action",
                f"Preserved action must be add or remove, got {operation!r}.",
            )
        if record_id in preserved:
            raise _single_issue(
                "preserved_record_already_exists",
                f"Preserved record {record_id} already exists and is retained by default.",
            )
        event = event_by_id.get(record_id)
        if event is None:
            raise _single_issue(
                "unknown_preserved_source",
                f"Cannot preserve unknown delta record {record_id}.",
                source_event_ids=(record_id,),
            )
        if event.role not in {"user", "assistant"} or not event.content:
            raise _single_issue(
                "invalid_preserved_source",
                f"Record {record_id} is not a non-empty user or assistant message.",
                source_event_ids=(record_id,),
            )
        if record_id not in evidence_event_ids:
            raise _single_issue(
                "preserved_source_not_cited",
                f"Preserved add action for {record_id} must cite that record as evidence.",
                source_event_ids=(record_id,),
            )
        preserved[record_id] = CompactionPreservedRecord(
            record_id=event.record_id,
            source_session_id=event.session_id,
            created_at=event.created_at,
            role=event.role,  # type: ignore[arg-type]
            content=event.content,
            content_sha256=_sha256_text(event.content),
            reason=reason,
            chronology=event.chronology,
        )
    return tuple(sorted(preserved.values(), key=lambda item: item.chronology))


def _apply_episode_actions(
    actions: Sequence[Mapping[str, Any]],
    *,
    previous_bundle: CompactionBundle | None,
    event_by_id: Mapping[str, CompactionSourceEvent],
    known_evidence_ids: set[str],
    delta_event_ids: set[str],
) -> tuple[CompactionEpisode, ...]:
    episodes = {
        item.episode_id: item
        for item in (previous_bundle.episodes if previous_bundle is not None else ())
    }
    prior_episode_ids = set(episodes)
    consumed_episode_ids: set[str] = set()
    consumed_delta_event_ids: set[str] = set()
    for action in actions:
        _require_exact_keys(
            action,
            {"action", "episode_id", "summary", "source_ids", "outcomes"},
            location="episode action",
        )
        operation = _required_string(action, "action")
        episode_id = _required_identifier(action, "episode_id")
        summary = _required_string(action, "summary")
        source_ids = _required_string_tuple(action, "source_ids")
        outcomes = _required_string_tuple(action, "outcomes", allow_empty=True)
        if episode_id in episodes:
            raise _single_issue(
                "duplicate_episode_id",
                f"Episode id {episode_id} already exists.",
            )

        if operation == "add":
            if not set(source_ids).issubset(delta_event_ids):
                raise _single_issue(
                    "episode_source_not_in_delta",
                    f"New episode {episode_id} may only summarize current delta events.",
                    source_event_ids=source_ids,
                )
            overlapping_ids = sorted(set(source_ids).intersection(consumed_delta_event_ids))
            if overlapping_ids:
                raise _single_issue(
                    "delta_event_in_multiple_episodes",
                    f"Delta events appear in more than one new episode: {', '.join(overlapping_ids)}.",
                    source_event_ids=tuple(overlapping_ids),
                )
            source_sequences = [event_by_id[event_id].sequence for event_id in source_ids]
            if source_sequences != sorted(source_sequences):
                raise _single_issue(
                    "episode_source_order_invalid",
                    f"New episode {episode_id} source events must follow delta chronology.",
                    source_event_ids=source_ids,
                )
            consumed_delta_event_ids.update(source_ids)
            chronology = min(event_by_id[event_id].chronology for event_id in source_ids)
            episode = CompactionEpisode(
                episode_id=episode_id,
                summary=summary,
                outcomes=outcomes,
                source_event_ids=source_ids,
                evidence_event_ids=source_ids,
                chronology=chronology,
            )
        elif operation == "consolidate":
            source_episodes: list[CompactionEpisode] = []
            for source_id in source_ids:
                if source_id not in prior_episode_ids:
                    raise _single_issue(
                        "consolidation_source_not_prior_episode",
                        f"Episode consolidation may only reference prior episode {source_id}.",
                    )
                source_episode = episodes.get(source_id)
                if source_episode is None:
                    raise _single_issue(
                        "unknown_consolidated_episode",
                        f"Cannot consolidate unknown episode {source_id}.",
                    )
                if source_id in consumed_episode_ids:
                    raise _single_issue(
                        "episode_consolidated_twice",
                        f"Episode {source_id} is consolidated more than once.",
                    )
                source_episodes.append(source_episode)
                consumed_episode_ids.add(source_id)
            inherited_source_ids = _ordered_unique(
                event_id for episode in source_episodes for event_id in episode.source_event_ids
            )
            inherited_evidence_ids = _ordered_unique(
                event_id for episode in source_episodes for event_id in episode.evidence_event_ids
            )
            if [episode.chronology for episode in source_episodes] != sorted(
                episode.chronology for episode in source_episodes
            ):
                raise _single_issue(
                    "consolidated_episode_order_invalid",
                    f"Consolidated episode {episode_id} sources must follow chronology.",
                )
            episode = CompactionEpisode(
                episode_id=episode_id,
                summary=summary,
                outcomes=outcomes,
                source_event_ids=inherited_source_ids,
                evidence_event_ids=inherited_evidence_ids,
                chronology=min(episode.chronology for episode in source_episodes),
            )
        else:
            raise _single_issue(
                "invalid_episode_action",
                f"Episode action must be add or consolidate, got {operation!r}.",
            )
        _validate_evidence_ids(
            episode.evidence_event_ids,
            known_evidence_ids=known_evidence_ids,
            location=f"episode {episode_id}",
        )
        episodes[episode_id] = episode

    for consumed_id in consumed_episode_ids:
        episodes.pop(consumed_id, None)
    return tuple(sorted(episodes.values(), key=lambda item: item.chronology))


def _apply_state_operations(
    operations: Sequence[Mapping[str, Any]],
    *,
    previous_bundle: CompactionBundle | None,
    event_by_id: Mapping[str, CompactionSourceEvent],
    known_evidence_ids: set[str],
    delta_event_ids: set[str],
) -> tuple[CompactionStateEntry, ...]:
    state = {
        item.entry_id: item
        for item in (previous_bundle.state_entries if previous_bundle is not None else ())
    }
    touched: set[str] = set()
    for operation in operations:
        expected_keys = {
            "action",
            "entry_id",
            "category",
            "summary",
            "evidence_event_ids",
            "supersedes_entry_id",
            "locator",
            "last_observed_state",
            "needs_verification",
            "blocker",
            "next_action",
        }
        _require_exact_keys(operation, expected_keys, location="state operation")
        action = _required_string(operation, "action")
        entry_id = _required_identifier(operation, "entry_id")
        category = _required_string(operation, "category")
        if category not in _STATE_CATEGORIES:
            raise _single_issue(
                "invalid_state_category",
                f"Unsupported state category: {category!r}.",
            )
        evidence_event_ids = _required_string_tuple(operation, "evidence_event_ids")
        _validate_evidence_ids(
            evidence_event_ids,
            known_evidence_ids=known_evidence_ids,
            location=f"state operation {entry_id}",
        )
        _require_delta_evidence(
            evidence_event_ids,
            delta_event_ids=delta_event_ids,
            location=f"state operation {entry_id}",
        )
        supersedes_entry_id = _optional_string(operation.get("supersedes_entry_id"))
        if entry_id in touched or (supersedes_entry_id is not None and supersedes_entry_id in touched):
            raise _single_issue(
                "duplicate_state_operation",
                f"State entry {entry_id} is changed more than once in one generation.",
            )

        summary = _optional_string(operation.get("summary"))
        locator = _optional_string(operation.get("locator"))
        last_observed_state = _optional_string(operation.get("last_observed_state"))
        blocker = _optional_string(operation.get("blocker"))
        next_action = _optional_string(operation.get("next_action"))
        needs_verification_raw = operation.get("needs_verification")
        if needs_verification_raw is not None and not isinstance(
            needs_verification_raw,
            bool,
        ):
            raise _single_issue(
                "invalid_field_type",
                "needs_verification must be boolean or null.",
            )
        needs_verification = (
            needs_verification_raw if isinstance(needs_verification_raw, bool) else None
        )
        if category == "artifact" and locator is not None and not _literal_appears_in_events(
            locator,
            evidence_event_ids=evidence_event_ids,
            event_by_id=event_by_id,
        ):
            raise _single_issue(
                "artifact_locator_not_in_evidence",
                f"Artifact locator {locator!r} does not appear in its cited delta evidence.",
                source_event_ids=evidence_event_ids,
            )

        if action == "add":
            if entry_id in state:
                raise _single_issue(
                    "state_entry_already_exists",
                    f"Cannot add existing state entry {entry_id}.",
                )
            if summary is None:
                raise _single_issue(
                    "missing_state_summary",
                    f"New state entry {entry_id} requires a summary.",
                )
            entry = CompactionStateEntry(
                entry_id=entry_id,
                category=category,  # type: ignore[arg-type]
                summary=summary,
                status="active",
                evidence_event_ids=evidence_event_ids,
                locator=locator,
                last_observed_state=last_observed_state,
                needs_verification=bool(needs_verification),
                blocker=blocker,
                next_action=next_action,
            )
        elif action in {"update", "resolve"}:
            existing = state.get(entry_id)
            if existing is None:
                raise _single_issue(
                    "unknown_state_entry",
                    f"Cannot {action} unknown state entry {entry_id}.",
                )
            if existing.category != category:
                raise _single_issue(
                    "state_category_changed",
                    f"State entry {entry_id} cannot change category.",
                )
            if existing.status != "active":
                raise _single_issue(
                    "inactive_state_entry_changed",
                    f"Cannot {action} non-active state entry {entry_id}.",
                )
            entry = replace(
                existing,
                summary=summary or existing.summary,
                status=("resolved" if action == "resolve" else existing.status),
                evidence_event_ids=_ordered_unique(
                    (*existing.evidence_event_ids, *evidence_event_ids)
                ),
                locator=locator if locator is not None else existing.locator,
                last_observed_state=(
                    last_observed_state
                    if last_observed_state is not None
                    else existing.last_observed_state
                ),
                needs_verification=(
                    needs_verification
                    if needs_verification is not None
                    else existing.needs_verification
                ),
                blocker=blocker if blocker is not None else existing.blocker,
                next_action=next_action if next_action is not None else existing.next_action,
            )
        elif action == "supersede":
            if entry_id in state:
                raise _single_issue(
                    "state_entry_already_exists",
                    f"Superseding state entry id {entry_id} must be new.",
                )
            if supersedes_entry_id is None or supersedes_entry_id not in state:
                raise _single_issue(
                    "unknown_superseded_state_entry",
                    f"State operation {entry_id} must supersede an existing entry.",
                )
            existing = state[supersedes_entry_id]
            if existing.category != category:
                raise _single_issue(
                    "state_category_changed",
                    f"State entry {entry_id} must use the superseded entry's category.",
                )
            if existing.status != "active":
                raise _single_issue(
                    "inactive_state_entry_superseded",
                    f"Cannot supersede non-active state entry {supersedes_entry_id}.",
                )
            if summary is None:
                raise _single_issue(
                    "missing_state_summary",
                    f"Superseding state entry {entry_id} requires a summary.",
                )
            state[supersedes_entry_id] = replace(existing, status="superseded")
            touched.add(supersedes_entry_id)
            entry = CompactionStateEntry(
                entry_id=entry_id,
                category=category,  # type: ignore[arg-type]
                summary=summary,
                status="active",
                evidence_event_ids=evidence_event_ids,
                supersedes_entry_ids=(supersedes_entry_id,),
                locator=locator,
                last_observed_state=last_observed_state,
                needs_verification=bool(needs_verification),
                blocker=blocker,
                next_action=next_action,
            )
        else:
            raise _single_issue(
                "invalid_state_operation",
                f"State action must be add, update, resolve, or supersede, got {action!r}.",
            )
        _validate_state_entry_shape(entry)
        state[entry_id] = entry
        touched.add(entry_id)
    return tuple(sorted(state.values(), key=lambda item: (item.category, item.entry_id)))


def _coverage_from_draft(
    payload: Mapping[str, Any],
    *,
    generation: int,
) -> CompactionCoverageGroup:
    _require_exact_keys(
        payload,
        {"source_event_ids", "disposition", "target_ids", "reason"},
        location="coverage group",
    )
    disposition = _required_string(payload, "disposition")
    if disposition not in _COVERAGE_DISPOSITIONS:
        raise _single_issue(
            "invalid_coverage_disposition",
            f"Unsupported coverage disposition: {disposition!r}.",
        )
    target_ids = _required_string_tuple(payload, "target_ids", allow_empty=True)
    if disposition == "omitted" and target_ids:
        raise _single_issue(
            "omitted_coverage_has_targets",
            "Omitted coverage groups must not have target ids.",
        )
    if disposition != "omitted" and not target_ids:
        raise _single_issue(
            "represented_coverage_missing_targets",
            "Non-omitted coverage groups require target ids.",
        )
    return CompactionCoverageGroup(
        generation=generation,
        source_event_ids=_required_string_tuple(payload, "source_event_ids"),
        disposition=disposition,  # type: ignore[arg-type]
        target_ids=target_ids,
        reason=_required_string(payload, "reason"),
    )


def _validate_delta_coverage(
    bundle: CompactionBundle,
    *,
    source_events: Sequence[CompactionSourceEvent],
    new_coverage: Sequence[CompactionCoverageGroup],
) -> None:
    issues: list[CompactionContractIssue] = []
    expected_ids = {event.event_id for event in source_events}
    covered_ids = [event_id for group in new_coverage for event_id in group.source_event_ids]
    missing = sorted(expected_ids - set(covered_ids))
    unknown = sorted(set(covered_ids) - expected_ids)
    duplicate = sorted(_duplicates(covered_ids))
    if missing:
        issues.append(
            CompactionContractIssue(
                code="missing_source_coverage",
                message=f"Delta events lack coverage: {', '.join(missing)}.",
                source_event_ids=tuple(missing),
            )
        )
    if unknown:
        issues.append(
            CompactionContractIssue(
                code="unknown_coverage_source",
                message=f"Coverage references unknown delta events: {', '.join(unknown)}.",
                source_event_ids=tuple(unknown),
            )
        )
    if duplicate:
        issues.append(
            CompactionContractIssue(
                code="duplicate_source_coverage",
                message=f"Delta events are covered more than once: {', '.join(duplicate)}.",
                source_event_ids=tuple(duplicate),
            )
        )

    valid_targets = {
        "preserved": {item.record_id for item in bundle.preserved_records},
        "episode": {item.episode_id for item in bundle.episodes},
        "state": {item.entry_id for item in bundle.state_entries},
        "objective": {"objective"},
        "handover": {"handover"},
        "omitted": set(),
    }
    source_by_id = {event.event_id: event for event in source_events}
    for group in new_coverage:
        bad_targets = sorted(set(group.target_ids) - valid_targets[group.disposition])
        if bad_targets:
            issues.append(
                CompactionContractIssue(
                    code="unknown_coverage_target",
                    message=(
                        f"Coverage group {group.disposition} references unknown targets: "
                        f"{', '.join(bad_targets)}."
                    ),
                    source_event_ids=group.source_event_ids,
                )
            )
        if group.disposition == "omitted":
            omitted_user_events = [
                event_id
                for event_id in group.source_event_ids
                if source_by_id.get(event_id) is not None
                and source_by_id[event_id].role == "user"
            ]
            if omitted_user_events:
                issues.append(
                    CompactionContractIssue(
                        code="user_event_omitted",
                        message="User-authored delta events cannot be omitted from compaction.",
                        source_event_ids=tuple(omitted_user_events),
                    )
                )
    if issues:
        raise CompactionContractError(issues)


def _validate_state_entry_shape(entry: CompactionStateEntry) -> None:
    issues: list[CompactionContractIssue] = []
    if entry.category == "artifact" and (
        entry.locator is None or entry.last_observed_state is None
    ):
        issues.append(
            CompactionContractIssue(
                code="incomplete_artifact_state",
                message=(
                    f"Artifact state entry {entry.entry_id} requires locator and "
                    "last_observed_state."
                ),
            )
        )
    if entry.category == "open_loop" and entry.next_action is None:
        issues.append(
            CompactionContractIssue(
                code="incomplete_open_loop",
                message=f"Open-loop state entry {entry.entry_id} requires next_action.",
            )
        )
    if issues:
        raise CompactionContractError(issues)


def _validate_evidence_ids(
    evidence_ids: Sequence[str],
    *,
    known_evidence_ids: set[str],
    location: str,
) -> None:
    missing = sorted(set(evidence_ids) - known_evidence_ids)
    if missing:
        raise _single_issue(
            "unknown_evidence_event",
            f"{location} references unknown evidence: {', '.join(missing)}.",
            source_event_ids=tuple(missing),
        )


def _literal_appears_in_events(
    literal: str,
    *,
    evidence_event_ids: Sequence[str],
    event_by_id: Mapping[str, CompactionSourceEvent],
) -> bool:
    for event_id in evidence_event_ids:
        event = event_by_id.get(event_id)
        if event is None:
            continue
        if literal in event.content:
            return True
        if event.metadata and literal in _canonical_json(event.metadata):
            return True
    return False


def _require_delta_evidence(
    evidence_ids: Sequence[str],
    *,
    delta_event_ids: set[str],
    location: str,
) -> None:
    if not set(evidence_ids).intersection(delta_event_ids):
        raise _single_issue(
            "missing_delta_evidence",
            f"{location} must cite at least one current delta event.",
        )


def _render_state_snapshot(bundle: CompactionBundle) -> str:
    lines = ["Canonical task state:", f"Objective: {bundle.objective.summary}"]
    headings = (
        ("constraint", "Active constraints"),
        ("decision", "Active decisions"),
        ("artifact", "Artifacts — last observed state"),
        ("open_loop", "Open work"),
        ("uncertainty", "Uncertainties"),
    )
    for category, heading in headings:
        entries = [
            entry
            for entry in bundle.state_entries
            if entry.category == category and entry.status == "active"
        ]
        if not entries:
            continue
        lines.append(f"{heading}:")
        for entry in entries:
            detail = entry.summary
            if entry.category == "artifact":
                detail = f"{entry.locator}: {entry.last_observed_state} — {entry.summary}"
                if entry.needs_verification:
                    detail += " [needs verification]"
            elif entry.category == "open_loop":
                detail += f" Next: {entry.next_action}"
                if entry.blocker:
                    detail += f" Blocker: {entry.blocker}"
            lines.append(f"- [{entry.entry_id}] {detail}")
    return "\n".join(lines)


def _render_handover(handover: CompactionHandover) -> str:
    lines = ["Continuation handover:", f"Current focus: {handover.current_focus}"]
    for heading, values in (
        ("Next actions", handover.next_actions),
        ("Do not repeat", handover.do_not_repeat),
        ("Verification needed", handover.verification_needed),
    ):
        if values:
            lines.append(f"{heading}:")
            lines.extend(f"- {value}" for value in values)
    return "\n".join(lines)


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise _single_issue("invalid_field_type", f"{key} must be an object.")
    return value


def _required_mapping_list(payload: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise _single_issue("invalid_field_type", f"{key} must be an array.")
    items: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise _single_issue(
                "invalid_field_type",
                f"{key}[{index}] must be an object.",
            )
        items.append(item)
    return tuple(items)


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _single_issue("invalid_field_type", f"{key} must be a non-empty string.")
    return value.strip()


def _required_identifier(payload: Mapping[str, Any], key: str) -> str:
    value = _required_string(payload, key)
    if len(value) > 128 or any(
        not (character.isalnum() or character in {"_", "-", ".", ":"})
        for character in value
    ):
        raise _single_issue(
            "invalid_identifier",
            f"{key} must use only letters, digits, _, -, ., or : and be at most 128 chars.",
        )
    return value


def _required_positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _single_issue("invalid_field_type", f"{key} must be a positive integer.")
    return value


def _required_string_tuple(
    payload: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise _single_issue("invalid_field_type", f"{key} must be an array of strings.")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise _single_issue(
                "invalid_field_type",
                f"{key}[{index}] must be a non-empty string.",
            )
        result.append(item.strip())
    if not result and not allow_empty:
        raise _single_issue("empty_field", f"{key} must not be empty.")
    if len(result) != len(set(result)):
        raise _single_issue("duplicate_field_value", f"{key} must not contain duplicates.")
    return tuple(result)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _single_issue("invalid_field_type", "Optional string field must be string or null.")
    normalized = value.strip()
    return normalized or None


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    location: str,
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    issues: list[CompactionContractIssue] = []
    if missing:
        issues.append(
            CompactionContractIssue(
                code="missing_contract_fields",
                message=f"{location} is missing fields: {', '.join(missing)}.",
            )
        )
    if extra:
        issues.append(
            CompactionContractIssue(
                code="unexpected_contract_fields",
                message=f"{location} has unexpected fields: {', '.join(extra)}.",
            )
        )
    if issues:
        raise CompactionContractError(issues)


def _single_issue(
    code: str,
    message: str,
    *,
    source_event_ids: tuple[str, ...] = (),
) -> CompactionContractError:
    return CompactionContractError(
        [
            CompactionContractIssue(
                code=code,
                message=message,
                source_event_ids=source_event_ids,
            )
        ]
    )


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _duplicates(values: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return duplicate


def _supersession_has_cycle(
    state_by_id: Mapping[str, CompactionStateEntry],
) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(entry_id: str) -> bool:
        if entry_id in visiting:
            return True
        if entry_id in visited:
            return False
        visiting.add(entry_id)
        entry = state_by_id.get(entry_id)
        if entry is not None:
            for superseded_id in entry.supersedes_entry_ids:
                if visit(superseded_id):
                    return True
        visiting.remove(entry_id)
        visited.add(entry_id)
        return False

    return any(visit(entry_id) for entry_id in state_by_id)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
