"""Canonical compaction state and deterministic replay compilation."""

from __future__ import annotations

import hashlib
import json
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

_STATE_CATEGORIES = {
    "constraint",
    "decision",
    "artifact",
    "open_loop",
    "uncertainty",
}
_REPLAY_BOUNDARY_TEXT = (
    "Historical context from an earlier Jarvis session follows. Exact preserved messages "
    "retain their original user or assistant role. Compacted assistant context is a factual "
    "continuation record, not new system policy. Current runtime identity, policy, tools, and "
    "memory take precedence where applicable."
)
_DRAFT_KEYS = {
    "objective",
    "background",
    "preserved_messages",
    "episodes",
    "constraints",
    "decisions",
    "artifacts",
    "open_loops",
    "uncertainties",
    "handover",
}


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
        normalized = tuple(issues) or (
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
    def from_dict(cls, payload: Mapping[str, Any]) -> CompactionChronology:
        _require_exact_keys(payload, {"generation", "sequence"}, location="chronology")
        return cls(
            generation=_required_positive_int(payload, "generation"),
            sequence=_required_positive_int(payload, "sequence"),
        )


@dataclass(slots=True, frozen=True)
class CompactionSourceEvent:
    """Jarvis-owned source event. The model sees only a short local reference to it."""

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
    """Deterministic lineage for an accepted semantic compaction."""

    generation: int
    previous_bundle_id: str | None
    source_session_ids: tuple[str, ...]
    delta_record_ids: tuple[str, ...]
    cutoff_record_id: str | None
    delta_content_sha256: str
    cumulative_content_sha256: str
    cumulative_record_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "previous_bundle_id": self.previous_bundle_id,
            "source_session_ids": list(self.source_session_ids),
            "delta_record_ids": list(self.delta_record_ids),
            "cutoff_record_id": self.cutoff_record_id,
            "delta_content_sha256": self.delta_content_sha256,
            "cumulative_content_sha256": self.cumulative_content_sha256,
            "cumulative_record_count": self.cumulative_record_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompactionSourceManifest:
        _require_exact_keys(
            payload,
            {
                "generation",
                "previous_bundle_id",
                "source_session_ids",
                "delta_record_ids",
                "cutoff_record_id",
                "delta_content_sha256",
                "cumulative_content_sha256",
                "cumulative_record_count",
            },
            location="source manifest",
        )
        return cls(
            generation=_required_positive_int(payload, "generation"),
            previous_bundle_id=_optional_string(payload.get("previous_bundle_id")),
            source_session_ids=_required_string_tuple(payload, "source_session_ids"),
            delta_record_ids=_required_string_tuple(
                payload,
                "delta_record_ids",
                allow_empty=True,
            ),
            cutoff_record_id=_optional_string(payload.get("cutoff_record_id")),
            delta_content_sha256=_required_string(payload, "delta_content_sha256"),
            cumulative_content_sha256=_required_string(
                payload,
                "cumulative_content_sha256",
            ),
            cumulative_record_count=_required_nonnegative_int(
                payload,
                "cumulative_record_count",
            ),
        )


@dataclass(slots=True, frozen=True)
class CompactionObjective:
    summary: str

    def to_dict(self) -> dict[str, str]:
        return {"summary": self.summary}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompactionObjective:
        _require_exact_keys(payload, {"summary"}, location="objective")
        return cls(summary=_required_string(payload, "summary"))


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
    def from_dict(cls, payload: Mapping[str, Any]) -> CompactionPreservedRecord:
        _require_exact_keys(
            payload,
            {
                "record_id",
                "source_session_id",
                "created_at",
                "role",
                "content",
                "content_sha256",
                "reason",
                "chronology",
            },
            location="preserved record",
        )
        role = _required_string(payload, "role")
        if role not in {"user", "assistant"}:
            raise _single_issue(
                "invalid_preserved_role",
                f"Preserved record role must be user or assistant, got {role!r}.",
            )
        content = _required_string(payload, "content", strip=False)
        content_hash = _required_string(payload, "content_sha256")
        if _sha256_text(content) != content_hash:
            raise _single_issue(
                "preserved_content_hash_mismatch",
                "Preserved record content does not match its stored hash.",
            )
        return cls(
            record_id=_required_string(payload, "record_id"),
            source_session_id=_required_string(payload, "source_session_id"),
            created_at=_required_string(payload, "created_at"),
            role=role,  # type: ignore[arg-type]
            content=content,
            content_sha256=content_hash,
            reason=_required_string(payload, "reason"),
            chronology=CompactionChronology.from_dict(
                _required_mapping(payload, "chronology")
            ),
        )


@dataclass(slots=True, frozen=True)
class CompactionEpisode:
    episode_id: str
    summary: str
    outcomes: tuple[str, ...]
    chronology: CompactionChronology

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "summary": self.summary,
            "outcomes": list(self.outcomes),
            "chronology": self.chronology.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompactionEpisode:
        _require_exact_keys(
            payload,
            {"episode_id", "summary", "outcomes", "chronology"},
            location="episode",
        )
        return cls(
            episode_id=_required_identifier(payload, "episode_id"),
            summary=_required_string(payload, "summary"),
            outcomes=_required_string_tuple(payload, "outcomes", allow_empty=True),
            chronology=CompactionChronology.from_dict(
                _required_mapping(payload, "chronology")
            ),
        )


@dataclass(slots=True, frozen=True)
class CompactionStateEntry:
    entry_id: str
    category: CompactionStateCategory
    summary: str
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
            "locator": self.locator,
            "last_observed_state": self.last_observed_state,
            "needs_verification": self.needs_verification,
            "blocker": self.blocker,
            "next_action": self.next_action,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompactionStateEntry:
        _require_exact_keys(
            payload,
            {
                "entry_id",
                "category",
                "summary",
                "locator",
                "last_observed_state",
                "needs_verification",
                "blocker",
                "next_action",
            },
            location="state entry",
        )
        category = _required_string(payload, "category")
        if category not in _STATE_CATEGORIES:
            raise _single_issue(
                "invalid_state_category",
                f"Unsupported compaction state category: {category!r}.",
            )
        needs_verification = payload.get("needs_verification")
        if not isinstance(needs_verification, bool):
            raise _single_issue(
                "invalid_field_type",
                "State entry needs_verification must be a boolean.",
            )
        entry = cls(
            entry_id=_required_identifier(payload, "entry_id"),
            category=category,  # type: ignore[arg-type]
            summary=_required_string(payload, "summary"),
            locator=_optional_string(payload.get("locator")),
            last_observed_state=_optional_string(payload.get("last_observed_state")),
            needs_verification=needs_verification,
            blocker=_optional_string(payload.get("blocker")),
            next_action=_optional_string(payload.get("next_action")),
        )
        _validate_state_entry(entry)
        return entry


@dataclass(slots=True, frozen=True)
class CompactionHandover:
    current_focus: str
    next_actions: tuple[str, ...]
    do_not_repeat: tuple[str, ...]
    verification_needed: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_focus": self.current_focus,
            "next_actions": list(self.next_actions),
            "do_not_repeat": list(self.do_not_repeat),
            "verification_needed": list(self.verification_needed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompactionHandover:
        _require_exact_keys(
            payload,
            {
                "current_focus",
                "next_actions",
                "do_not_repeat",
                "verification_needed",
            },
            location="handover",
        )
        return cls(
            current_focus=_required_string(payload, "current_focus"),
            next_actions=_required_string_tuple(payload, "next_actions", allow_empty=True),
            do_not_repeat=_required_string_tuple(
                payload,
                "do_not_repeat",
                allow_empty=True,
            ),
            verification_needed=_required_string_tuple(
                payload,
                "verification_needed",
                allow_empty=True,
            ),
        )


@dataclass(slots=True, frozen=True)
class CompactionBundle:
    schema_version: int
    bundle_id: str
    created_at: str
    source_manifest: CompactionSourceManifest
    objective: CompactionObjective
    background: tuple[str, ...]
    preserved_records: tuple[CompactionPreservedRecord, ...]
    episodes: tuple[CompactionEpisode, ...]
    state_entries: tuple[CompactionStateEntry, ...]
    handover: CompactionHandover

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
            "background": list(self.background),
            "preserved_records": [item.to_dict() for item in self.preserved_records],
            "episodes": [item.to_dict() for item in self.episodes],
            "state_entries": [item.to_dict() for item in self.state_entries],
            "handover": self.handover.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompactionBundle:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "bundle_id",
                "created_at",
                "source_manifest",
                "objective",
                "background",
                "preserved_records",
                "episodes",
                "state_entries",
                "handover",
            },
            location="compaction bundle",
        )
        schema_version = _required_positive_int(payload, "schema_version")
        if schema_version != 3:
            raise _single_issue(
                "unsupported_compaction_schema",
                f"Expected compaction schema version 3, got {schema_version}.",
            )
        bundle = cls(
            schema_version=schema_version,
            bundle_id=_required_string(payload, "bundle_id"),
            created_at=_required_string(payload, "created_at"),
            source_manifest=CompactionSourceManifest.from_dict(
                _required_mapping(payload, "source_manifest")
            ),
            objective=CompactionObjective.from_dict(
                _required_mapping(payload, "objective")
            ),
            background=_required_string_tuple(payload, "background", allow_empty=True),
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
            handover=CompactionHandover.from_dict(
                _required_mapping(payload, "handover")
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "kind": self.kind,
            "content": self.content,
            "bundle_id": self.bundle_id,
            "generation": self.generation,
            "exact_copy": self.exact_copy,
            "source_record_ids": list(self.source_record_ids),
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
        }


def build_source_manifest(
    *,
    generation: int,
    previous_bundle: CompactionBundle | None,
    source_events: Sequence[CompactionSourceEvent],
) -> CompactionSourceManifest:
    serialized_events = "\n".join(_canonical_json(event.to_dict()) for event in source_events)
    delta_hash = _sha256_text(serialized_events)
    previous_manifest = previous_bundle.source_manifest if previous_bundle is not None else None
    if previous_manifest is not None and not source_events:
        cumulative_hash = previous_manifest.cumulative_content_sha256
    elif previous_manifest is not None:
        cumulative_hash = _sha256_text(
            previous_manifest.cumulative_content_sha256 + "\n" + delta_hash
        )
    else:
        cumulative_hash = delta_hash
    previous_sessions = previous_manifest.source_session_ids if previous_manifest else ()
    source_sessions = _ordered_unique(
        (*previous_sessions, *(event.session_id for event in source_events))
    )
    cutoff = (
        source_events[-1].record_id
        if source_events
        else (previous_manifest.cutoff_record_id if previous_manifest else None)
    )
    return CompactionSourceManifest(
        generation=generation,
        previous_bundle_id=previous_bundle.bundle_id if previous_bundle is not None else None,
        source_session_ids=source_sessions,
        delta_record_ids=tuple(event.record_id for event in source_events),
        cutoff_record_id=cutoff,
        delta_content_sha256=delta_hash,
        cumulative_content_sha256=cumulative_hash,
        cumulative_record_count=(
            (previous_manifest.cumulative_record_count if previous_manifest else 0)
            + len(source_events)
        ),
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
    """Turn the model's semantic record into Jarvis-owned canonical state."""

    _require_exact_keys(payload, _DRAFT_KEYS, location="compaction submission")
    objective = CompactionObjective(summary=_required_string(payload, "objective"))
    background = _ordered_unique(
        _required_string_tuple(payload, "background", allow_empty=True)
    )
    preserved = _preserved_records_from_draft(
        _required_mapping_list(payload, "preserved_messages"),
        source_events=source_events,
        previous_bundle=previous_bundle,
    )
    episodes = _episodes_from_draft(
        _required_mapping_list(payload, "episodes"),
        generation=source_manifest.generation,
    )
    state_entries = _state_entries_from_draft(payload)
    handover = CompactionHandover.from_dict(_required_mapping(payload, "handover"))
    bundle = CompactionBundle(
        schema_version=3,
        bundle_id=bundle_id,
        created_at=created_at,
        source_manifest=source_manifest,
        objective=objective,
        background=background,
        preserved_records=preserved,
        episodes=episodes,
        state_entries=state_entries,
        handover=handover,
    )
    validate_compaction_bundle(bundle)
    return bundle


def validate_compaction_bundle(bundle: CompactionBundle) -> None:
    issues: list[CompactionContractIssue] = []
    if bundle.schema_version != 3:
        issues.append(
            CompactionContractIssue(
                code="unsupported_compaction_schema",
                message=f"Expected compaction schema version 3, got {bundle.schema_version}.",
            )
        )
    if bundle.source_manifest.generation <= 0:
        issues.append(
            CompactionContractIssue(
                code="invalid_generation",
                message="Compaction generation must be positive.",
            )
        )
    for label, values in (
        ("preserved record", [item.record_id for item in bundle.preserved_records]),
        ("episode", [item.episode_id for item in bundle.episodes]),
        ("state entry", [item.entry_id for item in bundle.state_entries]),
    ):
        duplicates = _duplicates(values)
        if duplicates:
            issues.append(
                CompactionContractIssue(
                    code=f"duplicate_{label.replace(' ', '_')}_id",
                    message=f"Duplicate {label} ids: {', '.join(sorted(duplicates))}.",
                )
            )
    for record in bundle.preserved_records:
        if _sha256_text(record.content) != record.content_sha256:
            issues.append(
                CompactionContractIssue(
                    code="preserved_content_hash_mismatch",
                    message=f"Preserved record {record.record_id} does not match its hash.",
                )
            )
    for entry in bundle.state_entries:
        try:
            _validate_state_entry(entry)
        except CompactionContractError as exc:
            issues.extend(exc.issues)
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
    for preserved in sorted(
        bundle.preserved_records,
        key=lambda item: item.chronology,
    ):
        items.append(
            CompactionReplayItem(
                role=preserved.role,
                kind="preserved_message",
                content=preserved.content,
                bundle_id=bundle.bundle_id,
                generation=bundle.generation,
                exact_copy=True,
                source_record_ids=(preserved.record_id,),
            )
        )
    for episode in bundle.episodes:
        content = "Prior-session episode:\n" + episode.summary
        if episode.outcomes:
            content += "\nOutcomes:\n" + "\n".join(
                f"- {outcome}" for outcome in episode.outcomes
            )
        items.append(
            CompactionReplayItem(
                role="assistant",
                kind="episode",
                content=content,
                bundle_id=bundle.bundle_id,
                generation=bundle.generation,
            )
        )
    items.extend(
        (
            CompactionReplayItem(
                role="assistant",
                kind="state_snapshot",
                content=_render_state_snapshot(bundle),
                bundle_id=bundle.bundle_id,
                generation=bundle.generation,
            ),
            CompactionReplayItem(
                role="assistant",
                kind="handover",
                content=_render_handover(bundle.handover),
                bundle_id=bundle.bundle_id,
                generation=bundle.generation,
            ),
        )
    )
    return tuple(items)


def _preserved_records_from_draft(
    items: Sequence[Mapping[str, Any]],
    *,
    source_events: Sequence[CompactionSourceEvent],
    previous_bundle: CompactionBundle | None,
) -> tuple[CompactionPreservedRecord, ...]:
    source_refs: dict[str, CompactionSourceEvent | CompactionPreservedRecord] = {
        f"E{index}": event for index, event in enumerate(source_events, start=1)
    }
    if previous_bundle is not None:
        source_refs.update(
            {
                f"P{index}": record
                for index, record in enumerate(previous_bundle.preserved_records, start=1)
            }
        )
    preserved: list[CompactionPreservedRecord] = []
    seen_record_ids: set[str] = set()
    for item in items:
        _require_exact_keys(
            item,
            {"source_ref", "reason"},
            location="preserved message selection",
        )
        source_ref = _required_string(item, "source_ref")
        reason = _required_string(item, "reason")
        source = source_refs.get(source_ref)
        if source is None:
            raise _single_issue(
                "unknown_preserved_source_ref",
                f"Preserved message references unknown source {source_ref!r}.",
            )
        if isinstance(source, CompactionPreservedRecord):
            record = replace(source, reason=reason)
        else:
            if source.role not in {"user", "assistant"} or not source.content.strip():
                raise _single_issue(
                    "invalid_preserved_source",
                    f"{source_ref} is not a non-empty user or assistant message.",
                    source_event_ids=(source.event_id,),
                )
            record = CompactionPreservedRecord(
                record_id=source.record_id,
                source_session_id=source.session_id,
                created_at=source.created_at,
                role=source.role,  # type: ignore[arg-type]
                content=source.content,
                content_sha256=_sha256_text(source.content),
                reason=reason,
                chronology=source.chronology,
            )
        if record.record_id in seen_record_ids:
            continue
        seen_record_ids.add(record.record_id)
        preserved.append(record)
    return tuple(sorted(preserved, key=lambda item: item.chronology))


def _episodes_from_draft(
    items: Sequence[Mapping[str, Any]],
    *,
    generation: int,
) -> tuple[CompactionEpisode, ...]:
    episodes: list[CompactionEpisode] = []
    seen: set[str] = set()
    for sequence, item in enumerate(items, start=1):
        _require_exact_keys(item, {"summary", "outcomes"}, location="episode")
        summary = _required_string(item, "summary")
        outcomes = _ordered_unique(
            _required_string_tuple(item, "outcomes", allow_empty=True)
        )
        fingerprint = _canonical_json({"summary": summary, "outcomes": outcomes})
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        episodes.append(
            CompactionEpisode(
                episode_id=_stable_id("episode", fingerprint),
                summary=summary,
                outcomes=outcomes,
                chronology=CompactionChronology(generation=generation, sequence=sequence),
            )
        )
    return tuple(episodes)


def _state_entries_from_draft(payload: Mapping[str, Any]) -> tuple[CompactionStateEntry, ...]:
    entries: list[CompactionStateEntry] = []
    for category, key in (
        ("constraint", "constraints"),
        ("decision", "decisions"),
        ("uncertainty", "uncertainties"),
    ):
        for summary in _ordered_unique(
            _required_string_tuple(payload, key, allow_empty=True)
        ):
            entries.append(
                CompactionStateEntry(
                    entry_id=_stable_id(category, summary),
                    category=category,  # type: ignore[arg-type]
                    summary=summary,
                )
            )
    for item in _required_mapping_list(payload, "artifacts"):
        _require_exact_keys(
            item,
            {"summary", "locator", "last_observed_state", "needs_verification"},
            location="artifact",
        )
        summary = _required_string(item, "summary")
        locator = _required_string(item, "locator")
        last_observed_state = _required_string(item, "last_observed_state")
        needs_verification = item.get("needs_verification")
        if not isinstance(needs_verification, bool):
            raise _single_issue(
                "invalid_field_type",
                "Artifact needs_verification must be a boolean.",
            )
        fingerprint = _canonical_json(
            {
                "summary": summary,
                "locator": locator,
                "last_observed_state": last_observed_state,
                "needs_verification": needs_verification,
            }
        )
        entries.append(
            CompactionStateEntry(
                entry_id=_stable_id("artifact", fingerprint),
                category="artifact",
                summary=summary,
                locator=locator,
                last_observed_state=last_observed_state,
                needs_verification=needs_verification,
            )
        )
    for item in _required_mapping_list(payload, "open_loops"):
        _require_exact_keys(
            item,
            {"summary", "next_action", "blocker"},
            location="open loop",
        )
        summary = _required_string(item, "summary")
        next_action = _required_string(item, "next_action")
        blocker = _optional_string(item.get("blocker"))
        fingerprint = _canonical_json(
            {"summary": summary, "next_action": next_action, "blocker": blocker}
        )
        entries.append(
            CompactionStateEntry(
                entry_id=_stable_id("open_loop", fingerprint),
                category="open_loop",
                summary=summary,
                blocker=blocker,
                next_action=next_action,
            )
        )
    unique: dict[tuple[Any, ...], CompactionStateEntry] = {}
    for entry in entries:
        identity = (
            entry.category,
            entry.summary,
            entry.locator,
            entry.last_observed_state,
            entry.needs_verification,
            entry.blocker,
            entry.next_action,
        )
        unique.setdefault(identity, entry)
    return tuple(unique.values())


def _validate_state_entry(entry: CompactionStateEntry) -> None:
    if entry.category == "artifact" and (
        entry.locator is None or entry.last_observed_state is None
    ):
        raise _single_issue(
            "incomplete_artifact_state",
            f"Artifact state entry {entry.entry_id} requires a locator and observed state.",
        )
    if entry.category == "open_loop" and entry.next_action is None:
        raise _single_issue(
            "incomplete_open_loop",
            f"Open-loop state entry {entry.entry_id} requires a next action.",
        )


def _render_state_snapshot(bundle: CompactionBundle) -> str:
    lines = ["Canonical task state:", f"Objective: {bundle.objective.summary}"]
    if bundle.background:
        lines.append("Background:")
        lines.extend(f"- {item}" for item in bundle.background)
    headings = {
        "constraint": "Active constraints",
        "decision": "Decisions",
        "artifact": "Artifacts",
        "open_loop": "Open work",
        "uncertainty": "Uncertainties",
    }
    for category in ("constraint", "decision", "artifact", "open_loop", "uncertainty"):
        entries = [item for item in bundle.state_entries if item.category == category]
        if not entries:
            continue
        lines.append(headings[category] + ":")
        for entry in entries:
            detail = entry.summary
            if entry.locator is not None:
                detail += f" [locator: {entry.locator}]"
            if entry.last_observed_state is not None:
                detail += f" [last observed: {entry.last_observed_state}]"
            if entry.needs_verification:
                detail += " [fresh verification required]"
            if entry.blocker is not None:
                detail += f" [blocker: {entry.blocker}]"
            if entry.next_action is not None:
                detail += f" [next: {entry.next_action}]"
            lines.append(f"- {detail}")
    return "\n".join(lines)


def _render_handover(handover: CompactionHandover) -> str:
    lines = ["Continuation handover:", f"Current focus: {handover.current_focus}"]
    for heading, values in (
        ("Next actions", handover.next_actions),
        ("Do not repeat", handover.do_not_repeat),
        ("Verification needed", handover.verification_needed),
    ):
        if values:
            lines.append(heading + ":")
            lines.extend(f"- {value}" for value in values)
    return "\n".join(lines)


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise _single_issue("invalid_field_type", f"{key} must be an object.")
    return value


def _required_mapping_list(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise _single_issue("invalid_field_type", f"{key} must be an array of objects.")
    return tuple(value)


def _required_string(
    payload: Mapping[str, Any],
    key: str,
    *,
    strip: bool = True,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise _single_issue("invalid_field_type", f"{key} must be a string.")
    normalized = value.strip() if strip else value
    if not normalized or (not strip and not normalized.strip()):
        raise _single_issue("empty_field", f"{key} must not be empty.")
    return normalized


def _required_identifier(payload: Mapping[str, Any], key: str) -> str:
    value = _required_string(payload, key)
    if any(not (character.isalnum() or character in "_-.:" ) for character in value):
        raise _single_issue(
            "invalid_identifier",
            f"{key} contains unsupported identifier characters: {value!r}.",
        )
    return value


def _required_positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _single_issue("invalid_field_type", f"{key} must be a positive integer.")
    return value


def _required_nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _single_issue(
            "invalid_field_type",
            f"{key} must be a non-negative integer.",
        )
    return value


def _required_string_tuple(
    payload: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)):
        raise _single_issue("invalid_field_type", f"{key} must be an array of strings.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise _single_issue(
                "invalid_field_type",
                f"Every item in {key} must be a non-empty string.",
            )
        normalized.append(item.strip())
    if not allow_empty and not normalized:
        raise _single_issue("empty_field", f"{key} must not be empty.")
    return tuple(normalized)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _single_issue("invalid_field_type", "Optional text values must be strings or null.")
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
    if not missing and not extra:
        return
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if extra:
        details.append("unexpected " + ", ".join(extra))
    raise _single_issue(
        "invalid_field_set",
        f"{location} has an invalid field set ({'; '.join(details)}).",
    )


def _single_issue(
    code: str,
    message: str,
    *,
    source_event_ids: tuple[str, ...] = (),
) -> CompactionContractError:
    return CompactionContractError(
        (
            CompactionContractIssue(
                code=code,
                message=message,
                source_event_ids=source_event_ids,
            ),
        )
    )


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _duplicates(values: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{_sha256_text(value)[:16]}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
