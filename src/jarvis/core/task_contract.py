"""Deterministic task contracts derived from explicit user requirements."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any


_REQUIREMENT_MARKER = re.compile(
    r"\b(?:must|should|required?|requirement|want you to|i want|do not|don't|never|"
    r"at least|at most|exactly|until|make use of|ensure|all tests?|lint|typecheck|build|"
    r"visually?|live inspection|finished|complete|fully|properly)\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_RESUME_PATTERN = re.compile(
    r"^\s*(?:please\s+)?(?:continue|resume|keep\s+going|carry\s+on|proceed|retry|try\s+again)\b",
    re.IGNORECASE,
)
_REQUIREMENT_REPLACEMENT_PATTERN = re.compile(
    r"\b(?:waive|drop|remove)\b.*\brequirement\b|\b(?:no longer|do not|don't|no)\s+need\b",
    re.IGNORECASE,
)
_TASK_REPLACEMENT_PATTERN = re.compile(
    r"\b(?:forget|cancel|stop|discard)\b.*\b(?:previous|prior|current|old)\b.*\btask\b|"
    r"^\s*(?:new task|instead[, :]|start over\b)",
    re.IGNORECASE,
)
_SIDE_QUERY_PATTERN = re.compile(
    r"^\s*(?:please\s+)?(?:give\s+me\s+)?(?:a\s+)?(?:quick\s+)?(?:status|progress)"
    r"(?:\s+update|\s+report)?\b|"
    r"^\s*(?:what\s+time\s+is\s+it\??\s*$|"
    r"what(?:'s|\s+is)\s+the\s+(?:current\s+)?(?:status|progress|time)|"
    r"how\s+(?:is|are)\b.*\bgoing\??\s*$)",
    re.IGNORECASE,
)
_MAX_REQUIREMENTS = 64
_MAX_CRITERION_CHARS = 1_000
_WITHDRAWAL_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "i",
        "it",
        "mean",
        "need",
        "no",
        "requirement",
        "requirements",
        "the",
        "to",
    }
)


@dataclass(slots=True, frozen=True)
class TaskRequirement:
    item_id: str
    criterion: str
    evidence_kind: str

    def to_state(self) -> dict[str, str]:
        return {
            "item_id": self.item_id,
            "criterion": self.criterion,
            "evidence_kind": self.evidence_kind,
        }

    @classmethod
    def from_state(cls, value: object) -> "TaskRequirement | None":
        if not isinstance(value, dict):
            return None
        item_id = str(value.get("item_id", "")).strip()
        criterion = str(value.get("criterion", "")).strip()
        evidence_kind = str(value.get("evidence_kind", "general")).strip() or "general"
        if not item_id or not criterion:
            return None
        return cls(item_id=item_id, criterion=criterion, evidence_kind=evidence_kind)


@dataclass(slots=True, frozen=True)
class TaskContract:
    task_id: str
    origin_turn_id: str
    user_message_sha256: str
    requirements: tuple[TaskRequirement, ...]

    def to_state(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "origin_turn_id": self.origin_turn_id,
            "user_message_sha256": self.user_message_sha256,
            "requirements": [item.to_state() for item in self.requirements],
        }

    @classmethod
    def from_state(cls, value: object) -> "TaskContract | None":
        if not isinstance(value, dict):
            return None
        task_id = str(value.get("task_id", "")).strip()
        origin_turn_id = str(value.get("origin_turn_id", "")).strip()
        user_message_sha256 = str(value.get("user_message_sha256", "")).strip()
        raw_requirements = value.get("requirements")
        if not task_id or not origin_turn_id or not user_message_sha256:
            return None
        requirements = tuple(
            requirement
            for raw in raw_requirements
            if (requirement := TaskRequirement.from_state(raw)) is not None
        ) if isinstance(raw_requirements, list) else ()
        return cls(
            task_id=task_id,
            origin_turn_id=origin_turn_id,
            user_message_sha256=user_message_sha256,
            requirements=requirements,
        )

    def render(self) -> str:
        lines = [
            "Task contract",
            f"task_id: {self.task_id}",
            f"user_message_sha256: {self.user_message_sha256}",
            "These user-owned requirements are mandatory. You may add checks, but you cannot "
            "omit or downgrade these items when recording acceptance.",
        ]
        if not self.requirements:
            lines.append("No separately structured hard requirements were detected.")
        else:
            lines.append("Required acceptance items:")
            lines.extend(
                f"- {item.item_id} [{item.evidence_kind}]: {item.criterion}"
                for item in self.requirements
            )
        return "\n".join(lines)


def build_task_contract(*, task_id: str, origin_turn_id: str, user_text: str) -> TaskContract:
    normalized_text = user_text.strip()
    message_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    requirements = _extract_requirements((normalized_text,), require_marker=True)
    return TaskContract(
        task_id=task_id,
        origin_turn_id=origin_turn_id,
        user_message_sha256=message_hash,
        requirements=requirements,
    )


def build_assignment_task_contract(
    *,
    task_id: str,
    origin_turn_id: str,
    assignment_texts: tuple[str, ...],
) -> TaskContract:
    """Build a child contract from its actual assignment rather than kickoff prose."""

    normalized_texts = tuple(
        normalized
        for text in assignment_texts
        if (normalized := text.strip())
    )
    message_hash = hashlib.sha256("\n".join(normalized_texts).encode("utf-8")).hexdigest()
    return TaskContract(
        task_id=task_id,
        origin_turn_id=origin_turn_id,
        user_message_sha256=message_hash,
        requirements=_extract_requirements(normalized_texts, require_marker=False),
    )


def _extract_requirements(
    source_texts: tuple[str, ...],
    *,
    require_marker: bool,
) -> tuple[TaskRequirement, ...]:
    requirements: list[TaskRequirement] = []
    seen: set[str] = set()
    for source_text in source_texts:
        for raw_fragment in _SENTENCE_BOUNDARY.split(source_text):
            fragment = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_fragment).strip()
            fragment = re.sub(r"\s+", " ", fragment)
            if (
                not fragment
                or _REQUIREMENT_REPLACEMENT_PATTERN.search(fragment)
                or (require_marker and not _REQUIREMENT_MARKER.search(fragment))
            ):
                continue
            criterion = fragment[:_MAX_CRITERION_CHARS].rstrip()
            normalized_criterion = criterion.casefold()
            if normalized_criterion in seen:
                continue
            seen.add(normalized_criterion)
            item_hash = hashlib.sha256(normalized_criterion.encode("utf-8")).hexdigest()[:12]
            evidence_kind = _classify_evidence_kind(criterion)
            if not require_marker and evidence_kind == "delegation":
                evidence_kind = "general"
            requirements.append(
                TaskRequirement(
                    item_id=f"user-{item_hash}",
                    criterion=criterion,
                    evidence_kind=evidence_kind,
                )
            )
            if len(requirements) >= _MAX_REQUIREMENTS:
                return tuple(requirements)
    return tuple(requirements)


def user_message_explicitly_resumes_task(user_text: str) -> bool:
    return bool(_RESUME_PATTERN.search(user_text)) and not bool(
        _REQUIREMENT_REPLACEMENT_PATTERN.search(user_text)
    )


def user_message_explicitly_replaces_task(user_text: str) -> bool:
    return bool(_TASK_REPLACEMENT_PATTERN.search(user_text))


def user_message_explicitly_replaces_requirements(user_text: str) -> bool:
    return bool(_REQUIREMENT_REPLACEMENT_PATTERN.search(user_text))


def user_message_is_side_query(user_text: str) -> bool:
    """Return whether a user message asks about, rather than changes, active work."""

    if not _SIDE_QUERY_PATTERN.search(user_text):
        return False
    return not bool(
        re.search(
            r"\b(?:then|and)\s+(?:continue|resume|keep|fix|implement|build|change|edit|run)\b",
            user_text,
            re.IGNORECASE,
        )
    )


def merge_task_contract(
    contract: TaskContract,
    *,
    user_text: str,
) -> TaskContract:
    if user_message_explicitly_replaces_requirements(user_text):
        remaining = _withdraw_requirements(contract.requirements, user_text=user_text)
        return TaskContract(
            task_id=contract.task_id,
            origin_turn_id=contract.origin_turn_id,
            user_message_sha256=contract.user_message_sha256,
            requirements=remaining,
        )
    supplemental = build_task_contract(
        task_id=contract.task_id,
        origin_turn_id=contract.origin_turn_id,
        user_text=user_text,
    )
    merged_requirements = {item.item_id: item for item in contract.requirements}
    merged_requirements.update({item.item_id: item for item in supplemental.requirements})
    return TaskContract(
        task_id=contract.task_id,
        origin_turn_id=contract.origin_turn_id,
        user_message_sha256=contract.user_message_sha256,
        requirements=tuple(merged_requirements.values()),
    )


def _withdraw_requirements(
    requirements: tuple[TaskRequirement, ...],
    *,
    user_text: str,
) -> tuple[TaskRequirement, ...]:
    if not requirements:
        return ()
    target_words = {
        word
        for word in re.findall(r"[a-z0-9]+", user_text.casefold())
        if word not in _WITHDRAWAL_STOP_WORDS
    }
    if not target_words:
        return requirements[:-1]
    scored = [
        (
            len(
                target_words
                & {
                    word
                    for word in re.findall(r"[a-z0-9]+", item.criterion.casefold())
                    if word not in _WITHDRAWAL_STOP_WORDS
                }
            ),
            index,
        )
        for index, item in enumerate(requirements)
    ]
    score, index = max(scored, key=lambda candidate: (candidate[0], candidate[1]))
    if score == 0:
        index = len(requirements) - 1
    return requirements[:index] + requirements[index + 1 :]


def _classify_evidence_kind(criterion: str) -> str:
    lowered = criterion.casefold()
    if "subagent" in lowered or "delegate" in lowered:
        return "delegation"
    if re.search(r"\b(?:loc|lines? of code|\d+[km]?\s+lines?)\b", lowered):
        return "source_line_count"
    if any(word in lowered for word in ("visual", "browser", "live inspection", "screenshot")):
        return "visual_inspection"
    if any(word in lowered for word in ("test", "lint", "typecheck", "build")):
        return "verification_gate"
    return "general"
