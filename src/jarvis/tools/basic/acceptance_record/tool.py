"""Transcript-persisted acceptance evidence for completed work."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from jarvis.llm import ToolDefinition

from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult
from ...workspace_revision import workspace_revision

_MAX_CHECKS = 24
_MAX_CHECK_TEXT_CHARS = 4_000
_MAX_SCOPE_CHARS = 800
_OUTCOMES = frozenset(
    {
        "open",
        "fixed",
        "not_a_bug",
        "deferred",
        "user_waived",
        # Backward-compatible gate vocabulary.
        "passed",
        "failed",
        "blocked",
        "not_run",
    }
)
_RESOLVED_OUTCOMES = frozenset({"fixed", "not_a_bug", "user_waived", "passed"})
_EVIDENCE_KINDS = frozenset(
    {"test_result", "artifact_inspection", "runtime_observation", "user_confirmation"}
)


class AcceptanceRecordToolExecutor:
    """Stores an append-only, model-visible acceptance ledger entry."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        scope = str(arguments.get("scope", "")).strip()
        observed_revision = str(arguments.get("workspace_revision", "")).strip()
        checks = arguments.get("checks")
        if not scope or len(scope) > _MAX_SCOPE_CHARS:
            return _failure(call_id, "scope must be a non-empty concise string.")
        current_revision = workspace_revision(context.workspace_dir)
        if not observed_revision:
            return _failure(
                call_id,
                "workspace_revision is required; use acceptance_run and copy its final revision.",
            )
        if observed_revision != current_revision:
            return _failure(
                call_id,
                "workspace revision changed after verification; rerun the acceptance gates.",
            )
        if not isinstance(checks, list) or not checks or len(checks) > _MAX_CHECKS:
            return _failure(
                call_id,
                f"checks must contain between 1 and {_MAX_CHECKS} entries.",
            )

        normalized_checks: list[dict[str, object]] = []
        for index, raw_check in enumerate(checks, start=1):
            if not isinstance(raw_check, dict):
                return _failure(call_id, f"checks[{index}] must be an object.")
            check, reason = _normalize_check(raw_check, context.workspace_dir)
            if reason is not None:
                return _failure(call_id, f"checks[{index}]: {reason}")
            normalized_checks.append(check)

        counts = {outcome: 0 for outcome in sorted(_OUTCOMES)}
        for check in normalized_checks:
            counts[str(check["outcome"])] += 1
        lines = [
            "Acceptance ledger recorded",
            f"scope: {scope}",
            "summary: " + ", ".join(f"{outcome}={count}" for outcome, count in counts.items()),
        ]
        for index, check in enumerate(normalized_checks, start=1):
            lines.extend(
                [
                    f"check_{index}.criterion: {check['criterion']}",
                    f"check_{index}.item_id: {check['item_id']}",
                    f"check_{index}.required: {str(check['required']).lower()}",
                    f"check_{index}.outcome: {check['outcome']}",
                    f"check_{index}.evidence_kind: {check['evidence_kind']}",
                    f"check_{index}.evidence: {check['evidence']}",
                ]
            )
            source_call_ids = _stored_string_list(check["source_tool_call_ids"])
            if source_call_ids:
                lines.append(
                    f"check_{index}.source_tool_call_ids: " + ", ".join(source_call_ids)
                )
            artifact_paths = _stored_string_list(check["artifact_paths"])
            if artifact_paths:
                lines.append(f"check_{index}.artifact_paths: " + ", ".join(artifact_paths))

        unresolved_required = sum(
            1
            for check in normalized_checks
            if bool(check["required"]) and str(check["outcome"]) not in _RESOLVED_OUTCOMES
        )
        return ToolExecutionResult(
            call_id=call_id,
            name="acceptance_record",
            ok=True,
            content="\n".join(lines),
            metadata={
                "acceptance_ledger": {
                    "scope": scope,
                    "checks": normalized_checks,
                    "summary": counts,
                    "unresolved_required_count": unresolved_required,
                    "complete": unresolved_required == 0,
                    "workspace_revision": current_revision,
                    "workspace_revision_verified": True,
                }
            },
        )


def build_acceptance_record_tool() -> RegisteredTool:
    """Build the basic acceptance-evidence tool."""

    return RegisteredTool(
        name="acceptance_record",
        exposure="basic",
        definition=ToolDefinition(
            name="acceptance_record",
            description=(
                "Record explicit acceptance evidence before claiming implementation work is "
                "complete. Each check must state its actual outcome and evidence; an exit code "
                "alone does not prove semantic success."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "minLength": 1, "maxLength": _MAX_SCOPE_CHARS},
                    "workspace_revision": {"type": "string", "minLength": 1},
                    "checks": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_CHECKS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "criterion": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": _MAX_CHECK_TEXT_CHARS,
                                },
                                "item_id": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 160,
                                },
                                "required": {"type": "boolean"},
                                "outcome": {
                                    "type": "string",
                                    "enum": sorted(_OUTCOMES),
                                },
                                "evidence_kind": {
                                    "type": "string",
                                    "enum": sorted(_EVIDENCE_KINDS),
                                },
                                "evidence": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": _MAX_CHECK_TEXT_CHARS,
                                },
                                "source_tool_call_ids": {
                                    "type": "array",
                                    "maxItems": 16,
                                    "items": {"type": "string", "minLength": 1},
                                },
                                "artifact_paths": {
                                    "type": "array",
                                    "maxItems": 16,
                                    "items": {"type": "string", "minLength": 1},
                                },
                            },
                            "required": [
                                "criterion",
                                "outcome",
                                "evidence_kind",
                                "evidence",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["scope", "workspace_revision", "checks"],
                "additionalProperties": False,
            },
        ),
        executor=AcceptanceRecordToolExecutor(),
    )


def _normalize_check(
    raw_check: dict[str, Any],
    workspace_dir: Path,
) -> tuple[dict[str, object], str | None]:
    criterion = str(raw_check.get("criterion", "")).strip()
    item_id = str(raw_check.get("item_id", "")).strip()
    if not item_id:
        item_id = "criterion-" + hashlib.sha256(criterion.encode("utf-8")).hexdigest()[:12]
    required = raw_check.get("required", True)
    outcome = str(raw_check.get("outcome", "")).strip()
    evidence_kind = str(raw_check.get("evidence_kind", "")).strip()
    evidence = str(raw_check.get("evidence", "")).strip()
    if not criterion or len(criterion) > _MAX_CHECK_TEXT_CHARS:
        return {}, "criterion must be a non-empty concise string."
    if len(item_id) > 160:
        return {}, "item_id must not exceed 160 characters."
    if not isinstance(required, bool):
        return {}, "required must be a boolean when supplied."
    if outcome not in _OUTCOMES:
        return {}, f"outcome must be one of: {', '.join(sorted(_OUTCOMES))}."
    if evidence_kind not in _EVIDENCE_KINDS:
        return {}, f"evidence_kind must be one of: {', '.join(sorted(_EVIDENCE_KINDS))}."
    if not evidence or len(evidence) > _MAX_CHECK_TEXT_CHARS:
        return {}, "evidence must be a non-empty concise string."
    source_tool_call_ids = _string_list(raw_check.get("source_tool_call_ids"))
    if source_tool_call_ids is None:
        return {}, "source_tool_call_ids must be a list of non-empty strings when supplied."
    artifact_paths = _workspace_paths(raw_check.get("artifact_paths"), workspace_dir)
    if artifact_paths is None:
        return {}, "artifact_paths must contain only workspace-relative paths when supplied."
    if outcome in {"passed", "fixed"} and evidence_kind == "user_confirmation":
        return {}, "a fixed implementation check needs test, artifact, or runtime evidence."
    if outcome == "user_waived" and evidence_kind != "user_confirmation":
        return {}, "user_waived requires user_confirmation evidence."
    if evidence_kind in {"test_result", "runtime_observation"} and not source_tool_call_ids:
        return {}, "test_result and runtime_observation evidence require source_tool_call_ids."
    return (
        {
            "criterion": criterion,
            "item_id": item_id,
            "required": required,
            "outcome": outcome,
            "evidence_kind": evidence_kind,
            "evidence": evidence,
            "source_tool_call_ids": source_tool_call_ids,
            "artifact_paths": artifact_paths,
        },
        None,
    )


def _string_list(value: object) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 16:
        return None
    values = [str(item).strip() for item in value]
    if not all(values):
        return None
    return values


def _stored_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _workspace_paths(value: object, workspace_dir: Path) -> list[str] | None:
    raw_paths = _string_list(value)
    if raw_paths is None:
        return None
    normalized: list[str] = []
    root = workspace_dir.resolve(strict=False)
    for raw_path in raw_paths:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            if candidate == Path("/workspace") or candidate.is_relative_to(Path("/workspace")):
                candidate = root / candidate.relative_to("/workspace")
        else:
            candidate = root / candidate
        resolved = candidate.resolve(strict=False)
        if resolved != root and not resolved.is_relative_to(root):
            return None
        normalized.append(str(Path("/workspace") / resolved.relative_to(root)))
    return normalized


def _failure(call_id: str, reason: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        call_id=call_id,
        name="acceptance_record",
        ok=False,
        content=(
            "Acceptance ledger was not recorded\n"
            "error_code: invalid_acceptance_evidence\n"
            f"reason: {reason}"
        ),
        metadata={
            "execution_failed": True,
            "error_code": "invalid_acceptance_evidence",
            "reason": reason,
        },
    )
