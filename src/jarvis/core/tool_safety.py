"""Turn-scoped safety accounting for repeated or non-progressing tool activity."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from jarvis.llm import ToolCall
from jarvis.llm.validation import TOOL_CALL_VALIDATION_ERROR_METADATA_KEY
from jarvis.tools import ToolExecutionResult

from .task_contract import TaskRequirement


_FILE_EDIT_TOOL_NAMES = frozenset({"file_patch", "file_write", "file_replace"})


@dataclass(slots=True, frozen=True)
class ToolSafetyObservation:
    """Safety result for one completed tool call."""

    repeated_invalid_call: bool = False
    blocked_invalid_signature: bool = False
    repeated_no_progress: bool = False
    made_progress: bool = False
    signature_id: str | None = None
    occurrence_count: int = 0
    progress_epoch: int = 0
    first_call_id: str | None = None


@dataclass(slots=True)
class ToolSafetyTracker:
    """Detect repeated invalid calls and identical no-progress tool activity."""

    _invalid_counts: dict[str, int] = field(default_factory=dict)
    _no_progress_counts: dict[str, int] = field(default_factory=dict)
    _seen_activity_signatures: set[str] = field(default_factory=set)
    _invalid_first_call_ids: dict[str, str] = field(default_factory=dict)
    _no_progress_first_call_ids: dict[str, str] = field(default_factory=dict)
    _blocked_call_reasons: dict[str, str] = field(default_factory=dict)
    _blocked_call_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    _progress_epoch: int = 0
    _progress_since_slice: bool = False
    _workspace_mutated: bool = False
    _acceptance_recorded: bool = False
    _passed_acceptance_run_call_ids: set[str] = field(default_factory=set)
    _passed_acceptance_run_scopes: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    _acceptance_items: dict[str, dict[str, Any]] = field(default_factory=dict)
    _contract_requirements: dict[str, dict[str, str]] = field(default_factory=dict)
    _subagent_invocation_count: int = 0
    _test_review_subagent_ids: set[str] = field(default_factory=set)
    _completed_test_review_subagent_ids: set[str] = field(default_factory=set)
    _passed_acceptance_gates: dict[str, str] = field(default_factory=dict)
    _visual_inspection_paths: set[str] = field(default_factory=set)
    _runtime_progress_signatures: list[str] = field(default_factory=list)

    def seed_contract_requirements(
        self,
        requirements: tuple[TaskRequirement, ...],
    ) -> None:
        """Install immutable user-owned acceptance requirements for this task."""

        supplied = {item.item_id: item.to_state() for item in requirements}
        if not self._contract_requirements:
            self._contract_requirements = supplied
            return
        for item_id, requirement in supplied.items():
            self._contract_requirements.setdefault(item_id, requirement)

    def blocked_call_reason(self, tool_call: ToolCall) -> str | None:
        return self._blocked_call_reasons.get(_tool_call_signature(tool_call))

    def blocked_call_details(self, tool_call: ToolCall) -> dict[str, Any]:
        return dict(self._blocked_call_details.get(_tool_call_signature(tool_call), {}))

    def record(self, tool_call: ToolCall, result: ToolExecutionResult) -> ToolSafetyObservation:
        workspace_mutated = _result_mutated_workspace(tool_call, result)
        if workspace_mutated:
            self._advance_progress_epoch(invalidate_acceptance=True)
            self._workspace_mutated = True
            self._track_changed_test_artifacts(tool_call, result)
        elif _result_is_material_progress(result):
            self._advance_progress_epoch(invalidate_acceptance=False)

        invalid_signature = _invalid_signature(tool_call, result)
        if invalid_signature is not None:
            count = self._invalid_counts.get(invalid_signature, 0) + 1
            self._invalid_counts[invalid_signature] = count
            first_call_id = self._invalid_first_call_ids.setdefault(
                invalid_signature,
                tool_call.call_id,
            )
            blocked = count >= 2
            if blocked:
                call_signature = _tool_call_signature(tool_call)
                self._blocked_call_reasons[call_signature] = "repeated_invalid_result"
                self._blocked_call_details[call_signature] = {
                    "reason": "repeated_invalid_result",
                    "tool": tool_call.name,
                    "call_signature": call_signature,
                    "result_signature": invalid_signature,
                    "first_call_id": first_call_id,
                    "threshold_call_id": tool_call.call_id,
                    "occurrence_count": count,
                    "progress_epoch": self._progress_epoch,
                }
            return ToolSafetyObservation(
                repeated_invalid_call=count >= 2,
                blocked_invalid_signature=blocked,
                signature_id=invalid_signature,
                occurrence_count=count,
                progress_epoch=self._progress_epoch,
                first_call_id=first_call_id,
            )

        if result.name == "acceptance_run" and result.ok:
            self._passed_acceptance_run_call_ids.add(result.call_id)
            run_scope = _acceptance_run_scope(result)
            if run_scope is not None:
                self._passed_acceptance_run_scopes[result.call_id] = run_scope
            self._passed_acceptance_gates.update(_passed_acceptance_gates(result))
        if result.name == "subagent_invoke" and result.ok:
            self._subagent_invocation_count += 1
            review_requirement = self._contract_requirements.get(
                "system-test-change-review"
            )
            if review_requirement is not None and _is_test_review_assignment(
                tool_call,
                review_requirement,
            ):
                subagent_id = str(result.metadata.get("subagent_id", "")).strip()
                if subagent_id:
                    self._test_review_subagent_ids.add(subagent_id)
        if result.name == "subagent_monitor" and result.ok:
            self._record_completed_test_reviews(result)
        if result.name == "view_image" and result.ok:
            paths = {
                str(result.metadata.get("path", "")).strip(),
                str(tool_call.arguments.get("path", "")).strip(),
            }
            self._visual_inspection_paths.update(path for path in paths if path)
        if result.name == "acceptance_record" and result.ok:
            self._acceptance_items.update(_acceptance_ledger_items(result))
            self._acceptance_recorded = _acceptance_ledger_resolved(
                result,
                valid_gate_call_ids=self._passed_acceptance_run_call_ids,
                valid_gate_scopes=self._passed_acceptance_run_scopes,
                durable_items=self._acceptance_items,
                contract_requirements=self._contract_requirements,
                subagent_invocation_count=self._subagent_invocation_count,
                completed_test_review_subagent_ids=(
                    self._completed_test_review_subagent_ids
                ),
                passed_gates=self._passed_acceptance_gates,
                visual_inspection_paths=self._visual_inspection_paths,
            )

        activity_signature = _tool_result_signature(tool_call, result)
        no_progress = _is_no_progress_result(result)
        if no_progress:
            count = self._no_progress_counts.get(activity_signature, 0) + 1
            self._no_progress_counts[activity_signature] = count
            repeated = count >= 3
            first_call_id = self._no_progress_first_call_ids.setdefault(
                activity_signature,
                tool_call.call_id,
            )
            first_seen = activity_signature not in self._seen_activity_signatures
            self._seen_activity_signatures.add(activity_signature)
            if first_seen:
                self._progress_since_slice = True
            if repeated:
                call_signature = _tool_call_signature(tool_call)
                self._blocked_call_reasons[call_signature] = "repeated_no_progress_result"
                self._blocked_call_details[call_signature] = {
                    "reason": "repeated_no_progress_result",
                    "tool": tool_call.name,
                    "call_signature": call_signature,
                    "result_signature": activity_signature,
                    "first_call_id": first_call_id,
                    "threshold_call_id": tool_call.call_id,
                    "occurrence_count": count,
                    "progress_epoch": self._progress_epoch,
                }
            return ToolSafetyObservation(
                repeated_no_progress=repeated,
                made_progress=first_seen,
                signature_id=activity_signature,
                occurrence_count=count,
                progress_epoch=self._progress_epoch,
                first_call_id=first_call_id,
            )

        self._seen_activity_signatures.add(activity_signature)
        self._progress_since_slice = True
        return ToolSafetyObservation(
            made_progress=True,
            progress_epoch=self._progress_epoch,
        )

    def _advance_progress_epoch(self, *, invalidate_acceptance: bool) -> None:
        self._progress_epoch += 1
        self._invalid_counts.clear()
        self._no_progress_counts.clear()
        self._seen_activity_signatures.clear()
        self._invalid_first_call_ids.clear()
        self._no_progress_first_call_ids.clear()
        self._blocked_call_reasons.clear()
        self._blocked_call_details.clear()
        self._progress_since_slice = True
        if invalidate_acceptance:
            self._acceptance_recorded = False
            self._passed_acceptance_run_call_ids.clear()
            self._passed_acceptance_run_scopes.clear()
            self._passed_acceptance_gates.clear()

    @property
    def progress_epoch(self) -> int:
        return self._progress_epoch

    def record_runtime_progress(
        self,
        *,
        content: str,
        metadata: dict[str, Any],
    ) -> bool:
        """Advance progress once for each materially distinct orchestrator update."""

        signature_payload: dict[str, Any]
        if metadata.get("bash_job_progress_update"):
            signature_payload = {
                "kind": "bash",
                "job_ids": metadata.get("detached_bash_job_ids", []),
                "notice_kinds": metadata.get("bash_job_notice_kinds", []),
                "running_ids": metadata.get("bash_job_running_ids", []),
                "terminal_ids": metadata.get("bash_job_terminal_ids", []),
                "recommended_action": metadata.get("recommended_action"),
                "progress_fingerprints": metadata.get(
                    "bash_job_progress_fingerprints",
                    [],
                ),
            }
        elif metadata.get("subagent_progress_update"):
            signature_payload = {
                "kind": "subagent",
                "subagent_id": metadata.get("subagent_id"),
                "notice_kind": metadata.get("subagent_notice_kind"),
                "pending_ids": metadata.get("pending_subagent_ids", []),
                "recommended_action": metadata.get("recommended_action"),
                "report_complete": metadata.get("latest_subagent_report_complete"),
                "content": content,
            }
        else:
            return False

        signature = _digest(signature_payload)
        if signature in self._runtime_progress_signatures:
            return False
        self._runtime_progress_signatures.append(signature)
        del self._runtime_progress_signatures[:-256]
        self._advance_progress_epoch(invalidate_acceptance=False)
        return True

    def consume_slice_progress(self) -> bool:
        made_progress = self._progress_since_slice
        self._progress_since_slice = False
        return made_progress

    def checkpoint_lines(self) -> tuple[str, ...]:
        """Return a compact model-visible checkpoint of unmet hard obligations."""

        lines = [
            f"progress_epoch: {self._progress_epoch}",
            f"subagents_invoked: {self._subagent_invocation_count}",
            f"passing_acceptance_gates: {len(self._passed_acceptance_gates)}",
            f"visual_artifacts_inspected: {len(self._visual_inspection_paths)}",
        ]
        outstanding = [
            requirement
            for item_id, requirement in self._contract_requirements.items()
            if str(self._acceptance_items.get(item_id, {}).get("outcome", ""))
            not in {"fixed", "passed"}
        ]
        if outstanding:
            lines.append("Outstanding hard requirements:")
            lines.extend(
                f"- {item['item_id']} [{item['evidence_kind']}]: {item['criterion']}"
                for item in outstanding
            )
        else:
            lines.append("Outstanding hard requirements: none recorded.")
        return tuple(lines)

    def _track_changed_test_artifacts(
        self,
        tool_call: ToolCall,
        result: ToolExecutionResult,
    ) -> None:
        paths = {
            str(result.metadata.get("path", "")).strip(),
            *(
                str(path).strip()
                for path in result.metadata.get("workspace_changed_paths", [])
                if str(path).strip()
            ),
            str(tool_call.arguments.get("path", "")).strip(),
        }
        test_paths = sorted(path for path in paths if path and _path_is_test_artifact(path))
        if not test_paths:
            return
        item_id = "system-test-change-review"
        self._contract_requirements.setdefault(
            item_id,
            {
                "item_id": item_id,
                "criterion": (
                    "Independently review semantic changes to test artifacts: "
                    + ", ".join(test_paths)
                ),
                "evidence_kind": "test_change_review",
            },
        )

    def _record_completed_test_reviews(self, result: ToolExecutionResult) -> None:
        raw_subagents = result.metadata.get("subagents")
        if not isinstance(raw_subagents, list):
            return
        for raw_subagent in raw_subagents:
            if not isinstance(raw_subagent, dict):
                continue
            subagent_id = str(raw_subagent.get("subagent_id", "")).strip()
            if (
                subagent_id in self._test_review_subagent_ids
                and str(raw_subagent.get("status", "")).strip() == "completed"
                and bool(raw_subagent.get("report_complete", False))
            ):
                self._completed_test_review_subagent_ids.add(subagent_id)

    @property
    def unverified_workspace_mutation(self) -> bool:
        return self._workspace_mutated and not self._acceptance_recorded

    def to_state(self) -> dict[str, Any]:
        """Return the bounded durable state needed to resume safety accounting."""

        return {
            "invalid_counts": dict(self._invalid_counts),
            "no_progress_counts": dict(self._no_progress_counts),
            "seen_activity_signatures": sorted(self._seen_activity_signatures),
            "invalid_first_call_ids": dict(self._invalid_first_call_ids),
            "no_progress_first_call_ids": dict(self._no_progress_first_call_ids),
            "blocked_call_reasons": dict(self._blocked_call_reasons),
            "blocked_call_details": dict(self._blocked_call_details),
            "progress_epoch": self._progress_epoch,
            "progress_since_slice": self._progress_since_slice,
            "workspace_mutated": self._workspace_mutated,
            "acceptance_recorded": self._acceptance_recorded,
            "passed_acceptance_run_call_ids": sorted(
                self._passed_acceptance_run_call_ids
            ),
            "passed_acceptance_run_scopes": dict(
                self._passed_acceptance_run_scopes
            ),
            "acceptance_items": dict(self._acceptance_items),
            "contract_requirements": dict(self._contract_requirements),
            "subagent_invocation_count": self._subagent_invocation_count,
            "test_review_subagent_ids": sorted(self._test_review_subagent_ids),
            "completed_test_review_subagent_ids": sorted(
                self._completed_test_review_subagent_ids
            ),
            "passed_acceptance_gates": dict(self._passed_acceptance_gates),
            "visual_inspection_paths": sorted(self._visual_inspection_paths),
            "runtime_progress_signatures": list(self._runtime_progress_signatures),
        }

    @classmethod
    def from_state(cls, value: object) -> "ToolSafetyTracker":
        """Restore validated state; malformed persisted data safely starts empty."""

        if not isinstance(value, dict):
            return cls()
        return cls(
            _invalid_counts=_bounded_count_map(value.get("invalid_counts")),
            _no_progress_counts=_bounded_count_map(value.get("no_progress_counts")),
            _seen_activity_signatures=_bounded_string_set(
                value.get("seen_activity_signatures")
            ),
            _invalid_first_call_ids=_bounded_string_map(
                value.get("invalid_first_call_ids")
            ),
            _no_progress_first_call_ids=_bounded_string_map(
                value.get("no_progress_first_call_ids")
            ),
            _blocked_call_reasons=_bounded_string_map(
                value.get("blocked_call_reasons")
            ),
            _blocked_call_details=_bounded_detail_map(
                value.get("blocked_call_details")
            ),
            _progress_epoch=_bounded_non_negative_int(value.get("progress_epoch")),
            _progress_since_slice=bool(value.get("progress_since_slice", False)),
            _workspace_mutated=bool(value.get("workspace_mutated", False)),
            _acceptance_recorded=bool(value.get("acceptance_recorded", False)),
            _passed_acceptance_run_call_ids=_bounded_string_set(
                value.get("passed_acceptance_run_call_ids")
            ),
            _passed_acceptance_run_scopes=_bounded_acceptance_run_scopes(
                value.get("passed_acceptance_run_scopes")
            ),
            _acceptance_items=_bounded_acceptance_items(
                value.get("acceptance_items")
            ),
            _contract_requirements=_bounded_contract_requirements(
                value.get("contract_requirements")
            ),
            _subagent_invocation_count=_bounded_non_negative_int(
                value.get("subagent_invocation_count")
            ),
            _test_review_subagent_ids=_bounded_string_set(
                value.get("test_review_subagent_ids"),
                limit=64,
            ),
            _completed_test_review_subagent_ids=_bounded_string_set(
                value.get("completed_test_review_subagent_ids"),
                limit=64,
            ),
            _passed_acceptance_gates=_bounded_string_map(
                value.get("passed_acceptance_gates"),
                limit=64,
            ),
            _visual_inspection_paths=_bounded_string_set(
                value.get("visual_inspection_paths"),
                limit=64,
            ),
            _runtime_progress_signatures=_bounded_string_list(
                value.get("runtime_progress_signatures"),
                limit=256,
            ),
        )


def build_blocked_repetition_result(
    *,
    tool_call: ToolCall,
    reason: str,
    diagnostics: dict[str, Any] | None = None,
) -> ToolExecutionResult:
    """Return deterministic feedback without executing an already-blocked action."""

    return ToolExecutionResult(
        call_id=tool_call.call_id,
        name=tool_call.name,
        ok=False,
        content=(
            "Tool call blocked\n"
            f"tool: {tool_call.name}\n"
            "error_code: blocked_repeated_tool_call\n"
            f"reason: {reason}. This exact action already crossed its unchanged-state retry "
            "limit. Replan with different arguments or gather material new evidence before "
            "trying it again."
        ),
        metadata={
            "tool_safety_blocked": True,
            "error_code": "blocked_repeated_tool_call",
            "reason": reason,
            "blocked_call_signature": _tool_call_signature(tool_call),
            "tool_safety_diagnostics": dict(diagnostics or {}),
            "arguments": dict(tool_call.arguments),
        },
    )


def _invalid_signature(tool_call: ToolCall, result: ToolExecutionResult) -> str | None:
    metadata = result.metadata
    if not (
        not result.ok
        or
        metadata.get("tool_call_validation_failed")
        or metadata.get("policy_denied")
        or metadata.get("execution_failed")
        or (tool_call.name in _FILE_EDIT_TOOL_NAMES and not result.ok)
        or tool_call.provider_metadata.get(TOOL_CALL_VALIDATION_ERROR_METADATA_KEY)
    ):
        return None
    if metadata.get("error_code") == "workspace_lease_conflict":
        return _digest(
            {
                "error_code": "workspace_lease_conflict",
                "conflict_key": str(
                    metadata.get("conflict_key")
                    or metadata.get("conflict_class")
                    or "workspace_access_conflict"
                ),
            }
        )
    reason = str(metadata.get("reason") or metadata.get("error") or "").strip()
    return _digest(
        {
            "tool": tool_call.name,
            "arguments": _normalized_tool_arguments(tool_call),
            "ok": result.ok,
            "content": _stable_result_content(result),
            "metadata": _stable_metadata(result.metadata),
            "reason": reason,
        }
    )


def _tool_result_signature(tool_call: ToolCall, result: ToolExecutionResult) -> str:
    return _digest(
        {
            "tool": tool_call.name,
            "arguments": _normalized_tool_arguments(tool_call),
            "ok": result.ok,
            "content": _stable_result_content(result),
            "metadata": _stable_metadata(result.metadata),
        }
    )


def _tool_call_signature(tool_call: ToolCall) -> str:
    return _digest(
        {
            "tool": tool_call.name,
            "arguments": _normalized_tool_arguments(tool_call),
        }
    )


def _normalized_tool_arguments(tool_call: ToolCall) -> Any:
    raw = tool_call.raw_arguments.strip()
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
    return dict(tool_call.arguments)


def _stable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    volatile_keys = {
        "job_id",
        "launched_at",
        "started_at",
        "finished_at",
        "last_update_at",
        "stdout_path",
        "stderr_path",
        "error_log_path",
        "duration_seconds",
        "observed_at",
    }
    return _stable_mapping(metadata, volatile_keys=volatile_keys)


def _stable_mapping(
    value: dict[Any, Any],
    *,
    volatile_keys: set[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key in volatile_keys:
            continue
        if isinstance(raw_value, dict):
            output[key] = _stable_mapping(raw_value, volatile_keys=volatile_keys)
        elif isinstance(raw_value, list):
            output[key] = [
                _stable_mapping(item, volatile_keys=volatile_keys)
                if isinstance(item, dict)
                else item
                for item in raw_value
            ]
        else:
            output[key] = raw_value
    return output


def _stable_result_content(result: ToolExecutionResult) -> Any:
    if result.name != "bash":
        return result.content
    return {
        "mode": result.metadata.get("mode"),
        "status": result.metadata.get("status") or result.metadata.get("state"),
        "exit_code": result.metadata.get("exit_code"),
        "timed_out": result.metadata.get("timed_out"),
        "stdout": result.metadata.get("stdout", ""),
        "stderr": result.metadata.get("stderr", ""),
    }


def _is_no_progress_result(result: ToolExecutionResult) -> bool:
    if not result.ok:
        return False
    if result.name in _FILE_EDIT_TOOL_NAMES:
        return not bool(result.metadata.get("changed"))
    if result.name == "bash":
        return (
            not bool(result.metadata.get("workspace_changed", False))
            and
            result.metadata.get("mode") == "foreground"
            and result.metadata.get("exit_code") == 0
            and result.metadata.get("status") != "running"
        )
    return not bool(result.metadata.get("changed"))


def _result_mutated_workspace(tool_call: ToolCall, result: ToolExecutionResult) -> bool:
    if "workspace_changed" in result.metadata:
        return bool(result.metadata.get("workspace_changed"))
    if tool_call.name in _FILE_EDIT_TOOL_NAMES:
        return bool(result.metadata.get("changed"))
    if tool_call.name == "bash":
        return bool(result.ok and tool_call.arguments.get("write_paths"))
    if tool_call.name == "acceptance_run":
        return bool(result.ok and result.metadata.get("changed"))
    if tool_call.name in {"generate_edit_image", "memory_write", "tool_register"}:
        return result.ok
    return False


def _result_is_material_progress(result: ToolExecutionResult) -> bool:
    if not result.ok:
        return False
    if result.name == "acceptance_run":
        return True
    if bool(result.metadata.get("changed", False)):
        return True
    return result.name in {
        "subagent_invoke",
        "subagent_step_in",
        "subagent_stop",
        "subagent_dispose",
    }


def _acceptance_ledger_resolved(
    result: ToolExecutionResult,
    *,
    valid_gate_call_ids: set[str],
    valid_gate_scopes: dict[str, dict[str, Any]],
    durable_items: dict[str, dict[str, Any]],
    contract_requirements: dict[str, dict[str, str]],
    subagent_invocation_count: int,
    completed_test_review_subagent_ids: set[str],
    passed_gates: dict[str, str],
    visual_inspection_paths: set[str],
) -> bool:
    ledger = result.metadata.get("acceptance_ledger")
    if not isinstance(ledger, dict):
        return False
    if not bool(ledger.get("workspace_revision_verified", False)):
        return False
    if not bool(ledger.get("complete", False)):
        return False
    checks = ledger.get("checks")
    if not isinstance(checks, list) or not checks:
        return False
    resolved = bool(durable_items) and all(
        not bool(item.get("required", True))
        or str(item.get("outcome", "")).strip()
        in {"fixed", "passed", "not_a_bug", "user_waived"}
        for item in durable_items.values()
    )
    if not resolved or not valid_gate_call_ids:
        return False
    for item_id, requirement in contract_requirements.items():
        item = durable_items.get(item_id)
        if item is None or str(item.get("outcome", "")) not in {
            "fixed",
            "passed",
        }:
            return False
        if not _contract_machine_evidence_resolved(
            requirement,
            item=item,
            subagent_invocation_count=subagent_invocation_count,
            completed_test_review_subagent_ids=completed_test_review_subagent_ids,
            passed_gates=passed_gates,
            visual_inspection_paths=visual_inspection_paths,
        ):
            return False
    supplied_source_ids = {
        str(call_id)
        for check in checks
        if isinstance(check, dict)
        for call_id in check.get("source_tool_call_ids", [])
    }
    cited_valid_ids = supplied_source_ids & valid_gate_call_ids
    if not cited_valid_ids:
        return False
    ledger_paths = _string_list(ledger.get("revision_paths"))
    ledger_revision = str(ledger.get("workspace_revision", "")).strip()
    if not ledger_paths or not ledger_revision:
        return False
    return any(
        scope.get("revision_paths") == ledger_paths
        and scope.get("workspace_revision") == ledger_revision
        for call_id in cited_valid_ids
        if (scope := valid_gate_scopes.get(call_id)) is not None
    )


def _acceptance_ledger_items(result: ToolExecutionResult) -> dict[str, dict[str, Any]]:
    ledger = result.metadata.get("acceptance_ledger")
    if not isinstance(ledger, dict):
        return {}
    checks = ledger.get("checks")
    if not isinstance(checks, list):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        item_id = str(check.get("item_id", "")).strip()
        outcome = str(check.get("outcome", "")).strip()
        if not item_id or not outcome:
            continue
        output[item_id] = {
            "required": bool(check.get("required", True)),
            "outcome": outcome,
            "evidence_kind": str(check.get("evidence_kind", "")).strip(),
            "source_tool_call_ids": _string_list(check.get("source_tool_call_ids")),
            "artifact_paths": _string_list(check.get("artifact_paths")),
        }
    return output


def _bounded_count_map(value: object, *, limit: int = 64) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, int] = {}
    for raw_key, raw_count in list(value.items())[-limit:]:
        key = str(raw_key).strip()
        if not key:
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            output[key] = min(count, 1_000_000)
    return output


def _bounded_string_set(value: object, *, limit: int = 64) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value[-limit:] if str(item).strip()}


def _bounded_string_list(value: object, *, limit: int = 64) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value[-limit:] if str(item).strip()]


def _bounded_acceptance_items(
    value: object,
    *,
    limit: int = 128,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for raw_item_id, raw_item in list(value.items())[-limit:]:
        item_id = str(raw_item_id).strip()
        if not item_id or not isinstance(raw_item, dict):
            continue
        outcome = str(raw_item.get("outcome", "")).strip()
        if not outcome:
            continue
        output[item_id] = {
            "required": bool(raw_item.get("required", True)),
            "outcome": outcome,
            "evidence_kind": str(raw_item.get("evidence_kind", "")).strip(),
            "source_tool_call_ids": _string_list(raw_item.get("source_tool_call_ids")),
            "artifact_paths": _string_list(raw_item.get("artifact_paths")),
        }
    return output


def _bounded_acceptance_run_scopes(
    value: object,
    *,
    limit: int = 64,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for raw_call_id, raw_scope in list(value.items())[-limit:]:
        call_id = str(raw_call_id).strip()
        if not call_id or not isinstance(raw_scope, dict):
            continue
        revision_paths = _bounded_string_list(raw_scope.get("revision_paths"))
        workspace_revision = str(raw_scope.get("workspace_revision", "")).strip()
        if revision_paths and workspace_revision:
            output[call_id] = {
                "revision_paths": revision_paths,
                "workspace_revision": workspace_revision,
            }
    return output


def _bounded_string_map(value: object, *, limit: int = 64) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, str] = {}
    for raw_key, raw_value in list(value.items())[-limit:]:
        key = str(raw_key).strip()
        item = str(raw_value).strip()
        if key and item:
            output[key] = item
    return output


def _bounded_contract_requirements(
    value: object,
    *,
    limit: int = 64,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, dict[str, str]] = {}
    for raw_item_id, raw_requirement in list(value.items())[-limit:]:
        item_id = str(raw_item_id).strip()
        if not item_id or not isinstance(raw_requirement, dict):
            continue
        criterion = str(raw_requirement.get("criterion", "")).strip()
        evidence_kind = str(raw_requirement.get("evidence_kind", "general")).strip()
        if criterion:
            output[item_id] = {
                "item_id": item_id,
                "criterion": criterion,
                "evidence_kind": evidence_kind or "general",
            }
    return output


def _passed_acceptance_gates(result: ToolExecutionResult) -> dict[str, str]:
    run = result.metadata.get("acceptance_run")
    if not isinstance(run, dict):
        return {}
    gates = run.get("gates")
    if not isinstance(gates, list):
        return {}
    output: dict[str, str] = {}
    for gate in gates:
        if not isinstance(gate, dict) or not bool(gate.get("passed")):
            continue
        gate_id = str(gate.get("gate_id", "")).strip()
        if not gate_id:
            continue
        metric = gate.get("source_line_count")
        if isinstance(metric, dict):
            output[gate_id] = (
                f"authored_source_lines={metric.get('line_count')} "
                f"minimum={metric.get('minimum')} files={metric.get('file_count')}"
            )
            continue
        command = str(gate.get("command", "")).strip()
        if command:
            output[gate_id] = command
    return output


def _acceptance_run_scope(result: ToolExecutionResult) -> dict[str, Any] | None:
    run = result.metadata.get("acceptance_run")
    if not isinstance(run, dict) or not bool(run.get("passed", False)):
        return None
    revision_paths = _string_list(run.get("revision_paths"))
    workspace_revision = str(run.get("workspace_revision_after", "")).strip()
    if not revision_paths or not workspace_revision:
        return None
    return {
        "revision_paths": revision_paths,
        "workspace_revision": workspace_revision,
    }


def _contract_machine_evidence_resolved(
    requirement: dict[str, str],
    *,
    item: dict[str, Any],
    subagent_invocation_count: int,
    completed_test_review_subagent_ids: set[str],
    passed_gates: dict[str, str],
    visual_inspection_paths: set[str],
) -> bool:
    kind = requirement.get("evidence_kind", "general")
    if kind == "delegation":
        return subagent_invocation_count > 0
    if kind == "verification_gate":
        criterion = requirement.get("criterion", "").casefold()
        required_words = {
            word
            for word in ("test", "lint", "typecheck", "build")
            if word in criterion
        }
        commands = "\n".join(passed_gates.values()).casefold()
        return all(word in commands for word in required_words)
    if kind == "visual_inspection":
        artifact_paths = {
            str(path).strip() for path in item.get("artifact_paths", []) if str(path).strip()
        }
        inspected = {
            path.removeprefix("/workspace/") for path in visual_inspection_paths
        } | visual_inspection_paths
        normalized_artifacts = {
            path.removeprefix("/workspace/") for path in artifact_paths
        } | artifact_paths
        return bool(normalized_artifacts & inspected) and item.get("evidence_kind") in {
            "artifact_inspection",
            "runtime_observation",
        }
    if kind == "source_line_count":
        required_lines = _required_source_lines(requirement.get("criterion", ""))
        observed_counts = [
            int(match.group(1))
            for evidence in passed_gates.values()
            if (match := re.search(r"authored_source_lines=(\d+)", evidence)) is not None
        ]
        return bool(observed_counts) and (
            required_lines is None or max(observed_counts) >= required_lines
        )
    if kind == "test_change_review":
        required_paths = _test_paths_from_requirement(requirement)
        cited_paths = {
            str(path).strip().removeprefix("/workspace/")
            for path in item.get("artifact_paths", [])
            if str(path).strip()
        }
        return (
            bool(completed_test_review_subagent_ids)
            and bool(required_paths)
            and required_paths.issubset(cited_paths)
            and item.get("evidence_kind") == "artifact_inspection"
        )
    return True


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _required_source_lines(criterion: str) -> int | None:
    match = re.search(
        r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*(?P<suffix>[km])?\s*(?:loc|lines?)",
        criterion.casefold(),
    )
    if match is None:
        return None
    value = float(match.group("number").replace(",", ""))
    suffix = match.group("suffix")
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    return int(value)


def _is_test_review_assignment(
    tool_call: ToolCall,
    requirement: dict[str, str],
) -> bool:
    assignment = "\n".join(
        str(tool_call.arguments.get(key, ""))
        for key in ("task_label", "instructions", "deliverable")
    ).casefold()
    required_paths = _test_paths_from_requirement(requirement)
    return "review" in assignment and bool(required_paths) and all(
        path.casefold() in assignment or Path(path).name.casefold() in assignment
        for path in required_paths
    )


def _test_paths_from_requirement(requirement: dict[str, str]) -> set[str]:
    criterion = requirement.get("criterion", "")
    _, separator, raw_paths = criterion.partition(":")
    if not separator:
        return set()
    return {
        path.strip().removeprefix("/workspace/")
        for path in raw_paths.split(",")
        if path.strip()
    }


def _path_is_test_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/").casefold()
    name = normalized.rsplit("/", 1)[-1]
    wrapped = "/" + normalized.strip("/") + "/"
    return (
        "/tests/" in wrapped
        or "/test/" in wrapped
        or name.startswith("test_")
        or ".test." in name
        or ".spec." in name
    )


def _bounded_non_negative_int(value: object) -> int:
    if not isinstance(value, (int, str)):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _bounded_detail_map(
    value: object,
    *,
    limit: int = 64,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for raw_key, raw_details in list(value.items())[-limit:]:
        key = str(raw_key).strip()
        if key and isinstance(raw_details, dict):
            output[key] = dict(raw_details)
    return output


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
