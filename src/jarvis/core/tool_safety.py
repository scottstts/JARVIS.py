"""Turn-scoped safety accounting for repeated or non-progressing tool activity."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from jarvis.llm import ToolCall
from jarvis.llm.validation import TOOL_CALL_VALIDATION_ERROR_METADATA_KEY
from jarvis.tools import ToolExecutionResult


_FILE_EDIT_TOOL_NAMES = frozenset({"file_patch", "file_write", "file_replace"})


@dataclass(slots=True, frozen=True)
class ToolSafetyObservation:
    """Safety result for one completed tool call."""

    repeated_invalid_call: bool = False
    blocked_invalid_signature: bool = False
    repeated_no_progress: bool = False
    made_progress: bool = False


@dataclass(slots=True)
class ToolSafetyTracker:
    """Detect repeated invalid calls and identical no-progress tool activity."""

    _invalid_counts: dict[str, int] = field(default_factory=dict)
    _disabled_tools: set[str] = field(default_factory=set)
    _no_progress_counts: dict[str, int] = field(default_factory=dict)
    _seen_activity_signatures: set[str] = field(default_factory=set)
    _progress_since_slice: bool = False
    _workspace_mutated: bool = False
    _acceptance_recorded: bool = False
    _passed_acceptance_run_call_ids: set[str] = field(default_factory=set)
    _acceptance_items: dict[str, dict[str, Any]] = field(default_factory=dict)

    def invalid_call_is_blocked(self, tool_call: ToolCall) -> bool:
        return tool_call.name in self._disabled_tools

    def record(self, tool_call: ToolCall, result: ToolExecutionResult) -> ToolSafetyObservation:
        invalid_signature = _invalid_signature(tool_call, result)
        if invalid_signature is not None:
            count = self._invalid_counts.get(invalid_signature, 0) + 1
            self._invalid_counts[invalid_signature] = count
            blocked = count >= 2 and tool_call.name in _FILE_EDIT_TOOL_NAMES
            if blocked:
                self._disabled_tools.add(tool_call.name)
            return ToolSafetyObservation(
                repeated_invalid_call=count >= 2,
                blocked_invalid_signature=blocked,
            )

        if _result_mutated_workspace(tool_call, result):
            self._workspace_mutated = True
            self._acceptance_recorded = False
            self._passed_acceptance_run_call_ids.clear()
        if result.name == "acceptance_run" and result.ok:
            self._passed_acceptance_run_call_ids.add(result.call_id)
        if result.name == "acceptance_record" and result.ok:
            self._acceptance_items.update(_acceptance_ledger_items(result))
            self._acceptance_recorded = _acceptance_ledger_resolved(
                result,
                valid_gate_call_ids=self._passed_acceptance_run_call_ids,
                durable_items=self._acceptance_items,
            )

        activity_signature = _tool_result_signature(tool_call, result)
        no_progress = _is_no_progress_result(result)
        if no_progress:
            count = self._no_progress_counts.get(activity_signature, 0) + 1
            self._no_progress_counts[activity_signature] = count
            repeated = count >= 3
            first_seen = activity_signature not in self._seen_activity_signatures
            self._seen_activity_signatures.add(activity_signature)
            if first_seen:
                self._progress_since_slice = True
            return ToolSafetyObservation(
                repeated_no_progress=repeated,
                made_progress=first_seen,
            )

        self._seen_activity_signatures.add(activity_signature)
        self._progress_since_slice = True
        return ToolSafetyObservation(made_progress=True)

    def consume_slice_progress(self) -> bool:
        made_progress = self._progress_since_slice
        self._progress_since_slice = False
        return made_progress

    @property
    def unverified_workspace_mutation(self) -> bool:
        return self._workspace_mutated and not self._acceptance_recorded

    def to_state(self) -> dict[str, Any]:
        """Return the bounded durable state needed to resume safety accounting."""

        return {
            "invalid_counts": dict(self._invalid_counts),
            "disabled_tools": sorted(self._disabled_tools),
            "no_progress_counts": dict(self._no_progress_counts),
            "seen_activity_signatures": sorted(self._seen_activity_signatures),
            "progress_since_slice": self._progress_since_slice,
            "workspace_mutated": self._workspace_mutated,
            "acceptance_recorded": self._acceptance_recorded,
            "passed_acceptance_run_call_ids": sorted(
                self._passed_acceptance_run_call_ids
            ),
            "acceptance_items": dict(self._acceptance_items),
        }

    @classmethod
    def from_state(cls, value: object) -> "ToolSafetyTracker":
        """Restore validated state; malformed persisted data safely starts empty."""

        if not isinstance(value, dict):
            return cls()
        return cls(
            _invalid_counts=_bounded_count_map(value.get("invalid_counts")),
            _disabled_tools=_bounded_string_set(value.get("disabled_tools")),
            _no_progress_counts=_bounded_count_map(value.get("no_progress_counts")),
            _seen_activity_signatures=_bounded_string_set(
                value.get("seen_activity_signatures")
            ),
            _progress_since_slice=bool(value.get("progress_since_slice", False)),
            _workspace_mutated=bool(value.get("workspace_mutated", False)),
            _acceptance_recorded=bool(value.get("acceptance_recorded", False)),
            _passed_acceptance_run_call_ids=_bounded_string_set(
                value.get("passed_acceptance_run_call_ids")
            ),
            _acceptance_items=_bounded_acceptance_items(
                value.get("acceptance_items")
            ),
        )


def build_blocked_invalid_result(*, tool_call: ToolCall) -> ToolExecutionResult:
    """Return deterministic feedback without executing a third identical bad edit."""

    return ToolExecutionResult(
        call_id=tool_call.call_id,
        name=tool_call.name,
        ok=False,
        content=(
            "Tool call blocked\n"
            f"tool: {tool_call.name}\n"
            "error_code: repeated_invalid_tool_call\n"
            "reason: This file-edit tool already repeated the same invalid call twice and is "
            "disabled for the rest of this turn. Reread the target and use file_write or "
            "file_replace when their simpler shape fits."
        ),
        metadata={
            "tool_safety_blocked": True,
            "error_code": "repeated_invalid_tool_call",
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
            result.metadata.get("mode") == "foreground"
            and result.metadata.get("exit_code") == 0
            and result.metadata.get("status") != "running"
        )
    return not bool(result.metadata.get("changed"))


def _result_mutated_workspace(tool_call: ToolCall, result: ToolExecutionResult) -> bool:
    if not result.ok:
        return False
    if tool_call.name in _FILE_EDIT_TOOL_NAMES:
        return bool(result.metadata.get("changed"))
    if tool_call.name == "bash":
        return bool(tool_call.arguments.get("write_paths"))
    if tool_call.name == "acceptance_run":
        return bool(result.metadata.get("changed"))
    if tool_call.name in {"generate_edit_image", "memory_write", "tool_register"}:
        return True
    return False


def _acceptance_ledger_resolved(
    result: ToolExecutionResult,
    *,
    valid_gate_call_ids: set[str],
    durable_items: dict[str, dict[str, Any]],
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
    supplied_source_ids = {
        str(call_id)
        for check in checks
        if isinstance(check, dict)
        for call_id in check.get("source_tool_call_ids", [])
    }
    return bool(supplied_source_ids & valid_gate_call_ids)


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
        }
    return output


def _bounded_count_map(value: object, *, limit: int = 512) -> dict[str, int]:
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


def _bounded_string_set(value: object, *, limit: int = 512) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value[-limit:] if str(item).strip()}


def _bounded_acceptance_items(
    value: object,
    *,
    limit: int = 512,
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
        }
    return output


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
