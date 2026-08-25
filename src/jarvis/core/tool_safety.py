"""Task-scoped liveness accounting for tool activity."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal

from jarvis.llm import ToolCall
from jarvis.llm.validation import TOOL_CALL_VALIDATION_ERROR_METADATA_KEY
from jarvis.tools import ToolExecutionResult


_FILE_EDIT_TOOL_NAMES = frozenset({"file_patch", "file_write", "file_replace"})


@dataclass(slots=True, frozen=True)
class ToolActivityObservation:
    """Liveness result for one completed tool call."""

    repeated_invalid_call: bool = False
    blocked_invalid_signature: bool = False
    repeated_no_progress: bool = False
    made_progress: bool = False
    signature_id: str | None = None
    occurrence_count: int = 0
    progress_epoch: int = 0
    first_call_id: str | None = None


@dataclass(slots=True)
class ToolActivityTracker:
    """Track exact repeated calls and material progress.

    This tracker deliberately has no semantic-completion authority. It exists only to
    stop unchanged invalid/no-progress tool loops and to reset those counters when new
    runtime evidence appears.
    """

    _actor_kind: Literal["main", "subagent"] = "main"
    _invalid_counts: dict[str, int] = field(default_factory=dict)
    _no_progress_counts: dict[str, int] = field(default_factory=dict)
    _seen_activity_signatures: set[str] = field(default_factory=set)
    _invalid_first_call_ids: dict[str, str] = field(default_factory=dict)
    _no_progress_first_call_ids: dict[str, str] = field(default_factory=dict)
    _blocked_call_reasons: dict[str, str] = field(default_factory=dict)
    _blocked_call_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    _progress_epoch: int = 0
    _progress_since_slice: bool = False
    _runtime_progress_signatures: list[str] = field(default_factory=list)

    def blocked_call_reason(self, tool_call: ToolCall) -> str | None:
        return self._blocked_call_reasons.get(_tool_call_signature(tool_call))

    def blocked_call_details(self, tool_call: ToolCall) -> dict[str, Any]:
        return dict(self._blocked_call_details.get(_tool_call_signature(tool_call), {}))

    def record(self, tool_call: ToolCall, result: ToolExecutionResult) -> ToolActivityObservation:
        if _result_mutated_workspace(tool_call, result) or _result_is_material_progress(result):
            self._advance_progress_epoch()

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
            return ToolActivityObservation(
                repeated_invalid_call=blocked,
                blocked_invalid_signature=blocked,
                signature_id=invalid_signature,
                occurrence_count=count,
                progress_epoch=self._progress_epoch,
                first_call_id=first_call_id,
            )

        activity_signature = _tool_result_signature(tool_call, result)
        if _is_no_progress_result(result):
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
            return ToolActivityObservation(
                repeated_no_progress=repeated,
                made_progress=first_seen,
                signature_id=activity_signature,
                occurrence_count=count,
                progress_epoch=self._progress_epoch,
                first_call_id=first_call_id,
            )

        self._seen_activity_signatures.add(activity_signature)
        self._progress_since_slice = True
        return ToolActivityObservation(
            made_progress=True,
            progress_epoch=self._progress_epoch,
        )

    def _advance_progress_epoch(self) -> None:
        self._progress_epoch += 1
        self._invalid_counts.clear()
        self._no_progress_counts.clear()
        self._seen_activity_signatures.clear()
        self._invalid_first_call_ids.clear()
        self._no_progress_first_call_ids.clear()
        self._blocked_call_reasons.clear()
        self._blocked_call_details.clear()
        self._progress_since_slice = True

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
                "progress_fingerprints": metadata.get("bash_job_progress_fingerprints", []),
            }
        elif metadata.get("subagent_progress_update"):
            signature_payload = {
                "kind": "subagent",
                "subagent_id": metadata.get("subagent_id"),
                "notice_kind": metadata.get("subagent_notice_kind"),
                "pending_ids": metadata.get("pending_subagent_ids", []),
                "recommended_action": metadata.get("recommended_action"),
                "report_complete": metadata.get("latest_subagent_report_complete"),
                "changed_test_artifact_paths": metadata.get("changed_test_artifact_paths", []),
                "subagents": metadata.get("subagents", []),
                "content": content,
            }
        else:
            return False

        signature = _digest(signature_payload)
        if signature in self._runtime_progress_signatures:
            return False
        self._runtime_progress_signatures.append(signature)
        del self._runtime_progress_signatures[:-256]
        self._advance_progress_epoch()
        return True

    def consume_slice_progress(self) -> bool:
        made_progress = self._progress_since_slice
        self._progress_since_slice = False
        return made_progress

    def checkpoint_lines(self) -> tuple[str, ...]:
        """Return compact liveness state for an automatic continuation checkpoint."""

        return (
            f"progress_epoch: {self._progress_epoch}",
            "Semantic completion is not runtime-gated; choose the next action from the task state.",
        )

    def to_state(self) -> dict[str, Any]:
        """Return bounded durable liveness state."""

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
            "runtime_progress_signatures": list(self._runtime_progress_signatures),
        }

    @classmethod
    def from_state(
        cls,
        value: object,
        *,
        actor_kind: Literal["main", "subagent"] = "main",
    ) -> "ToolActivityTracker":
        """Restore bounded durable liveness state."""

        if not isinstance(value, dict):
            return cls(_actor_kind=actor_kind)
        return cls(
            _actor_kind=actor_kind,
            _invalid_counts=_bounded_count_map(value.get("invalid_counts")),
            _no_progress_counts=_bounded_count_map(value.get("no_progress_counts")),
            _seen_activity_signatures=_bounded_string_set(value.get("seen_activity_signatures")),
            _invalid_first_call_ids=_bounded_string_map(value.get("invalid_first_call_ids")),
            _no_progress_first_call_ids=_bounded_string_map(value.get("no_progress_first_call_ids")),
            _blocked_call_reasons=_bounded_string_map(value.get("blocked_call_reasons")),
            _blocked_call_details=_bounded_detail_map(value.get("blocked_call_details")),
            _progress_epoch=_bounded_non_negative_int(value.get("progress_epoch")),
            _progress_since_slice=bool(value.get("progress_since_slice", False)),
            _runtime_progress_signatures=_bounded_string_list(
                value.get("runtime_progress_signatures"),
                limit=256,
            ),
        )


def build_suppressed_repetition_result(
    *,
    tool_call: ToolCall,
    reason: str,
    diagnostics: dict[str, Any] | None = None,
) -> ToolExecutionResult:
    """Return deterministic feedback without re-executing an exact repeated action."""

    return ToolExecutionResult(
        call_id=tool_call.call_id,
        name=tool_call.name,
        ok=False,
        content=(
            "Exact tool call suppressed\n"
            f"tool: {tool_call.name}\n"
            "error_code: suppressed_repeated_tool_call\n"
            f"reason: {reason}. This exact action already crossed its unchanged-state retry "
            "limit. Continue the task by choosing a materially different action or gathering "
            "new evidence."
        ),
        metadata={
            "tool_liveness_suppressed": True,
            "tool_liveness_replan_required": True,
            "error_code": "suppressed_repeated_tool_call",
            "reason": reason,
            "suppressed_call_signature": _tool_call_signature(tool_call),
            "tool_liveness_diagnostics": dict(diagnostics or {}),
            "arguments": dict(tool_call.arguments),
        },
    )


def _invalid_signature(tool_call: ToolCall, result: ToolExecutionResult) -> str | None:
    metadata = result.metadata
    if not (
        not result.ok
        or metadata.get("tool_call_validation_failed")
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


def _tool_call_signature(tool_call: ToolCall) -> str:
    return _digest({"tool": tool_call.name, "arguments": _normalized_tool_arguments(tool_call)})


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


def _stable_mapping(value: dict[Any, Any], *, volatile_keys: set[str]) -> dict[str, Any]:
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
    if not result.ok or result.turn_disposition == "yield_turn":
        return False
    if result.name in _FILE_EDIT_TOOL_NAMES:
        return not bool(result.metadata.get("changed"))
    if result.name == "bash":
        return (
            not bool(result.metadata.get("workspace_changed", False))
            and result.metadata.get("mode") == "foreground"
            and result.metadata.get("exit_code") == 0
            and result.metadata.get("status") != "running"
        )
    return not bool(result.metadata.get("changed"))


def _result_mutated_workspace(tool_call: ToolCall, result: ToolExecutionResult) -> bool:
    if "workspace_changed" in result.metadata:
        return bool(result.metadata.get("workspace_changed"))
    if tool_call.name in _FILE_EDIT_TOOL_NAMES:
        return bool(result.metadata.get("changed"))
    if tool_call.name in {"generate_edit_image", "memory_write", "tool_register"}:
        return result.ok
    return False


def _result_is_material_progress(result: ToolExecutionResult) -> bool:
    if not result.ok:
        return False
    if bool(result.metadata.get("changed", False)):
        return True
    return result.name in {
        "subagent_invoke",
        "subagent_step_in",
        "subagent_stop",
        "subagent_dispose",
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


def changed_test_artifact_paths_from_result(result: ToolExecutionResult) -> tuple[str, ...]:
    """Return normalized test artifacts observed as changed by a tool result."""

    candidates = {
        str(result.metadata.get("path", "")).strip(),
        *(
            str(path).strip()
            for path in result.metadata.get("workspace_changed_paths", [])
            if str(path).strip()
        ),
    }
    return tuple(
        sorted(
            path.removeprefix("/workspace/")
            for path in candidates
            if path and _path_is_test_artifact(path)
        )
    )


def _bounded_count_map(value: object, *, limit: int = 64) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, int] = {}
    for raw_key, raw_count in list(value.items())[-limit:]:
        key = str(raw_key).strip()
        count = _bounded_non_negative_int(raw_count)
        if key and count:
            output[key] = count
    return output


def _bounded_string_set(value: object, *, limit: int = 64) -> set[str]:
    return set(_bounded_string_list(value, limit=limit))


def _bounded_string_list(value: object, *, limit: int = 64) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [normalized for item in list(value)[-limit:] if (normalized := str(item).strip())]


def _bounded_string_map(value: object, *, limit: int = 64) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, str] = {}
    for raw_key, raw_value in list(value.items())[-limit:]:
        key = str(raw_key).strip()
        normalized = str(raw_value).strip()
        if key and normalized:
            output[key] = normalized
    return output


def _bounded_non_negative_int(value: object) -> int:
    if not isinstance(value, (int, str)):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _bounded_detail_map(value: object, *, limit: int = 64) -> dict[str, dict[str, Any]]:
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
