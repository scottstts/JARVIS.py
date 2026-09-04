"""File-patch tool definition and execution runtime."""

from __future__ import annotations

import difflib
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from jarvis.llm import ToolDefinition

from ...config import ToolSettings
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult

_MAX_OPERATIONS = 32
_MAX_PATH_CHARS = 1_024
_MAX_OPERATION_TEXT_CHARS = 200_000
_MAX_DIAGNOSTIC_CANDIDATES = 3
_MAX_DIAGNOSTIC_LINE_CHARS = 240


class FilePatchError(RuntimeError):
    """Raised when a file patch cannot be applied safely."""

    error_code = "file_edit_failed"
    inspect_current_sha256 = True


class InvalidFilePreconditionError(FilePatchError):
    """Raised when mutually exclusive file-state preconditions are combined."""

    error_code = "invalid_file_precondition"
    inspect_current_sha256 = False


class FilePatchToolExecutor:
    """Applies structured text edits to a single workspace file."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        return _execute_file_edit(
            call_id=call_id,
            tool_name="file_patch",
            raw_path=str(arguments["path"]).strip(),
            operations=arguments.get("operations"),
            expected_sha256=arguments.get("expected_sha256"),
            expected_file_absent=arguments.get("expected_file_absent", False),
            context=context,
        )


class FileWriteToolExecutor:
    """Writes one complete UTF-8 file with a minimal flat argument shape."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        return _execute_file_edit(
            call_id=call_id,
            tool_name="file_write",
            raw_path=str(arguments["path"]).strip(),
            operations=[
                {
                    "type": "write",
                    "match": "",
                    "replacement": arguments["content"],
                }
            ],
            expected_sha256=arguments.get("expected_sha256"),
            expected_file_absent=arguments.get("expected_file_absent", False),
            context=context,
        )


class FileReplaceToolExecutor:
    """Replaces one exact text occurrence with a minimal flat argument shape."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        return _execute_file_edit(
            call_id=call_id,
            tool_name="file_replace",
            raw_path=str(arguments["path"]).strip(),
            operations=[
                {
                    "type": "replace",
                    "match": arguments["match"],
                    "replacement": arguments["replacement"],
                }
            ],
            expected_sha256=arguments.get("expected_sha256"),
            expected_file_absent=arguments.get("expected_file_absent", False),
            context=context,
        )


def build_file_patch_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the file_patch registry entry."""

    return RegisteredTool(
        name="file_patch",
        exposure="basic",
        definition=ToolDefinition(
            name="file_patch",
            description=_build_file_patch_tool_description(settings),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": _MAX_PATH_CHARS,
                        "description": "Workspace file to edit.",
                    },
                    "expected_sha256": {
                        "type": "string",
                        "minLength": 64,
                        "maxLength": 64,
                        "pattern": "^[0-9a-fA-F]{64}$",
                        "description": (
                            "Optional SHA-256 of the file content inspected before this edit. "
                            "The patch fails if the current file differs. Never invent a digest; "
                            "do not combine this with expected_file_absent=true."
                        ),
                    },
                    "expected_file_absent": _expected_file_absent_schema(),
                    "operations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_OPERATIONS,
                        "description": "Ordered literal-text edits for one file.",
                        "items": {
                            "type": "object",
                            "description": (
                                "Every operation uses type, match, and replacement. For write, "
                                "match must be empty and replacement is the full file. For "
                                "replace, match is replaced by replacement. For insert_before "
                                "or insert_after, match is the anchor and replacement is inserted. "
                                "For delete, match is removed and replacement must be empty."
                            ),
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "write",
                                        "replace",
                                        "insert_before",
                                        "insert_after",
                                        "delete",
                                    ],
                                    "description": "Edit kind.",
                                },
                                "match": {
                                    "type": "string",
                                    "maxLength": _MAX_OPERATION_TEXT_CHARS,
                                    "description": "Exact target text, or empty for write.",
                                },
                                "replacement": {
                                    "type": "string",
                                    "maxLength": _MAX_OPERATION_TEXT_CHARS,
                                    "description": (
                                        "New, inserted, or full-file text; empty for delete."
                                    ),
                                },
                            },
                            "required": ["type", "match", "replacement"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["path", "operations"],
                "additionalProperties": False,
            },
        ),
        executor=FilePatchToolExecutor(),
    )


def build_file_write_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the flat whole-file writing tool."""

    return RegisteredTool(
        name="file_write",
        exposure="basic",
        definition=ToolDefinition(
            name="file_write",
            description=(
                "Write the complete UTF-8 content of one workspace file atomically. "
                f"Only files inside {settings.workspace_dir} are allowed. Use this for "
                "new files or broad rewrites. expected_file_absent and expected_sha256 are "
                "mutually exclusive: use expected_file_absent=true alone for a verified new "
                "file, or pass only a real observed SHA-256; never invent a digest."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": _path_schema(),
                    "content": {
                        "type": "string",
                        "maxLength": _MAX_OPERATION_TEXT_CHARS,
                    },
                    "expected_sha256": _expected_sha256_schema(),
                    "expected_file_absent": _expected_file_absent_schema(),
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        ),
        executor=FileWriteToolExecutor(),
    )


def build_file_replace_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the flat one-replacement editing tool."""

    return RegisteredTool(
        name="file_replace",
        exposure="basic",
        definition=ToolDefinition(
            name="file_replace",
            description=(
                "Replace one unique exact text match in one workspace UTF-8 file. "
                f"Only files inside {settings.workspace_dir} are allowed. Use this for "
                "simple targeted edits. expected_file_absent and expected_sha256 are mutually "
                "exclusive: use expected_file_absent=true alone for a verified new file, or "
                "pass only a real observed SHA-256; never invent a digest."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": _path_schema(),
                    "match": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": _MAX_OPERATION_TEXT_CHARS,
                    },
                    "replacement": {
                        "type": "string",
                        "maxLength": _MAX_OPERATION_TEXT_CHARS,
                    },
                    "expected_sha256": _expected_sha256_schema(),
                    "expected_file_absent": _expected_file_absent_schema(),
                },
                "required": ["path", "match", "replacement"],
                "additionalProperties": False,
            },
        ),
        executor=FileReplaceToolExecutor(),
    )


def _path_schema() -> dict[str, object]:
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": _MAX_PATH_CHARS,
        "description": "Workspace file to edit.",
    }


def _expected_sha256_schema() -> dict[str, object]:
    return {
        "type": "string",
        "minLength": 64,
        "maxLength": 64,
        "pattern": "^[0-9a-fA-F]{64}$",
        "description": (
            "Optional SHA-256 observed before this edit. Never invent a digest; do not combine "
            "with expected_file_absent=true."
        ),
    }


def _expected_file_absent_schema() -> dict[str, object]:
    return {
        "type": "boolean",
        "description": (
            "Set true only when inspection established that the target does not exist. "
            "The write fails if a file appeared before execution. This is mutually exclusive "
            "with expected_sha256; for a new file, use this alone."
        ),
    }


def _build_file_patch_tool_description(settings: ToolSettings) -> str:
    return (
        "Apply structured text edits to exactly one workspace file. "
        f"Only files inside {settings.workspace_dir} are allowed. "
        "Use this instead of shell editing when you can express the change as explicit patch "
        "operations. Supported operations: write, replace, insert_before, insert_after, delete. "
        "For broad rewrites of long prose or document files, prefer a single write operation "
        "instead of many small operations. For small-to-medium targeted edits, prefer one "
        "file_patch call with a modest set of operations. Split edits across multiple "
        "file_patch calls only when one patch payload would otherwise become too large or "
        "unreliable. "
        "Matching is exact literal text only and edit operations fail when the target text is "
        "missing or ambiguous. The expected_sha256 and expected_file_absent preconditions are "
        "mutually exclusive: use expected_file_absent=true alone for a verified new file, or "
        "pass only a real observed SHA-256; never invent a digest. Example: "
        '{"path":"src/app.py","operations":[{"type":"replace",'
        '"match":"x = 1","replacement":"x = 2"}]}.'
    )


def _execute_file_edit(
    *,
    call_id: str,
    tool_name: str,
    raw_path: str,
    operations: object,
    expected_sha256: object,
    expected_file_absent: object,
    context: ToolExecutionContext,
) -> ToolExecutionResult:
    file_path = context.workspace_dir
    try:
        normalized_operations = _normalize_operations(operations)
        normalized_expected_sha256 = _normalize_expected_sha256(expected_sha256)
        if not isinstance(expected_file_absent, bool):
            raise FilePatchError("expected_file_absent must be a boolean when supplied.")
        _validate_file_preconditions(
            expected_sha256=normalized_expected_sha256,
            expected_file_absent=expected_file_absent,
        )
        file_path = _resolve_workspace_relative_path(raw_path, context)
        outcome = _apply_file_patch(
            file_path=file_path,
            operations=normalized_operations,
            expected_sha256=normalized_expected_sha256,
            expected_file_absent=expected_file_absent,
        )
    except FilePatchError as exc:
        return _file_patch_error(
            call_id=call_id,
            tool_name=tool_name,
            raw_path=raw_path,
            file_path=file_path,
            reason=str(exc),
            error_code=exc.error_code,
            inspect_current_sha256=exc.inspect_current_sha256,
        )

    content_lines = [
        "File patch applied" if tool_name == "file_patch" else "File edit applied",
        f"path: {file_path}",
        f"status: {outcome['status']}",
        f"operations_applied: {outcome['operations_applied']}",
        "operation_types: " + ", ".join(cast(list[str], outcome["operation_types"])),
        f"changed: {str(outcome['changed']).lower()}",
        f"bytes_written: {outcome['bytes_written']}",
        f"content_sha256: {outcome['content_sha256']}",
    ]
    return ToolExecutionResult(
        call_id=call_id,
        name=tool_name,
        ok=True,
        content="\n".join(content_lines),
        metadata={
            "path": str(file_path),
            "status": outcome["status"],
            "file_created": outcome["file_created"],
            "changed": outcome["changed"],
            "operations_applied": outcome["operations_applied"],
            "operation_types": list(cast(list[str], outcome["operation_types"])),
            "bytes_written": outcome["bytes_written"],
            "content_sha256": outcome["content_sha256"],
            "artifact_provenance": {
                "path": str(file_path),
                "tool_name": tool_name,
                "observed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "origin_session_id": context.session_id,
                "origin_turn_id": context.turn_id,
                "actor_kind": context.agent_kind,
                "actor_name": context.agent_name,
                "subagent_id": context.subagent_id,
                "content_sha256": outcome["content_sha256"],
            },
        },
    )


def _normalize_operations(raw_operations: object) -> list[dict[str, str]]:
    if not isinstance(raw_operations, list) or not raw_operations:
        raise FilePatchError("operations must be a non-empty array.")
    if len(raw_operations) > _MAX_OPERATIONS:
        raise FilePatchError(f"operations may not contain more than {_MAX_OPERATIONS} items.")

    normalized: list[dict[str, str]] = []
    for index, raw_operation in enumerate(raw_operations, start=1):
        if not isinstance(raw_operation, dict):
            raise FilePatchError(f"operation {index} must be an object.")

        operation_type = _require_non_empty_string(
            raw_operation.get("type"),
            field_name=f"operations[{index}].type",
        )

        if operation_type not in {
            "write",
            "replace",
            "insert_before",
            "insert_after",
            "delete",
        }:
            raise FilePatchError(
                f"operations[{index}].type '{operation_type}' is not supported."
            )

        match = _require_string(
            raw_operation.get("match"),
            field_name=f"operations[{index}].match",
        )
        replacement = _require_string(
            raw_operation.get("replacement"),
            field_name=f"operations[{index}].replacement",
        )
        if operation_type == "write" and match:
            raise FilePatchError(
                f"operations[{index}].match must be empty for write."
            )
        if operation_type != "write" and not match:
            raise FilePatchError(
                f"operations[{index}].match must not be empty for {operation_type}."
            )
        if operation_type == "delete" and replacement:
            raise FilePatchError(
                f"operations[{index}].replacement must be empty for delete."
            )
        if operation_type in {"insert_before", "insert_after"} and not replacement:
            raise FilePatchError(
                f"operations[{index}].replacement must not be empty for {operation_type}."
            )
        normalized.append(
            {
                "type": operation_type,
                "match": match,
                "replacement": replacement,
            }
        )

    write_count = sum(1 for operation in normalized if operation["type"] == "write")
    if write_count > 0 and len(normalized) != 1:
        raise FilePatchError("write must be the only operation in a file_patch call.")

    return normalized


def _normalize_expected_sha256(value: object) -> str | None:
    if value is None:
        return None
    digest = _require_non_empty_string(value, field_name="expected_sha256").lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise FilePatchError(
            "expected_sha256 must be a 64-character hexadecimal SHA-256."
        )
    return digest


def _require_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise FilePatchError(f"{field_name} must be a string.")
    return value


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    text = _require_string(value, field_name=field_name)
    if not text:
        raise FilePatchError(f"{field_name} must not be empty.")
    return text


def _validate_file_preconditions(
    *,
    expected_sha256: str | None,
    expected_file_absent: bool,
) -> None:
    if expected_file_absent and expected_sha256 is not None:
        raise InvalidFilePreconditionError(
            "expected_file_absent and expected_sha256 are mutually exclusive preconditions; "
            "use expected_file_absent=true alone for a verified missing file, or pass only a "
            "real observed SHA-256. Never invent a digest."
        )


def _apply_file_patch(
    *,
    file_path: Path,
    operations: list[dict[str, str]],
    expected_sha256: str | None,
    expected_file_absent: bool,
) -> dict[str, object]:
    _validate_file_preconditions(
        expected_sha256=expected_sha256,
        expected_file_absent=expected_file_absent,
    )
    parent_dir = file_path.parent
    if not parent_dir.exists():
        raise FilePatchError(
            f"parent directory does not exist: {parent_dir}"
        )
    if not parent_dir.is_dir():
        raise FilePatchError(
            f"parent path is not a directory: {parent_dir}"
        )
    if file_path.exists() and not file_path.is_file():
        raise FilePatchError("path must point to a regular file.")

    file_existed = file_path.exists()
    if expected_file_absent and file_existed:
        raise FilePatchError(
            "expected_file_absent precondition failed; a file now exists. Reread before retrying."
        )
    operation_types = tuple(operation["type"] for operation in operations)

    existing_content = ""
    existing_mode: int | None = None
    if file_existed:
        existing_content = _read_utf8_text(file_path)
        existing_mode = file_path.stat().st_mode
    existing_sha256 = _content_sha256(existing_content) if file_existed else None
    if expected_sha256 is not None and existing_sha256 != expected_sha256:
        actual = existing_sha256 or "(file does not exist)"
        raise FilePatchError(
            "expected_sha256 precondition failed; "
            f"current_sha256: {actual}. Reread the file before retrying."
        )

    if operations[0]["type"] == "write":
        final_content = operations[0]["replacement"]
    else:
        if not file_existed:
            raise FilePatchError(
                "target file does not exist; use a single write operation to create it."
            )

        final_content = existing_content
        for index, operation in enumerate(operations, start=1):
            final_content = _apply_operation(
                content=final_content,
                operation=operation,
                index=index,
            )

    changed = (not file_existed) or final_content != existing_content
    if changed:
        _write_text_atomically(
            file_path=file_path,
            content=final_content,
            existing_mode=existing_mode,
        )

    bytes_written = len(final_content.encode("utf-8"))
    content_sha256 = _content_sha256(final_content)
    if not file_existed:
        status = "created"
    elif changed:
        status = "updated"
    else:
        status = "unchanged"

    return {
        "status": status,
        "file_created": not file_existed,
        "changed": changed,
        "operations_applied": len(operations),
        "operation_types": operation_types,
        "bytes_written": bytes_written,
        "content_sha256": content_sha256,
    }


def _apply_operation(
    *,
    content: str,
    operation: dict[str, str],
    index: int,
) -> str:
    operation_type = operation["type"]

    if operation_type == "replace":
        match = operation["match"]
        match_index = _require_unique_match(
            content=content,
            needle=match,
            index=index,
            operation_type=operation_type,
        )
        return (
            content[:match_index]
            + operation["replacement"]
            + content[match_index + len(match) :]
        )

    if operation_type == "insert_before":
        match = operation["match"]
        match_index = _require_unique_match(
            content=content,
            needle=match,
            index=index,
            operation_type=operation_type,
        )
        return content[:match_index] + operation["replacement"] + content[match_index:]

    if operation_type == "insert_after":
        match = operation["match"]
        match_index = _require_unique_match(
            content=content,
            needle=match,
            index=index,
            operation_type=operation_type,
        )
        insert_at = match_index + len(match)
        return content[:insert_at] + operation["replacement"] + content[insert_at:]

    if operation_type == "delete":
        match = operation["match"]
        match_index = _require_unique_match(
            content=content,
            needle=match,
            index=index,
            operation_type=operation_type,
        )
        return content[:match_index] + content[match_index + len(match) :]

    raise FilePatchError(f"operation {index} has unsupported type '{operation_type}'.")


def _require_unique_match(
    *,
    content: str,
    needle: str,
    index: int,
    operation_type: str,
) -> int:
    matches = _find_all_match_indexes(content, needle)
    if not matches:
        candidates = _similar_line_candidates(content=content, needle=needle)
        candidate_text = (
            "\nnearest line candidates:\n" + "\n".join(candidates)
            if candidates
            else "\nnearest line candidates: none"
        )
        raise FilePatchError(
            f"operation {index} ({operation_type}) could not find a unique exact match."
            f"{candidate_text}"
        )
    if len(matches) > 1:
        locations = _exact_match_locations(
            content=content,
            match_indexes=matches,
        )
        raise FilePatchError(
            f"operation {index} ({operation_type}) matched multiple exact occurrences.\n"
            "exact match locations:\n"
            + "\n".join(locations)
        )
    return matches[0]


def _find_all_match_indexes(content: str, needle: str) -> list[int]:
    indexes: list[int] = []
    start = 0
    while True:
        match_index = content.find(needle, start)
        if match_index < 0:
            return indexes
        indexes.append(match_index)
        start = match_index + 1


def _similar_line_candidates(*, content: str, needle: str) -> list[str]:
    target = next(
        (line.strip() for line in needle.splitlines() if line.strip()),
        needle.strip(),
    )
    if not target:
        return []

    scored: list[tuple[float, int, str]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        candidate = line.strip()
        if not candidate:
            continue
        score = difflib.SequenceMatcher(
            None,
            target[:_MAX_DIAGNOSTIC_LINE_CHARS],
            candidate[:_MAX_DIAGNOSTIC_LINE_CHARS],
        ).ratio()
        if score >= 0.45:
            scored.append((score, line_number, line))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        _format_line_candidate(line_number=line_number, line=line)
        for _, line_number, line in scored[:_MAX_DIAGNOSTIC_CANDIDATES]
    ]


def _exact_match_locations(
    *,
    content: str,
    match_indexes: list[int],
) -> list[str]:
    lines = content.splitlines()
    locations: list[str] = []
    for match_index in match_indexes[:8]:
        line_number = content.count("\n", 0, match_index) + 1
        line = lines[line_number - 1] if line_number <= len(lines) else ""
        locations.append(
            _format_line_candidate(line_number=line_number, line=line)
        )
    if len(match_indexes) > 8:
        locations.append(f"... and {len(match_indexes) - 8} more")
    return locations


def _format_line_candidate(*, line_number: int, line: str) -> str:
    bounded = line
    if len(bounded) > _MAX_DIAGNOSTIC_LINE_CHARS:
        bounded = bounded[: _MAX_DIAGNOSTIC_LINE_CHARS - 3] + "..."
    return f"line {line_number}: {bounded!r}"


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_utf8_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FilePatchError(f"file is not valid UTF-8 text: {path}") from exc
    except OSError as exc:
        raise FilePatchError(f"failed to read file: {exc}") from exc


def _write_text_atomically(
    *,
    file_path: Path,
    content: str,
    existing_mode: int | None,
) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=file_path.parent,
            prefix=f".{file_path.name}.jarvis.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temp_path = Path(handle.name)

        if temp_path is None:
            raise FilePatchError("failed to prepare an atomic write target.")

        if existing_mode is not None:
            os.chmod(temp_path, existing_mode & 0o777)

        os.replace(temp_path, file_path)
    except OSError as exc:
        raise FilePatchError(f"failed to write file: {exc}") from exc
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _resolve_workspace_relative_path(raw_path: str, context: ToolExecutionContext) -> Path:
    candidate = Path(raw_path)
    workspace = context.workspace_dir.resolve(strict=False)
    if candidate.is_absolute():
        if candidate == Path("/workspace") or candidate.is_relative_to(Path("/workspace")):
            candidate = workspace / candidate.relative_to("/workspace")
    else:
        candidate = workspace / candidate
    resolved = candidate.resolve(strict=False)
    if resolved != workspace and not resolved.is_relative_to(workspace):
        raise FilePatchError("path must stay inside /workspace.")
    return resolved


def _file_patch_error(
    *,
    call_id: str,
    tool_name: str,
    raw_path: str,
    file_path: Path,
    reason: str,
    error_code: str = "file_edit_failed",
    inspect_current_sha256: bool = True,
) -> ToolExecutionResult:
    current_sha256 = (
        _current_file_sha256(file_path) if inspect_current_sha256 else None
    )
    current_sha256_display = (
        current_sha256
        or ("(not checked)" if not inspect_current_sha256 else "(file unavailable)")
    )
    next_action = (
        "Remove expected_sha256 and retry with expected_file_absent=true if inspection proved "
        "the file is absent. Otherwise retry with only a real observed expected_sha256."
        if error_code == "invalid_file_precondition"
        else "Reread the affected region and retry with current exact text. "
        "Do not repeat the unchanged failed operation."
    )
    return ToolExecutionResult(
        call_id=call_id,
        name=tool_name,
        ok=False,
        content=(
            "File patch failed\n"
            f"path: {raw_path}\n"
            f"reason: {reason}\n"
            f"current_sha256: {current_sha256_display}\n"
            f"next_action: {next_action}"
        ),
        metadata={
            "path": raw_path,
            "error_code": error_code,
            "error": reason,
            "current_sha256": current_sha256,
        },
    )


def _current_file_sha256(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
