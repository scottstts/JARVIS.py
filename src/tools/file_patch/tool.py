"""File-patch tool definition and execution runtime."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from llm import ToolDefinition

from ..config import ToolSettings
from ..types import RegisteredTool, ToolExecutionContext, ToolExecutionResult

_MAX_OPERATIONS = 32
_MAX_PATH_CHARS = 1_024
_MAX_OPERATION_TEXT_CHARS = 200_000


class FilePatchError(RuntimeError):
    """Raised when a file patch cannot be applied safely."""


class FilePatchToolExecutor:
    """Applies structured text edits to a single workspace file."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        raw_path = str(arguments["path"]).strip()
        file_path = _resolve_workspace_relative_path(raw_path, context)

        try:
            operations = _normalize_operations(arguments.get("operations"))
            outcome = _apply_file_patch(file_path=file_path, operations=operations)
        except FilePatchError as exc:
            return _file_patch_error(
                call_id=call_id,
                raw_path=raw_path,
                reason=str(exc),
            )

        content_lines = [
            "File patch applied",
            f"path: {file_path}",
            f"status: {outcome['status']}",
            f"operations_applied: {outcome['operations_applied']}",
            "operation_types: " + ", ".join(outcome["operation_types"]),
            f"changed: {str(outcome['changed']).lower()}",
            f"bytes_written: {outcome['bytes_written']}",
        ]

        return ToolExecutionResult(
            call_id=call_id,
            name="file_patch",
            ok=True,
            content="\n".join(content_lines),
            metadata={
                "path": str(file_path),
                "status": outcome["status"],
                "file_created": outcome["file_created"],
                "changed": outcome["changed"],
                "operations_applied": outcome["operations_applied"],
                "operation_types": list(outcome["operation_types"]),
                "bytes_written": outcome["bytes_written"],
            },
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
                        "description": (
                            "One workspace file path to edit. The file may be created or fully "
                            "overwritten with a single write operation."
                        ),
                    },
                    "operations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_OPERATIONS,
                        "description": (
                            "Ordered patch operations for exactly one file. Use literal text "
                            "matching only; regex and fuzzy matching are not supported."
                        ),
                        "items": {
                            "type": "object",
                            "description": (
                                "One operation object. Use type=write with content only; "
                                "type=replace with old/new; type=insert_before or "
                                "insert_after with anchor/text; type=delete with text."
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
                                    "description": "Operation kind.",
                                },
                                "content": {
                                    "type": "string",
                                    "maxLength": _MAX_OPERATION_TEXT_CHARS,
                                    "description": (
                                        "Required only for type=write. Replaces the full file "
                                        "content and may create the file."
                                    ),
                                },
                                "old": {
                                    "type": "string",
                                    "maxLength": _MAX_OPERATION_TEXT_CHARS,
                                    "description": (
                                        "Required only for type=replace. Exact literal text to replace."
                                    ),
                                },
                                "new": {
                                    "type": "string",
                                    "maxLength": _MAX_OPERATION_TEXT_CHARS,
                                    "description": (
                                        "Required only for type=replace. Replacement text."
                                    ),
                                },
                                "anchor": {
                                    "type": "string",
                                    "maxLength": _MAX_OPERATION_TEXT_CHARS,
                                    "description": (
                                        "Required only for type=insert_before or "
                                        "type=insert_after. Exact literal anchor text."
                                    ),
                                },
                                "text": {
                                    "type": "string",
                                    "maxLength": _MAX_OPERATION_TEXT_CHARS,
                                    "description": (
                                        "Required for type=insert_before, type=insert_after, "
                                        "and type=delete. Inserted or deleted exact literal text."
                                    ),
                                },
                            },
                            "required": ["type"],
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
        "missing or ambiguous."
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

        if operation_type == "write":
            normalized.append(
                {
                    "type": "write",
                    "content": _require_string(
                        raw_operation.get("content"),
                        field_name=f"operations[{index}].content",
                    ),
                }
            )
            continue

        if operation_type == "replace":
            old_text = _require_non_empty_string(
                raw_operation.get("old"),
                field_name=f"operations[{index}].old",
            )
            normalized.append(
                {
                    "type": "replace",
                    "old": old_text,
                    "new": _require_string(
                        raw_operation.get("new"),
                        field_name=f"operations[{index}].new",
                    ),
                }
            )
            continue

        if operation_type == "insert_before":
            normalized.append(
                {
                    "type": "insert_before",
                    "anchor": _require_non_empty_string(
                        raw_operation.get("anchor"),
                        field_name=f"operations[{index}].anchor",
                    ),
                    "text": _require_non_empty_string(
                        raw_operation.get("text"),
                        field_name=f"operations[{index}].text",
                    ),
                }
            )
            continue

        if operation_type == "insert_after":
            normalized.append(
                {
                    "type": "insert_after",
                    "anchor": _require_non_empty_string(
                        raw_operation.get("anchor"),
                        field_name=f"operations[{index}].anchor",
                    ),
                    "text": _require_non_empty_string(
                        raw_operation.get("text"),
                        field_name=f"operations[{index}].text",
                    ),
                }
            )
            continue

        if operation_type == "delete":
            normalized.append(
                {
                    "type": "delete",
                    "text": _require_non_empty_string(
                        raw_operation.get("text"),
                        field_name=f"operations[{index}].text",
                    ),
                }
            )
            continue

        raise FilePatchError(
            f"operations[{index}].type '{operation_type}' is not supported."
        )

    write_count = sum(1 for operation in normalized if operation["type"] == "write")
    if write_count > 0 and len(normalized) != 1:
        raise FilePatchError("write must be the only operation in a file_patch call.")

    return normalized


def _require_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise FilePatchError(f"{field_name} must be a string.")
    return value


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    text = _require_string(value, field_name=field_name)
    if not text:
        raise FilePatchError(f"{field_name} must not be empty.")
    return text


def _apply_file_patch(
    *,
    file_path: Path,
    operations: list[dict[str, str]],
) -> dict[str, object]:
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
    operation_types = tuple(operation["type"] for operation in operations)

    existing_content = ""
    existing_mode: int | None = None
    if file_existed:
        existing_content = _read_utf8_text(file_path)
        existing_mode = file_path.stat().st_mode

    if operations[0]["type"] == "write":
        final_content = operations[0]["content"]
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
    }


def _apply_operation(
    *,
    content: str,
    operation: dict[str, str],
    index: int,
) -> str:
    operation_type = operation["type"]

    if operation_type == "replace":
        old_text = operation["old"]
        match_index = _require_unique_match(
            content=content,
            needle=old_text,
            index=index,
            operation_type=operation_type,
            label="old",
        )
        return (
            content[:match_index]
            + operation["new"]
            + content[match_index + len(old_text) :]
        )

    if operation_type == "insert_before":
        anchor = operation["anchor"]
        match_index = _require_unique_match(
            content=content,
            needle=anchor,
            index=index,
            operation_type=operation_type,
            label="anchor",
        )
        return content[:match_index] + operation["text"] + content[match_index:]

    if operation_type == "insert_after":
        anchor = operation["anchor"]
        match_index = _require_unique_match(
            content=content,
            needle=anchor,
            index=index,
            operation_type=operation_type,
            label="anchor",
        )
        insert_at = match_index + len(anchor)
        return content[:insert_at] + operation["text"] + content[insert_at:]

    if operation_type == "delete":
        text = operation["text"]
        match_index = _require_unique_match(
            content=content,
            needle=text,
            index=index,
            operation_type=operation_type,
            label="text",
        )
        return content[:match_index] + content[match_index + len(text) :]

    raise FilePatchError(f"operation {index} has unsupported type '{operation_type}'.")


def _require_unique_match(
    *,
    content: str,
    needle: str,
    index: int,
    operation_type: str,
    label: str,
) -> int:
    matches = _find_all_match_indexes(content, needle)
    if not matches:
        raise FilePatchError(
            f"operation {index} ({operation_type}) could not find a unique {label} match."
        )
    if len(matches) > 1:
        raise FilePatchError(
            f"operation {index} ({operation_type}) matched multiple {label} occurrences."
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
    if not candidate.is_absolute():
        candidate = context.workspace_dir / candidate
    return candidate.resolve(strict=False)


def _file_patch_error(
    *,
    call_id: str,
    raw_path: str,
    reason: str,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        call_id=call_id,
        name="file_patch",
        ok=False,
        content=(
            "File patch failed\n"
            f"path: {raw_path}\n"
            f"reason: {reason}"
        ),
        metadata={
            "path": raw_path,
            "error": reason,
        },
    )
