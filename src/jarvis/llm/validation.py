"""Tool-call parsing and schema validation utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .errors import LLMConfigurationError, ToolCallValidationError
from .types import ToolCall, ToolDefinition

TOOL_CALL_VALIDATION_ERROR_METADATA_KEY = "tool_call_validation_error"


def build_tool_schema_map(tools: Sequence[ToolDefinition]) -> dict[str, Mapping[str, Any]]:
    """Builds name->schema map and rejects duplicate tool names."""
    schemas: dict[str, Mapping[str, Any]] = {}
    for tool in tools:
        if tool.name in schemas:
            raise LLMConfigurationError(f"Duplicate tool name: {tool.name}")
        schemas[tool.name] = tool.input_schema
    return schemas


def get_tool_schema(
    *,
    call_id: str,
    name: str,
    tool_schemas: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Looks up a tool schema and raises a consistent error if missing."""
    schema = tool_schemas.get(name)
    if schema is None:
        raise ToolCallValidationError(
            f"Model called unknown tool '{name}'.",
            tool_name=name,
            call_id=call_id,
        )
    return schema


def load_tool_call_arguments(
    *,
    call_id: str,
    name: str,
    raw_arguments: str,
) -> dict[str, Any]:
    """Parses raw tool-call arguments into a JSON object."""
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ToolCallValidationError(
            f"Invalid JSON for tool '{name}': {exc.msg}",
            tool_name=name,
            call_id=call_id,
        ) from exc

    if not isinstance(parsed, dict):
        raise ToolCallValidationError(
            f"Tool '{name}' arguments must decode to a JSON object.",
            tool_name=name,
            call_id=call_id,
        )

    return parsed


def validate_tool_call_arguments(
    *,
    call_id: str,
    name: str,
    arguments: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    """Validates parsed tool-call arguments against a JSON schema."""
    if not isinstance(arguments, dict):
        raise ToolCallValidationError(
            f"Tool '{name}' arguments must decode to a JSON object.",
            tool_name=name,
            call_id=call_id,
        )

    try:
        Draft202012Validator(schema).validate(arguments)
    except ValidationError as exc:
        instance_path = _format_json_path(exc.absolute_path)
        schema_path = _format_json_path(exc.absolute_schema_path, root="$schema")
        received = _format_validation_value(exc.instance)
        raise ToolCallValidationError(
            f"Tool '{name}' arguments failed schema validation at {instance_path}: "
            f"{exc.message}. Received: {received}. Schema rule: {schema_path}.",
            tool_name=name,
            call_id=call_id,
        ) from exc


def parse_and_validate_tool_call(
    *,
    call_id: str,
    name: str,
    raw_arguments: str,
    tool_schemas: Mapping[str, Mapping[str, Any]],
) -> ToolCall:
    """Parses JSON args and validates them against the tool JSON schema."""
    schema = get_tool_schema(
        call_id=call_id,
        name=name,
        tool_schemas=tool_schemas,
    )
    parsed = load_tool_call_arguments(
        call_id=call_id,
        name=name,
        raw_arguments=raw_arguments,
    )
    normalized_arguments, normalization = _normalize_legacy_tool_arguments(
        name=name,
        arguments=parsed,
    )
    validate_tool_call_arguments(
        call_id=call_id,
        name=name,
        arguments=normalized_arguments,
        schema=schema,
    )

    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=normalized_arguments,
        raw_arguments=raw_arguments,
        provider_metadata=(
            {"tool_call_normalized": normalization} if normalization is not None else {}
        ),
    )


def parse_and_validate_tool_call_or_recover(
    *,
    call_id: str,
    name: str,
    raw_arguments: str,
    tool_schemas: Mapping[str, Mapping[str, Any]],
    provider_metadata: Mapping[str, Any] | None = None,
) -> ToolCall:
    """Parses and validates a tool call, or returns a recoverable invalid call."""
    try:
        tool_call = parse_and_validate_tool_call(
            call_id=call_id,
            name=name,
            raw_arguments=raw_arguments,
            tool_schemas=tool_schemas,
        )
    except ToolCallValidationError as exc:
        return build_recoverable_invalid_tool_call(
            call_id=call_id,
            name=name,
            raw_arguments=raw_arguments,
            error=exc,
            provider_metadata=provider_metadata,
        )
    if not provider_metadata:
        return tool_call
    merged_metadata = dict(provider_metadata)
    merged_metadata.update(tool_call.provider_metadata)
    return ToolCall(
        call_id=tool_call.call_id,
        name=tool_call.name,
        arguments=tool_call.arguments,
        raw_arguments=tool_call.raw_arguments,
        provider_metadata=merged_metadata,
    )


def build_recoverable_invalid_tool_call(
    *,
    call_id: str,
    name: str,
    raw_arguments: str,
    error: ToolCallValidationError,
    provider_metadata: Mapping[str, Any] | None = None,
) -> ToolCall:
    """Builds a synthetic tool call that will surface a recoverable tool error."""
    metadata = dict(provider_metadata or {})
    metadata[TOOL_CALL_VALIDATION_ERROR_METADATA_KEY] = str(error)
    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=_best_effort_load_tool_call_arguments(raw_arguments),
        raw_arguments=raw_arguments,
        provider_metadata=metadata,
    )


def _best_effort_load_tool_call_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _normalize_legacy_tool_arguments(
    *,
    name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply only unambiguous compatibility normalization for legacy file patches."""

    if name != "file_patch":
        return arguments, None
    raw_operations = arguments.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        return arguments, None
    if not all(isinstance(item, dict) for item in raw_operations):
        return arguments, None

    normalized = dict(arguments)
    operations = [dict(item) for item in raw_operations]
    changes: list[str] = []
    root_path = normalized.get("path")
    nested_paths = [
        str(operation["path"]).strip()
        for operation in operations
        if isinstance(operation.get("path"), str) and str(operation["path"]).strip()
    ]
    if (
        root_path is None
        and len(nested_paths) == len(operations)
        and len(set(nested_paths)) == 1
    ):
        root_path = nested_paths[0]
        normalized["path"] = root_path
        changes.append("hoisted identical operation paths")

    for operation in operations:
        if root_path is not None and operation.get("path") == root_path:
            operation.pop("path", None)
            changes.append("removed duplicated operation path")
        if "type" in operation:
            continue
        aliases = [
            alias
            for alias in ("write", "replace", "insert_before", "insert_after", "delete")
            if isinstance(operation.get(alias), str)
        ]
        if len(aliases) != 1:
            continue
        alias = aliases[0]
        if "match" in operation:
            continue
        operation["type"] = alias
        alias_value = operation.pop(alias)
        if alias == "write":
            operation["match"] = ""
            operation.setdefault("replacement", alias_value)
        else:
            operation["match"] = alias_value
            if alias == "delete":
                operation.setdefault("replacement", "")
        changes.append(f"inferred operation type '{alias}'")

    if not changes:
        return arguments, None
    normalized["operations"] = operations
    return normalized, {"kind": "legacy_file_patch", "changes": sorted(set(changes))}


def _format_json_path(path: Sequence[object], *, root: str = "$") -> str:
    rendered = root
    for component in path:
        if isinstance(component, int):
            rendered += f"[{component}]"
            continue
        name = str(component)
        if name.isidentifier():
            rendered += f".{name}"
        else:
            rendered += "[" + json.dumps(name, ensure_ascii=False) + "]"
    return rendered


def _format_validation_value(value: Any, *, max_chars: int = 240) -> str:
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 15].rstrip() + "...[truncated]"
