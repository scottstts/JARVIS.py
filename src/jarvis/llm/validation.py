"""Tool-call parsing and schema validation utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .errors import LLMConfigurationError, ToolCallValidationError
from .types import ToolCall, ToolDefinition


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
        raise ToolCallValidationError(
            f"Tool '{name}' arguments failed schema validation: {exc.message}",
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
    validate_tool_call_arguments(
        call_id=call_id,
        name=name,
        arguments=parsed,
        schema=schema,
    )

    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=parsed,
        raw_arguments=raw_arguments,
    )
