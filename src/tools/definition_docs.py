"""Human-readable rendering helpers for tool definitions."""

from __future__ import annotations

from collections.abc import Sequence

from llm import ToolDefinition


def render_tool_definition_docs(
    definitions: Sequence[ToolDefinition],
    *,
    intro_lines: Sequence[str] = (),
) -> str:
    """Render tool definitions into concise starter-context style text."""

    lines = [line for line in intro_lines if line]
    if not definitions:
        if lines:
            lines.extend(("", "No tool definitions available."))
            return "\n".join(lines)
        return "No tool definitions available."

    for definition in definitions:
        if lines:
            lines.append("")
        lines.append(definition.name)
        description = (definition.description or "").strip()
        if description:
            lines.append(description)
        lines.append(_render_schema_summary(definition))
    return "\n".join(lines)


def _render_schema_summary(definition: ToolDefinition) -> str:
    schema = definition.input_schema
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = ["Arguments:"]
    for name, payload in properties.items():
        type_name = payload.get("type", "object")
        required_suffix = " required" if name in required else " optional"
        enum = payload.get("enum")
        if isinstance(enum, list) and enum:
            type_name = f"{type_name} ({', '.join(str(item) for item in enum)})"
        lines.append(f"- {name}: {type_name};{required_suffix}")
    if len(lines) == 1:
        lines.append("- none")
    return "\n".join(lines)
