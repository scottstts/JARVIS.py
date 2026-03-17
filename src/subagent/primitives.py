"""Synthetic subagent control primitive definitions and runtime docs."""

from __future__ import annotations

from llm import ToolDefinition

SUBAGENT_PRIMITIVE_NAMES = (
    "subagent_invoke",
    "subagent_monitor",
    "subagent_stop",
    "subagent_step_in",
    "subagent_dispose",
)


def build_subagent_primitive_definitions() -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="subagent_invoke",
            description=(
                "Start a background subagent for a bounded side task. Use when the work is "
                "self-contained and can run independently while you continue supervising."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "instructions": {"type": "string"},
                    "context": {"type": "string"},
                    "deliverable": {"type": "string"},
                },
                "required": ["instructions"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="subagent_monitor",
            description=(
                "Inspect subagent status without changing it. Omit agent to get a summary "
                "of all non-disposed subagents."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "detail": {
                        "type": "string",
                        "enum": ["summary", "full"],
                    },
                },
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="subagent_stop",
            description=(
                "Request cooperative stop for a running or approval-blocked subagent so it "
                "settles into a paused state."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["agent"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="subagent_step_in",
            description=(
                "Stop a subagent if needed, wait for the turn to settle, then start a fresh "
                "subagent turn with updated direction."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "instructions": {"type": "string"},
                },
                "required": ["agent", "instructions"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="subagent_dispose",
            description=(
                "Permanently remove a non-running subagent from the active set and release "
                "its codename."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                },
                "required": ["agent"],
                "additionalProperties": False,
            },
        ),
    )


def render_subagent_primitive_docs() -> str:
    lines = [
        "Subagent control primitives are available only to Jarvis.",
        "Use them for bounded side work, monitor them, and dispose them when done.",
        "Subagents cannot spawn subagents.",
    ]
    for definition in build_subagent_primitive_definitions():
        lines.extend(
            [
                "",
                definition.name,
                definition.description or "",
                _render_schema_summary(definition),
            ]
        )
    return "\n".join(line for line in lines if line is not None)


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
