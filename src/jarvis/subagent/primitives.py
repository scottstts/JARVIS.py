"""Synthetic subagent control primitive definitions and runtime docs."""

from __future__ import annotations

from jarvis.llm import ToolDefinition

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
                "Start a background subagent for bounded side work that can run independently "
                "while you supervise."
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
                "Inspect subagent status without changing it."
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
                "Request cooperative stop for a running or approval-blocked subagent."
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
                "Cooperatively stop a subagent, wait for the turn to settle, then start a fresh "
                "turn with updated direction."
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
                "Permanently remove a non-running subagent and release its codename."
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
    definitions = {definition.name: definition for definition in build_subagent_primitive_definitions()}
    lines = [
        "Subagent control primitives are available only to Jarvis.",
        "Use subagents only for bounded side work; Jarvis remains responsible for the final answer.",
        f"- `{definitions['subagent_invoke'].name}`: start bounded background side work. "
        "After invoking, wait for orchestrator updates before polling by default.",
        f"- `{definitions['subagent_monitor'].name}`: inspect on demand. Omit `agent` to summarize all active "
        "subagents; use `detail=\"full\"` only when you need current internals.",
        f"- `{definitions['subagent_stop'].name}`: cooperatively pause a running or approval-blocked child.",
        f"- `{definitions['subagent_step_in'].name}`: stop, settle, then start a new child turn with updated "
        "instructions; it is not live prompt injection.",
        f"- `{definitions['subagent_dispose'].name}`: dispose completed, failed, or no-longer-needed children "
        "to free their slots.",
        "Subagents cannot spawn subagents.",
    ]
    return "\n".join(lines)
