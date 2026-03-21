"""Synthetic subagent control primitive definitions and runtime docs."""

from __future__ import annotations

from llm import ToolDefinition
from tools.definition_docs import render_tool_definition_docs

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
                "self-contained and can run independently while you continue supervising. "
                "After invocation, let the orchestrator surface meaningful progress instead of "
                "polling the child."
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
                "of all non-disposed subagents. Use this on demand when immediate detail is "
                "required; do not poll unchanged subagents while orchestration updates are flowing."
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
    return render_tool_definition_docs(
        build_subagent_primitive_definitions(),
        intro_lines=(
            "Subagent control primitives are available only to Jarvis.",
            "Use them for bounded side work, monitor them, and dispose them when done.",
            "Subagents cannot spawn subagents.",
        ),
    )
