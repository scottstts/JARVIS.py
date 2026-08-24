"""Synthetic subagent control primitive definitions and runtime docs."""

from __future__ import annotations

from jarvis.llm import ToolDefinition

SUBAGENT_PRIMITIVE_NAMES = (
    "subagent_invoke",
    "subagent_monitor",
    "subagent_stop",
    "subagent_step_in",
    "subagent_dispose",
    "orchestrator_wait",
)


def build_subagent_primitive_definitions() -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="subagent_invoke",
            description=(
                "Start a background subagent for bounded side work that can run independently "
                "while you supervise. Supply a stable task label and only the context the child "
                "needs: explicit user constraints, shared interfaces/environment, owned paths, "
                "and selected skill ids. Only Jarvis may select skills: name the exact top-level "
                "installed skill id, never infer one from SKILL.md, references, or other paths. "
                "Skills opened by Jarvis in this turn are inherited automatically."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "task_label": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 120,
                        "description": "Short stable identity for the delegated task.",
                    },
                    "instructions": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Bounded work the child must perform.",
                    },
                    "user_constraints": {
                        "type": "string",
                        "description": "Exact user constraints relevant to this child.",
                    },
                    "shared_context": {
                        "type": "string",
                        "description": (
                            "Relevant environment facts, shared interfaces, dependencies, and "
                            "coordination boundaries."
                        ),
                    },
                    "owned_paths": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 32,
                        "uniqueItems": True,
                        "description": (
                            "Existing workspace files or directories the child exclusively owns "
                            "and may edit; create them before invoking the child."
                        ),
                    },
                    "skill_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 4,
                        "uniqueItems": True,
                        "description": (
                            "Exact top-level installed skill ids Jarvis selected; file or folder "
                            "names inside a skill are not skill ids."
                        ),
                    },
                    "deliverable": {
                        "type": "string",
                        "description": "Concrete completion evidence Jarvis expects back.",
                    },
                },
                "required": ["task_label", "instructions"],
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
        ToolDefinition(
            name="orchestrator_wait",
            description=(
                "Park Jarvis while route-owned subagents or detached jobs continue and there is "
                "no actionable main-agent work. Material actor events wake Jarvis immediately; "
                "wake_after_seconds is only a bounded liveness review deadline. Call this only "
                "after other actions, as the final control tool in the response."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "wake_after_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Preferred maximum seconds before a liveness review.",
                    },
                    "reason": {"type": "string", "minLength": 1, "maxLength": 240},
                    "watch_actor_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 32,
                        "uniqueItems": True,
                    },
                },
                "required": ["wake_after_seconds", "reason"],
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
        "Give it a stable `task_label`, explicit constraints, shared interfaces, owned paths, "
        "exact top-level `skill_ids` you selected, and a concrete deliverable. Never infer a "
        "skill from a referenced skill file or directory. Continue independent main-task work "
        "after invoking; do not poll, and let orchestrator updates drive supervision.",
        f"- `{definitions['subagent_monitor'].name}`: inspect on demand. Omit `agent` to summarize all active "
        "subagents; use `detail=\"full\"` only when you need current internals.",
        f"- `{definitions['subagent_stop'].name}`: cooperatively pause a running or approval-blocked child.",
        f"- `{definitions['subagent_step_in'].name}`: stop, settle, then start a new child turn with updated "
        "instructions; it is not live prompt injection.",
        f"- `{definitions['subagent_dispose'].name}`: dispose completed, failed, or no-longer-needed children "
        "to free their slots.",
        f"- `{definitions['orchestrator_wait'].name}`: when route-owned work is still running but "
        "you have no actionable work, choose a liveness deadline and park. Material events wake "
        "you immediately; never create sleep jobs to poll.",
        "Subagents cannot spawn subagents.",
        "A paused, rejected, or failed child requires inspection and a Jarvis decision; only a "
        "completed child with a complete report is ready to finalize.",
    ]
    return "\n".join(lines)
