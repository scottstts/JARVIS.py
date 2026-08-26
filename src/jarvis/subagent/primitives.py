"""Synthetic subagent control primitive definitions and runtime docs."""

from __future__ import annotations

from jarvis.llm import ToolDefinition

SUBAGENT_PRIMITIVE_NAMES = (
    "subagent_invoke",
    "subagent_monitor",
    "subagent_stop",
    "subagent_step_in",
    "subagent_handoff",
    "subagent_dispose",
    "orchestrator_wait",
)


def build_subagent_primitive_definitions() -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="subagent_invoke",
            description=(
                "Start a background subagent for bounded side work while you supervise. Use it "
                "for genuinely independent work; if the work depends on another child, launch it "
                "after the prerequisite boundary is available and inspected. Supply a stable task "
                "label, explicit constraints, shared interfaces/environment, owned paths, a small "
                "seam contract, and selected skill ids. Only Jarvis may select skills: name the "
                "exact top-level installed skill id, never infer one from SKILL.md, references, or "
                "other paths. Skills opened by Jarvis in this turn are inherited automatically; "
                "explicitly repeat skill ids for later turns."
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
                    "phase": {
                        "type": "string",
                        "maxLength": 80,
                        "description": (
                            "Optional coordination phase, such as foundation, feature, integration, "
                            "or review. This is recorded for supervision and does not schedule work."
                        ),
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 16,
                        "uniqueItems": True,
                        "description": (
                            "Optional task labels, codenames, or subagent ids whose boundary this "
                            "work depends on. Jarvis decides when the dependency is satisfied."
                        ),
                    },
                    "seam_contract": {
                        "type": "string",
                        "description": (
                            "Optional minimal boundary contract: what the child owns, consumes, "
                            "provides, assumes about lifecycle, and how the boundary will be checked. "
                            "Do not turn this into an implementation plan."
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
                "Inspect subagent status without changing it. A child report is self-reported "
                "evidence, not semantic acceptance. Use full detail to review the assignment, "
                "seam, changed paths, validation scope, and remaining assumptions before "
                "integrating code-producing work."
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
            name="subagent_handoff",
            description=(
                "Release a settled subagent's workspace write lease while preserving "
                "its transcript, report, and monitoring state so Jarvis can inspect and integrate "
                "the work. Stop the child first if it is still running. This does not decide whether "
                "the work is correct and does not dispose the child."
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
        "a minimal seam contract, phase/dependencies when useful, exact top-level `skill_ids`, "
        "and a concrete deliverable. Stage dependent work after its prerequisite boundary is "
        "available; choose waves by dependency, not a fixed fan-out count. Never infer skills "
        "from referenced files. Continue independent work after invoking; do not poll.",
        f"- `{definitions['subagent_monitor'].name}`: inspect on demand. Omit `agent` to summarize all active "
        "subagents; use `detail=\"full\"` only when you need current internals.",
        f"- `{definitions['subagent_stop'].name}`: cooperatively pause a running or approval-blocked child.",
        f"- `{definitions['subagent_step_in'].name}`: stop, settle, then start a new child turn with updated "
        "instructions; it is not live prompt injection.",
        f"- `{definitions['subagent_handoff'].name}`: release a settled child's workspace lease while "
        "retaining its report and transcript for main-agent review and integration.",
        f"- `{definitions['subagent_dispose'].name}`: dispose completed, failed, or no-longer-needed children "
        "to free their slots.",
        f"- `{definitions['orchestrator_wait'].name}`: when route-owned work is still running but "
        "you have no actionable work, choose a liveness deadline and park. Material events wake "
        "you immediately; never create sleep jobs to poll.",
        "Subagents cannot spawn subagents.",
        "Review every completed implementation child's changed paths, seam, validation, and "
        "assumptions before integrating; a report is evidence, not acceptance. Inspect paused, "
        "rejected, or failed children and decide whether to continue, revise, or stop.",
    ]
    return "\n".join(lines)
