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
                "for genuinely independent work; for coupled work, establish enough shared "
                "reality before broad fan-out, and launch work that depends on another child only "
                "after the prerequisite boundary is available and inspected. Supply a stable task "
                "label, explicit constraints, shared interfaces/environment, owned paths, a small "
                "seam contract describing how the work fits, and selected skill ids. The seam "
                "contract should cover ownership, canonical inputs, consumers, and important "
                "lifecycle/data-flow assumptions without prescribing implementation. If a required "
                "seam is missing, have the child surface it rather than silently inventing a "
                "competing substitute. Only Jarvis may select skills: name the exact top-level "
                "installed skill id, never infer one from SKILL.md, references, or other paths. A "
                "routing skill helps select expertise but does not replace relevant specialist "
                "skills. Skills opened by Jarvis in this turn are inherited automatically; "
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
                            "Relevant environment facts, canonical data, shared interfaces, "
                            "dependencies, downstream consumers, and coordination boundaries."
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
                            "work depends on. This records coordination knowledge; Jarvis decides "
                            "when the dependency is semantically satisfied."
                        ),
                    },
                    "seam_contract": {
                        "type": "string",
                        "description": (
                            "Optional minimal boundary contract: how the work fits, what the child "
                            "owns, consumes, and provides, which inputs are canonical, what downstream "
                            "consumers rely on, important lifecycle/data-flow assumptions, and how the "
                            "boundary will be checked. If the seam is inadequate, surface it instead "
                            "of inventing a competing substitute. Do not turn this into an implementation "
                            "plan."
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
                        "description": (
                            "Concrete local verification and handoff evidence Jarvis expects back; "
                            "this is not product acceptance."
                        ),
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
                "Cooperatively stop an existing subagent, wait for its turn to settle, then start "
                "a fresh turn with updated direction. Optionally replace the child's complete "
                "workspace write scope atomically; use owned_paths when a prior write denial says "
                "the child needs access, and use an empty list to continue explicitly read-only."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "instructions": {"type": "string"},
                    "owned_paths": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 32,
                        "uniqueItems": True,
                        "description": (
                            "Optional complete replacement for the child's workspace write scope; "
                            "omit to preserve the current scope, or pass [] for an explicit "
                            "read-only continuation. Paths must already exist before assignment."
                        ),
                    },
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
        "and a concrete local-evidence deliverable. For coupled work, establish enough shared "
        "reality before broad fan-out; stage dependent work after its prerequisite boundary is "
        "available; choose waves by dependency, not a fixed fan-out count. A child should surface "
        "an inadequate canonical seam instead of inventing a competing substitute. A routing skill "
        "does not replace relevant specialist skills. Never infer skills from referenced files. "
        "Continue independent work after invoking; do not poll.",
        f"- `{definitions['subagent_monitor'].name}`: inspect on demand. Omit `agent` to summarize all active "
        "subagents; use `detail=\"full\"` only when you need current internals.",
        f"- `{definitions['subagent_stop'].name}`: cooperatively pause a running or approval-blocked child.",
        f"- `{definitions['subagent_step_in'].name}`: stop, settle, and continue the same child; "
        "it is not live prompt injection. `owned_paths` replaces its scope when needed.",
        f"- `{definitions['subagent_handoff'].name}`: release a settled child's workspace lease while "
        "retaining its report and transcript for main-agent review and integration.",
        f"- `{definitions['subagent_dispose'].name}`: dispose completed, failed, or no-longer-needed children "
        "to free their slots.",
        f"- `{definitions['orchestrator_wait'].name}`: when route-owned work is still running but "
        "you have no actionable work, choose a liveness deadline and park. Material events wake "
        "you immediately; never create sleep jobs to poll.",
        "Subagents cannot spawn subagents.",
        "Review every completed implementation child's producer changes, seam, consumers, "
        "changed paths, validation, and assumptions before integrating; a report is evidence, not "
        "acceptance. Inspect paused, "
        "rejected, or failed children and decide whether to continue, revise, or stop.",
    ]
    return "\n".join(lines)
