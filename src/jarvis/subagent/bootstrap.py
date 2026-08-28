"""Bootstrap prompt loading and directive message helpers for subagents."""

from __future__ import annotations

from pathlib import Path

from jarvis.core.agent_loop import AgentRuntimeMessage
from jarvis.llm import LLMMessage

from .acceptance_notes import AcceptanceNotes

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class SubagentBootstrapLoader:
    """Loads the static subagent prompt files plus an optional assignment message."""

    def __init__(self, *, assignment_message: LLMMessage | None = None) -> None:
        self._assignment_message = assignment_message

    def load_bootstrap_messages(self) -> list[LLMMessage]:
        messages = [
            LLMMessage.text("system", _read_prompt("SYSTEM.md")),
            LLMMessage.text("system", _read_prompt("OPERATING_RULES.md")),
        ]
        if self._assignment_message is not None:
            messages.append(self._assignment_message)
        return messages


def build_assignment_message(
    *,
    codename: str,
    subagent_id: str,
    task_label: str,
    instructions: str,
    user_constraints: str | None = None,
    shared_context: str | None = None,
    owned_paths: tuple[str, ...] = (),
    skill_documents: tuple[tuple[str, str], ...] = (),
    phase: str | None = None,
    depends_on: tuple[str, ...] = (),
    seam_contract: str | None = None,
    deliverable: str | None = None,
) -> LLMMessage:
    return LLMMessage.text(
        "system",
        _render_assignment_text(
            codename=codename,
            subagent_id=subagent_id,
            task_label=task_label,
            instructions=instructions,
            user_constraints=user_constraints,
            shared_context=shared_context,
            owned_paths=owned_paths,
            skill_documents=skill_documents,
            phase=phase,
            depends_on=depends_on,
            seam_contract=seam_contract,
            deliverable=deliverable,
        ),
    )


def build_step_in_message(
    *,
    instructions: str,
    owned_paths: tuple[str, ...] | None = None,
) -> AgentRuntimeMessage:
    content_lines = [
        "Updated direction from Jarvis for the next turn.",
        "",
        instructions.strip(),
    ]
    metadata: dict[str, object] = {"subagent_step_in": True}
    if owned_paths is not None:
        metadata["owned_paths_replaced"] = True
        metadata["owned_paths"] = list(owned_paths)
        if owned_paths:
            content_lines.extend(
                [
                    "",
                    "Your complete workspace write scope is now:",
                    *(f"- {path}" for path in owned_paths),
                ]
            )
        else:
            content_lines.extend(
                [
                    "",
                    "Your workspace write scope is now empty; continue read-only unless Jarvis assigns paths.",
                ]
            )
    return AgentRuntimeMessage(
        role="system",
        metadata=metadata,
        content="\n".join(content_lines),
    )


def build_subagent_kickoff_text() -> str:
    return "Start the assigned task now. Work until complete, blocked, or awaiting approval."


def _read_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Subagent prompt file is empty: {path}")
    return content


def _render_assignment_text(
    *,
    codename: str,
    subagent_id: str,
    task_label: str,
    instructions: str,
    user_constraints: str | None,
    shared_context: str | None,
    owned_paths: tuple[str, ...],
    skill_documents: tuple[tuple[str, str], ...],
    phase: str | None,
    depends_on: tuple[str, ...],
    seam_contract: str | None,
    deliverable: str | None,
) -> str:
    lines = [
        "Task assignment from Jarvis.",
        f"codename: {codename}",
        f"subagent_id: {subagent_id}",
        f"task_label: {task_label.strip()}",
        "instructions:",
        instructions.strip(),
    ]
    if user_constraints is not None and user_constraints.strip():
        lines.extend(
            [
                "",
                "user_constraints (preserve exactly; do not reinterpret):",
                user_constraints.strip(),
            ]
        )
    if shared_context is not None and shared_context.strip():
        lines.extend(
            [
                "",
                "shared_context (interfaces, environment, and coordination facts):",
                shared_context.strip(),
            ]
        )
    if owned_paths:
        lines.extend(["", "owned_paths:"])
        lines.extend(f"- {path}" for path in owned_paths)
    if phase or depends_on or (seam_contract is not None and seam_contract.strip()):
        lines.extend(["", "coordination:"])
        if phase:
            lines.append(f"phase: {phase.strip()}")
        if depends_on:
            lines.append("depends_on:")
            lines.extend(f"- {dependency}" for dependency in depends_on)
        if seam_contract is not None and seam_contract.strip():
            lines.extend(["seam_contract:", seam_contract.strip()])
    if skill_documents:
        lines.extend(
            [
                "",
                "selected_skills:",
                "Jarvis selected the following skills for this assignment. Follow them.",
            ]
        )
        for skill_id, document in skill_documents:
            lines.extend(
                [
                    "",
                    f"--- BEGIN SKILL {skill_id} ---",
                    document.strip(),
                    f"--- END SKILL {skill_id} ---",
                ]
            )
    if deliverable is not None and deliverable.strip():
        lines.extend(
            [
                "",
                "deliverable:",
                deliverable.strip(),
            ]
        )
    lines.extend(
        [
            "",
            AcceptanceNotes(
                instructions=instructions,
                user_constraints=user_constraints,
                deliverable=deliverable,
            ).render(),
        ]
    )
    return "\n".join(lines)
