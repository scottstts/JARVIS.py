"""Bootstrap prompt loading and directive message helpers for subagents."""

from __future__ import annotations

from pathlib import Path

from jarvis.core.agent_loop import AgentRuntimeMessage
from jarvis.llm import LLMMessage

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
    instructions: str,
    context: str | None = None,
    deliverable: str | None = None,
) -> LLMMessage:
    return LLMMessage.text(
        "system",
        _render_assignment_text(
            codename=codename,
            subagent_id=subagent_id,
            instructions=instructions,
            context=context,
            deliverable=deliverable,
        ),
    )


def build_step_in_message(*, instructions: str) -> AgentRuntimeMessage:
    return AgentRuntimeMessage(
        role="system",
        metadata={"subagent_step_in": True},
        content=(
            "Updated direction from Jarvis for the next turn.\n\n"
            f"{instructions.strip()}"
        ),
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
    instructions: str,
    context: str | None,
    deliverable: str | None,
) -> str:
    lines = [
        "Task assignment from Jarvis.",
        f"codename: {codename}",
        f"subagent_id: {subagent_id}",
        "instructions:",
        instructions.strip(),
    ]
    if context is not None and context.strip():
        lines.extend(
            [
                "",
                "context:",
                context.strip(),
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
    return "\n".join(lines)
