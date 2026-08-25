"""Passive self-check notes supplied to delegated subagents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class AcceptanceNotes:
    """Assignment reminders with no runtime completion semantics."""

    instructions: str
    user_constraints: str | None = None
    deliverable: str | None = None

    def render(self) -> str:
        lines = [
            "Acceptance Notes (informational only)",
            "Use these notes to self-check before handing work back. They never block handoff.",
            "You may return complete, partial, or blocked work; clearly report anything unverified or blocked.",
            "Assigned work:",
            self.instructions.strip(),
        ]
        if self.user_constraints is not None and self.user_constraints.strip():
            lines.extend(["", "Constraints to preserve:", self.user_constraints.strip()])
        if self.deliverable is not None and self.deliverable.strip():
            lines.extend(["", "Requested deliverable:", self.deliverable.strip()])
        lines.extend(
            [
                "",
                "Self-check where useful, but do not treat any specific verification command or environment failure as a runtime handoff requirement.",
            ]
        )
        return "\n".join(lines)
