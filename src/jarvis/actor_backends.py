"""Shared actor-runtime protocol and backend-selection helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, Protocol

from jarvis.core.agent_loop import (
    AgentRuntimeMessage,
    AgentTurnResult,
    AgentTurnStreamEvent,
    InterruptionReason,
)

ActorBackendKind = Literal["llm", "codex"]


class ActorRuntime(Protocol):
    """Minimal runtime surface shared by AgentLoop and Codex-backed actors."""

    async def aclose(self) -> None:
        """Release backend-specific resources held by this actor runtime."""

    async def handle_user_input(self, user_text: str) -> AgentTurnResult:
        """Run one parsed user input through the actor runtime."""

    async def stream_user_input(self, user_text: str) -> AsyncIterator[AgentTurnStreamEvent]:
        """Stream one parsed user input through the actor runtime."""

    async def stream_turn(
        self,
        *,
        user_text: str,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        """Stream one explicit user turn."""

    async def stream_runtime_turn(
        self,
        *,
        force_session_id: str | None = None,
        command_override: str | None = None,
        pre_turn_messages: Sequence[AgentRuntimeMessage] = (),
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        """Stream one runtime-initiated turn."""

    async def prepare_session(self, *, start_reason: str = "initial") -> str:
        """Ensure an active actor session exists and return its id."""

    def active_session_id(self) -> str | None:
        """Return the active persisted session id, if any."""

    def active_turn_id(self) -> str | None:
        """Return the active turn id, if any."""

    def has_active_turn(self) -> bool:
        """Return whether this actor currently has an in-flight turn."""

    def append_system_note(
        self,
        content: str,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Persist a system note for the next turn."""

    def request_stop(
        self,
        *,
        reason: InterruptionReason = "user_stop",
    ) -> bool:
        """Request cooperative interruption for the active turn, if any."""

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        """Resolve a pending Jarvis approval request."""


def backend_kind_for_provider(provider: str) -> ActorBackendKind:
    normalized = provider.strip().lower()
    if normalized == "codex":
        return "codex"
    return "llm"
