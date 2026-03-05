"""Session routing for gateway websocket clients."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from core import AgentTurnResult, AgentTurnStreamEvent

_ROUTE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class InvalidRouteIDError(ValueError):
    """Raised when a route id is invalid."""


class AgentLoopLike(Protocol):
    """Minimal protocol expected by SessionRouter."""

    async def handle_user_input(self, user_text: str) -> AgentTurnResult:
        """Process one user turn and return assistant output."""

    async def stream_user_input(self, user_text: str) -> AsyncIterator[AgentTurnStreamEvent]:
        """Process one user turn and stream delta/done events."""

    def active_session_id(self) -> str | None:
        """Return active session id for this route."""


def validate_route_id(route_id: str) -> str:
    normalized = route_id.strip()
    if not _ROUTE_ID_PATTERN.fullmatch(normalized):
        raise InvalidRouteIDError(
            "route_id must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
        )
    return normalized


@dataclass(slots=True)
class RouteContext:
    agent_loop: AgentLoopLike
    lock: asyncio.Lock


class SessionRouter:
    """Maps inbound route ids to dedicated AgentLoop instances."""

    def __init__(self, agent_loop_factory: Callable[[str], AgentLoopLike]) -> None:
        self._agent_loop_factory = agent_loop_factory
        self._routes: dict[str, RouteContext] = {}

    def get_or_create(self, route_id: str) -> RouteContext:
        validated = validate_route_id(route_id)
        context = self._routes.get(validated)
        if context is None:
            context = RouteContext(
                agent_loop=self._agent_loop_factory(validated),
                lock=asyncio.Lock(),
            )
            self._routes[validated] = context
        return context

    def active_session_id(self, route_id: str) -> str | None:
        context = self.get_or_create(route_id)
        return context.agent_loop.active_session_id()

    async def run_turn(self, route_id: str, user_text: str) -> AgentTurnResult:
        context = self.get_or_create(route_id)
        async with context.lock:
            return await context.agent_loop.handle_user_input(user_text)

    async def stream_turn(
        self,
        route_id: str,
        user_text: str,
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        context = self.get_or_create(route_id)
        async with context.lock:
            async for event in context.agent_loop.stream_user_input(user_text):
                yield event
