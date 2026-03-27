"""Session routing for gateway websocket clients."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

from jarvis.core import AgentTurnResult, AgentTurnStreamEvent

from .route_events import RouteEvent

_ROUTE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class InvalidRouteIDError(ValueError):
    """Raised when a route id is invalid."""


class RouteRuntimeLike(Protocol):
    """Minimal protocol expected by SessionRouter."""

    async def enqueue_user_message(self, user_text: str) -> None:
        """Queue one user message for the persistent route runtime."""

    async def run_turn(self, user_text: str) -> AgentTurnResult:
        """Compatibility helper for one full main-agent turn."""

    async def stream_turn(self, user_text: str) -> AsyncIterator[AgentTurnStreamEvent]:
        """Compatibility helper for one streamed main-agent turn."""

    def active_session_id(self) -> str | None:
        """Return active main session id for this route."""

    def request_stop(self) -> bool:
        """Request cooperative stop for the active main turn, if any."""

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        """Resolve one pending approval request for the active route."""

    def subscribe(self) -> tuple[str, asyncio.Queue[RouteEvent]]:
        """Subscribe to route-scoped outbound events."""

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove one route-scoped outbound event subscription."""

    async def handle_user_input(self, user_text: str) -> AgentTurnResult:
        """Legacy compatibility helper for old fake loop tests."""

    async def stream_user_input(self, user_text: str) -> AsyncIterator[AgentTurnStreamEvent]:
        """Legacy compatibility helper for old fake loop tests."""


def validate_route_id(route_id: str) -> str:
    normalized = route_id.strip()
    if not _ROUTE_ID_PATTERN.fullmatch(normalized):
        raise InvalidRouteIDError(
            "route_id must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
        )
    return normalized


@dataclass(slots=True)
class RouteContext:
    runtime: RouteRuntimeLike
    lock: asyncio.Lock


class SessionRouter:
    """Maps inbound route ids to dedicated route runtimes."""

    def __init__(self, route_runtime_factory: Callable[[str], RouteRuntimeLike]) -> None:
        self._route_runtime_factory = route_runtime_factory
        self._routes: dict[str, RouteContext] = {}

    def get_or_create(self, route_id: str) -> RouteContext:
        validated = validate_route_id(route_id)
        context = self._routes.get(validated)
        if context is None:
            context = RouteContext(
                runtime=self._route_runtime_factory(validated),
                lock=asyncio.Lock(),
            )
            self._routes[validated] = context
        return context

    def active_session_id(self, route_id: str) -> str | None:
        return self.get_or_create(route_id).runtime.active_session_id()

    def request_stop(self, route_id: str) -> bool:
        return self.get_or_create(route_id).runtime.request_stop()

    def resolve_approval(self, route_id: str, approval_id: str, approved: bool) -> bool:
        return self.get_or_create(route_id).runtime.resolve_approval(approval_id, approved)

    def subscribe(self, route_id: str) -> tuple[str, asyncio.Queue[RouteEvent]]:
        return self.get_or_create(route_id).runtime.subscribe()

    def unsubscribe(self, route_id: str, subscriber_id: str) -> None:
        self.get_or_create(route_id).runtime.unsubscribe(subscriber_id)

    async def enqueue_message(self, route_id: str, user_text: str) -> None:
        await self.get_or_create(route_id).runtime.enqueue_user_message(user_text)

    async def run_turn(self, route_id: str, user_text: str) -> AgentTurnResult:
        context = self.get_or_create(route_id)
        async with context.lock:
            runtime = context.runtime
            if hasattr(runtime, "run_turn"):
                return await runtime.run_turn(user_text)
            return await runtime.handle_user_input(user_text)

    async def stream_turn(
        self,
        route_id: str,
        user_text: str,
    ) -> AsyncIterator[AgentTurnStreamEvent]:
        context = self.get_or_create(route_id)
        async with context.lock:
            runtime = context.runtime
            if hasattr(runtime, "stream_turn"):
                async for event in runtime.stream_turn(user_text):
                    yield event
                return
            async for event in runtime.stream_user_input(user_text):
                yield event
