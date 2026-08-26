"""Session routing for gateway websocket clients."""

from __future__ import annotations

import asyncio
from inspect import isawaitable
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

from jarvis.core import AgentTurnResult, AgentTurnStreamEvent
from jarvis.logging_setup import get_application_logger

from .route_events import RouteEvent

_ROUTE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
LOGGER = get_application_logger(__name__)


class InvalidRouteIDError(ValueError):
    """Raised when a route id is invalid."""


class RouteRuntimeLike(Protocol):
    """Minimal protocol expected by SessionRouter."""

    async def enqueue_user_message(
        self,
        user_text: str,
        *,
        client_message_id: str | None = None,
    ) -> None:
        """Queue one user message for the persistent route runtime."""

    async def run_turn(self, user_text: str) -> AgentTurnResult:
        """Compatibility helper for one full main-agent turn."""

    async def stream_turn(self, user_text: str) -> AsyncIterator[AgentTurnStreamEvent]:
        """Compatibility helper for one streamed main-agent turn."""

    def active_session_id(self) -> str | None:
        """Return active main session id for this route."""

    async def request_stop(self) -> bool:
        """Request stop for active route work, if any."""
        ...

    async def initialize(self) -> None:
        """Restore durable route-owned state before accepting work."""
        ...

    async def graceful_shutdown(self) -> bool:
        """Gracefully quiesce this route before process shutdown."""
        ...

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
        self._shutdown_lock = asyncio.Lock()
        self._shutdown_started = False
        self._shutdown_complete = False

    def get_or_create(self, route_id: str) -> RouteContext:
        validated = validate_route_id(route_id)
        context = self._routes.get(validated)
        if context is None:
            if self._shutdown_started:
                raise RuntimeError("Session router is shutting down.")
            context = RouteContext(
                runtime=self._route_runtime_factory(validated),
                lock=asyncio.Lock(),
            )
            self._routes[validated] = context
        return context

    def active_session_id(self, route_id: str) -> str | None:
        return self.get_or_create(route_id).runtime.active_session_id()

    async def initialize(self, route_id: str) -> None:
        self._ensure_accepting_work()
        context = self.get_or_create(route_id)
        initializer = getattr(context.runtime, "initialize", None)
        if callable(initializer):
            result = initializer()
            if isawaitable(result):
                await result

    async def request_stop(self, route_id: str) -> bool:
        if self._shutdown_started:
            return False
        return await self.get_or_create(route_id).runtime.request_stop()

    async def graceful_shutdown(self) -> None:
        """Gracefully quiesce every route that exists in this process."""
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._shutdown_started = True
            contexts = tuple(self._routes.values())
            results = await asyncio.gather(
                *(self._graceful_shutdown_route(context.runtime) for context in contexts),
                return_exceptions=True,
            )
            shutdown_failed = False
            for context, result in zip(contexts, results, strict=True):
                if isinstance(result, BaseException):
                    shutdown_failed = True
                    LOGGER.exception(
                        "Graceful shutdown failed for a route runtime.",
                        exc_info=(type(result), result, result.__traceback__),
                    )
            self._shutdown_complete = not shutdown_failed

    @staticmethod
    async def _graceful_shutdown_route(runtime: RouteRuntimeLike) -> bool:
        shutdown = getattr(runtime, "graceful_shutdown", None)
        if callable(shutdown):
            result = shutdown()
            if isawaitable(result):
                return bool(await result)
            return bool(result)
        return bool(await runtime.request_stop())

    def resolve_approval(self, route_id: str, approval_id: str, approved: bool) -> bool:
        if self._shutdown_started:
            return False
        return self.get_or_create(route_id).runtime.resolve_approval(approval_id, approved)

    def subscribe(self, route_id: str) -> tuple[str, asyncio.Queue[RouteEvent]]:
        self._ensure_accepting_work()
        return self.get_or_create(route_id).runtime.subscribe()

    def unsubscribe(self, route_id: str, subscriber_id: str) -> None:
        self.get_or_create(route_id).runtime.unsubscribe(subscriber_id)

    async def enqueue_message(
        self,
        route_id: str,
        user_text: str,
        *,
        client_message_id: str | None = None,
    ) -> None:
        self._ensure_accepting_work()
        await self.get_or_create(route_id).runtime.enqueue_user_message(
            user_text,
            client_message_id=client_message_id,
        )

    async def run_turn(self, route_id: str, user_text: str) -> AgentTurnResult:
        context = self.get_or_create(route_id)
        async with context.lock:
            self._ensure_accepting_work()
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
            self._ensure_accepting_work()
            runtime = context.runtime
            if hasattr(runtime, "stream_turn"):
                async for event in runtime.stream_turn(user_text):
                    yield event
                return
            async for event in runtime.stream_user_input(user_text):
                yield event

    def _ensure_accepting_work(self) -> None:
        if self._shutdown_started:
            raise RuntimeError("Session router is shutting down.")
