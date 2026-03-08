"""Unit tests for gateway session routing behavior."""

from __future__ import annotations

import asyncio
import unittest

from core import AgentAssistantMessageEvent, AgentTextDeltaEvent, AgentTurnDoneEvent, AgentTurnResult
from gateway.session_router import SessionRouter, validate_route_id


class _TrackingLoop:
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self.messages: list[str] = []
        self.active_calls = 0
        self.max_concurrency = 0

    def active_session_id(self) -> str | None:
        return self._session_id

    async def handle_user_input(self, user_text: str) -> AgentTurnResult:
        self.active_calls += 1
        self.max_concurrency = max(self.max_concurrency, self.active_calls)
        await asyncio.sleep(0.01)
        self.messages.append(user_text)
        self.active_calls -= 1
        return AgentTurnResult(
            session_id=self._session_id,
            response_text=f"echo:{user_text}",
        )

    async def stream_user_input(self, user_text: str):
        self.active_calls += 1
        self.max_concurrency = max(self.max_concurrency, self.active_calls)
        await asyncio.sleep(0.01)
        self.messages.append(user_text)
        yield AgentTextDeltaEvent(session_id=self._session_id, delta="echo:")
        yield AgentAssistantMessageEvent(
            session_id=self._session_id,
            text=f"echo:{user_text}",
        )
        yield AgentTurnDoneEvent(
            session_id=self._session_id,
            response_text=f"echo:{user_text}",
        )
        self.active_calls -= 1


class SessionRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_same_loop_for_same_route(self) -> None:
        created_routes: list[str] = []
        loops: dict[str, _TrackingLoop] = {}

        def factory(route_id: str) -> _TrackingLoop:
            created_routes.append(route_id)
            loop = _TrackingLoop(session_id=f"{route_id}-session")
            loops[route_id] = loop
            return loop

        router = SessionRouter(factory)
        first = await router.run_turn("alpha", "one")
        second = await router.run_turn("alpha", "two")

        self.assertEqual(first.session_id, "alpha-session")
        self.assertEqual(second.session_id, "alpha-session")
        self.assertEqual(created_routes, ["alpha"])
        self.assertEqual(loops["alpha"].messages, ["one", "two"])

    async def test_serializes_turns_within_route(self) -> None:
        loop = _TrackingLoop(session_id="alpha-session")
        router = SessionRouter(lambda _route_id: loop)

        await asyncio.gather(
            router.run_turn("alpha", "one"),
            router.run_turn("alpha", "two"),
            router.run_turn("alpha", "three"),
        )

        self.assertEqual(loop.max_concurrency, 1)
        self.assertCountEqual(loop.messages, ["one", "two", "three"])

    async def test_isolates_routes(self) -> None:
        loops: dict[str, _TrackingLoop] = {}

        def factory(route_id: str) -> _TrackingLoop:
            loop = _TrackingLoop(session_id=f"{route_id}-session")
            loops[route_id] = loop
            return loop

        router = SessionRouter(factory)
        alpha_result, beta_result = await asyncio.gather(
            router.run_turn("alpha", "hello"),
            router.run_turn("beta", "world"),
        )

        self.assertEqual(alpha_result.session_id, "alpha-session")
        self.assertEqual(beta_result.session_id, "beta-session")
        self.assertEqual(loops["alpha"].messages, ["hello"])
        self.assertEqual(loops["beta"].messages, ["world"])

    async def test_active_session_id_uses_route_loop(self) -> None:
        router = SessionRouter(lambda route_id: _TrackingLoop(session_id=f"{route_id}-session"))
        self.assertEqual(router.active_session_id("alpha"), "alpha-session")

    async def test_stream_turn_yields_delta_message_and_done(self) -> None:
        router = SessionRouter(lambda route_id: _TrackingLoop(session_id=f"{route_id}-session"))
        events = [event async for event in router.stream_turn("alpha", "hello")]

        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].type, "text_delta")
        self.assertEqual(events[1].type, "assistant_message")
        self.assertEqual(events[2].type, "done")


class ValidateRouteIDTests(unittest.TestCase):
    def test_accepts_expected_characters(self) -> None:
        self.assertEqual(validate_route_id("user_01-prod"), "user_01-prod")

    def test_rejects_bad_route_id(self) -> None:
        with self.assertRaises(ValueError):
            validate_route_id(" bad id ")
        with self.assertRaises(ValueError):
            validate_route_id("../escape")
