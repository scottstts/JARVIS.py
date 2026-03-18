"""Unit tests for gateway session routing behavior."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import (
    AgentAssistantMessageEvent,
    AgentRuntimeMessage,
    AgentTextDeltaEvent,
    AgentTurnDoneEvent,
    AgentTurnResult,
)
from gateway.route_events import RouteAssistantMessageEvent, RouteSystemNoticeEvent
from gateway.route_runtime import (
    RouteEventBus,
    RouteRuntime,
    _RouteTurnRequest,
    _SUBAGENT_SUPERVISOR_FOLLOWUP_TEXT,
    _tool_result_for_payload,
)
from gateway.session_router import SessionRouter, validate_route_id
from tests.helpers import build_core_settings


class _TrackingLoop:
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self.messages: list[str] = []
        self.active_calls = 0
        self.max_concurrency = 0
        self.stop_requests = 0
        self.approval_resolutions: list[tuple[str, bool]] = []

    def active_session_id(self) -> str | None:
        return self._session_id

    def request_stop(self) -> bool:
        self.stop_requests += 1
        return True

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        self.approval_resolutions.append((approval_id, approved))
        return approval_id == "approval_1"

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

    async def test_request_stop_delegates_to_route_loop(self) -> None:
        loop = _TrackingLoop(session_id="alpha-session")
        router = SessionRouter(lambda _route_id: loop)

        stop_requested = router.request_stop("alpha")

        self.assertTrue(stop_requested)
        self.assertEqual(loop.stop_requests, 1)

    async def test_resolve_approval_delegates_to_route_loop(self) -> None:
        loop = _TrackingLoop(session_id="alpha-session")
        router = SessionRouter(lambda _route_id: loop)

        resolved = router.resolve_approval("alpha", "approval_1", True)

        self.assertTrue(resolved)
        self.assertEqual(loop.approval_resolutions, [("approval_1", True)])


class ValidateRouteIDTests(unittest.TestCase):
    def test_accepts_expected_characters(self) -> None:
        self.assertEqual(validate_route_id("user_01-prod"), "user_01-prod")

    def test_rejects_bad_route_id(self) -> None:
        with self.assertRaises(ValueError):
            validate_route_id(" bad id ")
        with self.assertRaises(ValueError):
            validate_route_id("../escape")


class RouteEventBusTests(unittest.IsolatedAsyncioTestCase):
    async def test_subscriber_ids_do_not_collide_after_unsubscribe(self) -> None:
        bus = RouteEventBus()

        first_id, _first_queue = bus.subscribe()
        second_id, second_queue = bus.subscribe()
        bus.unsubscribe(first_id)
        third_id, third_queue = bus.subscribe()

        self.assertNotEqual(second_id, third_id)

        event = RouteAssistantMessageEvent(
            route_id="route_1",
            agent_kind="main",
            agent_name="Jarvis",
            session_id="session_1",
            text="hello",
        )
        await bus.publish(event)

        self.assertEqual((await second_queue.get()).text, "hello")
        self.assertEqual((await third_queue.get()).text, "hello")


class RouteRuntimeToolResultTests(unittest.TestCase):
    def test_subagent_tool_results_mark_control_metadata(self) -> None:
        result = _tool_result_for_payload(
            call_id="call_1",
            name="subagent_invoke",
            title="Subagent invoked",
            payload={
                "subagent_id": "sub_1",
                "codename": "Friday",
            },
        )

        self.assertTrue(result.metadata["subagent_control"])
        self.assertEqual(result.metadata["subagent_action"], "invoke")
        self.assertEqual(result.metadata["subagent_id"], "sub_1")
        self.assertEqual(result.metadata["codename"], "Friday")


class RouteRuntimeSupervisorFollowupTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_suppresses_new_terminal_subagent_followup_until_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )

            with patch.object(runtime._main_loop, "request_stop", return_value=True):
                self.assertTrue(runtime.request_stop())

            await runtime.publish_event(
                RouteSystemNoticeEvent(
                    route_id="route_1",
                    agent_kind="subagent",
                    agent_name="Ultron",
                    subagent_id="sub_1",
                    session_id="sub_session",
                    notice_kind="subagent_completed",
                    text="Ultron completed.",
                )
            )

            self.assertTrue(runtime._message_queue.empty())

    async def test_stop_discards_stale_internal_followups_before_next_user_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )

            observed_calls: list[tuple[str, str]] = []

            async def _stream_user_input(user_text: str):
                observed_calls.append(("user", user_text))
                yield AgentTurnDoneEvent(
                    session_id="main_session",
                    response_text=f"user:{user_text}",
                )

            async def _stream_turn(
                *,
                user_text: str,
                force_session_id: str | None = None,
                pre_turn_messages=(),
            ):
                observed_calls.append(("internal", user_text))
                yield AgentTurnDoneEvent(
                    session_id=force_session_id or "main_session",
                    response_text=f"internal:{user_text}",
                )

            runtime._main_loop.stream_user_input = _stream_user_input  # type: ignore[method-assign]
            runtime._main_loop.stream_turn = _stream_turn  # type: ignore[method-assign]

            await runtime._message_queue.put(
                _RouteTurnRequest(
                    user_text=_SUBAGENT_SUPERVISOR_FOLLOWUP_TEXT,
                    force_session_id="main_session",
                    parse_commands=False,
                    user_initiated=False,
                    internal_generation=runtime._internal_followup_generation,
                )
            )

            with patch.object(runtime._main_loop, "request_stop", return_value=True):
                self.assertTrue(runtime.request_stop())

            await runtime.enqueue_user_message("continue")
            runtime._ensure_message_worker()
            await asyncio.wait_for(runtime._message_queue.join(), timeout=1)

            self.assertEqual(observed_calls, [("user", "continue")])

            worker = runtime._message_worker
            if worker is not None:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    async def test_terminal_subagent_notice_enqueues_internal_main_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )

            followup_message = AgentRuntimeMessage(
                role="developer",
                content="Subagent finished and needs supervision.",
            )

            with patch.object(runtime, "_ensure_message_worker"):
                with patch.object(
                    runtime._subagent_manager,
                    "main_followup_runtime_messages",
                    return_value=(followup_message,),
                ) as build_followup:
                    await runtime.publish_event(
                        RouteSystemNoticeEvent(
                            route_id="route_1",
                            agent_kind="subagent",
                            agent_name="Ultron",
                            subagent_id="sub_1",
                            session_id="sub_session",
                            notice_kind="subagent_completed",
                            text="Ultron completed.",
                        )
                    )

            queued = runtime._message_queue.get_nowait()
            self.assertEqual(queued.user_text, _SUBAGENT_SUPERVISOR_FOLLOWUP_TEXT)
            self.assertFalse(queued.parse_commands)
            self.assertEqual(queued.pre_turn_messages, (followup_message,))
            build_followup.assert_called_once_with(
                agent="sub_1",
                notice_kind="subagent_completed",
                notice_text="Ultron completed.",
            )
