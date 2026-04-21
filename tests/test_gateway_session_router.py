"""Unit tests for gateway session routing behavior."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from jarvis.codex_backend.types import CodexConnectionError
from jarvis.core import (
    AgentAssistantMessageEvent,
    AgentTextDeltaEvent,
    AgentTurnDoneEvent,
    AgentTurnResult,
)
from jarvis.llm import TextPart
from jarvis.core.compaction import CompactionOutcome, CompactionReplacementItem
from jarvis.gateway.bash_job_supervisor import BashJobNotice, _classify_notice_kind
from jarvis.gateway.route_events import (
    RouteAssistantMessageEvent,
    RouteErrorEvent,
    RouteLocalNoticeEvent,
    RouteSystemNoticeEvent,
    RouteTaskStatusEvent,
)
from jarvis.gateway.route_runtime import (
    CompositeMainBootstrapLoader,
    RouteEventBus,
    RouteRuntime,
    _RouteTurnRequest,
    _tool_result_for_payload,
)
from jarvis.gateway.session_router import SessionRouter, validate_route_id
from jarvis.subagent.types import SubagentSnapshot
from tests.helpers import build_core_settings
from jarvis.tools import ToolSettings
from jarvis.tools.basic.bash.jobs import BashJobRecord, claim_job_owner, create_background_job


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


class CompositeMainBootstrapLoaderTests(unittest.TestCase):
    def test_subagent_bootstrap_docs_stay_heuristic_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loader = CompositeMainBootstrapLoader(build_core_settings(root_dir=Path(tmp)))
            messages = loader.load_bootstrap_messages()

        subagent_text = next(
            (
                part.text
                for message in messages
                if message.role == "system"
                for part in message.parts
                if isinstance(part, TextPart)
                and "Subagent control primitives are available only to Jarvis." in part.text
            ),
            None,
        )

        self.assertIsNotNone(subagent_text)
        if subagent_text is None:
            self.fail("Expected subagent control bootstrap text.")
        self.assertIn("wait for orchestrator updates before polling", subagent_text)
        self.assertIn("not live prompt injection", subagent_text)
        self.assertIn("detail=\"full\"", subagent_text)
        self.assertNotIn("Arguments:", subagent_text)
        self.assertNotIn("Subagent runtime control reference:", subagent_text)
        self.assertLess(len(subagent_text), 900)


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
    async def test_user_message_publishes_task_status_until_turn_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            subscriber_id, queue = runtime.subscribe()

            async def _stream_user_input(_user_text: str):
                yield AgentTurnDoneEvent(
                    session_id="main_session",
                    response_text="done",
                )

            runtime._main_loop.stream_user_input = _stream_user_input  # type: ignore[method-assign]

            await runtime.enqueue_user_message("hi", client_message_id="msg_1")
            await asyncio.wait_for(runtime._user_message_queue.join(), timeout=1)

            seen_task_status: list[RouteTaskStatusEvent] = []
            while len(seen_task_status) < 2:
                event = await asyncio.wait_for(queue.get(), timeout=1)
                if isinstance(event, RouteTaskStatusEvent):
                    seen_task_status.append(event)

            self.assertTrue(seen_task_status[0].active)
            self.assertEqual(seen_task_status[0].reason, "user_message_queued")
            self.assertFalse(seen_task_status[1].active)
            self.assertEqual(seen_task_status[1].reason, "turn_worker_idle")

            runtime.unsubscribe(subscriber_id)
            worker = runtime._message_worker
            if worker is not None:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    async def test_task_status_stays_active_while_subagent_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            subscriber_id, queue = runtime.subscribe()
            running_snapshot = SubagentSnapshot(
                subagent_id="sub_1",
                codename="Ultron",
                status="running",
                owner_main_session_id="main_session",
                owner_main_turn_id="turn_1",
                current_subagent_session_id="sub_session",
            )

            async def _stream_user_input(_user_text: str):
                yield AgentTurnDoneEvent(
                    session_id="main_session",
                    response_text="waiting on subagent",
                )

            runtime._main_loop.stream_user_input = _stream_user_input  # type: ignore[method-assign]

            with patch.object(
                runtime._subagent_manager,
                "active_snapshots",
                return_value=(running_snapshot,),
            ):
                await runtime.enqueue_user_message("hi", client_message_id="msg_1")
                await asyncio.wait_for(runtime._user_message_queue.join(), timeout=1)

            queued_events = []
            while not queue.empty():
                queued_events.append(queue.get_nowait())
            seen_task_status = [
                event for event in queued_events if isinstance(event, RouteTaskStatusEvent)
            ]
            self.assertEqual([event.active for event in seen_task_status], [True])

            completed_snapshot = replace(running_snapshot, status="completed")
            with patch.object(
                runtime._subagent_manager,
                "active_snapshots",
                return_value=(completed_snapshot,),
            ):
                await runtime._publish_task_status_if_changed(reason="subagent_completed")

            event = await asyncio.wait_for(queue.get(), timeout=1)
            self.assertIsInstance(event, RouteTaskStatusEvent)
            if not isinstance(event, RouteTaskStatusEvent):
                self.fail("Expected task status event.")
            self.assertFalse(event.active)
            self.assertEqual(event.reason, "subagent_completed")

            runtime.unsubscribe(subscriber_id)
            worker = runtime._message_worker
            if worker is not None:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    async def test_enqueue_user_message_supersedes_active_main_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )

            with patch.object(runtime._main_loop, "has_active_turn", return_value=True):
                with patch.object(
                    runtime._main_loop,
                    "request_stop",
                    return_value=True,
                ) as request_stop:
                    with patch.object(
                        runtime._subagent_manager,
                        "request_stop_all_for_superseded_user_message",
                        return_value=(),
                    ) as stop_subagents:
                        await runtime.enqueue_user_message(
                            "continue",
                            client_message_id="msg_2",
                        )

            request_stop.assert_called_once_with(reason="superseded_by_user_message")
            stop_subagents.assert_called_once_with()
            queued = runtime._user_message_queue.get_nowait()
            self.assertEqual(queued.user_text, "continue")
            self.assertEqual(queued.client_message_id, "msg_2")
            self.assertFalse(runtime._main_resume_requires_user_message)

    async def test_enqueue_new_user_message_uses_user_stop_subagent_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )

            with patch.object(runtime._main_loop, "request_stop", return_value=True) as request_stop:
                with patch.object(
                    runtime._subagent_manager,
                    "request_stop_all_for_user_stop",
                    return_value=(),
                ) as stop_for_new:
                    with patch.object(
                        runtime._subagent_manager,
                        "request_stop_all_for_superseded_user_message",
                        return_value=(),
                    ) as stop_superseded:
                        await runtime.enqueue_user_message(
                            "/new continue here",
                            client_message_id="msg_new",
                        )

            request_stop.assert_called_once_with(reason="superseded_by_user_message")
            stop_for_new.assert_called_once_with()
            stop_superseded.assert_not_called()
            queued = runtime._user_message_queue.get_nowait()
            self.assertEqual(queued.user_text, "/new continue here")
            self.assertEqual(queued.client_message_id, "msg_new")
            self.assertTrue(queued.parse_commands)

    async def test_enqueue_user_message_supersedes_background_subagent_work_when_main_idle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )

            with patch.object(
                runtime._main_loop,
                "request_stop",
                return_value=False,
            ) as request_stop:
                with patch.object(
                    runtime._subagent_manager,
                    "request_stop_all_for_superseded_user_message",
                    return_value=(),
                ) as stop_subagents:
                    await runtime.enqueue_user_message(
                        "redirect the task",
                        client_message_id="msg_3",
                    )

            request_stop.assert_called_once_with(reason="superseded_by_user_message")
            stop_subagents.assert_called_once_with()
            queued = runtime._user_message_queue.get_nowait()
            self.assertEqual(queued.user_text, "redirect the task")
            self.assertEqual(queued.client_message_id, "msg_3")
            self.assertFalse(runtime._main_resume_requires_user_message)

    async def test_new_command_resets_subagents_before_streaming_main_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            observed: list[object] = []

            async def _reset() -> dict[str, object]:
                observed.append("reset")
                return {
                    "disposed_subagent_ids": [],
                    "cancelled_job_ids": [],
                    "disposed_count": 0,
                    "cancelled_job_count": 0,
                }

            async def _stream_user_input(user_text: str):
                observed.append(("user", user_text))
                yield AgentTurnDoneEvent(
                    session_id="main_session",
                    response_text="started",
                )

            runtime._main_loop.stream_user_input = _stream_user_input  # type: ignore[method-assign]

            with patch.object(
                runtime._subagent_manager,
                "reset_for_new_session",
                side_effect=_reset,
            ):
                await runtime._user_message_queue.put(
                    _RouteTurnRequest(
                        user_text="/new continue here",
                        client_message_id="msg_new",
                        parse_commands=True,
                        user_initiated=True,
                    )
                )
                runtime._queue_wakeup.set()
                runtime._ensure_message_worker()
                await asyncio.wait_for(runtime._user_message_queue.join(), timeout=1)

            self.assertEqual(observed, ["reset", ("user", "/new continue here")])

            worker = runtime._message_worker
            if worker is not None:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    async def test_new_command_reset_failure_is_unbound_from_client_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            subscriber_id, queue = runtime.subscribe()
            stream_called = False

            async def _stream_user_input(_user_text: str):
                nonlocal stream_called
                stream_called = True
                yield AgentTurnDoneEvent(
                    session_id="main_session",
                    response_text="should not happen",
                )

            async def _reset() -> dict[str, object]:
                raise RuntimeError("reset failed")

            runtime._main_loop.stream_user_input = _stream_user_input  # type: ignore[method-assign]

            with patch.object(
                runtime._subagent_manager,
                "reset_for_new_session",
                side_effect=_reset,
            ):
                await runtime._user_message_queue.put(
                    _RouteTurnRequest(
                        user_text="/new",
                        client_message_id="msg_new",
                        parse_commands=True,
                        user_initiated=True,
                    )
                )
                runtime._queue_wakeup.set()
                runtime._ensure_message_worker()
                await asyncio.wait_for(runtime._user_message_queue.join(), timeout=1)

            event = await asyncio.wait_for(queue.get(), timeout=1)
            self.assertIsInstance(event, RouteErrorEvent)
            if not isinstance(event, RouteErrorEvent):
                self.fail("Expected /new reset failure to publish a route error event.")
            self.assertEqual(event.code, "internal_error")
            self.assertIsNone(event.turn_kind)
            self.assertIsNone(event.client_message_id)
            self.assertFalse(stream_called)

            runtime.unsubscribe(subscriber_id)
            worker = runtime._message_worker
            if worker is not None:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    async def test_internal_error_writes_session_scoped_error_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            subscriber_id, queue = runtime.subscribe()
            session_id_holder: dict[str, str] = {}

            async def _stream_user_input(_user_text: str):
                session_id = await runtime._main_loop.prepare_session()
                session_id_holder["session_id"] = session_id
                raise RuntimeError("boom")
                yield  # pragma: no cover

            runtime._main_loop.stream_user_input = _stream_user_input  # type: ignore[method-assign]

            with patch.object(runtime, "_print_runtime_error_notice") as print_notice:
                await runtime._user_message_queue.put(
                    _RouteTurnRequest(
                        user_text="hello",
                        client_message_id="msg_1",
                        parse_commands=True,
                        user_initiated=True,
                    )
                )
                runtime._queue_wakeup.set()
                runtime._ensure_message_worker()
                await asyncio.wait_for(runtime._user_message_queue.join(), timeout=1)

            event = await asyncio.wait_for(queue.get(), timeout=1)
            self.assertIsInstance(event, RouteErrorEvent)
            if not isinstance(event, RouteErrorEvent):
                self.fail("Expected internal failure to publish a route error event.")
            self.assertEqual(event.code, "internal_error")
            self.assertEqual(event.client_message_id, "msg_1")

            session_id = session_id_holder["session_id"]
            error_log_path = (
                runtime._core_settings.transcript_archive_dir.parent / "error_logs" / f"{session_id}.jsonl"
            )
            self.assertEqual(
                print_notice.call_args.kwargs["error_log_path"],
                error_log_path,
            )
            self.assertTrue(error_log_path.exists())
            entries = [
                json.loads(line)
                for line in error_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry["schema"], "jarvis.runtime_error.v1")
            self.assertEqual(entry["route_id"], "route_1")
            self.assertEqual(entry["session_id"], session_id)
            self.assertEqual(entry["client_message_id"], "msg_1")
            self.assertEqual(entry["published_client_message_id"], "msg_1")
            self.assertEqual(entry["request_turn_kind"], "user")
            self.assertEqual(entry["published_turn_kind"], "user")
            self.assertEqual(entry["exception_type"], "RuntimeError")
            self.assertEqual(entry["exception_message"], "boom")
            self.assertEqual(
                entry["message"],
                "Route route_1 main turn failed while processing client_message_id=msg_1.",
            )
            self.assertIn("RuntimeError: boom", entry["traceback"])
            self.assertIn("Traceback (most recent call last):", entry["traceback"])
            self.assertIn("logged_at", entry)

            runtime.unsubscribe(subscriber_id)
            worker = runtime._message_worker
            if worker is not None:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    async def test_codex_backend_error_is_published_without_falling_back_to_internal_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            subscriber_id, queue = runtime.subscribe()

            async def _stream_user_input(_user_text: str):
                raise CodexConnectionError("Codex app-server unavailable")
                yield  # pragma: no cover

            runtime._main_loop.stream_user_input = _stream_user_input  # type: ignore[method-assign]

            await runtime._user_message_queue.put(
                _RouteTurnRequest(
                    user_text="hello",
                    client_message_id="msg_1",
                    parse_commands=True,
                    user_initiated=True,
                )
            )
            runtime._queue_wakeup.set()
            runtime._ensure_message_worker()
            await asyncio.wait_for(runtime._user_message_queue.join(), timeout=1)

            event = await asyncio.wait_for(queue.get(), timeout=1)
            self.assertIsInstance(event, RouteErrorEvent)
            if not isinstance(event, RouteErrorEvent):
                self.fail("Expected Codex backend failure to publish a route error event.")
            self.assertEqual(event.code, "codex_backend_error")
            self.assertEqual(event.client_message_id, "msg_1")
            self.assertEqual(event.turn_kind, "user")
            self.assertIn("Codex app-server unavailable", event.message)

            runtime.unsubscribe(subscriber_id)
            worker = runtime._message_worker
            if worker is not None:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    async def test_stop_requested_when_only_subagent_is_running_appends_main_transcript_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            session_id = await runtime._main_loop.prepare_session()
            snapshot = SubagentSnapshot(
                subagent_id="sub_1",
                codename="Friday",
                status="running",
                owner_main_session_id=session_id,
                owner_main_turn_id="main_turn",
                current_subagent_session_id="sub_session",
            )

            with patch.object(runtime._main_loop, "request_stop", return_value=False):
                with patch.object(
                    runtime._subagent_manager,
                    "request_stop_all_for_user_stop",
                    return_value=(snapshot,),
                ):
                    self.assertTrue(runtime.request_stop())

            self.assertTrue(runtime._main_resume_requires_user_message)
            records = runtime._main_loop._storage.load_records(session_id)
            stop_notes = [
                record
                for record in records
                if record.role == "system" and record.metadata.get("user_stop_subagents") is True
            ]
            self.assertEqual(len(stop_notes), 1)
            self.assertIn("The user issued /stop.", stop_notes[0].content)
            self.assertIn("Friday (sub_1)", stop_notes[0].content)
            self.assertEqual(stop_notes[0].metadata["subagent_ids"], ["sub_1"])

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

    async def test_stop_suppresses_paused_subagent_followup_until_user_message(self) -> None:
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
                    notice_kind="subagent_paused",
                    text="paused (user_stop).",
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

            async def _stream_runtime_turn(*, force_session_id: str | None = None, pre_turn_messages=()):
                _ = pre_turn_messages
                observed_calls.append(("runtime", force_session_id or "main_session"))
                yield AgentTurnDoneEvent(
                    session_id=force_session_id or "main_session",
                    response_text="runtime",
                )

            runtime._main_loop.stream_user_input = _stream_user_input  # type: ignore[method-assign]
            runtime._main_loop.stream_runtime_turn = _stream_runtime_turn  # type: ignore[method-assign]

            await runtime._message_queue.put(
                _RouteTurnRequest(
                    user_text=None,
                    force_session_id="main_session",
                    parse_commands=False,
                    user_initiated=False,
                    internal_generation=runtime._internal_followup_generation,
                    runtime_turn_kind="main_subagent_progress",
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

    async def test_subagent_progress_notice_enqueues_runtime_main_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            session_id = await runtime._main_loop.prepare_session()

            snapshot = SubagentSnapshot(
                subagent_id="sub_1",
                codename="Ultron",
                status="waiting_background",
                owner_main_session_id=session_id,
                owner_main_turn_id="turn_1",
                current_subagent_session_id="sub_session",
                pending_background_job_count=1,
                pending_background_job_ids=("deadbeefdeadbeefdeadbeefdeadbeef",),
            )

            with patch.object(runtime, "_ensure_message_worker"):
                with patch.object(
                    runtime._subagent_manager,
                    "snapshot_for",
                    return_value=snapshot,
                ):
                    await runtime.publish_event(
                        RouteSystemNoticeEvent(
                            route_id="route_1",
                            agent_kind="subagent",
                            agent_name="Ultron",
                            subagent_id="sub_1",
                            session_id="sub_session",
                            notice_kind="subagent_waiting_background",
                            text="waiting on detached bash jobs.",
                        )
                    )

            queued = runtime._message_queue.get_nowait()
            self.assertIsNone(queued.user_text)
            self.assertFalse(queued.parse_commands)
            self.assertEqual(queued.pre_turn_messages, ())
            self.assertEqual(queued.force_session_id, session_id)
            self.assertEqual(queued.runtime_turn_kind, "main_subagent_progress")

    async def test_subagent_paused_notice_enqueues_runtime_main_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            session_id = await runtime._main_loop.prepare_session()

            snapshot = SubagentSnapshot(
                subagent_id="sub_1",
                codename="Ultron",
                status="paused",
                owner_main_session_id=session_id,
                owner_main_turn_id="turn_1",
                current_subagent_session_id="sub_session",
                pause_reason="superseded_by_user_message",
            )

            with patch.object(runtime, "_ensure_message_worker"):
                with patch.object(
                    runtime._subagent_manager,
                    "snapshot_for",
                    return_value=snapshot,
                ):
                    await runtime.publish_event(
                        RouteSystemNoticeEvent(
                            route_id="route_1",
                            agent_kind="subagent",
                            agent_name="Ultron",
                            subagent_id="sub_1",
                            session_id="sub_session",
                            notice_kind="subagent_paused",
                            text="paused (superseded_by_user_message).",
                        )
                    )

            queued = runtime._message_queue.get_nowait()
            self.assertIsNone(queued.user_text)
            self.assertFalse(queued.parse_commands)
            self.assertEqual(queued.pre_turn_messages, ())
            self.assertEqual(queued.force_session_id, session_id)
            self.assertEqual(queued.runtime_turn_kind, "main_subagent_progress")

    async def test_subagent_resumed_notice_does_not_enqueue_main_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )

            with patch.object(runtime, "_ensure_message_worker"):
                await runtime.publish_event(
                    RouteSystemNoticeEvent(
                        route_id="route_1",
                        agent_kind="subagent",
                        agent_name="Ultron",
                        subagent_id="sub_1",
                        session_id="sub_session",
                        notice_kind="subagent_resumed_after_bash_update",
                        text="resumed after detached bash update.",
                    )
                )

            self.assertTrue(runtime._message_queue.empty())

    async def test_main_subagent_runtime_turn_persists_concise_agent_only_system_notice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            session_id = await runtime._main_loop.prepare_session()
            snapshot = SubagentSnapshot(
                subagent_id="sub_1",
                codename="Ultron",
                status="waiting_background",
                owner_main_session_id=session_id,
                owner_main_turn_id="turn_1",
                current_subagent_session_id="sub_session",
                pending_background_job_count=1,
                pending_background_job_ids=("deadbeefdeadbeefdeadbeefdeadbeef",),
            )
            notice = RouteSystemNoticeEvent(
                route_id="route_1",
                agent_kind="subagent",
                agent_name="Ultron",
                subagent_id="sub_1",
                session_id="sub_session",
                notice_kind="subagent_waiting_background",
                text="waiting on detached bash jobs.",
                public=False,
            )
            published_events: list[RouteSystemNoticeEvent] = []

            async def _publish_event(event):
                if isinstance(event, RouteSystemNoticeEvent):
                    published_events.append(event)

            async def _stream_runtime_turn(*, force_session_id=None, command_override=None, pre_turn_messages=()):
                _ = command_override, pre_turn_messages
                yield AgentTurnDoneEvent(
                    session_id=force_session_id or session_id,
                    response_text="waiting on subagent",
                )

            runtime.publish_event = _publish_event  # type: ignore[method-assign]
            runtime._main_loop.stream_runtime_turn = _stream_runtime_turn  # type: ignore[method-assign]

            with patch.object(runtime._subagent_manager, "snapshot_for", return_value=snapshot):
                await runtime._enqueue_main_subagent_followup(notice)
                runtime._ensure_message_worker()
                await asyncio.wait_for(runtime._message_queue.join(), timeout=1)

            records = runtime._main_loop._storage.load_records(session_id)
            system_notes = [
                record
                for record in records
                if record.role == "system"
                and record.metadata.get("subagent_progress_update") is True
            ]
            user_records = [record for record in records if record.role == "user"]
            self.assertEqual(len(system_notes), 1)
            self.assertIn("subagent=Ultron", system_notes[0].content)
            self.assertIn("recommendation=wait", system_notes[0].content)
            self.assertIn("not a new user message", system_notes[0].content)
            self.assertLess(len(system_notes[0].content), 600)
            self.assertEqual(user_records, [])
            self.assertEqual(len(published_events), 1)
            self.assertEqual(published_events[0].notice_kind, "subagent_progress_update")
            self.assertFalse(published_events[0].public)

            worker = runtime._message_worker
            if worker is not None:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    async def test_stale_disposed_subagent_followup_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            session_id = await runtime._main_loop.prepare_session()
            snapshot = SubagentSnapshot(
                subagent_id="sub_1",
                codename="Ultron",
                status="disposed",
                owner_main_session_id=session_id,
                owner_main_turn_id="turn_1",
                current_subagent_session_id="sub_session",
            )
            notice = RouteSystemNoticeEvent(
                route_id="route_1",
                agent_kind="subagent",
                agent_name="Ultron",
                subagent_id="sub_1",
                session_id="sub_session",
                notice_kind="subagent_completed",
                text="completed.",
                public=False,
            )

            async def _publish_event(event):
                raise AssertionError(f"Unexpected published event: {event!r}")

            async def _stream_runtime_turn(*, force_session_id=None, command_override=None, pre_turn_messages=()):
                raise AssertionError("Unexpected runtime followup turn.")
                yield force_session_id, command_override, pre_turn_messages  # pragma: no cover

            runtime.publish_event = _publish_event  # type: ignore[method-assign]
            runtime._main_loop.stream_runtime_turn = _stream_runtime_turn  # type: ignore[method-assign]

            with patch.object(runtime._subagent_manager, "snapshot_for", return_value=snapshot):
                await runtime._enqueue_main_subagent_followup(notice)
                runtime._ensure_message_worker()
                await asyncio.wait_for(runtime._message_queue.join(), timeout=1)

            records = runtime._main_loop._storage.load_records(session_id)
            self.assertFalse(
                any(
                    record.role == "system"
                    and record.metadata.get("subagent_progress_update") is True
                    for record in records
                )
            )

            worker = runtime._message_worker
            if worker is not None:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    async def test_manual_compaction_publishes_local_notice_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            session_id = await runtime._main_loop.prepare_session()
            self.assertTrue(
                runtime._main_loop.append_system_note(
                    "Compact this context.",
                    session_id=session_id,
                )
            )
            subscriber_id, event_queue = runtime.subscribe()

            try:
                with patch.object(
                    runtime._main_loop._compactor,
                    "compact",
                    return_value=CompactionOutcome(
                        items=(
                            CompactionReplacementItem(
                                role="system",
                                kind="session_frame",
                                content="Session frame",
                            ),
                            CompactionReplacementItem(
                                role="system",
                                kind="handover_state",
                                content="Handover state",
                            ),
                        ),
                        response_payload={"items": []},
                        model="fake-model",
                        provider="fake-provider",
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        response_id="resp_fake",
                    ),
                ):
                    await runtime.enqueue_user_message(
                        "/compact",
                        client_message_id="client_1",
                    )
                    await asyncio.wait_for(runtime._user_message_queue.join(), timeout=1)

                events: list[object] = []
                while not event_queue.empty():
                    events.append(event_queue.get_nowait())
            finally:
                runtime.unsubscribe(subscriber_id)
                worker = runtime._message_worker
                if worker is not None:
                    worker.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await worker

            local_notices = [
                event
                for event in events
                if isinstance(event, RouteLocalNoticeEvent)
            ]
            self.assertEqual(
                [
                    (
                        event.notice_kind,
                        event.text,
                        event.client_message_id,
                        event.turn_kind,
                    )
                    for event in local_notices
                ],
                [
                    (
                        "compaction_started",
                        "Compacting...",
                        "client_1",
                        "user",
                    ),
                    (
                        "compaction_completed",
                        "Context compacted into a new session.",
                        "client_1",
                        "user",
                    ),
                ],
            )

    async def test_stop_latches_when_detached_bash_jobs_are_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            session_id = await runtime._main_loop.prepare_session()
            tool_settings = ToolSettings.from_workspace_dir(runtime._core_settings.workspace_dir)
            job = create_background_job(
                workspace_dir=tool_settings.workspace_dir,
                bash_executable="/bin/bash",
                command="sleep 5",
                cwd="/workspace",
                log_max_bytes=tool_settings.bash_job_log_max_bytes,
                total_storage_budget_bytes=tool_settings.bash_job_total_storage_budget_bytes,
                retention_seconds=tool_settings.bash_job_retention_seconds,
            )
            claim_job_owner(
                workspace_dir=tool_settings.workspace_dir,
                job_id=job.job_id,
                route_id="route_1",
                session_id=session_id,
                turn_id="turn_1",
                agent_kind="main",
                agent_name="Jarvis",
            )

            with patch.object(runtime._main_loop, "request_stop", return_value=False):
                self.assertTrue(runtime.request_stop())

            self.assertTrue(runtime._main_resume_requires_user_message)
            records = runtime._main_loop._storage.load_records(session_id)
            stop_notes = [
                record
                for record in records
                if record.role == "system" and record.metadata.get("user_stop_bash_jobs") is True
            ]
            self.assertEqual(len(stop_notes), 1)
            self.assertIn("detached bash jobs", stop_notes[0].content)
            self.assertIn(job.job_id, stop_notes[0].content)

    async def test_detached_bash_notice_enqueues_internal_main_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            session_id = await runtime._main_loop.prepare_session()
            notice = BashJobNotice(
                job_id="deadbeefdeadbeefdeadbeefdeadbeef",
                notice_kind="bash_job_completed",
                owner_route_id="route_1",
                owner_session_id=session_id,
                owner_turn_id="turn_1",
                owner_agent_kind="main",
                owner_agent_name="Jarvis",
                owner_subagent_id=None,
                status="finished",
                command="sleep 1; echo done",
                started_at="2026-03-21T10:00:00Z",
                last_update_at="2026-03-21T10:00:02Z",
                finished_at="2026-03-21T10:00:02Z",
                cancelled_at=None,
                exit_code=0,
                stdout="done\n",
                stderr="",
                stdout_bytes_seen=5,
                stderr_bytes_seen=0,
                stdout_bytes_dropped=0,
                stderr_bytes_dropped=0,
                progress_hint="done",
            )

            with patch.object(runtime, "_ensure_message_worker"):
                await runtime._enqueue_main_bash_job_followup((notice,))

            queued = runtime._message_queue.get_nowait()
            self.assertIsNone(queued.user_text)
            self.assertFalse(queued.parse_commands)
            self.assertEqual(queued.force_session_id, session_id)
            self.assertEqual(queued.internal_generation, runtime._internal_followup_generation)
            self.assertEqual(queued.runtime_turn_kind, "main_bash_progress")
            self.assertEqual(queued.pre_turn_messages, ())
            self.assertIn(notice.job_id, runtime._pending_main_bash_notices)

    async def test_detached_bash_notice_is_suppressed_while_route_is_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            runtime._main_resume_requires_user_message = True
            notice = BashJobNotice(
                job_id="deadbeefdeadbeefdeadbeefdeadbeef",
                notice_kind="bash_job_completed",
                owner_route_id="route_1",
                owner_session_id="main_session",
                owner_turn_id="turn_1",
                owner_agent_kind="main",
                owner_agent_name="Jarvis",
                owner_subagent_id=None,
                status="finished",
                command="sleep 1; echo done",
                started_at="2026-03-21T10:00:00Z",
                last_update_at="2026-03-21T10:00:02Z",
                finished_at="2026-03-21T10:00:02Z",
                cancelled_at=None,
                exit_code=0,
                stdout="done\n",
                stderr="",
                stdout_bytes_seen=5,
                stderr_bytes_seen=0,
                stdout_bytes_dropped=0,
                stderr_bytes_dropped=0,
                progress_hint="done",
            )

            await runtime._enqueue_main_bash_job_followup((notice,))

            self.assertTrue(runtime._message_queue.empty())

    async def test_main_bash_runtime_turn_persists_concise_agent_only_system_notice_and_uses_runtime_turn(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RouteRuntime(
                route_id="route_1",
                llm_service=object(),  # type: ignore[arg-type]
                core_settings=build_core_settings(root_dir=Path(tmp)),
            )
            session_id = await runtime._main_loop.prepare_session()
            notice = BashJobNotice(
                job_id="deadbeefdeadbeefdeadbeefdeadbeef",
                notice_kind="bash_job_output_started",
                owner_route_id="route_1",
                owner_session_id=session_id,
                owner_turn_id="turn_1",
                owner_agent_kind="main",
                owner_agent_name="Jarvis",
                owner_subagent_id=None,
                status="running",
                command="sleep 1; echo done",
                started_at="2026-03-21T10:00:00Z",
                last_update_at="2026-03-21T10:00:02Z",
                finished_at=None,
                cancelled_at=None,
                exit_code=None,
                stdout="done\n",
                stderr="",
                stdout_bytes_seen=5,
                stderr_bytes_seen=0,
                stdout_bytes_dropped=0,
                stderr_bytes_dropped=0,
                progress_hint="done",
            )
            published_events: list[RouteSystemNoticeEvent] = []

            async def _publish_event(event):
                if isinstance(event, RouteSystemNoticeEvent):
                    published_events.append(event)

            async def _stream_runtime_turn(*, force_session_id=None, command_override=None, pre_turn_messages=()):
                _ = command_override, pre_turn_messages
                yield AgentTurnDoneEvent(
                    session_id=force_session_id or session_id,
                    response_text="waiting on background",
                )

            runtime.publish_event = _publish_event  # type: ignore[method-assign]
            runtime._main_loop.stream_runtime_turn = _stream_runtime_turn  # type: ignore[method-assign]

            await runtime._enqueue_main_bash_job_followup((notice,))
            runtime._ensure_message_worker()
            await asyncio.wait_for(runtime._message_queue.join(), timeout=1)

            records = runtime._main_loop._storage.load_records(session_id)
            system_notes = [
                record
                for record in records
                if record.role == "system"
                and record.metadata.get("bash_job_progress_update") is True
            ]
            user_records = [record for record in records if record.role == "user"]
            self.assertEqual(len(system_notes), 1)
            self.assertIn(notice.job_id, system_notes[0].content)
            self.assertNotIn("command:", system_notes[0].content)
            self.assertNotIn("stdout tail:", system_notes[0].content)
            self.assertIn("not a new user message", system_notes[0].content)
            self.assertLess(len(system_notes[0].content), 500)
            self.assertEqual(user_records, [])
            self.assertEqual(len(published_events), 1)
            self.assertEqual(published_events[0].notice_kind, "bash_job_progress_update")
            self.assertIn(notice.job_id, published_events[0].text)
            self.assertFalse(published_events[0].public)

            worker = runtime._message_worker
            if worker is not None:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    def test_bash_heartbeat_backoff_uses_progress_notice_count(self) -> None:
        now = datetime.now(UTC)
        base_record = BashJobRecord(
            job_id="deadbeefdeadbeefdeadbeefdeadbeef",
            command="sleep 1; echo done",
            pid=1,
            pgid=1,
            runner_pid=1,
            runner_pgid=1,
            launched_at=(now - timedelta(minutes=10)).isoformat(),
            cwd="/workspace",
            stdout_path="/workspace/stdout.log",
            stderr_path="/workspace/stderr.log",
            job_dir="/workspace/job",
            owner_route_id="route_1",
            owner_session_id="main_session",
            owner_turn_id="turn_1",
            owner_agent_kind="main",
            owner_agent_name="Jarvis",
            owner_subagent_id=None,
            terminal_notice_kind=None,
            terminal_notice_dispatched_at=None,
        )

        first_heartbeat_record = replace(
            base_record,
            last_progress_notice_kind="bash_job_started",
            last_progress_notice_at=(now - timedelta(seconds=31)).isoformat(),
            last_progress_notice_status="running",
            last_progress_notice_stdout_bytes_seen=0,
            last_progress_notice_stderr_bytes_seen=0,
            last_progress_notice_last_update_at=(now - timedelta(seconds=31)).isoformat(),
            progress_notice_count=0,
        )
        self.assertEqual(
            _classify_notice_kind(
                record=first_heartbeat_record,
                status_metadata={"status": "running", "stdout_bytes_seen": 0, "stderr_bytes_seen": 0},
            ),
            "bash_job_heartbeat",
        )

        second_heartbeat_not_due = replace(
            first_heartbeat_record,
            last_progress_notice_at=(now - timedelta(seconds=45)).isoformat(),
            progress_notice_count=1,
        )
        self.assertIsNone(
            _classify_notice_kind(
                record=second_heartbeat_not_due,
                status_metadata={"status": "running", "stdout_bytes_seen": 0, "stderr_bytes_seen": 0},
            )
        )

        second_heartbeat_due = replace(
            first_heartbeat_record,
            last_progress_notice_at=(now - timedelta(seconds=61)).isoformat(),
            progress_notice_count=1,
        )
        self.assertEqual(
            _classify_notice_kind(
                record=second_heartbeat_due,
                status_metadata={"status": "running", "stdout_bytes_seen": 0, "stderr_bytes_seen": 0},
            ),
            "bash_job_heartbeat",
        )

        third_heartbeat_not_due = replace(
            first_heartbeat_record,
            last_progress_notice_at=(now - timedelta(seconds=120)).isoformat(),
            progress_notice_count=2,
        )
        self.assertIsNone(
            _classify_notice_kind(
                record=third_heartbeat_not_due,
                status_metadata={"status": "running", "stdout_bytes_seen": 0, "stderr_bytes_seen": 0},
            )
        )

        third_heartbeat_due = replace(
            first_heartbeat_record,
            last_progress_notice_at=(now - timedelta(seconds=181)).isoformat(),
            progress_notice_count=2,
        )
        self.assertEqual(
            _classify_notice_kind(
                record=third_heartbeat_due,
                status_metadata={"status": "running", "stdout_bytes_seen": 0, "stderr_bytes_seen": 0},
            ),
            "bash_job_needs_attention",
        )
