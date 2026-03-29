"""Unit tests for AgentLoop streaming turn behavior."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from jarvis.core import AgentAssistantMessageEvent, AgentLoop, AgentTurnDoneEvent
from jarvis.llm import DoneEvent, LLMRequest, LLMResponse, LLMUsage, TextDeltaEvent, TextPart
from jarvis.storage import SessionStorage
from tests.helpers import build_core_settings


def _build_response(text: str) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="fake-chat",
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        response_id="resp_fake",
    )


class _FakeStreamingLLMService:
    async def generate(self, _request):
        return _build_response("non-stream-reply")

    async def stream_generate(self, _request):
        yield TextDeltaEvent(delta="stream-")
        yield TextDeltaEvent(delta="reply")
        yield DoneEvent(response=_build_response("stream-reply"))


class _FakeTurnContextLLMService:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest):
        self.requests.append(request)
        return _build_response("ok")

    async def stream_generate(self, _request):
        raise AssertionError("Streaming is not expected in this test.")


class _InterruptibleStreamingLLMService:
    def __init__(self) -> None:
        self.stream_started = asyncio.Event()
        self.release_stream = asyncio.Event()
        self.generate_requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest):
        self.generate_requests.append(request)
        return _build_response("next-turn")

    async def stream_generate(self, _request):
        yield TextDeltaEvent(delta="stream-")
        self.stream_started.set()
        await self.release_stream.wait()
        yield TextDeltaEvent(delta="reply")
        yield DoneEvent(response=_build_response("stream-reply"))


class AgentLoopStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_user_input_emits_delta_and_done_and_persists_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeStreamingLLMService(),
                settings=settings,
                storage=storage,
            )

            events = [event async for event in loop.stream_user_input("hello")]
            self.assertEqual(
                [event.type for event in events],
                ["turn_started", "text_delta", "text_delta", "assistant_message", "done"],
            )
            self.assertTrue(events[0].turn_id)
            self.assertIsInstance(events[-2], AgentAssistantMessageEvent)
            self.assertIsInstance(events[-1], AgentTurnDoneEvent)
            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertEqual(done.response_text, "stream-reply")

            records = storage.load_records(done.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-2].role, "user")
            self.assertEqual(message_records[-2].content, "hello")
            self.assertEqual(message_records[-1].role, "assistant")
            self.assertEqual(message_records[-1].content, "stream-reply")

    async def test_stream_new_with_body_marks_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeStreamingLLMService(),
                settings=settings,
                storage=storage,
            )

            events = [event async for event in loop.stream_user_input("/new continue")]
            self.assertEqual(events[-2].type, "assistant_message")
            self.assertIsInstance(events[-1], AgentTurnDoneEvent)
            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertEqual(done.command, "/new")

    async def test_handle_user_input_persists_turn_datetime_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = _FakeTurnContextLLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )

            result = await loop.handle_user_input("hello")

            self.assertEqual(result.response_text, "ok")
            self.assertEqual(len(llm_service.requests), 1)
            request = llm_service.requests[0]
            turn_context_entries: list[tuple[int, str]] = []
            for index, message in enumerate(request.messages):
                if message.role != "system":
                    continue
                for part in message.parts:
                    if not isinstance(part, TextPart):
                        continue
                    if "System context auto-appended for this turn only." in part.text:
                        turn_context_entries.append((index, part.text))
            self.assertEqual(len(turn_context_entries), 1)
            context_index, context_text = turn_context_entries[0]
            self.assertIn("Current date/time:", context_text)
            self.assertIn(f"| {settings.turn_timezone} time", context_text)

            user_index = next(
                index
                for index, message in enumerate(request.messages)
                if message.role == "user"
                and any(
                    isinstance(part, TextPart) and part.text == "hello"
                    for part in message.parts
                )
            )
            self.assertLess(context_index, user_index)

            persisted_records = [
                record
                for record in storage.load_records(result.session_id)
                if record.metadata.get("turn_context") == "datetime"
            ]
            self.assertEqual(len(persisted_records), 1)
            self.assertIn(
                "System context auto-appended for this turn only.",
                persisted_records[0].content,
            )

    async def test_handle_user_input_logs_basic_tool_bootstrap_into_transcript_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = _FakeTurnContextLLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )

            result = await loop.handle_user_input("hello")

            records = storage.load_records(result.session_id)
            tool_bootstrap_records = [
                record
                for record in records
                if record.role == "system" and record.metadata.get("tool_bootstrap") == "basic"
            ]
            self.assertEqual(len(tool_bootstrap_records), 1)
            serialized_tools = json.loads(tool_bootstrap_records[0].content)
            self.assertIn("tool_definitions", tool_bootstrap_records[0].metadata)
            self.assertEqual(serialized_tools, tool_bootstrap_records[0].metadata["tool_definitions"])
            self.assertIn("bash", [tool["name"] for tool in serialized_tools])
            self.assertIn("tool_search", [tool["name"] for tool in serialized_tools])

            request = llm_service.requests[0]
            request_text = "\n".join(
                part.text
                for message in request.messages
                for part in message.parts
                if isinstance(part, TextPart)
            )
            self.assertNotIn(tool_bootstrap_records[0].content, request_text)
            self.assertIn("bash", [tool.name for tool in request.tools])
            self.assertIn("tool_search", [tool.name for tool in request.tools])

    async def test_stream_user_input_interrupts_after_current_streamed_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = _InterruptibleStreamingLLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )

            async def _collect_events():
                return [event async for event in loop.stream_user_input("hello")]

            task = asyncio.create_task(_collect_events())
            await llm_service.stream_started.wait()

            self.assertTrue(loop.request_stop())
            llm_service.release_stream.set()
            events = await task

            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertTrue(done.interrupted)
            self.assertEqual(done.response_text, "stream-reply")

            visible_records = storage.load_records(done.session_id)
            visible_message_records = [record for record in visible_records if record.kind == "message"]
            self.assertEqual(visible_message_records[-3].role, "user")
            self.assertEqual(visible_message_records[-3].content, "hello")
            self.assertEqual(visible_message_records[-2].role, "assistant")
            self.assertEqual(visible_message_records[-2].content, "stream-reply")
            self.assertEqual(visible_message_records[-1].role, "system")

            all_records = storage.load_records(done.session_id, include_all_turns=True)
            message_records = [record for record in all_records if record.kind == "message"]
            self.assertEqual(message_records[-3].role, "user")
            self.assertEqual(message_records[-2].role, "assistant")
            self.assertEqual(message_records[-1].role, "system")
            self.assertEqual(
                message_records[-1].content,
                "The user interrupted this turn before it completed.",
            )

            session = storage.get_session(done.session_id)
            self.assertIsNotNone(session)
            self.assertTrue(session.pending_interruption_notice)  # type: ignore[union-attr]

    async def test_next_turn_includes_previous_task_interruption_notice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = _InterruptibleStreamingLLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )

            async def _collect_events():
                return [event async for event in loop.stream_user_input("hello")]

            task = asyncio.create_task(_collect_events())
            await llm_service.stream_started.wait()
            self.assertTrue(loop.request_stop())
            llm_service.release_stream.set()
            _ = await task

            result = await loop.handle_user_input("continue")
            self.assertEqual(result.response_text, "next-turn")
            self.assertEqual(len(llm_service.generate_requests), 1)

            request = llm_service.generate_requests[0]
            self.assertTrue(
                any(
                    message.role == "user"
                    and any(isinstance(part, TextPart) and part.text == "hello" for part in message.parts)
                    for message in request.messages
                )
            )
            self.assertTrue(
                any(
                    message.role == "assistant"
                    and any(isinstance(part, TextPart) and part.text == "stream-reply" for part in message.parts)
                    for message in request.messages
                )
            )
            interruption_messages = [
                part.text
                for message in request.messages
                if message.role == "system"
                for part in message.parts
                if isinstance(part, TextPart)
                and "The user interrupted the previous task." in part.text
            ]
            self.assertEqual(
                interruption_messages,
                [
                    "The user interrupted the previous task. Treat any partial output from it as incomplete."
                ],
            )

            session = storage.get_session(result.session_id)
            self.assertIsNotNone(session)
            self.assertFalse(session.pending_interruption_notice)  # type: ignore[union-attr]

    async def test_next_turn_includes_previous_task_superseded_notice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = _InterruptibleStreamingLLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )

            async def _collect_events():
                return [event async for event in loop.stream_user_input("hello")]

            task = asyncio.create_task(_collect_events())
            await llm_service.stream_started.wait()
            self.assertTrue(
                loop.request_stop(reason="superseded_by_user_message")
            )
            llm_service.release_stream.set()
            _ = await task

            active_session_id = loop.active_session_id()
            self.assertIsNotNone(active_session_id)
            all_records = storage.load_records(active_session_id, include_all_turns=True)
            self.assertTrue(
                any(
                    record.content
                    == "A newer user message superseded this turn before it completed."
                    for record in all_records
                    if record.role == "system"
                )
            )

            result = await loop.handle_user_input("continue")
            self.assertEqual(result.response_text, "next-turn")
            self.assertEqual(len(llm_service.generate_requests), 1)

            request = llm_service.generate_requests[0]
            superseded_messages = [
                part.text
                for message in request.messages
                if message.role == "system"
                for part in message.parts
                if isinstance(part, TextPart)
                and "superseded the previous task" in part.text
            ]
            self.assertIn(
                (
                    "A newer user message superseded the previous task. Handle the "
                    "current user message first. Use completed results from the older "
                    "task only if they are directly relevant."
                ),
                superseded_messages,
            )

            session = storage.get_session(result.session_id)
            self.assertIsNotNone(session)
            self.assertFalse(session.pending_interruption_notice)  # type: ignore[union-attr]
            self.assertIsNone(session.pending_interruption_notice_reason)  # type: ignore[union-attr]

    async def test_compaction_preserves_pending_superseded_notice_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = _InterruptibleStreamingLLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )

            async def _collect_events():
                return [event async for event in loop.stream_user_input("hello")]

            task = asyncio.create_task(_collect_events())
            await llm_service.stream_started.wait()
            self.assertTrue(
                loop.request_stop(reason="superseded_by_user_message")
            )
            llm_service.release_stream.set()
            _ = await task

            active_session_id = loop.active_session_id()
            self.assertIsNotNone(active_session_id)
            active_session = storage.get_session(active_session_id)
            self.assertIsNotNone(active_session)

            compacted = await loop._compact_session(active_session, reason="manual")
            self.assertIsNotNone(compacted)
            self.assertTrue(compacted.pending_interruption_notice)  # type: ignore[union-attr]
            self.assertEqual(
                compacted.pending_interruption_notice_reason,
                "superseded_by_user_message",
            )
