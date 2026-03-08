"""Unit tests for AgentLoop streaming turn behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core import AgentAssistantMessageEvent, AgentLoop, AgentTurnDoneEvent
from llm import DoneEvent, LLMRequest, LLMResponse, LLMUsage, TextDeltaEvent, TextPart
from storage import SessionStorage
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


class AgentLoopStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_user_input_emits_delta_and_done_and_persists_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.storage_dir)
            loop = AgentLoop(
                llm_service=_FakeStreamingLLMService(),
                settings=settings,
                storage=storage,
            )

            events = [event async for event in loop.stream_user_input("hello")]
            self.assertEqual(
                [event.type for event in events],
                ["text_delta", "text_delta", "assistant_message", "done"],
            )
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
            storage = SessionStorage(settings.storage_dir)
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

    async def test_handle_user_input_injects_transient_turn_datetime_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.storage_dir)
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

            persisted_records = storage.load_records(result.session_id)
            self.assertFalse(
                any(
                    "System context auto-appended for this turn only." in record.content
                    for record in persisted_records
                )
            )
