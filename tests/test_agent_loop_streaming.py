"""Unit tests for AgentLoop streaming turn behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core import AgentAssistantMessageEvent, AgentLoop, AgentTurnDoneEvent
from llm import DoneEvent, LLMResponse, LLMUsage, TextDeltaEvent
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
