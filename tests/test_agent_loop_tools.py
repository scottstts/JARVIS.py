"""AgentLoop tests covering tool execution rounds."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core import AgentLoop, AgentTurnDoneEvent
from llm import DoneEvent, LLMRequest, LLMResponse, LLMUsage, ToolCall, ToolResultPart
from storage import SessionStorage
from tests.helpers import build_core_settings


def _build_response(
    text: str,
    *,
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="fake-chat",
        text=text,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,  # type: ignore[arg-type]
        usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        response_id="resp_fake",
    )


class _FakeToolLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        self._assert_bash_registered(request)

        if self.generate_calls == 1 and self.stream_calls == 0:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="bash_1",
                        name="bash",
                        arguments={"command": "printf 'hello' | tee note.txt"},
                        raw_arguments='{"command":"printf \\"hello\\" | tee note.txt"}',
                    )
                ],
                finish_reason="tool_calls",
            )

        assistant_message = request.messages[-2]
        tool_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected tool result message before follow-up model call.")

        assistant_tool_parts = [
            part for part in assistant_message.parts if isinstance(part, ToolCall)
        ]
        if not assistant_tool_parts:
            raise AssertionError("Expected assistant message to contain tool-call parts.")
        if assistant_tool_parts[0].name != "bash":
            raise AssertionError("Expected bash tool call in assistant history.")

        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected one tool result part before follow-up model call.")
        if "note.txt" not in tool_result_parts[0].content:
            raise AssertionError("Expected tool result content to mention written file.")
        return _build_response("File written.")

    async def stream_generate(self, request: LLMRequest):
        self.stream_calls += 1
        self._assert_bash_registered(request)
        yield DoneEvent(
            response=_build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="bash_stream_1",
                        name="bash",
                        arguments={"command": "printf 'hello' | tee note.txt"},
                        raw_arguments='{"command":"printf \\"hello\\" | tee note.txt"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        )

    def _assert_bash_registered(self, request: LLMRequest) -> None:
        names = [tool.name for tool in request.tools]
        if names != ["bash"]:
            raise AssertionError(f"Expected only bash tool to be registered, got {names}.")


class AgentLoopToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_handle_user_input_executes_bash_tool_round_and_persists_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.storage_dir)
            loop = AgentLoop(
                llm_service=_FakeToolLLMService(),
                settings=settings,
                storage=storage,
            )

            result = await loop.handle_user_input("Write hello into note.txt.")

            self.assertEqual(result.response_text, "File written.")
            self.assertEqual(
                (settings.workspace_dir / "note.txt").read_text(encoding="utf-8"),
                "hello",
            )

            records = storage.load_records(result.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-4].role, "user")
            self.assertEqual(message_records[-3].role, "assistant")
            self.assertEqual(message_records[-2].role, "tool")
            self.assertEqual(message_records[-1].role, "assistant")
            self.assertEqual(message_records[-3].content, "")
            self.assertEqual(message_records[-3].metadata["tool_calls"][0]["name"], "bash")
            self.assertIn("Bash execution result", message_records[-2].content)

    async def test_stream_user_input_completes_tool_round_and_emits_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.storage_dir)
            loop = AgentLoop(
                llm_service=_FakeToolLLMService(),
                settings=settings,
                storage=storage,
            )

            events = [event async for event in loop.stream_user_input("Write hello into note.txt.")]

            self.assertEqual([event.type for event in events], ["text_delta", "done"])
            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertEqual(done.response_text, "File written.")
