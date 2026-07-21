"""Native streaming tests for the Anthropic provider adapter."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from jarvis.llm.config import AnthropicProviderSettings
from jarvis.llm.providers.anthropic_provider import AnthropicProvider
from jarvis.llm.types import (
    DoneEvent,
    LLMMessage,
    LLMRequest,
    ProviderActivityEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolDefinition,
    UsageDeltaEvent,
)


class _FakeAnthropicStream:
    def __init__(self, events: list[Any], final_message: Any) -> None:
        self._events = iter(events)
        self._final_message = final_message

    async def __aenter__(self) -> "_FakeAnthropicStream":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def __aiter__(self) -> "_FakeAnthropicStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def get_final_message(self) -> Any:
        return self._final_message


class _FakeAnthropicMessages:
    def __init__(self, stream: _FakeAnthropicStream) -> None:
        self._stream = stream
        self.stream_kwargs: dict[str, Any] | None = None

    def stream(self, **kwargs: Any) -> _FakeAnthropicStream:
        self.stream_kwargs = kwargs
        return self._stream


class AnthropicProviderStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_generate_uses_native_events_and_hides_reasoning(self) -> None:
        provider = AnthropicProvider(
            settings=AnthropicProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="claude-sonnet-4-6",
            max_output_tokens=1024,
            messages=(LLMMessage.text("user", "run pwd"),),
            tools=(
                ToolDefinition(
                    name="bash",
                    input_schema={
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                ),
            ),
        )

        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(id="msg_123"),
            ),
            SimpleNamespace(type="ping"),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(
                    type="thinking_delta",
                    thinking="private reasoning",
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Done"),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=1,
                content_block=SimpleNamespace(
                    type="tool_use",
                    id="call_1",
                    name="bash",
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(
                    type="input_json_delta",
                    partial_json='{"command":"pwd"}',
                ),
            ),
            SimpleNamespace(type="message_stop"),
        ]
        final_message = SimpleNamespace(
            id="msg_123",
            model="claude-sonnet-4-6",
            stop_reason="tool_use",
            content=[
                SimpleNamespace(type="text", text="Done"),
                SimpleNamespace(
                    type="tool_use",
                    id="call_1",
                    name="bash",
                    input={"command": "pwd"},
                ),
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        fake_messages = _FakeAnthropicMessages(
            _FakeAnthropicStream(events, final_message)
        )
        provider._client = SimpleNamespace(messages=fake_messages)  # type: ignore[assignment]

        streamed = [event async for event in provider.stream_generate(request)]

        self.assertIsNotNone(fake_messages.stream_kwargs)
        self.assertEqual(fake_messages.stream_kwargs["model"], request.model)
        self.assertNotIn("timeout", fake_messages.stream_kwargs)
        self.assertEqual(
            [event.delta for event in streamed if isinstance(event, TextDeltaEvent)],
            ["Done"],
        )
        tool_events = [
            event for event in streamed if isinstance(event, ToolCallDeltaEvent)
        ]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].call_id, "call_1")
        self.assertEqual(tool_events[0].arguments_delta, '{"command":"pwd"}')
        activity_types = [
            event.provider_event_type
            for event in streamed
            if isinstance(event, ProviderActivityEvent)
        ]
        self.assertIn("ping", activity_types)
        self.assertIn("content_block_delta", activity_types)
        self.assertNotIn("private reasoning", repr(streamed))
        self.assertEqual(
            len([event for event in streamed if isinstance(event, UsageDeltaEvent)]),
            1,
        )
        self.assertIsInstance(streamed[-1], DoneEvent)
        self.assertEqual(streamed[-1].response.response_id, "msg_123")
        self.assertEqual(streamed[-1].response.finish_reason, "tool_calls")

