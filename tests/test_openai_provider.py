"""Unit tests for OpenAI provider request shaping."""

from __future__ import annotations

import base64
import unittest
from pathlib import Path
from types import SimpleNamespace

from jarvis.llm.config import OpenAIProviderSettings
from jarvis.llm.errors import ProviderTemporaryError, StreamProtocolError
from jarvis.llm.providers.openai_provider import OpenAIProvider
from jarvis.llm.types import (
    DoneEvent,
    ImagePart,
    LLMMessage,
    LLMRequest,
    ToolCall,
    ToolDefinition,
    ToolResultPart,
    UsageDeltaEvent,
)
from jarvis.tools.basic.file_patch.tool import build_file_patch_tool
from jarvis.tools.config import ToolSettings


class OpenAIProviderRequestShapeTests(unittest.TestCase):
    def test_assistant_history_uses_output_text_content_items(self) -> None:
        provider = OpenAIProvider(
            settings=OpenAIProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="gpt-5.2-2025-12-11",
            messages=(
                LLMMessage.text("system", "System prompt"),
                LLMMessage.text("user", "Hello"),
                LLMMessage.text("assistant", "Hi there"),
                LLMMessage.text("user", "Second turn"),
            ),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)
        content = kwargs["input"]

        self.assertEqual(content[0]["content"][0]["type"], "input_text")
        self.assertEqual(content[1]["content"][0]["type"], "input_text")
        self.assertEqual(content[2]["content"][0]["type"], "output_text")
        self.assertEqual(content[3]["content"][0]["type"], "input_text")

    def test_tool_roundtrip_uses_function_call_and_function_call_output_items(self) -> None:
        provider = OpenAIProvider(
            settings=OpenAIProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="gpt-5.2-2025-12-11",
            messages=(
                LLMMessage(
                    role="assistant",
                    parts=(
                        ToolCall(
                            call_id="bash_1",
                            name="bash",
                            arguments={"command": "pwd"},
                            raw_arguments='{"command":"pwd"}',
                        ),
                    ),
                ),
                LLMMessage(
                    role="tool",
                    parts=(
                        ToolResultPart(
                            call_id="bash_1",
                            name="bash",
                            content="Bash execution result\nstatus: success",
                        ),
                    ),
                ),
            ),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)
        content = kwargs["input"]

        self.assertEqual(content[0]["type"], "function_call")
        self.assertEqual(content[0]["call_id"], "bash_1")
        self.assertEqual(content[1]["type"], "function_call_output")
        self.assertEqual(content[1]["call_id"], "bash_1")

    def test_strict_tool_schema_normalizes_optional_fields_for_openai(self) -> None:
        provider = OpenAIProvider(
            settings=OpenAIProviderSettings(),
            default_timeout_seconds=60.0,
        )
        tool = ToolDefinition(
            name="bash",
            description="Run bash.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "number"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        )

        payload = provider._to_openai_tool(tool)
        parameters = payload["parameters"]

        self.assertEqual(parameters["required"], ["command", "timeout_seconds"])
        self.assertEqual(
            parameters["properties"]["timeout_seconds"],
            {
                "anyOf": [
                    {"type": "number"},
                    {"type": "null"},
                ]
            },
        )
        self.assertFalse(parameters["additionalProperties"])

    def test_extract_tool_calls_drops_null_for_optional_fields_in_strict_mode(self) -> None:
        provider = OpenAIProvider(
            settings=OpenAIProviderSettings(),
            default_timeout_seconds=60.0,
        )
        tool = ToolDefinition(
            name="bash",
            description="Run bash.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "number"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        )

        calls = provider._extract_tool_calls(
            response_output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_123",
                    name="bash",
                    arguments='{"command":"pwd","timeout_seconds":null}',
                )
            ],
            request_tools=(tool,),
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, {"command": "pwd"})

    def test_extract_tool_calls_recovers_required_null_validation_as_invalid_tool_call(self) -> None:
        provider = OpenAIProvider(
            settings=OpenAIProviderSettings(),
            default_timeout_seconds=60.0,
        )
        tool = ToolDefinition(
            name="bash",
            description="Run bash.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        )

        calls = provider._extract_tool_calls(
            response_output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_123",
                    name="bash",
                    arguments='{"command":null}',
                )
            ],
            request_tools=(tool,),
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call_id, "call_123")
        self.assertEqual(calls[0].name, "bash")
        self.assertEqual(calls[0].arguments, {"command": None})
        self.assertIn(
            "arguments failed schema validation",
            calls[0].provider_metadata["tool_call_validation_error"],
        )

    def test_file_patch_tool_schema_avoids_openai_unsupported_one_of(self) -> None:
        provider = OpenAIProvider(
            settings=OpenAIProviderSettings(),
            default_timeout_seconds=60.0,
        )
        settings = ToolSettings.from_workspace_dir(Path("/workspace"))
        tool = build_file_patch_tool(settings).definition

        payload = provider._to_openai_tool(tool)
        parameters = payload["parameters"]
        operation_items = parameters["properties"]["operations"]["items"]

        self.assertNotIn("oneOf", operation_items)
        self.assertEqual(operation_items["type"], "object")
        self.assertEqual(
            operation_items["properties"]["type"]["enum"],
            ["write", "replace", "insert_before", "insert_after", "delete"],
        )

    def test_image_input_uses_input_image_items(self) -> None:
        provider = OpenAIProvider(
            settings=OpenAIProviderSettings(),
            default_timeout_seconds=60.0,
        )
        image_url = f"data:image/png;base64,{base64.b64encode(b'png-bytes').decode('ascii')}"
        request = LLMRequest(
            model="gpt-5.4-2026-03-05",
            messages=(
                LLMMessage(
                    role="user",
                    parts=(
                        ImagePart(image_url=image_url, detail="original"),
                    ),
                ),
            ),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)
        image_item = kwargs["input"][0]["content"][0]
        self.assertEqual(image_item["type"], "input_image")
        self.assertEqual(image_item["image_url"], image_url)
        self.assertEqual(image_item["detail"], "original")

    def test_original_detail_downgrades_for_models_without_original_support(self) -> None:
        provider = OpenAIProvider(
            settings=OpenAIProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="gpt-5.2-2025-12-11",
            messages=(
                LLMMessage(
                    role="user",
                    parts=(
                        ImagePart.from_base64(
                            media_type="image/png",
                            data_base64=base64.b64encode(b"png-bytes").decode("ascii"),
                            detail="original",
                        ),
                    ),
                ),
            ),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)
        image_item = kwargs["input"][0]["content"][0]
        self.assertEqual(image_item["detail"], "high")


class _FakeAsyncEventStream:
    def __init__(self, events: list[object]) -> None:
        self._events = list(events)

    def __aiter__(self) -> "_FakeAsyncEventStream":
        return self

    async def __anext__(self) -> object:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class OpenAIProviderStreamingTests(unittest.IsolatedAsyncioTestCase):
    def _provider_with_stream(self, *events: object) -> OpenAIProvider:
        provider = OpenAIProvider(
            settings=OpenAIProviderSettings(),
            default_timeout_seconds=60.0,
        )

        async def _create(**_kwargs):
            return _FakeAsyncEventStream(list(events))

        provider._client = SimpleNamespace(responses=SimpleNamespace(create=_create))  # type: ignore[assignment]
        return provider

    def _request(self) -> LLMRequest:
        return LLMRequest(
            model="gpt-5.4-2026-03-05",
            messages=(LLMMessage.text("user", "hello"),),
        )

    def _response(
        self,
        *,
        status: str,
        incomplete_reason: str | None = None,
        output_text: str = "done",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id="resp_123",
            model="gpt-5.4-2026-03-05",
            status=status,
            output_text=output_text,
            output=[],
            usage=SimpleNamespace(input_tokens=11, output_tokens=3, total_tokens=14),
            incomplete_details=(
                SimpleNamespace(reason=incomplete_reason)
                if incomplete_reason is not None
                else None
            ),
        )

    async def test_stream_generate_treats_response_incomplete_as_terminal(self) -> None:
        provider = self._provider_with_stream(
            SimpleNamespace(
                type="response.incomplete",
                response=self._response(
                    status="incomplete",
                    incomplete_reason="max_tokens",
                    output_text="partial",
                ),
            )
        )

        events = [event async for event in provider.stream_generate(self._request())]

        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], UsageDeltaEvent)
        self.assertIsInstance(events[1], DoneEvent)
        self.assertEqual(events[1].response.text, "partial")
        self.assertEqual(events[1].response.finish_reason, "length")
        self.assertEqual(
            events[1].response.provider_metadata["incomplete_reason"],
            "max_output_tokens",
        )

    async def test_stream_generate_raises_temporary_error_when_stream_closes_before_any_output(
        self,
    ) -> None:
        provider = self._provider_with_stream()

        with self.assertRaisesRegex(ProviderTemporaryError, "without a terminal event"):
            _events = [event async for event in provider.stream_generate(self._request())]

    async def test_stream_generate_raises_protocol_error_after_partial_output_without_terminal_event(
        self,
    ) -> None:
        provider = self._provider_with_stream(
            SimpleNamespace(type="response.output_text.delta", delta="hel")
        )

        with self.assertRaisesRegex(
            StreamProtocolError,
            "last_event=response.output_text.delta",
        ):
            _events = [event async for event in provider.stream_generate(self._request())]


class GeminiNormalizationRegressionTests(unittest.TestCase):
    def test_gemini_normalization_does_not_require_response_text_accessor(self) -> None:
        from jarvis.llm.config import GeminiProviderSettings
        from jarvis.llm.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(
            settings=GeminiProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="gemini-3-flash-preview",
            messages=(LLMMessage.text("user", "hello"),),
            tools=(
                ToolDefinition(
                    name="bash",
                    description="Run bash.",
                    input_schema={
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                ),
            ),
        )

        class _ResponseWithoutTextAccessor:
            response_id = "resp_123"
            model_version = "gemini-3-flash-preview"
            usage_metadata = None

            def __init__(self) -> None:
                function_call = SimpleNamespace(name="bash", id="bash", args={"command": "pwd"})
                self.candidates = [
                    SimpleNamespace(
                        finish_reason="STOP",
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    text=None,
                                    function_call=function_call,
                                    thought_signature=b"sig_123",
                                ),
                                SimpleNamespace(text="done", function_call=None),
                            ]
                        ),
                    )
                ]

            @property
            def text(self) -> str:
                raise AssertionError("response.text should not be accessed during normalization.")

        response = provider._normalize_generate_response(
            request=request,
            response=_ResponseWithoutTextAccessor(),
        )
        self.assertEqual(response.text, "done")
        self.assertEqual(response.tool_calls[0].name, "bash")
        self.assertEqual(
            response.tool_calls[0].provider_metadata["thought_signature_b64"],
            base64.b64encode(b"sig_123").decode("ascii"),
        )
