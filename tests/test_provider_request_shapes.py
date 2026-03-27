"""Unit tests for provider-specific multi-turn request shaping."""

from __future__ import annotations

import base64
import unittest

from jarvis.llm.config import (
    AnthropicProviderSettings,
    GeminiProviderSettings,
    LMStudioProviderSettings,
    OpenRouterProviderSettings,
)
from jarvis.llm.providers.anthropic_provider import AnthropicProvider
from jarvis.llm.providers.gemini_provider import GeminiProvider
from jarvis.llm.providers.lmstudio_provider import LMStudioProvider
from jarvis.llm.providers.openrouter_provider import OpenRouterProvider
from jarvis.llm.types import ImagePart, LLMMessage, LLMRequest, TextPart, ToolCall, ToolDefinition, ToolResultPart


class AnthropicProviderRequestShapeTests(unittest.TestCase):
    def test_prompt_caching_uses_top_level_cache_control(self) -> None:
        provider = AnthropicProvider(
            settings=AnthropicProviderSettings(prompt_cache_ttl="5m"),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="claude-sonnet-4-6",
            max_output_tokens=1024,
            messages=(LLMMessage.text("user", "Hello"),),
        )

        kwargs = provider._build_messages_create_kwargs(request)
        self.assertEqual(
            kwargs["cache_control"],
            {
                "type": "ephemeral",
                "ttl": "5m",
            },
        )

    def test_prompt_caching_marks_final_system_block(self) -> None:
        provider = AnthropicProvider(
            settings=AnthropicProviderSettings(prompt_cache_ttl="5m"),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="claude-sonnet-4-6",
            max_output_tokens=1024,
            messages=(
                LLMMessage.text("system", "System prompt A"),
                LLMMessage.text("system", "System prompt B"),
                LLMMessage.text("user", "Hello"),
            ),
        )

        kwargs = provider._build_messages_create_kwargs(request)
        self.assertEqual(
            kwargs["system"],
            [
                {"type": "text", "text": "System prompt A"},
                {
                    "type": "text",
                    "text": "System prompt B",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
            ],
        )

    def test_multi_turn_history_preserves_assistant_role(self) -> None:
        provider = AnthropicProvider(
            settings=AnthropicProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="claude-sonnet-4-6",
            max_output_tokens=1024,
            messages=(
                LLMMessage.text("system", "System prompt"),
                LLMMessage.text("user", "Hello"),
                LLMMessage.text("assistant", "Hi there"),
                LLMMessage.text("user", "Second turn"),
            ),
        )

        kwargs = provider._build_messages_create_kwargs(request)
        self.assertEqual(kwargs["system"], "System prompt")
        self.assertEqual(kwargs["messages"][0]["role"], "user")
        self.assertEqual(kwargs["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(kwargs["messages"][1]["role"], "assistant")
        self.assertEqual(kwargs["messages"][1]["content"][0]["type"], "text")
        self.assertEqual(kwargs["messages"][2]["role"], "user")

    def test_tool_roundtrip_uses_tool_use_and_tool_result_blocks(self) -> None:
        provider = AnthropicProvider(
            settings=AnthropicProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="claude-sonnet-4-6",
            max_output_tokens=1024,
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

        kwargs = provider._build_messages_create_kwargs(request)
        self.assertEqual(kwargs["messages"][0]["role"], "assistant")
        self.assertEqual(kwargs["messages"][0]["content"][0]["type"], "tool_use")
        self.assertEqual(kwargs["messages"][1]["role"], "user")
        self.assertEqual(kwargs["messages"][1]["content"][0]["type"], "tool_result")
        self.assertEqual(kwargs["messages"][1]["content"][0]["tool_use_id"], "bash_1")

    def test_image_input_uses_base64_image_block(self) -> None:
        provider = AnthropicProvider(
            settings=AnthropicProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="claude-sonnet-4-6",
            max_output_tokens=1024,
            messages=(
                LLMMessage(
                    role="user",
                    parts=(
                        ImagePart.from_base64(
                            media_type="image/png",
                            data_base64=base64.b64encode(b"png-bytes").decode("ascii"),
                        ),
                        TextPart(text="Describe this image."),
                    ),
                ),
            ),
        )

        kwargs = provider._build_messages_create_kwargs(request)
        image_block = kwargs["messages"][0]["content"][0]
        self.assertEqual(image_block["type"], "image")
        self.assertEqual(image_block["source"]["type"], "base64")
        self.assertEqual(image_block["source"]["media_type"], "image/png")
        self.assertEqual(
            image_block["source"]["data"],
            base64.b64encode(b"png-bytes").decode("ascii"),
        )

    def test_normalize_message_response_surfaces_cache_usage_metadata(self) -> None:
        provider = AnthropicProvider(
            settings=AnthropicProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="claude-sonnet-4-6",
            max_output_tokens=1024,
            messages=(LLMMessage.text("user", "Hello"),),
        )

        class _FakeCacheCreation:
            ephemeral_5m_input_tokens = 1200
            ephemeral_1h_input_tokens = 0

        class _FakeUsage:
            input_tokens = 10
            output_tokens = 5
            cache_read_input_tokens = 1200
            cache_creation_input_tokens = 1200
            cache_creation = _FakeCacheCreation()

        class _FakeTextBlock:
            type = "text"
            text = "Hi"

        class _FakeResponse:
            model = "claude-sonnet-4-6"
            id = "msg_123"
            stop_reason = "end_turn"
            content = [_FakeTextBlock()]
            usage = _FakeUsage()

        response = provider._normalize_message_response(
            request=request,
            response=_FakeResponse(),
        )
        self.assertEqual(response.provider_metadata["stop_reason"], "end_turn")
        self.assertEqual(response.provider_metadata["cache_read_input_tokens"], 1200)
        self.assertEqual(response.provider_metadata["cache_creation_input_tokens"], 1200)
        self.assertEqual(
            response.provider_metadata["cache_creation"],
            {"ephemeral_5m_input_tokens": 1200, "ephemeral_1h_input_tokens": 0},
        )


class GeminiProviderRequestShapeTests(unittest.TestCase):
    def test_multi_turn_history_maps_assistant_to_model_role(self) -> None:
        provider = GeminiProvider(
            settings=GeminiProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="gemini-3-flash-preview",
            messages=(
                LLMMessage.text("system", "System prompt"),
                LLMMessage.text("user", "Hello"),
                LLMMessage.text("assistant", "Hi there"),
                LLMMessage.text("user", "Second turn"),
            ),
        )

        contents, config = provider._build_generate_payload(request)
        self.assertEqual(config["system_instruction"], "System prompt")
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[0]["parts"][0]["text"], "Hello")
        self.assertEqual(contents[1]["role"], "model")
        self.assertEqual(contents[1]["parts"][0]["text"], "Hi there")
        self.assertEqual(contents[2]["role"], "user")

    def test_tool_definitions_use_parameters_json_schema(self) -> None:
        provider = GeminiProvider(
            settings=GeminiProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="gemini-3-flash-preview",
            messages=(LLMMessage.text("user", "List files."),),
            tools=(
                ToolDefinition(
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
                ),
            ),
        )

        _contents, config = provider._build_generate_payload(request)
        declaration = config["tools"][0]["function_declarations"][0]
        self.assertIn("parameters_json_schema", declaration)
        self.assertNotIn("parameters", declaration)
        self.assertEqual(
            declaration["parameters_json_schema"]["additionalProperties"],
            False,
        )

    def test_tool_roundtrip_uses_function_call_and_function_response_parts(self) -> None:
        provider = GeminiProvider(
            settings=GeminiProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="gemini-3-flash-preview",
            messages=(
                LLMMessage(
                    role="assistant",
                    parts=(
                        ToolCall(
                            call_id="bash_1",
                            name="bash",
                            arguments={"command": "pwd"},
                            raw_arguments='{"command":"pwd"}',
                            provider_metadata={
                                "thought_signature_b64": base64.b64encode(b"sig_123").decode(
                                    "ascii"
                                )
                            },
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

        contents, _config = provider._build_generate_payload(request)
        self.assertEqual(contents[0]["role"], "model")
        self.assertIn("function_call", contents[0]["parts"][0])
        self.assertEqual(contents[0]["parts"][0]["function_call"]["name"], "bash")
        self.assertEqual(contents[0]["parts"][0]["thought_signature"], b"sig_123")
        self.assertEqual(contents[1]["role"], "user")
        self.assertIn("function_response", contents[1]["parts"][0])
        self.assertEqual(contents[1]["parts"][0]["function_response"]["name"], "bash")

    def test_image_input_uses_inline_data_part(self) -> None:
        provider = GeminiProvider(
            settings=GeminiProviderSettings(),
            default_timeout_seconds=60.0,
        )
        image_bytes = b"png-bytes"
        request = LLMRequest(
            model="gemini-3-flash-preview",
            messages=(
                LLMMessage(
                    role="user",
                    parts=(
                        ImagePart.from_base64(
                            media_type="image/png",
                            data_base64=base64.b64encode(image_bytes).decode("ascii"),
                        ),
                        TextPart(text="Describe this image."),
                    ),
                ),
            ),
        )

        contents, _config = provider._build_generate_payload(request)
        image_part = contents[0]["parts"][0]
        self.assertEqual(image_part.inline_data.mime_type, "image/png")
        self.assertEqual(image_part.inline_data.data, image_bytes)


class OpenRouterProviderRequestShapeTests(unittest.TestCase):
    def test_multi_turn_history_preserves_assistant_role(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="minimax/minimax-m2.5",
            messages=(
                LLMMessage.text("system", "System prompt"),
                LLMMessage.text("user", "Hello"),
                LLMMessage.text("assistant", "Hi there"),
                LLMMessage.text("user", "Second turn"),
            ),
        )

        payload = provider._build_chat_payload(request)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "System prompt"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "Hello"})
        self.assertEqual(payload["messages"][2], {"role": "assistant", "content": "Hi there"})
        self.assertEqual(payload["messages"][3], {"role": "user", "content": "Second turn"})

    def test_tool_roundtrip_uses_assistant_tool_calls_and_tool_role_messages(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="minimax/minimax-m2.5",
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

        payload = provider._build_chat_payload(request)
        self.assertEqual(payload["messages"][0]["role"], "assistant")
        self.assertEqual(payload["messages"][0]["tool_calls"][0]["id"], "bash_1")
        self.assertIsNone(payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][1]["role"], "tool")
        self.assertEqual(payload["messages"][1]["tool_call_id"], "bash_1")

    def test_image_input_uses_image_url_content_item(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            default_timeout_seconds=60.0,
        )
        image_url = f"data:image/png;base64,{base64.b64encode(b'png-bytes').decode('ascii')}"
        request = LLMRequest(
            model="minimax/minimax-m2.5",
            messages=(
                LLMMessage(
                    role="user",
                    parts=(
                        ImagePart(image_url=image_url),
                        TextPart(text="Describe this image."),
                    ),
                ),
            ),
        )

        payload = provider._build_chat_payload(request)
        content = payload["messages"][0]["content"]
        image_item = next(item for item in content if item["type"] == "image_url")
        self.assertEqual(image_item["image_url"]["url"], image_url)


class LMStudioProviderRequestShapeTests(unittest.TestCase):
    def test_multi_turn_history_preserves_assistant_role(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="loaded-model",
            messages=(
                LLMMessage.text("system", "System prompt"),
                LLMMessage.text("user", "Hello"),
                LLMMessage.text("assistant", "Hi there"),
                LLMMessage.text("user", "Second turn"),
            ),
        )

        input_items = [
            item
            for message in request.messages
            for item in provider._to_response_input_items(message)
        ]
        payload = provider._build_response_payload(
            request,
            input_items=input_items,
            previous_response_id=None,
            stream=False,
        )
        self.assertEqual(
            payload["input"][0],
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "System prompt"}],
            },
        )
        self.assertEqual(
            payload["input"][1],
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            },
        )
        self.assertEqual(
            payload["input"][2],
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there"}],
            },
        )
        self.assertEqual(
            payload["input"][3],
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Second turn"}],
            },
        )

    def test_tool_roundtrip_uses_assistant_tool_calls_and_tool_role_messages(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="loaded-model",
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

        input_items = [
            item
            for message in request.messages
            for item in provider._to_response_input_items(message)
        ]
        payload = provider._build_response_payload(
            request,
            input_items=input_items,
            previous_response_id=None,
            stream=False,
        )
        self.assertEqual(
            payload["input"][0],
            {
                "type": "function_call",
                "call_id": "bash_1",
                "name": "bash",
                "arguments": '{"command":"pwd"}',
            },
        )
        self.assertEqual(
            payload["input"][1],
            {
                "type": "function_call_output",
                "call_id": "bash_1",
                "output": "Bash execution result\nstatus: success",
            },
        )

    def test_image_input_uses_image_url_content_item(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )
        image_url = f"data:image/png;base64,{base64.b64encode(b'png-bytes').decode('ascii')}"
        request = LLMRequest(
            model="loaded-model",
            messages=(
                LLMMessage(
                    role="user",
                    parts=(
                        ImagePart(image_url=image_url),
                        TextPart(text="Describe this image."),
                    ),
                ),
            ),
        )

        input_items = [
            item
            for message in request.messages
            for item in provider._to_response_input_items(message)
        ]
        payload = provider._build_response_payload(
            request,
            input_items=input_items,
            previous_response_id=None,
            stream=False,
        )
        content = payload["input"][0]["content"]
        image_item = next(item for item in content if item["type"] == "input_image")
        self.assertEqual(image_item["image_url"], image_url)
