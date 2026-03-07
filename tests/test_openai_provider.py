"""Unit tests for OpenAI provider request shaping."""

from __future__ import annotations

import base64
import unittest
from types import SimpleNamespace

from llm.config import OpenAIProviderSettings
from llm.errors import ToolCallValidationError
from llm.providers.openai_provider import OpenAIProvider
from llm.types import ImagePart, LLMMessage, LLMRequest, ToolCall, ToolDefinition, ToolResultPart


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

    def test_extract_tool_calls_keeps_required_null_validation(self) -> None:
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

        with self.assertRaises(ToolCallValidationError):
            provider._extract_tool_calls(
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


class GeminiNormalizationRegressionTests(unittest.TestCase):
    def test_gemini_normalization_does_not_require_response_text_accessor(self) -> None:
        from llm.config import GeminiProviderSettings
        from llm.providers.gemini_provider import GeminiProvider

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
