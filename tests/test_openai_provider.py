"""Unit tests for OpenAI provider request shaping."""

from __future__ import annotations

import unittest

from llm.config import OpenAIProviderSettings
from llm.providers.openai_provider import OpenAIProvider
from llm.types import LLMMessage, LLMRequest


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
