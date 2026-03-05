"""Unit tests for provider-specific multi-turn request shaping."""

from __future__ import annotations

import unittest

from llm.config import (
    AnthropicProviderSettings,
    GeminiProviderSettings,
    OpenRouterProviderSettings,
)
from llm.providers.anthropic_provider import AnthropicProvider
from llm.providers.gemini_provider import GeminiProvider
from llm.providers.openrouter_provider import OpenRouterProvider
from llm.types import LLMMessage, LLMRequest


class AnthropicProviderRequestShapeTests(unittest.TestCase):
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
