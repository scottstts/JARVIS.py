"""Unit tests for LLM provider configuration models."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from jarvis.llm.config import (
    EmbeddingSettings,
    GrokProviderSettings,
    LLMSettings,
    LMStudioProviderSettings,
    OpenRouterProviderSettings,
)
from jarvis.llm.errors import LLMConfigurationError


class LMStudioConfigTests(unittest.TestCase):
    def test_lmstudio_is_not_allowed_as_embedding_provider(self) -> None:
        with self.assertRaisesRegex(LLMConfigurationError, "JARVIS_EMBEDDING_PROVIDER must be one of"):
            EmbeddingSettings(provider="lmstudio", model="embed-model")

    def test_lmstudio_can_be_the_default_provider(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = LLMSettings(
                default_provider="lmstudio",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            )

        self.assertEqual(settings.default_provider, "lmstudio")
        self.assertEqual(settings.lmstudio.base_url, "http://127.0.0.1:1234")

    def test_lmstudio_settings_reads_base_url_override(self) -> None:
        with patch.dict(
            os.environ,
            {"JARVIS_LMSTUDIO_BASE_URL": "http://127.0.0.1:4321"},
            clear=True,
        ):
            settings = LMStudioProviderSettings.from_env()

        self.assertEqual(settings.base_url, "http://127.0.0.1:4321")


class OpenRouterConfigTests(unittest.TestCase):
    def test_openrouter_settings_read_local_site_url_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = OpenRouterProviderSettings.from_env()

        self.assertEqual(settings.site_url, "https://github.com/scottstts/JARVIS.py")
        self.assertEqual(settings.app_name, "Jarvis")

    def test_openrouter_settings_reads_site_url_override(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_SITE_URL": "https://jarvis.example",
                "OPENROUTER_APP_NAME": "Jarvis Dev",
            },
            clear=True,
        ):
            settings = OpenRouterProviderSettings.from_env()

        self.assertEqual(settings.site_url, "https://jarvis.example")
        self.assertEqual(settings.app_name, "Jarvis Dev")

    def test_openrouter_settings_reads_reasoning_effort_override(self) -> None:
        with patch.dict(
            os.environ,
            {"JARVIS_OPENROUTER_REASONING_EFFORT": "xhigh"},
            clear=True,
        ):
            settings = OpenRouterProviderSettings.from_env()

        self.assertEqual(settings.reasoning_effort, "xhigh")


class LLMSettingsTests(unittest.TestCase):
    def test_timeout_policy_defaults_are_long_running_safe(self) -> None:
        with patch.dict(
            os.environ,
            {
                "JARVIS_LLM_DEFAULT_PROVIDER": "openai",
                "JARVIS_EMBEDDING_PROVIDER": "openai",
                "JARVIS_EMBEDDING_MODEL": "text-embedding-test",
            },
            clear=True,
        ):
            settings = LLMSettings.from_env()

        self.assertEqual(settings.request_deadline_seconds, 3600.0)
        self.assertEqual(settings.connect_timeout_seconds, 30.0)
        self.assertEqual(settings.read_timeout_seconds, 3600.0)

    def test_timeout_policy_reads_env_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "JARVIS_LLM_DEFAULT_PROVIDER": "openai",
                "JARVIS_EMBEDDING_PROVIDER": "openai",
                "JARVIS_EMBEDDING_MODEL": "text-embedding-test",
                "JARVIS_LLM_REQUEST_DEADLINE_SECONDS": "120",
                "JARVIS_LLM_CONNECT_TIMEOUT_SECONDS": "12",
                "JARVIS_LLM_READ_TIMEOUT_SECONDS": "90",
            },
            clear=True,
        ):
            settings = LLMSettings.from_env()

        self.assertEqual(settings.request_deadline_seconds, 120.0)
        self.assertEqual(settings.connect_timeout_seconds, 12.0)
        self.assertEqual(settings.read_timeout_seconds, 90.0)

    def test_timeout_policy_rejects_non_positive_values(self) -> None:
        with self.assertRaisesRegex(
            LLMConfigurationError,
            "JARVIS_LLM_CONNECT_TIMEOUT_SECONDS",
        ):
            LLMSettings(
                default_provider="openai",
                embedding=EmbeddingSettings(
                    provider="openai",
                    model="text-embedding-test",
                ),
                connect_timeout_seconds=0,
            )


class GrokConfigTests(unittest.TestCase):
    def test_grok_can_be_the_default_provider(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = LLMSettings(
                default_provider="grok",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            )

        self.assertEqual(settings.default_provider, "grok")
        self.assertEqual(settings.grok.chat_model, "grok-4.3")
        self.assertEqual(settings.grok.reasoning_effort, "high")

    def test_grok_settings_reads_model_override(self) -> None:
        with patch.dict(
            os.environ,
            {
                "JARVIS_GROK_CHAT_MODEL": "grok-test-model",
                "JARVIS_GROK_TEMPERATURE": "0.2",
                "JARVIS_GROK_MAX_OUTPUT_TOKENS": "2048",
                "JARVIS_GROK_REASONING_EFFORT": "medium",
            },
            clear=True,
        ):
            settings = GrokProviderSettings.from_env()

        self.assertEqual(settings.chat_model, "grok-test-model")
        self.assertEqual(settings.temperature, 0.2)
        self.assertEqual(settings.max_output_tokens, 2048)
        self.assertEqual(settings.reasoning_effort, "medium")
