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


class LLMSettingsTests(unittest.TestCase):
    def test_generation_timeout_is_optional_and_read_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "JARVIS_LLM_DEFAULT_PROVIDER": "openai",
                "JARVIS_EMBEDDING_PROVIDER": "openai",
                "JARVIS_EMBEDDING_MODEL": "text-embedding-test",
                "JARVIS_LLM_GENERATION_TIMEOUT_SECONDS": "300",
            },
            clear=True,
        ):
            settings = LLMSettings.from_env()

        self.assertEqual(settings.request_timeout_seconds, 60.0)
        self.assertEqual(settings.generation_timeout_seconds, 300.0)

    def test_generation_timeout_must_be_positive(self) -> None:
        with self.assertRaisesRegex(
            LLMConfigurationError,
            "JARVIS_LLM_GENERATION_TIMEOUT_SECONDS must be > 0",
        ):
            LLMSettings(
                default_provider="openai",
                generation_timeout_seconds=0,
                embedding=EmbeddingSettings(
                    provider="openai",
                    model="text-embedding-test",
                ),
            )


class GrokConfigTests(unittest.TestCase):
    def test_grok_can_be_the_default_provider(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = LLMSettings(
                default_provider="grok",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            )

        self.assertEqual(settings.default_provider, "grok")
        self.assertEqual(settings.grok.chat_model, "grok-4.20-0309-non-reasoning")

    def test_grok_settings_reads_model_override(self) -> None:
        with patch.dict(
            os.environ,
            {
                "JARVIS_GROK_CHAT_MODEL": "grok-test-model",
                "JARVIS_GROK_TEMPERATURE": "0.2",
                "JARVIS_GROK_MAX_OUTPUT_TOKENS": "2048",
            },
            clear=True,
        ):
            settings = GrokProviderSettings.from_env()

        self.assertEqual(settings.chat_model, "grok-test-model")
        self.assertEqual(settings.temperature, 0.2)
        self.assertEqual(settings.max_output_tokens, 2048)
