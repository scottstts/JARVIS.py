"""Unit tests for LLM provider configuration models."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from jarvis import settings as app_settings

from jarvis.llm.config import (
    EmbeddingSettings,
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
        settings = LLMSettings(
            default_provider="lmstudio",
            embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
        )

        self.assertEqual(settings.default_provider, "lmstudio")
        self.assertEqual(settings.lmstudio.base_url, app_settings.JARVIS_LMSTUDIO_BASE_URL)

    def test_lmstudio_settings_reads_base_url_override(self) -> None:
        with patch.dict(
            os.environ,
            {"JARVIS_LMSTUDIO_BASE_URL": "http://127.0.0.1:4321"},
            clear=True,
        ):
            settings = LMStudioProviderSettings.from_env()

        self.assertEqual(settings.base_url, "http://127.0.0.1:4321")


class OpenRouterConfigTests(unittest.TestCase):
    def test_openrouter_settings_read_yaml_site_url_default(self) -> None:
        settings = OpenRouterProviderSettings.from_env()

        self.assertEqual(settings.site_url, app_settings.OPENROUTER_SITE_URL)
        self.assertEqual(settings.app_name, app_settings.OPENROUTER_APP_NAME)

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
