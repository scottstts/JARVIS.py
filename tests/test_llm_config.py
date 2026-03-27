"""Unit tests for LLM provider configuration models."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from jarvis import settings as app_settings

from jarvis.llm.config import EmbeddingSettings, LLMSettings, LMStudioProviderSettings
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
