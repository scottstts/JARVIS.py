"""Lifecycle tests for LLMService provider cleanup."""

from __future__ import annotations

import unittest

from llm.config import EmbeddingSettings, LLMSettings
from llm.protocols import ProviderCapabilities
from llm.service import LLMService


class _FakeProvider:
    def __init__(self, name: str) -> None:
        self._name = name
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    async def generate(self, request):
        raise NotImplementedError

    async def stream_generate(self, request):
        raise NotImplementedError
        yield None

    async def embed(self, request):
        raise NotImplementedError

    async def aclose(self) -> None:
        self.closed = True


class LLMServiceLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_aclose_calls_aclose_on_all_registered_providers(self) -> None:
        provider_one = _FakeProvider("openai")
        provider_two = _FakeProvider("anthropic")
        settings = LLMSettings(
            default_provider="openai",
            embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
        )
        service = LLMService(
            settings=settings,
            providers=(provider_one, provider_two),
        )

        await service.aclose()
        self.assertTrue(provider_one.closed)
        self.assertTrue(provider_two.closed)

    async def test_default_registry_includes_lmstudio_provider(self) -> None:
        service = LLMService(
            settings=LLMSettings(
                default_provider="lmstudio",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            )
        )

        try:
            self.assertEqual(service.registry.get("lmstudio").name, "lmstudio")
        finally:
            await service.aclose()
