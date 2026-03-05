"""Provider interface contracts for the LLM service layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable

from .types import EmbeddingRequest, EmbeddingResponse, LLMRequest, LLMResponse, LLMStreamEvent


@dataclass(slots=True, frozen=True)
class ProviderCapabilities:
    streaming: bool = True
    tools: bool = True
    embeddings: bool = False
    image_input: bool = False


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str:
        """Provider key used for routing (e.g. 'openai')."""

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Feature capabilities surfaced to service layer."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """One-shot generation API."""

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Streaming generation API."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embeddings API."""

    async def aclose(self) -> None:
        """Release any underlying provider client resources."""
