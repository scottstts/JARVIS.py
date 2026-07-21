"""Built-in provider context strategy and persisted provider session state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Mapping


class ProviderContextStrategy(StrEnum):
    PROVIDER_STATEFUL_CONTINUATION = "provider_stateful_continuation"
    PROVIDER_CACHED_CONTEXT = "provider_cached_context"
    LOCAL_REPLAY_WITH_PROMPT_CACHE = "local_replay_with_prompt_cache"


BUILTIN_PROVIDER_CONTEXT_STRATEGIES: dict[str, ProviderContextStrategy] = {
    "openai": ProviderContextStrategy.PROVIDER_STATEFUL_CONTINUATION,
    "grok": ProviderContextStrategy.PROVIDER_STATEFUL_CONTINUATION,
    "gemini": ProviderContextStrategy.PROVIDER_CACHED_CONTEXT,
    "anthropic": ProviderContextStrategy.LOCAL_REPLAY_WITH_PROMPT_CACHE,
    "openrouter": ProviderContextStrategy.LOCAL_REPLAY_WITH_PROMPT_CACHE,
}


@dataclass(slots=True, frozen=True)
class OpenAIProviderSessionState:
    conversation_id: str | None = None
    previous_response_id: str | None = None
    last_response_record_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversationId": self.conversation_id,
            "previousResponseId": self.previous_response_id,
            "lastResponseRecordId": self.last_response_record_id,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "OpenAIProviderSessionState":
        if not isinstance(value, Mapping):
            return cls()
        return cls(
            conversation_id=_optional_str(value.get("conversationId")),
            previous_response_id=_optional_str(value.get("previousResponseId")),
            last_response_record_id=_optional_str(value.get("lastResponseRecordId")),
        )


@dataclass(slots=True, frozen=True)
class GrokProviderSessionState:
    previous_response_id: str | None = None
    last_response_record_id: str | None = None
    durable_response_id: str | None = None
    durable_response_record_id: str | None = None
    storage_mode: Literal["durable", "ephemeral"] = "durable"
    websocket_generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "previousResponseId": self.previous_response_id,
            "lastResponseRecordId": self.last_response_record_id,
            "durableResponseId": self.durable_response_id,
            "durableResponseRecordId": self.durable_response_record_id,
            "storageMode": self.storage_mode,
            "websocketGeneration": self.websocket_generation,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "GrokProviderSessionState":
        if not isinstance(value, Mapping):
            return cls()
        previous_response_id = _optional_str(value.get("previousResponseId"))
        last_response_record_id = _optional_str(value.get("lastResponseRecordId"))
        raw_storage_mode = _optional_str(value.get("storageMode"))
        storage_mode: Literal["durable", "ephemeral"] = (
            "ephemeral" if raw_storage_mode == "ephemeral" else "durable"
        )
        raw_generation = value.get("websocketGeneration", 0)
        websocket_generation = raw_generation if isinstance(raw_generation, int) else 0
        is_legacy_state = not any(
            key in value
            for key in (
                "durableResponseId",
                "durableResponseRecordId",
                "storageMode",
                "websocketGeneration",
            )
        )
        return cls(
            previous_response_id=previous_response_id,
            last_response_record_id=last_response_record_id,
            durable_response_id=(
                previous_response_id
                if is_legacy_state
                else _optional_str(value.get("durableResponseId"))
            ),
            durable_response_record_id=(
                last_response_record_id
                if is_legacy_state
                else _optional_str(value.get("durableResponseRecordId"))
            ),
            storage_mode=storage_mode,
            websocket_generation=max(0, websocket_generation),
        )


@dataclass(slots=True, frozen=True)
class GeminiProviderSessionState:
    cached_content_name: str | None = None
    cache_expires_at: str | None = None
    cached_media_ids: tuple[str, ...] = ()
    model: str | None = None
    source_record_ids: tuple[str, ...] = ()
    source_signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cachedContentName": self.cached_content_name,
            "cacheExpiresAt": self.cache_expires_at,
            "cachedMediaIds": list(self.cached_media_ids),
            "model": self.model,
            "sourceRecordIds": list(self.source_record_ids),
            "sourceSignature": self.source_signature,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "GeminiProviderSessionState":
        if not isinstance(value, Mapping):
            return cls()
        return cls(
            cached_content_name=_optional_str(value.get("cachedContentName")),
            cache_expires_at=_optional_str(value.get("cacheExpiresAt")),
            cached_media_ids=_string_tuple(value.get("cachedMediaIds")),
            model=_optional_str(value.get("model")),
            source_record_ids=_string_tuple(value.get("sourceRecordIds")),
            source_signature=_optional_str(value.get("sourceSignature")),
        )


@dataclass(slots=True, frozen=True)
class AnthropicProviderSessionState:
    cache_control_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"cacheControlMode": self.cache_control_mode}

    @classmethod
    def from_mapping(cls, value: Any) -> "AnthropicProviderSessionState":
        if not isinstance(value, Mapping):
            return cls(cache_control_mode="prompt_cache_blocks")
        return cls(
            cache_control_mode=(
                _optional_str(value.get("cacheControlMode")) or "prompt_cache_blocks"
            )
        )


@dataclass(slots=True, frozen=True)
class OpenRouterProviderSessionState:
    session_id: str | None = None
    prompt_cache_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "promptCacheEnabled": self.prompt_cache_enabled,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "OpenRouterProviderSessionState":
        if not isinstance(value, Mapping):
            return cls()
        return cls(
            session_id=_optional_str(value.get("sessionId")),
            prompt_cache_enabled=bool(value.get("promptCacheEnabled", True)),
        )


@dataclass(slots=True, frozen=True)
class ProviderSessionState:
    provider: str
    strategy: ProviderContextStrategy
    openai: OpenAIProviderSessionState = field(default_factory=OpenAIProviderSessionState)
    grok: GrokProviderSessionState = field(default_factory=GrokProviderSessionState)
    gemini: GeminiProviderSessionState = field(default_factory=GeminiProviderSessionState)
    anthropic: AnthropicProviderSessionState = field(
        default_factory=AnthropicProviderSessionState
    )
    openrouter: OpenRouterProviderSessionState = field(
        default_factory=OpenRouterProviderSessionState
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "strategy": self.strategy.value,
            "openai": self.openai.to_dict(),
            "grok": self.grok.to_dict(),
            "gemini": self.gemini.to_dict(),
            "anthropic": self.anthropic.to_dict(),
            "openrouter": self.openrouter.to_dict(),
        }

    @classmethod
    def for_provider(cls, provider: str) -> "ProviderSessionState | None":
        normalized = provider.strip().lower()
        strategy = BUILTIN_PROVIDER_CONTEXT_STRATEGIES.get(normalized)
        if strategy is None:
            return None
        return cls(provider=normalized, strategy=strategy)

    @classmethod
    def from_mapping(cls, value: Any) -> "ProviderSessionState | None":
        if not isinstance(value, Mapping):
            return None
        provider = _optional_str(value.get("provider"))
        if provider is None:
            return None
        provider = provider.lower()
        strategy = BUILTIN_PROVIDER_CONTEXT_STRATEGIES.get(provider)
        if strategy is None:
            return None
        raw_strategy = _optional_str(value.get("strategy"))
        if raw_strategy != strategy.value:
            return cls.for_provider(provider)
        return cls(
            provider=provider,
            strategy=strategy,
            openai=OpenAIProviderSessionState.from_mapping(value.get("openai")),
            grok=GrokProviderSessionState.from_mapping(value.get("grok")),
            gemini=GeminiProviderSessionState.from_mapping(value.get("gemini")),
            anthropic=AnthropicProviderSessionState.from_mapping(value.get("anthropic")),
            openrouter=OpenRouterProviderSessionState.from_mapping(value.get("openrouter")),
        )


def strategy_for_provider(provider: str | None) -> ProviderContextStrategy | None:
    if provider is None:
        return None
    return BUILTIN_PROVIDER_CONTEXT_STRATEGIES.get(provider.strip().lower())


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[str] = []
    for item in value:
        normalized = _optional_str(item)
        if normalized is not None:
            out.append(normalized)
    return tuple(out)
