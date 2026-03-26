"""Canonical provider-name constants shared across the LLM subsystem."""

from __future__ import annotations

LLM_PROVIDER_NAMES: tuple[str, ...] = (
    "openai",
    "anthropic",
    "gemini",
    "openrouter",
    "lmstudio",
)
LLM_PROVIDER_NAME_SET: frozenset[str] = frozenset(LLM_PROVIDER_NAMES)
LLM_PROVIDER_NAMES_TEXT = ", ".join(LLM_PROVIDER_NAMES)

EMBEDDING_PROVIDER_NAMES: tuple[str, ...] = (
    "openai",
    "gemini",
    "openrouter",
)
EMBEDDING_PROVIDER_NAME_SET: frozenset[str] = frozenset(EMBEDDING_PROVIDER_NAMES)
EMBEDDING_PROVIDER_NAMES_TEXT = ", ".join(EMBEDDING_PROVIDER_NAMES)
