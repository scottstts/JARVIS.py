"""Non-secret project runtime settings.

Secrets and machine-specific paths stay in `.env`.
"""

from __future__ import annotations

from typing import Final

# Gateway websocket runtime.
JARVIS_GATEWAY_HOST: Final = "0.0.0.0"
JARVIS_GATEWAY_PORT: Final = 8080
JARVIS_GATEWAY_WS_PATH: Final = "/ws"
JARVIS_GATEWAY_MAX_MESSAGE_CHARS: Final = 32_000

# Core workspace paths.
AGENT_WORKSPACE: Final[str | None] = "/workspace"
JARVIS_STORAGE_DIR: Final[str | None] = (
    f"{AGENT_WORKSPACE}/storage" if AGENT_WORKSPACE is not None else None
)
JARVIS_IDENTITIES_DIR: Final = "/workspace/identities"

# Telegram UI runtime.
TELEGRAM_API_BASE_URL: Final = "https://api.telegram.org"
JARVIS_UI_TELEGRAM_ALLOWED_USER_ID: Final[int | None] = None
JARVIS_UI_TELEGRAM_TEMP_DIR: Final[str | None] = (
    f"{AGENT_WORKSPACE}/temp" if AGENT_WORKSPACE is not None else None
)
JARVIS_UI_TELEGRAM_POLL_TIMEOUT_SECONDS: Final = 30
JARVIS_UI_TELEGRAM_POLL_LIMIT: Final = 100
JARVIS_UI_POLL_ERROR_BACKOFF_SECONDS: Final = 2.0
JARVIS_UI_STREAM_DRAFT_MIN_INTERVAL_SECONDS: Final = 0.4
JARVIS_UI_STREAM_DRAFT_MIN_CHARS: Final = 20
JARVIS_UI_TELEGRAM_MAX_MESSAGE_CHARS: Final = 4_096

# Leave unset to derive from the gateway host/port/path settings above.
JARVIS_UI_GATEWAY_WS_BASE_URL: Final[str | None] = None
JARVIS_UI_GATEWAY_CONNECT_TIMEOUT_SECONDS: Final = 15.0

# Core context/session policy.
JARVIS_CONTEXT_WINDOW_TOKENS: Final = 400_000
JARVIS_COMPACT_THRESHOLD_TOKENS: Final = 350_000
JARVIS_COMPACT_RESERVE_OUTPUT_TOKENS: Final = 16_000
JARVIS_COMPACT_RESERVE_OVERHEAD_TOKENS: Final = 10_000

# Core LLM routing/runtime.
# The provider chosen here will determine which API key is needed below
JARVIS_LLM_DEFAULT_PROVIDER: Final = "gemini"
JARVIS_LLM_TIMEOUT_SECONDS: Final = 60.0
JARVIS_LLM_RETRY_ATTEMPTS: Final = 2
JARVIS_LLM_RETRY_BACKOFF_SECONDS: Final = 0.5

# Global embedding runtime.
JARVIS_EMBEDDING_PROVIDER: Final = "openai"
JARVIS_EMBEDDING_MODEL: Final = "text-embedding-3-small"

# OpenAI provider defaults.
JARVIS_OPENAI_CHAT_MODEL: Final = "gpt-5.2-2025-12-11"
JARVIS_OPENAI_TEMPERATURE: Final = 1.0
JARVIS_OPENAI_MAX_OUTPUT_TOKENS: Final = 64_000
JARVIS_OPENAI_REASONING_EFFORT: Final = "none"
JARVIS_OPENAI_REASONING_SUMMARY: Final[str | None] = None
JARVIS_OPENAI_TEXT_VERBOSITY: Final[str | None] = None

# Anthropic provider defaults.
JARVIS_ANTHROPIC_CHAT_MODEL: Final = "claude-sonnet-4-6"
JARVIS_ANTHROPIC_TEMPERATURE: Final = 1.0
JARVIS_ANTHROPIC_MAX_OUTPUT_TOKENS: Final = 32_000
JARVIS_ANTHROPIC_THINKING_MODE: Final = "adaptive"
JARVIS_ANTHROPIC_EFFORT: Final = "medium"
JARVIS_ANTHROPIC_THINKING_BUDGET_TOKENS: Final[int | None] = None

# Gemini provider defaults.
JARVIS_GEMINI_CHAT_MODEL: Final = "gemini-3-flash-preview"
JARVIS_GEMINI_TEMPERATURE: Final = 1.0
JARVIS_GEMINI_MAX_OUTPUT_TOKENS: Final = 64_000
JARVIS_GEMINI_THINKING_LEVEL: Final = "medium"
JARVIS_GEMINI_THINKING_BUDGET: Final[int | None] = None

# OpenRouter provider defaults.
JARVIS_OPENROUTER_CHAT_MODEL: Final = "minimax/minimax-m2.5"
OPENROUTER_APP_NAME: Final = "Jarvis"
JARVIS_OPENROUTER_TEMPERATURE: Final = 0.8
JARVIS_OPENROUTER_MAX_OUTPUT_TOKENS: Final = 10_000
