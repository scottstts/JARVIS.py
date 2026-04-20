"""Non-secret project runtime settings loaded from metadata-aware YAML.

Secrets are loaded separately from Docker secret files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import yaml


class SettingsConfigurationError(RuntimeError):
    """Raised when the Jarvis settings file is missing or invalid."""


_PACKAGED_SETTINGS_FILE_NAME = "settings.yml"
_WORKSPACE_SETTINGS_RELATIVE_PATH = Path("settings") / _PACKAGED_SETTINGS_FILE_NAME
_SETTINGS_SOURCE_OVERRIDE_ENV = "JARVIS_SETTINGS_FILE"


def _resolve_workspace_root() -> Path:
    raw = os.getenv("AGENT_WORKSPACE")
    if raw is None:
        return Path("/workspace")

    value = raw.strip()
    if not value:
        return Path("/workspace")
    return Path(value).expanduser()


def _packaged_settings_path() -> Path:
    return Path(__file__).resolve().with_name(_PACKAGED_SETTINGS_FILE_NAME)


def _workspace_settings_path() -> Path:
    return _resolve_workspace_root() / _WORKSPACE_SETTINGS_RELATIVE_PATH


PACKAGED_SETTINGS_PATH: Final = _packaged_settings_path()
WORKSPACE_SETTINGS_PATH: Final = _workspace_settings_path()


def _resolve_settings_source_path() -> Path:
    explicit_path = os.getenv(_SETTINGS_SOURCE_OVERRIDE_ENV)
    if explicit_path is not None:
        value = explicit_path.strip()
        if not value:
            raise SettingsConfigurationError(
                f"{_SETTINGS_SOURCE_OVERRIDE_ENV} cannot be blank when set."
            )
        return Path(value).expanduser().resolve(strict=False)

    workspace_path = WORKSPACE_SETTINGS_PATH.resolve(strict=False)
    if workspace_path.exists():
        if not workspace_path.is_file():
            raise SettingsConfigurationError(
                f"Expected Jarvis settings file at '{workspace_path}', but found a non-file entry."
            )
        return workspace_path

    return PACKAGED_SETTINGS_PATH.resolve(strict=False)


def _path_label(path_segments: list[str]) -> str:
    return ".".join(path_segments) or "<root>"


def _extract_runtime_values(
    source_path: Path, node: object, path_segments: list[str]
) -> object:
    if not isinstance(node, dict):
        return node

    path_label = _path_label(path_segments)
    has_type = "type" in node
    has_value = "value" in node
    if has_type or has_value:
        if not has_type or not has_value:
            raise SettingsConfigurationError(
                f"Jarvis settings file '{source_path}' is invalid: "
                f"field '{path_label}' must define both 'type' and 'value'."
            )

        field_type = node["type"]
        if not isinstance(field_type, str) or not field_type.strip():
            raise SettingsConfigurationError(
                f"Jarvis settings file '{source_path}' is invalid: "
                f"field '{path_label}' must define a non-empty string 'type'."
            )
        return node["value"]

    extracted: dict[str, object] = {}
    if "fields" in node:
        fields = node["fields"]
        if not isinstance(fields, dict):
            raise SettingsConfigurationError(
                f"Jarvis settings file '{source_path}' is invalid: "
                f"'{path_label}' must define 'fields' as a mapping."
            )
        for key, value in fields.items():
            if not isinstance(key, str) or not key:
                raise SettingsConfigurationError(
                    f"Jarvis settings file '{source_path}' is invalid: "
                    f"'{path_label}' contains an invalid field key."
                )
            extracted[key] = _extract_runtime_values(source_path, value, [*path_segments, key])

    if "groups" in node:
        groups = node["groups"]
        if not isinstance(groups, dict):
            raise SettingsConfigurationError(
                f"Jarvis settings file '{source_path}' is invalid: "
                f"'{path_label}' must define 'groups' as a mapping."
            )
        for key, value in groups.items():
            if not isinstance(key, str) or not key:
                raise SettingsConfigurationError(
                    f"Jarvis settings file '{source_path}' is invalid: "
                    f"'{path_label}' contains an invalid group key."
                )
            extracted[key] = _extract_runtime_values(source_path, value, [*path_segments, key])

    if extracted:
        return extracted

    if "fields" in node or "groups" in node:
        raise SettingsConfigurationError(
            f"Jarvis settings file '{source_path}' is invalid: "
            f"'{path_label}' must contain at least one field or group."
        )

    if "title" in node or "description" in node:
        raise SettingsConfigurationError(
            f"Jarvis settings file '{source_path}' is invalid: "
            f"'{path_label}' defines metadata but no fields or groups."
        )

    # Backwards-compatible fallback for plain nested mappings.
    for key, value in node.items():
        if not isinstance(key, str) or not key:
            raise SettingsConfigurationError(
                f"Jarvis settings file '{source_path}' is invalid: "
                f"'{path_label}' contains an invalid key."
            )
        extracted[key] = _extract_runtime_values(source_path, value, [*path_segments, key])
    return extracted


def _load_settings_payload() -> tuple[Path, dict[str, object], dict[str, object]]:
    source_path = _resolve_settings_source_path()

    try:
        raw_text = source_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SettingsConfigurationError(
            f"Jarvis settings file not found at '{source_path}'."
        ) from exc

    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise SettingsConfigurationError(
            f"Invalid Jarvis settings file '{source_path}': {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise SettingsConfigurationError(
            f"Jarvis settings file '{source_path}' must contain a top-level mapping."
        )

    runtime_values = _extract_runtime_values(source_path, payload, [])
    if not isinstance(runtime_values, dict):
        raise SettingsConfigurationError(
            f"Jarvis settings file '{source_path}' must resolve to a top-level mapping."
        )

    return source_path, payload, runtime_values


SETTINGS_SOURCE_PATH, _SETTINGS_DOCUMENT, _SETTINGS = _load_settings_payload()


def _setting(path: str) -> object:
    current: object = _SETTINGS
    traversed: list[str] = []
    for segment in path.split("."):
        traversed.append(segment)
        if not isinstance(current, dict):
            joined = ".".join(traversed[:-1])
            raise SettingsConfigurationError(
                f"Jarvis settings file '{SETTINGS_SOURCE_PATH}' is invalid: "
                f"'{joined}' must be a mapping to reach '{path}'."
            )
        if segment not in current:
            raise SettingsConfigurationError(
                f"Jarvis settings file '{SETTINGS_SOURCE_PATH}' is missing required setting "
                f"'{path}'."
            )
        current = current[segment]
    return current


def _type_error(path: str, expected: str, actual: object) -> SettingsConfigurationError:
    actual_type = type(actual).__name__
    return SettingsConfigurationError(
        f"Jarvis settings file '{SETTINGS_SOURCE_PATH}' has invalid value for '{path}': "
        f"expected {expected}, got {actual_type}."
    )


def _string(path: str) -> str:
    value = _setting(path)
    if not isinstance(value, str):
        raise _type_error(path, "a string", value)
    return value


def _optional_string(path: str) -> str | None:
    value = _setting(path)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _type_error(path, "a string or null", value)
    return value


def _integer(path: str) -> int:
    value = _setting(path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _type_error(path, "an integer", value)
    return value


def _optional_integer(path: str) -> int | None:
    value = _setting(path)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _type_error(path, "an integer or null", value)
    return value


def _float(path: str) -> float:
    value = _setting(path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _type_error(path, "a number", value)
    return float(value)


def _boolean(path: str) -> bool:
    value = _setting(path)
    if type(value) is not bool:
        raise _type_error(path, "a boolean", value)
    return value


# Tool defaults that remain user-facing.
JARVIS_TOOL_EMAIL_SMTP_HOST: Final = _string("tools.email.smtp_host")
JARVIS_TOOL_EMAIL_SMTP_PORT: Final = _integer("tools.email.smtp_port")
JARVIS_TOOL_EMAIL_SMTP_SECURITY: Final = _string("tools.email.smtp_security")

# Core session settings.
JARVIS_CONTEXT_WINDOW_TOKENS: Final = _integer("core.context_window_tokens")
JARVIS_CORE_TIMEZONE: Final = _string("core.timezone")
JARVIS_COMPACTION_PROVIDER: Final = _string("core.compaction.provider")

# Main-provider and embedding routing.
JARVIS_LLM_DEFAULT_PROVIDER: Final = _string("main_agent_provider")
JARVIS_EMBEDDING_PROVIDER: Final = _string("embedding.provider")
JARVIS_EMBEDDING_MODEL: Final = _string("embedding.model")

# Memory behavior settings.
JARVIS_MEMORY_MAINTENANCE_LLM_PROVIDER: Final = _string("memory.maintenance.provider")
JARVIS_MEMORY_MAINTENANCE_LLM_MODEL: Final = _string("memory.maintenance.model")
JARVIS_MEMORY_MAINTENANCE_LLM_MAX_OUTPUT_TOKENS: Final = _integer(
    "memory.maintenance.max_output_tokens"
)
JARVIS_MEMORY_SEARCH_DEFAULT_TOP_K: Final = _integer(
    "memory.retrieval.search_default_top_k"
)
JARVIS_MEMORY_DAILY_LOOKBACK_DAYS: Final = _integer("memory.retrieval.daily_lookback_days")
JARVIS_MEMORY_ENABLE_REFLECTION: Final = _boolean("memory.automation.enable_reflection")
JARVIS_MEMORY_ENABLE_AUTO_APPLY_CORE: Final = _boolean(
    "memory.automation.enable_auto_apply_core"
)
JARVIS_MEMORY_ENABLE_AUTO_APPLY_ONGOING: Final = _boolean(
    "memory.automation.enable_auto_apply_ongoing"
)

# Subagent behavior settings.
JARVIS_SUBAGENT_PROVIDER: Final[str | None] = _optional_string("subagent_provider")

# OpenAI provider defaults.
JARVIS_OPENAI_CHAT_MODEL: Final = _string("providers.openai.chat_model")
JARVIS_OPENAI_TEMPERATURE: Final = _float("providers.openai.temperature")
JARVIS_OPENAI_MAX_OUTPUT_TOKENS: Final = _integer("providers.openai.max_output_tokens")
JARVIS_OPENAI_REASONING_EFFORT: Final = _string("providers.openai.reasoning_effort")
JARVIS_OPENAI_REASONING_SUMMARY: Final[str | None] = _optional_string(
    "providers.openai.reasoning_summary"
)
JARVIS_OPENAI_TEXT_VERBOSITY: Final[str | None] = _optional_string(
    "providers.openai.text_verbosity"
)

# Anthropic provider defaults.
JARVIS_ANTHROPIC_CHAT_MODEL: Final = _string("providers.anthropic.chat_model")
JARVIS_ANTHROPIC_TEMPERATURE: Final = _float("providers.anthropic.temperature")
JARVIS_ANTHROPIC_MAX_OUTPUT_TOKENS: Final = _integer(
    "providers.anthropic.max_output_tokens"
)
JARVIS_ANTHROPIC_THINKING_MODE: Final = _string("providers.anthropic.thinking_mode")
JARVIS_ANTHROPIC_EFFORT: Final = _string("providers.anthropic.effort")
JARVIS_ANTHROPIC_THINKING_BUDGET_TOKENS: Final[int | None] = _optional_integer(
    "providers.anthropic.thinking_budget_tokens"
)
JARVIS_ANTHROPIC_PROMPT_CACHE_TTL: Final[str | None] = _optional_string(
    "providers.anthropic.prompt_cache_ttl"
)

# Gemini provider defaults.
JARVIS_GEMINI_CHAT_MODEL: Final = _string("providers.gemini.chat_model")
JARVIS_GEMINI_TEMPERATURE: Final = _float("providers.gemini.temperature")
JARVIS_GEMINI_MAX_OUTPUT_TOKENS: Final = _integer("providers.gemini.max_output_tokens")
JARVIS_GEMINI_THINKING_LEVEL: Final = _string("providers.gemini.thinking_level")
JARVIS_GEMINI_THINKING_BUDGET: Final[int | None] = _optional_integer(
    "providers.gemini.thinking_budget"
)

# Codex provider defaults.
JARVIS_CODEX_MODEL: Final[str | None] = _optional_string("providers.codex.model")
JARVIS_CODEX_REASONING_EFFORT: Final[str | None] = _optional_string(
    "providers.codex.reasoning_effort"
)

# Grok provider defaults.
JARVIS_GROK_CHAT_MODEL: Final = _string("providers.grok.chat_model")
JARVIS_GROK_TEMPERATURE: Final = _float("providers.grok.temperature")
JARVIS_GROK_MAX_OUTPUT_TOKENS: Final = _integer("providers.grok.max_output_tokens")

# OpenRouter provider defaults.
JARVIS_OPENROUTER_CHAT_MODEL: Final = _string("providers.openrouter.chat_model")
JARVIS_OPENROUTER_TEMPERATURE: Final = _float("providers.openrouter.temperature")
JARVIS_OPENROUTER_MAX_OUTPUT_TOKENS: Final = _integer(
    "providers.openrouter.max_output_tokens"
)
