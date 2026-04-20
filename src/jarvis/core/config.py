"""Configuration for core loop context/session policies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jarvis import settings as app_settings
from jarvis.llm.provider_names import LLM_PROVIDER_NAME_SET, LLM_PROVIDER_NAMES_TEXT
from jarvis.storage.layout import resolve_transcript_archive_root
from jarvis.workspace_paths import resolve_workspace_child, resolve_workspace_dir

from .errors import CoreConfigurationError


_COMPACT_THRESHOLD_PERCENT = 90
_COMPACT_RESERVE_OUTPUT_PERCENT = 6
_COMPACT_RESERVE_OUTPUT_MIN_TOKENS = 10_000
_COMPACT_RESERVE_OVERHEAD_PERCENT = 3
_COMPACT_RESERVE_OVERHEAD_MIN_TOKENS = 5_000


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise CoreConfigurationError(f"{name} must be an integer, got: {raw}") from exc
    return value


def _optional_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _required_lower_choice_env(name: str, default: str, *, disallowed: set[str] | None = None) -> str:
    raw = os.getenv(name)
    value = raw if raw is not None else default
    normalized = str(value).strip().lower()
    if not normalized:
        raise CoreConfigurationError(f"{name} must not be blank.")
    if normalized not in LLM_PROVIDER_NAME_SET:
        raise CoreConfigurationError(
            f"{name} must be one of: {LLM_PROVIDER_NAMES_TEXT}. Got: {normalized}"
        )
    blocked = disallowed or set()
    if normalized in blocked:
        blocked_text = ", ".join(sorted(blocked))
        raise CoreConfigurationError(
            f"{name} must not be one of: {blocked_text}. Got: {normalized}"
        )
    return normalized


def _ceil_percentage(value: int, percent: int) -> int:
    return (value * percent + 99) // 100


@dataclass(slots=True, frozen=True)
class ContextPolicySettings:
    """Global, provider-agnostic context compaction policy."""

    context_window_tokens: int

    @property
    def compact_threshold_tokens(self) -> int:
        threshold_from_window = (self.context_window_tokens * _COMPACT_THRESHOLD_PERCENT) // 100
        threshold_from_preflight = (
            self.preflight_limit_tokens - self.compact_reserve_overhead_tokens
        )
        return min(threshold_from_window, threshold_from_preflight)

    @property
    def compact_reserve_output_tokens(self) -> int:
        return max(
            _ceil_percentage(self.context_window_tokens, _COMPACT_RESERVE_OUTPUT_PERCENT),
            _COMPACT_RESERVE_OUTPUT_MIN_TOKENS,
        )

    @property
    def compact_reserve_overhead_tokens(self) -> int:
        return max(
            _ceil_percentage(self.context_window_tokens, _COMPACT_RESERVE_OVERHEAD_PERCENT),
            _COMPACT_RESERVE_OVERHEAD_MIN_TOKENS,
        )

    @property
    def reserve_tokens(self) -> int:
        return self.compact_reserve_output_tokens + self.compact_reserve_overhead_tokens

    @property
    def preflight_limit_tokens(self) -> int:
        return self.context_window_tokens - self.reserve_tokens

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0:
            raise CoreConfigurationError("JARVIS_CONTEXT_WINDOW_TOKENS must be > 0.")
        if self.reserve_tokens >= self.context_window_tokens:
            raise CoreConfigurationError(
                "JARVIS_CONTEXT_WINDOW_TOKENS must be greater than the derived reserve budget."
            )
        if self.compact_threshold_tokens <= 0:
            raise CoreConfigurationError(
                "JARVIS_CONTEXT_WINDOW_TOKENS must be large enough for the derived compaction threshold."
            )
        if self.compact_threshold_tokens >= self.preflight_limit_tokens:
            raise CoreConfigurationError(
                "Derived compact threshold must be less than the preflight context budget."
            )

    @classmethod
    def from_env(cls) -> "ContextPolicySettings":
        return cls(
            context_window_tokens=_parse_int_env(
                "JARVIS_CONTEXT_WINDOW_TOKENS",
                app_settings.JARVIS_CONTEXT_WINDOW_TOKENS,
            ),
        )


@dataclass(slots=True, frozen=True)
class CompactionSettings:
    """Provider selection for dedicated session compaction requests."""

    provider: str

    def __post_init__(self) -> None:
        normalized = self.provider.strip().lower()
        if not normalized:
            raise CoreConfigurationError("JARVIS_COMPACTION_PROVIDER must not be blank.")
        if normalized not in LLM_PROVIDER_NAME_SET:
            raise CoreConfigurationError(
                f"JARVIS_COMPACTION_PROVIDER must be one of: {LLM_PROVIDER_NAMES_TEXT}."
            )
        if normalized == "codex":
            raise CoreConfigurationError(
                "JARVIS_COMPACTION_PROVIDER cannot be 'codex'; compaction uses normal LLM providers only."
            )
        object.__setattr__(self, "provider", normalized)

    @classmethod
    def from_env(cls) -> "CompactionSettings":
        return cls(
            provider=_required_lower_choice_env(
                "JARVIS_COMPACTION_PROVIDER",
                app_settings.JARVIS_COMPACTION_PROVIDER,
                disallowed={"codex"},
            )
        )


@dataclass(slots=True, frozen=True)
class CoreSettings:
    """Core loop runtime settings."""

    context_policy: ContextPolicySettings
    compaction: CompactionSettings
    workspace_dir: Path
    transcript_archive_dir: Path
    identities_dir: Path
    turn_timezone: str = app_settings.JARVIS_CORE_TIMEZONE
    program_file_name: str = "PROGRAM.md"
    reactor_file_name: str = "REACTOR.md"
    user_file_name: str = "USER.md"
    armor_file_name: str = "ARMOR.md"

    def __post_init__(self) -> None:
        timezone_name = self.turn_timezone.strip()
        if not timezone_name:
            raise CoreConfigurationError("JARVIS_CORE_TIMEZONE must not be blank.")
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise CoreConfigurationError(
                f"JARVIS_CORE_TIMEZONE must be a valid IANA timezone, got: {timezone_name}"
            ) from exc
        object.__setattr__(self, "turn_timezone", timezone_name)

    @classmethod
    def from_env(cls) -> "CoreSettings":
        workspace_dir = resolve_workspace_dir(error_type=CoreConfigurationError)
        transcript_archive_dir = resolve_transcript_archive_root(workspace_dir)
        identities_dir = resolve_workspace_child(
            env_name="JARVIS_IDENTITIES_DIR",
            configured_default=None,
            workspace_dir=workspace_dir,
            child_name="identities",
        )

        return cls(
            context_policy=ContextPolicySettings.from_env(),
            compaction=CompactionSettings.from_env(),
            workspace_dir=workspace_dir,
            transcript_archive_dir=transcript_archive_dir,
            identities_dir=identities_dir,
            turn_timezone=_optional_env("JARVIS_CORE_TIMEZONE")
            or app_settings.JARVIS_CORE_TIMEZONE,
        )
