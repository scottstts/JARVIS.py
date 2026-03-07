"""Configuration for core loop context/session policies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import settings as app_settings
from workspace_paths import resolve_workspace_child, resolve_workspace_dir

from .errors import CoreConfigurationError


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


@dataclass(slots=True, frozen=True)
class ContextPolicySettings:
    """Global, provider-agnostic context compaction policy."""

    context_window_tokens: int
    compact_threshold_tokens: int
    compact_reserve_output_tokens: int
    compact_reserve_overhead_tokens: int

    @property
    def reserve_tokens(self) -> int:
        return self.compact_reserve_output_tokens + self.compact_reserve_overhead_tokens

    @property
    def preflight_limit_tokens(self) -> int:
        return self.context_window_tokens - self.reserve_tokens

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0:
            raise CoreConfigurationError("JARVIS_CONTEXT_WINDOW_TOKENS must be > 0.")
        if self.compact_threshold_tokens <= 0:
            raise CoreConfigurationError("JARVIS_COMPACT_THRESHOLD_TOKENS must be > 0.")
        if self.compact_reserve_output_tokens <= 0:
            raise CoreConfigurationError("JARVIS_COMPACT_RESERVE_OUTPUT_TOKENS must be > 0.")
        if self.compact_reserve_overhead_tokens < 0:
            raise CoreConfigurationError("JARVIS_COMPACT_RESERVE_OVERHEAD_TOKENS must be >= 0.")
        if self.compact_threshold_tokens >= self.context_window_tokens:
            raise CoreConfigurationError(
                "JARVIS_COMPACT_THRESHOLD_TOKENS must be less than JARVIS_CONTEXT_WINDOW_TOKENS."
            )
        if self.reserve_tokens >= self.context_window_tokens:
            raise CoreConfigurationError(
                "Combined reserve tokens must be less than JARVIS_CONTEXT_WINDOW_TOKENS."
            )

    @classmethod
    def from_env(cls) -> "ContextPolicySettings":
        return cls(
            context_window_tokens=_parse_int_env(
                "JARVIS_CONTEXT_WINDOW_TOKENS",
                app_settings.JARVIS_CONTEXT_WINDOW_TOKENS,
            ),
            compact_threshold_tokens=_parse_int_env(
                "JARVIS_COMPACT_THRESHOLD_TOKENS",
                app_settings.JARVIS_COMPACT_THRESHOLD_TOKENS,
            ),
            compact_reserve_output_tokens=_parse_int_env(
                "JARVIS_COMPACT_RESERVE_OUTPUT_TOKENS",
                app_settings.JARVIS_COMPACT_RESERVE_OUTPUT_TOKENS,
            ),
            compact_reserve_overhead_tokens=_parse_int_env(
                "JARVIS_COMPACT_RESERVE_OVERHEAD_TOKENS",
                app_settings.JARVIS_COMPACT_RESERVE_OVERHEAD_TOKENS,
            ),
        )


@dataclass(slots=True, frozen=True)
class CoreSettings:
    """Core loop runtime settings."""

    context_policy: ContextPolicySettings
    workspace_dir: Path
    storage_dir: Path
    identities_dir: Path
    program_file_name: str = "PROGRAM.md"
    reactor_file_name: str = "REACTOR.md"

    @classmethod
    def from_env(cls) -> "CoreSettings":
        workspace_dir = resolve_workspace_dir(error_type=CoreConfigurationError)
        storage_dir = resolve_workspace_child(
            env_name="JARVIS_STORAGE_DIR",
            configured_default=app_settings.JARVIS_STORAGE_DIR,
            workspace_dir=workspace_dir,
            child_name="storage",
        )
        identities_dir = resolve_workspace_child(
            env_name="JARVIS_IDENTITIES_DIR",
            configured_default=app_settings.JARVIS_IDENTITIES_DIR,
            workspace_dir=workspace_dir,
            child_name="identities",
        )

        return cls(
            context_policy=ContextPolicySettings.from_env(),
            workspace_dir=workspace_dir,
            storage_dir=storage_dir,
            identities_dir=identities_dir,
        )
