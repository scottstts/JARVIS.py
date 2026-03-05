"""Configuration for core loop context/session policies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import CoreConfigurationError


def _required_int_env(name: str) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        raise CoreConfigurationError(f"{name} must be explicitly set in the environment.")
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
            context_window_tokens=_required_int_env("JARVIS_CONTEXT_WINDOW_TOKENS"),
            compact_threshold_tokens=_required_int_env("JARVIS_COMPACT_THRESHOLD_TOKENS"),
            compact_reserve_output_tokens=_required_int_env(
                "JARVIS_COMPACT_RESERVE_OUTPUT_TOKENS"
            ),
            compact_reserve_overhead_tokens=_required_int_env(
                "JARVIS_COMPACT_RESERVE_OVERHEAD_TOKENS"
            ),
        )


@dataclass(slots=True, frozen=True)
class CoreSettings:
    """Core loop runtime settings."""

    context_policy: ContextPolicySettings
    storage_dir: Path
    identities_dir: Path
    program_file_name: str = "PROGRAM.md"
    reactor_file_name: str = "REACTOR.md"

    @classmethod
    def from_env(cls) -> "CoreSettings":
        storage_root = _optional_env("JARVIS_STORAGE_DIR")
        if storage_root is None:
            agent_workspace = _optional_env("AGENT_WORKSPACE")
            if agent_workspace is not None:
                storage_root = str(Path(agent_workspace).expanduser() / "storage")
            else:
                storage_root = "~/.jarvis/storage"

        identities_dir = _optional_env("JARVIS_IDENTITIES_DIR") or "src/identities"

        return cls(
            context_policy=ContextPolicySettings.from_env(),
            storage_dir=Path(storage_root).expanduser(),
            identities_dir=Path(identities_dir).expanduser(),
        )
