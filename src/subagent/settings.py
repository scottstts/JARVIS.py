"""Runtime settings for subagents."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import settings as app_settings
from llm.provider_names import LLM_PROVIDER_NAME_SET, LLM_PROVIDER_NAMES_TEXT
from workspace_paths import resolve_workspace_child


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _parse_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or default


def _parse_optional_provider_env(name: str, default: str | None) -> str | None:
    raw = os.getenv(name)
    value = raw if raw is not None else default
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized not in LLM_PROVIDER_NAME_SET:
        raise ValueError(
            f"{name} must be one of: {LLM_PROVIDER_NAMES_TEXT}. Got: {value}"
        )
    return normalized


@dataclass(slots=True, frozen=True)
class SubagentSettings:
    provider: str | None
    max_active: int
    codename_pool: tuple[str, ...]
    archive_dir: Path
    builtin_tool_blocklist: tuple[str, ...]
    main_context_event_limit: int

    @classmethod
    def from_workspace_dir(cls, workspace_dir: Path) -> "SubagentSettings":
        archive_dir = resolve_workspace_child(
            env_name="JARVIS_SUBAGENT_ARCHIVE_DIR",
            configured_default=app_settings.JARVIS_SUBAGENT_ARCHIVE_DIR,
            workspace_dir=workspace_dir,
            child_name="archive/subagents",
        )
        return cls(
            provider=_parse_optional_provider_env(
                "JARVIS_SUBAGENT_PROVIDER",
                app_settings.JARVIS_SUBAGENT_PROVIDER,
            ),
            max_active=_parse_int_env(
                "JARVIS_SUBAGENT_MAX_ACTIVE",
                app_settings.JARVIS_SUBAGENT_MAX_ACTIVE,
            ),
            codename_pool=_parse_csv_env(
                "JARVIS_SUBAGENT_CODENAME_POOL",
                app_settings.JARVIS_SUBAGENT_CODENAME_POOL,
            ),
            archive_dir=archive_dir,
            builtin_tool_blocklist=_parse_csv_env(
                "JARVIS_SUBAGENT_BUILTIN_TOOL_BLOCKLIST",
                app_settings.JARVIS_SUBAGENT_BUILTIN_TOOL_BLOCKLIST,
            ),
            main_context_event_limit=_parse_int_env(
                "JARVIS_SUBAGENT_MAIN_CONTEXT_EVENT_LIMIT",
                app_settings.JARVIS_SUBAGENT_MAIN_CONTEXT_EVENT_LIMIT,
            ),
        )
