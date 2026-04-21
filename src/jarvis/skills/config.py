"""Configuration for workspace-backed agent skills."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from jarvis import settings as app_settings

DEFAULT_MAX_SKILL_MD_CHARS = 120_000
DEFAULT_MAX_RESOURCE_LISTING_COUNT = 200
DEFAULT_MAX_IMPORT_FILE_COUNT = 500
DEFAULT_MAX_IMPORT_FILE_BYTES = 2_000_000
DEFAULT_MAX_IMPORT_TOTAL_BYTES = 20_000_000
DEFAULT_MAX_SEARCH_QUERY_CHARS = 200
DEFAULT_MAX_SEARCH_QUERY_WORDS = 24
DEFAULT_SKILL_ID_CHARS = 128
IGNORED_IMPORT_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got: {raw}")


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


@dataclass(slots=True, frozen=True)
class SkillsSettings:
    """Runtime settings for skills discovery and import."""

    workspace_dir: Path
    bootstrap_headers: bool
    max_skill_md_chars: int = DEFAULT_MAX_SKILL_MD_CHARS
    max_resource_listing_count: int = DEFAULT_MAX_RESOURCE_LISTING_COUNT
    max_import_file_count: int = DEFAULT_MAX_IMPORT_FILE_COUNT
    max_import_file_bytes: int = DEFAULT_MAX_IMPORT_FILE_BYTES
    max_import_total_bytes: int = DEFAULT_MAX_IMPORT_TOTAL_BYTES
    max_search_query_chars: int = DEFAULT_MAX_SEARCH_QUERY_CHARS
    max_search_query_words: int = DEFAULT_MAX_SEARCH_QUERY_WORDS
    max_skill_id_chars: int = DEFAULT_SKILL_ID_CHARS

    def __post_init__(self) -> None:
        if self.max_skill_md_chars <= 0:
            raise ValueError("max_skill_md_chars must be > 0.")
        if self.max_resource_listing_count <= 0:
            raise ValueError("max_resource_listing_count must be > 0.")
        if self.max_import_file_count <= 0:
            raise ValueError("max_import_file_count must be > 0.")
        if self.max_import_file_bytes <= 0:
            raise ValueError("max_import_file_bytes must be > 0.")
        if self.max_import_total_bytes <= 0:
            raise ValueError("max_import_total_bytes must be > 0.")
        if self.max_search_query_chars <= 0:
            raise ValueError("max_search_query_chars must be > 0.")
        if self.max_search_query_words <= 0:
            raise ValueError("max_search_query_words must be > 0.")
        if self.max_skill_id_chars <= 0:
            raise ValueError("max_skill_id_chars must be > 0.")

    @property
    def skills_dir(self) -> Path:
        return self.workspace_dir / "skills"

    @property
    def staging_roots(self) -> tuple[Path, ...]:
        return (
            self.workspace_dir / ".codex" / "skills",
            self.workspace_dir / ".claude" / "skills",
            self.workspace_dir / ".agents" / "skills",
            self.workspace_dir / ".mdskills" / "skills",
        )

    @classmethod
    def from_workspace_dir(cls, workspace_dir: Path) -> "SkillsSettings":
        return cls(
            workspace_dir=workspace_dir.expanduser(),
            bootstrap_headers=_parse_bool_env(
                "JARVIS_SKILLS_BOOTSTRAP_HEADERS",
                app_settings.JARVIS_SKILLS_BOOTSTRAP_HEADERS,
            ),
            max_skill_md_chars=_parse_int_env(
                "JARVIS_SKILLS_MAX_SKILL_MD_CHARS",
                DEFAULT_MAX_SKILL_MD_CHARS,
            ),
            max_resource_listing_count=_parse_int_env(
                "JARVIS_SKILLS_MAX_RESOURCE_LISTING_COUNT",
                DEFAULT_MAX_RESOURCE_LISTING_COUNT,
            ),
            max_import_file_count=_parse_int_env(
                "JARVIS_SKILLS_MAX_IMPORT_FILE_COUNT",
                DEFAULT_MAX_IMPORT_FILE_COUNT,
            ),
            max_import_file_bytes=_parse_int_env(
                "JARVIS_SKILLS_MAX_IMPORT_FILE_BYTES",
                DEFAULT_MAX_IMPORT_FILE_BYTES,
            ),
            max_import_total_bytes=_parse_int_env(
                "JARVIS_SKILLS_MAX_IMPORT_TOTAL_BYTES",
                DEFAULT_MAX_IMPORT_TOTAL_BYTES,
            ),
        )
