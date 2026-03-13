"""Runtime configuration for the memory subsystem."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import settings as app_settings
from workspace_paths import resolve_workspace_child, resolve_workspace_dir


def _optional_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value.")


@dataclass(slots=True, frozen=True)
class MemorySettings:
    workspace_dir: Path
    memory_dir: Path
    index_dir: Path
    default_timezone: str
    maintenance_provider: str
    maintenance_model: str
    maintenance_max_output_tokens: int
    bootstrap_max_tokens: int
    core_bootstrap_max_tokens: int
    ongoing_bootstrap_max_tokens: int
    search_default_top_k: int
    daily_lookback_days: int
    enable_reflection: bool
    enable_auto_apply_core: bool
    enable_auto_apply_ongoing: bool
    graph_default_expand: int
    lexical_candidate_count: int = 30
    semantic_candidate_count: int = 30
    graph_candidate_count: int = 20
    hybrid_result_count: int = 8

    def __post_init__(self) -> None:
        if self.bootstrap_max_tokens <= 0:
            raise ValueError("bootstrap_max_tokens must be > 0.")
        if self.core_bootstrap_max_tokens <= 0:
            raise ValueError("core_bootstrap_max_tokens must be > 0.")
        if self.ongoing_bootstrap_max_tokens <= 0:
            raise ValueError("ongoing_bootstrap_max_tokens must be > 0.")
        if self.core_bootstrap_max_tokens + self.ongoing_bootstrap_max_tokens > self.bootstrap_max_tokens:
            raise ValueError(
                "Combined core/ongoing bootstrap budgets cannot exceed bootstrap_max_tokens."
            )
        if self.search_default_top_k <= 0:
            raise ValueError("search_default_top_k must be > 0.")
        if self.daily_lookback_days <= 0:
            raise ValueError("daily_lookback_days must be > 0.")
        if self.graph_default_expand not in {0, 1, 2}:
            raise ValueError("graph_default_expand must be one of: 0, 1, 2.")
        if self.maintenance_max_output_tokens <= 0:
            raise ValueError("maintenance_max_output_tokens must be > 0.")
        if self.lexical_candidate_count <= 0:
            raise ValueError("lexical_candidate_count must be > 0.")
        if self.semantic_candidate_count <= 0:
            raise ValueError("semantic_candidate_count must be > 0.")
        if self.graph_candidate_count <= 0:
            raise ValueError("graph_candidate_count must be > 0.")
        if self.hybrid_result_count <= 0:
            raise ValueError("hybrid_result_count must be > 0.")
        if not self.maintenance_provider.strip():
            raise ValueError("maintenance_provider cannot be empty.")
        if not self.maintenance_model.strip():
            raise ValueError("maintenance_model cannot be empty.")
        if not self.default_timezone.strip():
            raise ValueError("default_timezone cannot be empty.")

    @classmethod
    def from_env(cls) -> "MemorySettings":
        workspace_dir = resolve_workspace_dir()
        return cls.from_workspace_dir(workspace_dir)

    @classmethod
    def from_workspace_dir(cls, workspace_dir: Path) -> "MemorySettings":
        memory_dir = resolve_workspace_child(
            env_name="JARVIS_MEMORY_DIR",
            configured_default=None,
            workspace_dir=workspace_dir,
            child_name="memory",
        )
        index_dir = resolve_workspace_child(
            env_name="JARVIS_MEMORY_INDEX_DIR",
            configured_default=None,
            workspace_dir=workspace_dir,
            child_name="memory/.index",
        )
        return cls(
            workspace_dir=workspace_dir,
            memory_dir=memory_dir,
            index_dir=index_dir,
            default_timezone=(
                _optional_env("JARVIS_CORE_TIMEZONE")
                or app_settings.JARVIS_CORE_TIMEZONE
            ),
            maintenance_provider=(
                _optional_env("JARVIS_MEMORY_MAINTENANCE_LLM_PROVIDER")
                or app_settings.JARVIS_MEMORY_MAINTENANCE_LLM_PROVIDER
            ),
            maintenance_model=(
                _optional_env("JARVIS_MEMORY_MAINTENANCE_LLM_MODEL")
                or app_settings.JARVIS_MEMORY_MAINTENANCE_LLM_MODEL
            ),
            maintenance_max_output_tokens=_parse_int_env(
                "JARVIS_MEMORY_MAINTENANCE_LLM_MAX_OUTPUT_TOKENS",
                app_settings.JARVIS_MEMORY_MAINTENANCE_LLM_MAX_OUTPUT_TOKENS,
            ),
            bootstrap_max_tokens=_parse_int_env(
                "JARVIS_MEMORY_BOOTSTRAP_MAX_TOKENS",
                app_settings.JARVIS_MEMORY_BOOTSTRAP_MAX_TOKENS,
            ),
            core_bootstrap_max_tokens=_parse_int_env(
                "JARVIS_MEMORY_CORE_BOOTSTRAP_MAX_TOKENS",
                app_settings.JARVIS_MEMORY_CORE_BOOTSTRAP_MAX_TOKENS,
            ),
            ongoing_bootstrap_max_tokens=_parse_int_env(
                "JARVIS_MEMORY_ONGOING_BOOTSTRAP_MAX_TOKENS",
                app_settings.JARVIS_MEMORY_ONGOING_BOOTSTRAP_MAX_TOKENS,
            ),
            search_default_top_k=_parse_int_env(
                "JARVIS_MEMORY_SEARCH_DEFAULT_TOP_K",
                app_settings.JARVIS_MEMORY_SEARCH_DEFAULT_TOP_K,
            ),
            daily_lookback_days=_parse_int_env(
                "JARVIS_MEMORY_DAILY_LOOKBACK_DAYS",
                app_settings.JARVIS_MEMORY_DAILY_LOOKBACK_DAYS,
            ),
            enable_reflection=_parse_bool_env(
                "JARVIS_MEMORY_ENABLE_REFLECTION",
                app_settings.JARVIS_MEMORY_ENABLE_REFLECTION,
            ),
            enable_auto_apply_core=_parse_bool_env(
                "JARVIS_MEMORY_ENABLE_AUTO_APPLY_CORE",
                app_settings.JARVIS_MEMORY_ENABLE_AUTO_APPLY_CORE,
            ),
            enable_auto_apply_ongoing=_parse_bool_env(
                "JARVIS_MEMORY_ENABLE_AUTO_APPLY_ONGOING",
                app_settings.JARVIS_MEMORY_ENABLE_AUTO_APPLY_ONGOING,
            ),
            graph_default_expand=_parse_int_env(
                "JARVIS_MEMORY_GRAPH_DEFAULT_EXPAND",
                app_settings.JARVIS_MEMORY_GRAPH_DEFAULT_EXPAND,
            ),
            lexical_candidate_count=_parse_int_env(
                "JARVIS_MEMORY_LEXICAL_CANDIDATE_COUNT",
                app_settings.JARVIS_MEMORY_LEXICAL_CANDIDATE_COUNT,
            ),
            semantic_candidate_count=_parse_int_env(
                "JARVIS_MEMORY_SEMANTIC_CANDIDATE_COUNT",
                app_settings.JARVIS_MEMORY_SEMANTIC_CANDIDATE_COUNT,
            ),
            graph_candidate_count=_parse_int_env(
                "JARVIS_MEMORY_GRAPH_CANDIDATE_COUNT",
                app_settings.JARVIS_MEMORY_GRAPH_CANDIDATE_COUNT,
            ),
            hybrid_result_count=_parse_int_env(
                "JARVIS_MEMORY_HYBRID_RESULT_COUNT",
                app_settings.JARVIS_MEMORY_HYBRID_RESULT_COUNT,
            ),
        )

    @property
    def core_dir(self) -> Path:
        return self.memory_dir / "core"

    @property
    def ongoing_dir(self) -> Path:
        return self.memory_dir / "ongoing"

    @property
    def daily_dir(self) -> Path:
        return self.memory_dir / "daily"

    @property
    def archive_dir(self) -> Path:
        return self.memory_dir / "archive"

    @property
    def archive_core_dir(self) -> Path:
        return self.archive_dir / "core"

    @property
    def archive_ongoing_dir(self) -> Path:
        return self.archive_dir / "ongoing"

    @property
    def archive_daily_dir(self) -> Path:
        return self.archive_dir / "daily"

    @property
    def main_index_path(self) -> Path:
        return self.index_dir / "index.sqlite"

    @property
    def embeddings_index_path(self) -> Path:
        return self.index_dir / "embeddings.sqlite"

    @property
    def state_path(self) -> Path:
        return self.index_dir / "state.json"
