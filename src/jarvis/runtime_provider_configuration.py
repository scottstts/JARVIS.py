"""Resolve the effective provider/model targets used by the Jarvis runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.core.config import CoreSettings
    from jarvis.llm.config import LLMSettings
    from jarvis.memory.config import MemorySettings
    from jarvis.subagent.settings import SubagentSettings


@dataclass(frozen=True, slots=True)
class RuntimeProviderTarget:
    role: str
    provider: str
    model: str


type RuntimeProviderConfiguration = tuple[RuntimeProviderTarget, ...]


def load_runtime_provider_configuration(
    *,
    core_settings: CoreSettings,
) -> RuntimeProviderConfiguration:
    from jarvis.llm.config import LLMSettings
    from jarvis.memory.config import MemorySettings
    from jarvis.subagent.settings import SubagentSettings

    llm_settings = LLMSettings.from_env()
    memory_settings = MemorySettings.from_workspace_dir(core_settings.workspace_dir)
    subagent_settings = SubagentSettings.from_workspace_dir(
        core_settings.workspace_dir,
        transcript_archive_root=core_settings.transcript_archive_dir,
    )
    return resolve_runtime_provider_configuration(
        core_settings=core_settings,
        llm_settings=llm_settings,
        memory_settings=memory_settings,
        subagent_settings=subagent_settings,
    )


def resolve_runtime_provider_configuration(
    *,
    core_settings: CoreSettings,
    llm_settings: LLMSettings,
    memory_settings: MemorySettings,
    subagent_settings: SubagentSettings,
) -> RuntimeProviderConfiguration:
    main_provider = llm_settings.default_provider
    subagent_provider = subagent_settings.provider or main_provider
    compaction_provider = core_settings.compaction.provider
    return (
        _provider_target(
            role="Main Agent",
            provider=main_provider,
            model=_chat_model_for_provider(
                llm_settings=llm_settings,
                provider=main_provider,
            ),
        ),
        _provider_target(
            role="Subagent",
            provider=subagent_provider,
            model=_chat_model_for_provider(
                llm_settings=llm_settings,
                provider=subagent_provider,
            ),
        ),
        _provider_target(
            role="Compaction",
            provider=compaction_provider,
            model=_chat_model_for_provider(
                llm_settings=llm_settings,
                provider=compaction_provider,
            ),
        ),
        _provider_target(
            role="Memory Maintenance",
            provider=memory_settings.maintenance_provider,
            model=memory_settings.maintenance_model,
        ),
        _provider_target(
            role="Embedding",
            provider=llm_settings.embedding.provider,
            model=llm_settings.embedding.model,
        ),
    )


def _chat_model_for_provider(*, llm_settings: LLMSettings, provider: str) -> str:
    if provider == "codex":
        from jarvis.codex_backend.config import CodexBackendSettings

        codex_settings = CodexBackendSettings.from_env()
        return codex_settings.model or "(server default)"
    if provider == "openai":
        return llm_settings.openai.chat_model or "(unconfigured)"
    if provider == "anthropic":
        return llm_settings.anthropic.chat_model or "(unconfigured)"
    if provider == "gemini":
        return llm_settings.gemini.chat_model or "(unconfigured)"
    if provider == "grok":
        return llm_settings.grok.chat_model or "(unconfigured)"
    if provider == "openrouter":
        return llm_settings.openrouter.chat_model or "(unconfigured)"
    if provider == "lmstudio":
        return "(provider-selected)"
    return "(unknown)"


def _provider_target(*, role: str, provider: str, model: str) -> RuntimeProviderTarget:
    return RuntimeProviderTarget(role=role, provider=provider, model=model)
