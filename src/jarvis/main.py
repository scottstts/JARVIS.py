"""Main system entrypoint for running gateway and Telegram UI together."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import replace

import uvicorn

from jarvis.codex_backend import CodexBackendSettings
from jarvis.core import CoreSettings
from jarvis.gateway import GatewaySettings, create_app
from jarvis.llm import LLMSettings
from jarvis.logging_setup import configure_application_logging, get_application_logger
from jarvis.memory import MemorySettings
from jarvis.runtime_env import load_docker_secrets_if_present
from jarvis.subagent.settings import SubagentSettings
from jarvis.ui.telegram import UISettings, run_telegram_ui

LOGGER = get_application_logger(__name__)


async def run_system(
    *,
    gateway_settings: GatewaySettings | None = None,
    ui_settings: UISettings | None = None,
) -> None:
    resolved_gateway_settings = gateway_settings or GatewaySettings.from_env()
    resolved_core_settings = CoreSettings.from_env()
    resolved_ui_settings = _bind_ui_to_gateway(
        ui_settings or UISettings.from_env(),
        resolved_gateway_settings,
    )
    _log_runtime_provider_configuration(core_settings=resolved_core_settings)

    app = create_app(
        gateway_settings=resolved_gateway_settings,
        core_settings=resolved_core_settings,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app=app,
            host=resolved_gateway_settings.host,
            port=resolved_gateway_settings.port,
            lifespan="on",
            access_log=False,
        )
    )

    server_task = asyncio.create_task(server.serve(), name="jarvis-gateway")
    ui_task: asyncio.Task[None] | None = None

    try:
        await _wait_for_gateway_start(
            server,
            server_task,
            startup_timeout_seconds=resolved_ui_settings.gateway_connect_timeout_seconds,
        )
        LOGGER.info(
            "Gateway ready on %s; starting Telegram UI.",
            resolved_ui_settings.gateway_ws_base_url,
        )
        ui_task = asyncio.create_task(
            run_telegram_ui(resolved_ui_settings),
            name="jarvis-ui",
        )

        done, _pending = await asyncio.wait(
            {server_task, ui_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if server_task in done:
            server_exception = server_task.exception()
            if server_exception is not None:
                raise server_exception
            LOGGER.info("Gateway server exited; shutting down Telegram UI.")

        if ui_task in done:
            ui_exception = ui_task.exception()
            if ui_exception is not None:
                raise ui_exception
            LOGGER.info("Telegram UI exited; shutting down gateway server.")
    finally:
        if ui_task is not None and not ui_task.done():
            ui_task.cancel()
            with suppress(asyncio.CancelledError):
                await ui_task

        if not server_task.done():
            server.should_exit = True
            with suppress(asyncio.CancelledError):
                await server_task


def main() -> None:
    load_docker_secrets_if_present()
    configure_application_logging()
    try:
        asyncio.run(run_system())
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested via Ctrl+C; exiting cleanly.")


async def _wait_for_gateway_start(
    server: uvicorn.Server,
    server_task: asyncio.Task[None],
    *,
    startup_timeout_seconds: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + startup_timeout_seconds
    while not server.started:
        if server_task.done():
            server_exception = server_task.exception()
            if server_exception is not None:
                raise server_exception
            raise RuntimeError("Gateway server exited before startup completed.")
        if asyncio.get_running_loop().time() >= deadline:
            server.should_exit = True
            raise TimeoutError("Gateway server did not start before startup timeout expired.")
        await asyncio.sleep(0.01)


def _bind_ui_to_gateway(
    ui_settings: UISettings,
    gateway_settings: GatewaySettings,
) -> UISettings:
    gateway_ws_base_url = _gateway_ws_base_url(gateway_settings)
    if ui_settings.gateway_ws_base_url != gateway_ws_base_url:
        LOGGER.info(
            "Using local gateway websocket URL %s for combined system run.",
            gateway_ws_base_url,
        )
    return replace(ui_settings, gateway_ws_base_url=gateway_ws_base_url)


def _gateway_ws_base_url(gateway_settings: GatewaySettings) -> str:
    host = gateway_settings.host.strip()
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    websocket_path = gateway_settings.websocket_path
    return f"ws://{host}:{gateway_settings.port}{websocket_path}"


def _log_runtime_provider_configuration(*, core_settings: CoreSettings) -> None:
    provider_configuration = _load_runtime_provider_configuration(
        core_settings=core_settings,
    )
    LOGGER.info("=====================================")
    LOGGER.info("  Main Agent LLM Provider: %s", provider_configuration["main_llm"])
    LOGGER.info("  Subagent LLM Provider: %s", provider_configuration["subagent_llm"])
    LOGGER.info(
        "  Memory Maintenance LLM Provider: %s",
        provider_configuration["memory_maintenance_llm"],
    )
    LOGGER.info("  Embedding Model Provider: %s", provider_configuration["embedding"])
    LOGGER.info("=====================================")


def _load_runtime_provider_configuration(*, core_settings: CoreSettings) -> dict[str, str]:
    llm_settings = LLMSettings.from_env()
    memory_settings = MemorySettings.from_workspace_dir(core_settings.workspace_dir)
    subagent_settings = SubagentSettings.from_workspace_dir(
        core_settings.workspace_dir,
        transcript_archive_root=core_settings.transcript_archive_dir,
    )
    return _resolve_runtime_provider_configuration(
        llm_settings=llm_settings,
        memory_settings=memory_settings,
        subagent_settings=subagent_settings,
    )


def _resolve_runtime_provider_configuration(
    *,
    llm_settings: LLMSettings,
    memory_settings: MemorySettings,
    subagent_settings: SubagentSettings,
) -> dict[str, str]:
    main_provider = llm_settings.default_provider
    subagent_provider = subagent_settings.provider or main_provider
    return {
        "main_llm": _format_provider_target(
            provider=main_provider,
            model=_chat_model_for_provider(llm_settings=llm_settings, provider=main_provider),
        ),
        "subagent_llm": _format_provider_target(
            provider=subagent_provider,
            model=_chat_model_for_provider(llm_settings=llm_settings, provider=subagent_provider),
        ),
        "memory_maintenance_llm": _format_provider_target(
            provider=memory_settings.maintenance_provider,
            model=memory_settings.maintenance_model,
        ),
        "embedding": _format_provider_target(
            provider=llm_settings.embedding.provider,
            model=llm_settings.embedding.model,
        ),
    }


def _chat_model_for_provider(*, llm_settings: LLMSettings, provider: str) -> str:
    if provider == "codex":
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


def _format_provider_target(*, provider: str, model: str) -> str:
    return f"{provider} | {model}"


if __name__ == "__main__":
    main()
