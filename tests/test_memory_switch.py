"""Tests for the memory subsystem master switch."""

from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis.core import AgentLoop
from jarvis.gateway.route_runtime import RouteRuntime
from jarvis.memory import MemorySettings
from jarvis.storage import SessionStorage
from jarvis.tools import MEMORY_TOOL_NAMES
from tests.helpers import build_core_settings


class MemorySwitchTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_switch_skips_memory_service_bootstrap_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            core_settings = build_core_settings(root_dir=root_dir)
            disabled_settings = replace(
                MemorySettings.from_workspace_dir(core_settings.workspace_dir),
                enabled=False,
            )

            with patch.object(
                MemorySettings,
                "from_workspace_dir",
                return_value=disabled_settings,
            ):
                loop = AgentLoop(
                    llm_service=object(),  # type: ignore[arg-type]
                    settings=core_settings,
                )

                self.assertFalse(loop._memory_mode.bootstrap)
                self.assertFalse(loop._memory_mode.maintenance)
                self.assertFalse(loop._memory_mode.reflection)
                self.assertIsNone(loop._memory_service)
                self.assertIsNone(loop._tool_context.memory_service)
                self.assertTrue(
                    MEMORY_TOOL_NAMES.isdisjoint(
                        {
                            definition.name
                            for definition in loop._tool_registry.basic_definitions()
                        }
                    )
                )
                self.assertIsNone(loop._tool_registry.get_discoverable("memory_admin"))
                self.assertFalse((core_settings.workspace_dir / "memory").exists())

                session_id = await loop.prepare_session()
                records = SessionStorage(core_settings.transcript_archive_dir).load_records(
                    session_id
                )

            self.assertFalse(
                any(record.metadata.get("memory_bootstrap") for record in records)
            )
            tool_bootstrap = next(
                (
                    record
                    for record in records
                    if record.metadata.get("tool_definitions") is not None
                ),
                None,
            )
            self.assertIsNotNone(tool_bootstrap)
            if tool_bootstrap is not None:
                tool_names = {
                    str(item["name"])
                    for item in tool_bootstrap.metadata["tool_definitions"]
                }
                self.assertTrue(MEMORY_TOOL_NAMES.isdisjoint(tool_names))

    async def test_disabled_switch_reaches_route_main_tool_definition_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            core_settings = build_core_settings(root_dir=root_dir)
            disabled_settings = replace(
                MemorySettings.from_workspace_dir(core_settings.workspace_dir),
                enabled=False,
            )

            with patch.object(
                MemorySettings,
                "from_workspace_dir",
                return_value=disabled_settings,
            ):
                route = RouteRuntime(
                    route_id="route_memory_disabled",
                    llm_service=object(),  # type: ignore[arg-type]
                    core_settings=core_settings,
                )

                self.assertTrue(
                    MEMORY_TOOL_NAMES.isdisjoint(
                        set(route._main_registry.registered_tool_names())
                    )
                )
                self.assertIsNone(route._main_registry.get_discoverable("memory_admin"))
                definitions = route._build_main_tool_definitions(["memory_admin"])
                self.assertTrue(
                    MEMORY_TOOL_NAMES.isdisjoint(
                        {definition.name for definition in definitions}
                    )
                )
                self.assertIsNone(route._main_loop._memory_service)

                await route.graceful_shutdown()
