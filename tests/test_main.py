"""Unit tests for the combined system entrypoint."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import main as repo_main
from gateway import GatewaySettings
from ui.telegram import UISettings

from runtime_env import load_docker_secrets_if_present

_SRC_MAIN_PATH = Path(__file__).resolve().parents[1] / "src" / "main.py"


def _load_src_main_module() -> ModuleType:
    src_dir = _SRC_MAIN_PATH.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    spec = importlib.util.spec_from_file_location("jarvis_test_src_main", _SRC_MAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load main module from {_SRC_MAIN_PATH}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SRC_MAIN = _load_src_main_module()


class _FakeServer:
    def __init__(
        self,
        config: object,
        *,
        startup_error: Exception | None = None,
    ) -> None:
        self.config = config
        self.started = False
        self.should_exit = False
        self._startup_error = startup_error

    async def serve(self) -> None:
        if self._startup_error is not None:
            raise self._startup_error
        self.started = True
        while not self.should_exit:
            await asyncio.sleep(0)


class RuntimeEnvTests(unittest.TestCase):
    def test_load_docker_secrets_if_present_sets_values_without_overriding_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            secrets_dir = Path(tmp) / "secrets"
            secrets_dir.mkdir()
            (secrets_dir / "FIRST").write_text("value-one\n", encoding="utf-8")
            (secrets_dir / "SECOND").write_text("value two\n", encoding="utf-8")
            (secrets_dir / "THIRD").write_text("from-docker-secret\n", encoding="utf-8")
            (secrets_dir / "EMPTY").write_text("\n", encoding="utf-8")
            (secrets_dir / "invalid-name").write_text("ignored\n", encoding="utf-8")

            with patch.dict(os.environ, {"THIRD": "from-env"}, clear=True):
                loaded = load_docker_secrets_if_present(secrets_dir)
                self.assertEqual(
                    loaded,
                    (
                        secrets_dir / "FIRST",
                        secrets_dir / "SECOND",
                    ),
                )
                self.assertEqual(os.environ["FIRST"], "value-one")
                self.assertEqual(os.environ["SECOND"], "value two")
                self.assertEqual(os.environ["THIRD"], "from-env")
                self.assertNotIn("EMPTY", os.environ)
                self.assertNotIn("invalid-name", os.environ)


class MainEntrypointTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_system_binds_ui_to_local_gateway_and_shuts_server_down(self) -> None:
        fake_server: _FakeServer | None = None
        captured_ui_settings: list[UISettings] = []
        captured_app_args: list[GatewaySettings] = []

        def fake_create_app(*, gateway_settings: GatewaySettings, **_: object) -> object:
            captured_app_args.append(gateway_settings)
            return object()

        def fake_config(**kwargs: object) -> dict[str, object]:
            return dict(kwargs)

        def fake_server_factory(config: object) -> _FakeServer:
            nonlocal fake_server
            fake_server = _FakeServer(config)
            return fake_server

        async def fake_run_telegram_ui(settings: UISettings) -> None:
            captured_ui_settings.append(settings)

        gateway_settings = GatewaySettings(
            host="0.0.0.0",
            port=8181,
            websocket_path="/ws",
            max_message_chars=32_000,
        )
        ui_settings = UISettings(
            telegram_token="token",
            telegram_allowed_user_id=777,
            gateway_ws_base_url="ws://example.com/remote",
        )

        with patch.object(SRC_MAIN, "create_app", side_effect=fake_create_app):
            with patch.object(SRC_MAIN.uvicorn, "Config", side_effect=fake_config):
                with patch.object(SRC_MAIN.uvicorn, "Server", side_effect=fake_server_factory):
                    with patch.object(SRC_MAIN, "run_telegram_ui", side_effect=fake_run_telegram_ui):
                        await SRC_MAIN.run_system(
                            gateway_settings=gateway_settings,
                            ui_settings=ui_settings,
                        )

        self.assertIsNotNone(fake_server)
        if fake_server is None:
            self.fail("Expected fake server to be created.")
        self.assertTrue(fake_server.should_exit)
        self.assertEqual(captured_app_args, [gateway_settings])
        self.assertEqual(len(captured_ui_settings), 1)
        self.assertEqual(
            captured_ui_settings[0].gateway_ws_base_url,
            "ws://127.0.0.1:8181/ws",
        )
        self.assertFalse(fake_server.config["access_log"])
        self.assertEqual(fake_server.config["host"], "0.0.0.0")
        self.assertEqual(fake_server.config["port"], 8181)

    async def test_run_system_propagates_gateway_startup_failure(self) -> None:
        startup_error = RuntimeError("gateway boom")

        def fake_config(**kwargs: object) -> dict[str, object]:
            return dict(kwargs)

        def fake_server_factory(config: object) -> _FakeServer:
            return _FakeServer(config, startup_error=startup_error)

        async def fake_run_telegram_ui(_settings: UISettings) -> None:
            self.fail("Telegram UI should not start when gateway startup fails.")

        with patch.object(SRC_MAIN, "create_app", return_value=object()):
            with patch.object(SRC_MAIN.uvicorn, "Config", side_effect=fake_config):
                with patch.object(SRC_MAIN.uvicorn, "Server", side_effect=fake_server_factory):
                    with patch.object(SRC_MAIN, "run_telegram_ui", side_effect=fake_run_telegram_ui):
                        with self.assertRaises(RuntimeError) as context:
                            await SRC_MAIN.run_system(
                                gateway_settings=GatewaySettings(
                                    host="127.0.0.1",
                                    port=8080,
                                    websocket_path="/ws",
                                    max_message_chars=32_000,
                                ),
                                ui_settings=UISettings(
                                    telegram_token="token",
                                    telegram_allowed_user_id=777,
                                ),
                            )
        self.assertEqual(str(context.exception), "gateway boom")


class MainFunctionTests(unittest.TestCase):
    def test_main_exits_cleanly_on_keyboard_interrupt(self) -> None:
        async def fake_run_system() -> None:
            raise KeyboardInterrupt

        with patch.object(SRC_MAIN, "load_docker_secrets_if_present"):
            with patch.object(SRC_MAIN, "configure_application_logging"):
                with patch.object(SRC_MAIN, "run_system", side_effect=fake_run_system):
                    with self.assertLogs(SRC_MAIN.LOGGER.name, level="INFO") as captured_logs:
                        SRC_MAIN.main()

        self.assertIn(
            "Shutdown requested via Ctrl+C; exiting cleanly.",
            captured_logs.output[0],
        )


class RepoRootMainShimTests(unittest.TestCase):
    def test_main_delegates_to_loaded_src_module(self) -> None:
        fake_module = Mock()

        with patch.object(repo_main, "_load_src_main_module", return_value=fake_module):
            repo_main.main()

        fake_module.main.assert_called_once_with()
