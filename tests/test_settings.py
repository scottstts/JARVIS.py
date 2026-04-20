"""Unit tests for YAML-backed app settings."""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from jarvis import settings as app_settings


_PACKAGED_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "src" / "jarvis" / "settings.yml"


def _field_value(document: dict[str, object], *path: str) -> object:
    current: object = document
    for segment in path:
        if not isinstance(current, dict):
            raise AssertionError(f"Expected mapping while walking path {'.'.join(path)}")
        current = current[segment]
    if not isinstance(current, dict):
        raise AssertionError(f"Expected field mapping at path {'.'.join(path)}")
    return current["value"]


def _set_field_value(document: dict[str, object], path: tuple[str, ...], value: object) -> None:
    current: object = document
    for segment in path:
        if not isinstance(current, dict):
            raise AssertionError(f"Expected mapping while walking path {'.'.join(path)}")
        current = current[segment]
    if not isinstance(current, dict):
        raise AssertionError(f"Expected field mapping at path {'.'.join(path)}")
    current["value"] = value


class SettingsModuleTests(unittest.TestCase):
    def tearDown(self) -> None:
        importlib.reload(app_settings)

    def test_uses_packaged_template_when_workspace_settings_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "AGENT_WORKSPACE": tmp,
                },
                clear=True,
            ):
                module = importlib.reload(app_settings)

        self.assertEqual(module.SETTINGS_SOURCE_PATH, _PACKAGED_TEMPLATE_PATH.resolve())
        self.assertEqual(module.JARVIS_LLM_DEFAULT_PROVIDER, "openai")

    def test_prefers_workspace_settings_file_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            settings_dir = workspace_dir / "settings"
            settings_dir.mkdir(parents=True)
            workspace_settings_path = settings_dir / "settings.yml"

            payload = yaml.safe_load(_PACKAGED_TEMPLATE_PATH.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                self.fail("Expected packaged settings template to be a mapping.")
            _set_field_value(payload, ("llm", "fields", "default_provider"), "grok")
            _set_field_value(payload, ("providers", "groups", "grok", "fields", "chat_model"), "grok-test")
            workspace_settings_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "AGENT_WORKSPACE": str(workspace_dir),
                },
                clear=True,
                ):
                    module = importlib.reload(app_settings)

        self.assertEqual(module.SETTINGS_SOURCE_PATH, workspace_settings_path.resolve())
        self.assertEqual(module.JARVIS_LLM_DEFAULT_PROVIDER, "grok")
        self.assertEqual(module.JARVIS_GROK_CHAT_MODEL, "grok-test")

    def test_invalid_workspace_settings_fail_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            settings_dir = workspace_dir / "settings"
            settings_dir.mkdir(parents=True)
            workspace_settings_path = settings_dir / "settings.yml"

            payload = yaml.safe_load(_PACKAGED_TEMPLATE_PATH.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                self.fail("Expected packaged settings template to be a mapping.")
            del payload["llm"]["fields"]["default_provider"]
            workspace_settings_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "AGENT_WORKSPACE": str(workspace_dir),
                },
                clear=True,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"{workspace_settings_path.resolve()}'.*llm\.default_provider",
                ):
                    importlib.reload(app_settings)
