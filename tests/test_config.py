"""Unit tests for core context policy and runtime settings."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis import settings as app_settings

from jarvis.core.config import CompactionSettings, ContextPolicySettings, CoreSettings
from jarvis.core.errors import CoreConfigurationError
from jarvis.gateway import GatewaySettings


class ContextPolicySettingsTests(unittest.TestCase):
    def test_derived_budgets_are_computed_from_context_window(self) -> None:
        settings = ContextPolicySettings(context_window_tokens=400_000)
        self.assertEqual(settings.compact_threshold_tokens, 352_000)
        self.assertEqual(settings.compact_reserve_output_tokens, 24_000)
        self.assertEqual(settings.compact_reserve_overhead_tokens, 12_000)
        self.assertEqual(settings.reserve_tokens, 36_000)
        self.assertEqual(settings.preflight_limit_tokens, 364_000)

    def test_minimum_reserves_apply_for_smaller_context_windows(self) -> None:
        settings = ContextPolicySettings(context_window_tokens=128_000)
        self.assertEqual(settings.compact_reserve_output_tokens, 10_000)
        self.assertEqual(settings.compact_reserve_overhead_tokens, 5_000)
        self.assertEqual(settings.preflight_limit_tokens, 113_000)
        self.assertEqual(settings.compact_threshold_tokens, 108_000)

    def test_context_window_must_fit_derived_reserve_budget(self) -> None:
        with self.assertRaisesRegex(CoreConfigurationError, "derived reserve budget"):
            ContextPolicySettings(context_window_tokens=15_000)

    def test_context_window_must_leave_room_for_derived_threshold(self) -> None:
        with self.assertRaisesRegex(CoreConfigurationError, "derived compaction threshold"):
            ContextPolicySettings(context_window_tokens=16_000)


class CoreSettingsTests(unittest.TestCase):
    def test_defaults_storage_and_identities_dirs_from_settings(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = CoreSettings.from_env()

        self.assertEqual(settings.workspace_dir, Path("/workspace"))
        self.assertEqual(
            settings.transcript_archive_dir,
            Path("/workspace/archive/transcripts"),
        )
        self.assertEqual(settings.identities_dir, Path("/workspace/identities"))
        self.assertEqual(settings.turn_timezone, app_settings.JARVIS_CORE_TIMEZONE)
        self.assertEqual(settings.compaction.provider, app_settings.JARVIS_COMPACTION_PROVIDER)

    def test_requires_agent_workspace_for_host_runs(self) -> None:
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ), patch("jarvis.workspace_paths._running_in_container", return_value=False):
            with self.assertRaisesRegex(
                CoreConfigurationError,
                "AGENT_WORKSPACE must be explicitly set for host runs",
            ):
                CoreSettings.from_env()

    def test_uses_explicit_agent_workspace_for_host_runs(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AGENT_WORKSPACE": "/tmp/jarvis-host-workspace",
            },
            clear=True,
        ), patch("jarvis.workspace_paths._running_in_container", return_value=False):
            settings = CoreSettings.from_env()

        self.assertEqual(settings.workspace_dir, Path("/tmp/jarvis-host-workspace"))
        self.assertEqual(
            settings.transcript_archive_dir,
            Path("/tmp/jarvis-host-workspace/archive/transcripts"),
        )
        self.assertEqual(
            settings.identities_dir,
            Path("/tmp/jarvis-host-workspace/identities"),
        )

    def test_reads_explicit_turn_timezone(self) -> None:
        with patch.dict(
            os.environ,
            {
                "JARVIS_CORE_TIMEZONE": "America/New_York",
            },
            clear=True,
        ):
            settings = CoreSettings.from_env()

        self.assertEqual(settings.turn_timezone, "America/New_York")

    def test_rejects_invalid_turn_timezone(self) -> None:
        with patch.dict(
            os.environ,
            {
                "JARVIS_CORE_TIMEZONE": "Mars/OlympusMons",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                CoreConfigurationError,
                "JARVIS_CORE_TIMEZONE must be a valid IANA timezone",
            ):
                CoreSettings.from_env()


class CompactionSettingsTests(unittest.TestCase):
    def test_rejects_codex_provider(self) -> None:
        with self.assertRaisesRegex(
            CoreConfigurationError,
            "JARVIS_COMPACTION_PROVIDER cannot be 'codex'",
        ):
            CompactionSettings(provider="codex")

    def test_normalizes_provider_name(self) -> None:
        settings = CompactionSettings(provider=" OpenAI ")
        self.assertEqual(settings.provider, "openai")


class GatewaySettingsTests(unittest.TestCase):
    def test_defaults_host_to_localhost(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = GatewaySettings.from_env()

        self.assertEqual(settings.host, "127.0.0.1")
