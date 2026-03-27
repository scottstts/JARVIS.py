"""Unit tests for core context policy and runtime settings."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis import settings as app_settings

from jarvis.core.config import ContextPolicySettings, CoreSettings
from jarvis.core.errors import CoreConfigurationError
from jarvis.gateway import GatewaySettings


class ContextPolicySettingsTests(unittest.TestCase):
    def test_reserve_and_preflight_limits_are_computed(self) -> None:
        settings = ContextPolicySettings(
            context_window_tokens=400_000,
            compact_threshold_tokens=350_000,
            compact_reserve_output_tokens=16_000,
            compact_reserve_overhead_tokens=10_000,
        )
        self.assertEqual(settings.reserve_tokens, 26_000)
        self.assertEqual(settings.preflight_limit_tokens, 374_000)

    def test_threshold_must_be_less_than_window(self) -> None:
        with self.assertRaises(CoreConfigurationError):
            ContextPolicySettings(
                context_window_tokens=1_000,
                compact_threshold_tokens=1_000,
                compact_reserve_output_tokens=100,
                compact_reserve_overhead_tokens=100,
            )

    def test_combined_reserve_must_fit_in_window(self) -> None:
        with self.assertRaises(CoreConfigurationError):
            ContextPolicySettings(
                context_window_tokens=1_000,
                compact_threshold_tokens=900,
                compact_reserve_output_tokens=700,
                compact_reserve_overhead_tokens=300,
            )


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


class GatewaySettingsTests(unittest.TestCase):
    def test_defaults_host_to_localhost(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = GatewaySettings.from_env()

        self.assertEqual(settings.host, "127.0.0.1")
