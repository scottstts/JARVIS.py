"""Unit tests for core context policy and runtime settings."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from core.config import ContextPolicySettings, CoreSettings
from core.errors import CoreConfigurationError


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
        self.assertEqual(settings.storage_dir, Path("/workspace/storage"))
        self.assertEqual(settings.identities_dir, Path("/workspace/identities"))
