"""Unit tests for core context policy settings."""

from __future__ import annotations

import unittest

from core.config import ContextPolicySettings
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
