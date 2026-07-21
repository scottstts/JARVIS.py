"""Tests for shared provider transport-timeout behavior."""

from __future__ import annotations

import unittest

import httpx
import requests

from jarvis.llm.timeouts import (
    ProviderTransportTimeouts,
    transport_timeout_metadata,
)


class ProviderTransportTimeoutTests(unittest.TestCase):
    def test_httpx_and_requests_receive_split_timeouts(self) -> None:
        timeouts = ProviderTransportTimeouts(
            connect_seconds=12.0,
            read_seconds=90.0,
        )

        self.assertEqual(
            timeouts.as_httpx().as_dict(),
            {
                "connect": 12.0,
                "read": 90.0,
                "write": 12.0,
                "pool": 12.0,
            },
        )
        self.assertEqual(timeouts.as_requests(), (12.0, 90.0))

    def test_timeout_metadata_classifies_connect_and_read_phases(self) -> None:
        timeouts = ProviderTransportTimeouts(
            connect_seconds=12.0,
            read_seconds=90.0,
        )

        connect_metadata = transport_timeout_metadata(
            requests.ConnectTimeout("connect"),
            timeouts=timeouts,
        )
        read_metadata = transport_timeout_metadata(
            httpx.ReadTimeout("read"),
            timeouts=timeouts,
        )

        self.assertEqual(connect_metadata["timeout_kind"], "connect")
        self.assertEqual(connect_metadata["timeout_limit_seconds"], 12.0)
        self.assertEqual(read_metadata["timeout_kind"], "read_idle")
        self.assertEqual(read_metadata["timeout_limit_seconds"], 90.0)
        self.assertEqual(read_metadata["connect_timeout_seconds"], 12.0)
        self.assertEqual(read_metadata["read_timeout_seconds"], 90.0)

    def test_non_positive_transport_timeout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "connect_seconds"):
            ProviderTransportTimeouts(connect_seconds=0, read_seconds=90)

