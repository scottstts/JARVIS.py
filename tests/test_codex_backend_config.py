"""Unit tests for Codex backend settings and path mapping."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis.codex_backend.config import CodexBackendSettings, CodexConfigurationError
from jarvis.codex_backend.path_mapping import CodexPathMapper


class CodexBackendSettingsTests(unittest.TestCase):
    def test_from_env_reads_overrides_and_builds_sandbox_policy(self) -> None:
        with patch.dict(
            os.environ,
            {
                "JARVIS_CODEX_WS_URL": "ws://host.docker.internal:5555",
                "JARVIS_CODEX_MODEL": "gpt-5-codex",
                "JARVIS_CODEX_REASONING_EFFORT": "high",
                "JARVIS_CODEX_REASONING_SUMMARY": "concise",
                "JARVIS_CODEX_PERSONALITY": "pragmatic",
                "JARVIS_CODEX_SERVICE_NAME": "Jarvis Dev",
                "JARVIS_CODEX_HOST_REPO_ROOT": "/Users/test/Jarvis",
                "JARVIS_CODEX_HOST_WORKSPACE_ROOT": "/Users/test/.jarvis/workspace",
                "JARVIS_CODEX_APPROVAL_POLICY": "on-request",
                "JARVIS_CODEX_SANDBOX_NETWORK_ACCESS": "true",
                "JARVIS_CODEX_WS_BEARER_TOKEN": "secret-token",
            },
            clear=True,
        ):
            settings = CodexBackendSettings.from_env()

        self.assertEqual(settings.ws_url, "ws://host.docker.internal:5555")
        self.assertEqual(settings.model, "gpt-5-codex")
        self.assertEqual(settings.reasoning_effort, "high")
        self.assertEqual(settings.reasoning_summary, "concise")
        self.assertEqual(settings.personality, "pragmatic")
        self.assertEqual(settings.service_name, "Jarvis Dev")
        self.assertEqual(settings.approval_policy, "on-request")
        self.assertTrue(settings.sandbox_network_access)
        self.assertEqual(settings.ws_bearer_token, "secret-token")
        self.assertEqual(settings.require_host_paths()[0], Path("/Users/test/Jarvis"))
        self.assertEqual(
            settings.require_host_paths()[1],
            Path("/Users/test/.jarvis/workspace"),
        )
        sandbox_policy = settings.sandbox_policy()
        self.assertEqual(sandbox_policy["type"], "workspaceWrite")
        self.assertTrue(sandbox_policy["networkAccess"])

    def test_require_host_paths_raises_when_unset(self) -> None:
        settings = CodexBackendSettings(
            ws_url="ws://host.docker.internal:4500",
            model=None,
            reasoning_effort="medium",
            reasoning_summary="none",
            personality="pragmatic",
            service_name="Jarvis",
            host_repo_root=None,
            host_workspace_root=None,
            approval_policy="untrusted",
            sandbox_network_access=False,
            ws_bearer_token=None,
        )

        with self.assertRaises(CodexConfigurationError):
            settings.require_host_paths()


class CodexPathMapperTests(unittest.TestCase):
    def test_maps_repo_and_workspace_paths_both_directions(self) -> None:
        mapper = CodexPathMapper(
            host_repo_root=Path("/Users/test/Jarvis"),
            host_workspace_root=Path("/Users/test/.jarvis/workspace"),
        )

        self.assertEqual(
            mapper.container_to_host("/repo/src/jarvis/main.py"),
            Path("/Users/test/Jarvis/src/jarvis/main.py"),
        )
        self.assertEqual(
            mapper.container_to_host("/workspace/archive/transcripts/s1.jsonl"),
            Path("/Users/test/.jarvis/workspace/archive/transcripts/s1.jsonl"),
        )
        self.assertEqual(
            mapper.host_to_container("/Users/test/Jarvis/dev_docs/codex_backend.md"),
            Path("/repo/dev_docs/codex_backend.md"),
        )
        self.assertEqual(
            mapper.host_to_container("/Users/test/.jarvis/workspace/runtime_tools/tool.json"),
            Path("/workspace/runtime_tools/tool.json"),
        )
