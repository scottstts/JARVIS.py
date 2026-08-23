"""Tests for the real-gateway headless development runner."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from jarvis.runtime_provider_configuration import RuntimeProviderTarget
from jarvis.ui.telegram.gateway_client import (
    GatewaySystemNoticeEvent,
    GatewayTaskStatusEvent,
    GatewayTurnDoneEvent,
)
from tests.headless.headless import (
    EventAuditor,
    HeadlessRunFailure,
    JsonlEventLog,
    _reset_route,
    build_sandboxed_prompt,
    validate_provider_configuration,
)


class HeadlessDevRunnerTests(unittest.TestCase):
    def test_provider_validation_requires_ox_alpha_for_both_agent_roles(self) -> None:
        valid = (
            RuntimeProviderTarget("Main Agent", "openrouter", "stealth/ox-alpha"),
            RuntimeProviderTarget("Subagent", "openrouter", "stealth/ox-alpha"),
        )
        validate_provider_configuration(
            valid,
            expected_main_provider="openrouter",
            expected_subagent_provider="openrouter",
            expected_model="stealth/ox-alpha",
        )

        invalid = (
            RuntimeProviderTarget("Main Agent", "openrouter", "stealth/ox-alpha"),
            RuntimeProviderTarget("Subagent", "openai", "gpt-5"),
        )
        with self.assertRaisesRegex(HeadlessRunFailure, "Subagent"):
            validate_provider_configuration(
                invalid,
                expected_main_provider="openrouter",
                expected_subagent_provider="openrouter",
                expected_model="stealth/ox-alpha",
            )

    def test_event_auditor_rejects_failure_signatures_and_notice_storms(self) -> None:
        with self.assertRaisesRegex(HeadlessRunFailure, "tool_progress_budget"):
            EventAuditor().observe(
                GatewaySystemNoticeEvent(
                    notice_kind="safety",
                    text="tool_progress_budget_exhausted",
                )
            )

        auditor = EventAuditor()
        notice = GatewaySystemNoticeEvent(
            notice_kind="bash_job_progress_update",
            text="unchanged",
        )
        auditor.observe(notice)
        auditor.observe(notice)
        with self.assertRaisesRegex(HeadlessRunFailure, "three times"):
            auditor.observe(notice)

    def test_prompt_places_all_mutation_inside_unique_test_workspace(self) -> None:
        workspace = Path("/workspace/jarvis-test-headless-example")
        prompt = build_sandboxed_prompt("Build and test an app.", workspace_dir=workspace)

        self.assertIn(str(workspace), prompt)
        self.assertIn("only inside", prompt)
        self.assertIn("real Jarvis tools and subagents", prompt)
        self.assertTrue(prompt.endswith("Build and test an app."))


class HeadlessResetTests(unittest.IsolatedAsyncioTestCase):
    async def test_reset_accepts_command_only_turn_without_turn_started(self) -> None:
        class _FakeRouteSession:
            def __init__(self) -> None:
                self.client_message_id = ""
                self.sent_text = ""

            async def send_user_message(self, *, text: str, client_message_id: str) -> None:
                self.sent_text = text
                self.client_message_id = client_message_id

            async def events(self):
                yield GatewayTaskStatusEvent(active=False)
                yield GatewayTurnDoneEvent(
                    client_message_id=self.client_message_id,
                    command="/new",
                )

        with tempfile.TemporaryDirectory() as tmp:
            session = _FakeRouteSession()
            event_log = JsonlEventLog(Path(tmp) / "events.jsonl")
            try:
                await _reset_route(  # pyright: ignore[reportArgumentType]
                    session,
                    event_log=event_log,
                )
            finally:
                event_log.close()

            self.assertEqual(session.sent_text, "/new")
