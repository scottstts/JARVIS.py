"""Integration tests for AgentLoop using real configured LLM provider."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import settings as app_settings

from core import AgentLoop
from llm import LLMService
from runtime_env import load_docker_secrets_if_present
from storage import SessionStorage
from tests.helpers import build_core_settings


class AgentLoopRealLLMTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _load_docker_secrets_if_present()
        provider = (
            os.getenv("JARVIS_LLM_DEFAULT_PROVIDER")
            or str(app_settings.JARVIS_LLM_DEFAULT_PROVIDER)
        ).strip().lower()
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise unittest.SkipTest("OPENAI_API_KEY is not configured.")
        if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            raise unittest.SkipTest("ANTHROPIC_API_KEY is not configured.")
        if provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            raise unittest.SkipTest("GOOGLE_API_KEY is not configured.")
        if provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            raise unittest.SkipTest("OPENROUTER_API_KEY is not configured.")

    async def test_initial_session_bootstraps_all_identity_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("README SHOULD NOT BE INJECTED", encoding="utf-8")

            settings = build_core_settings(
                root_dir=root,
                compact_reserve_output_tokens=256,
                compact_reserve_overhead_tokens=64,
            )
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = LLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )
            try:
                result = await loop.handle_user_input("Reply with ACK only.")
                self.assertTrue(result.response_text.strip())

                records = storage.load_records(result.session_id)
                message_records = [record for record in records if record.kind == "message"]
                self.assertGreaterEqual(len(message_records), 6)
                self.assertEqual(message_records[0].role, "system")
                self.assertEqual(message_records[0].content, "PROGRAM PROMPT")
                self.assertEqual(message_records[1].role, "system")
                self.assertEqual(message_records[1].content, "REACTOR PROMPT")
                self.assertEqual(message_records[2].role, "system")
                self.assertEqual(message_records[2].content, "USER PROMPT")
                self.assertEqual(message_records[3].role, "system")
                self.assertEqual(message_records[3].content, "ARMOR PROMPT")
                self.assertEqual(message_records[4].role, "user")
                self.assertEqual(message_records[4].content, "Reply with ACK only.")
                self.assertEqual(message_records[5].role, "assistant")
                self.assertNotIn(
                    "README SHOULD NOT BE INJECTED",
                    "\n".join(record.content for record in message_records),
                )
            finally:
                await llm_service.aclose()

    async def test_manual_compaction_creates_new_session_with_summary_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(
                root_dir=Path(tmp),
                compact_reserve_output_tokens=256,
                compact_reserve_overhead_tokens=64,
            )
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = LLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )
            try:
                first = await loop.handle_user_input("Remember marker CODE-12345. Reply with OK.")
                old_session_id = first.session_id

                compacted = await loop.handle_user_input("/compact keep constraints and marker")
                self.assertTrue(compacted.compaction_performed)
                self.assertNotEqual(compacted.session_id, old_session_id)

                old_session = storage.get_session(old_session_id)
                self.assertIsNotNone(old_session)
                self.assertEqual(old_session.status, "archived")  # type: ignore[union-attr]
                old_records = storage.load_records(old_session_id)
                self.assertTrue(any(record.kind == "compaction" for record in old_records))

                new_records = storage.load_records(compacted.session_id)
                summary_records = [
                    record
                    for record in new_records
                    if record.role == "developer" and record.metadata.get("summary_seed")
                ]
                self.assertEqual(len(summary_records), 1)
                self.assertTrue(summary_records[0].content.strip())

                follow_up = await loop.handle_user_input(
                    "What marker do you have from previous context?"
                )
                self.assertEqual(follow_up.session_id, compacted.session_id)
                self.assertTrue(follow_up.response_text.strip())
            finally:
                await llm_service.aclose()

    async def test_new_command_starts_fresh_session_without_summary_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(
                root_dir=Path(tmp),
                compact_reserve_output_tokens=256,
                compact_reserve_overhead_tokens=64,
            )
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = LLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )
            try:
                first = await loop.handle_user_input("Say OK.")
                reset = await loop.handle_user_input("/new")

                self.assertEqual(reset.command, "/new")
                self.assertNotEqual(first.session_id, reset.session_id)

                new_records = storage.load_records(reset.session_id)
                message_records = [record for record in new_records if record.kind == "message"]
                self.assertEqual(
                    [record.role for record in message_records],
                    ["system", "system", "system", "system"],
                )
                self.assertFalse(any(record.metadata.get("summary_seed") for record in message_records))

                continued = await loop.handle_user_input("Confirm reset session is active.")
                self.assertEqual(continued.session_id, reset.session_id)
                self.assertTrue(continued.response_text.strip())
            finally:
                await llm_service.aclose()


def _load_docker_secrets_if_present() -> None:
    load_docker_secrets_if_present(Path(__file__).resolve().parents[1] / "secrets")
