"""Unit tests for the runtime memory subsystem."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core import AgentLoop
from llm import LLMResponse, LLMUsage
from memory import MemoryService, MemorySettings
from storage import SessionStorage
from tests.helpers import build_core_settings


class _FakeMemoryBootstrapLLMService:
    async def generate(self, request):
        _ = request
        return LLMResponse(
            provider="fake",
            model="fake-chat",
            text="OK",
            tool_calls=[],
            finish_reason="stop",
            usage=LLMUsage(input_tokens=8, output_tokens=2, total_tokens=10),
            response_id="resp_fake",
        )


class MemoryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_search_and_get_core_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(
                settings=MemorySettings.from_workspace_dir(workspace_dir),
                llm_service=None,
            )

            created = await service.write(
                operation="create",
                target_kind="core",
                title="User Communication Style",
                summary="User prefers direct and concise engineering feedback.",
                body_sections={
                    "Summary": "User prefers direct and concise engineering feedback.",
                    "Details": "- Skip hedging.\n- Surface tradeoffs clearly.",
                    "Notes": "- Reinforce with concrete next steps.",
                },
                facts=[
                    {
                        "text": "User prefers direct and concise engineering feedback.",
                    }
                ],
            )

            results = await service.search(
                query="direct and concise engineering feedback",
                mode="lexical",
            )

            self.assertTrue(results)
            self.assertEqual(results[0].document_id, created.document_id)
            self.assertEqual(results[0].kind, "core")

            document_text = await service.get_document(document_id=created.document_id)
            self.assertIn("User Communication Style", document_text)
            self.assertIn("direct and concise engineering feedback", document_text)

    async def test_out_of_band_edit_is_reconciled_before_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(
                settings=MemorySettings.from_workspace_dir(workspace_dir),
                llm_service=None,
            )

            created = await service.write(
                operation="create",
                target_kind="core",
                title="Development Preferences",
                summary="User likes clear code review findings first.",
                body_sections={
                    "Summary": "User likes clear code review findings first.",
                    "Details": "- Findings before summary.",
                    "Notes": "- Mention real risks explicitly.",
                },
            )

            original_text = created.path.read_text(encoding="utf-8")
            edited_text = original_text.replace(
                "User likes clear code review findings first.",
                "User wants blunt review findings before any summary.",
            )
            created.path.write_text(edited_text, encoding="utf-8")

            results = await service.search(
                query="blunt review findings",
                mode="lexical",
            )

            self.assertTrue(results)
            self.assertEqual(results[0].document_id, created.document_id)
            reconciled_text = await service.get_document(document_id=created.document_id)
            self.assertIn("blunt review findings", reconciled_text)

    async def test_bootstrap_preview_includes_core_and_ongoing_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(
                settings=MemorySettings.from_workspace_dir(workspace_dir),
                llm_service=None,
            )

            await service.write(
                operation="create",
                target_kind="core",
                title="User Development Preferences",
                summary="User wants concise, practical implementation updates.",
                body_sections={
                    "Summary": "User wants concise, practical implementation updates.",
                    "Details": "- Prefer direct status reports.",
                    "Notes": "- Keep solutions pragmatic.",
                },
            )
            await service.write(
                operation="create",
                target_kind="ongoing",
                title="Jarvis Memory System",
                summary="The runtime memory system is actively being implemented.",
                body_sections={
                    "Summary": "The runtime memory system is actively being implemented.",
                    "Current State": "- Core package and tool wiring are in progress.",
                    "Open Loops": "- Finish tool integration.\n- Add tests and docs.",
                    "Artifacts": "",
                    "Notes": "",
                },
                review_after="2026-03-13T00:00:00+00:00",
            )

            preview = await service.render_bootstrap_preview()

            self.assertIn("User Development Preferences", preview)
            self.assertIn("Jarvis Memory System", preview)
            self.assertIn("Finish tool integration", preview)

    async def test_agent_loop_injects_memory_bootstrap_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = build_core_settings(
                root_dir=root,
                compact_reserve_output_tokens=256,
                compact_reserve_overhead_tokens=64,
            )
            memory_service = MemoryService(
                settings=MemorySettings.from_workspace_dir(settings.workspace_dir),
                llm_service=None,
            )
            await memory_service.write(
                operation="create",
                target_kind="core",
                title="User Communication Style",
                summary="User prefers direct implementation updates.",
                body_sections={
                    "Summary": "User prefers direct implementation updates.",
                    "Details": "- Keep updates factual.",
                    "Notes": "- Avoid unnecessary framing.",
                },
            )

            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeMemoryBootstrapLLMService(),
                settings=settings,
                storage=storage,
            )

            result = await loop.handle_user_input("Say OK.")
            records = storage.load_records(result.session_id)

            memory_bootstrap_records = [
                record
                for record in records
                if record.role == "developer" and record.metadata.get("memory_bootstrap") == "core"
            ]

            self.assertEqual(len(memory_bootstrap_records), 1)
            self.assertIn("User Communication Style", memory_bootstrap_records[0].content)

