"""Unit tests for the runtime memory subsystem."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core import AgentLoop
from llm import EmbeddingResponse, LLMResponse, LLMUsage
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


class _FakeEmbeddingLLMService(_FakeMemoryBootstrapLLMService):
    async def embed(self, request):
        inputs = request.inputs if isinstance(request.inputs, list) else [request.inputs]
        embeddings = [[float(len(str(item))), 1.0, 0.5] for item in inputs]
        return EmbeddingResponse(
            provider="fake",
            model="fake-embedding",
            embeddings=embeddings,
            usage=LLMUsage(input_tokens=4, output_tokens=0, total_tokens=4),
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

            response = await service.search(
                query="direct and concise engineering feedback",
                mode="lexical",
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, created.document_id)
            self.assertEqual(response.results[0].kind, "core")

            document_text = await service.get_document(document_id=created.document_id)
            self.assertIn("User Communication Style", document_text)
            self.assertIn("direct and concise engineering feedback", document_text)

    async def test_summary_only_ongoing_memory_is_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(
                settings=MemorySettings.from_workspace_dir(workspace_dir),
                llm_service=None,
            )

            created = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Developing Jarvis as the best AI assistant",
                summary="Scott is currently developing Jarvis as the best AI assistant.",
            )

            response = await service.search(
                query="developing Jarvis as the best AI assistant",
                mode="lexical",
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, created.document_id)
            document_text = await service.get_document(document_id=created.document_id)
            self.assertIn("Scott is currently developing Jarvis as the best AI assistant.", document_text)

    async def test_search_repairs_stale_summary_only_index_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(
                settings=MemorySettings.from_workspace_dir(workspace_dir),
                llm_service=None,
            )

            created = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Developing Jarvis as the best AI assistant",
                summary="Scott is currently developing Jarvis as the best AI assistant.",
            )

            historical_text = created.path.read_text(encoding="utf-8").replace(
                "\n## Summary\nScott is currently developing Jarvis as the best AI assistant.",
                "\n## Summary\n",
                1,
            )
            created.path.write_text(historical_text, encoding="utf-8")

            historical_document = service._store.read_document(created.path)
            service._index_db.upsert_document(historical_document, ())

            response = await service.search(
                query="developing Jarvis as the best AI assistant",
                mode="lexical",
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, created.document_id)

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

            response = await service.search(
                query="blunt review findings",
                mode="lexical",
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, created.document_id)
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

    async def test_append_daily_with_title_and_summary_persists_notable_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(
                settings=MemorySettings.from_workspace_dir(workspace_dir),
                llm_service=None,
            )

            written = await service.write(
                operation="append_daily",
                target_kind="daily",
                date="2026-03-12",
                timezone_name="Europe/Dublin",
                title="Jarvis development finished",
                summary="Scott said the Jarvis development project is finished.",
            )

            daily_text = written.path.read_text(encoding="utf-8")
            self.assertIn(
                "- Jarvis development finished: Scott said the Jarvis development project is finished.",
                daily_text,
            )

            response = await service.search(
                query="Jarvis development project is finished",
                mode="lexical",
                scopes=("daily",),
                daily_lookback_days=365,
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, written.document_id)

    async def test_close_archives_ongoing_memory_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(
                settings=MemorySettings.from_workspace_dir(workspace_dir),
                llm_service=None,
            )

            created = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Jarvis Memory System",
                summary="The Jarvis memory system is actively being implemented.",
                body_sections={
                    "Summary": "The Jarvis memory system is actively being implemented.",
                    "Current State": "- Core package and tool wiring are in progress.",
                    "Open Loops": "- Finish tool integration.",
                    "Artifacts": "",
                    "Notes": "",
                },
                review_after="2026-03-13T00:00:00+00:00",
            )

            closed = await service.write(
                operation="close",
                target_kind="ongoing",
                document_id=created.document_id,
                close_reason="finished",
            )

            self.assertFalse(created.path.exists())
            self.assertIn("/archive/ongoing/", closed.path.as_posix())

            archived_document = service._store.read_document(closed.path)
            self.assertEqual(archived_document.status, "closed")
            self.assertEqual(archived_document.close_reason, "finished")

            active_response = await service.search(
                query="Jarvis memory system",
                mode="lexical",
            )
            self.assertFalse(active_response.results)

            archive_response = await service.search(
                query="Jarvis memory system",
                mode="lexical",
                scopes=("archive",),
            )
            self.assertTrue(archive_response.results)
            self.assertEqual(archive_response.results[0].document_id, created.document_id)

    async def test_hybrid_search_falls_back_when_semantic_index_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(
                settings=MemorySettings.from_workspace_dir(workspace_dir),
                llm_service=_FakeEmbeddingLLMService(),
            )

            created = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Developing Jarvis as the best AI assistant",
                summary="Scott is currently developing Jarvis as the best AI assistant.",
            )

            service._index_db.write_state(
                {
                    "semantic_enabled": True,
                    "semantic_error": None,
                    "embedding_dimensions": 3,
                }
            )
            with service._index_db._connect_embeddings(load_vec=True) as conn:
                conn.execute("drop table if exists embedding_items_vec")

            response = await service.search(
                query="developing Jarvis as the best AI assistant",
                mode="auto",
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, created.document_id)
            self.assertTrue(response.semantic_disabled)
            self.assertTrue(response.warnings)
            self.assertIn("embedding vector table is missing", response.warnings[0])

    async def test_maintenance_repairs_missing_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(
                settings=MemorySettings.from_workspace_dir(workspace_dir),
                llm_service=_FakeEmbeddingLLMService(),
            )

            created = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Developing Jarvis as the best AI assistant",
                summary="Scott is currently developing Jarvis as the best AI assistant.",
            )

            service._index_db.write_state(
                {
                    "semantic_enabled": True,
                    "semantic_error": None,
                    "embedding_dimensions": 3,
                }
            )
            service._index_db.delete_embeddings_for_document(created.document_id)

            runs = await service.run_due_maintenance()

            repair_run = next(run for run in runs if run.job_name == "repair_missing_embeddings")
            self.assertEqual(repair_run.status, "ok")
            self.assertEqual(repair_run.summary["repaired_documents"], 1)
            self.assertGreater(
                service._index_db.embedding_vector_count_for_document(created.document_id),
                0,
            )

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
