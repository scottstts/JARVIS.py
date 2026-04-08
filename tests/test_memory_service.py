"""Unit tests for the runtime memory subsystem."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jarvis.core import AgentLoop
from jarvis.llm import EmbeddingResponse, LLMResponse, LLMUsage
from jarvis.memory import MemoryService, MemorySettings
from jarvis.storage import SessionStorage
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

    async def test_upsert_summary_only_retitled_memory_updates_summary_section_and_path(self) -> None:
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
                title="User routine: morning run",
                summary="Scott runs in the morning.",
            )

            updated = await service.write(
                operation="upsert",
                target_kind="ongoing",
                document_id=created.document_id,
                title="User routine: evening run",
                summary="Scott runs in the evening.",
            )

            updated_document = service._store.read_document(updated.path)

            self.assertFalse(created.path.exists())
            self.assertTrue(updated.path.exists())
            self.assertEqual(updated.changed_paths, (created.path, updated.path))
            self.assertEqual(updated.path.name, "user-routine-evening-run.md")
            self.assertEqual(updated_document.title, "User routine: evening run")
            self.assertEqual(updated_document.summary, "Scott runs in the evening.")
            self.assertEqual(updated_document.sections["Summary"], "Scott runs in the evening.")

            document_text = await service.get_document(document_id=created.document_id)
            self.assertIn("Scott runs in the evening.", document_text)
            self.assertNotIn("Scott runs in the morning.", document_text)

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

    async def test_integrity_check_warns_on_summary_and_title_path_drift(self) -> None:
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
                title="User routine: morning run",
                summary="Scott runs in the morning.",
            )

            drifted_text = created.path.read_text(encoding="utf-8")
            drifted_text = drifted_text.replace(
                "title: 'User routine: morning run'",
                "title: 'User routine: evening run'",
                1,
            ).replace(
                "# User routine: morning run",
                "# User routine: evening run",
                1,
            ).replace(
                "summary: Scott runs in the morning.",
                "summary: Scott runs in the evening.",
                1,
            )
            created.path.write_text(drifted_text, encoding="utf-8")

            issues = await service.integrity_check()
            issue_codes = {issue.code for issue in issues}

            self.assertIn("title_path_mismatch", issue_codes)
            self.assertIn("summary_section_mismatch", issue_codes)

            runs = await service.run_due_maintenance()
            integrity_run = next(run for run in runs if run.job_name == "integrity_check")
            repair_run = next(run for run in runs if run.job_name == "repair_canonical_drift")

            self.assertEqual(repair_run.status, "ok")
            self.assertEqual(repair_run.summary["repaired_documents"], 1)
            self.assertEqual(integrity_run.status, "warning")
            self.assertGreaterEqual(integrity_run.summary["issue_count"], 1)

    async def test_repair_canonical_drift_repairs_legacy_summary_section_gap(self) -> None:
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
                title="User communication preference",
                summary="User prefers direct, honest explanations.",
            )

            drifted_text = created.path.read_text(encoding="utf-8").replace(
                "\n## Summary\nUser prefers direct, honest explanations.",
                "\n## Summary\n",
                1,
            )
            created.path.write_text(drifted_text, encoding="utf-8")

            issues_before = await service.integrity_check()
            self.assertEqual({issue.code for issue in issues_before}, {"summary_section_empty"})

            repair_summary = await service.repair_canonical_drift()
            repaired_document = service._store.read_document(created.path)
            issues_after = await service.integrity_check()

            self.assertEqual(repair_summary["repaired_documents"], 1)
            self.assertEqual(repair_summary["repaired_issue_counts"]["summary_section_empty"], 1)
            self.assertEqual(repaired_document.sections["Summary"], "User prefers direct, honest explanations.")
            self.assertFalse(issues_after)

    async def test_close_rewrites_terminal_summary_before_archiving(self) -> None:
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
                title="Jarvis memory system improvement",
                summary="Scott is actively working on the Jarvis memory system improvement.",
            )

            closed = await service.write(
                operation="close",
                target_kind="ongoing",
                document_id=created.document_id,
                close_reason="Completed on 2026-03-13: Scott finished the Jarvis memory experiment.",
            )

            archived_document = service._store.read_document(closed.path)

            self.assertEqual(
                archived_document.summary,
                "Completed on 2026-03-13: Scott finished the Jarvis memory experiment.",
            )
            self.assertEqual(
                archived_document.sections["Summary"],
                "Completed on 2026-03-13: Scott finished the Jarvis memory experiment.",
            )
            self.assertEqual(
                archived_document.sections["Current State"],
                "Completed on 2026-03-13: Scott finished the Jarvis memory experiment.",
            )
            self.assertTrue(
                archived_document.sections["Notes"].startswith(
                    "System terminal rewrite fallback applied on "
                )
            )
            self.assertIn(
                "because no rewritten superseding content was provided.",
                archived_document.sections["Notes"],
            )
            self.assertFalse(any(issue.code == "closed_summary_present_tense" for issue in await service.integrity_check()))

    async def test_close_uses_explicit_transition_rewrite_instead_of_preserving_old_body(self) -> None:
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
                title="Three.js game project",
                summary="Scott is starting a new project to build a three.js game.",
                body_sections={
                    "Summary": "Scott is starting a new project to build a three.js game.",
                    "Current State": "- Project setup has started.",
                    "Open Loops": "- Build the first level.",
                    "Artifacts": "- Prototype notes.",
                    "Notes": "- Active project.",
                },
            )

            closed = await service.write(
                operation="close",
                target_kind="ongoing",
                document_id=created.document_id,
                summary="Scott completed the three.js game project.",
                body_sections={
                    "Current State": "- Project completed and no longer active.",
                    "Open Loops": "",
                    "Artifacts": "- Final build archived.",
                    "Notes": "- Closed after completion.",
                },
                close_reason="Completed. Scott said the three.js game has been finished.",
            )

            archived_document = service._store.read_document(closed.path)

            self.assertEqual(archived_document.summary, "Scott completed the three.js game project.")
            self.assertEqual(archived_document.sections["Summary"], "Scott completed the three.js game project.")
            self.assertEqual(archived_document.sections["Current State"], "- Project completed and no longer active.")
            self.assertEqual(archived_document.sections["Open Loops"], "")
            self.assertEqual(archived_document.sections["Artifacts"], "- Final build archived.")
            self.assertEqual(archived_document.sections["Notes"], "- Closed after completion.")
            self.assertNotIn("starting a new project", archived_document.body_markdown.lower())
            self.assertNotIn("build the first level", archived_document.body_markdown.lower())

    async def test_archive_without_rewrite_uses_generic_superseding_fallback(self) -> None:
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
                title="Frontend stack preference",
                summary="Scott currently prefers React for frontend work.",
                body_sections={
                    "Summary": "Scott currently prefers React for frontend work.",
                    "Details": "- React is the default choice.",
                    "Notes": "- Active preference.",
                },
            )

            archived = await service.write(
                operation="archive",
                target_kind="core",
                document_id=created.document_id,
                close_reason="Superseded by a newer frontend preference record.",
            )

            archived_document = service._store.read_document(archived.path)

            self.assertEqual(archived_document.status, "archived")
            self.assertEqual(archived_document.summary, "Superseded by a newer frontend preference record.")
            self.assertEqual(archived_document.sections["Summary"], "Superseded by a newer frontend preference record.")
            self.assertEqual(archived_document.sections["Details"], "Superseded by a newer frontend preference record.")
            self.assertIn(
                "System archive rewrite fallback applied on",
                archived_document.sections["Notes"],
            )
            self.assertNotIn("currently prefers React", archived_document.body_markdown)

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

    async def test_daily_upsert_requires_body_sections_for_corrections(self) -> None:
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
                title="Bike ride",
                summary="Scott went to Phoenix Park for a bike ride.",
            )

            with self.assertRaises(ValueError) as exc_info:
                await service.write(
                    operation="upsert",
                    target_kind="daily",
                    document_id=written.document_id,
                    summary="Scott went to Bray for a bike ride.",
                )

            self.assertIn("body_sections", str(exc_info.exception))
            self.assertIn("summary alone does not rewrite prior daily content", str(exc_info.exception))
            self.assertIn("append_daily", str(exc_info.exception))
            daily_text = written.path.read_text(encoding="utf-8")
            self.assertIn("Phoenix Park", daily_text)
            self.assertNotIn("Bray", daily_text)

    async def test_daily_upsert_rewrites_existing_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(
                settings=MemorySettings.from_workspace_dir(workspace_dir),
                llm_service=None,
            )

            created = await service.write(
                operation="create",
                target_kind="daily",
                date="2026-03-12",
                timezone_name="Europe/Dublin",
                body_sections={
                    "Notable Events": "- Scott went to Phoenix Park for a bike ride.",
                    "Decisions": "- Bring the gravel bike next time.",
                },
            )

            updated = await service.write(
                operation="upsert",
                target_kind="daily",
                document_id=created.document_id,
                body_sections={
                    "Notable Events": "- Scott went to Bray for a bike ride.",
                },
            )

            document = service._store.read_document(updated.path)
            self.assertEqual(
                document.sections["Notable Events"],
                "- Scott went to Bray for a bike ride.",
            )
            self.assertEqual(
                document.sections["Decisions"],
                "- Bring the gravel bike next time.",
            )
            self.assertNotIn("Phoenix Park", document.body_markdown)
            self.assertIn("Bray", document.body_markdown)

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
                if record.role == "system" and record.metadata.get("memory_bootstrap") == "core"
            ]

            self.assertEqual(len(memory_bootstrap_records), 1)
            self.assertIn("User Communication Style", memory_bootstrap_records[0].content)
