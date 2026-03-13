"""Focused tests for memory improvement pass 2."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import re
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

from llm import EmbeddingRequest, EmbeddingResponse, LLMResponse, LLMUsage
from memory import MemoryService, MemorySettings
from memory.bootstrap import render_core_bootstrap, render_ongoing_bootstrap
from storage.types import ConversationRecord


class _FakeEmbeddingReflectionLLM:
    def __init__(self, *, response_text: str = "{\"actions\": [{\"action\": \"ignore\", \"confidence\": \"high\"}]}") -> None:
        self.response_text = response_text
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return LLMResponse(
            provider="fake",
            model="fake-chat",
            text=self.response_text,
            tool_calls=[],
            finish_reason="stop",
            usage=LLMUsage(input_tokens=8, output_tokens=8, total_tokens=16),
            response_id="resp_fake",
        )

    async def embed(self, request: EmbeddingRequest):
        inputs = request.inputs if isinstance(request.inputs, list) else [request.inputs]
        embeddings = []
        for item in inputs:
            text = str(item)
            embeddings.append(
                [
                    float(len(text)),
                    float(text.lower().count("python") + text.lower().count("jarvis")),
                    float(text.lower().count("typescript") + text.lower().count("ts")),
                ]
            )
        return EmbeddingResponse(
            provider="fake",
            model="fake-embedding",
            embeddings=embeddings,
            usage=LLMUsage(input_tokens=4, output_tokens=0, total_tokens=4),
        )


class _SemanticTuningLLM(_FakeEmbeddingReflectionLLM):
    _TOKEN_GROUPS = {
        "graphics": 0,
        "renderer": 0,
        "render": 0,
        "shader": 0,
        "shaders": 0,
        "threejs": 0,
        "tsl": 0,
        "webgpu": 0,
        "hike": 1,
        "hikes": 1,
        "morning": 1,
        "routine": 1,
        "run": 1,
        "runs": 1,
        "archive": 2,
        "bootstrap": 2,
        "improvement": 2,
        "memory": 2,
        "retrieval": 2,
    }

    async def embed(self, request: EmbeddingRequest):
        inputs = request.inputs if isinstance(request.inputs, list) else [request.inputs]
        embeddings = [self._embed_text(str(item)) for item in inputs]
        return EmbeddingResponse(
            provider="fake",
            model="semantic-tuning",
            embeddings=embeddings,
            usage=LLMUsage(input_tokens=4, output_tokens=0, total_tokens=4),
        )

    @classmethod
    def _embed_text(cls, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vector = [0.0, 0.0, 0.0]
        for token in tokens:
            index = cls._TOKEN_GROUPS.get(token)
            if index is not None:
                vector[index] += 1.0
        return vector


class _FallbackPlanningLLM(_FakeEmbeddingReflectionLLM):
    async def embed(self, request: EmbeddingRequest):
        inputs = request.inputs if isinstance(request.inputs, list) else [request.inputs]
        embeddings = [self._embed_text(str(item)) for item in inputs]
        return EmbeddingResponse(
            provider="fake",
            model="fallback-planning",
            embeddings=embeddings,
            usage=LLMUsage(input_tokens=4, output_tokens=0, total_tokens=4),
        )

    @staticmethod
    def _embed_text(text: str) -> list[float]:
        normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
        if "graphics renderer" in normalized:
            return [1.0, 0.0, 0.0]
        if any(token in normalized for token in ("threejs", "webgpu", "tsl", "shader")):
            return [1.0, 0.0, 0.0]
        if any(token in normalized for token in ("morning", "run", "routine", "hike")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class _WeakSemanticLeakLLM(_FakeEmbeddingReflectionLLM):
    async def embed(self, request: EmbeddingRequest):
        inputs = request.inputs if isinstance(request.inputs, list) else [request.inputs]
        embeddings = [self._embed_text(str(item)) for item in inputs]
        return EmbeddingResponse(
            provider="fake",
            model="weak-semantic-leak",
            embeddings=embeddings,
            usage=LLMUsage(input_tokens=4, output_tokens=0, total_tokens=4),
        )

    @staticmethod
    def _embed_text(text: str) -> list[float]:
        normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
        if any(token in normalized for token in ("threejs", "webgpu", "shader", "playground")):
            return [1.0, 0.0]
        if "breakfast" in normalized and any(token in normalized for token in ("hike", "hiking")):
            return [0.08, 1.0]
        return [0.0, 1.0]


class MemoryPass2Tests(unittest.IsolatedAsyncioTestCase):
    async def test_natural_language_lexical_query_is_planned_for_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            created = await service.write(
                operation="create",
                target_kind="core",
                title="Backend Preferences",
                summary="Scott prefers Python and FastAPI for backend work.",
                body_sections={
                    "Summary": "Scott prefers Python and FastAPI for backend work.",
                    "Details": "- FastAPI is the default backend stack.",
                    "Notes": "",
                },
            )

            response = await service.search(
                query="What does Scott prefer for backend work?",
                mode="lexical",
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, created.document_id)
            self.assertIn("lexical_", response.results[0].match_reasons[0])

    async def test_punctuation_heavy_query_does_not_break_lexical_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            created = await service.write(
                operation="create",
                target_kind="core",
                title="Systems Preferences",
                summary="Scott likes Go and C for systems work.",
                body_sections={
                    "Summary": "Scott likes Go and C for systems work.",
                    "Details": "- Go for tools.\n- C for low-level code.",
                    "Notes": "",
                },
            )

            response = await service.search(
                query="What about Scott's Go/C preferences?",
                mode="lexical",
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, created.document_id)

    async def test_pinned_current_memory_outranks_stale_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            current = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Current Backend Project",
                summary="Jarvis currently uses Python services for backend work.",
                pinned=True,
            )
            stale = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Previous Backend Project",
                summary="Jarvis currently uses Python services for backend work.",
                pinned=False,
                review_after="2024-01-01T00:00:00+00:00",
            )
            stale_document = service._store.read_document(stale.path)
            stale_updated = replace(
                stale_document,
                updated_at="2024-01-01T00:00:00+00:00",
                review_after="2024-01-01T00:00:00+00:00",
            )
            await service._persist_document(
                stale_updated,
                allow_locked=True,
                existing_document=stale_document,
            )

            response = await service.search(
                query="python backend work",
                mode="hybrid",
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, current.document_id)

    async def test_graph_resolution_uses_full_sentence_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            created = await service.write(
                operation="create",
                target_kind="core",
                title="Scott TypeScript Preference",
                summary="Scott has a strong opinion about TypeScript.",
                entity_refs=[
                    {"name": "Scott", "entity_type": "person", "aliases": ["scott"]},
                    {"name": "TypeScript", "entity_type": "technology", "aliases": ["ts"]},
                ],
                relations=[
                    {
                        "subject": "Scott",
                        "predicate": "prefers",
                        "object": "TypeScript",
                        "status": "current",
                        "confidence": "high",
                        "cardinality": "single",
                    }
                ],
            )

            response = await service.search(
                query="What does Scott think about TypeScript?",
                mode="graph",
                scopes=("core",),
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, created.document_id)
            self.assertEqual(response.results[0].section_path, "relations")

    async def test_memory_get_defaults_to_body_and_logs_synthetic_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            created = await service.write(
                operation="create",
                target_kind="core",
                title="User Communication Style",
                summary="Scott prefers direct and concise engineering feedback.",
                facts=[{"text": "Scott prefers direct and concise engineering feedback."}],
                relations=[
                    {
                        "subject": "Scott",
                        "predicate": "prefers",
                        "object": "direct engineering feedback",
                    }
                ],
            )

            full_text = await service.get_document(document_id=created.document_id)
            facts_text = await service.get_document(
                document_id=created.document_id,
                section_path="facts",
                route_id="route_memory",
                session_id="session_memory",
            )
            relations_text = await service.get_document(
                document_id=created.document_id,
                section_path="relations",
                route_id="route_memory",
                session_id="session_memory",
            )

            self.assertTrue(full_text.startswith("# User Communication Style"))
            self.assertFalse(full_text.startswith("---"))
            self.assertIn("## Facts", facts_text)
            self.assertIn("## Relations", relations_text)

            with service._index_db._connect_main() as conn:
                rows = conn.execute(
                    """
                    select tool_name, chunk_id
                    from access_log
                    where tool_name = 'memory_get'
                    order by access_id desc
                    limit 2
                    """
                ).fetchall()
            self.assertEqual(str(rows[0]["chunk_id"]), f"relations:{created.document_id}")
            self.assertEqual(str(rows[1]["chunk_id"]), f"facts:{created.document_id}")

    async def test_memory_search_logs_real_chunk_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            created = await service.write(
                operation="create",
                target_kind="core",
                title="Development Preferences",
                summary="Scott wants blunt review findings before summaries.",
            )

            response = await service.search(
                query="blunt review findings",
                mode="lexical",
                route_id="route_search",
                session_id="session_search",
            )

            self.assertTrue(response.results)
            with service._index_db._connect_main() as conn:
                row = conn.execute(
                    """
                    select chunk_id
                    from access_log
                    where tool_name = 'memory_search'
                    order by access_id desc
                    limit 1
                    """
                ).fetchone()
            self.assertEqual(str(row["chunk_id"]), response.results[0].chunk_id)
            self.assertNotEqual(str(row["chunk_id"]), response.results[0].section_path)
            self.assertEqual(response.results[0].document_id, created.document_id)

    async def test_demote_archives_source_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            created = await service.write(
                operation="create",
                target_kind="core",
                title="Jarvis Memory Architecture",
                summary="The Jarvis memory architecture is being improved.",
            )

            demoted = await service.write(
                operation="demote",
                target_kind="ongoing",
                document_id=created.document_id,
            )

            self.assertFalse(created.path.exists())
            self.assertTrue(demoted.path.exists())
            archive_response = await service.search(
                query="Jarvis memory architecture",
                mode="lexical",
                scopes=("archive",),
            )
            self.assertTrue(archive_response.results)
            self.assertEqual(archive_response.results[0].document_id, created.document_id)

    async def test_daily_rollover_closes_previous_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            settings = MemorySettings.from_workspace_dir(workspace_dir)
            service = MemoryService(settings=settings, llm_service=None)

            today = datetime.now(ZoneInfo(settings.default_timezone)).date()
            previous_day = (today - timedelta(days=1)).isoformat()
            previous = await service.write(
                operation="append_daily",
                target_kind="daily",
                date=previous_day,
                timezone_name=settings.default_timezone,
                summary="Worked on memory improvements.",
            )

            runs = await service.run_due_maintenance()
            rollover = next(run for run in runs if run.job_name == "daily_rollover")
            previous_document = service._store.read_document(previous.path)
            today_document = service._store.read_document(settings.daily_dir / f"{today.isoformat()}.md")

            self.assertEqual(previous_document.status, "closed")
            self.assertEqual(rollover.summary["closed_documents"], 1)
            self.assertEqual(today_document.status, "active")

    async def test_priority_recompute_no_longer_drifts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            created = await service.write(
                operation="create",
                target_kind="core",
                title="Priority Drift Check",
                summary="Priority should stay stable.",
                priority=11,
            )

            runs = await service.run_due_maintenance()
            recompute = next(run for run in runs if run.job_name == "recompute_priority_from_usage")
            document = service._store.read_document(created.path)

            self.assertEqual(document.priority, 11)
            self.assertEqual(recompute.status, "skipped")

    async def test_bootstrap_includes_current_state_relations_and_freshness(self) -> None:
        reference_time = datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            core = await service.write(
                operation="create",
                target_kind="core",
                title="Scott TypeScript Preference",
                summary="Scott has a TypeScript preference.",
                relations=[
                    {
                        "subject": "Scott",
                        "predicate": "prefers",
                        "object": "TypeScript",
                    }
                ],
            )
            ongoing = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Jarvis Memory System",
                summary="The memory system is in improvement pass 2.",
                body_sections={
                    "Summary": "The memory system is in improvement pass 2.",
                    "Current State": "- Retrieval tuning is underway.",
                    "Open Loops": "- Finish pass 2 tests.",
                    "Artifacts": "",
                    "Notes": "",
                },
            )
            core_document = service._store.read_document(core.path)
            ongoing_document = service._store.read_document(ongoing.path)
            core_document = replace(core_document, updated_at="2026-03-11T12:00:00+00:00")
            ongoing_document = replace(ongoing_document, updated_at="2026-03-12T12:00:00+00:00")

            core_bootstrap = render_core_bootstrap((core_document,), token_budget=200, reference_time=reference_time)
            ongoing_bootstrap = render_ongoing_bootstrap((ongoing_document,), token_budget=200, reference_time=reference_time)

            self.assertIn("relation: Scott prefers TypeScript", core_bootstrap)
            self.assertIn("freshness: updated 2d ago", core_bootstrap)
            self.assertIn("current_state: - Retrieval tuning is underway.", ongoing_bootstrap)
            self.assertIn("open_loop: - Finish pass 2 tests.", ongoing_bootstrap)
            self.assertLessEqual(max(1, len(core_bootstrap) // 4), 200)
            self.assertLessEqual(max(1, len(ongoing_bootstrap) // 4), 200)

    async def test_reflection_prefers_update_over_duplicate_create(self) -> None:
        fake_llm = _FakeEmbeddingReflectionLLM(
            response_text=(
                "{\"actions\": [{\"action\": \"create_ongoing\", \"confidence\": \"high\", "
                "\"payload\": {\"title\": \"Jarvis Memory System\", \"summary\": \"Pass 2 is underway.\"}}]}"
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=fake_llm)

            created = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Jarvis Memory System",
                summary="Pass 1 already exists.",
            )

            applied = await service.reflect_completed_turn(
                route_id="route_reflect",
                session_id="session_reflect",
                records=(
                    ConversationRecord(
                        record_id="rec_1",
                        session_id="session_reflect",
                        created_at="2026-03-13T12:00:00+00:00",
                        role="user",
                        content="Continue the Jarvis memory system work.",
                    ),
                ),
            )

            documents = [
                document
                for document in service._store.read_all_documents()
                if document.kind == "ongoing" and document.status == "active" and not document.archived
            ]
            self.assertEqual(len(applied), 1)
            self.assertEqual(applied[0].document_id, created.document_id)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].summary, "Pass 2 is underway.")

    async def test_reflection_skips_locked_documents(self) -> None:
        fake_llm = _FakeEmbeddingReflectionLLM(
            response_text=(
                "{\"actions\": [{\"action\": \"update_ongoing\", \"confidence\": \"high\", "
                "\"payload\": {\"title\": \"Locked Memory\", \"summary\": \"Should not change.\"}}]}"
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=fake_llm)

            created = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Locked Memory",
                summary="Original summary.",
                locked=True,
            )

            applied = await service.reflect_completed_turn(
                route_id="route_reflect",
                session_id="session_reflect",
                records=(
                    ConversationRecord(
                        record_id="rec_1",
                        session_id="session_reflect",
                        created_at="2026-03-13T12:00:00+00:00",
                        role="user",
                        content="Update the locked memory automatically.",
                    ),
                ),
            )

            document = service._store.read_document(created.path)
            self.assertFalse(applied)
            self.assertEqual(document.summary, "Original summary.")

    async def test_maintenance_skips_locked_expired_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            created = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Locked Expiring Memory",
                summary="This should stay active because it is locked.",
                locked=True,
                expires_at="2024-01-01T00:00:00+00:00",
            )

            runs = await service.run_due_maintenance()
            expire_run = next(run for run in runs if run.job_name == "expire_due_ongoing")
            document = service._store.read_document(created.path)

            self.assertEqual(document.status, "active")
            self.assertEqual(expire_run.summary["changed_documents"], 0)

    async def test_candidate_counts_use_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            settings = replace(
                MemorySettings.from_workspace_dir(workspace_dir),
                lexical_candidate_count=1,
                semantic_candidate_count=1,
                graph_candidate_count=1,
                hybrid_result_count=1,
            )
            service = MemoryService(settings=settings, llm_service=None)

            for title in ("Alpha Preference", "Beta Preference", "Gamma Preference"):
                await service.write(
                    operation="create",
                    target_kind="core",
                    title=title,
                    summary="Scott prefers Python for backend work.",
                )

            response = await service.search(
                query="python backend work",
                mode="lexical",
                top_k=5,
            )

            self.assertEqual(len(response.results), 1)

    async def test_semantic_candidates_return_real_distance_when_available(self) -> None:
        fake_llm = _FakeEmbeddingReflectionLLM()
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=fake_llm)

            await service.write(
                operation="create",
                target_kind="core",
                title="Python Backend Preference",
                summary="Scott prefers Python for backend APIs.",
            )
            await service.write(
                operation="create",
                target_kind="core",
                title="TypeScript UI Preference",
                summary="Scott prefers TypeScript for frontend work.",
            )

            ready, reason = service._index_db.semantic_search_status()
            if not ready:
                self.skipTest(f"semantic search is unavailable in this runtime: {reason}")

            embedding_response = await fake_llm.embed(EmbeddingRequest(inputs="python backend"))
            rows = service._index_db.semantic_candidates(
                embedding=embedding_response.embeddings[0],
                scopes=("core",),
                include_expired=False,
                daily_lookback_days=30,
                limit=5,
            )
            hybrid_response = await service.search(query="python backend", mode="hybrid", scopes=("core",))

            self.assertGreaterEqual(len(rows), 2)
            self.assertLessEqual(float(rows[0]["distance"]), float(rows[1]["distance"]))
            self.assertTrue(any(float(row["distance"]) > 0.0 for row in rows[1:]))
            self.assertFalse(hybrid_response.semantic_disabled)

    async def test_lexical_search_does_not_report_semantic_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            await service.write(
                operation="create",
                target_kind="core",
                title="Backend Preferences",
                summary="Scott prefers Python for backend APIs.",
            )

            response = await service.search(
                query="python backend",
                mode="lexical",
                scopes=("core",),
            )

            self.assertTrue(response.results)
            self.assertFalse(response.semantic_disabled)

    async def test_truth_signals_are_populated_for_supported_and_conflicting_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=None)

            supporting_one = await service.write(
                operation="create",
                target_kind="core",
                title="Backend Language Preference",
                summary="Scott prefers Python for backend APIs.",
                facts=[{"text": "Scott prefers Python for backend APIs."}],
                relations=[
                    {
                        "subject": "Scott",
                        "predicate": "uses",
                        "object": "Python",
                        "status": "current",
                        "confidence": "high",
                        "cardinality": "multi",
                    }
                ],
            )
            supporting_two = await service.write(
                operation="create",
                target_kind="ongoing",
                title="Backend Stack Notes",
                summary="Scott prefers Python for backend APIs.",
                facts=[{"text": "Scott prefers Python for backend APIs."}],
                relations=[
                    {
                        "subject": "Scott",
                        "predicate": "uses",
                        "object": "Python",
                        "status": "current",
                        "confidence": "high",
                        "cardinality": "multi",
                    }
                ],
            )
            conflicting_locked = await service.write(
                operation="create",
                target_kind="core",
                title="Editor Preference Baseline",
                summary="Scott currently prefers Vim.",
                locked=True,
                relations=[
                    {
                        "subject": "Scott",
                        "predicate": "prefers_editor",
                        "object": "Vim",
                        "status": "current",
                        "confidence": "high",
                        "cardinality": "single",
                    }
                ],
            )
            conflicting_new = await service.write(
                operation="create",
                target_kind="core",
                title="Editor Preference Update",
                summary="Scott currently prefers Neovim.",
                relations=[
                    {
                        "subject": "Scott",
                        "predicate": "prefers_editor",
                        "object": "Neovim",
                        "status": "current",
                        "confidence": "high",
                        "cardinality": "single",
                    }
                ],
            )

            with service._index_db._connect_main() as conn:
                fact_rows = conn.execute(
                    """
                    select document_id, support_count, last_confirmed_at
                    from facts
                    where document_id in (?, ?)
                    order by document_id asc
                    """,
                    (supporting_one.document_id, supporting_two.document_id),
                ).fetchall()
                relation_rows = conn.execute(
                    """
                    select document_id, support_count, last_confirmed_at
                    from relations
                    where document_id in (?, ?)
                      and predicate = 'uses'
                    order by document_id asc
                    """,
                    (supporting_one.document_id, supporting_two.document_id),
                ).fetchall()
                document_rows = conn.execute(
                    """
                    select document_id, support_count, contradiction_count, last_confirmed_at
                    from documents
                    where document_id in (?, ?)
                    order by document_id asc
                    """,
                    (supporting_one.document_id, supporting_two.document_id),
                ).fetchall()
                conflict_rows = conn.execute(
                    """
                    select document_id, contradiction_count, last_contradicted_at
                    from relations
                    where document_id in (?, ?)
                      and predicate = 'prefers_editor'
                    order by document_id asc
                    """,
                    (conflicting_locked.document_id, conflicting_new.document_id),
                ).fetchall()
                conflicting_document_rows = conn.execute(
                    """
                    select document_id, contradiction_count, last_contradicted_at
                    from documents
                    where document_id in (?, ?)
                    order by document_id asc
                    """,
                    (conflicting_locked.document_id, conflicting_new.document_id),
                ).fetchall()

            self.assertEqual([int(row["support_count"]) for row in fact_rows], [1, 1])
            self.assertTrue(all(row["last_confirmed_at"] for row in fact_rows))
            self.assertEqual([int(row["support_count"]) for row in relation_rows], [1, 1])
            self.assertTrue(all(row["last_confirmed_at"] for row in relation_rows))
            self.assertTrue(all(int(row["support_count"]) >= 2 for row in document_rows))
            self.assertTrue(all(row["last_confirmed_at"] for row in document_rows))
            self.assertEqual([int(row["contradiction_count"]) for row in conflict_rows], [1, 1])
            self.assertTrue(all(row["last_contradicted_at"] for row in conflict_rows))
            self.assertTrue(all(int(row["contradiction_count"]) >= 1 for row in conflicting_document_rows))
            self.assertTrue(all(row["last_contradicted_at"] for row in conflicting_document_rows))

    async def test_hybrid_search_keeps_strong_semantic_hit_and_suppresses_weak_tail(self) -> None:
        fake_llm = _SemanticTuningLLM()
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=fake_llm)

            target = await service.write(
                operation="create",
                target_kind="core",
                title="Three.js WebGPU Playground",
                summary="Build a threejs webgpu tsl shader playground.",
            )
            await service.write(
                operation="create",
                target_kind="ongoing",
                title="Morning Run Routine",
                summary="Morning run and hike routine before breakfast.",
            )
            await service.write(
                operation="create",
                target_kind="core",
                title="Memory Retrieval Cleanup",
                summary="Memory retrieval and archive cleanup work.",
            )

            response = await service.search(
                query="graphics renderer thing",
                mode="hybrid",
                scopes=("core", "ongoing"),
                top_k=5,
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, target.document_id)
            self.assertEqual(len(response.results), 1)
            self.assertFalse(response.semantic_disabled)

    async def test_hybrid_retrieval_runs_bounded_fallback_for_noisy_query(self) -> None:
        fake_llm = _FallbackPlanningLLM()
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=fake_llm)

            target = await service.write(
                operation="create",
                target_kind="core",
                title="Visual Pipeline Project",
                summary="Build a threejs webgpu tsl shader playground.",
            )
            await service.write(
                operation="create",
                target_kind="ongoing",
                title="Morning Routine",
                summary="Morning run routine before breakfast.",
            )

            response = await service.search(
                query="What was the thing about graphics and renderer again?",
                mode="hybrid",
                scopes=("core", "ongoing"),
            )

            self.assertTrue(response.results)
            self.assertEqual(response.results[0].document_id, target.document_id)
            self.assertIn("retrieval_fallback", response.results[0].match_reasons)

    async def test_hybrid_search_drops_weak_semantic_only_junk_hit(self) -> None:
        fake_llm = _WeakSemanticLeakLLM()
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            service = MemoryService(settings=MemorySettings.from_workspace_dir(workspace_dir), llm_service=fake_llm)

            await service.write(
                operation="create",
                target_kind="ongoing",
                title="Visual Pipeline Project",
                summary="Build a threejs webgpu tsl shader playground.",
            )

            response = await service.search(
                query="breakfast hiking",
                mode="hybrid",
                scopes=("core", "ongoing"),
                top_k=5,
            )

            self.assertFalse(response.results)
            self.assertFalse(response.semantic_disabled)
