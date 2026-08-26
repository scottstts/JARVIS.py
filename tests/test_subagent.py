"""Unit tests for the subagent manager and storage behavior."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from jarvis.core import (
    AgentApprovalRequestEvent,
    AgentAssistantMessageEvent,
    AgentToolCallEvent,
    AgentTurnDoneEvent,
)
from jarvis.gateway.bash_job_supervisor import BashJobNotice
from jarvis.gateway.route_events import (
    RouteApprovalRequestEvent,
    RouteSystemNoticeEvent,
    RouteToolCallEvent,
)
from jarvis.llm import (
    DoneEvent,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderBadRequestError,
    TextDeltaEvent,
    TextPart,
)
from jarvis.storage import ConversationRecord, SessionStorage
from jarvis.subagent.manager import SubagentManager
from jarvis.subagent.runtime import SubagentRuntime
from jarvis.subagent.settings import SubagentSettings
from jarvis.subagent.storage import SubagentCatalogStorage
from jarvis.subagent.types import SubagentCatalogEntry
from tests.helpers import build_core_settings
from jarvis.tools import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolRegistry,
    ToolSettings,
    WorkspaceAccessCoordinator,
    WorkspaceLeaseError,
)
from jarvis.tools.basic.bash.jobs import claim_job_owner, create_background_job, load_job


def _build_response(text: str) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="fake-chat",
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        response_id="resp_fake",
    )


class _FakeSubagentLLMService:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return _build_response("done")

    async def stream_generate(self, request: LLMRequest):
        self.requests.append(request)
        yield TextDeltaEvent(delta="done")
        yield DoneEvent(response=_build_response("done"))


class _FakeSubagentLoop:
    def __init__(self, events: list[object], *, session_id: str = "subagent_session") -> None:
        self._events = tuple(events)
        self._session_id = session_id
        self.closed = False
        self.stop_requests = 0
        self.stop_reasons: list[str] = []
        self.hard_stop_requests = 0
        self.hard_stop_reasons: list[str] = []
        self.system_notes: list[tuple[str, str | None, dict[str, object] | None]] = []
        self.prepare_session_reasons: list[str] = []

    async def prepare_session(self, *, start_reason: str) -> str:
        self.prepare_session_reasons.append(start_reason)
        return self._session_id

    async def stream_turn(self, *, user_text: str, force_session_id: str | None, pre_turn_messages):
        _ = (user_text, force_session_id, pre_turn_messages)
        for event in self._events:
            yield event

    async def stream_runtime_turn(self, *, force_session_id: str | None, pre_turn_messages):
        _ = (force_session_id, pre_turn_messages)
        for event in self._events:
            yield event

    def active_session_id(self) -> str | None:
        return self._session_id

    def append_system_note(
        self,
        content: str,
        *,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> bool:
        self.system_notes.append((content, session_id, metadata))
        return True

    def request_stop(self, *, reason: str = "user_stop") -> bool:
        self.stop_requests += 1
        self.stop_reasons.append(reason)
        return True

    def request_hard_stop(self, *, reason: str = "new_session") -> bool:
        self.hard_stop_requests += 1
        self.hard_stop_reasons.append(reason)
        return True

    async def aclose(self) -> None:
        self.closed = True


class _WritingSubagentLoop(_FakeSubagentLoop):
    def __init__(self, workspace_dir: Path) -> None:
        super().__init__(
            [
                AgentTurnDoneEvent(
                    session_id="subagent_session",
                    response_text="done",
                )
            ]
        )
        self._workspace_dir = workspace_dir
        self.turn_count = 0

    async def stream_turn(self, *, user_text: str, force_session_id: str | None, pre_turn_messages):
        _ = (user_text, force_session_id, pre_turn_messages)
        self.turn_count += 1
        source_dir = self._workspace_dir / "src"
        if self.turn_count == 1:
            (source_dir / "modified.txt").write_text("after", encoding="utf-8")
            (source_dir / "deleted.txt").unlink()
            (source_dir / "created.txt").write_text("new", encoding="utf-8")
        else:
            (source_dir / "continued.txt").write_text("continued", encoding="utf-8")
        async for event in super().stream_turn(
            user_text=user_text,
            force_session_id=force_session_id,
            pre_turn_messages=pre_turn_messages,
        ):
            yield event


class _FailingSubagentLoop(_FakeSubagentLoop):
    async def stream_turn(self, *, user_text: str, force_session_id: str | None, pre_turn_messages):
        _ = (user_text, force_session_id, pre_turn_messages)
        if False:
            yield None
        raise ProviderBadRequestError(
            "provider rejected child request",
            metadata={"provider": "fake", "request_id": "req_child_1"},
        )


class SubagentSettingsTests(unittest.TestCase):
    def test_archive_dir_resolves_under_shared_transcript_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            settings = SubagentSettings.from_workspace_dir(workspace_dir)

        self.assertEqual(
            settings.archive_dir,
            workspace_dir / "archive" / "transcripts" / "subagents",
        )

    def test_reads_provider_override_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            with patch.dict(
                "os.environ",
                {"JARVIS_SUBAGENT_PROVIDER": "gemini"},
                clear=False,
            ):
                settings = SubagentSettings.from_workspace_dir(workspace_dir)

        self.assertEqual(settings.provider, "gemini")

    def test_reads_lmstudio_provider_override_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            with patch.dict(
                "os.environ",
                {"JARVIS_SUBAGENT_PROVIDER": "lmstudio"},
                clear=False,
            ):
                settings = SubagentSettings.from_workspace_dir(workspace_dir)

        self.assertEqual(settings.provider, "lmstudio")


class SubagentManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_restore_reconstitutes_only_current_main_lineage_and_pauses_in_flight_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            route_id = "route_restore"
            core_settings = build_core_settings(root_dir=Path(tmp))
            registry = ToolRegistry.default(
                ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            )
            manager = SubagentManager(
                route_id=route_id,
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=AsyncMock(),
                register_approval_target=lambda _approval_id, _loop: None,
            )

            main_storage = SessionStorage(core_settings.transcript_archive_dir)
            ancestor = main_storage.create_session(start_reason="initial")
            current = main_storage.create_session(
                parent_session_id=ancestor.session_id,
                start_reason="compaction",
            )
            loops: dict[str, _FakeSubagentLoop] = {}
            entry_specs = (
                (
                    "child_ancestor",
                    "Edith",
                    "completed",
                    ancestor.session_id,
                    None,
                ),
                (
                    "child_clean",
                    "Friday",
                    "paused",
                    current.session_id,
                    "process_shutdown",
                ),
                (
                    "child_interrupted",
                    "Karen",
                    "running",
                    current.session_id,
                    None,
                ),
            )
            for subagent_id, codename, status, owner_session_id, pause_reason in entry_specs:
                child_storage = manager._catalog.session_storage(
                    owner_main_session_id=owner_session_id,
                    subagent_id=subagent_id,
                )
                child_session = child_storage.create_session(
                    start_reason="subagent_initial"
                )
                loops[subagent_id] = _FakeSubagentLoop(
                    [],
                    session_id=child_session.session_id,
                )
                manager._catalog.create_entry(
                    SubagentCatalogEntry(
                        subagent_id=subagent_id,
                        codename=codename,
                        status=status,  # type: ignore[arg-type]
                        created_at="2026-08-25T12:00:00+00:00",
                        updated_at="2026-08-25T12:00:00+00:00",
                        route_id=route_id,
                        owner_main_session_id=owner_session_id,
                        owner_main_turn_id="main_turn",
                        current_subagent_session_id=child_session.session_id,
                        pause_reason=pause_reason,  # type: ignore[arg-type]
                    )
                )

            manager._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id="child_other_route_session",
                    codename="Ultron",
                    status="running",
                    created_at="2026-08-25T12:00:00+00:00",
                    updated_at="2026-08-25T12:00:00+00:00",
                    route_id=route_id,
                    owner_main_session_id="not-in-current-lineage",
                    owner_main_turn_id="old_turn",
                )
            )
            manager._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id="child_disposed",
                    codename="Homer",
                    status="disposed",
                    created_at="2026-08-25T12:00:00+00:00",
                    updated_at="2026-08-25T12:00:00+00:00",
                    route_id=route_id,
                    owner_main_session_id=current.session_id,
                    owner_main_turn_id="old_turn",
                )
            )

            with patch.object(
                manager,
                "_build_subagent_loop",
                side_effect=lambda **kwargs: loops[kwargs["subagent_id"]],
            ) as build_loop:
                await manager.restore(owner_main_session_id=current.session_id)
                await manager.restore(owner_main_session_id=current.session_id)

            self.assertEqual(build_loop.call_count, 3)
            self.assertCountEqual(
                [snapshot.subagent_id for snapshot in manager.active_snapshots()],
                ["child_ancestor", "child_clean", "child_interrupted"],
            )
            clean = manager._subagents["child_clean"]
            self.assertEqual(clean.status, "paused")
            self.assertEqual(clean.pause_reason, "process_shutdown")
            self.assertEqual(
                loops["child_clean"].prepare_session_reasons,
                ["subagent_recovery"],
            )
            interrupted = manager._subagents["child_interrupted"]
            self.assertEqual(interrupted.status, "paused")
            self.assertEqual(interrupted.pause_reason, "process_restart")
            self.assertFalse(interrupted.report_complete)
            self.assertTrue(
                any(
                    "ended before this subagent completed" in note[0]
                    for note in loops["child_interrupted"].system_notes
                )
            )
            self.assertTrue(
                any(
                    "graceful Jarvis process shutdown" in note[0]
                    for note in loops["child_clean"].system_notes
                )
            )
            interrupted_entry = manager._catalog.get_entry("child_interrupted")
            self.assertIsNotNone(interrupted_entry)
            if interrupted_entry is not None:
                self.assertEqual(interrupted_entry.status, "paused")
                self.assertEqual(interrupted_entry.pause_reason, "process_restart")

    async def test_restore_reacquires_held_lease_and_pauses_conflicting_in_flight_child(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            route_id = "route_restore_lease"
            core_settings = build_core_settings(root_dir=Path(tmp))
            (core_settings.workspace_dir / "src").mkdir()
            workspace_access = WorkspaceAccessCoordinator(
                workspace_dir=core_settings.workspace_dir
            )
            manager = SubagentManager(
                route_id=route_id,
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=ToolRegistry.default(
                    ToolSettings.from_workspace_dir(core_settings.workspace_dir)
                ),
                tool_execution_guard=asyncio.Semaphore(1),
                workspace_access=workspace_access,
                publish_event=AsyncMock(),
                register_approval_target=lambda _approval_id, _loop: None,
            )
            main_session = SessionStorage(
                core_settings.transcript_archive_dir
            ).create_session(start_reason="initial")
            loops: dict[str, _FakeSubagentLoop] = {}
            for subagent_id, status in (
                ("child_a", "paused"),
                ("child_b", "running"),
            ):
                child_storage = manager._catalog.session_storage(
                    owner_main_session_id=main_session.session_id,
                    subagent_id=subagent_id,
                )
                child_session = child_storage.create_session(
                    start_reason="subagent_initial"
                )
                loops[subagent_id] = _FakeSubagentLoop(
                    [],
                    session_id=child_session.session_id,
                )
                manager._catalog.create_entry(
                    SubagentCatalogEntry(
                        subagent_id=subagent_id,
                        codename="Friday" if subagent_id == "child_a" else "Edith",
                        status=status,  # type: ignore[arg-type]
                        created_at="2026-08-25T12:00:00+00:00",
                        updated_at="2026-08-25T12:00:00+00:00",
                        route_id=route_id,
                        owner_main_session_id=main_session.session_id,
                        owner_main_turn_id="main_turn",
                        owned_paths=("src",),
                        workspace_lease_status="held",
                        current_subagent_session_id=child_session.session_id,
                        pause_reason=(
                            "process_shutdown" if subagent_id == "child_a" else None
                        ),
                    )
                )

            with patch.object(
                manager,
                "_build_subagent_loop",
                side_effect=lambda **kwargs: loops[kwargs["subagent_id"]],
            ):
                await manager.restore(owner_main_session_id=main_session.session_id)

            first = manager._subagents["child_a"]
            second = manager._subagents["child_b"]
            self.assertEqual(first.workspace_lease_status, "held")
            self.assertEqual(second.workspace_lease_status, "released")
            self.assertEqual(second.status, "paused")
            self.assertEqual(second.pause_reason, "external_blocked")
            second_entry = manager._catalog.get_entry("child_b")
            self.assertIsNotNone(second_entry)
            if second_entry is not None:
                self.assertEqual(second_entry.workspace_lease_status, "released")
                self.assertEqual(second_entry.pause_reason, "external_blocked")
            with self.assertRaises(WorkspaceLeaseError):
                await workspace_access.claim_paths(owner="main", paths=("src",))

    async def test_build_main_progress_message_includes_latest_subagent_report_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)

            async def publish_event(_event: object) -> None:
                return None

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )

            storage = manager._catalog.session_storage(
                owner_main_session_id="main_session",
                subagent_id="sub_1",
            )
            session = storage.create_session(start_reason="subagent_initial")
            storage.append_record(
                session.session_id,
                ConversationRecord(
                    record_id="assistant_record",
                    session_id=session.session_id,
                    created_at="2026-04-07T00:00:00+00:00",
                    role="assistant",
                    content="one\ntwo\nUsed bash.",
                    metadata={"turn_id": "turn_1"},
                ),
            )
            storage.set_turn_status(session.session_id, turn_id="turn_1", status="completed")
            fake_loop = _FakeSubagentLoop([], session_id=session.session_id)
            manager._subagents["sub_1"] = SubagentRuntime(
                subagent_id="sub_1",
                codename="Friday",
                loop=fake_loop,  # type: ignore[arg-type]
                storage=storage,
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
                status="completed",
                created_at="2026-04-07T00:00:00+00:00",
                updated_at="2026-04-07T00:00:00+00:00",
                task_label="Inspect workspace",
                report_complete=True,
            )

            payload = manager.build_main_progress_message(
                agent="sub_1",
                notice_kind="subagent_completed",
                notice_text="completed.",
            )

            self.assertIsNotNone(payload)
            if payload is None:
                self.fail("Expected a progress payload.")
            _session_id, message = payload
            self.assertIn("Complete subagent report:", message.content)
            self.assertIn("one\ntwo\nUsed bash.", message.content)
            self.assertIn(
                "The report is self-reported completion, not semantic acceptance.",
                message.content,
            )
            self.assertIn(
                "inspect the producer's actual changes, its boundary, and its consumers",
                message.content,
            )
            self.assertIn("recommendation=inspect", message.content)
            self.assertEqual(message.metadata["latest_subagent_report_included"], True)
            self.assertEqual(message.metadata["latest_subagent_report_complete"], True)
            self.assertEqual(message.metadata["coordination_nudge"], "upstream_available")

    async def test_paused_subagent_checkpoint_requires_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            registry = ToolRegistry.default(
                ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            )

            async def publish_event(_event: object) -> None:
                return None

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )
            storage = manager._catalog.session_storage(
                owner_main_session_id="main_session",
                subagent_id="sub_paused",
            )
            manager._subagents["sub_paused"] = SubagentRuntime(
                subagent_id="sub_paused",
                codename="Friday",
                loop=_FakeSubagentLoop([], session_id="paused_session"),  # type: ignore[arg-type]
                storage=storage,
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
                status="paused",
                created_at="2026-04-07T00:00:00+00:00",
                updated_at="2026-04-07T00:00:00+00:00",
                task_label="Partial review",
                latest_report="I inspected the schema but did not run validation.",
                report_complete=False,
                pause_reason="main_stop",
            )

            payload = manager.build_main_progress_message(
                agent="sub_paused",
                notice_kind="subagent_paused",
                notice_text="paused.",
            )

            self.assertIsNotNone(payload)
            if payload is None:
                self.fail("Expected paused progress payload.")
            _session_id, message = payload
            self.assertEqual(message.metadata["recommended_action"], "inspect")
            self.assertIn("Latest subagent checkpoint:", message.content)
            self.assertIn("incomplete or truncated", message.content)

    async def test_main_status_snapshot_is_suppressed_until_subagent_state_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            registry = ToolRegistry.default(
                ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            )

            async def publish_event(_event: object) -> None:
                return None

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )
            runtime = SubagentRuntime(
                subagent_id="sub_1",
                codename="Friday",
                loop=_FakeSubagentLoop([], session_id="running_session"),  # type: ignore[arg-type]
                storage=manager._catalog.session_storage(
                    owner_main_session_id="main_session",
                    subagent_id="sub_1",
                ),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
                status="running",
                created_at="2026-04-07T00:00:00+00:00",
                updated_at="2026-04-07T00:00:00+00:00",
                task_label="Inspect runtime",
            )
            manager._subagents[runtime.subagent_id] = runtime

            first = manager.main_turn_runtime_messages(session_id="main_session")
            unchanged = manager.main_turn_runtime_messages(session_id="main_session")
            compacted_session = manager.main_turn_runtime_messages(
                session_id="compacted_session"
            )
            manager._append_notable_event(
                runtime,
                kind="checkpoint",
                summary="Inspected the runtime interface.",
            )
            changed = manager.main_turn_runtime_messages(session_id="compacted_session")

            self.assertEqual(len(first), 1)
            self.assertIn("Inspect runtime", first[0].content)
            self.assertNotIn("coordination_nudge", first[0].metadata)
            self.assertEqual(unchanged, ())
            self.assertEqual(len(compacted_session), 1)
            self.assertEqual(len(changed), 1)
            self.assertIn("Inspected the runtime interface.", changed[0].content)

    async def test_coordination_fanout_nudge_is_advisory_and_not_for_unrelated_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=ToolRegistry.default(
                    ToolSettings.from_workspace_dir(core_settings.workspace_dir)
                ),
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=AsyncMock(),
                register_approval_target=lambda _approval_id, _loop: None,
            )

            first = await manager.invoke(
                requester_kind="main",
                task_label="Independent one",
                instructions="Complete the first independent task.",
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )
            second = await manager.invoke(
                requester_kind="main",
                task_label="Independent two",
                instructions="Complete the second independent task.",
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )

            self.assertNotIn("coordination_nudge", first)
            self.assertNotIn("coordination_nudge", second)

            related = await manager.invoke(
                requester_kind="main",
                task_label="Dependent work",
                instructions="Build on the completed foundation.",
                phase="feature",
                depends_on=(first["subagent_id"],),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )
            self.assertEqual(related["coordination_nudge_kind"], "before_meaningful_fanout")
            self.assertIn("surface missing canonical dependencies", related["coordination_nudge"])

            fourth = await manager.invoke(
                requester_kind="main",
                task_label="Another dependent task",
                instructions="Continue the dependent work.",
                phase="feature",
                depends_on=(second["subagent_id"],),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )
            self.assertNotIn("coordination_nudge", fourth)

            for payload in (first, second, related, fourth):
                runtime = manager._subagents[payload["subagent_id"]]
                if runtime.task is not None:
                    await asyncio.wait_for(runtime.task, timeout=1)

    async def test_final_coordination_nudge_is_emitted_once_after_delegated_work_settles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=ToolRegistry.default(
                    ToolSettings.from_workspace_dir(core_settings.workspace_dir)
                ),
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=AsyncMock(),
                register_approval_target=lambda _approval_id, _loop: None,
            )
            runtime = SubagentRuntime(
                subagent_id="sub_completed",
                codename="Friday",
                loop=_FakeSubagentLoop([], session_id="child_session"),  # type: ignore[arg-type]
                storage=manager._catalog.session_storage(
                    owner_main_session_id="main_session",
                    subagent_id="sub_completed",
                ),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
                status="completed",
                created_at="2026-04-07T00:00:00+00:00",
                updated_at="2026-04-07T00:00:00+00:00",
                task_label="Completed foundation",
                latest_report="The foundation is locally verified.",
                report_complete=True,
            )
            manager._subagents[runtime.subagent_id] = runtime

            first = manager.main_turn_runtime_messages(session_id="main_session")
            second = manager.main_turn_runtime_messages(session_id="main_session")

            self.assertEqual(len(first), 1)
            self.assertEqual(first[0].metadata["coordination_nudge"], "before_final_completion")
            self.assertIn("Review the assembled result", first[0].content)
            self.assertEqual(second, ())

    async def test_invoke_returns_session_id_and_catalog_owner_linkage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)
            published_events: list[object] = []

            async def publish_event(event: object) -> None:
                published_events.append(event)

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )

            payload = await manager.invoke(
                requester_kind="main",
                task_label="Inspect workspace",
                instructions="Inspect the workspace and report back.",
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )

            self.assertTrue(payload["session_id"])
            runtime = manager._subagents[payload["subagent_id"]]
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)

            subagent_settings = SubagentSettings.from_workspace_dir(
                core_settings.workspace_dir,
                transcript_archive_root=core_settings.transcript_archive_dir,
            )
            catalog = SubagentCatalogStorage(
                archive_dir=subagent_settings.archive_dir,
                route_id="route_1",
            )
            entry = catalog.get_entry(payload["subagent_id"])

            self.assertIsNotNone(entry)
            if entry is None:
                self.fail("Expected subagent catalog entry to exist.")
            self.assertEqual(entry.owner_main_session_id, "main_session")
            self.assertEqual(entry.owner_main_turn_id, "main_turn")
            self.assertEqual(entry.task_label, "Inspect workspace")
            self.assertEqual(entry.current_subagent_session_id, payload["session_id"])
            self.assertEqual(entry.status, "completed")
            self.assertTrue(
                (
                    subagent_settings.archive_dir
                    / "route_1"
                    / "main_session"
                    / payload["subagent_id"]
                    / "sessions_index.json"
                ).exists()
            )
            self.assertTrue(
                (
                    subagent_settings.archive_dir
                    / "route_1"
                    / "main_session"
                    / payload["subagent_id"]
                    / "sessions"
                    / f"{payload['session_id']}.jsonl"
                ).exists()
            )

            self.assertEqual(len(published_events), 2)
            self.assertIsInstance(published_events[0], RouteSystemNoticeEvent)
            notice = published_events[0]
            if not isinstance(notice, RouteSystemNoticeEvent):
                self.fail("Expected invoke notice to be a route system notice.")
            self.assertEqual(notice.session_id, payload["session_id"])
            self.assertEqual(notice.notice_kind, "subagent_invoked")
            self.assertTrue(notice.public)
            completion_notice = published_events[1]
            self.assertIsInstance(completion_notice, RouteSystemNoticeEvent)
            if not isinstance(completion_notice, RouteSystemNoticeEvent):
                self.fail("Expected completion notice to be a route system notice.")
            self.assertEqual(completion_notice.notice_kind, "subagent_completed")
            self.assertEqual(completion_notice.text, "completed.")
            self.assertFalse(completion_notice.public)

    async def test_invoke_forwards_structured_context_and_selected_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            skill_dir = core_settings.workspace_dir / "skills" / "review-check"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: Review Check\n"
                "description: Enforce the selected review protocol.\n"
                "---\n\n"
                "Always include SKILL_FORWARDING_SENTINEL in the review.\n",
                encoding="utf-8",
            )
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)
            llm_service = _FakeSubagentLLMService()

            async def publish_event(_event: object) -> None:
                return None

            manager = SubagentManager(
                route_id="route_1",
                llm_service=llm_service,
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )

            payload = await manager.invoke(
                requester_kind="main",
                task_label="Review storage contract",
                instructions="Review the storage boundary.",
                user_constraints="Do not edit production files.",
                shared_context="SessionStorage is the shared interface.",
                owned_paths=("tests/test_storage.py",),
                skill_ids=("review-check",),
                phase="review",
                depends_on=("storage-foundation",),
                seam_contract=(
                    "Consumes the storage interface; provides an evidence-backed review; "
                    "does not edit production files."
                ),
                deliverable="A concise evidence-backed review.",
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )
            runtime = manager._subagents[payload["subagent_id"]]
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)

            bootstrap_text = "\n".join(
                part.text
                for message in llm_service.requests[0].messages
                for part in message.parts
                if isinstance(part, TextPart)
            )
            self.assertIn("task_label: Review storage contract", bootstrap_text)
            self.assertIn("Do not edit production files.", bootstrap_text)
            self.assertIn("SessionStorage is the shared interface.", bootstrap_text)
            self.assertIn("- tests/test_storage.py", bootstrap_text)
            self.assertIn("phase: review", bootstrap_text)
            self.assertIn("- storage-foundation", bootstrap_text)
            self.assertIn("Consumes the storage interface", bootstrap_text)
            self.assertIn("--- BEGIN SKILL review-check ---", bootstrap_text)
            self.assertIn("SKILL_FORWARDING_SENTINEL", bootstrap_text)
            self.assertEqual(payload["skill_ids"], ["review-check"])
            self.assertEqual(payload["skill_selection_reason"], "main_selected")

            entry = manager._catalog.get_entry(payload["subagent_id"])
            self.assertIsNotNone(entry)
            if entry is None:
                self.fail("Expected structured assignment to be persisted.")
            self.assertEqual(entry.task_label, "Review storage contract")
            self.assertEqual(entry.skill_ids, ("review-check",))
            self.assertEqual(entry.skill_selection_reason, "main_selected")
            self.assertEqual(entry.owned_paths, ("tests/test_storage.py",))
            self.assertEqual(entry.phase, "review")
            self.assertEqual(entry.depends_on, ("storage-foundation",))
            self.assertIn("Consumes the storage interface", entry.seam_contract or "")

    async def test_invoke_never_infers_skill_from_assignment_text_or_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            skill_dir = core_settings.workspace_dir / "skills" / "threejs-procedural-fields"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: threejs-procedural-fields\n"
                "description: Build coherent procedural scalar and vector fields for Three.js "
                "materials, terrain, and geometry.\n"
                "---\n",
                encoding="utf-8",
            )
            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=ToolRegistry.default(
                    ToolSettings.from_workspace_dir(core_settings.workspace_dir)
                ),
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=AsyncMock(),
                register_approval_target=lambda _approval_id, _loop: None,
            )

            payload = await manager.invoke(
                requester_kind="main",
                task_label="Build procedural fields",
                instructions=(
                    "Read skills/threejs-procedural-fields/SKILL.md and its references, then "
                    "build a Three.js procedural terrain."
                ),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )
            runtime = manager._subagents[payload["subagent_id"]]
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)

            bootstrap_text = "\n".join(
                part.text
                for request in manager._llm_service.requests
                for message in request.messages
                for part in message.parts
                if isinstance(part, TextPart)
            )
            self.assertEqual(payload["skill_ids"], [])
            self.assertEqual(
                payload["skill_selection_reason"],
                "none:not_selected_by_main",
            )
            self.assertNotIn("--- BEGIN SKILL threejs-procedural-fields ---", bootstrap_text)

    async def test_changed_child_tests_are_exposed_to_main_monitor_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=ToolRegistry.default(
                    ToolSettings.from_workspace_dir(core_settings.workspace_dir)
                ),
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=AsyncMock(),
                register_approval_target=lambda _approval_id, _loop: None,
            )
            payload = await manager.invoke(
                requester_kind="main",
                task_label="Build vehicle tests",
                instructions="Implement the vehicle subsystem and its tests.",
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )
            runtime = manager._subagents[payload["subagent_id"]]
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)

            await manager._observe_tool_result(
                subagent_id=runtime.subagent_id,
                result=ToolExecutionResult(
                    call_id="write-tests",
                    name="file_write",
                    ok=True,
                    content="changed",
                    metadata={
                        "changed": True,
                        "path": "/workspace/src/vehicles/vehicle.ts",
                        "workspace_changed_paths": [
                            "/workspace/src/vehicles/__tests__/vehicle.test.ts",
                        ],
                    },
                ),
                context=ToolExecutionContext(
                    workspace_dir=core_settings.workspace_dir,
                    agent_kind="subagent",
                    subagent_id=runtime.subagent_id,
                ),
            )

            monitor = await manager.monitor(agent=runtime.subagent_id)
            snapshot = monitor["subagents"][0]
            self.assertEqual(
                snapshot["changed_test_artifact_paths"],
                ["src/vehicles/__tests__/vehicle.test.ts"],
            )
            self.assertEqual(
                snapshot["changed_paths"],
                [
                    "src/vehicles/__tests__/vehicle.test.ts",
                    "src/vehicles/vehicle.ts",
                ],
            )
            progress = manager.build_main_progress_message(
                agent=runtime.subagent_id,
                notice_kind="subagent_completed",
                notice_text="completed.",
            )
            self.assertIsNotNone(progress)
            if progress is None:
                self.fail("Expected a main progress message.")
            _session_id, message = progress
            self.assertEqual(
                message.metadata["changed_test_artifact_paths"],
                ["src/vehicles/__tests__/vehicle.test.ts"],
            )
            self.assertEqual(
                message.metadata["changed_paths"],
                [
                    "src/vehicles/__tests__/vehicle.test.ts",
                    "src/vehicles/vehicle.ts",
                ],
            )
            self.assertIn("Subagent changed test artifacts", message.content)
            self.assertIn("Subagent changed paths", message.content)

    async def test_scoped_snapshot_reports_arbitrary_writes_and_resets_after_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            source_dir = core_settings.workspace_dir / "src"
            source_dir.mkdir(parents=True)
            (source_dir / "modified.txt").write_text("before", encoding="utf-8")
            (source_dir / "deleted.txt").write_text("delete me", encoding="utf-8")

            writing_loop = _WritingSubagentLoop(core_settings.workspace_dir)
            workspace_access = WorkspaceAccessCoordinator(
                workspace_dir=core_settings.workspace_dir
            )
            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=ToolRegistry.default(
                    ToolSettings.from_workspace_dir(core_settings.workspace_dir)
                ),
                tool_execution_guard=asyncio.Semaphore(1),
                workspace_access=workspace_access,
                publish_event=AsyncMock(),
                register_approval_target=lambda _approval_id, _loop: None,
            )

            with patch.object(manager, "_build_subagent_loop", return_value=writing_loop):
                payload = await manager.invoke(
                    requester_kind="main",
                    task_label="Track arbitrary writes",
                    instructions="Implement the assigned change and report what changed.",
                    owned_paths=("src",),
                    owner_main_session_id="main_session",
                    owner_main_turn_id="main_turn",
                )

            runtime = manager._subagents[payload["subagent_id"]]
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)
            self.assertEqual(
                runtime.changed_paths,
                {
                    "src/created.txt",
                    "src/deleted.txt",
                    "src/modified.txt",
                },
            )
            self.assertTrue(runtime.changed_paths_complete)
            self.assertEqual(runtime.changed_paths_source, "scoped_workspace_snapshot")

            handoff = await manager.handoff(agent=runtime.subagent_id)
            self.assertTrue(handoff["handoff_ready"])
            self.assertEqual(
                handoff["changed_paths"],
                [
                    "src/created.txt",
                    "src/deleted.txt",
                    "src/modified.txt",
                ],
            )
            self.assertTrue(handoff["changed_paths_complete"])
            self.assertEqual(
                handoff["changed_paths_source"],
                "scoped_workspace_snapshot",
            )
            await workspace_access.claim_paths(owner="main", paths=("src",))
            (source_dir / "main-integration.txt").write_text("main", encoding="utf-8")
            await workspace_access.release_owner(owner="main")

            await manager.step_in(
                agent=runtime.subagent_id,
                instructions="Continue from the integrated workspace state.",
            )
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)

            self.assertIn("src/continued.txt", runtime.changed_paths)
            self.assertNotIn("src/main-integration.txt", runtime.changed_paths)
            self.assertTrue(runtime.changed_paths_complete)
            self.assertEqual(runtime.changed_paths_source, "scoped_workspace_snapshot")
            await manager.dispose(agent=runtime.subagent_id)

    async def test_handoff_releases_lease_and_step_in_reacquires_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            owned_dir = core_settings.workspace_dir / "src"
            owned_dir.mkdir(parents=True)
            (owned_dir / "contract.txt").write_text("contract", encoding="utf-8")
            workspace_access = WorkspaceAccessCoordinator(
                workspace_dir=core_settings.workspace_dir
            )
            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=ToolRegistry.default(
                    ToolSettings.from_workspace_dir(core_settings.workspace_dir)
                ),
                tool_execution_guard=asyncio.Semaphore(1),
                workspace_access=workspace_access,
                publish_event=AsyncMock(),
                register_approval_target=lambda _approval_id, _loop: None,
            )

            payload = await manager.invoke(
                requester_kind="main",
                task_label="Implement the contract",
                instructions="Make the assigned change and report what was verified.",
                owned_paths=("src",),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )
            runtime = manager._subagents[payload["subagent_id"]]
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)
            self.assertEqual(runtime.status, "completed")
            self.assertEqual(runtime.workspace_lease_status, "held")

            with self.assertRaises(WorkspaceLeaseError):
                await workspace_access.claim_paths(owner="main", paths=("src",))

            handoff = await manager.handoff(agent=payload["subagent_id"])
            self.assertTrue(handoff["handoff_ready"])
            self.assertEqual(handoff["workspace_lease_status"], "released")
            self.assertEqual(runtime.workspace_lease_status, "released")
            entry = manager._catalog.get_entry(payload["subagent_id"])
            self.assertIsNotNone(entry)
            if entry is not None:
                self.assertEqual(entry.workspace_lease_status, "released")
            await workspace_access.claim_paths(owner="main", paths=("src",))
            await workspace_access.release_owner(owner="main")

            await manager.step_in(
                agent=payload["subagent_id"],
                instructions="Recheck the contract and report the result.",
            )
            self.assertEqual(runtime.workspace_lease_status, "held")
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)
            self.assertEqual(runtime.status, "completed")
            entry = manager._catalog.get_entry(payload["subagent_id"])
            self.assertIsNotNone(entry)
            if entry is not None:
                self.assertEqual(entry.workspace_lease_status, "held")
            await manager.dispose(agent=payload["subagent_id"])

    async def test_subagent_failure_persists_provider_metadata_and_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)

            async def publish_event(_event: object) -> None:
                return None

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )
            failing_loop = _FailingSubagentLoop([], session_id="failed_session")

            with patch.object(manager, "_build_subagent_loop", return_value=failing_loop):
                payload = await manager.invoke(
                    requester_kind="main",
                    task_label="Fail visibly",
                    instructions="Exercise provider error reporting.",
                    owner_main_session_id="main_session",
                    owner_main_turn_id="main_turn",
                )

            runtime = manager._subagents[payload["subagent_id"]]
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)

            self.assertEqual(runtime.status, "failed")
            self.assertEqual(runtime.last_error_metadata["request_id"], "req_child_1")
            self.assertIsNotNone(runtime.error_log_path)
            if runtime.error_log_path is None:
                self.fail("Expected a child error log path.")
            error_log = Path(runtime.error_log_path)
            self.assertTrue(error_log.exists())
            error_text = error_log.read_text(encoding="utf-8")
            self.assertIn("ProviderBadRequestError", error_text)
            self.assertIn("req_child_1", error_text)
            self.assertIn("traceback", error_text)

            monitored = await manager.monitor(agent=payload["subagent_id"], detail="full")
            snapshot = monitored["subagents"][0]
            self.assertEqual(snapshot["last_error_metadata"]["provider"], "fake")
            self.assertEqual(snapshot["error_log_path"], str(error_log))
            entry = manager._catalog.get_entry(payload["subagent_id"])
            self.assertIsNotNone(entry)
            if entry is not None:
                self.assertEqual(entry.last_error_metadata["request_id"], "req_child_1")

    async def test_subagent_publishes_resume_and_completion_notices_after_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)
            published_events: list[object] = []
            registered_approvals: list[str] = []

            async def publish_event(event: object) -> None:
                published_events.append(event)

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda approval_id, _loop: registered_approvals.append(
                    approval_id
                ),
            )

            fake_loop = _FakeSubagentLoop(
                [
                    AgentApprovalRequestEvent(
                        session_id="subagent_session",
                        approval_id="approval_1",
                        kind="bash_command",
                        summary="Install a dependency.",
                        details="Need one install before continuing.",
                        command="apt-get install xz-utils",
                        tool_name="bash",
                        inspection_url="https://example.com/xz-utils",
                    ),
                    AgentAssistantMessageEvent(
                        session_id="subagent_session",
                        text="Resumed after approval.",
                    ),
                    AgentToolCallEvent(
                        session_id="subagent_session",
                        tool_names=("bash",),
                        turn_id="sub_turn_1",
                    ),
                    AgentTurnDoneEvent(
                        session_id="subagent_session",
                        response_text="done",
                    ),
                ]
            )

            with patch.object(manager, "_build_subagent_loop", return_value=fake_loop):
                payload = await manager.invoke(
                    requester_kind="main",
                    task_label="Approval workflow",
                    instructions="Do the task.",
                    owner_main_session_id="main_session",
                    owner_main_turn_id="main_turn",
                )

            runtime = manager._subagents[payload["subagent_id"]]
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)

            self.assertEqual(runtime.status, "completed")
            self.assertEqual(registered_approvals, ["approval_1"])

            self.assertEqual(
                [
                    event.notice_kind
                    for event in published_events
                    if isinstance(event, RouteSystemNoticeEvent)
                ],
                [
                    "subagent_invoked",
                    "subagent_resumed",
                    "subagent_completed",
                ],
            )
            self.assertEqual(
                [
                    event.text
                    for event in published_events
                    if isinstance(event, RouteSystemNoticeEvent)
                ][1:],
                [
                    "resumed after approval.",
                    "completed.",
                ],
            )
            self.assertTrue(
                any(
                    isinstance(event, RouteApprovalRequestEvent)
                    and event.approval_id == "approval_1"
                    for event in published_events
                )
            )
            self.assertTrue(
                any(
                    isinstance(event, RouteToolCallEvent)
                    and event.tool_names == ("bash",)
                    and event.turn_id == "sub_turn_1"
                    for event in published_events
                )
            )

    async def test_request_stop_all_for_user_stop_targets_only_active_subagents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)

            async def publish_event(_event: object) -> None:
                return None

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )

            running_loop = _FakeSubagentLoop([], session_id="running_session")
            paused_loop = _FakeSubagentLoop([], session_id="paused_session")
            awaiting_loop = _FakeSubagentLoop([], session_id="awaiting_session")
            completed_loop = _FakeSubagentLoop([], session_id="completed_session")

            manager._subagents = {
                "sub_running": SubagentRuntime(
                    subagent_id="sub_running",
                    codename="Friday",
                    loop=running_loop,  # type: ignore[arg-type]
                    storage=manager._catalog.session_storage(
                        owner_main_session_id="main_session",
                        subagent_id="sub_running",
                    ),
                    owner_main_session_id="main_session",
                    owner_main_turn_id="main_turn",
                    status="running",
                    created_at="2026-03-19T12:00:00+00:00",
                    updated_at="2026-03-19T12:00:00+00:00",
                ),
                "sub_paused": SubagentRuntime(
                    subagent_id="sub_paused",
                    codename="Karen",
                    loop=paused_loop,  # type: ignore[arg-type]
                    storage=manager._catalog.session_storage(
                        owner_main_session_id="main_session",
                        subagent_id="sub_paused",
                    ),
                    owner_main_session_id="main_session",
                    owner_main_turn_id="main_turn",
                    status="paused",
                    created_at="2026-03-19T12:00:00+00:00",
                    updated_at="2026-03-19T12:00:00+00:00",
                ),
                "sub_awaiting": SubagentRuntime(
                    subagent_id="sub_awaiting",
                    codename="Ultron",
                    loop=awaiting_loop,  # type: ignore[arg-type]
                    storage=manager._catalog.session_storage(
                        owner_main_session_id="main_session",
                        subagent_id="sub_awaiting",
                    ),
                    owner_main_session_id="main_session",
                    owner_main_turn_id="main_turn",
                    status="awaiting_approval",
                    created_at="2026-03-19T12:00:00+00:00",
                    updated_at="2026-03-19T12:00:00+00:00",
                ),
                "sub_completed": SubagentRuntime(
                    subagent_id="sub_completed",
                    codename="Edith",
                    loop=completed_loop,  # type: ignore[arg-type]
                    storage=manager._catalog.session_storage(
                        owner_main_session_id="main_session",
                        subagent_id="sub_completed",
                    ),
                    owner_main_session_id="main_session",
                    owner_main_turn_id="main_turn",
                    status="completed",
                    created_at="2026-03-19T12:00:00+00:00",
                    updated_at="2026-03-19T12:00:00+00:00",
                ),
            }

            affected = manager.request_stop_all_for_user_stop()

            self.assertEqual([snapshot.subagent_id for snapshot in affected], ["sub_running", "sub_awaiting"])
            self.assertEqual(running_loop.stop_requests, 1)
            self.assertEqual(awaiting_loop.stop_requests, 1)
            self.assertEqual(paused_loop.stop_requests, 0)
            self.assertEqual(completed_loop.stop_requests, 0)
            self.assertEqual(running_loop.stop_reasons, ["user_stop"])
            self.assertEqual(awaiting_loop.stop_reasons, ["user_stop"])
            self.assertEqual(manager._subagents["sub_running"].pending_pause_reason, "main_stop")
            self.assertEqual(manager._subagents["sub_awaiting"].pending_pause_reason, "main_stop")
            self.assertIsNone(manager._subagents["sub_paused"].pending_pause_reason)
            self.assertIsNone(manager._subagents["sub_completed"].pending_pause_reason)

    async def test_reset_for_new_session_stops_running_and_silently_disposes_route_subagents(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            route_id = "route_reset_new_session"
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)
            published_events: list[object] = []

            async def publish_event(event: object) -> None:
                published_events.append(event)

            manager = SubagentManager(
                route_id=route_id,
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )

            running_loop = _FakeSubagentLoop([], session_id="running_session")
            completed_loop = _FakeSubagentLoop([], session_id="completed_session")
            manager._subagents = {
                "sub_running": SubagentRuntime(
                    subagent_id="sub_running",
                    codename="Friday",
                    loop=running_loop,  # type: ignore[arg-type]
                    storage=manager._catalog.session_storage(
                        owner_main_session_id="main_session",
                        subagent_id="sub_running",
                    ),
                    owner_main_session_id="main_session",
                    owner_main_turn_id="turn_1",
                    status="running",
                    created_at="2026-03-29T12:00:00+00:00",
                    updated_at="2026-03-29T12:00:00+00:00",
                ),
                "sub_completed": SubagentRuntime(
                    subagent_id="sub_completed",
                    codename="Jocasta",
                    loop=completed_loop,  # type: ignore[arg-type]
                    storage=manager._catalog.session_storage(
                        owner_main_session_id="main_session",
                        subagent_id="sub_completed",
                    ),
                    owner_main_session_id="main_session",
                    owner_main_turn_id="turn_1",
                    status="completed",
                    created_at="2026-03-29T12:00:00+00:00",
                    updated_at="2026-03-29T12:00:00+00:00",
                ),
            }

            manager._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id="sub_running",
                    codename="Friday",
                    status="running",
                    created_at="2026-03-29T12:00:00+00:00",
                    updated_at="2026-03-29T12:00:00+00:00",
                    route_id=route_id,
                    owner_main_session_id="main_session",
                    owner_main_turn_id="turn_1",
                    current_subagent_session_id="running_session",
                )
            )
            manager._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id="sub_completed",
                    codename="Jocasta",
                    status="completed",
                    created_at="2026-03-29T12:00:00+00:00",
                    updated_at="2026-03-29T12:00:00+00:00",
                    route_id=route_id,
                    owner_main_session_id="main_session",
                    owner_main_turn_id="turn_1",
                    current_subagent_session_id="completed_session",
                )
            )
            manager._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id="sub_stale",
                    codename="Ultron",
                    status="failed",
                    created_at="2026-03-29T12:00:00+00:00",
                    updated_at="2026-03-29T12:00:00+00:00",
                    route_id=route_id,
                    owner_main_session_id="older_session",
                    owner_main_turn_id="older_turn",
                    current_subagent_session_id="stale_session",
                )
            )

            async def _wait_for_turn_settle(runtime: SubagentRuntime) -> None:
                if runtime.subagent_id == "sub_running":
                    runtime.status = "paused"
                    runtime.pause_reason = "main_stop"
                runtime.task = None

            with patch.object(
                manager,
                "_wait_for_turn_settle",
                side_effect=_wait_for_turn_settle,
            ):
                result = await manager.reset_for_new_session()

            self.assertCountEqual(
                result["disposed_subagent_ids"],
                ["sub_running", "sub_completed", "sub_stale"],
            )
            self.assertEqual(result["disposed_count"], 3)
            self.assertEqual(result["cancelled_job_ids"], [])
            self.assertEqual(running_loop.stop_reasons, [])
            self.assertEqual(running_loop.hard_stop_reasons, ["new_session"])
            self.assertEqual(completed_loop.stop_requests, 0)
            self.assertEqual(completed_loop.hard_stop_requests, 0)
            self.assertEqual(manager._subagents["sub_running"].status, "disposed")
            self.assertEqual(manager._subagents["sub_completed"].status, "disposed")

            stale_entry = manager._catalog.get_entry("sub_stale")
            self.assertIsNotNone(stale_entry)
            if stale_entry is None:
                self.fail("Expected stale catalog entry to be retained and disposed.")
            self.assertEqual(stale_entry.status, "disposed")
            self.assertIsNotNone(stale_entry.disposed_at)

            dispose_notices = [
                event
                for event in published_events
                if isinstance(event, RouteSystemNoticeEvent)
                and event.notice_kind == "subagent_disposed"
            ]
            self.assertEqual(dispose_notices, [])

    async def test_reset_for_new_session_leaves_job_cancellation_to_route_supervisor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            route_id = "route_reset_background_jobs"
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)
            published_events: list[object] = []

            async def publish_event(event: object) -> None:
                published_events.append(event)

            manager = SubagentManager(
                route_id=route_id,
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )

            runtime = SubagentRuntime(
                subagent_id="sub_waiting",
                codename="Friday",
                loop=_FakeSubagentLoop([], session_id="waiting_session"),  # type: ignore[arg-type]
                storage=manager._catalog.session_storage(
                    owner_main_session_id="main_session",
                    subagent_id="sub_waiting",
                ),
                owner_main_session_id="main_session",
                owner_main_turn_id="turn_1",
                status="waiting_background",
                created_at="2026-03-29T12:00:00+00:00",
                updated_at="2026-03-29T12:00:00+00:00",
            )
            runtime.pending_background_job_ids.add("deadbeefdeadbeefdeadbeefdeadbeef")
            manager._subagents = {"sub_waiting": runtime}
            manager._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id="sub_waiting",
                    codename="Friday",
                    status="waiting_background",
                    created_at="2026-03-29T12:00:00+00:00",
                    updated_at="2026-03-29T12:00:00+00:00",
                    route_id=route_id,
                    owner_main_session_id="main_session",
                    owner_main_turn_id="turn_1",
                    current_subagent_session_id="waiting_session",
                )
            )

            job = create_background_job(
                workspace_dir=tool_settings.workspace_dir,
                bash_executable="/bin/bash",
                command="true",
                cwd="/workspace",
                log_max_bytes=tool_settings.bash_job_log_max_bytes,
                total_storage_budget_bytes=tool_settings.bash_job_total_storage_budget_bytes,
                retention_seconds=tool_settings.bash_job_retention_seconds,
            )
            claim_job_owner(
                workspace_dir=tool_settings.workspace_dir,
                job_id=job.job_id,
                route_id=route_id,
                session_id="waiting_session",
                turn_id="turn_1",
                agent_kind="subagent",
                agent_name="Friday",
                subagent_id="sub_waiting",
            )

            result = await manager.reset_for_new_session()

            self.assertEqual(result["cancelled_job_ids"], [])
            self.assertEqual(result["cancelled_job_count"], 0)
            self.assertEqual(manager._subagents["sub_waiting"].status, "disposed")
            self.assertEqual(manager._subagents["sub_waiting"].pending_background_job_ids, set())
            _, retained_job = load_job(tool_settings.workspace_dir, job.job_id)
            self.assertIsNone(retained_job.terminal_notice_dispatched_at)

            catalog_entry = manager._catalog.get_entry("sub_waiting")
            self.assertIsNotNone(catalog_entry)
            if catalog_entry is None:
                self.fail("Expected waiting-background subagent catalog entry to exist.")
            self.assertEqual(catalog_entry.status, "disposed")
            self.assertIsNotNone(catalog_entry.disposed_at)

            dispose_notices = [
                event
                for event in published_events
                if isinstance(event, RouteSystemNoticeEvent)
                and event.notice_kind == "subagent_disposed"
            ]
            self.assertEqual(dispose_notices, [])

    async def test_subagent_waits_for_detached_bash_jobs_before_reporting_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)
            published_events: list[object] = []

            async def publish_event(event: object) -> None:
                published_events.append(event)

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )

            runtime = SubagentRuntime(
                subagent_id="sub_1",
                codename="Friday",
                loop=_FakeSubagentLoop(
                    [AgentTurnDoneEvent(session_id="subagent_session", response_text="done")]
                ),  # type: ignore[arg-type]
                storage=manager._catalog.session_storage(
                    owner_main_session_id="main_session",
                    subagent_id="sub_1",
                ),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
                status="running",
                created_at="2026-03-21T10:00:00+00:00",
                updated_at="2026-03-21T10:00:00+00:00",
            )
            runtime.pending_background_job_ids.add("deadbeefdeadbeefdeadbeefdeadbeef")
            manager._subagents[runtime.subagent_id] = runtime
            manager._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id=runtime.subagent_id,
                    codename=runtime.codename,
                    status=runtime.status,
                    created_at=runtime.created_at,
                    updated_at=runtime.updated_at,
                    route_id="route_1",
                    owner_main_session_id=runtime.owner_main_session_id,
                    owner_main_turn_id=runtime.owner_main_turn_id,
                    current_subagent_session_id="subagent_session",
                )
            )

            await manager._run_turn(
                runtime,
                run_generation=runtime.run_generation,
                user_text="Continue.",
                force_session_id="subagent_session",
                pre_turn_messages=(),
            )

            self.assertEqual(runtime.status, "waiting_background")
            waiting_events = [
                event
                for event in published_events
                if isinstance(event, RouteSystemNoticeEvent)
                and event.notice_kind == "subagent_waiting_background"
            ]
            self.assertEqual(len(waiting_events), 1)
            self.assertIn("deadbeefdeadbeefdeadbeefdeadbeef", waiting_events[0].text)
            self.assertFalse(
                any(
                    isinstance(event, RouteSystemNoticeEvent)
                    and event.notice_kind == "subagent_completed"
                    for event in published_events
                )
            )

    async def test_subagent_bash_results_are_forwarded_to_shared_observer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)
            observed: list[tuple[ToolExecutionResult, ToolExecutionContext]] = []

            async def publish_event(_event: object) -> None:
                return None

            async def observe_tool_result(
                *,
                result: ToolExecutionResult,
                context: ToolExecutionContext,
            ) -> None:
                observed.append((result, context))

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
                tool_result_observer=observe_tool_result,
            )

            runtime = SubagentRuntime(
                subagent_id="sub_1",
                codename="Friday",
                loop=_FakeSubagentLoop([], session_id="subagent_session"),  # type: ignore[arg-type]
                storage=manager._catalog.session_storage(
                    owner_main_session_id="main_session",
                    subagent_id="sub_1",
                ),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
                status="running",
                created_at="2026-03-21T10:00:00+00:00",
                updated_at="2026-03-21T10:00:00+00:00",
            )
            manager._subagents[runtime.subagent_id] = runtime
            manager._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id=runtime.subagent_id,
                    codename=runtime.codename,
                    status=runtime.status,
                    created_at=runtime.created_at,
                    updated_at=runtime.updated_at,
                    route_id="route_1",
                    owner_main_session_id=runtime.owner_main_session_id,
                    owner_main_turn_id=runtime.owner_main_turn_id,
                    current_subagent_session_id="subagent_session",
                )
            )

            result = ToolExecutionResult(
                call_id="call_1",
                name="bash",
                ok=True,
                content="background running",
                metadata={
                    "mode": "foreground",
                    "promoted_to_background": True,
                    "job_id": "deadbeefdeadbeefdeadbeefdeadbeef",
                    "status": "running",
                    "state": "running",
                },
            )
            context = ToolExecutionContext(
                workspace_dir=core_settings.workspace_dir,
                route_id="route_1",
                session_id="subagent_session",
                turn_id="turn_1",
                agent_kind="subagent",
                agent_name="Friday",
                subagent_id="sub_1",
            )

            await manager._observe_tool_result(
                subagent_id="sub_1",
                result=result,
                context=context,
            )

            self.assertEqual(len(observed), 1)
            self.assertIs(observed[0][0], result)
            self.assertEqual(observed[0][1], context)
            self.assertEqual(
                runtime.pending_background_job_ids,
                {"deadbeefdeadbeefdeadbeefdeadbeef"},
            )
            self.assertTrue(
                any(
                    note.kind == "tool_result"
                    and "bash succeeded" in note.summary
                    and "status=running" in note.summary
                    for note in runtime.notable_events
                )
            )

    async def test_bash_job_followup_resumes_waiting_subagent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)

            async def publish_event(_event: object) -> None:
                return None

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )

            runtime = SubagentRuntime(
                subagent_id="sub_1",
                codename="Friday",
                loop=_FakeSubagentLoop([], session_id="subagent_session"),  # type: ignore[arg-type]
                storage=manager._catalog.session_storage(
                    owner_main_session_id="main_session",
                    subagent_id="sub_1",
                ),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
                status="waiting_background",
                created_at="2026-03-21T10:00:00+00:00",
                updated_at="2026-03-21T10:00:00+00:00",
            )
            runtime.pending_background_job_ids.add("deadbeefdeadbeefdeadbeefdeadbeef")
            manager._subagents[runtime.subagent_id] = runtime
            manager._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id=runtime.subagent_id,
                    codename=runtime.codename,
                    status=runtime.status,
                    created_at=runtime.created_at,
                    updated_at=runtime.updated_at,
                    route_id="route_1",
                    owner_main_session_id=runtime.owner_main_session_id,
                    owner_main_turn_id=runtime.owner_main_turn_id,
                    current_subagent_session_id="subagent_session",
                )
            )

            launched: dict[str, object] = {}

            def fake_launch_runtime_task(
                runtime_arg,
                *,
                user_text,
                force_session_id,
                pre_turn_messages,
                runtime_turn,
                name,
            ):
                launched["runtime"] = runtime_arg
                launched["user_text"] = user_text
                launched["force_session_id"] = force_session_id
                launched["pre_turn_messages"] = pre_turn_messages
                launched["runtime_turn"] = runtime_turn
                launched["name"] = name

            notice = BashJobNotice(
                job_id="deadbeefdeadbeefdeadbeefdeadbeef",
                notice_kind="bash_job_completed",
                owner_route_id="route_1",
                owner_session_id="subagent_session",
                owner_turn_id="turn_1",
                owner_agent_kind="subagent",
                owner_agent_name="Friday",
                owner_subagent_id="sub_1",
                status="finished",
                command="sleep 1; echo done",
                started_at="2026-03-21T10:00:00Z",
                last_update_at="2026-03-21T10:00:02Z",
                finished_at="2026-03-21T10:00:02Z",
                cancelled_at=None,
                exit_code=0,
                stdout="done\n",
                stderr="",
                stdout_bytes_seen=5,
                stderr_bytes_seen=0,
                stdout_bytes_dropped=0,
                stderr_bytes_dropped=0,
                progress_hint="done",
            )

            with patch.object(manager, "_launch_runtime_task", side_effect=fake_launch_runtime_task):
                await manager.enqueue_bash_job_followup((notice,))

            self.assertEqual(runtime.status, "running")
            self.assertEqual(runtime.pending_background_job_ids, set())
            self.assertEqual(launched["runtime"], runtime)
            self.assertIsNone(launched["user_text"])
            self.assertEqual(launched["pre_turn_messages"], ())
            self.assertTrue(bool(launched["runtime_turn"]))
            self.assertEqual(len(runtime.loop.system_notes), 1)
            note_content, note_session_id, note_metadata = runtime.loop.system_notes[0]
            self.assertIn(notice.job_id, note_content)
            self.assertNotIn("command:", note_content)
            self.assertNotIn("stdout tail:", note_content)
            self.assertIn("recommendation=finalize", note_content)
            self.assertIn("not a new user message or a new instruction from Jarvis", note_content)
            self.assertEqual(note_session_id, "subagent_session")
            self.assertEqual(note_metadata["notice_kind"], "bash_job_progress_update")
            self.assertEqual(note_metadata["recommended_action"], "finalize")

            runtime.status = "paused"
            runtime.pause_reason = None
            paused_job_id = "feedbeeffeedbeeffeedbeeffeedbeef"
            runtime.pending_background_job_ids.add(paused_job_id)
            paused_notice = replace(notice, job_id=paused_job_id)
            with patch.object(manager, "_launch_runtime_task", side_effect=fake_launch_runtime_task):
                accepted = await manager.enqueue_bash_job_followup((paused_notice,))

            self.assertTrue(accepted)
            self.assertEqual(runtime.status, "running")
            self.assertNotIn(paused_job_id, runtime.pending_background_job_ids)

            runtime.status = "paused"
            runtime.pause_reason = "main_stop"
            stopped_job_id = "cafebabecafebabecafebabecafebabe"
            runtime.pending_background_job_ids.add(stopped_job_id)
            stopped_notice = replace(notice, job_id=stopped_job_id)
            launched.clear()
            with patch.object(manager, "_launch_runtime_task", side_effect=fake_launch_runtime_task):
                accepted = await manager.enqueue_bash_job_followup((stopped_notice,))

            self.assertTrue(accepted)
            self.assertEqual(runtime.status, "paused")
            self.assertNotIn(stopped_job_id, runtime.pending_background_job_ids)
            self.assertEqual(launched, {})

    async def test_running_bash_job_followup_keeps_pending_job_until_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)

            async def publish_event(_event: object) -> None:
                return None

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )

            runtime = SubagentRuntime(
                subagent_id="sub_1",
                codename="Friday",
                loop=_FakeSubagentLoop([], session_id="subagent_session"),  # type: ignore[arg-type]
                storage=manager._catalog.session_storage(
                    owner_main_session_id="main_session",
                    subagent_id="sub_1",
                ),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
                status="waiting_background",
                created_at="2026-03-21T10:00:00+00:00",
                updated_at="2026-03-21T10:00:00+00:00",
            )
            runtime.pending_background_job_ids.add("deadbeefdeadbeefdeadbeefdeadbeef")
            manager._subagents[runtime.subagent_id] = runtime
            manager._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id=runtime.subagent_id,
                    codename=runtime.codename,
                    status=runtime.status,
                    created_at=runtime.created_at,
                    updated_at=runtime.updated_at,
                    route_id="route_1",
                    owner_main_session_id=runtime.owner_main_session_id,
                    owner_main_turn_id=runtime.owner_main_turn_id,
                    current_subagent_session_id="subagent_session",
                )
            )

            def fake_launch_runtime_task(
                runtime_arg,
                *,
                user_text,
                force_session_id,
                pre_turn_messages,
                runtime_turn,
                name,
            ):
                _ = (runtime_arg, user_text, force_session_id, pre_turn_messages, runtime_turn, name)

            notice = BashJobNotice(
                job_id="deadbeefdeadbeefdeadbeefdeadbeef",
                notice_kind="bash_job_output_grew",
                owner_route_id="route_1",
                owner_session_id="subagent_session",
                owner_turn_id="turn_1",
                owner_agent_kind="subagent",
                owner_agent_name="Friday",
                owner_subagent_id="sub_1",
                status="running",
                command="sleep 60",
                started_at="2026-03-21T10:00:00Z",
                last_update_at="2026-03-21T10:01:00Z",
                finished_at=None,
                cancelled_at=None,
                exit_code=None,
                stdout="",
                stderr="",
                stdout_bytes_seen=0,
                stderr_bytes_seen=0,
                stdout_bytes_dropped=0,
                stderr_bytes_dropped=0,
                progress_hint=None,
            )

            with patch.object(manager, "_launch_runtime_task", side_effect=fake_launch_runtime_task):
                await manager.enqueue_bash_job_followup((notice,))

            self.assertEqual(runtime.status, "waiting_background")
            self.assertEqual(
                runtime.pending_background_job_ids,
                {"deadbeefdeadbeefdeadbeefdeadbeef"},
            )
            self.assertEqual(runtime.loop.system_notes, [])

    async def test_child_silent_job_attention_resumes_child_without_escalating_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            published_events: list[object] = []

            async def publish_event(event: object) -> None:
                published_events.append(event)

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=ToolRegistry.default(
                    ToolSettings.from_workspace_dir(core_settings.workspace_dir)
                ),
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )
            runtime = SubagentRuntime(
                subagent_id="sub_1",
                codename="Ultron",
                loop=_FakeSubagentLoop([], session_id="subagent_session"),  # type: ignore[arg-type]
                storage=manager._catalog.session_storage(
                    owner_main_session_id="main_session",
                    subagent_id="sub_1",
                ),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
                status="waiting_background",
                created_at="2026-03-21T10:00:00+00:00",
                updated_at="2026-03-21T10:00:00+00:00",
            )
            runtime.pending_background_job_ids.add("deadbeefdeadbeefdeadbeefdeadbeef")
            manager._subagents[runtime.subagent_id] = runtime
            manager._catalog.create_entry(
                SubagentCatalogEntry(
                    subagent_id=runtime.subagent_id,
                    codename=runtime.codename,
                    status=runtime.status,
                    created_at=runtime.created_at,
                    updated_at=runtime.updated_at,
                    route_id="route_1",
                    owner_main_session_id=runtime.owner_main_session_id,
                    owner_main_turn_id=runtime.owner_main_turn_id,
                )
            )
            notice = BashJobNotice(
                job_id="deadbeefdeadbeefdeadbeefdeadbeef",
                notice_kind="bash_job_needs_attention",
                owner_route_id="route_1",
                owner_session_id="subagent_session",
                owner_turn_id="turn_1",
                owner_agent_kind="subagent",
                owner_agent_name="Ultron",
                owner_subagent_id="sub_1",
                status="running",
                command="run long stability test",
                started_at="2026-03-21T10:00:00Z",
                last_update_at="2026-03-21T10:05:00Z",
                finished_at=None,
                cancelled_at=None,
                exit_code=None,
                stdout="",
                stderr="",
                stdout_bytes_seen=0,
                stderr_bytes_seen=0,
                stdout_bytes_dropped=0,
                stderr_bytes_dropped=0,
                progress_hint=None,
            )

            with patch.object(manager, "_launch_runtime_task"):
                await manager.enqueue_bash_job_followup((notice,))

            lifecycle_notices = [
                event.notice_kind
                for event in published_events
                if isinstance(event, RouteSystemNoticeEvent)
            ]
            self.assertEqual(runtime.status, "running")
            self.assertIn("subagent_resumed_after_bash_update", lifecycle_notices)
            self.assertNotIn("subagent_needs_attention", lifecycle_notices)
            self.assertIn("recommendation=inspect", runtime.loop.system_notes[0][0])

    async def test_monitor_returns_full_pending_job_ids_and_nudges_on_unchanged_poll(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)

            async def publish_event(_event: object) -> None:
                return None

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
            )

            runtime = SubagentRuntime(
                subagent_id="sub_1",
                codename="Friday",
                loop=_FakeSubagentLoop([], session_id="subagent_session"),  # type: ignore[arg-type]
                storage=manager._catalog.session_storage(
                    owner_main_session_id="main_session",
                    subagent_id="sub_1",
                ),
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
                status="waiting_background",
                created_at="2026-03-21T10:00:00+00:00",
                updated_at="2026-03-21T10:00:00+00:00",
                task_label="Wait for validation",
                instructions="Run the detached validation and report results.",
                user_constraints="Do not modify source files.",
                shared_context="Validation owns the shared test database.",
                owned_paths=("tests/",),
                skill_ids=("testing",),
                deliverable="Validation evidence.",
                latest_report="Validation started; terminal result is pending.",
                report_complete=False,
                last_activity_at="2026-03-21T10:00:01+00:00",
            )
            runtime.pending_background_job_ids.add("deadbeefdeadbeefdeadbeefdeadbeef")
            manager._subagents[runtime.subagent_id] = runtime

            first = await manager.monitor(agent="sub_1", detail="summary")
            second = await manager.monitor(agent="sub_1", detail="summary")
            full = await manager.monitor(agent="sub_1", detail="full")

            self.assertTrue(first["changed"])
            self.assertEqual(
                first["subagents"][0]["pending_background_job_ids"],
                ["deadbeefdeadbeefdeadbeefdeadbeef"],
            )
            self.assertFalse(second["changed"])
            self.assertIn("Wait for orchestrator updates", second["message"])
            full_snapshot = full["subagents"][0]
            self.assertEqual(full_snapshot["task_label"], "Wait for validation")
            self.assertEqual(full_snapshot["latest_report"], runtime.latest_report)
            self.assertFalse(full_snapshot["report_complete"])
            self.assertEqual(full_snapshot["owned_paths"], ["tests/"])
            self.assertEqual(full_snapshot["skill_ids"], ["testing"])
            self.assertTrue(full_snapshot["transcript_path"].endswith("subagent_session.jsonl"))

    async def test_completed_subagent_counts_until_dispose_and_codename_reuses_after_dispose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)
            defaults = SubagentSettings.from_workspace_dir(
                core_settings.workspace_dir,
                transcript_archive_root=core_settings.transcript_archive_dir,
            )
            settings = SubagentSettings(
                provider=defaults.provider,
                max_active=1,
                codename_pool=("Friday",),
                archive_dir=defaults.archive_dir,
                builtin_tool_blocklist=defaults.builtin_tool_blocklist,
                main_context_event_limit=defaults.main_context_event_limit,
            )
            published_events: list[object] = []

            async def publish_event(event: object) -> None:
                published_events.append(event)

            manager = SubagentManager(
                route_id="route_1",
                llm_service=_FakeSubagentLLMService(),
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
                settings=settings,
            )

            first = await manager.invoke(
                requester_kind="main",
                task_label="First task",
                instructions="First task.",
                owner_main_session_id="main_session",
                owner_main_turn_id="turn_1",
            )
            first_runtime = manager._subagents[first["subagent_id"]]
            if first_runtime.task is not None:
                await asyncio.wait_for(first_runtime.task, timeout=1)

            with self.assertRaisesRegex(ValueError, "Subagent limit reached"):
                await manager.invoke(
                    requester_kind="main",
                    task_label="Blocked second task",
                    instructions="Second task should wait.",
                    owner_main_session_id="main_session",
                    owner_main_turn_id="turn_2",
                )

            dispose_result = await manager.dispose(agent=first["subagent_id"])
            self.assertEqual(dispose_result["status"], "disposed")
            dispose_notices = [
                event
                for event in published_events
                if isinstance(event, RouteSystemNoticeEvent)
                and event.notice_kind == "subagent_disposed"
            ]
            self.assertEqual(len(dispose_notices), 1)
            self.assertTrue(dispose_notices[0].public)

            second = await manager.invoke(
                requester_kind="main",
                task_label="Second task",
                instructions="Second task.",
                owner_main_session_id="main_session",
                owner_main_turn_id="turn_2",
            )
            self.assertEqual(second["codename"], "Friday")
            second_runtime = manager._subagents[second["subagent_id"]]
            if second_runtime.task is not None:
                await asyncio.wait_for(second_runtime.task, timeout=1)

            dispose_current = await manager.dispose(agent="Friday")
            self.assertEqual(dispose_current["subagent_id"], second["subagent_id"])
            self.assertTrue(dispose_current["changed"])
            self.assertEqual(second_runtime.status, "disposed")

    async def test_subagent_loop_uses_configured_provider_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)
            llm_service = _FakeSubagentLLMService()
            defaults = SubagentSettings.from_workspace_dir(
                core_settings.workspace_dir,
                transcript_archive_root=core_settings.transcript_archive_dir,
            )
            settings = SubagentSettings(
                provider="gemini",
                max_active=defaults.max_active,
                codename_pool=defaults.codename_pool,
                archive_dir=defaults.archive_dir,
                builtin_tool_blocklist=defaults.builtin_tool_blocklist,
                main_context_event_limit=defaults.main_context_event_limit,
            )

            async def publish_event(_event: object) -> None:
                return None

            manager = SubagentManager(
                route_id="route_1",
                llm_service=llm_service,
                core_settings=core_settings,
                tool_registry=registry,
                tool_execution_guard=asyncio.Semaphore(1),
                publish_event=publish_event,
                register_approval_target=lambda _approval_id, _loop: None,
                settings=settings,
            )

            payload = await manager.invoke(
                requester_kind="main",
                task_label="Provider override",
                instructions="Do the task.",
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )
            runtime = manager._subagents[payload["subagent_id"]]
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)

            self.assertTrue(llm_service.requests)
            self.assertEqual(llm_service.requests[0].provider, "gemini")
