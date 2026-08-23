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
from jarvis.storage import ConversationRecord
from jarvis.subagent.manager import SubagentManager
from jarvis.subagent.runtime import SubagentRuntime
from jarvis.subagent.settings import SubagentSettings
from jarvis.subagent.storage import SubagentCatalogStorage
from jarvis.subagent.types import SubagentCatalogEntry
from tests.helpers import build_core_settings
from jarvis.tools import ToolExecutionContext, ToolExecutionResult, ToolRegistry, ToolSettings
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

    async def prepare_session(self, *, start_reason: str) -> str:
        _ = start_reason
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
    async def test_build_main_progress_message_includes_latest_subagent_report_for_finalize(self) -> None:
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
                "The complete report is included above.",
                message.content,
            )
            self.assertEqual(message.metadata["latest_subagent_report_included"], True)
            self.assertEqual(message.metadata["latest_subagent_report_complete"], True)

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
            self.assertEqual(unchanged, ())
            self.assertEqual(len(compacted_session), 1)
            self.assertEqual(len(changed), 1)
            self.assertIn("Inspected the runtime interface.", changed[0].content)

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
            self.assertIn("--- BEGIN SKILL review-check ---", bootstrap_text)
            self.assertIn("SKILL_FORWARDING_SENTINEL", bootstrap_text)
            self.assertEqual(payload["skill_ids"], ["review-check"])
            self.assertEqual(payload["skill_selection_reason"], "explicit_or_inherited")

            entry = manager._catalog.get_entry(payload["subagent_id"])
            self.assertIsNotNone(entry)
            if entry is None:
                self.fail("Expected structured assignment to be persisted.")
            self.assertEqual(entry.task_label, "Review storage contract")
            self.assertEqual(entry.skill_ids, ("review-check",))
            self.assertEqual(entry.skill_selection_reason, "explicit_or_inherited")
            self.assertEqual(entry.owned_paths, ("tests/test_storage.py",))

    async def test_auto_skill_selection_requires_assignment_domain_identity(self) -> None:
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

            self.assertEqual(
                manager._auto_select_skill_ids(
                    "Implement a Python HTTP status server with JSON responses, pytest tests, "
                    "environment-variable configuration, and clean signal shutdown."
                ),
                (),
            )
            self.assertEqual(
                manager._auto_select_skill_ids(
                    "Build a Three.js procedural terrain whose height fields drive geometry."
                ),
                ("threejs-procedural-fields",),
            )

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
