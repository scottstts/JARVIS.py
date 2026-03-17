"""Unit tests for the subagent manager and storage behavior."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import (
    AgentApprovalRequestEvent,
    AgentAssistantMessageEvent,
    AgentToolCallEvent,
    AgentTurnDoneEvent,
)
from gateway.route_events import (
    RouteApprovalRequestEvent,
    RouteSystemNoticeEvent,
    RouteToolCallEvent,
)
from llm import DoneEvent, LLMRequest, LLMResponse, LLMUsage, TextDeltaEvent
from subagent.manager import SubagentManager
from subagent.settings import SubagentSettings
from subagent.storage import SubagentCatalogStorage
from tests.helpers import build_core_settings
from tools import ToolRegistry, ToolSettings


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

    async def prepare_session(self, *, start_reason: str) -> str:
        _ = start_reason
        return self._session_id

    async def stream_turn(self, *, user_text: str, force_session_id: str | None, pre_turn_messages):
        _ = (user_text, force_session_id, pre_turn_messages)
        for event in self._events:
            yield event

    def active_session_id(self) -> str | None:
        return self._session_id

    def request_stop(self) -> bool:
        return True


class SubagentSettingsTests(unittest.TestCase):
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


class SubagentManagerTests(unittest.IsolatedAsyncioTestCase):
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
                instructions="Inspect the workspace and report back.",
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )

            self.assertTrue(payload["session_id"])
            runtime = manager._subagents[payload["subagent_id"]]
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)

            subagent_settings = SubagentSettings.from_workspace_dir(core_settings.workspace_dir)
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
            self.assertEqual(entry.current_subagent_session_id, payload["session_id"])
            self.assertEqual(entry.status, "completed")

            self.assertEqual(len(published_events), 2)
            self.assertIsInstance(published_events[0], RouteSystemNoticeEvent)
            notice = published_events[0]
            if not isinstance(notice, RouteSystemNoticeEvent):
                self.fail("Expected invoke notice to be a route system notice.")
            self.assertEqual(notice.session_id, payload["session_id"])
            self.assertEqual(notice.notice_kind, "subagent_invoked")
            completion_notice = published_events[1]
            self.assertIsInstance(completion_notice, RouteSystemNoticeEvent)
            if not isinstance(completion_notice, RouteSystemNoticeEvent):
                self.fail("Expected completion notice to be a route system notice.")
            self.assertEqual(completion_notice.notice_kind, "subagent_completed")
            self.assertEqual(completion_notice.text, "completed.")

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
                    for event in published_events
                )
            )

    async def test_completed_subagent_counts_until_dispose_and_codename_reuses_after_dispose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)
            defaults = SubagentSettings.from_workspace_dir(core_settings.workspace_dir)
            settings = SubagentSettings(
                provider=defaults.provider,
                max_active=1,
                codename_pool=("Friday",),
                archive_dir=defaults.archive_dir,
                builtin_tool_blocklist=defaults.builtin_tool_blocklist,
                main_context_event_limit=defaults.main_context_event_limit,
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
                settings=settings,
            )

            first = await manager.invoke(
                requester_kind="main",
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
                    instructions="Second task should wait.",
                    owner_main_session_id="main_session",
                    owner_main_turn_id="turn_2",
                )

            dispose_result = await manager.dispose(agent=first["subagent_id"])
            self.assertEqual(dispose_result["status"], "disposed")

            second = await manager.invoke(
                requester_kind="main",
                instructions="Second task.",
                owner_main_session_id="main_session",
                owner_main_turn_id="turn_2",
            )
            self.assertEqual(second["codename"], "Friday")
            second_runtime = manager._subagents[second["subagent_id"]]
            if second_runtime.task is not None:
                await asyncio.wait_for(second_runtime.task, timeout=1)

    async def test_subagent_loop_uses_configured_provider_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))
            tool_settings = ToolSettings.from_workspace_dir(core_settings.workspace_dir)
            registry = ToolRegistry.default(tool_settings)
            llm_service = _FakeSubagentLLMService()
            defaults = SubagentSettings.from_workspace_dir(core_settings.workspace_dir)
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
                instructions="Do the task.",
                owner_main_session_id="main_session",
                owner_main_turn_id="main_turn",
            )
            runtime = manager._subagents[payload["subagent_id"]]
            if runtime.task is not None:
                await asyncio.wait_for(runtime.task, timeout=1)

            self.assertTrue(llm_service.requests)
            self.assertEqual(llm_service.requests[0].provider, "gemini")
