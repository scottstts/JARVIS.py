"""Unit tests for Codex-backed actor runtime behavior."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from jarvis.codex_backend.actor_runtime import CodexActorRuntime
from jarvis.codex_backend.config import CodexBackendSettings
from jarvis.codex_backend.types import CodexNativeCapabilityError
from jarvis.core import AgentIdentity, AgentMemoryMode
from jarvis.llm import LLMMessage, ToolDefinition
from jarvis.storage import SessionStorage
from jarvis.tools import ToolExecutionResult, ToolRegistry, ToolRuntime, ToolSettings
from tests.helpers import build_core_settings


class _FakeBootstrapLoader:
    def load_bootstrap_messages(self):
        return [LLMMessage.text("system", "You are Jarvis.")]


class _FakeCoordinator:
    def __init__(self, *, turn_start_ids: list[str] | None = None) -> None:
        self.requests: list[tuple[str, dict[str, object] | None]] = []
        self.registered_threads: list[str] = []
        self.turn_start_requests: list[dict[str, object]] = []
        self._turn_start_ids = list(turn_start_ids or ["turn_1"])

    async def ensure_authenticated(self, *, on_challenge) -> None:
        _ = on_challenge

    async def request(self, method: str, params: dict[str, object] | None) -> object:
        self.requests.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": "thread_1"}}
        if method == "thread/resume":
            return {"thread": {"id": "thread_1"}}
        if method == "turn/start":
            if params is not None:
                self.turn_start_requests.append(params)
            if not self._turn_start_ids:
                raise AssertionError("Unexpected extra turn/start request.")
            return {
                "turn": {
                    "id": self._turn_start_ids.pop(0),
                    "items": [],
                    "status": "inProgress",
                    "error": None,
                }
            }
        if method == "turn/interrupt":
            return {}
        raise AssertionError(f"Unexpected coordinator request: {method}")

    def register_actor(self, *, thread_id: str, actor) -> None:
        _ = actor
        self.registered_threads.append(thread_id)

    def unregister_actor(self, *, thread_id: str, actor) -> None:
        _ = actor
        if thread_id in self.registered_threads:
            self.registered_threads.remove(thread_id)


async def _collect_events(async_iterable) -> list[object]:
    return [event async for event in async_iterable]


def _build_runtime(
    *,
    root_dir: Path,
    coordinator: _FakeCoordinator | None = None,
    tool_definitions_provider=None,
    tool_executor=None,
) -> tuple[CodexActorRuntime, SessionStorage, _FakeCoordinator]:
    core_settings = build_core_settings(root_dir=root_dir)
    workspace_dir = root_dir / "workspace"
    tool_settings = ToolSettings.from_workspace_dir(workspace_dir)
    registry = ToolRegistry.default(tool_settings)
    storage = SessionStorage(core_settings.transcript_archive_dir)
    coordinator = coordinator or _FakeCoordinator()
    runtime = CodexActorRuntime(
        coordinator=coordinator,
        settings=CodexBackendSettings(
            ws_url="ws://host.docker.internal:4500",
            model="gpt-5-codex",
            reasoning_effort="medium",
            reasoning_summary="none",
            personality="pragmatic",
            service_name="Jarvis",
            host_repo_root=root_dir,
            host_workspace_root=workspace_dir,
            approval_policy="untrusted",
            sandbox_network_access=False,
        ),
        llm_service=object(),  # type: ignore[arg-type]
        storage=storage,
        core_settings=core_settings,
        route_id="route_1",
        identity=AgentIdentity(kind="main", name="Jarvis"),
        bootstrap_loader=_FakeBootstrapLoader(),
        memory_mode=AgentMemoryMode(
            bootstrap=False,
            maintenance=False,
            reflection=False,
        ),
        tool_registry=registry,
        tool_runtime=ToolRuntime(registry=registry),
        tool_definitions_provider=tool_definitions_provider
        or (lambda _activated: ()),
        tool_executor=tool_executor
        or (lambda tool_call, context: asyncio.sleep(0)),  # pragma: no cover
        publish_route_event=lambda _event: asyncio.sleep(0),
    )
    return runtime, storage, coordinator


class CodexActorRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_session_persists_bootstrap_and_codex_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime, storage, _coordinator = _build_runtime(
                root_dir=Path(tmp),
                tool_definitions_provider=lambda _activated: (
                    ToolDefinition(
                        name="bash",
                        description="Run shell commands.",
                        input_schema={"type": "object"},
                    ),
                ),
            )

            session_id = await runtime.prepare_session()
            records = storage.load_records(session_id)

            self.assertEqual(len(records), 2)
            developer_snapshot = next(
                record
                for record in records
                if record.metadata.get("codex_bootstrap") == "developer_instructions"
            )
            self.assertIn("You are Jarvis.", developer_snapshot.content)
            self.assertIn("Tooling boundary:", developer_snapshot.content)
            self.assertEqual(developer_snapshot.metadata["transcript_only"], True)
            tool_snapshot = next(
                record
                for record in records
                if record.metadata.get("codex_bootstrap") == "dynamic_tools"
            )
            self.assertIn("Codex dynamic tools bootstrap snapshot:", tool_snapshot.content)
            self.assertEqual(tool_snapshot.metadata["transcript_only"], True)
            self.assertEqual(tool_snapshot.metadata["tool_definitions"][0]["name"], "bash")

    async def test_stream_turn_maps_deltas_completion_and_persists_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime, storage, coordinator = _build_runtime(root_dir=Path(tmp))

            events_task = asyncio.create_task(
                _collect_events(runtime.stream_turn(user_text="hello"))
            )
            await asyncio.sleep(0)
            await runtime.handle_notification(
                "item/agentMessage/delta",
                {
                    "threadId": "thread_1",
                    "turnId": "turn_1",
                    "itemId": "msg_1",
                    "delta": "Hello",
                },
            )
            await runtime.handle_notification(
                "turn/completed",
                {
                    "threadId": "thread_1",
                    "turn": {
                        "id": "turn_1",
                        "items": [],
                        "status": "completed",
                        "error": None,
                    },
                },
            )
            events = await events_task

            self.assertEqual([event.type for event in events], ["turn_started", "text_delta", "assistant_message", "done"])
            self.assertEqual(events[1].delta, "Hello")
            self.assertEqual(events[2].text, "Hello")
            self.assertEqual(events[3].response_text, "Hello")
            active = storage.get_active_session()
            self.assertIsNotNone(active)
            if active is None:
                self.fail("Expected an active session.")
            self.assertEqual(active.backend_state["thread_id"], "thread_1")
            self.assertEqual(active.backend_state["last_turn_id"], "turn_1")
            assistant_record = storage.load_records(active.session_id)[-1]
            self.assertEqual(assistant_record.content, "Hello")
            self.assertEqual(assistant_record.metadata["provider"], "codex")
            self.assertEqual(assistant_record.metadata["model"], "gpt-5-codex")
            self.assertEqual(assistant_record.metadata["response_id"], "turn_1")
            self.assertEqual(assistant_record.metadata["finish_reason"], "stop")
            self.assertEqual(assistant_record.metadata["tool_calls"], [])
            self.assertIn("thread_1", coordinator.registered_threads)

    async def test_handle_server_request_waits_for_jarvis_approval_and_retries_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = _FakeCoordinator()
            executor_calls: list[object] = []

            async def tool_executor(tool_call, context):
                executor_calls.append(context.approved_action)
                if context.approved_action is None:
                    return ToolExecutionResult(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        ok=False,
                        content="Approval required",
                        metadata={
                            "approval_required": True,
                            "approval_request": {
                                "approval_id": "approval_1",
                                "kind": "bash_command",
                                "summary": "Run one shell command.",
                                "details": "Need a one-off command before continuing.",
                                "command": "echo hi",
                                "tool_name": "bash",
                            },
                        },
                    )
                return ToolExecutionResult(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    ok=True,
                    content="bash ok",
                )

            runtime, _storage, _coordinator = _build_runtime(
                root_dir=Path(tmp),
                coordinator=coordinator,
                tool_definitions_provider=lambda _activated: (
                    ToolDefinition(
                        name="bash",
                        description="Run shell commands.",
                        input_schema={"type": "object", "properties": {"cmd": {"type": "string"}}},
                    ),
                ),
                tool_executor=tool_executor,
            )

            events_task = asyncio.create_task(
                _collect_events(runtime.stream_turn(user_text="run bash"))
            )
            await asyncio.sleep(0)
            response_task = asyncio.create_task(
                runtime.handle_server_request(
                    "item/tool/call",
                    {
                        "threadId": "thread_1",
                        "turnId": "turn_1",
                        "callId": "call_1",
                        "tool": "bash",
                        "arguments": {"cmd": "echo hi"},
                    },
                )
            )
            await asyncio.sleep(0)
            self.assertTrue(runtime.resolve_approval("approval_1", True))
            response = await response_task
            await runtime.handle_notification(
                "item/agentMessage/delta",
                {
                    "threadId": "thread_1",
                    "turnId": "turn_1",
                    "itemId": "msg_1",
                    "delta": "done",
                },
            )
            await runtime.handle_notification(
                "turn/completed",
                {
                    "threadId": "thread_1",
                    "turn": {
                        "id": "turn_1",
                        "items": [],
                        "status": "completed",
                        "error": None,
                    },
                },
            )
            events = await events_task

            self.assertEqual(executor_calls, [None, {"approval_id": "approval_1", "kind": "bash_command", "summary": "Run one shell command.", "details": "Need a one-off command before continuing.", "command": "echo hi", "tool_name": "bash"}])
            self.assertEqual(response["success"], True)
            self.assertEqual(response["contentItems"][0], {"type": "inputText", "text": "bash ok"})
            self.assertEqual(
                [event.type for event in events],
                ["turn_started", "tool_call", "approval_request", "text_delta", "assistant_message", "done"],
            )

    async def test_native_tool_attempt_interrupts_corrects_and_retries_same_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = _FakeCoordinator(turn_start_ids=["turn_1", "turn_2"])
            runtime, storage, _coordinator = _build_runtime(
                root_dir=Path(tmp),
                coordinator=coordinator,
            )

            events_task = asyncio.create_task(
                _collect_events(runtime.stream_turn(user_text="delegate this"))
            )
            await asyncio.sleep(0)
            await runtime.handle_notification(
                "item/started",
                {
                    "threadId": "thread_1",
                    "turnId": "turn_1",
                    "item": {
                        "id": "item_1",
                        "type": "collabAgentToolCall",
                    },
                },
            )
            await runtime.handle_notification(
                "turn/completed",
                {
                    "threadId": "thread_1",
                    "turn": {
                        "id": "turn_1",
                        "items": [],
                        "status": "completed",
                        "error": None,
                    },
                },
            )
            await runtime.handle_notification(
                "item/agentMessage/delta",
                {
                    "threadId": "thread_1",
                    "turnId": "turn_2",
                    "itemId": "msg_2",
                    "delta": "Recovered",
                },
            )
            await runtime.handle_notification(
                "turn/completed",
                {
                    "threadId": "thread_1",
                    "turn": {
                        "id": "turn_2",
                        "items": [],
                        "status": "completed",
                        "error": None,
                    },
                },
            )
            events = await events_task

            self.assertEqual(
                [event.type for event in events],
                ["turn_started", "text_delta", "assistant_message", "done"],
            )
            self.assertEqual(events[0].turn_id, "turn_1")
            self.assertEqual(events[-1].turn_id, "turn_1")
            self.assertEqual(events[-1].response_text, "Recovered")
            self.assertEqual(
                [method for method, _params in coordinator.requests if method == "turn/interrupt"],
                ["turn/interrupt"],
            )
            self.assertEqual(
                [method for method, _params in coordinator.requests if method == "turn/start"],
                ["turn/start", "turn/start"],
            )
            retry_input = coordinator.turn_start_requests[1]["input"][0]["text"]
            self.assertIn("unsupported native Codex capability 'collabAgentToolCall'", retry_input)
            active = storage.get_active_session()
            self.assertIsNotNone(active)
            if active is None:
                self.fail("Expected an active session.")
            self.assertEqual(active.backend_state["last_turn_id"], "turn_2")
            recovery_records = [
                record
                for record in storage.load_records(active.session_id)
                if record.metadata.get("codex_native_tool_recovery") is True
            ]
            self.assertEqual(len(recovery_records), 1)
            self.assertEqual(recovery_records[0].metadata["turn_id"], "turn_1")
            self.assertEqual(recovery_records[0].metadata["item_type"], "collabAgentToolCall")

    async def test_native_server_request_interrupts_corrects_and_retries_same_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = _FakeCoordinator(turn_start_ids=["turn_1", "turn_2"])
            runtime, storage, _coordinator = _build_runtime(
                root_dir=Path(tmp),
                coordinator=coordinator,
            )

            events_task = asyncio.create_task(
                _collect_events(runtime.stream_turn(user_text="edit a file"))
            )
            await asyncio.sleep(0)
            with self.assertRaises(CodexNativeCapabilityError):
                await runtime.handle_server_request(
                    "commandExecution",
                    {
                        "threadId": "thread_1",
                        "turnId": "turn_1",
                    },
                )
            await runtime.handle_notification(
                "turn/completed",
                {
                    "threadId": "thread_1",
                    "turn": {
                        "id": "turn_1",
                        "items": [],
                        "status": "failed",
                        "error": {"message": "Interrupted"},
                    },
                },
            )
            await runtime.handle_notification(
                "item/agentMessage/delta",
                {
                    "threadId": "thread_1",
                    "turnId": "turn_2",
                    "itemId": "msg_2",
                    "delta": "Retried",
                },
            )
            await runtime.handle_notification(
                "turn/completed",
                {
                    "threadId": "thread_1",
                    "turn": {
                        "id": "turn_2",
                        "items": [],
                        "status": "completed",
                        "error": None,
                    },
                },
            )
            events = await events_task

            self.assertEqual(
                [event.type for event in events],
                ["turn_started", "text_delta", "assistant_message", "done"],
            )
            self.assertEqual(events[-1].response_text, "Retried")
            active = storage.get_active_session()
            self.assertIsNotNone(active)
            if active is None:
                self.fail("Expected an active session.")
            recovery_records = [
                record
                for record in storage.load_records(active.session_id)
                if record.metadata.get("codex_native_tool_recovery") is True
            ]
            self.assertEqual(recovery_records[0].metadata["item_type"], "commandExecution")

    async def test_aclose_unregisters_loaded_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime, _storage, coordinator = _build_runtime(root_dir=Path(tmp))

            events_task = asyncio.create_task(
                _collect_events(runtime.stream_turn(user_text="hello"))
            )
            await asyncio.sleep(0)
            await runtime.handle_notification(
                "turn/completed",
                {
                    "threadId": "thread_1",
                    "turn": {
                        "id": "turn_1",
                        "items": [],
                        "status": "completed",
                        "error": None,
                    },
                },
            )
            await events_task
            self.assertIn("thread_1", coordinator.registered_threads)

            await runtime.aclose()

            self.assertNotIn("thread_1", coordinator.registered_threads)
