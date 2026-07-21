"""Focused unit tests for AgentLoop request-building transcript seams."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

from jarvis.core.agent_loop import (
    AgentLoop,
    _record_to_llm_message,
    _records_to_grok_recovery_messages,
)
from jarvis.llm.provider_context import (
    ProviderContextStrategy,
    ProviderSessionState,
    strategy_for_provider,
)
from jarvis.llm.types import LLMResponse, LocalImagePart, ToolCall, ToolResultPart
from jarvis.storage import ConversationRecord, SessionStorage
from jarvis.tools import ToolExecutionResult
from tests.helpers import build_core_settings


class AgentLoopRequestBuildingTests(unittest.TestCase):
    def test_provider_context_strategy_assignments_are_builtin(self) -> None:
        self.assertEqual(
            strategy_for_provider("openai"),
            ProviderContextStrategy.PROVIDER_STATEFUL_CONTINUATION,
        )
        self.assertEqual(
            strategy_for_provider("grok"),
            ProviderContextStrategy.PROVIDER_STATEFUL_CONTINUATION,
        )
        self.assertEqual(
            strategy_for_provider("gemini"),
            ProviderContextStrategy.PROVIDER_CACHED_CONTEXT,
        )
        self.assertEqual(
            strategy_for_provider("anthropic"),
            ProviderContextStrategy.LOCAL_REPLAY_WITH_PROMPT_CACHE,
        )
        self.assertEqual(
            strategy_for_provider("openrouter"),
            ProviderContextStrategy.LOCAL_REPLAY_WITH_PROMPT_CACHE,
        )

    def test_build_assistant_record_persists_provider_metadata_for_replay(self) -> None:
        loop = object.__new__(AgentLoop)
        response = LLMResponse(
            provider="grok",
            model="grok-4.20-reasoning",
            text="Hello",
            tool_calls=[],
            finish_reason="stop",
            usage=None,
            provider_metadata={
                "response_output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ]
            },
        )

        record = AgentLoop._build_assistant_record(
            loop, session_id="session_123", response=response
        )
        replay_message = _record_to_llm_message(record)

        self.assertIsNotNone(replay_message)
        self.assertEqual(record.metadata["provider_metadata"], response.provider_metadata)
        self.assertEqual(replay_message.metadata["provider_metadata"], response.provider_metadata)

    def test_build_request_uses_session_id_as_prompt_cache_key(self) -> None:
        loop = object.__new__(AgentLoop)
        loop._llm_provider = "grok"
        loop._compose_request_tools = lambda activated_names: ()
        record = ConversationRecord(
            record_id="rec_1",
            session_id="session_123",
            created_at="2026-04-09T00:00:00Z",
            role="user",
            content="hello",
        )

        request = AgentLoop._build_request(loop, [record])

        self.assertEqual(request.prompt_cache_key, "session_123")

    def test_openai_contextual_request_uses_provider_response_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp))
            session = storage.create_session()
            state = ProviderSessionState.for_provider("openai")
            self.assertIsNotNone(state)
            state = replace(
                state,
                openai=replace(
                    state.openai,
                    previous_response_id="resp_1",
                    last_response_record_id="assistant_1",
                ),
            )
            storage.update_session(
                session.session_id,
                provider_session_state=state.to_dict(),
            )
            loop = object.__new__(AgentLoop)
            loop._storage = storage
            loop._llm_provider = "openai"
            loop._compose_request_tools = lambda activated_names: ()

            current_records = (
                _record(session.session_id, "ctx", "system", "Turn context"),
                _record(session.session_id, "user_1", "user", "Run pwd"),
                _record(session.session_id, "assistant_1", "assistant", ""),
                ConversationRecord(
                    record_id="tool_1",
                    session_id=session.session_id,
                    created_at="2026-04-09T00:00:03Z",
                    role="tool",
                    content="Bash execution result\nstatus: success",
                    metadata={"tool_name": "bash", "call_id": "call_1", "ok": True},
                ),
            )

            request = AgentLoop._build_contextual_request(
                loop,
                session_id=session.session_id,
                base_records=(),
                current_records=current_records,
            )

            self.assertEqual(request.previous_response_id, "resp_1")
            self.assertIsNone(request.prompt_cache_key)
            self.assertEqual(len(request.messages), 1)
            self.assertEqual(request.messages[0].role, "tool")
            self.assertIsInstance(request.messages[0].parts[0], ToolResultPart)

    def test_grok_ephemeral_context_uses_live_delta_and_bounded_recovery_tail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp))
            session = storage.create_session()
            state = ProviderSessionState.for_provider("grok")
            self.assertIsNotNone(state)
            state = replace(
                state,
                grok=replace(
                    state.grok,
                    previous_response_id="resp_live",
                    last_response_record_id="assistant_live",
                    durable_response_id="resp_durable",
                    durable_response_record_id="assistant_durable",
                    storage_mode="ephemeral",
                    websocket_generation=2,
                ),
            )
            storage.update_session(
                session.session_id,
                provider_session_state=state.to_dict(),
            )
            loop = object.__new__(AgentLoop)
            loop._storage = storage
            loop._llm_provider = "grok"
            loop._compose_request_tools = lambda activated_names: ()

            base_records = (
                _record(session.session_id, "assistant_durable", "assistant", "Anchor"),
                _record(session.session_id, "tail_user", "user", "Tail input"),
                _record(session.session_id, "assistant_live", "assistant", "Tail output"),
            )
            current_records = (_record(session.session_id, "user_current", "user", "Continue"),)

            request = AgentLoop._build_contextual_request(
                loop,
                session_id=session.session_id,
                base_records=base_records,
                current_records=current_records,
            )

            self.assertEqual(request.previous_response_id, "resp_live")
            self.assertEqual(request.prompt_cache_key, session.session_id)
            self.assertEqual([message.role for message in request.messages], ["user"])
            continuation = request.stateful_continuation
            self.assertIsNotNone(continuation)
            self.assertEqual(continuation.storage_mode, "ephemeral")
            self.assertEqual(continuation.durable_response_id, "resp_durable")
            self.assertEqual(continuation.generation, 2)
            self.assertEqual(tuple(continuation.recovery_messages), ())
            self.assertEqual(
                [
                    message.role
                    for message in continuation.materialize_recovery_messages()
                ],
                ["user", "assistant"],
            )

    def test_grok_recovery_allows_terminal_tool_call_completed_by_current_delta(
        self,
    ) -> None:
        assistant = ConversationRecord(
            record_id="assistant_live",
            session_id="session_123",
            created_at="2026-04-09T00:00:00Z",
            role="assistant",
            content="Starting background work.",
            metadata={
                "tool_calls": [
                    {
                        "call_id": "call_1",
                        "name": "bash",
                        "arguments": {"command": "run"},
                        "raw_arguments": '{"command":"run"}',
                    }
                ]
            },
        )

        messages = _records_to_grok_recovery_messages((assistant,))

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "assistant")
        self.assertEqual(messages[0].parts[0].type, "text")
        self.assertIsInstance(messages[0].parts[1], ToolCall)

    def test_grok_context_build_defers_terminal_tool_call_recovery_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp))
            session = storage.create_session()
            state = ProviderSessionState.for_provider("grok")
            self.assertIsNotNone(state)
            assert state is not None
            storage.update_session(
                session.session_id,
                provider_session_state=replace(
                    state,
                    grok=replace(
                        state.grok,
                        previous_response_id="resp_live",
                        last_response_record_id="assistant_live",
                        durable_response_id="resp_durable",
                        durable_response_record_id="assistant_durable",
                        storage_mode="ephemeral",
                    ),
                ).to_dict(),
            )
            loop = object.__new__(AgentLoop)
            loop._storage = storage
            loop._llm_provider = "grok"
            loop._compose_request_tools = lambda activated_names: ()
            assistant_live = ConversationRecord(
                record_id="assistant_live",
                session_id=session.session_id,
                created_at="2026-04-09T00:00:01Z",
                role="assistant",
                content="Starting background work.",
                metadata={
                    "tool_calls": [
                        {
                            "call_id": "call_1",
                            "name": "bash",
                            "arguments": {"command": "run"},
                            "raw_arguments": '{"command":"run"}',
                        }
                    ]
                },
            )
            tool_result = ConversationRecord(
                record_id="tool_1",
                session_id=session.session_id,
                created_at="2026-04-09T00:00:02Z",
                role="tool",
                content="Bash execution result\nstatus: background",
                metadata={"call_id": "call_1", "tool_name": "bash", "ok": True},
            )

            request = AgentLoop._build_contextual_request(
                loop,
                session_id=session.session_id,
                base_records=(
                    _record(
                        session.session_id,
                        "assistant_durable",
                        "assistant",
                        "Anchor",
                    ),
                    assistant_live,
                ),
                current_records=(tool_result,),
            )

            self.assertEqual([message.role for message in request.messages], ["tool"])
            continuation = request.stateful_continuation
            self.assertIsNotNone(continuation)
            assert continuation is not None
            self.assertEqual(tuple(continuation.recovery_messages), ())
            recovery = continuation.materialize_recovery_messages()
            self.assertEqual([message.role for message in recovery], ["assistant"])
            self.assertIsInstance(recovery[0].parts[-1], ToolCall)

    def test_legacy_grok_state_migrates_previous_response_to_durable_anchor(
        self,
    ) -> None:
        state = ProviderSessionState.from_mapping(
            {
                "provider": "grok",
                "strategy": "provider_stateful_continuation",
                "grok": {
                    "previousResponseId": "resp_legacy",
                    "lastResponseRecordId": "assistant_legacy",
                },
            }
        )

        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.grok.durable_response_id, "resp_legacy")
        self.assertEqual(state.grok.durable_response_record_id, "assistant_legacy")
        self.assertEqual(state.grok.storage_mode, "durable")

    def test_current_grok_state_preserves_an_empty_durable_anchor(self) -> None:
        state = ProviderSessionState.from_mapping(
            {
                "provider": "grok",
                "strategy": "provider_stateful_continuation",
                "grok": {
                    "previousResponseId": "resp_live",
                    "lastResponseRecordId": "assistant_live",
                    "durableResponseId": None,
                    "durableResponseRecordId": None,
                    "storageMode": "ephemeral",
                    "websocketGeneration": 1,
                },
            }
        )

        self.assertIsNotNone(state)
        assert state is not None
        self.assertIsNone(state.grok.durable_response_id)
        self.assertIsNone(state.grok.durable_response_record_id)
        self.assertEqual(state.grok.storage_mode, "ephemeral")

    def test_grok_image_snapshot_is_compressed_recoverable_and_deduplicated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            source_path = settings.workspace_dir / "large-screenshot.png"
            image = Image.effect_noise((2_000, 1_200), 100).convert("RGB")
            image.save(source_path, format="PNG")
            source_size = source_path.stat().st_size
            self.assertGreater(source_size, 512 * 1024)

            storage = SessionStorage(settings.transcript_archive_dir)
            session = storage.create_session()
            loop = object.__new__(AgentLoop)
            loop._settings = settings
            loop._storage = storage
            loop._llm_provider = "grok"
            result = ToolExecutionResult(
                call_id="view_image_1",
                name="view_image",
                ok=True,
                content="Image attachment prepared",
                metadata={
                    "image_attachment": {
                        "path": str(source_path),
                        "media_type": "image/png",
                        "detail": "high",
                    }
                },
            )

            records = AgentLoop._build_ephemeral_image_records_from_tool_result(
                loop,
                session.session_id,
                result,
                turn_id="turn_1",
            )

            self.assertEqual(len(records), 2)
            live_record, recovery_record = records
            self.assertEqual(live_record.kind, "message")
            self.assertEqual(recovery_record.kind, "provider_context")
            self.assertTrue(recovery_record.metadata["transcript_only"])
            self.assertTrue(recovery_record.metadata["transcoded"])
            snapshot = Path(recovery_record.metadata["image_input"]["path"])
            self.assertTrue(snapshot.is_file())
            self.assertEqual(snapshot.suffix, ".jpg")
            self.assertLess(snapshot.stat().st_size, source_size)

            recovery_messages = _records_to_grok_recovery_messages((recovery_record,))
            self.assertEqual(len(recovery_messages), 1)
            self.assertIsInstance(recovery_messages[0].parts[0], LocalImagePart)

            storage.append_record(session.session_id, recovery_record)
            storage.append_record(
                session.session_id,
                _record(
                    session.session_id,
                    "assistant_live",
                    "assistant",
                    "Image inspected.",
                ),
            )
            state = ProviderSessionState.for_provider("grok")
            self.assertIsNotNone(state)
            storage.update_session(
                session.session_id,
                provider_session_state=replace(
                    state,
                    grok=replace(
                        state.grok,
                        previous_response_id="resp_live",
                        last_response_record_id="assistant_live",
                        storage_mode="ephemeral",
                    ),
                ).to_dict(),
            )
            duplicate_records = AgentLoop._build_ephemeral_image_records_from_tool_result(
                loop,
                session.session_id,
                result,
                turn_id="turn_2",
            )
            self.assertEqual(len(duplicate_records), 1)
            self.assertIn("already present", duplicate_records[0].content)
            self.assertNotIn("image_input", duplicate_records[0].metadata)

            AgentLoop._cleanup_grok_provider_media(loop, session.session_id)
            self.assertFalse(snapshot.exists())

    def test_anthropic_contextual_request_uses_full_local_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp))
            session = storage.create_session()
            loop = object.__new__(AgentLoop)
            loop._storage = storage
            loop._llm_provider = "anthropic"
            loop._compose_request_tools = lambda activated_names: ()
            base_record = _record(session.session_id, "system_1", "system", "System prompt")
            current_record = _record(session.session_id, "user_1", "user", "Hello")

            request = AgentLoop._build_contextual_request(
                loop,
                session_id=session.session_id,
                base_records=(base_record,),
                current_records=(current_record,),
            )

            self.assertEqual(request.prompt_cache_key, session.session_id)
            self.assertIsNone(request.previous_response_id)
            self.assertEqual([message.role for message in request.messages], ["system", "user"])

    def test_gemini_contextual_request_uses_cached_base_and_current_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp))
            session = storage.create_session()
            loop = object.__new__(AgentLoop)
            loop._storage = storage
            loop._llm_provider = "gemini"
            loop._compose_request_tools = lambda activated_names: ()
            base_record = _record(session.session_id, "system_1", "system", "System prompt")
            current_record = _record(session.session_id, "user_1", "user", "Hello")

            first_request = AgentLoop._build_contextual_request(
                loop,
                session_id=session.session_id,
                base_records=(base_record,),
                current_records=(current_record,),
            )
            self.assertIsNone(first_request.cached_content_name)
            self.assertEqual(len(first_request.cached_content_messages), 1)
            self.assertEqual([message.role for message in first_request.messages], ["user"])

            state = ProviderSessionState.from_mapping(
                storage.get_session(session.session_id).provider_session_state
            )
            self.assertIsNotNone(state)
            state = replace(
                state,
                gemini=replace(
                    state.gemini,
                    cached_content_name="cachedContents/abc123",
                    source_signature=first_request.cached_content_source_signature,
                    model="gemini-3-flash-preview",
                ),
            )
            storage.update_session(
                session.session_id,
                provider_session_state=state.to_dict(),
            )

            second_request = AgentLoop._build_contextual_request(
                loop,
                session_id=session.session_id,
                base_records=(base_record,),
                current_records=(current_record,),
            )

            self.assertEqual(second_request.cached_content_name, "cachedContents/abc123")
            self.assertEqual(second_request.cached_content_model, "gemini-3-flash-preview")
            self.assertEqual([message.role for message in second_request.messages], ["user"])
            self.assertEqual(len(second_request.cached_content_messages), 1)

    def test_provider_session_state_updates_from_openai_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp))
            session = storage.create_session()
            loop = object.__new__(AgentLoop)
            loop._storage = storage
            loop._llm_provider = "openai"
            response = LLMResponse(
                provider="openai",
                model="gpt-5.2-2025-12-11",
                text="Hello",
                tool_calls=[],
                finish_reason="stop",
                usage=None,
                response_id="resp_2",
                provider_metadata={"conversation_id": "conv_1"},
            )
            assistant_record = _record(session.session_id, "assistant_2", "assistant", "Hello")

            AgentLoop._persist_provider_session_state_from_response(
                loop,
                session_id=session.session_id,
                response=response,
                assistant_record=assistant_record,
            )

            state = ProviderSessionState.from_mapping(
                storage.get_session(session.session_id).provider_session_state
            )
            self.assertIsNotNone(state)
            self.assertEqual(state.openai.conversation_id, "conv_1")
            self.assertEqual(state.openai.previous_response_id, "resp_2")
            self.assertEqual(state.openai.last_response_record_id, "assistant_2")

    def test_provider_session_state_latches_grok_ephemeral_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp))
            session = storage.create_session()
            state = ProviderSessionState.for_provider("grok")
            self.assertIsNotNone(state)
            state = replace(
                state,
                grok=replace(
                    state.grok,
                    previous_response_id="resp_durable",
                    last_response_record_id="assistant_durable",
                    durable_response_id="resp_durable",
                    durable_response_record_id="assistant_durable",
                ),
            )
            storage.update_session(
                session.session_id,
                provider_session_state=state.to_dict(),
            )
            loop = object.__new__(AgentLoop)
            loop._storage = storage
            loop._llm_provider = "grok"
            response = LLMResponse(
                provider="grok",
                model="grok-4.5",
                text="Image inspected.",
                tool_calls=[],
                finish_reason="stop",
                usage=None,
                response_id="resp_live",
                provider_metadata={
                    "response_storage_mode": "ephemeral",
                    "durable_response_id": "resp_durable",
                    "websocket_generation": 4,
                },
            )
            assistant_record = _record(
                session.session_id,
                "assistant_live",
                "assistant",
                "Image inspected.",
            )

            AgentLoop._persist_provider_session_state_from_response(
                loop,
                session_id=session.session_id,
                response=response,
                assistant_record=assistant_record,
            )

            updated = ProviderSessionState.from_mapping(
                storage.get_session(session.session_id).provider_session_state
            )
            self.assertIsNotNone(updated)
            self.assertEqual(updated.grok.previous_response_id, "resp_live")
            self.assertEqual(updated.grok.last_response_record_id, "assistant_live")
            self.assertEqual(updated.grok.durable_response_id, "resp_durable")
            self.assertEqual(
                updated.grok.durable_response_record_id,
                "assistant_durable",
            )
            self.assertEqual(updated.grok.storage_mode, "ephemeral")
            self.assertEqual(updated.grok.websocket_generation, 4)


def _record(
    session_id: str,
    record_id: str,
    role: str,
    content: str,
) -> ConversationRecord:
    return ConversationRecord(
        record_id=record_id,
        session_id=session_id,
        created_at="2026-04-09T00:00:00Z",
        role=role,  # type: ignore[arg-type]
        content=content,
    )
