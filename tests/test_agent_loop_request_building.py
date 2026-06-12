"""Focused unit tests for AgentLoop request-building transcript seams."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from jarvis.core.agent_loop import AgentLoop, _record_to_llm_message
from jarvis.llm.provider_context import (
    ProviderContextStrategy,
    ProviderSessionState,
    strategy_for_provider,
)
from jarvis.llm.types import LLMResponse, ToolResultPart
from jarvis.storage import ConversationRecord, SessionStorage


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

        record = AgentLoop._build_assistant_record(loop, session_id="session_123", response=response)
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
