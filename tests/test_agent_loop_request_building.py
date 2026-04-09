"""Focused unit tests for AgentLoop request-building transcript seams."""

from __future__ import annotations

import unittest

from jarvis.core.agent_loop import AgentLoop, _record_to_llm_message
from jarvis.llm.types import LLMResponse
from jarvis.storage import ConversationRecord


class AgentLoopRequestBuildingTests(unittest.TestCase):
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

