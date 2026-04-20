"""Unit tests for structured session compaction."""

from __future__ import annotations

import json
import unittest

from jarvis.core.compaction import ContextCompactor, prune_compaction_source_records
from jarvis.core.config import ContextPolicySettings
from jarvis.llm import LLMResponse, LLMUsage
from jarvis.storage import ConversationRecord


def _record(
    *,
    record_id: str,
    role: str,
    content: str,
    kind: str = "message",
    metadata: dict[str, object] | None = None,
) -> ConversationRecord:
    return ConversationRecord(
        record_id=record_id,
        session_id="session_1",
        created_at="2026-04-11T00:00:00+00:00",
        role=role,  # type: ignore[arg-type]
        content=content,
        kind=kind,  # type: ignore[arg-type]
        metadata=dict(metadata or {}),
    )


class _FakeCompactionLLMService:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    async def generate(self, request):  # type: ignore[no-untyped-def]
        return LLMResponse(
            provider=request.provider or "openai",
            model="fake-compactor",
            text=self.response_text,
            tool_calls=[],
            finish_reason="stop",
            usage=LLMUsage(input_tokens=12, output_tokens=8, total_tokens=20),
            response_id="resp_compact",
        )


class PruneCompactionSourceRecordsTests(unittest.TestCase):
    def test_drops_only_obvious_transient_items(self) -> None:
        kept_user = _record(record_id="user_1", role="user", content="Keep this exactly.")
        kept_compaction_history = _record(
            record_id="compact_1",
            role="system",
            content="Older compacted history that must remain available.",
            metadata={
                "type": "compaction",
                "compaction_item": True,
                "compaction_kind": "session_frame",
            },
        )
        records = [
            _record(
                record_id="audit_1",
                role="system",
                content="Compaction audit",
                kind="compaction",
            ),
            _record(
                record_id="bootstrap_1",
                role="system",
                content="Bootstrap",
                metadata={"bootstrap_identity": True},
            ),
            _record(
                record_id="tools_1",
                role="system",
                content="Tool bootstrap",
                metadata={"transcript_only": True},
            ),
            _record(
                record_id="memory_1",
                role="system",
                content="Memory bootstrap",
                metadata={"memory_bootstrap": "core"},
            ),
            _record(
                record_id="legacy_1",
                role="system",
                content="Legacy summary seed",
                metadata={"summary_seed": True},
            ),
            _record(
                record_id="time_1",
                role="system",
                content="System context auto-appended for this turn only.\nCurrent date/time: ...",
                metadata={"turn_context": "datetime"},
            ),
            _record(
                record_id="subagent_1",
                role="system",
                content="Subagent status snapshot:\n- Friday: running",
                metadata={"subagent_status_snapshot": True},
            ),
            _record(
                record_id="validation_1",
                role="tool",
                content="Bad call",
                metadata={"tool_call_validation_failed": True},
            ),
            _record(record_id="empty_1", role="assistant", content="   "),
            kept_compaction_history,
            kept_user,
        ]

        kept = prune_compaction_source_records(records)

        self.assertEqual(
            [record.record_id for record in kept],
            ["compact_1", "user_1"],
        )
        self.assertEqual(kept[0].content, kept_compaction_history.content)
        self.assertEqual(kept[1].content, kept_user.content)


class ContextCompactorTests(unittest.IsolatedAsyncioTestCase):
    async def test_compact_returns_structured_replacement_history(self) -> None:
        payload = {
            "items": [
                {
                    "type": "compaction",
                    "role": "system",
                    "kind": "session_frame",
                    "content": "Mission and durable constraints.",
                },
                {
                    "type": "compaction",
                    "role": "user",
                    "kind": "preserved_message",
                    "content": "Keep marker CODE-12345.",
                    "verbatim": True,
                    "source_record_ids": ["user_1"],
                    "source_range": {"start": 1, "end": 1},
                },
                {
                    "type": "compaction",
                    "role": "system",
                    "kind": "handover_state",
                    "content": "Resume by checking the latest build result.",
                },
            ]
        }
        compactor = ContextCompactor(
            llm_service=_FakeCompactionLLMService(json.dumps(payload)),
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
            provider="openai",
        )

        outcome = await compactor.compact(
            (
                _record(record_id="user_1", role="user", content="Keep marker CODE-12345."),
            )
        )

        self.assertEqual(outcome.provider, "openai")
        self.assertEqual(outcome.model, "fake-compactor")
        self.assertEqual(len(outcome.items), 3)
        self.assertEqual(outcome.items[0].kind, "session_frame")
        self.assertEqual(outcome.items[1].kind, "preserved_message")
        self.assertTrue(outcome.items[1].verbatim)
        self.assertEqual(outcome.items[1].source_record_ids, ("user_1",))
        self.assertEqual(outcome.items[2].kind, "handover_state")

    async def test_compact_rejects_invalid_replacement_history(self) -> None:
        payload = {
            "items": [
                {
                    "type": "compaction",
                    "role": "assistant",
                    "kind": "condensed_span",
                    "content": "Missing frame and handover.",
                }
            ]
        }
        compactor = ContextCompactor(
            llm_service=_FakeCompactionLLMService(json.dumps(payload)),
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
            provider="openai",
        )

        with self.assertRaisesRegex(ValueError, "must start with a system session_frame item"):
            await compactor.compact(
                (_record(record_id="user_1", role="user", content="Hello"),)
            )
