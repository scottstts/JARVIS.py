"""Tests for schema-constrained semantic session compaction."""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

from jarvis.core.compaction import (
    ContextCompactor,
    build_compaction_bundle_record,
    build_compaction_source_events,
    load_compaction_bundle,
    prune_compaction_source_records,
)
from jarvis.core.compaction_contract import (
    CompactionBundle,
    CompactionContractError,
    apply_compaction_draft,
    build_source_manifest,
    compile_compaction_replay,
)
from jarvis.core.config import ContextPolicySettings
from jarvis.core.errors import ContextBudgetError
from jarvis.core.token_estimator import estimate_request_input_tokens
from jarvis.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    TextPart,
    ToolCall,
    ToolChoiceMode,
)
from jarvis.llm.errors import ProviderTimeoutError
from jarvis.storage import ConversationRecord


def _record(
    *,
    record_id: str,
    role: str,
    content: str,
    session_id: str = "session_1",
    kind: str = "message",
    metadata: dict[str, object] | None = None,
) -> ConversationRecord:
    return ConversationRecord(
        record_id=record_id,
        session_id=session_id,
        created_at="2026-08-21T00:00:00+00:00",
        role=role,  # type: ignore[arg-type]
        content=content,
        kind=kind,  # type: ignore[arg-type]
        metadata=dict(metadata or {}),
    )


def _draft(
    *,
    objective: str = "Continue the current task.",
    preserved_refs: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "objective": objective,
        "background": ["The task has prior implementation context."],
        "preserved_messages": [
            {"source_ref": source_ref, "reason": "Exact wording is material."}
            for source_ref in preserved_refs
        ],
        "episodes": [
            {
                "summary": "The prior work was inspected and advanced.",
                "outcomes": ["A concrete continuation point was established."],
            }
        ],
        "constraints": ["Preserve the user's requested behavior."],
        "decisions": ["Continue with the current architecture."],
        "artifacts": [
            {
                "summary": "The implementation artifact was modified.",
                "locator": "/workspace/project/example.py",
                "last_observed_state": "The file existed after the last tool result.",
                "needs_verification": True,
            }
        ],
        "open_loops": [
            {
                "summary": "Final verification remains.",
                "next_action": "Run the focused test suite.",
                "blocker": None,
            }
        ],
        "uncertainties": ["The latest external state has not been rechecked."],
        "handover": {
            "current_focus": "Finish and verify the implementation.",
            "next_actions": ["Run the focused test suite."],
            "do_not_repeat": ["Do not redo the completed inspection."],
            "verification_needed": ["Recheck the artifact before claiming completion."],
        },
    }


def _response(
    payload: dict[str, Any],
    *,
    as_text: bool = False,
    tool_name: str = "submit_compaction",
) -> LLMResponse:
    raw = json.dumps(payload, ensure_ascii=False)
    return LLMResponse(
        provider="openai",
        model="fake-compactor",
        text=raw if as_text else "",
        tool_calls=(
            []
            if as_text
            else [
                ToolCall(
                    call_id="call_compact",
                    name=tool_name,
                    arguments=payload,
                    raw_arguments=raw,
                )
            ]
        ),
        finish_reason="stop" if as_text else "tool_calls",
        usage=LLMUsage(input_tokens=12, output_tokens=8, total_tokens=20),
        response_id="resp_compact",
    )


class _QueuedCompactionLLMService:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("Unexpected extra compaction request.")
        return self.responses.pop(0)


class _BlockingCompactionLLMService:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        _ = request
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("Blocked compaction unexpectedly resumed.")


def _request_user_text(request: LLMRequest) -> str:
    return "\n".join(
        part.text
        for message in request.messages
        if message.role == "user"
        for part in message.parts
        if isinstance(part, TextPart)
    )


def _bundle_from_draft(
    draft: dict[str, Any],
    records: tuple[ConversationRecord, ...],
    *,
    previous_bundle: CompactionBundle | None = None,
) -> CompactionBundle:
    generation = previous_bundle.generation + 1 if previous_bundle else 1
    events = build_compaction_source_events(records, generation=generation)
    manifest = build_source_manifest(
        generation=generation,
        previous_bundle=previous_bundle,
        source_events=events,
    )
    return apply_compaction_draft(
        draft,
        bundle_id=f"bundle_{generation}",
        created_at="2026-08-21T00:00:00+00:00",
        source_manifest=manifest,
        source_events=events,
        previous_bundle=previous_bundle,
    )


class PruneCompactionSourceRecordsTests(unittest.TestCase):
    def test_drops_runtime_scaffolding_and_prior_replay_but_keeps_causal_failures(self) -> None:
        records = [
            _record(
                record_id="replay",
                role="assistant",
                content="Old generated replay.",
                metadata={"compaction_item": True, "type": "compaction_replay"},
            ),
            _record(
                record_id="bootstrap",
                role="system",
                content="Bootstrap",
                metadata={"bootstrap_identity": True},
            ),
            _record(
                record_id="snapshot",
                role="system",
                content="Subagent status snapshot:\n- child running",
                metadata={"subagent_status_snapshot": True},
            ),
            _record(
                record_id="validation_failure",
                role="tool",
                content="The tool call failed schema validation.",
                metadata={
                    "tool_call_validation_failed": True,
                    "call_id": "call_1",
                    "tool_name": "bash",
                    "ok": False,
                },
            ),
            _record(
                record_id="child_terminal",
                role="system",
                content="Subagent completed with a report.",
                metadata={
                    "subagent_progress_update": True,
                    "pending_subagent_ids": [],
                },
            ),
            _record(record_id="user", role="user", content="Continue."),
        ]

        kept = prune_compaction_source_records(records)

        self.assertEqual(
            [record.record_id for record in kept],
            ["validation_failure", "child_terminal", "user"],
        )

    def test_source_events_keep_internal_lineage_without_provider_noise(self) -> None:
        records = (
            _record(
                record_id="assistant_call",
                role="assistant",
                content="I will inspect it.",
                metadata={
                    "turn_id": "turn_1",
                    "tool_calls": [
                        {
                            "call_id": "call_1",
                            "name": "bash",
                            "arguments": {"command": "pwd"},
                            "raw_arguments": "ignored",
                            "provider_metadata": {"ignored": True},
                        }
                    ],
                },
            ),
            _record(
                record_id="tool_result",
                role="tool",
                content="/workspace",
                metadata={
                    "turn_id": "turn_1",
                    "call_id": "call_1",
                    "tool_name": "bash",
                    "ok": True,
                },
            ),
        )

        events = build_compaction_source_events(records, generation=3)

        self.assertEqual(events[0].event_type, "assistant_tool_call")
        self.assertEqual(events[1].event_type, "tool_result")
        self.assertEqual(events[0].causal_ids, ("call_1",))
        self.assertEqual(events[1].causal_ids, ("call_1",))
        self.assertEqual(events[0].turn_id, "turn_1")
        self.assertNotIn("raw_arguments", json.dumps(events[0].to_dict()))
        self.assertNotIn("provider_metadata", json.dumps(events[0].to_dict()))


class ContextCompactorTests(unittest.IsolatedAsyncioTestCase):
    async def test_total_deadline_covers_the_entire_workflow(self) -> None:
        service = _BlockingCompactionLLMService()
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
            deadline_seconds=0.01,
        )

        with self.assertRaises(ProviderTimeoutError) as raised:
            await compactor.compact(
                (_record(record_id="user_1", role="user", content="Continue."),)
            )

        self.assertTrue(service.started.is_set())
        self.assertTrue(service.cancelled.is_set())
        self.assertEqual(raised.exception.metadata["compaction_phase"], "generate")
        self.assertEqual(raised.exception.metadata["compaction_call_count"], 0)

    async def test_normal_compaction_is_one_schema_constrained_model_call(self) -> None:
        service = _QueuedCompactionLLMService([_response(_draft())])
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
            provider="openai",
        )

        outcome = await compactor.compact(
            (_record(record_id="user_1", role="user", content="Build it."),)
        )

        self.assertEqual(len(service.requests), 1)
        request = service.requests[0]
        self.assertEqual(request.tool_choice.mode, ToolChoiceMode.TOOL)
        self.assertEqual(request.tool_choice.tool_name, "submit_compaction")
        self.assertEqual([tool.name for tool in request.tools], ["submit_compaction"])
        self.assertFalse(request.parallel_tool_calls)
        self.assertEqual([trace.phase for trace in outcome.call_traces], ["generate"])
        self.assertEqual(outcome.repair_count, 0)
        self.assertEqual(
            outcome.verification_payload["method"],
            "jarvis_deterministic_contract",
        )

    async def test_exact_message_selection_uses_short_ref_and_jarvis_copies_bytes(self) -> None:
        exact_text = "Use Jarvis’s exact punctuation and trailing space. "
        service = _QueuedCompactionLLMService(
            [_response(_draft(preserved_refs=("E1",)))]
        )
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )

        outcome = await compactor.compact(
            (_record(record_id="internal_record_uuid", role="user", content=exact_text),)
        )

        preserved = outcome.bundle.preserved_records[0]
        self.assertEqual(preserved.content, exact_text)
        self.assertEqual(preserved.record_id, "internal_record_uuid")
        replay = next(item for item in outcome.items if item.kind == "preserved_message")
        self.assertEqual(replay.content, exact_text)
        self.assertTrue(replay.exact_copy)
        request_text = _request_user_text(service.requests[0])
        self.assertIn("[EVENT E1 |", request_text)
        self.assertNotIn("internal_record_uuid", request_text)
        self.assertNotIn("session_1", request_text)

    async def test_paired_tool_arguments_are_not_duplicated_in_compaction_input(self) -> None:
        command = "python -c \"" + ("x = 1; " * 2_000) + "\""
        records = (
            _record(
                record_id="assistant_call",
                role="assistant",
                content="Writing the implementation.",
                metadata={
                    "tool_calls": [
                        {
                            "call_id": "call_1",
                            "name": "bash",
                            "arguments": {"command": command},
                        }
                    ]
                },
            ),
            _record(
                record_id="tool_result",
                role="tool",
                content=f"Bash execution result\ncommand: {command}\nstatus: success",
                metadata={"call_id": "call_1", "tool_name": "bash", "ok": True},
            ),
        )
        service = _QueuedCompactionLLMService([_response(_draft())])
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )

        await compactor.compact(records)

        request_text = _request_user_text(service.requests[0])
        self.assertEqual(request_text.count(command), 1)
        self.assertIn("<exact value appears in paired tool result>", request_text)
        self.assertIn("tool call: bash -> result E2", request_text)

    async def test_large_tool_evidence_is_bounded_before_provider_call(self) -> None:
        huge_result = "begin\n" + ("middle-data\n" * 50_000) + "end"
        records = (
            _record(record_id="user_1", role="user", content="Continue."),
            _record(
                record_id="tool_1",
                role="tool",
                content=huge_result,
                metadata={"call_id": "call_1", "tool_name": "bash", "ok": True},
            ),
        )
        service = _QueuedCompactionLLMService([_response(_draft())])
        policy = ContextPolicySettings(context_window_tokens=100_000)
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=policy,
        )

        await compactor.compact(records)

        request = service.requests[0]
        request_text = _request_user_text(request)
        self.assertIn("source characters omitted", request_text)
        self.assertIn("sha256=", request_text)
        self.assertLess(
            estimate_request_input_tokens(request),
            policy.preflight_limit_tokens * 70 // 100,
        )

    async def test_long_session_shape_is_globally_bounded_and_semantically_coalesced(
        self,
    ) -> None:
        contract = "Task contract\ntask_id: long-task\nRequired acceptance items:\n- run tests"
        records: list[ConversationRecord] = [
            _record(record_id="user_1", role="user", content="Implement the full repair."),
            _record(
                record_id="assistant_1",
                role="assistant",
                content="I am coordinating the workstreams.",
            ),
        ]
        records.extend(
            _record(
                record_id=f"contract_{index}",
                role="system",
                content=contract,
                metadata={
                    "task_contract": True,
                    "task_id": "long-task",
                    "task_contract_revision": "revision-1",
                },
            )
            for index in range(131)
        )
        for index in range(40):
            records.append(
                _record(
                    record_id=f"bash_progress_{index}",
                    role="system",
                    content=(f"Job job_1 progress update {index}\n" + "log line\n" * 2_000),
                    metadata={
                        "bash_job_progress_update": True,
                        "bash_job_running_ids": ["job_1"],
                        "bash_job_terminal_ids": [],
                    },
                )
            )
        service = _QueuedCompactionLLMService([_response(_draft())])
        policy = ContextPolicySettings(context_window_tokens=100_000)
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=policy,
        )

        await compactor.compact(tuple(records))

        request = service.requests[0]
        request_text = _request_user_text(request)
        self.assertEqual(request_text.count(contract), 1)
        self.assertIn("Earlier coalesced lifecycle or repeated evidence", request_text)
        self.assertLess(
            estimate_request_input_tokens(request),
            policy.preflight_limit_tokens * 70 // 100,
        )

    async def test_global_budget_accounts_for_multiple_tool_arguments_per_event(
        self,
    ) -> None:
        records: list[ConversationRecord] = [
            _record(record_id="user_1", role="user", content="Continue the repair.")
        ]
        for event_index in range(20):
            records.append(
                _record(
                    record_id=f"assistant_tools_{event_index}",
                    role="assistant",
                    content="",
                    metadata={
                        "tool_calls": [
                            {
                                "call_id": f"call_{event_index}_{call_index}",
                                "name": "bash",
                                "arguments": {
                                    "command": "x" * 20_000,
                                    "write_paths": [f"src/{call_index}.py"],
                                },
                            }
                            for call_index in range(5)
                        ]
                    },
                )
            )
        service = _QueuedCompactionLLMService([_response(_draft())])
        policy = ContextPolicySettings(context_window_tokens=30_000)
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=policy,
        )

        await compactor.compact(tuple(records))

        request = service.requests[0]
        self.assertLess(
            estimate_request_input_tokens(request),
            policy.preflight_limit_tokens * 70 // 100,
        )
        self.assertIn("source characters omitted", _request_user_text(request))

    async def test_oversized_non_tool_evidence_fails_before_calling_provider(self) -> None:
        service = _QueuedCompactionLLMService([])
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )

        with self.assertRaises(ContextBudgetError) as raised:
            await compactor.compact(
                (_record(record_id="user_1", role="user", content="x" * 400_000),)
            )

        self.assertEqual(service.requests, [])
        self.assertGreater(
            raised.exception.metadata["compaction_estimated_input_tokens"],
            raised.exception.metadata["compaction_safe_input_limit_tokens"],
        )

    async def test_invalid_short_ref_gets_exceptional_targeted_retry(self) -> None:
        invalid = _draft(preserved_refs=("E999",))
        repaired = _draft(preserved_refs=("E1",))
        service = _QueuedCompactionLLMService(
            [_response(invalid), _response(repaired)]
        )
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )

        outcome = await compactor.compact(
            (_record(record_id="user_1", role="user", content="Keep this."),)
        )

        self.assertEqual(len(service.requests), 2)
        self.assertEqual([trace.phase for trace in outcome.call_traces], ["generate", "repair"])
        self.assertEqual(outcome.repair_count, 1)
        retry_text = _request_user_text(service.requests[1])
        self.assertIn("unknown_preserved_source_ref", retry_text)
        self.assertIn("REJECTED SUBMISSION", retry_text)

    async def test_invalid_submissions_are_strictly_bounded(self) -> None:
        invalid = _draft(preserved_refs=("missing",))
        service = _QueuedCompactionLLMService([_response(invalid) for _ in range(3)])
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )

        with self.assertRaises(CompactionContractError) as raised:
            await compactor.compact(
                (_record(record_id="user_1", role="user", content="Continue."),)
            )

        self.assertEqual(len(service.requests), 3)
        self.assertEqual(raised.exception.metadata["compaction_call_count"], 3)
        self.assertEqual(raised.exception.metadata["compaction_draft_attempt"], 3)
        self.assertEqual(
            [trace["phase"] for trace in raised.exception.metadata["compaction_call_traces"]],
            ["generate", "repair", "repair"],
        )

    async def test_valid_json_text_is_accepted_as_provider_fallback(self) -> None:
        service = _QueuedCompactionLLMService([_response(_draft(), as_text=True)])
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )

        outcome = await compactor.compact(
            (_record(record_id="user_1", role="user", content="Continue."),)
        )

        self.assertEqual(outcome.bundle.objective.summary, "Continue the current task.")
        self.assertEqual(len(service.requests), 1)

    async def test_later_generation_receives_semantic_prior_state_and_delta_only(self) -> None:
        first_service = _QueuedCompactionLLMService(
            [_response(_draft(preserved_refs=("E1",)))]
        )
        first_compactor = ContextCompactor(
            llm_service=first_service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )
        first = await first_compactor.compact(
            (_record(record_id="user_1", role="user", content="Keep this exact."),)
        )
        second_draft = _draft(
            objective="Continue with the correction.",
            preserved_refs=("P1",),
        )
        second_service = _QueuedCompactionLLMService([_response(second_draft)])
        second_compactor = ContextCompactor(
            llm_service=second_service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )

        second = await second_compactor.compact(
            (_record(record_id="user_2", role="user", content="Apply the correction."),),
            previous_bundle=first.bundle,
        )

        request_text = _request_user_text(second_service.requests[0])
        self.assertIn('"source_ref":"P1"', request_text)
        self.assertIn("[EVENT E1 |", request_text)
        self.assertNotIn("user_1", request_text)
        self.assertNotIn("user_2", request_text)
        self.assertEqual(second.bundle.generation, 2)
        self.assertEqual(second.bundle.preserved_records[0].content, "Keep this exact.")
        self.assertEqual(second.bundle.source_manifest.cumulative_record_count, 2)
        self.assertEqual(second.bundle.source_manifest.delta_record_ids, ("user_2",))


class CompactionContractTests(unittest.TestCase):
    def test_draft_builds_rich_structured_state_with_jarvis_owned_ids(self) -> None:
        bundle = _bundle_from_draft(
            _draft(preserved_refs=("E1",)),
            (_record(record_id="user_1", role="user", content="Exact Ω text. "),),
        )

        categories = {entry.category for entry in bundle.state_entries}
        self.assertEqual(
            categories,
            {"constraint", "decision", "artifact", "open_loop", "uncertainty"},
        )
        self.assertTrue(all(entry.entry_id for entry in bundle.state_entries))
        artifact = next(entry for entry in bundle.state_entries if entry.category == "artifact")
        self.assertEqual(artifact.locator, "/workspace/project/example.py")
        self.assertTrue(artifact.needs_verification)
        self.assertEqual(bundle.preserved_records[0].content, "Exact Ω text. ")

    def test_duplicate_semantic_items_are_deduplicated_deterministically(self) -> None:
        draft = _draft()
        draft["constraints"] = ["Keep it.", "Keep it."]
        draft["episodes"] = [draft["episodes"][0], dict(draft["episodes"][0])]

        bundle = _bundle_from_draft(
            draft,
            (_record(record_id="user_1", role="user", content="Continue."),),
        )

        constraints = [
            entry for entry in bundle.state_entries if entry.category == "constraint"
        ]
        self.assertEqual(len(constraints), 1)
        self.assertEqual(len(bundle.episodes), 1)

    def test_manifest_keeps_complete_hash_lineage_without_cumulative_id_bloat(self) -> None:
        first = _bundle_from_draft(
            _draft(),
            (_record(record_id="record_1", role="user", content="First."),),
        )
        second = _bundle_from_draft(
            _draft(objective="Second."),
            (_record(record_id="record_2", role="user", content="Second."),),
            previous_bundle=first,
        )

        self.assertEqual(second.source_manifest.previous_bundle_id, first.bundle_id)
        self.assertEqual(second.source_manifest.delta_record_ids, ("record_2",))
        self.assertEqual(second.source_manifest.cumulative_record_count, 2)
        self.assertNotEqual(
            second.source_manifest.cumulative_content_sha256,
            first.source_manifest.cumulative_content_sha256,
        )

    def test_invalid_field_set_is_rejected_atomically(self) -> None:
        draft = _draft()
        draft["coverage"] = []
        events = build_compaction_source_events(
            (_record(record_id="user_1", role="user", content="Continue."),),
            generation=1,
        )
        manifest = build_source_manifest(
            generation=1,
            previous_bundle=None,
            source_events=events,
        )

        with self.assertRaises(CompactionContractError) as raised:
            apply_compaction_draft(
                draft,
                bundle_id="bundle_1",
                created_at="2026-08-21T00:00:00+00:00",
                source_manifest=manifest,
                source_events=events,
                previous_bundle=None,
            )

        self.assertEqual(raised.exception.issues[0].code, "invalid_field_set")

    def test_bundle_anchor_round_trip_and_replay_authority_are_deterministic(self) -> None:
        bundle = _bundle_from_draft(
            _draft(preserved_refs=("E1",)),
            (_record(record_id="user_1", role="user", content="Keep me exact."),),
        )
        anchor = build_compaction_bundle_record(session_id="new_session", bundle=bundle)

        loaded = load_compaction_bundle((anchor,))
        replay = compile_compaction_replay(bundle)

        self.assertEqual(loaded, CompactionBundle.from_dict(bundle.to_dict()))
        self.assertEqual(replay[0].role, "system")
        self.assertEqual(replay[0].kind, "history_boundary")
        self.assertEqual(
            [item.role for item in replay if item.role == "system"],
            ["system"],
        )
        self.assertEqual(
            [item.kind for item in replay],
            [
                "history_boundary",
                "preserved_message",
                "episode",
                "state_snapshot",
                "handover",
            ],
        )
        self.assertEqual(replay[1].source_record_ids, ("user_1",))

    def test_historical_schema_is_intentionally_not_accepted(self) -> None:
        bundle = _bundle_from_draft(
            _draft(),
            (_record(record_id="user_1", role="user", content="Continue."),),
        )
        payload = bundle.to_dict()
        payload["schema_version"] = 2

        with self.assertRaises(CompactionContractError) as raised:
            CompactionBundle.from_dict(payload)

        self.assertEqual(raised.exception.issues[0].code, "unsupported_compaction_schema")


if __name__ == "__main__":
    unittest.main()
