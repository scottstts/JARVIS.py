"""Tests for evidence-backed canonical session compaction."""

from __future__ import annotations

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
from jarvis.llm import LLMRequest, LLMResponse, LLMUsage, TextPart
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


def _response(payload: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        provider="openai",
        model="fake-compactor",
        text=json.dumps(payload, ensure_ascii=False),
        tool_calls=[],
        finish_reason="stop",
        usage=LLMUsage(input_tokens=12, output_tokens=8, total_tokens=20),
        response_id="resp_compact",
    )


def _draft(
    event_ids: list[str],
    *,
    marker: str = "Continue the task.",
    preserved_record_ids: list[str] | None = None,
    episode_id: str = "episode_1",
    state_operations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    preserved_ids = preserved_record_ids or []
    remaining_ids = [event_id for event_id in event_ids if event_id not in preserved_ids]
    episode_actions = (
        [
            {
                "action": "add",
                "episode_id": episode_id,
                "summary": marker,
                "source_ids": remaining_ids,
                "outcomes": ["The task has a verified continuation point."],
            }
        ]
        if remaining_ids
        else []
    )
    coverage: list[dict[str, Any]] = []
    if preserved_ids:
        coverage.append(
            {
                "source_event_ids": preserved_ids,
                "disposition": "preserved",
                "target_ids": preserved_ids,
                "reason": "Exact user wording remains authoritative.",
            }
        )
    if remaining_ids:
        coverage.append(
            {
                "source_event_ids": remaining_ids,
                "disposition": "episode",
                "target_ids": [episode_id],
                "reason": "Represented in the chronological episode.",
            }
        )
    return {
        "objective": {
            "summary": marker,
            "evidence_event_ids": [event_ids[0]],
        },
        "preserved_actions": [
            {
                "action": "add",
                "record_id": record_id,
                "reason": "Exact user wording remains authoritative.",
                "evidence_event_ids": [record_id],
            }
            for record_id in preserved_ids
        ],
        "episode_actions": episode_actions,
        "state_operations": state_operations or [],
        "handover": {
            "current_focus": marker,
            "next_actions": ["Resume from the verified state."],
            "do_not_repeat": [],
            "verification_needed": [],
            "evidence_event_ids": [event_ids[-1]],
        },
        "coverage": coverage,
    }


class _QueuedCompactionLLMService:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.payloads:
            raise AssertionError("Unexpected extra compaction request.")
        return _response(self.payloads.pop(0))


def _request_json(request: LLMRequest) -> dict[str, Any]:
    text = "\n".join(
        part.text
        for message in request.messages
        if message.role == "user"
        for part in message.parts
        if isinstance(part, TextPart)
    )
    return json.loads(text[text.index("{") :])


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

    def test_source_events_preserve_time_turn_and_tool_causality(self) -> None:
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
        self.assertEqual(events[0].generation, 3)
        self.assertNotIn("raw_arguments", json.dumps(events[0].to_dict()))
        self.assertNotIn("provider_metadata", json.dumps(events[0].to_dict()))


class ContextCompactorTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_preserved_message_is_copied_by_jarvis_without_normalization(self) -> None:
        exact_text = "Use Jarvis’s exact punctuation and trailing space. "
        draft = _draft(["user_1"], preserved_record_ids=["user_1"])
        service = _QueuedCompactionLLMService(
            [draft, {"valid": True, "issues": []}]
        )
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
            provider="openai",
        )

        outcome = await compactor.compact(
            (_record(record_id="user_1", role="user", content=exact_text),)
        )

        preserved = outcome.bundle.preserved_records[0]
        self.assertEqual(preserved.content, exact_text)
        replay = next(item for item in outcome.items if item.kind == "preserved_message")
        self.assertEqual(replay.content, exact_text)
        self.assertEqual(replay.role, "user")
        self.assertTrue(replay.exact_copy)
        self.assertEqual(replay.source_record_ids, ("user_1",))
        self.assertEqual(outcome.items[0].kind, "history_boundary")
        self.assertEqual(
            [item.role for item in outcome.items if item.kind != "history_boundary"],
            ["user", "assistant", "assistant"],
        )

    async def test_structural_failure_gets_targeted_repair_instead_of_silent_drop(self) -> None:
        invalid = _draft(["user_1"], preserved_record_ids=["missing_record"])
        repaired = _draft(["user_1"], preserved_record_ids=["user_1"])
        service = _QueuedCompactionLLMService(
            [invalid, repaired, {"valid": True, "issues": []}]
        )
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )

        outcome = await compactor.compact(
            (_record(record_id="user_1", role="user", content="Keep this."),)
        )

        self.assertEqual(outcome.repair_count, 1)
        repair_input = _request_json(service.requests[1])
        self.assertEqual(repair_input["mode"], "repair")
        self.assertTrue(repair_input["validation_issues"])
        self.assertEqual(outcome.bundle.preserved_records[0].record_id, "user_1")

    async def test_semantic_verifier_rejects_then_accepts_repair(self) -> None:
        first = _draft(["user_1"], marker="Incorrect completion claim.")
        repaired = _draft(["user_1"], marker="Task remains in progress.")
        service = _QueuedCompactionLLMService(
            [
                first,
                {
                    "valid": False,
                    "issues": [
                        {
                            "code": "false_completion",
                            "message": "Do not claim completion.",
                            "source_event_ids": ["user_1"],
                        }
                    ],
                },
                repaired,
                {"valid": True, "issues": []},
            ]
        )
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )

        outcome = await compactor.compact(
            (_record(record_id="user_1", role="user", content="Keep working."),)
        )

        self.assertEqual(outcome.bundle.objective.summary, "Task remains in progress.")
        self.assertEqual(
            [trace.phase for trace in outcome.call_traces],
            ["generate", "verify", "repair", "verify"],
        )
        self.assertEqual(outcome.repair_count, 1)

    async def test_malformed_verifier_output_is_retried_without_regenerating_bundle(self) -> None:
        draft = _draft(["user_1"])
        service = _QueuedCompactionLLMService(
            [draft, {"unexpected": "shape"}, {"valid": True, "issues": []}]
        )
        compactor = ContextCompactor(
            llm_service=service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )

        outcome = await compactor.compact(
            (_record(record_id="user_1", role="user", content="Continue."),)
        )

        self.assertEqual(
            [trace.phase for trace in outcome.call_traces],
            ["generate", "verify", "verify"],
        )
        retry_input = _request_json(service.requests[2])
        self.assertTrue(retry_input["verifier_contract_retry"])
        self.assertIn("unexpected", retry_input["previous_invalid_verifier_output"])

    async def test_later_generation_merges_only_delta_and_retains_verified_history(self) -> None:
        first_draft = _draft(["user_1", "assistant_1"], marker="Initial work.")
        first_service = _QueuedCompactionLLMService(
            [first_draft, {"valid": True, "issues": []}]
        )
        first_compactor = ContextCompactor(
            llm_service=first_service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )
        first = await first_compactor.compact(
            (
                _record(record_id="user_1", role="user", content="Build it."),
                _record(record_id="assistant_1", role="assistant", content="Started."),
            )
        )
        original_episode = first.bundle.episodes[0]

        second_draft = _draft(
            ["user_2"],
            marker="Continue with the new correction.",
            episode_id="episode_2",
        )
        second_service = _QueuedCompactionLLMService(
            [second_draft, {"valid": True, "issues": []}]
        )
        second_compactor = ContextCompactor(
            llm_service=second_service,  # type: ignore[arg-type]
            context_policy=ContextPolicySettings(context_window_tokens=100_000),
        )
        second = await second_compactor.compact(
            (_record(record_id="user_2", role="user", content="Use the correction."),),
            previous_bundle=first.bundle,
        )

        self.assertEqual(second.bundle.generation, 2)
        self.assertEqual(second.bundle.episodes[0], original_episode)
        self.assertEqual([item.episode_id for item in second.bundle.episodes], ["episode_1", "episode_2"])
        generator_input = _request_json(second_service.requests[0])
        self.assertEqual(
            [event["event_id"] for event in generator_input["delta_events"]],
            ["user_2"],
        )
        self.assertEqual(
            second.bundle.source_manifest.evidence_event_ids,
            ("user_1", "assistant_1", "user_2"),
        )


class CompactionContractTests(unittest.TestCase):
    def test_ten_generations_retain_exact_records_and_original_episode_without_drift(self) -> None:
        first_events = build_compaction_source_events(
            (
                _record(record_id="user_1", role="user", content="Exact constraint: use Ω. "),
                _record(
                    record_id="assistant_1",
                    role="assistant",
                    content="Initial implementation began.",
                ),
            ),
            generation=1,
        )
        first_manifest = build_source_manifest(
            generation=1,
            previous_bundle=None,
            source_events=first_events,
        )
        bundle = apply_compaction_draft(
            _draft(
                ["user_1", "assistant_1"],
                marker="Initial objective.",
                preserved_record_ids=["user_1"],
            ),
            bundle_id="bundle_1",
            created_at="2026-08-21T00:00:00+00:00",
            source_manifest=first_manifest,
            source_events=first_events,
            previous_bundle=None,
        )
        exact_record = bundle.preserved_records[0]
        original_episode = bundle.episodes[0]

        for generation in range(2, 11):
            event_id = f"user_{generation}"
            events = build_compaction_source_events(
                (
                    _record(
                        record_id=event_id,
                        role="user",
                        content=f"Generation {generation} update.",
                        session_id=f"session_{generation}",
                    ),
                ),
                generation=generation,
            )
            manifest = build_source_manifest(
                generation=generation,
                previous_bundle=bundle,
                source_events=events,
            )
            bundle = apply_compaction_draft(
                _draft(
                    [event_id],
                    marker=f"Objective at generation {generation}.",
                    episode_id=f"episode_{generation}",
                ),
                bundle_id=f"bundle_{generation}",
                created_at=f"2026-08-21T00:{generation:02d}:00+00:00",
                source_manifest=manifest,
                source_events=events,
                previous_bundle=bundle,
            )

        self.assertEqual(bundle.generation, 10)
        self.assertEqual(bundle.preserved_records, (exact_record,))
        self.assertEqual(bundle.episodes[0], original_episode)
        self.assertEqual(bundle.preserved_records[0].content, "Exact constraint: use Ω. ")
        self.assertEqual(
            bundle.source_manifest.evidence_event_ids,
            (
                "user_1",
                "assistant_1",
                *(f"user_{generation}" for generation in range(2, 11)),
            ),
        )

    def test_coverage_must_account_for_every_delta_event_exactly_once(self) -> None:
        records = (
            _record(record_id="user_1", role="user", content="First."),
            _record(record_id="assistant_1", role="assistant", content="Second."),
        )
        events = build_compaction_source_events(records, generation=1)
        manifest = build_source_manifest(
            generation=1,
            previous_bundle=None,
            source_events=events,
        )
        draft = _draft(["user_1", "assistant_1"])
        draft["coverage"][0]["source_event_ids"] = ["user_1"]

        with self.assertRaisesRegex(CompactionContractError, "missing_source_coverage"):
            apply_compaction_draft(
                draft,
                bundle_id="bundle_1",
                created_at="2026-08-21T00:00:00+00:00",
                source_manifest=manifest,
                source_events=events,
                previous_bundle=None,
            )

    def test_artifact_locator_must_be_exactly_grounded_in_delta_evidence(self) -> None:
        events = build_compaction_source_events(
            (
                _record(
                    record_id="tool_1",
                    role="tool",
                    content="Wrote /workspace/output/report.json successfully.",
                    metadata={"call_id": "call_1", "tool_name": "bash", "ok": True},
                ),
            ),
            generation=1,
        )
        manifest = build_source_manifest(
            generation=1,
            previous_bundle=None,
            source_events=events,
        )
        artifact_operation = {
            "action": "add",
            "entry_id": "artifact_report",
            "category": "artifact",
            "summary": "Generated report artifact.",
            "evidence_event_ids": ["tool_1"],
            "supersedes_entry_id": None,
            "locator": "/workspace/output/wrong.json",
            "last_observed_state": "written successfully",
            "needs_verification": True,
            "blocker": None,
            "next_action": None,
        }
        draft = _draft(["tool_1"], state_operations=[artifact_operation])

        with self.assertRaisesRegex(
            CompactionContractError,
            "artifact_locator_not_in_evidence",
        ):
            apply_compaction_draft(
                draft,
                bundle_id="bundle_1",
                created_at="2026-08-21T00:00:00+00:00",
                source_manifest=manifest,
                source_events=events,
                previous_bundle=None,
            )

        artifact_operation["locator"] = "/workspace/output/report.json"
        bundle = apply_compaction_draft(
            draft,
            bundle_id="bundle_1",
            created_at="2026-08-21T00:00:00+00:00",
            source_manifest=manifest,
            source_events=events,
            previous_bundle=None,
        )
        artifact = bundle.state_entries[0]
        self.assertEqual(artifact.locator, "/workspace/output/report.json")
        self.assertTrue(artifact.needs_verification)

    def test_state_supersession_keeps_lineage_and_marks_prior_entry(self) -> None:
        first_events = build_compaction_source_events(
            (_record(record_id="user_1", role="user", content="Use blue."),),
            generation=1,
        )
        first_manifest = build_source_manifest(
            generation=1,
            previous_bundle=None,
            source_events=first_events,
        )
        first_operation = {
            "action": "add",
            "entry_id": "constraint_color_blue",
            "category": "constraint",
            "summary": "Use blue.",
            "evidence_event_ids": ["user_1"],
            "supersedes_entry_id": None,
            "locator": None,
            "last_observed_state": None,
            "needs_verification": None,
            "blocker": None,
            "next_action": None,
        }
        first_draft = _draft(
            ["user_1"],
            preserved_record_ids=["user_1"],
            state_operations=[first_operation],
        )
        first_bundle = apply_compaction_draft(
            first_draft,
            bundle_id="bundle_1",
            created_at="2026-08-21T00:00:00+00:00",
            source_manifest=first_manifest,
            source_events=first_events,
            previous_bundle=None,
        )

        second_events = build_compaction_source_events(
            (_record(record_id="user_2", role="user", content="Correction: use green."),),
            generation=2,
        )
        second_manifest = build_source_manifest(
            generation=2,
            previous_bundle=first_bundle,
            source_events=second_events,
        )
        supersede = {
            "action": "supersede",
            "entry_id": "constraint_color_green",
            "category": "constraint",
            "summary": "Use green.",
            "evidence_event_ids": ["user_2"],
            "supersedes_entry_id": "constraint_color_blue",
            "locator": None,
            "last_observed_state": None,
            "needs_verification": None,
            "blocker": None,
            "next_action": None,
        }
        second_draft = _draft(
            ["user_2"],
            marker="Apply the corrected green constraint.",
            preserved_record_ids=["user_2"],
            state_operations=[supersede],
        )
        second_bundle = apply_compaction_draft(
            second_draft,
            bundle_id="bundle_2",
            created_at="2026-08-21T00:01:00+00:00",
            source_manifest=second_manifest,
            source_events=second_events,
            previous_bundle=first_bundle,
        )

        state = {entry.entry_id: entry for entry in second_bundle.state_entries}
        self.assertEqual(state["constraint_color_blue"].status, "superseded")
        self.assertEqual(state["constraint_color_green"].status, "active")
        self.assertEqual(
            state["constraint_color_green"].supersedes_entry_ids,
            ("constraint_color_blue",),
        )

    def test_bundle_anchor_round_trip_and_replay_have_deterministic_authority(self) -> None:
        events = build_compaction_source_events(
            (_record(record_id="user_1", role="user", content="Keep me exact."),),
            generation=1,
        )
        manifest = build_source_manifest(
            generation=1,
            previous_bundle=None,
            source_events=events,
        )
        bundle = apply_compaction_draft(
            _draft(["user_1"], preserved_record_ids=["user_1"]),
            bundle_id="bundle_1",
            created_at="2026-08-21T00:00:00+00:00",
            source_manifest=manifest,
            source_events=events,
            previous_bundle=None,
        )
        anchor = build_compaction_bundle_record(session_id="new_session", bundle=bundle)

        loaded = load_compaction_bundle((anchor,))
        replay = compile_compaction_replay(bundle)

        self.assertEqual(loaded, CompactionBundle.from_dict(bundle.to_dict()))
        self.assertEqual([item.role for item in replay], ["system", "user", "assistant", "assistant"])
        self.assertEqual([item.kind for item in replay], [
            "history_boundary",
            "preserved_message",
            "state_snapshot",
            "handover",
        ])
        self.assertEqual(
            [item.role for item in replay if item.role == "system"],
            ["system"],
        )
