"""Task identity, contract, and machine-evidence regression tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from jarvis.core import AgentLoop
from jarvis.core.task_contract import (
    TaskRequirement,
    build_assignment_task_contract,
    build_task_contract,
)
from jarvis.core.tool_safety import ToolSafetyTracker
from jarvis.llm import LLMResponse, ToolCall
from jarvis.storage import SessionStorage
from jarvis.tools import ToolExecutionResult
from tests.helpers import build_core_settings


def _call(name: str, arguments: dict[str, object], call_id: str) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments="",
    )


class TaskContractTests(unittest.TestCase):
    def test_contract_extracts_explicit_machine_evidence_requirements(self) -> None:
        contract = build_task_contract(
            task_id="task-1",
            origin_turn_id="turn-1",
            user_text=(
                "You must use subagents. The project must contain at least 50k LOC. "
                "All tests, lint, typecheck, and build must pass. "
                "Visually inspect the finished application."
            ),
        )

        self.assertEqual(
            {item.evidence_kind for item in contract.requirements},
            {
                "delegation",
                "source_line_count",
                "verification_gate",
                "visual_inspection",
            },
        )
        self.assertIn("user_message_sha256", contract.render())

    def test_assignment_contract_uses_all_assignment_sentences(self) -> None:
        contract = build_assignment_task_contract(
            task_id="child-task",
            origin_turn_id="child-turn",
            assignment_texts=(
                "Implement the parser.\n- Run tests and lint.",
                "Do not change the public API.",
                "Return a concise report.",
            ),
        )

        self.assertEqual(
            [item.criterion for item in contract.requirements],
            [
                "Implement the parser.",
                "Run tests and lint.",
                "Do not change the public API.",
                "Return a concise report.",
            ],
        )

    def test_superseding_clarification_and_side_query_preserve_active_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            session = storage.create_session()
            loop = AgentLoop(
                llm_service=SimpleNamespace(),  # type: ignore[arg-type]
                settings=settings,
                storage=storage,
            )

            first = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-one",
                user_text="You must build the application fully.",
            )
            storage.update_session(
                session.session_id,
                pending_interruption_notice=True,
                pending_interruption_notice_reason="superseded_by_user_message",
            )
            clarified = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-two",
                user_text="I want you to decide when and how to use subagents.",
            )
            side_query = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-three",
                user_text="Give me a status update.",
            )
            loop._begin_turn(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                turn_id="task-three",
            )
            loop._persist_successful_turn(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                turn_id="task-three",
                response=LLMResponse(
                    provider="fake",
                    model="fake",
                    text="Still working.",
                    tool_calls=[],
                    finish_reason="stop",
                    usage=None,
                ),
                estimated_input_tokens=1,
            )
            resumed = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-four",
                user_text="Continue and make sure all tests pass.",
            )

            self.assertIsNotNone(first)
            self.assertIsNotNone(clarified)
            self.assertIsNotNone(side_query)
            self.assertIsNotNone(resumed)
            assert first is not None and clarified is not None and resumed is not None
            assert side_query is not None
            self.assertEqual(first.task_id, clarified.task_id)
            self.assertNotEqual(side_query.task_id, first.task_id)
            self.assertEqual(resumed.task_id, first.task_id)
            self.assertTrue(
                any("subagents" in item.criterion for item in clarified.requirements)
            )
            self.assertTrue(
                any("tests pass" in item.criterion for item in resumed.requirements)
            )
            refreshed = storage.get_session(session.session_id)
            assert refreshed is not None
            self.assertEqual(
                refreshed.backend_state.get("active_tool_task_id"),
                "task-one",
            )
            self.assertIsNotNone(storage.load_tool_task_state("task-one"))

    def test_explicit_task_replacement_does_not_restore_previous_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            session = storage.create_session()
            loop = AgentLoop(
                llm_service=SimpleNamespace(),  # type: ignore[arg-type]
                settings=settings,
                storage=storage,
            )

            first = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-one",
                user_text="You must build the application fully.",
            )
            replacement = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-two",
                user_text="New task: tell me what time it is.",
            )

            assert first is not None and replacement is not None
            self.assertNotEqual(first.task_id, replacement.task_id)
            state = storage.load_tool_task_state(replacement.task_id)
            assert state is not None
            self.assertNotIn("restore_parent_task_id", state)

    def test_explicit_requirement_replacement_does_not_merge_superseded_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            session = storage.create_session()
            loop = AgentLoop(
                llm_service=SimpleNamespace(),  # type: ignore[arg-type]
                settings=settings,
                storage=storage,
            )

            first = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-one",
                user_text="You must use subagents. All tests must pass.",
            )
            storage.update_session(
                session.session_id,
                pending_interruption_notice=True,
                pending_interruption_notice_reason="superseded_by_user_message",
            )
            replacement = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-two",
                user_text="No need I mean.",
            )

            assert first is not None and replacement is not None
            self.assertEqual(first.task_id, replacement.task_id)
            self.assertEqual(len(replacement.requirements), 1)
            self.assertIn("subagents", replacement.requirements[0].criterion)

    def test_contract_acceptance_requires_runtime_observed_evidence(self) -> None:
        requirements = (
            TaskRequirement("delegation", "You must use subagents.", "delegation"),
            TaskRequirement(
                "quality-gates",
                "All tests, lint, typecheck, and build must pass.",
                "verification_gate",
            ),
            TaskRequirement(
                "line-count",
                "The project must contain at least 50k LOC.",
                "source_line_count",
            ),
            TaskRequirement(
                "visual",
                "Visually inspect the finished application.",
                "visual_inspection",
            ),
        )
        tracker = ToolSafetyTracker(_actor_kind="subagent")
        tracker.seed_contract_requirements(requirements)
        tracker.record(
            _call("file_write", {"path": "src/main.ts"}, "write"),
            ToolExecutionResult(
                call_id="write",
                name="file_write",
                ok=True,
                content="changed",
                metadata={"changed": True},
            ),
        )
        tracker.record(
            _call("subagent_invoke", {"task_label": "review"}, "subagent"),
            ToolExecutionResult(
                call_id="subagent",
                name="subagent_invoke",
                ok=True,
                content="started",
                metadata={},
            ),
        )
        tracker.record(
            _call("view_image", {"path": "renders/final.png"}, "visual"),
            ToolExecutionResult(
                call_id="visual",
                name="view_image",
                ok=True,
                content="inspected",
                metadata={"path": "/workspace/renders/final.png"},
            ),
        )
        tracker.record(
            _call("acceptance_run", {}, "gates"),
            ToolExecutionResult(
                call_id="gates",
                name="acceptance_run",
                ok=True,
                content="passed",
                metadata={
                    "acceptance_run": {
                        "passed": True,
                        "revision_paths": ["."],
                        "workspace_revision_after": "revision-1",
                        "gates": [
                            {"gate_id": "tests", "command": "npm test", "passed": True},
                            {"gate_id": "lint", "command": "npm run lint", "passed": True},
                            {
                                "gate_id": "typecheck",
                                "command": "npm run typecheck",
                                "passed": True,
                            },
                            {"gate_id": "build", "command": "npm run build", "passed": True},
                            {
                                "gate_id": "loc",
                                "command": "",
                                "passed": True,
                                "source_line_count": {
                                    "line_count": 50_100,
                                    "minimum": 50_000,
                                    "file_count": 100,
                                },
                            },
                        ]
                    },
                    "changed": False,
                },
            ),
        )
        checks = [
            {
                "item_id": f"criterion-{index}",
                "criterion": requirement.criterion,
                "required": True,
                "outcome": "passed",
                "evidence_kind": (
                    "artifact_inspection"
                    if requirement.item_id == "visual"
                    else "runtime_observation"
                ),
                "artifact_paths": (
                    ["/workspace/renders/final.png"]
                    if requirement.item_id == "visual"
                    else []
                ),
            }
            for index, requirement in enumerate(requirements, start=1)
        ]
        tracker.record(
            _call("acceptance_record", {}, "record"),
            ToolExecutionResult(
                call_id="record",
                name="acceptance_record",
                ok=True,
                content="recorded",
                metadata={
                    "acceptance_ledger": {
                        "workspace_revision_verified": True,
                        "workspace_revision": "revision-1",
                        "revision_paths": ["."],
                        "complete": True,
                        "checks": checks,
                    }
                },
            ),
        )

        self.assertFalse(tracker.unverified_workspace_mutation)

        missing_delegation = ToolSafetyTracker(_actor_kind="subagent")
        missing_delegation.seed_contract_requirements(requirements[:1])
        missing_delegation.record(
            _call("file_write", {"path": "src/main.ts"}, "write"),
            ToolExecutionResult(
                call_id="write",
                name="file_write",
                ok=True,
                content="changed",
                metadata={"changed": True},
            ),
        )
        missing_delegation.record(
            _call("acceptance_run", {}, "gates"),
            ToolExecutionResult(
                call_id="gates",
                name="acceptance_run",
                ok=True,
                content="passed",
                metadata={
                    "acceptance_run": {
                        "passed": True,
                        "revision_paths": ["."],
                        "workspace_revision_after": "revision-1",
                        "gates": [
                            {"gate_id": "tests", "command": "pytest", "passed": True}
                        ]
                    },
                    "changed": False,
                },
            ),
        )
        missing_delegation.record(
            _call("acceptance_record", {}, "record"),
            ToolExecutionResult(
                call_id="record",
                name="acceptance_record",
                ok=True,
                content="recorded",
                metadata={
                    "acceptance_ledger": {
                        "workspace_revision_verified": True,
                        "workspace_revision": "revision-1",
                        "revision_paths": ["."],
                        "complete": True,
                        "checks": [checks[0]],
                    }
                },
            ),
        )
        self.assertTrue(missing_delegation.unverified_workspace_mutation)

    def test_contract_items_require_distinct_corresponding_ledger_checks(self) -> None:
        tracker = ToolSafetyTracker(_actor_kind="subagent")
        tracker.seed_contract_requirements(
            (
                TaskRequirement("first", "Implement the parser.", "general"),
                TaskRequirement("second", "Document the parser.", "general"),
            )
        )
        tracker.record(
            _call("file_write", {"path": "src/parser.py"}, "write"),
            ToolExecutionResult(
                call_id="write",
                name="file_write",
                ok=True,
                content="changed",
                metadata={"changed": True},
            ),
        )
        tracker.record(
            _call("acceptance_run", {}, "gates"),
            ToolExecutionResult(
                call_id="gates",
                name="acceptance_run",
                ok=True,
                content="passed",
                metadata={
                    "acceptance_run": {
                        "passed": True,
                        "revision_paths": ["src"],
                        "workspace_revision_after": "revision-1",
                        "gates": [
                            {"gate_id": "tests", "command": "pytest", "passed": True}
                        ],
                    }
                },
            ),
        )
        tracker.record(
            _call("acceptance_record", {}, "record"),
            ToolExecutionResult(
                call_id="record",
                name="acceptance_record",
                ok=True,
                content="recorded",
                metadata={
                    "acceptance_ledger": {
                        "workspace_revision_verified": True,
                        "workspace_revision": "revision-1",
                        "revision_paths": ["src"],
                        "complete": True,
                        "checks": [
                            {
                                "item_id": "parser-implemented",
                                "criterion": "Implement the parser.",
                                "required": True,
                                "outcome": "passed",
                                "evidence_kind": "runtime_observation",
                            }
                        ],
                    }
                },
            ),
        )

        self.assertTrue(tracker.unverified_workspace_mutation)

    def test_legacy_run_scope_without_scoped_gates_fails_closed(self) -> None:
        tracker = ToolSafetyTracker(_actor_kind="subagent")
        tracker.seed_contract_requirements(
            (
                TaskRequirement(
                    "quality",
                    "All tests must pass.",
                    "verification_gate",
                ),
            )
        )
        tracker.record(
            _call("file_write", {"path": "src/app.py"}, "write"),
            ToolExecutionResult(
                call_id="write",
                name="file_write",
                ok=True,
                content="changed",
                metadata={"changed": True},
            ),
        )
        tracker.record(
            _call("acceptance_run", {}, "gates"),
            ToolExecutionResult(
                call_id="gates",
                name="acceptance_run",
                ok=True,
                content="passed",
                metadata={
                    "acceptance_run": {
                        "passed": True,
                        "revision_paths": ["src"],
                        "workspace_revision_after": "revision-1",
                        "gates": [
                            {"gate_id": "tests", "command": "pytest", "passed": True}
                        ],
                    }
                },
            ),
        )
        legacy_state = tracker.to_state()
        legacy_state["passed_acceptance_run_scopes"]["gates"].pop(
            "passed_gates",
            None,
        )
        restored = ToolSafetyTracker.from_state(legacy_state, actor_kind="subagent")
        restored.record(
            _call("acceptance_record", {}, "record"),
            ToolExecutionResult(
                call_id="record",
                name="acceptance_record",
                ok=True,
                content="recorded",
                metadata={
                    "acceptance_ledger": {
                        "workspace_revision_verified": True,
                        "workspace_revision": "revision-1",
                        "revision_paths": ["src"],
                        "complete": True,
                        "checks": [
                            {
                                "item_id": "quality-check",
                                "criterion": "All tests must pass.",
                                "required": True,
                                "outcome": "passed",
                                "evidence_kind": "test_result",
                            }
                        ],
                    }
                },
            ),
        )

        self.assertTrue(restored.unverified_workspace_mutation)

    def test_tool_task_sidecar_skips_identical_rewrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp))
            state = {"schema_version": 1, "rounds": 4, "tracker": {}}

            self.assertTrue(storage.write_tool_task_state("task-1", state))
            self.assertFalse(storage.write_tool_task_state("task-1", state))
            self.assertEqual(storage.load_tool_task_state("task-1"), state)

    def test_legacy_cumulative_round_count_migrates_without_poisoning_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            session = storage.create_session()
            loop = AgentLoop(
                llm_service=SimpleNamespace(),  # type: ignore[arg-type]
                settings=settings,
                storage=storage,
            )
            contract = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="legacy-task",
                user_text="You must finish the project.",
            )
            assert contract is not None
            state = storage.load_tool_task_state(contract.task_id)
            assert state is not None
            tracker = ToolSafetyTracker.from_state(state["tracker"])
            storage.write_tool_task_state(
                contract.task_id,
                {
                    **state,
                    "schema_version": 1,
                    "rounds": 338,
                    "tracker": tracker.to_state(),
                },
            )

            rounds, restored_tracker = loop._load_tool_task_state(  # pyright: ignore[reportPrivateUsage]
                session.session_id
            )
            self.assertEqual(rounds, 0)
            loop._persist_tool_task_state(  # pyright: ignore[reportPrivateUsage]
                session.session_id,
                rounds=rounds,
                tracker=restored_tracker,
            )
            migrated = storage.load_tool_task_state(contract.task_id)
            assert migrated is not None
            self.assertEqual(migrated["schema_version"], 2)
            self.assertEqual(migrated["stalled_rounds"], 0)

    def test_distinct_runtime_progress_resets_rounds_once(self) -> None:
        tracker = ToolSafetyTracker()
        metadata = {
            "bash_job_progress_update": True,
            "detached_bash_job_ids": ["job-1"],
            "bash_job_notice_kinds": ["bash_job_needs_attention"],
            "bash_job_running_ids": ["job-1"],
            "bash_job_terminal_ids": [],
            "recommended_action": "inspect",
            "bash_job_progress_fingerprints": [
                "job-1:bash_job_needs_attention:running:0:0:0:0:"
            ],
        }

        self.assertTrue(
            tracker.record_runtime_progress(content="runtime=300", metadata=metadata)
        )
        epoch = tracker.progress_epoch
        self.assertFalse(
            tracker.record_runtime_progress(content="runtime=600", metadata=metadata)
        )
        self.assertEqual(tracker.progress_epoch, epoch)
        self.assertTrue(
            tracker.record_runtime_progress(
                content="output grew",
                metadata={
                    **metadata,
                    "bash_job_notice_kinds": ["bash_job_output_grew"],
                    "recommended_action": "continue",
                    "bash_job_progress_fingerprints": [
                        "job-1:bash_job_output_grew:running:8192:0:0:0:"
                    ],
                },
            )
        )
        self.assertEqual(tracker.progress_epoch, epoch + 1)
        self.assertTrue(
            tracker.record_runtime_progress(
                content="finished",
                metadata={
                    **metadata,
                    "bash_job_notice_kinds": ["bash_job_completed"],
                    "bash_job_running_ids": [],
                    "bash_job_terminal_ids": ["job-1"],
                    "recommended_action": "finalize",
                    "bash_job_progress_fingerprints": [
                        "job-1:bash_job_completed:finished:8192:0:0:0:0"
                    ],
                },
            )
        )
        self.assertEqual(tracker.progress_epoch, epoch + 2)

    def test_persisted_orchestrator_note_resets_stalled_rounds_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            session = storage.create_session()
            loop = AgentLoop(
                llm_service=SimpleNamespace(),  # type: ignore[arg-type]
                settings=settings,
                storage=storage,
            )
            contract = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-one",
                user_text="You must complete the verification.",
            )
            assert contract is not None
            _rounds, tracker = loop._load_tool_task_state(  # pyright: ignore[reportPrivateUsage]
                session.session_id
            )
            loop._persist_tool_task_state(  # pyright: ignore[reportPrivateUsage]
                session.session_id,
                rounds=4,
                tracker=tracker,
            )
            metadata = {
                "bash_job_progress_update": True,
                "detached_bash_job_ids": ["job-1"],
                "bash_job_notice_kinds": ["bash_job_output_grew"],
                "bash_job_running_ids": ["job-1"],
                "bash_job_terminal_ids": [],
                "recommended_action": "continue",
                "bash_job_progress_fingerprints": [
                    "job-1:bash_job_output_grew:running:8192:0:0:0:"
                ],
            }

            self.assertTrue(
                loop.append_system_note(
                    "Detached bash output grew.",
                    session_id=session.session_id,
                    metadata=metadata,
                )
            )
            rounds, progressed_tracker = loop._load_tool_task_state(  # pyright: ignore[reportPrivateUsage]
                session.session_id
            )
            self.assertEqual(rounds, 0)
            self.assertGreater(progressed_tracker.progress_epoch, tracker.progress_epoch)

            loop._persist_tool_task_state(  # pyright: ignore[reportPrivateUsage]
                session.session_id,
                rounds=3,
                tracker=progressed_tracker,
            )
            self.assertTrue(
                loop.append_system_note(
                    "Duplicate detached bash update.",
                    session_id=session.session_id,
                    metadata=metadata,
                )
            )
            duplicate_rounds, _duplicate_tracker = loop._load_tool_task_state(  # pyright: ignore[reportPrivateUsage]
                session.session_id
            )
            self.assertEqual(duplicate_rounds, 3)

    def test_changed_test_artifact_adds_independent_review_requirement(self) -> None:
        tracker = ToolSafetyTracker()
        tracker.record(
            _call("file_write", {"path": "tests/vehicle.test.ts"}, "test-write"),
            ToolExecutionResult(
                call_id="test-write",
                name="file_write",
                ok=True,
                content="changed",
                metadata={"changed": True, "path": "/workspace/tests/vehicle.test.ts"},
            ),
        )

        self.assertIn("system-test-change-review", "\n".join(tracker.checkpoint_lines()))

        tracker.record(
            _call(
                "subagent_invoke",
                {
                    "task_label": "Review implementation",
                    "instructions": "Review src/vehicle.ts.",
                },
                "wrong-review",
            ),
            ToolExecutionResult(
                call_id="wrong-review",
                name="subagent_invoke",
                ok=True,
                content="started",
                metadata={"subagent_id": "wrong-agent"},
            ),
        )
        self.assertEqual(tracker.to_state()["test_review_subagent_ids"], [])

        tracker.record(
            _call(
                "subagent_invoke",
                {
                    "task_label": "Independent test review",
                    "instructions": "Review tests/vehicle.test.ts and report semantic issues.",
                },
                "test-review",
            ),
            ToolExecutionResult(
                call_id="test-review",
                name="subagent_invoke",
                ok=True,
                content="started",
                metadata={"subagent_id": "review-agent"},
            ),
        )
        tracker.record(
            _call("subagent_monitor", {"agent": "review-agent"}, "monitor"),
            ToolExecutionResult(
                call_id="monitor",
                name="subagent_monitor",
                ok=True,
                content="completed",
                metadata={
                    "subagents": [
                        {
                            "subagent_id": "review-agent",
                            "status": "completed",
                            "report_complete": True,
                        }
                    ]
                },
            ),
        )
        state = tracker.to_state()
        self.assertEqual(state["test_review_subagent_ids"], ["review-agent"])
        self.assertEqual(
            state["completed_test_review_subagent_ids"],
            ["review-agent"],
        )

        tracker.record(
            _call("acceptance_run", {}, "tests"),
            ToolExecutionResult(
                call_id="tests",
                name="acceptance_run",
                ok=True,
                content="passed",
                metadata={
                    "acceptance_run": {
                        "passed": True,
                        "revision_paths": ["."],
                        "workspace_revision_after": "revision-1",
                        "gates": [
                            {"gate_id": "tests", "command": "pytest", "passed": True}
                        ]
                    },
                    "changed": False,
                },
            ),
        )
        tracker.record(
            _call("acceptance_record", {}, "record"),
            ToolExecutionResult(
                call_id="record",
                name="acceptance_record",
                ok=True,
                content="recorded",
                metadata={
                    "acceptance_ledger": {
                        "workspace_revision_verified": True,
                        "workspace_revision": "revision-1",
                        "revision_paths": ["."],
                        "complete": True,
                        "checks": [
                            {
                                "item_id": "system-test-change-review",
                                "required": True,
                                "outcome": "passed",
                                "evidence_kind": "artifact_inspection",
                                "source_tool_call_ids": ["tests"],
                                "artifact_paths": ["tests/vehicle.test.ts"],
                            }
                        ],
                    }
                },
            ),
        )
        self.assertFalse(tracker.unverified_workspace_mutation)

    def test_subagent_test_changes_transfer_review_obligation_to_main_tracker(self) -> None:
        child_tracker = ToolSafetyTracker(_actor_kind="subagent")
        child_tracker.record(
            _call("file_write", {"path": "tests/vehicle.test.ts"}, "child-write"),
            ToolExecutionResult(
                call_id="child-write",
                name="file_write",
                ok=True,
                content="changed",
                metadata={
                    "changed": True,
                    "path": "/workspace/tests/vehicle.test.ts",
                },
            ),
        )

        self.assertNotIn(
            "system-test-change-review",
            "\n".join(child_tracker.checkpoint_lines()),
        )
        self.assertTrue(child_tracker.unverified_workspace_mutation)

        main_tracker = ToolSafetyTracker()
        made_progress = main_tracker.record_runtime_progress(
            content="Vehicle builder completed.",
            metadata={
                "subagent_progress_update": True,
                "subagent_id": "builder-agent",
                "subagent_notice_kind": "subagent_completed",
                "recommended_action": "finalize",
                "changed_test_artifact_paths": [
                    "tests/vehicle.test.ts",
                    "src/physics/__tests__/physics.test.ts",
                ],
            },
        )

        checkpoint = "\n".join(main_tracker.checkpoint_lines())
        self.assertTrue(made_progress)
        self.assertFalse(main_tracker.unverified_workspace_mutation)
        self.assertIn("system-test-change-review", checkpoint)
        self.assertIn("tests/vehicle.test.ts", checkpoint)
        self.assertIn("src/physics/__tests__/physics.test.ts", checkpoint)

        main_tracker.record(
            _call(
                "subagent_invoke",
                {
                    "task_label": "Independent test review",
                    "instructions": (
                        "Review tests/vehicle.test.ts and "
                        "src/physics/__tests__/physics.test.ts."
                    ),
                },
                "review-invoke",
            ),
            ToolExecutionResult(
                call_id="review-invoke",
                name="subagent_invoke",
                ok=True,
                content="started",
                metadata={"subagent_id": "review-agent"},
            ),
        )
        main_tracker.record_runtime_progress(
            content="Independent review completed.",
            metadata={
                "subagent_progress_update": True,
                "subagent_id": "review-agent",
                "subagent_status": "completed",
                "subagent_notice_kind": "subagent_completed",
                "recommended_action": "finalize",
                "latest_subagent_report_complete": True,
                "changed_test_artifact_paths": [],
            },
        )
        self.assertEqual(
            main_tracker.to_state()["completed_test_review_subagent_ids"],
            ["review-agent"],
        )

        editing_reviewer = ToolSafetyTracker()
        editing_reviewer.record_runtime_progress(
            content="Builder completed.",
            metadata={
                "subagent_progress_update": True,
                "subagent_id": "builder-agent",
                "subagent_status": "completed",
                "subagent_notice_kind": "subagent_completed",
                "recommended_action": "finalize",
                "latest_subagent_report_complete": True,
                "changed_test_artifact_paths": ["tests/vehicle.test.ts"],
            },
        )
        editing_reviewer.record(
            _call(
                "subagent_invoke",
                {
                    "task_label": "Independent test review",
                    "instructions": "Review tests/vehicle.test.ts.",
                },
                "editing-review-invoke",
            ),
            ToolExecutionResult(
                call_id="editing-review-invoke",
                name="subagent_invoke",
                ok=True,
                content="started",
                metadata={"subagent_id": "editing-review-agent"},
            ),
        )
        editing_reviewer.record_runtime_progress(
            content="Review completed after editing the target test.",
            metadata={
                "subagent_progress_update": True,
                "subagent_id": "editing-review-agent",
                "subagent_status": "completed",
                "subagent_notice_kind": "subagent_completed",
                "recommended_action": "finalize",
                "latest_subagent_report_complete": True,
                "changed_test_artifact_paths": ["tests/vehicle.test.ts"],
            },
        )
        self.assertEqual(
            editing_reviewer.to_state()["completed_test_review_subagent_ids"],
            [],
        )

        expanding_obligation = ToolSafetyTracker()
        expanding_obligation.record_runtime_progress(
            content="First builder completed.",
            metadata={
                "subagent_progress_update": True,
                "subagent_id": "first-builder",
                "subagent_status": "completed",
                "subagent_notice_kind": "subagent_completed",
                "recommended_action": "finalize",
                "latest_subagent_report_complete": True,
                "changed_test_artifact_paths": ["tests/first.test.ts"],
            },
        )
        expanding_obligation.record(
            _call(
                "subagent_invoke",
                {
                    "task_label": "Review first test",
                    "instructions": "Review tests/first.test.ts.",
                },
                "first-review-invoke",
            ),
            ToolExecutionResult(
                call_id="first-review-invoke",
                name="subagent_invoke",
                ok=True,
                content="started",
                metadata={"subagent_id": "first-reviewer"},
            ),
        )
        expanding_obligation.record_runtime_progress(
            content="First review completed.",
            metadata={
                "subagent_progress_update": True,
                "subagent_id": "first-reviewer",
                "subagent_status": "completed",
                "subagent_notice_kind": "subagent_completed",
                "recommended_action": "finalize",
                "latest_subagent_report_complete": True,
                "changed_test_artifact_paths": [],
            },
        )
        expanding_obligation.record_runtime_progress(
            content="Second builder completed.",
            metadata={
                "subagent_progress_update": True,
                "subagent_id": "second-builder",
                "subagent_status": "completed",
                "subagent_notice_kind": "subagent_completed",
                "recommended_action": "finalize",
                "latest_subagent_report_complete": True,
                "changed_test_artifact_paths": ["tests/second.test.ts"],
            },
        )
        expanded_state = expanding_obligation.to_state()
        self.assertEqual(
            expanded_state["completed_test_review_paths"],
            ["tests/first.test.ts"],
        )
        self.assertIn(
            "tests/second.test.ts",
            expanded_state["contract_requirements"]["system-test-change-review"][
                "criterion"
            ],
        )

    def test_failed_wait_result_transfers_delegated_test_review_obligation(self) -> None:
        tracker = ToolSafetyTracker()

        tracker.record(
            _call(
                "orchestrator_wait",
                {
                    "wake_after_seconds": 60,
                    "reason": "Waiting for productive child work.",
                },
                "wait-review",
            ),
            ToolExecutionResult(
                call_id="wait-review",
                name="orchestrator_wait",
                ok=False,
                content="A child completion requires review.",
                metadata={
                    "execution_failed": True,
                    "error_code": "orchestrator_review_required",
                    "changed_test_artifact_paths": ["tests/vehicle.test.ts"],
                },
            ),
        )

        state = tracker.to_state()
        self.assertFalse(tracker.unverified_workspace_mutation)
        self.assertIn(
            "tests/vehicle.test.ts",
            state["contract_requirements"]["system-test-change-review"][
                "criterion"
            ],
        )

    def test_aggregated_subagent_runtime_progress_preserves_test_review_evidence(
        self,
    ) -> None:
        tracker = ToolSafetyTracker()

        tracker.record_runtime_progress(
            content="Aggregated builder update.",
            metadata={
                "subagent_progress_update": True,
                "recommended_action": "finalize",
                "changed_test_artifact_paths": ["tests/vehicle.test.ts"],
                "subagents": [
                    {
                        "subagent_id": "builder-agent",
                        "status": "completed",
                        "report_complete": True,
                        "changed_test_artifact_paths": [
                            "tests/vehicle.test.ts"
                        ],
                    }
                ],
            },
        )
        tracker.record(
            _call(
                "subagent_invoke",
                {
                    "task_label": "Independent test review",
                    "instructions": "Review tests/vehicle.test.ts.",
                },
                "review-invoke",
            ),
            ToolExecutionResult(
                call_id="review-invoke",
                name="subagent_invoke",
                ok=True,
                content="started",
                metadata={"subagent_id": "review-agent"},
            ),
        )
        tracker.record_runtime_progress(
            content="Aggregated reviewer update.",
            metadata={
                "subagent_progress_update": True,
                "recommended_action": "finalize",
                "changed_test_artifact_paths": [],
                "subagents": [
                    {
                        "subagent_id": "review-agent",
                        "status": "completed",
                        "report_complete": True,
                        "changed_test_artifact_paths": [],
                    }
                ],
            },
        )

        state = tracker.to_state()
        self.assertEqual(
            state["completed_test_review_subagent_ids"],
            ["review-agent"],
        )
        self.assertEqual(
            state["completed_test_review_paths"],
            ["tests/vehicle.test.ts"],
        )

    def test_loading_subagent_tracker_drops_legacy_local_test_review_gate(self) -> None:
        legacy = ToolSafetyTracker()
        legacy.record(
            _call("file_write", {"path": "tests/legacy.test.ts"}, "legacy-write"),
            ToolExecutionResult(
                call_id="legacy-write",
                name="file_write",
                ok=True,
                content="changed",
                metadata={"changed": True, "path": "tests/legacy.test.ts"},
            ),
        )

        restored = ToolSafetyTracker.from_state(
            legacy.to_state(),
            actor_kind="subagent",
        )

        self.assertNotIn(
            "system-test-change-review",
            "\n".join(restored.checkpoint_lines()),
        )

    def test_turn_yield_results_never_enter_no_progress_suppression(self) -> None:
        tracker = ToolSafetyTracker()
        for index in range(5):
            call = _call(
                "orchestrator_wait",
                {"wake_after_seconds": 600, "reason": "Children are running."},
                f"wait-{index}",
            )
            observation = tracker.record(
                call,
                ToolExecutionResult(
                    call_id=call.call_id,
                    name=call.name,
                    ok=True,
                    content="registered",
                    turn_disposition="yield_turn",
                ),
            )
            self.assertFalse(observation.repeated_no_progress)
            self.assertIsNone(tracker.blocked_call_reason(call))
