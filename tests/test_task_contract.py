"""Task identity, contract, and machine-evidence regression tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from jarvis.core import AgentLoop
from jarvis.core.task_contract import TaskRequirement, build_task_contract
from jarvis.core.tool_safety import ToolSafetyTracker
from jarvis.llm import ToolCall
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

    def test_unrelated_user_turn_replaces_parked_task_but_resume_reuses_it(self) -> None:
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
            second = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-two",
                user_text="What time is it?",
            )
            resumed = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-three",
                user_text="Continue and make sure all tests pass.",
            )
            waived = loop._prepare_tool_task(  # pyright: ignore[reportPrivateUsage]
                session_id=session.session_id,
                proposed_task_id="task-four",
                user_text="Continue, but I no longer need the subagent requirement.",
            )

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertIsNotNone(resumed)
            self.assertIsNotNone(waived)
            assert first is not None and second is not None and resumed is not None
            assert waived is not None
            self.assertNotEqual(first.task_id, second.task_id)
            self.assertEqual(resumed.task_id, second.task_id)
            self.assertNotEqual(waived.task_id, resumed.task_id)
            self.assertTrue(
                any("tests pass" in item.criterion for item in resumed.requirements)
            )
            refreshed = storage.get_session(session.session_id)
            assert refreshed is not None
            self.assertEqual(
                refreshed.backend_state.get("active_tool_task_id"),
                "task-four",
            )
            self.assertIsNotNone(storage.load_tool_task_state("task-one"))

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
        tracker = ToolSafetyTracker()
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
                "item_id": requirement.item_id,
                "required": True,
                "outcome": "passed",
                "evidence_kind": (
                    "artifact_inspection"
                    if requirement.item_id == "visual"
                    else "runtime_observation"
                ),
                "source_tool_call_ids": ["gates"],
                "artifact_paths": (
                    ["/workspace/renders/final.png"]
                    if requirement.item_id == "visual"
                    else []
                ),
            }
            for requirement in requirements
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
                        "complete": True,
                        "checks": checks,
                    }
                },
            ),
        )

        self.assertFalse(tracker.unverified_workspace_mutation)

        missing_delegation = ToolSafetyTracker()
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
                        "complete": True,
                        "checks": [checks[0]],
                    }
                },
            ),
        )
        self.assertTrue(missing_delegation.unverified_workspace_mutation)

    def test_tool_task_sidecar_skips_identical_rewrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = SessionStorage(Path(tmp))
            state = {"schema_version": 1, "rounds": 4, "tracker": {}}

            self.assertTrue(storage.write_tool_task_state("task-1", state))
            self.assertFalse(storage.write_tool_task_state("task-1", state))
            self.assertEqual(storage.load_tool_task_state("task-1"), state)

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
