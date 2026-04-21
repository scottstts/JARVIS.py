"""Unit tests for workspace-backed agent skills."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis.core import AgentLoop
from jarvis.core.compaction import prune_compaction_source_records
from jarvis.llm import LLMRequest, LLMResponse, LLMUsage, TextPart, ToolCall, ToolDefinition
from jarvis.skills import (
    SkillsSettings,
    import_staged_skills,
    load_skill_catalog,
    search_skills,
)
from jarvis.storage import ConversationRecord, SessionStorage
from jarvis.tools import (
    RegisteredTool,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolPolicyDecision,
    ToolRegistry,
    ToolRuntime,
    ToolSettings,
)
from tests.helpers import build_core_settings


def _write_skill(workspace_dir: Path, skill_id: str, *, description: str) -> Path:
    skill_dir = workspace_dir / "skills" / skill_id
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {skill_id}",
                f"description: {description}",
                "compatibility: Jarvis",
                "---",
                "",
                f"Use {skill_id} carefully.",
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir


def _write_staged_skill(workspace_dir: Path, skill_id: str, *, description: str) -> Path:
    skill_dir = workspace_dir / ".claude" / "skills" / skill_id
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {skill_id}",
                f"description: {description}",
                "---",
                "",
                f"Use {skill_id} carefully.",
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir


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


class _FakeLLMService:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return _build_response("ok")

    async def stream_generate(self, _request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _AllowAllPolicy:
    def authorize(
        self,
        *,
        tool_name: str,
        arguments: dict,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = tool_name, arguments, context
        return ToolPolicyDecision(allowed=True)


class _SkillInstallBashExecutor:
    def __init__(self, workspace_dir: Path, *, exit_code: int = 0) -> None:
        self._workspace_dir = workspace_dir
        self._exit_code = exit_code

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        _ = context
        command = str(arguments.get("command", ""))
        if self._exit_code == 0:
            _write_staged_skill(
                self._workspace_dir,
                "ai-image-prompts-skill",
                description="Recommend image prompts.",
            )
        stdout = (
            "Installing from: YouMind-OpenLab/ai-image-prompts-skill\n"
            "Location: project (.claude/skills)\n"
            "Installed: ai-image-prompts-skill\n"
            "---POST-INSTALL---\n"
            "/workspace/skills\n"
        )
        stderr = "" if self._exit_code == 0 else "clone failed"
        return ToolExecutionResult(
            call_id=call_id,
            name="bash",
            ok=self._exit_code == 0,
            content=(
                "Bash execution result\n"
                f"status: {'success' if self._exit_code == 0 else 'failed'}\n"
                f"command: {command}\n"
                f"exit_code: {self._exit_code}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr or '(empty)'}"
            ),
            metadata={
                "mode": "foreground",
                "status": "success" if self._exit_code == 0 else "failed",
                "command": command,
                "exit_code": self._exit_code,
                "stdout": stdout,
                "stderr": stderr,
            },
        )


def _fake_bash_registry(executor) -> ToolRegistry:
    return ToolRegistry(
        tools=[
            RegisteredTool(
                name="bash",
                exposure="basic",
                definition=ToolDefinition(
                    name="bash",
                    description="Run bash.",
                    input_schema={"type": "object"},
                ),
                executor=executor,
            )
        ]
    )


class SkillCatalogTests(unittest.TestCase):
    def test_load_catalog_skips_invalid_skills_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            _write_skill(
                workspace_dir,
                "slides",
                description="Create and edit slide decks.",
            )
            invalid_dir = workspace_dir / "skills" / "invalid"
            invalid_dir.mkdir(parents=True)
            (invalid_dir / "SKILL.md").write_text(
                "---\nname: invalid\n---\n\nMissing description.",
                encoding="utf-8",
            )

            catalog = load_skill_catalog(SkillsSettings.from_workspace_dir(workspace_dir))

        self.assertEqual([skill.skill_id for skill in catalog.skills], ["slides"])
        self.assertEqual(len(catalog.warnings), 1)
        self.assertIn("missing frontmatter description", catalog.warnings[0])

    def test_search_prioritizes_name_and_description_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            _write_skill(
                workspace_dir,
                "slides",
                description="Create and edit presentation decks.",
            )
            _write_skill(
                workspace_dir,
                "spreadsheet",
                description="Analyze CSV and workbook data.",
            )

            catalog = load_skill_catalog(SkillsSettings.from_workspace_dir(workspace_dir))
            matches = search_skills(catalog, "deck")

        self.assertEqual([skill.skill_id for skill in matches], ["slides"])

    def test_load_catalog_skips_symlink_skill_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            outside = Path(tmp) / "outside"
            _write_skill(outside, "slides", description="Create decks.")
            skills_dir = workspace_dir / "skills"
            skills_dir.mkdir(parents=True, exist_ok=True)
            (skills_dir / "slides").symlink_to(outside / "skills" / "slides")

            catalog = load_skill_catalog(SkillsSettings.from_workspace_dir(workspace_dir))

        self.assertEqual(catalog.skills, ())
        self.assertEqual(len(catalog.warnings), 1)
        self.assertIn("may not be a symlink", catalog.warnings[0])

    def test_importer_copies_staged_npx_home_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            staged = workspace_dir / ".codex" / "skills" / "deck-maker"
            staged.mkdir(parents=True)
            (staged / "SKILL.md").write_text(
                "---\nname: deck-maker\ndescription: Build decks.\n---\n\nInstructions.",
                encoding="utf-8",
            )
            (staged / "scripts").mkdir()
            (staged / "scripts" / "build.py").write_text("print('ok')\n", encoding="utf-8")

            result = import_staged_skills(SkillsSettings.from_workspace_dir(workspace_dir))

            self.assertEqual(result.imported, ("deck-maker",))
            self.assertTrue((workspace_dir / "skills" / "deck-maker" / "SKILL.md").exists())
            self.assertTrue(
                (workspace_dir / "skills" / "deck-maker" / "scripts" / "build.py").exists()
            )
            self.assertFalse(staged.exists())

    def test_importer_reports_conflict_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            _write_skill(workspace_dir, "deck-maker", description="Existing skill.")
            staged = workspace_dir / ".codex" / "skills" / "deck-maker"
            staged.mkdir(parents=True)
            (staged / "SKILL.md").write_text(
                "---\nname: deck-maker\ndescription: Different skill.\n---\n\nNew.",
                encoding="utf-8",
            )

            result = import_staged_skills(SkillsSettings.from_workspace_dir(workspace_dir))

            self.assertEqual(result.conflicts, ("deck-maker",))
            installed_text = (workspace_dir / "skills" / "deck-maker" / "SKILL.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Existing skill", installed_text)


class GetSkillsToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_bash_skill_install_returns_normalized_success_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            runtime = ToolRuntime(
                registry=_fake_bash_registry(_SkillInstallBashExecutor(workspace_dir)),
                policy=_AllowAllPolicy(),
            )

            result = await runtime.execute(
                tool_call=ToolCall(
                    call_id="call_install_skill",
                    name="bash",
                    arguments={
                        "mode": "foreground",
                        "command": (
                            "cd /workspace && HOME=/workspace npx openskills install "
                            "YouMind-OpenLab/ai-image-prompts-skill && "
                            "find /workspace/skills -maxdepth 3 -print"
                        ),
                    },
                    raw_arguments="{}",
                ),
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )

            self.assertTrue(result.ok)
            self.assertEqual(
                result.content,
                "\n".join(
                    [
                        "Skill install result",
                        "status: success",
                        "skill: ai-image-prompts-skill",
                        "installed_at: "
                        f"{workspace_dir / 'skills' / 'ai-image-prompts-skill' / 'SKILL.md'}",
                    ]
                ),
            )
            self.assertEqual(result.metadata["skill_install"]["status"], "success")
            self.assertEqual(
                result.metadata["skill_import"]["imported"],
                ["ai-image-prompts-skill"],
            )
            self.assertTrue(
                (workspace_dir / "skills" / "ai-image-prompts-skill" / "SKILL.md").exists()
            )
            self.assertFalse(
                (workspace_dir / ".claude" / "skills" / "ai-image-prompts-skill").exists()
            )

    async def test_bash_skill_install_failure_identifies_install_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            runtime = ToolRuntime(
                registry=_fake_bash_registry(
                    _SkillInstallBashExecutor(workspace_dir, exit_code=1)
                ),
                policy=_AllowAllPolicy(),
            )

            result = await runtime.execute(
                tool_call=ToolCall(
                    call_id="call_install_skill_failed",
                    name="bash",
                    arguments={
                        "mode": "foreground",
                        "command": (
                            "cd /workspace && HOME=/workspace npx openskills install "
                            "YouMind-OpenLab/ai-image-prompts-skill"
                        ),
                    },
                    raw_arguments="{}",
                ),
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )

        self.assertFalse(result.ok)
        self.assertIn("status: failed", result.content)
        self.assertIn("failed_stage: install", result.content)
        self.assertEqual(result.metadata["skill_install"]["failed_stage"], "install")

    async def test_bash_skill_install_failure_identifies_normalization_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            _write_skill(
                workspace_dir,
                "ai-image-prompts-skill",
                description="Existing different skill.",
            )
            runtime = ToolRuntime(
                registry=_fake_bash_registry(_SkillInstallBashExecutor(workspace_dir)),
                policy=_AllowAllPolicy(),
            )

            result = await runtime.execute(
                tool_call=ToolCall(
                    call_id="call_install_skill_conflict",
                    name="bash",
                    arguments={
                        "mode": "foreground",
                        "command": (
                            "cd /workspace && HOME=/workspace npx openskills install "
                            "YouMind-OpenLab/ai-image-prompts-skill"
                        ),
                    },
                    raw_arguments="{}",
                ),
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )

            self.assertFalse(result.ok)
            self.assertIn("status: failed", result.content)
            self.assertIn("failed_stage: normalization", result.content)
            self.assertIn("conflicts: ai-image-prompts-skill", result.content)
            self.assertEqual(
                result.metadata["skill_install"]["failed_stage"],
                "normalization",
            )
            self.assertTrue(
                (workspace_dir / ".claude" / "skills" / "ai-image-prompts-skill").exists()
            )

    async def test_search_is_hidden_and_policy_denied_when_headers_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            _write_skill(workspace_dir, "slides", description="Create decks.")
            registry = ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))
            tool = registry.require("get_skills")

            self.assertEqual(
                tool.definition.input_schema["properties"]["mode"]["enum"],
                ["get"],
            )
            self.assertNotIn("query", tool.definition.input_schema["properties"])

            runtime = ToolRuntime(registry=registry)
            result = await runtime.execute(
                tool_call=ToolCall(
                    call_id="call_get_skills_search",
                    name="get_skills",
                    arguments={"mode": "search", "query": "slides"},
                    raw_arguments='{"mode":"search","query":"slides"}',
                ),
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )

            self.assertFalse(result.ok)
            self.assertIn("not available", result.content)

    async def test_search_available_when_headers_are_not_bootstrapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            _write_skill(workspace_dir, "slides", description="Create decks.")
            with patch.dict(
                "os.environ",
                {"JARVIS_SKILLS_BOOTSTRAP_HEADERS": "false"},
            ):
                registry = ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))
                tool = registry.require("get_skills")
                runtime = ToolRuntime(registry=registry)

                result = await runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_get_skills_search",
                        name="get_skills",
                        arguments={"mode": "search", "query": "deck"},
                        raw_arguments='{"mode":"search","query":"deck"}',
                    ),
                    context=ToolExecutionContext(workspace_dir=workspace_dir),
                )

        self.assertEqual(
            tool.definition.input_schema["properties"]["mode"]["enum"],
            ["search", "get"],
        )
        self.assertTrue(result.ok)
        self.assertIn("slides", result.content)
        self.assertEqual(result.metadata["match_count"], 1)

    async def test_get_returns_skill_and_bounded_resource_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            skill_dir = _write_skill(workspace_dir, "slides", description="Create decks.")
            (skill_dir / "references").mkdir()
            (skill_dir / "references" / "style.md").write_text("Use short titles.", encoding="utf-8")
            registry = ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))
            runtime = ToolRuntime(registry=registry)

            result = await runtime.execute(
                tool_call=ToolCall(
                    call_id="call_get_skills_get",
                    name="get_skills",
                    arguments={"mode": "get", "skill_id": "slides"},
                    raw_arguments='{"mode":"get","skill_id":"slides"}',
                ),
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )

        self.assertTrue(result.ok)
        self.assertIn("SKILL.md:", result.content)
        self.assertIn("Use slides carefully.", result.content)
        self.assertIn("references/style.md", result.content)
        self.assertEqual(result.metadata["skill"]["skill_id"], "slides")

    async def test_get_rejects_path_like_skill_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            registry = ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))
            runtime = ToolRuntime(registry=registry)

            result = await runtime.execute(
                tool_call=ToolCall(
                    call_id="call_get_skills_bad",
                    name="get_skills",
                    arguments={"mode": "get", "skill_id": "../secret"},
                    raw_arguments='{"mode":"get","skill_id":"../secret"}',
                ),
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )

        self.assertFalse(result.ok)
        self.assertIn("canonical skill directory name", result.content)


class SkillBootstrapTests(unittest.IsolatedAsyncioTestCase):
    async def test_skill_headers_are_persisted_and_sent_to_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            _write_skill(settings.workspace_dir, "slides", description="Create decks.")
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = _FakeLLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )

            result = await loop.handle_user_input("hello")

            records = storage.load_records(result.session_id)
            skill_records = [
                record
                for record in records
                if record.metadata.get("skills_bootstrap") == "headers"
            ]
            self.assertEqual(len(skill_records), 1)
            self.assertIn("slides: Create decks.", skill_records[0].content)
            self.assertFalse(skill_records[0].metadata.get("transcript_only", False))
            request_text = "\n".join(
                part.text
                for message in llm_service.requests[0].messages
                for part in message.parts
                if isinstance(part, TextPart)
            )
            self.assertIn("Installed skills:", request_text)
            self.assertIn("slides: Create decks.", request_text)

    async def test_skill_headers_are_omitted_when_setting_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            _write_skill(settings.workspace_dir, "slides", description="Create decks.")
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = _FakeLLMService()
            with patch.dict(
                "os.environ",
                {"JARVIS_SKILLS_BOOTSTRAP_HEADERS": "false"},
            ):
                loop = AgentLoop(
                    llm_service=llm_service,
                    settings=settings,
                    storage=storage,
                )
                result = await loop.handle_user_input("hello")

            records = storage.load_records(result.session_id)
            skill_records = [
                record for record in records if record.metadata.get("skills_bootstrap")
            ]
            self.assertEqual(
                [record.metadata.get("skills_bootstrap") for record in skill_records],
                ["search_guidance"],
            )
            self.assertIn("mode=search", skill_records[0].content)
            self.assertNotIn("slides: Create decks.", skill_records[0].content)

    def test_skill_bootstrap_is_pruned_from_compaction_source(self) -> None:
        record = ConversationRecord(
            record_id="record_1",
            session_id="session_1",
            created_at="2026-04-21T00:00:00+00:00",
            role="system",
            content="Installed skills:\n- slides: Create decks.",
            metadata={"skills_bootstrap": "headers"},
        )

        self.assertEqual(prune_compaction_source_records([record]), ())
