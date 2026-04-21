"""Execution runtime for registered tools."""

from __future__ import annotations

import re

from jarvis.logging_setup import get_application_logger
from jarvis.llm import ToolCall
from jarvis.llm.validation import TOOL_CALL_VALIDATION_ERROR_METADATA_KEY
from jarvis.skills import (
    SkillsSettings,
    import_staged_skills,
    load_skill_catalog,
    render_skill_import_notice,
    skill_import_metadata,
)
from jarvis.skills.catalog import is_valid_skill_id

from .policy import ToolPolicy
from .registry import ToolRegistry
from .types import ToolExecutionContext, ToolExecutionResult

LOGGER = get_application_logger(__name__)
_SKILL_INSTALL_COMMAND_MARKERS = (
    "openskills install",
    "add-skill",
    "skill-installer",
    "install-skill",
    "skills install",
    "skill install",
)
_SKILL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_INSTALLED_LINE_PATTERN = re.compile(r"Installed:\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_WORKSPACE_SKILL_PATH_PATTERN = re.compile(
    r"/workspace/(?:(?:\.codex|\.claude|\.agents|\.mdskills)/skills|skills)"
    r"/([A-Za-z0-9][A-Za-z0-9._-]*)"
)


class ToolRuntime:
    """Runs tool calls through registry lookup and policy checks."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        policy: ToolPolicy | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy or ToolPolicy()

    async def execute(
        self,
        *,
        tool_call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        validation_error = str(
            tool_call.provider_metadata.get(TOOL_CALL_VALIDATION_ERROR_METADATA_KEY, "")
        ).strip()
        if validation_error:
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                ok=False,
                content=(
                    "Tool execution failed\n"
                    f"tool: {tool_call.name}\n"
                    "error_type: ToolCallValidationError\n"
                    f"error: {validation_error}\n"
                    f"raw_arguments: {tool_call.raw_arguments}\n"
                    "fix: emit a new tool call whose arguments match the tool schema."
                ),
                metadata={
                    "tool_call_validation_failed": True,
                    "reason": validation_error,
                    "raw_arguments": tool_call.raw_arguments,
                    "arguments": dict(tool_call.arguments),
                },
            )
        tool = self._registry.require(tool_call.name)
        skill_install_call = _looks_like_skill_install_call(tool_call)
        before_skill_ids = (
            _installed_skill_ids(context.workspace_dir)
            if skill_install_call
            else None
        )
        decision = self._policy.authorize(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            context=context,
        )
        if not decision.allowed:
            if isinstance(decision.approval_request, dict):
                approval_request = dict(decision.approval_request)
                return ToolExecutionResult(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    ok=False,
                    content=_format_approval_required_content(
                        tool_name=tool_call.name,
                        approval_request=approval_request,
                    ),
                    metadata={
                        "approval_required": True,
                        "approval_request": approval_request,
                        "arguments": dict(tool_call.arguments),
                    },
                )
            reason = decision.reason or "Tool execution denied by policy."
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                ok=False,
                content=(
                    "Tool execution denied by policy\n"
                    f"tool: {tool_call.name}\n"
                    f"reason: {reason}"
                ),
                metadata={
                    "policy_denied": True,
                    "reason": reason,
                    "arguments": dict(tool_call.arguments),
                },
            )

        try:
            result = await tool.executor(
                call_id=tool_call.call_id,
                arguments=tool_call.arguments,
                context=context,
            )
            return _postprocess_bash_result_with_skill_imports(
                result,
                context=context,
                skill_install_call=skill_install_call,
                before_skill_ids=before_skill_ids,
            )
        except Exception as exc:
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                ok=False,
                content=(
                    "Tool execution failed\n"
                    f"tool: {tool_call.name}\n"
                    f"error_type: {type(exc).__name__}\n"
                    f"error: {exc}"
                ),
                metadata={
                    "execution_failed": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "arguments": dict(tool_call.arguments),
                },
            )


def _postprocess_bash_result_with_skill_imports(
    result: ToolExecutionResult,
    *,
    context: ToolExecutionContext,
    skill_install_call: bool = False,
    before_skill_ids: set[str] | None = None,
) -> ToolExecutionResult:
    if result.name != "bash":
        return result
    if skill_install_call and not _is_successful_terminal_bash_result(result):
        return _format_skill_install_failure_result(
            result,
            failed_stage="install",
            reason="bash command did not complete successfully.",
        )
    if not _is_successful_terminal_bash_result(result):
        return result

    try:
        settings = SkillsSettings.from_workspace_dir(context.workspace_dir)
        import_result = import_staged_skills(settings)
    except Exception:
        LOGGER.exception("Skill import scan after bash result failed.")
        if skill_install_call:
            return _format_skill_install_failure_result(
                result,
                failed_stage="normalization",
                reason="skill import normalization raised an exception.",
            )
        return result
    if skill_install_call:
        return _format_skill_install_normalized_result(
            result,
            settings=settings,
            import_result=import_result,
            before_skill_ids=before_skill_ids,
        )

    notice = render_skill_import_notice(import_result)
    if notice is None:
        return result
    metadata = dict(result.metadata)
    metadata["skill_import"] = skill_import_metadata(import_result)
    return ToolExecutionResult(
        call_id=result.call_id,
        name=result.name,
        ok=result.ok,
        content=result.content.rstrip() + "\n\n" + notice,
        metadata=metadata,
    )


def _is_successful_terminal_bash_result(result: ToolExecutionResult) -> bool:
    if result.name != "bash" or not result.ok:
        return False
    mode = str(result.metadata.get("mode", "")).strip()
    if mode == "foreground":
        status = str(result.metadata.get("status", "")).strip()
        return status != "running" and result.metadata.get("exit_code") == 0
    if mode == "status":
        status = str(
            result.metadata.get("status") or result.metadata.get("state") or ""
        ).strip()
        return status == "finished" and result.metadata.get("exit_code") == 0
    return False


def _looks_like_skill_install_call(tool_call: ToolCall) -> bool:
    if tool_call.name != "bash":
        return False
    command = str(tool_call.arguments.get("command", "")).strip().lower()
    if not command:
        return False
    return any(marker in command for marker in _SKILL_INSTALL_COMMAND_MARKERS)


def _installed_skill_ids(workspace_dir) -> set[str]:
    settings = SkillsSettings.from_workspace_dir(workspace_dir)
    catalog = load_skill_catalog(settings)
    return {skill.skill_id for skill in catalog.skills}


def _format_skill_install_normalized_result(
    result: ToolExecutionResult,
    *,
    settings: SkillsSettings,
    import_result,
    before_skill_ids: set[str] | None,
) -> ToolExecutionResult:
    metadata = dict(result.metadata)
    metadata["skill_import"] = skill_import_metadata(import_result)
    installed_skill_ids = _normalized_installed_skill_ids(
        result,
        settings=settings,
        import_result=import_result,
        before_skill_ids=before_skill_ids,
    )

    if import_result.conflicts or import_result.warnings:
        metadata["skill_install"] = {
            "status": "failed",
            "failed_stage": "normalization",
            "skill_ids": installed_skill_ids,
        }
        return ToolExecutionResult(
            call_id=result.call_id,
            name=result.name,
            ok=False,
            content=_skill_install_normalization_failure_content(
                result,
                settings=settings,
                import_result=import_result,
                installed_skill_ids=installed_skill_ids,
            ),
            metadata=metadata,
        )

    if not installed_skill_ids:
        metadata["skill_install"] = {
            "status": "failed",
            "failed_stage": "normalization",
        }
        return ToolExecutionResult(
            call_id=result.call_id,
            name=result.name,
            ok=False,
            content=_skill_install_no_skill_found_content(result),
            metadata=metadata,
        )

    metadata["skill_install"] = {
        "status": "success",
        "skill_ids": installed_skill_ids,
        "installed_at": [
            str(settings.skills_dir / skill_id / "SKILL.md")
            for skill_id in installed_skill_ids
        ],
    }
    return ToolExecutionResult(
        call_id=result.call_id,
        name=result.name,
        ok=True,
        content=_skill_install_success_content(
            settings=settings,
            skill_ids=installed_skill_ids,
        ),
        metadata=metadata,
    )


def _normalized_installed_skill_ids(
    result: ToolExecutionResult,
    *,
    settings: SkillsSettings,
    import_result,
    before_skill_ids: set[str] | None,
) -> list[str]:
    skill_ids = set(import_result.imported) | set(import_result.already_present)
    after_catalog = load_skill_catalog(settings)
    after_skill_ids = {skill.skill_id for skill in after_catalog.skills}
    if before_skill_ids is not None:
        skill_ids.update(after_skill_ids - before_skill_ids)
    for candidate in _candidate_skill_ids_from_bash_result(result, settings=settings):
        if candidate in after_skill_ids:
            skill_ids.add(candidate)
    return sorted(skill_ids)


def _candidate_skill_ids_from_bash_result(
    result: ToolExecutionResult,
    *,
    settings: SkillsSettings,
) -> set[str]:
    text = "\n".join(
        str(part)
        for part in (
            result.metadata.get("command", ""),
            result.metadata.get("stdout", ""),
            result.content,
        )
        if str(part).strip()
    )
    candidates: set[str] = set()
    for match in _INSTALLED_LINE_PATTERN.finditer(text):
        _add_skill_candidate(candidates, match.group(1), settings=settings)
    for match in _WORKSPACE_SKILL_PATH_PATTERN.finditer(text):
        _add_skill_candidate(candidates, match.group(1), settings=settings)
    command = str(result.metadata.get("command", "")).strip()
    for token in re.split(r"\s+", command):
        token = token.strip("'\"")
        if "/" not in token:
            continue
        _add_skill_candidate(candidates, token.rsplit("/", 1)[-1], settings=settings)
    return candidates


def _add_skill_candidate(
    candidates: set[str],
    value: str,
    *,
    settings: SkillsSettings,
) -> None:
    normalized = value.strip().strip("/").rsplit("/", 1)[-1]
    if _SKILL_ID_PATTERN.fullmatch(normalized) and is_valid_skill_id(
        normalized,
        max_chars=settings.max_skill_id_chars,
    ):
        candidates.add(normalized)


def _skill_install_success_content(
    *,
    settings: SkillsSettings,
    skill_ids: list[str],
) -> str:
    lines = ["Skill install result", "status: success"]
    if len(skill_ids) == 1:
        skill_id = skill_ids[0]
        lines.append(f"skill: {skill_id}")
        lines.append(f"installed_at: {settings.skills_dir / skill_id / 'SKILL.md'}")
        return "\n".join(lines)
    lines.append("skills:")
    for skill_id in skill_ids:
        lines.append(f"- skill: {skill_id}")
        lines.append(f"  installed_at: {settings.skills_dir / skill_id / 'SKILL.md'}")
    return "\n".join(lines)


def _format_skill_install_failure_result(
    result: ToolExecutionResult,
    *,
    failed_stage: str,
    reason: str,
) -> ToolExecutionResult:
    metadata = dict(result.metadata)
    metadata["skill_install"] = {
        "status": "failed",
        "failed_stage": failed_stage,
        "reason": reason,
    }
    return ToolExecutionResult(
        call_id=result.call_id,
        name=result.name,
        ok=False,
        content="\n".join(
            [
                "Skill install result",
                "status: failed",
                f"failed_stage: {failed_stage}",
                f"reason: {reason}",
                "",
                "bash_result:",
                _trim_tool_content(result.content),
            ]
        ),
        metadata=metadata,
    )


def _skill_install_normalization_failure_content(
    result: ToolExecutionResult,
    *,
    settings: SkillsSettings,
    import_result,
    installed_skill_ids: list[str],
) -> str:
    lines = [
        "Skill install result",
        "status: failed",
        "failed_stage: normalization",
    ]
    if installed_skill_ids:
        lines.append("installed_at:")
        for skill_id in installed_skill_ids:
            lines.append(f"- {settings.skills_dir / skill_id / 'SKILL.md'}")
    if import_result.conflicts:
        lines.append("conflicts: " + ", ".join(import_result.conflicts))
    if import_result.warnings:
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in import_result.warnings[:5])
    lines.extend(["", "bash_result:", _trim_tool_content(result.content)])
    return "\n".join(lines)


def _skill_install_no_skill_found_content(result: ToolExecutionResult) -> str:
    return "\n".join(
        [
            "Skill install result",
            "status: failed",
            "failed_stage: normalization",
            "reason: installer finished but no valid skill was found in /workspace/skills.",
            "",
            "bash_result:",
            _trim_tool_content(result.content),
        ]
    )


def _trim_tool_content(content: str, *, max_chars: int = 6000) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "\n[truncated]"


def _format_approval_required_content(
    *,
    tool_name: str,
    approval_request: dict[str, object],
) -> str:
    summary = str(approval_request.get("summary", "")).strip() or "Approval required."
    lines = [
        "Approval required",
        f"tool: {tool_name}",
        f"summary: {summary}",
    ]
    details = str(approval_request.get("details", "")).strip()
    if details:
        lines.append(f"details: {details}")
    command = str(approval_request.get("command", "")).strip()
    if command:
        lines.append(f"command: {command}")
    tool_label = str(approval_request.get("tool_name", "")).strip()
    if tool_label:
        lines.append(f"tool_name: {tool_label}")
    return "\n".join(lines)
