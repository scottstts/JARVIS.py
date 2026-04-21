"""Skill discovery and read tool definition."""

from __future__ import annotations

from typing import Any

from jarvis.llm import ToolDefinition
from jarvis.skills import (
    SkillsSettings,
    get_skill,
    import_staged_skills,
    list_skill_resources,
    load_skill_catalog,
    render_skill_get_result,
    render_skill_import_notice,
    render_skill_search_result,
    search_skills,
    skill_import_metadata,
)

from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult


class GetSkillsToolExecutor:
    """Searches installed skills or opens one skill."""

    def __init__(self, settings: SkillsSettings) -> None:
        self._settings = settings

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        settings = SkillsSettings.from_workspace_dir(context.workspace_dir)
        import_result = import_staged_skills(settings)
        mode = _normalize_mode(arguments.get("mode"))

        if mode == "search":
            query = _optional_string(arguments.get("query"))
            catalog = load_skill_catalog(settings)
            matches = search_skills(catalog, query or "")
            content = render_skill_search_result(
                query=query,
                matches=matches,
                warnings=catalog.warnings,
            )
            import_notice = render_skill_import_notice(import_result)
            if import_notice is not None:
                content += "\n\n" + import_notice
            return ToolExecutionResult(
                call_id=call_id,
                name="get_skills",
                ok=True,
                content=content,
                metadata={
                    "mode": "search",
                    "query": query,
                    "match_count": len(matches),
                    "matches": [_skill_metadata(skill) for skill in matches],
                    "scanner_warnings": list(catalog.warnings),
                    "skill_import": skill_import_metadata(import_result),
                },
            )

        skill_id = _optional_string(arguments.get("skill_id"))
        if skill_id is None:
            return ToolExecutionResult(
                call_id=call_id,
                name="get_skills",
                ok=False,
                content="get_skills failed: mode=get requires skill_id.",
                metadata={"mode": "get"},
            )
        skill = get_skill(settings, skill_id)
        if skill is None:
            return ToolExecutionResult(
                call_id=call_id,
                name="get_skills",
                ok=False,
                content=f"get_skills failed: skill not found: {skill_id}",
                metadata={
                    "mode": "get",
                    "skill_id": skill_id,
                    "skill_import": skill_import_metadata(import_result),
                },
            )
        try:
            resources = list_skill_resources(settings, skill)
            content = render_skill_get_result(
                settings=settings,
                skill=skill,
                resources=resources,
            )
        except Exception as exc:
            return ToolExecutionResult(
                call_id=call_id,
                name="get_skills",
                ok=False,
                content=f"get_skills failed: {exc}",
                metadata={"mode": "get", "skill_id": skill_id},
            )
        import_notice = render_skill_import_notice(import_result)
        if import_notice is not None:
            content += "\n\n" + import_notice
        return ToolExecutionResult(
            call_id=call_id,
            name="get_skills",
            ok=True,
            content=content,
            metadata={
                "mode": "get",
                "skill": _skill_metadata(skill),
                "resource_count": len(resources),
                "resources": [
                    {
                        "path": resource.path,
                        "kind": resource.kind,
                        "size_bytes": resource.size_bytes,
                    }
                    for resource in resources
                ],
                "skill_import": skill_import_metadata(import_result),
            },
        )


def build_get_skills_tool(settings: SkillsSettings) -> RegisteredTool:
    modes = ["get"] if settings.bootstrap_headers else ["search", "get"]
    properties: dict[str, Any] = {
        "mode": {
            "type": "string",
            "enum": modes,
        },
        "skill_id": {
            "type": "string",
            "description": "Skill id for mode=get.",
        },
    }
    if not settings.bootstrap_headers:
        properties["query"] = {
            "type": "string",
            "description": "Search text for mode=search.",
        }
    return RegisteredTool(
        name="get_skills",
        exposure="basic",
        definition=ToolDefinition(
            name="get_skills",
            description=(
                "Open an installed skill by id after matching a bootstrapped skill header."
                if settings.bootstrap_headers
                else "Search installed skills, then open one by id before using it."
            ),
            input_schema={
                "type": "object",
                "properties": properties,
                "required": ["mode"],
                "additionalProperties": False,
            },
        ),
        executor=GetSkillsToolExecutor(settings),
        allowed_agent_kinds=("main", "subagent"),
    )


def _normalize_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized or "get"


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _skill_metadata(skill) -> dict[str, Any]:
    payload = {
        "skill_id": skill.skill_id,
        "name": skill.name,
        "description": skill.description,
        "path": str(skill.path),
    }
    if skill.compatibility:
        payload["compatibility"] = skill.compatibility
    if skill.metadata:
        payload["metadata"] = dict(skill.metadata)
    return payload
