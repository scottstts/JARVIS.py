"""Agent-facing renderers for workspace-backed skills."""

from __future__ import annotations

from typing import Any

from .catalog import read_skill_markdown
from .config import SkillsSettings
from .types import SkillCatalog, SkillHeader, SkillImportResult, SkillResource


def render_skill_bootstrap_headers(catalog: SkillCatalog) -> str | None:
    if not catalog.skills:
        return None
    lines = ["Installed skills:"]
    for skill in catalog.skills:
        lines.append("- " + _format_skill_header_line(skill))
    lines.extend(["", "Use get_skills mode=get before applying a skill."])
    return "\n".join(lines)


def render_skill_search_guidance() -> str:
    return (
        "Skills are available through get_skills. Skill headers are not preloaded; "
        "use mode=search before tasks where an installed skill may apply, then "
        "mode=get for the chosen skill."
    )


def render_skill_search_result(
    *,
    query: str | None,
    matches: tuple[SkillHeader, ...],
    warnings: tuple[str, ...] = (),
) -> str:
    lines = [
        "Skill search result",
        f"query: {query if query else '(all installed skills)'}",
        f"match_count: {len(matches)}",
    ]
    if not matches:
        lines.append("No installed skills matched.")
    for index, skill in enumerate(matches, start=1):
        lines.extend(
            [
                f"{index}. {skill.skill_id}",
                f"name: {skill.name}",
                f"description: {skill.description}",
            ]
        )
        if skill.compatibility:
            lines.append(f"compatibility: {skill.compatibility}")
    if warnings:
        lines.append(f"warnings: {len(warnings)} scanner warning(s)")
    return "\n".join(lines)


def render_skill_get_result(
    *,
    settings: SkillsSettings,
    skill: SkillHeader,
    resources: tuple[SkillResource, ...],
) -> str:
    skill_text = read_skill_markdown(settings, skill)
    lines = [
        "Skill",
        f"skill_id: {skill.skill_id}",
        f"name: {skill.name}",
        f"description: {skill.description}",
    ]
    if skill.compatibility:
        lines.append(f"compatibility: {skill.compatibility}")
    lines.extend(["", "SKILL.md:", skill_text.rstrip()])
    if resources:
        lines.extend(["", "Bundled resources:"])
        for resource in resources:
            detail = f"- {resource.path} ({resource.kind}"
            if resource.size_bytes is not None:
                detail += f", {resource.size_bytes} bytes"
            detail += ")"
            lines.append(detail)
    return "\n".join(lines)


def render_skill_import_notice(result: SkillImportResult) -> str | None:
    if not result.has_reportable_changes:
        return None
    lines = ["Skill import normalization"]
    if result.imported:
        lines.append("imported: " + ", ".join(result.imported))
    if result.conflicts:
        lines.append("conflicts: " + ", ".join(result.conflicts))
    if result.warnings:
        lines.append(f"warnings: {len(result.warnings)}")
    return "\n".join(lines)


def skill_import_metadata(result: SkillImportResult) -> dict[str, Any]:
    return {
        "imported": list(result.imported),
        "already_present": list(result.already_present),
        "conflicts": list(result.conflicts),
        "skipped": list(result.skipped),
        "warnings": list(result.warnings),
        "cleaned": list(result.cleaned),
        "imported_details": [
            _skill_import_detail_metadata(detail)
            for detail in result.imported_details
        ],
        "already_present_details": [
            _skill_import_detail_metadata(detail)
            for detail in result.already_present_details
        ],
        "conflict_details": [
            _skill_import_detail_metadata(detail)
            for detail in result.conflict_details
        ],
    }


def skill_catalog_metadata(catalog: SkillCatalog) -> dict[str, Any]:
    return {
        "skill_count": len(catalog.skills),
        "warnings": list(catalog.warnings),
    }


def _format_skill_header_line(skill: SkillHeader) -> str:
    line = f"{skill.skill_id}: {skill.description}"
    if skill.compatibility:
        line += f" (compatibility: {skill.compatibility})"
    return line


def _skill_import_detail_metadata(detail) -> dict[str, str]:
    return {
        "skill_id": detail.skill_id,
        "source_dir": str(detail.source_dir),
        "target_dir": str(detail.target_dir),
        "installed_at": str(detail.target_dir / "SKILL.md"),
    }
