"""Workspace-backed agent skills support."""

from .catalog import (
    get_skill,
    list_skill_resources,
    load_skill_catalog,
    search_skills,
)
from .config import SkillsSettings
from .importer import import_staged_skills
from .rendering import (
    render_skill_bootstrap_headers,
    render_skill_get_result,
    render_skill_import_notice,
    render_skill_search_guidance,
    render_skill_search_result,
    skill_import_metadata,
)
from .types import (
    SkillCatalog,
    SkillHeader,
    SkillImportDetail,
    SkillImportResult,
    SkillResource,
)

__all__ = [
    "SkillCatalog",
    "SkillHeader",
    "SkillImportDetail",
    "SkillImportResult",
    "SkillResource",
    "SkillsSettings",
    "get_skill",
    "import_staged_skills",
    "list_skill_resources",
    "load_skill_catalog",
    "render_skill_bootstrap_headers",
    "render_skill_get_result",
    "render_skill_import_notice",
    "render_skill_search_guidance",
    "render_skill_search_result",
    "search_skills",
    "skill_import_metadata",
]
