"""Skills catalog parsing and deterministic search."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

from jarvis.logging_setup import get_application_logger

from .config import SkillsSettings
from .types import SkillCatalog, SkillHeader, SkillResource

LOGGER = get_application_logger(__name__)

_FRONTMATTER_BOUNDARY = "---"
_SKILL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_valid_skill_id(value: str, *, max_chars: int) -> bool:
    return bool(value) and len(value) <= max_chars and bool(_SKILL_ID_PATTERN.fullmatch(value))


def load_skill_catalog(settings: SkillsSettings) -> SkillCatalog:
    """Load valid skill headers from the canonical workspace skills directory."""

    skills_dir = settings.skills_dir
    try:
        skills_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        warning = f"Could not create skills directory {skills_dir}: {exc}"
        LOGGER.warning(warning)
        return SkillCatalog(skills=(), warnings=(warning,))

    warnings: list[str] = []
    skills: list[SkillHeader] = []
    try:
        children = sorted(skills_dir.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        warning = f"Could not scan skills directory {skills_dir}: {exc}"
        LOGGER.warning(warning)
        return SkillCatalog(skills=(), warnings=(warning,))

    for child in children:
        if not child.is_dir():
            continue
        if child.is_symlink():
            warning = f"Skipped skill {child.name}: skill directory may not be a symlink."
            LOGGER.warning(warning)
            warnings.append(warning)
            continue
        skill_id = child.name
        if not is_valid_skill_id(skill_id, max_chars=settings.max_skill_id_chars):
            warning = f"Skipped skill with invalid directory name: {skill_id}"
            LOGGER.warning(warning)
            warnings.append(warning)
            continue
        parsed = parse_skill_header(child / "SKILL.md", skill_id=skill_id, settings=settings)
        if isinstance(parsed, str):
            LOGGER.warning(parsed)
            warnings.append(parsed)
            continue
        skills.append(parsed)

    return SkillCatalog(skills=tuple(skills), warnings=tuple(warnings))


def get_skill(settings: SkillsSettings, skill_id: str) -> SkillHeader | None:
    normalized = skill_id.strip()
    if not is_valid_skill_id(normalized, max_chars=settings.max_skill_id_chars):
        return None
    catalog = load_skill_catalog(settings)
    for skill in catalog.skills:
        if skill.skill_id == normalized:
            return skill
    return None


def parse_skill_header(
    skill_md_path: Path,
    *,
    skill_id: str,
    settings: SkillsSettings,
) -> SkillHeader | str:
    if not skill_md_path.exists():
        return f"Skipped skill {skill_id}: missing SKILL.md."
    if not skill_md_path.is_file():
        return f"Skipped skill {skill_id}: SKILL.md is not a file."
    if skill_md_path.is_symlink():
        return f"Skipped skill {skill_id}: SKILL.md may not be a symlink."

    try:
        raw_text = skill_md_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Skipped skill {skill_id}: SKILL.md is not valid UTF-8."
    except OSError as exc:
        return f"Skipped skill {skill_id}: could not read SKILL.md: {exc}"

    if len(raw_text) > settings.max_skill_md_chars:
        return f"Skipped skill {skill_id}: SKILL.md exceeds configured size limit."

    frontmatter = _parse_frontmatter(raw_text)
    raw_name = frontmatter.get("name")
    name = str(raw_name).strip() if raw_name is not None else skill_id
    if not name:
        name = skill_id

    raw_description = frontmatter.get("description")
    description = str(raw_description).strip() if raw_description is not None else ""
    if not description:
        return f"Skipped skill {skill_id}: missing frontmatter description."

    raw_compatibility = frontmatter.get("compatibility")
    compatibility = (
        str(raw_compatibility).strip()
        if raw_compatibility is not None and str(raw_compatibility).strip()
        else None
    )
    raw_metadata = frontmatter.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else None

    return SkillHeader(
        skill_id=skill_id,
        name=name,
        description=description,
        path=skill_md_path.parent,
        compatibility=compatibility,
        metadata=dict(metadata) if metadata is not None else None,
    )


def read_skill_markdown(settings: SkillsSettings, skill: SkillHeader) -> str:
    skill_dir = _resolve_skill_dir(settings, skill.skill_id)
    skill_md_path = skill_dir / "SKILL.md"
    if skill_md_path.is_symlink():
        raise ValueError("SKILL.md may not be a symlink.")
    text = skill_md_path.read_text(encoding="utf-8")
    if len(text) <= settings.max_skill_md_chars:
        return text
    return (
        text[: settings.max_skill_md_chars]
        + "\n\n[SKILL.md truncated at configured character limit.]"
    )


def list_skill_resources(settings: SkillsSettings, skill: SkillHeader) -> tuple[SkillResource, ...]:
    skill_dir = _resolve_skill_dir(settings, skill.skill_id)
    resources: list[SkillResource] = []
    for path in sorted(skill_dir.rglob("*"), key=lambda item: item.relative_to(skill_dir).as_posix()):
        if path.name == "SKILL.md":
            continue
        if path.name.startswith(".env"):
            continue
        if path.is_symlink():
            continue
        if path == skill_dir:
            continue
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(skill_dir.resolve(strict=False))
        except ValueError:
            continue
        relative = path.relative_to(skill_dir).as_posix()
        if path.is_dir():
            kind = "dir"
            size_bytes = None
        elif path.is_file():
            kind = "file"
            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = None
        else:
            continue
        resources.append(SkillResource(path=relative, kind=kind, size_bytes=size_bytes))
        if len(resources) >= settings.max_resource_listing_count:
            break
    return tuple(resources)


def search_skills(catalog: SkillCatalog, query: str) -> tuple[SkillHeader, ...]:
    normalized = _normalize_query(query)
    if not normalized:
        return tuple(sorted(catalog.skills, key=lambda skill: skill.skill_id))

    tokens = tuple(token for token in normalized.split() if token)
    scored: list[tuple[int, SkillHeader]] = []
    for skill in catalog.skills:
        score = _score_skill(skill, query=normalized, tokens=tokens)
        if score > 0:
            scored.append((score, skill))
    scored.sort(key=lambda item: (-item[0], item[1].skill_id))
    return tuple(skill for _score, skill in scored)


def _parse_frontmatter(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_BOUNDARY:
        return {}
    for index in range(1, len(lines)):
        if lines[index].strip() == _FRONTMATTER_BOUNDARY:
            raw_frontmatter = "\n".join(lines[1:index])
            try:
                payload = yaml.safe_load(raw_frontmatter)
            except yaml.YAMLError:
                return {}
            return dict(payload) if isinstance(payload, dict) else {}
    return {}


def _score_skill(
    skill: SkillHeader,
    *,
    query: str,
    tokens: tuple[str, ...],
) -> int:
    skill_id = skill.skill_id.lower()
    name = skill.name.lower()
    description = skill.description.lower()
    compatibility = (skill.compatibility or "").lower()
    searchable = " ".join(part for part in (skill_id, name, description, compatibility) if part)

    if query == skill_id or query == name:
        return 1000
    score = 0
    if query in skill_id or query in name:
        score += 700
    if query in description:
        score += 400
    if compatibility and query in compatibility:
        score += 250
    token_hits = sum(1 for token in tokens if token in searchable)
    score += token_hits * 50
    return score


def _normalize_query(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _resolve_skill_dir(settings: SkillsSettings, skill_id: str) -> Path:
    if not is_valid_skill_id(skill_id, max_chars=settings.max_skill_id_chars):
        raise ValueError("skill_id must be a canonical skill directory name.")
    skill_dir = (settings.skills_dir / skill_id).resolve(strict=False)
    skills_root = settings.skills_dir.resolve(strict=False)
    try:
        skill_dir.relative_to(skills_root)
    except ValueError as exc:
        raise ValueError("skill path must stay inside the workspace skills directory.") from exc
    return skill_dir
