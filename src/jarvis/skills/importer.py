"""Import staged installer output into the canonical skills directory."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile

from jarvis.logging_setup import get_application_logger

from .catalog import is_valid_skill_id, parse_skill_header
from .config import IGNORED_IMPORT_NAMES, SkillsSettings
from .types import SkillImportDetail, SkillImportResult

LOGGER = get_application_logger(__name__)

_SLUG_INVALID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def import_staged_skills(settings: SkillsSettings) -> SkillImportResult:
    """Copy recognized staged skills into /workspace/skills."""

    settings.skills_dir.mkdir(parents=True, exist_ok=True)
    imported: list[str] = []
    already_present: list[str] = []
    conflicts: list[str] = []
    skipped: list[str] = []
    warnings: list[str] = []
    cleaned: list[str] = []
    imported_details: list[SkillImportDetail] = []
    already_present_details: list[SkillImportDetail] = []
    conflict_details: list[SkillImportDetail] = []

    for staging_root in settings.staging_roots:
        if not staging_root.exists() or not staging_root.is_dir():
            continue
        try:
            children = sorted(staging_root.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            warning = f"Could not scan staged skills at {staging_root}: {exc}"
            LOGGER.warning(warning)
            warnings.append(warning)
            continue
        for source_dir in children:
            if not source_dir.is_dir() or source_dir.is_symlink():
                continue
            if source_dir.name in IGNORED_IMPORT_NAMES:
                continue
            parsed = parse_skill_header(
                source_dir / "SKILL.md",
                skill_id=source_dir.name,
                settings=settings,
            )
            if isinstance(parsed, str):
                skipped.append(f"{source_dir}: {parsed}")
                continue
            skill_id = _target_skill_id(
                source_dir=source_dir,
                skill_name=parsed.name,
                settings=settings,
            )
            if skill_id is None:
                warning = f"Could not determine a safe skill id for staged skill {source_dir}."
                LOGGER.warning(warning)
                warnings.append(warning)
                continue

            target_dir = settings.skills_dir / skill_id
            detail = SkillImportDetail(
                skill_id=skill_id,
                source_dir=source_dir,
                target_dir=target_dir,
            )
            try:
                source_digest = _skill_payload_digest(source_dir, settings=settings)
            except OSError as exc:
                warning = f"Could not hash staged skill {source_dir}: {exc}"
                LOGGER.warning(warning)
                warnings.append(warning)
                continue
            if target_dir.exists() or target_dir.is_symlink():
                if target_dir.is_symlink():
                    conflicts.append(skill_id)
                    conflict_details.append(detail)
                    continue
                try:
                    target_digest = _skill_payload_digest(target_dir, settings=settings)
                except OSError as exc:
                    warning = f"Could not hash installed skill {target_dir}: {exc}"
                    LOGGER.warning(warning)
                    warnings.append(warning)
                    continue
                if source_digest == target_digest:
                    already_present.append(skill_id)
                    already_present_details.append(detail)
                    _cleanup_import_source(
                        source_dir=source_dir,
                        skill_id=skill_id,
                        cleaned=cleaned,
                        warnings=warnings,
                    )
                else:
                    conflicts.append(skill_id)
                    conflict_details.append(detail)
                continue

            try:
                _copy_skill_payload(source_dir, target_dir, settings=settings)
            except OSError as exc:
                warning = f"Could not import staged skill {source_dir} to {target_dir}: {exc}"
                LOGGER.warning(warning)
                warnings.append(warning)
                continue
            imported.append(skill_id)
            imported_details.append(detail)
            _cleanup_import_source(
                source_dir=source_dir,
                skill_id=skill_id,
                cleaned=cleaned,
                warnings=warnings,
            )

    return SkillImportResult(
        imported=tuple(sorted(set(imported))),
        already_present=tuple(sorted(set(already_present))),
        conflicts=tuple(sorted(set(conflicts))),
        skipped=tuple(skipped),
        warnings=tuple(warnings),
        cleaned=tuple(sorted(set(cleaned))),
        imported_details=_dedupe_details(imported_details),
        already_present_details=_dedupe_details(already_present_details),
        conflict_details=_dedupe_details(conflict_details),
    )


def _target_skill_id(
    *,
    source_dir: Path,
    skill_name: str,
    settings: SkillsSettings,
) -> str | None:
    if is_valid_skill_id(source_dir.name, max_chars=settings.max_skill_id_chars):
        return source_dir.name
    slug = _slugify_skill_name(skill_name)
    if slug and is_valid_skill_id(slug, max_chars=settings.max_skill_id_chars):
        return slug
    return None


def _slugify_skill_name(value: str) -> str:
    slug = _SLUG_INVALID_CHARS.sub("-", value.strip()).strip(".-_")
    return slug[:128]


def _skill_payload_digest(source_dir: Path, *, settings: SkillsSettings) -> str:
    digest = hashlib.sha256()
    for file_path in _iter_payload_files(source_dir, settings=settings):
        relative = file_path.relative_to(source_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_skill_payload(source_dir: Path, target_dir: Path, *, settings: SkillsSettings) -> None:
    temp_root = Path(
        tempfile.mkdtemp(
            prefix=f".{target_dir.name}.import-",
            dir=str(settings.skills_dir),
        )
    )
    try:
        for file_path in _iter_payload_files(source_dir, settings=settings):
            relative = file_path.relative_to(source_dir)
            target_path = temp_root / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target_path, follow_symlinks=False)
        temp_root.rename(target_dir)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _cleanup_import_source(
    *,
    source_dir: Path,
    skill_id: str,
    cleaned: list[str],
    warnings: list[str],
) -> None:
    try:
        shutil.rmtree(source_dir)
    except OSError as exc:
        warning = f"Could not clean staged skill source {source_dir}: {exc}"
        LOGGER.warning(warning)
        warnings.append(warning)
        return
    cleaned.append(skill_id)


def _dedupe_details(details: list[SkillImportDetail]) -> tuple[SkillImportDetail, ...]:
    deduped: dict[tuple[str, str, str], SkillImportDetail] = {}
    for detail in details:
        key = (detail.skill_id, str(detail.source_dir), str(detail.target_dir))
        deduped[key] = detail
    return tuple(deduped[key] for key in sorted(deduped))


def _iter_payload_files(source_dir: Path, *, settings: SkillsSettings) -> tuple[Path, ...]:
    root = source_dir.resolve(strict=False)
    files: list[Path] = []
    total_bytes = 0
    for current_root, dir_names, file_names in os.walk(source_dir, followlinks=False):
        current_path = Path(current_root)
        dir_names[:] = [
            name
            for name in dir_names
            if name not in IGNORED_IMPORT_NAMES and not name.startswith(".cache")
        ]
        for file_name in sorted(file_names):
            if file_name in IGNORED_IMPORT_NAMES or file_name.startswith(".env"):
                continue
            file_path = current_path / file_name
            if file_path.is_symlink() or not file_path.is_file():
                continue
            resolved = file_path.resolve(strict=False)
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            size = file_path.stat().st_size
            if size > settings.max_import_file_bytes:
                continue
            total_bytes += size
            if total_bytes > settings.max_import_total_bytes:
                raise OSError("staged skill payload exceeds total import byte limit")
            files.append(file_path)
            if len(files) > settings.max_import_file_count:
                raise OSError("staged skill payload exceeds file count limit")
    return tuple(sorted(files, key=lambda path: path.relative_to(source_dir).as_posix()))
