"""Shared types for workspace-backed skills."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class SkillHeader:
    skill_id: str
    name: str
    description: str
    path: Path
    compatibility: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class SkillCatalog:
    skills: tuple[SkillHeader, ...]
    warnings: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class SkillResource:
    path: str
    kind: str
    size_bytes: int | None = None


@dataclass(slots=True, frozen=True)
class SkillImportDetail:
    skill_id: str
    source_dir: Path
    target_dir: Path


@dataclass(slots=True, frozen=True)
class SkillImportResult:
    imported: tuple[str, ...] = ()
    already_present: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    cleaned: tuple[str, ...] = ()
    imported_details: tuple[SkillImportDetail, ...] = ()
    already_present_details: tuple[SkillImportDetail, ...] = ()
    conflict_details: tuple[SkillImportDetail, ...] = ()

    @property
    def has_reportable_changes(self) -> bool:
        return bool(self.imported or self.conflicts or self.warnings)
