"""Shared constants and helpers for the transcribe discoverable tool."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...types import ToolExecutionContext

SUPPORTED_AUDIO_SUFFIXES = {
    ".flac": "flac",
    ".m4a": "m4a",
    ".mp3": "mp3",
    ".mp4": "mp4",
    ".mpeg": "mpeg",
    ".mpga": "mpga",
    ".ogg": "ogg",
    ".wav": "wav",
    ".webm": "webm",
}
SUPPORTED_AUDIO_FORMATS = tuple(
    sorted(dict.fromkeys(SUPPORTED_AUDIO_SUFFIXES.values()))
)
MAX_AUDIO_FILE_SIZE_BYTES = 25 * 1024 * 1024
GLOB_PATTERN = re.compile(r"[*?\[]")


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_audio_path(value: Any) -> str | None:
    return normalize_optional_string(value)


def detect_supported_audio_format(path: Path) -> str | None:
    return SUPPORTED_AUDIO_SUFFIXES.get(path.suffix.lower())


def resolve_workspace_relative_path(raw_path: str, context: ToolExecutionContext) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = context.workspace_dir / candidate
    return candidate.resolve(strict=False)


def is_within_workspace(path: Path, workspace_dir: Path) -> bool:
    workspace = workspace_dir.resolve(strict=False)
    try:
        path.relative_to(workspace)
        return True
    except ValueError:
        return False
