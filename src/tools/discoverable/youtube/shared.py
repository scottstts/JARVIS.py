"""Shared constants and helpers for the YouTube discoverable tool."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

GEMINI_YOUTUBE_MODEL = "gemini-3-flash-preview"
MAX_YOUTUBE_VIDEO_URLS = 10
YOUTUBE_TOOL_SYSTEM_INSTRUCTION = (
    "You are helping another agent understand one or more provided public YouTube videos. "
    "Ground your response in the provided videos, keep it useful for downstream agent "
    "reasoning, and be explicit when a detail is unclear, absent, or uncertain instead of "
    "guessing. When multiple videos are provided, keep per-video observations clearly "
    "distinguishable and call out meaningful cross-video agreements or conflicts."
)
DEFAULT_YOUTUBE_OBJECTIVES = (
    "Summarize each provided video clearly and concisely, focusing on all main topics, key "
    "claims, important details, and actionable takeaways. When multiple videos are provided, "
    "also include a short combined synthesis that compares or connects them."
)

_YOUTUBE_URL_PATTERNS = (
    re.compile(
        r"^https?://(?:www\.|m\.|music\.)?youtube\.com/watch\?(?:[^\s#]*&)?v="
        r"[A-Za-z0-9_-]{11}(?:[&#?][^\s]*)?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^https?://(?:www\.)?youtu\.be/[A-Za-z0-9_-]{11}(?:[?#][^\s]*)?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^https?://(?:www\.|m\.|music\.)?youtube\.com/shorts/"
        r"[A-Za-z0-9_-]{11}(?:[?#][^\s]*)?$",
        re.IGNORECASE,
    ),
)


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_video_urls(value: Any) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    normalized = [str(item).strip() for item in value]
    if not normalized:
        return None
    return normalized


def is_valid_youtube_video_url(url: str) -> bool:
    normalized = url.strip()
    if not normalized:
        return False
    return any(pattern.fullmatch(normalized) for pattern in _YOUTUBE_URL_PATTERNS)


def collect_invalid_video_urls(video_urls: Sequence[str]) -> list[tuple[int, str]]:
    invalid: list[tuple[int, str]] = []
    for index, video_url in enumerate(video_urls, start=1):
        if not is_valid_youtube_video_url(video_url):
            invalid.append((index, video_url))
    return invalid


def format_invalid_video_urls(invalid_video_urls: Sequence[tuple[int, str]]) -> str:
    return "; ".join(
        f"[{index}] {video_url or '<empty>'}"
        for index, video_url in invalid_video_urls
    )


def build_youtube_system_instruction(objectives: str) -> str:
    normalized_objectives = objectives.strip()
    return (
        f"{YOUTUBE_TOOL_SYSTEM_INSTRUCTION}\n\n"
        "Current task:\n"
        f"{normalized_objectives}"
    )
