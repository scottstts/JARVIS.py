"""Policy checks for the youtube tool."""

from __future__ import annotations

from typing import Any

from ...types import ToolExecutionContext, ToolPolicyDecision
from .shared import (
    MAX_YOUTUBE_VIDEO_URLS,
    collect_invalid_video_urls,
    format_invalid_video_urls,
    normalize_video_urls,
)


class YouTubePolicy:
    """Restricts youtube to explicit valid public YouTube video URLs."""

    def authorize(
        self,
        *,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context
        video_urls = normalize_video_urls(arguments.get("video_urls"))
        if video_urls is None:
            return ToolPolicyDecision(
                allowed=False,
                reason="youtube requires a non-empty 'video_urls' list.",
            )

        invalid_video_urls = collect_invalid_video_urls(video_urls)
        if invalid_video_urls:
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "youtube video_urls must all be valid YouTube video URLs. "
                    f"Invalid entries: {format_invalid_video_urls(invalid_video_urls)}."
                ),
            )

        if len(video_urls) > MAX_YOUTUBE_VIDEO_URLS:
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "youtube supports at most "
                    f"{MAX_YOUTUBE_VIDEO_URLS} video_urls per call."
                ),
            )

        return ToolPolicyDecision(allowed=True)
