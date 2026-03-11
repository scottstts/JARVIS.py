"""ffmpeg_cli discoverable entry."""

from __future__ import annotations

from ...types import DiscoverableTool

_USAGE = (
    "Use the basic `bash` tool to run `ffmpeg` or `ffprobe` directly inside the container. "
    "Example: `ffmpeg -i /workspace/input.mp4 /workspace/output.mp3`. Keep all input and "
    "output paths inside `/workspace`."
)


def build_ffmpeg_cli_discoverable() -> DiscoverableTool:
    """Build the discoverable catalog entry for ffmpeg_cli."""

    return DiscoverableTool(
        name="ffmpeg_cli",
        aliases=("ffmpeg", "ffprobe", "media_convert"),
        purpose=(
            "Use the installed ffmpeg or ffprobe CLI through bash for audio or video "
            "conversion, trimming, muxing, probing, and stream extraction."
        ),
        detailed_description=(
            "This is a docs-only discoverable entry with no separate runtime. After "
            "discovery, invoke `ffmpeg` or `ffprobe` through the basic `bash` tool."
        ),
        usage=_USAGE,
        metadata={
            "operator": "bash",
            "commands": ["ffmpeg", "ffprobe"],
            "runtime": "docs_only_discoverable",
        },
    )
