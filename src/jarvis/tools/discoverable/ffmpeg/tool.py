"""ffmpeg discoverable entry."""

from __future__ import annotations

from ...types import DiscoverableTool

_USAGE = (
    "Use the basic `bash` tool to run `ffmpeg` or `ffprobe` on files in `/workspace`."
)


def build_ffmpeg_discoverable() -> DiscoverableTool:
    """Build the discoverable catalog entry for ffmpeg."""

    return DiscoverableTool(
        name="ffmpeg",
        aliases=("ffprobe", "media convert"),
        purpose=(
            "Use the installed ffmpeg or ffprobe CLI through bash for audio or video "
            "conversion, trimming, muxing, probing, and stream extraction."
        ),
        detailed_description="Docs-only entry. Run `ffmpeg` or `ffprobe` through `bash`.",
        usage=_USAGE,
    )
