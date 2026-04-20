"""Discoverable tool packages."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .email import EmailPolicy, build_email_discoverable, build_email_tool
    from .ffmpeg import build_ffmpeg_discoverable
    from .generate_edit_image import (
        GenerateEditImagePolicy,
        build_generate_edit_image_discoverable,
        build_generate_edit_image_tool,
    )
    from .transcribe import (
        TranscribePolicy,
        build_transcribe_discoverable,
        build_transcribe_tool,
    )

__all__ = [
    "EmailPolicy",
    "build_email_discoverable",
    "build_email_tool",
    "build_ffmpeg_discoverable",
    "GenerateEditImagePolicy",
    "TranscribePolicy",
    "build_generate_edit_image_discoverable",
    "build_generate_edit_image_tool",
    "build_transcribe_discoverable",
    "build_transcribe_tool",
]

_EXPORTS = {
    "EmailPolicy": (".email", "EmailPolicy"),
    "build_email_discoverable": (".email", "build_email_discoverable"),
    "build_email_tool": (".email", "build_email_tool"),
    "build_ffmpeg_discoverable": (".ffmpeg", "build_ffmpeg_discoverable"),
    "GenerateEditImagePolicy": (
        ".generate_edit_image",
        "GenerateEditImagePolicy",
    ),
    "TranscribePolicy": (".transcribe", "TranscribePolicy"),
    "build_generate_edit_image_discoverable": (
        ".generate_edit_image",
        "build_generate_edit_image_discoverable",
    ),
    "build_generate_edit_image_tool": (
        ".generate_edit_image",
        "build_generate_edit_image_tool",
    ),
    "build_transcribe_discoverable": (
        ".transcribe",
        "build_transcribe_discoverable",
    ),
    "build_transcribe_tool": (".transcribe", "build_transcribe_tool"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
