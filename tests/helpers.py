"""Shared testing helpers."""

from __future__ import annotations

from pathlib import Path

from core.config import ContextPolicySettings, CoreSettings


def build_core_settings(
    *,
    root_dir: Path,
    context_window_tokens: int = 400_000,
    compact_threshold_tokens: int = 350_000,
    compact_reserve_output_tokens: int = 16_000,
    compact_reserve_overhead_tokens: int = 10_000,
) -> CoreSettings:
    identities_dir = root_dir / "identities"
    workspace_dir = root_dir / "workspace"
    identities_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (identities_dir / "PROGRAM.md").write_text("PROGRAM PROMPT", encoding="utf-8")
    (identities_dir / "REACTOR.md").write_text("REACTOR PROMPT", encoding="utf-8")
    (identities_dir / "USER.md").write_text("USER PROMPT", encoding="utf-8")
    (identities_dir / "ARMOR.md").write_text("ARMOR PROMPT", encoding="utf-8")

    return CoreSettings(
        context_policy=ContextPolicySettings(
            context_window_tokens=context_window_tokens,
            compact_threshold_tokens=compact_threshold_tokens,
            compact_reserve_output_tokens=compact_reserve_output_tokens,
            compact_reserve_overhead_tokens=compact_reserve_overhead_tokens,
        ),
        workspace_dir=workspace_dir,
        transcript_archive_dir=root_dir / "archive" / "transcripts",
        identities_dir=identities_dir,
    )
