"""Shared transcript archive path helpers."""

from __future__ import annotations

from pathlib import Path

from jarvis.workspace_paths import resolve_workspace_child

_MAIN_AGENT_NAMESPACE = "jarvis"
_SUBAGENT_NAMESPACE = "subagents"


def resolve_transcript_archive_root(workspace_dir: Path) -> Path:
    """Resolve the shared transcript archive root."""

    return resolve_workspace_child(
        env_name="JARVIS_TRANSCRIPT_ARCHIVE_DIR",
        configured_default=None,
        workspace_dir=workspace_dir,
        child_name="archive/transcripts",
    )


def main_agent_transcript_dir(*, transcript_archive_root: Path, route_id: str) -> Path:
    """Return the route-local main-agent transcript directory."""

    return transcript_archive_root / _MAIN_AGENT_NAMESPACE / route_id


def transcript_archive_root_from_runtime_path(
    *,
    transcript_archive_dir: Path,
    route_id: str,
) -> Path:
    """Recover the shared transcript root from either a base root or a route-local main path."""

    if (
        transcript_archive_dir.name == route_id
        and transcript_archive_dir.parent.name == _MAIN_AGENT_NAMESPACE
    ):
        return transcript_archive_dir.parent.parent
    return transcript_archive_dir


def subagent_transcript_route_dir(*, transcript_archive_root: Path, route_id: str) -> Path:
    """Return the route-local subagent transcript directory."""

    return transcript_archive_root / _SUBAGENT_NAMESPACE / route_id


def subagent_session_storage_dir(
    *,
    transcript_archive_root: Path,
    route_id: str,
    owner_main_session_id: str,
    subagent_id: str,
) -> Path:
    """Return one subagent SessionStorage root for a specific owning main session."""

    return (
        subagent_transcript_route_dir(
            transcript_archive_root=transcript_archive_root,
            route_id=route_id,
        )
        / owner_main_session_id
        / subagent_id
    )
