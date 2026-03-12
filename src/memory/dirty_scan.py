"""Checksum-based dirty-file detection for opportunistic memory reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .parser import checksum_file
from .types import DirtyDocument


def scan_dirty_documents(
    *,
    markdown_paths: tuple[Path, ...],
    indexed_checksums: dict[str, str],
) -> tuple[DirtyDocument, ...]:
    detected_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    dirty: list[DirtyDocument] = []
    seen_paths: set[str] = set()

    for path in markdown_paths:
        resolved = str(path)
        seen_paths.add(resolved)
        checksum = checksum_file(path)
        indexed_checksum = indexed_checksums.get(resolved)
        if indexed_checksum is None:
            dirty.append(DirtyDocument(path=path, detected_at=detected_at, reason="missing_from_index"))
            continue
        if indexed_checksum != checksum:
            dirty.append(DirtyDocument(path=path, detected_at=detected_at, reason="checksum_mismatch"))

    for indexed_path in sorted(indexed_checksums):
        if indexed_path in seen_paths:
            continue
        dirty.append(
            DirtyDocument(
                path=Path(indexed_path),
                detected_at=detected_at,
                reason="missing_from_disk",
            )
        )
    return tuple(dirty)
