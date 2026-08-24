"""Durable, de-duplicated runtime exception records."""

from __future__ import annotations

import json
import traceback
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from jarvis.storage.layout import transcript_archive_root_from_runtime_path

_ERROR_LOG_PATH_ATTRIBUTE = "_jarvis_runtime_error_log_path"
_ERROR_LOG_METADATA_KEY = "runtime_error_log_path"


def record_runtime_error(
    *,
    transcript_archive_dir: Path,
    route_id: str,
    session_id: str | None,
    component: str,
    event: str,
    agent_kind: str,
    exc: Exception,
    error_code: str,
    message: str,
    context: Mapping[str, Any] | None = None,
) -> Path:
    """Append one JSONL error record and reuse it if the exception propagates upward."""

    existing_path = _recorded_error_path(exc)
    if existing_path is not None:
        return existing_path

    transcript_root = transcript_archive_root_from_runtime_path(
        transcript_archive_dir=transcript_archive_dir,
        route_id=route_id,
    )
    stem = (session_id or f"route_{route_id}_unbound").strip()
    sanitized = (stem or f"route_{route_id}_unbound").replace("/", "_")
    error_log_path = transcript_root.parent / "error_logs" / f"{sanitized}.jsonl"
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema": "jarvis.runtime_error.v1",
        "logged_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "component": component,
        "event": event,
        "level": "ERROR",
        "route_id": route_id,
        "agent_kind": agent_kind,
        "message": message,
        "error_code": error_code,
        "session_id": session_id,
        **dict(context or {}),
        "exception_type": type(exc).__name__,
        "exception_module": type(exc).__module__,
        "exception_message": str(exc),
        "exception_metadata": _exception_metadata(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }
    with error_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str))
        handle.write("\n")
    _mark_error_recorded(exc, error_log_path)
    return error_log_path


def _recorded_error_path(exc: Exception) -> Path | None:
    raw = getattr(exc, _ERROR_LOG_PATH_ATTRIBUTE, None)
    if isinstance(raw, str) and raw:
        return Path(raw)
    metadata = getattr(exc, "metadata", None)
    if isinstance(metadata, Mapping):
        raw = metadata.get(_ERROR_LOG_METADATA_KEY)
        if isinstance(raw, str) and raw:
            return Path(raw)
    return None


def _mark_error_recorded(exc: Exception, path: Path) -> None:
    rendered = str(path)
    try:
        setattr(exc, _ERROR_LOG_PATH_ATTRIBUTE, rendered)
        return
    except (AttributeError, TypeError):
        pass
    metadata = getattr(exc, "metadata", None)
    if isinstance(metadata, dict):
        metadata.setdefault(_ERROR_LOG_METADATA_KEY, rendered)


def _exception_metadata(exc: Exception) -> dict[str, Any]:
    metadata = getattr(exc, "metadata", None)
    if not isinstance(metadata, Mapping):
        return {}
    return json.loads(json.dumps(dict(metadata), ensure_ascii=False, default=str))
