"""Shared helpers for the email discoverable tool."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mimetypes
from pathlib import Path
import re
from typing import Any

from ...config import ToolSettings
from ...types import ToolExecutionContext

_EMAIL_ADDRESS_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
_GLOB_PATTERN = re.compile(r"[*?\[]")
_DEFAULT_ATTACHMENT_MEDIA_TYPE = "application/octet-stream"


class EmailArgumentError(ValueError):
    """Raised when email arguments fail validation."""


@dataclass(slots=True, frozen=True)
class EmailAttachmentSpec:
    """Normalized attachment metadata used by both policy and execution."""

    raw_path: str
    resolved_path: Path
    filename: str
    media_type: str
    size_bytes: int
    sha256: str


@dataclass(slots=True, frozen=True)
class EmailRequest:
    """Validated email request payload."""

    to_email: str
    subject: str
    body_markdown: str
    attachment_specs: tuple[EmailAttachmentSpec, ...]
    request_hash: str


def build_email_request(
    *,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
    settings: ToolSettings,
) -> EmailRequest:
    """Normalize and validate one email tool request."""

    to_email = normalize_email_address(arguments.get("to_email"))
    if to_email is None:
        raise EmailArgumentError("email requires a valid 'to_email' address.")

    subject = normalize_non_empty_string(arguments.get("subject"))
    if subject is None:
        raise EmailArgumentError("email requires a non-empty 'subject'.")
    if len(subject) > settings.email_max_subject_chars:
        raise EmailArgumentError(
            "email subject length must be <= "
            f"{settings.email_max_subject_chars} characters."
        )

    body_markdown = normalize_non_empty_string(arguments.get("body"))
    if body_markdown is None:
        raise EmailArgumentError("email requires a non-empty 'body'.")
    if len(body_markdown) > settings.email_max_body_chars:
        raise EmailArgumentError(
            "email body length must be <= "
            f"{settings.email_max_body_chars} characters."
        )

    raw_attachment_paths = normalize_attachment_paths(arguments.get("attachment_paths"))
    if len(raw_attachment_paths) > settings.email_max_attachment_count:
        raise EmailArgumentError(
            "email supports at most "
            f"{settings.email_max_attachment_count} attachment_paths per call."
        )

    attachment_specs = build_attachment_specs(
        raw_paths=raw_attachment_paths,
        context=context,
        max_total_bytes=settings.email_max_total_attachment_bytes,
    )
    request_hash = build_email_request_hash(
        to_email=to_email,
        subject=subject,
        body_markdown=body_markdown,
        sender_email=settings.email_sender_address,
        attachment_specs=attachment_specs,
    )
    return EmailRequest(
        to_email=to_email,
        subject=subject,
        body_markdown=body_markdown,
        attachment_specs=attachment_specs,
        request_hash=request_hash,
    )


def normalize_non_empty_string(value: Any) -> str | None:
    """Return a stripped non-empty string or None."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_email_address(value: Any) -> str | None:
    """Return a normalized single email address or None."""

    normalized = normalize_non_empty_string(value)
    if normalized is None:
        return None
    if " " in normalized or "," in normalized or ";" in normalized:
        return None
    if not _EMAIL_ADDRESS_PATTERN.fullmatch(normalized):
        return None
    return normalized


def normalize_attachment_paths(value: Any) -> tuple[str, ...]:
    """Return normalized attachment paths or raise for invalid input."""

    if value is None:
        return ()
    if not isinstance(value, list):
        raise EmailArgumentError(
            "attachment_paths must be a list of workspace file paths when provided."
        )

    normalized: list[str] = []
    for index, item in enumerate(value, start=1):
        raw_path = normalize_non_empty_string(item)
        if raw_path is None:
            raise EmailArgumentError(
                f"attachment_paths[{index}] must be a non-empty string."
            )
        normalized.append(raw_path)
    return tuple(normalized)


def build_attachment_specs(
    *,
    raw_paths: tuple[str, ...],
    context: ToolExecutionContext,
    max_total_bytes: int,
) -> tuple[EmailAttachmentSpec, ...]:
    """Validate attachment paths and collect file metadata."""

    specs: list[EmailAttachmentSpec] = []
    total_bytes = 0
    for raw_path in raw_paths:
        if raw_path == "-":
            raise EmailArgumentError("email attachment_paths '-' is not allowed.")
        if raw_path.startswith("~") or _GLOB_PATTERN.search(raw_path):
            raise EmailArgumentError(
                f"email does not allow shell-expanded attachment path '{raw_path}'."
            )

        resolved_path = resolve_workspace_relative_path(raw_path, context)
        if not is_within_workspace(resolved_path, context.workspace_dir):
            raise EmailArgumentError(
                f"email attachments may only read files inside {context.workspace_dir}."
            )
        if contains_dot_env_path(resolved_path):
            raise EmailArgumentError(
                "email does not allow .env files or paths inside .env directories as attachments."
            )
        if not resolved_path.exists():
            raise EmailArgumentError(
                f"email attachment '{raw_path}' does not exist."
            )
        if not resolved_path.is_file():
            raise EmailArgumentError(
                f"email attachment '{raw_path}' must point to a file."
            )

        try:
            size_bytes = resolved_path.stat().st_size
        except OSError as exc:
            raise EmailArgumentError(
                f"failed to inspect attachment '{raw_path}': {exc}"
            ) from exc

        total_bytes += size_bytes
        if total_bytes > max_total_bytes:
            raise EmailArgumentError(
                "email attachments exceed the total size limit of "
                f"{max_total_bytes} bytes."
            )

        specs.append(
            EmailAttachmentSpec(
                raw_path=raw_path,
                resolved_path=resolved_path,
                filename=resolved_path.name,
                media_type=guess_attachment_media_type(resolved_path),
                size_bytes=size_bytes,
                sha256=sha256_file(resolved_path),
            )
        )

    return tuple(specs)


def build_email_request_hash(
    *,
    to_email: str,
    subject: str,
    body_markdown: str,
    sender_email: str | None,
    attachment_specs: tuple[EmailAttachmentSpec, ...],
) -> str:
    """Build an exact-approval hash for one email request."""

    payload = {
        "to_email": to_email,
        "subject": subject,
        "body_markdown": body_markdown,
        "sender_email": sender_email,
        "attachments": [
            {
                "raw_path": spec.raw_path,
                "sha256": spec.sha256,
                "size_bytes": spec.size_bytes,
            }
            for spec in attachment_specs
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def resolve_workspace_relative_path(raw_path: str, context: ToolExecutionContext) -> Path:
    """Resolve a tool path against the workspace."""

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = context.workspace_dir / candidate
    return candidate.resolve(strict=False)


def is_within_workspace(path: Path, workspace_dir: Path) -> bool:
    """Return whether the path stays inside the workspace."""

    workspace = workspace_dir.resolve(strict=False)
    try:
        path.relative_to(workspace)
        return True
    except ValueError:
        return False


def contains_dot_env_path(path: Path) -> bool:
    """Return whether the path targets a .env file or directory subtree."""

    return any(part == ".env" for part in path.parts)


def guess_attachment_media_type(path: Path) -> str:
    """Guess the MIME type for one attachment path."""

    media_type, _encoding = mimetypes.guess_type(path.name)
    return media_type or _DEFAULT_ATTACHMENT_MEDIA_TYPE


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(65_536):
                digest.update(chunk)
    except OSError as exc:
        raise EmailArgumentError(f"failed to read attachment '{path}': {exc}") from exc
    return digest.hexdigest()
