"""Stable content-derived workspace revision fingerprints."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

_IGNORED_REVISION_PARTS = frozenset(
    {
        ".git",
        ".jarvis_internal",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)


def workspace_revision(workspace_dir: Path) -> str:
    """Return Git identity plus a deterministic fingerprint of material file content."""

    root = workspace_dir.resolve(strict=False)
    revision = _git_revision(root)
    fingerprint = sha256()
    try:
        paths = sorted(root.rglob("*"))
    except OSError as exc:
        fingerprint.update(f"scan-error:{type(exc).__name__}".encode("ascii"))
        return f"{revision}:{fingerprint.hexdigest()}"

    for path in paths:
        relative = path.relative_to(root)
        if _revision_path_is_ignored(relative):
            continue
        try:
            if path.is_symlink():
                fingerprint.update(
                    str(relative).encode("utf-8", errors="surrogateescape")
                )
                fingerprint.update(b"\0symlink\0")
                fingerprint.update(
                    str(path.readlink()).encode("utf-8", errors="surrogateescape")
                )
                fingerprint.update(b"\n")
                continue
            if not path.is_file():
                continue
            fingerprint.update(str(relative).encode("utf-8", errors="surrogateescape"))
            fingerprint.update(b"\0")
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    fingerprint.update(chunk)
            fingerprint.update(b"\n")
        except OSError as exc:
            fingerprint.update(str(relative).encode("utf-8", errors="surrogateescape"))
            fingerprint.update(b"\0unreadable\0")
            fingerprint.update(type(exc).__name__.encode("ascii"))
            fingerprint.update(b"\n")
    return f"{revision}:{fingerprint.hexdigest()}"


def _revision_path_is_ignored(relative: Path) -> bool:
    if any(part in _IGNORED_REVISION_PARTS for part in relative.parts):
        return True
    return relative.parts[:2] == ("archive", "transcripts")


def _git_revision(workspace_dir: Path) -> str:
    git_dir = workspace_dir / ".git"
    try:
        if git_dir.is_file():
            raw = git_dir.read_text(encoding="utf-8", errors="replace").strip()
            if raw.startswith("gitdir:"):
                candidate = Path(raw.removeprefix("gitdir:").strip())
                git_dir = (
                    candidate
                    if candidate.is_absolute()
                    else (workspace_dir / candidate).resolve(strict=False)
                )
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or "unborn"
        ref = head.removeprefix("ref:").strip()
        ref_path = git_dir / ref
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip() or "unborn"
        packed = (git_dir / "packed-refs").read_text(
            encoding="utf-8",
            errors="replace",
        )
        for line in packed.splitlines():
            if line and not line.startswith(("#", "^")) and line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except OSError:
        return "unversioned"
    return "unborn"
