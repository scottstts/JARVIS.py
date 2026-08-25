"""Stable content-derived workspace revision fingerprints."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

_IGNORED_REVISION_PARTS = frozenset(
    {
        ".git",
        ".cache",
        ".coverage",
        ".jarvis_internal",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".npm",
        ".tox",
        "__pycache__",
        "coverage",
        "htmlcov",
        "node_modules",
    }
)


def workspace_revision(workspace_dir: Path) -> str:
    """Return Git identity plus a deterministic fingerprint of material file content."""

    return workspace_revision_excluding(workspace_dir, ())


def workspace_revision_excluding(
    workspace_dir: Path,
    excluded_paths: Iterable[Path],
) -> str:
    """Return a workspace revision while excluding concurrently owned path roots."""

    root = workspace_dir.resolve(strict=False)
    excluded = tuple(
        sorted(
            {
                path.resolve(strict=False)
                for path in excluded_paths
                if path.resolve(strict=False) == root
                or path.resolve(strict=False).is_relative_to(root)
            }
        )
    )
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
        resolved = path.resolve(strict=False)
        if any(
            resolved == excluded_path or resolved.is_relative_to(excluded_path)
            for excluded_path in excluded
        ):
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


def workspace_paths_revision(workspace_dir: Path, paths: Iterable[Path]) -> str:
    """Return a content fingerprint for a bounded set of workspace paths."""

    root = workspace_dir.resolve(strict=False)
    fingerprint = sha256()
    resolved_paths = tuple(sorted({path.resolve(strict=False) for path in paths}))
    for path in resolved_paths:
        if path != root and not path.is_relative_to(root):
            raise ValueError("workspace revision paths must stay inside the workspace.")
        relative = path.relative_to(root)
        fingerprint.update(str(relative).encode("utf-8", errors="surrogateescape"))
        fingerprint.update(b"\0")
        if not path.exists() and not path.is_symlink():
            fingerprint.update(b"missing\n")
            continue
        candidates = (path,) if not path.is_dir() else tuple(sorted(path.rglob("*")))
        for candidate in candidates:
            candidate_relative = candidate.relative_to(root)
            if _revision_path_is_ignored(candidate_relative):
                continue
            _update_path_fingerprint(
                fingerprint,
                path=candidate,
                relative=candidate_relative,
            )
    return fingerprint.hexdigest()


def _update_path_fingerprint(fingerprint: Any, *, path: Path, relative: Path) -> None:
    try:
        if path.is_symlink():
            fingerprint.update(str(relative).encode("utf-8", errors="surrogateescape"))
            fingerprint.update(b"\0symlink\0")
            fingerprint.update(str(path.readlink()).encode("utf-8", errors="surrogateescape"))
            fingerprint.update(b"\n")
            return
        if not path.is_file():
            return
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


def _revision_path_is_ignored(relative: Path) -> bool:
    if any(part in _IGNORED_REVISION_PARTS for part in relative.parts):
        return True
    return relative.parts[:1] == ("archive",)


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
