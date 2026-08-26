"""Stable content-derived workspace revision fingerprints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
import os
from pathlib import Path
import stat
from typing import Any

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
_RUNTIME_MANAGED_SNAPSHOT_ROOTS = frozenset({"archive", ".jarvis_internal"})

WorkspaceSnapshot = dict[str, str]


class WorkspaceSnapshotError(RuntimeError):
    """Raised when a workspace snapshot cannot be captured consistently."""


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


def workspace_snapshot_paths(
    workspace_dir: Path,
    paths: Iterable[Path | str],
) -> WorkspaceSnapshot:
    """Capture file-state fingerprints for bounded workspace paths.

    The snapshot is deliberately independent of Git. It records material path shape and
    content, including untracked files, so a later snapshot can identify net changes in a
    lease scope even when the workspace was already dirty before the actor started. Runtime-
    managed ``archive`` and ``.jarvis_internal`` roots are excluded from the manifest.
    """

    root = workspace_dir.resolve(strict=False)
    resolved_paths: set[Path] = set()
    for raw_path in paths:
        candidate = Path(raw_path)
        if candidate.is_absolute() and (
            candidate == Path("/workspace")
            or candidate.is_relative_to(Path("/workspace"))
        ):
            candidate = root / candidate.relative_to(Path("/workspace"))
        elif not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceSnapshotError(
                f"Could not resolve workspace snapshot path {candidate}: {exc}"
            ) from exc
        if resolved != root and not resolved.is_relative_to(root):
            raise WorkspaceSnapshotError(
                f"Workspace snapshot path escapes the workspace: {resolved}"
            )
        resolved_paths.add(resolved)

    snapshot: WorkspaceSnapshot = {}
    for path in sorted(resolved_paths):
        relative = path.relative_to(root)
        _capture_snapshot_entry(
            path=path,
            relative=relative,
            snapshot=snapshot,
        )
    return snapshot


def diff_workspace_snapshots(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> tuple[str, ...]:
    """Return sorted workspace-relative paths whose snapshot state changed."""

    paths = set(before) | set(after)
    return tuple(sorted(path for path in paths if before.get(path) != after.get(path)))


def _capture_snapshot_entry(
    *,
    path: Path,
    relative: Path,
    snapshot: WorkspaceSnapshot,
) -> None:
    if _snapshot_path_is_ignored(relative):
        return
    key = str(relative)
    try:
        info = path.lstat()
    except FileNotFoundError:
        snapshot[key] = "missing"
        return
    except OSError as exc:
        raise WorkspaceSnapshotError(
            f"Could not inspect workspace snapshot path {path}: {exc}"
        ) from exc

    mode = info.st_mode
    if stat.S_ISLNK(mode):
        try:
            target = os.readlink(path)
            after = path.lstat()
        except OSError as exc:
            raise WorkspaceSnapshotError(
                f"Could not read workspace snapshot symlink {path}: {exc}"
            ) from exc
        if _snapshot_entry_identity(info) != _snapshot_entry_identity(after):
            raise WorkspaceSnapshotError(
                f"Workspace snapshot symlink changed while being read: {path}"
            )
        snapshot[key] = f"symlink:{target}"
        return
    if stat.S_ISDIR(mode):
        snapshot[key] = f"directory:{stat.S_IMODE(mode):o}"
        try:
            children = sorted(path.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise WorkspaceSnapshotError(
                f"Could not enumerate workspace snapshot directory {path}: {exc}"
            ) from exc
        for child in children:
            _capture_snapshot_entry(
                path=child,
                relative=relative / child.name,
                snapshot=snapshot,
            )
        try:
            after = path.lstat()
            children_after = sorted(path.iterdir(), key=lambda child: child.name)
        except FileNotFoundError as exc:
            raise WorkspaceSnapshotError(
                f"Workspace snapshot directory disappeared while being read: {path}"
            ) from exc
        except OSError as exc:
            raise WorkspaceSnapshotError(
                f"Could not re-check workspace snapshot directory {path}: {exc}"
            ) from exc
        visible_children = tuple(
            child.name
            for child in children
            if not _snapshot_path_is_ignored(relative / child.name)
        )
        visible_children_after = tuple(
            child.name
            for child in children_after
            if not _snapshot_path_is_ignored(relative / child.name)
        )
        if (
            _snapshot_directory_identity(info) != _snapshot_directory_identity(after)
            or visible_children != visible_children_after
        ):
            raise WorkspaceSnapshotError(
                f"Workspace snapshot directory changed while being read: {path}"
            )
        return
    if stat.S_ISREG(mode):
        snapshot[key] = _fingerprint_snapshot_file(path, info)
        return
    snapshot[key] = (
        f"special:{stat.S_IFMT(mode):o}:{stat.S_IMODE(mode):o}:{info.st_size}"
    )


def _fingerprint_snapshot_file(path: Path, before: os.stat_result) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = path.lstat()
    except FileNotFoundError as exc:
        raise WorkspaceSnapshotError(
            f"Workspace snapshot file disappeared while being read: {path}"
        ) from exc
    except OSError as exc:
        raise WorkspaceSnapshotError(
            f"Could not read workspace snapshot file {path}: {exc}"
        ) from exc

    if _snapshot_entry_identity(before) != _snapshot_entry_identity(after):
        raise WorkspaceSnapshotError(
            f"Workspace snapshot file changed while being read: {path}"
        )
    return f"file:{stat.S_IMODE(before.st_mode):o}:{digest.hexdigest()}"


def _snapshot_entry_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_mode,
    )


def _snapshot_directory_identity(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode)


def _snapshot_path_is_ignored(relative: Path) -> bool:
    """Ignore only runtime-owned roots; all other owned content is material evidence."""

    return bool(relative.parts) and relative.parts[0] in _RUNTIME_MANAGED_SNAPSHOT_ROOTS


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
