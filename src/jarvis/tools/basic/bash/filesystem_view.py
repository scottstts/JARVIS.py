"""Mount-namespace filesystem capabilities for isolated Bash executions."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit("filesystem view requires CONFIG_JSON -- COMMAND [ARGS...]")
    config = json.loads(sys.argv[1])
    if not isinstance(config, dict):
        raise SystemExit("filesystem view configuration must be an object")
    command = sys.argv[3:]
    if not command:
        raise SystemExit("filesystem view requires a command")
    _install_view(config)
    cwd = _required_path(config, "cwd")
    os.chdir(cwd)
    os.execvpe(command[0], command, os.environ)


def _install_view(config: dict[str, Any]) -> None:
    workspace = _required_path(config, "workspace")
    agent_kind = str(config.get("agent_kind", "main"))
    allowed = _path_list(config, "allowed", workspace=workspace)
    denied = _path_list(config, "denied", workspace=workspace)
    runtime_allowed = _path_list(config, "runtime_allowed", workspace=workspace)

    _mount("--make-rprivate", "/")
    source = Path(f"/tmp/.jarvis-workspace-source-{os.getpid()}")
    source.mkdir(mode=0o700)
    try:
        _mount("--bind", str(workspace), str(source))
        if agent_kind == "subagent":
            _bind_remount(workspace, read_only=True)
            for path in allowed:
                _bind_from_source(
                    source=source,
                    workspace=workspace,
                    target=path,
                    read_only=False,
                    use_existing_ancestor=True,
                )
        for path in denied:
            _bind_from_source(
                source=source,
                workspace=workspace,
                target=path,
                read_only=True,
                use_existing_ancestor=True,
            )
        for path in runtime_allowed:
            _bind_from_source(
                source=source,
                workspace=workspace,
                target=path,
                read_only=False,
                use_existing_ancestor=False,
            )
    finally:
        subprocess.run(
            ["umount", "-l", str(source)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with contextlib.suppress(OSError):
            source.rmdir()


def _bind_from_source(
    *,
    source: Path,
    workspace: Path,
    target: Path,
    read_only: bool,
    use_existing_ancestor: bool,
) -> None:
    mount_target = _existing_target(
        target,
        workspace=workspace,
        use_existing_ancestor=use_existing_ancestor,
    )
    if mount_target is None:
        return
    if mount_target == workspace:
        _bind_remount(workspace, read_only=read_only)
        return
    relative = mount_target.relative_to(workspace)
    source_target = source / relative
    _mount("--bind", str(source_target), str(mount_target))
    _mount(
        "-o",
        f"remount,bind,{'ro' if read_only else 'rw'}",
        str(mount_target),
    )


def _existing_target(
    target: Path,
    *,
    workspace: Path,
    use_existing_ancestor: bool,
) -> Path | None:
    if target.exists() or target.is_symlink():
        return target
    if not use_existing_ancestor:
        return None
    candidate = target.parent
    while candidate != workspace and not candidate.exists():
        candidate = candidate.parent
    return candidate if candidate.exists() else None


def _bind_remount(path: Path, *, read_only: bool) -> None:
    _mount("--bind", str(path), str(path))
    _mount("-o", f"remount,bind,{'ro' if read_only else 'rw'}", str(path))


def _mount(*arguments: str) -> None:
    subprocess.run(
        ["mount", *arguments],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _required_path(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"filesystem view field {key!r} must be a path string")
    return Path(value).resolve(strict=False)


def _path_list(
    config: dict[str, Any],
    key: str,
    *,
    workspace: Path,
) -> tuple[Path, ...]:
    value = config.get(key, [])
    if not isinstance(value, list):
        raise SystemExit(f"filesystem view field {key!r} must be a list")
    paths: list[Path] = []
    for raw in value:
        if not isinstance(raw, str) or not raw:
            raise SystemExit(f"filesystem view field {key!r} contains an invalid path")
        path = Path(raw).resolve(strict=False)
        if path != workspace and not path.is_relative_to(workspace):
            raise SystemExit(f"filesystem capability path escapes workspace: {path}")
        paths.append(path)
    return tuple(sorted(set(paths)))


if __name__ == "__main__":
    main()
