"""Runtime helpers for loading Docker secrets into process environment."""

from __future__ import annotations

import os
from pathlib import Path
import re

_DEFAULT_DOCKER_SECRETS_DIR = Path("/run/secrets")
_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def load_docker_secrets_if_present(
    secrets_dir: Path | None = None,
    *,
    override: bool = False,
) -> tuple[Path, ...]:
    resolved_dir = secrets_dir or default_docker_secrets_dir()
    if not resolved_dir.exists() or not resolved_dir.is_dir():
        return ()

    loaded_paths: list[Path] = []
    for candidate in sorted(resolved_dir.iterdir()):
        if not candidate.is_file():
            continue
        key = candidate.name.strip()
        if not _ENV_VAR_NAME_PATTERN.fullmatch(key):
            continue
        if not override and key in os.environ:
            continue
        value = candidate.read_text(encoding="utf-8").rstrip("\r\n")
        if not value.strip():
            continue

        os.environ[key] = value
        loaded_paths.append(candidate)

    return tuple(loaded_paths)


def default_docker_secrets_dir() -> Path:
    return _DEFAULT_DOCKER_SECRETS_DIR
