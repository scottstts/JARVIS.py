"""Small runtime helpers for loading local .env files."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_if_present(
    env_path: Path | None = None,
    *,
    override: bool = False,
) -> Path | None:
    resolved_path = env_path or default_dotenv_path()
    if not resolved_path.exists():
        return None

    for line in resolved_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].strip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            continue
        if not override and key in os.environ:
            continue
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ[key] = value

    return resolved_path


def default_dotenv_path() -> Path:
    return Path(__file__).resolve().parents[1] / ".env"
