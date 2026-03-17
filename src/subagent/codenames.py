"""Codename allocation helpers for active subagents."""

from __future__ import annotations

import secrets


def allocate_codename(
    *,
    pool: tuple[str, ...],
    active_codenames: set[str],
) -> str:
    available = [name for name in pool if name not in active_codenames]
    if not available:
        raise ValueError("No subagent codenames are currently available.")
    return secrets.choice(available)
