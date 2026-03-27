"""User command parsing for the core agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CommandKind = Literal["message", "new", "compact"]


@dataclass(slots=True, frozen=True)
class ParsedCommand:
    kind: CommandKind
    body: str


def parse_user_command(raw_text: str) -> ParsedCommand:
    stripped = raw_text.strip()
    if not stripped.startswith("/"):
        return ParsedCommand(kind="message", body=raw_text)

    parts = stripped.split(maxsplit=1)
    keyword = parts[0]
    argument = parts[1].strip() if len(parts) > 1 else ""

    if keyword == "/new":
        return ParsedCommand(kind="new", body=argument)
    if keyword == "/compact":
        return ParsedCommand(kind="compact", body=argument)
    return ParsedCommand(kind="message", body=raw_text)
