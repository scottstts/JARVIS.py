"""Small helpers for deciding whether a user message continues the active tool task."""

from __future__ import annotations

import re


_RESUME_PATTERN = re.compile(
    r"^\s*(?:please\s+)?(?:continue|resume|keep\s+going|carry\s+on|proceed|retry|try\s+again)\b",
    re.IGNORECASE,
)
_TASK_REPLACEMENT_PATTERN = re.compile(
    r"\b(?:forget|cancel|stop|discard)\b.*\b(?:previous|prior|current|old)\b.*\btask\b|"
    r"^\s*(?:new task|instead[, :]|start over\b)",
    re.IGNORECASE,
)
_SIDE_QUERY_PATTERN = re.compile(
    r"^\s*(?:please\s+)?(?:give\s+me\s+)?(?:a\s+)?(?:quick\s+)?(?:status|progress)"
    r"(?:\s+update|\s+report)?\b|"
    r"^\s*(?:what\s+time\s+is\s+it\??\s*$|"
    r"what(?:'s|\s+is)\s+the\s+(?:current\s+)?(?:status|progress|time)|"
    r"how\s+(?:is|are)\b.*\bgoing\??\s*$)",
    re.IGNORECASE,
)


def user_message_explicitly_resumes_task(user_text: str) -> bool:
    return bool(_RESUME_PATTERN.search(user_text))


def user_message_explicitly_replaces_task(user_text: str) -> bool:
    return bool(_TASK_REPLACEMENT_PATTERN.search(user_text))


def user_message_is_side_query(user_text: str) -> bool:
    """Return whether a user message asks about, rather than changes, active work."""

    if not _SIDE_QUERY_PATTERN.search(user_text):
        return False
    return not bool(
        re.search(
            r"\b(?:then|and)\s+(?:continue|resume|keep|fix|implement|build|change|edit|run)\b",
            user_text,
            re.IGNORECASE,
        )
    )
