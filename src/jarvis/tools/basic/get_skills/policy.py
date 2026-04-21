"""Policy checks for the get_skills tool."""

from __future__ import annotations

from jarvis.skills import SkillsSettings
from jarvis.skills.catalog import is_valid_skill_id

from ...types import ToolExecutionContext, ToolPolicyDecision


class GetSkillsPolicy:
    """Restricts get_skills to configured modes and canonical skill ids."""

    def __init__(self, settings: SkillsSettings) -> None:
        self._settings = settings

    def authorize(
        self,
        *,
        mode: str,
        query: str | None,
        skill_id: str | None,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context

        normalized_mode = mode.strip().lower()
        allowed_modes = {"get"}
        if not self._settings.bootstrap_headers:
            allowed_modes.add("search")
        if normalized_mode not in allowed_modes:
            return ToolPolicyDecision(
                allowed=False,
                reason="get_skills mode is not available with the current skills settings.",
            )

        if normalized_mode == "search":
            if query is not None:
                normalized_query = query.strip()
                if len(normalized_query) > self._settings.max_search_query_chars:
                    return ToolPolicyDecision(
                        allowed=False,
                        reason=(
                            "get_skills query length must be <= "
                            f"{self._settings.max_search_query_chars} characters."
                        ),
                    )
                if len(normalized_query.split()) > self._settings.max_search_query_words:
                    return ToolPolicyDecision(
                        allowed=False,
                        reason=(
                            "get_skills query length must be <= "
                            f"{self._settings.max_search_query_words} words."
                        ),
                    )
            return ToolPolicyDecision(allowed=True)

        if skill_id is None or not skill_id.strip():
            return ToolPolicyDecision(
                allowed=False,
                reason="get_skills mode=get requires skill_id.",
            )
        if not is_valid_skill_id(
            skill_id.strip(),
            max_chars=self._settings.max_skill_id_chars,
        ):
            return ToolPolicyDecision(
                allowed=False,
                reason="get_skills skill_id must be a canonical skill directory name.",
            )
        return ToolPolicyDecision(allowed=True)
