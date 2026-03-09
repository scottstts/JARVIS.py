"""Identity bootstrap loader for session initialization."""

from __future__ import annotations

from pathlib import Path

from llm.types import LLMMessage

from .config import CoreSettings
from .errors import CoreConfigurationError


class IdentityBootstrapLoader:
    """Loads startup identity files injected at the start of each session."""

    def __init__(self, settings: CoreSettings) -> None:
        self._settings = settings

    def load_bootstrap_messages(self, *, summary_text: str | None = None) -> list[LLMMessage]:
        identities_dir = self._settings.identities_dir
        program = self._read_identity_file(identities_dir / self._settings.program_file_name)
        reactor = self._read_identity_file(identities_dir / self._settings.reactor_file_name)
        user = self._read_identity_file(identities_dir / self._settings.user_file_name)
        armor = self._read_identity_file(identities_dir / self._settings.armor_file_name)

        messages = [
            LLMMessage.text("system", program),
            LLMMessage.text("system", reactor),
            LLMMessage.text("system", user),
            LLMMessage.text("system", armor),
        ]
        if summary_text:
            messages.append(
                LLMMessage.text(
                    "developer",
                    (
                        "Summarized context from previous session compaction.\n"
                        "Use this as prior conversational state:\n\n"
                        f"{summary_text.strip()}"
                    ),
                )
            )
        return messages

    def _read_identity_file(self, path: Path) -> str:
        if not path.exists():
            raise CoreConfigurationError(f"Missing identity file: {path}")
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise CoreConfigurationError(f"Identity file is empty: {path}")
        return content
