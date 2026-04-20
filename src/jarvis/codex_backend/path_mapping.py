"""Host/container path translation for the Codex backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import CodexBackendSettings
from .types import CodexConfigurationError

_CONTAINER_REPO_ROOT = Path("/repo")
_CONTAINER_WORKSPACE_ROOT = Path("/workspace")


@dataclass(slots=True, frozen=True)
class CodexPathMapper:
    """Maps Jarvis container-visible paths to host-visible Codex paths."""

    host_repo_root: Path
    host_workspace_root: Path
    container_repo_root: Path = _CONTAINER_REPO_ROOT
    container_workspace_root: Path = _CONTAINER_WORKSPACE_ROOT

    @classmethod
    def from_settings(cls, settings: CodexBackendSettings) -> "CodexPathMapper":
        host_repo_root, host_workspace_root = settings.require_host_paths()
        return cls(
            host_repo_root=host_repo_root,
            host_workspace_root=host_workspace_root,
        )

    def container_to_host(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise CodexConfigurationError(
                f"Codex backend requires an absolute container path. Got: {candidate}"
            )
        if candidate == self.container_repo_root or self._is_relative_to(
            candidate, self.container_repo_root
        ):
            relative = candidate.relative_to(self.container_repo_root)
            return (self.host_repo_root / relative).resolve(strict=False)
        if candidate == self.container_workspace_root or self._is_relative_to(
            candidate, self.container_workspace_root
        ):
            relative = candidate.relative_to(self.container_workspace_root)
            return (self.host_workspace_root / relative).resolve(strict=False)
        raise CodexConfigurationError(
            "Codex backend can only translate paths inside /repo or the configured workspace root. "
            f"Got: {candidate}"
        )

    def host_to_container(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise CodexConfigurationError(
                f"Codex backend requires an absolute host path. Got: {candidate}"
            )
        if candidate == self.host_repo_root or self._is_relative_to(candidate, self.host_repo_root):
            relative = candidate.relative_to(self.host_repo_root)
            return (self.container_repo_root / relative).resolve(strict=False)
        if candidate == self.host_workspace_root or self._is_relative_to(
            candidate, self.host_workspace_root
        ):
            relative = candidate.relative_to(self.host_workspace_root)
            return (self.container_workspace_root / relative).resolve(strict=False)
        raise CodexConfigurationError(
            "Codex backend can only reverse-map paths inside the configured host repo/workspace roots. "
            f"Got: {candidate}"
        )

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True
