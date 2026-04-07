"""Runtime configuration for the Codex app-server backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from jarvis import settings as app_settings

from .types import CodexConfigurationError

_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_REASONING_SUMMARIES = {"none", "auto", "concise", "detailed"}
_PERSONALITIES = {"none", "friendly", "pragmatic"}
_APPROVAL_POLICIES = {"untrusted", "on-failure", "on-request", "never"}


def _optional_env(name: str, default: object) -> str | None:
    raw = os.getenv(name)
    value = raw if raw is not None else default
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise CodexConfigurationError(f"{name} must be a boolean-like value.")


def _optional_choice(
    *,
    name: str,
    default: object,
    allowed: set[str],
) -> str | None:
    value = _optional_env(name, default)
    if value is None:
        return None
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise CodexConfigurationError(f"{name} must be one of: {allowed_values}. Got: {value}")
    return value


def _optional_path(name: str, default: object) -> Path | None:
    value = _optional_env(name, default)
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise CodexConfigurationError(f"{name} must be an absolute path. Got: {value}")
    return path


@dataclass(slots=True, frozen=True)
class CodexBackendSettings:
    ws_url: str
    model: str | None
    reasoning_effort: str | None
    reasoning_summary: str | None
    personality: str | None
    service_name: str
    host_repo_root: Path | None
    host_workspace_root: Path | None
    approval_policy: str
    sandbox_network_access: bool
    ws_bearer_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.ws_url.strip():
            raise CodexConfigurationError("JARVIS_CODEX_WS_URL cannot be blank.")
        if not self.service_name.strip():
            raise CodexConfigurationError("JARVIS_CODEX_SERVICE_NAME cannot be blank.")

    @classmethod
    def from_env(cls) -> "CodexBackendSettings":
        return cls(
            ws_url=_optional_env("JARVIS_CODEX_WS_URL", app_settings.JARVIS_CODEX_WS_URL)
            or "ws://host.docker.internal:4500",
            model=_optional_env("JARVIS_CODEX_MODEL", app_settings.JARVIS_CODEX_MODEL),
            reasoning_effort=_optional_choice(
                name="JARVIS_CODEX_REASONING_EFFORT",
                default=app_settings.JARVIS_CODEX_REASONING_EFFORT,
                allowed=_REASONING_EFFORTS,
            ),
            reasoning_summary=_optional_choice(
                name="JARVIS_CODEX_REASONING_SUMMARY",
                default=app_settings.JARVIS_CODEX_REASONING_SUMMARY,
                allowed=_REASONING_SUMMARIES,
            ),
            personality=_optional_choice(
                name="JARVIS_CODEX_PERSONALITY",
                default=app_settings.JARVIS_CODEX_PERSONALITY,
                allowed=_PERSONALITIES,
            ),
            service_name=_optional_env(
                "JARVIS_CODEX_SERVICE_NAME",
                app_settings.JARVIS_CODEX_SERVICE_NAME,
            )
            or "Jarvis",
            host_repo_root=_optional_path(
                "JARVIS_CODEX_HOST_REPO_ROOT",
                app_settings.JARVIS_CODEX_HOST_REPO_ROOT,
            ),
            host_workspace_root=_optional_path(
                "JARVIS_CODEX_HOST_WORKSPACE_ROOT",
                app_settings.JARVIS_CODEX_HOST_WORKSPACE_ROOT,
            ),
            approval_policy=_optional_choice(
                name="JARVIS_CODEX_APPROVAL_POLICY",
                default=app_settings.JARVIS_CODEX_APPROVAL_POLICY,
                allowed=_APPROVAL_POLICIES,
            )
            or "untrusted",
            sandbox_network_access=_bool_env(
                "JARVIS_CODEX_SANDBOX_NETWORK_ACCESS",
                app_settings.JARVIS_CODEX_SANDBOX_NETWORK_ACCESS,
            ),
            ws_bearer_token=_optional_env("JARVIS_CODEX_WS_BEARER_TOKEN", None),
        )

    def require_host_paths(self) -> tuple[Path, Path]:
        if self.host_repo_root is None:
            raise CodexConfigurationError(
                "Codex backend requires JARVIS_CODEX_HOST_REPO_ROOT when provider 'codex' is selected."
            )
        if self.host_workspace_root is None:
            raise CodexConfigurationError(
                "Codex backend requires JARVIS_CODEX_HOST_WORKSPACE_ROOT when provider 'codex' is selected."
            )
        return self.host_repo_root, self.host_workspace_root

    def sandbox_policy(self) -> dict[str, object]:
        host_repo_root, host_workspace_root = self.require_host_paths()
        return {
            "type": "workspaceWrite",
            "writableRoots": [str(host_repo_root), str(host_workspace_root)],
            "readOnlyAccess": {
                "type": "restricted",
                "includePlatformDefaults": True,
                "readableRoots": [str(host_repo_root), str(host_workspace_root)],
            },
            "networkAccess": self.sandbox_network_access,
            "excludeTmpdirEnvVar": False,
            "excludeSlashTmp": False,
        }
