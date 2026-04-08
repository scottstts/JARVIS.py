"""Runtime tool manifest normalization and serialization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .types import DiscoverableTool

_ALLOWED_TOP_LEVEL_KEYS = {
    "name",
    "purpose",
    "aliases",
    "detailed_description",
    "usage",
    "notes",
    "operator",
    "invocation",
    "provisioning",
    "artifacts",
    "rebuild",
    "safety",
}
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class RuntimeToolManifestError(ValueError):
    """Raised when a runtime tool manifest is invalid."""


@dataclass(slots=True, frozen=True)
class RuntimeToolManifest:
    name: str
    purpose: str
    operator: str
    aliases: tuple[str, ...] = ()
    detailed_description: str | None = None
    usage: Any = None
    notes: Any = None
    invocation: Any = None
    provisioning: Any = None
    artifacts: Any = None
    rebuild: Any = None
    safety: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "purpose": self.purpose,
            "operator": self.operator,
        }
        if self.aliases:
            payload["aliases"] = list(self.aliases)
        if self.detailed_description is not None:
            payload["detailed_description"] = self.detailed_description
        if self.usage is not None:
            payload["usage"] = self.usage
        if self.notes is not None:
            payload["notes"] = self.notes
        if self.invocation is not None:
            payload["invocation"] = self.invocation
        if self.provisioning is not None:
            payload["provisioning"] = self.provisioning
        if self.artifacts is not None:
            payload["artifacts"] = self.artifacts
        if self.rebuild is not None:
            payload["rebuild"] = self.rebuild
        if self.safety is not None:
            payload["safety"] = self.safety
        return payload

    def manifest_hash(self) -> str:
        normalized = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def to_discoverable(self, *, manifest_path: Path) -> DiscoverableTool:
        usage = self.usage
        if usage is None:
            usage = {"operator": self.operator}
            if self.invocation is not None:
                usage["invocation"] = self.invocation
        elif isinstance(usage, dict):
            usage = dict(usage)
            usage.setdefault("operator", self.operator)
            if self.invocation is not None:
                usage.setdefault("invocation", self.invocation)
        else:
            usage = {
                "operator": self.operator,
                "details": usage,
            }
            if self.invocation is not None:
                usage["invocation"] = self.invocation

        _ = manifest_path
        metadata: dict[str, Any] = {"source": "runtime_tools"}

        return DiscoverableTool(
            name=self.name,
            aliases=self.aliases,
            purpose=self.purpose,
            detailed_description=self.detailed_description,
            usage=usage,
            metadata=metadata,
        )


def validate_runtime_tool_manifest_payload(raw_payload: object) -> RuntimeToolManifest:
    if not isinstance(raw_payload, dict):
        raise RuntimeToolManifestError("runtime tool manifest must be an object.")

    unknown_keys = sorted(set(raw_payload) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown_keys:
        raise RuntimeToolManifestError(
            f"runtime tool manifest contains unknown keys: {', '.join(unknown_keys)}."
        )

    name = _require_tool_name(raw_payload.get("name"))
    purpose = _require_non_empty_string(raw_payload.get("purpose"), field_name="purpose")
    operator = _require_non_empty_string(raw_payload.get("operator"), field_name="operator")
    aliases = _normalize_aliases(raw_payload.get("aliases"))
    detailed_description = _normalize_optional_string(raw_payload.get("detailed_description"))
    usage = _normalize_json_value(raw_payload.get("usage"), field_name="usage")
    notes = _normalize_json_value(raw_payload.get("notes"), field_name="notes")
    invocation = _normalize_json_value(raw_payload.get("invocation"), field_name="invocation")
    provisioning = _normalize_json_value(
        raw_payload.get("provisioning"),
        field_name="provisioning",
    )
    artifacts = _normalize_json_value(raw_payload.get("artifacts"), field_name="artifacts")
    rebuild = _normalize_json_value(raw_payload.get("rebuild"), field_name="rebuild")
    safety = _normalize_json_value(raw_payload.get("safety"), field_name="safety")

    return RuntimeToolManifest(
        name=name,
        purpose=purpose,
        operator=operator,
        aliases=aliases,
        detailed_description=detailed_description,
        usage=usage,
        notes=notes,
        invocation=invocation,
        provisioning=provisioning,
        artifacts=artifacts,
        rebuild=rebuild,
        safety=safety,
    )


def load_runtime_tool_manifest_file(path: Path) -> RuntimeToolManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeToolManifestError(f"could not read manifest file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeToolManifestError(f"invalid JSON: {exc.msg}") from exc

    return validate_runtime_tool_manifest_payload(payload)


def dump_runtime_tool_manifest(manifest: RuntimeToolManifest) -> str:
    return json.dumps(
        manifest.to_dict(),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def runtime_tools_dir(workspace_dir: Path) -> Path:
    return workspace_dir / "runtime_tools"


def runtime_tool_manifest_path(workspace_dir: Path, name: str) -> Path:
    validated_name = _require_tool_name(name)
    return runtime_tools_dir(workspace_dir) / f"{validated_name}.json"


def _require_tool_name(value: object) -> str:
    name = _require_non_empty_string(value, field_name="name")
    if not _TOOL_NAME_PATTERN.fullmatch(name):
        raise RuntimeToolManifestError(
            "name must match ^[a-z][a-z0-9_]{0,63}$."
        )
    return name


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise RuntimeToolManifestError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise RuntimeToolManifestError(f"{field_name} cannot be empty.")
    return normalized


def _normalize_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeToolManifestError("optional string fields must be strings when set.")
    normalized = value.strip()
    return normalized or None


def _normalize_aliases(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RuntimeToolManifestError("aliases must be an array of strings.")
    aliases: list[str] = []
    seen: set[str] = set()
    for index, raw_alias in enumerate(value, start=1):
        if not isinstance(raw_alias, str):
            raise RuntimeToolManifestError(f"aliases[{index}] must be a string.")
        alias = raw_alias.strip()
        if not alias or alias in seen:
            continue
        aliases.append(alias)
        seen.add(alias)
    return tuple(aliases)


def _normalize_json_value(value: object, *, field_name: str) -> Any:
    if value is None:
        return None
    try:
        normalized = json.loads(json.dumps(value, ensure_ascii=True))
    except (TypeError, ValueError) as exc:
        raise RuntimeToolManifestError(
            f"{field_name} must be JSON-serializable."
        ) from exc
    return normalized
