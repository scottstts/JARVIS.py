"""Dynamic loading of runtime tool discoverable entries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .runtime_tool_manifest import (
    RuntimeToolManifest,
    RuntimeToolManifestError,
    load_runtime_tool_manifest_file,
    runtime_tools_dir,
)
from .types import DiscoverableTool


@dataclass(slots=True, frozen=True)
class RuntimeToolCatalogLoadResult:
    entries: tuple[DiscoverableTool, ...]
    manifests: tuple[RuntimeToolManifest, ...]
    errors: tuple[str, ...]


def load_runtime_tool_catalog(
    workspace_dir: Path,
    *,
    reserved_names: set[str] | None = None,
) -> RuntimeToolCatalogLoadResult:
    reserved = set(reserved_names or ())
    entries: list[DiscoverableTool] = []
    manifests: list[RuntimeToolManifest] = []
    errors: list[str] = []
    seen_names: set[str] = set()

    directory = runtime_tools_dir(workspace_dir)
    if not directory.exists():
        return RuntimeToolCatalogLoadResult(entries=(), manifests=(), errors=())

    for path in sorted(directory.glob("*.json")):
        try:
            manifest = load_runtime_tool_manifest_file(path)
        except RuntimeToolManifestError as exc:
            errors.append(f"{path.name}: {exc}")
            continue

        if manifest.name in reserved:
            errors.append(
                f"{path.name}: runtime tool name '{manifest.name}' conflicts with a built-in tool."
            )
            continue
        if manifest.name in seen_names:
            errors.append(
                f"{path.name}: duplicate runtime tool name '{manifest.name}'."
            )
            continue

        manifests.append(manifest)
        entries.append(manifest.to_discoverable(manifest_path=path))
        seen_names.add(manifest.name)

    return RuntimeToolCatalogLoadResult(
        entries=tuple(entries),
        manifests=tuple(manifests),
        errors=tuple(errors),
    )
