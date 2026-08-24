"""Route-local workspace leases and concurrent tool access coordination."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import AsyncIterator

from jarvis.llm import ToolCall

from .types import ToolExecutionContext, ToolExecutionResult
from .workspace_revision import (
    workspace_paths_revision,
    workspace_revision,
    workspace_revision_excluding,
)


class WorkspaceLeaseError(RuntimeError):
    """Raised when an actor attempts to write a path leased by another actor."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.conflict_class = _workspace_lease_conflict_class(message)
        self.conflict_key = _workspace_lease_conflict_key(
            conflict_class=self.conflict_class,
            message=message,
        )
        self.remediation = _workspace_lease_remediation(self.conflict_class)


@dataclass(slots=True)
class WorkspaceAccessObservation:
    """Runtime evidence about workspace state surrounding one tool execution."""

    mode: str
    scope: str | None = None
    revision_before: str | None = None
    revision_after: str | None = None
    write_allowed_paths: tuple[Path, ...] = ()
    write_denied_paths: tuple[Path, ...] = ()
    lease_generation: int | None = None
    revision_paths: tuple[Path, ...] = ()

    @property
    def may_mutate(self) -> bool:
        return self.mode in {"path_write", "global_write", "actor_workspace"}

    @property
    def changed(self) -> bool:
        return (
            self.revision_before is not None
            and self.revision_after is not None
            and self.revision_before != self.revision_after
        )


def with_workspace_observation(
    result: ToolExecutionResult,
    observation: WorkspaceAccessObservation,
) -> ToolExecutionResult:
    """Attach runtime-owned access and mutation evidence to a tool result."""

    metadata = dict(result.metadata)
    metadata["workspace_access_mode"] = observation.mode
    if observation.scope is not None:
        metadata["workspace_revision_scope"] = observation.scope
    if observation.revision_before is not None:
        metadata["workspace_revision_before"] = observation.revision_before
    if observation.revision_after is not None:
        metadata["workspace_revision_after"] = observation.revision_after
    if observation.may_mutate:
        metadata["workspace_changed"] = observation.changed
    return replace(result, metadata=metadata)


def with_workspace_capabilities(
    context: ToolExecutionContext,
    observation: WorkspaceAccessObservation,
) -> ToolExecutionContext:
    """Attach the coordinator-owned filesystem view to one executor context."""

    return replace(
        context,
        workspace_write_allowed_paths=observation.write_allowed_paths,
        workspace_write_denied_paths=observation.write_denied_paths,
        workspace_lease_generation=observation.lease_generation,
    )


@dataclass(slots=True, frozen=True)
class _AccessRequest:
    mode: str
    read_paths: tuple[Path, ...] = ()
    write_paths: tuple[Path, ...] = ()


class _AsyncReadWriteLock:
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        async with self._condition:
            while self._writer or self._writers_waiting:
                await self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            async with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        async with self._condition:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers:
                    await self._condition.wait()
                self._writer = True
            finally:
                self._writers_waiting -= 1
        try:
            yield
        finally:
            async with self._condition:
                self._writer = False
                self._condition.notify_all()


class WorkspaceAccessCoordinator:
    """Allows concurrent reads/disjoint writes while protecting assigned paths.

    The coordinator is route-local because all actors in a route share one workspace.
    Child `owned_paths` are durable write leases until explicit disposal. Main-agent
    writes require an explicit lease release through the existing child control flow.
    """

    def __init__(self, *, workspace_dir: Path) -> None:
        self._workspace_dir = workspace_dir.resolve(strict=False)
        for relative in ("archive", ".jarvis_internal"):
            (self._workspace_dir / relative).mkdir(parents=True, exist_ok=True)
        self._global_access = _AsyncReadWriteLock()
        self._lease_guard = asyncio.Lock()
        self._leases: dict[str, tuple[Path, ...]] = {}
        self._lease_generation = 0
        self._path_condition = asyncio.Condition()
        self._active_path_reads: dict[int, tuple[Path, ...]] = {}
        self._active_path_writes: dict[int, tuple[Path, ...]] = {}
        self._next_path_write_id = 0

    async def claim_paths(self, *, owner: str, paths: tuple[str, ...]) -> None:
        normalized_owner = _require_owner(owner)
        normalized_paths = tuple(
            sorted({_resolve_workspace_path(self._workspace_dir, raw) for raw in paths})
        )
        if not normalized_paths:
            return
        for path in normalized_paths:
            if not path.exists():
                raise WorkspaceLeaseError(
                    f"{_display_path(path, self._workspace_dir)} does not exist. Create an "
                    "owned directory or file before assigning it to a subagent."
                )
            if any(
                _paths_overlap(path, self._workspace_dir / relative)
                for relative in ("archive", ".jarvis_internal")
            ):
                raise WorkspaceLeaseError(
                    f"{_display_path(path, self._workspace_dir)} overlaps runtime-managed "
                    "workspace state and cannot be assigned to a subagent."
                )
        async with self._global_access.write():
            async with self._lease_guard:
                for other_owner, other_paths in self._leases.items():
                    if other_owner == normalized_owner:
                        continue
                    for candidate in normalized_paths:
                        if any(_paths_overlap(candidate, existing) for existing in other_paths):
                            raise WorkspaceLeaseError(
                                f"{_display_path(candidate, self._workspace_dir)} is already leased "
                                f"by {other_owner}."
                            )
                if self._leases.get(normalized_owner) != normalized_paths:
                    self._leases[normalized_owner] = normalized_paths
                    self._lease_generation += 1

    async def release_owner(self, *, owner: str) -> None:
        normalized_owner = _require_owner(owner)
        async with self._global_access.write():
            async with self._lease_guard:
                if self._leases.pop(normalized_owner, None) is not None:
                    self._lease_generation += 1

    async def lease_generation(self) -> int:
        """Return the current route-local lease generation."""

        async with self._lease_guard:
            return self._lease_generation

    @asynccontextmanager
    async def execute(
        self,
        *,
        tool_call: ToolCall,
        context: ToolExecutionContext,
    ) -> AsyncIterator[WorkspaceAccessObservation]:
        request = _access_request(
            tool_call=tool_call,
            context=context,
            workspace_dir=self._workspace_dir,
        )
        if request.mode == "read":
            async with self._global_access.read():
                yield WorkspaceAccessObservation(mode=request.mode)
            return
        if request.mode == "exclusive_read":
            async with self._global_access.write():
                yield WorkspaceAccessObservation(mode=request.mode)
            return
        if request.mode == "path_read":
            async with self._global_access.read():
                read_id = await self._acquire_path_read(request.read_paths)
                try:
                    yield WorkspaceAccessObservation(mode=request.mode)
                finally:
                    await self._release_path_read(read_id)
            return
        if request.mode == "global_write":
            async with self._global_access.write():
                await self._assert_global_write_allowed(owner=_context_owner(context))
                observation = self._begin_observation(request)
                try:
                    yield observation
                finally:
                    self._finish_observation(observation, request)
            return
        if request.mode == "actor_workspace":
            async with self._global_access.read():
                observation = await self._actor_workspace_observation(context=context)
                try:
                    yield observation
                finally:
                    self._finish_actor_workspace_observation(observation)
            return

        async with self._global_access.read():
            owner = _context_owner(context)
            await self._assert_write_leases(
                owner=owner,
                paths=request.write_paths,
            )
            write_id = await self._acquire_path_write(request.write_paths)
            observation = self._begin_observation(request)
            try:
                yield observation
            finally:
                self._finish_observation(observation, request)
                await self._release_path_write(write_id)

    def _begin_observation(self, request: _AccessRequest) -> WorkspaceAccessObservation:
        if request.mode == "global_write":
            return WorkspaceAccessObservation(
                mode=request.mode,
                scope="workspace",
                revision_before=workspace_revision(self._workspace_dir),
            )
        if request.mode == "path_write":
            return WorkspaceAccessObservation(
                mode=request.mode,
                scope="declared_paths",
                revision_before=workspace_paths_revision(
                    self._workspace_dir,
                    request.write_paths,
                ),
            )
        return WorkspaceAccessObservation(mode=request.mode)

    def _finish_observation(
        self,
        observation: WorkspaceAccessObservation,
        request: _AccessRequest,
    ) -> None:
        if request.mode == "global_write":
            observation.revision_after = workspace_revision(self._workspace_dir)
        elif request.mode == "path_write":
            observation.revision_after = workspace_paths_revision(
                self._workspace_dir,
                request.write_paths,
            )

    async def _assert_write_leases(
        self,
        *,
        owner: str,
        paths: tuple[Path, ...],
    ) -> None:
        async with self._lease_guard:
            if owner.startswith("subagent:"):
                owned_paths = self._leases.get(owner, ())
                for path in paths:
                    if not any(
                        path == owned_path or path.is_relative_to(owned_path)
                        for owned_path in owned_paths
                    ):
                        raise WorkspaceLeaseError(
                            f"{_display_path(path, self._workspace_dir)} is outside the "
                            "subagent's owned workspace paths. Ask Jarvis to assign the path "
                            "before writing it."
                        )
            for lease_owner, lease_paths in self._leases.items():
                if lease_owner == owner:
                    continue
                for path in paths:
                    if any(_paths_overlap(path, leased_path) for leased_path in lease_paths):
                        raise WorkspaceLeaseError(
                            f"Write denied for {_display_path(path, self._workspace_dir)}: "
                            f"it is leased by {lease_owner}. Stop and dispose that subagent "
                            "to revoke its lease before editing this path."
                        )

    async def _actor_workspace_observation(
        self,
        *,
        context: ToolExecutionContext,
    ) -> WorkspaceAccessObservation:
        owner = _context_owner(context)
        async with self._lease_guard:
            own_paths = self._leases.get(owner, ())
            other_paths = tuple(
                path
                for lease_owner, paths in self._leases.items()
                if lease_owner != owner
                for path in paths
            )
            generation = self._lease_generation
        runtime_paths = tuple(
            path
            for relative in ("archive", ".jarvis_internal")
            if (path := self._workspace_dir / relative).exists()
        )
        denied_paths = tuple(sorted({*other_paths, *runtime_paths}))
        if context.agent_kind == "subagent":
            revision_paths = own_paths
            revision_before = (
                workspace_paths_revision(self._workspace_dir, revision_paths)
                if revision_paths
                else None
            )
            scope = "actor_owned_paths"
        else:
            revision_paths = denied_paths
            revision_before = workspace_revision_excluding(
                self._workspace_dir,
                revision_paths,
            )
            scope = "workspace"
        observation = WorkspaceAccessObservation(
            mode="actor_workspace",
            scope=scope,
            revision_before=revision_before,
            write_allowed_paths=own_paths,
            write_denied_paths=denied_paths,
            lease_generation=generation,
            revision_paths=revision_paths,
        )
        return observation

    def _finish_actor_workspace_observation(
        self,
        observation: WorkspaceAccessObservation,
    ) -> None:
        revision_paths = observation.revision_paths
        if observation.scope == "actor_owned_paths":
            if revision_paths:
                observation.revision_after = workspace_paths_revision(
                    self._workspace_dir,
                    revision_paths,
                )
            return
        observation.revision_after = workspace_revision_excluding(
            self._workspace_dir,
            revision_paths,
        )

    async def _acquire_path_write(self, paths: tuple[Path, ...]) -> int:
        async with self._path_condition:
            while any(
                _path_sets_overlap(paths, active_paths)
                for active_paths in self._active_path_writes.values()
            ) or any(
                _path_sets_overlap(paths, active_paths)
                for active_paths in self._active_path_reads.values()
            ):
                await self._path_condition.wait()
            self._next_path_write_id += 1
            write_id = self._next_path_write_id
            self._active_path_writes[write_id] = paths
            return write_id

    async def _acquire_path_read(self, paths: tuple[Path, ...]) -> int:
        async with self._path_condition:
            while any(
                _path_sets_overlap(paths, active_paths)
                for active_paths in self._active_path_writes.values()
            ):
                await self._path_condition.wait()
            self._next_path_write_id += 1
            read_id = self._next_path_write_id
            self._active_path_reads[read_id] = paths
            return read_id

    async def _release_path_read(self, read_id: int) -> None:
        async with self._path_condition:
            self._active_path_reads.pop(read_id, None)
            self._path_condition.notify_all()

    async def _release_path_write(self, write_id: int) -> None:
        async with self._path_condition:
            self._active_path_writes.pop(write_id, None)
            self._path_condition.notify_all()

    async def _assert_global_write_allowed(self, *, owner: str) -> None:
        """Reject unknown global writers while another actor owns workspace paths."""

        async with self._lease_guard:
            if owner.startswith("subagent:"):
                raise WorkspaceLeaseError(
                    "A subagent cannot use an unscoped workspace writer. Use a tool with an "
                    "explicit path inside its owned workspace paths."
                )
            other_owners = sorted(
                lease_owner
                for lease_owner, paths in self._leases.items()
                if lease_owner != owner and paths
            )
        if other_owners:
            raise WorkspaceLeaseError(
                "An unknown global-write tool cannot run while workspace paths are owned by "
                + ", ".join(other_owners)
                + ". Wait for or dispose the owners before retrying."
            )


def _access_request(
    *,
    tool_call: ToolCall,
    context: ToolExecutionContext,
    workspace_dir: Path,
) -> _AccessRequest:
    if tool_call.name in {"file_patch", "file_write", "file_replace"}:
        raw_path = str(tool_call.arguments.get("path", "")).strip()
        if raw_path:
            return _AccessRequest(
                mode="path_write",
                write_paths=(_resolve_workspace_path(workspace_dir, raw_path),),
            )
        return _AccessRequest(mode="global_write")
    if tool_call.name == "bash":
        mode = str(tool_call.arguments.get("mode", "foreground")).strip().lower()
        if mode in {"status", "tail"}:
            return _AccessRequest(mode="read")
        if mode == "cancel":
            return _AccessRequest(mode="read")
        return _AccessRequest(mode="actor_workspace")
    if tool_call.name == "acceptance_run":
        return _AccessRequest(mode="actor_workspace")
    if tool_call.name == "tool_register":
        manifest = tool_call.arguments.get("manifest")
        name = str(manifest.get("name", "")).strip() if isinstance(manifest, dict) else ""
        if name:
            return _AccessRequest(
                mode="path_write",
                write_paths=(workspace_dir / "runtime_tools" / f"{name}.json",),
            )
        return _AccessRequest(mode="global_write")
    if tool_call.name == "generate_edit_image":
        output_path = str(tool_call.arguments.get("output_path", "")).strip()
        if output_path:
            return _AccessRequest(
                mode="path_write",
                write_paths=(_resolve_workspace_path(workspace_dir, output_path),),
            )
        return _AccessRequest(mode="global_write")
    if tool_call.name == "memory_write":
        return _AccessRequest(mode="global_write")
    if tool_call.name == "email":
        return _AccessRequest(mode="exclusive_read")
    if tool_call.name == "acceptance_record":
        return _AccessRequest(mode="exclusive_read")
    if tool_call.name in {"send_file", "view_image"}:
        raw_path = str(tool_call.arguments.get("path", "")).strip()
        if raw_path:
            return _AccessRequest(
                mode="path_read",
                read_paths=(_resolve_workspace_path(workspace_dir, raw_path),),
            )
        return _AccessRequest(mode="exclusive_read")
    if tool_call.name in {
        "get_skills",
        "memory_get",
        "memory_search",
        "tool_search",
        "transcribe",
        "web_fetch",
        "web_search",
    }:
        return _AccessRequest(mode="read")
    # Runtime-manifest and future tools are mutation-unknown. Keep them behind the
    # global write barrier until they declare a narrower access contract.
    return _AccessRequest(mode="global_write")


def _workspace_lease_conflict_class(message: str) -> str:
    normalized = message.lower()
    if "leased by" in normalized or "already leased" in normalized:
        return "path_owned_by_other_actor"
    if "stay inside the workspace" in normalized:
        return "invalid_workspace_path"
    if "lease owner" in normalized:
        return "invalid_lease_owner"
    return "workspace_access_conflict"


def _workspace_lease_remediation(conflict_class: str) -> str:
    if conflict_class == "path_owned_by_other_actor":
        return (
            "Wait for the owning actor, or explicitly stop and dispose it before editing its "
            "leased path."
        )
    if conflict_class == "invalid_workspace_path":
        return "Use only normalized paths inside the shared workspace."
    if conflict_class == "invalid_lease_owner":
        return "Provide a non-empty stable actor owner id."
    return "Inspect current workspace ownership and replan before retrying."


def _workspace_lease_conflict_key(*, conflict_class: str, message: str) -> str:
    if conflict_class == "path_owned_by_other_actor":
        return f"{conflict_class}:{' '.join(message.lower().split())}"
    return conflict_class


def _resolve_workspace_path(workspace_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        if candidate == Path("/workspace") or candidate.is_relative_to(Path("/workspace")):
            candidate = workspace_dir / candidate.relative_to("/workspace")
    else:
        candidate = workspace_dir / candidate
    resolved = candidate.resolve(strict=False)
    if resolved != workspace_dir and not resolved.is_relative_to(workspace_dir):
        raise WorkspaceLeaseError("Owned paths must stay inside the workspace.")
    return resolved


def _context_owner(context: ToolExecutionContext) -> str:
    if context.agent_kind == "subagent" and context.subagent_id:
        return f"subagent:{context.subagent_id}"
    return "main"


def _require_owner(owner: str) -> str:
    normalized = owner.strip()
    if not normalized:
        raise WorkspaceLeaseError("Lease owner cannot be empty.")
    return normalized


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _path_sets_overlap(left: tuple[Path, ...], right: tuple[Path, ...]) -> bool:
    return any(_paths_overlap(left_path, right_path) for left_path in left for right_path in right)


def _display_path(path: Path, workspace_dir: Path) -> str:
    try:
        return str(Path("/workspace") / path.relative_to(workspace_dir))
    except ValueError:
        return str(path)
