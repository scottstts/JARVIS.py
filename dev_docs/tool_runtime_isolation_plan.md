# Isolated Tool Runtime Plan

## Status

This plan targets the real problem:

- give the agent much broader `bash` capability
- keep the Jarvis app runtime stable
- keep the bind-mounted repo protected from accidental mutation through shell access
- avoid the blast radius of moving the whole app out of Docker

This plan covers both `bash` and `python_interpreter`, but they should not necessarily end up with the same capability envelope.

## Problem Statement

Today the Jarvis app, the bind-mounted repo, the shared workspace, and the tool execution runtime all live inside the same `dev` container.

That creates a bad tradeoff:

- if `bash` stays heavily restricted, it is often too weak to be useful
- if `bash` becomes broad enough to install packages and mutate the OS, it can also disrupt the Jarvis app runtime because it is mutating the same container OS

The current local sandbox approach helps with some path hiding and environment scrubbing, but it does not solve the core issue when broad system mutation is desired.

## Goals

- Keep the Jarvis app in Docker.
- Isolate risky tool execution away from the app container.
- Route `bash` into a dedicated sibling container with no repo mount.
- Route `python_interpreter` into that same sibling container so it shares the same Linux runtime and installed toolchain context.
- Keep `/workspace` as the shared bind-mounted handoff boundary.
- Preserve the current tool call UX as much as possible from the model's perspective.
- Keep policy checks in the main app before remote execution.
- Run both remote tools natively inside the isolated `tool_runtime` container.
- Slim the `dev` container so it contains only what the Jarvis app itself needs at runtime.

## Recommended Target Topology

Use two containers in the same `docker compose` project:

1. `dev`
2. `tool_runtime`

Recommended responsibilities:

- `dev`
  - runs the Jarvis app
  - mounts the repo at `/repo`
  - mounts the workspace at `/workspace`
  - owns the main LLM loop, storage, memory, approval flow, tool policy, and non-remote tools
  - should not include general shell or media utilities that exist only for agent tool execution
- `tool_runtime`
  - does not mount `/repo`
  - mounts the same workspace at `/workspace`
  - runs a small internal HTTP service
  - executes only `bash` and `python_interpreter`
  - may mutate its own container OS without affecting the app container
  - owns the CLI, archive, media, and build utilities needed by remote tool execution

## Core Design

### Shared Boundary

Keep `~/.jarvis/workspace` bind-mounted into both containers as `/workspace`.

That gives both the app and the tool runtime a stable shared filesystem contract for:

- user files
- runtime tool manifests
- temporary outputs
- artifacts intended to be consumed by other tools or sent back to the user

### Repo Boundary

Mount the repo only into `dev`.

The `tool_runtime` container should not see `/repo` at all.

This removes the main accidental-damage concern without complicated path masking.

### Tool Transport

Run a small internal HTTP service inside `tool_runtime`.

Recommended endpoints:

- `GET /health`
- `POST /tools/bash/execute`
- `POST /tools/python_interpreter/execute`

Request payload should be minimal and explicit:

- `call_id`
- `arguments`
- `session_id` optional
- `route_id` optional

Response payload should mirror `ToolExecutionResult`:

- `call_id`
- `name`
- `ok`
- `content`
- `metadata`

Policy should remain in `dev`. The remote runtime should execute only already-authorized calls.

## Tool Capability Model

### `bash`

Recommended target:

- runs directly in `tool_runtime`
- uses the real container filesystem of `tool_runtime`
- can install packages and tools into that container
- can access the network
- can write freely inside `/workspace`
- cannot see `/repo` because it is not mounted there

Recommended simplification:

- run `bash` directly in `tool_runtime` with no extra inner sandbox layer
- keep timeouts, output truncation, and structured result formatting
- keep environment scrubbing unless there is a strong reason to loosen it

Important note:

- broad `bash` is now acceptable because it only threatens the disposable tool-runtime container, not the Jarvis app runtime

### `python_interpreter`

Recommended target:

- route execution to `tool_runtime`
- use the Python environment and Linux toolchain available inside `tool_runtime`
- keep the tool one-shot and non-kernelized
- keep its high-level schema for compatibility (`code` or `script_path`, `args`, timeout)

Recommended initial capability split:

- allow broader package availability by running against the tool-runtime Python environment
- keep `script_path` limited to `/workspace`
- keep process-spawn blocking
- keep direct shell escape routes blocked

Reason:

- if `python_interpreter` can freely spawn subprocesses or mutate the full container OS, it becomes a bypass around any `bash` approval policy
- once that happens, `bash` policy no longer means anything

Recommended phase ordering:

1. Move `python_interpreter` to the tool-runtime container.
2. Keep its current safety posture around subprocess escape and direct OS mutation.
3. Re-evaluate whether additional freedom is desirable after the new split is stable.

This keeps the control model coherent.

## Policy Design

## Main Principle

Policy remains local in `dev`.

The remote container should not decide what is allowed. It should only execute what the main runtime already approved.

### `bash` Policy

Recommended direction:

- keep approval for installation and broader mutation commands
- allow those commands to target the tool-runtime container
- add hard-deny rules for commands that are pointless or harmful even inside the tool-runtime container

Recommended hard-deny categories:

- OS upgrade commands
  - `apt upgrade`
  - `apt-get upgrade`
  - `apt full-upgrade`
  - `apt-get full-upgrade`
  - `apt-get dist-upgrade`
  - `do-release-upgrade`
- service or init control
  - `systemctl`
  - `service`
  - `init`
  - `telinit`
  - `reboot`
  - `shutdown`
  - `poweroff`
  - `halt`
- mount and kernel/admin operations
  - `mount`
  - `umount`
  - `swapon`
  - `swapoff`
  - `modprobe`
  - `insmod`
  - `sysctl -w`
- container-runtime recursion unless explicitly wanted later
  - `docker`
  - `podman`
  - `nerdctl`

Recommended approval-gated categories:

- `apt install`
- `apt-get install`
- `apt-get update`
- package-manager installs for language ecosystems
- writes to important non-workspace locations inside `tool_runtime`
- remote installer pipelines like `curl ... | sh`

Recommended policy metadata changes:

- include `target_runtime: tool_runtime`
- include a new detector reason vocabulary that explains risk in terms of container mutation, not host/app mutation

### `python_interpreter` Policy

Recommended direction:

- keep exactly-one-of `code` or `script_path`
- keep workspace path validation for `script_path`
- keep argument limits and size limits
- do not attempt brittle static code scanning for "bad" Python

Important policy note:

- if `python_interpreter` keeps subprocess blocked, its existing policy can remain relatively simple
- if subprocess is ever allowed, policy must be reconsidered at the capability level because Python can then do almost everything `bash` can do

## Runtime Service Design

Implement a small service inside `tool_runtime`.

Suggested module layout:

- `src/tool_runtime_service/`
- `src/tool_runtime_service/app.py`
- `src/tool_runtime_service/models.py`
- `src/tool_runtime_service/bash_executor.py`
- `src/tool_runtime_service/python_executor.py`

Suggested transport rules:

- internal use only
- no public exposure outside compose unless intentionally added later
- JSON request/response only
- no streaming required in phase 1

Suggested executor rules:

- keep tool result formatting compatible with current transcript expectations
- preserve `call_id` and tool name
- include remote-runtime metadata so debugging is obvious

Recommended metadata additions:

- `runtime_location: tool_runtime_container`
- `runtime_transport: http`
- `container_mutation_boundary: isolated_from_app_runtime`

## Docker And Image Changes

### `docker-compose.yml`

Add a new service, tentatively named `tool_runtime`.

Recommended properties:

- mounts `/workspace`
- does not mount `/repo`
- does not need app secrets by default
- runs the internal HTTP service as its main command
- may keep package caches in named volumes if useful

Keep `dev` for the main Jarvis app.

Recommended follow-up:

- remove `SYS_ADMIN` and `seccomp:unconfined` from `dev` once `bash` and `python_interpreter` no longer execute locally

Open question:

- whether `python_interpreter` needs any additional in-container limits beyond subprocess blocking, timeouts, and normal service-level safeguards

### Dockerfiles

Recommended structure:

- either create a dedicated `Dockerfile.tool_runtime`
- or factor a shared base image and build `dev` and `tool_runtime` from it

Recommended image split:

- `dev` should contain only the runtime dependencies required to run the Jarvis app itself
- `tool_runtime` should contain the shell, CLI, build, archive, and media utilities needed by `bash` and `python_interpreter`
- if a package is not needed to run the Jarvis app, it should not be installed in `dev`

Packages that should move out of `dev` and live in `tool_runtime` include:

- `ripgrep`
- `curl`
- `zip`
- `unzip`
- `ffmpeg`
- other agent-facing CLI helpers or build tools that exist only to support remote tool execution

Important distinction:

- build-stage tooling is separate from runtime tooling
- if the `dev` image needs temporary build tools during image construction, use multi-stage builds or builder layers rather than leaving those packages installed in the final `dev` runtime image

Likely `tool_runtime` package set:

- `bash`
- `curl`
- `git`
- `ripgrep`
- `zip`
- `unzip`
- `ffmpeg`
- build tools needed by expected runtime installs
- Python runtime and project dependencies needed for the tool-runtime service

Likely `dev` runtime package set:

- Python runtime and project dependencies needed by the Jarvis app
- only system libraries actually required by the app runtime itself
- no general-purpose agent shell utilities unless the app directly depends on them

## App-Side Refactor Inventory

### Tool Runtime Routing

Add a remote execution client in the main app.

Suggested module layout:

- `src/tools/remote_runtime_client.py`
- `src/tools/remote_models.py`

Recommended approach:

- keep `ToolRuntime` as the central orchestrator
- keep policy where it is
- swap the executor implementation for `bash` and `python_interpreter` so those executors call the remote service instead of executing locally

This keeps the rest of the agent loop mostly unchanged.

### Settings

Add app settings for the remote tool runtime, for example:

- `JARVIS_TOOL_RUNTIME_BASE_URL`
- `JARVIS_TOOL_RUNTIME_TIMEOUT_SECONDS`
- `JARVIS_TOOL_RUNTIME_HEALTHCHECK_TIMEOUT_SECONDS`

Keep existing per-tool timeout and output settings unless they become redundant.

### Startup Health

Recommended app startup behavior:

- optionally health-check the `tool_runtime` service
- fail fast or log loudly if `bash` and `python_interpreter` are configured remote but the service is unavailable

## `bash` Tool Refactor Inventory

Current local behavior to retire:

- local inner sandbox invocation
- path masking logic tied to the app container filesystem
- local dependency on sandbox-specific runtime behavior

Behavior to keep:

- timeout handling
- output truncation
- standardized result shape
- approval-aware metadata

Behavior to add:

- remote HTTP execution
- metadata indicating remote isolated runtime

## `python_interpreter` Tool Refactor Inventory

Current local behavior to review:

- local sandboxed execution
- dedicated image-level interpreter venv assumptions
- local runner/config staging

Recommended keep-or-retain items:

- inline code vs `script_path` schema
- timeout behavior
- output truncation
- structured execution result
- script-path validation against `/workspace`
- process-spawn blocking at least in the initial remote version

Recommended simplification options:

- stop requiring a special dedicated venv path if the tool-runtime image already provides the needed Python environment
- keep the runner but point it at the tool-runtime Python instead of a special sandbox venv

## Test Impact

### Tests To Update

- `tests/test_tools.py`
  - remove assumptions that `bash` local execution depends on a special sandbox binary in the app container
  - replace local-runtime skips with either remote-runtime test doubles or explicit integration gating
  - add remote-client failure-path coverage
  - add policy tests for the new hard-deny command categories
- any tests that assert `/repo` invisibility via the old local sandbox path
- any tests that assert `python_interpreter` is only available in the dev container because of the old local sandboxed runtime

### New Tests To Add

- remote `bash` executor serializes request and parses result correctly
- remote `python_interpreter` executor serializes request and parses result correctly
- main runtime returns structured tool errors when the remote service is unavailable or returns malformed output
- `bash` policy hard-denies upgrade and service-control commands
- `bash` approval flow still works for install commands
- `python_interpreter` still blocks subprocess escape in the initial remote design

## Docs To Update If This Plan Is Adopted

- `.codex/AGENTS.md`
  - remove or rewrite the currently added host-app-refactor workflow guidance
  - document the new split as "app container + isolated tool_runtime container"
- `dev_docs/tool_dev_doc.md`
  - rewrite `bash` and `python_interpreter` executor sections
  - update policy descriptions
  - remove stale local-sandbox claims where no longer true
- `dev_docs/runtime_tools_plan.md`
  - update runtime-tool operational guidance because runtime tools will now usually be provisioned inside `tool_runtime`
- `src/identities/PROGRAM.md`
  - explain that `bash` and `python_interpreter` run in an isolated tool runtime container
  - explain repo visibility accurately
  - keep `/workspace` guidance clear
- `README.md`
  - update developer workflow and runtime topology
- `dev_docs/refactor_plan.md`
  - mark as superseded for this problem statement or archive clearly so future agents do not treat it as the active direction

## Migration Phases

### Phase 1: Infrastructure

1. Add `tool_runtime` service to compose.
2. Create the internal HTTP runtime service.
3. Add health endpoint and basic request/response models.

### Phase 2: Route `bash`

1. Add remote client code in the app runtime.
2. Switch `bash` executor to remote execution.
3. Remove local `bash` dependency on inner sandboxing.
4. Implement the new `bash` hard-deny and approval categories.

### Phase 3: Route `python_interpreter`

1. Add remote `python_interpreter` executor path.
2. Keep the current high-level schema and transcript contract.
3. Keep subprocess escape blocked in the initial remote version.
4. Rebase Python execution on the tool-runtime Python environment.

### Phase 4: Cleanup

1. Drop now-unneeded kernel privileges from `dev`.
2. Remove app-container packages that existed only for local `bash` or local `python_interpreter`.
3. Slim the final `dev` runtime image so utilities like `rg`, `curl`, `zip`, `unzip`, and `ffmpeg` live only in `tool_runtime` unless the Jarvis app itself proves it needs one of them.
4. Update docs, prompts, and tests.

### Phase 5: Re-evaluate Capability Expansion

After the split is stable:

1. Decide whether `python_interpreter` should remain constrained.
2. Decide whether the tool-runtime container should be long-lived, resettable, or ephemeral per session.
3. Decide whether some currently discoverable tools should also move there later.

## Open Decisions Answered

- service name: `tool_runtime`
- whether `tool_runtime` is long-lived or rebuilt/reset often -- long lived (when dev container is rebuilt tool_runtime container is auto rebuilt, but leave the option to rebuilt tool_runtime container only without touching dev container)
- whether `python_interpreter` keeps network disabled initially -- disabled
- whether `python_interpreter` should keep any extra in-container limits beyond the initial subprocess-blocking and schema constraints -- for now keep the ones discussed only
- whether install commands require approval by default or whether some approved-safe install subset should be automatic -- all installation command require user approval
- whether the tool-runtime container should have any secrets mounted at all -- no secret mounted in tool-runtime container

## Acceptance Criteria

This plan is successful when all of the following are true:

- `bash` can install and use tooling without risking the Jarvis app container runtime.
- `bash` cannot accidentally mutate the bind-mounted repo because the repo is absent from the tool-runtime container.
- `python_interpreter` uses the same isolated tool runtime container and environment.
- the main app still owns approvals and policy.
- the `dev` container no longer includes shell and media dependencies that are only needed for agent tool execution.
- docs and tests no longer describe the rejected host-app refactor as the active direction.
