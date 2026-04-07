# Jarvis Project Structure

## Purpose

This document describes the current intended repo structure after the `uv` packaging refactor.

Use this as the default map of the project when deciding where new code, tests, docs, or container changes should go.

If this document and the code ever disagree, treat the code as source of truth and update this file.

## Core Rules

- The project is strict container-first for Python.
- The installable Python package lives under `src/jarvis/`.
- Do not add new repo-root Python entrypoint shims.
- Add runtime entrypoints through `pyproject.toml` `[project.scripts]`.
- Run `uv` only inside the `jarvis_runtime` container against `/repo`.
- The `tool_runtime` container does not mount the repo; it runs the installed package from its image.

## High-Level Repo Layout

```text
.
├── pyproject.toml
├── uv.lock
├── README.md
├── docker-compose.yml
├── Dockerfile.jarvis_runtime
├── Dockerfile.tool_runtime
├── src/
│   └── jarvis/
├── tests/
├── dev_docs/
├── notes/
├── secrets/
├── assets/
├── utils/
└── vendor/
```

## Root-Level Responsibilities

### Packaging and dependency files

- `pyproject.toml`
  - package metadata
  - build-system configuration
  - `uv` dependency declarations
  - script entrypoints
  - tool config for `pytest` and `ruff`
- `uv.lock`
  - locked dependency graph for container builds and `uv sync`

### Container/runtime files

- `docker-compose.yml`
  - defines the `jarvis_runtime` and `tool_runtime` services
  - mounts `/repo` only into `jarvis_runtime`
  - mounts the shared `/workspace` into both services
- `Dockerfile.jarvis_runtime`
  - Linux development environment
  - installs `uv`
  - prepares `/opt/venv`
  - installs Playwright Chromium for app-runtime-only browser rendering fallbacks such as `web_fetch`
  - installs a `/usr/local/bin/jarvis` shell wrapper that prefers the tool-installed artifact when present
  - intended for `uv run ...` against the bind-mounted repo
- `Dockerfile.tool_runtime`
  - isolated runtime image for the HTTP tool runtime service
  - imports Node 22 tooling from the official `node:22-*-slim` image through a staging copy that preserves launcher symlinks and bundled targets
  - installs pinned global `defuddle` and the CA env needed for Node-based HTTPS fetches in `tool_runtime`
  - installs the `jarvis` package into `/opt/venv`
  - does not rely on `PYTHONPATH`

### Documentation and reference files

- `README.md`
  - developer/operator quickstart
- `dev_docs/`
  - design docs, implementation plans, and subsystem docs
- `notes/notes.md`
  - concise running notes and lessons learned for future agents
- `.codex/AGENTS.md`
  - project-wide coding and workflow rules for coding agents

### Supporting directories

- `tests/`
  - automated test suite
- `secrets/`
  - local secret-file inputs for Docker Compose
- `assets/`
  - repo assets such as images
- `utils/`
  - supporting utilities/scripts that are not part of the installable package
  - includes `settings_gui.html`, a Chrome-oriented metadata-driven settings renderer copied into `/workspace/settings/`
  - includes `install_build.sh`, a helper that execs into `jarvis_runtime` from the host and ensures the built wheel is installed when needed
  - includes `jarvis_shell_wrapper.sh`, the container-level `jarvis` wrapper installed into `/usr/local/bin/`
  - includes `migrate.sh`, a workspace archival helper copied into `/workspace/migrate.sh`
- `vendor/`
  - vendored third-party source/assets used during builds
  - currently includes `vendor/sqlite-vec/`

## Installable Package Layout

All runtime Python code lives under `src/jarvis/`.

```text
src/jarvis/
├── __init__.py
├── __main__.py
├── main.py
├── settings.py
├── settings.yml
├── runtime_env.py
├── logging_setup.py
├── workspace_paths.py
├── core/
├── gateway/
├── identities/
├── llm/
├── memory/
├── storage/
├── subagent/
├── tool_runtime_service/
├── tools/
└── ui/
```

### Package entrypoints and top-level support modules

- `__main__.py`
  - package-level `python -m jarvis` support
- `main.py`
  - combined runtime entrypoint for running gateway + UI together
  - backs the `jarvis` script
- `settings.py`
  - compatibility layer that exports the runtime setting constants consumed by the app
  - extracts grouped runtime values from metadata-rich `settings.yml` or the workspace override file
- `settings.yml`
  - shipped non-secret runtime settings template plus UI metadata
  - grouped, user-facing YAML source for both the `settings.py` compatibility layer and `settings_gui.html`
  - provider-specific HTTP attribution defaults such as OpenRouter `site_url` and `app_name` live here, and workspace overrides take precedence at runtime
- `runtime_env.py`
  - Docker secret loading and runtime environment bootstrap
- `logging_setup.py`
  - application logging configuration
- `workspace_paths.py`
  - shared workspace path helpers

## Subpackage Responsibilities

### `src/jarvis/core/`

Core agent loop, command handling, compaction, token estimation, and identity/bootstrap loading.

Also contains packaged prompt resources under `core/prompts/`.

### `src/jarvis/gateway/`

Starlette websocket gateway and route/session coordination.

Key responsibilities:

- websocket protocol
- route runtime lifecycle
- session routing
- route event publication
- detached bash-job observation

### `src/jarvis/identities/`

Source-controlled identity/bootstrap prompt files:

- `PROGRAM.md`
- `REACTOR.md`
- `USER.md`
- `ARMOR.md`

These files are part of the installed package, but at runtime the `jarvis_runtime` container also overwrites the workspace copies in `/workspace/identities/` so the shared workspace stays aligned with the repo starter files.

### `src/jarvis/llm/`

Provider-agnostic LLM interfaces and provider adapters.

The ethos of this part: all LLM provider quirks are dealt with right here, the agent loop only deals with normalized and unified i/o. No Provider specific quirks should be leaked beyond this part, unless absolutely necessary.

Includes:

- shared config
- request/response types
- validation
- service lifecycle
- provider implementations under `llm/providers/`
- provider-local translation for cache/state quirks such as LM Studio `previous_response_id` reuse and Grok `x-grok-conv-id` routing plus system-message collapsing

### `src/jarvis/memory/`

Long-term memory subsystem.

Includes:

- canonical Markdown memory
- dirty scanning
- indexing
- retrieval
- maintenance
- reflection
- graph handling
- memory-specific config and types

### `src/jarvis/storage/`

Conversation/session persistence and related storage types.

### `src/jarvis/subagent/`

Subagent runtime, lifecycle management, storage, prompts, and settings.

Packaged prompt resources live under `subagent/prompts/`.

### `src/jarvis/tool_runtime_service/`

HTTP service used by the isolated `tool_runtime` container.

This package is launched inside the runtime container via `python -m jarvis.tool_runtime_service`.

Current built-in remote endpoints execute `bash` and `web_fetch`.

### `src/jarvis/tools/`

Agent tool system.

Top-level responsibilities:

- tool registry
- runtime abstraction
- policy interface
- runtime manifest handling
- remote runtime client

Substructure:

- `tools/basic/`
  - always-available built-in tools
- `tools/discoverable/`
  - discoverable executable tools and docs-only discoverables

### `src/jarvis/ui/`

User-facing interfaces.

Current implementation:

- `ui/telegram/`
  - Telegram API client
  - bot bridge
  - gateway client
  - formatting and config

## Tests Layout

The `tests/` directory is repo-root, separate from `src/`, and imports the installed package namespace (`jarvis.*`).

General rules:

- keep tests near the subsystem they exercise by filename and naming
- add new tests under `tests/`, not inside `src/jarvis/`
- test package imports should target `jarvis...`, not old flat top-level module names

## Runtime vs Repo Data

It is important to keep repo structure and runtime workspace structure separate.

### Repo-controlled code and resources

Examples:

- `src/jarvis/...`
- `tests/...`
- `dev_docs/...`
- `notes/notes.md`

### Runtime workspace data

Examples inside `/workspace`:

- transcript archives
- memory state
- copied identities
- temporary files
- tool artifacts
- subagent archives

Do not mix runtime-generated data back into `src/jarvis/`.

## `uv` and Entry Point Model

The project is a normal installable Python package named `jarvis`.

Current project script entrypoint:

- `jarvis`

When adding a new runnable component:

1. add the module under `src/jarvis/`
2. expose a `main()` when appropriate
3. only add a `pyproject.toml` script if it is a real user-facing entrypoint
4. otherwise prefer explicit module invocation for internal/container-only processes
5. update docs if the new entrypoint is user-facing

## Where New Code Should Go

### Add code under `src/jarvis/` when it is:

- runtime application code
- package-owned prompts/resources
- config or path helpers for the app
- testable logic used by the runtime

### Add code under `tests/` when it is:

- unit, integration, or regression coverage

### Add docs under `dev_docs/` when they are:

- subsystem design docs
- implementation plans
- architecture or layout documentation

### Add notes under `notes/notes.md` when they are:

- short lessons learned
- sharp design constraints
- implementation gotchas for future agents

## Non-Structural Local Artifacts

You may see local/generated directories such as:

- `__pycache__/`
- `.pytest_cache/`
- `.ruff_cache/`
- `.venv/`
- `src/jarvis.egg-info/`
- `.DS_Store`

These are not part of the intended project structure and should not drive design decisions.
