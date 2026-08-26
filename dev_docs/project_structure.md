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
  - post-implementation subsystem reference docs
- `notes/notes.md`
  - concise running notes and lessons learned for future agents
- `.codex/AGENTS.md`
  - project-wide coding and workflow rules for coding agents

### Supporting directories

- `tests/`
  - automated test suite
  - `tests/headless/` contains the high-fidelity real-gateway development runner; it is test-only, is not imported by `src/jarvis/`, and is not a packaged script
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
├── runtime_provider_configuration.py
├── logging_setup.py
├── workspace_paths.py
├── core/
├── gateway/
├── identities/
├── llm/
├── memory/
├── skills/
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
  - compatibility layer that exports the user-facing runtime setting constants consumed by the app
  - extracts grouped runtime values from metadata-rich `settings.yml` or the workspace override file
- `settings.yml`
  - shipped non-secret user-facing settings template plus UI metadata
  - grouped YAML source for `settings.py` and `settings_gui.html`
  - internal runtime defaults live in subsystem-local config modules instead of here
- `runtime_env.py`
  - Docker secret loading and runtime environment bootstrap
- `runtime_provider_configuration.py`
  - resolves the effective startup provider/model targets shared by terminal and UI status rendering
- `logging_setup.py`
  - application logging configuration
- `runtime_errors.py`
  - central durable JSONL exception recorder with propagated-exception de-duplication
  - writes route/session-aware records under `archive/error_logs/`
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
- route-wide hard-quiesce for `/stop` (preserve session) and destructive `/new` (replace session); both terminate owned jobs/services, while `/new` also disposes children and archives the old lineage
- bounded adaptive `orchestrator_wait` timers and route-wide `task_status` snapshots for UI liveness
- route-scoped runtime error capture, with full tracebacks and exception-chain metadata persisted as JSONL entries under `/workspace/archive/error_logs/<session_id>.jsonl`; Telegram/UI error boundaries keep those diagnostics out of terminal output and user-facing error text

### `src/jarvis/identities/`

Source-controlled identity/bootstrap prompt files:

- `PROGRAM.md`
- `REACTOR.md`
- `USER.md`
- `ARMOR.md`

These files are part of the installed package, and at runtime the `jarvis_runtime` container seeds missing workspace copies in `/workspace/identities/` from them without overwriting existing workspace Markdown files.

### `src/jarvis/llm/`

Provider-agnostic LLM interfaces and provider adapters.

The ethos of this part: all LLM provider quirks are dealt with right here, the agent loop only deals with normalized and unified i/o. No Provider specific quirks should be leaked beyond this part, unless absolutely necessary.

Includes:

- shared config
- request/response types
- validation
- service lifecycle
- provider implementations under `llm/providers/`
- provider-local translation for cache/state quirks such as OpenRouter response-cache headers, LM Studio `previous_response_id` reuse, and Grok Responses continuation via persisted `previous_response_id`

Current OpenRouter note:

- the OpenRouter adapter always sends `X-OpenRouter-Cache: true` on provider HTTP requests so OpenRouter can reuse identical response payloads independently of prompt caching; its two bounded empty-response retries additionally send `X-OpenRouter-Cache-Clear: true` to remove only the matching response-cache entry while preserving provider prompt-cache locality
- OpenRouter Claude/Anthropic requests also include top-level `cache_control: {"type": "ephemeral"}` so OpenRouter strictly routes them to Anthropic prompt caching instead of third-party Anthropic-compatible vendors
- configured OpenRouter reasoning effort is sent through the normalized `reasoning.effort` field; accepted effort levels remain model-specific
- when the agent loop supplies a prompt-cache key, the OpenRouter adapter passes it as `session_id` for sticky routing and better provider prompt-cache reuse, and it does not request provider throughput sorting because OpenRouter prioritizes sorting over sticky routing
- chat responses surface OpenRouter response-cache headers in normalized `provider_metadata` for cache hit/miss inspection, while the agent loop remains unaware of the provider-specific header contract
- terminal OpenRouter responses containing neither visible text nor usable tool calls are invalid, never successful; retry-safe empty attempts retain no transient transcript content, and diagnostics include generation id, raw finish reason, usage, cache status, reasoning activity, and terminal source
- OpenRouter SSE errors preserve generation, response, endpoint, HTTP, typed-error, and upstream-code metadata in runtime logs; timeout, provider-unavailable, provider-overloaded, and rate-limit failures map to their normalized exception classes and explicitly terminal failures may retry only before normalized output is exposed
- provider request budgets are split into a 3600-second absolute logical deadline (`JARVIS_LLM_REQUEST_DEADLINE_SECONDS`), a 30-second connection timeout (`JARVIS_LLM_CONNECT_TIMEOUT_SECONDS`), and a 3600-second raw-read inactivity timeout (`JARVIS_LLM_READ_TIMEOUT_SECONDS`)
- the absolute deadline spans all attempts/backoff and never resets on stream activity; provider adapters never receive it as a provider payload timeout
- provider lifecycle, keepalive, reasoning, and empty-signature chunks become internal `ProviderActivityEvent` values that `LLMService` consumes without forwarding or persisting
- Anthropic and Gemini use native incremental streaming rather than buffering a complete response before producing normalized events
- streaming retries are normally allowed only before provider acceptance and normalized output; explicitly terminal provider failures may opt into retry after acceptance but never after normalized output, and ambiguous read/write timeouts are not blindly replayed

Current Grok note:

- Grok now uses xAI Responses, not chat-completions
- configured Grok reasoning effort is sent as Responses `reasoning.effort`
- native Grok uses provider-owned stateful continuation through persisted response ids; normal turns send only the new input plus `previous_response_id`
- xAI durable Responses storage (`store=true`) is distinct from its prompt/KV cache: durable storage is what makes a response id rehydratable on a later connection, while prompt caching only accelerates repeated token prefixes
- ordinary text-only Grok turns remain on HTTP Responses with `store=true`; the adapter sends the stable Jarvis session id as the Responses-body `prompt_cache_key`
- image-bearing turns proactively latch that Jarvis session into an ephemeral Grok mode and use a per-session Responses WebSocket with `store=false`, because xAI explicitly warns that storing image request/response history may fail
- once latched, later turns continue incrementally on the same WebSocket from its live response id instead of toggling back to durable storage or rebuilding context locally
- Grok session metadata keeps the live response/record id, last durable response/record anchor, storage mode, and WebSocket generation
- after a socket expiry, disconnect, or process restart, the adapter lazily assembles the bounded ephemeral tail, hydrates the last durable response id, and uses a `generate=false` warmup before sending the current delta; the normal live-socket path neither assembles nor sends that tail
- a recovery tail may validly end with assistant tool calls whose results begin the current delta; this narrow recovery-only case is supported without weakening normal transcript replay validation
- recovery-only image records persist path/hash metadata as `kind=provider_context`, never raw bytes or base64; their compressed snapshots are materialized lazily only during reconnect and are cleaned when the Jarvis session is replaced or compacted
- large PNGs that are clearly tool-produced screenshots (`shots`, `renders`, `screenshots`, or `captures`) may be bounded and transcoded to JPEG; arbitrary JPEG/WebP inputs and unmarked PNGs are preserved, and unchanged screenshots already in the live chain are replaced by a small reuse notice
- the exact xAI `Response is too large to store` failure is classified in the Grok adapter and retried once over `store=false`; matching partial text is suppressed by content prefix and already-announced tool deltas are not emitted twice, while tool execution still waits for the retried response's terminal `DoneEvent`
- Jarvis compaction/new-session boundaries deliberately start a fresh durable Grok chain and release the old recovery media
- Jarvis does not use a public URL or Files upload as the primary workaround, so local workspace images do not need to be exposed externally
- Jarvis no longer manually rebuilds Grok provider context from unified transcript history for normal native Grok turns; local tail reconstruction is a reconnect-only fallback after the durable anchor
- transcript records remain the archive/debug/audit source, and assistant records may still carry opaque Grok `response.output` metadata for inspection
- the older encrypted-reasoning replay path was only needed while Grok was treated as stateless; with xAI Responses continuation, reasoning history is provider-managed instead of Jarvis-owned

Future note:

- keep WebSocket/storage/error behavior Grok-specific; only the generic stateful-continuation descriptor, deferred local-image part, and `provider_context` archive seam are shared infrastructure
- if xAI changes model-family naming or Responses continuation behavior, update the Grok adapter heuristics there rather than leaking model-specific branching into the agent loop

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

### `src/jarvis/skills/`

Workspace-backed agent skill support.

Includes:

- parsing `/workspace/skills/<skill_id>/SKILL.md`
- deterministic skill search
- staged installer output import into `/workspace/skills`
- compact renderers for skill bootstrap and `get_skills`

### `src/jarvis/storage/`

Conversation/session persistence and related storage types. Session JSONL and the session index remain the conversation boundary; task-scoped liveness state—stalled-round counts, progress epochs, bounded runtime-progress signatures, and exact-call suppression state—lives in atomic, change-deduplicated `tool_tasks/<task_id>.json` sidecars so long tool runs do not rewrite the full session index for every call.

### `src/jarvis/subagent/`

Subagent runtime, lifecycle management, storage, prompts, and settings.

Packaged prompt resources live under `subagent/prompts/`.

### `src/jarvis/tool_runtime_service/`

HTTP service used by the isolated `tool_runtime` container.

This package is launched inside the runtime container via `python -m jarvis.tool_runtime_service`.

Its health response implements the versioned contract in `src/jarvis/tool_runtime_protocol.py`, including required tool capabilities. The app validates that contract before use so a stale isolated image cannot silently run an older bash protocol.

The test-only Ox Alpha runner is launched from the host with `utils/run_dev_headless.sh`. It executes `tests/headless/headless.py` inside `jarvis_runtime`, requires the unchanged `/workspace/settings/settings.yml` routing both main and subagents through `openrouter/stealth/ox-alpha`, uses a unique `/workspace/jarvis-test-headless-*` mutation boundary, audits real websocket events, and performs `/new` plus local artifact cleanup on completion or handled shutdown signals.

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
- installed skills
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

- subsystem reference docs
- architecture or layout documentation
- durable maintenance notes for implemented behavior

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
