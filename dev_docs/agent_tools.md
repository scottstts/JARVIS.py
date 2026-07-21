# Agent Tools

## Purpose

This document describes Jarvis's tool system, built-in tools, discoverable tools, runtime tools, policy, approvals, and background-work orchestration.

Tool code lives under `src/jarvis/tools/`. Subagent control primitives live under `src/jarvis/subagent/` and are injected by the route runtime; they are not normal registered tools.

## Tool Packages

Built-in tools are organized as:

- `src/jarvis/tools/basic/<tool_name>/`
- `src/jarvis/tools/discoverable/<tool_name>/`
- shared modules under `src/jarvis/tools/`

Shared modules include:

- `registry.py`
- `runtime.py`
- `policy.py`
- `config.py`
- `types.py`
- `runtime_tools.py`
- `runtime_tool_manifest.py`

## Registry And Exposure

Executable tool exposure classes:

- `basic`: exposed at the start of every normal turn
- `discoverable`: hidden by default; surfaced through `tool_search`

The initial `LLMRequest` contains only basic tools. The session transcript logs the raw basic `ToolDefinition` payloads as transcript-only audit data; that record is not replayed as model-visible history.

Discoverable catalog entries are separate from executable `ToolDefinition`s. They provide compact search docs for `tool_search` and may optionally link to a backing executable tool through `backing_tool_name`.

`ToolRegistry.filtered_view(...)` provides actor-scoped visibility. Every executable and backed discoverable tool declares `allowed_agent_kinds`; new tools should not rely on implicit defaults.

Subagent filtered views hide the configured built-in blocklist while still allowing runtime manifest discoverables.

## Discoverable Catalog

Discoverable entries contain:

- `name`
- `purpose`
- optional `aliases`
- optional `detailed_description`
- optional `usage`
- optional `metadata`
- optional `backing_tool_name`

Search indexes `name`, `aliases`, `purpose`, and `detailed_description`. `usage` and `metadata` are returned but not indexed.

Discoverable text is optimized for model runtime:

- one search-friendly purpose sentence
- compact detailed description
- minimal usage fields
- no obvious examples
- no repeated defaults or limits across fields
- metadata only when it adds unique operational signal

## Discoverable Activation

Backed discoverable tools activate only through high-verbosity `tool_search`.

Flow:

1. initial request exposes basic tools
2. model calls `tool_search`
3. low verbosity returns information only
4. high verbosity returns richer docs and activation metadata for backed matches
5. the agent loop scans current-turn tool results for activation metadata
6. matching executable discoverable tools are added to the next follow-up request
7. activation lasts only for the current turn

Activation does not persist across user turns.

## Runtime Tools

Runtime tools are data-driven discoverable entries loaded dynamically from:

```text
/workspace/runtime_tools/*.json
```

They are not repo-defined executable tools and do not create new Python executors. They usually describe capabilities installed or built at runtime and then used through existing operators such as `bash`.

Runtime manifests are loaded at `tool_search` execution time, so no restart is required.

Manifest fields include:

- `name`
- `purpose`
- `aliases`
- `detailed_description`
- `usage`
- `notes`
- `operator`
- `invocation`
- `provisioning`
- `artifacts`
- `rebuild`
- `safety`

`name`, `purpose`, and `operator` are mandatory. The manifest must contain enough information to rebuild the capability in a fresh workspace. If it depends on a custom script, the manifest stores the script content or pinned source plus integrity data.

`tool_search` labels runtime entries with `source: runtime_tools`, `manifest_path`, and `operator`.

## `tool_register`

`tool_register` is a basic tool that writes one runtime tool manifest under `/workspace/runtime_tools/`.

It:

- validates the manifest
- requires exact-action approval bound to the manifest payload hash
- writes atomically
- supports update/replace behavior
- returns the written path and normalized manifest metadata

It does not install software, build artifacts, or create repo executors.

## Policy

Policy routing lives in `src/jarvis/tools/policy.py`; per-tool policy lives beside each tool.

Current policy highlights:

- `bash` is checked in `jarvis_runtime` and executed in isolated `tool_runtime`
- `bash` requires approval for install/build/system-mutation commands unless `BASH_DANGEROUSLY_SKIP_PERMISSION=True`
- `bash` hard-denies upgrade, service/init-control, mount/kernel-admin, and container-runtime-recursion commands
- `view_image`, `send_file`, generated image paths, transcribe inputs, and email attachments must stay inside `/workspace`
- `.env` files and `.env` directories are denied for file-send and patch-like surfaces
- `tool_register` always requires exact manifest approval
- `email` always requires exact request approval
- `get_skills` reads only from `/workspace/skills`

Tool failures are returned as structured tool-error results instead of crashing the turn.

## Approvals

Approvals are enforced inside sensitive tools, not by a public model-callable approval tool.

Approval request payloads include fields such as:

- `approval_id`
- `kind`
- `summary`
- `details`
- `command`
- `tool_name`
- `inspection_url`
- `manifest_hash`

For `bash`, approval is bound to the exact command and relevant execution parameters. For `tool_register`, approval is bound to the exact manifest payload hash. For `email`, approval is bound to the exact send request hash.

Rejected approvals do not execute the action. The transcript records both approval requests and decisions for auditability.

Telegram renders approval requests with inline approve/reject controls and sends `approval_response` frames through the gateway.

## Tool Runtime Boundary

`bash` executes in the sibling `tool_runtime` container over internal HTTP.

Important boundaries:

- `/workspace` is shared
- `/repo` is not mounted in `tool_runtime`
- app secrets are not mounted in `tool_runtime`
- shell startup files are disabled
- the environment is scrubbed
- agent-facing Python resolves through `/opt/venv`

Files written outside `/workspace` stay local to the long-lived `tool_runtime` container and are not durable Jarvis artifacts.

## Detached Bash Jobs

`bash` supports:

- `foreground`
- `background`
- `status`
- `tail`
- `cancel`

Foreground jobs that exceed the soft timeout are promoted to background and return a `job_id`. Background job metadata persists under `.jarvis_internal/bash_jobs/` with owner route/session/turn/agent identity.

Route-scoped supervision owns detached bash monitoring outside the model loop. After a detached start or promotion, the current turn parks. Later progress notices enqueue runtime turns for the owning main agent or subagent.

Progress pacing combines immediate signal-driven updates with fallback heartbeats. Notices are batched and deduplicated by owner so multiple job updates coalesce into one revival.

Route-level `/stop` preempts foreground tool awaits. Foreground bash gets best-effort process-group/job cancellation when its active await is cancelled. Already-detached bash jobs are different: `/stop` suppresses their auto-followups until the next user message, but it does not cancel those jobs.

`/new` is destructive: it closes the route follow-up gate immediately, cancels every detached job still owned by the route through the configured bash runtime, marks terminal notice state as finalized, clears retained notices, and only then creates the fresh main session. Old job metadata and logs remain as archive artifacts, but the supervisor cannot recover them into a later turn.

## Implemented Basic Tools

### `bash`

Runs shell commands in `tool_runtime` with `/workspace` as the durable boundary. Supports foreground/background execution, status, tail, cancellation, output truncation, log retention, approval-gated installs/builds, and route-level background supervision.

### `file_patch`

Applies structured one-file text edits inside `/workspace` through literal operations:

- `write`
- `replace`
- `insert_before`
- `insert_after`
- `delete`

Writes are atomic. Non-write operations require exact single matches.

### `view_image`

Attaches one local workspace image to the next model follow-up request. Supports PNG, JPEG, and WebP. The image payload is current-turn-only and is not persisted in transcript history.

### `send_file`

Sends a workspace file to the user through Telegram, using the active route when available and falling back to the configured owner chat.

### `web_search`

Runs Brave web search and returns concise normalized web results.

### `web_fetch`

Fetches one public HTTP(S) URL and returns clean markdown through a staged fallback flow using direct markdown, Defuddle in `tool_runtime`, Cloudflare HTML-to-Markdown, and local Playwright rendering when needed.

### `tool_search`

Searches built-in discoverables and runtime tool manifests. Low verbosity is informational; high verbosity can activate backed discoverable executable tools for the rest of the current turn.

### `get_skills`

Searches or opens installed workspace skills. `search` is exposed only when skill headers are not bootstrapped. `get` returns one full `SKILL.md` plus a bounded resource listing.

### `memory_search`, `memory_get`, `memory_write`

Dedicated memory tools described in `agent_memory.md`.

### `tool_register`

Approval-gated runtime tool manifest registration.

## Implemented Discoverable Tools

### `memory_admin`

Manual memory maintenance and inspection actions. Discoverable only and intended for explicit memory-admin requests.

### `generate_edit_image`

Generates a new image or edits one workspace image through OpenAI or Gemini after discovery. Writes output inside `/workspace`.

### `transcribe`

Transcribes one workspace audio/media file through OpenAI after discovery. Supports endpoint-safe formats and enforces the current upload size limit.

### `email`

Sends one approved email through the configured SMTP account. Supports one recipient and optional workspace file attachments.

### `ffmpeg`

Docs-only discoverable entry telling the agent to use installed `ffmpeg` or `ffprobe` through `bash`.

## Tool Documentation Rules

When adding a tool, document:

- status
- exposure
- package
- purpose
- input schema
- executor behavior
- policy
- limitations

When adding a discoverable entry, document:

- name
- aliases
- purpose
- detailed description
- usage
- metadata
- backing tool

When adding a runtime tool capability, update the manifest contract or examples only if the data format changes.
