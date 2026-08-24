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

Invalid model tool arguments are recoverable tool results. Schema diagnostics identify the exact JSON argument path, received value, and failing schema rule so the model can correct nested payloads instead of guessing from a generic validation error.

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
- `bash` commands are admitted by default; unfamiliar commands, pipelines, redirects, installs, builds, cleanup, and container-local system writes do not need allowlist membership or approval
- high-confidence destructive repository erasure and recursive deletion of `/`, `/workspace`, or the active working directory require exact-command approval unless `BASH_DANGEROUSLY_SKIP_PERMISSION=True`
- `bash` hard-denies upgrade, service/init-control, mount/kernel-admin, and container-runtime-recursion commands
- unmanaged shell detachment is denied because it breaks process ownership; use supervised background/service modes
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

For `bash`, approval is bound to the exact command and resolved `/workspace` working directory. Changing either requires a new approval. For `tool_register`, approval is bound to the exact manifest payload hash. For `email`, approval is bound to the exact send request hash.

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

Files written outside `/workspace` stay local to the long-lived `tool_runtime` container and are not durable Jarvis artifacts. Bash filesystem ownership is enforced in a per-execution mount namespace supplied by the route coordinator, independently of shell parsing: main actors see active child-owned paths read-only, and subagents see the workspace read-only except for their existing owned roots. Runtime-owned job directories are separately writable for supervised process bookkeeping.

The app and isolated service share a versioned compatibility contract from `src/jarvis/tool_runtime_protocol.py`. `/health` declares the protocol version and supported tool capabilities; app startup fails with an actionable rebuild/restart error when the service is stale or missing a required mode. Remote status returned by `tool_runtime` is authoritative because app and tool containers do not share a PID namespace. Routine internal `httpx` request INFO records are context-locally suppressed, while warnings and failures remain logged.

## Detached Bash Jobs

`bash` supports:

- `foreground`
- `background`
- `service`
- `status`
- `tail`
- `cancel`

Foreground jobs that exceed the soft timeout are promoted to background and return a `job_id`. Background job metadata persists under `.jarvis_internal/bash_jobs/` with owner route/session/turn/agent identity and the resolved display working directory.

Route-scoped supervision owns detached bash monitoring outside the model loop. After a detached start or promotion, the current turn parks. Later progress notices enqueue runtime turns for the owning main agent or subagent.

Notices are batched and deduplicated by owner so multiple job updates coalesce into one revival. Accepted notices have an in-memory delivery latch until the owner queue records them, preventing a terminal or unchanged update from being redispatched on every supervisor poll. Deferred running notices are rechecked after 30 seconds so terminal transition is still discovered; deferred terminal notices remain latched until owner state changes.

Running jobs do not emit model-facing heartbeats merely because output started or grew. Those observations update durable supervisor state and keep terminal detection alive without reviving the owner model. A no-output/stalled job emits at most one needs-attention wake after five minutes; terminal transitions always wake the owner. Terminal notices include process exit/signal/runtime, command and output hashes, launch revision, bounded output tails, and durable stdout/stderr log paths. Exit status is process evidence, not semantic readiness evidence; record the actual acceptance check before reporting completion.

Readiness-verified `service` jobs are route-managed resources rather than pending task work. Healthy services do not keep task activity open or wake the model, and their health status is polled at a lower cadence. Both `/stop` and `/new` cancel them through the same isolated-runtime path. The shared route supervisor is the only process-cancellation authority; subagent reset/disposal never inspects or signals tool-runtime PIDs locally, and route cancellation still runs if child disposal fails.

Route-level `/stop` hard-preempts foreground tool awaits and then asks the route supervisor to cancel and finalize every route-owned detached job and service before acknowledging completion. Foreground bash gets best-effort process-group/job cancellation when its active await is cancelled.

`/new` is destructive: it closes the route follow-up gate immediately, cancels every detached job still owned by the route through the configured bash runtime, marks terminal notice state as finalized, clears retained notices, and only then creates the fresh main session. Old job metadata and logs remain as archive artifacts, but the supervisor cannot recover them into a later turn.

## Implemented Basic Tools

### `bash`

Runs shell commands in `tool_runtime` with `/workspace` as the durable boundary. Foreground and background calls may provide `cwd` as a relative path or `/workspace`-absolute path; it must resolve to an existing directory inside the workspace, applies only to that call, and defaults to `/workspace`. Job-control calls do not accept `cwd`. The tool also supports status, tail, cancellation, output truncation, log retention, exact approval for the narrow destructive blacklist, and route-level background supervision.

The command shell starts with `set -e -o pipefail`, so a failing verification command cannot be hidden by a later successful command in the same invocation. A shared non-executing shell lexer distinguishes asynchronous-list operators from descriptor/combined redirects such as `2>&1`, `<&3`, `&>`, and `>|`, and ignores quoted text, comments, arithmetic, and heredoc bodies. Shell-managed backgrounding (`&`, `nohup`, `disown`, `setsid`, nested shell/eval payloads, and equivalent detach wrappers) is rejected. Use `mode="background"` for supervised jobs or `mode="service"` for a long-lived process with allocated/preflighted loopback port, PID/PGID ownership, readiness URL, command/revision provenance, logs, status, and cancellation. Bash has no model-declared write scope or lease-generation arguments; the route runtime derives actor capabilities and the isolated executor applies them to foreground commands, promoted/background jobs, services, and Bash invoked inside acceptance gates.

### `file_patch`

Applies structured one-file text edits inside `/workspace` through literal operations:

- `write`
- `replace`
- `insert_before`
- `insert_after`
- `delete`

Writes are atomic. Non-write operations require exact single matches.

Every operation has the same required shape: `type`, `match`, and `replacement`. `write` uses an empty `match` and full-file `replacement`; `delete` uses an empty `replacement`; the other operations apply their documented literal semantics. This avoids provider-incompatible operation unions and reduces malformed model calls.

Callers may pass `expected_sha256` from a prior inspection to fail closed when the file changed before application, or `expected_file_absent=true` when inspection established a new target. These content preconditions are optional compare-and-swap controls and are independent of actor ownership. Successful results return artifact provenance and the resulting content digest. Missing and ambiguous exact matches return bounded line candidates or exact line locations, the current digest, and an explicit reread-and-retry instruction; no operation is written until the entire ordered patch succeeds.

The tool description includes a minimal valid replace payload because nested patch operations are a common model-shape failure; runtime schema errors use stable error codes, canonical examples, and a bounded retry budget without reflecting malformed raw payloads into the transcript.

### `file_write` and `file_replace`

Use `file_write(path, content)` for a whole-file create or rewrite, and `file_replace(path, match, replacement)` for one exact unique replacement. These flat shapes cover the common edit cases that previously caused malformed nested `file_patch` calls. All three file-edit tools are atomic, support `expected_sha256`, and participate in route-level path coordination.

### `acceptance_run` and `acceptance_record`

`acceptance_run` executes each required gate independently, rejecting top-level shell chaining/pipes and nested shell command wrappers so one successful command cannot mask another failure. Callers must provide explicit material `revision_paths`; the ledger records those paths, exact commands, exit status, duration, bounded output plus hash, and one scoped content-derived revision before/after each gate. Runtime archives, caches, dependencies, and generated logs are excluded so unrelated supervisor churn cannot invalidate valid evidence. It also supports a deterministic `source_line_count` gate for authored source only: selected paths must remain inside `/workspace`, source symlinks cannot escape it, and common test, probe, generated, build, dependency, and vendor trees are excluded by default. This metric deliberately does not execute shell code.

`acceptance_record` records the durable issue/acceptance ledger with statuses such as `open`, `fixed`, `not_a_bug`, `deferred`, and `user_waived`; it requires the exact `revision_paths` and still-current scoped revision from the matching `acceptance_run`, and cites a passing gate call from the current mutation epoch. Items accumulate by stable `item_id`, so a later partial ledger cannot hide an older required open item. User-contract items specifically require `fixed` or `passed` plus matching observed evidence; they cannot be waived by a model-authored ledger. Required unresolved items prevent a completion claim. A passing process exit alone is not sufficient evidence of an externally observable requirement.

## Workspace Coordination

Route actors share `/workspace`, but no longer share one global tool mutex. Persistent `owned_paths` assigned to a subagent remain write leases until the child is disposed. Owned roots must already exist, preventing a missing-path capability from widening to a writable parent. Direct path tools coordinate their targets automatically; subagents cannot write outside their owned roots, and main writes that overlap child ownership are rejected. Bash receives a runtime-derived filesystem view, while unknown unscoped writers use the global barrier and are unavailable to subagents.

Lease denials return a structured `conflict_class`, diagnostic `conflict_key`, and remediation. Tool liveness accounting never merges different commands or tools by a broad semantic class: only the exact normalized tool and arguments with the same stable result accumulate toward suppression.

Every routed tool call also receives a content-derived before/after workspace observation covering its coordinated access scope. Main Bash observations exclude concurrently child-owned roots, while child Bash observations cover only that child’s owned roots, so another actor’s concurrent mutation is not misattributed. The observation is attached even when execution fails; tool-declared mutation metadata is only a fallback when direct observation is unavailable.

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
