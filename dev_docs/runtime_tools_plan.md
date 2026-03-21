# Runtime Tools Plan

## Purpose

This document defines the implementation plan for runtime tool registration and approval-gated tool installation/building in Jarvis.

Status note:

- The active execution topology now routes `bash` into the isolated sibling `tool_runtime` container.
- Parts of this document still reflect the older in-app sandbox shape; where they conflict, current code takes precedence.
- Current capability model also differs from parts of this older plan: all agent-facing Python work now goes through `bash` using the central `/opt/venv` environment, and `bash` includes built-in non-PTY background job control.

The target outcome is:

- the agent can install or build tools at runtime, mostly through `bash`
- the agent can register those tools as discoverable runtime tools without changing repo code
- `tool_search` surfaces both built-in discoverable tools and runtime-registered tools dynamically
- install/build/register actions are approval-gated with real session state and Telegram inline-button UX
- broad `bash` access is allowed, but a small blacklist still blocks clearly sensitive paths

## Locked Decisions

- Do not introduce protocol naming like `v1` or `v2`.
- Runtime tools are data-driven discoverable entries, not repo-defined executable tools.
- Runtime tools are loaded dynamically from `/workspace/runtime_tools/*.json` when `tool_search` runs.
- Registering a runtime tool must not require restarting the agent system.
- Runtime tool execution uses existing runtimes, mainly `bash`, or agent-created scripts/binaries.
- Approval is enforced inside sensitive tools, especially `bash` and `tool_register`, not by a standalone public approval tool.
- `/workspace` has no hard filesystem restrictions at the `bash` sandbox level.
- `/repo` is hard-blocked for `bash` with no read or write access.
- Sensitive system paths such as `/run/secrets` stay hard-blocked.
- Installing/building a tool and registering a tool both require explicit user approval.
- Approval must have real session state and a Telegram inline keyboard approve/reject flow.
- The `bash` approval detector should be moderately aggressive: strong enough to catch common install/build/system-mutation commands, but not so aggressive that ordinary shell use becomes constantly noisy.
- Add `BASH_DANGEROUSLY_SKIP_PERMISSION` to [src/settings.py](/Users/scott/Documents/Projects/Python/Jarvis/src/settings.py), default `False`. When `True`, the built-in `bash` approval detector is bypassed entirely.
- If the user rejects an approval request, the current turn stops waiting and the system remains idle until the next user-initiated message.

## High-Level Design

### 1. Runtime tools are a manifest protocol

Each runtime tool is represented by one JSON file under:

- `/workspace/runtime_tools/<tool_name>.json`

That JSON is the source of truth for:

- discoverability through `tool_search`
- agent-facing usage guidance
- reproducibility metadata for rebuilding the tool in a fresh workspace/runtime

Runtime tools are not added to the built-in executable registry in code. They exist as discoverable runtime data only.

### 2. Built-in discoverables and runtime discoverables stay separate internally

The current built-in discoverable catalog in [src/tools/registry.py](/Users/scott/Documents/Projects/Python/Jarvis/src/tools/registry.py) should remain the source for repo-defined discoverables.

Runtime discoverables should be loaded separately from `/workspace/runtime_tools/` and merged at `tool_search` execution time. This avoids:

- startup-time caching
- restart requirements
- accidental coupling between repo code registration and runtime data registration

### 3. Runtime tools are usually used through `bash`

The common runtime-tool flow is:

1. Agent attempts a `bash` install/build command.
2. If the command trips the built-in approval detector, `bash` does not execute it and instead emits an approval request for that exact command.
3. User approves via Telegram inline keyboard.
4. The agent loop resumes and executes that exact approved `bash` command.
5. Agent attempts `tool_register` with the manifest payload.
6. `tool_register` requires approval for that exact manifest payload if it has not already been approved.
7. User approves via Telegram inline keyboard.
8. The agent loop resumes and `tool_register` writes `/workspace/runtime_tools/<name>.json`.
9. Future `tool_search` calls surface that runtime tool dynamically.

If the agent later wants to use that tool, it usually does so through `bash` following the manifest’s documented invocation pattern.

## Runtime Tool Manifest Shape

The manifest should be strict enough to support deterministic discovery and rebuild, but not so abstract that it becomes a second programming language.

Recommended top-level shape:

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

Required expectations:

- `name`, `purpose`, and `operator` are mandatory.
- `operator` will usually be `"bash"`.
- `usage` is what `tool_search` renders for the agent.
- `invocation` explains how the tool is actually used after discovery.
- `provisioning` records how the tool was installed or built.
- `rebuild` must contain enough information for another fresh agent to restore the capability from the JSON alone.

Important rebuild rule:

- If the runtime tool depends on a custom script created by the agent, the manifest must store the script content directly or store enough pinned source information plus integrity data to recreate it deterministically.

That rule is what makes `/workspace/runtime_tools/*.json` portable instead of being just a label catalog.

## New Basic Tools

### `tool_register`

Add a new basic tool at:

- `src/tools/basic/tool_register/`

Responsibilities:

- validate runtime tool manifest input
- enforce approval before registration
- write the manifest atomically to `/workspace/runtime_tools/<name>.json`
- support update/replace semantics for an existing runtime tool
- return a normalized result including the written manifest path

Non-responsibilities:

- it does not install or build the tool
- it does not create repo code
- it does not create a new runtime executor

Expected input responsibilities:

- full runtime tool manifest payload
- user-facing approval context fields such as summary, details, and optional inspection URL
- optional replace/update mode for existing manifest names

## Internal Approval Subsystem

Do not add a standalone public `approval_request` basic tool as the security boundary.

Reason:

- a compromised or misaligned model could skip that tool and call `bash` directly
- approval would then be advisory rather than enforced

Instead, add an internal approval subsystem used by sensitive tools and the agent loop.

Responsibilities:

- create a pending approval request in session state
- capture the agent-written rationale text shown to the user
- capture the exact command or registration action being approved
- capture optional inspection URL such as the tool website
- allow `bash` and `tool_register` to return `approval_required` results instead of executing immediately
- emit structured metadata that the agent loop can translate into a gateway/UI approval event
- suspend progress until approved or rejected

Recommended internal approval request payload:

- `kind`
- `summary`
- `details`
- `command`
- `tool_name`
- `inspection_url`
- `manifest_hash`

This approval subsystem may live in `src/core/` or `src/tools/`, but it should not be directly model-callable as a separate tool.

## Approval Model

## Approval State

Approval state should be explicit and persisted.

Add session-level approval metadata in [src/storage/types.py](/Users/scott/Documents/Projects/Python/Jarvis/src/storage/types.py) and [src/storage/service.py](/Users/scott/Documents/Projects/Python/Jarvis/src/storage/service.py).

Implemented persisted shape:

- a single `pending_approval` object on session metadata

Current behavior:

- `pending_approval` is present only while a turn is actively waiting for approval
- once approved, rejected, interrupted, or consumed, `pending_approval` is cleared from session metadata
- approval requests and decisions are recorded in the transcript for auditability instead of being expanded into multiple parallel session fields

The transcript should also record approval requests and outcomes so they remain auditable.

## Approval Binding

Approval must be bound to the exact action, not just to a vague category.

For `bash`:

- approval should bind to the exact command string
- optionally also bind to timeout and shell executable
- approval should be one-time-use unless explicitly marked reusable

For `tool_register`:

- approval should bind to the exact manifest payload hash

That prevents approval for one action from silently authorizing a different one.

## Approval Flow

Planned end-to-end flow:

1. The model calls `bash` or `tool_register`.
2. The called tool decides that approval is required for the exact requested action.
3. The tool runtime writes pending approval state and returns a structured `approval_required` result instead of executing.
4. The agent loop emits a new stream event carrying the approval payload.
5. The gateway forwards that event to the Telegram UI.
6. The Telegram UI sends a message with inline `Approve` and `Reject` buttons.
7. The current turn waits on approval resolution.
8. If approved, the gateway/UI notifies the agent loop, which resumes the same turn and replays the exact approved action.
9. If rejected, the current turn stops waiting and ends without executing the action.
10. The next user-initiated message starts the next turn normally.

## Bash Changes

### 1. Sandbox model changes

The current `bash` tool in [src/tools/basic/bash/tool.py](/Users/scott/Documents/Projects/Python/Jarvis/src/tools/basic/bash/tool.py) uses a workspace-only mount model.

Change it to a broad-access blacklist model while still keeping:

- `bubblewrap`
- scrubbed environment
- no shell startup files
- explicit timeout handling
- structured stdout/stderr result capture

The new filesystem rule is:

- `/workspace` is broadly available with no hard path restrictions
- `/repo` is completely hidden
- sensitive system locations are hidden

Initial hard-block list:

- `/repo`
- `/run/secrets`
- `/proc/kcore`

Additional blocked locations should be evaluated during implementation, but the blocklist must stay small and explicit.

The plan should avoid reintroducing a broad whitelist-by-mount design through the back door.

### 2. Approval enforcement in `bash`

Broader `bash` access requires real enforcement for install/build behavior.

Implement a `bash` approval gate with two parts:

- exact-command approval token validation
- a moderately aggressive detector for install/build/system-mutation commands that must have approval

To make approval prompts legible, the `bash` tool schema should be extended with optional user-facing approval context fields such as:

- `approval_summary`
- `approval_details`
- `inspection_url`

If approval is required and no such fields are provided, the runtime may fall back to a generic summary built from the command, but the preferred path is for the model to provide them explicitly.

Examples that should require approval when detected:

- package-manager installs
- shell installer pipelines
- writes into system executable locations
- creation of persistent system-level binaries or scripts outside `/workspace`

The detector should be moderately aggressive:

- it should catch common installation and system-mutation behavior reliably
- it should not trigger on ordinary inspection, search, listing, file reading, or common workspace-local editing commands

The exact command list can be tuned during implementation, but the policy must still fail closed for obvious install/build patterns.

Recommended behavior:

- if a command matches install/build heuristics and no valid approval token is attached, deny execution
- if approval token is attached but does not match the exact approved command, deny execution
- ordinary analysis commands remain runnable without approval

### 3. Settings

Add a new runtime setting in [src/settings.py](/Users/scott/Documents/Projects/Python/Jarvis/src/settings.py):

- `BASH_DANGEROUSLY_SKIP_PERMISSION: Final = False`

Behavior:

- when `False`, the built-in `bash` approval detector is active
- when `True`, the detector is entirely bypassed
- exact path blacklisting for `/repo`, `/run/secrets`, and other hard-blocked locations still remains in force

This setting is explicitly dangerous and should be documented that way.

### 4. Result metadata

`bash` result metadata should expose whether approval was required and whether an approval token was consumed.

## Dynamic Runtime Tool Loading

Create a runtime tool loader module under `src/tools/` that:

- reads `/workspace/runtime_tools/*.json`
- validates each manifest
- converts each valid manifest into a runtime discoverable entry shape consumable by `tool_search`
- ignores invalid files without crashing the turn
- returns validation diagnostics for logging

This loader should be called inside the `tool_search` execution path, not only at registry creation time.

Likely new modules:

- `src/tools/runtime_tools.py`
- `src/tools/runtime_tool_manifest.py`

## `tool_search` Changes

Modify [src/tools/basic/tool_search/tool.py](/Users/scott/Documents/Projects/Python/Jarvis/src/tools/basic/tool_search/tool.py) so each search merges:

- built-in discoverables from the registry
- runtime discoverables loaded from `/workspace/runtime_tools/`

Behavior rules:

- runtime tools must appear in both low and high verbosity output
- runtime tools do not activate new callable repo tools
- built-in backed discoverables keep the current activation behavior
- runtime tools should clearly identify that they are runtime-registered and usually operator-driven via `bash`

Recommended metadata additions for runtime entries:

- `source: runtime_tools`
- `manifest_path`
- `operator`

Search ranking should treat built-in and runtime discoverables uniformly.

## Gateway And Core Streaming Changes

Approval requires a new stream event type across core, gateway, and UI.

Add a new core event in [src/core/agent_loop.py](/Users/scott/Documents/Projects/Python/Jarvis/src/core/agent_loop.py):

- `AgentApprovalRequestEvent`

Extend:

- [src/core/__init__.py](/Users/scott/Documents/Projects/Python/Jarvis/src/core/__init__.py)
- [src/gateway/protocol.py](/Users/scott/Documents/Projects/Python/Jarvis/src/gateway/protocol.py)
- [src/gateway/app.py](/Users/scott/Documents/Projects/Python/Jarvis/src/gateway/app.py)
- [src/ui/telegram/gateway_client.py](/Users/scott/Documents/Projects/Python/Jarvis/src/ui/telegram/gateway_client.py)

New gateway protocol events:

- server-to-client: `approval_request`
- client-to-server: `approval_response`
- optional server-to-client acknowledgement: `approval_ack`

`approval_request` payload should include:

- `approval_id`
- `kind`
- `summary`
- `details`
- `command`
- `tool_name`
- `inspection_url`

`approval_response` payload should include:

- `approval_id`
- `approved`

## Agent Loop Integration

The approval lifecycle should live in the agent loop, not only in the UI.

Add an approval manager inside [src/core/agent_loop.py](/Users/scott/Documents/Projects/Python/Jarvis/src/core/agent_loop.py) or as a small dedicated helper module.

Responsibilities:

- create pending approval waiters keyed by route/session/turn
- expose a method to resolve approval from the gateway route
- suspend the active turn while waiting
- resume execution if approved by replaying the exact approved action
- stop waiting cleanly if rejected
- ensure stale approvals cannot be consumed by later unrelated actions

Recommended public route-level method additions:

- `SessionRouter.resolve_approval(route_id, approval_id, approved)`
- `AgentLoop.resolve_approval(approval_id, approved)`

If the user rejects:

- mark the approval rejected in session state
- clear the pending approval
- end the waiting section of the turn without executing the action
- leave the system idle until a new user message arrives

## Telegram UI Changes

The current Telegram client only polls `"message"` updates in [src/ui/telegram/api.py](/Users/scott/Documents/Projects/Python/Jarvis/src/ui/telegram/api.py). It must be extended to support inline keyboard approval UX.

Required Telegram additions:

- poll `callback_query` updates
- send messages with `reply_markup.inline_keyboard`
- answer callback queries so Telegram stops the loading spinner
- edit or follow up on the approval message after approval/rejection

Primary files:

- [src/ui/telegram/api.py](/Users/scott/Documents/Projects/Python/Jarvis/src/ui/telegram/api.py)
- [src/ui/telegram/bot.py](/Users/scott/Documents/Projects/Python/Jarvis/src/ui/telegram/bot.py)

Planned UI behavior:

- on `approval_request`, send a Telegram message containing:
  - the agent-written rationale
  - the exact command
  - optional inspection URL
  - `Approve` and `Reject` inline buttons
- on button press, send `approval_response` to the gateway
- on approval, edit the approval message to show approved state
- on rejection, edit the approval message to show rejected state

The inline button callback payload should be compact and encode:

- approval id
- approval decision

Route resolution should be derived from the Telegram chat/message context rather than duplicated inside the callback payload.

## Policy Changes

Update [src/tools/policy.py](/Users/scott/Documents/Projects/Python/Jarvis/src/tools/policy.py) to support:

- `tool_register`
- revised `bash` approval checks

Add per-tool policy packages for:

- `src/tools/basic/tool_register/policy.py`

`tool_register` policy should require a valid approval token every time.

## Storage And Transcript Changes

Session storage changes are needed for durable approval state and auditability.

Update:

- [src/storage/types.py](/Users/scott/Documents/Projects/Python/Jarvis/src/storage/types.py)
- [src/storage/service.py](/Users/scott/Documents/Projects/Python/Jarvis/src/storage/service.py)
- approval-related record creation in [src/core/agent_loop.py](/Users/scott/Documents/Projects/Python/Jarvis/src/core/agent_loop.py)

Transcript expectations:

- approval requests are recorded
- approval decisions are recorded
- consumed approval ids are recorded
- rejected actions are clearly visible in the transcript

## Tooling And Docs Updates

Update:

- [dev_docs/tool_dev_doc.md](/Users/scott/Documents/Projects/Python/Jarvis/dev_docs/tool_dev_doc.md)
- [src/identities/PROGRAM.md](/Users/scott/Documents/Projects/Python/Jarvis/src/identities/PROGRAM.md)
- optionally [notes/notes.md](/Users/scott/Documents/Projects/Python/Jarvis/notes/notes.md) if implementation decisions become relevant memory for later agents

Documentation changes should cover:

- runtime tool manifest contract
- `tool_register`
- tool-enforced approval flow for `bash` and `tool_register`
- dynamic runtime discoverable loading
- revised `bash` sandbox and approval behavior
- `BASH_DANGEROUSLY_SKIP_PERMISSION`
- Telegram approval UX

`PROGRAM.md` specifically should be updated in the `Tool Uses` section so the agent is explicitly aware of:

- the runtime tool protocol
- the fact that runtime tools are surfaced dynamically through `tool_search`
- when to register a newly installed or created tool
- that install/build/register actions may trigger approval UI
- that runtime tools are usually used through existing tools such as `bash` after discovery

## Test Plan

Add or update tests in:

- `tests/test_tools.py`
- `tests/test_agent_loop_tools.py`
- `tests/test_gateway_app.py`
- `tests/test_gateway_session_router.py`
- `tests/test_ui_gateway_client.py`
- `tests/test_ui_telegram_api.py`
- `tests/test_ui_telegram_bot.py`
- `tests/test_storage_service.py`

Required coverage:

- runtime manifest validation accepts valid manifests
- invalid runtime manifests are ignored safely and reported cleanly
- `tool_search` merges built-in and runtime discoverables dynamically
- runtime tool appears immediately after `tool_register` without restart
- `tool_register` denies registration without approval
- `tool_register` accepts exact approved manifest payload
- `bash` creates pending approval state when the detector fires
- `tool_register` creates pending approval state when registration approval is missing
- gateway forwards approval requests
- Telegram UI renders inline approve/reject buttons
- Telegram callback responses reach the gateway and resolve approval state
- approved turns resume and execute the exact approved action
- rejected turns stop waiting and do not execute the action
- `bash` denies obvious install/build commands without approval
- `bash` denies mismatched approved-command tokens
- `bash` moderate-aggression detector does not trigger on common non-mutating shell commands
- `BASH_DANGEROUSLY_SKIP_PERMISSION=True` bypasses the approval detector
- `bash` still blocks `/repo`
- `bash` still blocks `/run/secrets`
- `bash` still allows broad access elsewhere

## Implementation Order

Recommended build order:

1. Add manifest models and runtime tool loader under `src/tools/`.
2. Add `tool_register` basic tool with manifest validation and atomic file writes.
3. Update `tool_search` to merge runtime tool manifests dynamically.
4. Rework `bash` sandbox from workspace-only to blacklist-based path hiding.
5. Add `BASH_DANGEROUSLY_SKIP_PERMISSION` to settings and tool config.
6. Add approval state models and storage persistence.
7. Build the internal approval subsystem and exact-action binding helpers.
8. Add approval waiting/resolution support in the agent loop and session router.
9. Extend gateway protocol and websocket handling for approval request/response events.
10. Extend Telegram API client and bot for inline keyboard approvals and callback queries.
11. Tighten `bash` approval enforcement for moderately aggressive install/build/system-mutation detection.
12. Enforce approval inside `tool_register` for exact manifest payloads.
13. Update [src/identities/PROGRAM.md](/Users/scott/Documents/Projects/Python/Jarvis/src/identities/PROGRAM.md) `Tool Uses` guidance so the agent can discover, register, and use runtime tools correctly.
14. Update docs and complete end-to-end tests.

## Acceptance Criteria

This work is complete when all of the following are true:

- the agent can install or build a new tool through approval-gated `bash`
- the agent can register the tool through approval-gated `tool_register`
- the runtime tool becomes searchable immediately through `tool_search`
- no restart is required for discovery
- runtime tool manifests under `/workspace/runtime_tools/` are sufficient to reconstruct the tool capability on a fresh agent setup
- `bash` itself enforces approval for moderate-risk install/build/system-mutation commands unless `BASH_DANGEROUSLY_SKIP_PERMISSION=True`
- `bash` cannot access `/repo`
- `bash` cannot access `/run/secrets`
- the Telegram UI shows inline approve/reject controls for install/build/register approvals
- rejection leaves the system idle until the next user message
