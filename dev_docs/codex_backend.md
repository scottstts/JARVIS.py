# Codex Backend Plan

## Status

This document is the implementation and maintenance source of truth for integrating OpenAI Codex app-server into Jarvis.

As of April 7, 2026, this backend is planned but not yet implemented.

After implementation lands, keep this file updated so it reflects the code exactly. Do not leave stale future-tense design notes here.

## Purpose

Jarvis already supports normal LLM providers through `src/jarvis/llm/`.

Codex app-server is different:

- it owns authentication
- it owns thread and turn state
- it streams rich item events
- it can request approvals
- it can invoke experimental client-executed dynamic tools

Because of that, Codex must not be forced into the existing `LLMProvider` adapter contract.

The design here is:

- `codex` remains a selectable provider in user-facing settings
- internally, selecting `codex` routes that actor onto a separate backend
- the same Codex backend implementation can power the main agent and subagents
- all Codex-specific state, auth, transport, path translation, and protocol handling stay isolated in that backend
- the existing `src/jarvis/llm/` stack remains the implementation for normal providers only

## Scope

This plan covers:

- main-agent support for provider `codex`
- subagent support for provider `codex`
- mixed runtime configurations where some actors use Codex and others use normal providers
- host-side Codex app-server connectivity over WebSocket
- first-run ChatGPT OAuth browser login through Codex app-server
- route/session persistence needed to resume Codex threads for main and subagent sessions
- mapping Jarvis tools into Codex `dynamicTools`
- path translation between container paths and host paths
- gateway/UI changes required to surface browser login and Codex approval state
- tests
- follow-up docs and notes updates

This plan does not cover:

- embeddings through Codex
- memory maintenance/reflection through Codex
- Codex native shell/file-change/computer-use execution as a first-class Jarvis path
- running Codex app-server inside `jarvis_runtime`

## Resolved Design Decisions

These are the settled decisions for this backend.

1. `codex` is a settings-level provider name, but not an `LLMProvider`.
2. Codex integration lives outside `src/jarvis/llm/`.
3. Codex app-server runs on the host machine, not in `jarvis_runtime`.
4. Jarvis connects to Codex app-server over WebSocket JSON-RPC.
5. Jarvis uses ChatGPT browser login through `account/login/start { "type": "chatgpt" }` by default.
6. Jarvis does not own or persist ChatGPT OAuth tokens; the host app-server owns auth lifecycle.
7. Browser login is user-driven: Jarvis surfaces the returned `authUrl`, and the user opens it in a browser.
8. Codex `dynamicTools` are the integration point for Jarvis tools.
9. Codex experimental API must be enabled because `dynamicTools` require it.
10. Codex-specific quirks must not leak into `src/jarvis/llm/`, the main agent loop logic, or the subagent manager logic except through explicit backend interfaces.
11. Host/container path translation is mandatory and must stay fully inside the Codex backend.
12. Jarvis transcript remains the durable app-visible history, but Codex thread continuity depends on persisted backend state, not transcript replay.
13. `/new` and compaction intentionally start fresh Codex threads instead of trying to mirror old thread state forward.
14. Memory embeddings and maintenance continue to use the existing non-Codex provider stack.
15. Subagents may use `codex` as their provider.
16. The Codex backend core should be actor-scoped and reusable across main and subagent actors, with the differences injected as actor identity, bootstrap, tool filtering, and memory mode.
17. One route-scoped Codex client connection may multiplex the main actor and any Codex-backed subagents by thread id.
18. Codex native command/file-change approvals are not the primary Jarvis tool path; Jarvis tool approvals remain authoritative for Jarvis tools.

## Why This Is A Separate Backend

The current `LLMProvider` contract in `src/jarvis/llm/protocols.py` assumes:

- Jarvis sends normalized message history
- the provider returns normalized text/tool calls
- Jarvis owns the tool round loop

Codex app-server does not fit that model. It is a richer agent protocol:

- `thread/start`, `thread/resume`, `turn/start`, `turn/steer`, `turn/interrupt`
- item streaming instead of plain text/tool-call streaming
- server-initiated approval requests
- server-initiated `item/tool/call` requests for `dynamicTools`

That is backend behavior, not provider-adapter behavior.

Trying to stuff this into `src/jarvis/llm/providers/` would either:

- leak Codex thread/auth/tool semantics into the core loop, or
- cripple Codex into a text-only path and lose the point of using Codex

## Architecture Summary

Keep the existing normal-provider path:

- `RouteRuntime` + `AgentLoop` + `LLMService`

Add a Codex backend path built around a shared route-scoped connection plus reusable actor runtimes:

- `CodexRouteCoordinator`
- `CodexClient`
- `CodexActorRuntime`

Recommended package:

- `src/jarvis/codex_backend/`

Recommended files:

- `src/jarvis/codex_backend/__init__.py`
- `src/jarvis/codex_backend/config.py`
- `src/jarvis/codex_backend/types.py`
- `src/jarvis/codex_backend/path_mapping.py`
- `src/jarvis/codex_backend/client.py`
- `src/jarvis/codex_backend/auth.py`
- `src/jarvis/codex_backend/tool_bridge.py`
- `src/jarvis/codex_backend/actor_runtime.py`
- `src/jarvis/codex_backend/runtime.py`

Recommended responsibilities:

- `CodexClient`
  - one route-scoped WebSocket JSON-RPC connection
  - `initialize` and `initialized`
  - request id tracking
  - incoming event dispatch
  - reconnect and overload retry handling
- `CodexRouteCoordinator`
  - owns the shared `CodexClient`
  - owns auth gating for the route
  - dispatches incoming events and server requests to actor runtimes by `threadId` and `turnId`
  - surfaces route-level auth events
- `CodexActorRuntime`
  - powers one actor: main or subagent
  - owns thread start/resume, turn orchestration, transcript persistence, and actor-specific event mapping
  - injects actor-specific bootstrap, tool set, and memory mode

This design supports all of these:

- main = Codex, subagents = normal providers
- main = normal provider, subagents = Codex
- main = Codex, some subagents = Codex, some subagents = normal providers

## Main And Subagent Reuse

The main agent and subagents are close enough that the Codex backend should be built once and parameterized per actor.

Actor-specific inputs:

- actor kind: `main` or `subagent`
- actor name: `Jarvis` or subagent codename
- actor session storage
- bootstrap loader
- tool registry view
- tool executor
- memory mode
- route-event labels

For the main actor:

- use the main identity/bootstrap path
- allow the full main-agent tool set
- keep current main-agent route event semantics

For a subagent actor:

- use the subagent bootstrap loader
- apply current subagent built-in tool blocklist
- keep memory bootstrap, maintenance, and reflection disabled
- keep current subagent route event semantics and lifecycle notices

The goal is not a large generic abstraction hierarchy. The goal is one Codex actor runtime implementation with a small number of explicit injected collaborators.

## Backend Selection

Provider selection remains user-facing.

Internally, add a small actor-level routing helper that maps provider names to backend kinds:

- `openai`, `anthropic`, `gemini`, `grok`, `openrouter`, `lmstudio` -> existing LLM backend
- `codex` -> Codex backend

This selection applies independently to:

- the main agent provider
- each subagent provider resolution

Do not create a large generic abstraction tree.

Keep the routing minimal:

- one selector
- one actor runtime factory
- two actor runtime implementations

Implication for current code:

- the main route runtime must choose between normal `AgentLoop` and `CodexActorRuntime`
- `SubagentManager._build_subagent_loop()` must become a backend-aware actor builder instead of always constructing `AgentLoop`

## Host-Side Codex App-Server

Jarvis should connect to a host-run Codex app-server over WebSocket.

Relevant current OpenAI docs checked on April 7, 2026:

- app-server supports `stdio` and experimental `websocket` transports
- WebSocket transport is started with `codex app-server --listen ws://127.0.0.1:4500`
- clients must send `initialize`, then `initialized`
- `dynamicTools` require `capabilities.experimentalApi = true`

Recommended runtime assumption:

- the operator starts app-server manually on the host
- Jarvis connects to a configured WebSocket URL such as `ws://host.docker.internal:4500`

Jarvis should not try to spawn the host process from inside the container.

## Shared Route Connection

Use one Codex app-server connection per Jarvis route, not one connection per actor.

Reasons:

- auth state is shared anyway
- one connection can host multiple threads
- app-server server requests and notifications already include `threadId` and `turnId`
- route-level approval and event routing are already central concepts in Jarvis

Route-level responsibilities:

- initialize once
- authenticate once
- multiplex main and subagent Codex threads
- serialize auth-sensitive startup
- dispatch incoming messages to the correct actor runtime

If a route has no Codex-backed actors, it should not create a Codex client.

## Critical Constraint: Host vs Container Paths

This is the most important non-obvious constraint.

Jarvis runs inside `jarvis_runtime`.
Codex app-server runs on the host.

That means:

- Jarvis sees repo paths like `/repo/...`
- Jarvis sees shared workspace paths like `/workspace/...`
- Codex app-server needs host-visible absolute paths like `/Users/.../Jarvis/...` and `/Users/.../.jarvis/workspace/...`

This affects:

- Codex `cwd`
- `localImage` input items
- skill paths if used later
- any approval/file path UI copied from Codex
- main and subagent actors equally

Add a dedicated path translation layer under `src/jarvis/codex_backend/path_mapping.py`.

Required mappings:

- container repo root `<->` host repo root
- container workspace root `<->` host workspace root

Required behavior:

- fail fast on unmappable paths
- never leak path-mapping logic into core loop, tools, subagent manager, or gateway
- keep user-facing paths normalized to Jarvis conventions where practical

## Settings Surface

User-facing settings should still present `codex` as a provider choice for:

- the main chat provider
- the subagent provider override

Recommended `settings.yml` additions under `providers.codex`:

- `ws_url`
- `model`
- `effort`
- `summary`
- `personality`
- `service_name`
- `host_repo_root`
- `host_workspace_root`
- `approval_policy`
- `sandbox_network_access`

Recommended semantics:

- `llm.default_provider` may be `codex`
- `subagent.provider` may be `codex`
- `providers.codex.model` is the default model for Codex-backed actors
- `host_repo_root` is the host absolute path matching container `/repo`
- `host_workspace_root` is the host absolute path matching container `/workspace`

Recommended constraints:

- do not include `codex` in embedding provider settings
- do not include `codex` in memory maintenance provider settings

## Authentication Flow

Use the official app-server ChatGPT browser flow.

Route-level flow:

1. A Codex-backed actor requests work.
2. The route coordinator connects and initializes the app-server connection if needed.
3. The route coordinator calls `account/read`.
4. If authenticated, continue immediately.
5. If `requiresOpenaiAuth` is true and there is no active account, call `account/login/start` with `{ "type": "chatgpt" }`.
6. Receive `{ loginId, authUrl }`.
7. Emit a structured route event so the UI can display the browser login URL.
8. Wait for `account/login/completed`.
9. On success, release the waiting actor start requests.
10. On failure or cancellation, surface a clear route error and leave no half-started actor-turn state behind.

Recommended behavior:

- only one login flow may be in flight per route connection
- if several actors are waiting on auth, queue them behind the same auth flow
- do not try to open the browser automatically from the container

Recommended UI addition:

- a new route event for external auth, not just raw text

Recommended event shape:

- `auth_kind`
- `provider`
- `login_id`
- `auth_url`
- `message`
- optional `waiting_agents`

## Codex Session Model

Jarvis session and Codex thread are related but not identical.

Persist Codex backend state in session metadata.

Recommended `SessionMetadata` addition:

- `backend_state: dict[str, Any] = {}`

For Codex-backed sessions, `backend_state` should contain at least:

- `backend_kind = "codex"`
- `thread_id`
- `last_turn_id` if useful
- `auth_mode` snapshot if useful

This applies to:

- main sessions in the main transcript store
- subagent sessions in subagent transcript storage

Session behavior:

- new Jarvis main session -> new Codex thread
- resumed Jarvis main session -> `thread/resume` with stored `thread_id`
- new subagent session -> new Codex thread
- resumed subagent session -> `thread/resume` with stored `thread_id`
- subagent `step in` continues on the same thread after the active turn settles
- subagent dispose archives its session and stops tracking the thread
- `/new` archives the old main session, disposes active subagents, and starts a fresh main Codex thread
- compaction boundary -> start a fresh Codex thread for the post-compaction session

Do not try to reconstruct Codex thread state by replaying the Jarvis transcript.

This is a deliberate exception to the transcript-replay rules because Codex is a backend, not an `LLMProvider`.

## Turn Construction

Each Codex-backed actor turn should be built from:

- the current actor input
- any current-turn image inputs translated to Codex `localImage`
- actor bootstrap guidance translated into Codex `settings.developer_instructions`
- actor-filtered dynamic tools derived from the current Jarvis tool registry

Recommended mapping:

- bootstrap instructions -> Codex `turn/start.settings.developer_instructions`
- user or actor text -> Codex `input: [{ "type": "text", ... }]`
- current-turn image attachments -> Codex `input: [{ "type": "localImage", "path": ... }]`
- provider config -> `model`, `effort`, `summary`, `personality`
- host repo root mapping -> `cwd`

For the main actor:

- use the main identity/bootstrap loader
- include the main-agent current-turn runtime messages as needed

For a subagent actor:

- use the subagent bootstrap loader
- keep memory disabled exactly like the current subagent `AgentLoop` path
- reuse current subagent task assignment and runtime-note behavior

Do not translate archived Jarvis transcript history into per-turn Codex text input for either main or subagent actors.

Codex thread state is the continuity mechanism.

## Tool Bridge

Use Codex `dynamicTools` as the bridge for Jarvis tools.

Recommended behavior:

- build dynamic tool definitions from the same `ToolRegistry` and `ToolDefinition` source used by the current actor
- keep existing Jarvis tool descriptions, schemas, and policy checks authoritative
- execute tool calls inside Jarvis, not inside Codex
- preserve actor-specific tool filtering

Flow:

1. `thread/start` or `thread/resume`
2. `turn/start` with actor-specific `dynamicTools`
3. Codex emits `item/tool/call`
4. Jarvis executes the corresponding tool through the existing tool runtime path
5. Jarvis responds to the server request with tool output content items
6. Codex continues the same turn

This keeps:

- Jarvis tool policies
- Jarvis remote runtime split
- Jarvis approval prompts for Jarvis tools
- Jarvis workspace restrictions
- subagent built-in tool blocklists

inside Jarvis.

### Tool approvals

When a Jarvis tool execution returns `approval_required`, the Codex backend must:

1. create the normal Jarvis route approval event for the correct actor
2. wait for the user response
3. if approved, re-run the tool with the approved action context
4. respond to Codex with the final tool result
5. if rejected, return a structured tool error result to Codex

Do not bypass Jarvis approval rules by delegating those approvals to Codex.

### Tool result content

Support at least:

- plain text tool output
- tool error text
- image attachments from `view_image` translated into Codex `localImage`

If a Jarvis tool returns metadata that has no Codex equivalent, keep that metadata local to Jarvis and only send the model-visible content items to Codex.

## Event Mapping

Codex actor runtimes must translate Codex item and turn events into the same route event family the gateway/UI already uses where possible.

Core mappings:

- `item/agentMessage/delta` -> `RouteAssistantDeltaEvent`
- final agent message item -> `RouteAssistantMessageEvent`
- turn start -> `RouteTurnStartedEvent`
- turn completion -> `RouteTurnDoneEvent`
- Jarvis-tool invocation via dynamic tool -> `RouteToolCallEvent`
- Jarvis-tool approval -> existing route approval event
- browser login required -> new auth route event

These route events must remain actor-aware:

- main actor uses `agent_kind="main"` and `agent_name="Jarvis"`
- subagent actor uses `agent_kind="subagent"` and its codename

Unexpected Codex-native items:

- `commandExecution`
- `fileChange`
- `mcpToolCall`
- `collabToolCall`

should not silently pass through as if Jarvis owned them.

If they appear unexpectedly:

- surface a clear error
- record it in logs
- keep the failure isolated to the Codex backend

## Subagent-Specific Behavior

Codex-backed subagents must preserve the current subagent product behavior.

Required behavior:

- subagent invocation still comes only from the main agent
- nested subagents remain forbidden
- route-level `/stop` must interrupt active Codex-backed subagent turns too
- `step in` must interrupt the active Codex subagent turn, wait for it to settle, then start a new turn on the same Codex thread with the updated tasking
- subagent lifecycle notices remain route events, not special Codex-only UI messages
- subagent archive linkage to the owning main session and subagent id remains unchanged

Do not add a second subagent product with different semantics just because the backend is Codex.

## Persistence Rules

Persist normalized Jarvis-side records, not raw JSON-RPC traffic.

Persist:

- user messages
- assistant messages
- Jarvis tool call records
- Jarvis tool results
- Codex thread ids and other opaque backend data only in session `backend_state`

This applies to both:

- main sessions
- subagent sessions

Do not persist:

- browser auth URLs as transcript-visible records
- raw protocol envelopes
- transport-only notifications

If the Codex backend needs extra opaque state, keep it in `backend_state`.

## Codex Native Features

Codex has native concepts that overlap with Jarvis features:

- approvals
- command execution
- file changes
- skills
- apps/connectors
- collaboration tools

For this backend, do not adopt them by default unless the integration explicitly owns them.

Initial rule set:

- use Codex for reasoning, turn management, auth, and dynamic tool orchestration
- use Jarvis for tools, tool approvals, workspace policy, actor lifecycle, and UI event shaping

That keeps one primary tool system instead of two overlapping ones.

## Recommended File-Level Changes

### New backend package

- `src/jarvis/codex_backend/config.py`
  - settings model for Codex backend
- `src/jarvis/codex_backend/path_mapping.py`
  - container-to-host path translation
- `src/jarvis/codex_backend/client.py`
  - route-scoped WebSocket JSON-RPC client
  - request id tracking
  - initialize and initialized handshake
  - reconnect and overload retry handling
- `src/jarvis/codex_backend/auth.py`
  - `account/read`
  - login start/completion/cancel orchestration
- `src/jarvis/codex_backend/tool_bridge.py`
  - actor-specific dynamic tool definitions
  - request/response handling for `item/tool/call`
- `src/jarvis/codex_backend/actor_runtime.py`
  - reusable main/subagent actor runtime
  - actor session orchestration
  - event translation
  - backend-state persistence
- `src/jarvis/codex_backend/runtime.py`
  - route-scoped coordinator around the shared client
  - auth gating
  - actor registration and dispatch

### Existing files to change

- `src/jarvis/settings.yml`
  - add `codex` provider settings
  - add `codex` to main provider choices
  - ensure subagent provider choices allow `codex`
- `src/jarvis/settings.py`
  - export new Codex settings
- `src/jarvis/main.py`
  - log Codex backend targets correctly for main and subagent provider resolution
- `src/jarvis/gateway/app.py`
  - route runtime factory must be backend-aware
- `src/jarvis/gateway/session_router.py`
  - if needed for runtime factory wiring
- `src/jarvis/gateway/route_runtime.py`
  - integrate the Codex route coordinator and main-actor backend selection
- `src/jarvis/gateway/route_events.py`
  - add external-auth event
- `src/jarvis/storage/types.py`
  - add `backend_state`
- `src/jarvis/storage/service.py`
  - persist and update `backend_state`
- `src/jarvis/subagent/settings.py`
  - keep `codex` valid as a subagent provider
- `src/jarvis/subagent/manager.py`
  - build subagent actor runtime from backend selection instead of always using `AgentLoop`
- `dev_docs/project_structure.md`
- `notes/notes.md`

## Testing Plan

Add a fake Codex app-server test harness.

Do not rely on live OpenAI/Codex connectivity in automated tests.

Required test coverage:

1. settings parsing for Codex backend fields
2. provider-to-backend routing for main and subagent providers
3. path translation for repo and workspace paths
4. shared route-client initialization handshake
5. auth required -> login URL event -> login completed -> waiting actors resume
6. auth failure and cancel handling
7. new main session creates a new Codex thread
8. resumed main session uses `thread/resume`
9. new subagent session creates a new Codex thread
10. resumed subagent session uses `thread/resume`
11. subagent `step in` resumes on the same Codex thread after interruption
12. `/new` or disposal starts fresh threads and archives old backend state appropriately
13. dynamic tool registration shape for main and filtered subagent tool sets
14. dynamic tool invocation success path
15. dynamic tool invocation approval-required path
16. dynamic tool invocation rejection path
17. `view_image` tool result -> `localImage` mapping
18. mixed runtime configurations where only some actors use Codex
19. unexpected native Codex item types fail loudly
20. WebSocket overload / retryable error handling
21. backend-state persistence survives process restart for main and subagent sessions

Recommended new test files:

- `tests/test_codex_backend_config.py`
- `tests/test_codex_path_mapping.py`
- `tests/test_codex_client.py`
- `tests/test_codex_auth.py`
- `tests/test_codex_tool_bridge.py`
- `tests/test_codex_actor_runtime.py`
- `tests/test_codex_route_runtime.py`

Also update:

- `tests/test_llm_config.py`
- `tests/test_llm_service_lifecycle.py` only if provider lists or config assumptions change
- `tests/test_agent_loop.py` only where provider lists or skip logic need adjustment
- subagent manager tests and any gateway route-event tests impacted by the new auth event

## Implementation Sequence

Implement in this order:

1. Add settings and config parsing for Codex backend plus host-path mapping inputs.
2. Extend session metadata with `backend_state`.
3. Build the path translation layer.
4. Build the shared JSON-RPC WebSocket client and handshake logic.
5. Build the route-scoped coordinator and auth gating.
6. Build the reusable `CodexActorRuntime`.
7. Integrate the main actor with backend-aware selection.
8. Add the dynamic tool bridge.
9. Add tool approval bridging.
10. Add image-path translation for `localImage`.
11. Refactor `SubagentManager` to build subagent runtimes from backend selection and wire Codex-backed subagents.
12. Add mixed-backend route tests.
13. Update `dev_docs/project_structure.md` and append concise notes to `notes/notes.md`.

## Maintenance Notes

- If OpenAI stabilizes `dynamicTools` outside experimental API, remove the experimental opt-in requirement here and update the docs.
- If later work chooses to adopt Codex native command/file-change items, that must be a separate design update; do not silently mix them into this backend.
- If host-side path conventions change, update the path-mapping rules here first.
- If app-server transport semantics change, regenerate schemas from the installed Codex CLI and keep test fixtures aligned to that version.
- If subagent product semantics change, update the actor-runtime injection rules here rather than forking separate Codex logic paths for main and subagents.

## Sources

Official OpenAI docs checked on April 7, 2026:

- https://developers.openai.com/codex/app-server
- https://developers.openai.com/codex/auth

Key behaviors taken from those docs:

- app-server uses JSON-RPC over `stdio` or WebSocket
- WebSocket mode is started with `codex app-server --listen ws://IP:PORT`
- clients must send `initialize` then `initialized`
- `account/read`, `account/login/start`, `account/login/completed`, and `account/logout` drive auth state
- ChatGPT browser login returns an `authUrl` whose callback is hosted by app-server
- `dynamicTools` and related `item/tool/call` flow are experimental and require `experimentalApi`
- Codex turn inputs support `text`, remote `image`, and `localImage`
