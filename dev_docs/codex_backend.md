# Codex Backend

## Status

Implemented on April 7, 2026.

This document is the maintenance source of truth for Jarvis support for OpenAI Codex app-server.
It describes the code as it exists now, not a future plan.

## Purpose

Jarvis exposes `codex` as a user-facing provider choice in `settings.yml`, but `codex` is not implemented as an `LLMProvider`.

Codex app-server owns:

- auth
- thread and turn state
- streamed item events
- dynamic tool callbacks

That does not fit the normal `src/jarvis/llm/` contract, where Jarvis sends normalized history and receives normalized text/tool-call output.

Jarvis therefore routes `codex` actors onto a separate backend under `src/jarvis/codex_backend/`.

## What Is Implemented

- main-agent support for provider `codex`
- subagent support for provider `codex`
- mixed routes where some actors use Codex and others use the normal LLM backend
- host-side Codex app-server connectivity over WebSocket JSON-RPC
- optional websocket bearer-token auth for host-side Codex app-server listeners
- ChatGPT OAuth browser login through Codex app-server
- route/session persistence for Codex thread ids
- dynamic-tool bridging from Jarvis tools to Codex `item/tool/call`
- Codex auth-required route events and Telegram UI handling
- host/container path mapping for Codex-facing host paths

## What Is Intentionally Not Implemented

- Codex as an `LLMProvider`
- embeddings through Codex
- memory maintenance/reflection through Codex
- current-turn user/runtime image inputs for Codex turns
- Codex-native shell execution, file changes, MCP tools, or collaboration tools as Jarvis-owned execution paths
- running Codex app-server inside `jarvis_runtime`

Those are deliberate boundaries in the current implementation.

## Architecture

Normal providers still use:

- `RouteRuntime`
- `AgentLoop`
- `LLMService`

Codex-backed actors use:

- `CodexRouteCoordinator`
- `CodexClient`
- `CodexActorRuntime`

Shared selection lives in `src/jarvis/actor_backends.py`.

- `codex` -> Codex backend
- everything else -> normal LLM backend

This selection is applied independently to:

- the main actor
- each subagent

## Key Files

- `src/jarvis/actor_backends.py`
- `src/jarvis/codex_backend/config.py`
- `src/jarvis/codex_backend/path_mapping.py`
- `src/jarvis/codex_backend/client.py`
- `src/jarvis/codex_backend/auth.py`
- `src/jarvis/codex_backend/tool_bridge.py`
- `src/jarvis/codex_backend/runtime.py`
- `src/jarvis/codex_backend/actor_runtime.py`
- `src/jarvis/gateway/route_runtime.py`
- `src/jarvis/subagent/manager.py`
- `src/jarvis/gateway/route_events.py`
- `src/jarvis/gateway/protocol.py`
- `src/jarvis/ui/telegram/gateway_client.py`
- `src/jarvis/ui/telegram/bot.py`

## Settings Surface

`settings.yml` keeps `codex` as a selectable/defaultable model provider for:

- `main_agent_provider`
- `subagent_provider`

User-facing Codex model settings live under `providers.codex`:

- `model`
- `reasoning_effort`

Codex backend transport/runtime defaults such as websocket URL, reasoning summary, sandbox policy, host-path mapping, and service identity now live in `src/jarvis/codex_backend/config.py` and env/secret inputs, not in `settings.yml`.

The websocket bearer token is intentionally not stored in `settings.yml`.
If needed, Jarvis reads it only from the secret/env var `JARVIS_CODEX_WS_BEARER_TOKEN`.

Embeddings and memory-maintenance provider settings intentionally do not accept `codex`.

## LLMService Boundary

`LLMService` still owns only normal provider adapters.

To keep the architectural boundary explicit, `LLMService.generate()`, `stream_generate()`, and `embed()` now fail fast with `LLMConfigurationError` if they are asked to use provider `codex`.

That avoids leaking the backend split as a confusing `ProviderNotFoundError`.

## Host App-Server Model

Jarvis expects Codex app-server to run on the host.

Default assumption:

- Codex app-server listens at `ws://host.docker.internal:4500`
- Jarvis connects from inside `jarvis_runtime`
- when the host listener is non-loopback, Jarvis can send `Authorization: Bearer <token>` from `JARVIS_CODEX_WS_BEARER_TOKEN`

Jarvis does not try to spawn the host process itself.

The implementation was built against the official Codex app-server protocol as of April 7, 2026:

- `initialize` then `initialized`
- `account/read`
- `account/login/start`
- `account/login/completed`
- `thread/start`
- `thread/resume`
- `turn/start`
- `turn/interrupt`
- server-driven `item/tool/call`

## Route-Scoped Connection

Each Jarvis route owns at most one Codex connection through `CodexRouteCoordinator`.

That coordinator:

- lazily creates the WebSocket client
- initializes the protocol
- owns route-level auth state
- dispatches notifications and server requests by `threadId`
- multiplexes the main actor and any Codex-backed subagents over one connection

## Authentication Flow

Auth is route-scoped and serialized.

Flow:

1. a Codex-backed actor needs a thread
2. the coordinator ensures the client is connected
3. the authenticator calls `account/read`
4. if auth is missing, it calls `account/login/start { "type": "chatgpt" }`
5. Jarvis emits a structured `auth_required` route event with the returned `authUrl`
6. the UI shows the URL to the user
7. Jarvis waits for `account/login/completed`
8. after success, waiting actors continue

Only one login flow is allowed in flight per route connection.

Jarvis does not own the OAuth tokens. The host Codex app-server does.

## Session And Thread Model

Jarvis transcript state stays authoritative for Jarvis.
Codex thread state stays authoritative for Codex continuity.

At session creation, Codex-backed sessions persist starter context as transcript-only snapshots, not as replayable prompt-history records:

- one transcript-only snapshot of the full Codex `developerInstructions`
- one transcript-only snapshot of the initial Codex `dynamicTools`

Codex still consumes its bootstrap through `developerInstructions` on `thread/start` or `thread/resume`.
Jarvis does not reconstruct that thread state by replaying transcript back into Codex.

`SessionMetadata` now includes `backend_state`.

For Codex-backed sessions, `backend_state` stores:

- `backend_kind = "codex"`
- `thread_id`
- `last_turn_id`

Behavior:

- new Jarvis session -> new Codex thread
- resumed Jarvis session -> `thread/resume` with stored `thread_id`
- `/new` -> archive current session and start a fresh Codex thread
- `/compact` -> archive the current session, persist structured replacement-history items into a fresh session, and mark the next Codex turn to seed that new thread exactly once with the compacted handover history

Jarvis still does not rebuild Codex continuity by replaying the full old transcript into a new thread.
Instead, post-compaction Codex sessions persist compacted replacement-history records locally and inject them once into the first turn of the fresh thread.

## Main And Subagent Reuse

`CodexActorRuntime` is shared by:

- the main actor
- subagents

Injected actor-specific differences are:

- identity
- bootstrap loader
- storage
- tool registry view
- tool executor
- memory mode
- route-event labels

Main actor behavior:

- full main tool surface
- normal main bootstrap
- normal main memory mode

Subagent behavior:

- subagent bootstrap
- current subagent built-in tool blocklist
- no memory bootstrap
- no memory maintenance
- no memory reflection

Subagent disposal now calls `loop.aclose()` so Codex-backed subagents unregister their thread from the route coordinator instead of leaving stale thread mappings behind.

## Tool Bridge

Jarvis tools are exposed to Codex through dynamic tools built from the same `ToolDefinition` source used by the normal agent loop.

Flow:

1. `thread/start` or `thread/resume`
2. the backend computes the current Jarvis dynamic-tool set and a signature for it
3. `dynamicTools` are always sent on `thread/start`
4. `thread/resume` sends `dynamicTools` only when Jarvis needs to change the stored tool set or when older session metadata has no recorded signature yet
5. `turn/start` sends `dynamicTools` only when the tool signature has changed since the last applied thread/tool state
6. Codex sends `item/tool/call`
7. Jarvis executes the tool through the existing tool runtime
8. if approval is required, Jarvis keeps its own approval flow authoritative
9. Jarvis replies to Codex with model-visible tool output content items

The last applied dynamic-tool signature is persisted in session `backend_state` so resumed Codex sessions can avoid resending unchanged tool definitions.

Supported tool result content sent back to Codex:

- text
- error text
- image attachments from Jarvis tools, encoded as `data:` URLs in `inputImage`

Jarvis does not hand off tool approvals to Codex.

Current Codex turn inputs are text-only.

- user text is sent as Codex `text` input items
- runtime/pre-turn messages are sent as Codex `text` input items
- user/runtime image inputs are not yet translated into Codex `localImage`
- route-appended external system notes are not merely persisted: before each Codex turn, the backend also syncs any unsent external orchestrator system notes from transcript into Codex input so runtime follow-up turns see the same state that normal provider turns get from transcript replay

## Async Orchestrator Yield

Codex-backed actors do not keep one provider turn alive once work has moved into an orchestrator-managed async state.

If a Jarvis tool result starts async work that the route orchestrator is meant to supervise, the backend yields the turn back to `RouteRuntime`.

Current yield-producing tool results are:

- `bash` results where the job is still `running` in background mode or was promoted to background
- `subagent_invoke` / `subagent_step_in` results whose status is `running`, `waiting_background`, or `awaiting_approval`

Behavior:

1. Jarvis sends the normal tool response back to Codex
2. the backend immediately requests `turn/interrupt` for that provider turn
3. any later Codex deltas or assistant items from that yielded provider turn are ignored locally
4. if Codex races and sends another `item/tool/call` before the interrupt lands, Jarvis returns a backend-local rejection response instead of executing more work
5. when Codex reports the interrupted/completed turn, Jarvis emits a normal `done` event, not `codex_backend_error`

This is how Codex-backed actors rejoin the same orchestration model used by normal providers:

- the current turn ends cleanly
- `RouteRuntime` regains control
- later bash/subagent notices can enqueue new runtime follow-up turns

For main-agent subagent completion/finalize notices, the orchestrator progress message now includes the latest persisted subagent assistant report when available.
That lets Codex finalize against the child’s actual reported result instead of reopening the child with `subagent_monitor` or `subagent_step_in` just to recover the already-persisted output.

The backend does not currently force a synthetic assistant progress message before yielding.
If no assistant text was already produced, the user-visible output for that turn comes from the normal tool/system notice path rather than a final assistant message.

## Native Codex Execution Boundary

Jarvis does not allow Codex-native execution items to become first-class Jarvis execution paths.

If Codex emits unsupported native item types such as:

- `commandExecution`
- `fileChange`
- `mcpToolCall`
- `collabAgentToolCall`

or unsupported native server-request methods other than `item/tool/call`, the Codex actor runtime does not allow the native path to continue.

Behavior:

1. request `turn/interrupt` for the active Codex provider turn
2. append a corrective system note to the Jarvis transcript on the same logical turn
3. start one corrected retry turn on the same Codex thread
4. keep the outward Jarvis turn id stable across the retry

The corrective note explicitly tells Codex to continue with Jarvis dynamic tools only, and it includes the original user request when available.

If the corrected retry still attempts a native Codex capability, or if the retry cannot be started, the runtime then fails closed with `codex_backend_error`.

Once native-capability recovery is pending, further message deltas and completed agent-message items from that offending provider turn are ignored locally.

The developer instruction sent to Codex explicitly tells it to use only client-provided dynamic tools.
That soft suppression is advisory only; the interrupt-and-retry path is the enforcement layer.
Codex tool responses for orchestrator-managed async work also include Codex-only advisory text that tells the model the turn is yielding back to Jarvis, so it should not keep polling or calling more tools in that turn.

## Path Mapping

Codex runs on the host while Jarvis runs in `jarvis_runtime`, so host/container path translation is isolated in `CodexPathMapper`.

Supported roots:

- container `/repo` <-> configured host repo root
- container workspace root <-> configured host workspace root

Current active use:

- host `cwd` for Codex thread and turn requests

The reverse mapping is implemented and kept backend-local for future host-originating path surfaces, but current unsupported native Codex items mean it is not yet exercised in normal runtime behavior.

## Event Mapping

Codex backend events are normalized back into the existing route event family.

Key mappings:

- `item/agentMessage/delta` -> `RouteAssistantDeltaEvent`
- final text -> `RouteAssistantMessageEvent`
- turn start -> `RouteTurnStartedEvent`
- turn completion -> `RouteTurnDoneEvent`
- `item/tool/call` -> `RouteToolCallEvent`
- Jarvis approval request -> existing route approval event
- browser login challenge -> `RouteAuthRequiredEvent`

Telegram now renders `auth_required` as a dedicated HTML message with the login URL.

Persisted Codex assistant transcript records also include provider metadata in the same basic shape used by normal LLM-provider transcripts:

- `provider = "codex"`
- `model = <configured Codex model>`
- `response_id = <Codex provider turn id>`
- `finish_reason`

When one Codex turn emits multiple assistant message items, the backend persists them as separate assistant transcript records on the same Jarvis turn instead of collapsing them into one fused blob.

## Memory Behavior

Main Codex actors still use the existing Jarvis memory stack.

That means:

- bootstrap memory rendering still comes from `MemoryService`
- due maintenance still uses the configured maintenance provider
- reflection still uses the configured maintenance provider

Codex is not used for those maintenance/reflection model calls.

Subagents keep memory disabled exactly like the non-Codex subagent path.

## Operational Notes

- Codex app-server must already be running on the host before Jarvis can use provider `codex`.
- For the default Docker setup, a host app-server bound only to `127.0.0.1` is not reachable from `jarvis_runtime`; the listener must be exposed on a host-reachable interface or proxied accordingly.
- For non-loopback listeners using `codex app-server --ws-auth capability-token`, Jarvis expects the same bearer token in the secret/env var `JARVIS_CODEX_WS_BEARER_TOKEN`.
- `providers.codex.host_repo_root` and `providers.codex.host_workspace_root` are required when an actor actually uses Codex.
- `dynamicTools` are still an experimental Codex app-server surface.
- Tool-result images are sent back as data URLs because that is the current model-visible image form used by the backend bridge.
- `CodexPathMapper` currently has active runtime use only for host `cwd`; reverse path mapping and `localImage` host-path translation are present but not exercised by the current turn-input path.
- `CodexRouteCoordinator.aclose()` exists, but route runtime/session-router teardown does not currently call it because routes do not yet have an explicit disposal lifecycle. In practice, the Codex route connection lives as long as the route runtime object.
- Expected Codex transport and handshake failures are surfaced to the route layer as `codex_backend_error` rather than falling through to the generic `internal_error` path.

## Tests

Codex backend coverage lives primarily in:

- `tests/test_codex_backend_config.py`
- `tests/test_codex_client.py`
- `tests/test_codex_auth.py`
- `tests/test_codex_tool_bridge.py`
- `tests/test_codex_actor_runtime.py`
- `tests/test_gateway_protocol.py`

Broader regression coverage comes from the existing route, subagent, gateway, main-runtime, and full-suite tests.

## Future Work

These are not implemented today, but they are the natural expansion points if Codex ownership grows later:

- Codex-driven embeddings
- Codex as a maintenance/reflection provider
- Codex `localImage` support for user/runtime turn inputs
- Codex-native command/file/computer-use as explicit Jarvis execution paths
- explicit route-runtime teardown that closes the per-route Codex coordinator connection
- running app-server inside `jarvis_runtime`
