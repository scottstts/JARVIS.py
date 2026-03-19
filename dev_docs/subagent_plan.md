# Subagent Implementation Plan

## Status

This document is the implementation handoff plan for the subagent system.

It captures the design choices agreed in discussion and should be treated as the implementation source of truth for this feature unless the user explicitly changes a decision later.

The goal is that a new Codex agent can pick up this file and implement the entire feature without having to re-open design questions that have already been settled.

## Scope

This plan covers:

- the `src/subagent/` subsystem
- the required `AgentLoop` refactors so subagents can reuse the existing loop
- main-agent-only subagent control primitives with tool-call semantics
- subagent starter context prompt files as runtime prompt assets
- the required `PROGRAM.md` update that teaches the main agent how to use subagents at a high level
- subagent storage under `workspace/archive/`
- tool filtering for subagents
- gateway and Telegram changes needed for background subagent events and approvals
- settings additions
- tests
- follow-up updates to related development docs after implementation lands

This plan does not cover:

- exposing subagents directly to the user
- nested subagents
- subagent memory bootstrap
- subagent access to memory tools
- a generic multi-user UI system beyond the current Telegram DM runtime

## Resolved Product Decisions

These are the settled design decisions and should be implemented exactly as follows.

1. Subagents are their own subsystem under `src/subagent/`, not a tool under `src/tools/`.
2. The main agent controls subagents through core runtime primitives that behave like tool calls, but these primitives are not part of the normal tool registry.
3. Only the Jarvis main agent may invoke subagents.
4. Nested subagents are forbidden. A subagent must not be able to invoke another subagent.
5. Maximum active subagents: `7`.
6. Active subagents use a random non-repeating codename from this pool:
   `Friday`, `Edith`, `Karen`, `Jocasta`, `Tadashi`, `Homer`, `Ultron`
7. Codenames must be unique among active subagents.
8. A codename may be reused only after the prior holder has been disposed.
9. A subagent uses the same underlying `AgentLoop` engine as the main agent.
10. A subagent does not reuse the starter files in `src/identities/`.
11. A subagent has its own starter context built from static prompt files under `src/subagent/` plus the main-agent-written task assignment.
12. General behavioral guidance about when Jarvis should use subagents belongs in `src/identities/PROGRAM.md`.
13. Detailed invocation and control usage docs for the subagent primitives belong in `src/subagent/` and are dynamically injected into the main agent bootstrap context.
14. Subagents must not receive memory bootstrap.
15. Subagents must not perform memory reflection.
16. Subagents must not have access to memory tools.
17. The initial built-in tool blocklist for subagents is:
   `memory_search`, `memory_get`, `memory_write`, `memory_admin`, `send_file`
18. Runtime tools registered through `workspace/runtime_tools/` remain available to subagents.
19. The tool filtering design must be scalable so future built-in tools can be blocked from subagents without reworking the architecture.
20. Subagent transcripts must be stored separately from main transcripts under `workspace/archive/`.
21. Subagent archives must still be linkable to the owning main session and turn in a foreign-key-style way even though storage remains file-based.
22. The main agent must be able to:
   invoke, monitor, stop, step in, and dispose subagents
23. `step in` means:
   cooperatively stop the subagent if it is currently running, wait for the turn to settle, then start a new subagent turn with updated direction
24. `/stop` from the user cooperatively stops the main agent and any active subagents.
25. The user does not directly converse with subagents.
26. Telegram should receive system messages when a subagent is invoked and when a subagent is disposed.
27. Telegram tool-use notices and approval notices must include the acting agent name.
28. The main agent display name for UI notices is `Jarvis`.
29. A subagent display name for UI notices is its codename.
30. If a subagent approval request is rejected by the user, that subagent pauses and emits state back to the main-agent-side runtime; the main agent later decides what to do with it.
31. The gateway and Telegram design should be unified now, not split into one transport for normal turns and another special transport for subagents.
32. The unified design should use a persistent duplex route websocket and a route-level event bus inside the gateway runtime.
33. Immediate user-visible subagent progress should stay minimal:
   spawn notice, dispose notice, agent-attributed tool use, and agent-attributed approval prompts
34. General subagent progress stays internal unless later surfaced by the main agent in its own turn.

## Core Design Summary

Implement a route-scoped supervisor runtime that owns:

- one main `AgentLoop`
- zero to seven subagent runtimes
- a shared tool execution coordinator
- a shared route event bus
- route-scoped approval resolution

Refactor the gateway from a one-request/one-stream websocket into a persistent route session. Telegram should keep one long-lived websocket per route and receive both main-turn events and background subagent events on the same connection.

Subagents should not have a separate bespoke reasoning engine. They should reuse the existing loop, but the loop must become configurable enough to support:

- different bootstrap builders
- different tool visibility
- memory disabled mode
- route-level approval and event integration

## High-Level Architecture

### 1. Route Runtime

Introduce a new route-scoped runtime object. Recommended file:

- `src/gateway/route_runtime.py`

Responsibilities:

- own the main `AgentLoop`
- own the `SubagentManager`
- expose route-level methods for:
  - enqueueing user messages
  - requesting main stop
  - resolving approvals
  - subscribing to outbound route events
- maintain a shared event bus for outbound websocket/UI events
- maintain a route-global tool execution coordinator
- map approval IDs to the correct target actor

Recommended supporting types:

- `RouteRuntime`
- `RouteEventBus`
- `RouteEvent`
- `RouteApprovalRegistry`

This route runtime should replace the current assumption in `SessionRouter` that a route is just one `AgentLoop`.

### 2. Main Agent Loop

The main agent continues to run through `AgentLoop`, but `AgentLoop` must be refactored so it can be configured rather than hardcoded as “the one Jarvis loop”.

Refactor goals:

- keep the current conversation/session/compaction logic
- make bootstrap injection configurable
- make tool exposure configurable
- make memory integration configurable
- preserve existing main-agent behavior as the default configuration

### 3. Subagent Manager

Introduce a dedicated manager under `src/subagent/` that owns all subagent lifecycle behavior.

Recommended file:

- `src/subagent/manager.py`

Responsibilities:

- allocate and release codenames
- enforce the max-active limit
- create child loops
- run child turns in background tasks
- translate child events into:
  - internal state updates
  - route UI events where applicable
  - main-agent-visible status snapshots
- expose the control actions behind the synthetic subagent primitives

### 4. Unified Route Websocket

Refactor the gateway and Telegram client to use one persistent websocket per route.

That websocket must support:

- client-to-server command frames
- server-to-client event frames

This is the unified transport for:

- normal main-agent streaming output
- main-agent tool and approval notices
- subagent tool and approval notices
- subagent spawn/dispose system notices

## Main Components To Add

Recommended new files:

- `src/subagent/__init__.py`
- `src/subagent/types.py`
- `src/subagent/manager.py`
- `src/subagent/runtime.py`
- `src/subagent/primitives.py`
- `src/subagent/bootstrap.py`
- `src/subagent/storage.py`
- `src/subagent/codenames.py`
- `src/subagent/prompts/SYSTEM.md`
- `src/subagent/prompts/OPERATING_RULES.md`
- `src/gateway/route_runtime.py`
- `src/gateway/route_events.py`

Potential additional helper files if useful:

- `src/subagent/settings.py`
- `src/subagent/status.py`

## Existing Files Expected To Change

- `src/core/agent_loop.py`
- `src/core/identities.py`
- `src/core/__init__.py`
- `src/gateway/session_router.py`
- `src/gateway/app.py`
- `src/gateway/protocol.py`
- `src/ui/telegram/gateway_client.py`
- `src/ui/telegram/bot.py`
- `src/ui/telegram/__init__.py`
- `src/tools/registry.py`
- `src/tools/types.py`
- `src/tools/basic/tool_search/tool.py`
- `src/settings.py`
- `src/identities/PROGRAM.md`
- `notes/notes.md`

Storage model changes may also touch:

- `src/storage/types.py`
- `src/storage/service.py`

Only change those if useful. Do not force subagent relations into the generic session store if a dedicated subagent catalog is cleaner.

## AgentLoop Refactor Plan

### Goal

Reuse `AgentLoop` for both main and subagent execution without duplicating the loop logic.

### Required Refactors

Refactor `AgentLoop` to accept injected collaborators or configuration for:

1. bootstrap message construction
2. tool definition construction
3. tool execution integration
4. memory bootstrap and reflection enablement
5. actor identity metadata
6. outbound event mapping hooks if needed

Recommended additions:

- an `agent_kind` field with values like `main` and `subagent`
- an `agent_name` field with values like `Jarvis` or the subagent codename
- a configurable bootstrap loader/builder
- a configurable tool definition provider or tool filter
- a configurable memory mode

### Main Agent Configuration

For the main agent:

- preserve current identity bootstrap from `PROGRAM.md`, `REACTOR.md`, `USER.md`, `ARMOR.md`
- preserve memory bootstrap and reflection
- dynamically append subagent primitive docs to the bootstrap context
- dynamically append subagent status snapshot to each new main-agent turn as transient runtime context

### Subagent Configuration

For a subagent:

- use static prompt files from `src/subagent/prompts/`
- append a generated task assignment written by the main agent when the subagent is invoked
- disable memory bootstrap
- disable memory reflection
- disable memory tools through tool filtering
- do not inject the subagent primitive docs

### No Nested Subagents

Enforce no nested subagents at two layers:

1. subagent primitive definitions are not exposed to subagents at all
2. the subagent manager rejects invocations from any non-main actor defensively

## Subagent Prompt Bootstrap

### Static Prompt Files

Recommended files:

- `src/subagent/prompts/SYSTEM.md`
- `src/subagent/prompts/OPERATING_RULES.md`

Important:

- these files are part of runtime behavior, not implementation documentation
- write them as prompt/instruction files for the agent to actually consume
- do not fill them with redundant explanatory prose meant for developers
- keep them concise, directive, and appropriate for model starter context
- when in doubt about tone and structure, inspect the existing prompt files in `src/identities/` and follow that style
- do not copy `src/identities/` content verbatim; write only what is appropriate for subagent behavior

These should describe:

- that the actor is a delegated subagent working for Jarvis
- that the user does not talk directly to it
- that it must stay narrowly scoped to the assigned task
- that it must not attempt to spawn subagents
- that it has no memory access
- that it should communicate concisely in a way suitable for runtime progress capture

### Dynamic Assignment Message

When `subagent_invoke` is executed, create a task-specific assignment message that includes:

- subagent codename
- stable `subagent_id`
- the main agent’s instruction text
- optional context
- optional deliverable/success criteria if included in the invoke arguments

The dynamic assignment should be injected as a developer or system message after the static subagent prompts.

### PROGRAM.md

Update `src/identities/PROGRAM.md` with high-level usage guidance only.

It should explain:

- Jarvis may use subagents for bounded side tasks
- Jarvis remains responsible for the final user-facing answer
- Jarvis should monitor and clean up subagents
- Jarvis should dispose subagents once finished

Important:

- this `PROGRAM.md` addition is part of runtime prompt behavior, not developer documentation
- write it as operating guidance for the main agent, in the same instruction style as the rest of `PROGRAM.md`
- do not add redundant implementation explanation or schema dumps there
- keep it to usage guidelines and decision heuristics only
- specific invocation details, argument patterns, and control semantics belong in the subagent primitive docs injected from `src/subagent/`, not in `PROGRAM.md`

## Subagent Primitive Design

Subagent control primitives are core-only synthetic tool definitions. They should not live in the normal `ToolRegistry`.

Recommended primitive names:

- `subagent_invoke`
- `subagent_monitor`
- `subagent_stop`
- `subagent_step_in`
- `subagent_dispose`

These definitions should be created in `src/subagent/primitives.py`.

The same module should also expose human-readable docs that can be injected into the main agent bootstrap, so the runtime docs stay in sync with the actual schemas.

### Recommended Schemas

#### `subagent_invoke`

Purpose:

- start a new subagent in the background with a fresh starter context and a main-agent-provided task

Recommended arguments:

- `instructions`: required string
- `context`: optional string
- `deliverable`: optional string

Recommended return content:

- `subagent_id`
- `codename`
- `status`
- `session_id`
- `active_count`

Behavior:

- allocate a unique codename
- create subagent archive/catalog entry
- create subagent loop
- start the subagent turn asynchronously
- emit a route `system_notice` for spawn

#### `subagent_monitor`

Purpose:

- inspect current subagent state without changing it

Recommended arguments:

- `agent`: optional string
  - accepts either codename or stable `subagent_id`
  - if omitted, return a summary of all non-disposed subagents
- `detail`: optional enum
  - `summary`
  - `full`

Recommended behavior:

- no side effects
- return current status, last activity, pause reason if any, and recent noteworthy events

#### `subagent_stop`

Purpose:

- request cooperative stop of a running subagent

Recommended arguments:

- `agent`: required string
- `reason`: optional string

Behavior:

- if running, request stop and move toward paused state when the turn settles
- if awaiting approval, cancel active waiting and pause the subagent
- if already paused/completed/failed, return a no-op result

#### `subagent_step_in`

Purpose:

- take over direction of a subagent that needs correction or further guidance

Recommended arguments:

- `agent`: required string
- `instructions`: required string

Behavior:

- if target is running, request cooperative stop
- wait until the target is no longer in the middle of a turn
- start a fresh turn on the same subagent loop with the new instructions
- preserve the same subagent identity and archive lineage

Important:

- `step in` is not mid-token prompt injection
- `step in` is stop, settle, then new turn

#### `subagent_dispose`

Purpose:

- permanently close and remove a subagent from the active set

Recommended arguments:

- `agent`: required string

Behavior:

- allowed only for non-running states
- release codename back to the pool
- mark subagent as disposed in the catalog
- emit a route `system_notice` for dispose

## Active-State Semantics

To keep the lifecycle explicit and force cleanup:

- any non-disposed subagent counts toward the max active limit
- this includes:
  - running
  - awaiting approval
  - paused
  - completed
  - failed

That means:

- a completed subagent still occupies a slot until explicit dispose
- a failed subagent still occupies a slot until explicit dispose

This matches the agreed “explicit dispose” lifecycle.

## Subagent Status Model

Recommended statuses:

- `running`
- `awaiting_approval`
- `paused`
- `completed`
- `failed`
- `disposed`

Recommended separate metadata fields:

- `pause_reason`
  - `main_stop`
  - `approval_rejected`
  - `gateway_disconnect_recovery` if ever needed later
- `last_error`
- `last_tool_name`
- `last_activity_at`
- `owner_main_session_id`
- `owner_main_turn_id`

Avoid overcomplicating the state machine with too many top-level statuses. Keep one stable set and use metadata for the reason fields.

## Tool Access Filtering

### Required Outcome

Subagents must:

- not see blocked built-in tools
- not be able to activate blocked built-in discoverable tools
- still see runtime tool manifests from `workspace/runtime_tools/`

### Recommended Design

Implement agent-scoped tool visibility rather than hardcoding a one-off subagent denylist inside `AgentLoop`.

Recommended approach:

1. Add access-scope metadata for built-in tools and built-in discoverable tools.
2. Add a subagent built-in blocklist in settings.
3. Build a filtered registry view for subagents.
4. Keep runtime manifest discoverables visible to subagents by default.

This lets future built-in tools be blocked by configuration rather than by rewriting core logic.

### Initial Subagent Blocklist

The initial built-in blocklist should be exactly:

- `memory_search`
- `memory_get`
- `memory_write`
- `memory_admin`
- `send_file`

Do not silently add more blocked tools unless a later design decision says to.

### `tool_search` Behavior

`tool_search` currently merges:

- built-in discoverables
- runtime manifest discoverables

For subagents, modify that merge so:

- blocked built-in discoverables are excluded
- runtime manifest discoverables remain included
- activation of matched backed built-ins only occurs if the built-in tool survives subagent filtering

### Runtime Tool Rule

Runtime tool manifests loaded from `workspace/runtime_tools/` remain visible to subagents by default.

That rule is intentional and should stay unless the user later changes it.

## Shared Tool Execution Coordinator

### Problem

Main and subagents must not race through tool execution against shared workspace state.

### Design

Add a route-global tool execution coordinator shared by the main loop and all subagent loops.

Recommended implementation:

- wrap actual tool execution behind an `asyncio.Semaphore(1)`

This should serialize tool execution across:

- main agent
- all subagents

LLM generation does not need the same serialization.

### Recommendation

Do not attempt fine-grained concurrency control in the first version. Route-global tool serialization is the correct first implementation.

## Subagent Storage Design

### Goals

- keep subagent transcripts separate from main transcripts
- preserve easy manual inspection
- preserve linkability to the owning main session and turn
- allow normal subagent session compaction chains

### Recommended Layout

Keep the current main transcript storage unchanged:

- `/workspace/archive/transcripts/<route_id>/...`

Add new subagent storage:

- `/workspace/archive/subagents/<route_id>/index.json`
- `/workspace/archive/subagents/<route_id>/<subagent_id>/sessions_index.json`
- `/workspace/archive/subagents/<route_id>/<subagent_id>/sessions/<session_id>.jsonl`

This means each subagent gets its own `SessionStorage` root.

### Route-Level Subagent Catalog

`index.json` should contain entries with:

- `subagent_id`
- `codename`
- `status`
- `created_at`
- `updated_at`
- `disposed_at`
- `route_id`
- `owner_main_session_id`
- `owner_main_turn_id`
- `current_subagent_session_id`
- `pause_reason`
- `last_error`

This is the foreign-key-style link layer for ad hoc inspection.

### Main Transcript Linking

When the main agent invokes or disposes a subagent, write a normal main-transcript record that includes:

- `subagent_id`
- `codename`
- the action taken

This creates a readable cross-reference in the main transcript without forcing all subagent data into the main session store.

### Subagent Session Compaction

Subagents should use ordinary `SessionStorage` behavior within their own storage root.

If a subagent compacts:

- the child session chain remains entirely within that subagent’s own archive root
- the route-level subagent catalog must update `current_subagent_session_id`

## Unified Gateway and Websocket Design

### Goal

Replace the current per-turn websocket RPC model with one persistent duplex route websocket.

This unified connection should be used for:

- user messages
- main-agent streaming output
- tool notices
- approval requests
- subagent spawn/dispose notices
- background subagent tool and approval notices

### Why

The current gateway app blocks inside one streamed turn after receiving a message.

That shape is not sufficient for:

- background subagent approvals
- asynchronous subagent notices
- persistent route supervision

### Required Gateway Refactor

Refactor the websocket endpoint so each client connection has:

1. a reader task
   - receives client frames
   - parses commands
   - forwards them into the route runtime
2. a writer task
   - subscribes to the route event bus
   - sends outbound events to the websocket

The route runtime becomes the orchestrator. The gateway should no longer directly loop over `AgentLoop.stream_user_input(...)` inside the websocket endpoint.

### Internal Route Event Bus

Introduce a route-level pub/sub event bus inside the route runtime.

This is not a “subagent-only side mechanism”.

It is the unified outbound event spine for the whole route.

### Outbound Event Types

Recommended route event set:

- `assistant_delta`
- `assistant_message`
- `turn_done`
- `tool_call`
- `approval_request`
- `system_notice`
- `error`

### Event Metadata

For relevant outbound events, include:

- `event_id`
- `created_at`
- `route_id`
- `session_id`
- `agent_kind`
- `agent_name`
- optional `subagent_id`

Notes:

- main agent uses `agent_kind=main`, `agent_name=Jarvis`
- subagent uses `agent_kind=subagent`, `agent_name=<codename>`

### UI Visibility Rules

Publish these subagent-originated events outward to the UI:

- `tool_call`
- `approval_request`
- `system_notice` for invoke/dispose

Do not publish these subagent-originated events outward to the UI by default:

- `assistant_delta`
- `assistant_message`
- `turn_done`

Those remain internal unless a later design decision changes it.

### Ready Event

Keep a `ready` event on connection open.

It may remain minimal:

- `route_id`
- `session_id`

Do not overengineer replay or durable event backfill in the first implementation.

Pending approvals already remain actionable because the Telegram approval message itself contains the durable `approval_id`.

## Approval Handling

### Current Problem

The current gateway resolves approvals against a single main loop per route.

That is insufficient once both main and subagents can await approval.

### Required Change

Move approval ownership to the route runtime.

When any actor emits an approval request:

- the route runtime registers `approval_id -> target actor`
- the route runtime publishes the approval request event

When the client sends `approval_response`:

- the route runtime looks up the target actor
- the route runtime forwards resolution to the correct loop instance

### Main-Agent Rejection Behavior

Keep current main-agent semantics unless a later design decision changes them.

That means if a main-agent tool approval is rejected:

- the main turn ends as it does today

### Subagent Rejection Behavior

If a subagent tool approval is rejected:

- the subagent leaves `awaiting_approval`
- the subagent enters `paused`
- `pause_reason` becomes `approval_rejected`
- the route runtime records a noteworthy internal status event
- the main agent can later inspect or intervene via `monitor` or `step_in`

## Main-Agent Awareness of Subagents

### Goal

The main agent should know what its subagents are doing without requiring the user to manage that manually.

### Design

The `SubagentManager` should maintain a concise internal activity log for each subagent and a route-level snapshot of all non-disposed subagents.

Before each main-agent turn, inject a transient system or developer runtime message summarizing:

- current active subagents
- state of each one
- noteworthy recent events
  - spawned
  - awaiting approval
  - approval rejected
  - paused
  - completed
  - failed

This snapshot should be appended to the main turn request similarly to the existing transient datetime turn context.

### Important Constraint

Do not implement strict main/subagent turn lockstep.

Correct behavior is:

- subagents run independently in background tasks
- the main agent learns about them at checkpoints
- `subagent_monitor` gives explicit current state on demand
- `subagent_step_in` gives a deliberate intervention path

## Telegram/UI Behavior

### Persistent Gateway Connection

Refactor the Telegram gateway client to maintain one persistent websocket per route instead of opening a new websocket for:

- every turn
- every stop request
- every approval response

Recommended new client abstraction:

- a route session client that can:
  - send command frames
  - receive event frames continuously

### Chat-Level Bridge Behavior

Per Telegram chat:

- maintain one persistent route session
- keep the current per-chat inbound message queue
- continue serializing user messages per chat
- process outbound route events continuously in the background

### Immediate Telegram Messages Required

Immediately send Telegram messages for:

- subagent invoked
- subagent disposed
- tool-use notice with agent attribution
- approval request with agent attribution

Do not send every internal subagent progress update to Telegram.

If a subagent is doing something noteworthy, the main agent may choose to surface it in a later normal answer.

### Tool Use Notice Format

Change the current notice format from:

- `Used <tool> tool.`

to:

- `Jarvis used <tool> tool.`
- `<codename> used <tool> tool.`

### Approval Notice Format

Approval messages should also identify the actor:

- `Jarvis requests approval...`
- `Friday requests approval...`

Continue editing the Telegram approval message in place when the user approves or rejects.

### User Visibility Rule

The user does not interact directly with subagents. All user-facing agent conversation remains through Jarvis.

## Settings Additions

Add subagent runtime settings to `src/settings.py`.

Recommended settings:

- `JARVIS_SUBAGENT_MAX_ACTIVE = 7`
- `JARVIS_SUBAGENT_CODENAME_POOL = ("Friday", "Edith", "Karen", "Jocasta", "Tadashi", "Homer", "Ultron")`
- `JARVIS_SUBAGENT_ARCHIVE_DIR = f"{AGENT_WORKSPACE}/archive/subagents"` when workspace is configured
- `JARVIS_SUBAGENT_BUILTIN_TOOL_BLOCKLIST = ("memory_search", "memory_get", "memory_write", "memory_admin", "send_file")`
- `JARVIS_SUBAGENT_MAIN_CONTEXT_EVENT_LIMIT`
  - small integer to bound how many recent noteworthy subagent events are injected into the main turn snapshot

If implementing separate settings dataclasses is useful, add them under `src/subagent/` or the relevant config module.

## Recommended Implementation Order

Implement in this order so the architecture lands cleanly.

### Phase 1: Route Runtime Skeleton

Create the route runtime layer first.

Tasks:

1. Add route runtime and route event types.
2. Make `SessionRouter` manage route runtimes instead of raw loops.
3. Keep the main-agent behavior working through the new route runtime.

Exit criteria:

- normal main-agent turns still work
- no subagent functionality yet

### Phase 2: Persistent Unified Websocket

Refactor gateway and Telegram transport next.

Tasks:

1. Change the gateway websocket endpoint to persistent reader/writer tasks.
2. Change the Telegram gateway client to maintain one persistent route session per chat.
3. Preserve existing main-agent streaming behavior over the unified connection.

Exit criteria:

- main-agent chat still works end to end
- `/stop` still cooperatively stops the main agent at assistant/tool-step boundaries
- active subagents are also cooperatively stopped and later settle into `paused`
- already-running subagent tool executions are still allowed to finish and log their results before pause
- the main transcript gets a persistent system note reminding Jarvis to inspect paused subagents after the user resumes
- approvals still work for the main agent

### Phase 3: AgentLoop Configurability

Refactor `AgentLoop` so it can power both actor types.

Tasks:

1. Extract or inject bootstrap loading.
2. Add actor identity metadata.
3. Add configurable memory enable/disable behavior.
4. Add configurable tool definition builder or filtered tool provider.

Exit criteria:

- main-agent behavior is unchanged
- a subagent-configured loop can be instantiated without using `src/identities/`

### Phase 4: Subagent Storage and Manager

Build the subagent catalog and lifecycle manager.

Tasks:

1. Add codename allocation and release.
2. Add subagent catalog persistence.
3. Add child loop construction.
4. Add background task running.
5. Add state tracking and internal noteworthy event tracking.

Exit criteria:

- manager can spawn and track subagents internally
- archive structure exists under `workspace/archive/subagents/`

### Phase 5: Subagent Primitives

Add the main-agent-only synthetic subagent primitives.

Tasks:

1. Implement primitive schemas and docs.
2. Inject them into the main-agent tool set.
3. Ensure they are absent from subagent tool sets.
4. Wire their execution into the route runtime / subagent manager.

Exit criteria:

- main agent can invoke, monitor, stop, step in, and dispose

### Phase 6: Tool Filtering

Add subagent tool visibility rules.

Tasks:

1. Implement built-in tool scope/filtering.
2. Add settings-backed built-in blocklist.
3. Update `tool_search` merging/activation logic for subagent context.
4. Confirm runtime manifests remain available to subagents.

Exit criteria:

- subagents cannot use the blocked memory tools
- runtime tools still show up to subagents

### Phase 7: UI Notices and Approval Routing

Wire the user-visible notices and background approval flow.

Tasks:

1. Add route event publication for invoke/dispose notices.
2. Add agent attribution to tool and approval events.
3. Route approval responses through the route runtime to the correct actor.
4. Pause subagents on rejection.

Exit criteria:

- Telegram shows:
  - invoke notice
  - dispose notice
  - attributed tool notices
  - attributed approval prompts

### Phase 8: Main-Agent Context Awareness

Add subagent state snapshot injection into main turns.

Tasks:

1. Render concise subagent status snapshot before each main turn.
2. Include recent noteworthy subagent events.
3. Keep the snapshot bounded and concise.

Exit criteria:

- Jarvis sees up-to-date subagent state in its next turn without the user managing that manually

### Phase 9: PROGRAM.md and Final Polish

Tasks:

1. Update `PROGRAM.md` with high-level subagent usage guidance.
2. Ensure all prompt and tool docs are consistent.
3. Verify storage, UI messages, and lifecycle semantics all match this plan.
4. Update related development docs that became stale because of this implementation.

Related doc reminder:

- after the implementation is complete, review the existing docs under `dev_docs/` and update any document that now describes outdated gateway, loop, tool-visibility, storage, or prompt-bootstrap behavior
- this is a post-implementation documentation sync task, not a substitute for the required runtime prompt files under `src/subagent/` or the required `PROGRAM.md` behavior update

## Testing Plan

Add automated tests for the following.

### Core Lifecycle

- main agent can invoke a subagent
- subagent gets unique codename
- invoking the eighth non-disposed subagent fails
- disposing a subagent releases its codename

### No Nested Subagents

- subagent tool definitions do not include subagent primitives
- subagent invocation attempt is rejected defensively if somehow reached

### Tool Filtering

- subagent cannot access `memory_search`
- subagent cannot access `memory_get`
- subagent cannot access `memory_write`
- subagent cannot activate `memory_admin`
- subagent cannot access `send_file`
- runtime manifest discoverables remain visible to subagents

### Storage

- subagent archives are written under `/workspace/archive/subagents/<route_id>/...`
- route-level subagent catalog contains owner main session/turn linkage
- main transcript contains invoke/dispose cross-reference records

### Control Primitives

- `subagent_monitor` returns summary for all when no target is provided
- `subagent_stop` pauses a running subagent
- `subagent_step_in` stops then starts a new turn with new instructions
- `subagent_dispose` is rejected for a running subagent

### Approval Handling

- main approval still resolves correctly
- subagent approval request routes to the correct child
- rejecting subagent approval pauses the child and does not stop the main route

### Gateway/UI

- persistent route websocket handles:
  - user message
  - stop
  - approval response
  - background subagent tool event
  - background subagent approval event
- Telegram tool notices include `Jarvis` or the codename
- Telegram spawn/dispose system notices are emitted

### Main-Agent Awareness

- main-turn runtime snapshot includes current subagent state
- snapshot stays bounded in size

## Implementation Notes

### Reuse Existing Logic Aggressively

Do not build a second reasoning loop. The existing `AgentLoop` should remain the single loop implementation and become configurable.

### Avoid Overloading `src/tools/`

Do not implement subagents as a normal basic or discoverable tool in `src/tools/`.

The control primitives should be synthetic core runtime definitions owned by `src/subagent/`.

### Prefer Generated Primitive Docs

The main-agent bootstrap docs for subagent primitives should be generated from the same definitions that power the synthetic tool schemas to avoid drift.

### Keep User Visibility Intentional

Do not suddenly surface subagent text streams to Telegram.

Only the agreed immediate notices should be user-visible.

## Done Criteria

This feature is done when all of the following are true:

1. Jarvis can invoke up to seven background subagents through main-agent-only synthetic primitives.
2. Subagents use the same core loop with different bootstrap/tool/memory configuration.
3. Functional subagent prompt files exist under `src/subagent/prompts/` and are written as real starter context, not as developer docs.
4. `PROGRAM.md` contains high-level subagent usage guidance for the main agent, but not detailed primitive schemas.
5. Subagents have their own archive storage under `workspace/archive/subagents/`.
6. Subagents cannot access memory tools.
7. Runtime manifest tools remain available to subagents.
8. No nested subagent path exists.
9. Gateway and Telegram use one persistent route websocket connection model.
10. Telegram immediately receives:
   - invoke notice
   - dispose notice
   - attributed tool notices
   - attributed approval requests
11. User `/stop` cooperatively affects the main agent and any active subagents, while still requiring a later user message before Jarvis resumes.
12. Rejected subagent approvals pause the subagent rather than collapsing the whole route.
13. Jarvis sees concise subagent state in later turns without the user manually managing it.
14. Related development docs have been updated where the implementation made them stale.

## Final Guidance For The Implementing Agent

If implementation details need to vary slightly, keep these external behaviors fixed:

- main-agent-only synthetic subagent control
- no subagent memory access
- no nested subagents
- separate subagent archive storage
- explicit dispose lifecycle
- unified persistent route websocket
- minimal but immediate Telegram notices for subagent lifecycle/tool/approval events

Do not re-open product decisions already settled in this document unless you hit a genuine contradiction in the codebase or a concrete blocker that cannot be resolved within the agreed design.
