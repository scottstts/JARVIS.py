# User Message Interruption Design And Implementation Notes

## Status

This feature is implemented.

This document now serves as the design-and-handoff source of truth for route-level user message interruption unless the user explicitly changes a decision later.

Current implementation notes:

- websocket `user_message` events require `client_message_id`; Telegram and compatibility helpers synthesize and forward one for every submitted turn
- every queued user message now runs through the route-level supersede path, so background subagent work is also asked to stop even when the main loop itself is not in an active turn
- superseded turns remain visible in normal transcript replay so completed outputs from the older task are still available to the next turn
- pending interruption notices preserve their explicit reason across compaction, so a pending supersede notice stays supersede-specific after session rollover
- Telegram text messages and Telegram file submissions both use the same immediate-submit, cooperative-supersede path

## Scope

This plan covers:

- route-level interruption semantics for new user messages that arrive while the main loop is already running
- persistent websocket protocol changes needed to support multiple queued user turns on one route session
- `RouteRuntime` queueing, prioritization, and interruption control changes
- `AgentLoop` interruption metadata and persisted prompt-visible notices
- Telegram bridge refactors needed so inbound Telegram messages are not blocked behind one active `handle_message()` call
- interaction with subagents, approvals, detached bash jobs, and in-flight tool execution
- persistence, replay, and prompt-priority labeling rules
- tests
- follow-up updates to related development docs after implementation lands

This plan does not cover:

- hard cancellation of provider streams or running tools
- changing the single-user Telegram DM product shape
- exposing subagents directly to the user
- reordering transcript chronology to make later user messages appear earlier than events that actually happened first

## Resolved Product Decisions

These decisions are settled and should be implemented exactly as follows.

1. User message interruption is cooperative only. It must not hard-cancel provider streams, tool calls, or subprocesses already in flight.
2. Every mid-turn user message is treated exactly like a normal user message submission, not a special command.
3. If a user message arrives while a main turn is active, that new message immediately requests interruption of the active work and is queued as the next normal user turn.
4. The active turn ends at the next available interruption checkpoint.
5. As soon as the interrupted turn reaches that checkpoint and emits its final interrupted event, the next queued user turn begins automatically without waiting for another user action.
6. Multiple mid-turn user messages remain distinct turns and must be processed in first-submitted, first-injected order.
7. No Telegram-specific reminder or UI-only acknowledgement message should be sent for this behavior.
8. Explicit `/stop` remains a separate product behavior from “superseded by newer user message”.
9. A newer user message should cooperatively stop active subagents from the superseded task.
10. Main-loop tools or subagent tool calls that have already started are allowed to finish their current execution and persist their results.
11. Completed tool results from a superseded turn must remain persisted and replayable so the next turn can still know how the superseded tool execution went.
12. Automatic model follow-up work triggered by superseded task progress must not outrank queued user turns.
13. Prompt-visible prioritization should be implemented with persisted system notes, not by mutating raw user message text.
14. Structured metadata labels may be added to records where useful, but if prompt-visible prioritization cues are added they must also persist according to the persistence rule.
15. Transcript chronology must remain truthful; do not reorder old tool results after newer user messages just to simulate priority.

## Problem Summary

The current code has two separate blockers.

### 1. Telegram Head-Of-Line Blocking

In `src/jarvis/ui/telegram/bot.py`, `dispatch_message()` pushes ordinary inbound Telegram messages into a per-chat queue and `_chat_worker()` waits for `handle_message()` to finish before sending the next one to the gateway.

That means a second Telegram message sent during an active main turn does not even reach the route runtime until the first turn is already done.

### 2. Route Runtime Does Not Treat New User Messages As Interruptions

In `src/jarvis/gateway/route_runtime.py`, a normal `enqueue_user_message()` just queues another `_RouteTurnRequest`.

Only explicit `/stop` currently requests interruption through `request_stop()`, and that path intentionally latches “wait for next user message before resuming”.

That is correct for `/stop`, but it is the wrong semantic for “new user message supersedes old task”.

## Target Runtime Semantics

### Canonical Flow

For one route:

1. User sends message `A`.
2. Main turn `T1` begins for `A`.
3. While `T1` is running, user sends message `B`.
4. `B` is accepted immediately as a normal queued user turn.
5. Route runtime immediately requests cooperative interruption of `T1` with reason `superseded_by_user_message`.
6. `T1` continues only until its next existing interruption checkpoint.
7. `T1` emits `turn_done(interrupted=true, interruption_reason="superseded_by_user_message")`.
8. Main turn `T2` begins automatically for `B`.

If user sends `C` while `T2` is running, the exact same rule applies again.

### Chronology Rule

Chronology stays truthful.

If a tool result from `T1` completes before `T1` reaches its interruption checkpoint, it remains part of `T1` in transcript order.

The system does not rewrite history to place `B` earlier than records that actually completed before `T1` stopped.

Priority is conveyed by:

- the fact that `T1` is explicitly marked interrupted
- a persisted interruption notice on the next turn
- a persisted priority system note telling the model that completed outputs from the superseded turn are lower priority than the new user request unless directly relevant

### User Priority Rule

When at least one user turn is queued:

- queued user turns outrank internal runtime followups
- stale internal followups from the superseded task generation are invalidated
- background work notices may still persist, but they must not consume the next main-model turn ahead of queued user input

## Architecture Plan

## 1. Turn Identity Must Become First-Class

The current route stream is session-scoped but not robustly turn-scoped.

That is acceptable when one route session handles one visible turn at a time, but it is not robust enough once multiple user turns can be queued on a persistent websocket while background runtime turns also exist.

### Required additions

Introduce first-class turn identity throughout the stream model:

- `turn_id: str`
- `turn_kind: Literal["user", "runtime"]`
- `client_message_id: str | None`

Add a new outbound lifecycle event:

- `turn_started`

`turn_started` should be emitted when the route worker actually begins executing a queued request, not when the client merely submits it.

This gives the Telegram bridge a reliable way to associate streamed output with the correct queued user message without relying on fragile implicit ordering.

### Files expected to change

- `src/jarvis/core/agent_loop.py`
- `src/jarvis/core/__init__.py`
- `src/jarvis/gateway/route_events.py`
- `src/jarvis/gateway/protocol.py`
- `src/jarvis/gateway/app.py`
- `src/jarvis/ui/telegram/gateway_client.py`

## 2. RouteRuntime Must Own The Canonical Supersede Semantics

Do not implement this behavior as a Telegram-only workaround.

The canonical rule belongs in `RouteRuntime`, because route-level session semantics should be UI-agnostic.

### Required behavior

`RouteRuntime.enqueue_user_message()` must do all of the following when a main turn is already active:

- accept the new user message immediately
- enqueue it as a normal user turn
- request cooperative interruption of the current work with reason `superseded_by_user_message`
- invalidate internal followups from the superseded task generation so they cannot jump ahead of the new user turn
- keep FIFO ordering among user-submitted turns

### Queue model

Replace the single undifferentiated `_message_queue` scheduling assumption with explicit prioritization between:

- queued user turns
- queued internal runtime turns

Recommended shape:

- keep user turns in one FIFO queue
- keep internal runtime turns in another FIFO queue
- worker always drains user turns first
- internal runtime turns run only when no user turns are pending

This is simpler and less error-prone than trying to emulate priority inside one plain `asyncio.Queue`.

### `/stop` must stay separate

Keep explicit `/stop` semantics as they are conceptually:

- user asked to stop and pause
- suppress automatic resume until a later user message

Superseding with a new user message is different:

- user is not asking for a pause
- user is replacing the active task with a newer task
- automatic transition into the next queued user turn is required

That means `request_stop()` should remain the explicit-stop path, and a separate internal supersede path should be introduced.

### Files expected to change

- `src/jarvis/gateway/route_runtime.py`
- `src/jarvis/gateway/session_router.py`
- `src/jarvis/gateway/app.py`
- `tests/test_gateway_session_router.py`
- `tests/test_gateway_app.py`

## 3. AgentLoop Needs Reason-Aware Interruption Metadata

`AgentLoop` already has most of the hard interruption mechanics:

- active-turn stop request storage
- cooperative checkpoint checks
- interrupted turn persistence
- next-turn interruption notice persistence
- interrupted tool-call normalization

Do not replace those mechanics. Extend them.

### Add explicit interruption reason

Introduce a first-class interruption reason, at minimum:

- `user_stop`
- `superseded_by_user_message`

This reason should flow through:

- in-memory turn control state
- persisted transcript metadata
- `turn_done` event payloads
- session metadata used for next-turn interruption notices

### Prompt-visible priority note

When the next user turn starts after a superseded turn, append a persisted system note that tells the model:

- the previous task was superseded by a newer user message
- completed tool results from the superseded turn remain available as context
- the new user message has higher priority and should be handled first unless those older results are directly relevant

This is the recommended solution to the “label or not” question.

Use both:

- structured metadata labels on records for machine bookkeeping
- one persisted prompt-visible system note for model prioritization

Do not mutate the raw user message text and do not rewrite tool result content just to embed labels.

### Suggested metadata

Recommended metadata additions where relevant:

- `interruption_reason`
- `superseded_by_user_message`
- `superseded_turn_output`
- `completed_after_interrupt_request`
- `turn_kind`
- `client_message_id`

These metadata labels are for storage, inspection, filtering, and tests.

### Files expected to change

- `src/jarvis/core/agent_loop.py`
- `src/jarvis/storage/types.py`
- `src/jarvis/storage/service.py`
- `tests/test_agent_loop_streaming.py`
- `tests/test_agent_loop_tools.py`

## 4. Telegram Bridge Must Stop Blocking Submission On One Active Turn

The current bridge design is centered on one `handle_message()` call owning the pending queue for one turn.

That model is not compatible with immediate submission of newer user messages while a previous turn is still streaming.

### Required redesign

The route event worker should become the sole owner of streamed route output for a chat.

Inbound message handling should instead do:

- parse and authorize Telegram update
- download file attachments if needed
- normalize to final user text
- assign a `client_message_id`
- submit the user message immediately over the persistent route session
- store local per-chat bookkeeping for that submitted user turn

It should not block waiting for that turn’s `turn_done` before allowing another inbound Telegram message to be submitted.

### Recommended state model

For each chat, maintain:

- one persistent route session
- one route event worker
- one FIFO of submitted user turns keyed by `client_message_id`
- one active display state keyed by `turn_id`

The gateway event stream then drives output rendering:

- `turn_started(turn_kind="user", client_message_id=...)` binds streamed output to the correct submitted user message
- assistant deltas/messages/tool calls/approval requests are attributed to the active turn id
- `turn_done` finalizes that turn

Runtime turns can still be handled as background/system route events, but they must not steal queued user-turn priority in the route runtime.

### No special Telegram message

Do not emit any “interrupting current task” Telegram message.

The only visible user-facing output should remain the normal streamed assistant/tool/approval/system outputs that already belong to the app.

### Files expected to change

- `src/jarvis/ui/telegram/bot.py`
- `src/jarvis/ui/telegram/gateway_client.py`
- `tests/test_ui_telegram_bot.py`
- `tests/test_ui_gateway_client.py`

## 5. Protocol Changes

The protocol needs to support multiple user turns in one persistent route session cleanly.

### Inbound client event changes

Extend `user_message` to accept:

- `text: str`
- `client_message_id: str | None`

`client_message_id` is required on the websocket protocol after this feature. The Telegram bridge always sends it, and legacy compatibility helpers should synthesize one when needed.

### Outbound event changes

Add `turn_started`.

Enrich all turn-scoped outbound events with:

- `turn_id`
- `turn_kind`
- `client_message_id`

For interrupted completions, include:

- `interruption_reason`

Turn-scoped events means:

- `assistant_delta`
- `assistant_message`
- `tool_call`
- `approval_request`
- `turn_done`
- `error` when tied to a specific active turn

### Files expected to change

- `src/jarvis/gateway/protocol.py`
- `src/jarvis/gateway/route_events.py`
- `src/jarvis/gateway/app.py`
- `src/jarvis/ui/telegram/gateway_client.py`
- `tests/test_gateway_app.py`
- `tests/test_ui_gateway_client.py`

## 6. Subagents, Approvals, And Detached Bash Jobs

### Subagents

When a newer user message supersedes the active main task:

- request cooperative stop for active subagents tied to that work
- allow any already-started child tool execution to finish and persist in child transcript
- persist a concise main-session system note that the older task was superseded and child state may need later inspection

Do not phrase this as `/stop`.

Use new reason-specific wording and metadata.

### Approval waits

If the main loop is blocked waiting for approval and a newer user message arrives:

- the newer user message still supersedes the waiting turn
- the approval wait ends cooperatively
- the new user turn begins at the next slot

This is already conceptually compatible with the approval polling loop; extend tests and reason metadata accordingly.

### Detached bash jobs and already-launched work

If work is already launched and not cooperatively cancellable:

- allow it to finish
- persist its records truthfully
- prevent old-task automatic runtime followups from outranking queued user turns
- keep those results visible to later replay and next-turn reasoning

The persisted priority system note on the next user turn should make it clear that those results belong to a superseded task and are context, not the new task’s primary objective.

### Files expected to change

- `src/jarvis/gateway/route_runtime.py`
- `src/jarvis/subagent/manager.py`
- `src/jarvis/core/agent_loop.py`
- `tests/test_gateway_session_router.py`
- `tests/test_subagent.py`
- `tests/test_agent_loop_tools.py`

## 7. Persistence And Labeling Rules

This feature must obey the global persistence rule:

- no transient prompt-visible content
- replayable transcript remains source of truth

### Required persisted prompt-visible records

For superseded-turn handling, persist as normal transcript records:

- the interrupted-turn note
- any explicit unexecuted-tool-call normalization record
- any completed tool result that finished before the interrupted turn settled
- the next-turn interruption notice
- the next-turn priority note that says the old task was superseded by a newer user message

### What must not happen

Do not:

- silently invent a non-persisted priority rule only in gateway memory
- mutate raw user text to prepend labels
- mutate raw tool result content to prepend labels unless there is a very strong reason
- reorder transcript chronology

### Recommended implementation choice

Use one concise persisted system note on the next user turn for prompt-visible prioritization, plus structured metadata on the affected interrupted turn and carried-forward outputs.

That provides:

- correct replay behavior
- clear model guidance
- minimal transcript distortion

### Docs expected to change

- `dev_docs/persistence_refector.md`
- potentially `dev_docs/design.md` if the route-session behavior section needs updating

## 8. Concrete Implementation Tasks

The following is the recommended implementation order.

1. Extend core turn event models with `turn_id`, `turn_kind`, `client_message_id`, and interruption reason plumbing.
2. Extend gateway route event models and websocket protocol payloads to carry the new turn identity fields and the new `turn_started` event.
3. Refactor `RouteRuntime` queueing into explicit user-vs-internal prioritization.
4. Add a supersede-on-new-user-message path inside `RouteRuntime` that differs from explicit `/stop`.
5. Add reason-aware interruption persistence and next-turn priority note generation inside `AgentLoop`.
6. Update approval wait, tool-round, and interrupted-tool-result handling so supersede reasons survive all existing paths.
7. Refactor Telegram bridge submission so inbound user messages are sent immediately over the persistent route session and are no longer blocked by one active `handle_message()` call.
8. Update route event handling in Telegram to be driven by `turn_started`/`turn_done` and turn identity rather than one global “current pending queue”.
9. Add regression tests across loop, gateway runtime, gateway protocol/client, and Telegram bridge.
10. Update the persistence doc and any other related development docs.

This ordering is for engineering workflow only. It is not a product-phasing plan. The feature should land in the settled final form described in this document.

## 9. Test Plan

At minimum, add or update tests for the following.

### AgentLoop

- interrupt reason `superseded_by_user_message` persists distinctly from `/stop`
- next turn includes the correct supersede-priority persisted system note
- completed tool results from a superseded turn remain replayable
- unresolved interrupted tool calls still normalize exactly once
- approval wait interrupted by newer user message works

### RouteRuntime

- new user message during active main turn requests supersede interruption automatically
- queued user turns are FIFO
- queued user turns outrank internal runtime followups
- stale internal followups from the superseded task generation are discarded or deferred correctly
- explicit `/stop` behavior remains unchanged
- superseding user message stops subagents cooperatively

### Gateway protocol and client

- `user_message` accepts and forwards `client_message_id`
- `turn_started` is emitted
- all turn-scoped events include `turn_id`, `turn_kind`, and `client_message_id` when appropriate
- interrupted `turn_done` includes `interruption_reason`

### Telegram bridge

- second Telegram message submitted during active turn is forwarded immediately, not blocked behind the first turn’s completion
- no special Telegram interruption reminder is sent
- multiple mid-turn Telegram messages remain separate and FIFO
- file attachments submitted mid-turn behave like normal user messages and still supersede active work
- queued user turns are associated with the correct streamed outputs via turn identity

## 10. Acceptance Criteria

The implementation is complete when all of the following are true.

- Sending a new Telegram message during an active main turn causes the active task to become cooperatively interrupted at the next checkpoint.
- The newer message starts the next normal user turn automatically without needing `/stop` or any extra user action.
- Multiple newer user messages remain distinct turns and run FIFO.
- Queued user turns beat internal runtime followups.
- Superseded-task tool results and notes persist truthfully and remain replayable.
- The next turn is explicitly told, through persisted prompt-visible context, that the old task was superseded and the new user message has higher priority.
- Explicit `/stop` behavior still works exactly as before.
- Telegram emits no extra interruption acknowledgement.
- Tests cover loop, gateway, Telegram, and protocol regression paths.

## Existing Files Expected To Change

- `src/jarvis/core/agent_loop.py`
- `src/jarvis/core/__init__.py`
- `src/jarvis/storage/types.py`
- `src/jarvis/storage/service.py`
- `src/jarvis/gateway/route_runtime.py`
- `src/jarvis/gateway/route_events.py`
- `src/jarvis/gateway/protocol.py`
- `src/jarvis/gateway/app.py`
- `src/jarvis/gateway/session_router.py`
- `src/jarvis/ui/telegram/gateway_client.py`
- `src/jarvis/ui/telegram/bot.py`
- `src/jarvis/subagent/manager.py`
- `tests/test_agent_loop_streaming.py`
- `tests/test_agent_loop_tools.py`
- `tests/test_gateway_session_router.py`
- `tests/test_gateway_app.py`
- `tests/test_ui_gateway_client.py`
- `tests/test_ui_telegram_bot.py`
- `tests/test_subagent.py`
- `dev_docs/persistence_refector.md`
- `dev_docs/design.md`
- `notes/notes.md`

## Final Notes For The Implementing Agent

- Preserve chronology. Priority is a reasoning rule, not a history rewrite.
- Prefer explicit, persisted, reason-aware system notes over clever implicit behavior.
- Keep `/stop` and superseding-user-message semantics separate all the way down.
- Do not let internal runtime followups steal the next model turn from queued user input.
- Do not ship a Telegram-only hack; the authoritative behavior belongs in route runtime and protocol, with Telegram updated to use it correctly.
