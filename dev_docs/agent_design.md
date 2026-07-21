# Agent Design

## Purpose

This document describes the runtime design for Jarvis agent execution, transcript persistence, compaction, provider context handling, route-level interruption, and approvals.

Jarvis has one shared agent loop architecture used by the main agent and subagents. Normal LLM providers run through `AgentLoop` and `LLMService`; Codex-backed actors use the Codex backend described in `codex_backend.md`.

## Main Runtime Shape

The normal provider path is:

- `RouteRuntime`
- `AgentLoop`
- `LLMService`
- provider adapter under `src/jarvis/llm/providers/`

`RouteRuntime` owns route-level scheduling and supervision. It coordinates:

- the main agent loop
- subagent manager
- route event publication
- approval routing
- detached bash job supervision
- queued user turns and runtime follow-up turns

`AgentLoop` owns:

- session lifecycle
- prompt/bootstrap construction
- provider request construction
- tool-call execution rounds
- streaming and non-streaming completion handling
- approval waits
- compaction
- interruption normalization
- transcript persistence
- provider continuation/cache state persistence

`LLMService` owns normal provider adapters only. The provider value `codex` is rejected by `LLMService` with `LLMConfigurationError` because Codex uses a separate backend.

## LLM Deadlines, Transport Timeouts, And Streaming Activity

Normal provider requests use three separate limits:

- `JARVIS_LLM_REQUEST_DEADLINE_SECONDS` defaults to 3600 seconds and is one absolute wall-clock budget for the logical request, including service retries and retry backoff. `LLMRequest.deadline_seconds` and `EmbeddingRequest.deadline_seconds` can override it per request.
- `JARVIS_LLM_CONNECT_TIMEOUT_SECONDS` defaults to 30 seconds and bounds connection establishment, connection-pool acquisition, and request writes.
- `JARVIS_LLM_READ_TIMEOUT_SECONDS` defaults to 3600 seconds and bounds transport-level inactivity while waiting for the next raw response chunk.

The service deadline never resets when a stream event arrives. Provider transports own connection/read timeouts; request deadlines are not copied into provider SDK payloads or per-request transport options.

Provider adapters may yield internal `ProviderActivityEvent` values for lifecycle, keepalive, reasoning, empty-signature, response-header, and other non-user-visible stream activity. `LLMService` consumes these values for acceptance state and diagnostics, but never forwards them to `AgentLoop` or transcript persistence. Anthropic and Gemini use their native streaming APIs; OpenAI/Grok Responses lifecycle events, OpenRouter SSE comments, and LM Studio Responses lifecycle events follow the same internal activity contract.

Automatic stream retry is allowed only before any provider acceptance or normalized output. A response header or provider lifecycle event marks the request accepted, so later failures propagate instead of replaying a possibly active generation. Transport read/write timeouts are also treated as ambiguous and are not blindly retried; only classified connection and pre-request pool-acquisition timeouts are automatically retry-safe.

## Provider Context Strategies

Replayable transcript records are always stored as Jarvis archive/debug/audit history. Provider context construction then follows the provider strategy:

- OpenAI native: provider-side continuation through stored `conversationId` and `previousResponseId`
- Grok/xAI native: provider-side continuation through stored `previousResponseId`
- Gemini native: cached base context and tool schemas through Gemini `cachedContent`
- Anthropic native: local transcript replay with prompt-cache blocks
- OpenRouter: local transcript replay with prompt-cache best effort, Anthropic `cache_control` when routing Claude models, sticky `session_id`, and OpenRouter response-cache headers

Provider session state is persisted on `SessionMetadata.provider_session_state`. Transcript JSONL stays normalized around Jarvis messages, tool calls, and tool results. Provider-specific replay aids may be stored in record metadata, but the core loop treats them as opaque and round-trips them through `LLMMessage.metadata`.

Provider request context construction is built in. It is not user-configurable.

## Persistence Rules

Core invariants:

1. Replayable transcript records are persisted as Jarvis archive/debug/audit history.
2. Every non-image prompt-visible record that should affect later replay is persisted.
3. `transcript_only` means archived but intentionally excluded from replay.
4. Provider quirks stay in `src/jarvis/llm/providers/`.
5. Replay must not invent prompt-visible content.
6. Provider continuation/cache handles live in session metadata.
7. Provider-local metadata inside transcript records is opaque to the core loop.

Prompt-visible records persisted before model calls include:

- turn datetime context
- interruption notices
- runtime/orchestrator messages
- skill bootstrap headers or skill search guidance
- subagent status snapshots

Prompt-visible records persisted during follow-up rounds include:

- assistant messages
- assistant tool calls
- tool results
- approval requests and decisions
- tool-round-limit notices
- orchestrator waiting notices
- explicit unexecuted-tool-call normalization notices

Completed streaming turns persist one canonical assistant record. Interrupted streams can persist a partial assistant-text checkpoint with interruption metadata so later replay reflects text that was already shown before the stop.

Internal provider activity events are transient control signals. They are never assistant content and must never be written to the transcript.

## Tool-Call Normalization

Replay expects unresolved assistant tool calls to be normalized by explicit transcript records.

If a stream or process interruption leaves assistant tool calls without matching tool results, the loop appends an explicit unexecuted-tool-call notice. Replay raises if it encounters unresolved tool calls without either matching results or that persisted notice.

Tool-round-limit recovery follows the same rule:

1. persist the unexecuted-tool-call notice
2. persist the tool-round-limit instruction
3. build the recovery request

## Crash And Restart Recovery

On session access, orphaned `in_progress` turns are reconciled before the session is used again.

The loop:

- finds `in_progress` turns that are not active in memory
- inspects their persisted records
- appends an explicit unexecuted-tool-call notice when needed
- appends an orphaned-turn recovery system note
- marks the turn `interrupted`

Same-process runtime failures use the same normalization immediately for the active turn.

## User Message Interruption

Route-level user message interruption is preemptive at active await boundaries.

When a new user message arrives while a main turn is active:

1. the new user message is accepted immediately as a normal queued user turn
2. `RouteRuntime` requests interruption of the active work with reason `superseded_by_user_message`
3. active subagents tied to the superseded task are also asked to stop
4. active provider streams, provider requests, and tool awaits race against the turn stop signal
5. completed tool results remain persisted and replayable; in-flight tool results are not fabricated
6. the interrupted turn emits `turn_done(interrupted=true, interruption_reason="superseded_by_user_message")`
7. the queued user turn begins automatically

Multiple mid-turn user messages remain distinct turns and run FIFO.

Explicit `/stop` is separate. `/stop` means stop and pause until the next user action. A newer user message means supersede the active task and continue automatically into the newer request. `/new` uses the same route stop request path as `/stop`, then resets subagents and starts the fresh session request.

Queued user turns outrank internal runtime follow-ups. Runtime follow-ups from superseded work are discarded or delayed so they cannot consume the next main-model turn ahead of user input.

Chronology stays truthful. Tool results completed by the superseded turn remain in the older turn. Priority is expressed through persisted interruption and priority notes, not by rewriting history.

Detached bash jobs are intentionally distinct from in-turn awaits. `/stop` suppresses their automatic follow-ups but does not cancel already-detached jobs; foreground bash execution is best-effort cancelled when the active tool await is preempted.

## Turn Identity And Gateway Events

The route protocol carries first-class turn identity:

- `turn_id`
- `turn_kind`: `user` or `runtime`
- `client_message_id`

Clients submit `user_message` events with a non-empty `client_message_id`.

The gateway emits `turn_started` when the route worker actually starts a queued request. Turn-scoped route events include turn identity so the Telegram bridge can attribute output correctly even when multiple user turns are queued.

Turn-scoped events include:

- `turn_started`
- `assistant_delta`
- `assistant_message`
- `tool_call`
- `approval_request`
- `turn_done`
- turn-scoped errors

Interrupted `turn_done` events include `interruption_reason`.

## Telegram Submission Model

Telegram inbound handling submits user messages immediately over the persistent route session. It does not wait for the current turn to finish before forwarding the next user message.

`/models` is handled entirely by the Telegram bridge from the immutable provider configuration snapshot resolved at process startup. It does not create a gateway route, agent turn, transcript record, or provider request.

For each chat, the bridge maintains:

- one persistent route session
- one route event worker
- submitted user turns keyed by `client_message_id`
- active display state keyed by `turn_id`

`turn_started` binds streamed output to the correct submitted Telegram message. No special Telegram “interrupting current task” acknowledgement is sent.

## Approvals

Approvals are enforced inside sensitive tools and coordinated by the agent loop and route runtime.

The current approval model supports:

- exact-action approval for `bash`
- exact-manifest approval for `tool_register`
- exact-request approval for `email`
- route-level approval routing for main and subagent actors
- Telegram inline approve/reject controls

Approval state is persisted as a single `pending_approval` object on session metadata while a turn is waiting. Approval requests and decisions are also recorded in transcript records for auditability.

Rejected approvals stop the waiting section without executing the action. For main turns, the system idles until the next user message. For subagents, the child pauses and reports state back through the route runtime.

## Compaction

Jarvis compacts sessions into structured replacement history, not a monolithic summary seed.

The compaction provider is configured separately as `core.compaction.provider`. It resolves the model from that provider's existing chat-model settings. `codex` is not a valid compaction provider because compaction uses the normal `LLMService` path.

Compaction output is JSON:

```json
{
  "items": [
    {
      "type": "compaction",
      "role": "system",
      "kind": "session_frame",
      "content": "..."
    }
  ]
}
```

Replacement item kinds:

- `session_frame`: first item, `system`, compact frame for the next session
- `preserved_message`: exact critical user or assistant wording, `verbatim=true`
- `condensed_span`: assistant-written compressed conversation beat
- `handover_state`: last item, `system`, exact continuation state

Validation requires:

- first item is `session_frame`
- last item is `handover_state`
- every item has `type="compaction"`
- role is `system`, `user`, or `assistant`
- no `tool` role items
- every item has non-empty content
- `preserved_message` uses `verbatim=true`
- ordering follows original chronology

## Compaction Pruning

Pre-compaction pruning is item-level only. It can keep or drop whole records; it never rewrites content.

Dropped source records include:

- old-session `kind="compaction"` audit records
- identity/bootstrap records
- `transcript_only` records
- memory bootstrap records
- legacy `summary_seed` records
- current-time turn-context records
- subagent status snapshots
- skill bootstrap records
- tool-call validation failure records
- empty assistant messages

Kept source records include user messages, meaningful assistant text, tool results, approvals, interruptions, supersede records, and material system notes.

Post-compaction pruning validates returned replacement items and drops invalid or exact consecutive duplicate items. It does not rewrite, shorten, or merge returned content.

## Session Rebuild After Compaction

After successful compaction, the fresh session is rebuilt in this order:

1. fresh identity bootstrap
2. fresh tool bootstrap
3. fresh skill bootstrap or skill search guidance
4. fresh memory bootstrap, when enabled
5. compacted replacement transcript items
6. carried-forward in-progress turn records, when compaction happened mid-turn

Replacement items are persisted as normal replayable transcript records with structural metadata such as:

- `type="compaction"`
- `compaction_item=true`
- `compaction_kind`
- `verbatim`
- `source_record_ids`
- `source_range`
- `compaction_generation`

The old session receives one `kind="compaction"` audit record containing compaction metadata and the full returned structured payload.

Mid-turn compaction excludes active in-progress turn records from compaction source. Active-turn carry-forward records remain separate so the same content is not duplicated into replacement history.

Codex-backed actors use the same compaction schema and provider selection, then start a fresh Codex thread and seed it once with the compacted replacement history.

## Intentional Cache Boundaries

The following can legitimately break provider cache continuity:

- `view_image` turns
- compaction into a new session
- model or provider changes
- provider setting changes
- tool registry or runtime-tool manifest changes
- skill installation or skill file changes
- current-turn-only discoverable tool activation

These are not transcript-fidelity bugs.

## Maintenance Rules

When changing the agent loop:

- persist prompt-visible replayable content before relying on it later
- keep replay-time synthesis out of prompt-visible paths
- keep provider-specific request shaping inside provider adapters
- persist continuation/cache handles in provider session state
- preserve truthful transcript chronology
- keep `/stop` and superseding-user-message semantics separate
- ensure subagent prompt-visible behavior follows the same shared-loop persistence rules
