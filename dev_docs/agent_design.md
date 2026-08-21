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

Automatic stream retry is normally allowed only before provider acceptance or normalized output. A response header or provider lifecycle event marks the request accepted, so ambiguous later failures propagate instead of replaying a possibly active generation. A provider adapter may explicitly mark a structured terminal failure as retry-safe after acceptance; `LLMService` still forbids that retry after any normalized output was exposed. Transport read/write timeouts remain ambiguous and are not blindly retried; only classified connection and pre-request pool-acquisition timeouts, plus explicitly terminal provider failures, are automatically retry-safe.

OpenRouter treats a terminal chat response with neither visible text nor a usable tool call as invalid. The adapter performs at most two provider-local retries when the failed attempt exposed no text or tool-call delta. Each retry preserves the request and sticky `session_id`, sends both `X-OpenRouter-Cache: true` and `X-OpenRouter-Cache-Clear: true`, and remains inside the original absolute request deadline. Empty attempts log generation id, raw finish reason, usage, response-cache status, reasoning activity, terminal signal, attempt number, and whether semantic output was exposed. Exhaustion raises `ProviderResponseError`; it never produces an empty successful turn.

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

Completed streaming turns persist one canonical assistant record. Interrupted streams and streams that end in a runtime failure persist a partial assistant-text checkpoint with incomplete-stream metadata so later replay reflects text that was already shown before the stop or failure.

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

Same-process runtime failures first persist any partial streamed assistant text, then use the same normalization immediately for the active turn.

## User Message Interruption

Route-level user message interruption is preemptive at active await boundaries.

When a new user message arrives while a main turn is active:

1. the new user message is accepted immediately as a normal queued user turn
2. `RouteRuntime` requests interruption of the active work with reason `superseded_by_user_message`
3. active subagents continue the assignments Jarvis gave them and receive no user-message injection
4. active main provider streams, provider requests, and tool awaits race against the turn stop signal
5. completed main tool results remain persisted and replayable; in-flight tool results are not fabricated
6. the interrupted main turn emits `turn_done(interrupted=true, interruption_reason="superseded_by_user_message")`
7. the queued user turn begins automatically and Jarvis decides what to do with any continuing children

Multiple mid-turn user messages remain distinct turns and run FIFO.

Explicit `/stop` is separate. `/stop` cooperatively stops active main and subagent work, suppresses automatic follow-ups, and pauses until the next user message. Already-detached bash jobs keep running. A newer ordinary user message supersedes only the active main task and continues automatically into the newer request.

`/new` is a control-only hard session boundary and does not reuse `/stop` semantics. As soon as the command is accepted, `RouteRuntime` closes the internal-follow-up gate, hard-preempts active main and subagent turns with reason `new_session`, and invalidates queued runtime continuations. Before the replacement session is created, the runtime terminates and finalizes all route-owned detached bash jobs, disposes every old subagent, clears retained bash/subagent notices, and persists a hard-reset trace in the old main transcript. The old main session is archived, provider/thread continuity is severed, and the fresh session remains idle. Only a later ordinary user message can start its agent loop.

Queued user turns outrank internal runtime follow-ups. Runtime follow-ups from superseded work are discarded or delayed so they cannot consume the next main-model turn ahead of user input.

Chronology stays truthful. Tool results completed by the superseded turn remain in the older turn. Priority is expressed through persisted interruption and priority notes, not by rewriting history.

Detached bash jobs are intentionally distinct from in-turn awaits. `/stop` suppresses their automatic follow-ups but does not cancel already-detached jobs; foreground bash execution is best-effort cancelled when the active tool await is preempted. `/new` terminates and finalizes detached jobs because none of their work may cross the hard session boundary.

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

Jarvis compacts sessions into a canonical, evidence-backed `CompactionBundle` and then deterministically compiles that bundle into replayable history. The provider never authors replay roles, system messages, exact copied content, bundle identity, chronology, lineage, or source hashes.

The compaction provider is configured separately as `core.compaction.provider`. It resolves the model from that provider's existing chat-model settings. `codex` is not a valid compaction provider because compaction uses the normal `LLMService` path. Generation, verification, malformed-verifier retry, and targeted repair all use this same user-selected provider and the existing user-controlled compaction output budget. Jarvis intentionally does not inspect the selected model's capacity, resize the budget, or select a fallback model.

### Source ledger

Jarvis first converts replayable transcript records into an ordered typed event ledger. Each event includes:

- stable event and record IDs
- source session, creation timestamp, generation, and sequence
- turn ID when present
- event type such as user message, assistant tool call, tool result, approval, interruption, subagent outcome, or system event
- tool call/result causal IDs
- exact record content
- a small allowlist of semantically useful metadata

Provider-specific metadata and raw provider argument encodings, bootstrap material, memory bootstrap, skill bootstrap, transcript-only records, transient datetime records, transient subagent snapshots, waiting boilerplate, prior replay items, and internal compaction records do not enter the delta ledger. Normalized tool names, call IDs, structured arguments, and exact result/error content remain. Tool-call validation failures and terminal subagent progress/outcomes are retained because they can explain later corrections or state.

### Canonical bundle

Schema version 2 contains:

- Jarvis-owned bundle ID, generation, creation time, prior-bundle lineage, ordered source sessions, delta IDs, cumulative evidence IDs, cutoff record ID, and a deterministic delta-content hash
- current objective with evidence IDs
- exact preserved user/assistant records copied by Jarvis, including source role, bytes, content hash, reason, and derived chronology
- chronological episodes with stable IDs, source/evidence IDs, outcomes, and derived chronology
- stable state entries for constraints, decisions, artifacts, open loops, and uncertainties
- current handover focus, next actions, work not to repeat, and required verification
- cumulative coverage groups recording how every delta event was represented or intentionally omitted

Artifact entries carry an exact locator, last-observed state, and `needs_verification`; they do not assert mutable external state as timeless truth. Open-loop entries require a next action and may carry a blocker.

New or changed artifact locators must appear exactly in the operation's cited delta event content or normalized metadata. This makes paths, URLs, and IDs deterministically grounded instead of trusting a rewritten literal.

### Incremental merge

The first compaction builds a bundle from the current session ledger. Later compactions load the verified bundle anchor from the active session and pass only new non-replay transcript records as the delta.

- Prior exact records, episodes, and state remain unchanged by default.
- Exact records can be added only by source record ID; Jarvis copies their stored role and content. Removing one requires current delta evidence.
- New episodes can summarize only current delta events.
- Old episodes may be deliberately consolidated hierarchically; source-event lineage and chronology are inherited.
- State changes use `add`, `update`, `resolve`, or `supersede` operations. Every operation requires current delta evidence.
- Supersession is same-category, single-successor, active-entry-only, and acyclic. Superseded entries remain in the canonical record.
- Objective changes require current delta evidence.

This prevents later generations from re-summarizing prompt-visible summaries and preserves original evidence lineage across repeated compactions.

### Validation, verification, and repair

Jarvis validates the entire draft atomically. It never drops invalid middle items. Deterministic validation checks exact field sets and types, identifiers, evidence existence, chronology, exact-copy hashes, episode sources, state shapes, supersession lineage, and coverage.

Every current delta event must appear in exactly one coverage group. User events cannot be omitted. Non-omitted groups must point to an existing preserved record, episode, state entry, objective, or handover.

After deterministic validation, a second request to the configured compaction provider verifies semantic fidelity against the prior bundle and delta ledger. It checks omissions, contradictions, unsupported claims, false completion, stale external state, active constraints, causal outcomes, and the proposed continuation. A malformed verifier response is retried once with an explicit contract retry. A negative verdict or deterministic validation failure produces a targeted full-draft repair request. Jarvis permits two repair attempts and activates nothing unless the final bundle passes deterministic and semantic verification.

## Compaction Pruning

Pre-compaction pruning is item-level only. It can keep or drop whole records; it never rewrites content.

Dropped source records include:

- internal `kind="compaction"` bundle/audit records
- prior `compaction_replay` records
- identity/bootstrap records
- `transcript_only` records
- memory bootstrap records
- current-time turn-context records
- transient subagent status snapshots and waiting boilerplate
- skill bootstrap records
- empty assistant messages

Kept source records include user messages, meaningful assistant text/tool calls, tool results and validation failures, approvals, interruptions, supersede records, terminal subagent outcomes, and material system notes.

## Session Rebuild After Compaction

After successful compaction, the fresh session is rebuilt in this order:

1. fresh identity bootstrap
2. fresh tool bootstrap
3. fresh skill bootstrap or skill search guidance
4. fresh memory bootstrap, when enabled
5. internal canonical bundle anchor (`kind="compaction"`, never provider-visible)
6. deterministic compaction replay items
7. carried-forward in-progress turn records, when compaction happened mid-turn

The replay compiler emits:

- one fixed Jarvis-authored `system` `history_boundary`
- exact `preserved_message` items using the source user/assistant role and bytes
- chronological assistant `episode` items
- one assistant `state_snapshot`
- one assistant `handover`

Replay records use `type="compaction_replay"` metadata with bundle/generation IDs, exact-copy status, source record IDs, and evidence event IDs. No model-authored text receives system authority.

The old session receives one `kind="compaction"` audit record containing the explicit manual instruction, provider/model, canonical bundle, accepted draft, verifier report, repair count, per-call traces, aggregate usage, and deterministic replay items. The old session remains active until the bundle has passed all validation and verification.

Mid-turn compaction excludes active in-progress turn records from compaction source. Active-turn carry-forward records remain separate so the same content is not duplicated into replacement history.

Codex-backed actors use the same bundle, provider selection, validation, persistence, and replay compiler. Because Codex thread input is text-only rather than native transcript replay, the Codex adapter renders the fixed history boundary, exact prior user/assistant labels, and assistant historical context deterministically, then seeds the new thread exactly once.

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
- never treat an ordinary user message as child input or a child interruption request
- keep `/new` on its distinct hard-reset path and never let the command itself open the runtime-follow-up gate
- ensure subagent prompt-visible behavior follows the same shared-loop persistence rules
