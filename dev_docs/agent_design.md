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

Internal provider activity is not meaningful progress. A stream must produce normalized semantic output within the fixed 300-second first-semantic-output watchdog or it fails as a typed `ProviderTimeoutError`; after output begins, a separate 120-second idle watchdog applies while the absolute request deadline remains authoritative. Keepalives and reasoning/lifecycle traffic do not satisfy semantic progress. Generation uses a per-provider/model bulkhead (three concurrent requests), capped backoff, and a transient-failure circuit breaker, so child fan-out cannot stampede an unhealthy provider.

Automatic stream retry is normally allowed only before provider acceptance or normalized output. A response header or provider lifecycle event marks the request accepted, so ambiguous later failures propagate instead of replaying a possibly active generation. A provider adapter may explicitly mark a structured terminal failure as retry-safe after acceptance; `LLMService` still forbids that retry after any normalized output was exposed. Transport read/write timeouts remain ambiguous and are not blindly retried; only classified connection and pre-request pool-acquisition timeouts, plus explicitly terminal provider failures, are automatically retry-safe.

After service-local retries are exhausted, the route may start up to three fresh checkpoint-based recovery turns. Partial visible output is persisted before recovery, completed tools are not replayed, and every recovery turn retains the original `client_message_id` so synchronous and websocket consumers follow one logical task. Exhaustion persists the paused recovery state, suppresses automatic follow-ups until a new user message, and emits a blocked terminal event instead of leaving the client waiting indefinitely.

OpenRouter treats a terminal chat response with neither visible text nor a usable tool call as invalid. The adapter performs at most two provider-local retries when the failed attempt exposed no text or tool-call delta. Each retry preserves the request and sticky `session_id`, sends both `X-OpenRouter-Cache: true` and `X-OpenRouter-Cache-Clear: true`, and remains inside the original absolute request deadline. Empty attempts log generation id, raw finish reason, usage, response-cache status, reasoning activity, terminal signal, attempt number, and whether semantic output was exposed. Exhaustion raises `ProviderEmptyResponseError`; post-acceptance replay is marked safe only when no semantic text or tool-call delta escaped, and otherwise the failure is not retried. A genuinely empty failure can continue through the normal bounded subagent checkpoint-recovery path, and no empty attempt becomes a successful turn.

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

Tool execution has two functional-liveness bounds: `JARVIS_TOOL_MAX_ROUNDS_PER_TURN` defaults to `64` and triggers an automatic persisted replan continuation, while `JARVIS_TOOL_MAX_ROUNDS_PER_TASK` defaults to `256` unchanged rounds and triggers one tools-disabled checkpoint response with `completion_block_reason="tool_liveness_exhausted"`. The latter is a true malfunction disposition, never a blank or successful completion and never a generic request for the user to send “Continue.”

Tool results can carry a first-class `yield_turn` disposition for runtime control primitives. The result and deterministic skipped results for any later calls in the same batch are persisted, the task remains deferred, and the loop makes no follow-up provider request. Yielding does not consume the unchanged-round liveness budget.

Tool activity state is task-scoped rather than session-scoped. Each user task has a compact schema-v2 sidecar under the transcript archive's `tool_tasks/` directory containing its task contract, stalled-round count, progress epoch, bounded runtime-progress signatures, exact-call liveness state, and acceptance evidence. The stalled count resets on material tool or distinct orchestrator progress; persisted bash/subagent system notes record that progress at delivery time, and stable job/status/byte fingerprints distinguish real output growth without treating changing wall-clock text as progress. Legacy schema-v1 cumulative `rounds` values migrate to zero so old long sessions cannot permanently poison a task. The session index stores only `active_tool_task_id`; unchanged sidecars and unchanged session metadata are not rewritten. Explicit resumes and superseding clarifications merge new requirements into the active contract, explicit task or requirement replacement creates a fresh contract, and short status/time side queries use a temporary child contract before restoring the active task. Compaction carries the active task pointer without copying the sidecar into every session record.

Repeated-result limits are evaluated inside progress epochs. A real workspace mutation or other material orchestration progress starts a new epoch and clears stale call/result signatures. The second identical invalid result or third identical no-progress result suppresses only future reuse of that exact normalized tool-and-arguments signature. Workspace conflicts are not merged across different tools, commands, or paths. Suppression returns a normal persisted tool result with deterministic signatures, call IDs, count, epoch, and remediation; the rest of the batch and later materially different actions remain available.

At task creation Jarvis deterministically extracts explicit user requirements into a prompt-visible, persisted contract with stable item IDs. A content-addressed contract revision is injected once per session, re-injected after compaction creates a new session, and injected again only when merged requirements materially change it; ordinary continuations reuse the persisted record instead of duplicating prompt bytes. The contract cannot be silently reduced by the model's acceptance ledger. Delegation, verification-gate, authored-source-line, visual-inspection, and changed-test-review requirements need matching runtime-observed evidence before a mutation can be marked accepted. Test changes made by a child are reported to the owning main task: the child retains only locally satisfiable acceptance gates, while Jarvis must obtain an independent non-editing child review before final acceptance. Slice rollover notices include the current progress epoch and outstanding contract items.

Slice rollover follows the same unresolved-call normalization rule:

1. persist the unexecuted-tool-call notice
2. persist an automatic-continuation instruction with boundary diagnostics
3. reset stateful provider continuation lineage so the normalized transcript can be replayed without leaving a server-side tool call unresolved
4. rebuild a normal follow-up request with basic tools and all current-turn discoverable activations still available
5. reset the internal slice counter and continue executing any newly returned tool calls

A completely empty continuation response is normalized to an explicit failure message. A text or tool-call continuation is preserved as returned. Streaming additionally emits the continued tool-call event so UI state remains accurate.

An unverified-mutation handoff is an automatic continuation, not a terminal block. The attempted assistant response remains persisted for truthful replay, a concise acceptance instruction is appended, and the loop continues with tools until current gates and the acceptance ledger resolve. Premature streamed completion text is buffered from the UI while evidence is outstanding. A turn truthfully waiting on orchestrator-owned bash or subagent work remains non-blocked, records `task_completion_deferred`, retains the active task contract and acceptance state, and defers the acceptance continuation until pending work becomes terminal. Only external/runtime blocks and the high unchanged-round liveness ceiling set `completion_blocked`, always with a structured `completion_block_reason`.

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

Explicit `/stop` is separate from ordinary supersession but now uses route-wide hard quiescence. It hard-preempts the active main turn and active child turns, converts waiting children to a paused `main_stop` state, terminates and finalizes route-owned detached jobs and services, drains queued user/runtime continuations with terminal interruption events, clears retained notices/timers, and leaves the existing session available only for an explicit later user resume. Concurrent stop requests serialize through one route lock.

`/new` is a control-only hard session boundary and does not reuse `/stop` semantics. As soon as the command is accepted, `RouteRuntime` closes the internal-follow-up gate, hard-preempts active main and subagent turns with reason `new_session`, and invalidates queued runtime continuations. Before the replacement session is created, the runtime terminates and finalizes all route-owned detached bash jobs, disposes every old subagent, clears retained bash/subagent notices, and persists a hard-reset trace in the old main transcript. The old main session is archived, provider/thread continuity is severed, and the fresh session remains idle. Only a later ordinary user message can start its agent loop.

Queued user turns outrank internal runtime follow-ups. Runtime follow-ups from superseded work are discarded or delayed so they cannot consume the next main-model turn ahead of user input.

Chronology stays truthful. Tool results completed by the superseded turn remain in the older turn. Priority is expressed through persisted interruption and priority notes, not by rewriting history.

Detached bash jobs are intentionally distinct from in-turn awaits, but both `/stop` and `/new` terminate and finalize route-owned detached jobs and services. Foreground bash execution is best-effort cancelled when its active tool await is preempted. `/new` additionally disposes children and replaces the session; `/stop` preserves paused child/session state for an explicit resume.

When route-owned children or detached jobs remain active but the main agent has no actionable work, Jarvis can call `orchestrator_wait` with a preferred liveness deadline and optional actor IDs. A successful call is a first-class turn yield: its result is persisted, the current turn completes as deferred, and no follow-up provider request is made. Before parking, routine queued notices are atomically persisted and acknowledged, stale notices are dropped, and material inspect/finalize notices are returned once as typed review evidence. The runtime clamps the deadline to 30 seconds–30 minutes and applies an exponential 60-second adaptive floor after unchanged reviews. Later material terminal/attention events cancel the timer and wake Jarvis immediately; routine running/output-growth observations are recorded by the supervisor without spending a main or child model turn. A deadline wake performs one runtime review, after which Jarvis must act or register another bounded wait. Narrow wait-only detached `sleep` polling commands are rejected.

## Turn Identity And Gateway Events

The route protocol carries first-class turn identity:

- `turn_id`
- `turn_kind`: `user` or `runtime`
- `client_message_id`

Clients submit `user_message` events with a non-empty `client_message_id`.

The gateway emits `turn_started` when the route worker actually starts a queued request. Turn-scoped route events include turn identity so the Telegram bridge can attribute output correctly even when multiple user turns are queued.

Every emitted route event also has a route-local monotonic `sequence`, a per-actor monotonic `actor_sequence`, an `actor_id`, and provenance (`origin_session_id`, `origin_turn_id`, and actor `run_generation` where applicable). Consumers discard events that do not belong to the actor’s current run generation, preventing superseded sessions, compacted lineages, and disposed/restarted children from updating current UI state.

Turn-scoped events include:

- `turn_started`
- `assistant_delta`
- `assistant_message`
- `tool_call`
- `approval_request`
- `turn_done`
- turn-scoped errors

Interrupted `turn_done` events include `interruption_reason`.

`task_status` is the route-wide activity latch, not a mirror of one main turn. It stays active while a main request, queued work, a pending detached job, or a running/waiting/approval-blocked child exists, even if the main agent itself is paused for user input. Every new subscriber receives an immediate authoritative snapshot before incremental events. Lifecycle transitions log route/task counts so a dropped UI indicator can be correlated with the server state.

## Telegram Submission Model

Telegram inbound handling submits user messages immediately over the persistent route session. It does not wait for the current turn to finish before forwarding the next user message.

`/models` is handled entirely by the Telegram bridge from the immutable provider configuration snapshot resolved at process startup. It does not create a gateway route, agent turn, transcript record, or provider request.

For each chat, the bridge maintains:

- one persistent route session
- one route event worker
- submitted user turns keyed by `client_message_id`
- active display state keyed by `turn_id`

`turn_started` binds streamed output to the correct submitted Telegram message. No special Telegram “interrupting current task” acknowledgement is sent.

Telegram treats `task_status` as the master typing indicator. Approval/auth prompts, visible progress updates, provider failures, and chat-output pause do not clear it; only an inactive route status does. If the heartbeat task exits unexpectedly while the route remains active, the bridge persists the exception and restarts the task. Gateway/runtime errors never become Telegram fallback text. `/stop` temporarily mutes stale in-flight output, waits for the hard-quiesce acknowledgement, and then sends the single functional confirmation `Session stopped.`

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

Jarvis compacts sessions into a canonical structured `CompactionBundle` and deterministically compiles it into replayable history. The model performs semantic compression only. Jarvis owns the output schema, replay roles, system boundary, exact-message copying, bundle identity, chronology, source lineage, and hashes.

The compaction provider is configured separately as `core.compaction.provider` and resolves its model from that provider's existing chat-model settings. `codex` is not valid because compaction uses the normal `LLMService` path. Jarvis does not inspect model capacity, resize the user-controlled output budget, select a fallback model, or change provider automatically.

### Source ledger

Jarvis first converts replayable transcript records into an internal ordered typed event ledger. Each event retains:

- stable event and record IDs
- source session, creation timestamp, generation, and sequence
- turn ID when present
- event type such as user message, assistant tool call, tool result, approval, interruption, subagent outcome, or system event
- tool call/result causal IDs
- exact record content
- a small allowlist of semantically useful metadata

Provider-specific metadata and raw provider argument encodings, bootstrap material, memory bootstrap, skill bootstrap, transcript-only records, transient datetime records, transient subagent snapshots, waiting boilerplate, prior replay items, and internal compaction records do not enter the delta ledger. Tool-call validation failures and terminal subagent outcomes remain because they can explain later state.

The provider does not receive this verbose internal representation. Jarvis renders a compact causal transcript with short local references (`E1`, `E2`, ...), short tool-call/result links, event type, role, time, and semantic content. Internal record, session, turn, and call IDs are not exposed. When an exact long tool argument already appears in its paired result, the renderer substitutes a reference instead of sending the bytes twice.

Every compaction request is estimated before dispatch with the intentionally lightweight four-characters-per-token heuristic. Its safety ceiling is 70% of the operator-selected normal preflight input limit. Jarvis first reserves the fixed instructions/prior-bundle cost, then applies one global source-character budget across all tool/system evidence; tool arguments receive both nested-string and whole-payload bounds. User messages and plain assistant messages are never shortened. Repeated task-contract records, routine bash lifecycle updates, identical system evidence, and selected repeated failure records are deterministically coalesced to their latest causal event before dispatch. If the exact messages plus minimum causal evidence cannot fit, compaction fails before contacting the provider and records detailed budget diagnostics rather than silently dropping exact content.

### Canonical bundle

Schema version 3 contains:

- Jarvis-owned bundle ID, generation, creation time, prior-bundle lineage, ordered source sessions, current delta record IDs, cutoff record ID, delta hash, rolling cumulative hash, and cumulative record count
- current objective and essential background
- exact preserved user/assistant records copied by Jarvis, including source role, bytes, content hash, reason, and derived chronology
- chronological episodes with Jarvis-derived stable IDs, outcomes, and chronology
- structured current entries for constraints, decisions, artifacts, open loops, and uncertainties, with Jarvis-derived IDs
- current handover focus, next actions, work not to repeat, and required verification

Artifact entries carry an exact locator, last-observed state, and `needs_verification`; they do not assert mutable external state as timeless truth. Open-loop entries require a next action and may carry a blocker.

### Incremental merge

The first compaction builds a bundle from the current session ledger. Later compactions load the canonical bundle anchor and pass its compact semantic state plus only new non-replay transcript records. The model returns one complete current semantic record rather than an incremental database mutation protocol. This lets it retire stale state and consolidate older episodes without authoring stable IDs, supersession graphs, evidence links, or exhaustive coverage bookkeeping.

Previously preserved exact messages receive short `P1`, `P2`, ... references. Current delta events receive `E1`, `E2`, ... references. The model lists only exact messages that should remain material; Jarvis resolves those local references and copies stored bytes. It never accepts model-authored exact content.

### Validation and exceptional repair

The provider is forced through the shared `submit_compaction` tool schema and instructed to perform its semantic fidelity review before the single submission. A normal compaction therefore makes one model call. Jarvis then atomically validates exact field sets and types, non-empty values, local preservation references, exact-copy hashes, state shapes, and canonical bundle integrity.

No second model verifier resends the source. The earlier verifier duplicated cost and latency while asking another model response to police bookkeeping that Jarvis should own. If a provider nevertheless returns malformed output or an invalid local reference, Jarvis may make at most two targeted full-record repairs. Those retries are fault containment, not a normal stage. Rejection issues and call traces are logged immediately.

The workflow shares one total deadline across all exceptional attempts. Compaction provider and memory awaits participate in route interruption. Mid-turn compaction races against the active turn stop signal. Reactive, preflight, and manual compaction register their own active operation before a normal turn exists, so `/stop`, a superseding user message, and `/new` can cancel them as well. A validated result is checked for interruption before the old session is archived; bundle activation then runs as a consistency-sensitive commit. Phase, attempt, call trace, validation issue, budget, deadline, session, reason, and turn metadata are attached to runtime failures.

Failed compaction records a durable JSONL error and stores a latch keyed by the exact source revision in session backend state. Automatic reactive/preflight attempts for the unchanged source are suppressed after the first failure and clear the reactive flag, preventing an error loop; `/compact` deliberately bypasses the latch for a user-authorized retry. New source evidence changes the revision and permits a new automatic attempt.

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

Replay records use `type="compaction_replay"` metadata with bundle/generation IDs, exact-copy status, and source record IDs. No model-authored text receives system authority.

The old session receives one `kind="compaction"` audit record containing the explicit manual instruction, provider/model, canonical bundle, accepted semantic submission, deterministic validation report, repair count, per-call traces, aggregate usage, and replay items. The old session remains active until the bundle has passed validation.

Mid-turn compaction excludes active in-progress turn records from compaction source. Active-turn carry-forward records remain separate, preserve their original record IDs/session/timestamps, and coalesce repeated identical tool-call/result cycles into one truthful system handover note so the same material is neither duplicated nor amplified into replacement history.

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
- preserve canonical tool arguments once; retain raw provider arguments only when they differ, and never echo malformed raw payloads back into model-visible errors
- preserve the intentionally lightweight 4-characters-per-token context estimate. The configured context window is an operator safety budget, not a claim about a provider tokenizer or physical model limit.
