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

Tool activity state is task-scoped rather than session-scoped. Each user task has a compact schema-v3 sidecar under the transcript archive's `tool_tasks/` directory containing only stalled-round count, progress epoch, bounded runtime-progress signatures, and exact-call liveness state. The stalled count resets on material tool or distinct orchestrator progress; persisted bash/subagent system notes record that progress at delivery time, and stable job/status/byte fingerprints distinguish real output growth without treating changing wall-clock text as progress. Only the current schema is read. The session index stores only `active_tool_task_id`; explicit resumes/superseding continuations reuse the active liveness task, while short status/time side queries temporarily use a child liveness task before restoring the parent. Compaction carries the active task pointer without copying sidecar state into every session record.

Repeated-result limits are evaluated inside progress epochs. A real workspace mutation or other material orchestration progress starts a new epoch and clears stale call/result signatures. The second identical invalid result or third identical no-progress result suppresses only future reuse of that exact normalized tool-and-arguments signature. Workspace conflicts are not merged across different tools, commands, or paths. Suppression returns a normal persisted tool result with deterministic signatures, call IDs, count, epoch, and remediation; the rest of the batch and later materially different actions remain available.

Subagent semantic completion is not decided by the runtime. The subagent bootstrap contains passive Acceptance Notes derived directly from the assignment instructions, user constraints, and deliverable. The notes remind the child to self-check where useful and to report anything unverified, environment-limited, partial, or blocked. They create no criterion IDs, evidence kinds, revision checks, or completion state. Test changes made by a child are reported to Jarvis for visibility only. Delegation metadata can describe a coordination phase, dependencies, and a minimal seam contract; these are context for Jarvis's scheduling and review decisions, not a static acceptance gate.

The main-agent coordination SOP is the primary guidance for delegated work. It teaches Jarvis to establish enough shared reality before parallelizing coupled work, define semantic seams without prescribing implementation, stage dependency waves, surface missing canonical dependencies, review producer/seam/consumer integration, and judge the assembled result. The runtime adds only sparse, non-blocking reminders at meaningful fan-out, upstream-work availability, and final-integration moments; it never determines whether an architecture or result is semantically correct.

Slice rollover follows the same unresolved-call normalization rule:

1. persist the unexecuted-tool-call notice
2. persist an automatic-continuation instruction with boundary diagnostics
3. reset stateful provider continuation lineage so the normalized transcript can be replayed without leaving a server-side tool call unresolved
4. rebuild a normal follow-up request with basic tools and all current-turn discoverable activations still available
5. reset the internal slice counter and continue executing any newly returned tool calls

A completely empty continuation response is normalized to an explicit failure message. A text or tool-call continuation is preserved as returned. Streaming additionally emits the continued tool-call event so UI state remains accurate.

A subagent final response is always a valid handoff, regardless of workspace mutation or verification outcome. The runtime does not intercept completion text, append acceptance continuations, retry because acceptance is incomplete, or force a handoff after a retry threshold. A settled child can release its write lease through `subagent_handoff` while retaining its state for main-agent review; disposal remains a separate lifecycle decision. Work waiting on orchestrator-owned bash or subagent activity still uses the separate `task_completion_deferred` mechanism, and genuine external/runtime blocks still use structured `completion_block_reason`; neither is semantic acceptance.

Lease-backed subagents also receive runtime-owned workspace change evidence. The manager captures a content/type manifest after lease acquisition and diffs it at settled lifecycle boundaries, yielding exact net changed paths inside the child’s owned scope without relying on Git, tool metadata, or Bash command parsing. Handoff closes the current capture segment; a later step-in starts a new one so main-agent integration edits are not attributed to the child. If a snapshot cannot be captured, the runtime marks the evidence incomplete rather than presenting a partial list as authoritative.

## Crash And Restart Recovery

On session access, orphaned `in_progress` turns are reconciled before the session is used again.

The loop:

- finds `in_progress` turns that are not active in memory
- inspects their persisted records
- appends an explicit unexecuted-tool-call notice when needed
- appends an orphaned-turn recovery system note
- marks the turn `interrupted`

Same-process runtime failures first persist any partial streamed assistant text, then use the same normalization immediately for the active turn.

The process lifecycle adds a route-level boundary around that actor recovery. The combined
entrypoint and ASGI lifespan call `SessionRouter.graceful_shutdown()` for every route created in
the process. Each route closes its automatic-follow-up gate and reuses the route hard-quiescence
path used by `/stop`: it hard-stops the main and child actors with the `process_shutdown`
interruption reason, pauses affected children with `pause_reason=process_shutdown`, terminates
route-owned detached jobs and services, drains queued work, and persists a durable process
shutdown note. The lifecycle reason is distinct from `/stop`, so user-facing stop events and
confirmations are not emitted during process shutdown.

Route initialization reconstitutes non-disposed child contracts owned by the active main session
or a compaction ancestor. The manager rebuilds each child actor from its durable assignment,
transcript root, selected skills, coordination metadata, and change evidence, then calls the
actor's recovery session preparation so orphaned child turns are normalized before future use.
Reconstitution never launches a child turn automatically. Persisted `running`,
`awaiting_approval`, and `waiting_background` entries are conservatively classified as
unexpected interruption and restored as paused `process_restart` children; cleanly shut down
children remain paused with `process_shutdown`. Held workspace leases are reacquired before
continuation, while lease conflicts leave an in-flight child paused as `external_blocked` and
preserve the conflict in its durable state. Explicitly released handoff leases remain released.
Only a later explicit main-agent decision can step a restored child in, hand it off, or dispose
it. Destructive termination is not covered by this recovery path.

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

Detached bash jobs are intentionally distinct from in-turn awaits, but `/stop`, `/new`, and
graceful process shutdown terminate and finalize route-owned detached jobs and services.
Foreground bash execution is best-effort cancelled when its active tool await is preempted.
`/new` additionally disposes children and replaces the session; `/stop` and graceful process
shutdown preserve durable child/session state for later explicit handling.

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

Telegram treats `task_status` as the master typing indicator. Approval/auth prompts, visible progress updates, provider failures, and chat-output pause do not clear it; only an inactive route status does. If the heartbeat task exits unexpectedly while the route remains active, the bridge persists the exception and restarts the task. A terminal main-loop error tied to a user or runtime turn produces only the generic functional notice `❌ Error occurred. Try again.`; unbound gateway/transport failures are persisted with route/session context and remain silent in both Telegram and terminal output. `/stop` temporarily mutes stale in-flight output, waits for the hard-quiesce acknowledgement, and then always sends the single functional confirmation `Session stopped.`, including when the route became quiescent just before the stop request was processed.

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

Jarvis compacts sessions into a canonical structured `CompactionBundle` and deterministically compiles it into replayable history. Context rollover is a harness primitive; the model contributes only a best-effort semantic refresh. Jarvis owns replay roles, the system boundary, exact recent-message selection and copying, bundle identity, chronology, source lineage, hashes, size enforcement, archive/activation, and authoritative runtime-state carry-forward.

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

The provider does not receive this verbose internal representation. Jarvis renders a compact causal transcript with local event labels used only for readability and tool-call/result links, plus event type, role, time, and semantic content. The model never returns those labels and no label is commit-critical. Internal record, session, turn, and call IDs are not exposed. When an exact long tool argument already appears in its paired result, the renderer substitutes a reference instead of sending the bytes twice.

Every semantic refresh request is estimated before dispatch with the intentionally lightweight four-characters-per-token heuristic. Its safety ceiling is 70% of the operator-selected normal preflight input limit. Jarvis first reserves the fixed instructions/prior-bundle cost, then applies one global source-character budget across all evidence; tool arguments receive both nested-string and whole-payload bounds. Routine bash lifecycle updates, identical system evidence, and selected repeated failure records are deterministically coalesced to their latest causal event. All source event bodies may be shortened for this lossy semantic input because the transcript archive remains authoritative. An input-budget failure rejects only the semantic refresh and does not block rollover.

### Canonical bundle and deterministic recent context

Schema version 4 contains:

- Jarvis-owned bundle ID, generation, creation time, prior-bundle lineage, ordered source sessions, current delta record IDs, cutoff record ID, delta hash, rolling cumulative hash, and cumulative record count
- current objective and essential background
- a bounded, causal recent tail of exact user/assistant records selected by Jarvis, including source role, bytes, content hash, deterministic selection reason, and chronology
- chronological episodes with Jarvis-derived stable IDs, outcomes, and chronology
- structured current entries for constraints, decisions, artifacts, open loops, and uncertainties, with Jarvis-derived IDs
- current handover focus, next actions, work not to repeat, and required verification
- semantic provenance recording accepted model output or fallback, the fallback source, and the rejection/failure code

Artifact entries carry an exact locator, last-observed state, and `needs_verification`; they do not assert mutable external state as timeless truth. Open-loop entries require a next action and may carry a blocker. Schema v4 is the only accepted bundle format; older development schemas are intentionally unsupported.

Recent selection is independent of the model, uses a derived token/character budget capped at 12,000 estimated tokens, walks backward by current-turn groups, and stops rather than splitting a causal group. Previously retained recent records participate in the same bounded tail on later generations. Oversized exact records remain authoritative in the archived transcript and are not forced into the next provider context.

### Incremental merge

The first compaction builds a bundle from the current session ledger. Later compactions load the canonical bundle anchor and pass its compact semantic state plus only new non-replay transcript records. The model returns one complete current semantic record rather than an incremental database mutation protocol. The model-facing tool requires only an objective and current focus; the richer background, episode, state, artifact, open-work, and handover fields are optional. Jarvis derives IDs and tolerates omitted optional sections.

The model does not select exact messages, return preservation references, or author exact source content. Extra legacy-style preservation hints are ignored by bundle assembly and cannot affect rollover.

### Validation, fallback, and rollover

The provider is asked for one `submit_compaction` call and instructed to perform its semantic fidelity review before submission. Normal compaction makes exactly one semantic model call. Jarvis validates harness-owned lineage, hashes, exact-copy integrity, bundle serialization, and state shapes. Semantic validation is deliberately lightweight: malformed, empty, placeholder-like, or extremely low-information output is rejected as a quality candidate rather than as a rollover transaction.

There is no semantic repair loop and no second model verifier. A malformed response, semantic timeout, provider failure, or semantic input-budget failure immediately selects a deterministic fallback. Later generations reuse the previous adequate semantic snapshot; the first generation uses a minimal harness-authored continuation marker. Both fallbacks are combined with authoritative runtime state, deterministic recent context, and archive lineage before activation.

The semantic call has one total deadline. Compaction provider and memory awaits participate in route interruption; cancellation still stops compaction rather than being converted into fallback. Mid-turn compaction races against the active turn stop signal. Reactive, preflight, and manual compaction register their own active operation before a normal turn exists, so `/stop`, a superseding user message, and `/new` can cancel them as well. The assembled checkpoint is checked for interruption before the old session is archived; bundle activation then runs as a consistency-sensitive commit.

Semantic degradation is recorded durably in the source session's compaction audit metadata (`semantic_status`, `semantic_source`, and `semantic_issue_code`) and does not create a failure latch. Only fundamental deterministic preparation, storage, integrity, interruption, or commit failures escape the compactor and may latch an unchanged source revision. Auto and manual compaction continue to call the same `ContextCompactor` and activation mechanism.

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
- exact harness-selected `recent_message` items using the source user/assistant role and bytes
- chronological assistant `episode` items
- one assistant `state_snapshot`
- one assistant `handover`

Replay records use `type="compaction_replay"` metadata with bundle/generation IDs, exact-copy status, and source record IDs. No model-authored text receives system authority.

The old session receives one `kind="compaction"` audit record containing the explicit manual instruction, provider/model when a response completed, canonical bundle, semantic candidate when parseable, deterministic validation report, semantic status/source/issue, per-call traces, aggregate usage, and replay items. The old session remains active until the complete deterministic checkpoint is ready for activation.

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
