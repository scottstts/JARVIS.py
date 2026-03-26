# Persistence And Cache-Hit Notes

## Purpose

This document is the current source of truth for how transcript persistence relates to prompt-cache stability in the main agent loop and subagent loops.

The goal is:

- keep replayable transcript history as the source of truth for rebuilding `LLMRequest.messages`
- persist every non-image prompt-visible record that should affect later replay
- let provider adapters in `src/llm/providers/` translate unified replayed history into provider-native payloads
- avoid avoidable provider cache misses caused by transcript/replay drift

This document is about unified replayable history, not about persisting provider-native wire JSON.

## Scope

Applies to:

- main agent turns
- subagent turns
- tool follow-up rounds
- streaming and non-streaming paths
- crash/restart recovery of persisted turn history

Primary code:

- `src/core/agent_loop.py`
- `src/storage/service.py`
- `src/llm/providers/`
- `src/gateway/route_runtime.py`
- `src/subagent/manager.py`
- `src/subagent/bootstrap.py`

## Core Rules

1. Replayable transcript records are the source of truth for rebuilding `LLMRequest.messages`.
2. Non-image prompt-visible records must persist.
3. `transcript_only` means archived but intentionally excluded from replay.
4. Provider quirks stay in `src/llm/providers/`.
5. If replay would require inventing prompt-visible content, that is a bug unless the behavior is explicitly documented as a non-replay feature.

## Implemented Behavior

### 1. Initial turn-visible records persist before the authoritative provider call

The following are persisted as normal transcript records before the model call:

- turn context datetime message
- interruption notice message
- runtime messages from orchestrator or subagent plumbing

This keeps initial request-visible context replayable on later turns.

### 2. Follow-up records that affect the model persist too

The following are persisted when they are used:

- orchestrator monitored-waiting notices
- tool-round-limit notices
- approval request/result history
- assistant messages
- tool results

This applies to both streaming and non-streaming paths.

### 3. Streaming assistant persistence is canonical

Normal streamed completions persist one canonical assistant record.

If a stream is interrupted after text but before tool execution:

- the partial assistant text is persisted
- an explicit persisted unexecuted-tool-call notice is appended

Replay does not rely on streaming checkpoints as canonical history.

### 4. Replay does not synthesize unexecuted-tool-call notices

For current code, replay expects unresolved assistant tool calls to be normalized by explicitly persisted transcript records.

If replay encounters assistant tool calls without either:

- matching tool results, or
- an explicit persisted unexecuted-tool-call notice

then replay raises rather than inventing a new prompt-visible system message.

This is intentional. The transcript must already contain the normalization record.

### 5. Orphaned `in_progress` turns are reconciled on session access

If the process dies or restarts while a turn is still marked `in_progress`, the loop reconciles that turn before using the session again.

Current reconciliation behavior:

- find `in_progress` turns that are not the live in-memory active turn
- inspect the persisted records for that turn
- if the turn ended with unresolved proposed tool calls, append an explicit persisted unexecuted-tool-call notice
- append a persisted orphaned-turn recovery system note
- mark the turn `interrupted`

This prevents persisted prompt-visible records from disappearing just because the old process died before it could finalize turn status.

### 6. Tool-round-limit recovery persists explicit normalization first

When the loop hits the per-turn tool-round limit while the latest assistant response still contains unexecuted tool calls:

- it first persists the explicit unexecuted-tool-call notice
- then it persists the tool-round-limit instruction
- then it builds the recovery request

That keeps recovery replay deterministic and avoids replay-time synthesis.

## Intentional Exceptions

### 1. `view_image` remains ephemeral

`view_image` attachments are still current-turn-only and are not persisted in transcript.

Implication:

- turns that use `view_image` may still lose exact cache continuity on later turns

This is accepted to avoid storing image bytes or stable image archives in transcript-adjacent storage.

### 2. Discoverable-tool activation is current-turn-only by design

Backed discoverable tools activated by `tool_search` high-verbosity results are only surfaced for the rest of the current turn.

Current design:

- initial user-turn requests expose only `basic` tools
- `tool_search` may attach `activated_discoverable_tool_names` metadata to its tool result
- follow-up requests in the same turn may merge those discoverable tools
- later user turns do not carry those activations forward

This is intentional and documented behavior, not a persistence bug.

Implication:

- tool availability can intentionally shrink again at a user-turn boundary
- that can reduce cache reuse across that boundary
- this is accepted because the product behavior keeps initial tool lists smaller and avoids permanent tool-list growth

### 3. Compaction is an intentional cache boundary

Once a session is compacted into a new summary-backed session, exact cache continuity is intentionally broken.

That is expected.

## `transcript_only`

`transcript_only` remains valid and separate from ephemerality.

Examples:

- tool bootstrap audit records
- debugging or audit records that should be archived but not replayed

Rules:

- `transcript_only` records are persisted
- `transcript_only` records are excluded from replay
- do not use `transcript_only` as a substitute for prompt-visible current-turn data

## Subagent Notes

Subagents use the same `AgentLoop`, so the persistence and replay guarantees above apply to both:

- main loop
- subagent loop

That means the following fixes automatically cover both:

- pre-turn persistence
- explicit unexecuted-tool-call normalization
- orphaned-turn reconciliation
- canonical streamed assistant persistence

Subagent-specific runtime messages such as:

- step-in developer messages
- subagent bash progress system messages
- subagent orchestration progress notices

must continue to enter the shared loop through normal persisted message paths.

Current review of subagent paths did not identify an additional subagent-only prompt-persistence gap beyond the shared-loop rules above.

## What Still Causes Cache Misses

The refactor removes avoidable transcript/replay drift, but it does not prevent misses caused by legitimate request changes such as:

- `view_image` turns
- compaction into a new session
- model change
- provider change
- provider setting change
- tool registry change
- runtime tool manifest change
- intentionally current-turn-only discoverable activation boundaries

Those are not transcript-fidelity bugs.

## Maintenance Rules

When changing the agent loop, preserve these invariants:

- if the model sees it and it is replayable later, persist it
- if replay needs it later, persist it in unified transcript form
- do not add new replay-time synthesis for prompt-visible content
- keep image ephemerality explicit and isolated
- if a new crash-recovery path can leave assistant tool calls unresolved, persist the normalization notice at the time of reconciliation
- if a change affects subagent prompt-visible behavior, review the shared-loop effect and document it here

## Tests To Keep

At minimum, regression coverage should continue to include:

- turn-context persistence
- orchestrator/runtime message persistence
- interrupted tool-call normalization across turns
- completed tool results surviving interrupted turns
- orphaned `in_progress` turn reconciliation
- tool-round-limit recovery with explicit unexecuted-tool-call notice
- discoverable-tool activation remaining current-turn-only
- image attachments remaining ephemeral

## Summary

Current contract:

- replay is transcript-driven
- non-image prompt-visible records persist
- unresolved tool-call normalization must be explicit and persisted
- orphaned turns are reconciled into visible interrupted history
- discoverable activation remains current-turn-only on purpose
- the same rules apply to main and subagent loops because both use the shared `AgentLoop`
