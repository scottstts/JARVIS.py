# Subagents

## Purpose

Subagents let the main Jarvis agent delegate bounded side work while remaining responsible for the final user-facing answer.

Subagents are implemented under `src/jarvis/subagent/`. They are not normal tools under `src/jarvis/tools/`, but the main agent controls them through synthetic core primitives with tool-call semantics.

## Product Rules

- Only the main agent can invoke subagents.
- Nested subagents are forbidden.
- Maximum active subagents is `7` by default (`JARVIS_SUBAGENT_MAX_ACTIVE` can lower or raise the operator-selected limit). Provider/model generation is independently bulkheaded at three concurrent requests.
- Active codenames are unique and come from: `Friday`, `Edith`, `Karen`, `Jocasta`, `Tadashi`, `Homer`, `Ultron`.
- A codename can be reused after the previous holder is disposed.
- Subagents use the same underlying `AgentLoop` engine as the main agent.
- Subagents do not use identity bootstrap files from `src/jarvis/identities/`.
- Subagents use prompt assets from `src/jarvis/subagent/prompts/` plus a task assignment written by the main agent.
- Subagents do not receive memory bootstrap.
- Subagents do not perform memory reflection.
- Subagents do not have access to memory tools.
- Runtime tool manifests under `/workspace/runtime_tools/` remain visible to subagents.
- The user does not directly converse with subagents.
- `/stop` hard-quiesces the route: main/child turns are preempted, waiting children pause, and detached jobs/services are terminated.
- `/new` hard-stops and disposes route subagents.
- Ordinary user messages supersede only the active main turn; they never redirect, pause, or stop a child.
- Jarvis decides whether to continue independent main-task work, inspect a child, step in, stop it, or wait.

## Architecture

`RouteRuntime` owns:

- one main actor
- one `SubagentManager`
- shared route event bus
- shared approval registry
- shared detached bash supervisor
- route-level scheduling for user and runtime turns

`SubagentManager` owns:

- codename allocation
- max-active enforcement
- child loop creation
- child background tasks
- child status and catalog updates
- subagent primitive execution
- routing child events into route events and main-agent progress notices

Each subagent uses a configured `AgentLoop` with:

- actor identity `kind="subagent"`
- its codename as display name
- subagent bootstrap builder
- filtered tool registry
- memory disabled
- reflection disabled
- subagent archive storage

## Prompt Bootstrap

Subagent static prompts live under:

- `src/jarvis/subagent/prompts/SYSTEM.md`
- `src/jarvis/subagent/prompts/OPERATING_RULES.md`

The dynamic assignment includes:

- codename
- stable `subagent_id`
- required stable `task_label`
- bounded main-agent instructions
- optional exact user constraints
- optional shared environment, dependency, and interface context
- optional owned workspace paths
- optional selected skill ids, whose full `SKILL.md` documents are embedded into bootstrap
- optional deliverable or success criteria

The assignment is injected after the static subagent prompts. Skills opened by the main agent earlier in the same turn are automatically inherited (up to four total, with explicit `skill_ids` taking precedence); when none were selected, the manager may attach a conservatively matched installed skill. Automatic matching requires assignment overlap with the skill's ID/name identity, not merely two generic words from a long description, so unrelated large skill prompts are not injected. Every child records `skill_selection_reason`, including the short no-match reason when no skill applies.

The main agent receives high-level subagent usage guidance through `PROGRAM.md` and detailed primitive docs from `src/jarvis/subagent/primitives.py`.

## Control Primitives

Subagent primitives are core-only synthetic tool definitions. They do not live in `ToolRegistry`.

### `subagent_invoke`

Starts a new background subagent with a fresh starter context.

Arguments:

- `task_label`
- `instructions`
- optional `user_constraints`
- optional `shared_context`
- optional `owned_paths`
- optional `skill_ids`
- optional `deliverable`

Returns:

- `subagent_id`
- codename
- task label
- status
- session id
- selected skill ids and owned paths
- active count

It allocates a codename, creates child storage/catalog entries, starts the child turn asynchronously, and emits a public route notice.

When the user explicitly requires delegation, task acceptance is coupled to an observed successful `subagent_invoke`; merely claiming delegation in `acceptance_record` is insufficient. Semantic edits to test artifacts additionally create a changed-test-review obligation that requires an independent subagent review and cited artifact evidence before completion.

### `subagent_monitor`

Inspects current subagent state without side effects.

It accepts an optional codename or `subagent_id`. Omitted `agent` summarizes all non-disposed subagents. Full output includes the durable assignment, selected skills, owned paths, recent activity, pause reason, complete report or latest checkpoint, report-completeness flag, provider error metadata, error-log path, transcript path, and `pending_background_job_ids`.

Repeated unchanged monitor calls return a minimal no-delta nudge instead of another full snapshot. Automatic main-context status snapshots are also suppressed while unchanged within one main session, but a fresh snapshot is emitted after `/new` or compaction because compaction prunes older status records.

### `subagent_stop`

Requests cooperative stop for a running or approval-blocked subagent.

Already paused, completed, failed, or disposed targets return a no-op result.

### `subagent_step_in`

Changes direction for an existing subagent.

If the target is running, Jarvis requests cooperative stop, waits for the turn to settle, and starts a fresh child turn on the same subagent loop with the new instructions.

Step-in is stop, settle, then new turn. It is not mid-token prompt injection.

### `subagent_dispose`

Permanently closes and removes a non-running subagent from the active set.

It releases the codename, marks the catalog entry disposed, closes the child loop resources, and emits a public route notice.

### `orchestrator_wait`

Parks only the main orchestrator when route-owned children or detached jobs are still active and there is no actionable main work. Jarvis supplies a preferred `wake_after_seconds`, a concise reason, and optional pending actor IDs. The runtime validates the actor set, clamps the liveness deadline, applies exponential backoff after unchanged reviews, and wakes immediately on terminal/attention events. Routine running/output-growth observations do not poll either model. This primitive is general across any pending actor mix; it is not special-cased to one child.

## Lifecycle And Status

Any non-disposed subagent counts toward the active limit, including completed and failed subagents. Explicit disposal is required to free a slot.

Statuses:

- `running`
- `awaiting_approval`
- `waiting_background`
- `paused`
- `completed`
- `failed`
- `disposed`

Important metadata:

- `pause_reason`
- `last_error`
- `last_error_metadata`
- `error_log_path`
- `last_tool_name`
- `last_activity_at`
- `task_label`
- `report_complete`
- `owner_main_session_id`
- `owner_main_turn_id`
- `current_subagent_session_id`
- `run_generation`
- `pending_background_job_ids`

## Tool Access

Subagents receive a filtered registry view.

Initial built-in blocklist:

- `memory_search`
- `memory_get`
- `memory_write`
- `memory_admin`
- `send_file`

Blocked built-in discoverables are excluded from `tool_search`, and backed discoverable activation only works when the backing built-in tool survives filtering.

Runtime manifest discoverables remain visible to subagents by default.

Subagent primitives are not exposed to subagents, and `SubagentManager` rejects invocations from non-main actors defensively.

## Tool Execution Coordination

Route-level tool execution uses a workspace coordinator rather than one global tool mutex. Exact-path reads and disjoint declared writes can run concurrently, while broad snapshot reads take an exclusive barrier. A child’s `owned_paths` are exclusive persistent write leases until disposal; direct file producers/consumers acquire exact paths automatically, while mutable `bash` must declare all `write_paths` whenever another actor holds a lease. Overlapping access is serialized or rejected with ownership guidance instead of racing shared state.

Detached bash jobs are supervised route-wide. Subagent background jobs carry owner metadata and terminal/attention evidence later revives the owning child loop through persisted child-system notes and runtime turns. Routine running, output-started, and output-growth observations do not spend a child model turn; accepted notices are latched while queued, and terminal notices carry bounded output plus stdout/stderr log paths for later inspection. A child waiting on detached work stays `waiting_background`; if an earlier acceptance handoff paused it without a user stop reason, terminal bash evidence may resume that same child so it can complete verification. `/stop` terminates the job and hard-pauses the affected child with reason `main_stop`.

## Storage

Main transcripts remain under:

```text
/workspace/archive/transcripts/jarvis/<route_id>/
```

Subagent transcripts live under:

```text
/workspace/archive/transcripts/subagents/<route_id>/index.json
/workspace/archive/transcripts/subagents/<route_id>/<main_session_id>/<subagent_id>/sessions_index.json
/workspace/archive/transcripts/subagents/<route_id>/<main_session_id>/<subagent_id>/sessions/<session_id>.jsonl
```

The route-level catalog stores:

- `subagent_id`
- codename
- status
- created/updated/disposed timestamps
- route id
- owner main session and turn ids
- durable task label and structured assignment fields
- current subagent session id
- current run generation
- pause reason
- last error, structured provider metadata, and error-log path

Each subagent has its own `SessionStorage` root and normal session compaction lineage.

## Route And UI Events

The unified route websocket carries main and subagent events. Every event carries route ordering plus origin session/turn provenance; child lifecycle, tool, and approval events also carry the child run generation. The runtime discards stale generations after stop, disposal, restart, `/new`, or a newer session lineage.

Public Telegram notices are intentionally minimal:

- subagent invoked
- subagent disposed
- agent-attributed tool use
- agent-attributed approval prompts

General progress stays internal unless the main agent reports it in its own turn.

Approval requests include the acting agent name. Rejected subagent approvals pause the child and report state to the main-agent side of the route runtime.

## User Stop And Supersede

`/stop` uses the same hard-preemption mechanism as the destructive part of `/new`, but preserves the current session and child objects for a later explicit resume. Active provider/tool awaits are cancelled, `waiting_background` children become `paused(main_stop)`, pending child notices are cleared, and the route supervisor terminates/finalizes detached jobs and services before the stop acknowledgement. Only children actually affected by the stop are reported in the persisted stop note.

When a newer ordinary user message arrives, it supersedes only the active main turn. Existing children continue their original Jarvis assignments unchanged. The user message cannot become child context or an implicit child stop/redirect. Jarvis can later inspect, stop, or step into a child explicitly.

`/new` does not use either cooperative path. It hard-preempts active child turns with reason `new_session`, waits for their persisted turn state to settle, terminates route-owned detached jobs, archives and disposes every remaining child, and clears pending child notices before the replacement main session is created.

## Codex-Backed Subagents

Subagents can use provider `codex`. Codex-backed subagents share the Codex route connection with the main actor when applicable and use `CodexActorRuntime` with subagent identity, subagent bootstrap, filtered tools, and memory disabled.

Starting or stepping into a child does not force a Codex main turn to yield. Jarvis can continue useful independent work in the same turn; child progress remains orchestrator-monitored and should not be polled.

Subagent disposal closes the child loop so Codex-backed subagents unregister their thread mapping from the route coordinator.

## Maintenance Rules

When changing subagents:

- keep subagent control main-agent-only
- preserve explicit dispose semantics
- keep memory disabled unless a new design changes it
- keep tool filtering actor-scoped rather than hardcoded in `AgentLoop`
- keep runtime tools visible unless policy changes
- keep public UI progress minimal
- preserve child transcript linkability to the owning main session and turn
- preserve the rule that ordinary user messages affect only the main turn
- treat paused, approval-rejected, and failed children as inspect decisions, not finalize decisions
- keep full child reports and failure evidence accessible through full monitoring
