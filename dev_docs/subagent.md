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
- Delegation is a coordination decision, not a parallelism target. Jarvis should identify prerequisites, stage dependent work after its boundary exists, and review a completed child before integrating it.
- A useful assignment defines a seam rather than prescribing implementation: ownership, consumed interfaces, provided interfaces, lifecycle assumptions, invariants, verification scope, and non-goals.
- Subagents receive passive Acceptance Notes derived from assignment instructions, user constraints, and the requested deliverable; the notes are reminders for self-checking and reporting, never a completion gate.
- A subagent may always hand work back, including partial or blocked work. Jarvis decides whether to accept the report, inspect the changes, or step the child back in.

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
- optional coordination phase, dependency references, and seam contract
- optional selected skill ids, whose full `SKILL.md` documents are embedded into bootstrap
- optional deliverable or success criteria

The assignment is injected after the static subagent prompts. Skills opened by the main agent earlier in the same turn are automatically inherited, and explicit `skill_ids` can add exact top-level installed skill IDs, up to four total. Same-turn inheritance is only a convenience; Jarvis should explicitly repeat skill ids for later orchestration turns. The harness never infers skills from assignment prose, `SKILL.md`, referenced resources, or file/directory names. Every child records whether Jarvis selected skills or selected none.

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
- optional `phase`
- optional `depends_on`
- optional `seam_contract`
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
- coordination metadata and workspace lease status
- active count

It allocates a codename, creates child storage/catalog entries, starts the child turn asynchronously, and emits a public route notice.

A successful `subagent_invoke` delegates the task. The child bootstrap includes passive Acceptance Notes covering the assignment, constraints, deliverable, useful self-checking, and explicit reporting of anything unverified or blocked. Semantic edits to test artifacts are reported upward for visibility only; they do not create a runtime acceptance obligation. Coordination metadata is informational runtime state; Jarvis decides whether dependencies are satisfied.

### `subagent_monitor`

Inspects current subagent state without side effects.

It accepts an optional codename or `subagent_id`. Omitted `agent` summarizes all non-disposed subagents. Full output includes the durable assignment, coordination metadata, selected skills, owned paths, workspace lease status, changed paths, changed-path completeness/source, recent activity, pause reason, complete report or latest checkpoint, report-completeness flag, provider error metadata, error-log path, transcript path, and `pending_background_job_ids`. For lease-backed children, settled lifecycle snapshots report exact net changes within the owned paths; otherwise changed paths are tool-result evidence and may be incomplete. A report is self-reported evidence, not semantic acceptance; inspect changed paths and the seam before integrating implementation work.

Repeated unchanged monitor calls return a minimal no-delta nudge instead of another full snapshot. Automatic main-context status snapshots are also suppressed while unchanged within one main session, but a fresh snapshot is emitted after `/new` or compaction because compaction prunes older status records.

### `subagent_stop`

Requests cooperative stop for a running or approval-blocked subagent.

Already paused, completed, failed, or disposed targets return a no-op result.

### `subagent_step_in`

Changes direction for an existing subagent.

If the target is running, Jarvis requests cooperative stop, waits for the turn to settle, and starts a fresh child turn on the same subagent loop with the new instructions.

Step-in is stop, settle, then new turn. It is not mid-token prompt injection.

### `subagent_handoff`

Releases a settled (completed, paused, or failed) child's workspace write lease while preserving its transcript, report, status, and monitoring state for main-agent review and integration. It does not decide whether the work is correct and does not dispose the child. A later `subagent_step_in` reacquires the lease if it is still available.

### `subagent_dispose`

Permanently closes and removes a non-running subagent from the active set.

It releases the codename, marks the catalog entry disposed, closes the child loop resources, and emits a public route notice. Use `subagent_handoff` when the main agent needs the files without destroying the child state.

### `orchestrator_wait`

Parks only the main orchestrator when route-owned children or detached jobs are still active and there is no actionable main work. Jarvis supplies a preferred `wake_after_seconds`, a concise reason, and optional pending actor IDs. A successful call persists its tool result and yields the current main turn without another provider request. The runtime atomically persists and consumes routine queued notices; material inspect/finalize notices return one typed review result instead of parking. The runtime validates the actor set, clamps the liveness deadline, applies exponential backoff after unchanged reviews, and wakes immediately on later material events. Routine running/output-growth observations do not poll either model. This primitive is general across any pending actor mix; it is not special-cased to one child.

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
- `phase`
- `depends_on`
- `seam_contract`
- `changed_paths`
- `changed_paths_complete`
- `changed_paths_source`
- `workspace_lease_status`
- `owner_main_session_id`
- `owner_main_turn_id`
- `current_subagent_session_id`
- `run_generation`
- `pending_background_job_ids`

Pause reasons distinguish user/orchestrator control (`main_stop`, `new_session`, `approval_rejected`) from typed runtime dispositions (`tool_liveness_exhausted`, `provider_recovery_exhausted`, `external_blocked`). Acceptance Notes are informational only; they do not pause or automatically continue a child. A child may hand work back as complete, partial, or blocked, and Jarvis decides whether to inspect or continue it.

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

Route-level tool execution uses a workspace coordinator rather than one global tool mutex. A child’s `owned_paths` are exclusive persistent write leases until explicit handoff or disposal and must name existing files or directories. Direct path tools coordinate automatically and reject child writes outside those roots. Bash receives a runtime-derived filesystem capability view: children see the shared workspace read-only except for their owned roots, while Jarvis sees child-owned roots read-only. This enforcement applies to foreground, background and service Bash without model-declared paths or lease generations. The subagent manager captures a content/type snapshot after acquiring a lease and diffs it when the lease segment settles, hands off, or is disposed; this produces exact net `changed_paths` for the owned scope, including Bash, untracked files, deletes, and renames. Runtime-managed `archive` and `.jarvis_internal` roots are excluded from this evidence; other directories, including project caches, remain material when they are explicitly owned. A resumed child starts a new snapshot segment so main-agent integration edits are not attributed to it. This is independent of Git and is evidence for review, not a semantic acceptance gate.

Detached bash jobs are supervised route-wide. Subagent background jobs carry owner metadata and terminal/attention evidence later revives the owning child loop through persisted child-system notes and runtime turns. Routine running, output-started, and output-growth observations do not spend a child model turn; accepted notices are latched while queued, and terminal notices carry bounded output plus stdout/stderr log paths for later inspection. A silent-job attention notice first resumes the owning child and remains routine from Jarvis's perspective; the main agent is escalated only if that child subsequently cannot recover, pauses, fails, needs approval, or reports an unresolved blocker. A child waiting on detached work stays `waiting_background`; terminal bash evidence may resume that same child when it was waiting on the job. `/stop` terminates the job and hard-pauses the affected child with reason `main_stop`.

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
- per-agent tool activity stacks using raw tool names
- agent-attributed approval prompts

Tool activity is rendered as one editable Telegram message per agent. Tool names are bold, repeated uses increment a count, and the most recently used tool is moved to the bottom of that agent's list. Every user-facing Jarvis message partitions all current tool-activity stacks, so the next main-agent or subagent tool use starts a fresh notice below that message. System notices remain separate and do not partition tool activity.

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
